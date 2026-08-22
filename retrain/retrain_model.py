import argparse
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from google.cloud import storage
from torch.optim import AdamW
from transformers import PatchTSTConfig

from src.ml.dataset import get_dataloaders
from src.ml.train_patchtst import PatchTSTRegressor


load_dotenv()

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_blob(
    storage_client: storage.Client,
    bucket_name: str,
    source_blob: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob)

    if not blob.exists():
        raise FileNotFoundError(
            f"Missing gs://{bucket_name}/{source_blob}"
        )

    blob.download_to_filename(destination)
    print(f"Downloaded gs://{bucket_name}/{source_blob}")


def upload_blob(
    storage_client: storage.Client,
    bucket_name: str,
    source_file: Path,
    destination_blob: str,
) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)
    uri = f"gs://{bucket_name}/{destination_blob}"

    print(f"Uploaded {source_file.name} to {uri}")
    return uri


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            predictions.append(preds.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    loss = total_loss / len(loader.dataset)
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    return loss, mae, rmse


def train_one_dataset(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
):
    train_csv = data_dir / f"train_{dataset}_processed.csv"
    val_csv = data_dir / f"val_{dataset}_processed.csv"
    test_csv = data_dir / f"test_{dataset}_processed.csv"

    train_loader, val_loader, test_loader, train_stats = get_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        context_length=56,
        batch_size=batch_size,
    )

    config = PatchTSTConfig(
        context_length=56,
        patch_length=16,
        patch_stride=8,
        num_input_channels=24,
        d_model=256,
        encoder_layers=4,
        encoder_attention_heads=8,
        encoder_ffn_dim=512,
        dropout=0.15,
        head_dropout=0.15,
    )

    model = PatchTSTRegressor(config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.SmoothL1Loss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
    )

    checkpoint_path = output_dir / f"best_model_{dataset}.pt"

    best_mae = float("inf")
    best_epoch = -1
    patience_counter = 0

    mlflow.set_experiment(f"turbineguard_{dataset.lower()}")

    with mlflow.start_run(run_name=f"scheduled_retrain_{dataset.lower()}"):
        mlflow.log_params(
            {
                "dataset": dataset,
                "context_length": 56,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "seed": seed,
                "num_input_channels": 24,
                "normalization": "operating_regime_kmeans",
            }
        )

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

                total_train_loss += loss.item() * x_batch.size(0)

            train_loss = total_train_loss / len(train_loader.dataset)

            val_loss, val_mae, val_rmse = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"{dataset} | Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"MAE: {val_mae:.2f} | "
                f"RMSE: {val_rmse:.2f} | "
                f"LR: {current_lr:.2e}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            scheduler.step(epoch + 1)

            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "dataset": dataset,
                        "model_state_dict": model.state_dict(),
                        "mae": val_mae,
                        "rmse": val_rmse,
                        "train_stats": train_stats,
                        "config": config.to_dict(),
                        "context_length": 56,
                        "num_input_channels": 24,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(
                        f"{dataset} early stopped at epoch {epoch + 1}"
                    )
                    break

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_mae, test_rmse = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        mlflow.log_metrics(
            {
                "best_val_mae": best_mae,
                "best_epoch": best_epoch + 1,
                "test_loss": test_loss,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
            }
        )

    return {
        "dataset": dataset,
        "checkpoint_path": checkpoint_path,
        "best_val_mae": best_mae,
        "best_epoch": best_epoch + 1,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASETS],
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("GCS_BUCKET", "turbineguard-data"),
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", ""),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_dir = Path("/tmp/turbineguard_retrain")
    data_dir = work_dir / "data"
    output_dir = work_dir / "models"

    if work_dir.exists():
        shutil.rmtree(work_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_client = storage.Client()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Retraining: {', '.join(datasets)}")
    print(f"Source bucket: gs://{args.bucket}")

    for dataset in datasets:
        for split in ["train", "val", "test"]:
            filename = f"{split}_{dataset}_processed.csv"

            download_blob(
                storage_client=storage_client,
                bucket_name=args.bucket,
                source_blob=f"processed/{filename}",
                destination=data_dir / filename,
            )

    results = []

    for dataset in datasets:
        result = train_one_dataset(
            dataset=dataset,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed,
        )

        model_blob = (
            f"models/{dataset}/"
            f"best_model_{dataset}_{timestamp}.pt"
        )

        result["gcs_model_uri"] = upload_blob(
            storage_client=storage_client,
            bucket_name=args.bucket,
            source_file=result["checkpoint_path"],
            destination_blob=model_blob,
        )

        results.append(result)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nRetraining complete")
    for result in results:
        print(
            f"{result['dataset']} | "
            f"Best Val MAE: {result['best_val_mae']:.2f} | "
            f"Test MAE: {result['test_mae']:.2f} | "
            f"Test RMSE: {result['test_rmse']:.2f} | "
            f"Model: {result['gcs_model_uri']}"
        )


if __name__ == "__main__":
    main()