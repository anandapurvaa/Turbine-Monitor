import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PatchTSTConfig, PatchTSTModel

from src.ml.dataset import get_dataloaders


class PatchTSTRegressor(nn.Module):
    """PatchTST encoder with a regression head for capped RUL prediction."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.encoder = PatchTSTModel(config)
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=(1, 2))
        return self.regression_head(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    loss = total_loss / len(data_loader.dataset)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return loss, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--context-length", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regime-normalize",
        action="store_true",
        help="Apply operating-regime-specific sensor normalization. Recommended for FD002/FD004.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all"
        else [args.dataset]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"Training on {dataset}")
        print(f"{'=' * 60}\n")

        train_csv = args.data_dir / f"train_{dataset}_processed.csv"
        val_csv = args.data_dir / f"val_{dataset}_processed.csv"
        test_csv = args.data_dir / f"test_{dataset}_processed.csv"

        for csv_path in (train_csv, val_csv, test_csv):
            if not csv_path.exists():
                raise FileNotFoundError(f"Processed dataset not found: {csv_path}")

        train_loader, val_loader, test_loader, train_stats = get_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            context_length=args.context_length,
            batch_size=args.batch_size,
        )

        num_input_channels = 24
        sample_x, _ = next(iter(train_loader))
        if sample_x.ndim != 3 or sample_x.shape[-1] != num_input_channels:
            raise ValueError(
                f"Expected batches shaped (B, context, {num_input_channels}), "
                f"received {tuple(sample_x.shape)}. Check dataset.py."
            )

        config = PatchTSTConfig(
            context_length=args.context_length,
            patch_length=16,
            patch_stride=8,
            num_input_channels=num_input_channels,
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
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6,
        )

        checkpoint_path = Path(f"best_model_{dataset}.pt")
        mlflow.set_experiment(f"turbineguard_{dataset.lower()}")

        best_mae = float("inf")
        best_epoch = -1
        patience_counter = 0

        with mlflow.start_run(run_name=f"patchtst_{dataset.lower()}"):
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "context_length": args.context_length,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "seed": args.seed,
                    "d_model": config.d_model,
                    "encoder_layers": config.encoder_layers,
                    "num_input_channels": num_input_channels,
                    "regime_normalize": args.regime_normalize,
                }
            )

            for epoch in range(args.epochs):
                model.train()
                total_train_loss = 0.0

                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += loss.item() * x_batch.size(0)

                train_loss = total_train_loss / len(train_loader.dataset)
                val_loss, val_mae, val_rmse = evaluate_model(
                    model, val_loader, criterion, device
                )

                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | "
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
                            "context_length": args.context_length,
                            "num_input_channels": num_input_channels,
                        },
                        checkpoint_path,
                    )
                    print(f"  -> New best validation MAE: {val_mae:.2f}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  -> Early stopping at epoch {epoch + 1}")
                        break

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,                #test
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            test_loss, test_mae, test_rmse = evaluate_model(
                model, test_loader, criterion, device
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

            print(f"\n{dataset} complete")
            print(f"Best validation MAE: {best_mae:.2f} at epoch {best_epoch + 1}")
            print(f"Test MAE: {test_mae:.2f} | Test RMSE: {test_rmse:.2f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

