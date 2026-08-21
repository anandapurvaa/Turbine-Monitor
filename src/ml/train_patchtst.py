import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PatchTSTModel, PatchTSTConfig
import mlflow
import numpy as np

from src.ml.dataset import get_dataloaders


class PatchTSTRegressor(nn.Module):
    """
    Custom wrapper around PatchTST encoder for regression.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = PatchTSTModel(config)
        # Regression head: takes d_model and outputs 1
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        
    def forward(self, x):
        # x: (B, context_length, num_channels)
        outputs = self.encoder(x)
        # outputs.last_hidden_state: (B, num_channels, num_patches, d_model)
        hidden = outputs.last_hidden_state  # (B, C, P, D)
        # Pool over channels and patches to get (B, D)
        pooled = hidden.mean(dim=(1, 2))  # (B, D)
        pred = self.regression_head(pooled)  # (B, 1)
        pred = pred.squeeze(-1)  # (B,)
        return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--context-length", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="patchtst_fd001")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = args.data_dir / "train_FD001_processed.csv"
    test_csv = args.data_dir / "test_FD001_processed.csv"

    train_loader, test_loader, train_stats = get_dataloaders(
        train_csv,
        test_csv,
        context_length=args.context_length,
        batch_size=args.batch_size,
    )

    # PatchTST config
    config = PatchTSTConfig(
        context_length=args.context_length,
        patch_length=16,
        patch_stride=8,
        num_input_channels=21,
        d_model=128,
        encoder_layers=3,
        encoder_attention_heads=4,
        encoder_ffn_dim=256,
        dropout=0.1,
        head_dropout=0.1,
    )

    model = PatchTSTRegressor(config)
    model = model.to(device)

    # Debug: check output shapes
    print("Testing model forward pass...")
    test_x = torch.randn(2, args.context_length, 21).to(device)
    with torch.no_grad():
        test_out = model(test_x)
    print(f"Input shape: {test_x.shape}")
    print(f"Output shape: {test_out.shape}")
    print("If output shape is (2,), we're good. Starting training...\n")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    mlflow.set_experiment("turbineguard_patchtst_fd001")

    best_mae = float("inf")
    patience_counter = 0

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "d_model": config.d_model,
                "encoder_layers": config.encoder_layers,
                "num_input_channels": config.num_input_channels,
            }
        )

        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)  # (B, context_length, 21)
                y_batch = y_batch.to(device)  # (B,)

                optimizer.zero_grad()
                preds = model(x_batch)

                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Eval
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    preds = model(x_batch)

                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * x_batch.size(0)

                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())

            val_loss /= len(test_loader.dataset)
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)

            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mae": mae,
                    "rmse": rmse,
                },
                step=epoch,
            )

            scheduler.step(val_loss)

            if mae < best_mae:
                best_mae = mae
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "mae": mae,
                        "rmse": rmse,
                        "train_stats": train_stats,
                        "config": config.to_dict(),
                    },
                    "best_model.pt",
                )
                print(f"New best MAE: {mae:.2f} — model saved")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        mlflow.log_metric("best_mae", best_mae)
        print(f"Training complete. Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()