import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PatchTSTForRegression, PatchTSTConfig

from src.ml.dataset import CMAPSSDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--context-length", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    config = PatchTSTConfig(**checkpoint["config"])
    model = PatchTSTForRegression(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_csv = args.data_dir / "test_FD001_processed.csv"

    test_ds = CMAPSSDataset(
        test_csv,
        context_length=args.context_length,
        stride=1,
        normalize=True,
        train_stats=checkpoint["train_stats"],
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = outputs.prediction.squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()