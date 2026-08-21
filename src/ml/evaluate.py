import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import PatchTSTConfig

from src.ml.dataset import CMAPSSDataset
from src.ml.train_patchtst import PatchTSTRegressor  # Import from training file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--context-length", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        model_path = Path(f"best_model_{fd}.pt")
        if not model_path.exists():
            print(f"⚠️  Skipping {fd} - no checkpoint found")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating {fd}")
        print(f"{'='*50}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = PatchTSTConfig(**checkpoint["config"])
        model = PatchTSTRegressor(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        test_csv = args.data_dir / f"test_{fd}_processed.csv"

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
                preds = model(x_batch).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(y_batch.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        results.append({"Dataset": fd, "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}", "Samples": len(all_targets)})
        print(f"Test MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for r in results:
        print(f"{r['Dataset']}: MAE={r['MAE']}, RMSE={r['RMSE']}")
    print("="*50)


if __name__ == "__main__":
    main()