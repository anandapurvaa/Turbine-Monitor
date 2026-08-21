import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from src.ml.dataset import CMAPSSDataset


class PatchTSTRegressor(nn.Module):
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


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = PatchTSTConfig.from_dict(checkpoint["config"])
    model = PatchTSTRegressor(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["train_stats"]


def predict(model, data_loader, device: torch.device) -> np.ndarray:
    predictions = []

    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            predictions.append(model(x_batch).cpu().numpy())

    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=36)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_stats = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

    dataset_name = args.checkpoint.stem.replace("best_model_", "")
    test_csv = args.data_dir / f"test_{dataset_name}_processed.csv"

    if not test_csv.exists():
        raise FileNotFoundError(f"Processed test file not found: {test_csv}")

    test_dataset = CMAPSSDataset(
        test_csv,
        context_length=args.context_length,
        stride=1,
        normalize=True,
        train_stats=train_stats,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    predictions = predict(model, test_loader, device)

    if len(predictions) != len(test_dataset.targets):
        raise RuntimeError(
            f"Prediction count ({len(predictions)}) does not match target count "
            f"({len(test_dataset.targets)})."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        {
            "unit_number": test_dataset.window_unit_ids,
            "cycle": test_dataset.window_end_cycles,
            "operating_regime": test_dataset.window_regime_ids,
            "predicted_rul": predictions,
            "actual_rul": test_dataset.targets,
        }
    )
    results.to_csv(args.output, index=False)

    mae = np.mean(np.abs(predictions - test_dataset.targets))
    rmse = np.sqrt(np.mean((predictions - test_dataset.targets) ** 2))

    print(f"Saved predictions to {args.output}")
    print(f"Window-level MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    print(f"Windows: {len(results)} | Engines: {results['unit_number'].nunique()}")


if __name__ == "__main__":
    main()