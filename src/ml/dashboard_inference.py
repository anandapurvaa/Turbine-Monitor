from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel


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


class DashboardPredictor:
    """Loads trained PatchTST checkpoints and predicts RUL for one engine window."""

    OP_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
    SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
    FEATURE_COLUMNS = OP_COLUMNS + SENSOR_COLUMNS
    VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: dict[str, dict[str, Any]] = {}

    def _load(self, dataset: str) -> dict[str, Any]:
        dataset = dataset.upper()
        if dataset not in self.VALID_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if dataset in self._cache:
            return self._cache[dataset]

        checkpoint_path = Path(f"best_model_{dataset}.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train {dataset} first."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        config = PatchTSTConfig.from_dict(checkpoint["config"])
        model = PatchTSTRegressor(config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        train_stats = checkpoint["train_stats"]
        required_stats = {
            "n_regimes",
            "op_mean",
            "op_std",
            "regime_centers",
            "sensor_means",
            "sensor_stds",
        }
        missing_stats = required_stats.difference(train_stats)
        if missing_stats:
            raise ValueError(
                f"{checkpoint_path} uses old normalization statistics. "
                "Retrain with regime-normalized dataset.py. "
                f"Missing: {sorted(missing_stats)}"
            )

        bundle = {
            "model": model,
            "context_length": int(checkpoint.get("context_length", config.context_length)),
            "train_stats": {
                "n_regimes": int(train_stats["n_regimes"]),
                "op_mean": np.asarray(train_stats["op_mean"], dtype=np.float32),
                "op_std": np.asarray(train_stats["op_std"], dtype=np.float32),
                "regime_centers": np.asarray(train_stats["regime_centers"], dtype=np.float32),
                "sensor_means": np.asarray(train_stats["sensor_means"], dtype=np.float32),
                "sensor_stds": np.asarray(train_stats["sensor_stds"], dtype=np.float32),
            },
        }
        self._cache[dataset] = bundle
        return bundle

    @staticmethod
    def _normalize_window(window: pd.DataFrame, train_stats: dict[str, Any]) -> tuple[np.ndarray, int]:
        op_values = window[["op_setting_1", "op_setting_2", "op_setting_3"]].to_numpy(dtype=np.float32)
        sensor_values = window[[f"sensor_{index}" for index in range(1, 22)]].to_numpy(dtype=np.float32)

        op_mean = train_stats["op_mean"]
        op_std = np.where(train_stats["op_std"] < 1e-8, 1.0, train_stats["op_std"])
        normalized_ops = (op_values - op_mean) / op_std

        centers = train_stats["regime_centers"]
        distances = ((normalized_ops[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        regime_ids = distances.argmin(axis=1)

        sensor_means = train_stats["sensor_means"]
        sensor_stds = np.where(train_stats["sensor_stds"] < 1e-8, 1.0, train_stats["sensor_stds"])
        normalized_sensors = np.empty_like(sensor_values, dtype=np.float32)

        for regime in range(train_stats["n_regimes"]):
            mask = regime_ids == regime
            if np.any(mask):
                normalized_sensors[mask] = (
                    sensor_values[mask] - sensor_means[regime]
                ) / sensor_stds[regime]

        features = np.concatenate([normalized_ops, normalized_sensors], axis=1).astype(np.float32)
        return features, int(regime_ids[-1])

    def predict_from_dataframe(self, dataset: str, engine_window: pd.DataFrame) -> dict[str, float | int]:
        bundle = self._load(dataset)
        context_length = bundle["context_length"]

        missing_columns = set(self.FEATURE_COLUMNS).difference(engine_window.columns)
        if missing_columns:
            raise ValueError(f"Engine data missing input columns: {sorted(missing_columns)}")

        window = engine_window.sort_values("time_in_cycles").tail(context_length).copy()
        if len(window) < context_length:
            first_row = window.iloc[[0]].copy()
            padding = pd.concat([first_row] * (context_length - len(window)), ignore_index=True)
            window = pd.concat([padding, window], ignore_index=True)

        features, regime_id = self._normalize_window(window, bundle["train_stats"])
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            predicted_rul = float(bundle["model"](x).item())

        predicted_rul = max(0.0, predicted_rul)

        return {
            "predicted_rul": predicted_rul,
            "operating_regime": regime_id,
            "context_length": context_length,
        }


_predictor = DashboardPredictor()


def predict_engine_rul(dataset: str, engine_window: pd.DataFrame) -> dict[str, float | int]:
    """Return a PatchTST RUL prediction for a recent engine trajectory."""
    return _predictor.predict_from_dataframe(dataset, engine_window)
