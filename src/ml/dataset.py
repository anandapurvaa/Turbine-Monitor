from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset


class CMAPSSDataset(Dataset):
    """Sliding-window C-MAPSS dataset with operating-regime-aware normalization.

    The model input contains 24 channels:
    - 3 globally standardized operating settings
    - 21 sensor values normalized within their assigned operating regime

    Regimes and normalization statistics are fitted on the training split only and
    reused for validation, test, and inference through ``train_stats``.
    """

    def __init__(
        self,
        csv_path: Path,
        context_length: int = 36,
        stride: int = 1,
        normalize: bool = True,
        train_stats: Optional[dict] = None,
        n_regimes: int = 6,
        random_state: int = 42,
    ):
        self.context_length = context_length
        self.stride = stride
        self.n_regimes = n_regimes

        df = pd.read_csv(csv_path)

        self.op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
        self.sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        self.feature_cols = self.op_cols + self.sensor_cols

        required_columns = self.feature_cols + ["unit_nr", "time_in_cycles", "RUL_capped"]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")

        op_values = df[self.op_cols].to_numpy(dtype=np.float32)
        sensor_values = df[self.sensor_cols].to_numpy(dtype=np.float32)

        self.rul = df["RUL_capped"].to_numpy(dtype=np.float32)
        self.unit_ids = df["unit_nr"].to_numpy()
        self.cycles = df["time_in_cycles"].to_numpy()

        if normalize:
            if train_stats is None:
                self.train_stats = self._fit_normalization(
                    op_values=op_values,
                    sensor_values=sensor_values,
                    n_regimes=n_regimes,
                    random_state=random_state,
                )
            else:
                self.train_stats = self._validate_train_stats(train_stats)

            self.regime_ids = self._assign_regimes(op_values, self.train_stats)
            normalized_ops = self._normalize_operating_settings(op_values, self.train_stats)
            normalized_sensors = self._normalize_sensors_by_regime(
                sensor_values,
                self.regime_ids,
                self.train_stats,
            )
            self.features = np.concatenate(
                [normalized_ops, normalized_sensors],
                axis=1,
            ).astype(np.float32)
        else:
            self.regime_ids = np.zeros(len(df), dtype=np.int64)
            self.features = np.concatenate([op_values, sensor_values], axis=1).astype(np.float32)
            self.train_stats = {
                "n_regimes": 1,
                "op_mean": np.zeros((1, 3), dtype=np.float32),
                "op_std": np.ones((1, 3), dtype=np.float32),
                "regime_centers": np.zeros((1, 3), dtype=np.float32),
                "sensor_means": np.zeros((1, 21), dtype=np.float32),
                "sensor_stds": np.ones((1, 21), dtype=np.float32),
            }

        windows = []
        targets = []
        window_unit_ids = []
        window_end_cycles = []
        window_regime_ids = []

        for unit in np.unique(self.unit_ids):
            indices = np.flatnonzero(self.unit_ids == unit)
            unit_features = self.features[indices]
            unit_rul = self.rul[indices]
            unit_cycles = self.cycles[indices]
            unit_regimes = self.regime_ids[indices]

            for start in range(0, len(indices) - context_length + 1, stride):
                end = start + context_length
                windows.append(unit_features[start:end])
                targets.append(unit_rul[end - 1])
                window_unit_ids.append(unit)
                window_end_cycles.append(unit_cycles[end - 1])
                window_regime_ids.append(unit_regimes[end - 1])

        if not windows:
            raise ValueError(
                f"No windows were created from {csv_path}. "
                f"Ensure each unit has at least context_length={context_length} rows."
            )

        self.windows = np.stack(windows).astype(np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.window_unit_ids = np.asarray(window_unit_ids)
        self.window_end_cycles = np.asarray(window_end_cycles)
        self.window_regime_ids = np.asarray(window_regime_ids, dtype=np.int64)

    @staticmethod
    def _fit_normalization(op_values, sensor_values, n_regimes, random_state):
        op_mean = op_values.mean(axis=0, keepdims=True)
        op_std = op_values.std(axis=0, keepdims=True)
        op_std = np.where(op_std < 1e-8, 1.0, op_std)

        standardized_ops = (op_values - op_mean) / op_std

        effective_regimes = min(n_regimes, len(standardized_ops))
        kmeans = KMeans(
            n_clusters=effective_regimes,
            random_state=random_state,
            n_init=20,
        )
        regime_ids = kmeans.fit_predict(standardized_ops)

        sensor_means = np.zeros((effective_regimes, sensor_values.shape[1]), dtype=np.float32)
        sensor_stds = np.ones((effective_regimes, sensor_values.shape[1]), dtype=np.float32)

        for regime in range(effective_regimes):
            regime_sensors = sensor_values[regime_ids == regime]
            sensor_means[regime] = regime_sensors.mean(axis=0)
            regime_std = regime_sensors.std(axis=0)
            sensor_stds[regime] = np.where(regime_std < 1e-8, 1.0, regime_std)

        return {
            "n_regimes": int(effective_regimes),
            "op_mean": op_mean.astype(np.float32),
            "op_std": op_std.astype(np.float32),
            "regime_centers": kmeans.cluster_centers_.astype(np.float32),
            "sensor_means": sensor_means,
            "sensor_stds": sensor_stds,
        }

    @staticmethod
    def _validate_train_stats(train_stats):
        required_keys = {
            "n_regimes",
            "op_mean",
            "op_std",
            "regime_centers",
            "sensor_means",
            "sensor_stds",
        }
        missing = required_keys.difference(train_stats)
        if missing:
            raise ValueError(
                "Checkpoint/train statistics use the old global-normalization format. "
                "Retrain with the regime-normalized dataset before evaluating or inferring. "
                f"Missing keys: {sorted(missing)}"
            )

        n_regimes = int(train_stats["n_regimes"])
        normalized_stats = {
            "n_regimes": n_regimes,
            "op_mean": np.asarray(train_stats["op_mean"], dtype=np.float32),
            "op_std": np.asarray(train_stats["op_std"], dtype=np.float32),
            "regime_centers": np.asarray(train_stats["regime_centers"], dtype=np.float32),
            "sensor_means": np.asarray(train_stats["sensor_means"], dtype=np.float32),
            "sensor_stds": np.asarray(train_stats["sensor_stds"], dtype=np.float32),
        }

        expected_shapes = {
            "op_mean": (1, 3),
            "op_std": (1, 3),
            "regime_centers": (n_regimes, 3),
            "sensor_means": (n_regimes, 21),
            "sensor_stds": (n_regimes, 21),
        }
        for key, expected_shape in expected_shapes.items():
            if normalized_stats[key].shape != expected_shape:
                raise ValueError(
                    f"Invalid train_stats[{key!r}] shape: "
                    f"expected {expected_shape}, got {normalized_stats[key].shape}."
                )

        normalized_stats["op_std"] = np.where(
            normalized_stats["op_std"] < 1e-8,
            1.0,
            normalized_stats["op_std"],
        )
        normalized_stats["sensor_stds"] = np.where(
            normalized_stats["sensor_stds"] < 1e-8,
            1.0,
            normalized_stats["sensor_stds"],
        )
        return normalized_stats

    @staticmethod
    def _normalize_operating_settings(op_values, train_stats):
        return (op_values - train_stats["op_mean"]) / train_stats["op_std"]

    @classmethod
    def _assign_regimes(cls, op_values, train_stats):
        standardized_ops = cls._normalize_operating_settings(op_values, train_stats)
        centers = train_stats["regime_centers"]
        distances = ((standardized_ops[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(np.int64)

    @staticmethod
    def _normalize_sensors_by_regime(sensor_values, regime_ids, train_stats):
        normalized = np.empty_like(sensor_values, dtype=np.float32)
        for regime in range(train_stats["n_regimes"]):
            mask = regime_ids == regime
            if np.any(mask):
                normalized[mask] = (
                    sensor_values[mask] - train_stats["sensor_means"][regime]
                ) / train_stats["sensor_stds"][regime]
        return normalized

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def get_dataloaders(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    context_length: int = 36,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    train_ds = CMAPSSDataset(
        train_csv,
        context_length=context_length,
        stride=1,
        normalize=True,
        train_stats=None,
    )

    val_ds = CMAPSSDataset(
        val_csv,
        context_length=context_length,
        stride=1,
        normalize=True,
        train_stats=train_ds.train_stats,
    )

    test_ds = CMAPSSDataset(
        test_csv,
        context_length=context_length,
        stride=1,
        normalize=True,
        train_stats=train_ds.train_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, train_ds.train_stats