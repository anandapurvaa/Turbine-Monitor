from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset for C-MAPSS FD001, windowed for PatchTST.
    Returns (X, y) where:
      X: (context_length, num_channels)
      y: scalar RUL (capped)
    """

    def __init__(
        self,
        csv_path: Path,
        context_length: int = 36,
        stride: int = 1,
        normalize: bool = True,
        train_stats: Optional[dict] = None,
    ):
        self.context_length = context_length
        self.stride = stride

        df = pd.read_csv(csv_path)

        # Select sensor columns only (sensor_1 .. sensor_21)
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        self.sensors = df[sensor_cols].values.astype(np.float32)  # (T, 21)
        self.rul = df["RUL_capped"].values.astype(np.float32)  # (T,)
        self.unit_ids = df["unit_nr"].values

        # Normalize sensors
        if normalize:
            if train_stats is None:
                # Compute stats from this dataset (assumed to be train)
                self.mean = self.sensors.mean(axis=0, keepdims=True)
                self.std = self.sensors.std(axis=0, keepdims=True) + 1e-8
            else:
                self.mean = train_stats["mean"]
                self.std = train_stats["std"]
            self.sensors = (self.sensors - self.mean) / self.std

        self.train_stats = {
            "mean": self.mean,
            "std": self.std,
        }

        # Build windows
        self.windows = []
        self.targets = []

        for unit in np.unique(self.unit_ids):
            mask = self.unit_ids == unit
            unit_sensors = self.sensors[mask]
            unit_rul = self.rul[mask]

            for i in range(0, len(unit_sensors) - context_length + 1, stride):
                x = unit_sensors[i : i + context_length]  # (context_length, 21)
                y = unit_rul[i + context_length - 1]  # RUL at last timestep
                self.windows.append(x)
                self.targets.append(y)

        self.windows = np.stack(self.windows)  # (N, context_length, 21)
        self.targets = np.array(self.targets, dtype=np.float32)  # (N,)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])  # (context_length, 21)
        y = torch.tensor(self.targets[idx])
        return x, y


def get_dataloaders(
    train_csv: Path,
    test_csv: Path,
    context_length: int = 36,
    batch_size: int = 64,
):
    train_ds = CMAPSSDataset(
        train_csv,
        context_length=context_length,
        stride=1,
        normalize=True,
        train_stats=None,
    )

    test_ds = CMAPSSDataset(
        test_csv,
        context_length=context_length,
        stride=1,
        normalize=True,
        train_stats=train_ds.train_stats,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, test_loader, train_ds.train_stats