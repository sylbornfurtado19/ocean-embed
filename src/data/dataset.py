"""Spatiotemporal PyTorch dataset and data utilities for OceanEmbed Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class OceanSpatiotemporalDataset(Dataset):
    """PyTorch Dataset yielding 5D surface sequences and 15-depth targets."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        climatology: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.climatology = torch.tensor(climatology, dtype=torch.float32)
        self.coords = torch.tensor(coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.climatology[idx], self.coords[idx]


def load_spatiotemporal_data(
    path: str | Path = "data/processed/bay_of_bengal_spatiotemporal.npz",
) -> dict[str, Any]:
    """Load the spatiotemporal NPZ dataset with error checking."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spatiotemporal dataset not found at: {p}")
    data = np.load(p, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "climatology": data["climatology"],
        "coords": data["coords"],
        "channels": data["channels"].tolist(),
        "target_depths": data["target_depths"].tolist(),
    }


def spatiotemporal_spatial_holdout_split(
    data: dict[str, Any],
    holdout_fraction: float = 0.2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split spatiotemporal dataset using the spatial coordinates (lat, lon) to prevent leakage.

    Uses the northeastern spatial quadrant as held-out validation, matching the Phase 1 & 2 protocol.
    """
    coords = data["coords"]  # shape (N, 3) -> [lat, lon, day]
    lat_mid = np.median(coords[:, 0])
    lon_mid = np.median(coords[:, 1])

    val_mask = (coords[:, 0] >= lat_mid) & (coords[:, 1] >= lon_mid)
    if val_mask.mean() < 0.05:
        val_mask = np.zeros_like(val_mask, dtype=bool)
        val_count = max(1, int(len(coords) * holdout_fraction))
        val_mask[:val_count] = True

    train_data = {
        "X": data["X"][~val_mask],
        "y": data["y"][~val_mask],
        "climatology": data["climatology"][~val_mask],
        "coords": coords[~val_mask],
    }
    val_data = {
        "X": data["X"][val_mask],
        "y": data["y"][val_mask],
        "climatology": data["climatology"][val_mask],
        "coords": coords[val_mask],
    }
    return train_data, val_data


def compute_channel_norm_stats(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-channel mean and std strictly on training patches across (N, T, H, W)."""
    # X_train shape: (N, T, C=5, H, W)
    mean = np.mean(X_train, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)  # (1, 1, C, 1, 1)
    std = np.std(X_train, axis=(0, 1, 3, 4), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}


def apply_channel_norm(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize spatiotemporal patches using precomputed training channel statistics."""
    return (X - stats["mean"]) / stats["std"]
