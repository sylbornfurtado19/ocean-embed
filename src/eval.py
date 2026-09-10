"""Evaluation entry point for OceanEmbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.models.v0_baseline import OceanBaselineV0


def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration and runtime arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the OceanEmbed baseline model.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def load_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def evaluate_model(config_path: str) -> None:
    """Evaluate the trained checkpoint on a held-out validation split and print RMSE by depth."""
    processed_path = Path("data/processed/bay_of_bengal.npz")
    checkpoint_path = Path("checkpoints/v0_baseline.pt")
    if not processed_path.exists():
        raise FileNotFoundError("No processed dataset found. Run src/data/prepare.py before evaluation.")
    if not checkpoint_path.exists():
        raise FileNotFoundError("No trained checkpoint found. Run src/train.py before evaluation.")

    data = np.load(processed_path)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()

    lat_idx = feature_names.index("lat")
    lon_idx = feature_names.index("lon")
    lat_mid = np.median(X[:, lat_idx])
    lon_mid = np.median(X[:, lon_idx])
    val_mask = (X[:, lat_idx] >= lat_mid) & (X[:, lon_idx] >= lon_mid)
    X_val = X[val_mask]
    y_val = y[val_mask]

    checkpoint = load_checkpoint(str(checkpoint_path))
    model = OceanBaselineV0(input_dim=X_val.shape[1], hidden_dim=128, output_dim=y_val.shape[1], dropout=0.1)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

    rmse_by_depth = np.sqrt(np.mean((preds - y_val) ** 2, axis=0))
    print("Depth level | RMSE")
    print("------------------")
    depth_levels = np.array([0.0, 10.0, 20.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
    for idx, depth in enumerate(depth_levels[: y_val.shape[1]]):
        print(f"{depth:>7.1f} m | {rmse_by_depth[idx]:.4f}")

    # If ARGO netCDF exists, it is loaded here for future validation alignment.
    argo_path = Path("data/raw/argo/argo_profiles.nc")
    if argo_path.exists():
        ds = xr.open_dataset(argo_path)
        print(f"[EVAL] ARGO file available: {argo_path} with variables: {list(ds.data_vars)[:10]}")


def main() -> None:
    """Entry point for model evaluation."""
    args = parse_args()
    evaluate_model(args.config)


if __name__ == "__main__":
    main()
