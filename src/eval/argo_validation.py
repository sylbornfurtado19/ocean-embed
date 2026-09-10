"""Independent ARGO validation for OceanEmbed.

This module validates the model against sparse in-situ Argo float profiles and
reports depth-wise RMSE/MAE/bias metrics with sample counts. The key detail is
that ARGO depths are irregular and sparse, so predictions are interpolated from
our 15 target depths onto the observed ARGO depths before comparing values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml


def _load_config(config: dict | str) -> dict:
    if isinstance(config, dict):
        return config
    with open(config, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_argo_profiles(config: dict | str) -> pd.DataFrame:
    """Load all ARGO NetCDF profiles in data/raw/argo and flatten them to a row-per-observation table."""
    cfg = _load_config(config)
    argo_dir = Path("data/raw/argo")
    files = sorted(argo_dir.glob("*.nc")) + sorted(argo_dir.glob("*.nc4"))
    if not files:
        raise FileNotFoundError("No ARGO NetCDF files found in data/raw/argo. Run download_argo first.")

    ds = xr.open_mfdataset([str(path) for path in files], combine="by_coords", decode_times=True)
    frames: list[pd.DataFrame] = []

    # Normalize likely xarray field names from ARGO exports.
    lat_name = next((name for name in ("latitude", "LATITUDE", "lat") if name in ds.coords or name in ds), None)
    lon_name = next((name for name in ("longitude", "LONGITUDE", "lon") if name in ds.coords or name in ds), None)
    time_name = next((name for name in ("time", "TIME", "datetime") if name in ds.coords or name in ds), None)
    depth_name = next((name for name in ("pressure", "PRES", "depth", "DEPTH") if name in ds.coords or name in ds), None)
    temp_name = next((name for name in ("temperature", "TEMP", "temp") if name in ds.coords or name in ds), None)

    if lat_name is None or lon_name is None or depth_name is None or temp_name is None:
        raise KeyError("ARGO dataset does not expose the expected lat/lon/depth/temperature variables.")

    for profile_id in range(ds.sizes.get("profile", 1)):
        try:
            profile = ds.isel(profile=profile_id)
        except Exception:
            profile = ds

        lat = float(profile[lat_name].values.reshape(-1)[0]) if np.asarray(profile[lat_name].values).size else np.nan
        lon = float(profile[lon_name].values.reshape(-1)[0]) if np.asarray(profile[lon_name].values).size else np.nan
        timestamp = profile[time_name].values if time_name in profile.coords or time_name in profile else np.nan
        if isinstance(timestamp, np.ndarray):
            timestamp = timestamp.reshape(-1)[0]

        depth_values = np.asarray(profile[depth_name].values).reshape(-1)
        temp_values = np.asarray(profile[temp_name].values).reshape(-1)

        for depth_value, temp_value in zip(depth_values, temp_values):
            if not np.isfinite(float(depth_value)) or not np.isfinite(float(temp_value)):
                continue
            frames.append(
                {
                    "profile_id": profile_id,
                    "lat": float(lat),
                    "lon": float(lon),
                    "date": pd.to_datetime(timestamp).date() if not pd.isna(timestamp) else pd.NaT,
                    "depth": float(depth_value),
                    "temperature": float(temp_value),
                }
            )

    if not frames:
        raise ValueError("No valid ARGO temperature observations were extracted from the downloaded files.")

    df = pd.DataFrame(frames)
    df = df[(df["lat"].between(cfg["region"]["lat_min"], cfg["region"]["lat_max"])) & (df["lon"].between(cfg["region"]["lon_min"], cfg["region"]["lon_max"]))]
    return df.reset_index(drop=True)


def collocate_with_predictions(argo_df: pd.DataFrame, model: Any, surface_data: dict) -> pd.DataFrame:
    """For each ARGO sample, find the nearest surface grid point and interpolate the predicted profile onto the measured depth.

    This is intentionally conservative: the trained model outputs temperatures at our 15 target depths,
    then a 1D interpolation is used to evaluate the prediction at the actual ARGO pressure/depth value.
    """
    if "X" not in surface_data or "feature_names" not in surface_data:
        raise KeyError("surface_data requires 'X' and 'feature_names' arrays.")

    model.eval()
    X = np.asarray(surface_data["X"], dtype=np.float32)
    feature_names = list(surface_data["feature_names"])
    target_depths = np.asarray(surface_data.get("depth_levels", [0.0, 10.0, 20.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]), dtype=np.float32)
    lat_idx = feature_names.index("lat")
    lon_idx = feature_names.index("lon")

    rows: list[dict[str, Any]] = []
    for _, row in argo_df.iterrows():
        delta_lat = np.abs(X[:, lat_idx] - float(row["lat"]))
        delta_lon = np.abs(X[:, lon_idx] - float(row["lon"]))
        nearest_idx = np.argmin(delta_lat + delta_lon)
        x_sample = torch.tensor(X[nearest_idx][None, :], dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, "forward") and isinstance(model.forward(x_sample), tuple):
                pred_mean, _ = model(x_sample)
                pred_values = pred_mean.cpu().numpy()[0]
            else:
                pred_values = model(x_sample).cpu().numpy()[0]

        predicted_depth_profile = pred_values
        # Interpolate to the actual ARGO depth level. This keeps the comparison valid even
        # when ARGO depth values are not exactly aligned to the 15 target depths in the model.
        interpolated = float(np.interp(float(row["depth"]), target_depths, predicted_depth_profile))
        row_out = row.to_dict()
        row_out["predicted_temperature"] = interpolated
        row_out["error"] = interpolated - float(row["temperature"])
        rows.append(row_out)

    return pd.DataFrame(rows)


def compute_metrics(collocated_df: pd.DataFrame) -> dict:
    """Compute RMSE, MAE, bias, and sample count per depth, along with an overall summary."""
    if collocated_df.empty:
        raise ValueError("No collocated ARGO predictions available for evaluation.")

    metrics: dict[str, Any] = {}
    per_depth: list[dict[str, Any]] = []
    grouped = collocated_df.groupby("depth")
    for depth, group in grouped:
        y_true = group["temperature"].to_numpy(dtype=np.float64)
        y_pred = group["predicted_temperature"].to_numpy(dtype=np.float64)
        err = y_pred - y_true
        rmse = float(np.sqrt(np.mean(np.square(err))))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        n = int(len(group))
        per_depth.append({"depth": float(depth), "rmse": rmse, "mae": mae, "bias": bias, "n_samples": n})

    metrics["per_depth"] = pd.DataFrame(per_depth).sort_values("depth").reset_index(drop=True)
    err = collocated_df["predicted_temperature"].to_numpy(dtype=np.float64) - collocated_df["temperature"].to_numpy(dtype=np.float64)
    metrics["overall_rmse"] = float(np.sqrt(np.mean(np.square(err))))
    metrics["overall_mae"] = float(np.mean(np.abs(err)))
    metrics["overall_bias"] = float(np.mean(err))
    metrics["n_samples"] = int(len(collocated_df))
    return metrics


def run_argo_validation(config: dict | str, checkpoint_path: str = "checkpoints/v0_baseline.pt") -> dict:
    """Run the full ARGO validation flow for the configured project and return summary metrics."""
    cfg = _load_config(config)
    processed_path = Path("data/processed/bay_of_bengal.npz")
    if not processed_path.exists():
        raise FileNotFoundError("Processed arrays are missing. Run src/data/prepare.py before validation.")

    data = np.load(processed_path)
    surface_data = {
        "X": data["X"],
        "feature_names": data["feature_names"].tolist(),
        "depth_levels": cfg["model"]["target_depths"],
    }

    argo_df = load_argo_profiles(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = torch.load(checkpoint_path, map_location="cpu") if isinstance(checkpoint, torch.nn.Module) else None

    if model is None:
        from src.models.v0_baseline import OceanBaselineV0
        model = OceanBaselineV0(input_dim=surface_data["X"].shape[1], hidden_dim=cfg["model"].get("hidden_dim", 128), output_dim=len(cfg["model"]["target_depths"]), dropout=cfg["model"].get("dropout", 0.1))
        model.load_state_dict(checkpoint["model_state"])

    collocated_df = collocate_with_predictions(argo_df, model, surface_data)
    metrics = compute_metrics(collocated_df)
    print("Depth | RMSE | MAE | bias | n_samples")
    print("----- | ---- | --- | ---- | ---------")
    for _, row in metrics["per_depth"].iterrows():
        print(f"{row['depth']:>5.0f} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['bias']:.4f} | {int(row['n_samples'])}")
    print(f"Overall RMSE={metrics['overall_rmse']:.4f}, MAE={metrics['overall_mae']:.4f}, bias={metrics['overall_bias']:.4f}, n={metrics['n_samples']}")
    return metrics


if __name__ == "__main__":
    run_argo_validation("configs/bay_of_bengal.yaml")
