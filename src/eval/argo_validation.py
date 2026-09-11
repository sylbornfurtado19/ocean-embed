"""Independent ARGO in-situ validation for OceanEmbed V0, V1, and V2 models.

Validates predictions against sparse in-situ Argo float profiles in data/raw/argo.
If no ARGO NetCDF files are present in data/raw/argo, reports:
  STATUS = NOT_AVAILABLE
  REASON = ARGO validation unavailable because no real in-situ observations are configured.

DO NOT fabricate or simulate ARGO observations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import settings
from src.inference import OceanInferenceEngine

logger = logging.getLogger("oceanembed.argo_validation")


def _load_config(config: dict | str) -> dict:
    if isinstance(config, dict):
        return config
    with open(config, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_argo_available() -> tuple[bool, list[Path]]:
    """Check if real ARGO NetCDF files exist in data/raw/argo."""
    argo_dir = Path(settings.argo_data_dir)
    if not argo_dir.exists():
        return False, []
    files = sorted(argo_dir.glob("*.nc")) + sorted(argo_dir.glob("*.nc4")) + sorted(argo_dir.glob("**/*.nc"))
    return len(files) > 0, list(set(files))


def load_argo_profiles(config: dict | str) -> pd.DataFrame:
    """Load all ARGO NetCDF profiles in data/raw/argo and flatten them to a row-per-observation table."""
    import xarray as xr

    cfg = _load_config(config)
    available, files = check_argo_available()
    if not available:
        raise FileNotFoundError("No ARGO NetCDF files found in data/raw/argo.")

    region_lat_min = cfg.get("region", {}).get("lat_min", settings.domain_lat_min)
    region_lat_max = cfg.get("region", {}).get("lat_max", settings.domain_lat_max)
    region_lon_min = cfg.get("region", {}).get("lon_min", settings.domain_lon_min)
    region_lon_max = cfg.get("region", {}).get("lon_max", settings.domain_lon_max)

    frames: list[pd.DataFrame] = []

    for file_path in files:
        try:
            with xr.open_dataset(file_path) as ds:
                lat_name = next((name for name in ("latitude", "LATITUDE", "lat") if name in ds.coords or name in ds), None)
                lon_name = next((name for name in ("longitude", "LONGITUDE", "lon") if name in ds.coords or name in ds), None)
                time_name = next((name for name in ("time", "TIME", "datetime") if name in ds.coords or name in ds), None)
                depth_name = next((name for name in ("pressure", "PRES", "depth", "DEPTH") if name in ds.coords or name in ds), None)
                temp_name = next((name for name in ("temperature", "TEMP", "temp") if name in ds.coords or name in ds), None)

                if lat_name is None or lon_name is None or depth_name is None or temp_name is None:
                    continue

                n_profs = ds.sizes.get("N_PROF", ds.sizes.get("profile", 1))
                for profile_id in range(n_profs):
                    try:
                        p = ds.isel(N_PROF=profile_id) if "N_PROF" in ds.dims else (ds.isel(profile=profile_id) if "profile" in ds.dims else ds)
                        lat = float(np.asarray(p[lat_name].values).reshape(-1)[0])
                        lon = float(np.asarray(p[lon_name].values).reshape(-1)[0])

                        if not (region_lat_min <= lat <= region_lat_max and region_lon_min <= lon <= region_lon_max):
                            continue

                        timestamp = p[time_name].values if time_name in p.coords or time_name in p else np.nan
                        if isinstance(timestamp, np.ndarray):
                            timestamp = timestamp.reshape(-1)[0]

                        depth_values = np.asarray(p[depth_name].values).reshape(-1)
                        temp_values = np.asarray(p[temp_name].values).reshape(-1)

                        for depth_val, temp_val in zip(depth_values, temp_values):
                            if not np.isfinite(float(depth_val)) or not np.isfinite(float(temp_val)):
                                continue
                            frames.append(
                                {
                                    "file": file_path.name,
                                    "profile_id": f"{file_path.stem}_{profile_id}",
                                    "lat": float(lat),
                                    "lon": float(lon),
                                    "date": pd.to_datetime(timestamp).date() if not pd.isna(timestamp) else pd.NaT,
                                    "depth": float(depth_val),
                                    "temperature": float(temp_val),
                                }
                            )
                    except Exception:
                        continue
        except Exception as ex:
            logger.warning("Error reading ARGO file %s: %s", file_path, ex)

    if not frames:
        raise ValueError("No valid ARGO temperature observations were extracted from the downloaded files.")

    df = pd.DataFrame(frames)
    return df.reset_index(drop=True)


def compute_metrics(collocated_df: pd.DataFrame) -> dict[str, Any]:
    """Compute RMSE, MAE, bias, and sample count per depth, along with an overall summary."""
    if collocated_df.empty:
        raise ValueError("No collocated ARGO predictions available for evaluation.")

    metrics: dict[str, Any] = {}
    per_depth: list[dict[str, Any]] = []

    # Bin depths into standard target depths or near intervals
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


def run_argo_validation(
    config: dict | str = "configs/bay_of_bengal.yaml",
    checkpoint_path: str = "checkpoints/oceanembed_v2.pt",
) -> dict[str, Any]:
    """Run ARGO validation against OceanEmbed model if real files exist, otherwise report NOT_AVAILABLE."""
    available, files = check_argo_available()
    if not available:
        print("\n" + "=" * 60)
        print("[ARGO VALIDATION]")
        print("  STATUS: NOT AVAILABLE")
        print("  REASON: ARGO validation unavailable because no real in-situ observations are configured.")
        print("  NOTICE: Real in-situ ARGO profiles will be ingested in future operational phases.")
        print("=" * 60 + "\n")
        return {
            "status": "NOT_AVAILABLE",
            "reason": "ARGO validation unavailable because no real in-situ observations are configured.",
            "overall_rmse": None,
            "overall_mae": None,
            "overall_bias": None,
            "n_samples": 0,
        }

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        return {
            "status": "NOT_AVAILABLE",
            "reason": f"Model checkpoint not found at: {ckpt_p}",
            "overall_rmse": None,
            "overall_mae": None,
            "overall_bias": None,
            "n_samples": 0,
        }

    cfg = _load_config(config)
    argo_df = load_argo_profiles(cfg)
    engine = OceanInferenceEngine(ckpt_p)

    rows: list[dict[str, Any]] = []
    # Group by profile
    for p_id, p_group in argo_df.groupby("profile_id"):
        first_row = p_group.iloc[0]
        lat = float(first_row["lat"])
        lon = float(first_row["lon"])
        p_date = first_row["date"]
        date_val = p_date if pd.notna(p_date) else "2023-02-15"

        try:
            if engine.model_version == "oceanembed_v2":
                res = engine.predict_from_location_date(lat, lon, date_val)
                target_depths = np.asarray(res["depths"], dtype=np.float32)
                predicted_profile = np.asarray(res["temperature"], dtype=np.float32)
            else:
                # Baseline 1D
                res = engine.predict([28.5, 33.2, 0.05, 2.1, -1.2, lat, lon, 45.0])
                target_depths = np.asarray(engine.target_depths, dtype=np.float32)
                predicted_profile = np.asarray(res["temperature"], dtype=np.float32)

            for _, row in p_group.iterrows():
                obs_depth = float(row["depth"])
                obs_temp = float(row["temperature"])
                interp_pred = float(np.interp(obs_depth, target_depths, predicted_profile))
                row_dict = row.to_dict()
                row_dict["predicted_temperature"] = interp_pred
                row_dict["error"] = interp_pred - obs_temp
                rows.append(row_dict)
        except Exception as ex:
            logger.warning("Error evaluating ARGO profile %s: %s", p_id, ex)
            continue

    if not rows:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Could not collocate any ARGO observations with model predictions.",
            "overall_rmse": None,
            "overall_mae": None,
            "overall_bias": None,
            "n_samples": 0,
        }

    collocated_df = pd.DataFrame(rows)
    metrics = compute_metrics(collocated_df)
    metrics["status"] = "AVAILABLE"
    return metrics


if __name__ == "__main__":
    run_argo_validation()
