"""Data preparation pipeline for OceanEmbed.

This module provides data preprocessing for the OceanEmbed framework:
1. Ingests or generates surface features (SST, SSS, SLA, Wind-U, Wind-V, Lat, Lon, Time).
2. Prepares target subsurface temperature profiles at the 15 standardized OceanEmbed depth levels.
3. Provides a deterministic --demo-sample mode to generate test arrays without external downloads.
4. Performs integrity checks (bounds, dimensions, NaN/Inf) before saving.
5. Saves the processed arrays to data/processed/<experiment>.npz for the training pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Standard OceanEmbed 15 target depths in metres (Single Source of Truth)
STANDARD_DEPTHS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare dataset for OceanEmbed training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bay_of_bengal.yaml",
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--demo-sample",
        action="store_true",
        help="Generate a deterministic synthetic dataset fixture for testing without downloading real data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/bay_of_bengal.npz",
        help="Destination path for the processed NPZ file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def generate_demo_data(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Generate a deterministic synthetic demonstration dataset for the configured domain.

    NOTE: This generates a smooth, physically-plausible mathematical approximation
    representing typical tropical ocean thermocline dynamics for development and smoke-testing.
    IT DOES NOT REPRESENT REAL OCEAN SATELLITE OR IN-SITU OBSERVATIONS.
    DO NOT USE FOR SCIENTIFIC VALIDATION.
    """
    region = cfg.get("region", {})
    lat_min = float(region.get("lat_min", 5.0))
    lat_max = float(region.get("lat_max", 23.0))
    lon_min = float(region.get("lon_min", 80.0))
    lon_max = float(region.get("lon_max", 100.0))

    demo_cfg = cfg.get("demo_sample", {})
    n_samples = int(demo_cfg.get("n_samples", 2000))
    seed = int(demo_cfg.get("seed", cfg.get("training", {}).get("seed", 42)))

    model_cfg = cfg.get("model", {})
    feature_names = model_cfg.get(
        "input_features",
        ["sst", "sss", "sla", "wind_u", "wind_v", "lat", "lon", "time"],
    )
    target_depths = np.array(model_cfg.get("target_depths", STANDARD_DEPTHS), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # 1. Geographic & temporal distribution across the specified domain
    lat = rng.uniform(lat_min, lat_max, size=n_samples).astype(np.float32)
    lon = rng.uniform(lon_min, lon_max, size=n_samples).astype(np.float32)
    # Day-of-year normalized in [-1, 1] for seasonal variation
    time_day = rng.uniform(1, 90, size=n_samples).astype(np.float32)
    time_norm = np.sin(2 * np.pi * time_day / 365.25).astype(np.float32)

    # 2. Surface ocean variables with realistic spatial gradients for Bay of Bengal
    # SST: 26°C to 30.5°C with meridional gradient (cooler north, warmer south)
    sst_base = 29.5 - 0.15 * (lat - lat_min) + 0.5 * time_norm
    sst = sst_base + rng.normal(0, 0.25, size=n_samples).astype(np.float32)

    # SSS: Sea Surface Salinity (31 PSU north river discharge, 34.5 PSU south)
    sss_base = 31.5 + 0.12 * (lat_max - lat)
    sss = sss_base + rng.normal(0, 0.2, size=n_samples).astype(np.float32)

    # SLA: Sea Level Anomaly in metres (-0.25m to +0.25m)
    sla = (0.12 * np.sin(lat / 3.0) * np.cos(lon / 4.0) + rng.normal(0, 0.03, size=n_samples)).astype(np.float32)

    # Wind-U and Wind-V (NE monsoon winds: -3 m/s to 4 m/s)
    wind_u = (-1.5 + 0.8 * np.cos(lat / 4.0) + rng.normal(0, 0.5, size=n_samples)).astype(np.float32)
    wind_v = (-2.0 + 0.5 * np.sin(lon / 5.0) + rng.normal(0, 0.5, size=n_samples)).astype(np.float32)

    # Assemble feature matrix X according to feature_names
    feature_dict = {
        "sst": sst,
        "sss": sss,
        "sla": sla,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "lat": lat,
        "lon": lon,
        "time": time_norm,
    }

    feature_cols = []
    for feat in feature_names:
        if feat in feature_dict:
            feature_cols.append(feature_dict[feat])
        else:
            raise KeyError(f"Feature '{feat}' configured in input_features is not supported by data generator.")
    X = np.column_stack(feature_cols).astype(np.float32)

    # 3. Physically-plausible subsurface temperature profile:
    # Upper mixed layer (0-50m): close to SST, slightly warm
    # Thermocline (50-200m): steep exponential decay
    # Deep ocean (300-1000m): asymptotic approach to deep bottom temperature (~5-7°C)
    # SLA positively modulates thermocline depth (downwelling warm anomaly)
    # Latitudes further north have slightly shallower thermoclines
    z = target_depths[np.newaxis, :]  # shape: (1, 15)
    t_deep = 5.5  # deep abyssal temperature in °C

    # Thermocline scale depth modulated by SLA and wind mixing
    thermocline_depth = 80.0 + 60.0 * sla[:, np.newaxis] - 0.8 * (lat[:, np.newaxis] - lat_min)
    thermocline_depth = np.clip(thermocline_depth, 40.0, 130.0)

    # Exponential thermocline transition: T(z) = T_deep + (SST - T_deep) * exp(- (z / z0)^1.2)
    temp_profile = t_deep + (sst[:, np.newaxis] - t_deep) * np.exp(-((z / thermocline_depth) ** 1.15))

    # Add small depth-dependent noise
    noise_amplitude = 0.25 * np.exp(-z / 250.0)
    noise = rng.normal(0, 1, size=(n_samples, len(target_depths))).astype(np.float32) * noise_amplitude
    y = np.clip(temp_profile + noise, 3.0, 33.0).astype(np.float32)

    return X, y, feature_names, target_depths


def load_real_data(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Ingest and preprocess real satellite and GLORYS reanalysis datasets.

    NOTE: Full operational satellite and reanalysis ingestion will be integrated in subsequent phases.
    """
    raise NotImplementedError(
        "Real satellite/GLORYS ingestion is not yet available in local workspace. "
        "Use '--demo-sample' to generate a verified development dataset."
    )


def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_depths: np.ndarray,
    cfg: dict[str, Any],
) -> None:
    """Perform validation checks on the processed dataset."""
    # 1. Samples consistency
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")

    if X.shape[0] == 0:
        raise ValueError("Processed dataset contains 0 samples.")

    # 2. Features count
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: X has {X.shape[1]} columns, feature_names has {len(feature_names)}.")

    # 3. Target depths count & values
    if y.shape[1] != 15:
        raise ValueError(f"Target dimension mismatch: expected 15 depths, got {y.shape[1]}.")

    if len(target_depths) != 15:
        raise ValueError(f"Target depths array must have 15 elements, got {len(target_depths)}.")

    expected_depths = np.array(STANDARD_DEPTHS, dtype=np.float32)
    if not np.allclose(target_depths, expected_depths, atol=1e-3):
        raise ValueError(f"Target depths do not match official specification: {target_depths.tolist()} vs {STANDARD_DEPTHS}")

    # 4. Check for NaN / Inf
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or Inf values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("y contains NaN or Inf values.")

    # 5. Geographic domain boundaries
    region = cfg.get("region", {})
    lat_min = float(region.get("lat_min", 5.0))
    lat_max = float(region.get("lat_max", 23.0))
    lon_min = float(region.get("lon_min", 80.0))
    lon_max = float(region.get("lon_max", 100.0))

    if "lat" in feature_names:
        lat_idx = feature_names.index("lat")
        if np.min(X[:, lat_idx]) < lat_min - 1e-3 or np.max(X[:, lat_idx]) > lat_max + 1e-3:
            raise ValueError(f"Latitude values exceed bounds [{lat_min}, {lat_max}]: [{np.min(X[:, lat_idx])}, {np.max(X[:, lat_idx])}]")

    if "lon" in feature_names:
        lon_idx = feature_names.index("lon")
        if np.min(X[:, lon_idx]) < lon_min - 1e-3 or np.max(X[:, lon_idx]) > lon_max + 1e-3:
            raise ValueError(f"Longitude values exceed bounds [{lon_min}, {lon_max}]: [{np.min(X[:, lon_idx])}, {np.max(X[:, lon_idx])}]")


def save_processed_dataset(
    output_path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_depths: np.ndarray,
    is_demo: bool = False,
) -> None:
    """Save processed arrays into NPZ format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        X=X,
        y=y,
        feature_names=np.array(feature_names, dtype=object),
        target_depths=target_depths,
        is_synthetic_demo=is_demo,
    )


def prepare_dataset(config_path: str, demo_sample: bool = False, output_path: str | None = None) -> None:
    """Main orchestration function to load, validate, and save dataset."""
    cfg = load_config(config_path)
    if output_path is None:
        output_path = "data/processed/bay_of_bengal.npz"

    print(f"[PREPARE] Configuration loaded from: {config_path}")
    print(f"[PREPARE] Target output file: {output_path}")

    if demo_sample:
        print("[PREPARE] Mode: Deterministic Synthetic DEMO Fixture (Phase 1)")
        print("  NOTICE: Generated data is synthetic for pipeline validation and baseline testing.")
        print("  DO NOT claim or use as real ocean observations.")
        X, y, feature_names, target_depths = generate_demo_data(cfg)
    else:
        print("[PREPARE] Mode: Real Data Ingestion")
        X, y, feature_names, target_depths = load_real_data(cfg)

    # Validate dataset before persisting
    validate_dataset(X, y, feature_names, target_depths, cfg)
    print(f"[PREPARE] Validation passed: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} depths.")

    # Save
    save_processed_dataset(output_path, X, y, feature_names, target_depths, is_demo=demo_sample)
    print(f"[PREPARE] Successfully wrote processed dataset to: {output_path}")

    # Summary report
    print("\n--- Processed Dataset Summary ---")
    print(f"X shape:          {X.shape} (float32)")
    print(f"y shape:          {y.shape} (float32)")
    print(f"Features ({len(feature_names)}):     {feature_names}")
    print(f"Target Depths (15): {target_depths.tolist()} m")
    lat_idx = feature_names.index("lat") if "lat" in feature_names else None
    lon_idx = feature_names.index("lon") if "lon" in feature_names else None
    if lat_idx is not None:
        print(f"Latitude range:   {X[:, lat_idx].min():.2f}N to {X[:, lat_idx].max():.2f}N")
    if lon_idx is not None:
        print(f"Longitude range:  {X[:, lon_idx].min():.2f}E to {X[:, lon_idx].max():.2f}E")
    print("---------------------------------\n")


def main() -> None:
    """CLI Entrypoint."""
    args = parse_args()
    prepare_dataset(args.config, demo_sample=args.demo_sample, output_path=args.output)


if __name__ == "__main__":
    main()
