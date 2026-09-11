"""Synthetic spatiotemporal dataset generator for OceanEmbed Phase 3 architecture.

Generates structured 5D tensor batches (N, T=31, C=5, H=32, W=32) representing:
- 31 consecutive days of daily surface variables (day -15 to day +15 around prediction date)
- 5 surface physical channels: SST, SSS, SLA, Wind-U, Wind-V
- 32x32 local spatial neighborhoods (approx 0.25 deg grid spacing across an 8x8 deg patch)
- Target: 15-depth temperature profile at the central location and central date (day 0)
- Explicit deterministic synthetic demo climatology prior at the central location

CRITICAL SCIENTIFIC NOTICE:
This dataset is a deterministic mathematical simulation created solely for software
architecture verification and model engineering. It does NOT represent real satellite
or in-situ ocean observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

STANDARD_DEPTHS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
CHANNELS = ["sst", "sss", "sla", "wind_u", "wind_v"]


def compute_demo_climatology_profile(lat: float, lon: float, target_depths: np.ndarray) -> np.ndarray:
    """Compute a deterministic reference climatology profile based on latitude and longitude in Bay of Bengal.

    This serves as the development climatology prior:
    Predicted T(z) = Climatology(z) + Predicted Anomaly(z).
    """
    z = np.asarray(target_depths, dtype=np.float32)
    # Mean annual surface temperature for latitude
    t_surf = 28.5 - 0.12 * (lat - 10.0)
    t_deep = 5.5
    z_clim = 85.0 - 0.5 * (lat - 10.0) + 0.2 * (lon - 85.0)
    z_clim = np.clip(z_clim, 50.0, 120.0)
    clim_profile = t_deep + (t_surf - t_deep) * np.exp(-((z / z_clim) ** 1.15))
    return clim_profile.astype(np.float32)


def generate_spatiotemporal_sample(
    center_lat: float,
    center_lon: float,
    center_day: float,
    target_depths: np.ndarray,
    rng: np.random.Generator,
    T: int = 31,
    H: int = 32,
    W: int = 32,
    grid_res: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single spatiotemporal patch and corresponding central subsurface temperature target."""
    # Coordinate offsets for 32x32 patch centered at (center_lat, center_lon)
    half_h = (H - 1) / 2.0 * grid_res
    half_w = (W - 1) / 2.0 * grid_res
    lats = np.linspace(center_lat - half_h, center_lat + half_h, H, dtype=np.float32)
    lons = np.linspace(center_lon - half_w, center_lon + half_w, W, dtype=np.float32)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")  # (H, W)

    # Time steps: -15 to +15 days
    time_offsets = np.arange(-15, 16, dtype=np.float32)  # shape (31,)
    days = center_day + time_offsets

    # Seasonal variation across 31-day window
    time_norm = np.sin(2.0 * np.pi * days / 365.25)[:, np.newaxis, np.newaxis]  # (T, 1, 1)

    # Base surface fields with spatial gradients across patch
    # 1. SST: (T, H, W)
    sst_base = 29.5 - 0.15 * (grid_lat - 5.0) + 0.5 * time_norm
    # Small synoptic temporal fluctuation
    synoptic_sst = 0.2 * np.cos(2.0 * np.pi * time_offsets / 10.0)[:, np.newaxis, np.newaxis]
    sst_noise = rng.normal(0, 0.05, size=(T, H, W)).astype(np.float32)
    sst = (sst_base + synoptic_sst + sst_noise).astype(np.float32)

    # 2. SSS: (T, H, W)
    sss_base = 31.5 + 0.12 * (23.0 - grid_lat)
    sss_noise = rng.normal(0, 0.04, size=(T, H, W)).astype(np.float32)
    sss = (sss_base + sss_noise).astype(np.float32)

    # 3. SLA: (T, H, W) in metres
    # Mesoscale eddy structure in patch
    eddy = 0.10 * np.sin((grid_lat - center_lat) * 4.0) * np.cos((grid_lon - center_lon) * 4.0)
    sla_time = 0.05 * np.sin(2.0 * np.pi * time_offsets / 20.0)[:, np.newaxis, np.newaxis]
    sla_noise = rng.normal(0, 0.01, size=(T, H, W)).astype(np.float32)
    sla = (eddy[np.newaxis, :, :] + sla_time + sla_noise).astype(np.float32)

    # 4. Wind-U & Wind-V: (T, H, W)
    wind_u = (-1.5 + 0.5 * np.cos(grid_lat / 4.0) + 0.4 * np.sin(time_offsets / 5.0)[:, np.newaxis, np.newaxis] + rng.normal(0, 0.1, size=(T, H, W))).astype(np.float32)
    wind_v = (-2.0 + 0.4 * np.sin(grid_lon / 5.0) + 0.4 * np.cos(time_offsets / 5.0)[:, np.newaxis, np.newaxis] + rng.normal(0, 0.1, size=(T, H, W))).astype(np.float32)

    # Stack into (T, C=5, H, W)
    X_patch = np.stack([sst, sss, sla, wind_u, wind_v], axis=1)  # (31, 5, 32, 32)

    # Target: Subsurface temperature profile at center coordinates and day 0 (index 15)
    c_sst = float(sst[15, H // 2, W // 2])
    c_sla = float(sla[15, H // 2, W // 2])

    z = target_depths[np.newaxis, :]  # (1, 15)
    t_deep = 5.5
    thermocline_depth = 80.0 + 60.0 * c_sla - 0.8 * (center_lat - 5.0)
    thermocline_depth = np.clip(thermocline_depth, 40.0, 130.0)

    temp_profile = t_deep + (c_sst - t_deep) * np.exp(-((z / thermocline_depth) ** 1.15))
    noise = rng.normal(0, 0.15, size=(1, len(target_depths))).astype(np.float32) * np.exp(-z / 300.0)
    y_target = np.clip(temp_profile + noise, 3.0, 33.0)[0].astype(np.float32)

    # Compute reference climatology at central point
    climatology = compute_demo_climatology_profile(center_lat, center_lon, target_depths)

    return X_patch, y_target, climatology


def generate_spatiotemporal_dataset(
    n_samples: int = 1200,
    seed: int = 42,
    output_path: str = "data/processed/bay_of_bengal_spatiotemporal.npz",
) -> Path:
    """Generate and save the deterministic spatiotemporal development dataset."""
    rng = np.random.default_rng(seed)
    target_depths = np.array(STANDARD_DEPTHS, dtype=np.float32)

    # Bounding box for center points (leaving buffer for 32x32 patch @ 0.25 deg = ~4 deg radius)
    lat_min, lat_max = 9.0, 19.0
    lon_min, lon_max = 84.0, 96.0

    print(f"[GEN] Generating {n_samples} spatiotemporal patches (T=31, C=5, H=32, W=32)...")
    center_lats = rng.uniform(lat_min, lat_max, size=n_samples).astype(np.float32)
    center_lons = rng.uniform(lon_min, lon_max, size=n_samples).astype(np.float32)
    center_days = rng.uniform(20, 80, size=n_samples).astype(np.float32)

    X_list = []
    y_list = []
    clim_list = []
    coords_list = []

    for i in range(n_samples):
        lat = float(center_lats[i])
        lon = float(center_lons[i])
        day = float(center_days[i])
        x_p, y_t, clim = generate_spatiotemporal_sample(lat, lon, day, target_depths, rng)
        X_list.append(x_p)
        y_list.append(y_t)
        clim_list.append(clim)
        coords_list.append([lat, lon, day])

    X = np.stack(X_list, axis=0)        # shape: (N, 31, 5, 32, 32)
    y = np.stack(y_list, axis=0)        # shape: (N, 15)
    climatology = np.stack(clim_list, axis=0)  # shape: (N, 15)
    coords = np.array(coords_list, dtype=np.float32)  # shape: (N, 3)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        X=X,
        y=y,
        climatology=climatology,
        coords=coords,
        channels=np.array(CHANNELS, dtype=object),
        target_depths=target_depths,
        is_synthetic_demo=True,
    )
    print(f"[GEN] Saved spatiotemporal dataset to: {out_file}")
    print(f"      X shape: {X.shape} (float32)")
    print(f"      y shape: {y.shape} (float32)")
    print(f"      Climatology shape: {climatology.shape}")
    print(f"      Channels ({len(CHANNELS)}): {CHANNELS}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic spatiotemporal dataset for OceanEmbed.")
    parser.add_argument("--samples", type=int, default=800, help="Number of spatiotemporal patches to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--output", type=str, default="data/processed/bay_of_bengal_spatiotemporal.npz", help="Output path.")
    args = parser.parse_args()
    generate_spatiotemporal_dataset(n_samples=args.samples, seed=args.seed, output_path=args.output)


if __name__ == "__main__":
    main()
