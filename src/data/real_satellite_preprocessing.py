"""Real Satellite Data Preprocessing Adapter for OceanEmbed.

Converts multi-source real satellite/reanalysis NetCDF/Zarr rasters into the
canonical 5D spatiotemporal tensor required by OceanEmbed V2:
    Shape: (B, T=31, C=5, H=32, W=32)
    Channels: [SST, SSS, SLA, Wind-U, Wind-V]

Key features:
1. Flexible variable name resolution (CMEMS, GLORYS, OSTIA, CCMP, ERA5).
2. Spatial cropping to local neighborhood and bilinear interpolation to 32x32.
3. Coordinate ordering and longitude convention alignment ([-180, 180] vs [0, 360]).
4. Temporal alignment around central prediction date (31 days: -15 to +15).
5. Documented missing-value handling (spatial nearest-neighbor + temporal forward/backward fill).
6. Clear validation reporting if real source data is incompatible or corrupt.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("oceanembed.preprocessing")

# Target Channels in canonical OceanEmbed order
CANONICAL_CHANNELS = ["SST", "SSS", "SLA", "Wind-U", "Wind-V"]

# Standard variable name aliases across satellite providers
VARIABLE_ALIASES: dict[str, list[str]] = {
    "SST": ["thetao", "sst", "tos", "sea_surface_temperature", "analysed_sst", "temperature_surface"],
    "SSS": ["so", "sss", "sos", "sea_surface_salinity", "salinity_surface"],
    "SLA": ["zos", "sla", "adt", "ssh", "sea_surface_height_above_sea_level", "sea_surface_height_above_geoid"],
    "Wind-U": ["usi", "uas", "wind_u", "eastward_wind", "u10", "uwnd"],
    "Wind-V": ["vsi", "vas", "wind_v", "northward_wind", "v10", "vwnd"],
}


class DataIncompatibilityError(Exception):
    """Raised when real satellite source data cannot be reliably preprocessed into model format."""
    pass


class RealSatellitePreprocessor:
    """Preprocessor for converting real satellite datasets into model inputs."""

    def __init__(
        self,
        spatial_patch_size: int = 32,
        temporal_window_days: int = 31,
        grid_resolution_deg: float = 0.25,
        max_missing_ratio: float = 0.40,
    ) -> None:
        self.patch_size = spatial_patch_size
        self.T = temporal_window_days
        self.half_T = temporal_window_days // 2  # 15 days
        self.resolution = grid_resolution_deg
        self.max_missing_ratio = max_missing_ratio

    def match_variable_name(self, ds_vars: list[str] | set[str], channel: str) -> str | None:
        """Find the matching dataset variable name for a canonical channel."""
        candidates = VARIABLE_ALIASES.get(channel, [])
        lower_map = {v.lower(): v for v in ds_vars}
        for alias in candidates:
            if alias.lower() in lower_map:
                return lower_map[alias.lower()]
        return None

    def validate_dataset_compatibility(self, ds: Any) -> dict[str, Any]:
        """Validate whether a real dataset contains required coordinates and variables."""
        available_vars = set(ds.data_vars) if hasattr(ds, "data_vars") else set(ds.variables)
        var_mapping: dict[str, str] = {}
        missing_channels: list[str] = []

        for ch in CANONICAL_CHANNELS:
            matched = self.match_variable_name(available_vars, ch)
            if matched:
                var_mapping[ch] = matched
            else:
                missing_channels.append(ch)

        # Coordinate checks
        coords = set(ds.coords) if hasattr(ds, "coords") else set(ds.variables)
        lat_coord = next((c for c in ["latitude", "lat", "LATITUDE", "lat_rho"] if c in coords), None)
        lon_coord = next((c for c in ["longitude", "lon", "LONGITUDE", "lon_rho"] if c in coords), None)
        time_coord = next((c for c in ["time", "TIME", "datetime", "date"] if c in coords), None)

        is_compatible = len(missing_channels) == 0 and lat_coord is not None and lon_coord is not None and time_coord is not None

        return {
            "compatible": is_compatible,
            "variable_mapping": var_mapping,
            "missing_channels": missing_channels,
            "lat_coord": lat_coord,
            "lon_coord": lon_coord,
            "time_coord": time_coord,
        }

    def preprocess_patch(
        self,
        ds: Any,
        center_lat: float,
        center_lon: float,
        center_date: datetime.date | str,
        norm_stats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Extract, resample, and normalize a (31, 5, 32, 32) patch centered at given coords and date.

        Missing value strategy:
        - Checks missing ratio per channel. If missing ratio exceeds `max_missing_ratio` (40%),
          raises DataIncompatibilityError (e.g., over land or outside coverage).
        - For minor gaps, applies temporal interpolation followed by spatial nearest-neighbor fill.
        """
        import pandas as pd
        import xarray as xr

        if isinstance(center_date, str):
            c_date = datetime.date.fromisoformat(center_date)
        elif isinstance(center_date, datetime.datetime):
            c_date = center_date.date()
        else:
            c_date = center_date

        compat = self.validate_dataset_compatibility(ds)
        if not compat["compatible"]:
            raise DataIncompatibilityError(
                f"Satellite dataset incompatible: missing channels {compat['missing_channels']} "
                f"or coordinates (lat={compat['lat_coord']}, lon={compat['lon_coord']}, time={compat['time_coord']})."
            )

        lat_name = compat["lat_coord"]
        lon_name = compat["lon_coord"]
        time_name = compat["time_coord"]
        var_map = compat["variable_mapping"]

        # 1. Coordinate normalization (e.g. 0..360 to -180..180)
        ds_work = ds
        lons = ds_work[lon_name].values
        if (lons > 180.0).any():
            ds_work = ds_work.assign_coords({lon_name: (((ds_work[lon_name] + 180) % 360) - 180)})
            ds_work = ds_work.sortby(lon_name)

        # 2. Compute spatial target grid: 32x32 centered at (center_lat, center_lon)
        half_h = (self.patch_size - 1) / 2.0 * self.resolution
        half_w = (self.patch_size - 1) / 2.0 * self.resolution
        target_lats = np.linspace(center_lat - half_h, center_lat + half_h, self.patch_size, dtype=np.float32)
        target_lons = np.linspace(center_lon - half_w, center_lon + half_w, self.patch_size, dtype=np.float32)

        # 3. Compute 31-day temporal sequence: [center_date - 15 days, center_date + 15 days]
        start_date = c_date - datetime.timedelta(days=self.half_T)
        end_date = c_date + datetime.timedelta(days=self.half_T)
        target_dates = pd.date_range(start_date, end_date, freq="D")

        # 4. Extract and resample each channel
        channel_arrays = []
        for ch in CANONICAL_CHANNELS:
            v_name = var_map[ch]
            da = ds_work[v_name]

            # If variable has depth dimension, take surface slice (depth=0)
            if "depth" in da.dims:
                da = da.isel(depth=0)
            elif "deptho" in da.dims:
                da = da.isel(deptho=0)

            # Interpolate spatially and temporally onto target grid
            try:
                da_interp = da.interp(
                    {
                        lat_name: target_lats,
                        lon_name: target_lons,
                        time_name: target_dates,
                    },
                    method="linear",
                )
            except Exception as ex:
                raise DataIncompatibilityError(f"Interpolation failed for channel {ch}: {ex}")

            arr = da_interp.values.astype(np.float32)  # shape (31, 32, 32)

            # Missing value inspection
            missing_mask = np.isnan(arr) | np.isinf(arr)
            missing_ratio = float(np.mean(missing_mask))
            if missing_ratio > self.max_missing_ratio:
                raise DataIncompatibilityError(
                    f"Channel {ch} has excessive missing values ({missing_ratio*100:.1f}% > {self.max_missing_ratio*100:.1f}%), "
                    "likely due to land mask or out-of-bounds coverage."
                )

            if missing_ratio > 0.0:
                # Strategy: Fill temporal gaps with nearest valid time step, then spatial mean fill
                logger.info("Imputing %.2f%% missing values for channel %s using documented temporal/spatial fill.", missing_ratio * 100, ch)
                # Fill along time axis if possible
                arr = self._fill_missing_values(arr)

            channel_arrays.append(arr)

        # Stack into (T=31, C=5, H=32, W=32)
        patch = np.stack(channel_arrays, axis=1)

        # 5. Normalization using stored training stats if provided
        if norm_stats is not None and "mean" in norm_stats and "std" in norm_stats:
            mean = norm_stats["mean"]
            std = norm_stats["std"]
            patch = (patch - mean) / np.maximum(std, 1e-6)

        return patch

    def _fill_missing_values(self, arr: np.ndarray) -> np.ndarray:
        """Robust fill strategy: temporal forward/backward fill, then spatial mean fallback."""
        filled = arr.copy()
        T, H, W = filled.shape
        # Temporal forward/backward fill per pixel
        for i in range(H):
            for j in range(W):
                pixel_series = filled[:, i, j]
                nans = np.isnan(pixel_series)
                if np.all(nans):
                    continue
                if np.any(nans):
                    # Valid values exist in temporal series: interpolate
                    valid_idx = np.where(~nans)[0]
                    filled[:, i, j] = np.interp(np.arange(T), valid_idx, pixel_series[valid_idx])

        # Remaining NaNs (e.g. land points) filled with channel spatial valid mean
        still_nan = np.isnan(filled)
        if np.any(still_nan):
            valid_mean = float(np.nanmean(filled))
            filled[still_nan] = valid_mean

        return filled
