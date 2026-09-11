"""ARGO in-situ NetCDF ingestion service for OceanEmbed Phase 5 & 6.

Discovers, validates, and parses real ARGO float NetCDF profiles in data/raw/argo/,
filters by Bay of Bengal spatial domain (5°N–23°N, 80°E–100°E), and provides
honest availability reporting and normalized profile representations.
"""

import datetime
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.config import WORKSPACE_ROOT, settings

logger = logging.getLogger("oceanembed.argo_service")

MAX_ARGO_FILES = 50
MAX_RETURNED_PROFILES = 100


class ArgoService:
    """Service discovering, inspecting, validating, and serving in-situ ARGO NetCDF float data."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else settings.argo_data_dir
        self._status_cache: dict[str, Any] | None = None
        self._status_cache_time: float = 0.0
        self._cache_ttl_seconds: float = 30.0

    def _sanitize_path(self, p: Path) -> str:
        """Return relative path to workspace root to avoid leaking host filesystem layout."""
        try:
            return str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
        except Exception:
            return p.name

    def discover_files(self) -> list[Path]:
        """Find all NetCDF files in the ARGO data directory with boundary and traversal guards."""
        if not self.data_dir.exists():
            return []

        resolved_root = self.data_dir.resolve()
        found: set[Path] = set()

        try:
            for pattern in ("*.nc", "*.nc4"):
                for p in self.data_dir.glob(pattern):
                    if p.is_file():
                        try:
                            if p.resolve().is_relative_to(resolved_root):
                                found.add(p)
                        except (ValueError, AttributeError):
                            if str(p.resolve()).startswith(str(resolved_root)):
                                found.add(p)

            # Check immediate subdirectories (1 level deep)
            for sub in self.data_dir.iterdir():
                if sub.is_dir():
                    try:
                        if not sub.resolve().is_relative_to(resolved_root):
                            continue
                    except (ValueError, AttributeError):
                        if not str(sub.resolve()).startswith(str(resolved_root)):
                            continue
                    for pattern in ("*.nc", "*.nc4"):
                        for p in sub.glob(pattern):
                            if p.is_file():
                                found.add(p)
        except Exception as err:
            logger.warning("Error discovering ARGO NetCDF files: %s", err)

        return sorted(list(found))[:MAX_ARGO_FILES]

    def validate_profile_file(self, file_path: Path) -> dict[str, Any]:
        """Validate a single NetCDF file for ARGO profiles adhering to Phase 6 requirements.

        Validates presence and validity of:
        - latitude
        - longitude
        - time
        - depth or pressure
        - temperature
        """
        result = {
            "file": file_path.name,
            "valid": False,
            "profile_count": 0,
            "profiles": [],
            "error": None,
        }

        try:
            import xarray as xr
            with xr.open_dataset(file_path) as ds:
                coords_and_vars = set(ds.coords) | set(ds.data_vars) | set(ds.variables)

                lat_name = next((c for c in ["latitude", "LATITUDE", "lat"] if c in coords_and_vars), None)
                lon_name = next((c for c in ["longitude", "LONGITUDE", "lon"] if c in coords_and_vars), None)
                time_name = next((c for c in ["time", "TIME", "date", "datetime"] if c in coords_and_vars), None)
                depth_name = next((c for c in ["pressure", "PRES", "depth", "DEPTH", "pres"] if c in coords_and_vars), None)
                temp_name = next((c for c in ["temperature", "TEMP", "temp", "temp_adjusted"] if c in coords_and_vars), None)

                if not (lat_name and lon_name and depth_name and temp_name):
                    missing = []
                    if not lat_name: missing.append("latitude")
                    if not lon_name: missing.append("longitude")
                    if not depth_name: missing.append("depth/pressure")
                    if not temp_name: missing.append("temperature")
                    result["error"] = f"Missing required variables: {', '.join(missing)}"
                    return result

                # Inspect profiles
                n_prof = ds.sizes.get("N_PROF", ds.sizes.get("profile", 1))
                valid_profiles = []

                for idx in range(n_prof):
                    try:
                        p_ds = ds.isel(N_PROF=idx) if "N_PROF" in ds.dims else (ds.isel(profile=idx) if "profile" in ds.dims else ds)
                        p_lat = float(np.asarray(p_ds[lat_name].values).reshape(-1)[0])
                        p_lon = float(np.asarray(p_ds[lon_name].values).reshape(-1)[0])

                        # Domain filter: 5°N–23°N, 80°E–100°E
                        if not (settings.domain_lat_min <= p_lat <= settings.domain_lat_max and settings.domain_lon_min <= p_lon <= settings.domain_lon_max):
                            continue

                        # Extract depths and temperatures
                        z_vals = np.asarray(p_ds[depth_name].values).reshape(-1)
                        t_vals = np.asarray(p_ds[temp_name].values).reshape(-1)

                        # Filter valid finite observations
                        mask = np.isfinite(z_vals) & np.isfinite(t_vals)
                        if np.sum(mask) < 3:
                            continue  # Need at least 3 depth levels for meaningful profile

                        valid_z = [round(float(z), 2) for z in z_vals[mask]]
                        valid_t = [round(float(t), 2) for t in t_vals[mask]]

                        # Time
                        date_str = "unknown"
                        if time_name and time_name in p_ds:
                            t_val = p_ds[time_name].values
                            if t_val is not None:
                                try:
                                    date_str = str(t_val)[:10]
                                except Exception:
                                    date_str = "unknown"

                        valid_profiles.append({
                            "profile_id": f"{file_path.stem}_{idx}",
                            "latitude": round(p_lat, 4),
                            "longitude": round(p_lon, 4),
                            "date": date_str,
                            "depth_min": float(np.min(valid_z)),
                            "depth_max": float(np.max(valid_z)),
                            "num_levels": len(valid_z),
                            "depths": valid_z,
                            "temperatures": valid_t,
                        })
                    except Exception as err:
                        logger.debug("Error extracting profile index %d from %s: %s", idx, file_path.name, err)
                        continue

                result["valid"] = len(valid_profiles) > 0
                result["profile_count"] = len(valid_profiles)
                result["profiles"] = valid_profiles

        except Exception as ex:
            result["error"] = str(ex)

        return result

    def get_argo_status(self) -> dict[str, Any]:
        """Return ARGO data availability status with short TTL caching to prevent repeated NetCDF parsing.

        Adheres strictly to scientific honesty:
        - Reports available: False and files_found: 0 if no NetCDF files exist.
        - Never fabricates observation records.
        """
        now = time.time()
        if self._status_cache is not None and (now - self._status_cache_time) < self._cache_ttl_seconds:
            return self._status_cache

        files = self.discover_files()
        files_found = len(files)
        sanitized_dir = self._sanitize_path(self.data_dir)

        if files_found == 0:
            result = {
                "available": False,
                "files_found": 0,
                "profiles_available": 0,
                "directory": sanitized_dir,
                "message": "No real ARGO NetCDF profiles found in data/raw/argo. In-situ verification is disabled.",
                "geographic_coverage": None,
                "depth_coverage": None,
                "temporal_coverage": None,
            }
            self._status_cache = result
            self._status_cache_time = now
            return result

        # Validate files and count valid profiles
        total_valid_profiles = 0
        all_lats, all_lons, all_depths, all_dates = [], [], [], []

        for f in files:
            val_res = self.validate_profile_file(f)
            if val_res["valid"]:
                total_valid_profiles += val_res["profile_count"]
                for p in val_res["profiles"]:
                    all_lats.append(p["latitude"])
                    all_lons.append(p["longitude"])
                    all_depths.extend([p["depth_min"], p["depth_max"]])
                    if p["date"] != "unknown":
                        all_dates.append(p["date"])

        if total_valid_profiles == 0:
            result = {
                "available": False,
                "files_found": files_found,
                "profiles_available": 0,
                "directory": sanitized_dir,
                "message": f"Found {files_found} ARGO file(s), but none contained valid Bay of Bengal profiles.",
                "geographic_coverage": None,
                "depth_coverage": None,
                "temporal_coverage": None,
            }
            self._status_cache = result
            self._status_cache_time = now
            return result

        geo_cov = {
            "lat_min": float(np.min(all_lats)),
            "lat_max": float(np.max(all_lats)),
            "lon_min": float(np.min(all_lons)),
            "lon_max": float(np.max(all_lons)),
        }
        depth_cov = {
            "min_depth_m": float(np.min(all_depths)),
            "max_depth_m": float(np.max(all_depths)),
        }
        temp_cov = {
            "start_date": min(all_dates) if all_dates else "unknown",
            "end_date": max(all_dates) if all_dates else "unknown",
        }

        result = {
            "available": True,
            "files_found": files_found,
            "profiles_available": total_valid_profiles,
            "directory": sanitized_dir,
            "message": f"Found {files_found} ARGO NetCDF file(s) with {total_valid_profiles} valid Bay of Bengal profile(s).",
            "geographic_coverage": geo_cov,
            "depth_coverage": depth_cov,
            "temporal_coverage": temp_cov,
        }
        self._status_cache = result
        self._status_cache_time = now
        return result

    def get_profiles(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        date_str: str | None = None,
        radius_deg: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Query ingested ARGO profiles within radius of target coordinate.

        Returns empty list if no real ARGO files are ingested.
        """
        files = self.discover_files()
        if not files:
            return []

        # Bound search radius to safe limits
        effective_radius = min(max(0.1, radius_deg), settings.max_argo_query_radius_deg)
        matched_profiles: list[dict[str, Any]] = []

        for f in files:
            if len(matched_profiles) >= MAX_RETURNED_PROFILES:
                break
            val_res = self.validate_profile_file(f)
            if not val_res["valid"]:
                continue
            for p in val_res["profiles"]:
                if latitude is not None and longitude is not None:
                    dist = ((p["latitude"] - latitude) ** 2 + (p["longitude"] - longitude) ** 2) ** 0.5
                    if dist > effective_radius:
                        continue
                matched_profiles.append(p)
                if len(matched_profiles) >= MAX_RETURNED_PROFILES:
                    break

        return matched_profiles

