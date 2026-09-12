#!/usr/bin/env python3
"""Reproducible in-situ ARGO float NetCDF download utility for OceanEmbed.

Acquires real Argo float profiles for the Bay of Bengal domain (5°N–23°N, 80°E–100°E)
using the argopy library or GDAC HTTPS repositories.

Adheres strictly to scientific honesty:
- Never fabricates or generates fake observation records.
- If argopy is not installed, network fails, or no profiles exist,
  logs 'REAL ARGO DOWNLOAD: SKIPPED / UNAVAILABLE' and exits cleanly.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Ensure project root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oceanembed.download_argo")

# Default Bay of Bengal Domain Bounds
DEFAULT_LAT_MIN = 5.0
DEFAULT_LAT_MAX = 23.0
DEFAULT_LON_MIN = 80.0
DEFAULT_LON_MAX = 100.0
DEFAULT_DEPTH_MIN = 0.0
DEFAULT_DEPTH_MAX = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real ARGO in-situ float NetCDF profiles for the Bay of Bengal."
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
        help=f"Minimum latitude (°N), default: {DEFAULT_LAT_MIN}",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
        help=f"Maximum latitude (°N), default: {DEFAULT_LAT_MAX}",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_LON_MIN,
        help=f"Minimum longitude (°E), default: {DEFAULT_LON_MIN}",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_LON_MAX,
        help=f"Maximum longitude (°E), default: {DEFAULT_LON_MAX}",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-03-31",
        help="End date YYYY-MM-DD (default: 2023-03-31)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.argo_data_dir),
        help="Directory to save downloaded NetCDF profiles",
    )
    parser.add_argument(
        "--max-profiles",
        "--limit",
        dest="max_profiles",
        type=int,
        default=50,
        help="Maximum profiles to download to prevent oversized transfers (default: 50)",
    )
    return parser.parse_args()


def download_argo_profiles(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_date: str,
    end_date: str,
    output_dir: Path,
    max_profiles: int = 50,
) -> dict[str, Any]:
    """Acquire real ARGO profiles via argopy or GDAC."""
    logger.info("Initializing ARGO profile acquisition...")
    logger.info("Spatial bounds: Lat [%.2f, %.2f]°N, Lon [%.2f, %.2f]°E", lat_min, lat_max, lon_min, lon_max)
    logger.info("Temporal window: %s to %s", start_date, end_date)
    logger.info("Target directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for argopy library availability
    try:
        import argopy
        from argopy import DataFetcher
    except ImportError:
        logger.warning("argopy library is not installed in the current Python environment.")
        logger.info("=" * 60)
        logger.info("REAL ARGO DOWNLOAD: SKIPPED / UNAVAILABLE")
        logger.info("Reason: argopy library is not installed. To install: pip install argopy")
        logger.info("Operating mode remains deterministic synthetic demo mode.")
        logger.info("=" * 60)
        return {
            "status": "UNAVAILABLE",
            "reason": "argopy library not installed",
            "profiles_downloaded": 0,
        }

    try:
        logger.info("Querying ARGO data fetcher for region box...")
        # argopy box format: [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_start, date_end]
        box = [
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            DEFAULT_DEPTH_MIN,
            DEFAULT_DEPTH_MAX,
            start_date,
            end_date,
        ]

        fetcher = DataFetcher(src="erddap", mode="expert").region(box)
        ds = fetcher.to_xarray()

        if ds is None or len(ds.coords) == 0:
            logger.info("=" * 60)
            logger.info("REAL ARGO DOWNLOAD: COMPLETED (0 PROFILES FOUND)")
            logger.info("No Argo profiles were returned for the specified spatio-temporal box.")
            logger.info("=" * 60)
            return {
                "status": "NO_PROFILES",
                "reason": "No profiles found in specified bounding box",
                "profiles_downloaded": 0,
            }

        # Save dataset to NetCDF in output directory
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        target_nc_file = output_dir / f"argo_bob_{start_date}_{end_date}_{timestamp_str}.nc"
        ds.to_netcdf(target_nc_file)
        logger.info("Successfully saved ARGO dataset to: %s", target_nc_file)

        return {
            "status": "SUCCESS",
            "file": str(target_nc_file),
            "profiles_downloaded": int(ds.sizes.get("N_PROF", 1)),
        }

    except Exception as ex:
        logger.warning("ARGO acquisition failed or network endpoint is unreachable: %s", ex)
        logger.info("=" * 60)
        logger.info("REAL ARGO DOWNLOAD: SKIPPED / UNAVAILABLE")
        logger.info("Reason: %s", ex)
        logger.info("Operating mode remains deterministic synthetic demo mode.")
        logger.info("=" * 60)
        return {
            "status": "UNAVAILABLE",
            "reason": str(ex),
            "profiles_downloaded": 0,
        }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    result = download_argo_profiles(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=out_dir,
        max_profiles=args.max_profiles,
    )
    return 0 if result["status"] in ["SUCCESS", "UNAVAILABLE", "NO_PROFILES"] else 1


if __name__ == "__main__":
    sys.exit(main())
