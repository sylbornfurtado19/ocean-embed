"""Data download for OceanEmbed.

Pulls GLORYS reanalysis (surface fields + 15-depth temperature target) via the
copernicusmarine toolbox, and matching in-situ ARGO float profiles via argopy,
for the region/date range defined in a config YAML (e.g. configs/bay_of_bengal.yaml).

Run directly:
    python -m src.data.download --config configs/bay_of_bengal.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

GLORYS_RAW_DIR = Path("data/raw/glorys")
ARGO_RAW_DIR = Path("data/raw/argo")

# Copernicus Marine dataset IDs for GLORYS12 reanalysis (physics, daily mean).
GLORYS_SURFACE_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def _load_config(config: dict | str) -> dict:
    if isinstance(config, dict):
        return config
    with open(config, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def download_glorys(config: dict | str) -> None:
    """Download GLORYS surface fields + temperature profile for the config's region/dates.

    Pulls SST, SSS, SSH/SLA (zos), wind is NOT part of GLORYS (ocean reanalysis, not
    atmospheric) so wind-U/V are pulled separately from a companion product below.
    Requires `copernicusmarine login` to have been run once beforehand.
    """
    cfg = _load_config(config)
    region = cfg["region"]
    dates = cfg["dates"]
    GLORYS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import copernicusmarine
    except ImportError as exc:
        raise ImportError(
            "copernicusmarine is not installed. Run `pip install copernicusmarine` "
            "and `copernicusmarine login` before calling download_glorys()."
        ) from exc

    # Surface fields (SST, SSS, SSH) + full-depth temperature (training target),
    # pulled from the physics reanalysis dataset.
    variables = ["thetao", "so", "zos"]  # temperature, salinity, sea surface height
    output_file = GLORYS_RAW_DIR / "glorys_bay_of_bengal.nc"

    print(f"[DOWNLOAD] Requesting GLORYS subset -> {output_file}")
    copernicusmarine.subset(
        dataset_id=GLORYS_SURFACE_DATASET_ID,
        variables=variables,
        minimum_longitude=region["lon_min"],
        maximum_longitude=region["lon_max"],
        minimum_latitude=region["lat_min"],
        maximum_latitude=region["lat_max"],
        start_datetime=dates["start"],
        end_datetime=dates["end"],
        minimum_depth=0,
        maximum_depth=1500,
        output_filename=output_file.name,
        output_directory=str(GLORYS_RAW_DIR),
        force_download=True,
    )
    print(f"[DOWNLOAD] GLORYS data saved to {output_file}")

    # Wind-U/V come from a separate ocean-atmosphere flux/wind product.
    wind_output_file = GLORYS_RAW_DIR / "wind_bay_of_bengal.nc"
    print(f"[DOWNLOAD] Requesting wind subset -> {wind_output_file}")
    try:
        copernicusmarine.subset(
            dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
            variables=["eastward_wind", "northward_wind"],
            minimum_longitude=region["lon_min"],
            maximum_longitude=region["lon_max"],
            minimum_latitude=region["lat_min"],
            maximum_latitude=region["lat_max"],
            start_datetime=dates["start"],
            end_datetime=dates["end"],
            output_filename=wind_output_file.name,
            output_directory=str(GLORYS_RAW_DIR),
            force_download=True,
        )
        print(f"[DOWNLOAD] Wind data saved to {wind_output_file}")
    except Exception as exc:  # pragma: no cover - network/dataset-id dependent
        print(f"[DOWNLOAD] WARNING: wind download failed ({exc}). "
              f"Check the wind dataset_id is available for your account/region.")


def download_argo(config: dict | str) -> None:
    """Download matching ARGO float temperature profiles for the config's region/dates."""
    cfg = _load_config(config)
    region = cfg["region"]
    dates = cfg["dates"]
    ARGO_RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import argopy
    except ImportError as exc:
        raise ImportError("argopy is not installed. Run `pip install argopy` first.") from exc

    print("[DOWNLOAD] Requesting ARGO profiles for region/date range...")
    fetcher = argopy.DataFetcher(src="erddap").region(
        [
            region["lon_min"],
            region["lon_max"],
            region["lat_min"],
            region["lat_max"],
            0,
            1500,  # depth/pressure range in dbar, matches GLORYS depth range
            dates["start"],
            dates["end"],
        ]
    )
    ds = fetcher.to_xarray()

    output_file = ARGO_RAW_DIR / "argo_bay_of_bengal.nc"
    ds.to_netcdf(output_file)
    print(f"[DOWNLOAD] ARGO data saved to {output_file} ({ds.sizes} )")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GLORYS + ARGO data for OceanEmbed.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml")
    parser.add_argument("--skip-glorys", action="store_true", help="Skip the GLORYS download.")
    parser.add_argument("--skip-argo", action="store_true", help="Skip the ARGO download.")
    args = parser.parse_args()

    if not args.skip_glorys:
        download_glorys(args.config)
    if not args.skip_argo:
        download_argo(args.config)


if __name__ == "__main__":
    main()
