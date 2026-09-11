"""Centralized configuration management for OceanEmbed Phase 5.

Loads environment variables with robust defaults for model serving,
data ingestion (satellite and ARGO NetCDF), and API runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# Determine Workspace Root (c:\...\ocean-embed)
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """OceanEmbed operational settings and environment configuration."""

    def __init__(self) -> None:
        # Operating Data Mode: 'synthetic' (default for reproducible demo) or 'real'
        mode_env = os.getenv("OCEAN_DATA_MODE", "synthetic").strip().lower()
        self.data_mode: Literal["synthetic", "real"] = "real" if mode_env == "real" else "synthetic"

        # Model Checkpoint Path
        ckpt_env = os.getenv("OCEANEMBED_CHECKPOINT", "checkpoints/oceanembed_v2.pt").strip()
        self.checkpoint_path: Path = Path(ckpt_env) if Path(ckpt_env).is_absolute() else (WORKSPACE_ROOT / ckpt_env)

        # Ingestion Directories
        argo_env = os.getenv("ARGO_DATA_DIR", "data/raw/argo").strip()
        self.argo_data_dir: Path = Path(argo_env) if Path(argo_env).is_absolute() else (WORKSPACE_ROOT / argo_env)

        sat_env = os.getenv("SATELLITE_DATA_DIR", "data/raw/satellite").strip()
        self.satellite_data_dir: Path = Path(sat_env) if Path(sat_env).is_absolute() else (WORKSPACE_ROOT / sat_env)

        # Copernicus Marine (CMEMS) Credentials (optional, never committed)
        self.cmems_username: str = os.getenv("CMEMS_USERNAME", "").strip()
        self.cmems_password: str = os.getenv("CMEMS_PASSWORD", "").strip()

        # API Server Runtime
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0").strip()
        self.api_port: int = int(os.getenv("API_PORT", "8000").strip())
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

        # Domain Constraints (Bay of Bengal Prototype)
        self.domain_lat_min: float = 5.0
        self.domain_lat_max: float = 23.0
        self.domain_lon_min: float = 80.0
        self.domain_lon_max: float = 100.0

        # Model Architecture Constants
        self.target_depths: list[float] = [
            0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0,
            125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
        ]
        self.embedding_dim: int = 512
        self.num_regimes: int = 4
        self.surface_channels: list[str] = ["SST", "SSS", "SLA", "Wind-U", "Wind-V"]

    @property
    def has_cmems_credentials(self) -> bool:
        """Check if Copernicus Marine credentials are provided."""
        return bool(self.cmems_username and self.cmems_password)

    def to_safe_dict(self) -> dict:
        """Export settings dictionary masking sensitive credentials."""
        return {
            "data_mode": self.data_mode,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "argo_data_dir": str(self.argo_data_dir),
            "satellite_data_dir": str(self.satellite_data_dir),
            "has_cmems_credentials": self.has_cmems_credentials,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "domain": {
                "lat_min": self.domain_lat_min,
                "lat_max": self.domain_lat_max,
                "lon_min": self.domain_lon_min,
                "lon_max": self.domain_lon_max,
            },
        }


# Global settings singleton
settings = Settings()
