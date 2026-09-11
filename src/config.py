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


def _parse_bool(val: str | None, default: bool) -> bool:
    """Parse boolean environment variables safely, ensuring 'false' remains False."""
    if val is None:
        return default
    cleaned = val.strip().lower()
    if cleaned in ("1", "true", "yes", "on"):
        return True
    if cleaned in ("0", "false", "no", "off"):
        return False
    return default


def _parse_list(val: str | None, default: list[str]) -> list[str]:
    """Parse comma-separated strings into a list of stripped strings."""
    if val is None or not val.strip():
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


def _sanitize_path(p: Path) -> str:
    """Convert path to relative path against workspace root to prevent host filesystem leakage."""
    try:
        return str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except Exception:
        return p.name


class Settings:
    """OceanEmbed operational settings and environment configuration."""

    def __init__(self) -> None:
        # Runtime Environment: 'development', 'demo', or 'production'
        self.api_env: Literal["development", "demo", "production"] = (
            os.getenv("API_ENV", "development").strip().lower()  # type: ignore
        )
        if self.api_env not in ("development", "demo", "production"):
            self.api_env = "development"

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

        # Security & Networking
        default_origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
        ]
        self.cors_allowed_origins: list[str] = _parse_list(
            os.getenv("CORS_ALLOWED_ORIGINS"),
            default=default_origins,
        )

        default_hosts = ["localhost", "127.0.0.1", "testserver", "0.0.0.0"]
        self.trusted_hosts: list[str] = _parse_list(
            os.getenv("TRUSTED_HOSTS"),
            default=default_hosts,
        )

        # API Documentation Control (enabled by default in dev, disabled in prod/demo)
        doc_env = os.getenv("ENABLE_DOCS")
        if doc_env is not None:
            self.enable_docs: bool = _parse_bool(doc_env, default=True)
        else:
            self.enable_docs = (self.api_env == "development")

        # Optional API Key (empty string means authentication disabled)
        self.api_key: str = os.getenv("API_KEY", "").strip()

        # Resource Bounds & Abuse Prevention
        self.max_request_body_bytes: int = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576").strip())
        self.request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0").strip())
        self.max_concurrent_inference: int = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", "4").strip()))
        self.rate_limit_per_minute: int = max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "60").strip()))
        self.max_argo_query_radius_deg: float = float(os.getenv("MAX_ARGO_QUERY_RADIUS_DEG", "5.0").strip())

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
        """Export settings dictionary masking sensitive credentials and sanitizing filesystem paths."""
        return {
            "api_env": self.api_env,
            "data_mode": self.data_mode,
            "checkpoint_path": _sanitize_path(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "argo_data_dir": _sanitize_path(self.argo_data_dir),
            "satellite_data_dir": _sanitize_path(self.satellite_data_dir),
            "has_cmems_credentials": self.has_cmems_credentials,
            "has_api_key": bool(self.api_key),
            "enable_docs": self.enable_docs,
            "cors_allowed_origins": self.cors_allowed_origins,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "max_concurrent_inference": self.max_concurrent_inference,
            "max_request_body_bytes": self.max_request_body_bytes,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "domain": {
                "lat_min": self.domain_lat_min,
                "lat_max": self.domain_lat_max,
                "lon_min": self.domain_lon_min,
                "lon_max": self.domain_lon_max,
            },
        }


# Global settings singleton
settings = Settings()
