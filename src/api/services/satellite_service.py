"""Satellite data service and provider abstractions for OceanEmbed Phase 5.

Defines the extensible interface for surface observation providers (CMEMS, synthetic demo)
and ensures honest reporting of data availability and operating mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger("oceanembed.satellite_service")


class SatelliteDataProvider(ABC):
    """Abstract base class for satellite surface observation providers."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider credentials/source paths are configured."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if live satellite raster data is currently reachable/ingested."""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return provider status and metadata."""
        pass


class SyntheticDemoProvider(SatelliteDataProvider):
    """Deterministic synthetic fixture generator provider for software verification & demo."""

    def is_configured(self) -> bool:
        return True

    def is_available(self) -> bool:
        # Synthetic generator is always executable locally
        return True

    def get_status(self) -> dict[str, Any]:
        return {
            "provider_name": "Synthetic Spatiotemporal Demo Generator",
            "is_configured": True,
            "is_available": True,
            "data_mode": "synthetic_demo",
            "channels": settings.surface_channels,
            "temporal_window_days": 31,
            "spatial_resolution_deg": 0.25,
            "notice": "Deterministic mathematical simulation for software demonstration only.",
        }


class CMEMSSatelliteProvider(SatelliteDataProvider):
    """Copernicus Marine Environment Monitoring Service (CMEMS) API provider adapter."""

    def __init__(self, username: str = "", password: str = "") -> None:
        self.username = username or settings.cmems_username
        self.password = password or settings.cmems_password

    def is_configured(self) -> bool:
        return bool(self.username and self.password)

    def is_available(self) -> bool:
        # Only available if real credentials exist and raster cache exists
        if not self.is_configured():
            return False
        # If credentials provided, check if local satellite raster data exists
        if not settings.satellite_data_dir.exists():
            return False
        # Check if actual satellite netcdf / zarr data files exist
        has_nc = next(settings.satellite_data_dir.glob("*.nc"), None) is not None
        has_zarr = next(settings.satellite_data_dir.glob("*.zarr"), None) is not None
        return has_nc or has_zarr


    def get_status(self) -> dict[str, Any]:
        configured = self.is_configured()
        available = self.is_available()

        if not configured:
            message = "Real satellite data provider is not configured. CMEMS credentials absent."
        elif not available:
            message = "CMEMS credentials provided but no local satellite raster files found in data/raw/satellite."
        else:
            message = "Real CMEMS satellite data ingested and available."

        return {
            "provider_name": "Copernicus Marine Service (CMEMS)",
            "is_configured": configured,
            "is_available": available,
            "data_mode": "real_satellite" if available else "synthetic_demo",
            "channels": settings.surface_channels,
            "message": message,
        }


class SatelliteService:
    """Service orchestrating satellite observation providers and runtime status."""

    def __init__(self) -> None:
        self.demo_provider = SyntheticDemoProvider()
        self.cmems_provider = CMEMSSatelliteProvider()

    def get_satellite_status(self) -> dict[str, Any]:
        """Return satellite status according to current configuration and data presence.

        Adheres strictly to scientific honesty:
        - If real satellite data is NOT ingested, available is False and mode is 'synthetic_demo'.
        """
        is_real_mode = (settings.data_mode == "real")
        cmems_available = self.cmems_provider.is_available()

        if is_real_mode and cmems_available:
            return {
                "available": True,
                "mode": "real_satellite",
                "provider": "Copernicus Marine Service (CMEMS)",
                "variables": settings.surface_channels,
                "coverage": {
                    "domain": "Bay of Bengal (5°N–23°N, 80°E–100°E)",
                    "resolution_deg": 0.25,
                },
                "message": "Real satellite observations active.",
            }

        return {
            "available": False,
            "mode": "synthetic_demo",
            "provider": None,
            "variables": settings.surface_channels,
            "coverage": {
                "domain": "Bay of Bengal (5°N–23°N, 80°E–100°E)",
                "resolution_deg": 0.25,
            },
            "message": (
                "Real satellite data provider is not configured. "
                "Operating in reproducible synthetic demo mode."
            ),
        }
