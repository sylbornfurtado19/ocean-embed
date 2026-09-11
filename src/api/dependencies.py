"""FastAPI dependency injection providers for OceanEmbed Phase 5."""

from __future__ import annotations

from src.api.services.argo_service import ArgoService
from src.api.services.model_service import ModelService
from src.api.services.satellite_service import SatelliteService


def get_model_service() -> ModelService:
    """Provide singleton ModelService instance."""
    return ModelService.get_instance()


def get_satellite_service() -> SatelliteService:
    """Provide SatelliteService instance."""
    return SatelliteService()


def get_argo_service() -> ArgoService:
    """Provide ArgoService instance."""
    return ArgoService()
