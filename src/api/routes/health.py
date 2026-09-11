"""Health and model status endpoints for OceanEmbed Phase 5 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_argo_service, get_model_service, get_satellite_service
from src.api.schemas import HealthResponse, ModelStatusResponse
from src.api.security import verify_api_key
from src.api.services.argo_service import ArgoService
from src.api.services.model_service import ModelService
from src.api.services.satellite_service import SatelliteService
from src.config import settings

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API and subsystem health",
    description="Returns service health, model availability, and honest status of real satellite and ARGO data sources.",
)
def get_health(
    model_service: ModelService = Depends(get_model_service),
    satellite_service: SatelliteService = Depends(get_satellite_service),
    argo_service: ArgoService = Depends(get_argo_service),
) -> HealthResponse:
    """Return system health and subsystem availability (public endpoint)."""
    model_avail = model_service.is_available
    sat_status = satellite_service.get_satellite_status()
    argo_status = argo_service.get_argo_status()

    # Overall service status
    status = "ok" if model_avail else "degraded"

    # Current mode
    current_mode = "real" if (settings.data_mode == "real" and sat_status["available"]) else "synthetic_demo"

    return HealthResponse(
        status=status,
        model={
            "available": model_avail,
            "version": "oceanembed_v2" if model_avail else "unknown",
        },
        data={
            "satellite": bool(sat_status["available"]),
            "argo": bool(argo_status["available"]),
        },
        mode=current_mode,
    )


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Inspect model checkpoint and architectural specifications",
    description="Returns metadata inspected directly from the loaded OceanEmbed V2 checkpoint and model architecture.",
)
def get_model_status(
    _auth: str | None = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service),
) -> ModelStatusResponse:
    """Return model status, checkpoint presence, and architectural dimensions."""
    status_dict = model_service.get_model_status()
    return ModelStatusResponse(**status_dict)

