"""Data status and ingestion endpoints for OceanEmbed Phase 5 API."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_argo_service, get_satellite_service
from src.api.schemas import ArgoStatusResponse, SatelliteStatusResponse
from src.api.services.argo_service import ArgoService
from src.api.services.satellite_service import SatelliteService

router = APIRouter(prefix="/data", tags=["Data Ingestion & Observations"])


@router.get(
    "/argo/status",
    response_model=ArgoStatusResponse,
    summary="Get ARGO in-situ float data status",
    description="Returns the ingestion and verification status of real ARGO float NetCDF profiles in data/raw/argo.",
)
def get_argo_status(
    argo_service: ArgoService = Depends(get_argo_service),
) -> ArgoStatusResponse:
    """Return in-situ ARGO float data status."""
    status_dict = argo_service.get_argo_status()
    return ArgoStatusResponse(**status_dict)


@router.get(
    "/satellite/status",
    response_model=SatelliteStatusResponse,
    summary="Get satellite observation provider status",
    description="Returns the status of satellite surface observation providers (e.g. CMEMS vs synthetic demo).",
)
def get_satellite_status(
    satellite_service: SatelliteService = Depends(get_satellite_service),
) -> SatelliteStatusResponse:
    """Return satellite provider configuration and availability status."""
    status_dict = satellite_service.get_satellite_status()
    return SatelliteStatusResponse(**status_dict)


@router.get(
    "/argo/profiles",
    summary="Query ingested in-situ ARGO profiles",
    description="Returns profiles within specified geographic radius. Returns empty list if no NetCDF profiles are ingested.",
)
def get_argo_profiles(
    latitude: float | None = Query(None, ge=-90.0, le=90.0, description="Center latitude (°N)"),
    longitude: float | None = Query(None, ge=-180.0, le=180.0, description="Center longitude (°E)"),
    radius_deg: float = Query(1.0, ge=0.1, le=10.0, description="Search radius in degrees"),
    argo_service: ArgoService = Depends(get_argo_service),
) -> list[dict[str, Any]]:
    """Return in-situ ARGO float profiles matching search criteria."""
    return argo_service.get_profiles(latitude=latitude, longitude=longitude, radius_deg=radius_deg)
