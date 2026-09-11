"""Pydantic request and response schemas for OceanEmbed Phase 5 API."""

from __future__ import annotations

import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Payload for subsurface ocean temperature reconstruction."""

    latitude: float = Field(
        ...,
        description="Target latitude in degrees North (supported Bay of Bengal range: 5.0 to 23.0).",
        examples=[14.5],
    )
    longitude: float = Field(
        ...,
        description="Target longitude in degrees East (supported Bay of Bengal range: 80.0 to 100.0).",
        examples=[88.0],
    )
    date: str = Field(
        ...,
        description="Target prediction date in ISO format YYYY-MM-DD (central date of 31-day temporal sequence).",
        examples=["2023-06-15"],
    )
    data_mode: Literal["synthetic", "real"] = Field(
        default="synthetic",
        description="Observation data mode: 'synthetic' demo data or 'real' satellite observations.",
    )

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (5.0 <= v <= 23.0):
            raise ValueError(f"Latitude {v:.4f}°N is outside the Bay of Bengal demo domain (5.0°N to 23.0°N).")
        return round(v, 4)

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (80.0 <= v <= 100.0):
            raise ValueError(f"Longitude {v:.4f}°E is outside the Bay of Bengal demo domain (80.0°E to 100.0°E).")
        return round(v, 4)

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            parsed = datetime.date.fromisoformat(v.strip())
            # Ensure within year 2023 for demo consistency
            if parsed.year < 2020 or parsed.year > 2026:
                raise ValueError(f"Date {v} is outside supported simulation window (2020-2026).")
            return parsed.isoformat()
        except Exception as ex:
            if isinstance(ex, ValueError) and "supported simulation window" in str(ex):
                raise ex
            raise ValueError(f"Invalid date format '{v}'. Expected ISO format YYYY-MM-DD.")


class PredictionResponse(BaseModel):
    """Reconstruction results for the 15-depth subsurface ocean temperature profile."""

    latitude: float = Field(..., description="Selected target latitude (°N).")
    longitude: float = Field(..., description="Selected target longitude (°E).")
    date: str = Field(..., description="Central prediction date (YYYY-MM-DD).")
    depths: list[float] = Field(..., description="Standardized target depths in meters (15 depths: 0 to 1000m).")
    temperatures: list[float] = Field(..., description="Model predicted temperature at each depth level in °C.")
    sigma: list[float] = Field(..., description="Predictive standard deviation uncertainty at each depth in °C.")
    lower_90: list[float] = Field(..., description="Lower 90% Gaussian predictive bound (mean - 1.645*sigma) in °C.")
    upper_90: list[float] = Field(..., description="Upper 90% Gaussian predictive bound (mean + 1.645*sigma) in °C.")
    regime_probs: list[float] = Field(..., description="Differentiable probability distribution across K=4 latent regimes.")
    embedding: list[float] = Field(..., description="Continuous 512-D latent Ocean Embedding vector.")
    climatology_prior: list[float] | None = Field(None, description="Background climatology reference prior in °C.")
    anomaly: list[float] | None = Field(None, description="Model-predicted subsurface temperature anomaly in °C.")
    surface_inputs: dict[str, float] | None = Field(None, description="Representative demo surface boundary conditions.")
    inference_latency_ms: float = Field(..., description="Measured forward pass inference execution time in milliseconds.")
    data_mode: str = Field(..., description="Operational data mode ('synthetic_demo' or 'real_satellite').")


class ModelInfo(BaseModel):
    available: bool
    version: str


class DataInfo(BaseModel):
    satellite: bool
    argo: bool


class HealthResponse(BaseModel):
    """System health and operational status response."""

    status: str = Field(..., description="Service health ('ok' or 'degraded').")
    model: ModelInfo
    data: DataInfo
    mode: str = Field(..., description="Current data mode ('synthetic_demo' or 'real').")


class DomainInfo(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class ModelStatusResponse(BaseModel):
    """Comprehensive metadata about the OceanEmbed neural model and checkpoint."""

    version: str
    checkpoint_path: str
    checkpoint_exists: bool
    checkpoint_size_mb: float
    is_loaded: bool
    embedding_dim: int
    num_depths: int
    target_depths: list[float]
    num_regimes: int
    domain: DomainInfo
    surface_channels: list[str]
    load_error: str | None = None


class ArgoStatusResponse(BaseModel):
    """Status of in-situ ARGO NetCDF observations."""

    available: bool
    files_found: int
    profiles_available: int
    directory: str
    message: str


class SatelliteStatusResponse(BaseModel):
    """Status of satellite surface observation providers."""

    available: bool
    mode: str
    provider: str | None
    variables: list[str]
    coverage: dict[str, Any]
    message: str


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    detail: str
    error_code: str
    timestamp: str
