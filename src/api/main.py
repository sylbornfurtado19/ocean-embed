"""OceanEmbed Phase 5 FastAPI Application.

Production-oriented API backend providing model status, real-time subsurface ocean temperature
reconstruction, satellite observation provider status, and in-situ ARGO NetCDF discovery.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import data_router, health_router, inference_router
from src.api.services.model_service import ModelService
from src.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oceanembed.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown events."""
    logger.info("Starting OceanEmbed API (Mode: %s)...", settings.data_mode)
    logger.info("Configured checkpoint path: %s", settings.checkpoint_path)

    # Pre-warm model in memory so first inference request is fast
    model_service = ModelService.get_instance()
    model_service.load_model()
    if model_service.is_available:
        logger.info("OceanEmbed V2 model loaded and ready on CPU.")
    else:
        logger.warning("OceanEmbed V2 model is not yet loaded or checkpoint is absent.")

    yield

    logger.info("Shutting down OceanEmbed API server.")


app = FastAPI(
    title="OceanEmbed API",
    description=(
        "Production-oriented backend for **OceanEmbed** — Satellite-Embedding-Based Deep Learning "
        "Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.\n\n"
        "Smart India Hackathon 2026 (Problem Statement ID: 26066 | Team: Bug Dealers)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for open/flexible integration (e.g. Streamlit or external frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Standardized handler for Pydantic input validation errors (HTTP 422)."""
    errors = exc.errors()
    error_msg = "; ".join([f"{e['loc'][-1]}: {e['msg']}" for e in errors]) if errors else "Validation error."
    logger.warning("Input validation error on %s: %s", request.url.path, error_msg)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": error_msg,
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized handler for HTTPExceptions."""
    logger.warning("HTTP %d error on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler suppressing raw Python tracebacks from API consumers."""
    logger.error("Unhandled exception processing %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error occurred during request processing.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


# Register Routers
app.include_router(health_router)
app.include_router(inference_router)
app.include_router(data_router)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Root endpoint welcoming users and directing to API documentation."""
    return {
        "project": "OceanEmbed",
        "description": "Satellite-Embedding-Based Subsurface Ocean Temperature Reconstruction API",
        "team": "Bug Dealers",
        "sih_id": "26066",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "mode": settings.data_mode,
    }
