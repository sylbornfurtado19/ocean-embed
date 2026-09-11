"""API Route definitions package."""

from src.api.routes.data import router as data_router
from src.api.routes.health import router as health_router
from src.api.routes.inference import router as inference_router

__all__ = ["health_router", "inference_router", "data_router"]
