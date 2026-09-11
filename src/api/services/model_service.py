"""Model service for OceanEmbed Phase 5 FastAPI backend.

Wraps OceanInferenceEngine in a singleton/cached service lifecycle,
ensuring that the trained checkpoint is loaded once and shared across API requests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.inference import OceanInferenceEngine

logger = logging.getLogger("oceanembed.model_service")


class ModelNotAvailableError(Exception):
    """Raised when OceanEmbed checkpoint is missing or cannot be loaded."""
    pass


class RealDataNotAvailableError(Exception):
    """Raised when real satellite data mode is requested but real data is not configured or available."""
    pass


class ModelService:
    """Singleton service managing OceanEmbed V2 model lifecycle and inference."""

    _instance: ModelService | None = None
    _engine: OceanInferenceEngine | None = None
    _load_error: str | None = None

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else settings.checkpoint_path

    @classmethod
    def get_instance(cls, checkpoint_path: str | Path | None = None) -> ModelService:
        """Return singleton instance of the ModelService."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def load_model(self) -> None:
        """Load and cache the trained OceanInferenceEngine into memory."""
        if self._engine is not None:
            return

        if not self.checkpoint_path.exists():
            self._load_error = f"Checkpoint file not found at: {self.checkpoint_path}"
            logger.warning("Model checkpoint not found: %s", self.checkpoint_path)
            return

        try:
            logger.info("Loading OceanEmbed model checkpoint from: %s", self.checkpoint_path)
            self._engine = OceanInferenceEngine(self.checkpoint_path)
            self._load_error = None
            logger.info("OceanEmbed V2 model loaded successfully into memory.")
        except Exception as ex:
            self._load_error = str(ex)
            self._engine = None
            logger.error("Failed to load OceanEmbed checkpoint: %s", ex, exc_info=True)

    @property
    def is_available(self) -> bool:
        """Check if model is currently loaded and ready for inference."""
        if self._engine is None and self.checkpoint_path.exists():
            self.load_model()
        return self._engine is not None

    def get_engine(self) -> OceanInferenceEngine:
        """Get the cached inference engine or raise ModelNotAvailableError."""
        if self._engine is None:
            self.load_model()
        if self._engine is None:
            raise ModelNotAvailableError(
                self._load_error or f"OceanEmbed model checkpoint not loaded from {self.checkpoint_path}"
            )
        return self._engine

    def get_model_status(self) -> dict[str, Any]:
        """Return comprehensive metadata about the current model and checkpoint."""
        exists = self.checkpoint_path.exists()
        size_bytes = self.checkpoint_path.stat().st_size if exists else 0
        size_mb = round(size_bytes / (1024 * 1024), 2) if exists else 0.0

        if self._engine is None and exists:
            self.load_model()

        return {
            "version": "oceanembed_v2" if self._engine else "unknown",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": exists,
            "checkpoint_size_mb": size_mb,
            "is_loaded": self.is_available,
            "embedding_dim": settings.embedding_dim,
            "num_depths": len(settings.target_depths),
            "target_depths": settings.target_depths,
            "num_regimes": settings.num_regimes,
            "domain": {
                "lat_min": settings.domain_lat_min,
                "lat_max": settings.domain_lat_max,
                "lon_min": settings.domain_lon_min,
                "lon_max": settings.domain_lon_max,
            },
            "surface_channels": settings.surface_channels,
            "load_error": self._load_error,
        }

    def predict(
        self,
        latitude: float,
        longitude: float,
        date_val: Any,
        data_mode: str = "synthetic",
    ) -> dict[str, Any]:
        """Execute model inference and return formatted prediction dictionary.

        Adheres strictly to Part K:
        - If data_mode == 'synthetic': use deterministic demo generator.
        - If data_mode == 'real': use real satellite data only if genuinely available.
          Otherwise raise RealDataNotAvailableError (which maps to HTTP 503).
          NEVER silently fall back to synthetic mode.
        """
        engine = self.get_engine()
        if data_mode == "real":
            sat_dir = settings.satellite_data_dir
            sat_files = (
                list(sat_dir.glob("*.nc"))
                + list(sat_dir.glob("**/*.nc"))
                + list(sat_dir.glob("*.zarr"))
            ) if sat_dir.exists() else []

            if not sat_files:
                raise RealDataNotAvailableError(
                    "Real satellite data pipeline is not configured. "
                    "No satellite raster files were found in data/raw/satellite. "
                    "In adherence to SIH scientific honesty rules, real-mode inference cannot proceed without verified observations."
                )

            try:
                from src.data.real_satellite_preprocessing import RealSatellitePreprocessor
                import xarray as xr
                preprocessor = RealSatellitePreprocessor()
                ds = xr.open_mfdataset([str(f) for f in sat_files], combine="by_coords")
                x_patch = preprocessor.preprocess_patch(
                    ds=ds,
                    center_lat=latitude,
                    center_lon=longitude,
                    center_date=date_val,
                    norm_stats=engine.norm_stats,
                )
                result = engine.predict_spatiotemporal(x_patch)
                result["latitude"] = latitude
                result["longitude"] = longitude
                result["prediction_date"] = str(date_val)
                result["data_mode"] = "real_satellite"
                return result
            except Exception as err:
                logger.error("Failed to process real satellite rasters for prediction: %s", err)
                raise RealDataNotAvailableError(
                    f"Real satellite data processing failed: {err}"
                )

        # Default synthetic mode
        result = engine.predict_from_location_date(latitude, longitude, date_val)
        result["data_mode"] = "synthetic_demo"
        return result
