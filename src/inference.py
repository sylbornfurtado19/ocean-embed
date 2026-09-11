"""Inference module for OceanEmbed baseline and spatiotemporal models.

Provides clean standalone inference utilities to run predictions on:
- Phase 2 1D point surface feature vectors (V0 MSE baseline, V1 Gaussian uncertainty)
- Phase 3 5D spatiotemporal surface sequences (B, 31, 5, 32, 32) using OceanEmbed V2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.models.oceanembed_v2 import OceanEmbedV2
from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1


class OceanInferenceEngine:
    """Unified inference engine supporting V0, V1, and OceanEmbed V2."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model_version = checkpoint.get("model_version", "v0")
        self.target_depths = checkpoint.get(
            "target_depths",
            [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
        )
        self.norm_stats = checkpoint.get("norm_stats", None)

        if self.model_version == "oceanembed_v2":
            self.embedding_dim = checkpoint.get("embedding_dim", 512)
            self.model = OceanEmbedV2(
                in_channels=checkpoint.get("in_channels", 5),
                temporal_length=checkpoint.get("temporal_length", 31),
                spatial_patch_size=checkpoint.get("spatial_patch_size", 32),
                embedding_dim=self.embedding_dim,
                num_regimes=4,
                num_depths=len(self.target_depths),
                target_depths=self.target_depths,
            )
        elif self.model_version == "v1_uncertainty":
            self.input_dim = checkpoint.get("input_dim", 8)
            self.output_dim = checkpoint.get("output_dim", 15)
            self.model = OceanBaselineV1(
                input_dim=self.input_dim,
                hidden_dim=checkpoint.get("hidden_dim", 128),
                output_dim=self.output_dim,
                dropout=checkpoint.get("dropout", 0.1),
            )
        elif self.model_version == "v0":
            self.input_dim = checkpoint.get("input_dim", 8)
            self.output_dim = checkpoint.get("output_dim", 15)
            self.model = OceanBaselineV0(
                input_dim=self.input_dim,
                hidden_dim=checkpoint.get("hidden_dim", 128),
                output_dim=self.output_dim,
                dropout=checkpoint.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown model version in checkpoint: {self.model_version}")

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply stored training normalization statistics if available."""
        if self.norm_stats is not None:
            mean = self.norm_stats["mean"]
            std = self.norm_stats["std"]
            return (x - mean) / std
        return x

    def predict_spatiotemporal(
        self,
        x_patch: np.ndarray,
        climatology_prior: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference on spatiotemporal surface patch (T=31, C=5, H=32, W=32) or batch (B, 31, 5, 32, 32)."""
        if self.model_version != "oceanembed_v2":
            raise ValueError(f"predict_spatiotemporal called on {self.model_version} model (requires oceanembed_v2).")

        single_sample = False
        if x_patch.ndim == 4:
            x_arr = x_patch[np.newaxis, ...].astype(np.float32)
            single_sample = True
        elif x_patch.ndim == 5:
            x_arr = x_patch.astype(np.float32)
        else:
            raise ValueError(f"Expected 4D or 5D tensor for spatiotemporal input, got {x_patch.ndim}D")

        if np.isnan(x_arr).any() or np.isinf(x_arr).any():
            raise ValueError("Input patch contains NaN or Inf values.")

        x_norm = self._normalize(x_arr)
        x_t = torch.tensor(x_norm, dtype=torch.float32)

        clim_t = None
        if climatology_prior is not None:
            c_arr = np.asarray(climatology_prior, dtype=np.float32)
            if c_arr.ndim == 1:
                c_arr = c_arr[np.newaxis, :]
            clim_t = torch.tensor(c_arr, dtype=torch.float32)

        with torch.no_grad():
            out = self.model(x_t, climatology_prior=clim_t)
            temp = out["temperature_mean"].cpu().numpy()
            var = out["variance"].cpu().numpy()
            sigma = out["uncertainty_sigma"].cpu().numpy()
            emb = out["embedding"].cpu().numpy()
            regimes = out["regime_probs"].cpu().numpy()
            anomaly = out["anomaly"].cpu().numpy()
            clim = out["climatology_prior"].cpu().numpy()

        if single_sample:
            temp = temp[0]
            var = var[0]
            sigma = sigma[0]
            emb = emb[0]
            regimes = regimes[0]
            anomaly = anomaly[0]
            clim = clim[0]

        return {
            "model_version": self.model_version,
            "depths": list(self.target_depths),
            "temperature": temp,
            "variance": var,
            "uncertainty_sigma": sigma,
            "embedding": emb,
            "regime_probs": regimes,
            "anomaly": anomaly,
            "climatology_prior": clim,
        }

    def predict_from_location_date(
        self,
        latitude: float,
        longitude: float,
        date_val: Any,
    ) -> dict[str, Any]:
        """Generate deterministic demo patch and predict subsurface temperature profile for given coordinate and date.

        Args:
            latitude: Center latitude (5.0 to 23.0)
            longitude: Center longitude (80.0 to 100.0)
            date_val: datetime.date, datetime.datetime, or ISO string (e.g. '2023-02-15')

        Returns:
            Dictionary with prediction results, surface representative metrics, and execution time.
        """
        import time
        from datetime import date as dt_date, datetime as dt_datetime
        from src.data.generate_spatiotemporal_demo import (
            compute_demo_climatology_profile,
            generate_spatiotemporal_sample,
        )

        if isinstance(date_val, str):
            parsed_date = dt_datetime.fromisoformat(date_val).date()
        elif isinstance(date_val, dt_datetime):
            parsed_date = date_val.date()
        elif isinstance(date_val, dt_date):
            parsed_date = date_val
        else:
            raise TypeError(f"Unsupported date type: {type(date_val)}")

        center_day = float(parsed_date.timetuple().tm_yday)

        # Deterministic seed unique to location and day of year
        seed = int(abs(round(latitude, 4) * 10000 + round(longitude, 4) * 100 + center_day * 17)) % (2**31 - 1)
        rng = np.random.default_rng(seed)

        target_depths = np.array(self.target_depths, dtype=np.float32)
        X_patch, _, climatology = generate_spatiotemporal_sample(
            center_lat=float(latitude),
            center_lon=float(longitude),
            center_day=center_day,
            target_depths=target_depths,
            rng=rng,
            T=31,
            H=32,
            W=32,
        )

        # Measure forward inference latency
        t0 = time.perf_counter()
        result = self.predict_spatiotemporal(X_patch, climatology_prior=climatology)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract central surface conditions at day 0 (index 15), patch center (16, 16)
        # Channels: 0: SST, 1: SSS, 2: SLA, 3: Wind-U, 4: Wind-V
        mid_t, mid_h, mid_w = 15, 16, 16
        result["surface_inputs"] = {
            "sst": float(X_patch[mid_t, 0, mid_h, mid_w]),
            "sss": float(X_patch[mid_t, 1, mid_h, mid_w]),
            "sla": float(X_patch[mid_t, 2, mid_h, mid_w]),
            "wind_u": float(X_patch[mid_t, 3, mid_h, mid_w]),
            "wind_v": float(X_patch[mid_t, 4, mid_h, mid_w]),
        }
        result["latitude"] = float(latitude)
        result["longitude"] = float(longitude)
        result["prediction_date"] = parsed_date.isoformat()
        result["inference_time_ms"] = elapsed_ms
        result["patch_shape"] = (1, 31, 5, 32, 32)
        return result

    def extract_embedding(self, x_patch: np.ndarray) -> np.ndarray:
        """Extract explicit 512-D Ocean Embedding without full decoder execution."""
        if self.model_version != "oceanembed_v2":
            raise ValueError("Embedding extraction is only supported for oceanembed_v2.")
        single_sample = False
        if x_patch.ndim == 4:
            x_arr = x_patch[np.newaxis, ...].astype(np.float32)
            single_sample = True
        else:
            x_arr = x_patch.astype(np.float32)

        x_norm = self._normalize(x_arr)
        x_t = torch.tensor(x_norm, dtype=torch.float32)

        with torch.no_grad():
            emb = self.model.encode_to_embedding(x_t).cpu().numpy()

        return emb[0] if single_sample else emb

    def predict(self, features: np.ndarray | list[float] | dict[str, float]) -> dict[str, Any]:
        """Run inference on 1D feature vectors for V0 and V1 baselines."""
        if self.model_version == "oceanembed_v2":
            raise ValueError("OceanEmbed V2 requires predict_spatiotemporal(x_patch).")

        feature_names = ["sst", "sss", "sla", "wind_u", "wind_v", "lat", "lon", "time"]
        if isinstance(features, dict):
            vector = [features[k] for k in feature_names]
            x_arr = np.array([vector], dtype=np.float32)
            single_sample = True
        elif isinstance(features, list):
            x_arr = np.array([features], dtype=np.float32)
            single_sample = True
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                x_arr = features[np.newaxis, :].astype(np.float32)
                single_sample = True
            elif features.ndim == 2:
                x_arr = features.astype(np.float32)
                single_sample = False
            else:
                raise ValueError(f"Features array must be 1D or 2D, got {features.ndim}D")
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        x_norm = self._normalize(x_arr)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)

        with torch.no_grad():
            if self.model_version == "v1_uncertainty":
                mean_t, logvar_t = self.model(x_tensor)
                clamped_logvar = torch.clamp(logvar_t, -10.0, 10.0)
                var_t = torch.exp(clamped_logvar)
                sigma_t = torch.sqrt(var_t)
                temp_out = mean_t.cpu().numpy()
                var_out = var_t.cpu().numpy()
                sigma_out = sigma_t.cpu().numpy()
            else:
                pred_t = self.model(x_tensor)
                temp_out = pred_t.cpu().numpy()
                var_out = None
                sigma_out = None

        if single_sample:
            temp_out = temp_out[0]
            if var_out is not None:
                var_out = var_out[0]
                sigma_out = sigma_out[0]

        return {
            "model_version": self.model_version,
            "depths": list(self.target_depths),
            "temperature": temp_out,
            "variance": var_out,
            "uncertainty_sigma": sigma_out,
        }


def predict_profile(
    features: np.ndarray | list[float] | dict[str, float],
    checkpoint_path: str | Path = "checkpoints/v1_uncertainty.pt",
) -> dict[str, Any]:
    """Convenience functional interface for 1D baseline inference."""
    engine = OceanInferenceEngine(checkpoint_path)
    return engine.predict(features)


def predict_oceanembed(
    x_patch: np.ndarray,
    climatology_prior: np.ndarray | None = None,
    checkpoint_path: str | Path = "checkpoints/oceanembed_v2.pt",
) -> dict[str, Any]:
    """Convenience functional interface for OceanEmbed V2 spatiotemporal inference."""
    engine = OceanInferenceEngine(checkpoint_path)
    return engine.predict_spatiotemporal(x_patch, climatology_prior=climatology_prior)
