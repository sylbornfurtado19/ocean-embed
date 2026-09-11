"""Inference module for OceanEmbed baseline models.

Provides clean standalone inference utilities to run predictions on single or batched
surface feature vectors using trained model checkpoints with normalization statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1


class OceanInferenceEngine:
    """Reusable inference engine for OceanEmbed baseline models."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model_version = checkpoint.get("model_version", "v0")
        self.input_dim = checkpoint.get("input_dim", 8)
        self.output_dim = checkpoint.get("output_dim", 15)
        self.target_depths = checkpoint.get(
            "target_depths",
            [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
        )
        self.feature_names = checkpoint.get(
            "feature_names",
            ["sst", "sss", "sla", "wind_u", "wind_v", "lat", "lon", "time"],
        )
        self.norm_stats = checkpoint.get("norm_stats", None)

        hidden_dim = checkpoint.get("hidden_dim", 128)
        dropout = checkpoint.get("dropout", 0.1)

        if self.model_version == "v0":
            self.model = OceanBaselineV0(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.output_dim,
                dropout=dropout,
            )
        elif self.model_version == "v1_uncertainty":
            self.model = OceanBaselineV1(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.output_dim,
                dropout=dropout,
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

    def predict(self, features: np.ndarray | list[float] | dict[str, float]) -> dict[str, Any]:
        """Run inference on input features.

        Args:
            features: Can be:
              - 1D array/list of 8 feature values matching self.feature_names order
              - 2D array of shape (N, 8)
              - dictionary mapping feature names to scalar values

        Returns:
            Dictionary containing:
              - 'depths': list of 15 target depths (m)
              - 'temperature': 1D or 2D array of predicted temperatures (°C)
              - 'variance': predicted variance (for v1_uncertainty), or None (for v0)
              - 'uncertainty_sigma': standard deviation sqrt(variance) (for v1_uncertainty), or None
        """
        # Format input to 2D numpy array
        if isinstance(features, dict):
            vector = [features[k] for k in self.feature_names]
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

        if x_arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features ({self.feature_names}), got {x_arr.shape[1]}")

        # Check for NaN / Inf
        if np.isnan(x_arr).any() or np.isinf(x_arr).any():
            raise ValueError("Input feature vector contains NaN or Inf values.")

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
    """Convenience functional interface for inference."""
    engine = OceanInferenceEngine(checkpoint_path)
    return engine.predict(features)
