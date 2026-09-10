"""Uncertainty-aware baseline model for OceanEmbed.

This keeps the same encoder/decoder structure as the v0 baseline and adds a
second head for predictive variance. The model predicts (mean, log_variance)
per depth level so Gaussian negative log likelihood can be used during training.
"""

from __future__ import annotations

import torch
from torch import nn


class OceanBaselineV1(nn.Module):
    """A v0-like baseline with a separate uncertainty head for Gaussian calibration."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, output_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log-variance predictions for each depth level."""
        hidden = self.shared(x)
        mean = self.mean_head(hidden)
        log_var = self.logvar_head(hidden)
        return mean, log_var
