"""Baseline model architecture for OceanEmbed."""

from __future__ import annotations

import torch
from torch import nn


class OceanBaselineV0(nn.Module):
    """A lightweight MLP baseline that maps surface features to 15-depth temperature predictions."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a 15-depth temperature vector for each example in the batch."""
        return self.net(x)
