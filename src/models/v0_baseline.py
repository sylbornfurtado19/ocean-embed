"""Baseline model architecture for OceanEmbed.

TODO: implement a simple MLP/CNN model that maps surface inputs to 15 depth outputs.
"""

from __future__ import annotations

import torch
from torch import nn


class OceanEmbedBaseline(nn.Module):
    """Simple baseline model mapping surface features to temperature at 15 depth levels."""

    def __init__(self, input_dim: int, depth_levels: int = 15) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.depth_levels = depth_levels
        # TODO: define encoder/decoder architecture.
        raise NotImplementedError("TODO: implement baseline architecture.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return depth-wise temperature predictions for the given input batch."""
        # TODO: implement forward pass.
        raise NotImplementedError("TODO: implement forward pass.")
