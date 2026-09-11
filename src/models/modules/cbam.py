"""Convolutional Block Attention Module (CBAM) implementation for OceanEmbed.

CBAM consists of two sequential sub-modules:
1. Channel Attention: models inter-channel relationships using both AvgPool and MaxPool.
2. Spatial Attention: models inter-spatial relationships across spatial feature positions.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Channel attention module using combined average and max pooling."""

    def __init__(self, in_planes: int, ratio: int = 8) -> None:
        super().__init__()
        reduced_planes = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention module utilizing inter-spatial feature pooling."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, in_planes: int, ratio: int = 8, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out
