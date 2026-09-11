"""Modules package for OceanEmbed deep neural networks."""

from src.models.modules.cbam import CBAM, ChannelAttention, SpatialAttention

__all__ = ["CBAM", "ChannelAttention", "SpatialAttention"]
