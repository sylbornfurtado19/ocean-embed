"""OceanEmbed V2: Satellite Embedding-Based Deep Learning Framework.

Architecture:
Input (B, T=31, C=5, H=32, W=32)
  │
  ▼
Spatiotemporal Surface Encoder (Spatial CNN + ConvLSTM temporal aggregation)
  │
  ▼
CBAM Attention (Channel & Spatial Attention on deep feature maps)
  │
  ▼
512-Dimensional Ocean Embedding (Global pooling + Bottleneck Projection)
  │
  ├───► Soft Regime Context Head (Differentiable K-regime soft probabilities)
  │
  ▼
Depth-Conditioned Decoder (Embeds 15 target depths + coupled MLP)
  │
  ▼
Temperature Anomaly + Climatology Prior = 15-Depth Temperature Profile
  +
Predictive Uncertainty Head (Gaussian Negative Log-Likelihood variance)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules.cbam import CBAM

STANDARD_DEPTHS = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]


class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM cell for temporal aggregation of spatial feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SpatiotemporalSurfaceEncoder(nn.Module):
    """Encodes (B, T, C, H, W) surface satellite fields using 2D CNN spatial blocks and ConvLSTM."""

    def __init__(
        self,
        in_channels: int = 5,
        cnn_features: int = 32,
        lstm_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden

        # Spatial feature extractor applied per time-step: (C=5, H=32, W=32) -> (32, 16, 16)
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 16x16
            nn.Conv2d(cnn_features, cnn_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(inplace=True),
        )

        # Temporal recurrent aggregator
        self.conv_lstm = ConvLSTMCell(in_channels=cnn_features, hidden_channels=lstm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Flatten time into batch for CNN: (B*T, C, H, W)
        x_reshaped = x.view(B * T, C, H, W)
        feat_spatial = self.spatial_cnn(x_reshaped)  # (B*T, cnn_features, H/2, W/2)
        _, fC, fH, fW = feat_spatial.shape
        feat_seq = feat_spatial.view(B, T, fC, fH, fW)

        # Iterate ConvLSTM through time dimension
        h = torch.zeros(B, self.lstm_hidden, fH, fW, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.lstm_hidden, fH, fW, device=x.device, dtype=x.dtype)

        for t in range(T):
            h, c = self.conv_lstm(feat_seq[:, t], h, c)

        return h  # Final hidden spatial state: (B, lstm_hidden=32, 16, 16)


class DepthConditionedDecoder(nn.Module):
    """Decodes the 512-D Ocean Embedding + Regime Context into 15 depths using explicit depth embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        regime_dim: int = 4,
        depth_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_depths: int = 15,
        target_depths: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_depths = num_depths
        depth_list = target_depths if target_depths is not None else STANDARD_DEPTHS
        # Register normalized target depths as buffer: (15, 1)
        z_norm = np.log1p(np.array(depth_list, dtype=np.float32) / 10.0)[:, np.newaxis]
        self.register_buffer("depth_values", torch.tensor(z_norm, dtype=torch.float32))

        # Explicit continuous depth projection to depth embedding
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, depth_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(depth_embed_dim, depth_embed_dim),
        )

        in_decoder = embedding_dim + regime_dim + depth_embed_dim
        self.coupled_decoder = nn.Sequential(
            nn.Linear(in_decoder, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.anomaly_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        ocean_embedding: torch.Tensor,
        regime_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ocean_embedding: (B, 512), regime_context: (B, 4)
        B = ocean_embedding.shape[0]
        K = self.num_depths

        # 1. Compute depth embeddings for all 15 depths: (15, depth_embed_dim)
        z_embed = self.depth_mlp(self.depth_values)  # (15, 32)
        z_expanded = z_embed.unsqueeze(0).expand(B, K, -1)  # (B, 15, 32)

        # 2. Replicate ocean embedding and regime context across 15 depths
        ctx = torch.cat([ocean_embedding, regime_context], dim=1)  # (B, 516)
        ctx_expanded = ctx.unsqueeze(1).expand(B, K, -1)  # (B, 15, 516)

        # 3. Concatenate and decode coupled depths in parallel
        combined = torch.cat([ctx_expanded, z_expanded], dim=-1)  # (B, 15, 548)
        decoded = self.coupled_decoder(combined)  # (B, 15, 64)

        anomaly = self.anomaly_head(decoded).squeeze(-1)  # (B, 15)
        logvar = self.logvar_head(decoded).squeeze(-1)  # (B, 15)
        return anomaly, logvar


class OceanEmbedV2(nn.Module):
    """Core OceanEmbed V2 Deep Learning Framework."""

    def __init__(
        self,
        in_channels: int = 5,
        temporal_length: int = 31,
        spatial_patch_size: int = 32,
        embedding_dim: int = 512,
        num_regimes: int = 4,
        num_depths: int = 15,
        target_depths: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_length = temporal_length
        self.spatial_patch_size = spatial_patch_size
        self.embedding_dim = embedding_dim
        self.num_regimes = num_regimes
        self.num_depths = num_depths
        self.target_depths = target_depths if target_depths is not None else STANDARD_DEPTHS

        # 1. Spatiotemporal Surface Encoder
        self.encoder = SpatiotemporalSurfaceEncoder(in_channels=in_channels, cnn_features=32, lstm_hidden=32)

        # 2. CBAM Attention Module on encoder spatial hidden state (32 channels, 16x16)
        self.cbam = CBAM(in_planes=32, ratio=8, kernel_size=7)

        # 3. 512-D Ocean Embedding Projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 32, 1, 1) -> (B, 32)
        self.embedding_proj = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),  # Explicit (B, 512)
        )

        # 4. Soft Regime Context Head (differentiable probability distribution over K=4 regimes)
        self.regime_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_regimes),
        )

        # 5. Depth-Conditioned Decoder
        self.decoder = DepthConditionedDecoder(
            embedding_dim=embedding_dim,
            regime_dim=num_regimes,
            depth_embed_dim=32,
            hidden_dim=128,
            num_depths=num_depths,
            target_depths=self.target_depths,
        )

    def encode_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract explicit 512-dimensional Ocean Embedding from input sequence."""
        # 1. Spatiotemporal encoding: (B, T, C, H, W) -> (B, 32, 16, 16)
        spatial_temporal_feat = self.encoder(x)
        # 2. CBAM attention refinement
        refined = self.cbam(spatial_temporal_feat)
        # 3. Global pooling & projection to 512-D
        pooled = self.global_pool(refined).flatten(1)  # (B, 32)
        embedding = self.embedding_proj(pooled)  # (B, 512)
        return embedding

    def forward(
        self,
        x: torch.Tensor,
        climatology_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the entire OceanEmbed pipeline.

        Args:
            x: Surface input tensor of shape (B, T=31, C=5, H=32, W=32)
            climatology_prior: Optional background prior of shape (B, 15)

        Returns:
            Dictionary containing:
              - 'temperature_mean': Final reconstructed 15-depth temperatures (B, 15)
              - 'log_variance': Predictive log-variance per depth (B, 15)
              - 'variance': Clamped positive variance (B, 15)
              - 'uncertainty_sigma': Predictive standard deviation (B, 15)
              - 'embedding': Explicit 512-D Ocean Embedding (B, 512)
              - 'regime_probs': Soft regime probabilities (B, K=4)
              - 'anomaly': Predicted temperature anomaly (B, 15)
              - 'climatology_prior': Background prior used (B, 15)
        """
        # 1. Extract 512-D Ocean Embedding
        embedding = self.encode_to_embedding(x)  # (B, 512)

        # 2. Soft Regime Context (differentiable softmax)
        regime_logits = self.regime_head(embedding)  # (B, 4)
        regime_probs = F.softmax(regime_logits, dim=-1)  # (B, 4)

        # 3. Depth-Conditioned Decoding
        predicted_anomaly, logvar = self.decoder(embedding, regime_probs)  # (B, 15), (B, 15)

        # 4. Numerical safety for uncertainty
        clamped_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        variance = torch.exp(clamped_logvar)
        sigma = torch.sqrt(variance)

        # 5. Climatology-Aware Residual Recombination
        if climatology_prior is not None:
            clim = climatology_prior
        else:
            # Fallback baseline constant prior ~12°C mean if not supplied
            clim = torch.full_like(predicted_anomaly, 12.0)

        temperature_mean = clim + predicted_anomaly

        return {
            "temperature_mean": temperature_mean,
            "log_variance": clamped_logvar,
            "variance": variance,
            "uncertainty_sigma": sigma,
            "embedding": embedding,
            "regime_probs": regime_probs,
            "anomaly": predicted_anomaly,
            "climatology_prior": clim,
        }
