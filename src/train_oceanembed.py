"""Training pipeline for OceanEmbed V2 Deep Learning Framework.

Trains the ConvLSTM + CBAM + 512-D Ocean Embedding + Depth-Conditioned Decoder model
on spatiotemporal surface sequences (B, T=31, C=5, H=32, W=32) and evaluates on the
contiguous spatial holdout split with Gaussian Negative Log-Likelihood loss.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.dataset import (
    OceanSpatiotemporalDataset,
    apply_channel_norm,
    compute_channel_norm_stats,
    load_spatiotemporal_data,
    spatiotemporal_spatial_holdout_split,
)
from src.models.oceanembed_v2 import OceanEmbedV2


def set_seed(seed: int = 42) -> None:
    """Set random seeds for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OceanEmbed V2 Spatiotemporal Architecture.")
    parser.add_argument("--data_path", type=str, default="data/processed/bay_of_bengal_spatiotemporal.npz")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Bottleneck ocean embedding dimension.")
    return parser.parse_args()


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss for mean and log-variance predictions."""
    clamped_logvar = torch.clamp(log_var, min=-10.0, max=10.0)
    var = torch.exp(clamped_logvar)
    return F.gaussian_nll_loss(mean, target, var, full=False, reduction="mean")


def train_oceanembed(
    data_path: str = "data/processed/bay_of_bengal_spatiotemporal.npz",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.001,
    seed: int = 42,
    embedding_dim: int = 512,
) -> Path:
    """Train OceanEmbed V2 model and save best checkpoint."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[OCEANEMBED V2 TRAIN] Starting on device: {device}")
    raw_data = load_spatiotemporal_data(data_path)
    train_data, val_data = spatiotemporal_spatial_holdout_split(raw_data, holdout_fraction=0.2)

    # Compute normalization strictly on training patches across (N, T, H, W)
    norm_stats = compute_channel_norm_stats(train_data["X"])
    X_train = apply_channel_norm(train_data["X"], norm_stats)
    X_val = apply_channel_norm(val_data["X"], norm_stats)

    train_ds = OceanSpatiotemporalDataset(X_train, train_data["y"], train_data["climatology"], train_data["coords"])
    val_ds = OceanSpatiotemporalDataset(X_val, val_data["y"], val_data["climatology"], val_data["coords"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[OCEANEMBED V2 TRAIN] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"[OCEANEMBED V2 TRAIN] Input shape: (B, T=31, C=5, H=32, W=32)")

    model = OceanEmbedV2(
        in_channels=5,
        temporal_length=31,
        spatial_patch_size=32,
        embedding_dim=embedding_dim,
        num_regimes=4,
        num_depths=15,
        target_depths=raw_data["target_depths"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OCEANEMBED V2 TRAIN] Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, clim_b, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            clim_b = clim_b.to(device)

            optimizer.zero_grad()
            out = model(xb, climatology_prior=clim_b)
            loss = gaussian_nll_loss(out["temperature_mean"], out["log_variance"], yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, clim_b, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                clim_b = clim_b.to(device)
                out = model(xb, climatology_prior=clim_b)
                v_loss = gaussian_nll_loss(out["temperature_mean"], out["log_variance"], yb)
                val_loss += v_loss.item() * xb.size(0)

        val_loss /= len(val_ds)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train GNLL: {train_loss:.6f} | Val GNLL: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "model_version": "oceanembed_v2",
                "model_state": model.state_dict(),
                "norm_stats": norm_stats,
                "embedding_dim": embedding_dim,
                "in_channels": 5,
                "temporal_length": 31,
                "spatial_patch_size": 32,
                "target_depths": raw_data["target_depths"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "seed": seed,
            }

    train_duration = time.time() - start_time
    print(f"[OCEANEMBED V2 TRAIN] Finished in {train_duration:.2f}s (Best Epoch: {best_epoch}, Best Val: {best_val_loss:.6f})")

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    out_checkpoint = checkpoints_dir / "oceanembed_v2.pt"
    torch.save(best_state, out_checkpoint)
    print(f"[OCEANEMBED V2 TRAIN] Checkpoint saved to: {out_checkpoint}")
    return out_checkpoint


def main() -> None:
    args = parse_args()
    train_oceanembed(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
    )


if __name__ == "__main__":
    main()
