"""Training entry point for OceanEmbed."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse the YAML config path and model version."""
    parser = argparse.ArgumentParser(description="Train the OceanEmbed baseline model.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_version", type=str, choices=["v0", "v1_uncertainty"], default="v0", help="Model version to train.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_processed_data(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()
    return X, y, feature_names


def spatial_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a contiguous spatial validation split using the lat/lon feature columns."""
    lat_idx = feature_names.index("lat")
    lon_idx = feature_names.index("lon")
    lat_mid = np.median(X[:, lat_idx])
    lon_mid = np.median(X[:, lon_idx])
    val_mask = (X[:, lat_idx] >= lat_mid) & (X[:, lon_idx] >= lon_mid)
    if val_mask.mean() < 0.05:
        val_mask = np.zeros_like(val_mask, dtype=bool)
        val_count = max(1, int(len(X) * holdout_fraction))
        val_mask[:val_count] = True
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]


def compute_norm_stats(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean and std normalization statistics strictly from training data."""
    mean = np.mean(X_train, axis=0).astype(np.float32)
    std = np.std(X_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}


def apply_norm(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize input features using precomputed training statistics."""
    return (X - stats["mean"]) / stats["std"]


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian negative log-likelihood loss for mean/log-variance predictions with stability clamp."""
    clamped_logvar = torch.clamp(log_var, min=-10.0, max=10.0)
    var = torch.exp(clamped_logvar)
    return F.gaussian_nll_loss(mean, target, var, full=False, reduction="mean")


def get_model(model_version: str, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    if model_version == "v0":
        return OceanBaselineV0(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    if model_version == "v1_uncertainty":
        return OceanBaselineV1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    raise ValueError(f"Unsupported model version: {model_version}")


def train_model(config_path: str, model_version: str = "v0", override_epochs: int | None = None) -> Path:
    """Train either the v0 MSE baseline or the v1 uncertainty-aware baseline."""
    config = load_config(config_path)
    training_cfg = config["training"]
    model_cfg = config["model"]
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    processed_path = Path("data/processed/bay_of_bengal.npz")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed arrays not found at {processed_path}. Run src/data/prepare.py first.")

    X, y, feature_names = load_processed_data(str(processed_path))
    X_train_raw, y_train, X_val_raw, y_val = spatial_holdout_split(
        X, y, feature_names, holdout_fraction=config["validation"].get("holdout_fraction", 0.2)
    )

    # TRAIN-ONLY feature normalization
    norm_stats = compute_norm_stats(X_train_raw)
    X_train = apply_norm(X_train_raw, norm_stats)
    X_val = apply_norm(X_val_raw, norm_stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.1))
    model = get_model(model_version, X_train.shape[1], hidden_dim, y_train.shape[1], dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(training_cfg["batch_size"]), shuffle=False)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs = override_epochs if override_epochs is not None else int(training_cfg["epochs"])

    print(f"\n[TRAIN] Model version: {model_version} on {device}")
    print(f"[TRAIN] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"[TRAIN] Features: {X_train.shape[1]}, Target depths: {y_train.shape[1]}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TRAIN] Total trainable parameters: {param_count:,}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            if model_version == "v1_uncertainty":
                mean, log_var = model(xb)
                loss = gaussian_nll_loss(mean, log_var, yb)
            else:
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if model_version == "v1_uncertainty":
                    mean, log_var = model(xb)
                    loss = gaussian_nll_loss(mean, log_var, yb)
                else:
                    pred = model(xb)
                    loss = nn.MSELoss()(pred, yb)
                val_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        print(f"Epoch {epoch + 1:02d}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {
                "model_version": model_version,
                "model_state": model.state_dict(),
                "config": config,
                "feature_names": feature_names,
                "norm_stats": norm_stats,
                "target_depths": model_cfg.get("target_depths"),
                "input_dim": X_train.shape[1],
                "output_dim": y_train.shape[1],
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "seed": seed,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name = "v0_baseline.pt" if model_version == "v0" else "v1_uncertainty.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(best_state, checkpoint_path)
    print(f"[TRAIN] Best checkpoint saved to {checkpoint_path} (Best epoch: {best_epoch}, val_loss: {best_val:.6f})")
    return checkpoint_path


def main() -> None:
    """Entry point for training the Bay of Bengal baseline."""
    args = parse_args()
    train_model(args.config, args.model_version, args.epochs)


if __name__ == "__main__":
    main()
