"""Training entry point for OceanEmbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1


def parse_args() -> argparse.Namespace:
    """Parse the YAML config path and model version."""
    parser = argparse.ArgumentParser(description="Train the OceanEmbed baseline model.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_version", type=str, choices=["v0", "v1_uncertainty"], default="v0", help="Model version to train.")
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


def spatial_holdout_split(X: np.ndarray, y: np.ndarray, feature_names: list[str], holdout_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian negative log-likelihood loss for mean/log-variance predictions."""
    return F.gaussian_nll_loss(mean, target, torch.exp(log_var), full=False, reduction="mean")


def get_model(model_version: str, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    if model_version == "v0":
        return OceanBaselineV0(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    if model_version == "v1_uncertainty":
        return OceanBaselineV1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    raise ValueError(f"Unsupported model version: {model_version}")


def train_model(config_path: str, model_version: str = "v0") -> None:
    """Train either the v0 MSE baseline or the v1 uncertainty-aware baseline."""
    config = load_config(config_path)
    training_cfg = config["training"]
    model_cfg = config["model"]

    processed_path = Path("data/processed/bay_of_bengal.npz")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed arrays not found at {processed_path}. Run src/data/prepare.py first.")

    X, y, feature_names = load_processed_data(str(processed_path))
    X_train, y_train, X_val, y_val = spatial_holdout_split(X, y, feature_names, holdout_fraction=config["validation"].get("holdout_fraction", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_version, X_train.shape[1], model_cfg.get("hidden_dim", 128), y_train.shape[1], model_cfg.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=int(training_cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(training_cfg["batch_size"]), shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(int(training_cfg["epochs"])):
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
        print(f"Epoch {epoch + 1:02d}/{training_cfg['epochs']}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {"model_state": model.state_dict(), "config": config, "feature_names": feature_names}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name = "v0_baseline.pt" if model_version == "v0" else "v1_uncertainty.pt"
    torch.save(best_state, checkpoint_dir / checkpoint_name)
    print(f"[TRAIN] Best checkpoint saved to {checkpoint_dir / checkpoint_name}")


def main() -> None:
    """Entry point for training the Bay of Bengal baseline."""
    args = parse_args()
    train_model(args.config, args.model_version)


if __name__ == "__main__":
    main()
