"""Training entry point for OceanEmbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.v0_baseline import OceanBaselineV0


def parse_args() -> argparse.Namespace:
    """Parse the YAML config path for the training run."""
    parser = argparse.ArgumentParser(description="Train the OceanEmbed baseline model.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Path to YAML config file.")
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


def train_model(config_path: str) -> None:
    """Load config, build train/val splits, train the baseline model, and save the best checkpoint."""
    config = load_config(config_path)
    training_cfg = config["training"]
    model_cfg = config["model"]

    processed_path = Path("data/processed/bay_of_bengal.npz")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed arrays not found at {processed_path}. Run src/data/prepare.py first.")

    X, y, feature_names = load_processed_data(str(processed_path))
    X_train, y_train, X_val, y_val = spatial_holdout_split(X, y, feature_names, holdout_fraction=config["validation"].get("holdout_fraction", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OceanBaselineV0(input_dim=X_train.shape[1], hidden_dim=model_cfg.get("hidden_dim", 128), output_dim=y_train.shape[1], dropout=model_cfg.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    loss_fn = nn.MSELoss()

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
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)

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
    torch.save(best_state, checkpoint_dir / "v0_baseline.pt")
    print(f"[TRAIN] Best checkpoint saved to {checkpoint_dir / 'v0_baseline.pt'}")


def main() -> None:
    """Entry point for training the Bay of Bengal baseline."""
    args = parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
