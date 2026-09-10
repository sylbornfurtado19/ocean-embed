"""Calibration diagnostics for uncertainty-aware OceanEmbed predictions.

A model is considered well calibrated when the empirical coverage of its
confidence intervals matches the nominal confidence level. For example, a 90%
interval should contain the true target roughly 90% of the time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_calibration_curve(predictions: np.ndarray, targets: np.ndarray, predicted_variance: np.ndarray) -> dict:
    """Bucket samples by predicted confidence and compare empirical coverage against nominal confidence."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    var = np.asarray(predicted_variance, dtype=np.float64).reshape(-1)
    sigma = np.sqrt(np.clip(var, 1e-8, None))

    # Lower variance corresponds to higher confidence. This gives a simple scalar confidence metric
    # that can be binned for a reliability diagram.
    confidence = np.clip(np.exp(-sigma), 0.0, 1.0)
    bucket_edges = np.linspace(0.0, 1.0, 11)
    points: list[dict[str, float]] = []

    for idx in range(len(bucket_edges) - 1):
        low, high = bucket_edges[idx], bucket_edges[idx + 1]
        mask = (confidence >= low) & (confidence < high)
        if not np.any(mask):
            continue
        abs_err = np.abs(pred[mask] - y[mask])
        empirical_coverage = np.mean(abs_err <= 1.645 * sigma[mask])
        points.append(
            {
                "predicted_confidence": float((low + high) / 2.0),
                "empirical_coverage": float(empirical_coverage),
                "n_samples": int(mask.sum()),
            }
        )

    calibration_data = {
        "curve": points,
        "well_calibrated": bool(
            len(points) > 0 and np.all(np.abs(np.array([p["empirical_coverage"] for p in points]) - np.array([p["predicted_confidence"] for p in points])) <= 0.10)
        ),
    }
    return calibration_data


def plot_calibration(calibration_data: dict) -> plt.Figure:
    """Create the reliability diagram and save it to outputs/calibration_plot.png."""
    curve = calibration_data["curve"]
    if not curve:
        raise ValueError("Calibration curve is empty; no confidence buckets were produced.")

    x = np.array([point["predicted_confidence"] for point in curve], dtype=np.float64)
    y = np.array([point["empirical_coverage"] for point in curve], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Ideal calibration")
    ax.plot(x, y, marker="o", color="tab:blue", label="Observed coverage")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("OceanEmbed uncertainty calibration")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.1)
    ax.grid(alpha=0.2)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "calibration_plot.png", dpi=200, bbox_inches="tight")
    print(f"[CAL] Calibration plot saved to {out_dir / 'calibration_plot.png'}")
    return fig


def run_calibration_summary(checkpoint_path: str = "checkpoints/v1_uncertainty.pt", processed_path: str = "data/processed/bay_of_bengal.npz") -> dict:
    """Load a trained uncertainty-aware checkpoint and summarize calibration quality."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data = np.load(processed_path)
    X = data["X"]
    y = data["y"]

    # For a quick sanity check, compute predictions and a proxy variance on the validation split.
    from src.models.v1_uncertainty import OceanBaselineV1
    model = OceanBaselineV1(input_dim=X.shape[1], hidden_dim=128, output_dim=y.shape[1], dropout=0.1)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        mean, log_var = model(torch.tensor(X[:512], dtype=torch.float32))
        preds = mean.numpy()
        var = np.exp(log_var.numpy())

    calibration = compute_calibration_curve(preds, y[:512], var)
    plot_calibration(calibration)
    print(f"[CAL] Well calibrated? {calibration['well_calibrated']}")
    return calibration


if __name__ == "__main__":
    run_calibration_summary()
