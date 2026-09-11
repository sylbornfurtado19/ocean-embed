"""Calibration diagnostics for uncertainty-aware OceanEmbed predictions.

A model is considered well calibrated when the empirical coverage of its
confidence intervals matches the nominal confidence level. For example, a 90%
Gaussian interval (mean ± 1.645 * sigma) should contain the true target
roughly 90% of the time.

NOTICE: All calibration evaluations performed on synthetic demonstration data
are for engineering validation only and do NOT represent real-world oceanic calibration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.v1_uncertainty import OceanBaselineV1
from src.train import apply_norm, spatial_holdout_split


def compute_calibration_curve(predictions: np.ndarray, targets: np.ndarray, predicted_variance: np.ndarray) -> dict:
    """Bucket samples by predicted confidence and compare empirical coverage against nominal confidence."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    var = np.asarray(predicted_variance, dtype=np.float64).reshape(-1)
    sigma = np.sqrt(np.clip(var, 1e-8, None))

    # Overall nominal 90% coverage check across all (N * 15) points
    abs_err_all = np.abs(pred - y)
    overall_coverage_90 = float(np.mean(abs_err_all <= 1.645 * sigma))

    # Binning by confidence for reliability diagram
    confidence = np.clip(np.exp(-sigma), 0.0, 1.0)
    bucket_edges = np.linspace(0.0, 1.0, 11)
    points: list[dict[str, float]] = []

    for idx in range(len(bucket_edges) - 1):
        low, high = bucket_edges[idx], bucket_edges[idx + 1]
        mask = (confidence >= low) & (confidence < high)
        if not np.any(mask):
            continue
        abs_err = abs_err_all[mask]
        empirical_cov = float(np.mean(abs_err <= 1.645 * sigma[mask]))
        points.append(
            {
                "predicted_confidence": float((low + high) / 2.0),
                "empirical_coverage": empirical_cov,
                "n_samples": int(mask.sum()),
            }
        )

    well_calibrated = bool(
        len(points) > 0 and np.all(np.abs(np.array([p["empirical_coverage"] for p in points]) - np.array([p["predicted_confidence"] for p in points])) <= 0.15)
    )

    return {
        "curve": points,
        "nominal_90_coverage": overall_coverage_90,
        "well_calibrated": well_calibrated,
    }


def plot_calibration(calibration_data: dict, output_path: str | Path = "outputs/evaluation/calibration_plot.png") -> Path:
    """Create the reliability diagram and save it to the specified output path."""
    curve = calibration_data["curve"]
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Ideal calibration")

    if curve:
        x = np.array([point["predicted_confidence"] for point in curve], dtype=np.float64)
        y = np.array([point["empirical_coverage"] for point in curve], dtype=np.float64)
        ax.plot(x, y, marker="o", color="tab:blue", label="Observed coverage")

    ax.set_xlabel("Predicted confidence (Nominal)")
    ax.set_ylabel("Empirical coverage (Observed)")
    ax.set_title("OceanEmbed Uncertainty Calibration\n(Synthetic Demo Evaluation)")
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.1)
    ax.grid(alpha=0.2)

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CAL] Calibration plot saved to {out_file}")
    return out_file


def run_calibration_summary(
    checkpoint_path: str = "checkpoints/v1_uncertainty.pt",
    processed_path: str = "data/processed/bay_of_bengal.npz",
    output_plot: str = "outputs/evaluation/calibration_plot.png",
) -> dict:
    """Load a trained uncertainty-aware checkpoint and evaluate calibration on held-out validation samples."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = np.load(processed_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()

    # Use spatial holdout validation split
    _, _, X_val_raw, y_val = spatial_holdout_split(X, y, feature_names)

    # Normalize validation features using train normalization statistics
    norm_stats = checkpoint.get("norm_stats")
    X_val = apply_norm(X_val_raw, norm_stats) if norm_stats else X_val_raw

    model = OceanBaselineV1(
        input_dim=checkpoint.get("input_dim", X_val.shape[1]),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        output_dim=checkpoint.get("output_dim", y_val.shape[1]),
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        mean, log_var = model(torch.tensor(X_val, dtype=torch.float32))
        preds = mean.numpy()
        clamped_logvar = torch.clamp(log_var, -10.0, 10.0)
        var = torch.exp(clamped_logvar).numpy()

    calibration = compute_calibration_curve(preds, y_val, var)
    plot_file = plot_calibration(calibration, output_path=output_plot)
    calibration["plot_path"] = str(plot_file)

    print(f"[CAL] Validation samples evaluated: {len(X_val)}")
    print(f"[CAL] Nominal 90% coverage: 0.9000 | Observed coverage: {calibration['nominal_90_coverage']:.4f}")
    print(f"[CAL] Well calibrated? {calibration['well_calibrated']}")
    print("  NOTICE: Evaluated on synthetic demo data fixture. Not a real ocean validation.")
    return calibration


if __name__ == "__main__":
    run_calibration_summary()
