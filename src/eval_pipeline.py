"""Canonical evaluation runner for OceanEmbed baseline models.

Evaluates trained checkpoints on held-out validation samples and computes:
1. Overall RMSE, MAE, and Bias.
2. Depth-wise metrics across all 15 standardized OceanEmbed depth levels.
3. Uncertainty calibration (reliability curves & 90% coverage) for uncertainty models.
4. Independent ARGO validation if NetCDF files are present (or reports NOT AVAILABLE gracefully).

CRITICAL NOTICE:
Results evaluated on the synthetic demo dataset are for engineering validation only.
They do NOT represent real-world oceanographic accuracy or disaster prediction skill.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.eval.argo_validation import run_argo_validation
from src.eval.uncertainty_calibration import run_calibration_summary
from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1
from src.train import apply_norm, spatial_holdout_split


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed baseline models.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Project config path.")
    parser.add_argument("--model_version", type=str, choices=["v0", "v1_uncertainty"], default="v0", help="Model version to evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path.")
    parser.add_argument("--processed_path", type=str, default="data/processed/bay_of_bengal.npz", help="Processed dataset path.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_held_out_validation(
    checkpoint_path: str | Path,
    processed_path: str | Path = "data/processed/bay_of_bengal.npz",
) -> dict[str, Any]:
    """Compute overall and depth-wise RMSE, MAE, and Bias on the spatial holdout validation split."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    data = np.load(processed_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"].tolist()

    # Spatial holdout split: matches training protocol
    _, _, X_val_raw, y_val = spatial_holdout_split(X, y, feature_names)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_version = checkpoint.get("model_version", "v0")
    norm_stats = checkpoint.get("norm_stats")
    target_depths = checkpoint.get(
        "target_depths",
        [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0],
    )

    X_val = apply_norm(X_val_raw, norm_stats) if norm_stats else X_val_raw
    input_dim = checkpoint.get("input_dim", X_val.shape[1])
    output_dim = checkpoint.get("output_dim", y_val.shape[1])
    hidden_dim = checkpoint.get("hidden_dim", 128)
    dropout = checkpoint.get("dropout", 0.1)

    if model_version == "v1_uncertainty":
        model = OceanBaselineV1(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    else:
        model = OceanBaselineV0(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(X_val, dtype=torch.float32)
        if model_version == "v1_uncertainty":
            mean_t, _ = model(x_tensor)
            preds = mean_t.cpu().numpy()
        else:
            preds = model(x_tensor).cpu().numpy()

    errors = preds - y_val  # shape: (N_val, 15)
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))

    depth_metrics = []
    rmse_per_depth = np.sqrt(np.mean(errors ** 2, axis=0))
    mae_per_depth = np.mean(np.abs(errors), axis=0)
    bias_per_depth = np.mean(errors, axis=0)

    for idx, depth in enumerate(target_depths[: y_val.shape[1]]):
        depth_metrics.append(
            {
                "depth": float(depth),
                "rmse": float(rmse_per_depth[idx]),
                "mae": float(mae_per_depth[idx]),
                "bias": float(bias_per_depth[idx]),
            }
        )

    return {
        "model_version": model_version,
        "n_validation_samples": len(X_val),
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "depth_metrics": depth_metrics,
        "target_depths": list(target_depths),
    }


def main() -> None:
    """Run canonical evaluation."""
    args = parse_args()
    checkpoint_name = "v0_baseline.pt" if args.model_version == "v0" else "v1_uncertainty.pt"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("checkpoints") / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train it first with src/train.py --model_version {args.model_version}.")

    print("================================================================")
    print(f"       OCEANEMBED CANONICAL BASELINE EVALUATION ({args.model_version.upper()})")
    print("================================================================")
    print("DATASET TYPE: SYNTHETIC DEMO DATA (BAY OF BENGAL)")
    print("NOTICE: These results verify the software pipeline and mathematical contracts.")
    print("        They are NOT real ocean measurements and do not represent real-world skill.")
    print("----------------------------------------------------------------")

    # 1. Held-out validation split metrics
    val_res = evaluate_held_out_validation(checkpoint_path, args.processed_path)
    print(f"Validation Samples: {val_res['n_validation_samples']} (Contiguous Northeast Spatial Holdout)")
    print(f"Overall RMSE:       {val_res['overall_rmse']:.4f} °C")
    print(f"Overall MAE:        {val_res['overall_mae']:.4f} °C")
    print(f"Overall Bias:       {val_res['overall_bias']:.4f} °C\n")

    print("DEPTH-WISE METRICS:")
    print("| Depth (m) | RMSE (°C) | MAE (°C) | Bias (°C) |")
    print("|-----------|-----------|----------|-----------|")
    for row in val_res["depth_metrics"]:
        print(f"| {row['depth']:>9.1f} | {row['rmse']:>9.4f} | {row['mae']:>8.4f} | {row['bias']:>9.4f} |")
    print("----------------------------------------------------------------")

    # 2. Uncertainty Calibration (for v1_uncertainty)
    if args.model_version == "v1_uncertainty":
        print("\nUNCERTAINTY CALIBRATION (GAUSSIAN NLL HEAD):")
        cal_res = run_calibration_summary(str(checkpoint_path), args.processed_path)
        print(f"Confidence Interval: 90% (mean ± 1.645 * sigma)")
        print(f"Nominal Coverage:    0.9000")
        print(f"Observed Coverage:   {cal_res['nominal_90_coverage']:.4f}")
        print(f"Well Calibrated?:    {cal_res['well_calibrated']}")
        print(f"Calibration Diagram: {cal_res['plot_path']}")
        print("----------------------------------------------------------------")

    # 3. Independent ARGO Validation
    print("\nINDEPENDENT ARGO VALIDATION:")
    run_argo_validation(args.config, str(checkpoint_path))
    print("================================================================\n")


if __name__ == "__main__":
    main()
