"""Canonical evaluation runner for OceanEmbed baseline and deep learning models.

Evaluates trained checkpoints on held-out validation samples and computes:
1. Overall RMSE, MAE, and Bias.
2. Depth-wise metrics across all 15 standardized OceanEmbed depth levels.
3. Uncertainty calibration (reliability curves & 90% coverage) for uncertainty models.
4. Independent ARGO validation check.
5. Comparative baseline reporting (V0 vs V1 vs OceanEmbed V2).

CRITICAL NOTICE:
All results evaluated on the synthetic demo datasets are for software verification
and engineering validation only. They do NOT represent real-world oceanic skill.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.dataset import (
    apply_channel_norm,
    load_spatiotemporal_data,
    spatiotemporal_spatial_holdout_split,
)
from src.eval.argo_validation import run_argo_validation
from src.eval.uncertainty_calibration import compute_calibration_curve, plot_calibration, run_calibration_summary
from src.models.oceanembed_v2 import OceanEmbedV2
from src.models.v0_baseline import OceanBaselineV0
from src.models.v1_uncertainty import OceanBaselineV1
from src.train import apply_norm, spatial_holdout_split


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed baseline and deep models.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Project config path.")
    parser.add_argument(
        "--model_version",
        type=str,
        choices=["v0", "v1_uncertainty", "oceanembed_v2", "compare_all"],
        default="oceanembed_v2",
        help="Model version to evaluate, or 'compare_all' for benchmark table.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path.")
    parser.add_argument("--processed_path", type=str, default=None, help="Custom dataset path.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_spatiotemporal_validation(
    checkpoint_path: str | Path = "checkpoints/oceanembed_v2.pt",
    data_path: str | Path = "data/processed/bay_of_bengal_spatiotemporal.npz",
) -> dict[str, Any]:
    """Evaluate OceanEmbed V2 on the spatiotemporal contiguous spatial holdout split."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"OceanEmbed V2 checkpoint not found: {checkpoint_path}")

    raw_data = load_spatiotemporal_data(data_path)
    _, val_data = spatiotemporal_spatial_holdout_split(raw_data, holdout_fraction=0.2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm_stats = checkpoint["norm_stats"]
    target_depths = checkpoint["target_depths"]

    X_val = apply_channel_norm(val_data["X"], norm_stats)
    y_val = val_data["y"]
    clim_val = val_data["climatology"]

    model = OceanEmbedV2(
        in_channels=checkpoint.get("in_channels", 5),
        temporal_length=checkpoint.get("temporal_length", 31),
        spatial_patch_size=checkpoint.get("spatial_patch_size", 32),
        embedding_dim=checkpoint.get("embedding_dim", 512),
        num_regimes=4,
        num_depths=len(target_depths),
        target_depths=target_depths,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(X_val, dtype=torch.float32)
        clim_t = torch.tensor(clim_val, dtype=torch.float32)
        out = model(x_t, climatology_prior=clim_t)
        preds = out["temperature_mean"].cpu().numpy()
        var = out["variance"].cpu().numpy()
        sigma = out["uncertainty_sigma"].cpu().numpy()
        embeddings = out["embedding"].cpu().numpy()

    errors = preds - y_val  # (N_val, 15)
    overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
    overall_mae = float(np.mean(np.abs(errors)))
    overall_bias = float(np.mean(errors))

    rmse_per_depth = np.sqrt(np.mean(errors ** 2, axis=0))
    mae_per_depth = np.mean(np.abs(errors), axis=0)
    bias_per_depth = np.mean(errors, axis=0)

    depth_metrics = []
    for idx, depth in enumerate(target_depths):
        depth_metrics.append(
            {
                "depth": float(depth),
                "rmse": float(rmse_per_depth[idx]),
                "mae": float(mae_per_depth[idx]),
                "bias": float(bias_per_depth[idx]),
            }
        )

    # Uncertainty calibration
    cal_data = compute_calibration_curve(preds, y_val, var)
    plot_path = plot_calibration(cal_data, "outputs/evaluation/oceanembed_v2_calibration.png")
    cal_data["plot_path"] = str(plot_path)

    return {
        "model_version": "oceanembed_v2",
        "n_validation_samples": len(X_val),
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "depth_metrics": depth_metrics,
        "target_depths": target_depths,
        "calibration": cal_data,
        "embeddings": embeddings,
    }


def evaluate_held_out_validation(
    checkpoint_path: str | Path,
    processed_path: str | Path = "data/processed/bay_of_bengal.npz",
) -> dict[str, Any]:
    """Evaluate baseline models (v0, v1) on flat spatial holdout split."""
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

    _, _, X_val_raw, y_val = spatial_holdout_split(X, y, feature_names)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_version = checkpoint.get("model_version", "v0")
    norm_stats = checkpoint.get("norm_stats")
    target_depths = checkpoint.get("target_depths")

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

    errors = preds - y_val
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


def compare_all_models() -> None:
    """Print comparative benchmark table across V0, V1, and OceanEmbed V2."""
    print("================================================================")
    print("     OCEANEMBED COMPREHENSIVE ARCHITECTURAL COMPARISON")
    print("================================================================")
    print("DATASET TYPE: SYNTHETIC DEVELOPMENT DATA ONLY")
    print("NOTICE: Engineering baseline comparison on held-out spatial test splits.")
    print("----------------------------------------------------------------\n")

    v0_res = evaluate_held_out_validation("checkpoints/v0_baseline.pt")
    v1_res = evaluate_held_out_validation("checkpoints/v1_uncertainty.pt")
    v2_res = evaluate_spatiotemporal_validation("checkpoints/oceanembed_v2.pt")

    print("| Model | Overall RMSE (°C) | Overall MAE (°C) | Overall Bias (°C) | Validation Samples |")
    print("|---|---|---|---|---|")
    print(f"| V0 (MSE Baseline) | {v0_res['overall_rmse']:.4f} | {v0_res['overall_mae']:.4f} | {v0_res['overall_bias']:+.4f} | {v0_res['n_validation_samples']} (1D points) |")
    print(f"| V1 (Gaussian Uncertainty) | {v1_res['overall_rmse']:.4f} | {v1_res['overall_mae']:.4f} | {v1_res['overall_bias']:+.4f} | {v1_res['n_validation_samples']} (1D points) |")
    print(f"| OceanEmbed V2 (ConvLSTM+CBAM+512D) | {v2_res['overall_rmse']:.4f} | {v2_res['overall_mae']:.4f} | {v2_res['overall_bias']:+.4f} | {v2_res['n_validation_samples']} (31x5x32x32 patches) |")
    print("\n----------------------------------------------------------------")


def main() -> None:
    args = parse_args()

    if args.model_version == "compare_all":
        compare_all_models()
        return

    if args.model_version == "oceanembed_v2":
        ckpt = args.checkpoint or "checkpoints/oceanembed_v2.pt"
        data_p = args.processed_path or "data/processed/bay_of_bengal_spatiotemporal.npz"

        print("================================================================")
        print("      CANONICAL EVALUATION: OCEANEMBED V2 (DEEP ARCHITECTURE)")
        print("================================================================")
        print("DATASET TYPE: SYNTHETIC SPATIOTEMPORAL DEMO DATA (BAY OF BENGAL)")
        print("INPUT TENSOR: (B, T=31, C=5, H=32, W=32)")
        print("----------------------------------------------------------------")

        res = evaluate_spatiotemporal_validation(ckpt, data_p)
        print(f"Validation Samples: {res['n_validation_samples']} (Contiguous Northeast Spatial Holdout)")
        print(f"Overall RMSE:       {res['overall_rmse']:.4f} °C")
        print(f"Overall MAE:        {res['overall_mae']:.4f} °C")
        print(f"Overall Bias:       {res['overall_bias']:.4f} °C\n")

        print("DEPTH-WISE METRICS:")
        print("| Depth (m) | RMSE (°C) | MAE (°C) | Bias (°C) |")
        print("|-----------|-----------|----------|-----------|")
        for row in res["depth_metrics"]:
            print(f"| {row['depth']:>9.1f} | {row['rmse']:>9.4f} | {row['mae']:>8.4f} | {row['bias']:>9.4f} |")
        print("----------------------------------------------------------------")

        print("\nUNCERTAINTY CALIBRATION (GAUSSIAN NLL HEAD):")
        cal = res["calibration"]
        print(f"Confidence Interval: 90% (mean ± 1.645 * sigma)")
        print(f"Nominal Coverage:    0.9000")
        print(f"Observed Coverage:   {cal['nominal_90_coverage']:.4f}")
        print(f"Well Calibrated?:    {cal['well_calibrated']}")
        print(f"Calibration Plot:    {cal['plot_path']}")
        print("----------------------------------------------------------------")

        print("\nINDEPENDENT ARGO VALIDATION:")
        run_argo_validation(args.config, ckpt)
        print("================================================================\n")
        return

    # Baseline evaluation fallback (v0 or v1_uncertainty)
    ckpt = args.checkpoint or ("checkpoints/v0_baseline.pt" if args.model_version == "v0" else "checkpoints/v1_uncertainty.pt")
    data_p = args.processed_path or "data/processed/bay_of_bengal.npz"

    print("================================================================")
    print(f"       OCEANEMBED CANONICAL BASELINE EVALUATION ({args.model_version.upper()})")
    print("================================================================")
    print("DATASET TYPE: SYNTHETIC DEMO DATA (BAY OF BENGAL)")
    print("----------------------------------------------------------------")

    val_res = evaluate_held_out_validation(ckpt, data_p)
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

    if args.model_version == "v1_uncertainty":
        print("\nUNCERTAINTY CALIBRATION (GAUSSIAN NLL HEAD):")
        cal_res = run_calibration_summary(str(ckpt), data_p)
        print(f"Nominal Coverage:    0.9000")
        print(f"Observed Coverage:   {cal_res['nominal_90_coverage']:.4f}")
        print(f"Well Calibrated?:    {cal_res['well_calibrated']}")
        print(f"Calibration Diagram: {cal_res['plot_path']}")
        print("----------------------------------------------------------------")

    print("\nINDEPENDENT ARGO VALIDATION:")
    run_argo_validation(args.config, str(ckpt))
    print("================================================================\n")


if __name__ == "__main__":
    main()
