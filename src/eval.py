"""Combined evaluation runner for OceanEmbed."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.argo_validation import run_argo_validation
from src.eval.uncertainty_calibration import run_calibration_summary


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser(description="Run OceanEmbed validation and uncertainty analysis.")
    parser.add_argument("--config", type=str, default="configs/bay_of_bengal.yaml", help="Project config path.")
    parser.add_argument("--model_version", type=str, choices=["v0", "v1_uncertainty"], default="v0", help="Model version to evaluate.")
    return parser.parse_args()


def main() -> None:
    """Run ARGO validation and uncertainty calibration and print a summary report."""
    args = parse_args()
    checkpoint_name = "v0_baseline.pt" if args.model_version == "v0" else "v1_uncertainty.pt"
    checkpoint_path = Path("checkpoints") / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train it first with src/train.py --model_version {args.model_version}.")

    print(f"[EVAL] Evaluating model version: {args.model_version}")
    validation_summary = run_argo_validation(args.config, str(checkpoint_path))
    calibration_summary = run_calibration_summary(str(checkpoint_path)) if args.model_version == "v1_uncertainty" else {"well_calibrated": None}

    print("\n=== Combined evaluation summary ===")
    print(f"Overall RMSE: {validation_summary['overall_rmse']:.4f}")
    print(f"Overall MAE:  {validation_summary['overall_mae']:.4f}")
    print(f"Overall bias: {validation_summary['overall_bias']:.4f}")
    print(f"Samples:      {validation_summary['n_samples']}")
    if args.model_version == "v1_uncertainty":
        print(f"Calibration status: {calibration_summary['well_calibrated']}")


if __name__ == "__main__":
    main()
