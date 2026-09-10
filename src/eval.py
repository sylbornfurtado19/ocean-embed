"""Evaluation entry point for OceanEmbed.

TODO: implement spatial and temporal holdout evaluation for model quality assessment.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration and runtime arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the OceanEmbed model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def evaluate_model(config_path: str) -> None:
    """Run validation and holdout evaluation against ARGO and reanalysis targets."""
    # TODO: load model, run evaluation, and report metrics by depth and region.
    raise NotImplementedError("TODO: implement evaluation pipeline.")


def main() -> None:
    """Entry point for model evaluation."""
    args = parse_args()
    evaluate_model(args.config)


if __name__ == "__main__":
    main()
