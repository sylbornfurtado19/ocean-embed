"""Training entry point for OceanEmbed.

TODO: implement the training script, configuration loading, and optimization loop.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse training configuration path and runtime arguments."""
    parser = argparse.ArgumentParser(description="Train the OceanEmbed model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def train_model(config_path: str) -> None:
    """Load a config and run the training loop for the configured experiment."""
    # TODO: read YAML config and initialize data loaders/model/trainer.
    raise NotImplementedError("TODO: implement training loop.")


def main() -> None:
    """Entry point for model training."""
    args = parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
