"""Evaluation utilities for OceanEmbed."""

# Exports argo_validation and uncertainty_calibration submodules
from src.eval.argo_validation import run_argo_validation
from src.eval.uncertainty_calibration import run_calibration_summary

__all__ = ["run_argo_validation", "run_calibration_summary"]
