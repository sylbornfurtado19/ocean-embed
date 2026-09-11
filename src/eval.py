"""Canonical CLI entry point for OceanEmbed evaluation.

Redirects directly to `src.eval_pipeline.main()`.
Can be called directly via:
  python src/eval.py --model_version v0
  python src/eval.py --model_version v1_uncertainty
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_pipeline import evaluate_held_out_validation, main

if __name__ == "__main__":
    main()
