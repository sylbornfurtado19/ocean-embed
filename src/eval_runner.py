"""Compatibility wrapper for evaluation.

NOTE: This script delegates to the canonical evaluation runner `src.eval.main()`.
Use `python src/eval.py --config configs/bay_of_bengal.yaml` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_pipeline import main

if __name__ == "__main__":
    print("[NOTE] src/eval_runner.py is a compatibility wrapper. Redirecting to canonical src/eval.py ...\n")
    main()
