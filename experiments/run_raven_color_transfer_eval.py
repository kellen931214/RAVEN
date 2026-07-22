#!/usr/bin/env python3
"""Formal paper-faithful/aligned color-transfer evaluation entrypoint."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.run_raven_aligned_color_eval import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
