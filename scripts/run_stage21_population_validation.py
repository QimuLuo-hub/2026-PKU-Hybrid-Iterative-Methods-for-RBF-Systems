#!/usr/bin/env python3
"""Run Stage 21 from the project root without changing frozen settings."""

from pathlib import Path

from lcp_rbffd.experiments.stage21_population_validation import run_stage21


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(run_stage21(root))

