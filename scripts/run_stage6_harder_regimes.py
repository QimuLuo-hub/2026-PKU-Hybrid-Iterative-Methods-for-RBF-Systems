#!/usr/bin/env python3
"""Run the harder-regime, subspace, and direct-complement A--F study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lcp_rbffd.experiments.stage6_harder_regimes import run_stage6_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute required Stages A--F and write the Stage 6 report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage6_harder_regimes_and_direct_ml",
        help="Output directory for raw data, plots, model, and report.",
    )
    args = parser.parse_args()
    run_stage6_experiment(args.output)


if __name__ == "__main__":
    main()

