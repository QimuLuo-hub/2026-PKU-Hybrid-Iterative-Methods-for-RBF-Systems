#!/usr/bin/env python3
"""Run Stage 19A from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage19a_spatial_coefficient_oracle import run_stage19a


if __name__ == "__main__":
    run_stage19a(Path(__file__).resolve().parents[1])
