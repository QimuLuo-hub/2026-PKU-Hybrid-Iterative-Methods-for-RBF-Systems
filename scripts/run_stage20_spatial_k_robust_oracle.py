#!/usr/bin/env python3
"""Run the Stage 20 spatial-K robust-oracle diagnostic."""

from pathlib import Path

from lcp_rbffd.experiments.stage20_spatial_k_robust_oracle import run_stage20


if __name__ == "__main__":
    run_stage20(Path(__file__).resolve().parents[1])
