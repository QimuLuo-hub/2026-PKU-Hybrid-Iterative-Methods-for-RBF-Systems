#!/usr/bin/env python3
"""Run the preregistered Stage-26 adaptive-rank diagnostic."""

from pathlib import Path

from lcp_rbffd.experiments.stage26_spatial_k_adaptive_rank import run_stage26


if __name__ == "__main__":
    print(run_stage26(Path(__file__).resolve().parents[1]))

