#!/usr/bin/env python3
"""Run the Stage 15 narrow-family W-target MLP feasibility experiment."""

from pathlib import Path

from lcp_rbffd.experiments.stage15_family_specific_w_target_mlp import run_stage15


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_stage15(root / "results" / "stage15_family_specific_W_target_MLP")
