#!/usr/bin/env python3
"""Run the controlled Stage 16 reduced-dictionary ablation."""

from pathlib import Path

from lcp_rbffd.experiments.stage16_w_aware_dictionary_ablation import run_stage16


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_stage16(root / "results" / "stage16_W_aware_dictionary_ablation")
