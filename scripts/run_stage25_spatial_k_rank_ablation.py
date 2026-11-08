#!/usr/bin/env python3
"""Run the preregistered Stage-25 nested-rank oracle ablation."""

from pathlib import Path

from lcp_rbffd.experiments.stage25_spatial_k_rank_ablation import run_stage25


if __name__ == "__main__":
    print(run_stage25(Path(__file__).resolve().parents[1]))

