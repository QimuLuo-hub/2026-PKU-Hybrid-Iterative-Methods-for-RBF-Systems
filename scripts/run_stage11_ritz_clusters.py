#!/usr/bin/env python3
"""Run Stage 11 Ritz-cluster stability experiments."""

from pathlib import Path

from lcp_rbffd.experiments.stage11_ritz_clusters import run_stage11


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    print(
        run_stage11(
            repository / "results" / "stage11_ritz_cluster_stability"
        )
    )
