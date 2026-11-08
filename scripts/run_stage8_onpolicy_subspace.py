#!/usr/bin/env python3
"""Run Stage 8 from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage8_onpolicy_subspace import run_stage8


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_stage8(
        root / "results" / "stage8_direct_onpolicy_and_subspace_stability",
        root / "results" / "stage7_solver_aware_mlp_vs_gnn",
    )
    print(result)
