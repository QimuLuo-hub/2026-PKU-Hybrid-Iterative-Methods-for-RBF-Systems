#!/usr/bin/env python3
"""Run the Stage 7 solver-aware MLP/GNN diagnostic."""

from pathlib import Path

from lcp_rbffd.experiments.stage7_solver_aware import run


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = run(root / "results" / "stage7_solver_aware_mlp_vs_gnn")
    print(report)
