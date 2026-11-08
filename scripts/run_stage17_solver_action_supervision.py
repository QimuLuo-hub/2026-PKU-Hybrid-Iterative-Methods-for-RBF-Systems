#!/usr/bin/env python3
"""Run Stage 17 solver-action supervision."""

from pathlib import Path

from lcp_rbffd.experiments.stage17_solver_action_supervision import run_stage17


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_stage17(root / "results" / "stage17_solver_action_supervision")
