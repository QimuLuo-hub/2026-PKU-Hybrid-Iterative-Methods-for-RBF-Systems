#!/usr/bin/env python3
"""Run Stage 14 solver-aware target/loss diagnostics."""

from pathlib import Path

from lcp_rbffd.experiments.stage14_solver_aware_targets import run_stage14


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    print(
        run_stage14(
            repository / "results" / "stage14_solver_aware_target_loss_diagnostics"
        )
    )

