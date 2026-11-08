#!/usr/bin/env python3
"""Run Stage 13 persistent Petrov versus fixed-space Krylov recycling."""

from pathlib import Path

from lcp_rbffd.experiments.stage13_persistent_vs_recycling import run_stage13


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    print(
        run_stage13(
            repository
            / "results"
            / "stage13_persistent_preconditioner_vs_krylov_recycling"
        )
    )
