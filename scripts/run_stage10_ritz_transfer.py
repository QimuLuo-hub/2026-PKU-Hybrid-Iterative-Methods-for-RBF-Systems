#!/usr/bin/env python3
"""Run Stage 10 Ritz transfer and solver-critical mode tracking."""

from pathlib import Path

from lcp_rbffd.experiments.stage10_ritz_transfer import run_stage10


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    print(
        run_stage10(
            repository / "results" / "stage10_ritz_transfer_and_mode_tracking"
        )
    )
