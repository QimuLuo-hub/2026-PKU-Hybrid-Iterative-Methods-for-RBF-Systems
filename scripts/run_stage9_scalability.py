#!/usr/bin/env python3
"""Run Stage 9 hard-regime discovery from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage9_scalability import run_stage9


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_stage9(
        root / "results" / "stage9_hard_regime_discovery_and_scalability"
    )
    print(result)
