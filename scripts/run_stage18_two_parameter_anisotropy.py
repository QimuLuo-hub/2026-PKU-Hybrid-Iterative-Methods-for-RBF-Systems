#!/usr/bin/env python3
"""Run Stage 18 two-parameter anisotropy feasibility study."""

from pathlib import Path

from lcp_rbffd.experiments.stage18_two_parameter_anisotropy import run_stage18


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_stage18(root / "results" / "stage18_two_parameter_anisotropy_family")
