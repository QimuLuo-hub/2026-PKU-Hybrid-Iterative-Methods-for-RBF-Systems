#!/usr/bin/env python3
"""Run Stage 12 nonsymmetric coarse-coupling validation."""

from pathlib import Path

from lcp_rbffd.experiments.stage12_nonsymmetric_coupling import run_stage12


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    print(
        run_stage12(
            repository
            / "results"
            / "stage12_nonsymmetric_petrov_galerkin_recycling_validation"
        )
    )
