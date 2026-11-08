#!/usr/bin/env python3
"""Run Stage 19C from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage19c_oracle_mechanism import run_stage19c


if __name__ == "__main__":
    run_stage19c(Path(__file__).resolve().parents[1])
