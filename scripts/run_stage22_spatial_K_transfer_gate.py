#!/usr/bin/env python3
"""Run Stage 22 frozen cross-system transfer validation."""

from pathlib import Path

from lcp_rbffd.experiments.stage22_transfer_gate import run_stage22_transfer


if __name__ == "__main__":
    print(run_stage22_transfer(Path(__file__).resolve().parents[1]))
