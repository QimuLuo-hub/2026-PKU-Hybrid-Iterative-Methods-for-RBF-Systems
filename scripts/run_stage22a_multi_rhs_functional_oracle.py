#!/usr/bin/env python3
"""Run checkpointed Stage 22A from the project root."""

from pathlib import Path

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import run_stage22a


if __name__ == "__main__":
    print(run_stage22a(Path(__file__).resolve().parents[1]))
