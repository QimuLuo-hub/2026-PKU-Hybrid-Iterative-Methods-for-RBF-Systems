#!/usr/bin/env python3
"""Run the preregistered Stage-24 Stage-22A attribution experiment."""

from pathlib import Path

from lcp_rbffd.experiments.stage24_stage22a_attribution import run_stage24


if __name__ == "__main__":
    print(run_stage24(Path(__file__).resolve().parents[1]))
