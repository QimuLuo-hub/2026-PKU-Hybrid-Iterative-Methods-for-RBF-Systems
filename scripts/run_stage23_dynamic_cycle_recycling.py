#!/usr/bin/env python3
"""Run the preregistered Stage-23 dynamic recycle-space diagnostic."""

from pathlib import Path

from lcp_rbffd.experiments.stage23_dynamic_cycle_recycling import run_stage23


if __name__ == "__main__":
    print(run_stage23(Path(__file__).resolve().parents[1]))
