#!/usr/bin/env python3
"""Workspace entry point for the reproducible Stage 1-5 study."""

from pathlib import Path
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lcp_rbffd.experiments.stages_1_5 import main


if __name__ == "__main__":
    main()

