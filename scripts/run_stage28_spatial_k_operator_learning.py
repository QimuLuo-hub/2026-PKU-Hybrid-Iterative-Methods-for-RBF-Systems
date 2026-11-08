"""Run Stage 28 from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage28_spatial_k_operator_learning import run_stage28


if __name__ == "__main__":
    print(run_stage28(Path(__file__).resolve().parents[1]))

