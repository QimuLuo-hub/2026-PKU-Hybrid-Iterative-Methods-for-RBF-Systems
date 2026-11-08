"""Run Stage 27 independent validation from the repository root."""

from pathlib import Path

from lcp_rbffd.experiments.stage27_spatial_k_adaptive_rank_validation import run_stage27


if __name__ == "__main__":
    print(run_stage27(Path(__file__).resolve().parents[1]))

