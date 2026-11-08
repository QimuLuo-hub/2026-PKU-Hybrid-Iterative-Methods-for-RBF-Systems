from pathlib import Path

from lcp_rbffd.experiments.stage34_neural_target_conditioned_selection import run_stage34


if __name__ == "__main__":
    print(run_stage34(Path(__file__).resolve().parents[1]))
