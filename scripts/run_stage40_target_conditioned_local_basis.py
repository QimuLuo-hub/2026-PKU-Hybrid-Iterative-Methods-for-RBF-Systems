from pathlib import Path

from lcp_rbffd.experiments.stage40_target_conditioned_local_basis import run_stage40


if __name__ == "__main__":
    print(run_stage40(Path(__file__).resolve().parents[1]))
