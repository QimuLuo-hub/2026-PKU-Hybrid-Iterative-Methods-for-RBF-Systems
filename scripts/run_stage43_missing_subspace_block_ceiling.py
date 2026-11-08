from pathlib import Path

from lcp_rbffd.experiments.stage43_missing_subspace_block_ceiling import run_stage43


if __name__ == "__main__":
    print(run_stage43(Path(__file__).resolve().parents[1]))
