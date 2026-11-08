from pathlib import Path

from lcp_rbffd.experiments.stage45_deployable_rank_stopping import run_stage45


if __name__ == "__main__":
    print(run_stage45(Path(__file__).resolve().parents[1]))

