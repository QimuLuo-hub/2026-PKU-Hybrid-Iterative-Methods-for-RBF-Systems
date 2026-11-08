from pathlib import Path

from lcp_rbffd.experiments.stage44_rank_retention_mechanism import run_stage44


if __name__ == "__main__":
    print(run_stage44(Path(__file__).resolve().parents[1]))

