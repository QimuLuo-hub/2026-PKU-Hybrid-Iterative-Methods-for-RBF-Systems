from pathlib import Path

from lcp_rbffd.experiments.stage32_action_aware_rank_sweep import run_stage32


if __name__ == "__main__":
    print(run_stage32(Path(__file__).resolve().parents[1]))
