from pathlib import Path

from lcp_rbffd.experiments.stage41_multi_rhs_slow_mode_target import run_stage41


if __name__ == "__main__":
    print(run_stage41(Path(__file__).resolve().parents[1]))
