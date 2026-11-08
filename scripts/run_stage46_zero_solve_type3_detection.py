from pathlib import Path

from lcp_rbffd.experiments.stage46_zero_solve_type3_detection import run_stage46


if __name__ == "__main__":
    print(run_stage46(Path(__file__).resolve().parents[1]))

