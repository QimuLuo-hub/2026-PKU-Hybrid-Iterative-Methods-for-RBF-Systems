from pathlib import Path

from lcp_rbffd.experiments.stage38_solver_response_guided_local_selection import run_stage38


if __name__ == "__main__":
    print(run_stage38(Path(__file__).resolve().parents[1]))
