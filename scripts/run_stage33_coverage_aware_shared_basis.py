from pathlib import Path

from lcp_rbffd.experiments.stage33_coverage_aware_shared_basis import run_stage33


if __name__ == "__main__":
    print(run_stage33(Path(__file__).resolve().parents[1]))
