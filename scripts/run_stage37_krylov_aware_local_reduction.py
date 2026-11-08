from pathlib import Path

from lcp_rbffd.experiments.stage37_krylov_aware_local_reduction import run_stage37


if __name__ == "__main__":
    print(run_stage37(Path(__file__).resolve().parents[1]))
