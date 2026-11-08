from pathlib import Path

from lcp_rbffd.experiments.stage31_krylov_aware_shared_basis import run_stage31


if __name__ == "__main__":
    print(run_stage31(Path(__file__).resolve().parents[1]))
