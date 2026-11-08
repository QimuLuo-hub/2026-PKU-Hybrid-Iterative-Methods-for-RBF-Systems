from pathlib import Path

from lcp_rbffd.experiments.stage52_basis_equivariant_raw_compression import run_stage52


if __name__ == "__main__":
    print(run_stage52(Path(__file__).resolve().parents[1]))

