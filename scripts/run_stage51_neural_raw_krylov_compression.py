from pathlib import Path

from lcp_rbffd.experiments.stage51_neural_raw_krylov_compression import run_stage51


if __name__ == "__main__":
    print(run_stage51(Path(__file__).resolve().parents[1]))
