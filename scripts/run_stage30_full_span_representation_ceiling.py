from pathlib import Path

from lcp_rbffd.experiments.stage30_full_span_representation_ceiling import run_stage30


if __name__ == "__main__":
    print(run_stage30(Path(__file__).resolve().parents[1]))
