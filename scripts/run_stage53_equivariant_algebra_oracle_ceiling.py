from pathlib import Path

from lcp_rbffd.experiments.stage53_equivariant_algebra_oracle_ceiling import run_stage53


if __name__ == "__main__":
    print(run_stage53(Path(__file__).resolve().parents[1]))

