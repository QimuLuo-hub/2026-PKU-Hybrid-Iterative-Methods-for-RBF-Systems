from pathlib import Path

from lcp_rbffd.experiments.stage54_missing_operator_analysis import run_stage54


if __name__ == "__main__":
    print(run_stage54(Path(__file__).resolve().parents[1]))

