from pathlib import Path

from lcp_rbffd.experiments.stage50_missing_tail_information_attribution import run_stage50


if __name__ == "__main__":
    print(run_stage50(Path(__file__).resolve().parents[1]))

