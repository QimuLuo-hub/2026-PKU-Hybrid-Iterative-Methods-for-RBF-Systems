from pathlib import Path

from lcp_rbffd.experiments.stage36_neural_donor_mixture import run_stage36


if __name__ == "__main__":
    print(run_stage36(Path(__file__).resolve().parents[1]))
