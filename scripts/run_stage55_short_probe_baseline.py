from pathlib import Path

from lcp_rbffd.experiments.stage55_short_probe_neural_selector import run_stage55


if __name__ == "__main__":
    print(run_stage55(Path(__file__).resolve().parents[1]))
