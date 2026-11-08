from pathlib import Path

from lcp_rbffd.experiments.stage49_target_local_correction_ceiling import run_stage49


if __name__ == "__main__":
    print(run_stage49(Path(__file__).resolve().parents[1]))

