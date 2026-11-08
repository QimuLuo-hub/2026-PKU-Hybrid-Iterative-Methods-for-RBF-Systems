from pathlib import Path

from lcp_rbffd.experiments.stage42_missing_critical_mode_diagnosis import run_stage42


if __name__ == "__main__":
    print(run_stage42(Path(__file__).resolve().parents[1]))
