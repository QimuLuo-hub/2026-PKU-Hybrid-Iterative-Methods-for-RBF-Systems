from pathlib import Path

from lcp_rbffd.experiments.stage39_direction_level_representation_ceiling import run_stage39


if __name__ == "__main__":
    print(run_stage39(Path(__file__).resolve().parents[1]))
