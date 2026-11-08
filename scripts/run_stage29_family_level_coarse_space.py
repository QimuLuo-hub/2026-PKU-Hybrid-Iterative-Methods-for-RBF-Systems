from pathlib import Path

from lcp_rbffd.experiments.stage29_family_level_coarse_space import run_stage29


if __name__ == "__main__":
    print(run_stage29(Path(__file__).resolve().parents[1]))

