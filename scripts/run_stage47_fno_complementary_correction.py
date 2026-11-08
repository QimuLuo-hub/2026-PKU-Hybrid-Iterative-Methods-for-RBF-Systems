from pathlib import Path

from lcp_rbffd.experiments.stage47_fno_complementary_correction import run_stage47


if __name__ == "__main__":
    print(run_stage47(Path(__file__).resolve().parents[1]))

