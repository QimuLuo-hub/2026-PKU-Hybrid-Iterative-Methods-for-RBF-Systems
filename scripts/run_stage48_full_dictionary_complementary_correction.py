from pathlib import Path

from lcp_rbffd.experiments.stage48_full_dictionary_complementary_correction import run_stage48


if __name__ == "__main__":
    print(run_stage48(Path(__file__).resolve().parents[1]))

