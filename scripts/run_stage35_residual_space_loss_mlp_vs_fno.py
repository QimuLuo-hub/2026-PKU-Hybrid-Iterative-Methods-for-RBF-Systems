from pathlib import Path

from lcp_rbffd.experiments.stage35_residual_space_loss_mlp_vs_fno import run_stage35


if __name__ == "__main__":
    print(run_stage35(Path(__file__).resolve().parents[1]))
