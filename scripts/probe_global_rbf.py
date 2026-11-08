"""Small exploratory scan used to select numerically valid Stage-1 regimes."""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import scipy.linalg as la

from global_rbf.collocation import assemble_poisson_system, matrix_diagnostics
from global_rbf.nodes import square_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=20)
    args = parser.parse_args()
    for distribution in ("tensor", "jittered"):
        for kernel in ("gaussian", "imq"):
            for side in (10, 15, 20, 25, 30):
                if side > args.max_side:
                    continue
                nodes, boundary = square_nodes(side, distribution, seed=20260830 + side)
                for epsilon in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
                    system = assemble_poisson_system(nodes, boundary, kernel=kernel, epsilon=epsilon)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", la.LinAlgWarning)
                        try:
                            x, seconds = system.solve_reference()
                            metrics = system.error_metrics(x, evaluation_side=31)
                        except Exception as exc:
                            print(distribution, kernel, side**2, epsilon, "FAIL", repr(exc), flush=True)
                            continue
                    diag = matrix_diagnostics(system.A, exact_svd=side <= 20)
                    print(
                        distribution,
                        kernel,
                        side**2,
                        epsilon,
                        f"grid={metrics['grid_rel_l2']:.3e}",
                        f"res={metrics['relative_residual']:.3e}",
                        f"cond={diag.get('condition_2', float('nan')):.3e}",
                        f"solve={seconds:.3e}",
                        f"warn={len(caught)}",
                        flush=True,
                    )
        for side in (10, 15, 20, 25, 30):
            if side > args.max_side:
                continue
            nodes, boundary = square_nodes(side, distribution, seed=20260830 + side)
            system = assemble_poisson_system(nodes, boundary, kernel="phs3", epsilon=None)
            try:
                x, seconds = system.solve_reference()
                metrics = system.error_metrics(x, evaluation_side=31)
                diag = matrix_diagnostics(system.A, exact_svd=side <= 20)
                print(
                    distribution,
                    "phs3",
                    side**2,
                    "none",
                    f"grid={metrics['grid_rel_l2']:.3e}",
                    f"res={metrics['relative_residual']:.3e}",
                    f"cond={diag.get('condition_2', float('nan')):.3e}",
                    f"solve={seconds:.3e}",
                    flush=True,
                )
            except Exception as exc:
                print(distribution, "phs3", side**2, "none", "FAIL", repr(exc), flush=True)


if __name__ == "__main__":
    main()
