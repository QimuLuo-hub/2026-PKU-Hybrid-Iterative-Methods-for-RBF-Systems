"""Probe dense preconditioner behavior before the full Stage-1 sweep."""

from __future__ import annotations

import argparse

from global_rbf.collocation import assemble_poisson_system
from global_rbf.krylov import fgmres
from global_rbf.nodes import square_nodes
from global_rbf.preconditioners import (
    dense_block_jacobi_preconditioner,
    global_residual_sparse_approximate_inverse,
    identity_preconditioner,
    jacobi_preconditioner,
    local_sparse_approximate_inverse,
    local_truncation_preconditioner,
    low_precision_lu_preconditioner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=15)
    parser.add_argument("--kernel", choices=["gaussian", "imq"], default="imq")
    parser.add_argument("--distribution", choices=["tensor", "jittered"], default="tensor")
    args = parser.parse_args()
    epsilon = (0.4 if args.kernel == "gaussian" else 0.2) * args.side
    nodes, boundary = square_nodes(args.side, args.distribution, seed=20260830 + args.side)
    system = assemble_poisson_system(nodes, boundary, kernel=args.kernel, epsilon=epsilon)
    builders = [
        lambda: identity_preconditioner(system.dimension),
        lambda: jacobi_preconditioner(system.A),
        lambda: dense_block_jacobi_preconditioner(system.A, nodes, blocks_per_axis=3),
        lambda: dense_block_jacobi_preconditioner(system.A, nodes, blocks_per_axis=5),
        lambda: local_truncation_preconditioner(system.A, nodes, neighbors=25),
        lambda: local_truncation_preconditioner(system.A, nodes, neighbors=49),
        lambda: local_sparse_approximate_inverse(system.A, nodes, support_size=25),
        lambda: global_residual_sparse_approximate_inverse(system.A, nodes, support_size=25),
        lambda: global_residual_sparse_approximate_inverse(system.A, nodes, support_size=49),
        lambda: low_precision_lu_preconditioner(system.A),
    ]
    for builder in builders:
        try:
            preconditioner = builder()
            result = fgmres(
                system.A,
                system.b,
                preconditioner.apply,
                rtol=1e-10,
                restart=80,
                max_iterations=240,
            )
            print(
                preconditioner.name,
                f"setup={preconditioner.setup_seconds:.3e}",
                f"apply={preconditioner.benchmark_application(system.dimension):.3e}",
                f"it={result.iterations}",
                f"conv={result.converged}",
                f"relres={result.relative_residual:.3e}",
                f"solve={result.solve_seconds:.3e}",
                flush=True,
            )
        except Exception as exc:
            print("FAIL", getattr(builder, "__name__", repr(builder)), repr(exc), flush=True)


if __name__ == "__main__":
    main()
