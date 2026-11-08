"""Probe correction POD and oracle spaces on one selected valid system."""

from __future__ import annotations

import argparse

import numpy as np

from global_rbf.collocation import assemble_poisson_system
from global_rbf.krylov import fgmres
from global_rbf.nodes import square_nodes
from global_rbf.preconditioners import low_precision_lu_preconditioner
from global_rbf.structure import (
    build_correction_data,
    harmonic_ritz_space,
    oracle_coarse_preconditioner,
    pod_metrics,
    residual_history_space,
    smallest_right_singular_space,
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
    base = low_precision_lu_preconditioner(system.A)
    trajectory = fgmres(
        system.A,
        system.b,
        base.apply,
        rtol=1e-11,
        restart=min(80, system.dimension),
        max_iterations=240,
        capture_residual_vectors=True,
        capture_arnoldi=True,
    )
    rng = np.random.default_rng(419)
    train = rng.standard_normal((system.dimension, 2 * system.dimension))
    if trajectory.residual_vectors:
        train = np.column_stack([train, *trajectory.residual_vectors])
    test = rng.standard_normal((system.dimension, system.dimension))
    data = build_correction_data(system.A, base, train_residuals=train, test_residuals=test)
    for row in pod_metrics(data, [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]):
        print(row, flush=True)
    for q in (4, 8, 12, 16, 24):
        spaces = {
            "correction_pod": data.pod_basis[:, :q],
            "residual_history": residual_history_space(trajectory.residual_vectors, base, q),
            "harmonic_ritz": harmonic_ritz_space(trajectory, q),
            "small_singular": smallest_right_singular_space(system.A, q),
        }
        for name, Z in spaces.items():
            if not Z.size:
                continue
            hybrid = oracle_coarse_preconditioner(
                system.A,
                base,
                Z,
                name=f"{name}_q{q}",
                composition="residual_complement",
            )
            result = fgmres(system.A, system.b, hybrid.apply, rtol=1e-10, restart=80, max_iterations=240)
            print(name, q, Z.shape[1], result.iterations, result.converged, result.relative_residual, flush=True)


if __name__ == "__main__":
    main()
