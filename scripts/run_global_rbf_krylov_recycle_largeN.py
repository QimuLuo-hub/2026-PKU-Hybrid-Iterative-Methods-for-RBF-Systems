#!/usr/bin/env python3
"""Test Krylov-derived harmonic-Ritz recycling on one large global-RBF case."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import scipy

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import jacobi_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import (
    GmresResult,
    build_augmentation,
    restarted_projected_gmres,
)


Q_VALUES = (4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=70)
    parser.add_argument("--epsilon-factor", type=float, default=0.4)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20240831)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_krylov_recycle_largeN"),
    )
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _solve_repeated(
    matrix: np.ndarray,
    rhs: np.ndarray,
    apply_preconditioner,
    *,
    u_basis: np.ndarray | None,
    c_basis: np.ndarray | None,
    rtol: float,
    restart: int,
    max_iterations: int,
    repeats: int,
    warmup: bool,
) -> tuple[GmresResult, float, list[float]]:
    kwargs = {
        "u_basis": u_basis,
        "c_basis": c_basis,
        "rtol": rtol,
        "restart": restart,
        "max_iterations": max_iterations,
    }
    if warmup:
        restarted_projected_gmres(matrix, rhs, apply_preconditioner, **kwargs)
    runs = [
        restarted_projected_gmres(matrix, rhs, apply_preconditioner, **kwargs)
        for _ in range(repeats)
    ]
    if len({run.iterations for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagree on iteration count")
    if len({run.converged for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagree on convergence")
    residuals = np.asarray([run.relative_residual for run in runs])
    if np.ptp(residuals) > 1.0e-13 * max(float(np.max(np.abs(residuals))), 1.0):
        raise RuntimeError("deterministic repeats disagree on final residual")
    times = [float(run.solve_seconds) for run in runs]
    return runs[0], float(np.median(times)), times


def _augmentation_diagnostics(
    matrix: np.ndarray,
    u_basis: np.ndarray,
    c_basis: np.ndarray,
) -> dict[str, float]:
    q = c_basis.shape[1]
    return {
        "AU_minus_C_frobenius": float(
            np.linalg.norm(matrix @ u_basis - c_basis, ord="fro")
        ),
        "CTC_minus_I_frobenius": float(
            np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")
        ),
    }


def main() -> int:
    args = parse_args()
    if args.side < 3:
        raise ValueError("side must be at least 3")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    epsilon = args.epsilon_factor * args.side
    system = assemble_tensor_gaussian_poisson_blocked(
        args.side,
        epsilon=epsilon,
    )
    matrix = system.A
    rhs = system.b
    dimension = matrix.shape[0]
    preconditioner = jacobi_preconditioner(matrix)

    extraction_start = perf_counter()
    baseline_trajectory = fgmres(
        matrix,
        rhs,
        preconditioner.apply,
        rtol=args.rtol,
        restart=args.restart,
        max_iterations=args.max_iterations,
        capture_arnoldi=True,
    )
    extraction_seconds = perf_counter() - extraction_start
    if not baseline_trajectory.arnoldi_cycles:
        raise RuntimeError("baseline trajectory did not retain Arnoldi data")

    harmonic_spaces: dict[int, np.ndarray] = {}
    harmonic_construction_seconds: dict[int, float] = {}
    for q in Q_VALUES:
        start = perf_counter()
        basis = harmonic_ritz_space(baseline_trajectory, q)
        harmonic_construction_seconds[q] = perf_counter() - start
        if basis.shape != (dimension, q):
            raise RuntimeError(
                f"harmonic-Ritz extraction returned {basis.shape}, expected {(dimension, q)}"
            )
        harmonic_spaces[q] = basis

    random_start = perf_counter()
    random_max, _ = np.linalg.qr(
        np.random.default_rng(args.random_seed).standard_normal(
            (dimension, max(Q_VALUES))
        ),
        mode="reduced",
    )
    random_construction_seconds = perf_counter() - random_start

    candidates: list[tuple[str, int, np.ndarray | None]] = [("baseline", 0, None)]
    for q in Q_VALUES:
        candidates.extend(
            [
                ("harmonic_ritz", q, harmonic_spaces[q]),
                ("random", q, random_max[:, :q]),
            ]
        )

    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {
        "rhs": rhs,
        "random_q16": random_max,
    }
    diagnostics: dict[str, dict[str, float]] = {}
    baseline_iterations: int | None = None

    for space_name, q, candidate_basis in candidates:
        augmentation_seconds = 0.0
        if candidate_basis is None:
            u_basis = c_basis = None
        else:
            start = perf_counter()
            u_basis, c_basis = build_augmentation(matrix, candidate_basis)
            augmentation_seconds = perf_counter() - start
            key = f"{space_name}_q{q}"
            arrays[f"{key}_Y"] = candidate_basis
            arrays[f"{key}_U"] = u_basis
            arrays[f"{key}_C"] = c_basis
            diagnostics[key] = _augmentation_diagnostics(matrix, u_basis, c_basis)

        result, solve_seconds, solve_repeats = _solve_repeated(
            matrix,
            rhs,
            preconditioner.apply,
            u_basis=u_basis,
            c_basis=c_basis,
            rtol=args.rtol,
            restart=args.restart,
            max_iterations=args.max_iterations,
            repeats=args.repeats,
            warmup=not args.no_warmup,
        )
        if space_name == "baseline":
            baseline_iterations = result.iterations
            if (
                baseline_trajectory.iterations != result.iterations
                or baseline_trajectory.converged != result.converged
            ):
                raise RuntimeError(
                    "capturing Arnoldi data changed the baseline iteration outcome"
                )
        assert baseline_iterations is not None
        label = {
            "baseline": "Baseline FGMRES",
            "harmonic_ritz": f"Exact harmonic Ritz q={q}",
            "random": f"Random orthonormal q={q}",
        }[space_name]
        row = {
            "method": label,
            "recycle_space": space_name,
            "q": q,
            "iterations": result.iterations,
            "true_relative_residual": result.relative_residual,
            "solve_seconds": solve_seconds,
            "iteration_ratio": result.iterations / baseline_iterations,
        }
        rows.append(row)
        arrays[f"solution_{space_name}_q{q}"] = result.x
        for history_index, relative_residual in enumerate(result.residual_history):
            histories.append(
                {
                    "recycle_space": space_name,
                    "q": q,
                    "history_index": history_index,
                    "relative_residual": relative_residual,
                }
            )
        diagnostics[f"timing_{space_name}_q{q}"] = {
            "augmentation_seconds_excluded": augmentation_seconds,
            "solve_seconds_min": float(min(solve_repeats)),
            "solve_seconds_max": float(max(solve_repeats)),
        }

    _write_csv(args.output_dir / "benchmark.csv", rows)
    _write_csv(args.output_dir / "residual_history.csv", histories)
    np.savez_compressed(args.output_dir / "audit_arrays.npz", **arrays)

    summary = {
        "configuration": {
            "side": args.side,
            "N": dimension,
            "epsilon": epsilon,
            "epsilon_times_h": epsilon / (args.side - 1),
            "pde": "2D Poisson, exact u=sin(pi x) sin(pi y), Dirichlet boundary rows",
            "kernel": "global Gaussian RBF",
            "preconditioner": "right Jacobi / coefficient-space diagonal scaling",
            "rtol": args.rtol,
            "atol": 0.0,
            "restart": args.restart,
            "max_iterations": args.max_iterations,
            "timing_repeats": args.repeats,
            "timing_warmup": not args.no_warmup,
            "q_values": list(Q_VALUES),
            "random_seed": args.random_seed,
            "solve_time_definition": "median warmed projected-Arnoldi solve; excludes shared Jacobi setup, baseline extraction, harmonic-Ritz construction, random-basis generation, and AU=C construction",
            "harmonic_ritz_definition": "smallest-magnitude harmonic Ritz modes of the right-preconditioned baseline Arnoldi operator A M^-1, mapped through the stored preconditioned Arnoldi vectors into coefficient-space candidate directions",
            "recycle_integration": "validated fixed-subspace NeurKItt/GCRO-DR formulation: AU=C, C^T C=I, Arnoldi on (I-CC^T)A over right-preconditioned directions",
            "bare_A_invariant_subspaces_used": False,
            "A_inverse_used": False,
            "ML_used": False,
        },
        "selection": {
            "reason": "N=4900 is the largest completed scaling case with an exact condition estimate; Jacobi has negligible setup and storage and is an exact linear right-preconditioner even though it did not cure the conditioning bottleneck.",
            "matrix_memory_bytes": int(matrix.nbytes),
            "condition_2_A_from_completed_scaling": 3.6339238664398896e12,
        },
        "construction_seconds_excluded": {
            "baseline_arnoldi_extraction": extraction_seconds,
            "harmonic_ritz": {
                str(q): harmonic_construction_seconds[q] for q in Q_VALUES
            },
            "random_q16": random_construction_seconds,
            "jacobi_setup": preconditioner.setup_seconds,
        },
        "baseline_trajectory": {
            "iterations": baseline_trajectory.iterations,
            "true_relative_residual": baseline_trajectory.relative_residual,
            "restart_cycles": len(baseline_trajectory.arnoldi_cycles),
            "last_cycle_dimension": int(
                baseline_trajectory.arnoldi_cycles[-1].hessenberg.shape[1]
            ),
        },
        "augmentation_diagnostics": diagnostics,
        "results": rows,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threading": "single-thread environment requested",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['method']:<29} "
            f"iterations={row['iterations']:4d} "
            f"residual={row['true_relative_residual']:.6e} "
            f"solve={row['solve_seconds']:.4f}s "
            f"ratio={row['iteration_ratio']:.3f}"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
