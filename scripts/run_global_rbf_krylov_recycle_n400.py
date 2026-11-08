#!/usr/bin/env python3
"""Test exact harmonic-Ritz recycling on the validated N=400 RBF case."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import scipy

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import (
    GmresResult,
    build_augmentation,
    restarted_projected_gmres,
)


SIDE = 20
DIMENSION = SIDE**2
EPSILON = 8.0
LS_SUPPORT_SIZE = 81
RTOL = 1.0e-10
RESTART = 100
MAX_ITERATIONS = 1000
Q_VALUES = (4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_krylov_recycle_n400"),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _solve_repeated(
    matrix: np.ndarray,
    rhs: np.ndarray,
    apply_preconditioner,
    *,
    u_basis: np.ndarray | None,
    c_basis: np.ndarray | None,
    repeats: int,
    warmup: bool,
) -> tuple[GmresResult, float, list[float]]:
    kwargs = {
        "u_basis": u_basis,
        "c_basis": c_basis,
        "rtol": RTOL,
        "atol": 0.0,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
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
    residual_spread = float(np.ptp(residuals))
    residual_scale = max(float(np.max(np.abs(residuals))), 1.0)
    if residual_spread > 1.0e-13 * residual_scale:
        raise RuntimeError("deterministic repeats disagree on final residual")
    times = [float(run.solve_seconds) for run in runs]
    return runs[0], float(np.median(times)), times


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system = assemble_tensor_gaussian_poisson_blocked(SIDE, epsilon=EPSILON)
    if system.A.shape != (DIMENSION, DIMENSION):
        raise RuntimeError("validated benchmark assembly changed dimension")
    matrix = system.A
    rhs = system.b
    rhs_norm = float(np.linalg.norm(rhs))
    preconditioner = rbf_local_cardinal_preconditioner(
        matrix,
        system.nodes,
        support_size=LS_SUPPORT_SIZE,
    )

    extraction_start = perf_counter()
    baseline_trajectory = fgmres(
        matrix,
        rhs,
        preconditioner.apply,
        rtol=RTOL,
        atol=0.0,
        restart=RESTART,
        max_iterations=MAX_ITERATIONS,
        capture_arnoldi=True,
    )
    extraction_seconds = perf_counter() - extraction_start
    if not baseline_trajectory.converged:
        raise RuntimeError("validated N=400 extraction baseline did not converge")
    if not baseline_trajectory.arnoldi_cycles:
        raise RuntimeError("baseline extraction did not retain Arnoldi cycles")

    cycle_arrays: dict[str, np.ndarray] = {}
    cycle_rows: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(baseline_trajectory.arnoldi_cycles):
        prefix = f"cycle_{cycle_index:02d}"
        cycle_arrays[f"{prefix}_V"] = cycle.arnoldi_vectors
        cycle_arrays[f"{prefix}_Z"] = cycle.preconditioned_vectors
        cycle_arrays[f"{prefix}_H"] = cycle.hessenberg
        cycle_rows.append(
            {
                "cycle_index": cycle_index,
                "first_iteration": cycle.first_iteration,
                "krylov_dimension": cycle.hessenberg.shape[1],
                "V_key": f"{prefix}_V",
                "V_shape": str(tuple(cycle.arnoldi_vectors.shape)),
                "Z_key": f"{prefix}_Z",
                "Z_shape": str(tuple(cycle.preconditioned_vectors.shape)),
                "H_key": f"{prefix}_H",
                "H_shape": str(tuple(cycle.hessenberg.shape)),
            }
        )
    np.savez_compressed(args.output_dir / "baseline_krylov_cycles.npz", **cycle_arrays)
    _write_csv(args.output_dir / "baseline_krylov_cycle_index.csv", cycle_rows)
    _write_csv(
        args.output_dir / "baseline_extraction_residual_history.csv",
        [
            {"history_index": index, "relative_residual": value}
            for index, value in enumerate(baseline_trajectory.residual_history)
        ],
    )

    harmonic_spaces: dict[int, np.ndarray] = {}
    harmonic_construction_seconds: dict[int, float] = {}
    for q in Q_VALUES:
        start = perf_counter()
        basis = harmonic_ritz_space(baseline_trajectory, q)
        harmonic_construction_seconds[q] = perf_counter() - start
        if basis.shape != (DIMENSION, q):
            raise RuntimeError(
                f"harmonic-Ritz extraction returned {basis.shape}, expected {(DIMENSION, q)}"
            )
        harmonic_spaces[q] = basis

    random_start = perf_counter()
    random_max, _ = np.linalg.qr(
        np.random.default_rng(args.random_seed).standard_normal(
            (DIMENSION, max(Q_VALUES))
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

    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "rhs": rhs,
        "random_q16": random_max,
        "baseline_extraction_solution": baseline_trajectory.x,
    }
    baseline_iterations: int | None = None
    baseline_solve_seconds: float | None = None
    random_iterations: dict[int, int] = {}

    for space_name, q, candidate_basis in candidates:
        augmentation_seconds = 0.0
        if candidate_basis is None:
            u_basis = c_basis = None
            au_defect = ctc_defect = 0.0
        else:
            start = perf_counter()
            u_basis, c_basis = build_augmentation(matrix, candidate_basis)
            augmentation_seconds = perf_counter() - start
            au_defect = float(np.linalg.norm(matrix @ u_basis - c_basis, ord="fro"))
            ctc_defect = float(
                np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")
            )
            key = f"{space_name}_q{q}"
            arrays[f"{key}_Y"] = candidate_basis
            arrays[f"{key}_U"] = u_basis
            arrays[f"{key}_C"] = c_basis

        result, solve_seconds, solve_repeats = _solve_repeated(
            matrix,
            rhs,
            preconditioner.apply,
            u_basis=u_basis,
            c_basis=c_basis,
            repeats=args.repeats,
            warmup=not args.no_warmup,
        )
        true_relative_residual = float(
            np.linalg.norm(rhs - matrix @ result.x) / rhs_norm
        )
        if space_name == "baseline":
            baseline_iterations = result.iterations
            baseline_solve_seconds = solve_seconds
            if (
                baseline_trajectory.iterations != result.iterations
                or baseline_trajectory.converged != result.converged
            ):
                raise RuntimeError(
                    "Arnoldi capture changed the baseline iteration outcome"
                )
        if space_name == "random":
            random_iterations[q] = result.iterations
        assert baseline_iterations is not None
        assert baseline_solve_seconds is not None

        label = {
            "baseline": "Baseline FGMRES",
            "harmonic_ritz": f"Exact harmonic Ritz q={q}",
            "random": f"Random orthonormal q={q}",
        }[space_name]
        row = {
            "method": label,
            "recycle_space": space_name,
            "q": q,
            "converged": bool(result.converged and true_relative_residual <= RTOL),
            "iterations": result.iterations,
            "true_relative_residual": true_relative_residual,
            "solve_seconds": solve_seconds,
            "iteration_ratio": result.iterations / baseline_iterations,
            "solve_time_ratio": solve_seconds / baseline_solve_seconds,
            "restart_cycles": result.restart_cycles,
        }
        rows.append(row)
        arrays[f"solution_{space_name}_q{q}"] = result.x
        histories.extend(
            {
                "recycle_space": space_name,
                "q": q,
                "history_index": history_index,
                "relative_residual": relative_residual,
            }
            for history_index, relative_residual in enumerate(result.residual_history)
        )
        diagnostics.append(
            {
                "recycle_space": space_name,
                "q": q,
                "AU_minus_C_frobenius": au_defect if candidate_basis is not None else None,
                "CHC_minus_I_frobenius": ctc_defect if candidate_basis is not None else None,
                "augmentation_setup_seconds_excluded": augmentation_seconds,
                "solve_seconds_min": min(solve_repeats),
                "solve_seconds_median": solve_seconds,
                "solve_seconds_max": max(solve_repeats),
            }
        )
        print(
            f"{label:<29} converged={row['converged']} "
            f"iterations={result.iterations:4d} "
            f"relres={true_relative_residual:.6e} "
            f"solve={solve_seconds:.6f}s "
            f"iter_ratio={row['iteration_ratio']:.3f} "
            f"time_ratio={row['solve_time_ratio']:.3f}",
            flush=True,
        )

    harmonic_rows = [row for row in rows if row["recycle_space"] == "harmonic_ritz"]
    for row in harmonic_rows:
        q = int(row["q"])
        row["beats_matched_random_iterations"] = (
            bool(row["converged"]) and int(row["iterations"]) < random_iterations[q]
        )
        row["at_least_20_percent_iteration_reduction"] = (
            bool(row["converged"]) and float(row["iteration_ratio"]) <= 0.8
        )
        row["success"] = bool(
            row["beats_matched_random_iterations"]
            and row["at_least_20_percent_iteration_reduction"]
        )

    best_harmonic = min(
        harmonic_rows,
        key=lambda row: (
            not bool(row["converged"]),
            float(row["iteration_ratio"]),
            int(row["q"]),
        ),
    )
    success = any(bool(row["success"]) for row in harmonic_rows)

    _write_csv(args.output_dir / "benchmark.csv", rows)
    _write_csv(args.output_dir / "residual_history.csv", histories)
    _write_csv(args.output_dir / "augmentation_diagnostics.csv", diagnostics)
    np.savez_compressed(args.output_dir / "audit_arrays.npz", **arrays)

    configuration = {
        "side": SIDE,
        "N": DIMENSION,
        "epsilon": EPSILON,
        "epsilon_times_h": EPSILON / (SIDE - 1),
        "kernel": "global Gaussian RBF",
        "nodes": "tensor square grid",
        "pde": "2D Poisson, exact u=sin(pi x) sin(pi y), Dirichlet boundary rows",
        "preconditioner": "right LS-ACBF / local least-squares approximate-cardinal inverse",
        "ls_acbf_support_size": LS_SUPPORT_SIZE,
        "rtol": RTOL,
        "atol": 0.0,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
        "initial_guess": "zero vector for every solve",
        "timing_repeats": args.repeats,
        "timing_warmup": not args.no_warmup,
        "q_values": list(Q_VALUES),
        "random_seed": args.random_seed,
        "harmonic_ritz_definition": (
            "smallest-magnitude harmonic Ritz modes of the right-preconditioned "
            "baseline Arnoldi operator A M^-1, mapped through stored preconditioned "
            "Arnoldi vectors into coefficient-space directions"
        ),
        "recycle_integration": (
            "fixed-subspace projected Arnoldi: AY=CR, U=Y R^-1, AU=C, "
            "C^H C=I, and Arnoldi on (I-CC^H)A over right-preconditioned directions"
        ),
        "solve_time_definition": (
            "median warmed online projected-Arnoldi solve; excludes LS-ACBF setup, "
            "baseline extraction, harmonic/random construction, and AU=C construction"
        ),
        "success_rule": (
            "harmonic space converges, beats its matched random control in iterations, "
            "and has iteration ratio <= 0.8"
        ),
        "A_inverse_used": False,
        "bare_A_invariant_subspaces_used": False,
        "ML_used": False,
    }
    summary = {
        "configuration": configuration,
        "offline_costs_excluded": {
            "ls_acbf_setup_seconds": preconditioner.setup_seconds,
            "baseline_arnoldi_extraction_seconds": extraction_seconds,
            "harmonic_ritz_construction_seconds": {
                str(q): harmonic_construction_seconds[q] for q in Q_VALUES
            },
            "random_q16_construction_seconds": random_construction_seconds,
        },
        "baseline_extraction": {
            "converged": baseline_trajectory.converged,
            "iterations": baseline_trajectory.iterations,
            "true_relative_residual": baseline_trajectory.relative_residual,
            "restart_cycle_count": len(baseline_trajectory.arnoldi_cycles),
            "cycle_dimensions": [
                cycle.hessenberg.shape[1]
                for cycle in baseline_trajectory.arnoldi_cycles
            ],
        },
        "results": rows,
        "augmentation_diagnostics": diagnostics,
        "decision": {
            "success": success,
            "best_harmonic_q": int(best_harmonic["q"]),
            "best_harmonic_iteration_ratio": float(best_harmonic["iteration_ratio"]),
            "best_harmonic_iterations": int(best_harmonic["iterations"]),
            "proceed_to_ML": False,
            "reason": (
                "ML remains out of scope for this experiment"
                if success
                else "no tested harmonic-Ritz rank met the declared acceleration rule"
            ),
        },
        "saved_diagnostics": {
            "full_online_residual_histories": "residual_history.csv",
            "full_extraction_residual_history": "baseline_extraction_residual_history.csv",
            "all_baseline_arnoldi_cycles": "baseline_krylov_cycles.npz",
            "cycle_array_index": "baseline_krylov_cycle_index.csv",
            "solutions_and_recycle_bases": "audit_arrays.npz",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"decision: success={success}, best_harmonic_q={best_harmonic['q']}, "
        f"best_iteration_ratio={best_harmonic['iteration_ratio']:.3f}; "
        f"wrote {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
