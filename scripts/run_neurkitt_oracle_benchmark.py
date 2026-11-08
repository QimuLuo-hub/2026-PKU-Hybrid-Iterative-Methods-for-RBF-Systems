#!/usr/bin/env python3
"""Run the oracle-only NeurKItt-style Darcy-flow benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import scipy
import scipy.sparse.linalg as spla

from neurkitt_oracle import (
    GmresResult,
    build_augmentation,
    build_darcy_system,
    restarted_projected_gmres,
)


VectorMap = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/neurkitt_oracle"),
    )
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def _build_preconditioner(
    name: str,
    matrix: scipy.sparse.spmatrix,
) -> tuple[VectorMap, float, dict[str, object]]:
    start = perf_counter()
    if name == "None":
        apply = lambda vector: vector.copy()
        details: dict[str, object] = {"kind": "identity"}
    elif name == "Jacobi":
        diagonal = matrix.diagonal()
        zero_diagonal_count = int(np.count_nonzero(diagonal == 0.0))
        if zero_diagonal_count:
            raise ValueError("Jacobi preconditioner encountered a zero diagonal")
        inverse_diagonal = 1.0 / diagonal
        apply = lambda vector: inverse_diagonal * vector
        details = {
            "kind": "right Jacobi",
            "zero_diagonal_count": zero_diagonal_count,
        }
    elif name == "ILU(0)":
        factor = spla.spilu(
            matrix.tocsc(),
            drop_tol=0.0,
            fill_factor=1.0,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"Equil": False},
        )
        apply = factor.solve
        details = {
            "kind": "right ILU(0)-style SuperLU",
            "drop_tol": 0.0,
            "fill_factor": 1.0,
            "permutation": "NATURAL",
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
        }
    else:
        raise ValueError(f"unknown preconditioner: {name}")
    return apply, perf_counter() - start, details


def _solve_repeated(
    matrix: scipy.sparse.spmatrix,
    rhs: np.ndarray,
    apply: VectorMap,
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
        restarted_projected_gmres(matrix, rhs, apply, **kwargs)
    runs = [
        restarted_projected_gmres(matrix, rhs, apply, **kwargs)
        for _ in range(repeats)
    ]
    iteration_counts = {run.iterations for run in runs}
    convergence_flags = {run.converged for run in runs}
    if len(iteration_counts) != 1 or len(convergence_flags) != 1:
        raise RuntimeError("repeated deterministic solves disagreed")
    times = [run.solve_seconds for run in runs]
    return runs[0], float(np.median(times)), times


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix, rhs, log_permeability, permeability = build_darcy_system(
        args.grid_size,
        seed=args.seed,
    )
    dimension = matrix.shape[0]
    if args.k >= dimension:
        raise ValueError("k must be smaller than the matrix dimension")

    eigen_start = perf_counter()
    eigenvalues, invariant_basis = spla.eigsh(
        matrix,
        k=args.k,
        sigma=0.0,
        which="LM",
        tol=1.0e-12,
        maxiter=50000,
        v0=np.full(dimension, 1.0 / np.sqrt(dimension)),
    )
    eigen_seconds = perf_counter() - eigen_start
    order = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    invariant_basis = invariant_basis[:, order]

    augmentation_start = perf_counter()
    u_basis, c_basis = build_augmentation(matrix, invariant_basis)
    augmentation_seconds = perf_counter() - augmentation_start

    eigen_residuals = np.linalg.norm(
        matrix @ invariant_basis - invariant_basis * eigenvalues[None, :],
        axis=0,
    )
    au_c_defect = float(np.linalg.norm(matrix @ u_basis - c_basis, ord="fro"))
    c_orthogonality_defect = float(
        np.linalg.norm(c_basis.T @ c_basis - np.eye(args.k), ord="fro")
    )
    symmetry_defect = float(
        spla.norm(matrix - matrix.T) / max(spla.norm(matrix), np.finfo(float).tiny)
    )

    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    preconditioner_metadata: dict[str, dict[str, object]] = {}
    baseline_by_pc: dict[str, dict[str, object]] = {}

    for preconditioner in ("None", "Jacobi", "ILU(0)"):
        apply, setup_seconds, details = _build_preconditioner(preconditioner, matrix)
        preconditioner_metadata[preconditioner] = {
            **details,
            "setup_seconds": setup_seconds,
        }

        for oracle in (False, True):
            result, krylov_median, krylov_times = _solve_repeated(
                matrix,
                rhs,
                apply,
                u_basis=u_basis if oracle else None,
                c_basis=c_basis if oracle else None,
                rtol=args.rtol,
                restart=args.restart,
                max_iterations=args.max_iterations,
                repeats=args.repeats,
                warmup=not args.no_warmup,
            )
            variant = (
                f"GMRES + {preconditioner} + oracle k={args.k}"
                if preconditioner != "None" and oracle
                else f"GMRES + oracle k={args.k}"
                if oracle
                else f"GMRES + {preconditioner}"
                if preconditioner != "None"
                else "GMRES"
            )
            total_solve_seconds = (
                setup_seconds
                + krylov_median
                + (augmentation_seconds if oracle else 0.0)
            )
            row: dict[str, object] = {
                "method": variant,
                "preconditioner": preconditioner,
                "oracle": oracle,
                "k": args.k if oracle else 0,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_true_relative_residual": result.relative_residual,
                "restart_cycles": result.restart_cycles,
                "preconditioner_setup_seconds": setup_seconds,
                "augmentation_setup_seconds": augmentation_seconds if oracle else 0.0,
                "krylov_seconds_median": krylov_median,
                "solve_time_seconds": total_solve_seconds,
                "iteration_ratio_oracle_over_baseline": None,
                "time_ratio_oracle_over_baseline": None,
            }
            if oracle:
                baseline = baseline_by_pc[preconditioner]
                row["iteration_ratio_oracle_over_baseline"] = (
                    result.iterations / int(baseline["iterations"])
                )
                row["time_ratio_oracle_over_baseline"] = (
                    total_solve_seconds / float(baseline["solve_time_seconds"])
                )
            else:
                baseline_by_pc[preconditioner] = row
            rows.append(row)

            for history_index, relative_residual in enumerate(result.residual_history):
                histories.append(
                    {
                        "method": variant,
                        "preconditioner": preconditioner,
                        "oracle": oracle,
                        "history_index": history_index,
                        "relative_residual": relative_residual,
                    }
                )
            row["krylov_seconds_repeats"] = krylov_times

    _write_csv(
        args.output_dir / "benchmark.csv",
        [
            {key: value for key, value in row.items() if key != "krylov_seconds_repeats"}
            for row in rows
        ],
    )
    _write_csv(args.output_dir / "residual_history.csv", histories)

    eigen_rows = [
        {
            "index": index + 1,
            "eigenvalue": float(value),
            "magnitude": float(abs(value)),
            "relative_eigen_residual": float(
                residual / max(abs(value), np.finfo(float).tiny)
            ),
        }
        for index, (value, residual) in enumerate(zip(eigenvalues, eigen_residuals))
    ]
    _write_csv(args.output_dir / "oracle_eigenpairs.csv", eigen_rows)

    summary = {
        "configuration": {
            "grid_size": args.grid_size,
            "interior_side": args.grid_size - 2,
            "matrix_dimension": dimension,
            "matrix_nnz": int(matrix.nnz),
            "seed": args.seed,
            "grf_alpha": 2.0,
            "grf_tau": 3.0,
            "forcing": 1.0,
            "boundary_condition": "zero Dirichlet",
            "rtol": args.rtol,
            "atol": 0.0,
            "restart": args.restart,
            "max_iterations": args.max_iterations,
            "timing_repeats": args.repeats,
            "timing_warmup": not args.no_warmup,
            "oracle_dimension": args.k,
            "oracle_target": "smallest-magnitude eigenvalues of A",
            "preconditioning": "right preconditioning inside projected Arnoldi",
            "solve_time_definition": (
                "preconditioner setup + median Krylov solve + AU=C thin-QR setup for "
                "oracle variants; excludes eigenspace computation"
            ),
        },
        "matrix_diagnostics": {
            "symmetry_relative_frobenius_defect": symmetry_defect,
            "diagonal_min": float(matrix.diagonal().min()),
            "diagonal_max": float(matrix.diagonal().max()),
            "log_permeability_min": float(log_permeability.min()),
            "log_permeability_max": float(log_permeability.max()),
            "permeability_min": float(permeability.min()),
            "permeability_max": float(permeability.max()),
        },
        "oracle_diagnostics": {
            "eigenspace_seconds_excluded": eigen_seconds,
            "augmentation_setup_seconds_included": augmentation_seconds,
            "max_relative_eigen_residual": max(
                row["relative_eigen_residual"] for row in eigen_rows
            ),
            "AU_minus_C_frobenius": au_c_defect,
            "CTC_minus_I_frobenius": c_orthogonality_defect,
            "eigenvalues": [float(value) for value in eigenvalues],
        },
        "preconditioners": preconditioner_metadata,
        "results": rows,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row['method']:<32} iterations={row['iterations']:>5} "
            f"converged={str(row['converged']):<5} "
            f"relres={row['final_true_relative_residual']:.3e} "
            f"solve={row['solve_time_seconds']:.4f}s"
        )
    print(f"oracle eigenspace time excluded: {eigen_seconds:.4f}s")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
