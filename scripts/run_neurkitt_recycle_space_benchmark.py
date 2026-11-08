#!/usr/bin/env python3
"""Compare A- and right-preconditioned invariant recycle spaces on Darcy N=9604."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from neurkitt_oracle import (
    GmresResult,
    apply_real_operator_to_complex,
    apply_to_columns,
    build_augmentation,
    build_darcy_system,
    invariant_subspace_defect,
    principal_angles_degrees,
    real_invariant_subspace,
    restarted_projected_gmres,
)


VectorMap = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Preconditioner:
    name: str
    apply: VectorMap
    setup_seconds: float
    details: dict[str, object]
    factor: object | None = None


@dataclass(frozen=True)
class OperatorSpace:
    basis: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    selected_indices: tuple[int, ...]
    relative_eigen_residuals: np.ndarray
    construction_seconds: float
    invariant_defect: float
    inverse_check_defect: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20240831)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/neurkitt_recycle_space"),
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


def _build_preconditioner(name: str, matrix: sp.spmatrix) -> Preconditioner:
    start = perf_counter()
    if name == "Jacobi":
        diagonal = matrix.diagonal()
        if np.any(diagonal == 0.0):
            raise ValueError("Jacobi encountered a zero diagonal")
        inverse_diagonal = 1.0 / diagonal

        def apply(array: np.ndarray) -> np.ndarray:
            values = np.asarray(array)
            if values.ndim == 1:
                return inverse_diagonal * values
            return inverse_diagonal[:, None] * values

        factor = inverse_diagonal
        details: dict[str, object] = {
            "kind": "right Jacobi",
            "zero_diagonal_count": 0,
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
            "row_permutation_identity": bool(
                np.array_equal(factor.perm_r, np.arange(matrix.shape[0]))
            ),
            "column_permutation_identity": bool(
                np.array_equal(factor.perm_c, np.arange(matrix.shape[0]))
            ),
        }
    else:
        raise ValueError(f"unknown preconditioner: {name}")
    return Preconditioner(
        name=name,
        apply=apply,
        setup_seconds=perf_counter() - start,
        details=details,
        factor=factor,
    )


def _smallest_a_space(matrix: sp.spmatrix, k: int) -> OperatorSpace:
    dimension = matrix.shape[0]
    start = perf_counter()
    eigenvalues, eigenvectors = spla.eigsh(
        matrix,
        k=k,
        sigma=0.0,
        which="LM",
        tol=1.0e-12,
        maxiter=50000,
        v0=np.full(dimension, 1.0 / np.sqrt(dimension)),
    )
    elapsed = perf_counter() - start
    order = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[order].astype(complex)
    eigenvectors = eigenvectors[:, order].astype(complex)
    basis = eigenvectors.real
    residuals = np.array(
        [
            np.linalg.norm(matrix @ eigenvectors[:, j] - eigenvalues[j] * eigenvectors[:, j])
            / max(abs(eigenvalues[j]), np.finfo(float).tiny)
            for j in range(k)
        ]
    )
    return OperatorSpace(
        basis=basis,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        selected_indices=tuple(range(k)),
        relative_eigen_residuals=residuals,
        construction_seconds=elapsed,
        invariant_defect=invariant_subspace_defect(lambda array: matrix @ array, basis),
        inverse_check_defect=None,
    )


def _smallest_b_space(
    matrix: sp.spmatrix,
    preconditioner: Preconditioner,
    k: int,
) -> OperatorSpace:
    """Compute the smallest-magnitude real invariant space of ``B=A M^-1``."""

    dimension = matrix.shape[0]
    requested = max(k + 6, 16)
    if requested >= dimension - 1:
        requested = dimension - 2
    start = perf_counter()
    inverse_check_defect: float | None = None

    def apply_b(array: np.ndarray) -> np.ndarray:
        return matrix @ apply_real_operator_to_complex(preconditioner.apply, array)

    if preconditioner.name == "Jacobi":
        inverse_diagonal = np.asarray(preconditioner.factor)
        b_operator = (matrix @ sp.diags(inverse_diagonal, format="csc")).tocsc()
        eigenvalues, eigenvectors = spla.eigs(
            b_operator,
            k=requested,
            sigma=0.0,
            which="LM",
            tol=1.0e-12,
            maxiter=50000,
            v0=np.full(dimension, 1.0 / np.sqrt(dimension)),
        )
    elif preconditioner.name == "ILU(0)":
        ilu = preconditioner.factor
        if not (
            np.array_equal(ilu.perm_r, np.arange(dimension))
            and np.array_equal(ilu.perm_c, np.arange(dimension))
        ):
            raise RuntimeError("the exact ILU inverse operator assumes identity permutations")
        approximate_matrix = (ilu.L @ ilu.U).tocsr()
        exact_factor = spla.splu(
            matrix.tocsc(),
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"Equil": False},
        )

        def apply_b_inverse(array: np.ndarray) -> np.ndarray:
            solved = apply_real_operator_to_complex(exact_factor.solve, array)
            return approximate_matrix @ solved

        b_operator = spla.LinearOperator(
            matrix.shape,
            matvec=apply_b,
            dtype=float,
        )
        inverse_operator = spla.LinearOperator(
            matrix.shape,
            matvec=apply_b_inverse,
            dtype=float,
        )
        probe = np.random.default_rng(314159).standard_normal(dimension)
        inverse_check_defect = float(
            np.linalg.norm(apply_b(apply_b_inverse(probe)) - probe)
            / np.linalg.norm(probe)
        )
        eigenvalues, eigenvectors = spla.eigs(
            b_operator,
            k=requested,
            sigma=0.0,
            which="LM",
            OPinv=inverse_operator,
            tol=1.0e-12,
            maxiter=50000,
            v0=np.full(dimension, 1.0 / np.sqrt(dimension)),
        )
    else:
        raise ValueError(preconditioner.name)

    elapsed = perf_counter() - start
    order = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    real_space = real_invariant_subspace(eigenvalues, eigenvectors, k)
    residuals = np.array(
        [
            np.linalg.norm(
                apply_b(eigenvectors[:, j]) - eigenvalues[j] * eigenvectors[:, j]
            )
            / max(abs(eigenvalues[j]), np.finfo(float).tiny)
            for j in range(eigenvalues.size)
        ]
    )
    return OperatorSpace(
        basis=real_space.basis,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        selected_indices=real_space.selected_indices,
        relative_eigen_residuals=residuals,
        construction_seconds=elapsed,
        invariant_defect=invariant_subspace_defect(apply_b, real_space.basis),
        inverse_check_defect=inverse_check_defect,
    )


def _solve_repeated(
    matrix: sp.spmatrix,
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
    if len({run.iterations for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagreed on iteration count")
    if len({run.converged for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagreed on convergence")
    times = [run.solve_seconds for run in runs]
    return runs[0], float(np.median(times)), times


def _augmentation_diagnostics(
    matrix: sp.spmatrix,
    u_basis: np.ndarray,
    c_basis: np.ndarray,
) -> dict[str, float]:
    k = c_basis.shape[1]
    return {
        "AU_minus_C_frobenius": float(
            np.linalg.norm(matrix @ u_basis - c_basis, ord="fro")
        ),
        "CTC_minus_I_frobenius": float(
            np.linalg.norm(c_basis.T @ c_basis - np.eye(k), ord="fro")
        ),
    }


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
    if not 1 <= args.k < dimension - 1:
        raise ValueError("k must be positive and smaller than matrix dimension minus one")

    qa_space = _smallest_a_space(matrix, args.k)
    random_start = perf_counter()
    random_basis, _ = la.qr(
        np.random.default_rng(args.random_seed).standard_normal((dimension, args.k)),
        mode="economic",
        check_finite=False,
    )
    random_construction_seconds = perf_counter() - random_start

    rows: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    angle_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    preconditioner_metadata: dict[str, dict[str, object]] = {}
    space_diagnostics: dict[str, dict[str, dict[str, float]]] = {}
    qb_metadata: dict[str, dict[str, object]] = {}

    for index in qa_space.selected_indices:
        value = qa_space.eigenvalues[index]
        eigen_rows.append(
            {
                "preconditioner": "None",
                "operator": "A",
                "selected_mode": len(eigen_rows) + 1,
                "eigenvalue_real": float(value.real),
                "eigenvalue_imag": float(value.imag),
                "eigenvalue_magnitude": float(abs(value)),
                "relative_eigen_residual": float(
                    qa_space.relative_eigen_residuals[index]
                ),
            }
        )

    for preconditioner_name in ("Jacobi", "ILU(0)"):
        preconditioner = _build_preconditioner(preconditioner_name, matrix)
        preconditioner_metadata[preconditioner_name] = {
            **preconditioner.details,
            "setup_seconds": preconditioner.setup_seconds,
        }
        qb_space = _smallest_b_space(matrix, preconditioner, args.k)
        angles = principal_angles_degrees(qa_space.basis, qb_space.basis)

        for ordinal, index in enumerate(qb_space.selected_indices, start=1):
            value = qb_space.eigenvalues[index]
            eigen_rows.append(
                {
                    "preconditioner": preconditioner_name,
                    "operator": "B=A M^-1",
                    "selected_mode": ordinal,
                    "eigenvalue_real": float(value.real),
                    "eigenvalue_imag": float(value.imag),
                    "eigenvalue_magnitude": float(abs(value)),
                    "relative_eigen_residual": float(
                        qb_space.relative_eigen_residuals[index]
                    ),
                }
            )
        for angle_index, angle in enumerate(angles, start=1):
            angle_rows.append(
                {
                    "preconditioner": preconditioner_name,
                    "canonical_angle_index_largest_first": angle_index,
                    "principal_angle_degrees": float(angle),
                }
            )

        qb_physical_basis = apply_to_columns(preconditioner.apply, qb_space.basis)
        candidate_bases: dict[str, np.ndarray | None] = {
            "baseline": None,
            "QA": qa_space.basis,
            "QB": qb_physical_basis,
            "random": random_basis,
        }
        augmentations: dict[str, tuple[np.ndarray | None, np.ndarray | None, float]] = {}
        pc_diagnostics: dict[str, dict[str, float]] = {}
        for space_name, candidate_basis in candidate_bases.items():
            if candidate_basis is None:
                augmentations[space_name] = (None, None, 0.0)
                continue
            augmentation_start = perf_counter()
            u_basis, c_basis = build_augmentation(matrix, candidate_basis)
            augmentation_seconds = perf_counter() - augmentation_start
            augmentations[space_name] = (u_basis, c_basis, augmentation_seconds)
            pc_diagnostics[space_name] = _augmentation_diagnostics(
                matrix,
                u_basis,
                c_basis,
            )

        c_qa = augmentations["QA"][1]
        c_qb = augmentations["QB"][1]
        projector_angles = principal_angles_degrees(c_qa, c_qb)
        qb_projector_alignment = float(
            principal_angles_degrees(c_qb, qb_space.basis)[0]
        )
        qa_projector_alignment = float(
            principal_angles_degrees(c_qa, qa_space.basis)[0]
        )
        mapped_qb_orthonormal, _ = la.qr(
            qb_physical_basis,
            mode="economic",
            check_finite=False,
        )
        physical_angles = principal_angles_degrees(
            qa_space.basis,
            mapped_qb_orthonormal,
        )
        qb_metadata[preconditioner_name] = {
            "eigenspace_seconds_excluded": qb_space.construction_seconds,
            "selected_eigenvalues": [
                {
                    "real": float(qb_space.eigenvalues[index].real),
                    "imag": float(qb_space.eigenvalues[index].imag),
                }
                for index in qb_space.selected_indices
            ],
            "max_selected_relative_eigen_residual": float(
                max(qb_space.relative_eigen_residuals[index] for index in qb_space.selected_indices)
            ),
            "real_invariant_subspace_defect": qb_space.invariant_defect,
            "shift_invert_inverse_check_defect": qb_space.inverse_check_defect,
            "qa_vs_qb_principal_angles_degrees_largest_first": angles.tolist(),
            "largest_principal_angle_degrees": float(angles[0]),
            "median_principal_angle_degrees": float(np.median(angles)),
            "smallest_principal_angle_degrees": float(angles[-1]),
            "qa_vs_m_inverse_qb_largest_physical_angle_degrees": float(
                physical_angles[0]
            ),
            "projector_angle_match_max_absolute_degrees": float(
                np.max(np.abs(projector_angles - angles))
            ),
            "qa_projector_alignment_degrees": qa_projector_alignment,
            "qb_projector_alignment_degrees": qb_projector_alignment,
        }
        space_diagnostics[preconditioner_name] = pc_diagnostics

        baseline_iterations: int | None = None
        baseline_solve_seconds: float | None = None
        for space_name in ("baseline", "QA", "QB", "random"):
            u_basis, c_basis, augmentation_seconds = augmentations[space_name]
            result, krylov_median, krylov_repeats = _solve_repeated(
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
            solve_seconds = (
                preconditioner.setup_seconds + augmentation_seconds + krylov_median
            )
            if space_name == "baseline":
                baseline_iterations = result.iterations
                baseline_solve_seconds = solve_seconds
            ratio = result.iterations / baseline_iterations
            time_ratio = solve_seconds / baseline_solve_seconds
            method = (
                f"GMRES + {preconditioner_name}"
                if space_name == "baseline"
                else f"GMRES + {preconditioner_name} + {space_name} k={args.k}"
            )
            rows.append(
                {
                    "method": method,
                    "preconditioner": preconditioner_name,
                    "recycle_space": space_name,
                    "k": 0 if space_name == "baseline" else args.k,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "final_true_relative_residual": result.relative_residual,
                    "restart_cycles": result.restart_cycles,
                    "preconditioner_setup_seconds": preconditioner.setup_seconds,
                    "augmentation_setup_seconds": augmentation_seconds,
                    "krylov_seconds_median": krylov_median,
                    "solve_time_seconds_excluding_space_construction": solve_seconds,
                    "iteration_ratio_over_baseline": ratio,
                    "time_ratio_over_baseline": time_ratio,
                }
            )
            for history_index, relative_residual in enumerate(result.residual_history):
                histories.append(
                    {
                        "preconditioner": preconditioner_name,
                        "recycle_space": space_name,
                        "history_index": history_index,
                        "relative_residual": relative_residual,
                    }
                )
            rows[-1]["krylov_seconds_repeats"] = krylov_repeats

    _write_csv(
        args.output_dir / "benchmark.csv",
        [
            {key: value for key, value in row.items() if key != "krylov_seconds_repeats"}
            for row in rows
        ],
    )
    _write_csv(args.output_dir / "principal_angles.csv", angle_rows)
    _write_csv(args.output_dir / "targeted_eigenpairs.csv", eigen_rows)
    _write_csv(args.output_dir / "residual_history.csv", histories)

    summary = {
        "configuration": {
            "grid_size": args.grid_size,
            "interior_side": args.grid_size - 2,
            "matrix_dimension": dimension,
            "matrix_nnz": int(matrix.nnz),
            "darcy_seed": args.seed,
            "random_control_seed": args.random_seed,
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
            "recycle_dimension": args.k,
            "qa_definition": "orthonormal invariant basis of A at its k smallest-magnitude eigenvalues",
            "qb_definition": "real orthonormal invariant basis of B=A M^-1 at its k smallest-magnitude modes; mapped to physical corrections by M^-1 before AU=C normalization",
            "random_definition": "fixed Gaussian orthonormal physical candidate basis, shared across preconditioners",
            "preconditioning": "right preconditioning inside the validated projected-Arnoldi implementation",
            "solve_time_definition": "preconditioner setup + AU=C thin-QR normalization + median Krylov solve; excludes QA/QB eigensolvers and random-basis construction",
            "principal_angle_definition": "canonical angles between orthonormal QA and QB; headline value is the largest angle",
        },
        "matrix_diagnostics": {
            "diagonal_min": float(matrix.diagonal().min()),
            "diagonal_max": float(matrix.diagonal().max()),
            "log_permeability_min": float(log_permeability.min()),
            "log_permeability_max": float(log_permeability.max()),
            "permeability_min": float(permeability.min()),
            "permeability_max": float(permeability.max()),
        },
        "qa_diagnostics": {
            "eigenspace_seconds_excluded": qa_space.construction_seconds,
            "selected_eigenvalues": [
                float(qa_space.eigenvalues[index].real)
                for index in qa_space.selected_indices
            ],
            "max_relative_eigen_residual": float(
                max(qa_space.relative_eigen_residuals)
            ),
            "invariant_subspace_defect": qa_space.invariant_defect,
        },
        "qb_diagnostics": qb_metadata,
        "random_control_construction_seconds_excluded": random_construction_seconds,
        "preconditioners": preconditioner_metadata,
        "augmentation_diagnostics": space_diagnostics,
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
            f"{row['preconditioner']:<7} {row['recycle_space']:<8} "
            f"iterations={row['iterations']:>4} "
            f"ratio={row['iteration_ratio_over_baseline']:.4f} "
            f"converged={str(row['converged']):<5} "
            f"solve={row['solve_time_seconds_excluding_space_construction']:.4f}s"
        )
    for name, diagnostics in qb_metadata.items():
        print(
            f"{name:<7} QA-QB principal angle: "
            f"largest={diagnostics['largest_principal_angle_degrees']:.3f} deg, "
            f"median={diagnostics['median_principal_angle_degrees']:.3f} deg"
        )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
