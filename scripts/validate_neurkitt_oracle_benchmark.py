#!/usr/bin/env python3
"""Independent validation checks for the NeurKItt oracle benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from neurkitt_oracle import build_augmentation, build_darcy_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/neurkitt_oracle"),
    )
    return parser.parse_args()


def _preconditioner(name: str, matrix):
    if name == "None":
        return lambda vector: vector.copy()
    if name == "Jacobi":
        inverse_diagonal = 1.0 / matrix.diagonal()
        return lambda vector: inverse_diagonal * vector
    if name == "ILU(0)":
        factor = spla.spilu(
            matrix.tocsc(),
            drop_tol=0.0,
            fill_factor=1.0,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"Equil": False},
        )
        return factor.solve
    raise ValueError(name)


def _scipy_gmres(
    matrix,
    rhs: np.ndarray,
    apply,
    *,
    restart: int,
    max_iterations: int,
    threshold: float,
    u_basis: np.ndarray | None = None,
    c_basis: np.ndarray | None = None,
) -> tuple[int, int, float]:
    n = matrix.shape[0]
    augmented = u_basis is not None
    if augmented:
        projected_rhs = rhs - c_basis @ (c_basis.T @ rhs)

        def matvec(vector):
            a_z = matrix @ apply(vector)
            return a_z - c_basis @ (c_basis.T @ a_z)

        system_rhs = projected_rhs
    else:

        def matvec(vector):
            return matrix @ apply(vector)

        system_rhs = rhs

    operator = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    callback_values: list[float] = []
    relative_tolerance = threshold / np.linalg.norm(system_rhs)
    transformed_solution, info = spla.gmres(
        operator,
        system_rhs,
        rtol=relative_tolerance,
        atol=0.0,
        restart=restart,
        maxiter=int(np.ceil(max_iterations / restart)),
        callback=callback_values.append,
        callback_type="pr_norm",
    )
    fine_solution = apply(transformed_solution)
    if augmented:
        fine_solution = (
            u_basis @ (c_basis.T @ rhs)
            + fine_solution
            - u_basis @ (c_basis.T @ (matrix @ fine_solution))
        )
    relative_residual = np.linalg.norm(rhs - matrix @ fine_solution) / np.linalg.norm(rhs)
    return info, len(callback_values), float(relative_residual)


def _projected_arnoldi_identity(matrix, rhs, u_basis, c_basis, steps: int = 20):
    residual = rhs - c_basis @ (c_basis.T @ rhs)
    beta = np.linalg.norm(residual)
    vectors = np.zeros((matrix.shape[0], steps + 1))
    directions = np.zeros((matrix.shape[0], steps))
    hessenberg = np.zeros((steps + 1, steps))
    coupling = np.zeros((c_basis.shape[1], steps))
    vectors[:, 0] = residual / beta
    used = steps
    for j in range(steps):
        directions[:, j] = vectors[:, j]
        a_z = matrix @ directions[:, j]
        coupling[:, j] = c_basis.T @ a_z
        work = a_z - c_basis @ coupling[:, j]
        for _ in range(2):
            coefficients = vectors[:, : j + 1].T @ work
            hessenberg[: j + 1, j] += coefficients
            work -= vectors[:, : j + 1] @ coefficients
        hessenberg[j + 1, j] = np.linalg.norm(work)
        if hessenberg[j + 1, j] <= 100.0 * np.finfo(float).eps:
            used = j + 1
            break
        vectors[:, j + 1] = work / hessenberg[j + 1, j]

    projected_left = matrix @ directions[:, :used]
    projected_left -= c_basis @ (c_basis.T @ projected_left)
    projected_right = vectors[:, : used + 1] @ hessenberg[: used + 1, :used]
    projected_defect = np.linalg.norm(projected_left - projected_right) / np.linalg.norm(
        projected_left
    )

    corrected_left = matrix @ (
        directions[:, :used] - u_basis @ coupling[:, :used]
    )
    corrected_defect = np.linalg.norm(corrected_left - projected_right) / np.linalg.norm(
        corrected_left
    )
    return float(projected_defect), float(corrected_defect)


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = summary["configuration"]
    with (args.results_dir / "benchmark.csv").open(encoding="utf-8") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    benchmark = {
        (row["preconditioner"], row["oracle"] == "True"): row
        for row in benchmark_rows
    }

    matrix, rhs, _, _ = build_darcy_system(
        int(configuration["grid_size"]),
        seed=int(configuration["seed"]),
    )
    k = int(configuration["oracle_dimension"])
    eigenvalues, invariant_basis = spla.eigsh(
        matrix,
        k=k,
        sigma=0.0,
        which="LM",
        tol=1.0e-12,
        maxiter=50000,
        v0=np.full(matrix.shape[0], 1.0 / np.sqrt(matrix.shape[0])),
    )
    order = np.argsort(np.abs(eigenvalues))
    u_basis, c_basis = build_augmentation(matrix, invariant_basis[:, order])

    threshold = float(configuration["rtol"]) * np.linalg.norm(rhs)
    cross_checks: list[dict[str, object]] = []
    all_iteration_matches = True
    all_residuals_pass = True
    for preconditioner in ("None", "Jacobi", "ILU(0)"):
        apply = _preconditioner(preconditioner, matrix)
        for oracle in (False, True):
            info, iterations, relative_residual = _scipy_gmres(
                matrix,
                rhs,
                apply,
                restart=int(configuration["restart"]),
                max_iterations=int(configuration["max_iterations"]),
                threshold=threshold,
                u_basis=u_basis if oracle else None,
                c_basis=c_basis if oracle else None,
            )
            expected_iterations = int(benchmark[(preconditioner, oracle)]["iterations"])
            iteration_match = iterations == expected_iterations
            residual_pass = relative_residual <= float(configuration["rtol"])
            all_iteration_matches &= iteration_match
            all_residuals_pass &= residual_pass and info == 0
            cross_checks.append(
                {
                    "preconditioner": preconditioner,
                    "oracle": oracle,
                    "benchmark_iterations": expected_iterations,
                    "scipy_iterations": iterations,
                    "iteration_match": iteration_match,
                    "scipy_info": info,
                    "scipy_true_relative_residual": relative_residual,
                    "residual_pass": residual_pass,
                }
            )

    projected_defect, corrected_defect = _projected_arnoldi_identity(
        matrix,
        rhs,
        u_basis,
        c_basis,
    )
    identity_pass = max(projected_defect, corrected_defect) <= 5.0e-12
    paper_identity_pass = (
        summary["oracle_diagnostics"]["AU_minus_C_frobenius"] <= 5.0e-11
        and summary["oracle_diagnostics"]["CTC_minus_I_frobenius"] <= 5.0e-11
    )

    validation = {
        "overall_assessment": (
            "Ready to share"
            if all_iteration_matches
            and all_residuals_pass
            and identity_pass
            and paper_identity_pass
            else "Needs revision"
        ),
        "checks": {
            "all_scipy_iteration_counts_match": all_iteration_matches,
            "all_scipy_true_residuals_pass": all_residuals_pass,
            "paper_AU_equals_C_and_CTC_equals_I_pass": paper_identity_pass,
            "projected_arnoldi_identity_pass": identity_pass,
            "projected_arnoldi_relative_defect": projected_defect,
            "corrected_full_operator_relative_defect": corrected_defect,
        },
        "independent_scipy_cross_checks": cross_checks,
        "caveats": [
            "This is one deterministic GRF realization, not the paper's test-set average.",
            "The matrix is 9,604-by-9,604 rather than the paper's main 160,000-by-160,000 Darcy case.",
            "The exact eigenspace cost is deliberately excluded; AU=C setup and preconditioner setup are included.",
            "Preconditioners are right-applied; ILU is SciPy SuperLU ILU(0)-style rather than PETSc's implementation.",
        ],
    }
    (args.results_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    return 0 if validation["overall_assessment"] == "Ready to share" else 1


if __name__ == "__main__":
    raise SystemExit(main())
