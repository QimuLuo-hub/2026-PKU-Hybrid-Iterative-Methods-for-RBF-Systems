#!/usr/bin/env python3
"""Independently validate the preconditioned recycle-space benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from neurkitt_oracle import (
    apply_to_columns,
    build_augmentation,
    build_darcy_system,
    principal_angles_degrees,
)
from run_neurkitt_recycle_space_benchmark import (
    _build_preconditioner,
    _smallest_a_space,
    _smallest_b_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/neurkitt_recycle_space"),
    )
    parser.add_argument(
        "--validated-oracle-results",
        type=Path,
        default=Path("results/neurkitt_oracle/benchmark.csv"),
    )
    return parser.parse_args()


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
    dimension = matrix.shape[0]
    if u_basis is None:
        projected_rhs = rhs

        def matvec(vector):
            return matrix @ apply(vector)

    else:
        projected_rhs = rhs - c_basis @ (c_basis.T @ rhs)

        def matvec(vector):
            image = matrix @ apply(vector)
            return image - c_basis @ (c_basis.T @ image)

    operator = spla.LinearOperator(matrix.shape, matvec=matvec, dtype=float)
    callbacks: list[float] = []
    relative_tolerance = threshold / np.linalg.norm(projected_rhs)
    transformed_solution, info = spla.gmres(
        operator,
        projected_rhs,
        rtol=relative_tolerance,
        atol=0.0,
        restart=restart,
        maxiter=int(np.ceil(max_iterations / restart)),
        callback=callbacks.append,
        callback_type="pr_norm",
    )
    fine_solution = apply(transformed_solution)
    if u_basis is not None:
        fine_solution = (
            u_basis @ (c_basis.T @ rhs)
            + fine_solution
            - u_basis @ (c_basis.T @ (matrix @ fine_solution))
        )
    relative_residual = np.linalg.norm(rhs - matrix @ fine_solution) / np.linalg.norm(rhs)
    return info, len(callbacks), float(relative_residual)


def _projected_arnoldi_identity(
    matrix,
    rhs: np.ndarray,
    apply,
    u_basis: np.ndarray,
    c_basis: np.ndarray,
    *,
    steps: int = 20,
) -> tuple[float, float]:
    residual = rhs - c_basis @ (c_basis.T @ rhs)
    beta = np.linalg.norm(residual)
    vectors = np.zeros((matrix.shape[0], steps + 1))
    directions = np.zeros((matrix.shape[0], steps))
    hessenberg = np.zeros((steps + 1, steps))
    coupling = np.zeros((c_basis.shape[1], steps))
    vectors[:, 0] = residual / beta
    used = steps
    for column in range(steps):
        directions[:, column] = apply(vectors[:, column])
        image = matrix @ directions[:, column]
        coupling[:, column] = c_basis.T @ image
        work = image - c_basis @ coupling[:, column]
        for _ in range(2):
            coefficients = vectors[:, : column + 1].T @ work
            hessenberg[: column + 1, column] += coefficients
            work -= vectors[:, : column + 1] @ coefficients
        hessenberg[column + 1, column] = np.linalg.norm(work)
        if hessenberg[column + 1, column] <= 100.0 * np.finfo(float).eps:
            used = column + 1
            break
        vectors[:, column + 1] = work / hessenberg[column + 1, column]

    projected_left = matrix @ directions[:, :used]
    projected_left -= c_basis @ (c_basis.T @ projected_left)
    projected_right = vectors[:, : used + 1] @ hessenberg[: used + 1, :used]
    projected_defect = np.linalg.norm(projected_left - projected_right) / np.linalg.norm(
        projected_left
    )
    corrected_directions = directions[:, :used] - u_basis @ coupling[:, :used]
    corrected_left = matrix @ corrected_directions
    corrected_defect = np.linalg.norm(corrected_left - projected_right) / np.linalg.norm(
        corrected_left
    )
    return float(projected_defect), float(corrected_defect)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = summary["configuration"]
    benchmark_rows = _read_csv(args.results_dir / "benchmark.csv")
    benchmark = {
        (row["preconditioner"], row["recycle_space"]): row
        for row in benchmark_rows
    }
    angle_rows = _read_csv(args.results_dir / "principal_angles.csv")

    matrix, rhs, _, _ = build_darcy_system(
        int(configuration["grid_size"]),
        seed=int(configuration["darcy_seed"]),
    )
    k = int(configuration["recycle_dimension"])
    qa_space = _smallest_a_space(matrix, k)
    random_basis, _ = la.qr(
        np.random.default_rng(int(configuration["random_control_seed"])).standard_normal(
            (matrix.shape[0], k)
        ),
        mode="economic",
        check_finite=False,
    )
    threshold = float(configuration["rtol"]) * np.linalg.norm(rhs)

    cross_checks: list[dict[str, object]] = []
    identity_checks: list[dict[str, object]] = []
    angle_checks: list[dict[str, object]] = []
    all_iteration_matches = True
    all_residuals_pass = True
    all_identities_pass = True
    all_angles_match = True
    all_eigenspaces_accurate = qa_space.invariant_defect <= 5.0e-10

    for preconditioner_name in ("Jacobi", "ILU(0)"):
        preconditioner = _build_preconditioner(preconditioner_name, matrix)
        qb_space = _smallest_b_space(matrix, preconditioner, k)
        all_eigenspaces_accurate &= qb_space.invariant_defect <= 5.0e-10
        qb_physical = apply_to_columns(preconditioner.apply, qb_space.basis)
        candidate_bases = {
            "baseline": None,
            "QA": qa_space.basis,
            "QB": qb_physical,
            "random": random_basis,
        }
        for space_name, candidate_basis in candidate_bases.items():
            if candidate_basis is None:
                u_basis = c_basis = None
            else:
                u_basis, c_basis = build_augmentation(matrix, candidate_basis)
                au_c_defect = float(
                    np.linalg.norm(matrix @ u_basis - c_basis, ord="fro")
                )
                orthogonality_defect = float(
                    np.linalg.norm(c_basis.T @ c_basis - np.eye(k), ord="fro")
                )
                projected_defect, corrected_defect = _projected_arnoldi_identity(
                    matrix,
                    rhs,
                    preconditioner.apply,
                    u_basis,
                    c_basis,
                )
                identity_pass = (
                    max(
                        au_c_defect,
                        orthogonality_defect,
                        projected_defect,
                        corrected_defect,
                    )
                    <= 5.0e-11
                )
                all_identities_pass &= identity_pass
                identity_checks.append(
                    {
                        "preconditioner": preconditioner_name,
                        "recycle_space": space_name,
                        "AU_minus_C_frobenius": au_c_defect,
                        "CTC_minus_I_frobenius": orthogonality_defect,
                        "projected_arnoldi_relative_defect": projected_defect,
                        "corrected_full_operator_relative_defect": corrected_defect,
                        "pass": identity_pass,
                    }
                )

            info, iterations, relative_residual = _scipy_gmres(
                matrix,
                rhs,
                preconditioner.apply,
                restart=int(configuration["restart"]),
                max_iterations=int(configuration["max_iterations"]),
                threshold=threshold,
                u_basis=u_basis,
                c_basis=c_basis,
            )
            expected = benchmark[(preconditioner_name, space_name)]
            iteration_match = iterations == int(expected["iterations"])
            residual_pass = relative_residual <= float(configuration["rtol"]) and info == 0
            all_iteration_matches &= iteration_match
            all_residuals_pass &= residual_pass
            cross_checks.append(
                {
                    "preconditioner": preconditioner_name,
                    "recycle_space": space_name,
                    "benchmark_iterations": int(expected["iterations"]),
                    "scipy_iterations": iterations,
                    "iteration_match": iteration_match,
                    "scipy_info": info,
                    "scipy_true_relative_residual": relative_residual,
                    "residual_pass": residual_pass,
                }
            )

        recomputed_angles = principal_angles_degrees(qa_space.basis, qb_space.basis)
        recorded_angles = np.array(
            [
                float(row["principal_angle_degrees"])
                for row in angle_rows
                if row["preconditioner"] == preconditioner_name
            ]
        )
        angle_error = float(np.max(np.abs(recomputed_angles - recorded_angles)))
        angle_match = angle_error <= 1.0e-8
        all_angles_match &= angle_match
        angle_checks.append(
            {
                "preconditioner": preconditioner_name,
                "largest_principal_angle_degrees": float(recomputed_angles[0]),
                "median_principal_angle_degrees": float(np.median(recomputed_angles)),
                "max_recorded_angle_error_degrees": angle_error,
                "match": angle_match,
            }
        )

    regression_checks: list[dict[str, object]] = []
    qa_regression_pass = True
    if args.validated_oracle_results.exists():
        validated_rows = _read_csv(args.validated_oracle_results)
        validated = {
            (row["preconditioner"], row["oracle"] == "True"): int(row["iterations"])
            for row in validated_rows
        }
        for preconditioner_name in ("Jacobi", "ILU(0)"):
            for space_name, oracle in (("baseline", False), ("QA", True)):
                current = int(benchmark[(preconditioner_name, space_name)]["iterations"])
                prior = validated[(preconditioner_name, oracle)]
                match = current == prior
                qa_regression_pass &= match
                regression_checks.append(
                    {
                        "preconditioner": preconditioner_name,
                        "recycle_space": space_name,
                        "current_iterations": current,
                        "validated_prior_iterations": prior,
                        "match": match,
                    }
                )
    else:
        qa_regression_pass = False

    ratio_checks_pass = True
    for preconditioner_name in ("Jacobi", "ILU(0)"):
        baseline_iterations = int(benchmark[(preconditioner_name, "baseline")]["iterations"])
        for space_name in ("baseline", "QA", "QB", "random"):
            row = benchmark[(preconditioner_name, space_name)]
            expected_ratio = int(row["iterations"]) / baseline_iterations
            ratio_checks_pass &= abs(
                float(row["iteration_ratio_over_baseline"]) - expected_ratio
            ) <= 1.0e-14

    random_control_neutral = all(
        0.95 <= float(benchmark[(name, "random")]["iteration_ratio_over_baseline"]) <= 1.05
        for name in ("Jacobi", "ILU(0)")
    )
    all_reported_converged = all(
        row["converged"] == "True"
        and float(row["final_true_relative_residual"]) <= float(configuration["rtol"])
        for row in benchmark_rows
    )
    ready = all(
        (
            all_iteration_matches,
            all_residuals_pass,
            all_identities_pass,
            all_angles_match,
            all_eigenspaces_accurate,
            qa_regression_pass,
            ratio_checks_pass,
            all_reported_converged,
        )
    )
    validation = {
        "overall_assessment": "Ready to share" if ready else "Needs revision",
        "checks": {
            "all_scipy_iteration_counts_match": all_iteration_matches,
            "all_scipy_true_residuals_pass": all_residuals_pass,
            "all_projected_arnoldi_identities_pass": all_identities_pass,
            "all_principal_angles_reproduce": all_angles_match,
            "all_targeted_eigenspaces_accurate": all_eigenspaces_accurate,
            "validated_QA_and_baseline_counts_reproduce": qa_regression_pass,
            "iteration_ratios_recompute_exactly": ratio_checks_pass,
            "all_reported_runs_converged": all_reported_converged,
            "random_control_within_5_percent_of_baseline": random_control_neutral,
        },
        "independent_scipy_cross_checks": cross_checks,
        "projected_arnoldi_identity_checks": identity_checks,
        "principal_angle_checks": angle_checks,
        "validated_benchmark_regression_checks": regression_checks,
        "caveats": [
            "This is one deterministic Darcy GRF realization, so the conclusion is instance-specific.",
            "N=9,604 is the validated manageable-grid surrogate, not the paper's larger Darcy matrix.",
            "The exact QA/QB construction cost is excluded by design; online AU=C normalization and preconditioner setup are included.",
            "ILU is SciPy SuperLU ILU(0)-style with fixed natural ordering, not PETSc ILU.",
        ],
    }
    (args.results_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
