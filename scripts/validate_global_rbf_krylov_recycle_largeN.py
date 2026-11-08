#!/usr/bin/env python3
"""Validate the large-N Krylov-derived recycle-space benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import jacobi_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import restarted_projected_gmres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_krylov_recycle_largeN"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _subspace_defect(left: np.ndarray, right: np.ndarray) -> float:
    q_left, _ = la.qr(left, mode="economic", check_finite=False)
    q_right, _ = la.qr(right, mode="economic", check_finite=False)
    return float(
        np.linalg.norm(
            q_left @ q_left.T - q_right @ q_right.T,
            ord="fro",
        )
    )


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _read_csv(args.results_dir / "benchmark.csv")
    arrays = np.load(args.results_dir / "audit_arrays.npz")
    config = summary["configuration"]

    system = assemble_tensor_gaussian_poisson_blocked(
        int(config["side"]),
        epsilon=float(config["epsilon"]),
    )
    matrix = system.A
    rhs = system.b
    preconditioner = jacobi_preconditioner(matrix)
    baseline = next(row for row in rows if row["recycle_space"] == "baseline")

    expected_pairs = {
        ("baseline", 0),
        ("harmonic_ritz", 4),
        ("random", 4),
        ("harmonic_ritz", 8),
        ("random", 8),
        ("harmonic_ritz", 16),
        ("random", 16),
    }
    actual_pairs = {(row["recycle_space"], int(row["q"])) for row in rows}

    residual_relative_errors: list[float] = []
    ratio_errors: list[float] = []
    for row in rows:
        key = f"solution_{row['recycle_space']}_q{row['q']}"
        solution = arrays[key]
        recomputed = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
        reported = float(row["true_relative_residual"])
        residual_relative_errors.append(
            abs(recomputed - reported) / max(abs(recomputed), np.finfo(float).tiny)
        )
        expected_ratio = int(row["iterations"]) / int(baseline["iterations"])
        ratio_errors.append(abs(float(row["iteration_ratio"]) - expected_ratio))

    maximum_au_defect = 0.0
    maximum_ctc_defect = 0.0
    maximum_corrected_operator_defect = 0.0
    rng = np.random.default_rng(8831)
    probes = rng.standard_normal((matrix.shape[0], 8))
    for space_name in ("harmonic_ritz", "random"):
        for q in (4, 8, 16):
            prefix = f"{space_name}_q{q}"
            u_basis = arrays[f"{prefix}_U"]
            c_basis = arrays[f"{prefix}_C"]
            au_defect = np.linalg.norm(matrix @ u_basis - c_basis, ord="fro")
            ctc_defect = np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")
            az = matrix @ probes
            projected = az - c_basis @ (c_basis.T @ az)
            corrected = matrix @ (
                probes - u_basis @ (c_basis.T @ az)
            )
            corrected_defect = np.linalg.norm(projected - corrected, ord="fro") / max(
                np.linalg.norm(projected, ord="fro"),
                np.finfo(float).tiny,
            )
            maximum_au_defect = max(maximum_au_defect, float(au_defect))
            maximum_ctc_defect = max(maximum_ctc_defect, float(ctc_defect))
            maximum_corrected_operator_defect = max(
                maximum_corrected_operator_defect,
                float(corrected_defect),
            )

    trajectory = fgmres(
        matrix,
        rhs,
        preconditioner.apply,
        rtol=float(config["rtol"]),
        restart=int(config["restart"]),
        max_iterations=int(config["max_iterations"]),
        capture_arnoldi=True,
    )
    harmonic_space_defects = {}
    for q in (4, 8, 16):
        recomputed = harmonic_ritz_space(trajectory, q)
        harmonic_space_defects[str(q)] = _subspace_defect(
            recomputed,
            arrays[f"harmonic_ritz_q{q}_Y"],
        )

    random_basis = arrays["random_q16"]
    random_orthogonality = float(
        np.linalg.norm(random_basis.T @ random_basis - np.eye(16), ord="fro")
    )

    right_operator = spla.LinearOperator(
        matrix.shape,
        matvec=lambda vector: matrix @ preconditioner.apply(vector),
        dtype=float,
    )
    scipy_iterations = 0

    def count_iteration(_value: float) -> None:
        nonlocal scipy_iterations
        scipy_iterations += 1

    y_scipy, scipy_info = spla.gmres(
        right_operator,
        rhs,
        rtol=float(config["rtol"]),
        atol=0.0,
        restart=int(config["restart"]),
        maxiter=int(config["max_iterations"]) // int(config["restart"]),
        callback=count_iteration,
        callback_type="pr_norm",
    )
    x_scipy = preconditioner.apply(y_scipy)
    scipy_residual = float(np.linalg.norm(rhs - matrix @ x_scipy) / np.linalg.norm(rhs))

    u16 = arrays["harmonic_ritz_q16_U"]
    c16 = arrays["harmonic_ritz_q16_C"]
    repeat = restarted_projected_gmres(
        matrix,
        rhs,
        preconditioner.apply,
        u_basis=u16,
        c_basis=c16,
        rtol=float(config["rtol"]),
        restart=int(config["restart"]),
        max_iterations=int(config["max_iterations"]),
    )
    reported_q16 = next(
        row
        for row in rows
        if row["recycle_space"] == "harmonic_ritz" and int(row["q"]) == 16
    )

    checks = {
        "all_requested_method_rank_pairs_present": actual_pairs == expected_pairs,
        "reported_true_residuals_recompute": max(residual_relative_errors) <= 1.0e-12,
        "iteration_ratios_recompute": max(ratio_errors) <= 1.0e-15,
        "all_solve_times_are_positive_and_finite": all(
            np.isfinite(float(row["solve_seconds"])) and float(row["solve_seconds"]) > 0.0
            for row in rows
        ),
        "AU_equals_C": maximum_au_defect <= 1.0e-10,
        "C_is_orthonormal": maximum_ctc_defect <= 1.0e-12,
        "corrected_operator_matches_projected_operator": maximum_corrected_operator_defect <= 1.0e-12,
        "harmonic_spaces_recompute_from_baseline_Arnoldi_data": max(harmonic_space_defects.values()) <= 1.0e-10,
        "random_q16_is_orthonormal": random_orthogonality <= 1.0e-12,
        "captured_baseline_matches_reported_iteration_outcome": (
            trajectory.iterations == int(baseline["iterations"])
            and trajectory.converged
            == (float(baseline["true_relative_residual"]) <= float(config["rtol"]))
        ),
        "independent_SciPy_right_GMRES_matches_cap_and_convergence": (
            scipy_iterations == int(baseline["iterations"])
            and (scipy_info == 0) == (float(baseline["true_relative_residual"]) <= float(config["rtol"]))
        ),
        "harmonic_q16_repeat_reproduces": (
            repeat.iterations == int(reported_q16["iterations"])
            and abs(repeat.relative_residual - float(reported_q16["true_relative_residual"]))
            <= 1.0e-13
        ),
        "no_bare_A_invariant_subspaces": not bool(config["bare_A_invariant_subspaces_used"]),
        "no_A_inverse": not bool(config["A_inverse_used"]),
        "no_ML": not bool(config["ML_used"]),
    }
    validation = {
        "overall_assessment": "Ready to share" if all(checks.values()) else "Needs review",
        "checks": checks,
        "diagnostics": {
            "maximum_reported_residual_relative_error": max(residual_relative_errors),
            "maximum_iteration_ratio_absolute_error": max(ratio_errors),
            "maximum_AU_minus_C_frobenius": maximum_au_defect,
            "maximum_CTC_minus_I_frobenius": maximum_ctc_defect,
            "maximum_corrected_operator_relative_defect": maximum_corrected_operator_defect,
            "harmonic_space_projector_defects": harmonic_space_defects,
            "random_q16_orthogonality_defect": random_orthogonality,
            "scipy_iterations": scipy_iterations,
            "scipy_info": int(scipy_info),
            "scipy_true_relative_residual": scipy_residual,
            "harmonic_q16_repeat_true_relative_residual": repeat.relative_residual,
        },
        "caveats": [
            "All iteration counts are right-censored at the shared 1,000-step cap.",
            "Recycle-space extraction and AU=C construction are excluded from the requested solve-time comparison.",
            "The harmonic Ritz space is extracted from the completed baseline run on the same matrix and right-hand side; this is an a-posteriori recycle test, not a transfer experiment.",
            "Wall-clock values are local single-thread medians.",
        ],
    }
    (args.results_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
