#!/usr/bin/env python3
"""Validate the N=400 harmonic-Ritz recycle-space experiment."""

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

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import restarted_projected_gmres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_krylov_recycle_n400"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    q_left, _ = la.qr(left, mode="economic", check_finite=False)
    q_right, _ = la.qr(right, mode="economic", check_finite=False)
    return float(
        np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord="fro")
    )


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _read_csv(args.results_dir / "benchmark.csv")
    diagnostics_rows = _read_csv(args.results_dir / "augmentation_diagnostics.csv")
    cycle_rows = _read_csv(args.results_dir / "baseline_krylov_cycle_index.csv")
    arrays = np.load(args.results_dir / "audit_arrays.npz")
    cycle_arrays = np.load(args.results_dir / "baseline_krylov_cycles.npz")
    config = summary["configuration"]

    system = assemble_tensor_gaussian_poisson_blocked(
        int(config["side"]),
        epsilon=float(config["epsilon"]),
    )
    matrix = system.A
    rhs = system.b
    rhs_norm = float(np.linalg.norm(rhs))
    preconditioner = rbf_local_cardinal_preconditioner(
        matrix,
        system.nodes,
        support_size=int(config["ls_acbf_support_size"]),
    )

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
    baseline = next(row for row in rows if row["recycle_space"] == "baseline")

    maximum_residual_relative_error = 0.0
    maximum_iteration_ratio_error = 0.0
    maximum_time_ratio_error = 0.0
    maximum_repeat_residual_error = 0.0
    repeat_outcomes_match = True
    convergence_flags_match = True
    maximum_au_defect = 0.0
    maximum_ctc_defect = 0.0
    maximum_projected_action_defect = 0.0
    rng = np.random.default_rng(4418)
    probes = rng.standard_normal((matrix.shape[0], 8))

    for row in rows:
        space_name = row["recycle_space"]
        q = int(row["q"])
        solution = arrays[f"solution_{space_name}_q{q}"]
        recomputed_residual = float(
            np.linalg.norm(rhs - matrix @ solution) / rhs_norm
        )
        reported_residual = float(row["true_relative_residual"])
        maximum_residual_relative_error = max(
            maximum_residual_relative_error,
            abs(recomputed_residual - reported_residual)
            / max(abs(recomputed_residual), np.finfo(float).tiny),
        )
        convergence_flags_match &= _as_bool(row["converged"]) == (
            recomputed_residual <= float(config["rtol"])
        )
        expected_iteration_ratio = int(row["iterations"]) / int(
            baseline["iterations"]
        )
        expected_time_ratio = float(row["solve_seconds"]) / float(
            baseline["solve_seconds"]
        )
        maximum_iteration_ratio_error = max(
            maximum_iteration_ratio_error,
            abs(float(row["iteration_ratio"]) - expected_iteration_ratio),
        )
        maximum_time_ratio_error = max(
            maximum_time_ratio_error,
            abs(float(row["solve_time_ratio"]) - expected_time_ratio),
        )

        if space_name == "baseline":
            u_basis = c_basis = None
        else:
            prefix = f"{space_name}_q{q}"
            u_basis = arrays[f"{prefix}_U"]
            c_basis = arrays[f"{prefix}_C"]
            au_defect = float(np.linalg.norm(matrix @ u_basis - c_basis, ord="fro"))
            ctc_defect = float(
                np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")
            )
            az = matrix @ probes
            projected = az - c_basis @ (c_basis.T @ az)
            corrected = matrix @ (probes - u_basis @ (c_basis.T @ az))
            projected_defect = float(
                np.linalg.norm(projected - corrected, ord="fro")
                / max(np.linalg.norm(projected, ord="fro"), np.finfo(float).tiny)
            )
            maximum_au_defect = max(maximum_au_defect, au_defect)
            maximum_ctc_defect = max(maximum_ctc_defect, ctc_defect)
            maximum_projected_action_defect = max(
                maximum_projected_action_defect,
                projected_defect,
            )

        repeat = restarted_projected_gmres(
            matrix,
            rhs,
            preconditioner.apply,
            u_basis=u_basis,
            c_basis=c_basis,
            x0=np.zeros(matrix.shape[0]),
            rtol=float(config["rtol"]),
            atol=float(config["atol"]),
            restart=int(config["restart"]),
            max_iterations=int(config["max_iterations"]),
        )
        repeat_residual = float(
            np.linalg.norm(rhs - matrix @ repeat.x) / rhs_norm
        )
        maximum_repeat_residual_error = max(
            maximum_repeat_residual_error,
            abs(repeat_residual - reported_residual),
        )
        repeat_outcomes_match &= (
            repeat.iterations == int(row["iterations"])
            and repeat.converged == _as_bool(row["converged"])
            and repeat_residual <= float(config["rtol"])
        )

    trajectory = fgmres(
        matrix,
        rhs,
        preconditioner.apply,
        rtol=float(config["rtol"]),
        atol=float(config["atol"]),
        restart=int(config["restart"]),
        max_iterations=int(config["max_iterations"]),
        capture_arnoldi=True,
    )
    cycle_count_matches = len(trajectory.arnoldi_cycles or []) == len(cycle_rows)
    maximum_saved_cycle_relative_defect = 0.0
    if cycle_count_matches:
        for row, cycle in zip(cycle_rows, trajectory.arnoldi_cycles or []):
            for key, recomputed in (
                (row["V_key"], cycle.arnoldi_vectors),
                (row["Z_key"], cycle.preconditioned_vectors),
                (row["H_key"], cycle.hessenberg),
            ):
                saved = cycle_arrays[key]
                defect = float(
                    np.linalg.norm(saved - recomputed)
                    / max(np.linalg.norm(saved), np.finfo(float).tiny)
                )
                maximum_saved_cycle_relative_defect = max(
                    maximum_saved_cycle_relative_defect,
                    defect,
                )

    harmonic_space_defects: dict[str, float] = {}
    for q in (4, 8, 16):
        recomputed = harmonic_ritz_space(trajectory, q)
        harmonic_space_defects[str(q)] = _projector_distance(
            recomputed,
            arrays[f"harmonic_ritz_q{q}_Y"],
        )

    random_basis = arrays["random_q16"]
    random_orthogonality_defect = float(
        np.linalg.norm(random_basis.T @ random_basis - np.eye(16), ord="fro")
    )
    timing_medians_match = True
    for diagnostic in diagnostics_rows:
        matching = next(
            row
            for row in rows
            if row["recycle_space"] == diagnostic["recycle_space"]
            and int(row["q"]) == int(diagnostic["q"])
        )
        timing_medians_match &= (
            abs(
                float(diagnostic["solve_seconds_median"])
                - float(matching["solve_seconds"])
            )
            <= 1.0e-15
        )

    harmonic_rows = [row for row in rows if row["recycle_space"] == "harmonic_ritz"]
    random_by_q = {
        int(row["q"]): row for row in rows if row["recycle_space"] == "random"
    }
    success_recomputes = all(
        int(row["iterations"]) < int(random_by_q[int(row["q"])]["iterations"])
        and float(row["iteration_ratio"]) <= 0.8
        and _as_bool(row["converged"])
        for row in harmonic_rows
    )

    checks = {
        "fixed_validated_configuration_is_exact": (
            int(config["N"]) == 400
            and float(config["epsilon"]) == 8.0
            and int(config["ls_acbf_support_size"]) == 81
            and float(config["rtol"]) == 1.0e-10
            and int(config["restart"]) == 100
            and int(config["max_iterations"]) == 1000
        ),
        "all_seven_requested_rows_present": actual_pairs == expected_pairs,
        "all_reported_true_residuals_recompute_from_original_A_b": (
            maximum_residual_relative_error <= 1.0e-12
        ),
        "convergence_flags_match_original_system_tolerance": convergence_flags_match,
        "iteration_ratios_recompute": maximum_iteration_ratio_error <= 1.0e-15,
        "solve_time_ratios_recompute": maximum_time_ratio_error <= 1.0e-15,
        "all_seven_solver_outcomes_repeat": repeat_outcomes_match,
        "AU_equals_C_for_all_recycle_spaces": maximum_au_defect <= 2.0e-10,
        "C_is_orthonormal_for_all_recycle_spaces": maximum_ctc_defect <= 1.0e-12,
        "corrected_and_projected_actions_match": (
            maximum_projected_action_defect <= float(config["rtol"])
        ),
        "all_baseline_Krylov_cycles_saved_and_reproduce": (
            cycle_count_matches and maximum_saved_cycle_relative_defect <= 1.0e-13
        ),
        "harmonic_spaces_recompute_from_saved_convention": (
            max(harmonic_space_defects.values()) <= 1.0e-10
        ),
        "matched_random_q16_basis_is_orthonormal": random_orthogonality_defect
        <= 1.0e-12,
        "timed_medians_are_online_solver_only": (
            timing_medians_match
            and "baseline_arnoldi_extraction_seconds"
            in summary["offline_costs_excluded"]
        ),
        "all_harmonic_spaces_meet_declared_success_rule": success_recomputes,
        "no_A_inverse_bare_A_space_or_ML": (
            not config["A_inverse_used"]
            and not config["bare_A_invariant_subspaces_used"]
            and not config["ML_used"]
        ),
    }
    validation = {
        "overall_assessment": "Ready to use" if all(checks.values()) else "Needs review",
        "checks": checks,
        "diagnostics": {
            "maximum_reported_residual_relative_error": maximum_residual_relative_error,
            "maximum_repeat_residual_absolute_error": maximum_repeat_residual_error,
            "maximum_iteration_ratio_absolute_error": maximum_iteration_ratio_error,
            "maximum_solve_time_ratio_absolute_error": maximum_time_ratio_error,
            "maximum_AU_minus_C_frobenius": maximum_au_defect,
            "maximum_CHC_minus_I_frobenius": maximum_ctc_defect,
            "maximum_projected_action_relative_defect": maximum_projected_action_defect,
            "maximum_saved_Krylov_cycle_relative_defect": maximum_saved_cycle_relative_defect,
            "harmonic_space_projector_defects": harmonic_space_defects,
            "random_q16_orthogonality_defect": random_orthogonality_defect,
        },
        "caveats": [
            "The harmonic spaces are a-posteriori oracle spaces extracted from the same converged baseline trajectory; transfer to a new system or right-hand side is not tested.",
            "Solve times are local single-thread medians and exclude all declared offline construction costs.",
            "The absolute AU-C defect reaches roughly 1e-10 at q=16, while C remains orthonormal to machine precision and the projected-action relative defect remains below the 1e-10 solver tolerance.",
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
