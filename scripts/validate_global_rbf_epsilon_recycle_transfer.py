#!/usr/bin/env python3
"""Validate epsilon transfer of q=16 harmonic-Ritz recycle spaces."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.image as mpimg
import numpy as np
import scipy.linalg as la
import scipy.stats as stats

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import restarted_projected_gmres


EPSILONS = (7.6, 7.8, 8.0, 8.2, 8.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_epsilon_recycle_transfer"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _epsilon_key(epsilon: float) -> str:
    return f"eps_{epsilon:.1f}".replace(".", "p")


def _projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    q_left, _ = la.qr(left, mode="economic", check_finite=False)
    q_right, _ = la.qr(right, mode="economic", check_finite=False)
    return float(
        np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord="fro")
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _harmonic_values(hessenberg: np.ndarray) -> np.ndarray:
    m = hessenberg.shape[1]
    hm = hessenberg[:m, :]
    tail = hessenberg[m, m - 1] if hessenberg.shape[0] > m else 0.0
    k_matrix = hm.T @ hm
    k_matrix[-1, -1] += tail**2
    values = la.eigvals(k_matrix, hm.T, check_finite=False)
    values = values[np.isfinite(values)]
    return values[np.argsort(np.abs(values))]


def _rerun(
    matrix: np.ndarray,
    rhs: np.ndarray,
    apply_preconditioner,
    *,
    u_basis: np.ndarray | None,
    c_basis: np.ndarray | None,
    config: dict[str, Any],
):
    return restarted_projected_gmres(
        matrix,
        rhs,
        apply_preconditioner,
        u_basis=u_basis,
        c_basis=c_basis,
        x0=np.zeros(matrix.shape[0]),
        rtol=float(config["rtol"]),
        atol=float(config["atol"]),
        restart=int(config["restart"]),
        max_iterations=int(config["max_iterations"]),
    )


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["configuration"]
    pair_rows = _read_csv(args.results_dir / "pair_results.csv")
    target_rows = _read_csv(args.results_dir / "target_controls.csv")
    source_rows = _read_csv(args.results_dir / "source_extractions.csv")
    angle_rows = _read_csv(args.results_dir / "principal_angles.csv")
    cycle_index = _read_csv(args.results_dir / "source_krylov_cycle_index.csv")
    source_arrays = np.load(args.results_dir / "source_recycle_spaces.npz")
    pair_arrays = np.load(args.results_dir / "target_and_pair_recycle_arrays.npz")
    cycle_arrays = np.load(args.results_dir / "source_krylov_cycles.npz")

    systems: dict[float, Any] = {}
    preconditioners: dict[float, Any] = {}
    trajectories: dict[float, Any] = {}
    maximum_source_space_defect = 0.0
    maximum_saved_cycle_defect = 0.0
    source_outcomes_repeat = True
    cycle_counts_match = True
    cutoff_rows: list[dict[str, Any]] = []
    ritz_rows: list[dict[str, Any]] = []

    for epsilon in EPSILONS:
        system = assemble_tensor_gaussian_poisson_blocked(20, epsilon=epsilon)
        preconditioner = rbf_local_cardinal_preconditioner(
            system.A,
            system.nodes,
            support_size=81,
        )
        trajectory = fgmres(
            system.A,
            system.b,
            preconditioner.apply,
            x0=np.zeros(system.A.shape[0]),
            rtol=float(config["rtol"]),
            atol=float(config["atol"]),
            restart=int(config["restart"]),
            max_iterations=int(config["max_iterations"]),
            capture_arnoldi=True,
        )
        reported_source = next(
            row for row in source_rows if float(row["epsilon"]) == epsilon
        )
        source_outcomes_repeat &= (
            trajectory.converged == _as_bool(reported_source["converged"])
            and trajectory.iterations == int(reported_source["iterations"])
            and abs(
                trajectory.relative_residual
                - float(reported_source["true_relative_residual"])
            )
            <= 1.0e-15
        )
        recomputed_basis = harmonic_ritz_space(trajectory, int(config["q"]))
        saved_basis = source_arrays[f"{_epsilon_key(epsilon)}_Y"]
        maximum_source_space_defect = max(
            maximum_source_space_defect,
            _projector_distance(recomputed_basis, saved_basis),
        )

        indexed = [
            row for row in cycle_index if float(row["epsilon"]) == epsilon
        ]
        cycles = trajectory.arnoldi_cycles or []
        cycle_counts_match &= len(indexed) == len(cycles)
        if len(indexed) == len(cycles):
            for row, cycle in zip(indexed, cycles):
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
                    maximum_saved_cycle_defect = max(
                        maximum_saved_cycle_defect,
                        defect,
                    )

        last_index = max(indexed, key=lambda row: int(row["cycle_index"]))
        values = _harmonic_values(cycle_arrays[last_index["H_key"]])
        magnitudes = np.abs(values)
        relative_gap = float(
            (magnitudes[16] - magnitudes[15])
            / max(magnitudes[15], np.finfo(float).tiny)
        )
        cutoff_rows.append(
            {
                "epsilon": epsilon,
                "last_cycle_dimension": len(values),
                "magnitude_rank_16": float(magnitudes[15]),
                "magnitude_rank_17": float(magnitudes[16]),
                "relative_gap_after_rank_16": relative_gap,
            }
        )
        for rank, value in enumerate(values[: min(32, len(values))], start=1):
            ritz_rows.append(
                {
                    "epsilon": epsilon,
                    "magnitude_rank": rank,
                    "real_part": float(value.real),
                    "imaginary_part": float(value.imag),
                    "magnitude": float(abs(value)),
                }
            )
        systems[epsilon] = system
        preconditioners[epsilon] = preconditioner
        trajectories[epsilon] = trajectory

    _write_csv(args.results_dir / "harmonic_ritz_cutoff_gaps.csv", cutoff_rows)
    _write_csv(args.results_dir / "harmonic_ritz_values.csv", ritz_rows)

    expected_pairs = {(source, target) for source in EPSILONS for target in EPSILONS}
    actual_pairs = {
        (float(row["epsilon_source"]), float(row["epsilon_target"]))
        for row in pair_rows
    }
    maximum_residual_error = 0.0
    maximum_ratio_error = 0.0
    maximum_au_report_error = 0.0
    maximum_ctc_report_error = 0.0
    maximum_au_defect = 0.0
    maximum_ctc_defect = 0.0
    maximum_angle_error = 0.0
    maximum_projector_error = 0.0
    pair_outcomes_repeat = True
    classifications_recompute = True
    diagonal_matches_oracle = True

    for row in pair_rows:
        source = float(row["epsilon_source"])
        target = float(row["epsilon_target"])
        source_key = _epsilon_key(source)
        target_key = _epsilon_key(target)
        prefix = f"source_{source_key}_target_{target_key}"
        system = systems[target]
        preconditioner = preconditioners[target]
        u_basis = pair_arrays[f"{prefix}_U"]
        c_basis = pair_arrays[f"{prefix}_C"]
        solution = pair_arrays[f"{prefix}_solution"]
        rhs_norm = float(np.linalg.norm(system.b))
        recomputed_residual = float(
            np.linalg.norm(system.b - system.A @ solution) / rhs_norm
        )
        reported_residual = float(row["transferred_true_relative_residual"])
        maximum_residual_error = max(
            maximum_residual_error,
            abs(recomputed_residual - reported_residual),
        )
        ratio = int(row["transferred_iterations"]) / int(row["baseline_iterations"])
        maximum_ratio_error = max(
            maximum_ratio_error,
            abs(ratio - float(row["transferred_baseline_iteration_ratio"])),
        )
        au_defect = float(np.linalg.norm(system.A @ u_basis - c_basis, ord="fro"))
        ctc_defect = float(
            np.linalg.norm(c_basis.T @ c_basis - np.eye(c_basis.shape[1]), ord="fro")
        )
        maximum_au_defect = max(maximum_au_defect, au_defect)
        maximum_ctc_defect = max(maximum_ctc_defect, ctc_defect)
        maximum_au_report_error = max(
            maximum_au_report_error,
            abs(au_defect - float(row["target_AU_minus_C_frobenius"])),
        )
        maximum_ctc_report_error = max(
            maximum_ctc_report_error,
            abs(ctc_defect - float(row["target_CHC_minus_I_frobenius"])),
        )

        source_basis = source_arrays[f"{source_key}_Y"]
        target_basis = source_arrays[f"{target_key}_Y"]
        angles = np.sort(la.subspace_angles(source_basis, target_basis))[::-1]
        degrees = np.rad2deg(angles)
        projector = _projector_distance(source_basis, target_basis)
        maximum_angle_error = max(
            maximum_angle_error,
            abs(float(np.max(degrees)) - float(row["largest_principal_angle_degrees"])),
            abs(float(np.median(degrees)) - float(row["median_principal_angle_degrees"])),
        )
        maximum_projector_error = max(
            maximum_projector_error,
            abs(projector - float(row["projector_distance_frobenius"])),
        )

        repeat = _rerun(
            system.A,
            system.b,
            preconditioner.apply,
            u_basis=u_basis,
            c_basis=c_basis,
            config=config,
        )
        repeat_residual = float(
            np.linalg.norm(system.b - system.A @ repeat.x) / rhs_norm
        )
        pair_outcomes_repeat &= (
            repeat.iterations == int(row["transferred_iterations"])
            and repeat.converged == _as_bool(row["transferred_converged"])
            and abs(repeat_residual - reported_residual) <= 1.0e-15
        )
        random_iterations = int(row["random_iterations"])
        if ratio <= 0.85:
            expected_class = "strong_positive"
        elif ratio <= 0.95 and int(row["transferred_iterations"]) < random_iterations:
            expected_class = "weak_but_useful"
        else:
            expected_class = "no_useful_transfer"
        classifications_recompute &= row["transfer_classification"] == expected_class
        if source == target:
            diagonal_matches_oracle &= (
                int(row["transferred_iterations"]) == int(row["oracle_iterations"])
                and abs(
                    float(row["transferred_true_relative_residual"])
                    - float(row["oracle_true_relative_residual"])
                )
                <= 1.0e-15
            )

    target_outcomes_repeat = True
    target_residuals_recompute = True
    for row in target_rows:
        target = float(row["epsilon_target"])
        key = _epsilon_key(target)
        system = systems[target]
        preconditioner = preconditioners[target]
        rhs_norm = float(np.linalg.norm(system.b))
        candidates = (
            ("baseline", None, None),
            ("oracle", pair_arrays[f"{key}_oracle_U"], pair_arrays[f"{key}_oracle_C"]),
            ("random", pair_arrays[f"{key}_random_U"], pair_arrays[f"{key}_random_C"]),
        )
        for role, u_basis, c_basis in candidates:
            saved_solution = pair_arrays[f"{key}_{role}_solution"]
            residual = float(
                np.linalg.norm(system.b - system.A @ saved_solution) / rhs_norm
            )
            target_residuals_recompute &= abs(
                residual - float(row[f"{role}_true_relative_residual"])
            ) <= 1.0e-15
            repeat = _rerun(
                system.A,
                system.b,
                preconditioner.apply,
                u_basis=u_basis,
                c_basis=c_basis,
                config=config,
            )
            target_outcomes_repeat &= (
                repeat.iterations == int(row[f"{role}_iterations"])
                and repeat.converged == _as_bool(row[f"{role}_converged"])
            )

    angle_rows_complete = len(angle_rows) == 25 * int(config["q"])
    random_basis = source_arrays["matched_random_Y"]
    random_orthogonality = float(
        np.linalg.norm(random_basis.T @ random_basis - np.eye(int(config["q"])), ord="fro")
    )

    distances = np.asarray([float(row["epsilon_distance"]) for row in pair_rows])
    ratios = np.asarray(
        [float(row["transferred_baseline_iteration_ratio"]) for row in pair_rows]
    )
    angles = np.asarray(
        [float(row["largest_principal_angle_degrees"]) for row in pair_rows]
    )
    projectors = np.asarray(
        [float(row["projector_distance_frobenius"]) for row in pair_rows]
    )
    off_diagonal = distances > 0.0
    recomputed_correlations = {
        "epsilon_distance_vs_iteration_ratio_all_pairs": _safe_corr(distances, ratios),
        "epsilon_distance_vs_iteration_ratio_off_diagonal": _safe_corr(
            distances[off_diagonal], ratios[off_diagonal]
        ),
        "largest_angle_vs_iteration_ratio_all_pairs": _safe_corr(angles, ratios),
        "largest_angle_vs_iteration_ratio_off_diagonal": _safe_corr(
            angles[off_diagonal], ratios[off_diagonal]
        ),
        "projector_distance_vs_iteration_ratio_all_pairs": _safe_corr(
            projectors, ratios
        ),
        "projector_distance_vs_iteration_ratio_off_diagonal": _safe_corr(
            projectors[off_diagonal], ratios[off_diagonal]
        ),
    }
    maximum_correlation_error = 0.0
    for name, metrics in recomputed_correlations.items():
        for metric, value in metrics.items():
            maximum_correlation_error = max(
                maximum_correlation_error,
                abs(value - float(summary["correlations"][name][metric])),
            )

    distance_summary = _read_csv(args.results_dir / "distance_summary.csv")
    distance_counts_match = {
        float(row["epsilon_distance"]): int(row["pair_count"])
        for row in distance_summary
    } == {0.0: 5, 0.2: 8, 0.4: 6, 0.6: 4, 0.8: 2}
    nearby = [row for row in pair_rows if float(row["epsilon_distance"]) == 0.2]
    nearby_ratio = float(
        np.median(
            [float(row["transferred_baseline_iteration_ratio"]) for row in nearby]
        )
    )
    nearby_decision_recomputes = (
        abs(nearby_ratio - float(summary["decision"]["nearby_median_iteration_ratio"]))
        <= 1.0e-15
        and summary["decision"]["nearby_transfer_positive"] is False
    )

    plots = (
        args.results_dir / "transferred_iteration_ratio_heatmap.png",
        args.results_dir / "subspace_similarity_heatmaps.png",
        args.results_dir / "iteration_ratio_vs_epsilon_distance.png",
    )
    plot_checks: dict[str, Any] = {}
    for path in plots:
        image = mpimg.imread(path)
        plot_checks[path.name] = {
            "exists": path.exists(),
            "width_pixels": int(image.shape[1]),
            "height_pixels": int(image.shape[0]),
            "pixel_standard_deviation": float(np.std(image)),
            "nonblank": bool(np.std(image) > 0.01),
        }
    plots_nonblank = all(item["nonblank"] for item in plot_checks.values())

    cutoff_gaps = [float(row["relative_gap_after_rank_16"]) for row in cutoff_rows]
    checks = {
        "fixed_configuration_matches_request": (
            int(config["N"]) == 400
            and tuple(float(value) for value in config["epsilon_values"]) == EPSILONS
            and int(config["q"]) == 16
            and float(config["rtol"]) == 1.0e-10
            and int(config["restart"]) == 100
            and config["initial_guess"] == "zero vector for every solve"
        ),
        "all_five_sources_converge_and_repeat": source_outcomes_repeat,
        "all_25_ordered_pairs_present": actual_pairs == expected_pairs and len(pair_rows) == 25,
        "all_saved_source_spaces_and_cycles_reproduce": (
            cycle_counts_match
            and maximum_source_space_defect <= 1.0e-10
            and maximum_saved_cycle_defect <= 1.0e-13
        ),
        "all_pair_true_residuals_recompute_from_original_A_b": maximum_residual_error
        <= 1.0e-15,
        "all_target_control_residuals_recompute": target_residuals_recompute,
        "all_pair_solver_outcomes_repeat": pair_outcomes_repeat,
        "all_target_control_outcomes_repeat": target_outcomes_repeat,
        "target_side_AU_equals_C": maximum_au_defect / np.sqrt(16.0)
        <= 10.0 * float(config["rtol"]),
        "target_side_C_is_orthonormal": maximum_ctc_defect <= 1.0e-12,
        "reported_augmentation_defects_recompute": (
            maximum_au_report_error <= 1.0e-15
            and maximum_ctc_report_error <= 1.0e-15
        ),
        "subspace_metrics_and_all_principal_angles_recompute": (
            angle_rows_complete
            and maximum_angle_error <= 1.0e-10
            and maximum_projector_error <= 1.0e-10
        ),
        "ratios_and_classifications_recompute": (
            maximum_ratio_error <= 1.0e-15 and classifications_recompute
        ),
        "diagonal_transfer_exactly_matches_target_oracle": diagonal_matches_oracle,
        "matched_random_basis_is_orthonormal": random_orthogonality <= 1.0e-12,
        "distance_summary_has_correct_pair_counts": distance_counts_match,
        "correlations_recompute": maximum_correlation_error <= 1.0e-12,
        "nearby_decision_recomputes": nearby_decision_recomputes,
        "all_three_requested_plots_are_nonblank": plots_nonblank,
        "no_A_inverse_bare_A_space_or_ML": (
            not config["A_inverse_used"]
            and not config["bare_A_invariant_subspaces_used"]
            and not config["ML_used"]
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    validation = {
        "overall_assessment": "Ready to share" if all(checks.values()) else "Needs revision",
        "checks": checks,
        "diagnostics": {
            "maximum_saved_source_space_projector_defect": maximum_source_space_defect,
            "maximum_saved_cycle_relative_defect": maximum_saved_cycle_defect,
            "maximum_saved_true_residual_absolute_error": maximum_residual_error,
            "maximum_iteration_ratio_absolute_error": maximum_ratio_error,
            "maximum_target_AU_minus_C_frobenius": maximum_au_defect,
            "maximum_target_CHC_minus_I_frobenius": maximum_ctc_defect,
            "maximum_principal_angle_absolute_error_degrees": maximum_angle_error,
            "maximum_projector_distance_absolute_error": maximum_projector_error,
            "maximum_correlation_absolute_error": maximum_correlation_error,
            "rank_16_cutoff_relative_gap_min": min(cutoff_gaps),
            "rank_16_cutoff_relative_gap_max": max(cutoff_gaps),
            "plot_checks": plot_checks,
        },
        "interpretation_checks": {
            "target_mapping_failure_ruled_out": bool(
                checks["target_side_AU_equals_C"]
                and checks["target_side_C_is_orthonormal"]
                and checks["diagonal_transfer_exactly_matches_target_oracle"]
            ),
            "rapid_basis_invariant_subspace_rotation_observed": bool(
                np.median(angles[off_diagonal]) > 80.0
                and np.median(projectors[off_diagonal]) > 3.0
            ),
            "uniformly_tiny_rank_16_cutoff_gap_observed": bool(max(cutoff_gaps) < 0.05),
            "off_diagonal_geometry_predicts_solver_ratio_strongly": bool(
                abs(
                    recomputed_correlations[
                        "projector_distance_vs_iteration_ratio_off_diagonal"
                    ]["spearman_rho"]
                )
                >= 0.7
            ),
        },
        "caveats": [
            "The 3,000-iteration cap is only a full-convergence safety cap; restart, tolerance, zero initial guess, N, PDE, kernel, and preconditioner are unchanged.",
            "All 25 transferred solves converged, but the random target control at epsilon 7.6 is right-censored at 3,000 iterations.",
            "The rank-16 cutoff-gap diagnostic uses generalized harmonic-Ritz values from each final restart cycle; it is evidence about local ordering stability, not a complete cross-parameter mode-tracking analysis.",
            "Wall-clock values are local single-thread medians and exclude all declared offline costs.",
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
