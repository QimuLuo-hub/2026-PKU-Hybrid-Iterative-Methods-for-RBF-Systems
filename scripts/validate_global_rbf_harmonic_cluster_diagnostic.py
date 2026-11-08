#!/usr/bin/env python3
"""Independently validate the harmonic-Ritz cluster diagnostic artifacts."""

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

from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from neurkitt_oracle.gcrodr import restarted_projected_gmres


EPSILONS = (7.8, 8.0, 8.2, 8.4)
PAIRS = ((7.8, 8.0), (8.0, 7.8), (8.0, 8.2), (8.2, 8.0), (8.2, 8.4), (8.4, 8.2))
GAP_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_harmonic_cluster_diagnostic"),
    )
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _eps_key(epsilon: float) -> str:
    return f"eps_{epsilon:.1f}".replace(".", "p")


def _gaps(magnitudes: np.ndarray) -> np.ndarray:
    return np.diff(magnitudes) / np.maximum(magnitudes[:-1], np.finfo(float).tiny)


def _cluster_end(magnitudes: np.ndarray, boundary: int) -> int:
    gaps = _gaps(magnitudes)
    end = int(boundary)
    while end < len(magnitudes) and gaps[end - 1] < GAP_THRESHOLD:
        end += 1
    return end


def _common_boundary(left: np.ndarray, right: np.ndarray) -> int:
    boundary = 16
    while True:
        extended = max(_cluster_end(left, boundary), _cluster_end(right, boundary))
        if extended == boundary:
            return boundary
        boundary = extended


def _spectrum(hbar: np.ndarray) -> np.ndarray:
    m = hbar.shape[1]
    hm = hbar[:m, :]
    tail = hbar[m, m - 1] if hbar.shape[0] > m else 0.0
    left = hm.T @ hm
    left[-1, -1] += tail**2
    values = la.eigvals(left, hm.T, check_finite=False)
    values = values[np.isfinite(values)]
    return values[np.argsort(np.abs(values))]


def _projector_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    q_left, _ = la.qr(left, mode="economic", check_finite=False)
    q_right, _ = la.qr(right, mode="economic", check_finite=False)
    angles = np.sort(np.rad2deg(la.subspace_angles(q_left, q_right)))[::-1]
    radians = np.deg2rad(angles)
    distance = float(np.sqrt(2.0 * np.sum(np.sin(radians) ** 2)))
    return float(np.max(angles)), float(np.median(angles)), distance, distance / np.sqrt(2.0 * len(angles))


def _rerun(matrix, rhs, apply_preconditioner, u_basis, c_basis, config):
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
    spectra = _read(args.results_dir / "harmonic_ritz_spectra.csv")
    source_rows = _read(args.results_dir / "source_extractions.csv")
    self_rows = _read(args.results_dir / "self_oracle_results.csv")
    transfer_rows = _read(args.results_dir / "adjacent_transfer_results.csv")
    angle_rows = _read(args.results_dir / "principal_angles.csv")
    cycle_index = _read(args.results_dir / "source_krylov_cycle_index.csv")
    arrays = np.load(args.results_dir / "recycle_spaces_and_solutions.npz")
    cycles = np.load(args.results_dir / "source_krylov_cycles.npz")

    systems: dict[float, Any] = {}
    preconditioners: dict[float, Any] = {}
    magnitudes: dict[float, np.ndarray] = {}
    maximum_spectrum_error = 0.0
    maximum_gap_error = 0.0
    local_ranks: dict[str, int] = {}
    for epsilon in EPSILONS:
        system = assemble_tensor_gaussian_poisson_blocked(20, epsilon=epsilon)
        preconditioner = rbf_local_cardinal_preconditioner(system.A, system.nodes, support_size=81)
        selected = [row for row in cycle_index if float(row["epsilon"]) == epsilon]
        eligible = [row for row in selected if int(row["dimension"]) >= 30]
        last = max(eligible, key=lambda row: int(row["cycle_index"]))
        values = _spectrum(cycles[last["H_key"]])
        mags = np.abs(values)
        reported = sorted(
            [row for row in spectra if float(row["epsilon"]) == epsilon],
            key=lambda row: int(row["mode_index"]),
        )
        for index, row in enumerate(reported):
            maximum_spectrum_error = max(
                maximum_spectrum_error,
                abs(float(row["real_part"]) - values[index].real),
                abs(float(row["imaginary_part"]) - values[index].imag),
                abs(float(row["magnitude"]) - mags[index]),
            )
            maximum_gap_error = max(
                maximum_gap_error,
                abs(float(row["relative_gap_to_next"]) - _gaps(mags)[index]),
            )
        local_ranks[f"{epsilon:.1f}"] = _cluster_end(mags, 16)
        systems[epsilon] = system
        preconditioners[epsilon] = preconditioner
        magnitudes[epsilon] = mags

    pair_ranks = {
        f"{source:.1f}->{target:.1f}": _common_boundary(magnitudes[source], magnitudes[target])
        for source, target in PAIRS
    }

    maximum_residual_error = 0.0
    maximum_au_defect = 0.0
    maximum_ctc_defect = 0.0
    maximum_geometry_error = 0.0
    all_saved_ranks_match = True
    all_reported_outcomes_repeat = True
    rerun_cache: dict[tuple[Any, ...], Any] = {}

    for row in self_rows:
        epsilon = float(row["epsilon"])
        q = int(row["q"])
        key = _eps_key(epsilon)
        system = systems[epsilon]
        preconditioner = preconditioners[epsilon]
        rhs_norm = float(np.linalg.norm(system.b))
        for role in ("oracle", "random"):
            u_basis = arrays[f"{key}_{role}_q{q}_U"]
            c_basis = arrays[f"{key}_{role}_q{q}_C"]
            solution = arrays[f"{key}_{role}_q{q}_solution"]
            all_saved_ranks_match &= u_basis.shape == (400, q) and c_basis.shape == (400, q)
            residual = float(np.linalg.norm(system.b - system.A @ solution) / rhs_norm)
            maximum_residual_error = max(
                maximum_residual_error,
                abs(residual - float(row[f"{role}_true_relative_residual"])),
            )
            maximum_au_defect = max(maximum_au_defect, float(np.linalg.norm(system.A @ u_basis - c_basis, ord="fro")))
            maximum_ctc_defect = max(maximum_ctc_defect, float(np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")))
            cache_key = (epsilon, q, role)
            if cache_key not in rerun_cache:
                rerun_cache[cache_key] = _rerun(system.A, system.b, preconditioner.apply, u_basis, c_basis, config)
            repeat = rerun_cache[cache_key]
            all_reported_outcomes_repeat &= (
                repeat.iterations == int(row[f"{role}_iterations"])
                and repeat.converged == (row[f"{role}_converged"] == "True")
            )

    for epsilon in EPSILONS:
        key = _eps_key(epsilon)
        row = next(item for item in self_rows if float(item["epsilon"]) == epsilon)
        system = systems[epsilon]
        preconditioner = preconditioners[epsilon]
        solution = arrays[f"{key}_baseline_solution"]
        residual = float(np.linalg.norm(system.b - system.A @ solution) / np.linalg.norm(system.b))
        maximum_residual_error = max(maximum_residual_error, abs(residual - float(row["baseline_true_relative_residual"])))
        repeat = _rerun(system.A, system.b, preconditioner.apply, None, None, config)
        all_reported_outcomes_repeat &= repeat.iterations == int(row["baseline_iterations"])

    unique_transfers: set[tuple[float, float, int]] = set()
    for row in transfer_rows:
        source = float(row["epsilon_source"])
        target = float(row["epsilon_target"])
        q = int(row["q"])
        source_key = _eps_key(source)
        target_key = _eps_key(target)
        prefix = f"source_{source_key}_target_{target_key}_q{q}"
        system = systems[target]
        preconditioner = preconditioners[target]
        u_basis = arrays[f"{prefix}_U"]
        c_basis = arrays[f"{prefix}_C"]
        solution = arrays[f"{prefix}_solution"]
        residual = float(np.linalg.norm(system.b - system.A @ solution) / np.linalg.norm(system.b))
        maximum_residual_error = max(
            maximum_residual_error,
            abs(residual - float(row["transferred_true_relative_residual"])),
        )
        au_defect = float(np.linalg.norm(system.A @ u_basis - c_basis, ord="fro"))
        ctc_defect = float(np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro"))
        maximum_au_defect = max(maximum_au_defect, au_defect)
        maximum_ctc_defect = max(maximum_ctc_defect, ctc_defect)
        metrics = _projector_metrics(arrays[f"{source_key}_Y_q{q}"], arrays[f"{target_key}_Y_q{q}"])
        reported_metrics = (
            float(row["largest_principal_angle_degrees"]),
            float(row["median_principal_angle_degrees"]),
            float(row["projector_distance_frobenius"]),
            float(row["projector_distance_normalized"]),
        )
        maximum_geometry_error = max(
            maximum_geometry_error,
            *(abs(left - right) for left, right in zip(metrics, reported_metrics)),
        )
        cache_key = (source, target, q)
        if cache_key not in unique_transfers:
            repeat = _rerun(system.A, system.b, preconditioner.apply, u_basis, c_basis, config)
            all_reported_outcomes_repeat &= (
                repeat.iterations == int(row["transferred_iterations"])
                and repeat.converged == (row["transferred_converged"] == "True")
            )
            unique_transfers.add(cache_key)

    cluster_rows = [row for row in transfer_rows if row["candidate"] == "cluster_aware"]
    q16_rows = [row for row in transfer_rows if row["candidate"] == "q16"]
    cluster_ratios = np.asarray([float(row["transferred_baseline_iteration_ratio"]) for row in cluster_rows])
    q16_ratios = np.asarray([float(row["transferred_baseline_iteration_ratio"]) for row in q16_rows])
    cluster_angles = np.asarray([float(row["largest_principal_angle_degrees"]) for row in cluster_rows])
    q16_angles = np.asarray([float(row["largest_principal_angle_degrees"]) for row in q16_rows])
    cluster_projectors = np.asarray([float(row["projector_distance_normalized"]) for row in cluster_rows])
    q16_projectors = np.asarray([float(row["projector_distance_normalized"]) for row in q16_rows])
    geometry_improves = bool(
        np.median(cluster_angles) < np.median(q16_angles)
        and np.median(cluster_projectors) < np.median(q16_projectors)
    )
    positive_count = int(np.count_nonzero(cluster_ratios <= 0.90))
    better_random = sum(int(row["transferred_iterations"]) < int(row["random_iterations"]) for row in cluster_rows)
    if positive_count >= 4 and np.median(cluster_ratios) < np.median(q16_ratios) and better_random >= 4:
        decision = "POSITIVE"
    elif 0.90 <= np.median(cluster_ratios) <= 0.95 and geometry_improves:
        decision = "WEAK"
    else:
        decision = "NEGATIVE"

    plots = (
        args.results_dir / "harmonic_ritz_spectra_and_gaps.png",
        args.results_dir / "adjacent_transfer_iteration_ratios.png",
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

    checks = {
        "fixed_configuration_matches_request": (
            int(config["N"]) == 400
            and tuple(config["epsilon_values"]) == EPSILONS
            and config["right_preconditioner"].startswith("LS-ACBF k=81")
            and float(config["rtol"]) == 1.0e-10
            and int(config["restart"]) == 100
        ),
        "four_sources_and_120_spectrum_rows_present": len(source_rows) == 4 and len(spectra) == 120,
        "six_pairs_and_four_candidates_present": len(transfer_rows) == 24,
        "spectra_and_gaps_recompute_from_saved_Arnoldi_data": maximum_spectrum_error <= 1.0e-14 and maximum_gap_error <= 1.0e-14,
        "local_cluster_ranks_recompute": local_ranks == summary["local_cluster_ranks"],
        "pair_common_cluster_ranks_recompute": pair_ranks == summary["pair_common_cluster_ranks"],
        "saved_basis_and_augmentation_ranks_match": all_saved_ranks_match,
        "all_true_residuals_recompute_from_original_A_b": maximum_residual_error <= 1.0e-15,
        "target_side_AU_equals_C": maximum_au_defect / np.sqrt(24.0) <= 1.0e-9,
        "target_side_C_is_orthonormal": maximum_ctc_defect <= 1.0e-12,
        "all_solver_outcomes_repeat": all_reported_outcomes_repeat,
        "all_geometry_metrics_recompute": maximum_geometry_error <= 1.0e-10,
        "all_principal_angles_saved": len(angle_rows) == sum(int(row["q"]) for row in transfer_rows),
        "decision_recomputes": decision == summary["decision"]["classification"],
        "plots_nonblank": all(item["nonblank"] for item in plot_checks.values()),
        "no_ML": config["ML_used"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    validation = {
        "overall_assessment": "Ready to share" if all(checks.values()) else "Needs revision",
        "checks": checks,
        "diagnostics": {
            "maximum_spectrum_absolute_error": maximum_spectrum_error,
            "maximum_gap_absolute_error": maximum_gap_error,
            "maximum_true_residual_absolute_error": maximum_residual_error,
            "maximum_AU_minus_C_frobenius": maximum_au_defect,
            "maximum_CHC_minus_I_frobenius": maximum_ctc_defect,
            "maximum_geometry_metric_absolute_error": maximum_geometry_error,
            "unique_transferred_solves_rerun": len(unique_transfers),
            "decision": decision,
            "plot_checks": plot_checks,
        },
    }
    (args.results_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
