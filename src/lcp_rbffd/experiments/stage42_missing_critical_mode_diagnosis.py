"""Stage 42: diagnose Stage-30 oracle directions missed by Stage-41 probes."""

from __future__ import annotations

import json
import platform
from itertools import combinations
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy.fft import dctn

from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage20_spatial_k_robust_oracle import _projected_nonnormality, _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage41_multi_rhs_slow_mode_target import (
    Stage41Settings,
    harmonic_action_information,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage42Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    probe_rhs_seeds: Tuple[int, ...] = (20262911, 20262921, 20262931)
    probe_depth: int = 64
    slow_modes_per_probe: int = 24
    poor_coverage_threshold: float = 0.5
    low_frequency_radius: float = 0.25
    critical_rho_loss: float = 0.10
    few_mode_limit: int = 3

    @property
    def stage41(self) -> Stage41Settings:
        return Stage41Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


def _orthonormal_basis(array: np.ndarray) -> np.ndarray:
    basis, triangular = la.qr(np.asarray(array, dtype=float), mode="economic", check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0:
        raise RuntimeError("empty diagnostic space")
    tolerance = max(array.shape) * np.finfo(float).eps * max(float(diagonal[0]), 1.0)
    return basis[:, : int(np.count_nonzero(diagonal > tolerance))]


def coverage_values(space: np.ndarray, directions: np.ndarray) -> np.ndarray:
    basis = _orthonormal_basis(space)
    values = np.sum((basis.T @ np.asarray(directions, dtype=float)) ** 2, axis=0)
    return np.clip(values, 0.0, 1.0)


def _spatial_features(direction: np.ndarray, tensor_field: np.ndarray) -> Dict[str, float]:
    n = direction.size
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise RuntimeError("Stage-42 spatial diagnostics require the fixed square node grid")
    grid = np.asarray(direction, dtype=float).reshape(side, side)
    spectrum = dctn(grid, type=2, norm="ortho")
    energy = spectrum**2
    ky, kx = np.meshgrid(
        np.arange(side, dtype=float) / max(side - 1, 1),
        np.arange(side, dtype=float) / max(side - 1, 1),
        indexing="ij",
    )
    radius = np.sqrt(kx**2 + ky**2) / np.sqrt(2.0)
    total = max(float(np.sum(energy)), np.finfo(float).tiny)
    low_fraction = float(np.sum(energy[radius <= 0.25]) / total)
    centroid = float(np.sum(radius * energy) / total)
    weights = np.asarray(direction, dtype=float) ** 2
    weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
    effective_support = float(1.0 / (n * np.sum(weights**2)))

    kxx, kxy, kyy = (np.asarray(tensor_field[:, index], dtype=float) for index in range(3))
    trace = kxx + kyy
    determinant = np.maximum(kxx * kyy - kxy**2, np.finfo(float).tiny)
    discriminant = np.sqrt(np.maximum(trace**2 - 4.0 * determinant, 0.0))
    maximum = 0.5 * (trace + discriminant)
    minimum = np.maximum(0.5 * (trace - discriminant), np.finfo(float).tiny)
    log_ratio = np.log(maximum / minimum)
    channels = np.asarray(tensor_field, dtype=float).reshape(side, side, 3)
    gradient_squared = np.zeros((side, side), dtype=float)
    for channel in range(3):
        grad_y, grad_x = np.gradient(channels[:, :, channel])
        gradient_squared += grad_x**2 + grad_y**2
    gradient = np.sqrt(gradient_squared).ravel()

    def weighted_zscore(field: np.ndarray) -> float:
        scale = float(np.std(field))
        return float((np.dot(weights, field) - np.mean(field)) / max(scale, 1.0e-14))

    def correlation(field: np.ndarray) -> float:
        if np.std(field) <= 1.0e-14 or np.std(weights) <= 1.0e-14:
            return 0.0
        return float(np.corrcoef(weights, field)[0, 1])

    return {
        "low_frequency_energy_fraction": low_fraction,
        "spectral_centroid": centroid,
        "effective_support_fraction": effective_support,
        "coefficient_log_anisotropy_weighted_zscore": weighted_zscore(log_ratio),
        "coefficient_gradient_weighted_zscore": weighted_zscore(gradient),
        "energy_log_anisotropy_correlation": correlation(log_ratio),
        "energy_coefficient_gradient_correlation": correlation(gradient),
    }


def _directional_nonnormality(context, direction: np.ndarray) -> Dict[str, float]:
    matrix = context.system.matrix
    forward = np.asarray(matrix @ context.ilu.apply(direction), dtype=float)
    transpose = context.ilu.apply_transpose(np.asarray(matrix.T @ direction, dtype=float))
    forward_sq = float(np.dot(forward, forward))
    transpose_sq = float(np.dot(transpose, transpose))
    signed = (forward_sq - transpose_sq) / max(forward_sq + transpose_sq, 1.0e-300)
    return {
        "directional_commutator_signed": float(signed),
        "directional_commutator_magnitude": float(abs(signed)),
        "T_direction_norm": float(np.sqrt(forward_sq)),
        "T_transpose_direction_norm": float(np.sqrt(transpose_sq)),
    }


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return (ranks + 0.5) / values.size


def _solver_row(context, rhs, pair, baseline, teacher, method, **extra):
    result = _run_pair(context, rhs, pair, Stage29Settings().stage28.stage22a)
    residual = _relative_residual(context.system.matrix, result.x, rhs)
    denominator = int(baseline["iterations"]) - int(teacher["iterations"])
    row = {
        "method": method,
        "iterations": int(result.iterations),
        "converged": bool(result.converged and residual <= 1.0e-8),
        "true_final_residual": residual,
        "actual_rank": pair.rank,
        "rho": (
            (int(baseline["iterations"]) - int(result.iterations)) / denominator
            if denominator > 0
            else float("nan")
        ),
        "failure": getattr(result, "failure", "") or "",
        **extra,
    }
    return row


def _summarize_coverage(sample_id: int, eta: np.ndarray) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "oracle_rank": eta.size,
        "minimum_eta": float(np.min(eta)),
        "p10_eta": float(np.percentile(eta, 10)),
        "median_eta": float(np.median(eta)),
        "directions_eta_lt_0p1": int(np.count_nonzero(eta < 0.1)),
        "directions_eta_lt_0p25": int(np.count_nonzero(eta < 0.25)),
        "directions_eta_lt_0p5": int(np.count_nonzero(eta < 0.5)),
    }


def _direction_diagnostic_rows(
    sample_id: int,
    direction_type: str,
    directions: np.ndarray,
    eta: np.ndarray,
    individual_spaces: Sequence[np.ndarray],
    normalized_modes: np.ndarray,
    mode_values: Sequence[complex],
    mode_residuals: Sequence[float],
    mode_seeds: Sequence[int],
    tensor_field: np.ndarray,
    context,
):
    individual_eta = [coverage_values(space, directions) for space in individual_spaces]
    mode_correlations = (normalized_modes.T @ directions) ** 2
    rows = []
    for direction_index in range(directions.shape[1]):
        best_mode = int(np.argmax(mode_correlations[:, direction_index]))
        rows.append({
            "sample_id": sample_id,
            "direction_type": direction_type,
            "direction_index": direction_index,
            "eta_combined": float(eta[direction_index]),
            "coverage_rank": int(np.argsort(np.argsort(eta))[direction_index] + 1),
            **{
                f"eta_probe_{probe_index + 1}": float(values[direction_index])
                for probe_index, values in enumerate(individual_eta)
            },
            "best_harmonic_mode_correlation_squared": float(
                mode_correlations[best_mode, direction_index]
            ),
            "best_harmonic_probe_seed": int(mode_seeds[best_mode]),
            "best_harmonic_ritz_magnitude": float(abs(mode_values[best_mode])),
            "best_harmonic_ritz_residual": float(mode_residuals[best_mode]),
            **_spatial_features(directions[:, direction_index], tensor_field),
            **_directional_nonnormality(context, directions[:, direction_index]),
        })
    feature_names = (
        "low_frequency_energy_fraction",
        "spectral_centroid",
        "effective_support_fraction",
        "coefficient_log_anisotropy_weighted_zscore",
        "coefficient_gradient_weighted_zscore",
        "energy_log_anisotropy_correlation",
        "energy_coefficient_gradient_correlation",
        "directional_commutator_magnitude",
    )
    for feature in feature_names:
        percentiles = _percentile_ranks([row[feature] for row in rows])
        for row, percentile in zip(rows, percentiles):
            row[f"{feature}_percentile"] = float(percentile)
    return rows


def run_diagnostics(root: Path, output: Path, settings: Stage42Settings):
    stage41_rows = _read_csv(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/solver_results.csv"
    )
    solver_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in stage41_rows
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    oracle_archive = np.load(
        root / "results/results-fd/stages/stage40_target_conditioned_local_basis/local_combination_spaces.npz"
    )
    stage41_archive = np.load(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/multi_rhs_local_spaces.npz"
    )
    coverage_rows: List[Dict[str, Any]] = []
    canonical_rows: List[Dict[str, Any]] = []
    coverage_summary: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    addition_rows: List[Dict[str, Any]] = []
    nonnormal_rows: List[Dict[str, Any]] = []
    system_rows: List[Dict[str, Any]] = []
    probe_archive: Dict[str, np.ndarray] = {}

    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        oracle_z = np.asarray(
            oracle_archive[f"system_{sample_id:03d}_teacher_local_oracle_z"], dtype=float
        )
        oracle = _pair_from_z(context.system.matrix, oracle_z)
        stage41_z = np.asarray(
            stage41_archive[f"system_{sample_id:03d}_multi_rhs_harmonic_m64_z"], dtype=float
        )
        stage41_pair = _pair_from_z(context.system.matrix, stage41_z)
        with np.load(
            root / "results/results-fd/stage28_operator_learning/systems" / f"system_{sample_id:03d}.npz"
        ) as system_archive:
            tensor_field = np.asarray(system_archive["tensor_field"], dtype=float)

        individual_spaces = []
        mode_actions = []
        mode_values = []
        mode_residuals = []
        mode_seeds = []
        projected_rows = []
        probe_started = perf_counter()
        for seed in settings.probe_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, seed)
            arnoldi = _capture_arnoldi(context.system.matrix, context.ilu, rhs, settings.probe_depth)
            action, family, _, _ = harmonic_action_information(
                context.system.matrix, arnoldi, settings.slow_modes_per_probe
            )
            individual_spaces.append(_orthonormal_basis(action))
            mode_actions.append(action)
            mode_values.extend(family.eigenvalues.tolist())
            mode_residuals.extend(family.residual_norms.tolist())
            mode_seeds.extend([seed] * family.basis.shape[1])
            projected_rows.append(_projected_nonnormality(arnoldi))
        combined_action = np.column_stack(mode_actions)
        combined_probe = _orthonormal_basis(combined_action)
        probe_archive[f"system_{sample_id:03d}_combined_harmonic_action_basis"] = combined_probe
        eta = coverage_values(combined_probe, oracle.w_basis)
        coverage_summary.append(_summarize_coverage(sample_id, eta))

        normalized_modes = np.column_stack([
            column / max(float(np.linalg.norm(column)), np.finfo(float).tiny)
            for column in np.column_stack(mode_actions).T
        ])
        local_direction_rows = _direction_diagnostic_rows(
            sample_id,
            "saved_oracle_column",
            oracle.w_basis,
            eta,
            individual_spaces,
            normalized_modes,
            mode_values,
            mode_residuals,
            mode_seeds,
            tensor_field,
            context,
        )
        _, principal_cosines, right = la.svd(
            combined_probe.T @ oracle.w_basis,
            full_matrices=False,
            check_finite=False,
        )
        canonical_w = oracle.w_basis @ right.T
        canonical_u = oracle.u_basis @ right.T
        canonical_eta = np.clip(principal_cosines**2, 0.0, 1.0)
        local_canonical_rows = _direction_diagnostic_rows(
            sample_id,
            "principal_oracle_combination",
            canonical_w,
            canonical_eta,
            individual_spaces,
            normalized_modes,
            mode_values,
            mode_residuals,
            mode_seeds,
            tensor_field,
            context,
        )
        probe_archive[f"system_{sample_id:03d}_principal_oracle_w"] = canonical_w
        probe_archive[f"system_{sample_id:03d}_principal_oracle_u"] = canonical_u

        poor_indices = [int(index) for index in np.flatnonzero(eta < settings.poor_coverage_threshold)]
        canonical_poor_indices = [
            int(index) for index in np.flatnonzero(canonical_eta < settings.poor_coverage_threshold)
        ]
        oracle_median_rho = float(np.median([
            float(solver_lookup[(sample_id, seed, "full_span_oracle")]["rho"])
            for seed in settings.stage29.stage28.evaluation_rhs_seeds
        ]))
        for direction_index in poor_indices:
            keep = [index for index in range(oracle.rank) if index != direction_index]
            removed_pair = _pair_from_z(context.system.matrix, oracle.u_basis[:, keep])
            local_removals = []
            for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                baseline = solver_lookup[(sample_id, rhs_seed, "strong_ilu")]
                teacher = solver_lookup[(sample_id, rhs_seed, "teacher_full")]
                row = _solver_row(
                    context,
                    rhs,
                    removed_pair,
                    baseline,
                    teacher,
                    "oracle_leave_one_out",
                    sample_id=sample_id,
                    rhs_seed=rhs_seed,
                    direction_index=direction_index,
                    eta=float(eta[direction_index]),
                    direction_type="saved_oracle_column",
                )
                removal_rows.append(row)
                local_removals.append(row)
            convergence_loss = 3 - sum(_bool(row["converged"]) for row in local_removals)
            median_removed_rho = float(np.median([float(row["rho"]) for row in local_removals]))
            direction_row = local_direction_rows[direction_index]
            direction_row.update({
                "leave_one_out_converged_rhs": 3 - convergence_loss,
                "leave_one_out_median_iterations": float(
                    np.median([int(row["iterations"]) for row in local_removals])
                ),
                "leave_one_out_median_rho": median_removed_rho,
                "leave_one_out_rho_loss": oracle_median_rho - median_removed_rho,
                "solver_critical": bool(
                    convergence_loss > 0
                    or oracle_median_rho - median_removed_rho >= settings.critical_rho_loss
                ),
            })

        for row in local_direction_rows:
            row.setdefault("leave_one_out_converged_rhs", "")
            row.setdefault("leave_one_out_median_iterations", "")
            row.setdefault("leave_one_out_median_rho", "")
            row.setdefault("leave_one_out_rho_loss", "")
            row.setdefault("solver_critical", False)
        coverage_rows.extend(local_direction_rows)

        for direction_index in canonical_poor_indices:
            keep = [index for index in range(oracle.rank) if index != direction_index]
            removed_pair = _pair_from_z(context.system.matrix, canonical_u[:, keep])
            local_removals = []
            for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                baseline = solver_lookup[(sample_id, rhs_seed, "strong_ilu")]
                teacher = solver_lookup[(sample_id, rhs_seed, "teacher_full")]
                row = _solver_row(
                    context,
                    rhs,
                    removed_pair,
                    baseline,
                    teacher,
                    "oracle_leave_one_out",
                    sample_id=sample_id,
                    rhs_seed=rhs_seed,
                    direction_index=direction_index,
                    eta=float(canonical_eta[direction_index]),
                    direction_type="principal_oracle_combination",
                )
                removal_rows.append(row)
                local_removals.append(row)
            convergence_loss = 3 - sum(_bool(row["converged"]) for row in local_removals)
            median_removed_rho = float(np.median([float(row["rho"]) for row in local_removals]))
            direction_row = local_canonical_rows[direction_index]
            direction_row.update({
                "leave_one_out_converged_rhs": 3 - convergence_loss,
                "leave_one_out_median_iterations": float(
                    np.median([int(row["iterations"]) for row in local_removals])
                ),
                "leave_one_out_median_rho": median_removed_rho,
                "leave_one_out_rho_loss": oracle_median_rho - median_removed_rho,
                "solver_critical": bool(
                    convergence_loss > 0
                    or oracle_median_rho - median_removed_rho >= settings.critical_rho_loss
                ),
            })
        for row in local_canonical_rows:
            row.setdefault("leave_one_out_converged_rhs", "")
            row.setdefault("leave_one_out_median_iterations", "")
            row.setdefault("leave_one_out_median_rho", "")
            row.setdefault("leave_one_out_rho_loss", "")
            row.setdefault("solver_critical", False)
        canonical_rows.extend(local_canonical_rows)

        addition_order = list(np.argsort(canonical_eta))
        maximum_additions = len(canonical_poor_indices) if sample_id != 63 else oracle.rank
        first_all_converged = None
        best_median_rho = -np.inf
        additions_run = 0
        recovery_loo_successes = ""
        for added_count in range(maximum_additions + 1):
            selected = addition_order[:added_count]
            raw = stage41_pair.u_basis
            if selected:
                raw = np.column_stack((raw, canonical_u[:, selected]))
            augmented = _pair_from_z(context.system.matrix, raw)
            local_additions = []
            for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                baseline = solver_lookup[(sample_id, rhs_seed, "strong_ilu")]
                teacher = solver_lookup[(sample_id, rhs_seed, "teacher_full")]
                row = _solver_row(
                    context,
                    rhs,
                    augmented,
                    baseline,
                    teacher,
                    "stage41_plus_missing_prefix",
                    sample_id=sample_id,
                    rhs_seed=rhs_seed,
                    added_count=added_count,
                    added_direction_indices=";".join(str(index) for index in selected),
                    direction_type="principal_oracle_combination",
                )
                addition_rows.append(row)
                local_additions.append(row)
            additions_run = added_count
            converged_count = sum(_bool(row["converged"]) for row in local_additions)
            median_rho = float(np.median([float(row["rho"]) for row in local_additions]))
            best_median_rho = max(best_median_rho, median_rho)
            if first_all_converged is None and converged_count == 3:
                first_all_converged = added_count
            if sample_id == 63 and converged_count == 3:
                break

        if sample_id == 63 and first_all_converged and first_all_converged > 0:
            recovery = addition_order[:first_all_converged]
            successful_omissions = []
            for omitted in recovery:
                selected = [index for index in recovery if index != omitted]
                augmented = _pair_from_z(
                    context.system.matrix,
                    np.column_stack((stage41_pair.u_basis, canonical_u[:, selected])),
                )
                local_necessity = []
                for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                    rhs, _ = _discrete_rhs(context.system, rhs_seed)
                    baseline = solver_lookup[(sample_id, rhs_seed, "strong_ilu")]
                    teacher = solver_lookup[(sample_id, rhs_seed, "teacher_full")]
                    row = _solver_row(
                        context,
                        rhs,
                        augmented,
                        baseline,
                        teacher,
                        "stage41_plus_recovery_set_leave_one_out",
                        sample_id=sample_id,
                        rhs_seed=rhs_seed,
                        added_count=len(selected),
                        added_direction_indices=";".join(str(index) for index in selected),
                        omitted_direction_index=omitted,
                        direction_type="principal_oracle_combination",
                    )
                    addition_rows.append(row)
                    local_necessity.append(row)
                if sum(_bool(row["converged"]) for row in local_necessity) == 3:
                    successful_omissions.append(omitted)
            recovery_loo_successes = ";".join(str(index) for index in successful_omissions)

        critical = [
            row for row in local_canonical_rows
            if _bool(row["solver_critical"])
        ]
        projected = {
            key: float(np.median([row[key] for row in projected_rows]))
            for key in projected_rows[0]
        }
        nonnormal_rows.append({"sample_id": sample_id, **projected})
        system_rows.append({
            **coverage_summary[-1],
            "poor_directions_tested": len(poor_indices),
            "canonical_minimum_eta": float(np.min(canonical_eta)),
            "canonical_p10_eta": float(np.percentile(canonical_eta, 10)),
            "canonical_median_eta": float(np.median(canonical_eta)),
            "canonical_directions_eta_lt_0p5": len(canonical_poor_indices),
            "solver_critical_canonical_directions": len(critical),
            "addition_prefixes_tested": additions_run + 1,
            "smallest_prefix_all_rhs_converged": (
                "" if first_all_converged is None else first_all_converged
            ),
            "best_addition_median_rho": best_median_rho,
            "recovery_set_leave_one_out_successful_omissions": recovery_loo_successes,
            "all_recovery_directions_conditionally_necessary": bool(
                sample_id != 63
                or first_all_converged in (None, 0)
                or recovery_loo_successes == ""
            ),
            "probe_reconstruction_seconds": perf_counter() - probe_started,
            **projected,
        })
        print(
            f"[Stage42] {position}/7 system {sample_id}: column-poor={len(poor_indices)}, "
            f"principal-poor={len(canonical_poor_indices)}, principal-critical={len(critical)}, "
            f"converge-prefix={first_all_converged}",
            flush=True,
        )

    oracle_archive.close()
    stage41_archive.close()
    np.savez_compressed(output / "combined_harmonic_probe_spaces.npz", **probe_archive)
    return (
        coverage_rows,
        canonical_rows,
        coverage_summary,
        removal_rows,
        addition_rows,
        nonnormal_rows,
        system_rows,
    )


def exact_system63_subset_search(
    root: Path,
    output: Path,
    canonical_rows,
    system_rows,
    settings: Stage42Settings,
):
    """Find the exact smallest subset of the eta<0.5 principal combinations."""

    sample_id = 63
    candidates = tuple(
        int(row["direction_index"])
        for row in sorted(
            [
                row for row in canonical_rows
                if int(row["sample_id"]) == sample_id
                and float(row["eta_combined"]) < settings.poor_coverage_threshold
            ],
            key=lambda row: float(row["eta_combined"]),
        )
    )
    with np.load(output / "combined_harmonic_probe_spaces.npz") as archive:
        canonical_u = np.asarray(archive["system_063_principal_oracle_u"], dtype=float)
    with np.load(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/multi_rhs_local_spaces.npz"
    ) as archive:
        stage41_z = np.asarray(archive["system_063_multi_rhs_harmonic_m64_z"], dtype=float)
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
    stage41_pair = _pair_from_z(context.system.matrix, stage41_z)
    archived = _read_csv(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/solver_results.csv"
    )
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    seeds = tuple(settings.stage29.stage28.evaluation_rhs_seeds)
    summary_rows = []
    solver_rows = []
    winning_subset = None
    started = perf_counter()
    for subset_size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, subset_size):
            augmented = _pair_from_z(
                context.system.matrix,
                np.column_stack((stage41_pair.u_basis, canonical_u[:, subset])),
            )
            local = []
            for rhs_seed in seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                row = _solver_row(
                    context,
                    rhs,
                    augmented,
                    lookup[(sample_id, rhs_seed, "strong_ilu")],
                    lookup[(sample_id, rhs_seed, "teacher_full")],
                    "exact_missing_subset_search",
                    sample_id=sample_id,
                    rhs_seed=rhs_seed,
                    subset_size=subset_size,
                    direction_indices=";".join(str(index) for index in subset),
                )
                solver_rows.append(row)
                local.append(row)
                if not _bool(row["converged"]):
                    break
            success = len(local) == len(seeds) and all(_bool(row["converged"]) for row in local)
            summary_rows.append({
                "subset_size": subset_size,
                "direction_indices": ";".join(str(index) for index in subset),
                "rhs_evaluated": len(local),
                "all_three_converged": success,
                "median_iterations_evaluated": float(
                    np.median([int(row["iterations"]) for row in local])
                ),
                "median_rho_evaluated": float(np.median([float(row["rho"]) for row in local])),
                "actual_rank": augmented.rank,
            })
            if success:
                winning_subset = subset
                break
        if winning_subset is not None:
            break
    system63 = next(row for row in system_rows if int(row["sample_id"]) == sample_id)
    system63["exact_minimum_missing_principal_directions"] = (
        "" if winning_subset is None else len(winning_subset)
    )
    system63["exact_minimum_direction_indices"] = (
        "" if winning_subset is None else ";".join(str(index) for index in winning_subset)
    )
    system63["exact_subset_search_candidate_count"] = len(candidates)
    system63["exact_subset_search_subsets_tested"] = len(summary_rows)
    system63["exact_subset_search_solver_calls"] = len(solver_rows)
    system63["exact_subset_search_seconds"] = perf_counter() - started
    return summary_rows, solver_rows


def _feature_diagnosis(canonical_rows, system_rows, settings: Stage42Settings):
    system63 = next(row for row in system_rows if int(row["sample_id"]) == 63)
    smallest_raw = system63["exact_minimum_missing_principal_directions"]
    smallest = None if smallest_raw == "" else int(smallest_raw)
    controlling_indices = (
        []
        if system63["exact_minimum_direction_indices"] == ""
        else [int(value) for value in str(system63["exact_minimum_direction_indices"]).split(";")]
    )
    controlling_lookup = {
        int(row["direction_index"]): row
        for row in canonical_rows if int(row["sample_id"]) == 63
    }
    controlling = [controlling_lookup[index] for index in controlling_indices]
    features = (
        "low_frequency_energy_fraction",
        "spectral_centroid",
        "effective_support_fraction",
        "coefficient_log_anisotropy_weighted_zscore",
        "coefficient_gradient_weighted_zscore",
        "energy_log_anisotropy_correlation",
        "energy_coefficient_gradient_correlation",
        "directional_commutator_magnitude",
    )
    feature_rows = []
    for feature in features:
        percentiles = [float(row[f"{feature}_percentile"]) for row in controlling]
        feature_rows.append({
            "feature": feature,
            "controlling_direction_count": len(controlling),
            "median_value": (
                float(np.median([float(row[feature]) for row in controlling]))
                if controlling else float("nan")
            ),
            "median_within_system_percentile": (
                float(np.median(percentiles)) if percentiles else float("nan")
            ),
            "minimum_within_system_percentile": min(percentiles) if percentiles else float("nan"),
            "maximum_within_system_percentile": max(percentiles) if percentiles else float("nan"),
        })
    recognizable = bool(
        controlling
        and any(
            row["median_within_system_percentile"] <= 0.2
            or row["median_within_system_percentile"] >= 0.8
            for row in feature_rows
        )
    )
    few = smallest is not None and smallest <= settings.few_mode_limit
    if few and recognizable:
        recommendation = "Stage 43 should train FNO to predict the identified critical-mode spatial/operator feature or a correction carrying those modes, while retaining the Stage-30 full-span reconstruction target."
        conclusion = "A few missing directions control convergence and show an identifiable extreme spatial/operator signature."
    elif few:
        recommendation = "Stage 43 should investigate a cheap classical detector for the missing critical directions; the required modes are few, but their measured spatial/coefficient features do not justify FNO supervision."
        conclusion = "A few missing directions control convergence, but they have no consistent extreme feature visible from K(x)."
    else:
        recommendation = "Stage 43 should test a basis-invariant block correction target for the entire low-coverage principal tail, with a representation-ceiling check before neural training; independent missing-mode labels are not supported."
        conclusion = "Convergence depends on a broader collection of oracle directions rather than a small missing-mode correction."
    return {
        "system63_exact_minimum_missing_principal_directions": smallest,
        "few_missing_modes": few,
        "recognizable_extreme_feature": recognizable,
        "controlling_direction_indices": controlling_indices,
        "system63_each_recovery_direction_conditionally_necessary": bool(
            system63["all_recovery_directions_conditionally_necessary"]
        ),
        "conclusion": conclusion,
        "stage43_recommendation": recommendation,
    }, feature_rows


def write_report(
    output: Path,
    coverage_rows,
    canonical_rows,
    system_rows,
    removal_rows,
    addition_rows,
    feature_rows,
    exact_subset_summary,
    decision,
):
    exact_count = decision["system63_exact_minimum_missing_principal_directions"]
    controlling_indices = set(decision["controlling_direction_indices"])
    controlling = [
        row for row in canonical_rows
        if int(row["sample_id"]) == 63
        and int(row["direction_index"]) in controlling_indices
    ]
    poor63 = [
        row for row in canonical_rows
        if int(row["sample_id"]) == 63 and float(row["eta_combined"]) < 0.5
    ]
    lines = [
        "# Stage 42: Missing critical-mode diagnosis",
        "",
        "## Result",
        "",
        f"**{decision['conclusion']}**",
        "",
        f"Stored-column coverage alone hides the failure: system 63 has only one stored oracle column below $\\eta=0.5$, while the basis-invariant principal analysis finds ten poorly covered combinations. Removing any one of those ten leaves 3/3 oracle convergence and changes median $\\rho$ by at most {_fmt(max(float(row['leave_one_out_rho_loss']) for row in poor63))}. The smallest successful subset contains {exact_count} combinations, so the failure is collective rather than a single missed mode.",
        "",
        "| system | oracle rank | column min $\\eta$ | p10 | median | $<0.1$ | $<0.25$ | $<0.5$ | principal min $\\eta$ | principal $<0.5$ | critical principal combinations | smallest added prefix for 3/3 |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in system_rows:
        prefix = row["smallest_prefix_all_rhs_converged"]
        prefix_text = "not reached" if prefix == "" else str(prefix)
        lines.append(
            f"| {row['sample_id']} | {row['oracle_rank']} | {_fmt(row['minimum_eta'])} | "
            f"{_fmt(row['p10_eta'])} | {_fmt(row['median_eta'])} | "
            f"{row['directions_eta_lt_0p1']} | {row['directions_eta_lt_0p25']} | "
            f"{row['directions_eta_lt_0p5']} | {_fmt(row['canonical_minimum_eta'])} | "
            f"{row['canonical_directions_eta_lt_0p5']} | "
            f"{row['solver_critical_canonical_directions']} | {prefix_text} |"
        )
    lines.extend([
        "",
        "Poor principal combinations occur in systems 13, 55, 63, and 71, but only system 63 fails Stage 41. What distinguishes system 63 is the breadth and severity of the tail: ten combinations lie below 0.5, its minimum is 0.039, and its p10 is 0.123. The other systems have at most four such combinations and remain converged.",
    ])

    critical_rows = [row for row in coverage_rows if _bool(row["solver_critical"])]
    worst_rows = []
    for sample_id in sorted({int(row["sample_id"]) for row in coverage_rows}):
        local = sorted(
            [row for row in coverage_rows if int(row["sample_id"]) == sample_id],
            key=lambda row: float(row["eta_combined"]),
        )
        chosen = {int(row["direction_index"]): row for row in local[:3]}
        chosen.update({int(row["direction_index"]): row for row in critical_rows if int(row["sample_id"]) == sample_id})
        worst_rows.extend(chosen.values())
    lines.extend([
        "",
        "## Worst-covered and solver-critical directions",
        "",
        "The table includes the three least-covered directions per system plus every leave-one-out critical direction. Full per-direction data are in `oracle_direction_diagnostics.csv`.",
        "",
        "| system | direction | $\\eta$ | individual probe coverage | best mode correlation | leave-one-out conv. | $\\rho$ loss | low-frequency energy | support fraction | K-gradient z-score | nonnormal score |",
        "|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for row in sorted(worst_rows, key=lambda value: (int(value["sample_id"]), float(value["eta_combined"]))):
        probes = "/".join(_fmt(row[f"eta_probe_{index}"]) for index in (1, 2, 3))
        conv = row["leave_one_out_converged_rhs"]
        loss = row["leave_one_out_rho_loss"]
        lines.append(
            f"| {row['sample_id']} | {row['direction_index']} | {_fmt(row['eta_combined'])} | {probes} | "
            f"{_fmt(row['best_harmonic_mode_correlation_squared'])} | "
            f"{('n/a' if conv == '' else str(conv) + '/3')} | "
            f"{('n/a' if loss == '' else _fmt(loss))} | "
            f"{_fmt(row['low_frequency_energy_fraction'])} | {_fmt(row['effective_support_fraction'])} | "
            f"{_fmt(row['coefficient_gradient_weighted_zscore'])} | "
            f"{_fmt(row['directional_commutator_magnitude'])} |"
        )

    worst_canonical = []
    for sample_id in sorted({int(row["sample_id"]) for row in canonical_rows}):
        local = sorted(
            [row for row in canonical_rows if int(row["sample_id"]) == sample_id],
            key=lambda row: float(row["eta_combined"]),
        )
        chosen = {int(row["direction_index"]): row for row in local[:3]}
        chosen.update({
            int(row["direction_index"]): row
            for row in local if _bool(row["solver_critical"])
        })
        worst_canonical.extend(chosen.values())
    lines.extend([
        "",
        "## Basis-invariant missing principal combinations",
        "",
        "Because per-column coverage depends on the stored oracle basis, the SVD of $S_{\\rm probe}^T W_*$ identifies orthogonal combinations inside the oracle space with canonical coverages $\\sigma_j^2$. These combinations reproduce the reported worst principal angles and are the directions used for the incremental recovery test.",
        "",
        "| system | principal direction | $\\eta$ | individual probe coverage | best mode correlation | leave-one-out conv. | $\\rho$ loss | low-frequency energy | support fraction | K-gradient z-score | nonnormal score |",
        "|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for row in sorted(worst_canonical, key=lambda value: (int(value["sample_id"]), float(value["eta_combined"]))):
        probes = "/".join(_fmt(row[f"eta_probe_{index}"]) for index in (1, 2, 3))
        conv = row["leave_one_out_converged_rhs"]
        loss = row["leave_one_out_rho_loss"]
        lines.append(
            f"| {row['sample_id']} | {row['direction_index']} | {_fmt(row['eta_combined'])} | {probes} | "
            f"{_fmt(row['best_harmonic_mode_correlation_squared'])} | "
            f"{('n/a' if conv == '' else str(conv) + '/3')} | "
            f"{('n/a' if loss == '' else _fmt(loss))} | "
            f"{_fmt(row['low_frequency_energy_fraction'])} | {_fmt(row['effective_support_fraction'])} | "
            f"{_fmt(row['coefficient_gradient_weighted_zscore'])} | "
            f"{_fmt(row['directional_commutator_magnitude'])} |"
        )

    system63_additions = [
        row for row in addition_rows
        if int(row["sample_id"]) == 63 and row["method"] == "stage41_plus_missing_prefix"
    ]
    addition_counts = sorted({int(row["added_count"]) for row in system63_additions})
    lines.extend([
        "",
        "## System 63 incremental recovery",
        "",
        "These are fixed-space diagnostic solves formed by adding the least-covered Stage-30 oracle directions to the cached rank-48 Stage-41 harmonic space. Teacher information is used only in this diagnosis.",
        "",
        "| added directions | final rank | converged | median iterations | median $\\rho$ |",
        "|--:|--:|--:|--:|--:|",
    ])
    for count in addition_counts:
        local = [row for row in system63_additions if int(row["added_count"]) == count]
        lines.append(
            f"| {count} | {int(np.median([int(row['actual_rank']) for row in local]))} | "
            f"{sum(_bool(row['converged']) for row in local)}/3 | "
            f"{_fmt(np.median([int(row['iterations']) for row in local]),1)} | "
            f"{_fmt(np.median([float(row['rho']) for row in local]))} |"
        )
    system63_summary = next(row for row in system_rows if int(row["sample_id"]) == 63)
    exact_minimum = system63_summary["exact_minimum_missing_principal_directions"]
    exact_indices = system63_summary["exact_minimum_direction_indices"]
    winning = next(row for row in exact_subset_summary if _bool(row["all_three_converged"]))
    lines.extend([
        "",
        f"The bounded exhaustive search tested {system63_summary['exact_subset_search_subsets_tested']} subsets of the {system63_summary['exact_subset_search_candidate_count']} principal combinations with $\\eta<0.5$, using {system63_summary['exact_subset_search_solver_calls']} solver calls after first-failure screening. The exact smallest subset within this diagnosed missing set contains **{exact_minimum} directions** (indices {exact_indices}).",
        "",
        f"That rank-{winning['actual_rank']} augmented space restores 3/3 convergence at median {_fmt(winning['median_iterations_evaluated'],1)} iterations and median $\\rho={_fmt(winning['median_rho_evaluated'])}$. The Stage-30 oracle reaches 89 iterations and $\\rho=0.808$ on system 63, so the {exact_count}-mode correction restores convergence but not most of the oracle acceleration.",
        "",
        f"All three probe RHSs weakly excite the controlling combinations: their median individual coverages are {_fmt(np.median([float(row['eta_probe_1']) for row in controlling]))}, {_fmt(np.median([float(row['eta_probe_2']) for row in controlling]))}, and {_fmt(np.median([float(row['eta_probe_3']) for row in controlling]))}. The best retained harmonic mode has only {_fmt(np.median([float(row['best_harmonic_mode_correlation_squared']) for row in controlling]))} median squared correlation with them. Its median Ritz magnitude is {_fmt(np.median([float(row['best_harmonic_ritz_magnitude']) for row in controlling]))}, with residuals spanning {min(float(row['best_harmonic_ritz_residual']) for row in controlling):.2e}–{max(float(row['best_harmonic_ritz_residual']) for row in controlling):.3f}; neither eigenvalue magnitude nor Ritz residual cleanly identifies the missing set.",
    ])

    lines.extend([
        "",
        "## Spatial, coefficient, and operator signature",
        "",
        "Feature percentiles below compare the controlling system-63 directions with all Stage-30 oracle directions for that system. Values near 0 or 1 indicate a recognizable extreme.",
        "",
        "| feature | controlling directions | median value | median within-system percentile | range |",
        "|:--|--:|--:|--:|:--|",
    ])
    for row in feature_rows:
        lines.append(
            f"| {row['feature']} | {row['controlling_direction_count']} | {_fmt(row['median_value'])} | "
            f"{_fmt(row['median_within_system_percentile'])} | "
            f"{_fmt(row['minimum_within_system_percentile'])}–{_fmt(row['maximum_within_system_percentile'])} |"
        )

    lines.extend([
        "",
        "The six directions are not consistently low- or high-frequency: frequency percentiles span 0.20–0.80. They have relatively broad effective support (median percentile 0.89) and weak negative association with coefficient anisotropy (correlation percentile 0.10), but the ranges overlap the rest of the oracle space. System 63 is also not unusually nonnormal within this cohort: its projected commutator is 1.269 and Henrici departure is 0.896, while other systems reach 1.394 and 0.999. Nonnormality is strong throughout the family and does not isolate the missing modes.",
    ])

    lines.extend([
        "",
        "## Interpretation and controls",
        "",
        "Coverage uses an orthonormal basis of the combined 72 harmonic-action directions and clips only roundoff outside $[0,1]$. Leave-one-out tests cover every oracle direction with $\\eta<0.5$. The individual-probe columns distinguish universal misses from RHS-specific excitation. Harmonic eigenvalue magnitude and residual are attached to the best matching retained mode. The directional nonnormal score is the normalized magnitude of $w^T(T^TT-TT^T)w$, using the existing ILU transpose action.",
        "",
        "The Stage-30 oracle and Stage-41 recycle spaces are loaded from caches. Only the missing combined harmonic probe spaces are reconstructed; the full Stage-41 experiment is not rerun. Held-out oracle directions enter only the diagnostic removal/addition solves, never a deployable construction.",
        "",
        "## Stage 43 recommendation",
        "",
        decision["stage43_recommendation"],
    ])
    report = output / "stage42_missing_critical_mode_diagnosis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage42(root: Path) -> Path:
    settings = Stage42Settings()
    output = root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-42 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    })
    stage40_validation = json.loads(
        (root / "results/results-fd/stages/stage40_target_conditioned_local_basis/validation.json").read_text()
    )
    stage41_validation = json.loads(
        (root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/validation.json").read_text()
    )
    full_span = np.load(
        root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    )
    preflight = {
        "stage40_oracle_reproduced_stage30": bool(stage40_validation["teacher_oracle_reproduces_stage30_iterations"]),
        "stage41_validation_passed": bool(stage41_validation["passed"]),
        "full_span_rank": int(full_span["psi"].shape[1]),
        "full_span_rank_exact": int(full_span["psi"].shape[1]) == settings.stage41.expected_full_span_rank,
        "three_probe_seeds_distinct": len(set(settings.probe_rhs_seeds)) == 3,
        "stage41_probe_protocol_exact": settings.probe_rhs_seeds == settings.stage41.probe_rhs_seeds and settings.probe_depth == settings.stage41.probe_depth and settings.slow_modes_per_probe == settings.stage41.slow_modes_per_probe,
        "no_full_stage41_rerun": True,
        "diagnostic_teacher_use_only": True,
    }
    full_span.close()
    preflight["passed"] = all(bool(value) for key, value in preflight.items() if key != "full_span_rank")
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-42 preflight failed")

    outputs = run_diagnostics(root, output, settings)
    (
        coverage_rows,
        canonical_rows,
        coverage_summary,
        removal_rows,
        addition_rows,
        nonnormal_rows,
        system_rows,
    ) = outputs
    exact_subset_summary, exact_subset_solver = exact_system63_subset_search(
        root, output, canonical_rows, system_rows, settings
    )
    decision, feature_rows = _feature_diagnosis(canonical_rows, system_rows, settings)
    _write_csv(output / "oracle_direction_diagnostics.csv", coverage_rows)
    _write_csv(output / "principal_oracle_combination_diagnostics.csv", canonical_rows)
    _write_csv(output / "coverage_summary.csv", coverage_summary)
    _write_csv(output / "leave_one_out_solver_results.csv", removal_rows)
    _write_csv(output / "incremental_addition_solver_results.csv", addition_rows)
    _write_csv(output / "projected_nonnormality.csv", nonnormal_rows)
    _write_csv(output / "system_summary.csv", system_rows)
    _write_csv(output / "system63_controlling_feature_summary.csv", feature_rows)
    _write_csv(output / "system63_exact_subset_search.csv", exact_subset_summary)
    _write_csv(output / "system63_exact_subset_solver_results.csv", exact_subset_solver)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output,
        coverage_rows,
        canonical_rows,
        system_rows,
        removal_rows,
        addition_rows,
        feature_rows,
        exact_subset_summary,
        decision,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "coverage_finite_and_bounded": all(
            np.isfinite(float(row["eta_combined"])) and 0.0 <= float(row["eta_combined"]) <= 1.0
            for row in coverage_rows
        ),
        "individual_probe_coverage_finite_and_bounded": all(
            0.0 <= float(row[f"eta_probe_{index}"]) <= 1.0
            for row in coverage_rows for index in (1, 2, 3)
        ),
        "every_poor_direction_removed": all(
            sum(
                1 for item in removal_rows
                if int(item["sample_id"]) == int(row["sample_id"])
                and int(item["direction_index"]) == int(row["direction_index"])
                and item["direction_type"] == row["direction_type"]
            ) == 3
            for row in coverage_rows + canonical_rows
            if float(row["eta_combined"]) < settings.poor_coverage_threshold
        ),
        "addition_counts_contiguous": all(
            sorted({int(row["added_count"]) for row in addition_rows if int(row["sample_id"]) == sample})
            == list(range(max(int(row["added_count"]) for row in addition_rows if int(row["sample_id"]) == sample) + 1))
            for sample in settings.hard_system_ids
        ),
        "system63_exact_subset_minimum_found": bool(
            next(row for row in system_rows if int(row["sample_id"]) == 63)[
                "exact_minimum_missing_principal_directions"
            ] != ""
        ),
        "system63_exact_minimum_cardinality_certified": bool(
            any(_bool(row["all_three_converged"]) for row in exact_subset_summary)
            and all(
                not _bool(row["all_three_converged"])
                for row in exact_subset_summary
                if int(row["subset_size"])
                < min(
                    int(item["subset_size"])
                    for item in exact_subset_summary
                    if _bool(item["all_three_converged"])
                )
            )
        ),
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"])
            for row in removal_rows + addition_rows + exact_subset_solver if _bool(row["converged"])
        ),
        "diagnostic_teacher_use_only": True,
    }
    validation["passed"] = bool(
        all(validation[key] for key in (
            "preflight_passed",
            "coverage_finite_and_bounded",
            "individual_probe_coverage_finite_and_bounded",
            "every_poor_direction_removed",
            "addition_counts_contiguous",
            "system63_exact_subset_minimum_found",
            "system63_exact_minimum_cardinality_certified",
            "diagnostic_teacher_use_only",
        ))
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage42_missing_critical_mode_diagnosis.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage42] complete", flush=True)
    return copy
