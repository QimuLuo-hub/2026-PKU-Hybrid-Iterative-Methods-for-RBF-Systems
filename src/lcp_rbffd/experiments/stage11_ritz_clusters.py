"""Stage 11: Ritz-cluster stability in the rotated-anisotropy transition.

No learned model is trained.  The Stage 10 discretization, node ordering,
stencil, anisotropy angle, strong ILU, solver tolerance, and held-out right-hand
sides are reused exactly.  The new object of study is a conjugacy-preserving,
spectral-gap-defined real Ritz invariant block rather than an individual Ritz
vector or an arbitrarily truncated prefix.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment

from ..coarse_space import random_orthonormal_basis
from ..preconditioners import InverseAction
from ..subspace_analysis import (
    generalized_subspace_metrics,
    interpolate_basis_to_probe,
    ritz_atomic_blocks,
)
from .stage9_scalability import _read_csv, _write_csv, _write_json
from .stage10_ritz_transfer import (
    BLUE,
    CHARCOAL,
    LIGHT_GRAY,
    OLIVE,
    ORANGE,
    PINK,
    PURPLE,
    RitzContext,
    Stage10Settings,
    _build_context,
    _coarse_action,
    _run_solver,
)


@dataclass(frozen=True)
class Stage11Settings:
    """Numerical controls and predeclared stability/relevance gates."""

    stage10: Stage10Settings = Stage10Settings()
    ratios: Tuple[float, ...] = (
        80.0,
        82.0,
        84.0,
        85.0,
        86.0,
        88.0,
        90.0,
        95.0,
        100.0,
        105.0,
        110.0,
        120.0,
        130.0,
        140.0,
    )
    gap_mad_multiplier: float = 1.5
    gap_sensitivity_multiplier: float = 2.0
    fixed_prefix_dimension: int = 8
    maximum_low_dimension: int = 8
    maximum_expanded_dimension: int = 16
    stable_projector_distance: float = 0.35
    stable_max_principal_angle_degrees: float = 35.0
    stable_spectral_factor: float = 1.5
    critical_iteration_reduction: float = 0.20
    critical_removal_penalty: float = 5.0
    stable_pair_fraction: float = 0.80
    critical_occurrence_fraction: float = 0.60
    minimum_track_occurrences: int = 3
    robustness_ratios: Tuple[float, ...] = (85.0, 90.0, 100.0, 120.0)
    construction_rhs_replicates: Tuple[int, ...] = (20261001, 20261011, 20261021)
    node_seeds: Tuple[int, ...] = (7, 17, 27)
    importance_rhs_seeds: Tuple[int, ...] = (20261101, 20261102)


@dataclass(frozen=True)
class RitzAtom:
    """One real Ritz direction or one real plane from a complex Ritz pair."""

    atom_id: int
    start: int
    stop: int
    magnitude: float
    eigenvalue_real: float
    eigenvalue_imag: float

    @property
    def dimension(self) -> int:
        return self.stop - self.start


@dataclass
class RitzCluster:
    """A consecutive collection of Ritz atoms separated by external gaps."""

    ratio: float
    cluster_id: int
    atom_start: int
    atom_stop: int
    start: int
    stop: int
    magnitude_min: float
    magnitude_max: float
    magnitude_geometric_mean: float
    gap_before_log10: float
    gap_after_log10: float
    detection_threshold_log10: float
    basis: np.ndarray

    @property
    def dimension(self) -> int:
        return self.stop - self.start


@dataclass
class ClusteredContext:
    """Stage 10 numerical context plus its Stage 11 atoms and clusters."""

    context: RitzContext
    atoms: Tuple[RitzAtom, ...]
    clusters: Tuple[RitzCluster, ...]
    log_gaps: np.ndarray
    gap_threshold: float


def _orthonormal_columns(basis: np.ndarray) -> np.ndarray:
    orthonormal, triangular = la.qr(np.asarray(basis, dtype=float), mode="economic")
    if np.min(np.abs(np.diag(triangular))) <= 1.0e-12:
        raise RuntimeError("candidate invariant block is rank deficient")
    return orthonormal


def _stage10_results_directory(output: Path) -> Path:
    """Resolve the immutable Stage 10 evidence used by this continuation."""

    sibling = output.parent / "stage10_ritz_transfer_and_mode_tracking"
    if sibling.exists():
        return sibling
    repository = Path(__file__).resolve().parents[3]
    fallback = repository / "results" / "stage10_ritz_transfer_and_mode_tracking"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Stage 10 results are required before Stage 11; checked "
        f"{sibling} and {fallback}"
    )


def detect_ritz_clusters(
    ratio: float,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    gap_mad_multiplier: float,
) -> Tuple[Tuple[RitzAtom, ...], Tuple[RitzCluster, ...], np.ndarray, float]:
    """Detect variable-rank clusters from robust outliers in log spectral gaps.

    Complex Ritz planes are first made indivisible.  If ``rho_k`` denotes the
    magnitude of atomic Ritz value ``k``, clustering uses
    ``g_k=log10(rho_{k+1}/rho_k)`` and places a boundary where ``g_k`` is at
    least ``median(g)+c*1.4826*MAD(g)``.  No cluster rank is prescribed.
    """

    spans = ritz_atomic_blocks(eigenvalues)
    atoms: List[RitzAtom] = []
    for atom_id, (start, stop) in enumerate(spans):
        value = complex(eigenvalues[start])
        atoms.append(
            RitzAtom(
                atom_id,
                start,
                stop,
                float(abs(value)),
                float(value.real),
                float(value.imag),
            )
        )
    magnitudes = np.asarray([atom.magnitude for atom in atoms], dtype=float)
    epsilon = np.finfo(float).eps
    log_gaps = np.log10(
        (magnitudes[1:] + epsilon) / (magnitudes[:-1] + epsilon)
    )
    median = float(np.median(log_gaps)) if log_gaps.size else 0.0
    scaled_mad = (
        float(1.4826 * np.median(np.abs(log_gaps - median)))
        if log_gaps.size
        else 0.0
    )
    threshold = median + gap_mad_multiplier * scaled_mad
    boundaries = np.flatnonzero(log_gaps >= threshold) + 1
    atomic_groups = np.split(np.arange(len(atoms)), boundaries)
    clusters: List[RitzCluster] = []
    for cluster_id, group in enumerate(atomic_groups):
        first = atoms[int(group[0])]
        last = atoms[int(group[-1])]
        group_magnitudes = magnitudes[group]
        before = float(log_gaps[int(group[0]) - 1]) if group[0] > 0 else float("inf")
        after = (
            float(log_gaps[int(group[-1])])
            if int(group[-1]) < len(atoms) - 1
            else float("inf")
        )
        clusters.append(
            RitzCluster(
                ratio=ratio,
                cluster_id=cluster_id,
                atom_start=int(group[0]),
                atom_stop=int(group[-1]) + 1,
                start=first.start,
                stop=last.stop,
                magnitude_min=float(np.min(group_magnitudes)),
                magnitude_max=float(np.max(group_magnitudes)),
                magnitude_geometric_mean=float(
                    np.exp(np.mean(np.log(group_magnitudes + epsilon)))
                ),
                gap_before_log10=before,
                gap_after_log10=after,
                detection_threshold_log10=threshold,
                basis=_orthonormal_columns(basis[:, first.start : last.stop]),
            )
        )
    return tuple(atoms), tuple(clusters), log_gaps, threshold


def build_clustered_contexts(
    output: Path,
    settings: Stage11Settings,
) -> Tuple[Dict[float, ClusteredContext], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild the deterministic Stage 10 contexts and cluster their spectra."""

    contexts: Dict[float, ClusteredContext] = {}
    atom_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    subspace_dir = output / "clusters"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    for ratio in settings.ratios:
        context = _build_context(ratio, settings.stage10)
        atoms, clusters, gaps, threshold = detect_ritz_clusters(
            ratio,
            context.modes.basis,
            context.modes.eigenvalues,
            settings.gap_mad_multiplier,
        )
        contexts[ratio] = ClusteredContext(context, atoms, clusters, gaps, threshold)
        for atom in atoms:
            atom_rows.append(
                {
                    "ratio": ratio,
                    "atom_id": atom.atom_id,
                    "start": atom.start,
                    "stop": atom.stop,
                    "dimension": atom.dimension,
                    "ritz_magnitude": atom.magnitude,
                    "ritz_value_real": atom.eigenvalue_real,
                    "ritz_value_imag": atom.eigenvalue_imag,
                    "is_complex_plane": atom.dimension == 2,
                    "crosses_q8_cut": atom.start < settings.fixed_prefix_dimension < atom.stop,
                }
            )
        for cluster in clusters:
            cluster_rows.append(
                {
                    "ratio": ratio,
                    "cluster_id": cluster.cluster_id,
                    "atom_start": cluster.atom_start,
                    "atom_stop": cluster.atom_stop,
                    "start": cluster.start,
                    "stop": cluster.stop,
                    "dimension": cluster.dimension,
                    "magnitude_min": cluster.magnitude_min,
                    "magnitude_max": cluster.magnitude_max,
                    "magnitude_geometric_mean": cluster.magnitude_geometric_mean,
                    "gap_before_log10": cluster.gap_before_log10,
                    "gap_after_log10": cluster.gap_after_log10,
                    "detection_threshold_log10": threshold,
                    "contains_q8_cut": cluster.start < settings.fixed_prefix_dimension <= cluster.stop,
                }
            )
        np.savez_compressed(
            subspace_dir / f"ritz_clusters_R{ratio:g}.npz",
            basis=context.modes.basis,
            eigenvalues=context.modes.eigenvalues,
            atom_spans=np.asarray([(atom.start, atom.stop) for atom in atoms]),
            cluster_spans=np.asarray(
                [(cluster.start, cluster.stop) for cluster in clusters]
            ),
            log_gaps=gaps,
            gap_threshold=threshold,
        )
    _write_csv(output / "ritz_atoms.csv", atom_rows)
    _write_csv(output / "ritz_clusters.csv", cluster_rows)
    return contexts, atom_rows, cluster_rows


def _cluster_pair_metrics(
    first: RitzCluster,
    second: RitzCluster,
) -> Dict[str, Any]:
    metrics = generalized_subspace_metrics(first.basis, second.basis)
    spectral_log_distance = float(
        abs(
            np.log10(
                (first.magnitude_geometric_mean + np.finfo(float).eps)
                / (second.magnitude_geometric_mean + np.finfo(float).eps)
            )
        )
    )
    dimension_penalty = abs(first.dimension - second.dimension) / (
        first.dimension + second.dimension
    )
    return {
        **{key: value for key, value in metrics.items() if key != "principal_angles_degrees"},
        "spectral_log10_distance": spectral_log_distance,
        "spectral_factor": float(10.0**spectral_log_distance),
        "dimension_penalty": float(dimension_penalty),
        "matching_cost": float(
            metrics["projector_distance_normalized"]
            + 0.25 * min(spectral_log_distance, 2.0)
            + 0.25 * dimension_penalty
        ),
    }


def match_cluster_tracks(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    settings: Stage11Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Track variable-count clusters through neighboring anisotropy ratios."""

    ratios = list(contexts)
    next_track_id = len(contexts[ratios[0]].clusters)
    current_tracks = {
        cluster.cluster_id: cluster.cluster_id
        for cluster in contexts[ratios[0]].clusters
    }
    assignments: List[Dict[str, Any]] = [
        {"ratio": ratios[0], "cluster_id": local, "track_id": track}
        for local, track in current_tracks.items()
    ]
    pair_rows: List[Dict[str, Any]] = []
    for source_ratio, target_ratio in zip(ratios[:-1], ratios[1:]):
        source_clusters = contexts[source_ratio].clusters
        target_clusters = contexts[target_ratio].clusters
        cost = np.zeros((len(source_clusters), len(target_clusters)), dtype=float)
        metric_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for source_index, source in enumerate(source_clusters):
            for target_index, target in enumerate(target_clusters):
                metrics = _cluster_pair_metrics(source, target)
                metric_cache[(source_index, target_index)] = metrics
                cost[source_index, target_index] = metrics["matching_cost"]
        source_indices, target_indices = linear_sum_assignment(cost)
        matched_targets = set()
        next_tracks: Dict[int, int] = {}
        for source_index, target_index in zip(source_indices, target_indices):
            source = source_clusters[int(source_index)]
            target = target_clusters[int(target_index)]
            metrics = metric_cache[(int(source_index), int(target_index))]
            track_id = current_tracks[source.cluster_id]
            matched_targets.add(target.cluster_id)
            next_tracks[target.cluster_id] = track_id
            stable = bool(
                source.dimension == target.dimension
                and metrics["projector_distance_normalized"]
                <= settings.stable_projector_distance
                and metrics["principal_angle_max_degrees"]
                <= settings.stable_max_principal_angle_degrees
                and metrics["spectral_factor"] <= settings.stable_spectral_factor
            )
            pair_rows.append(
                {
                    "track_id": track_id,
                    "source_ratio": source_ratio,
                    "target_ratio": target_ratio,
                    "source_cluster_id": source.cluster_id,
                    "target_cluster_id": target.cluster_id,
                    "source_start": source.start,
                    "source_stop": source.stop,
                    "target_start": target.start,
                    "target_stop": target.stop,
                    "source_dimension": source.dimension,
                    "target_dimension": target.dimension,
                    **metrics,
                    "stable_pair": stable,
                }
            )
        for target in target_clusters:
            if target.cluster_id not in matched_targets:
                next_tracks[target.cluster_id] = next_track_id
                next_track_id += 1
        current_tracks = next_tracks
        assignments.extend(
            {
                "ratio": target_ratio,
                "cluster_id": cluster_id,
                "track_id": track_id,
            }
            for cluster_id, track_id in current_tracks.items()
        )

    track_rows: List[Dict[str, Any]] = []
    for track_id in sorted({int(row["track_id"]) for row in assignments}):
        occurrences = [row for row in assignments if int(row["track_id"]) == track_id]
        pairs = [row for row in pair_rows if int(row["track_id"]) == track_id]
        dimensions = []
        for occurrence in occurrences:
            cluster = contexts[float(occurrence["ratio"])].clusters[
                int(occurrence["cluster_id"])
            ]
            dimensions.append(cluster.dimension)
        stable_fraction = (
            float(np.mean([bool(row["stable_pair"]) for row in pairs]))
            if pairs
            else 0.0
        )
        track_rows.append(
            {
                "track_id": track_id,
                "occurrences": len(occurrences),
                "adjacent_matches": len(pairs),
                "dimension_min": min(dimensions),
                "dimension_max": max(dimensions),
                "dimension_constant": len(set(dimensions)) == 1,
                "stable_pair_fraction": stable_fraction,
                "projector_distance_median": (
                    float(
                        np.median(
                            [row["projector_distance_normalized"] for row in pairs]
                        )
                    )
                    if pairs
                    else float("nan")
                ),
                "principal_angle_max_p90": (
                    float(
                        np.quantile(
                            [row["principal_angle_max_degrees"] for row in pairs],
                            0.9,
                        )
                    )
                    if pairs
                    else float("nan")
                ),
                "stable_track": bool(
                    len(occurrences) >= settings.minimum_track_occurrences
                    and len(set(dimensions)) == 1
                    and stable_fraction >= settings.stable_pair_fraction
                ),
            }
        )
    _write_csv(output / "cluster_track_assignments.csv", assignments)
    _write_csv(output / "cluster_neighbor_matches.csv", pair_rows)
    _write_csv(output / "cluster_track_summary.csv", track_rows)
    return assignments, pair_rows, track_rows


def _summarize_results(results: Sequence[Any]) -> Dict[str, Any]:
    return {
        "iterations_median": float(np.median([result.iterations for result in results])),
        "iterations_max": int(max(result.iterations for result in results)),
        "convergence_rate": float(np.mean([result.converged for result in results])),
        "solve_seconds_median": float(
            np.median([result.solve_seconds for result in results])
        ),
        "final_relative_residual_max": float(
            max(result.relative_residuals[-1] for result in results)
        ),
        "failures": "; ".join(
            sorted({result.failure for result in results if result.failure})
        ),
    }


def run_cluster_solver_importance(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    assignments: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Add and remove every gap-defined cluster on held-out solver RHSs."""

    stage10_dir = _stage10_results_directory(output)
    stage10_solves = _read_csv(stage10_dir / "q_sweep_results.csv")
    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    for ratio, clustered in contexts.items():
        context = clustered.context
        baseline_source = [
            row
            for row in stage10_solves
            if float(row["ratio"]) == ratio
            and int(row["q"]) == 0
            and int(row["rhs_seed"]) in settings.importance_rhs_seeds
        ]
        full_source = [
            row
            for row in stage10_solves
            if float(row["ratio"]) == ratio
            and int(row["q"]) == 32
            and int(row["rhs_seed"]) in settings.importance_rhs_seeds
        ]
        baseline_iterations = float(
            np.median([float(row["iterations"]) for row in baseline_source])
        )
        baseline_convergence = float(
            np.mean([bool(row["converged"]) for row in baseline_source])
        )
        full_iterations = float(
            np.median([float(row["iterations"]) for row in full_source])
        )
        full_convergence = float(
            np.mean([bool(row["converged"]) for row in full_source])
        )
        track_map = {
            int(row["cluster_id"]): int(row["track_id"])
            for row in assignments
            if float(row["ratio"]) == ratio
        }
        for cluster in clustered.clusters:
            add_action, add_diagnostic = _coarse_action(
                context,
                cluster.basis,
                settings.stage10,
                f"add_cluster_{cluster.cluster_id}",
            )
            keep_indices = np.r_[0 : cluster.start, cluster.stop : 32]
            remove_basis = context.modes.basis[:, keep_indices]
            remove_action: Optional[InverseAction] = None
            remove_setup = float("nan")
            if remove_basis.shape[1] > 0:
                remove_action, remove_diagnostic = _coarse_action(
                    context,
                    remove_basis,
                    settings.stage10,
                    f"q32_without_cluster_{cluster.cluster_id}",
                )
                remove_setup = remove_diagnostic["coarse_setup_seconds"]
            method_results: Dict[str, Sequence[Any]] = {}
            add_results = [
                _run_solver(context, rhs_seed, add_action, settings.stage10)
                for rhs_seed in settings.importance_rhs_seeds
            ]
            method_results["add_cluster"] = add_results
            if remove_action is not None:
                remove_results = [
                    _run_solver(context, rhs_seed, remove_action, settings.stage10)
                    for rhs_seed in settings.importance_rhs_seeds
                ]
                method_results["remove_from_q32"] = remove_results
            for method, results in method_results.items():
                for rhs_seed, result in zip(settings.importance_rhs_seeds, results):
                    for iteration, residual in enumerate(result.relative_residuals):
                        histories.append(
                            {
                                "ratio": ratio,
                                "cluster_id": cluster.cluster_id,
                                "track_id": track_map[cluster.cluster_id],
                                "method": method,
                                "rhs_seed": rhs_seed,
                                "iteration": iteration,
                                "relative_residual": float(residual),
                            }
                        )
            add_summary = _summarize_results(add_results)
            remove_summary = (
                _summarize_results(method_results["remove_from_q32"])
                if "remove_from_q32" in method_results
                else {
                    "iterations_median": float("nan"),
                    "convergence_rate": 0.0,
                }
            )
            add_reduction = (
                baseline_iterations - add_summary["iterations_median"]
            ) / max(baseline_iterations, 1.0)
            remove_penalty = remove_summary["iterations_median"] - full_iterations
            solver_critical = bool(
                (
                    add_reduction >= settings.critical_iteration_reduction
                    and add_summary["convergence_rate"] >= 0.9
                )
                or remove_penalty >= settings.critical_removal_penalty
                or remove_summary["convergence_rate"] < full_convergence
            )
            rows.append(
                {
                    "ratio": ratio,
                    "cluster_id": cluster.cluster_id,
                    "track_id": track_map[cluster.cluster_id],
                    "start": cluster.start,
                    "stop": cluster.stop,
                    "dimension": cluster.dimension,
                    "magnitude_min": cluster.magnitude_min,
                    "magnitude_max": cluster.magnitude_max,
                    "baseline_iterations_median": baseline_iterations,
                    "baseline_convergence_rate": baseline_convergence,
                    "q32_iterations_median": full_iterations,
                    "q32_convergence_rate": full_convergence,
                    "add_iterations_median": add_summary["iterations_median"],
                    "add_convergence_rate": add_summary["convergence_rate"],
                    "add_iteration_reduction": add_reduction,
                    "add_coarse_setup_seconds": add_diagnostic[
                        "coarse_setup_seconds"
                    ],
                    "remove_iterations_median": remove_summary[
                        "iterations_median"
                    ],
                    "remove_convergence_rate": remove_summary[
                        "convergence_rate"
                    ],
                    "remove_iteration_penalty": remove_penalty,
                    "remove_coarse_setup_seconds": remove_setup,
                    "solver_critical": solver_critical,
                    "low_dimensional": cluster.dimension
                    <= settings.maximum_low_dimension,
                }
            )

    track_rows: List[Dict[str, Any]] = []
    for track_id in sorted({int(row["track_id"]) for row in rows}):
        local = [row for row in rows if int(row["track_id"]) == track_id]
        critical_fraction = float(
            np.mean([bool(row["solver_critical"]) for row in local])
        )
        track_rows.append(
            {
                "track_id": track_id,
                "occurrences": len(local),
                "dimension_min": min(int(row["dimension"]) for row in local),
                "dimension_max": max(int(row["dimension"]) for row in local),
                "critical_occurrence_fraction": critical_fraction,
                "add_iteration_reduction_median": float(
                    np.median([row["add_iteration_reduction"] for row in local])
                ),
                "remove_iteration_penalty_median": float(
                    np.median([row["remove_iteration_penalty"] for row in local])
                ),
                "solver_critical_track": bool(
                    len(local) >= settings.minimum_track_occurrences
                    and critical_fraction >= settings.critical_occurrence_fraction
                ),
            }
        )
    _write_csv(output / "cluster_solver_importance.csv", rows)
    _write_csv(output / "cluster_solver_histories.csv", histories)
    _write_csv(output / "cluster_solver_track_summary.csv", track_rows)
    return rows, histories, track_rows


def select_candidate_clusters(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    importance_rows: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> List[Dict[str, Any]]:
    """Select the strongest low-rank cluster per system without hiding failures."""

    rows: List[Dict[str, Any]] = []
    for ratio in contexts:
        local = [
            row
            for row in importance_rows
            if float(row["ratio"]) == ratio
            and int(row["dimension"]) <= settings.maximum_low_dimension
        ]
        if not local:
            continue
        for row in local:
            row["selection_score"] = float(
                row["add_iteration_reduction"]
                + max(float(row["remove_iteration_penalty"]), 0.0)
                / max(float(row["baseline_iterations_median"]), 1.0)
            )
        selected = max(
            local,
            key=lambda row: (
                bool(row["solver_critical"]),
                float(row["selection_score"]),
                -int(row["dimension"]),
            ),
        )
        rows.append(
            {
                "ratio": ratio,
                "cluster_id": int(selected["cluster_id"]),
                "track_id": int(selected["track_id"]),
                "dimension": int(selected["dimension"]),
                "solver_critical": bool(selected["solver_critical"]),
                "selection_score": float(selected["selection_score"]),
                "add_iteration_reduction": float(
                    selected["add_iteration_reduction"]
                ),
                "remove_iteration_penalty": float(
                    selected["remove_iteration_penalty"]
                ),
            }
        )
    _write_csv(output / "selected_cluster_candidates.csv", rows)
    return rows


def _crossing_atom_index(atoms: Sequence[RitzAtom], cutoff: int) -> int:
    return next(
        index
        for index, atom in enumerate(atoms)
        if atom.start < cutoff <= atom.stop
    )


def _atomic_windows_containing_cutoff(
    clustered: ClusteredContext,
    cutoff: int,
    maximum_dimension: int,
) -> List[Tuple[int, int, int]]:
    crossing = _crossing_atom_index(clustered.atoms, cutoff)
    windows: List[Tuple[int, int, int]] = []
    for first_atom in range(crossing + 1):
        for last_atom in range(crossing, len(clustered.atoms)):
            start = clustered.atoms[first_atom].start
            stop = clustered.atoms[last_atom].stop
            dimension = stop - start
            if dimension <= maximum_dimension:
                windows.append((start, stop, dimension))
    return windows


def run_enlarged_crossing_blocks(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    settings: Stage11Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Search an optimistic slightly larger block around the q=8 crossing.

    For every adjacent system pair, all conjugacy-preserving atomic windows
    containing the q=8 cutoff and having equal rank at most 16 are compared.
    The best match is an oracle stability upper bound, not a deployable target.
    Its own and reused solver actions are then tested on two held-out RHSs.
    """

    ratios = list(contexts)
    stage10_pairs = _read_csv(
        _stage10_results_directory(output) / "subspace_pairs.csv"
    )
    rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    for source_ratio, target_ratio in zip(ratios[:-1], ratios[1:]):
        source = contexts[source_ratio]
        target = contexts[target_ratio]
        source_windows = _atomic_windows_containing_cutoff(
            source,
            settings.fixed_prefix_dimension,
            settings.maximum_expanded_dimension,
        )
        target_windows = _atomic_windows_containing_cutoff(
            target,
            settings.fixed_prefix_dimension,
            settings.maximum_expanded_dimension,
        )
        overlap = source.context.modes.basis.T @ target.context.modes.basis
        best: Optional[Tuple[float, Tuple[int, int, int], Tuple[int, int, int]]] = None
        for source_window in source_windows:
            for target_window in target_windows:
                if source_window[2] != target_window[2]:
                    continue
                s0, s1, dimension = source_window
                t0, t1, _ = target_window
                shared = float(np.linalg.norm(overlap[s0:s1, t0:t1], ord="fro") ** 2)
                distance = float(
                    np.sqrt(max(0.0, 2.0 * dimension - 2.0 * shared))
                    / np.sqrt(2.0 * dimension)
                )
                if best is None or distance < best[0]:
                    best = (distance, source_window, target_window)
        q8_row = next(
            row
            for row in stage10_pairs
            if int(row["q"]) == settings.fixed_prefix_dimension
            and float(row["ratio_i"]) == source_ratio
            and float(row["ratio_j"]) == target_ratio
        )
        source_crossing_atom = source.atoms[
            _crossing_atom_index(source.atoms, settings.fixed_prefix_dimension)
        ]
        target_crossing_atom = target.atoms[
            _crossing_atom_index(target.atoms, settings.fixed_prefix_dimension)
        ]
        pair_complete_source_q = source_crossing_atom.stop
        pair_complete_target_q = target_crossing_atom.stop
        pair_metrics = generalized_subspace_metrics(
            source.context.modes.basis[:, :pair_complete_source_q],
            target.context.modes.basis[:, :pair_complete_target_q],
        )
        if best is None:
            rows.append(
                {
                    "source_ratio": source_ratio,
                    "target_ratio": target_ratio,
                    "q8_projector_distance": q8_row[
                        "normalized_projector_distance"
                    ],
                    "pair_complete_source_q": pair_complete_source_q,
                    "pair_complete_target_q": pair_complete_target_q,
                    "pair_complete_projector_distance": pair_metrics[
                        "projector_distance_normalized"
                    ],
                    "expanded_available": False,
                }
            )
            continue
        distance, source_window, target_window = best
        source_basis = source.context.modes.basis[
            :, source_window[0] : source_window[1]
        ]
        target_basis = target.context.modes.basis[
            :, target_window[0] : target_window[1]
        ]
        expanded_metrics = generalized_subspace_metrics(source_basis, target_basis)
        stable = bool(
            distance <= settings.stable_projector_distance
            and expanded_metrics["principal_angle_max_degrees"]
            <= settings.stable_max_principal_angle_degrees
        )
        rows.append(
            {
                "source_ratio": source_ratio,
                "target_ratio": target_ratio,
                "q8_projector_distance": q8_row[
                    "normalized_projector_distance"
                ],
                "q8_principal_angle_max_degrees": q8_row[
                    "principal_angle_max_degrees"
                ],
                "pair_complete_source_q": pair_complete_source_q,
                "pair_complete_target_q": pair_complete_target_q,
                "pair_complete_projector_distance": pair_metrics[
                    "projector_distance_normalized"
                ],
                "pair_complete_principal_angle_max_degrees": pair_metrics[
                    "principal_angle_max_degrees"
                ],
                "expanded_available": True,
                "expanded_source_start": source_window[0],
                "expanded_source_stop": source_window[1],
                "expanded_target_start": target_window[0],
                "expanded_target_stop": target_window[1],
                "expanded_dimension": source_window[2],
                "expanded_projector_distance": distance,
                "expanded_principal_angle_mean_degrees": expanded_metrics[
                    "principal_angle_mean_degrees"
                ],
                "expanded_principal_angle_max_degrees": expanded_metrics[
                    "principal_angle_max_degrees"
                ],
                "expanded_stable": stable,
            }
        )
        own_action, _ = _coarse_action(
            target.context,
            target_basis,
            settings.stage10,
            "own_expanded_crossing_block",
        )
        reuse_action, _ = _coarse_action(
            target.context,
            source_basis,
            settings.stage10,
            "reused_expanded_crossing_block",
        )
        for method, action in (("own_expanded", own_action), ("reused_expanded", reuse_action)):
            results = [
                _run_solver(target.context, rhs_seed, action, settings.stage10)
                for rhs_seed in settings.importance_rhs_seeds
            ]
            summary = _summarize_results(results)
            solver_rows.append(
                {
                    "source_ratio": source_ratio,
                    "target_ratio": target_ratio,
                    "method": method,
                    "dimension": source_window[2],
                    **summary,
                }
            )
    _write_csv(output / "expanded_crossing_blocks.csv", rows)
    _write_csv(output / "expanded_crossing_solver_results.csv", solver_rows)
    return rows, solver_rows


def _match_reference_to_context(
    reference: RitzCluster,
    candidate_context: ClusteredContext,
    reference_nodes: np.ndarray,
    interpolate: bool,
) -> Tuple[RitzCluster, np.ndarray, Dict[str, Any]]:
    """Find the candidate cluster closest to one reference block."""

    best: Optional[Tuple[float, RitzCluster, np.ndarray, Dict[str, Any]]] = None
    for candidate in candidate_context.clusters:
        candidate_basis = (
            interpolate_basis_to_probe(
                candidate_context.context.system.nodes,
                candidate.basis,
                reference_nodes,
            )
            if interpolate
            else candidate.basis
        )
        metrics = generalized_subspace_metrics(reference.basis, candidate_basis)
        spectral_log_distance = float(
            abs(
                np.log10(
                    (reference.magnitude_geometric_mean + np.finfo(float).eps)
                    / (
                        candidate.magnitude_geometric_mean
                        + np.finfo(float).eps
                    )
                )
            )
        )
        dimension_penalty = abs(reference.dimension - candidate.dimension) / (
            reference.dimension + candidate.dimension
        )
        cost = float(
            metrics["projector_distance_normalized"]
            + 0.25 * min(spectral_log_distance, 2.0)
            + 0.25 * dimension_penalty
        )
        enriched = {
            **{key: value for key, value in metrics.items() if key != "principal_angles_degrees"},
            "spectral_log10_distance": spectral_log_distance,
            "spectral_factor": float(10.0**spectral_log_distance),
            "matching_cost": cost,
        }
        if best is None or cost < best[0]:
            best = (cost, candidate, candidate_basis, enriched)
    if best is None:
        raise RuntimeError("candidate context has no Ritz clusters")
    return best[1], best[2], best[3]


def _cluster_context_from_stage10(
    context: RitzContext,
    settings: Stage11Settings,
) -> ClusteredContext:
    atoms, clusters, gaps, threshold = detect_ritz_clusters(
        context.ratio,
        context.modes.basis,
        context.modes.eigenvalues,
        settings.gap_mad_multiplier,
    )
    return ClusteredContext(context, atoms, clusters, gaps, threshold)


def run_cluster_robustness(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    selected_rows: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Repeat selected critical candidates across RHS and node perturbations."""

    match_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    for ratio in settings.robustness_ratios:
        if ratio not in contexts:
            continue
        selected = next(row for row in selected_rows if float(row["ratio"]) == ratio)
        base_context = contexts[ratio]
        reference = base_context.clusters[int(selected["cluster_id"])]
        rhs_contexts: List[Tuple[int, ClusteredContext]] = [
            (settings.stage10.construction_rhs_seed, base_context)
        ]
        rhs_contexts.extend(
            (
                seed,
                _cluster_context_from_stage10(
                    _build_context(
                        ratio,
                        settings.stage10,
                        construction_seed=seed,
                        node_seed=settings.stage10.node_seed,
                    ),
                    settings,
                ),
            )
            for seed in settings.construction_rhs_replicates
            if seed != settings.stage10.construction_rhs_seed
        )
        node_contexts: List[Tuple[int, ClusteredContext]] = [
            (settings.stage10.node_seed, base_context)
        ]
        node_contexts.extend(
            (
                node_seed,
                _cluster_context_from_stage10(
                    _build_context(
                        ratio,
                        settings.stage10,
                        construction_seed=settings.stage10.construction_rhs_seed,
                        node_seed=node_seed,
                    ),
                    settings,
                ),
            )
            for node_seed in settings.node_seeds
            if node_seed != settings.stage10.node_seed
        )
        matched_by_variation: Dict[str, List[Tuple[int, ClusteredContext, RitzCluster, np.ndarray]]] = {}
        for variation, replicates, interpolate in (
            ("construction_rhs", rhs_contexts, False),
            ("node_perturbation", node_contexts, True),
        ):
            matched: List[Tuple[int, ClusteredContext, RitzCluster, np.ndarray]] = []
            for replicate, replicate_context in replicates:
                if replicate_context is base_context:
                    candidate = reference
                    candidate_basis = reference.basis
                    metrics = generalized_subspace_metrics(
                        reference.basis, reference.basis
                    )
                    spectral_factor = 1.0
                    matching_cost = 0.0
                else:
                    candidate, candidate_basis, metrics = _match_reference_to_context(
                        reference,
                        replicate_context,
                        base_context.context.system.nodes,
                        interpolate,
                    )
                    spectral_factor = metrics["spectral_factor"]
                    matching_cost = metrics["matching_cost"]
                matched.append((replicate, replicate_context, candidate, candidate_basis))
                match_rows.append(
                    {
                        "ratio": ratio,
                        "variation": variation,
                        "replicate": replicate,
                        "reference_cluster_id": reference.cluster_id,
                        "matched_cluster_id": candidate.cluster_id,
                        "reference_dimension": reference.dimension,
                        "matched_dimension": candidate.dimension,
                        "projector_distance_normalized": metrics[
                            "projector_distance_normalized"
                        ],
                        "principal_angle_mean_degrees": metrics[
                            "principal_angle_mean_degrees"
                        ],
                        "principal_angle_max_degrees": metrics[
                            "principal_angle_max_degrees"
                        ],
                        "spectral_factor": spectral_factor,
                        "matching_cost": matching_cost,
                    }
                )
                action, diagnostic = _coarse_action(
                    replicate_context.context,
                    candidate.basis,
                    settings.stage10,
                    f"robust_{variation}_cluster",
                )
                result = _run_solver(
                    replicate_context.context,
                    settings.importance_rhs_seeds[0],
                    action,
                    settings.stage10,
                )
                solver_rows.append(
                    {
                        "ratio": ratio,
                        "variation": variation,
                        "replicate": replicate,
                        "cluster_dimension": candidate.dimension,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "final_relative_residual": float(
                            result.relative_residuals[-1]
                        ),
                        "solve_seconds": result.solve_seconds,
                        "coarse_setup_seconds": diagnostic[
                            "coarse_setup_seconds"
                        ],
                    }
                )
            matched_by_variation[variation] = matched
            for first_index in range(len(matched)):
                for second_index in range(first_index + 1, len(matched)):
                    first = matched[first_index]
                    second = matched[second_index]
                    first_basis = first[3]
                    second_basis = second[3]
                    metrics = generalized_subspace_metrics(first_basis, second_basis)
                    pair_rows.append(
                        {
                            "ratio": ratio,
                            "variation": variation,
                            "replicate_i": first[0],
                            "replicate_j": second[0],
                            "dimension_i": first[2].dimension,
                            "dimension_j": second[2].dimension,
                            **{
                                key: value
                                for key, value in metrics.items()
                                if key != "principal_angles_degrees"
                            },
                        }
                    )
    _write_csv(output / "cluster_robustness_matches.csv", match_rows)
    _write_csv(output / "cluster_robustness_pairs.csv", pair_rows)
    _write_csv(output / "cluster_robustness_solver.csv", solver_rows)
    return match_rows, pair_rows, solver_rows


def _neighbor_cluster_for_target(
    target_ratio: float,
    target_cluster: RitzCluster,
    source_ratio: float,
    contexts: Mapping[float, ClusteredContext],
    pair_rows: Sequence[Mapping[str, Any]],
) -> RitzCluster:
    if source_ratio < target_ratio:
        matches = [
            row
            for row in pair_rows
            if float(row["source_ratio"]) == source_ratio
            and float(row["target_ratio"]) == target_ratio
            and int(row["target_cluster_id"]) == target_cluster.cluster_id
        ]
        if matches:
            return contexts[source_ratio].clusters[
                int(matches[0]["source_cluster_id"])
            ]
    else:
        matches = [
            row
            for row in pair_rows
            if float(row["source_ratio"]) == target_ratio
            and float(row["target_ratio"]) == source_ratio
            and int(row["source_cluster_id"]) == target_cluster.cluster_id
        ]
        if matches:
            return contexts[source_ratio].clusters[
                int(matches[0]["target_cluster_id"])
            ]
    candidates = contexts[source_ratio].clusters
    return min(
        candidates,
        key=lambda candidate: _cluster_pair_metrics(target_cluster, candidate)[
            "matching_cost"
        ],
    )


def run_cluster_transfer(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    selected_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare Stage 10 full-q8 controls with candidate cluster transfer."""

    stage10_dir = _stage10_results_directory(output)
    stage10_transfer = _read_csv(stage10_dir / "transfer_results.csv")
    stage10_histories = _read_csv(stage10_dir / "transfer_histories.csv")
    method_map = {
        "ILU": "ILU",
        "own_oracle": "own_full_q8",
        "nearby_reuse": "nearby_full_q8_reuse",
        "random_control": "random_q8",
    }
    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    for row in stage10_transfer:
        ratio = float(row["target_ratio"])
        if ratio in contexts and row["method"] in method_map:
            copied = dict(row)
            copied["method"] = method_map[row["method"]]
            copied["cluster_dimension"] = ""
            rows.append(copied)
    for row in stage10_histories:
        ratio = float(row["target_ratio"])
        if ratio in contexts and row["method"] in method_map:
            copied = dict(row)
            copied["method"] = method_map[row["method"]]
            histories.append(copied)

    ratios = list(contexts)
    selected_map = {float(row["ratio"]): row for row in selected_rows}
    for target_index, target_ratio in enumerate(ratios):
        target = contexts[target_ratio]
        target_selection = selected_map[target_ratio]
        target_cluster = target.clusters[int(target_selection["cluster_id"])]
        other_ratios = [ratio for ratio in ratios if ratio != target_ratio]
        source_ratio = min(
            other_ratios,
            key=lambda ratio: (abs(ratio - target_ratio), ratio > target_ratio, ratio),
        )
        source_cluster = _neighbor_cluster_for_target(
            target_ratio,
            target_cluster,
            source_ratio,
            contexts,
            pair_rows,
        )
        random_basis = random_orthonormal_basis(
            settings.stage10.n_nodes,
            source_cluster.dimension,
            settings.stage10.seed + 7000 + target_index,
        )
        method_specs = (
            ("own_cluster", target_ratio, target_cluster.basis),
            ("nearby_cluster_reuse", source_ratio, source_cluster.basis),
            ("random_cluster", None, random_basis),
        )
        for method, source, basis in method_specs:
            action, diagnostic = _coarse_action(
                target.context, basis, settings.stage10, method
            )
            results = []
            for rhs_seed in settings.stage10.evaluation_rhs_seeds:
                result = _run_solver(
                    target.context, rhs_seed, action, settings.stage10
                )
                results.append(result)
                for iteration, residual in enumerate(result.relative_residuals):
                    histories.append(
                        {
                            "target_ratio": target_ratio,
                            "method": method,
                            "source_ratio": "" if source is None else source,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "relative_residual": float(residual),
                        }
                    )
            summary = _summarize_results(results)
            rows.append(
                {
                    "target_ratio": target_ratio,
                    "method": method,
                    "source_ratio": "" if source is None else source,
                    "q": basis.shape[1],
                    "cluster_dimension": basis.shape[1],
                    **summary,
                    "setup_seconds": diagnostic["coarse_setup_seconds"],
                    "target_cluster_id": target_cluster.cluster_id,
                    "source_cluster_id": (
                        "" if source is None else source_cluster.cluster_id
                    ),
                    "target_cluster_solver_critical": target_selection[
                        "solver_critical"
                    ],
                }
            )
    _write_csv(output / "cluster_transfer_results.csv", rows)
    _write_csv(output / "cluster_transfer_histories.csv", histories)
    return rows, histories


def run_gap_sensitivity(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    settings: Stage11Settings,
) -> List[Dict[str, Any]]:
    """Check whether cluster conclusions depend on the MAD multiplier."""

    rows: List[Dict[str, Any]] = []
    for ratio, primary in contexts.items():
        _, alternate, _, alternate_threshold = detect_ritz_clusters(
            ratio,
            primary.context.modes.basis,
            primary.context.modes.eigenvalues,
            settings.gap_sensitivity_multiplier,
        )
        primary_crossing = next(
            cluster
            for cluster in primary.clusters
            if cluster.start < settings.fixed_prefix_dimension <= cluster.stop
        )
        alternate_crossing = next(
            cluster
            for cluster in alternate
            if cluster.start < settings.fixed_prefix_dimension <= cluster.stop
        )
        primary_boundaries = {cluster.stop for cluster in primary.clusters[:-1]}
        alternate_boundaries = {cluster.stop for cluster in alternate[:-1]}
        union = primary_boundaries | alternate_boundaries
        intersection = primary_boundaries & alternate_boundaries
        rows.append(
            {
                "ratio": ratio,
                "primary_multiplier": settings.gap_mad_multiplier,
                "alternate_multiplier": settings.gap_sensitivity_multiplier,
                "primary_threshold_log10": primary.gap_threshold,
                "alternate_threshold_log10": alternate_threshold,
                "primary_cluster_count": len(primary.clusters),
                "alternate_cluster_count": len(alternate),
                "boundary_jaccard": (
                    len(intersection) / len(union) if union else 1.0
                ),
                "primary_q8_cluster_start": primary_crossing.start,
                "primary_q8_cluster_stop": primary_crossing.stop,
                "primary_q8_cluster_dimension": primary_crossing.dimension,
                "alternate_q8_cluster_start": alternate_crossing.start,
                "alternate_q8_cluster_stop": alternate_crossing.stop,
                "alternate_q8_cluster_dimension": alternate_crossing.dimension,
            }
        )
    _write_csv(output / "gap_sensitivity.csv", rows)
    return rows


def build_decision(
    output: Path,
    cluster_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    importance_rows: Sequence[Mapping[str, Any]],
    importance_track_rows: Sequence[Mapping[str, Any]],
    robustness_pair_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    expanded_rows: Sequence[Mapping[str, Any]],
    gap_sensitivity_rows: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> Dict[str, Any]:
    """Apply the solver relevance, stability, robustness, and transfer gates."""

    stage10_mode_rows = _read_csv(
        _stage10_results_directory(output) / "mode_tracking_adjacent.csv"
    )
    relevant_steps = {
        (first, second)
        for first, second in zip(settings.ratios[:-1], settings.ratios[1:])
    }
    individual_rows = [
        row
        for row in stage10_mode_rows
        if (float(row["source_ratio"]), float(row["target_ratio"]))
        in relevant_steps
    ]
    individual_angles = [
        float(np.degrees(np.arccos(np.clip(float(row["absolute_cosine"]), 0.0, 1.0))))
        for row in individual_rows
    ]
    low_rank_pairs = [
        row
        for row in pair_rows
        if int(row["source_dimension"]) <= settings.maximum_low_dimension
        and int(row["target_dimension"]) <= settings.maximum_low_dimension
    ]
    stable_tracks = {
        int(row["track_id"])
        for row in track_rows
        if bool(row["stable_track"])
    }
    critical_tracks = {
        int(row["track_id"])
        for row in importance_track_rows
        if bool(row["solver_critical_track"])
        and int(row["dimension_max"]) <= settings.maximum_low_dimension
    }
    stable_critical_tracks = stable_tracks & critical_tracks
    robust_rhs = [
        float(row["projector_distance_normalized"])
        for row in robustness_pair_rows
        if row["variation"] == "construction_rhs"
    ]
    robust_nodes = [
        float(row["projector_distance_normalized"])
        for row in robustness_pair_rows
        if row["variation"] == "node_perturbation"
    ]
    rhs_p90 = float(np.quantile(robust_rhs, 0.9))
    node_p90 = float(np.quantile(robust_nodes, 0.9))
    robustness_pass = bool(
        rhs_p90 <= settings.stable_projector_distance
        and node_p90 <= settings.stable_projector_distance
    )
    hard_ratios = {
        float(row["target_ratio"])
        for row in transfer_rows
        if row["method"] == "ILU" and float(row["iterations_median"]) >= 50.0
    }
    cluster_reuse = [
        row
        for row in transfer_rows
        if row["method"] == "nearby_cluster_reuse"
        and float(row["target_ratio"]) in hard_ratios
    ]
    full_reuse = [
        row
        for row in transfer_rows
        if row["method"] == "nearby_full_q8_reuse"
        and float(row["target_ratio"]) in hard_ratios
    ]
    baselines = {
        float(row["target_ratio"]): float(row["iterations_median"])
        for row in transfer_rows
        if row["method"] == "ILU"
    }

    def reductions(rows: Sequence[Mapping[str, Any]]) -> List[float]:
        return [
            (
                baselines[float(row["target_ratio"])]
                - float(row["iterations_median"])
            )
            / max(baselines[float(row["target_ratio"])], 1.0)
            for row in rows
        ]

    cluster_reductions = reductions(cluster_reuse)
    full_reductions = reductions(full_reuse)
    expanded_solver_rows = _read_csv(output / "expanded_crossing_solver_results.csv")
    own_expanded = [
        row
        for row in expanded_solver_rows
        if row["method"] == "own_expanded"
        and float(row["target_ratio"]) in hard_ratios
    ]
    reused_expanded = [
        row
        for row in expanded_solver_rows
        if row["method"] == "reused_expanded"
        and float(row["target_ratio"]) in hard_ratios
    ]
    own_expanded_reductions = reductions(own_expanded)
    reused_expanded_reductions = reductions(reused_expanded)
    transfer_pass = bool(
        cluster_reuse
        and np.median(cluster_reductions)
        >= settings.critical_iteration_reduction
        and np.mean([float(row["convergence_rate"]) for row in cluster_reuse])
        >= 0.9
    )
    expanded_available = [row for row in expanded_rows if bool(row["expanded_available"])]
    expanded_stable = [row for row in expanded_available if bool(row["expanded_stable"])]
    crossing_pair_cuts = sum(
        bool(row["crosses_q8_cut"])
        for row in _read_csv(output / "ritz_atoms.csv")
    )
    low_dimensional_critical_occurrences = [
        row
        for row in importance_rows
        if bool(row["solver_critical"])
        and int(row["dimension"]) <= settings.maximum_low_dimension
    ]
    cluster_stability_better = bool(
        low_rank_pairs
        and np.median(
            [float(row["principal_angle_max_degrees"]) for row in low_rank_pairs]
        )
        < np.median(individual_angles)
    )
    gap_boundary_jaccard_median = float(
        np.median([float(row["boundary_jaccard"]) for row in gap_sensitivity_rows])
    )
    target_exists = bool(
        stable_critical_tracks
        and robustness_pass
        and transfer_pass
        and cluster_stability_better
    )
    decision = {
        "individual_adjacent_angle_median_degrees": float(
            np.median(individual_angles)
        ),
        "low_rank_cluster_match_count": len(low_rank_pairs),
        "low_rank_cluster_projector_distance_median": float(
            np.median(
                [row["projector_distance_normalized"] for row in low_rank_pairs]
            )
        ),
        "low_rank_cluster_max_angle_median_degrees": float(
            np.median(
                [row["principal_angle_max_degrees"] for row in low_rank_pairs]
            )
        ),
        "low_rank_cluster_spectral_factor_median": float(
            np.median([row["spectral_factor"] for row in low_rank_pairs])
        ),
        "low_rank_cluster_spectral_gate_fraction": float(
            np.mean(
                [
                    float(row["spectral_factor"])
                    <= settings.stable_spectral_factor
                    for row in low_rank_pairs
                ]
            )
        ),
        "cluster_stability_better_than_individual": cluster_stability_better,
        "stable_track_count": len(stable_tracks),
        "critical_track_count": len(critical_tracks),
        "stable_critical_track_count": len(stable_critical_tracks),
        "stable_critical_track_ids": sorted(stable_critical_tracks),
        "low_dimensional_critical_occurrences": len(
            low_dimensional_critical_occurrences
        ),
        "rhs_robustness_p90_projector_distance": rhs_p90,
        "node_robustness_p90_projector_distance": node_p90,
        "robustness_pass": robustness_pass,
        "hard_ratio_count": len(hard_ratios),
        "cluster_reuse_iteration_reduction_median": float(
            np.median(cluster_reductions)
        ),
        "cluster_reuse_convergence_rate_mean": float(
            np.mean([float(row["convergence_rate"]) for row in cluster_reuse])
        ),
        "full_q8_reuse_iteration_reduction_median": float(
            np.median(full_reductions)
        ),
        "full_q8_reuse_convergence_rate_mean": float(
            np.mean([float(row["convergence_rate"]) for row in full_reuse])
        ),
        "own_expanded_iteration_reduction_median": float(
            np.median(own_expanded_reductions)
        ),
        "own_expanded_convergence_rate_mean": float(
            np.mean([float(row["convergence_rate"]) for row in own_expanded])
        ),
        "reused_expanded_iteration_reduction_median": float(
            np.median(reused_expanded_reductions)
        ),
        "reused_expanded_convergence_rate_mean": float(
            np.mean([float(row["convergence_rate"]) for row in reused_expanded])
        ),
        "cluster_transfer_pass": transfer_pass,
        "q8_complex_plane_cut_count": crossing_pair_cuts,
        "expanded_pair_count": len(expanded_available),
        "expanded_stable_pair_count": len(expanded_stable),
        "expanded_stable_pair_fraction": (
            len(expanded_stable) / len(expanded_available)
            if expanded_available
            else 0.0
        ),
        "expanded_projector_distance_median": float(
            np.median(
                [row["expanded_projector_distance"] for row in expanded_available]
            )
        ),
        "q8_projector_distance_median_same_pairs": float(
            np.median([row["q8_projector_distance"] for row in expanded_available])
        ),
        "gap_boundary_jaccard_median": gap_boundary_jaccard_median,
        "new_learned_subspace_target_exists": target_exists,
        "pause_current_ritz_learned_subspace_route": not target_exists,
        "decision_rule": (
            "A new target requires at least one rank<=8 track that is both stable "
            "and solver-critical, p90 RHS/node projector distances <=0.35, nearby "
            "cluster reuse with >=20% median hard-regime reduction and >=0.90 "
            "convergence, and better angular stability than individual vectors."
        ),
    }
    _write_json(output / "decision.json", decision)
    return decision


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    pair_rows: Sequence[Mapping[str, Any]],
    importance_rows: Sequence[Mapping[str, Any]],
    robustness_pair_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    transfer_histories: Sequence[Mapping[str, Any]],
    expanded_rows: Sequence[Mapping[str, Any]],
    settings: Stage11Settings,
) -> List[str]:
    """Render the compact, neutral Stage 11 evidence figures."""

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []

    selected_ratios = [84.0, 90.0, 100.0, 120.0]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    for axis, ratio in zip(axes.flat, selected_ratios):
        clustered = contexts[ratio]
        magnitudes = np.abs(clustered.context.modes.eigenvalues)
        axis.semilogy(np.arange(1, len(magnitudes) + 1), magnitudes, "o-", color=BLUE)
        for cluster in clustered.clusters[:-1]:
            axis.axvline(cluster.stop + 0.5, color=ORANGE, linestyle="--", alpha=0.8)
        axis.axvline(
            settings.fixed_prefix_dimension + 0.5,
            color=PINK,
            linewidth=1.8,
            label="q=8 cut" if ratio == selected_ratios[0] else None,
        )
        axis.set_title(f"R={ratio:g}: {len(clustered.clusters)} gap clusters", loc="left")
        axis.set_ylabel("|Ritz value|")
        axis.grid(color=LIGHT_GRAY, alpha=0.55)
    axes[-1, 0].set_xlabel("real Ritz direction index")
    axes[-1, 1].set_xlabel("real Ritz direction index")
    axes[0, 0].legend(frameon=False)
    name = "ritz_spectra_and_gap_clusters.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    ratios = list(contexts)
    max_clusters = max(len(context.clusters) for context in contexts.values())
    dimension_matrix = np.full((len(ratios), max_clusters), np.nan)
    for row_index, ratio in enumerate(ratios):
        for cluster in contexts[ratio].clusters:
            dimension_matrix[row_index, cluster.cluster_id] = cluster.dimension
    figure, axis = plt.subplots(figsize=(10.0, 6.4))
    handle = axis.imshow(dimension_matrix, aspect="auto", cmap="Blues")
    axis.set_xticks(np.arange(max_clusters), [str(index) for index in range(max_clusters)])
    axis.set_yticks(np.arange(len(ratios)), [f"{ratio:g}" for ratio in ratios])
    axis.set(xlabel="local cluster ID", ylabel="anisotropy ratio R")
    axis.set_title("Gap-detected cluster dimensions vary with R", loc="left")
    for i in range(len(ratios)):
        for j in range(max_clusters):
            if np.isfinite(dimension_matrix[i, j]):
                axis.text(j, i, f"{dimension_matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)
    figure.colorbar(handle, ax=axis, label="cluster dimension")
    name = "cluster_dimension_heatmap.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    stage10_modes = _read_csv(
        _stage10_results_directory(output) / "mode_tracking_adjacent.csv"
    )
    steps = {(a, b) for a, b in zip(settings.ratios[:-1], settings.ratios[1:])}
    individual_angles = [
        float(np.degrees(np.arccos(np.clip(float(row["absolute_cosine"]), 0.0, 1.0))))
        for row in stage10_modes
        if (float(row["source_ratio"]), float(row["target_ratio"])) in steps
    ]
    cluster_angles = [
        float(row["principal_angle_max_degrees"])
        for row in pair_rows
        if int(row["source_dimension"]) <= settings.maximum_low_dimension
        and int(row["target_dimension"]) <= settings.maximum_low_dimension
    ]
    expanded_angles = [
        float(row["expanded_principal_angle_max_degrees"])
        for row in expanded_rows
        if bool(row["expanded_available"])
    ]
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    boxes = axis.boxplot(
        [individual_angles, cluster_angles, expanded_angles],
        labels=["individual Ritz\ndirections", "matched low-rank\nclusters", "oracle enlarged\nblocks (≤16)"],
        patch_artist=True,
    )
    for box, color in zip(boxes["boxes"], [PINK, BLUE, ORANGE]):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axis.axhline(
        settings.stable_max_principal_angle_degrees,
        color=CHARCOAL,
        linestyle="--",
        label="stability gate",
    )
    axis.set_ylabel("angle (degrees)")
    axis.set_title("Neighboring-system angular instability", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False)
    name = "individual_cluster_enlarged_stability.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    noncritical = [row for row in importance_rows if not bool(row["solver_critical"])]
    critical = [row for row in importance_rows if bool(row["solver_critical"])]
    axis.scatter(
        [int(row["dimension"]) for row in noncritical],
        [100.0 * float(row["add_iteration_reduction"]) for row in noncritical],
        facecolors="none",
        edgecolors=BLUE,
        label="not solver-critical",
    )
    axis.scatter(
        [int(row["dimension"]) for row in critical],
        [100.0 * float(row["add_iteration_reduction"]) for row in critical],
        color=ORANGE,
        label="solver-critical by add/remove gate",
    )
    axis.axhline(100.0 * settings.critical_iteration_reduction, color=LIGHT_GRAY, linestyle="--")
    axis.axvline(settings.maximum_low_dimension, color=LIGHT_GRAY, linestyle=":")
    axis.set(xlabel="cluster dimension", ylabel="iteration reduction from cluster alone (%)")
    axis.set_title("Cluster dimension versus held-out solver importance", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.45)
    axis.legend(frameon=False)
    name = "cluster_solver_importance.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    methods = [
        "ILU",
        "own_full_q8",
        "nearby_full_q8_reuse",
        "own_cluster",
        "nearby_cluster_reuse",
        "random_cluster",
    ]
    colors = [CHARCOAL, BLUE, ORANGE, PURPLE, PINK, OLIVE]
    figure, axis = plt.subplots(figsize=(10.7, 5.5))
    for method, color in zip(methods, colors):
        local = sorted(
            [row for row in transfer_rows if row["method"] == method],
            key=lambda row: float(row["target_ratio"]),
        )
        axis.plot(
            [float(row["target_ratio"]) for row in local],
            [float(row["iterations_median"]) for row in local],
            marker="o",
            linewidth=1.8,
            label=method.replace("_", " "),
            color=color,
        )
    axis.set(xlabel="target anisotropy ratio R", ylabel="median FGMRES iterations")
    axis.set_title("Full-q8 and candidate-cluster transfer", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False, ncol=3)
    name = "cluster_transfer_comparison.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    robustness_values = [
        [
            float(row["projector_distance_normalized"])
            for row in robustness_pair_rows
            if row["variation"] == variation
        ]
        for variation in ("construction_rhs", "node_perturbation")
    ]
    boxes = axis.boxplot(
        robustness_values,
        labels=["construction RHS", "node perturbation\n(interpolated)"],
        patch_artist=True,
    )
    for box, color in zip(boxes["boxes"], [BLUE, ORANGE]):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    axis.axhline(settings.stable_projector_distance, color=CHARCOAL, linestyle="--")
    axis.set_ylabel("normalized projector distance")
    axis.set_title("Selected-cluster robustness", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    name = "cluster_robustness.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(10.2, 5.0))
    labels = [
        f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}"
        for row in expanded_rows
    ]
    axis.plot(
        labels,
        [float(row["q8_projector_distance"]) for row in expanded_rows],
        marker="o",
        color=PINK,
        label="q=8 prefix",
    )
    axis.plot(
        labels,
        [float(row["pair_complete_projector_distance"]) for row in expanded_rows],
        marker="s",
        color=BLUE,
        label="conjugacy-completed prefix",
    )
    axis.plot(
        labels,
        [
            float(row["expanded_projector_distance"])
            if bool(row["expanded_available"])
            else np.nan
            for row in expanded_rows
        ],
        marker="^",
        color=ORANGE,
        label="best oracle block ≤16",
    )
    axis.axhline(settings.stable_projector_distance, color=CHARCOAL, linestyle="--")
    axis.set(xlabel="neighboring R pair", ylabel="normalized projector distance")
    axis.set_title("Completing and enlarging the q=8 crossing block", loc="left")
    axis.tick_params(axis="x", rotation=55)
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False)
    name = "expanded_crossing_block_stability.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    residual_ratio = 100.0
    residual_seed = settings.stage10.evaluation_rhs_seeds[0]
    figure, axis = plt.subplots(figsize=(8.8, 4.9))
    for method, color in zip(methods, colors):
        local = [
            row
            for row in transfer_histories
            if float(row["target_ratio"]) == residual_ratio
            and int(row["rhs_seed"]) == residual_seed
            and row["method"] == method
        ]
        if local:
            axis.semilogy(
                [int(row["iteration"]) for row in local],
                [float(row["relative_residual"]) for row in local],
                color=color,
                label=method.replace("_", " "),
            )
    axis.axhline(settings.stage10.solver.relative_tolerance, color=LIGHT_GRAY, linestyle="--")
    axis.set(xlabel="FGMRES iteration", ylabel="true relative residual")
    axis.set_title("Held-out convergence at R=100", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.5)
    axis.legend(frameon=False, ncol=2)
    name = "cluster_transfer_residual_history.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)
    return names


def _format(value: Any, digits: int = 3) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if value is None or value == "":
        return "—"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "—"
    if numeric != 0.0 and (abs(numeric) < 1.0e-3 or abs(numeric) >= 1.0e4):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    contexts: Mapping[float, ClusteredContext],
    cluster_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    importance_rows: Sequence[Mapping[str, Any]],
    importance_track_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    robustness_pair_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    expanded_rows: Sequence[Mapping[str, Any]],
    gap_sensitivity_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage11Settings,
) -> Path:
    """Write the requested methods-focused Markdown report."""

    target_exists = bool(decision["new_learned_subspace_target_exists"])
    if target_exists:
        headline = (
            "A low-dimensional Ritz block passes the stability, solver, robustness, "
            "and transfer gates and is defined as the new learned-subspace target."
        )
    else:
        headline = (
            "Ritz clustering does not rescue the learned-subspace target: some vector "
            "rotation is explained by complex planes, but no low-rank block is jointly "
            "stable, solver-critical, robust, and transferable."
        )

    cluster_protocol_rows = []
    selected_map = {float(row["ratio"]): row for row in selected_rows}
    atom_rows = _read_csv(output / "ritz_atoms.csv")
    for ratio, context in contexts.items():
        ranks = ",".join(str(cluster.dimension) for cluster in context.clusters)
        crossing_cluster = next(
            cluster
            for cluster in context.clusters
            if cluster.start < settings.fixed_prefix_dimension <= cluster.stop
        )
        pair_cut = any(
            bool(row["crosses_q8_cut"])
            for row in atom_rows
            if float(row["ratio"]) == ratio
        )
        selected = selected_map[ratio]
        cluster_protocol_rows.append(
            (
                f"{ratio:g}",
                str(len(context.atoms)),
                str(len(context.clusters)),
                ranks,
                "yes" if pair_cut else "no",
                f"[{crossing_cluster.start},{crossing_cluster.stop}) / {crossing_cluster.dimension}",
                f"{int(selected['cluster_id'])} / {int(selected['dimension'])}",
                "yes" if bool(selected["solver_critical"]) else "no",
            )
        )

    importance_track_map = {
        int(row["track_id"]): row for row in importance_track_rows
    }
    track_table_rows = []
    for row in sorted(
        track_rows,
        key=lambda item: (
            bool(item["stable_track"]),
            float(item["stable_pair_fraction"]),
            int(item["occurrences"]),
        ),
        reverse=True,
    )[:10]:
        importance = importance_track_map.get(int(row["track_id"]), {})
        track_table_rows.append(
            (
                str(int(row["track_id"])),
                str(int(row["occurrences"])),
                f"{int(row['dimension_min'])}–{int(row['dimension_max'])}",
                _format(row["stable_pair_fraction"], 2),
                _format(row["projector_distance_median"]),
                _format(row["principal_angle_max_p90"], 1),
                _format(importance.get("critical_occurrence_fraction", float("nan")), 2),
                "yes" if bool(row["stable_track"]) else "no",
                "yes" if bool(importance.get("solver_critical_track", False)) else "no",
            )
        )

    selected_table_rows = [
        (
            f"{float(row['ratio']):g}",
            str(int(row["cluster_id"])),
            str(int(row["dimension"])),
            _format(100.0 * float(row["add_iteration_reduction"]), 1),
            _format(row["remove_iteration_penalty"], 1),
            "yes" if bool(row["solver_critical"]) else "no",
        )
        for row in selected_rows
    ]

    hard_ratios = sorted(
        {
            float(row["target_ratio"])
            for row in transfer_rows
            if row["method"] == "ILU"
            and float(row["iterations_median"]) >= 50.0
        }
    )
    transfer_table_rows = []
    for ratio in hard_ratios:
        local = {
            row["method"]: row
            for row in transfer_rows
            if float(row["target_ratio"]) == ratio
        }
        transfer_table_rows.append(
            (
                f"{ratio:g}",
                _format(local["ILU"]["iterations_median"], 0),
                _format(local["own_full_q8"]["iterations_median"], 0),
                _format(local["nearby_full_q8_reuse"]["iterations_median"], 0),
                _format(local["own_cluster"]["iterations_median"], 0),
                str(int(local["own_cluster"]["cluster_dimension"])),
                _format(local["nearby_cluster_reuse"]["iterations_median"], 0),
                _format(local["nearby_cluster_reuse"]["convergence_rate"], 2),
                _format(local["random_cluster"]["iterations_median"], 0),
            )
        )

    robustness_table_rows = []
    for ratio in settings.robustness_ratios:
        rhs = [
            float(row["projector_distance_normalized"])
            for row in robustness_pair_rows
            if float(row["ratio"]) == ratio
            and row["variation"] == "construction_rhs"
        ]
        node = [
            float(row["projector_distance_normalized"])
            for row in robustness_pair_rows
            if float(row["ratio"]) == ratio
            and row["variation"] == "node_perturbation"
        ]
        robustness_table_rows.append(
            (
                f"{ratio:g}",
                _format(np.median(rhs)),
                _format(max(rhs)),
                _format(np.median(node)),
                _format(max(node)),
            )
        )

    expanded_table_rows = [
        (
            f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}",
            _format(row["q8_projector_distance"]),
            f"{int(row['pair_complete_source_q'])}/{int(row['pair_complete_target_q'])}",
            _format(row["pair_complete_projector_distance"]),
            str(int(row["expanded_dimension"])) if bool(row["expanded_available"]) else "—",
            _format(row.get("expanded_projector_distance", float("nan"))),
            _format(row.get("expanded_principal_angle_max_degrees", float("nan")), 1),
            "yes" if bool(row.get("expanded_stable", False)) else "no",
        )
        for row in expanded_rows
    ]

    sensitivity_table_rows = [
        (
            f"{float(row['ratio']):g}",
            f"{int(row['primary_cluster_count'])}/{int(row['alternate_cluster_count'])}",
            _format(row["boundary_jaccard"], 2),
            f"{int(row['primary_q8_cluster_dimension'])}/{int(row['alternate_q8_cluster_dimension'])}",
        )
        for row in gap_sensitivity_rows
    ]

    pair_complete_distances = [
        float(row["pair_complete_projector_distance"]) for row in expanded_rows
    ]
    q8_distances = [float(row["q8_projector_distance"]) for row in expanded_rows]
    report = rf"""# Stage 11: Ritz-cluster stability and invariant-block transfer

## Executive conclusion

{headline}

The Stage 10 (q=8) prefix did cut a complex Ritz plane at **{int(decision['q8_complex_plane_cut_count'])} of 14 systems**, confirming that some individual-vector instability was a basis artifact. Completing those conjugate planes changed the median neighboring-system projector distance from **{np.median(q8_distances):.3f}** to **{np.median(pair_complete_distances):.3f}**, but it did not make the family stable. Gap-defined low-rank clusters have median neighboring maximum principal angle **{float(decision['low_rank_cluster_max_angle_median_degrees']):.1f}°**, versus **{float(decision['individual_adjacent_angle_median_degrees']):.1f}°** for individual directions, and no track is simultaneously stable and solver-critical.

Candidate cluster reuse reduces hard-regime iterations by a median **{100.0 * float(decision['cluster_reuse_iteration_reduction_median']):.1f}%**, but converges at rate only **{float(decision['cluster_reuse_convergence_rate_mean']):.2f}**. Its p90 projector variability is **{float(decision['rhs_robustness_p90_projector_distance']):.3f}** across construction RHSs and **{float(decision['node_robustness_p90_projector_distance']):.3f}** across node perturbations. Even an optimistic oracle search over conjugacy-preserving blocks of rank at most 16 finds a stable block for only **{int(decision['expanded_stable_pair_count'])}/{int(decision['expanded_pair_count'])}** neighboring pairs.

**Decision:** {'define tracks ' + str(decision['stable_critical_track_ids']) + ' as the new learned-subspace target' if target_exists else 'pause the current Ritz-based learned-subspace route'}. No neural network was trained.

## 1. Fixed protocol and invariant-block definition

The [Stage 10 report](../stage10_ritz_transfer_and_mode_tracking/stage10_ritz_transfer_and_mode_tracking_report.md) is the empirical starting point. This round reuses its N={settings.stage10.n_nodes}, node seed={settings.stage10.node_seed}, stencil={settings.stage10.stencil_size}, jitter={settings.stage10.jitter_fraction}, 30° rotated anisotropy, strong ILU (drop tolerance {settings.stage10.classical_ilu.drop_tolerance:g}, fill factor {settings.stage10.classical_ilu.fill_factor:g}), FGMRES tolerance {settings.stage10.solver.relative_tolerance:.0e}, and held-out RHS seeds. Only R varies in the primary family.

For the right-preconditioned operator (AM^{{-1}}), a non-real Ritz vector and its conjugate are represented by one indivisible two-dimensional real plane. After forming these atomic blocks, with magnitudes ($\rho_k$), the log gap is

\[
g_k=\log_{{10}}(\rho_{{k+1}}/\rho_k).
\]

A cluster boundary is placed when (g_k\geq \operatorname{{median}}(g)+1.5\times1.4826\,\operatorname{{MAD}}(g)). This determines a different cluster count and rank for every system. Sensitivity at multiplier 2.0 is reported rather than hiding boundary instability.

For unequal ranks p and q, projector comparison uses

\[
d_{{\rm proj}}=\frac{{\sqrt{{p+q-2\lVert Z_i^T Z_j\rVert_F^2}}}}{{\sqrt{{p+q}}}},
\]

so a smaller subspace contained inside a larger one is not falsely reported as identical. A neighboring pair is stable only when ranks agree, (d_{{\rm proj}}\leq{settings.stable_projector_distance:.2f}), largest principal angle ≤{settings.stable_max_principal_angle_degrees:.0f}°, and geometric-mean Ritz magnitudes differ by at most a factor {settings.stable_spectral_factor:.1f}.

{_table(
    ['R', 'atoms', 'clusters', 'cluster ranks', 'q8 cuts plane', 'q8 gap cluster [start,stop)/rank', 'selected ID/rank', 'critical'],
    cluster_protocol_rows,
)}

![Ritz spectra and clusters](plots/ritz_spectra_and_gap_clusters.png)

![Cluster dimensions](plots/cluster_dimension_heatmap.png)

## 2. Are clusters more stable than individual vectors?

There are **{int(decision['low_rank_cluster_match_count'])}** neighboring matched pairs with both ranks ≤8. Their median normalized projector distance is **{float(decision['low_rank_cluster_projector_distance_median']):.3f}** and median largest principal angle is **{float(decision['low_rank_cluster_max_angle_median_degrees']):.1f}°**. The individual-direction median adjacent angle from Stage 10 is **{float(decision['individual_adjacent_angle_median_degrees']):.1f}°**. Thus cluster stability is **{'better by the angular median test' if decision['cluster_stability_better_than_individual'] else 'not better by the angular median test'}**, but stability alone is insufficient without persistent rank, solver importance, robustness, and transfer.

Ritz-value proximity is not the sole failure: the median matched-cluster spectral factor is **{float(decision['low_rank_cluster_spectral_factor_median']):.2f}**, and **{100.0 * float(decision['low_rank_cluster_spectral_gate_fraction']):.1f}%** of low-rank matches pass the 1.5 spectral-factor gate, yet their projectors and principal angles remain unstable.

{_table(
    ['track', 'occurrences', 'rank range', 'stable-pair fraction', 'median d_proj', 'p90 max angle', 'critical fraction', 'stable track', 'critical track'],
    track_table_rows,
)}

Stable low-rank tracks: **{int(decision['stable_track_count'])}**. Solver-critical low-rank tracks: **{int(decision['critical_track_count'])}**. Intersection: **{int(decision['stable_critical_track_count'])}**.

![Stability comparison](plots/individual_cluster_enlarged_stability.png)

## 3. Solver importance by adding and removing clusters

Each cluster was tested in two ways on held-out RHSs: add it to ILU as the only coarse block, and remove it from the own q=32 reference. A cluster is solver-critical when adding it reduces iterations by at least 20% with ≥0.90 convergence, or removing it adds at least five iterations or reduces convergence. Nonmonotone outcomes and cap hits remain in the raw CSV.

{_table(
    ['R', 'selected cluster', 'rank', 'add reduction (%)', 'removal penalty', 'critical'],
    selected_table_rows,
)}

Across all systems there are **{int(decision['low_dimensional_critical_occurrences'])}** low-rank critical occurrences, but they do not form a stable cross-R track.

![Cluster solver importance](plots/cluster_solver_importance.png)

## 4. Transfer against Stage 10 controls

“Own full q8” and “nearby full q8 reuse” are the exact Stage 10 held-out results. “Own cluster” is the strongest rank≤8 cluster for the target system; “nearby cluster reuse” uses its matched neighbor block. Random-cluster controls have the reused block's rank.

{_table(
    ['R', 'ILU', 'own q8', 'near q8', 'own cluster', 'rank', 'near cluster', 'cluster conv.', 'random cluster'],
    transfer_table_rows,
)}

On the hard systems, full q8 reuse has median iteration reduction **{100.0 * float(decision['full_q8_reuse_iteration_reduction_median']):.1f}%** and mean convergence **{float(decision['full_q8_reuse_convergence_rate_mean']):.2f}**. Candidate cluster reuse has **{100.0 * float(decision['cluster_reuse_iteration_reduction_median']):.1f}%** and **{float(decision['cluster_reuse_convergence_rate_mean']):.2f}**, so it **{'passes' if decision['cluster_transfer_pass'] else 'fails'}** the solver transfer gate.

![Cluster transfer](plots/cluster_transfer_comparison.png)

![Residual history](plots/cluster_transfer_residual_history.png)

## 5. Robustness to construction RHS and node perturbations

At R∈{{85,90,100,120}}, the strongest low-rank cluster was reconstructed from three Arnoldi construction RHSs. Node seeds {{7,17,27}} were compared after interpolation to the seed-7 nodes. These are target-reconstruction tests; node perturbations intentionally leave the fixed-node primary family.

{_table(['R', 'RHS median d', 'RHS max d', 'node median d', 'node max d'], robustness_table_rows)}

The aggregate p90 distances are **{float(decision['rhs_robustness_p90_projector_distance']):.3f}** for RHS variation and **{float(decision['node_robustness_p90_projector_distance']):.3f}** for node variation, versus the 0.35 gate. Robustness therefore **{'passes' if decision['robustness_pass'] else 'fails'}**.

![Cluster robustness](plots/cluster_robustness.png)

## 6. Completing crossing values and enlarging the block

Because primary clusters remain unstable, each neighboring pair received an optimistic fallback. First, q=8 was extended only when needed to complete a complex plane. Second, all equal-rank, conjugacy-preserving atomic windows containing the q=8 crossing were searched up to rank 16, and the most stable pair was retained. This is an oracle upper bound chosen using both systems—not a deployable predictor.

{_table(
    ['R pair', 'q8 d', 'pair-complete ranks', 'pair-complete d', 'oracle rank', 'oracle d', 'oracle max angle', 'stable'],
    expanded_table_rows,
)}

The oracle enlargement reduces median projector distance from **{float(decision['q8_projector_distance_median_same_pairs']):.3f}** to **{float(decision['expanded_projector_distance_median']):.3f}**, but only **{100.0 * float(decision['expanded_stable_pair_fraction']):.1f}%** of neighboring pairs pass both distance and angle gates.

The oracle-expanded own blocks reduce hard-regime iterations by a median **{100.0 * float(decision['own_expanded_iteration_reduction_median']):.1f}%** with mean convergence **{float(decision['own_expanded_convergence_rate_mean']):.2f}**. Reusing the oracle-expanded neighbor blocks gives **{100.0 * float(decision['reused_expanded_iteration_reduction_median']):.1f}%** with convergence **{float(decision['reused_expanded_convergence_rate_mean']):.2f}**. This confirms solver relevance but not a stable target: the reused convergence rate remains below the 0.90 gate, and the block itself was selected with knowledge of both systems.

![Expanded crossing blocks](plots/expanded_crossing_block_stability.png)

## 7. Spectral-gap sensitivity

The primary 1.5-MAD and stricter 2.0-MAD definitions have median boundary Jaccard agreement **{float(decision['gap_boundary_jaccard_median']):.2f}**. The table reports primary/alternate cluster counts and q8-crossing ranks.

{_table(['R', 'cluster counts', 'boundary Jaccard', 'q8 cluster ranks'], sensitivity_table_rows)}

Boundary sensitivity is part of the target uncertainty; it is not tuned away.

## 8. Decision

| Gate | Requirement | Result | Pass |
| --- | --- | --- | --- |
| Stable and critical block | at least one rank≤8 track | {int(decision['stable_critical_track_count'])} | {'yes' if decision['stable_critical_track_count'] else 'no'} |
| RHS robustness | p90 d≤0.35 | {float(decision['rhs_robustness_p90_projector_distance']):.3f} | {'yes' if decision['rhs_robustness_p90_projector_distance'] <= 0.35 else 'no'} |
| Node robustness | p90 d≤0.35 | {float(decision['node_robustness_p90_projector_distance']):.3f} | {'yes' if decision['node_robustness_p90_projector_distance'] <= 0.35 else 'no'} |
| Neighbor transfer | ≥20% reduction and ≥0.90 convergence | {100.0 * float(decision['cluster_reuse_iteration_reduction_median']):.1f}%, {float(decision['cluster_reuse_convergence_rate_mean']):.2f} | {'yes' if decision['cluster_transfer_pass'] else 'no'} |
| Better than individual vectors | lower median angular instability | {decision['cluster_stability_better_than_individual']} | {'yes' if decision['cluster_stability_better_than_individual'] else 'no'} |

**Main answer:** {'A new learned target is defined by stable critical tracks ' + str(decision['stable_critical_track_ids']) + '.' if target_exists else 'No low-dimensional solver-critical invariant block is substantially stable and transferable across this family. The current Ritz-based learned-subspace route should be paused.'}

## Limitations and reproducibility

- These are approximate Ritz invariant blocks of (AM^{{-1}}), mapped through (M^{{-1}}) into solution coordinates; they are not exact eigenspaces of A.
- Arnoldi dimension remains 64, as in Stage 10. Larger Krylov constructions could change the reference target, but would also increase its classical setup cost.
- Gap clustering is one-dimensional in |Ritz value|. Complex-plane conjugacy is exact, while broader near-degeneracy in the complex plane may require a different, prospectively defined clustering rule.
- The oracle enlarged-block search is deliberately optimistic and must not be interpreted as an online algorithm.
- All cap hits are retained. Solver claims use true residuals and held-out RHSs.

Raw evidence is saved in `ritz_atoms.csv`, `ritz_clusters.csv`, `cluster_neighbor_matches.csv`, `cluster_solver_importance.csv`, `cluster_transfer_results.csv`, `cluster_robustness_pairs.csv`, `expanded_crossing_blocks.csv`, and `gap_sensitivity.csv`. Thresholds and the final gate are in `settings.json` and `decision.json`.
"""
    path = output / "stage11_ritz_cluster_stability_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_stage11(output: Path) -> Path:
    """Run the complete Stage 11 no-ML invariant-block study."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = Stage11Settings()
    np.random.seed(settings.stage10.seed)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    print("[1/9] rebuilding the fixed Stage 10 hard-transition systems", flush=True)
    contexts, _, cluster_rows = build_clustered_contexts(output, settings)
    print("[2/9] checking spectral-gap sensitivity", flush=True)
    gap_sensitivity_rows = run_gap_sensitivity(output, contexts, settings)
    print("[3/9] matching variable-rank clusters across neighboring R", flush=True)
    assignments, pair_rows, track_rows = match_cluster_tracks(
        output, contexts, settings
    )
    print("[4/9] adding and removing every cluster in held-out solves", flush=True)
    importance_rows, _, importance_track_rows = run_cluster_solver_importance(
        output, contexts, assignments, settings
    )
    selected_rows = select_candidate_clusters(
        output, contexts, importance_rows, settings
    )
    print("[5/9] testing RHS and node-perturbation robustness", flush=True)
    _, robustness_pair_rows, _ = run_cluster_robustness(
        output, contexts, selected_rows, settings
    )
    print("[6/9] comparing cluster transfer with Stage 10 controls", flush=True)
    transfer_rows, transfer_histories = run_cluster_transfer(
        output, contexts, selected_rows, pair_rows, settings
    )
    print("[7/9] testing conjugacy completion and enlarged crossing blocks", flush=True)
    expanded_rows, _ = run_enlarged_crossing_blocks(output, contexts, settings)
    print("[8/9] applying the stability/relevance/robustness gates", flush=True)
    decision = build_decision(
        output,
        cluster_rows,
        pair_rows,
        track_rows,
        importance_rows,
        importance_track_rows,
        robustness_pair_rows,
        transfer_rows,
        expanded_rows,
        gap_sensitivity_rows,
        settings,
    )
    print("[9/9] rendering figures and the Markdown report", flush=True)
    make_plots(
        output,
        contexts,
        pair_rows,
        importance_rows,
        robustness_pair_rows,
        transfer_rows,
        transfer_histories,
        expanded_rows,
        settings,
    )
    report = write_report(
        output,
        contexts,
        cluster_rows,
        pair_rows,
        track_rows,
        importance_rows,
        importance_track_rows,
        selected_rows,
        robustness_pair_rows,
        transfer_rows,
        expanded_rows,
        gap_sensitivity_rows,
        decision,
        settings,
    )
    print(f"completed: {report}", flush=True)
    return report
