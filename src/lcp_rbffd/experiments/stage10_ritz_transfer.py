"""Stage 10: Ritz-space transfer and mode tracking for rotated anisotropy.

No learned model is trained.  The fixed RBF-FD discretization and strong ILU
from Stage 9 are held constant while only the anisotropy ratio changes in the
main sweep.  Every candidate space is a nested prefix of Ritz directions from
the right-preconditioned operator ``A M^{-1}``.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment

from ..coarse_space import (
    build_multiplicative_coarse_preconditioner,
    random_orthonormal_basis,
)
from ..config import ILUConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..pde import ConstantAnisotropicDiffusion, ManufacturedSolution, random_smooth_solution
from ..preconditioners import InverseAction, ilu_preconditioner
from ..rbffd import RBFSystem, assemble_system
from ..subspace_analysis import (
    RitzModeResult,
    interpolate_basis_to_probe,
    nested_krylov_ritz_modes,
    subspace_metrics,
)
from .stage9_scalability import Stage9Spec, _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
PURPLE = "#7553A6"
CHARCOAL = "#30343B"
LIGHT_GRAY = "#D7DBE2"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
    }
)


@dataclass(frozen=True)
class Stage10Settings:
    """Fixed numerical controls and decision thresholds."""

    seed: int = 20260901
    n_nodes: int = 5041
    stencil_size: int = 25
    node_seed: int = 7
    jitter_fraction: float = 0.15
    anisotropy_angle_degrees: float = 30.0
    base_ratios: Tuple[float, ...] = (
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
    )
    transition_ratios: Tuple[float, ...] = (
        82.0,
        84.0,
        85.0,
        86.0,
        88.0,
        95.0,
        105.0,
    )
    q_values: Tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32)
    arnoldi_dimension: int = 64
    construction_rhs_seed: int = 20261001
    evaluation_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    solver: SolverConfig = SolverConfig(
        relative_tolerance=1.0e-8,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=300,
        breakdown_tolerance=1.0e-14,
    )
    classical_ilu: ILUConfig = ILUConfig(
        drop_tolerance=1.0e-3,
        fill_factor=5.0,
        diagonal_pivot_threshold=0.0,
        permutation="COLAMD",
    )
    hard_iteration_threshold: int = 50
    stable_direction_cosine: float = 0.85
    structured_distance_correlation: float = 0.70
    within_between_ratio_gate: float = 0.50
    useful_iteration_reduction: float = 0.20
    reuse_oracle_gap_iterations: float = 10.0
    selected_robustness_ratios: Tuple[float, ...] = (80.0, 85.0, 90.0, 100.0, 120.0)
    construction_rhs_replicates: Tuple[int, ...] = (20261001, 20261011, 20261021)
    node_seeds: Tuple[int, ...] = (7, 17, 27)

    @property
    def ratios(self) -> Tuple[float, ...]:
        return tuple(sorted(set(self.base_ratios + self.transition_ratios)))


@dataclass
class RitzContext:
    """One fixed-ratio system, ILU, and its nested q<=32 Ritz directions."""

    ratio: float
    spec: Stage9Spec
    system: RBFSystem
    ilu: InverseAction
    modes: RitzModeResult
    assembly_seconds: float


def _system_spec(
    ratio: float,
    settings: Stage10Settings,
    node_seed: Optional[int] = None,
) -> Stage9Spec:
    return Stage9Spec(
        name=f"anisotropy_R{ratio:g}_seed{settings.node_seed if node_seed is None else node_seed}",
        family="stage10_anisotropy",
        n_nodes=settings.n_nodes,
        stencil_size=settings.stencil_size,
        node_distribution="jittered_grid",
        jitter_fraction=settings.jitter_fraction,
        coefficient_kind="anisotropic",
        coefficient_contrast=ratio,
        anisotropy_angle=settings.anisotropy_angle_degrees,
        seed=settings.node_seed if node_seed is None else node_seed,
    )


def _discrete_rhs(system: RBFSystem, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    exact = random_smooth_solution(seed, 6).values(system.nodes)
    exact[system.boundary] = 0.0
    return np.asarray(system.matrix @ exact, dtype=float), exact


def _build_context(
    ratio: float,
    settings: Stage10Settings,
    construction_seed: Optional[int] = None,
    node_seed: Optional[int] = None,
) -> RitzContext:
    spec = _system_spec(ratio, settings, node_seed=node_seed)
    start = perf_counter()
    system = assemble_system(spec.config(), spec.coefficient())
    assembly_seconds = perf_counter() - start
    ilu = ilu_preconditioner(system.matrix, settings.classical_ilu, name="strong_ilu")
    rhs, _ = _discrete_rhs(
        system,
        settings.construction_rhs_seed if construction_seed is None else construction_seed,
    )
    modes = nested_krylov_ritz_modes(
        system.matrix,
        ilu,
        max_modes=max(settings.q_values),
        arnoldi_dimension=settings.arnoldi_dimension,
        seed=settings.seed + int(10 * ratio),
        initial_vector=rhs,
    )
    return RitzContext(ratio, spec, system, ilu, modes, assembly_seconds)


def _run_solver(
    context: RitzContext,
    rhs_seed: int,
    preconditioner: InverseAction,
    settings: Stage10Settings,
) -> KrylovResult:
    rhs, _ = _discrete_rhs(context.system, rhs_seed)
    return restarted_gmres(
        context.system.matrix,
        rhs,
        settings.solver,
        preconditioner,
    )


def _coarse_action(
    context: RitzContext,
    basis: np.ndarray,
    settings: Stage10Settings,
    name: str,
) -> Tuple[InverseAction, Dict[str, Any]]:
    return build_multiplicative_coarse_preconditioner(
        context.system.matrix,
        context.ilu,
        basis,
        name=name,
    )


def build_main_contexts(
    output: Path,
    settings: Stage10Settings,
) -> Tuple[Dict[float, RitzContext], List[Dict[str, Any]]]:
    """Assemble every fixed-node ratio and save its q=32 Ritz reference."""

    contexts: Dict[float, RitzContext] = {}
    rows: List[Dict[str, Any]] = []
    subspace_dir = output / "subspaces"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    manufactured = ManufacturedSolution()
    for ratio in settings.ratios:
        context = _build_context(ratio, settings)
        contexts[ratio] = context
        coefficient = ConstantAnisotropicDiffusion(
            ratio, settings.anisotropy_angle_degrees
        )
        rhs = manufactured.anisotropic_right_hand_side(
            context.system.nodes, context.system.boundary, coefficient
        )
        direct_start = perf_counter()
        numerical = spla.spsolve(context.system.matrix, rhs)
        direct_seconds = perf_counter() - direct_start
        exact = manufactured.values(context.system.nodes)
        error = float(
            np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), 1.0e-300)
        )
        row = {
            "ratio": ratio,
            "n_nodes": settings.n_nodes,
            "node_seed": settings.node_seed,
            "stencil_size": settings.stencil_size,
            "jitter_fraction": settings.jitter_fraction,
            "anisotropy_angle_degrees": settings.anisotropy_angle_degrees,
            "nnz": int(context.system.matrix.nnz),
            "symmetry_measure": context.system.diagnostics["symmetry_measure"],
            "assembly_seconds": context.assembly_seconds,
            "ilu_setup_seconds": context.ilu.setup_seconds,
            "ilu_factor_fill_ratio": context.ilu.diagnostics["factor_fill_ratio"],
            "ilu_factor_memory_mib": context.ilu.diagnostics["factor_memory_bytes"] / 2.0**20,
            "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
            "arnoldi_dimension": context.modes.diagnostics["arnoldi_dimension"],
            "basis_orthogonality_error": context.modes.diagnostics["basis_orthogonality_error"],
            "relative_discretization_error": error,
            "reference_direct_solve_seconds": direct_seconds,
        }
        rows.append(row)
        np.savez_compressed(
            subspace_dir / f"ritz_R{ratio:g}_q32.npz",
            basis=context.modes.basis,
            eigenvalues=context.modes.eigenvalues,
            mode_labels=np.asarray(context.modes.mode_labels),
            ratio=ratio,
            construction_rhs_seed=settings.construction_rhs_seed,
            target_definition="Ritz modes of A M^-1 mapped through M^-1",
        )
    _write_csv(output / "system_metadata.csv", rows)
    return contexts, rows


def run_q_sweep(
    output: Path,
    contexts: Mapping[float, RitzContext],
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Record held-out FGMRES convergence for q=0 and every requested prefix."""

    solve_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for ratio, context in contexts.items():
        for q in (0,) + settings.q_values:
            if q == 0:
                action = context.ilu
                coarse_setup = 0.0
                reduced_condition = float("nan")
                method = "ILU"
            else:
                action, diagnostic = _coarse_action(
                    context,
                    context.modes.basis[:, :q],
                    settings,
                    name=f"own_ritz_q{q}",
                )
                coarse_setup = diagnostic["coarse_setup_seconds"]
                reduced_condition = diagnostic["reduced_condition"]
                method = f"own_ritz_q{q}"
            local = []
            for rhs_seed in settings.evaluation_rhs_seeds:
                result = _run_solver(context, rhs_seed, action, settings)
                local.append(result)
                solve_rows.append(
                    {
                        "ratio": ratio,
                        "q": q,
                        "method": method,
                        "rhs_seed": rhs_seed,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "final_relative_residual": float(result.relative_residuals[-1]),
                        "solve_seconds": result.solve_seconds,
                        "preconditioner_seconds": result.preconditioner_seconds,
                        "coarse_setup_seconds": coarse_setup,
                        "reduced_condition": reduced_condition,
                        "failure": result.failure,
                    }
                )
                for iteration, residual in enumerate(result.relative_residuals):
                    histories.append(
                        {
                            "ratio": ratio,
                            "q": q,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "relative_residual": float(residual),
                        }
                    )
            summaries.append(
                {
                    "ratio": ratio,
                    "q": q,
                    "iterations_median": float(np.median([r.iterations for r in local])),
                    "iterations_max": int(max(r.iterations for r in local)),
                    "convergence_rate": float(np.mean([r.converged for r in local])),
                    "solve_seconds_median": float(np.median([r.solve_seconds for r in local])),
                    "coarse_setup_seconds": coarse_setup,
                    "reduced_condition": reduced_condition,
                }
            )

    elbows: List[Dict[str, Any]] = []
    q80_values = []
    for ratio in contexts:
        rows = sorted(
            [row for row in summaries if row["ratio"] == ratio],
            key=lambda row: int(row["q"]),
        )
        baseline = next(row for row in rows if row["q"] == 0)
        q32 = next(row for row in rows if row["q"] == max(settings.q_values))
        ordered = [next(row for row in rows if row["q"] == q) for q in (0,) + settings.q_values]
        drops = [
            ordered[index - 1]["iterations_median"] - ordered[index]["iterations_median"]
            for index in range(1, len(ordered))
        ]
        major_index = int(np.argmax(drops)) + 1
        full_gain = baseline["iterations_median"] - q32["iterations_median"]
        q80 = max(settings.q_values)
        if full_gain > 0.0:
            for row in ordered[1:]:
                gain = baseline["iterations_median"] - row["iterations_median"]
                if gain >= 0.8 * full_gain:
                    q80 = int(row["q"])
                    break
        if baseline["iterations_median"] >= settings.hard_iteration_threshold:
            q80_values.append(q80)
        elbows.append(
            {
                "ratio": ratio,
                "baseline_iterations": baseline["iterations_median"],
                "q32_iterations": q32["iterations_median"],
                "full_q32_gain": full_gain,
                "major_drop_from_q": int(ordered[major_index - 1]["q"]),
                "major_drop_to_q": int(ordered[major_index]["q"]),
                "major_iteration_drop": drops[major_index - 1],
                "q_for_80pct_full_gain": q80,
            }
        )
    candidates = np.asarray(settings.q_values)
    target = float(np.median(q80_values)) if q80_values else 24.0
    q_track = int(candidates[np.argmin(np.abs(candidates - target))])
    q_track = max(q_track, 4)
    _write_csv(output / "q_sweep_results.csv", solve_rows)
    _write_csv(output / "q_sweep_summary.csv", summaries)
    _write_csv(output / "q_sweep_histories.csv", histories)
    _write_csv(output / "q_elbows.csv", elbows)
    _write_json(
        output / "tracking_dimension.json",
        {
            "q_track": q_track,
            "rule": "nearest requested q to median q achieving 80% of q32 gain among systems with baseline median >=50 iterations",
            "hard_q80_values": q80_values,
        },
    )
    return solve_rows, summaries, elbows, q_track


def _match_modes(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-overlap one-to-one matching with sign-invariant cosines."""

    overlap = np.abs(source.T @ target)
    source_indices, target_indices = linear_sum_assignment(-overlap)
    order = np.argsort(source_indices)
    source_indices = source_indices[order]
    target_indices = target_indices[order]
    cosines = overlap[source_indices, target_indices]
    return source_indices, target_indices, cosines


def run_cross_system_analysis(
    output: Path,
    contexts: Mapping[float, RitzContext],
    q_track: int,
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute subspace distances for every q and track individual q_track modes."""

    ratios = list(contexts)
    pair_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    matrix_dir = output / "distance_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for q in settings.q_values:
        distances = np.zeros((len(ratios), len(ratios)), dtype=float)
        angle_means = np.zeros_like(distances)
        parameter_distances = []
        projector_distances = []
        for i, first_ratio in enumerate(ratios):
            first = contexts[first_ratio].modes.basis[:, :q]
            for j, second_ratio in enumerate(ratios):
                second = contexts[second_ratio].modes.basis[:, :q]
                metric = subspace_metrics(first, second)
                distances[i, j] = metric["projector_distance_normalized"]
                angle_means[i, j] = metric["principal_angle_mean_degrees"]
                if i < j:
                    delta = abs(first_ratio - second_ratio)
                    parameter_distances.append(delta)
                    projector_distances.append(
                        metric["projector_distance_normalized"]
                    )
                    pair_rows.append(
                        {
                            "q": q,
                            "ratio_i": first_ratio,
                            "ratio_j": second_ratio,
                            "absolute_parameter_distance": delta,
                            "projector_distance": metric["projector_distance"],
                            "normalized_projector_distance": metric[
                                "projector_distance_normalized"
                            ],
                            "principal_angle_mean_degrees": metric[
                                "principal_angle_mean_degrees"
                            ],
                            "principal_angle_max_degrees": metric[
                                "principal_angle_max_degrees"
                            ],
                        }
                    )
        if len(parameter_distances) >= 2:
            spearman = stats.spearmanr(parameter_distances, projector_distances)
            pearson = stats.pearsonr(parameter_distances, projector_distances)
            spearman_statistic = float(spearman.statistic)
            spearman_pvalue = float(spearman.pvalue)
            pearson_statistic = float(pearson.statistic)
            pearson_pvalue = float(pearson.pvalue)
        else:
            spearman_statistic = spearman_pvalue = float("nan")
            pearson_statistic = pearson_pvalue = float("nan")
        correlation_rows.append(
            {
                "q": q,
                "pairs": len(parameter_distances),
                "spearman_correlation": spearman_statistic,
                "spearman_pvalue": spearman_pvalue,
                "pearson_correlation": pearson_statistic,
                "pearson_pvalue": pearson_pvalue,
                "median_projector_distance": float(np.median(projector_distances)),
            }
        )
        np.savez_compressed(
            matrix_dir / f"distances_q{q}.npz",
            ratios=np.asarray(ratios),
            normalized_projector_distance=distances,
            principal_angle_mean_degrees=angle_means,
        )

    adjacent_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = [
        {
            "ratio": ratios[0],
            "local_mode_index": mode_index,
            "track_id": mode_index,
        }
        for mode_index in range(q_track)
    ]
    track_ids = np.arange(q_track, dtype=int)
    current_order = np.arange(q_track, dtype=int)
    for pair_index in range(len(ratios) - 1):
        source_ratio = ratios[pair_index]
        target_ratio = ratios[pair_index + 1]
        source_basis = contexts[source_ratio].modes.basis[:, :q_track]
        target_basis = contexts[target_ratio].modes.basis[:, :q_track]
        source_indices, target_indices, cosines = _match_modes(
            source_basis, target_basis
        )
        next_order = np.empty(q_track, dtype=int)
        for source_index, target_index, cosine in zip(
            source_indices, target_indices, cosines
        ):
            track_id = int(current_order[source_index])
            next_order[target_index] = track_id
            source_value = contexts[source_ratio].modes.eigenvalues[source_index]
            target_value = contexts[target_ratio].modes.eigenvalues[target_index]
            adjacent_rows.append(
                {
                    "track_id": track_id,
                    "source_ratio": source_ratio,
                    "target_ratio": target_ratio,
                    "source_mode_index": int(source_index),
                    "target_mode_index": int(target_index),
                    "absolute_cosine": float(cosine),
                    "source_ritz_magnitude": float(abs(source_value)),
                    "target_ritz_magnitude": float(abs(target_value)),
                    "parameter_step": target_ratio - source_ratio,
                }
            )
        current_order = next_order
        assignment_rows.extend(
            {
                "ratio": target_ratio,
                "local_mode_index": local_mode_index,
                "track_id": int(track_id),
            }
            for local_mode_index, track_id in enumerate(current_order)
        )

    track_summaries: List[Dict[str, Any]] = []
    for track_id in track_ids:
        rows = [row for row in adjacent_rows if row["track_id"] == track_id]
        cosines = np.asarray([row["absolute_cosine"] for row in rows])
        track_summaries.append(
            {
                "track_id": int(track_id),
                "adjacent_steps": len(rows),
                "cosine_median": float(np.median(cosines)),
                "cosine_p10": float(np.quantile(cosines, 0.1)),
                "cosine_min": float(np.min(cosines)),
                "globally_stable": bool(
                    np.quantile(cosines, 0.1)
                    >= settings.stable_direction_cosine
                ),
            }
        )
    _write_csv(output / "subspace_pairs.csv", pair_rows)
    _write_csv(output / "subspace_correlations.csv", correlation_rows)
    _write_csv(output / "mode_tracking_adjacent.csv", adjacent_rows)
    _write_csv(output / "mode_track_assignments.csv", assignment_rows)
    _write_csv(output / "mode_tracking_summary.csv", track_summaries)
    return pair_rows, correlation_rows, track_summaries


def _method_summary(
    ratio: float,
    method: str,
    source_ratio: Optional[float],
    results: Sequence[KrylovResult],
    setup_seconds: float,
    q: int,
) -> Dict[str, Any]:
    return {
        "target_ratio": ratio,
        "method": method,
        "source_ratio": "" if source_ratio is None else source_ratio,
        "q": q,
        "iterations_median": float(np.median([result.iterations for result in results])),
        "iterations_max": int(max(result.iterations for result in results)),
        "convergence_rate": float(np.mean([result.converged for result in results])),
        "solve_seconds_median": float(
            np.median([result.solve_seconds for result in results])
        ),
        "final_relative_residual_max": float(
            max(result.relative_residuals[-1] for result in results)
        ),
        "setup_seconds": setup_seconds,
        "failure": "; ".join(
            sorted({result.failure for result in results if result.failure})
        ),
    }


def run_transfer_controls(
    output: Path,
    contexts: Mapping[float, RitzContext],
    q_track: int,
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compare own, nearby, far, and random q_track spaces on held-out RHSs."""

    ratios = list(contexts)
    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    for target_index, target_ratio in enumerate(ratios):
        context = contexts[target_ratio]
        other = [ratio for ratio in ratios if ratio != target_ratio]
        nearby_ratio = min(
            other,
            key=lambda ratio: (abs(ratio - target_ratio), ratio > target_ratio, ratio),
        )
        far_ratio = max(other, key=lambda ratio: abs(ratio - target_ratio))
        method_specs: List[Tuple[str, Optional[float], Optional[np.ndarray]]] = [
            ("ILU", None, None),
            ("own_oracle", target_ratio, context.modes.basis[:, :q_track]),
            (
                "nearby_reuse",
                nearby_ratio,
                contexts[nearby_ratio].modes.basis[:, :q_track],
            ),
            (
                "far_reuse",
                far_ratio,
                contexts[far_ratio].modes.basis[:, :q_track],
            ),
            (
                "random_control",
                None,
                random_orthonormal_basis(
                    settings.n_nodes,
                    q_track,
                    settings.seed + 1000 + target_index,
                ),
            ),
        ]
        for method, source_ratio, basis in method_specs:
            if basis is None:
                action = context.ilu
                setup_seconds = action.setup_seconds
            else:
                action, diagnostic = _coarse_action(
                    context, basis, settings, name=method
                )
                setup_seconds = diagnostic["coarse_setup_seconds"]
            results = []
            for rhs_seed in settings.evaluation_rhs_seeds:
                result = _run_solver(context, rhs_seed, action, settings)
                results.append(result)
                for iteration, residual in enumerate(result.relative_residuals):
                    histories.append(
                        {
                            "target_ratio": target_ratio,
                            "method": method,
                            "source_ratio": "" if source_ratio is None else source_ratio,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "relative_residual": float(residual),
                        }
                    )
            rows.append(
                _method_summary(
                    target_ratio,
                    method,
                    source_ratio,
                    results,
                    setup_seconds,
                    q_track,
                )
            )

    for target_ratio in ratios:
        target_rows = [row for row in rows if row["target_ratio"] == target_ratio]
        baseline = next(row for row in target_rows if row["method"] == "ILU")
        own = next(row for row in target_rows if row["method"] == "own_oracle")
        for row in target_rows:
            row["iteration_reduction_vs_ilu"] = (
                baseline["iterations_median"] - row["iterations_median"]
            ) / max(baseline["iterations_median"], 1.0)
            row["iteration_gap_vs_own"] = (
                row["iterations_median"] - own["iterations_median"]
            )
    nearby = [row for row in rows if row["method"] == "nearby_reuse"]
    hard_nearby = [
        row
        for row in nearby
        if next(
            baseline
            for baseline in rows
            if baseline["target_ratio"] == row["target_ratio"]
            and baseline["method"] == "ILU"
        )["iterations_median"]
        >= settings.hard_iteration_threshold
    ]
    full_reuse_failed = bool(
        not hard_nearby
        or np.median(
            [row["iteration_reduction_vs_ilu"] for row in hard_nearby]
        )
        < settings.useful_iteration_reduction
        or np.mean([row["convergence_rate"] for row in hard_nearby]) < 0.9
        or np.median([row["iteration_gap_vs_own"] for row in hard_nearby])
        > settings.reuse_oracle_gap_iterations
    )
    gate = {
        "q_track": q_track,
        "hard_targets": len(hard_nearby),
        "nearby_iteration_reduction_median": (
            float(
                np.median(
                    [row["iteration_reduction_vs_ilu"] for row in hard_nearby]
                )
            )
            if hard_nearby
            else float("nan")
        ),
        "nearby_iteration_gap_vs_own_median": (
            float(np.median([row["iteration_gap_vs_own"] for row in hard_nearby]))
            if hard_nearby
            else float("nan")
        ),
        "nearby_convergence_rate_mean": (
            float(np.mean([row["convergence_rate"] for row in hard_nearby]))
            if hard_nearby
            else 0.0
        ),
        "full_reuse_failed": full_reuse_failed,
    }
    _write_csv(output / "transfer_results.csv", rows)
    _write_csv(output / "transfer_histories.csv", histories)
    _write_json(output / "full_reuse_gate.json", gate)
    return rows, histories, gate


def _orthonormalize_columns(columns: Sequence[np.ndarray], q: int) -> np.ndarray:
    """Stable two-pass Gram-Schmidt without silently changing dimension."""

    accepted: List[np.ndarray] = []
    for column in columns:
        work = np.asarray(column, dtype=float).copy()
        for _ in range(2):
            for previous in accepted:
                work -= previous * np.dot(previous, work)
        norm = float(np.linalg.norm(work))
        if norm > 1.0e-10:
            accepted.append(work / norm)
        if len(accepted) == q:
            break
    if len(accepted) != q:
        raise RuntimeError(
            f"mixed partial-reuse basis has rank {len(accepted)}, expected {q}"
        )
    return np.column_stack(accepted)


def run_partial_reuse(
    output: Path,
    contexts: Mapping[float, RitzContext],
    transfer_rows: Sequence[Mapping[str, Any]],
    full_reuse_gate: Mapping[str, Any],
    q_track: int,
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reuse matched stable directions and recompute only sensitive directions."""

    if not bool(full_reuse_gate["full_reuse_failed"]):
        gate = {
            "performed": False,
            "reason": "Full nearby reuse passed the preregistered gate.",
        }
        _write_csv(output / "partial_reuse_results.csv", [])
        _write_json(output / "partial_reuse_gate.json", gate)
        return [], gate

    ratios = list(contexts)
    rows: List[Dict[str, Any]] = []
    for target_ratio in ratios:
        context = contexts[target_ratio]
        nearby_row = next(
            row
            for row in transfer_rows
            if row["target_ratio"] == target_ratio
            and row["method"] == "nearby_reuse"
        )
        source_ratio = float(nearby_row["source_ratio"])
        source_basis = contexts[source_ratio].modes.basis[:, :q_track]
        target_basis = context.modes.basis[:, :q_track]
        source_indices, target_indices, cosines = _match_modes(
            source_basis, target_basis
        )
        stable_pairs = [
            (int(source), int(target), float(cosine))
            for source, target, cosine in zip(
                source_indices, target_indices, cosines
            )
            if cosine >= settings.stable_direction_cosine
        ]
        sensitive_targets = [
            int(target)
            for target, cosine in zip(target_indices, cosines)
            if cosine < settings.stable_direction_cosine
        ]
        mixed_columns = [source_basis[:, source] for source, _, _ in stable_pairs]
        mixed_columns.extend(target_basis[:, target] for target in sensitive_targets)
        mixed_basis = _orthonormalize_columns(mixed_columns, q_track)
        stable_only_basis = (
            _orthonormalize_columns(
                [source_basis[:, source] for source, _, _ in stable_pairs],
                len(stable_pairs),
            )
            if stable_pairs
            else None
        )
        method_specs: List[Tuple[str, np.ndarray]] = [("partial_update", mixed_basis)]
        if stable_only_basis is not None:
            method_specs.append(("stable_reuse_only", stable_only_basis))
        for method, basis in method_specs:
            action, diagnostic = _coarse_action(
                context, basis, settings, name=method
            )
            results = [
                _run_solver(context, rhs_seed, action, settings)
                for rhs_seed in settings.evaluation_rhs_seeds
            ]
            row = _method_summary(
                target_ratio,
                method,
                source_ratio,
                results,
                diagnostic["coarse_setup_seconds"],
                basis.shape[1],
            )
            row.update(
                {
                    "stable_directions_reused": len(stable_pairs),
                    "sensitive_directions_recomputed": len(sensitive_targets),
                    "stable_fraction": len(stable_pairs) / q_track,
                    "matched_cosine_median": float(np.median(cosines)),
                    "matched_cosine_min": float(np.min(cosines)),
                }
            )
            baseline = next(
                item
                for item in transfer_rows
                if item["target_ratio"] == target_ratio
                and item["method"] == "ILU"
            )
            own = next(
                item
                for item in transfer_rows
                if item["target_ratio"] == target_ratio
                and item["method"] == "own_oracle"
            )
            row["iteration_reduction_vs_ilu"] = (
                float(baseline["iterations_median"])
                - row["iterations_median"]
            ) / max(float(baseline["iterations_median"]), 1.0)
            row["iteration_gap_vs_own"] = (
                row["iterations_median"] - float(own["iterations_median"])
            )
            rows.append(row)

    hard_rows = [
        row
        for row in rows
        if row["method"] == "partial_update"
        and next(
            item
            for item in transfer_rows
            if item["target_ratio"] == row["target_ratio"]
            and item["method"] == "ILU"
        )["iterations_median"]
        >= settings.hard_iteration_threshold
    ]
    gate = {
        "performed": True,
        "stable_direction_cosine_threshold": settings.stable_direction_cosine,
        "hard_targets": len(hard_rows),
        "stable_fraction_median": (
            float(np.median([row["stable_fraction"] for row in hard_rows]))
            if hard_rows
            else float("nan")
        ),
        "iteration_reduction_median": (
            float(np.median([row["iteration_reduction_vs_ilu"] for row in hard_rows]))
            if hard_rows
            else float("nan")
        ),
        "iteration_gap_vs_own_median": (
            float(np.median([row["iteration_gap_vs_own"] for row in hard_rows]))
            if hard_rows
            else float("nan")
        ),
        "convergence_rate_mean": (
            float(np.mean([row["convergence_rate"] for row in hard_rows]))
            if hard_rows
            else 0.0
        ),
    }
    gate["partial_reuse_useful"] = bool(
        gate["iteration_reduction_median"]
        >= settings.useful_iteration_reduction
        and gate["iteration_gap_vs_own_median"]
        <= settings.reuse_oracle_gap_iterations
        and gate["convergence_rate_mean"] >= 0.9
    )
    _write_csv(output / "partial_reuse_results.csv", rows)
    _write_json(output / "partial_reuse_gate.json", gate)
    return rows, gate


def run_mode_ablation(
    output: Path,
    contexts: Mapping[float, RitzContext],
    q_track: int,
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Identify solver-critical individual directions by leave-one-out tests."""

    selected = [
        ratio
        for ratio in (85.0, 90.0, 100.0, 120.0)
        if ratio in contexts
    ]
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    ratios = list(contexts)
    track_maps: Dict[float, np.ndarray] = {
        ratios[0]: np.arange(q_track, dtype=int)
    }
    current_order = track_maps[ratios[0]]
    for source_ratio, target_ratio in zip(ratios[:-1], ratios[1:]):
        _, target_indices, _ = _match_modes(
            contexts[source_ratio].modes.basis[:, :q_track],
            contexts[target_ratio].modes.basis[:, :q_track],
        )
        next_order = np.empty(q_track, dtype=int)
        for source_index, target_index in enumerate(target_indices):
            next_order[target_index] = current_order[source_index]
        current_order = next_order
        track_maps[target_ratio] = current_order.copy()
    rhs_seeds = settings.evaluation_rhs_seeds[:2]
    for ratio in selected:
        context = contexts[ratio]
        full_action, _ = _coarse_action(
            context, context.modes.basis[:, :q_track], settings, "full_reference"
        )
        full_results = [
            _run_solver(context, rhs_seed, full_action, settings)
            for rhs_seed in rhs_seeds
        ]
        full_iterations = float(np.median([result.iterations for result in full_results]))
        for mode_index in range(q_track):
            basis = np.delete(
                context.modes.basis[:, :q_track], mode_index, axis=1
            )
            action, _ = _coarse_action(
                context, basis, settings, f"without_mode{mode_index}"
            )
            results = [
                _run_solver(context, rhs_seed, action, settings)
                for rhs_seed in rhs_seeds
            ]
            without = float(np.median([result.iterations for result in results]))
            rows.append(
                {
                    "ratio": ratio,
                    "mode_index": mode_index,
                    "track_id": int(track_maps[ratio][mode_index]),
                    "ritz_magnitude": float(abs(context.modes.eigenvalues[mode_index])),
                    "full_iterations_median": full_iterations,
                    "without_mode_iterations_median": without,
                    "iteration_penalty": without - full_iterations,
                    "without_mode_convergence_rate": float(
                        np.mean([result.converged for result in results])
                    ),
                }
            )
        local = [row for row in rows if row["ratio"] == ratio]
        critical = [row for row in local if row["iteration_penalty"] >= 5.0]
        summaries.append(
            {
                "ratio": ratio,
                "q_track": q_track,
                "full_iterations_median": full_iterations,
                "critical_mode_count": len(critical),
                "critical_mode_indices": ";".join(
                    str(row["mode_index"]) for row in critical
                ),
                "max_iteration_penalty": float(
                    max(row["iteration_penalty"] for row in local)
                ),
            }
        )
    _write_csv(output / "mode_ablation.csv", rows)
    _write_csv(output / "mode_ablation_summary.csv", summaries)
    return rows, summaries


def run_robustness(
    output: Path,
    base_contexts: Mapping[float, RitzContext],
    q_track: int,
    settings: Stage10Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Repeat selected ratios across construction RHSs and node perturbations."""

    variability: List[Dict[str, Any]] = []
    performance: List[Dict[str, Any]] = []
    for ratio in settings.selected_robustness_ratios:
        if ratio not in base_contexts:
            continue
        base = base_contexts[ratio]
        rhs_contexts = [
            _build_context(
                ratio,
                settings,
                construction_seed=construction_seed,
                node_seed=settings.node_seed,
            )
            for construction_seed in settings.construction_rhs_replicates
        ]
        for i in range(len(rhs_contexts)):
            for j in range(i + 1, len(rhs_contexts)):
                metric = subspace_metrics(
                    rhs_contexts[i].modes.basis[:, :q_track],
                    rhs_contexts[j].modes.basis[:, :q_track],
                )
                variability.append(
                    {
                        "ratio": ratio,
                        "variation": "construction_rhs",
                        "replicate_i": settings.construction_rhs_replicates[i],
                        "replicate_j": settings.construction_rhs_replicates[j],
                        **{
                            key: value
                            for key, value in metric.items()
                            if key != "principal_angles_degrees"
                        },
                    }
                )
        node_contexts = [
            _build_context(
                ratio,
                settings,
                construction_seed=settings.construction_rhs_seed,
                node_seed=node_seed,
            )
            for node_seed in settings.node_seeds
        ]
        interpolated = [
            interpolate_basis_to_probe(
                context.system.nodes,
                context.modes.basis[:, :q_track],
                base.system.nodes,
            )
            for context in node_contexts
        ]
        for i in range(len(node_contexts)):
            for j in range(i + 1, len(node_contexts)):
                metric = subspace_metrics(interpolated[i], interpolated[j])
                variability.append(
                    {
                        "ratio": ratio,
                        "variation": "node_perturbation_interpolated",
                        "replicate_i": settings.node_seeds[i],
                        "replicate_j": settings.node_seeds[j],
                        **{
                            key: value
                            for key, value in metric.items()
                            if key != "principal_angles_degrees"
                        },
                    }
                )
        for variation, contexts in (
            ("construction_rhs", rhs_contexts),
            ("node_perturbation", node_contexts),
        ):
            for replicate, context in enumerate(contexts):
                action, diagnostic = _coarse_action(
                    context,
                    context.modes.basis[:, :q_track],
                    settings,
                    name=f"robust_{variation}",
                )
                results = [
                    _run_solver(context, rhs_seed, action, settings)
                    for rhs_seed in settings.evaluation_rhs_seeds[:2]
                ]
                performance.append(
                    {
                        "ratio": ratio,
                        "variation": variation,
                        "replicate": replicate,
                        "iterations_median": float(
                            np.median([result.iterations for result in results])
                        ),
                        "convergence_rate": float(
                            np.mean([result.converged for result in results])
                        ),
                        "solve_seconds_median": float(
                            np.median([result.solve_seconds for result in results])
                        ),
                        "coarse_setup_seconds": diagnostic[
                            "coarse_setup_seconds"
                        ],
                    }
                )

    rhs_distances = [
        row["projector_distance_normalized"]
        for row in variability
        if row["variation"] == "construction_rhs"
    ]
    node_distances = [
        row["projector_distance_normalized"]
        for row in variability
        if row["variation"] == "node_perturbation_interpolated"
    ]
    summary = {
        "q_track": q_track,
        "rhs_variability_median": float(np.median(rhs_distances)),
        "rhs_variability_p90": float(np.quantile(rhs_distances, 0.9)),
        "node_variability_median": float(np.median(node_distances)),
        "node_variability_p90": float(np.quantile(node_distances, 0.9)),
        "oracle_convergence_rate": float(
            np.mean([row["convergence_rate"] for row in performance])
        ),
    }
    _write_csv(output / "robustness_variability.csv", variability)
    _write_csv(output / "robustness_performance.csv", performance)
    _write_json(output / "robustness_summary.json", summary)
    return variability, performance, summary


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    q_summaries: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    transfer_histories: Sequence[Mapping[str, Any]],
    partial_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    q_track: int,
) -> List[str]:
    """Create the compact figure set cited by the Markdown report."""

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    ratios = sorted({float(row["ratio"]) for row in q_summaries})
    qs = sorted({int(row["q"]) for row in q_summaries})

    baseline = [
        next(
            row
            for row in q_summaries
            if float(row["ratio"]) == ratio and int(row["q"]) == 0
        )
        for ratio in ratios
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    axis.plot(
        ratios,
        [float(row["iterations_median"]) for row in baseline],
        marker="o",
        color=BLUE,
        linewidth=2.2,
    )
    failed_x = [
        ratio
        for ratio, row in zip(ratios, baseline)
        if float(row["convergence_rate"]) < 1.0
    ]
    failed_y = [
        float(row["iterations_median"])
        for row in baseline
        if float(row["convergence_rate"]) < 1.0
    ]
    if failed_x:
        axis.scatter(failed_x, failed_y, marker="x", s=70, color=PINK, label="hit cap in ≥1 RHS")
        axis.legend(frameon=False)
    axis.axhline(50, color=LIGHT_GRAY, linewidth=1.2, linestyle="--")
    axis.set(xlabel="anisotropy ratio R", ylabel="median FGMRES iterations")
    axis.set_title("Strong-ILU transition under one fixed RBF-FD discretization", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.65)
    name = "ilu_transition.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    iteration_matrix = np.full((len(ratios), len(qs)), np.nan)
    for row in q_summaries:
        i = ratios.index(float(row["ratio"]))
        j = qs.index(int(row["q"]))
        iteration_matrix[i, j] = float(row["iterations_median"])
    figure, axis = plt.subplots(figsize=(10.8, 7.5))
    image_handle = axis.imshow(iteration_matrix, aspect="auto", cmap="magma_r")
    axis.set_xticks(np.arange(len(qs)), ["ILU" if q == 0 else str(q) for q in qs])
    axis.set_yticks(np.arange(len(ratios)), [f"{ratio:g}" for ratio in ratios])
    axis.set(xlabel="Ritz dimension q", ylabel="anisotropy ratio R")
    axis.set_title("Held-out FGMRES iterations: own-system nested Ritz spaces", loc="left")
    for i in range(len(ratios)):
        for j in range(len(qs)):
            value = iteration_matrix[i, j]
            text_color = "white" if value > np.nanmedian(iteration_matrix) else CHARCOAL
            axis.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color=text_color)
    figure.colorbar(image_handle, ax=axis, label="median iterations")
    name = "q_sweep_iteration_heatmap.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    selected_ratios = [ratio for ratio in (80.0, 85.0, 90.0, 100.0, 120.0, 140.0) if ratio in ratios]
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    palette = plt.cm.viridis(np.linspace(0.05, 0.9, len(selected_ratios)))
    for color, ratio in zip(palette, selected_ratios):
        local = sorted(
            [row for row in q_summaries if float(row["ratio"]) == ratio],
            key=lambda row: int(row["q"]),
        )
        axis.plot(
            [int(row["q"]) for row in local],
            [float(row["iterations_median"]) for row in local],
            marker="o",
            label=f"R={ratio:g}",
            color=color,
        )
    axis.set(xlabel="Ritz dimension q (0 = ILU only)", ylabel="median FGMRES iterations")
    axis.set_title("Where the solver-level iteration drop occurs", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.6)
    axis.legend(frameon=False, ncol=3)
    name = "q_sweep_selected_curves.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    q_pairs = [row for row in pair_rows if int(row["q"]) == q_track]
    distance_matrix = np.zeros((len(ratios), len(ratios)))
    angle_matrix = np.zeros_like(distance_matrix)
    for row in q_pairs:
        i = ratios.index(float(row["ratio_i"]))
        j = ratios.index(float(row["ratio_j"]))
        distance_matrix[i, j] = distance_matrix[j, i] = float(
            row["normalized_projector_distance"]
        )
        angle_matrix[i, j] = angle_matrix[j, i] = float(
            row["principal_angle_mean_degrees"]
        )
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 6.2))
    for axis, matrix, title, label in (
        (axes[0], distance_matrix, "Normalized projector distance", "distance"),
        (axes[1], angle_matrix, "Mean principal angle", "degrees"),
    ):
        handle = axis.imshow(matrix, vmin=0.0, aspect="equal", cmap="viridis")
        axis.set_xticks(np.arange(len(ratios)), [f"{ratio:g}" for ratio in ratios], rotation=60)
        axis.set_yticks(np.arange(len(ratios)), [f"{ratio:g}" for ratio in ratios])
        axis.set(xlabel="R_j", ylabel="R_i", title=f"{title}, q={q_track}")
        figure.colorbar(handle, ax=axis, label=label, shrink=0.82)
    name = "subspace_distance_and_angles_heatmaps.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    figure, axis = plt.subplots(figsize=(8.7, 4.8))
    axis.scatter(
        [float(row["absolute_parameter_distance"]) for row in q_pairs],
        [float(row["normalized_projector_distance"]) for row in q_pairs],
        color=BLUE,
        alpha=0.65,
        edgecolor="none",
    )
    correlation = next(row for row in correlation_rows if int(row["q"]) == q_track)
    axis.text(
        0.03,
        0.95,
        f"Spearman ρ = {float(correlation['spearman_correlation']):.3f}",
        transform=axis.transAxes,
        va="top",
    )
    axis.set(xlabel="|R_i − R_j|", ylabel="normalized projector distance")
    axis.set_title(f"Parameter distance versus Ritz-space change (q={q_track})", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.55)
    name = "subspace_distance_vs_parameter_distance.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    figure, axis = plt.subplots(figsize=(10.2, 5.3))
    methods = ["ILU", "own_oracle", "nearby_reuse", "far_reuse", "random_control"]
    colors = [CHARCOAL, BLUE, ORANGE, PINK, OLIVE]
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
            color=color,
            label=method.replace("_", " "),
        )
    if partial_rows:
        local = sorted(
            [row for row in partial_rows if row["method"] == "partial_update"],
            key=lambda row: float(row["target_ratio"]),
        )
        axis.plot(
            [float(row["target_ratio"]) for row in local],
            [float(row["iterations_median"]) for row in local],
            marker="s",
            linestyle="--",
            color=PURPLE,
            label="partial update",
        )
    axis.set(xlabel="target anisotropy ratio R", ylabel="median FGMRES iterations")
    axis.set_title(f"Own, reused, and control spaces at q={q_track}", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.6)
    axis.legend(frameon=False, ncol=3)
    name = "transfer_comparison.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    residual_ratio = 120.0 if 120.0 in ratios else ratios[-1]
    residual_seed = min(int(row["rhs_seed"]) for row in transfer_histories)
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
    axis.axhline(1.0e-8, color=LIGHT_GRAY, linestyle="--")
    axis.set(xlabel="FGMRES iteration", ylabel="true relative residual")
    axis.set_title(f"Held-out convergence at R={residual_ratio:g}", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.5)
    axis.legend(frameon=False)
    name = "transfer_residual_history.png"
    _save_figure(figure, plot_dir / name)
    paths.append(name)

    if track_rows:
        steps = sorted(
            {(float(row["source_ratio"]), float(row["target_ratio"])) for row in track_rows}
        )
        tracks = sorted({int(row["track_id"]) for row in track_rows})
        matrix = np.full((len(tracks), len(steps)), np.nan)
        for row in track_rows:
            i = tracks.index(int(row["track_id"]))
            j = steps.index((float(row["source_ratio"]), float(row["target_ratio"])))
            matrix[i, j] = float(row["absolute_cosine"])
        figure, axis = plt.subplots(figsize=(13.5, 6.4))
        handle = axis.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
        axis.set_xticks(
            np.arange(len(steps)),
            [f"{first:g}→{second:g}" for first, second in steps],
            rotation=60,
            ha="right",
        )
        axis.set_yticks(np.arange(len(tracks)), [str(track) for track in tracks])
        axis.set(xlabel="adjacent R step", ylabel="tracked direction ID")
        axis.set_title("One-to-one Ritz-direction overlap", loc="left")
        figure.colorbar(handle, ax=axis, label="absolute cosine")
        name = "mode_tracking_cosine_heatmap.png"
        _save_figure(figure, plot_dir / name)
        paths.append(name)

    if ablation_rows:
        selected = sorted({float(row["ratio"]) for row in ablation_rows})
        figure, axes = plt.subplots(len(selected), 1, figsize=(10.0, 2.5 * len(selected)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, ratio in zip(axes, selected):
            local = sorted(
                [row for row in ablation_rows if float(row["ratio"]) == ratio],
                key=lambda row: int(row["mode_index"]),
            )
            penalties = [float(row["iteration_penalty"]) for row in local]
            axis.bar(np.arange(len(local)), penalties, color=[PINK if value >= 5 else BLUE for value in penalties])
            axis.axhline(5, color=LIGHT_GRAY, linestyle="--")
            axis.set_ylabel(f"R={ratio:g}\nΔ iters")
        axes[-1].set_xlabel("local Ritz direction removed")
        axes[0].set_title("Leave-one-direction-out solver penalty", loc="left")
        name = "mode_ablation_penalties.png"
        _save_figure(figure, plot_dir / name)
        paths.append(name)

    if robustness_rows:
        labels = ["construction RHS", "node perturbation\n(interpolated)"]
        values = [
            [
                float(row["projector_distance_normalized"])
                for row in robustness_rows
                if row["variation"] == "construction_rhs"
            ],
            [
                float(row["projector_distance_normalized"])
                for row in robustness_rows
                if row["variation"] == "node_perturbation_interpolated"
            ],
        ]
        figure, axis = plt.subplots(figsize=(7.3, 4.8))
        boxes = axis.boxplot(values, labels=labels, patch_artist=True)
        for box, color in zip(boxes["boxes"], [BLUE, ORANGE]):
            box.set_facecolor(color)
            box.set_alpha(0.75)
        axis.set_ylabel("normalized projector distance")
        axis.set_title(f"Within-system/reference variability at q={q_track}", loc="left")
        axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
        name = "robustness_variability.png"
        _save_figure(figure, plot_dir / name)
        paths.append(name)
    return paths


def build_decision(
    q_track: int,
    pair_rows: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    track_summaries: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    transfer_gate: Mapping[str, Any],
    partial_gate: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]],
    robustness_summary: Mapping[str, Any],
    settings: Stage10Settings,
) -> Dict[str, Any]:
    """Apply explicit, conservative gates to the Stage 10 evidence."""

    q_pairs = [row for row in pair_rows if int(row["q"]) == q_track]
    all_between = [float(row["normalized_projector_distance"]) for row in q_pairs]
    adjacent_between = [
        float(row["normalized_projector_distance"])
        for row in q_pairs
        if float(row["absolute_parameter_distance"]) <= 2.0
    ]
    correlation = next(row for row in correlation_rows if int(row["q"]) == q_track)
    hard_ratios = {
        float(row["target_ratio"])
        for row in transfer_rows
        if row["method"] == "ILU"
        and float(row["iterations_median"]) >= settings.hard_iteration_threshold
    }
    own_rows = [
        row
        for row in transfer_rows
        if row["method"] == "own_oracle"
        and float(row["target_ratio"]) in hard_ratios
    ]
    oracle_reductions = [float(row["iteration_reduction_vs_ilu"]) for row in own_rows]
    stable_tracks = {
        int(row["track_id"])
        for row in track_summaries
        if bool(row["globally_stable"])
    }
    critical_rows = [row for row in ablation_rows if float(row["iteration_penalty"]) >= 5.0]
    critical_tracks = {int(row["track_id"]) for row in critical_rows}
    stable_critical = stable_tracks & critical_tracks
    between_median = float(np.median(all_between))
    rhs_within_median = float(robustness_summary["rhs_variability_median"])
    rhs_within_p90 = float(robustness_summary["rhs_variability_p90"])
    node_variability_median = float(robustness_summary["node_variability_median"])
    within_between_ratio = rhs_within_p90 / max(between_median, 1.0e-300)
    node_between_ratio = node_variability_median / max(between_median, 1.0e-300)
    oracle_useful = bool(
        own_rows
        and np.median(oracle_reductions) >= settings.useful_iteration_reduction
        and np.mean([float(row["convergence_rate"]) for row in own_rows]) >= 0.9
    )
    structured = bool(
        float(correlation["spearman_correlation"])
        >= settings.structured_distance_correlation
    )
    target_stable = bool(within_between_ratio < settings.within_between_ratio_gate)
    node_robust = bool(node_between_ratio < settings.within_between_ratio_gate)
    full_reuse_useful = not bool(transfer_gate["full_reuse_failed"])
    partial_reuse_useful = bool(partial_gate.get("partial_reuse_useful", False))
    reuse_supported = full_reuse_useful or partial_reuse_useful
    learned_prediction_justified = bool(
        oracle_useful
        and structured
        and target_stable
        and node_robust
        and reuse_supported
        and stable_critical
    )
    return {
        "q_track": q_track,
        "hard_ratio_count": len(hard_ratios),
        "hard_ratios": sorted(hard_ratios),
        "oracle_iteration_reduction_median": float(np.median(oracle_reductions)) if oracle_reductions else float("nan"),
        "oracle_convergence_rate_mean": float(np.mean([float(row["convergence_rate"]) for row in own_rows])) if own_rows else 0.0,
        "oracle_useful": oracle_useful,
        "spearman_parameter_subspace_correlation": float(correlation["spearman_correlation"]),
        "structured_variation": structured,
        "between_projector_distance_median": between_median,
        "between_principal_angle_mean_median_degrees": float(
            np.median([float(row["principal_angle_mean_degrees"]) for row in q_pairs])
        ),
        "between_principal_angle_max_median_degrees": float(
            np.median([float(row["principal_angle_max_degrees"]) for row in q_pairs])
        ),
        "adjacent_delta_le_2_distance_median": float(np.median(adjacent_between)) if adjacent_between else float("nan"),
        "rhs_within_projector_distance_median": rhs_within_median,
        "rhs_within_projector_distance_p90": rhs_within_p90,
        "within_between_ratio": within_between_ratio,
        "target_stable": target_stable,
        "node_variability_projector_distance_median": node_variability_median,
        "node_variability_between_ratio": node_between_ratio,
        "node_robust": node_robust,
        "globally_stable_track_count": len(stable_tracks),
        "critical_track_count": len(critical_tracks),
        "stable_critical_track_count": len(stable_critical),
        "stable_critical_track_ids": sorted(stable_critical),
        "full_reuse_useful": full_reuse_useful,
        "partial_reuse_useful": partial_reuse_useful,
        "reuse_supported": reuse_supported,
        "learned_subspace_prediction_justified": learned_prediction_justified,
        "cheap_partial_update_promising": bool(
            oracle_useful
            and structured
            and partial_reuse_useful
            and float(partial_gate.get("stable_fraction_median", 0.0)) >= 0.5
        ),
        "decision_rule": (
            "Prediction requires useful own-system Ritz correction, Spearman rho >= 0.70, "
            "construction-RHS p90/between and node-variability/between projector ratios < 0.50, "
            "useful full/partial reuse, and at least one direction that is both globally stable "
            "and solver-critical. A cheap partial update must reuse at least half the directions."
        ),
    }


def _format_number(value: Any, digits: int = 3) -> str:
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


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    metadata_rows: Sequence[Mapping[str, Any]],
    q_summaries: Sequence[Mapping[str, Any]],
    elbows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    track_summaries: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    transfer_gate: Mapping[str, Any],
    partial_gate: Mapping[str, Any],
    ablation_summaries: Sequence[Mapping[str, Any]],
    robustness_summary: Mapping[str, Any],
    robustness_rows: Sequence[Mapping[str, Any]],
    robustness_performance: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage10Settings,
) -> Path:
    """Write the concise, evidence-linked Stage 10 research report."""

    q_track = int(decision["q_track"])
    ratios = sorted(float(row["ratio"]) for row in metadata_rows)
    hard_ratios = set(float(value) for value in decision["hard_ratios"])
    prediction = bool(decision["learned_subspace_prediction_justified"])
    update_promising = bool(decision["cheap_partial_update_promising"])
    if prediction:
        headline = (
            "The fixed-node Ritz target passes the conservative prediction gate: own-system "
            "spaces are useful, variation is systematic, and stable solver-critical directions exist."
        )
    elif update_promising:
        headline = (
            "Full learned-space prediction is not yet justified, but a partial-update route is supported: "
            "stable directions can be reused while sensitive directions are recomputed."
        )
    else:
        headline = (
            "The evidence does not yet justify learned Ritz-space prediction; the measured target fails "
            "at least one solver relevance, stability, or reuse gate."
        )

    system_rows = []
    for ratio in ratios:
        metadata = next(row for row in metadata_rows if float(row["ratio"]) == ratio)
        baseline = next(
            row
            for row in q_summaries
            if float(row["ratio"]) == ratio and int(row["q"]) == 0
        )
        own = next(
            row
            for row in q_summaries
            if float(row["ratio"]) == ratio and int(row["q"]) == q_track
        )
        elbow = next(row for row in elbows if float(row["ratio"]) == ratio)
        system_rows.append(
            (
                f"{ratio:g}",
                _format_number(baseline["iterations_median"], 0),
                _format_number(baseline["convergence_rate"], 2),
                _format_number(own["iterations_median"], 0),
                _format_number(own["convergence_rate"], 2),
                f"{int(elbow['major_drop_from_q'])}→{int(elbow['major_drop_to_q'])}",
                _format_number(elbow["major_iteration_drop"], 0),
                f"{float(metadata['relative_discretization_error']):.2e}",
                "hard" if ratio in hard_ratios else "easy/moderate",
            )
        )

    q_correlation_rows = [
        (
            str(int(row["q"])),
            _format_number(row["spearman_correlation"]),
            _format_number(row["pearson_correlation"]),
            _format_number(row["median_projector_distance"]),
        )
        for row in correlation_rows
    ]
    transfer_table_rows = []
    for ratio in sorted(hard_ratios):
        local = {row["method"]: row for row in transfer_rows if float(row["target_ratio"]) == ratio}
        transfer_table_rows.append(
            (
                f"{ratio:g}",
                _format_number(local["ILU"]["iterations_median"], 0),
                _format_number(local["own_oracle"]["iterations_median"], 0),
                _format_number(local["nearby_reuse"]["iterations_median"], 0),
                f"R={float(local['nearby_reuse']['source_ratio']):g}",
                _format_number(local["far_reuse"]["iterations_median"], 0),
                _format_number(local["random_control"]["iterations_median"], 0),
            )
        )

    stable_tracks = [row for row in track_summaries if bool(row["globally_stable"])]
    track_table_rows = [
        (
            str(int(row["track_id"])),
            _format_number(row["cosine_median"]),
            _format_number(row["cosine_p10"]),
            _format_number(row["cosine_min"]),
        )
        for row in sorted(stable_tracks, key=lambda row: int(row["track_id"]))
    ]
    if not track_table_rows:
        track_table_rows = [("none", "—", "—", "—")]
    ablation_table_rows = []
    for row in ablation_summaries:
        indices = row["critical_mode_indices"]
        if isinstance(indices, (float, np.floating)) and np.isfinite(indices):
            indices = str(int(indices))
        ablation_table_rows.append(
            (
                f"{float(row['ratio']):g}",
                str(int(row["critical_mode_count"])),
                str(indices) or "none",
                _format_number(row["max_iteration_penalty"], 1),
            )
        )
    robustness_table_rows = []
    for variation in ("construction_rhs", "node_perturbation"):
        local = [row for row in robustness_performance if row["variation"] == variation]
        robustness_table_rows.append(
            (
                variation.replace("_", " "),
                _format_number(np.median([float(row["iterations_median"]) for row in local]), 1),
                _format_number(np.mean([float(row["convergence_rate"]) for row in local]), 2),
                _format_number(np.median([float(row["coarse_setup_seconds"]) for row in local]), 3),
            )
        )
    robustness_variability_rows = []
    for ratio in sorted({float(row["ratio"]) for row in robustness_rows}):
        rhs_values = [
            float(row["projector_distance_normalized"])
            for row in robustness_rows
            if float(row["ratio"]) == ratio
            and row["variation"] == "construction_rhs"
        ]
        node_values = [
            float(row["projector_distance_normalized"])
            for row in robustness_rows
            if float(row["ratio"]) == ratio
            and row["variation"] == "node_perturbation_interpolated"
        ]
        robustness_variability_rows.append(
            (
                f"{ratio:g}",
                _format_number(np.median(rhs_values)),
                _format_number(max(rhs_values)),
                _format_number(np.median(node_values)),
                _format_number(max(node_values)),
            )
        )

    report = f"""# Stage 10: Ritz transfer and solver-critical mode tracking

## Executive conclusion

{headline}

The experiment holds the main RBF-FD discretization, node coordinates and ordering, stencil, anisotropy angle, and strong ILU configuration fixed. Only the anisotropy ratio varies in the primary sweep. The tracking dimension selected by a solver-level rule is **q={q_track}**. The own-system reference gives a median iteration reduction of **{100.0 * float(decision['oracle_iteration_reduction_median']):.1f}%** on the {int(decision['hard_ratio_count'])} hard ratios, but its mean convergence rate is only **{float(decision['oracle_convergence_rate_mean']):.2f}**. Parameter distance and q={q_track} projector distance have Spearman correlation **ρ={float(decision['spearman_parameter_subspace_correlation']):.3f}**. The construction-RHS p90/between variability ratio is **{float(decision['within_between_ratio']):.3f}**, the node-perturbation/between ratio is **{float(decision['node_variability_between_ratio']):.3f}**, full nearby reuse is **{'useful' if decision['full_reuse_useful'] else 'not useful'}**, and partial update is **{'useful' if decision['partial_reuse_useful'] else 'not useful or not needed'}** under the declared gates.

**Decision:** learned-subspace prediction is **{'justified for a first small prototype' if prediction else 'not justified yet'}**. A cheap stable-mode reuse/sensitive-mode update is **{'promising' if update_promising else 'not supported'}**. No neural network was trained in this round.

## Starting point and fixed protocol

The [Stage 9 report](../stage9_hard_regime_discovery_and_scalability/stage9_hard_regime_discovery_and_scalability_report.md) identified the rotated-anisotropy family as the strongest nontrivial regime: strong ILU became unreliable near R≈100–120, while a q=24 own-system Ritz correction was materially useful. Stage 9 also found that transferring a whole fixed q=24 space was weak, motivating this denser transition sweep and direction-level tracking.

The main protocol is:

- N={settings.n_nodes}, jittered-grid seed={settings.node_seed}, jitter fraction={settings.jitter_fraction}, stencil={settings.stencil_size}, PHS RBF-FD and polynomial augmentation inherited unchanged from the current codebase;
- rotated tensor anisotropy angle={settings.anisotropy_angle_degrees:g}° and ratios R={', '.join(f'{ratio:g}' for ratio in settings.ratios)};
- strong ILU: drop tolerance={settings.classical_ilu.drop_tolerance:g}, fill factor={settings.classical_ilu.fill_factor:g}, COLAMD, zero diagonal-pivot threshold;
- FGMRES tolerance={settings.solver.relative_tolerance:.0e}, restart={settings.solver.restart}, cap={settings.solver.max_iterations};
- each Z\*(R) is a nested real solution-space Ritz basis from Arnoldi on A(R)M(R)^−1, ordered by increasing |Ritz value| and mapped through M(R)^−1;
- the construction RHS is disjoint from three held-out evaluation RHSs. “Oracle” means the target system uses its own reference basis; it does not mean a globally optimized best-q space;
- q∈{{1,2,4,8,12,16,24,32}}. q={q_track} was selected as the requested q nearest the median dimension achieving 80% of the q=32 gain on hard systems—not chosen after inspecting transfer scores.

The denser points {{82,84,85,86,88,95,105}} were added after a diagnostic probe found an abrupt ILU iteration increase between R=80 and R=85 and first cap hits near R=110. They refine the transition rather than changing any other numerical parameter.

## 1. Solver-critical dimension

{_markdown_table(
    ['R', 'ILU it.', 'ILU conv.', f'own q={q_track} it.', 'own conv.', 'largest drop q', 'drop', 'rel. disc. error', 'regime'],
    system_rows,
)}

![ILU transition](plots/ilu_transition.png)

![q sweep heatmap](plots/q_sweep_iteration_heatmap.png)

![selected q curves](plots/q_sweep_selected_curves.png)

The “largest drop” is the largest decrease between adjacent requested q values for that ratio. It is a solver-level diagnostic, not a claim that every direction before the elbow is individually essential. Leave-one-out tests below isolate individual critical directions.

## 2. Cross-system variation

{_markdown_table(
    ['q', 'Spearman ρ', 'Pearson r', 'median normalized projector distance'],
    q_correlation_rows,
)}

At q={q_track}, the median pairwise normalized projector distance is **{float(decision['between_projector_distance_median']):.3f}**; for the most local sampled pairs (ΔR≤2) it is **{float(decision['adjacent_delta_le_2_distance_median']):.3f}**. The median pairwise mean principal angle is **{float(decision['between_principal_angle_mean_median_degrees']):.1f}°**, while the median of the largest principal angle is **{float(decision['between_principal_angle_max_median_degrees']):.1f}°**. Correlation establishes monotone structure at the subspace level but does not, by itself, establish transferability or a stable supervised target.

![subspace heatmaps](plots/subspace_distance_and_angles_heatmaps.png)

![distance correlation](plots/subspace_distance_vs_parameter_distance.png)

## 3. Full-space transfer and controls

{_markdown_table(
    ['target R', 'ILU', 'own oracle', 'near reuse', 'near source', 'far reuse', 'random'],
    transfer_table_rows,
)}

On hard targets, nearby reuse has median relative iteration reduction **{100.0 * float(transfer_gate['nearby_iteration_reduction_median']):.1f}%**, median gap **{float(transfer_gate['nearby_iteration_gap_vs_own_median']):.1f}** iterations to the own-system reference, and mean convergence rate **{float(transfer_gate['nearby_convergence_rate_mean']):.2f}**. Therefore full nearby reuse **{'passes' if not transfer_gate['full_reuse_failed'] else 'fails'}** the predeclared combined gate (≥20% reduction, ≥0.90 convergence, ≤10-iteration median oracle gap). Far-away and random spaces are controls against attributing improvement merely to adding q vectors.

![transfer comparison](plots/transfer_comparison.png)

![held-out residual history](plots/transfer_residual_history.png)

## 4. Individual mode tracking and partial update

Globally stable tracks require the 10th percentile of adjacent-step absolute cosine to be at least {settings.stable_direction_cosine:.2f}. There are **{int(decision['globally_stable_track_count'])}/{q_track}** such tracks. The stable tracks are:

{_markdown_table(['track ID', 'median cosine', 'p10 cosine', 'minimum cosine'], track_table_rows)}

![mode tracking](plots/mode_tracking_cosine_heatmap.png)

The leave-one-direction-out test calls a direction solver-critical when its removal adds at least five median iterations across two held-out RHSs:

{_markdown_table(['R', 'critical count', 'local indices', 'maximum iteration penalty'], ablation_table_rows)}

Across the tested ratios there are **{int(decision['critical_track_count'])}** distinct critical tracks, of which **{int(decision['stable_critical_track_count'])}** are globally stable (IDs {decision['stable_critical_track_ids']}). This explicit intersection is the evidence relevant to “stable and solver-critical”; high directional cosine alone is not enough.

![mode ablation](plots/mode_ablation_penalties.png)

{'Because full reuse failed, the oracle partial-update diagnostic retained matched source directions with cosine ≥0.85 and recomputed the sensitive target directions. It reused a median fraction of ' + _format_number(partial_gate.get('stable_fraction_median', float('nan')), 2) + ' on hard targets, achieved median iteration reduction ' + _format_number(100.0 * float(partial_gate.get('iteration_reduction_median', float('nan'))), 1) + '%, had a median ' + _format_number(partial_gate.get('iteration_gap_vs_own_median', float('nan')), 1) + '-iteration oracle gap, and converged at rate ' + _format_number(partial_gate.get('convergence_rate_mean', float('nan')), 2) + '. Because it recomputes the target-sensitive directions from the target oracle, it is an upper-bound diagnostic rather than an available online algorithm; reusing only ' + _format_number(partial_gate.get('stable_fraction_median', float('nan')), 2) + ' of the directions is not a cheap update.' if partial_gate.get('performed', False) else 'The partial-update experiment was not triggered because full nearby reuse already passed the combined gate.'}

## 5. Robustness to construction RHS and node perturbations

For each selected R∈{{80,85,90,100,120}}, three independently constructed RHS-based Ritz spaces were compared on the identical system. Separately, node seeds {{7,17,27}} were assembled and their bases interpolated to the seed-7 nodes before projector comparison; this is a robustness test outside the primary fixed-node sweep, not evidence that only R changed.

- construction-RHS median/p90 normalized projector variability: **{float(robustness_summary['rhs_variability_median']):.3f} / {float(robustness_summary['rhs_variability_p90']):.3f}**;
- node-perturbation median/p90 interpolated variability: **{float(robustness_summary['node_variability_median']):.3f} / {float(robustness_summary['node_variability_p90']):.3f}**;
- construction-RHS p90/between ratio: **{float(decision['within_between_ratio']):.3f}**; node-perturbation median/between ratio: **{float(decision['node_variability_between_ratio']):.3f}** (each required <{settings.within_between_ratio_gate:.2f});
- mean own-reference convergence rate across robustness replicas: **{float(robustness_summary['oracle_convergence_rate']):.2f}**.

{_markdown_table(['R', 'RHS median', 'RHS maximum', 'node median', 'node maximum'], robustness_variability_rows)}

{_markdown_table(['variation', 'median oracle it.', 'mean convergence', 'median coarse setup (s)'], robustness_table_rows)}

![robustness variability](plots/robustness_variability.png)

## 6. Decision against the research question

The conservative gates are deliberately conjunctive:

| Gate | Threshold | Result | Pass |
| --- | --- | --- | --- |
| Own-system solver relevance | median hard-regime reduction ≥20%, convergence ≥0.90 | {100.0 * float(decision['oracle_iteration_reduction_median']):.1f}%, {float(decision['oracle_convergence_rate_mean']):.2f} | {'yes' if decision['oracle_useful'] else 'no'} |
| Systematic R dependence | Spearman ρ≥0.70 | {float(decision['spearman_parameter_subspace_correlation']):.3f} | {'yes' if decision['structured_variation'] else 'no'} |
| Same-system target stability | construction-RHS p90/between <0.50 | {float(decision['within_between_ratio']):.3f} | {'yes' if decision['target_stable'] else 'no'} |
| Node-perturbation robustness | node median/between <0.50 | {float(decision['node_variability_between_ratio']):.3f} | {'yes' if decision['node_robust'] else 'no'} |
| Reuse/update value | full reuse or partial update passes solver gate | full={decision['full_reuse_useful']}, partial={decision['partial_reuse_useful']} | {'yes' if decision['reuse_supported'] else 'no'} |
| Stable solver-critical direction | at least one | {int(decision['stable_critical_track_count'])} | {'yes' if decision['stable_critical_track_count'] else 'no'} |

**Answer:** {'Yes—within this fixed-node, fixed-angle family, a first learned predictor has a sufficiently stable and solver-relevant target. The evidence remains local to R∈[60,140], so the next prototype should remain small and be tested against cheap reuse/update.' if prediction else 'No—not yet. The current Ritz construction does not simultaneously satisfy solver relevance, stability, and transfer/update requirements. Training a learned-subspace network now would conflate target noise or mode sensitivity with approximation error.'}

## 7. Limitations and reproducibility

- The study covers one resolution, node ordering, stencil, anisotropy angle, ILU budget, and R interval. It establishes or rejects a local map R↦range(Z\*(R)); it does not establish transfer over arbitrary geometries.
- Individual real Ritz directions can rotate within clustered invariant subspaces. Projector/principal-angle conclusions are therefore primary; mode tracking is used only with one-to-one matching and sign-invariant cosine.
- The node-perturbation comparison uses linear interpolation plus nearest-neighbor fallback to a shared probe node set; interpolation error is part of the reported variability.
- Every solver result uses a true-residual convergence history and failures/cap hits remain in the CSV files. No failed solve was removed.
- Timings are single-process wall-clock measurements on the environment recorded in `environment.json`; they are diagnostics, not hardware-independent complexity claims.

Raw results are in `system_metadata.csv`, `q_sweep_results.csv`, `q_sweep_histories.csv`, `subspace_pairs.csv`, `mode_tracking_adjacent.csv`, `mode_track_assignments.csv`, `transfer_results.csv`, `transfer_histories.csv`, `partial_reuse_results.csv`, `mode_ablation.csv`, and `robustness_variability.csv`. Reference bases are stored under `subspaces/`; decision inputs and thresholds are stored in `decision.json` and `settings.json`.
"""
    path = output / "stage10_ritz_transfer_and_mode_tracking_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_stage10(output: Path) -> Path:
    """Run the complete no-ML Stage 10 study and return the report path."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = Stage10Settings()
    np.random.seed(settings.seed)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__ if "scipy" in sys.modules else "unknown",
            "matplotlib": matplotlib.__version__,
        },
    )
    print("[1/8] assembling fixed-node ratio sweep and Ritz references", flush=True)
    contexts, metadata_rows = build_main_contexts(output, settings)
    print("[2/8] testing q-dependent solver relevance", flush=True)
    _, q_summaries, elbows, q_track = run_q_sweep(output, contexts, settings)
    print(f"[3/8] measuring cross-system variation at selected q={q_track}", flush=True)
    pair_rows, correlation_rows, track_summaries = run_cross_system_analysis(
        output, contexts, q_track, settings
    )
    print("[4/8] evaluating own, near, far, and random transfer controls", flush=True)
    transfer_rows, transfer_histories, transfer_gate = run_transfer_controls(
        output, contexts, q_track, settings
    )
    print("[5/8] testing partial reuse when full reuse fails", flush=True)
    partial_rows, partial_gate = run_partial_reuse(
        output, contexts, transfer_rows, transfer_gate, q_track, settings
    )
    print("[6/8] ablating individual tracked directions", flush=True)
    ablation_rows, ablation_summaries = run_mode_ablation(
        output, contexts, q_track, settings
    )
    print("[7/8] repeating construction RHS and node perturbations", flush=True)
    robustness_rows, robustness_performance, robustness_summary = run_robustness(
        output, contexts, q_track, settings
    )
    decision = build_decision(
        q_track,
        pair_rows,
        correlation_rows,
        track_summaries,
        transfer_rows,
        transfer_gate,
        partial_gate,
        ablation_rows,
        robustness_summary,
        settings,
    )
    _write_json(output / "decision.json", decision)
    print("[8/8] rendering figures and report", flush=True)
    make_plots(
        output,
        q_summaries,
        pair_rows,
        correlation_rows,
        [
            row
            for row in csv.DictReader(
                (output / "mode_tracking_adjacent.csv").open(encoding="utf-8")
            )
        ],
        transfer_rows,
        transfer_histories,
        partial_rows,
        ablation_rows,
        robustness_rows,
        q_track,
    )
    report = write_report(
        output,
        metadata_rows,
        q_summaries,
        elbows,
        pair_rows,
        correlation_rows,
        track_summaries,
        transfer_rows,
        transfer_gate,
        partial_gate,
        ablation_summaries,
        robustness_summary,
        robustness_rows,
        robustness_performance,
        decision,
        settings,
    )
    print(f"completed: {report}", flush=True)
    return report
