"""Stage 21: population validation of the frozen spatial-K Ritz oracle.

This stage performs no model training and no oracle selection.  It applies the
Stage-20 ordinary-Ritz construction with Arnoldi depth 128 and requested rank
24 to every numerically valid Stage-19A coefficient field.  Three independent
construction right-hand sides are retained solely to measure construction
stability; the first is the frozen primary construction used for the population
gate.
"""

from __future__ import annotations

import csv
import hashlib
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
from matplotlib.lines import Line2D

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..krylov import KrylovResult
from ..preconditioners import ilu_preconditioner
from ..rbffd import RBFSystem, assemble_on_nodes
from ..subspace_analysis import (
    generalized_subspace_metrics,
    nested_krylov_ritz_modes,
    ritz_atomic_blocks,
)
from .stage10_ritz_transfer import Stage10Settings
from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage20_spatial_k_robust_oracle import (
    SystemContext,
    _coefficient_from_row,
    _complete_block_indices,
    _read_rows,
    _relative_residual,
    _run_baseline,
    _run_recycle,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PURPLE = "#7553A6"
PINK = "#B14D75"
CHARCOAL = "#30343B"
GRAY = "#8A9099"
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
class Stage21Settings:
    """Frozen oracle, cohorts, RHS protocol, and validation thresholds."""

    arnoldi_dimension: int = 128
    requested_q: int = 24
    extraction: str = "ordinary_ritz_smallest_magnitude_pair_preserving"
    maximum_extracted_modes: int = 64
    construction_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    stage20_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    easy_iteration_upper_exclusive: int = 50
    hard_iteration_lower_inclusive: int = 200
    system_reduction_gate: float = 0.40
    unseen_pass_fraction_gate: float = 0.70
    unseen_iteration_ratio_gate: float = 0.65
    seed: int = 20262101

    @property
    def solver(self):
        return Stage10Settings().solver

    @property
    def ilu_config(self):
        return Stage10Settings().classical_ilu

    @property
    def discretization(self):
        from .stage20_spatial_k_robust_oracle import Stage20Settings

        return Stage20Settings().discretization


@dataclass
class Construction:
    """One frozen Ritz/recycle construction and its measured setup costs."""

    rhs_seed: int
    basis: np.ndarray
    factors: AZPetrovFactors
    actual_q: int
    ritz_seconds: float
    pair_seconds: float
    selected_values: np.ndarray
    basis_orthogonality_error: float

    @property
    def total_setup_seconds(self) -> float:
        return self.ritz_seconds + self.pair_seconds


def classify_difficulty(iterations: float, settings: Stage21Settings) -> str:
    """Classify systems from archived strong-ILU results, before oracle use."""

    if iterations < settings.easy_iteration_upper_exclusive:
        return "easy"
    if iterations < settings.hard_iteration_lower_inclusive:
        return "transition"
    return "hard"


def system_gate(
    convergence_flags: Sequence[bool],
    relative_reductions: Sequence[float],
    settings: Stage21Settings,
) -> bool:
    """Apply the predeclared system gate without outcome-dependent changes."""

    if not convergence_flags or not relative_reductions:
        return False
    return bool(
        all(_bool(value) for value in convergence_flags)
        and float(np.median(relative_reductions)) >= settings.system_reduction_gate
    )


def population_gate(
    system_rows: Sequence[Mapping[str, Any]], settings: Stage21Settings
) -> Dict[str, Any]:
    """Apply the frozen unseen hard/transition population gate."""

    cohort = [
        row
        for row in system_rows
        if str(row["provenance_cohort"]) == "unseen"
        and str(row["difficulty_class"]) in {"transition", "hard"}
    ]
    if not cohort:
        return {
            "systems": 0,
            "systems_passing": 0,
            "pass_fraction": float("nan"),
            "median_iteration_ratio": float("nan"),
            "pass_fraction_gate": False,
            "iteration_ratio_gate": False,
            "population_gate_passed": False,
        }
    pass_fraction = float(np.mean([_bool(row["system_gate_passed"]) for row in cohort]))
    median_ratio = float(np.median([float(row["iteration_ratio_median"]) for row in cohort]))
    fraction_ok = pass_fraction >= settings.unseen_pass_fraction_gate
    ratio_ok = median_ratio <= settings.unseen_iteration_ratio_gate
    return {
        "systems": len(cohort),
        "systems_passing": int(sum(_bool(row["system_gate_passed"]) for row in cohort)),
        "pass_fraction": pass_fraction,
        "median_iteration_ratio": median_ratio,
        "pass_fraction_gate": bool(fraction_ok),
        "iteration_ratio_gate": bool(ratio_ok),
        "population_gate_passed": bool(fraction_ok and ratio_ok),
    }


def estimate_break_even(setup_seconds: float, baseline_seconds: float, recycle_seconds: float) -> float:
    """Return incremental repeated-RHS break-even count, or infinity.

    ILU setup is common to both methods.  The numerator is therefore only the
    Ritz plus recycle-pair setup, and the denominator is the median online time
    saved per right-hand side.
    """

    saving = baseline_seconds - recycle_seconds
    if not np.isfinite(saving) or saving <= 0.0:
        return float("inf")
    return float(math.ceil(setup_seconds / saving))


def _safe_spearman(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(finite))
    if count < 3 or np.ptp(x[finite]) == 0.0 or np.ptp(y[finite]) == 0.0:
        return float("nan"), float("nan"), count
    result = stats.spearmanr(x[finite], y[finite])
    return float(result.statistic), float(result.pvalue), count


def _archived_median(row: Mapping[str, str]) -> float:
    return float(row["ilu_iterations_median"])


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_population(root: Path, settings: Stage21Settings):
    stage19a = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage19A_spatial_coefficient_oracle_validation"
    )
    node_data = np.load(stage19a / "fixed_nodes.npz")
    nodes = np.asarray(node_data["nodes"], dtype=float)
    boundary = np.asarray(node_data["boundary"], dtype=bool)
    coefficients = {
        int(row["sample_id"]): row
        for row in _read_rows(stage19a / "coefficient_samples.csv")
    }
    summaries = {
        int(row["sample_id"]): row
        for row in _read_rows(stage19a / "hardness_scan_summary.csv")
    }
    rhs_rows = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in _read_rows(stage19a / "hardness_scan_rhs.csv")
    }
    valid = [sample_id for sample_id, row in summaries.items() if _bool(row["numerically_valid"])]
    invalid = [sample_id for sample_id, row in summaries.items() if not _bool(row["numerically_valid"])]
    return stage19a, nodes, boundary, coefficients, summaries, rhs_rows, sorted(valid), sorted(invalid)


def _assemble_context(
    sample_id: int,
    coefficient_row: Mapping[str, str],
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage21Settings,
) -> Tuple[SystemContext, float]:
    coefficient = _coefficient_from_row(coefficient_row)
    start = perf_counter()
    system = assemble_on_nodes(nodes, boundary, settings.discretization, coefficient)
    assembly_seconds = perf_counter() - start
    ilu = ilu_preconditioner(system.matrix, settings.ilu_config, name="strong_ilu")
    return SystemContext(sample_id, system, ilu), assembly_seconds


def _construct(
    context: SystemContext,
    construction_rhs_seed: int,
    settings: Stage21Settings,
) -> Construction:
    rhs, _ = _discrete_rhs(context.system, construction_rhs_seed)
    modes = nested_krylov_ritz_modes(
        context.system.matrix,
        context.ilu,
        max_modes=settings.maximum_extracted_modes,
        arnoldi_dimension=settings.arnoldi_dimension,
        seed=settings.seed + context.sample_id + construction_rhs_seed,
        initial_vector=rhs,
    )
    blocks = ritz_atomic_blocks(modes.eigenvalues)
    indices = _complete_block_indices(blocks, settings.requested_q)
    basis = np.asarray(modes.basis[:, indices], dtype=float)
    factor_start = perf_counter()
    factors = factor_az_petrov(context.system.matrix, basis)
    factor_seconds = perf_counter() - factor_start
    return Construction(
        rhs_seed=construction_rhs_seed,
        basis=basis,
        factors=factors,
        actual_q=basis.shape[1],
        ritz_seconds=float(modes.diagnostics["setup_seconds"]),
        pair_seconds=factor_seconds,
        selected_values=np.asarray(modes.eigenvalues[list(indices)], dtype=complex),
        basis_orthogonality_error=float(modes.diagnostics["basis_orthogonality_error"]),
    )


def _run_population(
    root: Path,
    output: Path,
    settings: Stage21Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    (
        _,
        nodes,
        boundary,
        coefficient_rows,
        archived_summaries,
        archived_rhs,
        valid_ids,
        invalid_ids,
    ) = _load_population(root, settings)

    exclusion_rows = []
    for sample_id in invalid_ids:
        source = archived_summaries[sample_id]
        exclusion_rows.append(
            {
                "sample_id": sample_id,
                "numerically_valid": False,
                "reason": source.get("failure", "archived Stage-19A validity failure"),
                "manufactured_relative_error": source.get("manufactured_relative_error", ""),
                "manufactured_direct_residual": source.get("manufactured_direct_residual", ""),
            }
        )
    _write_csv(output / "excluded_systems.csv", exclusion_rows)

    per_rhs_rows: List[Dict[str, Any]] = []
    construction_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ordinal, sample_id in enumerate(valid_ids, start=1):
        archived = archived_summaries[sample_id]
        archived_median = _archived_median(archived)
        difficulty = classify_difficulty(archived_median, settings)
        provenance = "stage20_diagnostic" if sample_id in settings.stage20_ids else "unseen"
        print(
            f"[Stage21 {ordinal:02d}/{len(valid_ids)}] ID {sample_id:02d} "
            f"({difficulty}, {provenance})",
            flush=True,
        )
        context, assembly_seconds = _assemble_context(
            sample_id, coefficient_rows[sample_id], nodes, boundary, settings
        )
        ilu_setup_seconds = float(context.ilu.setup_seconds)

        baseline_results: Dict[int, KrylovResult] = {}
        baseline_rows: Dict[int, Dict[str, Any]] = {}
        for rhs_seed in settings.heldout_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            result = _run_baseline(context, rhs, settings.solver)
            true_residual = _relative_residual(context.system.matrix, result.solution, rhs)
            old = archived_rhs[(sample_id, rhs_seed)]
            exact_iteration_match = result.iterations == int(old["iterations"])
            exact_convergence_match = result.converged == _bool(old["converged"])
            if not exact_iteration_match or not exact_convergence_match:
                raise RuntimeError(
                    f"Stage-19A baseline reproduction failed for ID {sample_id}, "
                    f"RHS {rhs_seed}: saved {old['iterations']}/{old['converged']}, "
                    f"got {result.iterations}/{result.converged}"
                )
            baseline_results[rhs_seed] = result
            baseline_rows[rhs_seed] = {
                "baseline_iterations": result.iterations,
                "baseline_converged": result.converged,
                "baseline_true_final_relative_residual": true_residual,
                "baseline_solve_seconds": result.solve_seconds,
                "baseline_preconditioner_apply_seconds": result.preconditioner_seconds,
                "baseline_reproduces_stage19a": bool(
                    exact_iteration_match and exact_convergence_match
                ),
            }

        constructions: List[Construction] = []
        performance_by_construction: Dict[int, Dict[str, Any]] = {}
        for construction_index, construction_seed in enumerate(
            settings.construction_rhs_seeds
        ):
            construction = _construct(context, construction_seed, settings)
            constructions.append(construction)
            factors = construction.factors
            identity = factors.diagnostics
            construction_rows.append(
                {
                    "sample_id": sample_id,
                    "difficulty_class": difficulty,
                    "provenance_cohort": provenance,
                    "construction_rhs_seed": construction_seed,
                    "primary_construction": construction_index == 0,
                    "arnoldi_dimension": settings.arnoldi_dimension,
                    "requested_q": settings.requested_q,
                    "actual_q": construction.actual_q,
                    "q_over_n": construction.actual_q / context.system.matrix.shape[0],
                    "ritz_setup_seconds": construction.ritz_seconds,
                    "pair_setup_seconds": construction.pair_seconds,
                    "total_recycle_setup_seconds": construction.total_setup_seconds,
                    "basis_orthogonality_error": construction.basis_orthogonality_error,
                    "test_orthogonality_error": identity["test_orthogonality_error"],
                    "au_equals_w_relative_error": identity["au_equals_w_relative_error"],
                    "wtaz_equals_r_relative_error": identity["wtaz_equals_r_relative_error"],
                    "reduced_condition": identity["reduced_condition"],
                    "selected_ritz_magnitude_min": float(
                        np.min(np.abs(construction.selected_values))
                    ),
                    "selected_ritz_magnitude_max": float(
                        np.max(np.abs(construction.selected_values))
                    ),
                }
            )
            local_rows: List[Dict[str, Any]] = []
            for rhs_seed in settings.heldout_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                result = _run_recycle(context, rhs, factors, settings.solver)
                baseline = baseline_results[rhs_seed]
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                iteration_ratio = result.iterations / max(baseline.iterations, 1)
                reduction = (baseline.iterations - result.iterations) / max(
                    baseline.iterations, 1
                )
                row = {
                    "sample_id": sample_id,
                    "difficulty_class": difficulty,
                    "provenance_cohort": provenance,
                    "construction_rhs_seed": construction_seed,
                    "primary_construction": construction_index == 0,
                    "heldout_rhs_seed": rhs_seed,
                    "construction_and_evaluation_disjoint": construction_seed
                    != rhs_seed,
                    "n": context.system.matrix.shape[0],
                    "nnz": context.system.matrix.nnz,
                    "assembly_seconds": assembly_seconds,
                    "ilu_setup_seconds": ilu_setup_seconds,
                    "ilu_fill_ratio": context.ilu.diagnostics["factor_fill_ratio"],
                    "ilu_factor_memory_mib": context.ilu.diagnostics[
                        "factor_memory_bytes"
                    ]
                    / 2**20,
                    "arnoldi_dimension": settings.arnoldi_dimension,
                    "requested_q": settings.requested_q,
                    "actual_q": construction.actual_q,
                    "q_over_n": construction.actual_q / context.system.matrix.shape[0],
                    **baseline_rows[rhs_seed],
                    "recycle_iterations": result.iterations,
                    "recycle_converged": result.converged,
                    "recycle_true_final_relative_residual": true_residual,
                    "recycle_solve_seconds_excluding_setup": result.solve_seconds,
                    "recycle_solve_seconds_including_setup": result.solve_seconds
                    + construction.total_setup_seconds,
                    "recycle_preconditioner_apply_seconds": context.ilu.apply_seconds,
                    "recycle_preconditioner_applications": context.ilu.apply_count,
                    "recycle_a_matvecs": (result.operation_counts or {}).get(
                        "a_matvecs", 0
                    ),
                    "iteration_absolute_reduction": baseline.iterations
                    - result.iterations,
                    "iteration_ratio": iteration_ratio,
                    "J_iter": reduction,
                    "ritz_setup_seconds": construction.ritz_seconds,
                    "pair_setup_seconds": construction.pair_seconds,
                    "total_recycle_setup_seconds": construction.total_setup_seconds,
                    "total_seconds_excluding_common_ilu_setup": result.solve_seconds
                    + construction.total_setup_seconds,
                    "total_seconds_including_common_ilu_setup": result.solve_seconds
                    + construction.total_setup_seconds
                    + ilu_setup_seconds,
                    "recycle_failure": "" if result.converged else "maximum iterations reached",
                }
                per_rhs_rows.append(row)
                local_rows.append(row)

            performance_by_construction[construction_seed] = {
                "iterations_median": float(
                    np.median([int(row["recycle_iterations"]) for row in local_rows])
                ),
                "iteration_ratio_median": float(
                    np.median([float(row["iteration_ratio"]) for row in local_rows])
                ),
                "J_iter_median": float(
                    np.median([float(row["J_iter"]) for row in local_rows])
                ),
                "all_converged": all(bool(row["recycle_converged"]) for row in local_rows),
            }

        for first_index, first in enumerate(constructions):
            for second in constructions[first_index + 1 :]:
                z_metrics = generalized_subspace_metrics(first.basis, second.basis)
                w_metrics = generalized_subspace_metrics(
                    first.factors.test, second.factors.test
                )
                perf_first = performance_by_construction[first.rhs_seed]
                perf_second = performance_by_construction[second.rhs_seed]
                per_rhs_difference = []
                for rhs_seed in settings.heldout_rhs_seeds:
                    first_row = next(
                        row
                        for row in per_rhs_rows
                        if int(row["sample_id"]) == sample_id
                        and int(row["construction_rhs_seed"]) == first.rhs_seed
                        and int(row["heldout_rhs_seed"]) == rhs_seed
                    )
                    second_row = next(
                        row
                        for row in per_rhs_rows
                        if int(row["sample_id"]) == sample_id
                        and int(row["construction_rhs_seed"]) == second.rhs_seed
                        and int(row["heldout_rhs_seed"]) == rhs_seed
                    )
                    per_rhs_difference.append(
                        abs(
                            float(first_row["iteration_ratio"])
                            - float(second_row["iteration_ratio"])
                        )
                    )
                geometry_rows.append(
                    {
                        "sample_id": sample_id,
                        "difficulty_class": difficulty,
                        "provenance_cohort": provenance,
                        "construction_rhs_seed_i": first.rhs_seed,
                        "construction_rhs_seed_j": second.rhs_seed,
                        "q_i": first.actual_q,
                        "q_j": second.actual_q,
                        "d_Z": z_metrics["projector_distance_normalized"],
                        "d_W": w_metrics["projector_distance_normalized"],
                        "projector_distance_Z": z_metrics["projector_distance"],
                        "projector_distance_W": w_metrics["projector_distance"],
                        "max_principal_angle_Z_degrees": z_metrics[
                            "principal_angle_max_degrees"
                        ],
                        "max_principal_angle_W_degrees": w_metrics[
                            "principal_angle_max_degrees"
                        ],
                        "mean_principal_angle_Z_degrees": z_metrics[
                            "principal_angle_mean_degrees"
                        ],
                        "mean_principal_angle_W_degrees": w_metrics[
                            "principal_angle_mean_degrees"
                        ],
                        "construction_i_iterations_median": perf_first[
                            "iterations_median"
                        ],
                        "construction_j_iterations_median": perf_second[
                            "iterations_median"
                        ],
                        "construction_i_J_iter_median": perf_first["J_iter_median"],
                        "construction_j_J_iter_median": perf_second["J_iter_median"],
                        "median_iterations_absolute_difference": abs(
                            perf_first["iterations_median"]
                            - perf_second["iterations_median"]
                        ),
                        "median_iteration_ratio_difference": abs(
                            perf_first["iteration_ratio_median"]
                            - perf_second["iteration_ratio_median"]
                        ),
                        "median_J_iter_difference": abs(
                            perf_first["J_iter_median"] - perf_second["J_iter_median"]
                        ),
                        "max_per_rhs_iteration_ratio_difference": max(
                            per_rhs_difference
                        ),
                        "both_constructions_converged_all_rhs": bool(
                            perf_first["all_converged"]
                            and perf_second["all_converged"]
                        ),
                    }
                )

        primary_seed = settings.construction_rhs_seeds[0]
        primary_rows = [
            row
            for row in per_rhs_rows
            if int(row["sample_id"]) == sample_id
            and int(row["construction_rhs_seed"]) == primary_seed
        ]
        baseline_iterations = [int(row["baseline_iterations"]) for row in primary_rows]
        recycle_iterations = [int(row["recycle_iterations"]) for row in primary_rows]
        reductions = [float(row["J_iter"]) for row in primary_rows]
        convergence = [bool(row["recycle_converged"]) for row in primary_rows]
        baseline_seconds = float(
            np.median([float(row["baseline_solve_seconds"]) for row in primary_rows])
        )
        recycle_seconds = float(
            np.median(
                [float(row["recycle_solve_seconds_excluding_setup"]) for row in primary_rows]
            )
        )
        construction = constructions[0]
        break_even = estimate_break_even(
            construction.total_setup_seconds, baseline_seconds, recycle_seconds
        )
        summary_rows.append(
            {
                "sample_id": sample_id,
                "difficulty_class": difficulty,
                "provenance_cohort": provenance,
                "numerically_valid": True,
                "n": context.system.matrix.shape[0],
                "nnz": context.system.matrix.nnz,
                "assembly_seconds": assembly_seconds,
                "ilu_setup_seconds": ilu_setup_seconds,
                "ilu_fill_ratio": context.ilu.diagnostics["factor_fill_ratio"],
                "ilu_factor_memory_mib": context.ilu.diagnostics[
                    "factor_memory_bytes"
                ]
                / 2**20,
                "baseline_iterations_min": min(baseline_iterations),
                "baseline_iterations_median": float(np.median(baseline_iterations)),
                "baseline_iterations_max": max(baseline_iterations),
                "baseline_convergence_rate": float(
                    np.mean([bool(row["baseline_converged"]) for row in primary_rows])
                ),
                "baseline_true_residual_max": max(
                    float(row["baseline_true_final_relative_residual"])
                    for row in primary_rows
                ),
                "baseline_solve_seconds_median": baseline_seconds,
                "requested_q": settings.requested_q,
                "actual_q": construction.actual_q,
                "q_over_n": construction.actual_q / context.system.matrix.shape[0],
                "recycle_iterations_min": min(recycle_iterations),
                "recycle_iterations_median": float(np.median(recycle_iterations)),
                "recycle_iterations_max": max(recycle_iterations),
                "recycle_convergence_rate": float(np.mean(convergence)),
                "recycle_true_residual_max": max(
                    float(row["recycle_true_final_relative_residual"])
                    for row in primary_rows
                ),
                "recycle_solve_seconds_median": recycle_seconds,
                "iteration_absolute_reduction_median": float(
                    np.median(
                        [int(row["iteration_absolute_reduction"]) for row in primary_rows]
                    )
                ),
                "iteration_ratio_median": float(
                    np.median([float(row["iteration_ratio"]) for row in primary_rows])
                ),
                "J_iter_median": float(np.median(reductions)),
                "system_gate_passed": system_gate(convergence, reductions, settings),
                "ritz_setup_seconds": construction.ritz_seconds,
                "pair_setup_seconds": construction.pair_seconds,
                "total_recycle_setup_seconds": construction.total_setup_seconds,
                "break_even_rhs_count": break_even,
                "break_even_is_finite": bool(np.isfinite(break_even)),
                "break_even_censored_baseline": not all(
                    bool(row["baseline_converged"]) for row in primary_rows
                ),
                "reproduces_stage19a_all_rhs": all(
                    bool(row["baseline_reproduces_stage19a"]) for row in primary_rows
                ),
                "au_equals_w_relative_error": construction.factors.diagnostics[
                    "au_equals_w_relative_error"
                ],
                "test_orthogonality_error": construction.factors.diagnostics[
                    "test_orthogonality_error"
                ],
                "reduced_condition": construction.factors.diagnostics[
                    "reduced_condition"
                ],
            }
        )

    _write_csv(output / "population_solver_results.csv", per_rhs_rows)
    _write_csv(output / "construction_results.csv", construction_rows)
    _write_csv(output / "construction_geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "system_summary.csv", summary_rows)
    return per_rhs_rows, construction_rows, geometry_rows, summary_rows


def _ranking_rows(
    geometry_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics = (
        "d_Z",
        "d_W",
        "max_principal_angle_Z_degrees",
        "max_principal_angle_W_degrees",
    )
    target = "median_iteration_ratio_difference"
    scopes = {
        "all_valid": lambda row: True,
        "hard": lambda row: str(row["difficulty_class"]) == "hard",
        "unseen_hard": lambda row: str(row["difficulty_class"]) == "hard"
        and str(row["provenance_cohort"]) == "unseen",
        "stage20_hard": lambda row: str(row["difficulty_class"]) == "hard"
        and str(row["provenance_cohort"]) == "stage20_diagnostic",
        "easy": lambda row: str(row["difficulty_class"]) == "easy",
    }
    correlation_rows: List[Dict[str, Any]] = []
    system_rows: List[Dict[str, Any]] = []
    sample_ids = sorted({int(row["sample_id"]) for row in geometry_rows})
    for scope, include in scopes.items():
        scoped = [row for row in geometry_rows if include(row)]
        for metric in metrics:
            coefficient, pvalue, count = _safe_spearman(
                [float(row[metric]) for row in scoped],
                [float(row[target]) for row in scoped],
            )
            correlation_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "performance_target": target,
                    "spearman_coefficient": coefficient,
                    "spearman_pvalue": pvalue,
                    "pair_count": count,
                }
            )

    for metric in metrics:
        per_system = []
        closest_matches = []
        for sample_id in sample_ids:
            local = [row for row in geometry_rows if int(row["sample_id"]) == sample_id]
            local_coefficient, local_pvalue, local_count = _safe_spearman(
                [float(row[metric]) for row in local],
                [float(row[target]) for row in local],
            )
            metric_values = np.asarray([float(row[metric]) for row in local])
            target_values = np.asarray([float(row[target]) for row in local])
            closest_index = int(np.argmin(metric_values))
            minimum_target = float(np.min(target_values))
            correct = bool(target_values[closest_index] <= minimum_target + 1.0e-12)
            closest_matches.append(correct)
            if np.isfinite(local_coefficient):
                per_system.append(local_coefficient)
            system_rows.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "spearman_coefficient": local_coefficient,
                    "spearman_pvalue": local_pvalue,
                    "pair_count": local_count,
                    "closest_geometry_pair_has_smallest_performance_difference": correct,
                }
            )
        correlation_rows.append(
            {
                "scope": "system_by_system_ranking",
                "metric": metric,
                "performance_target": target,
                "spearman_coefficient": float("nan"),
                "spearman_pvalue": float("nan"),
                "pair_count": len(geometry_rows),
                "systems_with_defined_spearman": len(per_system),
                "median_system_spearman": float(np.median(per_system))
                if per_system
                else float("nan"),
                "closest_pair_ranking_accuracy": float(np.mean(closest_matches))
                if closest_matches
                else float("nan"),
            }
        )
    return correlation_rows, system_rows


def _aggregate(
    output: Path,
    settings: Stage21Settings,
    per_rhs_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    gate = population_gate(summary_rows, settings)
    correlations, ranking = _ranking_rows(geometry_rows)
    _write_csv(output / "geometry_performance_correlations.csv", correlations)
    _write_csv(output / "system_geometry_ranking.csv", ranking)

    primary = [row for row in per_rhs_rows if _bool(row["primary_construction"])]
    easy = [row for row in summary_rows if str(row["difficulty_class"]) == "easy"]
    hard = [row for row in summary_rows if str(row["difficulty_class"]) == "hard"]
    unseen_hard = [
        row
        for row in hard
        if str(row["provenance_cohort"]) == "unseen"
    ]
    stage20_hard = [
        row
        for row in hard
        if str(row["provenance_cohort"]) == "stage20_diagnostic"
    ]
    def geometry_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        z_distances = np.asarray([float(row["d_Z"]) for row in rows])
        w_distances = np.asarray([float(row["d_W"]) for row in rows])
        z_angles = np.asarray(
            [float(row["max_principal_angle_Z_degrees"]) for row in rows]
        )
        w_angles = np.asarray(
            [float(row["max_principal_angle_W_degrees"]) for row in rows]
        )
        return {
            "pair_count": len(rows),
            "d_Z_median": float(np.median(z_distances)),
            "d_W_median": float(np.median(w_distances)),
            "d_W_less_than_d_Z_fraction": float(np.mean(w_distances < z_distances)),
            "max_angle_Z_median_degrees": float(np.median(z_angles)),
            "max_angle_W_median_degrees": float(np.median(w_angles)),
            "max_angle_W_less_than_Z_fraction": float(np.mean(w_angles < z_angles)),
        }

    hard_geometry = [
        row for row in geometry_rows if str(row["difficulty_class"]) == "hard"
    ]
    unseen_hard_geometry = [
        row
        for row in hard_geometry
        if str(row["provenance_cohort"]) == "unseen"
    ]
    primary_q = [int(row["actual_q"]) for row in summary_rows]

    construction_gate_rows = []
    for system in summary_rows:
        sample_id = int(system["sample_id"])
        local = [row for row in per_rhs_rows if int(row["sample_id"]) == sample_id]
        passes = []
        medians = []
        for construction_seed in settings.construction_rhs_seeds:
            construction_local = [
                row
                for row in local
                if int(row["construction_rhs_seed"]) == construction_seed
            ]
            reductions = [float(row["J_iter"]) for row in construction_local]
            passes.append(
                system_gate(
                    [_bool(row["recycle_converged"]) for row in construction_local],
                    reductions,
                    settings,
                )
            )
            medians.append(float(np.median(reductions)))
        construction_gate_rows.append(
            {
                "sample_id": sample_id,
                "difficulty_class": system["difficulty_class"],
                "provenance_cohort": system["provenance_cohort"],
                "construction_gates_passed": int(sum(passes)),
                "all_three_construction_gates_passed": bool(all(passes)),
                "J_iter_across_constructions_min": min(medians),
                "J_iter_across_constructions_max": max(medians),
            }
        )
    _write_csv(output / "construction_stability_summary.csv", construction_gate_rows)
    aggregate = {
        "population": {
            "valid_systems": len(summary_rows),
            "invalid_systems": 0,
            "easy_systems": len(easy),
            "transition_systems": sum(
                str(row["difficulty_class"]) == "transition" for row in summary_rows
            ),
            "hard_systems": len(hard),
            "stage20_hard_systems": len(stage20_hard),
            "unseen_hard_systems": len(unseen_hard),
            "heldout_rhs_per_system": len(settings.heldout_rhs_seeds),
            "independent_constructions_per_system": len(
                settings.construction_rhs_seeds
            ),
        },
        "unseen_population_gate": gate,
        "all_hard": {
            "systems_passing": sum(_bool(row["system_gate_passed"]) for row in hard),
            "pass_fraction": float(
                np.mean([_bool(row["system_gate_passed"]) for row in hard])
            ),
            "median_iteration_ratio": float(
                np.median([float(row["iteration_ratio_median"]) for row in hard])
            ),
            "median_J_iter": float(
                np.median([float(row["J_iter_median"]) for row in hard])
            ),
        },
        "easy_systems": {
            "systems_passing_40pct_gate": sum(
                _bool(row["system_gate_passed"]) for row in easy
            ),
            "median_baseline_iterations": float(
                np.median([float(row["baseline_iterations_median"]) for row in easy])
            ),
            "median_recycle_iterations": float(
                np.median([float(row["recycle_iterations_median"]) for row in easy])
            ),
            "median_absolute_reduction": float(
                np.median(
                    [float(row["iteration_absolute_reduction_median"]) for row in easy]
                )
            ),
        },
        "rank": {
            "requested_q": settings.requested_q,
            "actual_q_min": min(primary_q),
            "actual_q_max": max(primary_q),
            "q_over_n_min": min(float(row["q_over_n"]) for row in summary_rows),
            "q_over_n_max": max(float(row["q_over_n"]) for row in summary_rows),
        },
        "construction_geometry": {
            "all_valid": geometry_summary(geometry_rows),
            "hard": geometry_summary(hard_geometry),
            "unseen_hard": geometry_summary(unseen_hard_geometry),
        },
        "construction_functional_stability": {
            "hard_systems_passing_all_three_constructions": sum(
                _bool(row["all_three_construction_gates_passed"])
                for row in construction_gate_rows
                if str(row["difficulty_class"]) == "hard"
            ),
            "unseen_hard_systems_passing_all_three_constructions": sum(
                _bool(row["all_three_construction_gates_passed"])
                for row in construction_gate_rows
                if str(row["difficulty_class"]) == "hard"
                and str(row["provenance_cohort"]) == "unseen"
            ),
        },
        "timing": {
            "ritz_setup_seconds_median": float(
                np.median([float(row["ritz_setup_seconds"]) for row in summary_rows])
            ),
            "pair_setup_seconds_median": float(
                np.median([float(row["pair_setup_seconds"]) for row in summary_rows])
            ),
            "total_setup_seconds_median": float(
                np.median(
                    [float(row["total_recycle_setup_seconds"]) for row in summary_rows]
                )
            ),
            "finite_break_even_systems": int(
                sum(_bool(row["break_even_is_finite"]) for row in summary_rows)
            ),
            "finite_break_even_median_rhs": float(
                np.median(
                    [
                        float(row["break_even_rhs_count"])
                        for row in summary_rows
                        if _bool(row["break_even_is_finite"])
                    ]
                )
            )
            if any(_bool(row["break_even_is_finite"]) for row in summary_rows)
            else float("nan"),
        },
        "correlations": {
            f"{row['scope']}:{row['metric']}": row for row in correlations
        },
        "true_residual_check": {
            "converged_primary_rows": sum(
                _bool(row["recycle_converged"]) for row in primary
            ),
            "max_true_residual_among_converged": max(
                float(row["recycle_true_final_relative_residual"])
                for row in primary
                if _bool(row["recycle_converged"])
            ),
        },
    }
    _write_json(output / "aggregate_results.json", aggregate)
    return aggregate


def _format_axis(ax) -> None:
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_results(
    output: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # 1. Baseline versus recycle iterations.
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for provenance, color, marker, label in (
        ("unseen", BLUE, "o", "Unseen after Stage 20"),
        ("stage20_diagnostic", ORANGE, "s", "Stage-20 diagnostic set"),
    ):
        local = [row for row in summary_rows if row["provenance_cohort"] == provenance]
        ax.scatter(
            [float(row["baseline_iterations_median"]) for row in local],
            [float(row["recycle_iterations_median"]) for row in local],
            color=color,
            marker=marker,
            s=54,
            alpha=0.88,
            label=label,
        )
    ax.plot([0, 310], [0, 310], color=GRAY, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 310)
    ax.set_xlabel("Strong-ILU GMRES median iterations")
    ax.set_ylabel("Frozen-oracle recycled GMRES median iterations")
    ax.set_title("Baseline versus frozen-oracle iterations")
    ax.legend(frameon=False, loc="upper left")
    _format_axis(ax)
    fig.savefig(plots / "baseline_vs_recycle_iterations.png", dpi=180)
    plt.close(fig)

    # 2. Distribution of reductions, separated by decision-relevant cohort.
    groups = [
        (
            "Easy",
            [float(row["J_iter_median"]) for row in summary_rows if row["difficulty_class"] == "easy"],
        ),
        (
            "Unseen hard",
            [
                float(row["J_iter_median"])
                for row in summary_rows
                if row["difficulty_class"] == "hard"
                and row["provenance_cohort"] == "unseen"
            ],
        ),
        (
            "Stage-20 hard",
            [
                float(row["J_iter_median"])
                for row in summary_rows
                if row["difficulty_class"] == "hard"
                and row["provenance_cohort"] == "stage20_diagnostic"
            ],
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    box = ax.boxplot(
        [values for _, values in groups],
        labels=[label for label, _ in groups],
        patch_artist=True,
        widths=0.55,
        showmeans=True,
    )
    for patch, color in zip(box["boxes"], (OLIVE, BLUE, ORANGE)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0.40, color=PINK, linestyle="--", linewidth=1.4, label="System gate (40%)")
    ax.set_ylabel("Median relative iteration reduction $J_{iter}$")
    ax.set_title("Frozen-oracle reduction distribution")
    ax.legend(frameon=False)
    _format_axis(ax)
    fig.savefig(plots / "oracle_reduction_distribution.png", dpi=180)
    plt.close(fig)

    # 3-4. Geometry versus functional performance difference.
    for metric, filename, xlabel in (
        ("d_Z", "dZ_vs_performance_difference.png", "Normalized solution-space distance $d_Z$"),
        ("d_W", "dW_vs_performance_difference.png", "Normalized residual-space distance $d_W$"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for provenance, color, marker, label in (
            ("unseen", BLUE, "o", "Unseen"),
            ("stage20_diagnostic", ORANGE, "s", "Stage-20 set"),
        ):
            local = [row for row in geometry_rows if row["provenance_cohort"] == provenance]
            ax.scatter(
                [float(row[metric]) for row in local],
                [float(row["median_iteration_ratio_difference"]) for row in local],
                color=color,
                marker=marker,
                s=42,
                alpha=0.72,
                label=label,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absolute difference in median iteration ratio")
        ax.set_title("Construction geometry versus solver-performance difference")
        ax.legend(frameon=False)
        _format_axis(ax)
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    # 5. Z/W maximum principal angles.
    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    z_angle = [float(row["max_principal_angle_Z_degrees"]) for row in geometry_rows]
    w_angle = [float(row["max_principal_angle_W_degrees"]) for row in geometry_rows]
    colors = [
        ORANGE if row["provenance_cohort"] == "stage20_diagnostic" else BLUE
        for row in geometry_rows
    ]
    ax.scatter(z_angle, w_angle, c=colors, s=42, alpha=0.72)
    ax.plot([0, 90], [0, 90], color=GRAY, linestyle="--", linewidth=1.2)
    ax.set_xlim(-2, 92)
    ax.set_ylim(-2, 92)
    ax.set_xlabel("Maximum principal angle in $Z$ (degrees)")
    ax.set_ylabel("Maximum principal angle in $W$ (degrees)")
    ax.set_title("Construction-RHS principal-angle stability")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="Unseen"),
            Line2D(
                [0], [0], marker="o", linestyle="", color=ORANGE, label="Stage-20 set"
            ),
        ],
        frameon=False,
        loc="lower right",
    )
    _format_axis(ax)
    fig.savefig(plots / "max_principal_angle_Z_vs_W.png", dpi=180)
    plt.close(fig)

    # 6. Setup and repeated-RHS break-even summary.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    ordering = sorted(summary_rows, key=lambda row: int(row["sample_id"]))
    axes[0].scatter(
        [int(row["sample_id"]) for row in ordering],
        [float(row["ritz_setup_seconds"]) for row in ordering],
        color=BLUE,
        s=32,
        label="Arnoldi/Ritz",
    )
    axes[0].scatter(
        [int(row["sample_id"]) for row in ordering],
        [float(row["pair_setup_seconds"]) for row in ordering],
        color=ORANGE,
        s=32,
        label="$AZ=WR$ pairing",
    )
    axes[0].set_xlabel("System ID")
    axes[0].set_ylabel("Setup time (s)")
    axes[0].set_title("Frozen-oracle setup cost")
    axes[0].legend(frameon=False)
    _format_axis(axes[0])
    finite = [row for row in ordering if _bool(row["break_even_is_finite"])]
    categories = (
        (
            "Converged time-to-solution",
            lambda row: float(row["baseline_convergence_rate"]) == 1.0
            and float(row["recycle_convergence_rate"]) == 1.0,
            ORANGE,
            "s",
        ),
        (
            "Censored baseline budget",
            lambda row: float(row["baseline_convergence_rate"]) < 1.0
            and float(row["recycle_convergence_rate"]) == 1.0,
            OLIVE,
            "o",
        ),
        (
            "Recycle not fully converged",
            lambda row: float(row["recycle_convergence_rate"]) < 1.0,
            PINK,
            "x",
        ),
    )
    for label, include, color, marker in categories:
        local = [row for row in finite if include(row)]
        if not local:
            continue
        axes[1].scatter(
            [int(row["sample_id"]) for row in local],
            [float(row["break_even_rhs_count"]) for row in local],
            color=color,
            marker=marker,
            s=42,
            label=label,
        )
    axes[1].set_xlabel("System ID")
    axes[1].set_ylabel("Estimated RHS count")
    axes[1].set_title("Incremental online break-even")
    axes[1].legend(frameon=False, fontsize=9)
    _format_axis(axes[1])
    fig.savefig(plots / "setup_cost_break_even.png", dpi=180)
    plt.close(fig)


def _write_chart_map(output: Path) -> None:
    rows = [
        {
            "figure": "baseline_vs_recycle_iterations.png",
            "report_segment": "population validation",
            "question": "Does the frozen oracle improve all saved systems?",
            "chart_type": "scatter with equality line",
            "fields": "baseline_iterations_median,recycle_iterations_median,provenance_cohort",
            "supported_claim": "iteration improvement and cap failures by seen/unseen cohort",
        },
        {
            "figure": "oracle_reduction_distribution.png",
            "report_segment": "predeclared gate",
            "question": "How does oracle reduction differ across easy and hard cohorts?",
            "chart_type": "box plot",
            "fields": "difficulty_class,provenance_cohort,J_iter_median",
            "supported_claim": "distribution relative to the 40% system gate",
        },
        {
            "figure": "dZ_vs_performance_difference.png",
            "report_segment": "construction stability",
            "question": "Does Z-space geometry predict functional differences?",
            "chart_type": "scatter",
            "fields": "d_Z,median_iteration_ratio_difference",
            "supported_claim": "rank association of Z distance with solver difference",
        },
        {
            "figure": "dW_vs_performance_difference.png",
            "report_segment": "construction stability",
            "question": "Does W-space geometry predict functional differences?",
            "chart_type": "scatter",
            "fields": "d_W,median_iteration_ratio_difference",
            "supported_claim": "rank association of W distance with solver difference",
        },
        {
            "figure": "max_principal_angle_Z_vs_W.png",
            "report_segment": "construction stability",
            "question": "Is W geometrically more stable than Z?",
            "chart_type": "paired scatter with equality line",
            "fields": "max_principal_angle_Z_degrees,max_principal_angle_W_degrees",
            "supported_claim": "pairwise maximum-angle comparison",
        },
        {
            "figure": "setup_cost_break_even.png",
            "report_segment": "cost",
            "question": "How costly is construction and when can it amortize?",
            "chart_type": "two-panel dot plot",
            "fields": "ritz_setup_seconds,pair_setup_seconds,break_even_rhs_count",
            "supported_claim": "measured setup and timing-based break-even",
        },
    ]
    _write_csv(output / "chart_map.csv", rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output / "manifest.json",
        {
            "stage": "stage21_spatial_K_population_validation",
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )


def _validate(
    root: Path,
    output: Path,
    report: Path,
    settings: Stage21Settings,
    per_rhs_rows: Sequence[Mapping[str, Any]],
    construction_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, bool]:
    _, _, _, _, archived_summaries, _, valid_ids, invalid_ids = _load_population(
        root, settings
    )
    primary = [row for row in per_rhs_rows if _bool(row["primary_construction"])]
    converged_primary = [row for row in primary if _bool(row["recycle_converged"])]
    stage20_reference_path = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage20_spatial_K_robust_oracle_search"
        / "arnoldi_rank_ablation.csv"
    )
    stage20_reference = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in _read_rows(stage20_reference_path)
        if row["method"] == "ordinary_m128_q24"
    }
    stage20_matches = []
    for row in primary:
        key = (int(row["sample_id"]), int(row["heldout_rhs_seed"]))
        if key in stage20_reference:
            stage20_matches.append(
                int(row["recycle_iterations"])
                == int(stage20_reference[key]["iterations"])
                and _bool(row["recycle_converged"])
                == _bool(stage20_reference[key]["converged"])
            )
    required_plots = (
        "baseline_vs_recycle_iterations.png",
        "oracle_reduction_distribution.png",
        "dZ_vs_performance_difference.png",
        "dW_vs_performance_difference.png",
        "max_principal_angle_Z_vs_W.png",
        "setup_cost_break_even.png",
    )
    checks = {
        "all_archived_valid_systems_included": {int(row["sample_id"]) for row in summary_rows}
        == set(valid_ids),
        "invalid_systems_excluded": not any(
            int(row["sample_id"]) in invalid_ids for row in summary_rows
        ),
        "all_36_valid_systems_present": len(summary_rows) == len(valid_ids) == 36,
        "three_heldout_rhs_for_each_primary_system": len(primary)
        == len(valid_ids) * len(settings.heldout_rhs_seeds),
        "three_constructions_for_each_system": len(construction_rows)
        == len(valid_ids) * len(settings.construction_rhs_seeds),
        "three_geometry_pairs_for_each_system": len(geometry_rows)
        == len(valid_ids) * 3,
        "construction_and_evaluation_rhs_disjoint": all(
            _bool(row["construction_and_evaluation_disjoint"]) for row in per_rhs_rows
        ),
        "baseline_reproduces_stage19a": all(
            _bool(row["baseline_reproduces_stage19a"]) for row in primary
        ),
        "stage20_m128_q24_reproduced": bool(stage20_matches)
        and all(stage20_matches)
        and len(stage20_matches)
        == len(settings.stage20_ids) * len(settings.heldout_rhs_seeds),
        "true_residual_stopping_preserved": all(
            float(row["recycle_true_final_relative_residual"])
            <= 1.1 * settings.solver.relative_tolerance
            for row in converged_primary
        ),
        "paired_recycle_identities_valid": all(
            float(row["au_equals_w_relative_error"]) <= 1.0e-10
            and float(row["test_orthogonality_error"]) <= 1.0e-10
            for row in construction_rows
        ),
        "requested_oracle_frozen": all(
            int(row["arnoldi_dimension"]) == settings.arnoldi_dimension
            and int(row["requested_q"]) == settings.requested_q
            for row in construction_rows
        ),
        "required_plots_exist": all((output / "plots" / name).is_file() for name in required_plots),
        "report_exists": report.is_file(),
        "supporting_tests_pass": (output / "test_results.txt").is_file()
        and "Ran 43 tests" in (output / "test_results.txt").read_text(encoding="utf-8")
        and (output / "test_results.txt").read_text(encoding="utf-8").rstrip().endswith("OK"),
        "no_neural_artifacts": not any(
            "model" in path.name.lower() or path.suffix in {".pt", ".pth"}
            for path in output.rglob("*")
        ),
    }
    _write_json(
        output / "validation.json",
        {
            "assessment": "ready_to_share" if all(checks.values()) else "needs_revision",
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "archived_valid_count": len(valid_ids),
            "archived_invalid_count": len(invalid_ids),
            "stage20_reproduction_rows": len(stage20_matches),
            "caveat": "Timing-based break-even for capped baseline solves is censored and optimistic; counts are preserved rather than extrapolated.",
        },
    )
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Stage 21 validation failed: {failed}")
    return checks


def run_stage21(root: Path) -> Path:
    """Run the frozen population experiment and save auditable raw outputs."""

    settings = Stage21Settings()
    fd_root = root / "results" / "results-fd"
    output = fd_root / "stages" / "stage21_spatial_K_population_validation"
    report = fd_root / "reports" / "stage21_spatial_K_population_validation.md"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)

    # This file is deliberately written before any Stage-21 oracle solve.  It
    # records the fixed algorithm and gates against outcome-dependent changes.
    predeclared = {
        "written_before_stage21_oracle_execution": True,
        "settings": asdict(settings),
        "cohort_definition": {
            "easy": "archived Stage-19A median strong-ILU iterations < 50",
            "transition": "50 <= archived median iterations < 200",
            "hard": "archived median iterations >= 200",
            "stage20_diagnostic": list(settings.stage20_ids),
            "unseen": "every other numerically valid Stage-19A system",
        },
        "system_gate": "all three held-out RHS converge and median J_iter >= 0.40",
        "unseen_population_gate": "at least 70% of unseen hard/transition systems pass and their median recycled/baseline iteration ratio <= 0.65",
        "primary_construction_rhs": settings.construction_rhs_seeds[0],
        "stability_only_construction_rhs": list(settings.construction_rhs_seeds[1:]),
    }
    _write_json(output / "predeclared_protocol.json", predeclared)
    _write_json(
        output / "settings.json",
        {
            **asdict(settings),
            "solver": asdict(settings.solver),
            "ilu": asdict(settings.ilu_config),
            "discretization": asdict(settings.discretization),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )

    per_rhs, constructions, geometry, summaries = _run_population(
        root, output, settings
    )
    aggregate = _aggregate(output, settings, per_rhs, geometry, summaries)
    _plot_results(output, summaries, geometry)
    _write_chart_map(output)
    _write_json(
        output / "decision.json",
        {
            "population_gate_passed": aggregate["unseen_population_gate"][
                "population_gate_passed"
            ],
            "decision_pending_representation_analysis": True,
            "allowed_decisions": {
                "A": "Oracle validation passed; proceed to Stage 22 transfer.",
                "B": "Oracle validation failed; return to target construction.",
                "C": "Oracle validation passed, but no stable functional representation was identified; transfer requires a representation study first.",
            },
        },
    )
    # The report is generated by the companion finalizer after the saved data
    # have passed numerical and visual QA.
    return output


__all__ = [
    "Stage21Settings",
    "classify_difficulty",
    "estimate_break_even",
    "population_gate",
    "run_stage21",
    "system_gate",
    "write_manifest",
    "_validate",
]
