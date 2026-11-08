"""Stage 22A: multi-RHS functional oracle for spatially varying K(x).

The experiment keeps the Stage-21 RBF-FD matrices, strong ILU action,
right-preconditioned Arnoldi convention, recycled GMRES implementation, true
residual stopping rule, and evaluation RHSs fixed.  It changes only the
construction information available to the own-system oracle.

No learned model and no cross-system transfer appear in this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from neurkitt_oracle.gcrodr import GmresResult, restarted_projected_gmres

from ..preconditioners import InverseAction
from ..subspace_analysis import (
    generalized_subspace_metrics,
    nested_krylov_ritz_modes,
    ritz_atomic_blocks,
)
from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage20_spatial_k_robust_oracle import (
    SystemContext,
    _complete_block_indices,
    _harmonic_from_arnoldi,
    _relative_residual,
)
from .stage21_population_validation import (
    Stage21Settings,
    _archived_median,
    _assemble_context,
    _bool,
    _load_population,
    classify_difficulty,
    estimate_break_even,
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
class Stage22ASettings:
    """Predeclared Stage-22A construction grid, data split, and gates."""

    construction_rhs_seeds: Tuple[int, ...] = (
        20261001,
        20261011,
        20261021,
        20261031,
    )
    selection_rhs_seeds: Tuple[int, ...] = (20261041, 20261051)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    n_construct_values: Tuple[int, ...] = (1, 2, 3, 4)
    q_values: Tuple[int, ...] = (16, 24, 32, 48, 64)
    primary_arnoldi_dimension: int = 128
    candidates_per_rhs: int = 24
    extraction_buffer: int = 32
    functional_greedy_n_construct: int = 4
    cost_matched_q_values: Tuple[int, ...] = (16, 24)
    system_reduction_gate: float = 0.40
    family_pass_fraction_gate: float = 0.80
    numerical_rank_tolerance: float = 1.0e-11
    functional_log_floor: float = 1.0e-12
    harmonic_secondary_q: int = 64
    seed: int = 20262201

    @property
    def stage21(self) -> Stage21Settings:
        return Stage21Settings()

    @property
    def solver(self):
        return self.stage21.solver

    @property
    def hard_transition_required(self) -> int:
        # The actual denominator is written before the run after the Stage-21
        # archived ILU-only classification is loaded.
        return 12


@dataclass
class AtomicRitzBlock:
    construction_seed: int
    source_order: int
    basis: np.ndarray
    eigenvalue_magnitude: float

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


@dataclass
class RitzConstruction:
    rhs_seed: int
    arnoldi_dimension: int
    basis: np.ndarray
    blocks: Tuple[AtomicRitzBlock, ...]
    eigenvalues: np.ndarray
    setup_seconds: float
    a_applications: int
    m_applications: int
    basis_orthogonality_error: float
    extraction: str = "ordinary_ritz"


@dataclass
class RecyclePair:
    """A paired nonsymmetric recycle object satisfying ``A U = W``."""

    z_basis: np.ndarray
    u_basis: np.ndarray
    w_basis: np.ndarray
    setup_seconds: float
    diagnostics: Dict[str, float]

    @property
    def rank(self) -> int:
        return int(self.w_basis.shape[1])


def system_gate(converged: Sequence[bool], reductions: Sequence[float], settings: Stage22ASettings) -> bool:
    """Require all held-out RHSs and at least 40% median iteration reduction."""

    return bool(
        converged
        and reductions
        and all(bool(value) for value in converged)
        and float(np.median(reductions)) >= settings.system_reduction_gate
    )


def family_gate(system_rows: Sequence[Mapping[str, Any]], settings: Stage22ASettings) -> Dict[str, Any]:
    """Apply the predeclared 80% family gate to one fixed rule."""

    count = len(system_rows)
    passing = sum(_bool(row["system_gate_passed"]) for row in system_rows)
    fraction = passing / count if count else float("nan")
    return {
        "systems": count,
        "systems_passing": int(passing),
        "pass_fraction": float(fraction),
        "required_systems": int(math.ceil(settings.family_pass_fraction_gate * count)),
        "passed": bool(count and fraction >= settings.family_pass_fraction_gate),
    }


def _orthonormalize(raw: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("raw dictionary must be a nonempty matrix")
    left, singular, _ = la.svd(array, full_matrices=False, check_finite=False)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank == 0:
        raise RuntimeError("candidate dictionary has zero numerical rank")
    return left[:, :rank], singular, rank


def _build_ordinary_construction(
    context: SystemContext,
    rhs_seed: int,
    arnoldi_dimension: int,
    settings: Stage22ASettings,
) -> RitzConstruction:
    rhs, _ = _discrete_rhs(context.system, rhs_seed)
    modes = nested_krylov_ritz_modes(
        context.system.matrix,
        context.ilu,
        max_modes=settings.extraction_buffer,
        arnoldi_dimension=arnoldi_dimension,
        seed=settings.seed + context.sample_id + rhs_seed + arnoldi_dimension,
        initial_vector=rhs,
    )
    all_blocks = ritz_atomic_blocks(modes.eigenvalues)
    indices = _complete_block_indices(all_blocks, settings.candidates_per_rhs)
    last = max(indices) + 1
    basis = np.asarray(modes.basis[:, :last], dtype=float)
    values = np.asarray(modes.eigenvalues[:last], dtype=complex)
    blocks = tuple(
        AtomicRitzBlock(
            construction_seed=rhs_seed,
            source_order=order,
            basis=basis[:, start:stop],
            eigenvalue_magnitude=float(abs(values[start])),
        )
        for order, (start, stop) in enumerate(ritz_atomic_blocks(values), start=1)
    )
    return RitzConstruction(
        rhs_seed=rhs_seed,
        arnoldi_dimension=int(modes.diagnostics["arnoldi_dimension"]),
        basis=basis,
        blocks=blocks,
        eigenvalues=values,
        setup_seconds=float(modes.diagnostics["setup_seconds"]),
        a_applications=int(modes.diagnostics["arnoldi_dimension"]),
        m_applications=int(modes.diagnostics["preconditioner_applications"]),
        basis_orthogonality_error=float(modes.diagnostics["basis_orthogonality_error"]),
    )


def _build_harmonic_construction(
    context: SystemContext,
    rhs_seed: int,
    arnoldi_dimension: int,
    settings: Stage22ASettings,
) -> RitzConstruction:
    """Limited secondary construction from the same right Arnoldi relation."""

    rhs, _ = _discrete_rhs(context.system, rhs_seed)
    context.ilu.reset_timing()
    arnoldi = _capture_arnoldi(
        context.system.matrix, context.ilu, rhs, arnoldi_dimension
    )
    arnoldi_m_applications = int(arnoldi.hessenberg.shape[1])
    family = _harmonic_from_arnoldi(
        context.system.matrix, arnoldi, settings.extraction_buffer
    )
    indices = _complete_block_indices(family.blocks, settings.candidates_per_rhs)
    last = max(indices) + 1
    basis = np.asarray(family.basis[:, :last], dtype=float)
    values = np.asarray(family.eigenvalues[:last], dtype=complex)
    blocks = tuple(
        AtomicRitzBlock(
            construction_seed=rhs_seed,
            source_order=order,
            basis=basis[:, start:stop],
            eigenvalue_magnitude=float(abs(values[start])),
        )
        for order, (start, stop) in enumerate(ritz_atomic_blocks(values), start=1)
    )
    return RitzConstruction(
        rhs_seed=rhs_seed,
        arnoldi_dimension=int(arnoldi.hessenberg.shape[1]),
        basis=basis,
        blocks=blocks,
        eigenvalues=values,
        setup_seconds=float(family.construction_seconds),
        a_applications=int(arnoldi.hessenberg.shape[1]),
        m_applications=arnoldi_m_applications,
        basis_orthogonality_error=float(
            np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord="fro")
        ),
        extraction="harmonic_ritz",
    )


def _pair_from_z(matrix, z_basis: np.ndarray) -> RecyclePair:
    start = perf_counter()
    z, _, rank = _orthonormalize(z_basis, 1.0e-13)
    az = np.asarray(matrix @ z, dtype=float)
    w, triangular = la.qr(az, mode="economic", check_finite=False)
    condition = float(np.linalg.cond(triangular))
    u = la.solve_triangular(triangular.T, z.T, lower=True, check_finite=False).T
    au_error = float(
        np.linalg.norm(np.asarray(matrix @ u, dtype=float) - w, ord="fro")
        / max(np.linalg.norm(w, ord="fro"), np.finfo(float).tiny)
    )
    w_error = float(np.linalg.norm(w.T @ w - np.eye(rank), ord="fro"))
    return RecyclePair(
        z_basis=z,
        u_basis=u,
        w_basis=w,
        setup_seconds=perf_counter() - start,
        diagnostics={
            "au_equals_w_relative_error": au_error,
            "test_orthogonality_error": w_error,
            "reduced_condition": condition,
            "dictionary_rank": float(rank),
        },
    )


def _z_union(matrix, raw: np.ndarray, requested_q: int, tolerance: float) -> Tuple[RecyclePair, Dict[str, Any]]:
    start = perf_counter()
    left, singular, dictionary_rank = _orthonormalize(raw, tolerance)
    actual = min(requested_q, dictionary_rank)
    pair = _pair_from_z(matrix, left[:, :actual])
    compression_seconds = perf_counter() - start - pair.setup_seconds
    return pair, {
        "dictionary_rank": dictionary_rank,
        "dictionary_columns": raw.shape[1],
        "compression_seconds": compression_seconds,
        "pair_setup_seconds": pair.setup_seconds,
        "singular_value_min_retained": float(singular[actual - 1]),
        "singular_value_max": float(singular[0]),
        "a_applications_pair": actual,
    }


def _w_union(matrix, raw: np.ndarray, requested_q: int, tolerance: float) -> Tuple[RecyclePair, Dict[str, Any]]:
    """Compress ``A Z_raw`` and recover U algebraically inside ``span(Z_raw)``."""

    start = perf_counter()
    az_raw = np.asarray(matrix @ raw, dtype=float)
    w_all, singular, vh = la.svd(az_raw, full_matrices=False, check_finite=False)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank == 0:
        raise RuntimeError("A Z_raw has zero numerical rank")
    actual = min(requested_q, rank)
    w_target = w_all[:, :actual]
    u_raw = (raw @ vh[:actual, :].T) / singular[:actual]
    # The back-map through small singular values can amplify sparse-matvec
    # roundoff.  Re-pair the recovered trial action with a thin QR and a
    # triangular solve, exactly as in the validated nonsymmetric construction.
    au_raw = np.asarray(matrix @ u_raw, dtype=float)
    w, correction = la.qr(au_raw, mode="economic", check_finite=False)
    u = la.solve_triangular(
        correction.T, u_raw.T, lower=True, check_finite=False
    ).T
    z, _, z_rank = _orthonormalize(u, 1.0e-13)
    if z_rank != actual:
        raise RuntimeError("W-union recovery lost solution-space rank")
    reduced = w.T @ np.asarray(matrix @ z, dtype=float)
    au_error = float(
        np.linalg.norm(np.asarray(matrix @ u, dtype=float) - w, ord="fro")
        / max(np.linalg.norm(w, ord="fro"), np.finfo(float).tiny)
    )
    w_error = float(np.linalg.norm(w.T @ w - np.eye(actual), ord="fro"))
    pair = RecyclePair(
        z_basis=z,
        u_basis=u,
        w_basis=w,
        setup_seconds=perf_counter() - start,
        diagnostics={
            "au_equals_w_relative_error": au_error,
            "test_orthogonality_error": w_error,
            "reduced_condition": float(np.linalg.cond(reduced)),
            "dictionary_rank": float(rank),
        },
    )
    return pair, {
        "dictionary_rank": rank,
        "dictionary_columns": raw.shape[1],
        "compression_seconds": pair.setup_seconds,
        "pair_setup_seconds": 0.0,
        "singular_value_min_retained": float(singular[actual - 1]),
        "singular_value_max": float(singular[0]),
        "a_applications_pair": raw.shape[1] + 3 * actual,
        "pre_repair_w_projector_error": float(
            np.linalg.norm(
                w @ (w.T @ w_target) - w_target,
                ord="fro",
            )
            / max(np.linalg.norm(w_target, ord="fro"), np.finfo(float).tiny)
        ),
    }


def _run_pair(context: SystemContext, rhs: np.ndarray, pair: RecyclePair, settings: Stage22ASettings) -> GmresResult:
    context.ilu.reset_timing()
    return restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        u_basis=pair.u_basis,
        c_basis=pair.w_basis,
        rtol=settings.solver.relative_tolerance,
        atol=settings.solver.absolute_tolerance,
        restart=settings.solver.restart,
        max_iterations=settings.solver.max_iterations,
        record_true_residuals=True,
    )


def _curve_from_result(result: GmresResult, cap: int, floor: float) -> np.ndarray:
    """Interpolate a true-residual trajectory on the common iteration grid."""

    iterations = np.asarray(result.true_residual_iterations, dtype=int)
    values = np.asarray(result.true_residual_history, dtype=float)
    if iterations.size == 0:
        raise RuntimeError("true residual history was not recorded")
    latest: Dict[int, float] = {}
    for iteration, value in zip(iterations, values):
        latest[int(iteration)] = float(value)
    xp = np.asarray(sorted(latest), dtype=float)
    fp = np.log10(np.maximum([latest[int(index)] for index in xp], floor))
    return np.interp(np.arange(cap + 1, dtype=float), xp, fp, left=fp[0], right=fp[-1])


def _functional_curve_distance(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("functional curves must be paired and nonempty")
    distances = [float(np.sqrt(np.mean((a - b) ** 2)) / 8.0) for a, b in zip(first, second)]
    return float(np.mean(distances))


def _evaluate_pair(
    context: SystemContext,
    pair: RecyclePair,
    metadata: Mapping[str, Any],
    baseline: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    settings: Stage22ASettings,
    phase: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    curves: Dict[int, np.ndarray] = {}
    for rhs_seed in seeds:
        rhs, _ = _discrete_rhs(context.system, rhs_seed)
        result = _run_pair(context, rhs, pair, settings)
        true_residual = _relative_residual(context.system.matrix, result.x, rhs)
        base = baseline[rhs_seed]
        k_base = int(base["iterations"])
        reduction = (k_base - result.iterations) / max(k_base, 1)
        counts = result.operation_counts or {}
        row = dict(metadata)
        row.update(
            {
                "sample_id": context.sample_id,
                "phase": phase,
                "heldout_rhs_seed": rhs_seed,
                "evaluation_disjoint_from_construction_and_selection": rhs_seed
                not in settings.construction_rhs_seeds + settings.selection_rhs_seeds,
                "baseline_iterations": k_base,
                "baseline_converged": _bool(base["converged"]),
                "baseline_true_final_relative_residual": float(base["true_residual"]),
                "baseline_solve_seconds": float(base["solve_seconds"]),
                "recycle_iterations": result.iterations,
                "recycle_converged": result.converged,
                "recycle_true_final_relative_residual": true_residual,
                "recycle_solve_seconds": result.solve_seconds,
                "iteration_absolute_reduction": k_base - result.iterations,
                "iteration_ratio": result.iterations / max(k_base, 1),
                "J_iter": reduction,
                "online_a_applications": counts.get("a_matvecs", 0),
                "online_m_applications": counts.get("preconditioner_applications", 0),
            }
        )
        rows.append(row)
        curve = _curve_from_result(
            result, settings.solver.max_iterations, settings.functional_log_floor
        )
        curves[rhs_seed] = curve
        for iteration, relative_residual in zip(
            result.true_residual_iterations, result.true_residual_history
        ):
            history_rows.append(
                {
                    "sample_id": context.sample_id,
                    "phase": phase,
                    "method": metadata["method"],
                    "rule_id": metadata["rule_id"],
                    "heldout_rhs_seed": rhs_seed,
                    "iteration": iteration,
                    "true_relative_residual": relative_residual,
                }
            )
    return rows, history_rows, curves


def _baseline_map(stage21_rows: Sequence[Mapping[str, str]], sample_id: int, seeds: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        matches = [
            row
            for row in stage21_rows
            if int(row["sample_id"]) == sample_id
            and int(row["heldout_rhs_seed"]) == seed
            and _bool(row["primary_construction"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one Stage-21 baseline row for ID {sample_id}, RHS {seed}")
        row = matches[0]
        result[seed] = {
            "iterations": int(row["baseline_iterations"]),
            "converged": _bool(row["baseline_converged"]),
            "true_residual": float(row["baseline_true_final_relative_residual"]),
            "solve_seconds": float(row["baseline_solve_seconds"]),
        }
    return result


def _raw_dictionary(constructions: Sequence[RitzConstruction]) -> np.ndarray:
    return np.column_stack([item.basis for item in constructions])


def _setup_metadata(
    method: str,
    rule_id: str,
    constructions: Sequence[RitzConstruction],
    pair: RecyclePair,
    details: Mapping[str, Any],
    requested_q: int,
    selection_seconds: float = 0.0,
    selection_a_applications: int = 0,
    selection_m_applications: int = 0,
) -> Dict[str, Any]:
    ritz_seconds = float(sum(item.setup_seconds for item in constructions))
    construction_a = int(sum(item.a_applications for item in constructions))
    construction_m = int(sum(item.m_applications for item in constructions))
    compression = float(details["compression_seconds"])
    pair_seconds = float(details["pair_setup_seconds"])
    deployable_setup = ritz_seconds + compression + pair_seconds
    diagnostic_setup = deployable_setup + selection_seconds
    return {
        "method": method,
        "rule_id": rule_id,
        "n_construct": len(constructions),
        "arnoldi_dimensions": "+".join(str(item.arnoldi_dimension) for item in constructions),
        "nominal_arnoldi_work": int(sum(item.arnoldi_dimension for item in constructions)),
        "requested_q": requested_q,
        "actual_q": pair.rank,
        "dictionary_columns": int(details["dictionary_columns"]),
        "dictionary_rank": int(details["dictionary_rank"]),
        "ritz_setup_seconds": ritz_seconds,
        "compression_seconds": compression,
        "pair_setup_seconds": pair_seconds,
        "selection_search_seconds": selection_seconds,
        "deployable_setup_seconds": deployable_setup,
        "oracle_diagnostic_setup_seconds": diagnostic_setup,
        "construction_a_applications": construction_a,
        "construction_m_applications": construction_m,
        "pair_a_applications": int(details["a_applications_pair"]),
        "selection_a_applications": selection_a_applications,
        "selection_m_applications": selection_m_applications,
        "total_setup_a_applications": construction_a
        + int(details["a_applications_pair"])
        + selection_a_applications,
        "total_setup_m_applications": construction_m + selection_m_applications,
        "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
        "test_orthogonality_error": pair.diagnostics["test_orthogonality_error"],
        "reduced_condition": pair.diagnostics["reduced_condition"],
        "singular_value_min_retained": details["singular_value_min_retained"],
        "singular_value_max": details["singular_value_max"],
    }


def _selection_score(results: Sequence[GmresResult]) -> Tuple[int, float, float]:
    failures = sum(not result.converged for result in results)
    return (
        int(failures),
        float(np.median([result.iterations for result in results])),
        float(np.mean([math.log10(max(result.relative_residual, 1.0e-300)) for result in results])),
    )


def _functional_greedy(
    context: SystemContext,
    constructions: Sequence[RitzConstruction],
    settings: Stage22ASettings,
) -> Tuple[List[AtomicRitzBlock], List[Dict[str, Any]], float, int, int]:
    """Constrained forward greedy over one exposed pair-preserving block per RHS.

    The within-RHS smallest-magnitude Ritz order is preserved.  At each step,
    the next block from every construction RHS is evaluated on two disjoint
    selection RHSs.  This keeps the search finite while making every selection
    decision from actual recycled-GMRES performance rather than geometry.
    """

    pointers = [0 for _ in constructions]
    selected: List[AtomicRitzBlock] = []
    trials: List[Dict[str, Any]] = []
    total_a = 0
    total_m = 0
    start = perf_counter()
    while True:
        selected_rank = sum(block.rank for block in selected)
        if selected_rank >= max(settings.q_values):
            break
        candidates: List[Tuple[Tuple[int, float, float], int, AtomicRitzBlock, RecyclePair, List[GmresResult]]] = []
        for construction_index, construction in enumerate(constructions):
            pointer = pointers[construction_index]
            if pointer >= len(construction.blocks):
                continue
            block = construction.blocks[pointer]
            if selected_rank + block.rank > max(settings.q_values):
                continue
            raw = np.column_stack([item.basis for item in selected] + [block.basis])
            pair, _ = _z_union(
                context.system.matrix,
                raw,
                raw.shape[1],
                settings.numerical_rank_tolerance,
            )
            selection_results: List[GmresResult] = []
            for rhs_seed in settings.selection_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                result = _run_pair(context, rhs, pair, settings)
                selection_results.append(result)
                counts = result.operation_counts or {}
                total_a += int(counts.get("a_matvecs", 0))
                total_m += int(counts.get("preconditioner_applications", 0))
            score = _selection_score(selection_results)
            candidates.append((score, construction_index, block, pair, selection_results))
            trials.append(
                {
                    "sample_id": context.sample_id,
                    "greedy_step": len(selected) + 1,
                    "rank_before": selected_rank,
                    "candidate_construction_seed": block.construction_seed,
                    "candidate_source_order": block.source_order,
                    "candidate_block_rank": block.rank,
                    "candidate_rank_after": pair.rank,
                    "selection_failures": score[0],
                    "selection_iterations_median": score[1],
                    "selection_log10_residual_mean": score[2],
                    "chosen": False,
                }
            )
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2].source_order))
        _, chosen_index, chosen_block, _, _ = candidates[0]
        selected.append(chosen_block)
        pointers[chosen_index] += 1
        # Mark the last matching trial from this step as selected.
        for trial in reversed(trials):
            if (
                int(trial["greedy_step"]) == len(selected)
                and int(trial["candidate_construction_seed"]) == chosen_block.construction_seed
                and int(trial["candidate_source_order"]) == chosen_block.source_order
            ):
                trial["chosen"] = True
                break
    return selected, trials, perf_counter() - start, total_a, total_m


def _greedy_prefix_raw(blocks: Sequence[AtomicRitzBlock], requested_q: int) -> np.ndarray | None:
    accepted: List[np.ndarray] = []
    rank = 0
    for block in blocks:
        if rank + block.rank > requested_q:
            break
        accepted.append(block.basis)
        rank += block.rank
    return np.column_stack(accepted) if accepted else None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_protocol(root: Path, output: Path, settings: Stage22ASettings) -> List[int]:
    (
        _,
        _,
        _,
        _,
        archived_summaries,
        _,
        valid_ids,
        invalid_ids,
    ) = _load_population(root, settings.stage21)
    selected = [
        sample_id
        for sample_id in valid_ids
        if classify_difficulty(_archived_median(archived_summaries[sample_id]), settings.stage21)
        in {"hard", "transition"}
    ]
    protocol = {
        "written_before_stage22a_execution": True,
        "stage": "22A_multi_RHS_functional_oracle",
        "neural_training_performed": False,
        "cross_system_transfer_performed": False,
        "systems": selected,
        "excluded_invalid_systems": invalid_ids,
        "classification_source": "Stage-19A archived strong-ILU medians",
        "construction": asdict(settings),
        "system_gate": {
            "all_heldout_rhs_converge_true_relative_residual_at_most": 1.0e-8,
            "median_iteration_reduction_at_least": settings.system_reduction_gate,
        },
        "family_gate": {
            "fixed_rule_pass_fraction_at_least": settings.family_pass_fraction_gate,
            "selected_system_count": len(selected),
            "required_pass_count": math.ceil(settings.family_pass_fraction_gate * len(selected)),
            "maximum_final_rank": 64,
        },
        "functional_greedy_scope": "four-RHS m=128 dictionary only",
        "cost_matched_final_ranks": list(settings.cost_matched_q_values),
    }
    _write_json(output / "predeclared_protocol.json", protocol)
    _write_json(output / "settings.json", asdict(settings))
    return selected


def _load_stage21_baselines(root: Path) -> List[Dict[str, str]]:
    return _read_csv(
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage21_spatial_K_population_validation"
        / "population_solver_results.csv"
    )


def _run_primary_system(
    root: Path,
    system_output: Path,
    sample_id: int,
    settings: Stage22ASettings,
    stage21_rows: Sequence[Mapping[str, str]],
) -> None:
    (
        _, nodes, boundary, coefficient_rows, archived_summaries, _, _, _
    ) = _load_population(root, settings.stage21)
    context, assembly_seconds = _assemble_context(
        sample_id, coefficient_rows[sample_id], nodes, boundary, settings.stage21
    )
    difficulty = classify_difficulty(
        _archived_median(archived_summaries[sample_id]), settings.stage21
    )
    baseline = _baseline_map(stage21_rows, sample_id, settings.heldout_rhs_seeds)
    construction_rows: List[Dict[str, Any]] = []
    setup_rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    singular_rows: List[Dict[str, Any]] = []
    greedy_trials: List[Dict[str, Any]] = []
    independent: Dict[int, Tuple[RecyclePair, Dict[int, np.ndarray], List[Dict[str, Any]]]] = {}

    primary_constructions: List[RitzConstruction] = []
    for rhs_seed in settings.construction_rhs_seeds:
        construction = _build_ordinary_construction(
            context, rhs_seed, settings.primary_arnoldi_dimension, settings
        )
        primary_constructions.append(construction)
        construction_rows.append(
            {
                "sample_id": sample_id,
                "construction_family": "multi_rhs_primary",
                "rhs_seed": rhs_seed,
                "arnoldi_dimension": construction.arnoldi_dimension,
                "candidate_rank": construction.basis.shape[1],
                "atomic_blocks": len(construction.blocks),
                "setup_seconds": construction.setup_seconds,
                "a_applications": construction.a_applications,
                "m_applications": construction.m_applications,
                "basis_orthogonality_error": construction.basis_orthogonality_error,
                "ritz_magnitude_min": float(np.min(np.abs(construction.eigenvalues))),
                "ritz_magnitude_max": float(np.max(np.abs(construction.eigenvalues))),
            }
        )

    # Main Z/W union grid.  Unavailable requested ranks are recorded by omission
    # and the actual dictionary rank remains explicit in setup_results.csv.
    for n_construct in settings.n_construct_values:
        selected_constructions = primary_constructions[:n_construct]
        raw = _raw_dictionary(selected_constructions)
        _, z_singular, z_rank = _orthonormalize(raw, settings.numerical_rank_tolerance)
        w_singular = la.svdvals(np.asarray(context.system.matrix @ raw, dtype=float))
        w_rank = int(np.count_nonzero(w_singular > settings.numerical_rank_tolerance * w_singular[0]))
        for index, value in enumerate(z_singular, start=1):
            singular_rows.append(
                {
                    "sample_id": sample_id,
                    "n_construct": n_construct,
                    "space": "Z_raw",
                    "index": index,
                    "singular_value": value,
                    "normalized_singular_value": value / z_singular[0],
                    "numerical_rank": z_rank,
                }
            )
        for index, value in enumerate(w_singular, start=1):
            singular_rows.append(
                {
                    "sample_id": sample_id,
                    "n_construct": n_construct,
                    "space": "AZ_raw",
                    "index": index,
                    "singular_value": value,
                    "normalized_singular_value": value / w_singular[0],
                    "numerical_rank": w_rank,
                }
            )
        for requested_q in settings.q_values:
            if requested_q <= n_construct * settings.candidates_per_rhs:
                pair, details = _z_union(
                    context.system.matrix, raw, requested_q, settings.numerical_rank_tolerance
                )
                rule_id = f"Z_union_n{n_construct}_m128_q{requested_q}"
                metadata = _setup_metadata(
                    "Z_union", rule_id, selected_constructions, pair, details, requested_q
                )
                metadata.update({"difficulty_class": difficulty, "assembly_seconds": assembly_seconds})
                setup_rows.append({"sample_id": sample_id, **metadata})
                rows, histories, curves = _evaluate_pair(
                    context, pair, metadata, baseline, settings.heldout_rhs_seeds,
                    settings, "main_union"
                )
                solver_rows.extend(rows)
                history_rows.extend(histories)
                if n_construct == 1 and requested_q == 24:
                    independent[selected_constructions[0].rhs_seed] = (pair, curves, rows)
            if requested_q <= n_construct * settings.candidates_per_rhs:
                pair, details = _w_union(
                    context.system.matrix, raw, requested_q, settings.numerical_rank_tolerance
                )
                rule_id = f"W_union_n{n_construct}_m128_q{requested_q}"
                metadata = _setup_metadata(
                    "W_union", rule_id, selected_constructions, pair, details, requested_q
                )
                metadata.update({"difficulty_class": difficulty, "assembly_seconds": assembly_seconds})
                setup_rows.append({"sample_id": sample_id, **metadata})
                rows, histories, _ = _evaluate_pair(
                    context, pair, metadata, baseline, settings.heldout_rhs_seeds,
                    settings, "main_union"
                )
                solver_rows.extend(rows)
                history_rows.extend(histories)

    # Independent single-RHS constructions for functional/geometric stability.
    for construction in primary_constructions:
        if construction.rhs_seed in independent:
            continue
        pair, details = _z_union(
            context.system.matrix,
            construction.basis,
            min(24, construction.basis.shape[1]),
            settings.numerical_rank_tolerance,
        )
        rule_id = f"independent_seed{construction.rhs_seed}_m128_q24"
        metadata = _setup_metadata(
            "independent_single_rhs", rule_id, [construction], pair, details, 24
        )
        metadata.update({"difficulty_class": difficulty, "assembly_seconds": assembly_seconds})
        setup_rows.append({"sample_id": sample_id, **metadata})
        rows, histories, curves = _evaluate_pair(
            context, pair, metadata, baseline, settings.heldout_rhs_seeds,
            settings, "construction_stability"
        )
        solver_rows.extend(rows)
        history_rows.extend(histories)
        independent[construction.rhs_seed] = (pair, curves, rows)

    # Functional greedy on the complete four-RHS dictionary only.
    selected_blocks, trials, selection_seconds, selection_a, selection_m = _functional_greedy(
        context, primary_constructions, settings
    )
    greedy_trials.extend(trials)
    for requested_q in settings.q_values:
        raw = _greedy_prefix_raw(selected_blocks, requested_q)
        if raw is None:
            continue
        pair, details = _z_union(
            context.system.matrix, raw, raw.shape[1], settings.numerical_rank_tolerance
        )
        rule_id = f"functional_greedy_n4_m128_q{requested_q}"
        metadata = _setup_metadata(
            "functional_greedy", rule_id, primary_constructions, pair, details,
            requested_q, selection_seconds, selection_a, selection_m
        )
        metadata.update({"difficulty_class": difficulty, "assembly_seconds": assembly_seconds})
        setup_rows.append({"sample_id": sample_id, **metadata})
        rows, histories, _ = _evaluate_pair(
            context, pair, metadata, baseline, settings.heldout_rhs_seeds,
            settings, "main_functional_greedy"
        )
        solver_rows.extend(rows)
        history_rows.extend(histories)

    # Cost-matched ablation.  Z-union is the fixed neutral compression rule.
    cost_constructions: Dict[str, List[RitzConstruction]] = {
        "1rhs_m128": [primary_constructions[0]],
        "2rhs_m128": primary_constructions[:2],
    }
    cost_constructions["1rhs_m256"] = [
        _build_ordinary_construction(
            context, settings.construction_rhs_seeds[0], 256, settings
        )
    ]
    cost_constructions["4rhs_m64"] = [
        _build_ordinary_construction(context, seed, 64, settings)
        for seed in settings.construction_rhs_seeds
    ]
    for label, constructions in cost_constructions.items():
        if label not in {"1rhs_m128", "2rhs_m128"}:
            for construction in constructions:
                construction_rows.append(
                    {
                        "sample_id": sample_id,
                        "construction_family": f"cost_matched_{label}",
                        "rhs_seed": construction.rhs_seed,
                        "arnoldi_dimension": construction.arnoldi_dimension,
                        "candidate_rank": construction.basis.shape[1],
                        "atomic_blocks": len(construction.blocks),
                        "setup_seconds": construction.setup_seconds,
                        "a_applications": construction.a_applications,
                        "m_applications": construction.m_applications,
                        "basis_orthogonality_error": construction.basis_orthogonality_error,
                        "ritz_magnitude_min": float(np.min(np.abs(construction.eigenvalues))),
                        "ritz_magnitude_max": float(np.max(np.abs(construction.eigenvalues))),
                    }
                )
        raw = _raw_dictionary(constructions)
        _, _, rank = _orthonormalize(raw, settings.numerical_rank_tolerance)
        for requested_q in settings.cost_matched_q_values:
            pair, details = _z_union(
                context.system.matrix, raw, requested_q, settings.numerical_rank_tolerance
            )
            rule_id = f"cost_{label}_q{requested_q}"
            metadata = _setup_metadata(
                "cost_matched_Z_union", rule_id, constructions, pair, details, requested_q
            )
            metadata.update({"difficulty_class": difficulty, "assembly_seconds": assembly_seconds})
            setup_rows.append({"sample_id": sample_id, **metadata})
            rows, histories, _ = _evaluate_pair(
                context, pair, metadata, baseline, settings.heldout_rhs_seeds,
                settings, "cost_matched"
            )
            solver_rows.extend(rows)
            history_rows.extend(histories)

    # Pairwise geometry and functional residual-curve distance for four fully
    # independent q=24 constructions.
    geometry_rows: List[Dict[str, Any]] = []
    seeds = list(settings.construction_rhs_seeds)
    for i, seed_i in enumerate(seeds):
        pair_i, curves_i, rows_i = independent[seed_i]
        for seed_j in seeds[i + 1 :]:
            pair_j, curves_j, rows_j = independent[seed_j]
            z_metric = generalized_subspace_metrics(pair_i.z_basis, pair_j.z_basis)
            w_metric = generalized_subspace_metrics(pair_i.w_basis, pair_j.w_basis)
            functional = _functional_curve_distance(
                [curves_i[seed] for seed in settings.heldout_rhs_seeds],
                [curves_j[seed] for seed in settings.heldout_rhs_seeds],
            )
            ratio_i = float(np.median([float(row["iteration_ratio"]) for row in rows_i]))
            ratio_j = float(np.median([float(row["iteration_ratio"]) for row in rows_j]))
            geometry_rows.append(
                {
                    "sample_id": sample_id,
                    "difficulty_class": difficulty,
                    "construction_rhs_seed_i": seed_i,
                    "construction_rhs_seed_j": seed_j,
                    "q_i": pair_i.rank,
                    "q_j": pair_j.rank,
                    "d_Z": z_metric["projector_distance_normalized"],
                    "d_W": w_metric["projector_distance_normalized"],
                    "max_principal_angle_Z_degrees": z_metric["principal_angle_max_degrees"],
                    "max_principal_angle_W_degrees": w_metric["principal_angle_max_degrees"],
                    "mean_principal_angle_Z_degrees": z_metric["principal_angle_mean_degrees"],
                    "mean_principal_angle_W_degrees": w_metric["principal_angle_mean_degrees"],
                    "functional_curve_distance": functional,
                    "median_iteration_ratio_i": ratio_i,
                    "median_iteration_ratio_j": ratio_j,
                    "median_iteration_ratio_difference": abs(ratio_i - ratio_j),
                }
            )

    system_output.mkdir(parents=True, exist_ok=True)
    _write_csv(system_output / "construction_results.csv", construction_rows)
    _write_csv(system_output / "setup_results.csv", setup_rows)
    _write_csv(system_output / "solver_results.csv", solver_rows)
    _write_csv(system_output / "residual_histories.csv", history_rows)
    _write_csv(system_output / "dictionary_singular_values.csv", singular_rows)
    _write_csv(system_output / "greedy_search_trials.csv", greedy_trials)
    _write_csv(system_output / "geometry_functional_diagnostics.csv", geometry_rows)
    _write_json(
        system_output / "primary_complete.json",
        {
            "sample_id": sample_id,
            "complete": True,
            "neural_training_performed": False,
            "cross_system_transfer_performed": False,
            "solver_rows": len(solver_rows),
            "geometry_rows": len(geometry_rows),
        },
    )


def _combine_system_files(output: Path, sample_ids: Sequence[int], name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for sample_id in sample_ids:
        path = output / "systems" / f"id_{sample_id:02d}" / name
        if path.exists() and path.stat().st_size:
            rows.extend(_read_csv(path))
    _write_csv(output / name, rows)
    return rows


def _repair_requested_rank_boundaries(
    root: Path,
    output: Path,
    sample_ids: Sequence[int],
    settings: Stage22ASettings,
    stage21_rows: Sequence[Mapping[str, str]],
) -> None:
    """Evaluate pair-preserving requested ranks even when actual rank is lower.

    A multi-RHS dictionary can have fewer independent columns than its nominal
    ``24 * n_construct`` budget.  Such a requested-rank experiment is valid at
    the maximum available rank and must be recorded as, for example,
    requested 64 / actual 47 rather than silently omitted.  This repair is
    checkpoint-aware and only fills combinations omitted by an older partial
    Stage-22A run.
    """

    (_, nodes, boundary, coefficient_rows, archived_summaries, _, _, _) = (
        _load_population(root, settings.stage21)
    )
    for ordinal, sample_id in enumerate(sample_ids, start=1):
        system_output = output / "systems" / f"id_{sample_id:02d}"
        solver_rows = _read_csv(system_output / "solver_results.csv")
        setup_rows = _read_csv(system_output / "setup_results.csv")
        history_rows = _read_csv(system_output / "residual_histories.csv")
        existing = {str(row["rule_id"]) for row in solver_rows}
        expected: List[Tuple[str, str, int, int]] = []
        for n_construct in settings.n_construct_values:
            for requested_q in settings.q_values:
                if requested_q > n_construct * settings.candidates_per_rhs:
                    continue
                expected.extend(
                    (
                        (
                            f"Z_union_n{n_construct}_m128_q{requested_q}",
                            "Z_union",
                            n_construct,
                            requested_q,
                        ),
                        (
                            f"W_union_n{n_construct}_m128_q{requested_q}",
                            "W_union",
                            n_construct,
                            requested_q,
                        ),
                    )
                )
        expected.extend(
            (
                ("cost_1rhs_m128_q24", "cost_matched_Z_union", 1, 24),
                ("cost_1rhs_m256_q24", "cost_matched_Z_union", 1, 24),
            )
        )
        missing = [item for item in expected if item[0] not in existing]
        if not missing:
            continue
        print(
            f"[Stage22A rank repair {ordinal:02d}/{len(sample_ids)}] "
            f"ID {sample_id:02d}: {len(missing)} combinations",
            flush=True,
        )
        context, assembly_seconds = _assemble_context(
            sample_id, coefficient_rows[sample_id], nodes, boundary, settings.stage21
        )
        difficulty = classify_difficulty(
            _archived_median(archived_summaries[sample_id]), settings.stage21
        )
        baseline = _baseline_map(stage21_rows, sample_id, settings.heldout_rhs_seeds)
        primary = [
            _build_ordinary_construction(
                context, seed, settings.primary_arnoldi_dimension, settings
            )
            for seed in settings.construction_rhs_seeds
        ]
        deep: RitzConstruction | None = None
        for rule_id, method, n_construct, requested_q in missing:
            if rule_id.startswith("cost_1rhs_m256"):
                if deep is None:
                    deep = _build_ordinary_construction(
                        context, settings.construction_rhs_seeds[0], 256, settings
                    )
                constructions = [deep]
                raw = deep.basis
                phase = "cost_matched"
            else:
                constructions = primary[:n_construct]
                raw = _raw_dictionary(constructions)
                phase = "cost_matched" if rule_id.startswith("cost_") else "main_union"
            if method == "W_union":
                pair, details = _w_union(
                    context.system.matrix,
                    raw,
                    requested_q,
                    settings.numerical_rank_tolerance,
                )
            else:
                pair, details = _z_union(
                    context.system.matrix,
                    raw,
                    requested_q,
                    settings.numerical_rank_tolerance,
                )
            metadata = _setup_metadata(
                method, rule_id, constructions, pair, details, requested_q
            )
            metadata.update(
                {"difficulty_class": difficulty, "assembly_seconds": assembly_seconds}
            )
            setup_rows.append({"sample_id": sample_id, **metadata})
            rows, histories, _ = _evaluate_pair(
                context,
                pair,
                metadata,
                baseline,
                settings.heldout_rhs_seeds,
                settings,
                phase,
            )
            solver_rows.extend(rows)
            history_rows.extend(histories)
        _write_csv(system_output / "setup_results.csv", setup_rows)
        _write_csv(system_output / "solver_results.csv", solver_rows)
        _write_csv(system_output / "residual_histories.csv", history_rows)


def _repair_w_union_pairing(
    root: Path,
    output: Path,
    sample_ids: Sequence[int],
    settings: Stage22ASettings,
    stage21_rows: Sequence[Mapping[str, str]],
) -> None:
    """Replace only W-union rows generated before stable QR re-pairing."""

    (_, nodes, boundary, coefficient_rows, archived_summaries, _, _, _) = (
        _load_population(root, settings.stage21)
    )
    for ordinal, sample_id in enumerate(sample_ids, start=1):
        system_output = output / "systems" / f"id_{sample_id:02d}"
        marker = system_output / "w_union_qr_repaired.json"
        if marker.exists():
            continue
        setup_rows = _read_csv(system_output / "setup_results.csv")
        solver_rows = _read_csv(system_output / "solver_results.csv")
        history_rows = _read_csv(system_output / "residual_histories.csv")
        if not any(str(row["method"]) == "W_union" for row in setup_rows):
            continue
        print(
            f"[Stage22A W-union repair {ordinal:02d}/{len(sample_ids)}] "
            f"ID {sample_id:02d}",
            flush=True,
        )
        setup_rows = [row for row in setup_rows if str(row["method"]) != "W_union"]
        solver_rows = [row for row in solver_rows if str(row["method"]) != "W_union"]
        history_rows = [row for row in history_rows if str(row["method"]) != "W_union"]
        context, assembly_seconds = _assemble_context(
            sample_id, coefficient_rows[sample_id], nodes, boundary, settings.stage21
        )
        difficulty = classify_difficulty(
            _archived_median(archived_summaries[sample_id]), settings.stage21
        )
        baseline = _baseline_map(stage21_rows, sample_id, settings.heldout_rhs_seeds)
        constructions = [
            _build_ordinary_construction(
                context, seed, settings.primary_arnoldi_dimension, settings
            )
            for seed in settings.construction_rhs_seeds
        ]
        for n_construct in settings.n_construct_values:
            selected = constructions[:n_construct]
            raw = _raw_dictionary(selected)
            for requested_q in settings.q_values:
                if requested_q > n_construct * settings.candidates_per_rhs:
                    continue
                pair, details = _w_union(
                    context.system.matrix,
                    raw,
                    requested_q,
                    settings.numerical_rank_tolerance,
                )
                rule_id = f"W_union_n{n_construct}_m128_q{requested_q}"
                metadata = _setup_metadata(
                    "W_union", rule_id, selected, pair, details, requested_q
                )
                metadata.update(
                    {
                        "difficulty_class": difficulty,
                        "assembly_seconds": assembly_seconds,
                    }
                )
                setup_rows.append({"sample_id": sample_id, **metadata})
                rows, histories, _ = _evaluate_pair(
                    context,
                    pair,
                    metadata,
                    baseline,
                    settings.heldout_rhs_seeds,
                    settings,
                    "main_union",
                )
                solver_rows.extend(rows)
                history_rows.extend(histories)
        _write_csv(system_output / "setup_results.csv", setup_rows)
        _write_csv(system_output / "solver_results.csv", solver_rows)
        _write_csv(system_output / "residual_histories.csv", history_rows)
        _write_json(
            marker,
            {
                "sample_id": sample_id,
                "stable_qr_repair": True,
                "identity": "A U = W",
            },
        )


def _summarize_rules(
    solver_rows: Sequence[Mapping[str, Any]],
    setup_rows: Sequence[Mapping[str, Any]],
    settings: Stage22ASettings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    setup_map = {
        (int(row["sample_id"]), str(row["rule_id"])): row for row in setup_rows
    }
    system_rows: List[Dict[str, Any]] = []
    keys = sorted({(str(row["rule_id"]), int(row["sample_id"])) for row in solver_rows})
    for rule_id, sample_id in keys:
        local = [
            row for row in solver_rows
            if str(row["rule_id"]) == rule_id and int(row["sample_id"]) == sample_id
        ]
        setup = setup_map[(sample_id, rule_id)]
        convergence = [_bool(row["recycle_converged"]) for row in local]
        reductions = [float(row["J_iter"]) for row in local]
        baseline_time = float(np.median([float(row["baseline_solve_seconds"]) for row in local]))
        recycle_time = float(np.median([float(row["recycle_solve_seconds"]) for row in local]))
        deployable_setup = float(setup["deployable_setup_seconds"])
        diagnostic_setup = float(setup["oracle_diagnostic_setup_seconds"])
        system_rows.append(
            {
                "sample_id": sample_id,
                "method": local[0]["method"],
                "rule_id": rule_id,
                "n_construct": int(setup["n_construct"]),
                "nominal_arnoldi_work": int(setup["nominal_arnoldi_work"]),
                "requested_q": int(setup["requested_q"]),
                "actual_q": int(setup["actual_q"]),
                "dictionary_rank": int(setup["dictionary_rank"]),
                "baseline_iterations_median": float(np.median([int(row["baseline_iterations"]) for row in local])),
                "recycle_iterations_min": min(int(row["recycle_iterations"]) for row in local),
                "recycle_iterations_median": float(np.median([int(row["recycle_iterations"]) for row in local])),
                "recycle_iterations_max": max(int(row["recycle_iterations"]) for row in local),
                "convergence_rate": float(np.mean(convergence)),
                "true_residual_max": max(float(row["recycle_true_final_relative_residual"]) for row in local),
                "J_iter_median": float(np.median(reductions)),
                "iteration_ratio_median": float(np.median([float(row["iteration_ratio"]) for row in local])),
                "system_gate_passed": system_gate(convergence, reductions, settings),
                "deployable_setup_seconds": deployable_setup,
                "oracle_diagnostic_setup_seconds": diagnostic_setup,
                "online_solve_seconds_median": recycle_time,
                "break_even_rhs_deployable": estimate_break_even(deployable_setup, baseline_time, recycle_time),
                "break_even_rhs_oracle_diagnostic": estimate_break_even(diagnostic_setup, baseline_time, recycle_time),
                "break_even_censored_baseline": not all(_bool(row["baseline_converged"]) for row in local),
            }
        )

    rule_rows: List[Dict[str, Any]] = []
    for rule_id in sorted({str(row["rule_id"]) for row in system_rows}):
        local = [row for row in system_rows if row["rule_id"] == rule_id]
        gate = family_gate(local, settings)
        rule_rows.append(
            {
                "rule_id": rule_id,
                "method": local[0]["method"],
                "n_construct": local[0]["n_construct"],
                "requested_q": local[0]["requested_q"],
                "systems_evaluated": len(local),
                "systems_passing": gate["systems_passing"],
                "pass_fraction": gate["pass_fraction"],
                "required_systems": gate["required_systems"],
                "family_gate_passed": gate["passed"],
                "median_iteration_ratio": float(np.median([float(row["iteration_ratio_median"]) for row in local])),
                "median_J_iter": float(np.median([float(row["J_iter_median"]) for row in local])),
                "median_actual_q": float(np.median([int(row["actual_q"]) for row in local])),
                "median_deployable_setup_seconds": float(np.median([float(row["deployable_setup_seconds"]) for row in local])),
                "median_oracle_diagnostic_setup_seconds": float(np.median([float(row["oracle_diagnostic_setup_seconds"]) for row in local])),
            }
        )
    return system_rows, rule_rows


def _correlation_rows(geometry_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    target = np.asarray([float(row["median_iteration_ratio_difference"]) for row in geometry_rows])
    metrics = (
        "d_Z",
        "d_W",
        "max_principal_angle_Z_degrees",
        "max_principal_angle_W_degrees",
        "functional_curve_distance",
    )
    rows = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in geometry_rows])
        if values.size < 3 or np.ptp(values) == 0 or np.ptp(target) == 0:
            coefficient = pvalue = float("nan")
        else:
            result = stats.spearmanr(values, target)
            coefficient, pvalue = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "scope": "all_hard_transition_construction_pairs",
                "metric": metric,
                "performance_target": "absolute median iteration-ratio difference",
                "spearman_coefficient": coefficient,
                "spearman_pvalue": pvalue,
                "pair_count": len(values),
            }
        )
    return rows


def _aggregate_primary(output: Path, settings: Stage22ASettings, sample_ids: Sequence[int]) -> Dict[str, Any]:
    constructions = _combine_system_files(output, sample_ids, "construction_results.csv")
    setup = _combine_system_files(output, sample_ids, "setup_results.csv")
    solver = _combine_system_files(output, sample_ids, "solver_results.csv")
    histories = _combine_system_files(output, sample_ids, "residual_histories.csv")
    singular = _combine_system_files(output, sample_ids, "dictionary_singular_values.csv")
    greedy = _combine_system_files(output, sample_ids, "greedy_search_trials.csv")
    geometry = _combine_system_files(output, sample_ids, "geometry_functional_diagnostics.csv")
    system_summary, rule_summary = _summarize_rules(solver, setup, settings)
    _write_csv(output / "system_rule_summary.csv", system_summary)
    _write_csv(output / "family_rule_summary.csv", rule_summary)
    correlations = _correlation_rows(geometry)
    _write_csv(output / "geometry_functional_correlations.csv", correlations)
    # Family candidates must be available on every system and are ranked without
    # per-system tuning: pass count, then median J, then lower rank and setup.
    eligible = [
        row for row in rule_summary
        if int(row["systems_evaluated"]) == len(sample_ids)
        and row["method"] in {"Z_union", "W_union", "functional_greedy"}
    ]
    eligible.sort(
        key=lambda row: (
            -int(row["systems_passing"]),
            -float(row["median_J_iter"]),
            float(row["median_actual_q"]),
            float(row["median_deployable_setup_seconds"]),
        )
    )
    best = eligible[0] if eligible else None
    aggregate = {
        "population": {
            "hard_transition_systems": len(sample_ids),
            "sample_ids": list(sample_ids),
            "heldout_rhs_per_system": len(settings.heldout_rhs_seeds),
            "construction_rhs_count_max": len(settings.construction_rhs_seeds),
        },
        "artifact_rows": {
            "construction": len(constructions),
            "setup": len(setup),
            "solver": len(solver),
            "residual_history": len(histories),
            "singular_value": len(singular),
            "greedy_trial": len(greedy),
            "geometry": len(geometry),
        },
        "best_fixed_ordinary_rule": best,
        "ordinary_family_gate_passed": bool(best and _bool(best["family_gate_passed"])),
        "persistent_failures_under_best_rule": (
            [
                int(row["sample_id"])
                for row in system_summary
                if best and row["rule_id"] == best["rule_id"] and not _bool(row["system_gate_passed"])
            ]
            if best else list(sample_ids)
        ),
        "correlations": {row["metric"]: row for row in correlations},
        "no_neural_training": True,
        "no_cross_system_transfer": True,
    }
    _write_json(output / "primary_aggregate.json", aggregate)
    return aggregate


def _run_secondary_system(
    root: Path,
    system_output: Path,
    sample_id: int,
    settings: Stage22ASettings,
    stage21_rows: Sequence[Mapping[str, str]],
) -> None:
    """Run the limited harmonic diagnostic on a persistent failure."""

    (_, nodes, boundary, coefficient_rows, archived_summaries, _, _, _) = (
        _load_population(root, settings.stage21)
    )
    context, assembly_seconds = _assemble_context(
        sample_id, coefficient_rows[sample_id], nodes, boundary, settings.stage21
    )
    difficulty = classify_difficulty(
        _archived_median(archived_summaries[sample_id]), settings.stage21
    )
    baseline = _baseline_map(stage21_rows, sample_id, settings.heldout_rhs_seeds)
    ordinary = [
        _build_ordinary_construction(
            context, seed, settings.primary_arnoldi_dimension, settings
        )
        for seed in settings.construction_rhs_seeds
    ]
    harmonic = [
        _build_harmonic_construction(
            context, seed, settings.primary_arnoldi_dimension, settings
        )
        for seed in settings.construction_rhs_seeds
    ]
    candidates = {
        "harmonic_Z_union_n4_m128_q64": harmonic,
        "ordinary_harmonic_Z_union_n4_m128_q64": ordinary + harmonic,
    }
    setup_rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for rule_id, constructions in candidates.items():
        raw = _raw_dictionary(constructions)
        _, _, rank = _orthonormalize(raw, settings.numerical_rank_tolerance)
        requested_q = min(settings.harmonic_secondary_q, rank)
        pair, details = _z_union(
            context.system.matrix, raw, requested_q, settings.numerical_rank_tolerance
        )
        method = (
            "harmonic_Z_union"
            if rule_id.startswith("harmonic_")
            else "ordinary_harmonic_Z_union"
        )
        metadata = _setup_metadata(
            method, rule_id, constructions, pair, details, requested_q
        )
        metadata.update(
            {"difficulty_class": difficulty, "assembly_seconds": assembly_seconds}
        )
        setup_rows.append({"sample_id": sample_id, **metadata})
        rows, histories, _ = _evaluate_pair(
            context,
            pair,
            metadata,
            baseline,
            settings.heldout_rhs_seeds,
            settings,
            "persistent_failure_secondary",
        )
        solver_rows.extend(rows)
        history_rows.extend(histories)
    _write_csv(system_output / "secondary_setup_results.csv", setup_rows)
    _write_csv(system_output / "secondary_solver_results.csv", solver_rows)
    _write_csv(system_output / "secondary_residual_histories.csv", history_rows)
    _write_json(
        system_output / "secondary_complete.json",
        {
            "sample_id": sample_id,
            "complete": True,
            "ordinary_rules_exhausted_before_secondary": True,
            "secondary_scope": "harmonic and ordinary+harmonic q=64 only",
        },
    )


def _aggregate_secondary(
    output: Path,
    settings: Stage22ASettings,
    persistent_ids: Sequence[int],
) -> Dict[str, Any]:
    setup = _combine_system_files(
        output, persistent_ids, "secondary_setup_results.csv"
    )
    solver = _combine_system_files(
        output, persistent_ids, "secondary_solver_results.csv"
    )
    histories = _combine_system_files(
        output, persistent_ids, "secondary_residual_histories.csv"
    )
    if solver:
        system_summary, rule_summary = _summarize_rules(solver, setup, settings)
    else:
        system_summary, rule_summary = [], []
    _write_csv(output / "secondary_system_summary.csv", system_summary)
    _write_csv(output / "secondary_rule_summary.csv", rule_summary)
    aggregate = {
        "persistent_systems": list(persistent_ids),
        "solver_rows": len(solver),
        "residual_history_rows": len(histories),
        "system_results": system_summary,
        "rule_results": rule_summary,
        "limited_secondary_only": True,
    }
    _write_json(output / "secondary_aggregate.json", aggregate)
    return aggregate


def run_stage22a(root: Path) -> Path:
    """Run checkpointed primary Stage-22A experiments and consolidate evidence."""

    settings = Stage22ASettings()
    output = (
        root / "results" / "results-fd" / "stages"
        / "stage22A_spatial_K_multi_rhs_functional_oracle"
    )
    output.mkdir(parents=True, exist_ok=True)
    sample_ids = _write_protocol(root, output, settings)
    stage21_rows = _load_stage21_baselines(root)
    for ordinal, sample_id in enumerate(sample_ids, start=1):
        system_output = output / "systems" / f"id_{sample_id:02d}"
        if (system_output / "primary_complete.json").exists():
            print(f"[Stage22A {ordinal:02d}/{len(sample_ids)}] ID {sample_id:02d} cached", flush=True)
            continue
        print(f"[Stage22A {ordinal:02d}/{len(sample_ids)}] ID {sample_id:02d} primary", flush=True)
        _run_primary_system(root, system_output, sample_id, settings, stage21_rows)
    _repair_requested_rank_boundaries(
        root, output, sample_ids, settings, stage21_rows
    )
    _repair_w_union_pairing(
        root, output, sample_ids, settings, stage21_rows
    )
    aggregate = _aggregate_primary(output, settings, sample_ids)
    persistent_ids = [
        int(value) for value in aggregate["persistent_failures_under_best_rule"]
    ]
    for ordinal, sample_id in enumerate(persistent_ids, start=1):
        system_output = output / "systems" / f"id_{sample_id:02d}"
        if (system_output / "secondary_complete.json").exists():
            print(
                f"[Stage22A secondary {ordinal:02d}/{len(persistent_ids)}] "
                f"ID {sample_id:02d} cached",
                flush=True,
            )
            continue
        print(
            f"[Stage22A secondary {ordinal:02d}/{len(persistent_ids)}] "
            f"ID {sample_id:02d}",
            flush=True,
        )
        _run_secondary_system(
            root, system_output, sample_id, settings, stage21_rows
        )
    _aggregate_secondary(output, settings, persistent_ids)
    return output


if __name__ == "__main__":
    print(run_stage22a(Path(__file__).resolve().parents[3]))
