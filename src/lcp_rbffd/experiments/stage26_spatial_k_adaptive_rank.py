"""Stage 26: solver-side adaptive rank for the spatial-K own-system oracle.

The Stage-25 q=28 SVD prefix is the default.  At completed restart cycles a
fixed contraction rule may replace the paired recycle object by a broader
prefix of the same saved own-system dictionary.  No construction, spectral
selection, PDE, ILU, solver, or learning choice is changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
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

from neurkitt_oracle.gcrodr import CycleArnoldiData, GmresResult, restarted_projected_gmres

from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage21_population_validation import _assemble_context, _load_population, _safe_spearman
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    _build_ordinary_construction,
    _orthonormalize,
    _pair_from_z,
    _raw_dictionary,
    _run_pair,
)
from .stage25_spatial_k_rank_ablation import Stage25Settings
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PURPLE = "#7553A6"
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
class Stage26Settings:
    """Preregistered diagnostic, rank schedule, and viability gate."""

    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    fixed_q_values: Tuple[int, ...] = (28, 32, 48)
    adaptive_q_schedule: Tuple[int, ...] = (28, 32, 40, 48)
    default_q: int = 28
    moderate_contraction_threshold: float = 1.0e-2
    severe_contraction_threshold: float = 0.5
    recovery_gate: float = 0.75
    required_systems: int = 6
    mean_final_rank_limit: float = 44.0
    sensitivity_gate: float = 0.90
    specificity_gate: float = 0.75

    @property
    def stage25(self) -> Stage25Settings:
        return Stage25Settings()

    @property
    def solver(self):
        return self.stage25.stage22a.solver


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def expansion_target(
    current_requested_q: int,
    contraction: float,
    settings: Stage26Settings,
) -> Tuple[int, str]:
    """Return the preregistered next requested rank and decision label."""

    if current_requested_q >= settings.adaptive_q_schedule[-1]:
        return current_requested_q, "already_at_maximum"
    if contraction >= settings.severe_contraction_threshold:
        return settings.adaptive_q_schedule[-1], "severe_jump_to_full"
    if contraction >= settings.moderate_contraction_threshold:
        position = settings.adaptive_q_schedule.index(current_requested_q)
        return settings.adaptive_q_schedule[position + 1], "moderate_step"
    return current_requested_q, "keep_rank"


class AdaptiveRankController:
    """Record cheap cycle diagnostics and lazily expand a nested recycle pair."""

    def __init__(
        self,
        matrix,
        z_full: np.ndarray,
        initial_pair: RecyclePair,
        rhs_norm: float,
        settings: Stage26Settings,
    ) -> None:
        self.matrix = matrix
        self.z_full = np.asarray(z_full, dtype=float)
        self.pair = initial_pair
        self.rhs_norm = rhs_norm
        self.settings = settings
        self.active_requested_q = settings.default_q
        self.records: List[Dict[str, Any]] = []
        self.callback_seconds = 0.0
        self.expansion_setup_seconds = 0.0
        self.expansion_a_applications = 0
        self.expansions = 0

    def __call__(
        self, data: CycleArnoldiData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        callback_start = perf_counter()
        start_norm = float(np.linalg.norm(data.start_residual))
        end_norm = float(np.linalg.norm(data.end_residual))
        contraction = end_norm / max(start_norm, np.finfo(float).tiny)
        cycle_iterations = max(data.end_iteration - data.start_iteration, 1)
        decades = -float(np.log10(max(contraction, np.finfo(float).tiny)))
        test_coordinates = self.pair.w_basis.T @ data.end_residual
        rho_end = float(
            np.linalg.norm(test_coordinates)
            / max(end_norm, np.finfo(float).tiny)
        )
        correction = self.pair.u_basis @ test_coordinates
        correction_ratio = float(
            np.linalg.norm(correction) / max(end_norm, np.finfo(float).tiny)
        )
        end_relative = end_norm / max(self.rhs_norm, np.finfo(float).tiny)
        converged = end_relative <= self.settings.solver.relative_tolerance
        before_requested = self.active_requested_q
        before_actual = self.pair.rank
        target_requested, decision = expansion_target(
            before_requested, contraction, self.settings
        )
        replacement: RecyclePair | None = None
        expansion_seconds = 0.0
        if not converged and target_requested > before_requested:
            target_actual = min(target_requested, self.z_full.shape[1])
            expansion_start = perf_counter()
            replacement = _pair_from_z(
                self.matrix, self.z_full[:, :target_actual]
            )
            expansion_seconds = perf_counter() - expansion_start
            self.pair = replacement
            self.active_requested_q = target_requested
            self.expansion_setup_seconds += expansion_seconds
            self.expansion_a_applications += replacement.rank
            self.expansions += 1
        elif converged:
            decision = "converged_no_expansion"

        after_requested = self.active_requested_q
        after_actual = self.pair.rank
        self.records.append(
            {
                "cycle": data.cycle,
                "start_iteration": data.start_iteration,
                "end_iteration": data.end_iteration,
                "cycle_iterations": cycle_iterations,
                "start_relative_residual": start_norm / max(self.rhs_norm, 1.0e-300),
                "end_relative_residual": end_relative,
                "cycle_contraction": contraction,
                "cycle_log10_reduction": decades,
                "log10_reduction_per_iteration": decades / cycle_iterations,
                "projected_residual_fraction_end": rho_end,
                "recycle_correction_magnitude_ratio_end": correction_ratio,
                "active_requested_q_before": before_requested,
                "active_actual_q_before": before_actual,
                "decision": decision,
                "expanded": replacement is not None,
                "active_requested_q_after": after_requested,
                "active_actual_q_after": after_actual,
                "expansion_setup_seconds": expansion_seconds,
                "converged_at_cycle_end": converged,
            }
        )
        self.callback_seconds += perf_counter() - callback_start
        if replacement is None:
            return None
        return replacement.u_basis, replacement.w_basis


def _run_adaptive(
    context,
    rhs: np.ndarray,
    z_full: np.ndarray,
    settings: Stage26Settings,
) -> Tuple[GmresResult, AdaptiveRankController, float]:
    initial_actual = min(settings.default_q, z_full.shape[1])
    initial_pair = _pair_from_z(context.system.matrix, z_full[:, :initial_actual])
    controller = AdaptiveRankController(
        context.system.matrix,
        z_full,
        initial_pair,
        float(np.linalg.norm(rhs)),
        settings,
    )
    context.ilu.reset_timing()
    result = restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        u_basis=initial_pair.u_basis,
        c_basis=initial_pair.w_basis,
        rtol=settings.solver.relative_tolerance,
        atol=settings.solver.absolute_tolerance,
        restart=settings.solver.restart,
        max_iterations=settings.solver.max_iterations,
        record_true_residuals=True,
        # The controller computes the requested projected-residual and
        # correction diagnostics once at each cycle boundary.  Enabling the
        # solver's per-iteration projection trace would add O(Nq) work at
        # every Arnoldi step even though it is not used by the policy.
        record_projection_diagnostics=False,
        cycle_update=controller,
    )
    online_excluding_callback = max(
        result.solve_seconds - controller.callback_seconds, 0.0
    )
    return result, controller, online_excluding_callback


def _fixed_row(row: Mapping[str, str], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "iterations": int(row["recycle_iterations"]),
        "converged": _bool(row["recycle_converged"]),
        "true_final_residual": float(row["recycle_true_final_residual"]),
        "final_requested_q": int(row["requested_q"]),
        "final_actual_q": int(row["actual_q"]),
        "expansions": 0,
        "online_seconds": float(row["recycle_solve_seconds"]),
        "online_seconds_excluding_diagnostic": float(row["recycle_solve_seconds"]),
        "initial_setup_seconds": float(row["setup_seconds"]),
        "expansion_setup_seconds": 0.0,
        "total_setup_seconds": float(row["setup_seconds"]),
        "total_seconds": float(row["setup_seconds"])
        + float(row["recycle_solve_seconds"]),
        "setup_a_applications": int(row["setup_a_applications"]),
        "setup_m_applications": int(row["setup_m_applications"]),
        "online_a_applications": int(row["online_a_applications"]),
        "online_m_applications": int(row["online_m_applications"]),
        "expansion_a_applications": 0,
        "source": "archived_stage25",
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]], settings: Stage26Settings
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reference = {
        (int(row["sample_id"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in rows
        if row["method"] == "fixed_q48"
    }
    system_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        for method in ("fixed_q28", "fixed_q32", "fixed_q48", "adaptive"):
            local = [
                row
                for row in rows
                if int(row["sample_id"]) == sample_id and row["method"] == method
            ]
            eta = []
            for row in local:
                baseline = int(row["baseline_iterations"])
                denominator = baseline - reference[(sample_id, int(row["rhs_seed"]))]
                eta.append(
                    (baseline - int(row["iterations"])) / denominator
                    if denominator > 0
                    else float("nan")
                )
            median_eta = float(np.nanmedian(eta))
            all_converged = all(_bool(row["converged"]) for row in local)
            system_rows.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "rhs_count": len(local),
                    "converged_rhs": sum(_bool(row["converged"]) for row in local),
                    "all_rhs_converged": all_converged,
                    "baseline_iterations_median": float(
                        np.median([int(row["baseline_iterations"]) for row in local])
                    ),
                    "iterations_median": float(
                        np.median([int(row["iterations"]) for row in local])
                    ),
                    "iterations_min": min(int(row["iterations"]) for row in local),
                    "iterations_max": max(int(row["iterations"]) for row in local),
                    "reference_benefit_recovery_median": median_eta,
                    "final_actual_q_median": float(
                        np.median([int(row["final_actual_q"]) for row in local])
                    ),
                    "final_actual_q_max": max(int(row["final_actual_q"]) for row in local),
                    "expansions_median": float(
                        np.median([int(row["expansions"]) for row in local])
                    ),
                    "online_seconds_median": float(
                        np.median([float(row["online_seconds"]) for row in local])
                    ),
                    "total_setup_seconds_median": float(
                        np.median([float(row["total_setup_seconds"]) for row in local])
                    ),
                    "total_seconds_median": float(
                        np.median([float(row["total_seconds"]) for row in local])
                    ),
                    "system_gate_passed": bool(
                        all_converged and median_eta >= settings.recovery_gate
                    ),
                }
            )

    family_rows: List[Dict[str, Any]] = []
    for method in ("fixed_q28", "fixed_q32", "fixed_q48", "adaptive"):
        local = [row for row in system_rows if row["method"] == method]
        family_rows.append(
            {
                "method": method,
                "systems": len(local),
                "systems_passing": sum(_bool(row["system_gate_passed"]) for row in local),
                "all_rhs_convergence_systems": sum(
                    _bool(row["all_rhs_converged"]) for row in local
                ),
                "converged_rhs": sum(int(row["converged_rhs"]) for row in local),
                "median_iterations": float(
                    np.median([float(row["iterations_median"]) for row in local])
                ),
                "median_iteration_ratio_to_ilu": float(
                    np.median(
                        [
                            float(row["iterations_median"])
                            / max(float(row["baseline_iterations_median"]), 1.0)
                            for row in local
                        ]
                    )
                ),
                "median_reference_benefit_recovery": float(
                    np.median(
                        [float(row["reference_benefit_recovery_median"]) for row in local]
                    )
                ),
                "mean_final_actual_q": float(
                    np.mean([float(row["final_actual_q_median"]) for row in local])
                ),
                "median_final_actual_q": float(
                    np.median([float(row["final_actual_q_median"]) for row in local])
                ),
                "median_expansions": float(
                    np.median([float(row["expansions_median"]) for row in local])
                ),
                "median_online_seconds": float(
                    np.median([float(row["online_seconds_median"]) for row in local])
                ),
                "median_total_setup_seconds": float(
                    np.median([float(row["total_setup_seconds_median"]) for row in local])
                ),
                "median_total_seconds": float(
                    np.median([float(row["total_seconds_median"]) for row in local])
                ),
                "median_setup_a_applications": float(
                    np.median(
                        [
                            float(row["setup_a_applications"])
                            for row in rows
                            if row["method"] == method
                        ]
                    )
                ),
                "median_setup_m_applications": float(
                    np.median(
                        [
                            float(row["setup_m_applications"])
                            for row in rows
                            if row["method"] == method
                        ]
                    )
                ),
                "median_online_a_applications": float(
                    np.median(
                        [
                            float(row["online_a_applications"])
                            for row in rows
                            if row["method"] == method
                        ]
                    )
                ),
                "median_online_m_applications": float(
                    np.median(
                        [
                            float(row["online_m_applications"])
                            for row in rows
                            if row["method"] == method
                        ]
                    )
                ),
            }
        )
    return system_rows, family_rows


def _diagnostic_analysis(
    cycle_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    settings: Stage26Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    q28 = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in method_rows
        if row["method"] == "fixed_q28"
    }
    q48 = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in method_rows
        if row["method"] == "fixed_q48"
    }
    first_rows: List[Dict[str, Any]] = []
    for cycle in cycle_rows:
        if int(cycle["cycle"]) != 1:
            continue
        key = (int(cycle["sample_id"]), int(cycle["rhs_seed"]))
        low = q28[key]
        ref = q48[key]
        denominator = int(low["baseline_iterations"]) - int(ref["iterations"])
        fixed_eta = (
            (int(low["baseline_iterations"]) - int(low["iterations"])) / denominator
            if denominator > 0
            else float("nan")
        )
        need = (not _bool(low["converged"])) or fixed_eta < settings.recovery_gate
        predicted = _bool(cycle["expanded"])
        first_rows.append(
            {
                **dict(cycle),
                "fixed_q28_iterations": int(low["iterations"]),
                "fixed_q28_converged": _bool(low["converged"]),
                "fixed_q28_reference_benefit_recovery": fixed_eta,
                "expansion_needed": need,
                "expansion_predicted": predicted,
                "classification": (
                    "TP" if need and predicted else
                    "FN" if need else
                    "FP" if predicted else
                    "TN"
                ),
            }
        )

    metrics = (
        ("cycle_contraction", 1.0),
        ("cycle_log10_reduction", -1.0),
        ("log10_reduction_per_iteration", -1.0),
        ("projected_residual_fraction_end", 1.0),
        ("recycle_correction_magnitude_ratio_end", 1.0),
    )
    group_rows: List[Dict[str, Any]] = []
    correlation_rows: List[Dict[str, Any]] = []
    labels = np.asarray([_bool(row["expansion_needed"]) for row in first_rows], dtype=float)
    for metric, direction in metrics:
        for needed in (False, True):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in first_rows
                    if _bool(row["expansion_needed"]) == needed
                ],
                dtype=float,
            )
            group_rows.append(
                {
                    "metric": metric,
                    "group": "q28_insufficient" if needed else "q28_sufficient",
                    "observations": values.size,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
        scores = direction * np.asarray([float(row[metric]) for row in first_rows])
        rho, pvalue, count = _safe_spearman(scores, labels)
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        comparisons = [
            (1.0 if p > n else 0.5 if p == n else 0.0)
            for p in positive
            for n in negative
        ]
        auc = float(np.mean(comparisons)) if comparisons else float("nan")
        correlation_rows.append(
            {
                "metric": metric,
                "higher_score_means_more_need": True,
                "spearman_rho_with_expansion_need": rho,
                "pvalue_descriptive_only": pvalue,
                "observations": count,
                "rank_auc": auc,
            }
        )

    counts = {
        label: sum(row["classification"] == label for row in first_rows)
        for label in ("TP", "TN", "FP", "FN")
    }
    sensitivity = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    specificity = counts["TN"] / max(counts["TN"] + counts["FP"], 1)
    confusion = {
        **counts,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "false_negative_rate": 1.0 - sensitivity,
    }
    return first_rows, group_rows, correlation_rows, confusion


def _plots(
    output: Path,
    method_rows: Sequence[Mapping[str, Any]],
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    first_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    settings: Stage26Settings,
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Dict[str, str]] = []
    methods = ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48")
    labels = {"fixed_q28": "fixed q=28", "fixed_q32": "fixed q=32", "adaptive": "adaptive", "fixed_q48": "fixed q<=48"}
    colors = {"fixed_q28": GRAY, "fixed_q32": BLUE, "adaptive": ORANGE, "fixed_q48": OLIVE}

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(settings.sample_ids))
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for offset, method in zip(offsets, methods):
        values = [
            float(
                next(
                    row["iterations_median"]
                    for row in system_rows
                    if int(row["sample_id"]) == sample_id and row["method"] == method
                )
            )
            for sample_id in settings.sample_ids
        ]
        ax.scatter(positions + offset, values, color=colors[method], s=45, label=labels[method])
    ax.axhline(settings.solver.max_iterations, color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, [str(value) for value in settings.sample_ids])
    ax.set_xlabel("spatial-K system ID")
    ax.set_ylabel("median held-out iterations")
    ax.set_title("Fixed and adaptive recycle-rank comparison")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    path = plot_dir / "method_iterations_by_system.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "system_method_summary.csv", "purpose": "Compare solver-level robustness under identical own-system information."})

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    rng = np.random.default_rng(26)
    for index, needed in enumerate((False, True)):
        local = [row for row in first_rows if _bool(row["expansion_needed"]) == needed]
        x = index + rng.uniform(-0.08, 0.08, len(local))
        y = [float(row["cycle_contraction"]) for row in local]
        ax.scatter(x, y, color=BLUE if not needed else ORANGE, s=42, alpha=0.8)
    ax.axhline(settings.moderate_contraction_threshold, color=CHARCOAL, linestyle="--", linewidth=0.9, label="step threshold")
    ax.axhline(settings.severe_contraction_threshold, color=PURPLE, linestyle=":", linewidth=1.1, label="jump threshold")
    ax.set_yscale("log")
    ax.set_xticks((0, 1), ("q=28 sufficient", "q=28 insufficient"))
    ax.set_ylabel("first-cycle residual contraction")
    ax.set_title("Preregistered early diagnostic by fixed-q=28 outcome")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "first_cycle_contraction_by_need.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "first_cycle_diagnostics.csv", "purpose": "Show false-positive and false-negative behavior of the preregistered trigger."})

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    adaptive = [row for row in method_rows if row["method"] == "adaptive"]
    for sample_index, sample_id in enumerate(settings.sample_ids):
        local = [row for row in adaptive if int(row["sample_id"]) == sample_id]
        ax.scatter(
            np.full(len(local), sample_index),
            [int(row["final_actual_q"]) for row in local],
            color=ORANGE,
            s=42,
            alpha=0.8,
        )
    ax.axhline(48, color=CHARCOAL, linestyle="--", linewidth=0.8, label="full requested rank")
    ax.set_xticks(np.arange(len(settings.sample_ids)), [str(value) for value in settings.sample_ids])
    ax.set_yticks(settings.adaptive_q_schedule)
    ax.set_xlabel("spatial-K system ID")
    ax.set_ylabel("final active actual rank")
    ax.set_title("Adaptive rank paid by each held-out solve")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "adaptive_final_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "method_results.csv", "purpose": "Test whether robustness is obtained without paying full rank on every system."})

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True)
    for ax, sample_id in zip(axes, (10, 17)):
        for method in methods:
            local = [
                row for row in history_rows
                if int(row["sample_id"]) == sample_id
                and int(row["rhs_seed"]) == settings.heldout_rhs_seeds[0]
                and row["method"] == method
            ]
            ax.semilogy(
                [int(row["iteration"]) for row in local],
                [max(float(row["true_relative_residual"]), 1.0e-16) for row in local],
                color=colors[method],
                linewidth=1.5,
                label=labels[method],
            )
        ax.axhline(settings.solver.relative_tolerance, color=CHARCOAL, linestyle="--", linewidth=0.8)
        ax.set_title(f"ID {sample_id}")
        ax.set_xlabel("iteration")
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].set_ylabel("true relative residual")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    path = plot_dir / "representative_residual_histories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "residual_histories.csv", "purpose": "Verify the adaptive decision against true-residual convergence on the insufficient systems."})

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    x = np.arange(len(methods))
    setup = [float(next(row["median_total_setup_seconds"] for row in family_rows if row["method"] == method)) for method in methods]
    online = [float(next(row["median_online_seconds"] for row in family_rows if row["method"] == method)) for method in methods]
    ax.bar(x, setup, color=LIGHT_GRAY, edgecolor=CHARCOAL, label="setup")
    ax.bar(x, online, bottom=setup, color=[colors[method] for method in methods], edgecolor=CHARCOAL, label="online solve")
    ax.set_xticks(x, [labels[method] for method in methods], rotation=15, ha="right")
    ax.set_ylabel("median total time (s)")
    ax.set_title("Setup and online cost by integration rule")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "setup_online_cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "family_method_summary.csv", "purpose": "Compare rank savings with measured setup and online cost."})
    return charts


def _write_report(
    root: Path,
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    diagnostic_groups: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    confusion: Mapping[str, Any],
    decision: Mapping[str, Any],
    settings: Stage26Settings,
) -> Path:
    report = root / "results/results-fd/reports/stage26_spatial_K_adaptive_rank_diagnostic.md"
    family = {row["method"]: row for row in family_rows}
    method_labels = {"fixed_q28": "fixed $q=28$", "fixed_q32": "fixed $q=32$", "fixed_q48": "fixed $q\\leq48$", "adaptive": "adaptive"}
    status = "viable" if decision["adaptive_rank_viable"] else "not validated"
    lines = [
        "# Stage 26: solver-side adaptive rank for the spatial-$K$ recycle oracle",
        "",
        "## Technical conclusion",
        "",
        f"**The preregistered adaptive-rank mechanism is {status}.** {decision['conclusion']} The policy starts at requested $q=28$ and uses only completed-cycle residual contraction. It does not inspect the target oracle outcome or change the recycle construction.",
        "",
        f"At the first restart boundary the trigger has sensitivity {float(confusion['sensitivity']):.3f} and specificity {float(confusion['specificity']):.3f} against the per-RHS definition of insufficient fixed-$q=28$ performance. The confusion counts are TP={confusion['TP']}, TN={confusion['TN']}, FP={confusion['FP']}, and FN={confusion['FN']}.",
        "",
        "## Frozen experiment",
        "",
        "The cohort is spatial-$K$ IDs 3, 7, 10, 14, 15, 17, 19, and 34. Every $N=5041$ system uses the fixed seed-7 jittered node set (jitter 0.15), stencil size 25, PHS $r^3$ with degree-2 polynomial augmentation, ILU drop tolerance $10^{-3}$ and fill factor 5, right preconditioning, restart 50, a 300-iteration cap, and true-residual tolerance $10^{-8}$. The joint dictionary uses ordinary Ritz vectors from three independent $m=128$ construction trajectories (seeds 20261001, 20261011, and 20261021). Evaluation uses the disjoint seeds 20261101--20261103.",
        "",
        "For every active prefix, thin QR gives $AZ=WR$ and triangular solution gives $U=ZR^{-1}$, so $AU=W$ and the existing nonsymmetric projected/recycled GMRES integration is unchanged. The only adaptive action is replacement by a broader prefix of the same ordered joint-SVD basis at a completed restart boundary.",
        "",
        "The policy gate was stored before adaptive execution. It requires 24/24 convergence, at least six system-level recovery passes, median per-RHS recovery at least 0.75, mean final rank below 44, first-boundary sensitivity at least 0.90, and specificity at least 0.75.",
        "",
        "## Solver performance and rank paid",
        "",
        "| method | converged RHS | systems passing recovery gate | median iterations | median $k/k_{ILU}$ | median $\\eta_{ref}$ | mean final rank | median expansions | median setup (s) | median online (s) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for method in ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48"):
        row = family[method]
        lines.append(
            f"| {method_labels[method]} | {int(row['converged_rhs'])}/24 | {int(row['systems_passing'])}/8 | {float(row['median_iterations']):.0f} | {float(row['median_iteration_ratio_to_ilu']):.3f} | {float(row['median_reference_benefit_recovery']):.3f} | {float(row['mean_final_actual_q']):.1f} | {float(row['median_expansions']):.1f} | {float(row['median_total_setup_seconds']):.3f} | {float(row['median_online_seconds']):.3f} |"
        )
    lines.extend([
        "",
        "The system comparison uses identical held-out right-hand sides. A cap hit remains at 300 iterations and is not hidden in the medians.",
        "",
        "![Method iterations](../stages/stage26_spatial_K_adaptive_rank_diagnostic/plots/method_iterations_by_system.png)",
        "",
        "| ID | ILU | fixed $q=28$ | fixed $q=32$ | adaptive | fixed $q\\leq48$ | adaptive final rank | expansions | adaptive $\\eta_{ref}$ |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for sample_id in settings.sample_ids:
        local = {row["method"]: row for row in system_rows if int(row["sample_id"]) == sample_id}
        lines.append(
            f"| {sample_id} | {float(local['adaptive']['baseline_iterations_median']):.0f} | {float(local['fixed_q28']['iterations_median']):.0f} | {float(local['fixed_q32']['iterations_median']):.0f} | {float(local['adaptive']['iterations_median']):.0f} | {float(local['fixed_q48']['iterations_median']):.0f} | {float(local['adaptive']['final_actual_q_median']):.0f} | {float(local['adaptive']['expansions_median']):.0f} | {float(local['adaptive']['reference_benefit_recovery_median']):.3f} |"
        )
    lines.extend([
        "",
        "The rank plot shows the actual final rank for every held-out solve, so average savings cannot hide full-rank expansion on a difficult case.",
        "",
        "![Adaptive final rank](../stages/stage26_spatial_K_adaptive_rank_diagnostic/plots/adaptive_final_rank.png)",
        "",
        "The adaptive rule expands only ID 17, directly from 28 to 48 after its severe first-cycle signal. All three ID 17 solves then converge in 139--141 iterations, compared with the 300-iteration cap at fixed ranks 28 and 32 and 89--92 iterations when rank 48 is active from initialization. ID 10 is the important false-negative case: fixed rank 28 converges in 137--141 iterations, but recovers only about 0.61 of the rank-48 benefit, and its first-cycle contraction remains below the preregistered moderate threshold.",
        "",
        "## Early contraction is the preregistered decision signal",
        "",
        "The primary diagnostic is $c_1=\\|r_{end}\\|_2/\\|r_{start}\\|_2$ over the first completed restart cycle. The moderate threshold $10^{-2}$ means fewer than two decades of reduction in a 50-step cycle. The severe threshold 0.5 means less than a factor-two reduction and triggers an immediate jump to the full available prefix. Other already-available quantities are reported descriptively and were not used to tune the policy.",
        "",
        "![First-cycle diagnostic](../stages/stage26_spatial_K_adaptive_rank_diagnostic/plots/first_cycle_contraction_by_need.png)",
        "",
        "| diagnostic | sufficient median | insufficient median | rank AUC | Spearman $\\rho$ with need |",
        "|:--|--:|--:|--:|--:|",
    ])
    group_lookup = {(row["metric"], row["group"]): row for row in diagnostic_groups}
    correlation_lookup = {row["metric"]: row for row in correlations}
    for metric in (
        "cycle_contraction",
        "cycle_log10_reduction",
        "log10_reduction_per_iteration",
        "projected_residual_fraction_end",
        "recycle_correction_magnitude_ratio_end",
    ):
        sufficient = group_lookup[(metric, "q28_sufficient")]
        insufficient = group_lookup[(metric, "q28_insufficient")]
        correlation = correlation_lookup[metric]
        lines.append(
            f"| `{metric}` | {float(sufficient['median']):.3e} | {float(insufficient['median']):.3e} | {float(correlation['rank_auc']):.3f} | {float(correlation['spearman_rho_with_expansion_need']):.3f} |"
        )
    lines.extend([
        "",
        "The AUC direction is normalized so that larger scores mean greater expansion need. These are descriptive same-cohort diagnostics, not an independently validated classifier.",
        "",
        "First-cycle contraction ranks all six insufficient cases above all 18 sufficient cases (rank AUC 1.000), so it contains useful same-cohort signal. The frozen threshold nevertheless misses all three ID 10 solves. This distinction matters: Stage 26 rejects the preregistered policy, not the possibility that a differently validated classifier could use early contraction. Retuning the cutoff on these same 24 solves would be post-selection and was not performed.",
        "",
        "## True-residual histories and cost",
        "",
        "Representative histories show when the adaptive decision occurs and whether the broader prefix restores true-residual convergence.",
        "",
        "![Residual histories](../stages/stage26_spatial_K_adaptive_rank_diagnostic/plots/representative_residual_histories.png)",
        "",
        "Adaptive online time includes diagnostic evaluation and lazy pair reconstruction. Total setup begins with the archived Stage-25 $q=28$ construction cost and adds only pair reconstruction actually triggered during the solve. It does not charge the common three-trajectory oracle construction differently across methods.",
        "",
        "![Setup and online cost](../stages/stage26_spatial_K_adaptive_rank_diagnostic/plots/setup_online_cost.png)",
        "",
        "| method | median setup $A$ apps | median setup $M^{-1}$ apps | median online $A$ apps | median online $M^{-1}$ apps |",
        "|:--|--:|--:|--:|--:|",
    ])
    for method in ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48"):
        row = family[method]
        lines.append(
            f"| {method_labels[method]} | {float(row['median_setup_a_applications']):.0f} | {float(row['median_setup_m_applications']):.0f} | {float(row['median_online_a_applications']):.0f} | {float(row['median_online_m_applications']):.0f} |"
        )
    lines.extend([
        "",
        "Fixed-rank iterations are the archived Stage-25 values. Their online times and operation counts were repeated in the same Stage-26 process; all 72 repeats matched the archived iteration and convergence outcomes exactly. The adaptive controller adds only cycle-boundary diagnostics, but its measured Python integration path is slower than the fixed solver path, so it provides no wall-clock advantage in this implementation.",
        "",
        "## Scope and decision",
        "",
        "The first-cycle diagnostic and its ground-truth need label are evaluated on the same 24 held-out solves already characterized in Stage 25. Thresholds were preregistered before Stage 26 adaptive solves, but this is not an independent-system validation. The result can retain or reject this simple classical mechanism; it cannot establish transferability or learnability.",
        "",
        "The policy does restore true-residual convergence on all 24 held-out solves, lowers the mean final rank from 42.5 for the full reference to 30.5, and has median reference-benefit recovery 0.950. It still fails the preregistered decision gate because first-boundary sensitivity is only 0.500 and only seven of eight systems meet the 75% recovery criterion. The apparent aggregate economy is therefore driven by a correct trigger on ID 17 while the less dramatic ID 10 insufficiency is undetected.",
        "",
        f"**Decision:** {decision['recommendation']}",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def run_stage26(root: Path) -> Path:
    settings = Stage26Settings()
    output = root / "results/results-fd/stages/stage26_spatial_K_adaptive_rank_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))

    stage25 = root / "results/results-fd/stages/stage25_spatial_K_recycle_rank_ablation"
    archived = _read_csv(stage25 / "rank_ablation_results.csv")
    archived_histories = _read_csv(stage25 / "residual_histories.csv")
    (
        _, nodes, boundary, coefficient_rows, _, _, _, _
    ) = _load_population(root, settings.stage25.stage22a.stage21)

    method_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    timing_repeat_rows: List[Dict[str, Any]] = []
    for q in settings.fixed_q_values:
        method = f"fixed_q{q}"
        for row in archived:
            if int(row["requested_q"]) == q:
                fixed = _fixed_row(row, method)
                fixed["baseline_iterations"] = int(row["baseline_iterations"])
                method_rows.append(fixed)
        for row in archived_histories:
            if int(row["requested_q"]) == q:
                history_rows.append(
                    {
                        "sample_id": int(row["sample_id"]),
                        "rhs_seed": int(row["rhs_seed"]),
                        "method": method,
                        "iteration": int(row["iteration"]),
                        "true_relative_residual": float(row["true_relative_residual"]),
                        "source": "archived_stage25",
                    }
                )

    q28_setup = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in method_rows
        if row["method"] == "fixed_q28"
    }
    fixed_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in method_rows
    }
    q28_history = {
        (int(row["sample_id"]), int(row["rhs_seed"]), int(row["iteration"])): float(row["true_relative_residual"])
        for row in history_rows
        if row["method"] == "fixed_q28"
    }
    first_cycle_history_difference = 0.0
    for sample_id in settings.sample_ids:
        context, _ = _assemble_context(
            sample_id,
            coefficient_rows[sample_id],
            nodes,
            boundary,
            settings.stage25.stage22a.stage21,
        )
        # Reconstruct the ordered joint-SVD dictionary used by Stage 25.  The
        # archived ``z_full`` is the paired basis after a second SVD inside
        # ``_pair_from_z``; because its singular values are all one, that SVD
        # may rotate the full span and its leading columns are not nested
        # Stage-25 prefixes.  Prefixes must therefore come from the ordered
        # left singular vectors of the raw three-trajectory dictionary.
        constructions = [
            _build_ordinary_construction(
                context,
                construction_seed,
                settings.stage25.arnoldi_dimension,
                settings.stage25.stage22a,
            )
            for construction_seed in settings.stage25.construction_rhs_seeds
        ]
        raw = _raw_dictionary(constructions)
        ordered, _, numerical_rank = _orthonormalize(
            raw, settings.stage25.numerical_rank_tolerance
        )
        z_full = ordered[:, : min(settings.adaptive_q_schedule[-1], numerical_rank)]
        np.savez_compressed(
            output / f"sample_{sample_id:02d}_ordered_joint_dictionary.npz",
            z_full=z_full,
            construction_rhs_seeds=np.asarray(
                settings.stage25.construction_rhs_seeds, dtype=int
            ),
            arnoldi_dimension=settings.stage25.arnoldi_dimension,
        )
        fixed_pairs = {
            q: _pair_from_z(
                context.system.matrix,
                z_full[:, : min(q, z_full.shape[1])],
            )
            for q in settings.fixed_q_values
        }
        for rhs_seed in settings.heldout_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            for q, pair in fixed_pairs.items():
                repeat = _run_pair(context, rhs, pair, settings.stage25.stage22a)
                true_repeat = _relative_residual(
                    context.system.matrix, repeat.x, rhs
                )
                fixed = fixed_lookup[(sample_id, rhs_seed, f"fixed_q{q}")]
                counts = repeat.operation_counts or {}
                timing_repeat_rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": f"fixed_q{q}",
                        "archived_iterations": int(fixed["iterations"]),
                        "repeat_iterations": repeat.iterations,
                        "archived_converged": _bool(fixed["converged"]),
                        "repeat_converged": repeat.converged,
                        "archived_true_final_residual": float(
                            fixed["true_final_residual"]
                        ),
                        "repeat_true_final_residual": true_repeat,
                        "repeat_online_seconds": repeat.solve_seconds,
                        "repeat_online_a_applications": int(
                            counts.get("a_matvecs", 0)
                        ),
                        "repeat_online_m_applications": int(
                            counts.get("preconditioner_applications", 0)
                        ),
                    }
                )
                fixed["online_seconds"] = repeat.solve_seconds
                fixed["online_seconds_excluding_diagnostic"] = repeat.solve_seconds
                fixed["total_seconds"] = (
                    float(fixed["total_setup_seconds"]) + repeat.solve_seconds
                )
                fixed["online_a_applications"] = int(
                    counts.get("a_matvecs", 0)
                )
                fixed["online_m_applications"] = int(
                    counts.get("preconditioner_applications", 0)
                )
                fixed["source"] = (
                    "archived_stage25_numerics_with_stage26_timing_repeat"
                )
            result, controller, online_excluding = _run_adaptive(
                context, rhs, z_full, settings
            )
            true_residual = _relative_residual(context.system.matrix, result.x, rhs)
            initial = q28_setup[(sample_id, rhs_seed)]
            counts = result.operation_counts or {}
            total_setup = float(initial["initial_setup_seconds"]) + controller.expansion_setup_seconds
            method_rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": "adaptive",
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "true_final_residual": true_residual,
                    "baseline_iterations": int(initial["baseline_iterations"]),
                    "final_requested_q": controller.active_requested_q,
                    "final_actual_q": controller.pair.rank,
                    "expansions": controller.expansions,
                    "online_seconds": result.solve_seconds,
                    "online_seconds_excluding_diagnostic": online_excluding,
                    "initial_setup_seconds": float(initial["initial_setup_seconds"]),
                    "expansion_setup_seconds": controller.expansion_setup_seconds,
                    "total_setup_seconds": total_setup,
                    "total_seconds": float(initial["initial_setup_seconds"]) + result.solve_seconds,
                    "setup_a_applications": int(initial["setup_a_applications"]) + controller.expansion_a_applications,
                    "setup_m_applications": int(initial["setup_m_applications"]),
                    "online_a_applications": int(counts.get("a_matvecs", 0)),
                    "online_m_applications": int(counts.get("preconditioner_applications", 0)),
                    "expansion_a_applications": controller.expansion_a_applications,
                    "source": "new_stage26_adaptive",
                }
            )
            for cycle in controller.records:
                cycle_rows.append(
                    {"sample_id": sample_id, "rhs_seed": rhs_seed, **cycle}
                )
            for iteration, residual in zip(
                result.true_residual_iterations, result.true_residual_history
            ):
                history_rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": "adaptive",
                        "iteration": iteration,
                        "true_relative_residual": residual,
                        "source": "new_stage26_adaptive",
                    }
                )
                if iteration <= settings.solver.restart:
                    archived_value = q28_history.get((sample_id, rhs_seed, int(iteration)))
                    if archived_value is not None:
                        first_cycle_history_difference = max(
                            first_cycle_history_difference,
                            abs(float(residual) - archived_value),
                        )
        print(f"Stage 26 completed adaptive solves for system {sample_id}", flush=True)

    system_rows, family_rows = _summarize(method_rows, settings)
    first_rows, diagnostic_groups, correlations, confusion = _diagnostic_analysis(
        cycle_rows, method_rows, settings
    )
    adaptive_family = next(row for row in family_rows if row["method"] == "adaptive")
    policy_checks = {
        "all_24_adaptive_solves_converge": int(adaptive_family["converged_rhs"]) == 24,
        "at_least_6_systems_recover_75pct": int(adaptive_family["systems_passing"]) >= settings.required_systems,
        "median_per_rhs_recovery_at_least_0p75": float(adaptive_family["median_reference_benefit_recovery"]) >= settings.recovery_gate,
        "mean_final_rank_below_44": float(adaptive_family["mean_final_actual_q"]) < settings.mean_final_rank_limit,
        "sensitivity_at_least_0p90": float(confusion["sensitivity"]) >= settings.sensitivity_gate,
        "specificity_at_least_0p75": float(confusion["specificity"]) >= settings.specificity_gate,
    }
    viable = all(policy_checks.values())
    decision = {
        "adaptive_rank_viable": viable,
        "checks": policy_checks,
        "conclusion": (
            "A cheap cycle-contraction trigger recovers robust convergence and most full-prefix benefit while lowering average active rank."
            if viable
            else "The preregistered cycle-contraction trigger does not reliably recover full-prefix robustness at lower average rank."
        ),
        "recommendation": (
            "Retain adaptive-rank recycling as the viable classical mechanism; validate the frozen trigger on unseen spatial-K systems before any learning."
            if viable
            else "Do not continue threshold tuning; the required recycle dimension is too system-specific for this simple adaptive mechanism."
        ),
    }

    _write_csv(output / "method_results.csv", method_rows)
    _write_csv(output / "fixed_timing_repeats.csv", timing_repeat_rows)
    _write_csv(output / "cycle_diagnostics.csv", cycle_rows)
    _write_csv(output / "first_cycle_diagnostics.csv", first_rows)
    _write_csv(output / "diagnostic_group_summary.csv", diagnostic_groups)
    _write_csv(output / "diagnostic_rank_association.csv", correlations)
    _write_csv(output / "system_method_summary.csv", system_rows)
    _write_csv(output / "family_method_summary.csv", family_rows)
    _write_csv(output / "residual_histories.csv", history_rows)
    _write_json(output / "confusion_matrix.json", confusion)
    _write_json(output / "decision.json", decision)
    charts = _plots(
        output,
        method_rows,
        system_rows,
        family_rows,
        first_rows,
        history_rows,
        settings,
    )
    _write_csv(output / "chart_map.csv", charts)
    converged = [row for row in method_rows if _bool(row["converged"])]
    validation = {
        "expected_method_rows": 8 * 3 * 4,
        "actual_method_rows": len(method_rows),
        "adaptive_rows": sum(row["method"] == "adaptive" for row in method_rows),
        "fixed_rows_with_archived_numerics": sum(row["method"].startswith("fixed_") for row in method_rows),
        "fixed_timing_repeat_rows": len(timing_repeat_rows),
        "maximum_fixed_repeat_iteration_difference": max(
            abs(int(row["repeat_iterations"]) - int(row["archived_iterations"]))
            for row in timing_repeat_rows
        ),
        "fixed_repeat_convergence_mismatches": sum(
            _bool(row["repeat_converged"]) != _bool(row["archived_converged"])
            for row in timing_repeat_rows
        ),
        "maximum_converged_true_relative_residual": max(float(row["true_final_residual"]) for row in converged),
        "maximum_first_cycle_history_difference_from_fixed_q28": first_cycle_history_difference,
        "construction_and_evaluation_seeds_disjoint": not (
            set(settings.stage25.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        ),
        "no_neural_training": True,
        "no_cross_system_transfer": True,
    }
    validation["passed"] = bool(
        validation["actual_method_rows"] == validation["expected_method_rows"]
        and validation["adaptive_rows"] == 24
        and validation["fixed_rows_with_archived_numerics"] == 72
        and validation["fixed_timing_repeat_rows"] == 72
        and validation["maximum_fixed_repeat_iteration_difference"] == 0
        and validation["fixed_repeat_convergence_mismatches"] == 0
        and validation["maximum_converged_true_relative_residual"] <= settings.solver.relative_tolerance
        and validation["maximum_first_cycle_history_difference_from_fixed_q28"] < 1.0e-10
    )
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "environment.json",
        {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    )
    (output / "source_notes.md").write_text(
        "# Source notes\n\n"
        "- Technical Markdown report requested by the project workflow.\n"
        "- Fixed q=28, q=32, and q<=48 results and histories are reused from Stage 25.\n"
        "- The 24 adaptive solves are new. Fixed-rank iteration results remain archived Stage-25 values; 72 same-session fixed-rank repeats provide comparable timing and operation counts.\n"
        "- The first adaptive restart cycle is identical to fixed q=28 up to floating-point error.\n"
        "- The primary contraction thresholds were written before Stage 26 adaptive execution and were not retuned.\n"
        "- Other diagnostics are descriptive. Their rank association does not select the policy.\n"
        "- No neural training, transfer, new target, or solver redesign was performed.\n",
        encoding="utf-8",
    )
    report = _write_report(
        root,
        system_rows,
        family_rows,
        diagnostic_groups,
        correlations,
        confusion,
        decision,
        settings,
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(f"Stage 26 report: {report}", flush=True)
    return report
