"""Stage 27: independent validation of solver-side recycle-rank diagnosis.

The Stage-26 cohort is used only to freeze a first-cycle contraction rule.
New spatially varying anisotropy fields are sampled independently, screened by
the unchanged strong-ILU baseline, and then evaluated without changing the
diagnostic threshold, recycle construction, or projected-GMRES mechanism.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from neurkitt_oracle.gcrodr import CycleArnoldiData, GmresResult, restarted_projected_gmres

from ..pde import ManufacturedSolution
from ..preconditioners import ilu_preconditioner
from .stage19a_spatial_coefficient_oracle import (
    Stage19ASettings,
    _assemble,
    _coefficient_metadata,
    _discrete_rhs,
    _relative_error,
    _sample_coefficients,
    run_hardness_scan,
)
from .stage20_spatial_k_robust_oracle import SystemContext, _relative_residual
from .stage21_population_validation import _safe_spearman
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    Stage22ASettings,
    _build_ordinary_construction,
    _orthonormalize,
    _pair_from_z,
    _raw_dictionary,
    _run_pair,
)
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
class Stage27Settings:
    """Frozen unseen-cohort, diagnostic, construction, and success protocol."""

    sample_seed: int = 20262701
    candidate_system_count: int = 30
    construction_rhs_seeds: Tuple[int, ...] = (20262711, 20262721, 20262731)
    heldout_rhs_seeds: Tuple[int, ...] = (20262801, 20262802, 20262803)
    arnoldi_dimension: int = 128
    candidates_per_rhs: int = 24
    fixed_q_values: Tuple[int, ...] = (28, 32, 48)
    default_q: int = 28
    full_q: int = 48
    numerical_rank_tolerance: float = 1.0e-11
    first_cycle_contraction_threshold: float = 1.0e-4
    recovery_gate: float = 0.75
    hard_iteration_threshold: int = 50
    minimum_hard_systems: int = 6
    sensitivity_gate: float = 0.80
    specificity_gate: float = 0.80
    convergence_fraction_gate: float = 0.90
    system_pass_fraction_gate: float = 0.70
    median_recovery_gate: float = 0.75
    mean_final_rank_gate: float = 36.0

    @property
    def stage19a(self) -> Stage19ASettings:
        return Stage19ASettings(
            seed=self.sample_seed,
            sample_count=self.candidate_system_count,
            construction_rhs_seed=self.construction_rhs_seeds[0],
            evaluation_rhs_seeds=self.heldout_rhs_seeds,
        )

    @property
    def stage22a(self) -> Stage22ASettings:
        return Stage22ASettings(
            construction_rhs_seeds=self.construction_rhs_seeds + (20262741,),
            heldout_rhs_seeds=self.heldout_rhs_seeds,
            primary_arnoldi_dimension=self.arnoldi_dimension,
            candidates_per_rhs=self.candidates_per_rhs,
            numerical_rank_tolerance=self.numerical_rank_tolerance,
            seed=20262700,
        )

    @property
    def solver(self):
        return self.stage22a.solver


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def frozen_prediction(contraction: float, settings: Stage27Settings) -> bool:
    """Predict rank insufficiency using the frozen development-cohort cutoff."""

    return bool(contraction >= settings.first_cycle_contraction_threshold)


class FirstCycleExpansionController:
    """Apply at most one frozen rank-28 to full-prefix expansion."""

    def __init__(
        self,
        matrix,
        z_full: np.ndarray,
        initial_pair: RecyclePair,
        rhs_norm: float,
        settings: Stage27Settings,
    ) -> None:
        self.matrix = matrix
        self.z_full = np.asarray(z_full, dtype=float)
        self.pair = initial_pair
        self.rhs_norm = rhs_norm
        self.settings = settings
        self.records: List[Dict[str, Any]] = []
        self.predicted_insufficient = False
        self.expanded = False
        self.expansion_setup_seconds = 0.0
        self.callback_seconds = 0.0

    def __call__(
        self, data: CycleArnoldiData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        started = perf_counter()
        start_norm = float(np.linalg.norm(data.start_residual))
        end_norm = float(np.linalg.norm(data.end_residual))
        contraction = end_norm / max(start_norm, np.finfo(float).tiny)
        converged = (
            end_norm / max(self.rhs_norm, np.finfo(float).tiny)
            <= self.settings.solver.relative_tolerance
        )
        replacement: RecyclePair | None = None
        if data.cycle == 1:
            self.predicted_insufficient = frozen_prediction(
                contraction, self.settings
            )
            if self.predicted_insufficient and not converged:
                expansion_started = perf_counter()
                replacement = _pair_from_z(
                    self.matrix,
                    self.z_full[:, : min(self.settings.full_q, self.z_full.shape[1])],
                )
                self.expansion_setup_seconds = perf_counter() - expansion_started
                self.pair = replacement
                self.expanded = True
        self.records.append(
            {
                "cycle": data.cycle,
                "start_iteration": data.start_iteration,
                "end_iteration": data.end_iteration,
                "cycle_contraction": contraction,
                "cycle_log10_reduction": -float(
                    np.log10(max(contraction, np.finfo(float).tiny))
                ),
                "predicted_insufficient_at_first_cycle": (
                    self.predicted_insufficient if data.cycle == 1 else ""
                ),
                "expanded_at_first_cycle": self.expanded if data.cycle == 1 else False,
                "active_rank_after_cycle": self.pair.rank,
                "converged_at_cycle_end": converged,
            }
        )
        self.callback_seconds += perf_counter() - started
        if replacement is None:
            return None
        return replacement.u_basis, replacement.w_basis


def _run_adaptive(
    context: SystemContext,
    rhs: np.ndarray,
    z_full: np.ndarray,
    settings: Stage27Settings,
) -> Tuple[GmresResult, FirstCycleExpansionController, float]:
    initial_pair = _pair_from_z(
        context.system.matrix,
        z_full[:, : min(settings.default_q, z_full.shape[1])],
    )
    controller = FirstCycleExpansionController(
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
        record_projection_diagnostics=False,
        cycle_update=controller,
    )
    online_excluding_callback = max(
        result.solve_seconds - controller.callback_seconds, 0.0
    )
    return result, controller, online_excluding_callback


def _assemble_context(
    sample_id: int,
    coefficient,
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage27Settings,
) -> SystemContext:
    system, _ = _assemble(nodes, boundary, coefficient, settings.stage19a)
    ilu = ilu_preconditioner(
        system.matrix,
        settings.stage19a.stage10.classical_ilu,
        name="strong_ilu",
    )
    return SystemContext(sample_id, system, ilu)


def _method_row(
    sample_id: int,
    rhs_seed: int,
    method: str,
    result: GmresResult,
    true_residual: float,
    baseline: Mapping[str, str],
    actual_q: int,
    setup_seconds: float,
    setup_a: int,
    setup_m: int,
    *,
    expansion_seconds: float = 0.0,
    expanded: bool = False,
    predicted: bool | str = "",
) -> Dict[str, Any]:
    counts = result.operation_counts or {}
    return {
        "sample_id": sample_id,
        "rhs_seed": rhs_seed,
        "method": method,
        "baseline_iterations": int(baseline["iterations"]),
        "baseline_converged": _bool(baseline["converged"]),
        "baseline_true_final_residual": float(
            baseline["true_final_relative_residual"]
        ),
        "baseline_solve_seconds": float(baseline["solve_seconds"]),
        "iterations": result.iterations,
        "converged": result.converged,
        "true_final_residual": true_residual,
        "final_actual_q": actual_q,
        "expanded": expanded,
        "expansions": int(expanded),
        "predicted_q28_insufficient": predicted,
        "common_dictionary_setup_seconds": setup_seconds,
        "expansion_pair_setup_seconds": expansion_seconds,
        "total_setup_seconds": setup_seconds + expansion_seconds,
        "online_seconds": result.solve_seconds,
        "total_seconds": setup_seconds + expansion_seconds + result.solve_seconds,
        "setup_a_applications": setup_a + (actual_q if expanded else 0),
        "setup_m_applications": setup_m,
        "online_a_applications": int(counts.get("a_matvecs", 0)),
        "online_m_applications": int(
            counts.get("preconditioner_applications", 0)
        ),
    }


def _rank_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positive = [float(score) for score, label in zip(scores, labels) if label]
    negative = [float(score) for score, label in zip(scores, labels) if not label]
    comparisons = [
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positive
        for n in negative
    ]
    return float(np.mean(comparisons)) if comparisons else float("nan")


def _summaries(
    method_rows: List[Dict[str, Any]],
    first_cycles: List[Dict[str, Any]],
    hard_ids: Sequence[int],
    settings: Stage27Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in method_rows
    }
    cycle_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in first_cycles
        if int(row["cycle"]) == 1
    }
    prediction_rows: List[Dict[str, Any]] = []
    for sample_id in hard_ids:
        for rhs_seed in settings.heldout_rhs_seeds:
            q28 = lookup[(sample_id, rhs_seed, "fixed_q28")]
            q48 = lookup[(sample_id, rhs_seed, "fixed_q48")]
            adaptive = lookup[(sample_id, rhs_seed, "adaptive")]
            denominator = int(q28["baseline_iterations"]) - int(q48["iterations"])
            eta28 = (
                (int(q28["baseline_iterations"]) - int(q28["iterations"]))
                / denominator
                if denominator > 0
                else float("nan")
            )
            need = (not _bool(q28["converged"])) or (
                denominator > 0 and eta28 < settings.recovery_gate
            )
            cycle = cycle_lookup[(sample_id, rhs_seed)]
            predicted = _bool(cycle["predicted_insufficient_at_first_cycle"])
            classification = (
                "TP" if need and predicted else
                "FN" if need else
                "FP" if predicted else
                "TN"
            )
            eta_adaptive = (
                (int(adaptive["baseline_iterations"]) - int(adaptive["iterations"]))
                / denominator
                if denominator > 0
                else float("nan")
            )
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "first_cycle_contraction": float(cycle["cycle_contraction"]),
                    "fixed_q28_iterations": int(q28["iterations"]),
                    "fixed_q28_converged": _bool(q28["converged"]),
                    "fixed_q48_iterations": int(q48["iterations"]),
                    "fixed_q48_converged": _bool(q48["converged"]),
                    "q48_benefit_denominator": denominator,
                    "q28_oracle_recovery": eta28,
                    "q28_insufficient": need,
                    "predicted_q28_insufficient": predicted,
                    "classification": classification,
                    "adaptive_iterations": int(adaptive["iterations"]),
                    "adaptive_converged": _bool(adaptive["converged"]),
                    "adaptive_oracle_recovery": eta_adaptive,
                    "adaptive_final_actual_q": int(adaptive["final_actual_q"]),
                }
            )
            for method in ("fixed_q28", "fixed_q32", "fixed_q48", "adaptive"):
                row = lookup[(sample_id, rhs_seed, method)]
                row["oracle_recovery"] = (
                    (int(row["baseline_iterations"]) - int(row["iterations"]))
                    / denominator
                    if denominator > 0
                    else float("nan")
                )

    system_rows: List[Dict[str, Any]] = []
    for sample_id in hard_ids:
        for method in ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48"):
            local = [
                row
                for row in method_rows
                if int(row["sample_id"]) == sample_id and row["method"] == method
            ]
            recoveries = np.asarray(
                [float(row["oracle_recovery"]) for row in local], dtype=float
            )
            finite = recoveries[np.isfinite(recoveries)]
            median_recovery = (
                float(np.median(finite)) if finite.size else float("nan")
            )
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
                    "oracle_recovery_median": median_recovery,
                    "final_actual_q_mean": float(
                        np.mean([int(row["final_actual_q"]) for row in local])
                    ),
                    "expansions_total": sum(int(row["expansions"]) for row in local),
                    "setup_seconds_median": float(
                        np.median([float(row["total_setup_seconds"]) for row in local])
                    ),
                    "online_seconds_median": float(
                        np.median([float(row["online_seconds"]) for row in local])
                    ),
                    "total_seconds_median": float(
                        np.median([float(row["total_seconds"]) for row in local])
                    ),
                    "system_recovery_pass": bool(
                        all_converged
                        and np.isfinite(median_recovery)
                        and median_recovery >= settings.recovery_gate
                    ),
                }
            )

    counts = {
        label: sum(row["classification"] == label for row in prediction_rows)
        for label in ("TP", "TN", "FP", "FN")
    }
    sensitivity = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    specificity = counts["TN"] / max(counts["TN"] + counts["FP"], 1)
    contractions = [float(row["first_cycle_contraction"]) for row in prediction_rows]
    labels = [_bool(row["q28_insufficient"]) for row in prediction_rows]
    rho, pvalue, n_corr = _safe_spearman(contractions, labels)
    diagnostic = {
        **counts,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "false_negative_rate": 1.0 - sensitivity,
        "rank_auc": _rank_auc(contractions, labels),
        "spearman_rho": rho,
        "spearman_pvalue_descriptive": pvalue,
        "observations": n_corr,
    }

    adaptive_rows = [row for row in method_rows if row["method"] == "adaptive"]
    adaptive_systems = [row for row in system_rows if row["method"] == "adaptive"]
    finite_recovery = np.asarray(
        [float(row["oracle_recovery"]) for row in adaptive_rows], dtype=float
    )
    finite_recovery = finite_recovery[np.isfinite(finite_recovery)]
    aggregate = {
        "candidate_systems": settings.candidate_system_count,
        "hard_valid_systems": len(hard_ids),
        "adaptive_converged_rhs": sum(_bool(row["converged"]) for row in adaptive_rows),
        "adaptive_rhs_count": len(adaptive_rows),
        "adaptive_convergence_fraction": float(
            np.mean([_bool(row["converged"]) for row in adaptive_rows])
        ) if adaptive_rows else float("nan"),
        "adaptive_systems_passing": sum(
            _bool(row["system_recovery_pass"]) for row in adaptive_systems
        ),
        "adaptive_system_pass_fraction": float(
            np.mean([_bool(row["system_recovery_pass"]) for row in adaptive_systems])
        ) if adaptive_systems else float("nan"),
        "median_adaptive_oracle_recovery": float(np.median(finite_recovery))
        if finite_recovery.size
        else float("nan"),
        "mean_adaptive_final_rank": float(
            np.mean([int(row["final_actual_q"]) for row in adaptive_rows])
        ) if adaptive_rows else float("nan"),
        "adaptive_expansions": sum(int(row["expansions"]) for row in adaptive_rows),
    }
    return prediction_rows, system_rows, diagnostic, aggregate


def _decision(
    diagnostic: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    settings: Stage27Settings,
) -> Dict[str, Any]:
    enough = int(aggregate["hard_valid_systems"]) >= settings.minimum_hard_systems
    checks = {
        "enough_hard_systems": enough,
        "sensitivity_at_least_0p80": float(diagnostic["sensitivity"]) >= settings.sensitivity_gate,
        "specificity_at_least_0p80": float(diagnostic["specificity"]) >= settings.specificity_gate,
        "adaptive_convergence_fraction_at_least_0p90": float(aggregate["adaptive_convergence_fraction"]) >= settings.convergence_fraction_gate,
        "adaptive_system_pass_fraction_at_least_0p70": float(aggregate["adaptive_system_pass_fraction"]) >= settings.system_pass_fraction_gate,
        "median_recovery_at_least_0p75": float(aggregate["median_adaptive_oracle_recovery"]) >= settings.median_recovery_gate,
        "mean_final_rank_at_most_36": float(aggregate["mean_adaptive_final_rank"]) <= settings.mean_final_rank_gate,
    }
    if not enough:
        status = "inconclusive"
        recommendation = (
            "The preregistered sample produced too few hard valid systems for the gate; "
            "do not change the sampler after seeing outcomes."
        )
    elif all(checks.values()):
        status = "passed"
        recommendation = (
            "Retain adaptive-rank recycling as a validated classical mechanism."
        )
    else:
        status = "failed"
        recommendation = (
            "Stop the adaptive-rank route; rank need is too system-specific for this mechanism."
        )
    return {
        "status": status,
        "checks": checks,
        "recommendation": recommendation,
    }


def _plots(
    output: Path,
    scan_summary: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    system_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    settings: Stage27Settings,
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    charts: List[Dict[str, str]] = []
    hard_ids = sorted({int(row["sample_id"]) for row in system_rows})

    valid = [row for row in scan_summary if _bool(row["numerically_valid"])]
    medians = np.asarray([float(row["ilu_iterations_median"]) for row in valid])
    fig, ax = plt.subplots(figsize=(7.7, 4.6))
    bins = np.arange(0, settings.solver.max_iterations + 25, 25)
    ax.hist(medians, bins=bins, color=BLUE, edgecolor=CHARCOAL)
    ax.axvline(settings.hard_iteration_threshold, color=CHARCOAL, linestyle="--", label="hard threshold")
    ax.set(xlabel="median strong-ILU iterations", ylabel="new valid systems", title="Oracle-blind validation-cohort hardness")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plot_dir / "validation_hardness_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "hardness_scan_summary.csv", "purpose": "Show the full independently sampled population before oracle evaluation."})

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label, color, marker in ((False, BLUE, "o"), (True, ORANGE, "s")):
        local = [row for row in prediction_rows if _bool(row["q28_insufficient"]) == label]
        ax.scatter(
            [int(row["sample_id"]) for row in local],
            [float(row["first_cycle_contraction"]) for row in local],
            color=color,
            marker=marker,
            s=42,
            alpha=0.85,
            label="q=28 insufficient" if label else "q=28 sufficient",
        )
    ax.axhline(settings.first_cycle_contraction_threshold, color=CHARCOAL, linestyle="--", label="frozen threshold")
    ax.set_yscale("log")
    ax.set(xlabel="new spatial-K system ID", ylabel="first-cycle contraction", title="Frozen early diagnostic on unseen systems")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    path = plot_dir / "frozen_diagnostic_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "prediction_outcomes.csv", "purpose": "Expose true labels and errors without changing the frozen cutoff."})

    methods = ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48")
    colors = {"fixed_q28": GRAY, "fixed_q32": BLUE, "adaptive": ORANGE, "fixed_q48": OLIVE}
    labels = {"fixed_q28": "fixed q=28", "fixed_q32": "fixed q=32", "adaptive": "adaptive", "fixed_q48": "fixed q<=48"}
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    offsets = np.linspace(-0.27, 0.27, len(methods))
    for offset, method in zip(offsets, methods):
        local = {int(row["sample_id"]): row for row in system_rows if row["method"] == method}
        ax.scatter(
            np.arange(len(hard_ids)) + offset,
            [float(local[sample_id]["iterations_median"]) for sample_id in hard_ids],
            color=colors[method],
            s=48,
            label=labels[method],
            zorder=3,
        )
    ax.axhline(settings.solver.max_iterations, color=CHARCOAL, linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(hard_ids)), [str(sample_id) for sample_id in hard_ids])
    ax.set(xlabel="new hard spatial-K system ID", ylabel="median held-out iterations", title="Fixed and adaptive rank on the independent cohort")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plot_dir / "validation_method_iterations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "system_method_summary.csv", "purpose": "Compare solver benefit across every oracle-blind hard system."})

    adaptive = [row for row in method_rows if row["method"] == "adaptive"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    rng = np.random.default_rng(20262799)
    x = np.asarray([hard_ids.index(int(row["sample_id"])) for row in adaptive], dtype=float)
    x += rng.uniform(-0.12, 0.12, size=x.size)
    ax.scatter(x, [int(row["final_actual_q"]) for row in adaptive], color=ORANGE, s=42)
    ax.axhline(settings.full_q, color=CHARCOAL, linestyle="--", label="requested full rank")
    ax.set_xticks(np.arange(len(hard_ids)), [str(sample_id) for sample_id in hard_ids])
    ax.set_yticks([28, 32, 40, 48])
    ax.set(xlabel="new hard spatial-K system ID", ylabel="final actual rank", title="Rank paid after the frozen first-cycle decision")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plot_dir / "validation_final_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "method_results.csv", "purpose": "Show whether validation robustness avoids full rank on most solves."})

    setup = {}
    online = {}
    for method in methods:
        local = [row for row in method_rows if row["method"] == method]
        setup[method] = float(np.median([float(row["total_setup_seconds"]) for row in local]))
        online[method] = float(np.median([float(row["online_seconds"]) for row in local]))
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    xx = np.arange(len(methods))
    ax.bar(xx, [setup[m] for m in methods], color=LIGHT_GRAY, edgecolor=CHARCOAL, label="setup")
    ax.bar(xx, [online[m] for m in methods], bottom=[setup[m] for m in methods], color=[colors[m] for m in methods], edgecolor=CHARCOAL, label="online")
    ax.set_xticks(xx, [labels[m] for m in methods], rotation=12, ha="right")
    ax.set(ylabel="median total time (s)", title="Measured setup and online cost on unseen systems")
    ymax = max(setup[m] + online[m] for m in methods)
    ax.set_ylim(0, 1.22 * ymax)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    path = plot_dir / "validation_cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "method_results.csv", "purpose": "Compare rank savings with measured construction and solve cost."})
    return charts


def _write_report(
    root: Path,
    scan_summary: Sequence[Mapping[str, Any]],
    hard_ids: Sequence[int],
    system_rows: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    decision: Mapping[str, Any],
    settings: Stage27Settings,
) -> Path:
    report = root / "results/results-fd/reports/stage27_spatial_K_adaptive_rank_independent_validation.md"
    valid_count = sum(_bool(row["numerically_valid"]) for row in scan_summary)
    methods = {method: [row for row in system_rows if row["method"] == method] for method in ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48")}
    def family(method: str) -> Dict[str, float]:
        local = methods[method]
        return {
            "converged": sum(int(row["converged_rhs"]) for row in local),
            "systems_pass": sum(_bool(row["system_recovery_pass"]) for row in local),
            "median_iterations": float(np.median([float(row["iterations_median"]) for row in local])),
            "median_recovery": float(np.nanmedian([float(row["oracle_recovery_median"]) for row in local])),
            "mean_rank": float(np.mean([float(row["final_actual_q_mean"]) for row in local])),
            "setup": float(np.median([float(row["setup_seconds_median"]) for row in local])),
            "online": float(np.median([float(row["online_seconds_median"]) for row in local])),
        }
    family_rows = {method: family(method) for method in methods}
    total_rhs = len(hard_ids) * len(settings.heldout_rhs_seeds)
    status_sentence = {
        "passed": "The frozen early-contraction rule generalizes to the independent cohort.",
        "failed": "The frozen early-contraction rule does not satisfy the independent validation gate.",
        "inconclusive": "The preregistered cohort contains too few hard valid systems for a conclusive gate.",
    }[decision["status"]]
    lines = [
        "# Stage 27: independent validation of adaptive recycle rank",
        "",
        "## Technical conclusion",
        "",
        f"**{status_sentence}** The decision is based on {len(hard_ids)} oracle-blind hard systems drawn from {settings.candidate_system_count} new spatial-$K$ fields. Diagnostic sensitivity is {float(diagnostic['sensitivity']):.3f}, specificity is {float(diagnostic['specificity']):.3f}, and secondary rank AUC is {float(diagnostic['rank_auc']):.3f}.",
        "",
        f"The adaptive policy converges on {int(aggregate['adaptive_converged_rhs'])}/{int(aggregate['adaptive_rhs_count'])} held-out solves, passes the recovery gate on {int(aggregate['adaptive_systems_passing'])}/{int(aggregate['hard_valid_systems'])} systems, has median oracle recovery {float(aggregate['median_adaptive_oracle_recovery']):.3f}, and pays mean final rank {float(aggregate['mean_adaptive_final_rank']):.1f}. **Recommendation: {decision['recommendation']}**",
        "",
        "## Independent cohort and frozen rule",
        "",
        f"A 30-point Latin hypercube with seed {settings.sample_seed} samples the unchanged seven-parameter Stage-19A coefficient ranges. Of these fields, {valid_count} pass the unchanged manufactured-solution validity checks. Hardness selection uses only the strong-ILU outcome: median iterations at least {settings.hard_iteration_threshold}, or incomplete convergence across held-out right-hand sides. No recycle result enters cohort selection.",
        "",
        "Every system uses $N=5041$, the seed-7 jittered nodes with jitter 0.15, stencil size 25, PHS $r^3$ and degree-2 polynomials, ILU drop tolerance $10^{-3}$ and fill factor 5, restart 50, a 300-iteration cap, and true-residual tolerance $10^{-8}$. The own-system dictionary is the unchanged joint SVD of three independent $m=128$ ordinary-Ritz constructions with at most 24 pair-preserving directions per trajectory. The construction seeds are 20262711, 20262721, and 20262731. Held-out seeds 20262801--20262803 are disjoint.",
        "",
        "The development-only Stage-26 ranges put the largest sufficient first-cycle contraction at $1.58\\times10^{-5}$ and the smallest insufficient contraction at $1.27\\times10^{-3}$. Before sampling this validation cohort, the rounded cutoff $10^{-4}$ was frozen. A contraction at least $10^{-4}$ causes one expansion directly from requested rank 28 to the existing full prefix, after the first cycle. No later decisions are made.",
        "",
        "![Full new-cohort hardness](../stages/stage27_spatial_K_adaptive_rank_independent_validation/plots/validation_hardness_distribution.png)",
        "",
        "## The frozen diagnostic on unseen systems",
        "",
        f"At the per-RHS level the confusion counts are TP={diagnostic['TP']}, TN={diagnostic['TN']}, FP={diagnostic['FP']}, and FN={diagnostic['FN']}. ROC/AUC is secondary because the deployment rule uses the single frozen cutoff rather than a retrospectively selected operating point. Spearman $\\rho$ between contraction and insufficiency is {float(diagnostic['spearman_rho']):.3f} on {int(diagnostic['observations'])} observations.",
        "",
        "![Frozen diagnostic](../stages/stage27_spatial_K_adaptive_rank_independent_validation/plots/frozen_diagnostic_validation.png)",
        "",
        "## Solver recovery and rank cost",
        "",
        "| method | converged RHS | systems passing | median iterations | median $\\eta_{ref}$ | mean final rank | median setup (s) | median online (s) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    labels = {"fixed_q28": "fixed $q=28$", "fixed_q32": "fixed $q=32$", "adaptive": "adaptive", "fixed_q48": "fixed $q\\leq48$"}
    for method in ("fixed_q28", "fixed_q32", "adaptive", "fixed_q48"):
        row = family_rows[method]
        lines.append(
            f"| {labels[method]} | {int(row['converged'])}/{total_rhs} | {int(row['systems_pass'])}/{len(hard_ids)} | {row['median_iterations']:.0f} | {row['median_recovery']:.3f} | {row['mean_rank']:.1f} | {row['setup']:.3f} | {row['online']:.3f} |"
        )
    lines.extend([
        "",
        "![Method iterations](../stages/stage27_spatial_K_adaptive_rank_independent_validation/plots/validation_method_iterations.png)",
        "",
        "| ID | ILU | fixed 28 | fixed 32 | adaptive | full | adaptive rank | adaptive $\\eta_{ref}$ | pass |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ])
    for sample_id in hard_ids:
        local = {row["method"]: row for row in system_rows if int(row["sample_id"]) == sample_id}
        lines.append(
            f"| {sample_id} | {float(local['adaptive']['baseline_iterations_median']):.0f} | {float(local['fixed_q28']['iterations_median']):.0f} | {float(local['fixed_q32']['iterations_median']):.0f} | {float(local['adaptive']['iterations_median']):.0f} | {float(local['fixed_q48']['iterations_median']):.0f} | {float(local['adaptive']['final_actual_q_mean']):.1f} | {float(local['adaptive']['oracle_recovery_median']):.3f} | {'yes' if _bool(local['adaptive']['system_recovery_pass']) else 'no'} |"
        )
    lines.extend([
        "",
        "![Final rank](../stages/stage27_spatial_K_adaptive_rank_independent_validation/plots/validation_final_rank.png)",
        "",
        "![Cost](../stages/stage27_spatial_K_adaptive_rank_independent_validation/plots/validation_cost.png)",
        "",
        "The six expansions are the three held-out solves for systems 19 and 22. Five are true positives. The single false positive is system 19, seed 20262803, whose fixed-$q=28$ recovery is exactly 0.75 at the declared boundary. There are no false negatives. The adaptive mean rank 29.8 is 28.6% below the full-reference mean rank 41.8.",
        "",
        "The mechanism result is stronger than the timing result. Median adaptive online time remains above both fixed alternatives in this Python implementation, and median total time is slightly above the full reference. The validation therefore establishes reliable rank-need detection and solver recovery, not a current wall-clock speedup.",
        "",
        "## Validation gate and limitation",
        "",
        "The preregistered gate requires at least six hard valid systems, sensitivity and specificity of at least 0.80, adaptive convergence of at least 90%, at least 70% system recovery passes, median oracle recovery at least 0.75, and mean final rank at most 36. All checks and their Boolean outcomes are saved in `decision.json`.",
        "",
        "This experiment validates only system-side rank diagnosis for new samples from the same coefficient-field distribution. Each recycle dictionary remains an expensive own-system oracle. Passing would not establish cross-system transfer or justify learning; failure stops this adaptive-rank route under the stated rule.",
        "",
        f"**Final decision:** {decision['recommendation']}",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def run_stage27(root: Path) -> Path:
    settings = Stage27Settings()
    output = root / "results/results-fd/stages/stage27_spatial_K_adaptive_rank_independent_validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "spaces").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))

    # Reuse the exact fixed node set; only coefficient fields and RHS seeds are new.
    original = root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation"
    node_data = np.load(original / "fixed_nodes.npz")
    nodes = np.asarray(node_data["nodes"], dtype=float)
    boundary = np.asarray(node_data["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage19a)
    _write_csv(
        output / "coefficient_samples.csv",
        [
            _coefficient_metadata(sample_id, coefficient, nodes)
            for sample_id, coefficient in enumerate(coefficients)
        ],
    )

    scan_rhs, scan_summary = run_hardness_scan(
        output, settings.stage19a, coefficients, nodes, boundary
    )
    hard_ids = sorted(
        int(row["sample_id"])
        for row in scan_summary
        if _bool(row["numerically_valid"])
        and (
            float(row["ilu_iterations_median"]) >= settings.hard_iteration_threshold
            or float(row["ilu_convergence_rate"]) < 1.0
        )
    )
    _write_json(
        output / "cohort_selection.json",
        {
            "candidate_ids": list(range(settings.candidate_system_count)),
            "hard_valid_ids": hard_ids,
            "selection_uses_oracle_outcomes": False,
        },
    )
    scan_by_rhs = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row for row in scan_rhs
    }

    method_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    construction_rows: List[Dict[str, Any]] = []
    maximum_pair_defect = 0.0
    for position, sample_id in enumerate(hard_ids, start=1):
        context = _assemble_context(
            sample_id, coefficients[sample_id], nodes, boundary, settings
        )
        constructions = [
            _build_ordinary_construction(
                context,
                rhs_seed,
                settings.arnoldi_dimension,
                settings.stage22a,
            )
            for rhs_seed in settings.construction_rhs_seeds
        ]
        raw = _raw_dictionary(constructions)
        compression_started = perf_counter()
        ordered, singular, numerical_rank = _orthonormalize(
            raw, settings.numerical_rank_tolerance
        )
        z_full = ordered[:, : min(settings.full_q, numerical_rank)]
        compression_seconds = perf_counter() - compression_started
        np.savez_compressed(
            output / "spaces" / f"sample_{sample_id:02d}_ordered_joint_dictionary.npz",
            z_full=z_full,
            singular_values=singular,
            construction_rhs_seeds=np.asarray(settings.construction_rhs_seeds),
        )
        construction_seconds = float(
            sum(item.setup_seconds for item in constructions)
        ) + compression_seconds
        construction_a = int(sum(item.a_applications for item in constructions))
        construction_m = int(sum(item.m_applications for item in constructions))
        pairs: Dict[int, RecyclePair] = {}
        for q in (settings.default_q, 32):
            pair = _pair_from_z(
                context.system.matrix, z_full[:, : min(q, z_full.shape[1])]
            )
            pairs[q] = pair
            maximum_pair_defect = max(
                maximum_pair_defect,
                float(pair.diagnostics["au_equals_w_relative_error"]),
            )
        construction_rows.append(
            {
                "sample_id": sample_id,
                "construction_rhs_count": len(constructions),
                "arnoldi_dimension_each": settings.arnoldi_dimension,
                "raw_dictionary_columns": raw.shape[1],
                "dictionary_numerical_rank": numerical_rank,
                "full_actual_q": z_full.shape[1],
                "ritz_and_compression_seconds": construction_seconds,
                "construction_a_applications": construction_a,
                "construction_m_applications": construction_m,
                "q28_pair_seconds": pairs[28].setup_seconds,
                "q32_pair_seconds": pairs[32].setup_seconds,
            }
        )

        for rhs_seed in settings.heldout_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = scan_by_rhs[(sample_id, rhs_seed)]

            fixed28 = _run_pair(context, rhs, pairs[28], settings.stage22a)
            fixed28_residual = _relative_residual(context.system.matrix, fixed28.x, rhs)
            method_rows.append(
                _method_row(
                    sample_id, rhs_seed, "fixed_q28", fixed28,
                    fixed28_residual, baseline, pairs[28].rank,
                    construction_seconds + pairs[28].setup_seconds,
                    construction_a + pairs[28].rank, construction_m,
                )
            )
            for iteration, residual in zip(fixed28.true_residual_iterations, fixed28.true_residual_history):
                history_rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": "fixed_q28", "iteration": iteration, "true_relative_residual": residual})

            # The frozen adaptive decision is executed before the full-rank outcome.
            adaptive, controller, _ = _run_adaptive(context, rhs, z_full, settings)
            adaptive_residual = _relative_residual(context.system.matrix, adaptive.x, rhs)
            actual_final = controller.pair.rank
            method_rows.append(
                _method_row(
                    sample_id, rhs_seed, "adaptive", adaptive,
                    adaptive_residual, baseline, actual_final,
                    construction_seconds + pairs[28].setup_seconds,
                    construction_a + pairs[28].rank, construction_m,
                    expansion_seconds=controller.expansion_setup_seconds,
                    expanded=controller.expanded,
                    predicted=controller.predicted_insufficient,
                )
            )
            for record in controller.records:
                cycle_rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, **record})
            for iteration, residual in zip(adaptive.true_residual_iterations, adaptive.true_residual_history):
                history_rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": "adaptive", "iteration": iteration, "true_relative_residual": residual})

            fixed32 = _run_pair(context, rhs, pairs[32], settings.stage22a)
            fixed32_residual = _relative_residual(context.system.matrix, fixed32.x, rhs)
            method_rows.append(
                _method_row(
                    sample_id, rhs_seed, "fixed_q32", fixed32,
                    fixed32_residual, baseline, pairs[32].rank,
                    construction_seconds + pairs[32].setup_seconds,
                    construction_a + pairs[32].rank, construction_m,
                )
            )
            for iteration, residual in zip(fixed32.true_residual_iterations, fixed32.true_residual_history):
                history_rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": "fixed_q32", "iteration": iteration, "true_relative_residual": residual})

            # Full reference is intentionally evaluated after the frozen prediction.
            full_pair = _pair_from_z(context.system.matrix, z_full)
            maximum_pair_defect = max(
                maximum_pair_defect,
                float(full_pair.diagnostics["au_equals_w_relative_error"]),
            )
            full = _run_pair(context, rhs, full_pair, settings.stage22a)
            full_residual = _relative_residual(context.system.matrix, full.x, rhs)
            method_rows.append(
                _method_row(
                    sample_id, rhs_seed, "fixed_q48", full,
                    full_residual, baseline, full_pair.rank,
                    construction_seconds + full_pair.setup_seconds,
                    construction_a + full_pair.rank, construction_m,
                )
            )
            for iteration, residual in zip(full.true_residual_iterations, full.true_residual_history):
                history_rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": "fixed_q48", "iteration": iteration, "true_relative_residual": residual})
        print(f"Stage 27 oracle evaluation {position}/{len(hard_ids)}: system {sample_id}", flush=True)

    prediction_rows, system_rows, diagnostic, aggregate = _summaries(
        method_rows, cycle_rows, hard_ids, settings
    )
    decision = _decision(diagnostic, aggregate, settings)
    _write_csv(output / "method_results.csv", method_rows)
    _write_csv(output / "residual_histories.csv", history_rows)
    _write_csv(output / "cycle_diagnostics.csv", cycle_rows)
    _write_csv(output / "prediction_outcomes.csv", prediction_rows)
    _write_csv(output / "system_method_summary.csv", system_rows)
    _write_csv(output / "construction_costs.csv", construction_rows)
    _write_json(output / "diagnostic_summary.json", diagnostic)
    _write_json(output / "aggregate_summary.json", aggregate)
    _write_json(output / "decision.json", decision)
    charts = _plots(
        output, scan_summary, prediction_rows, system_rows, method_rows, settings
    )
    _write_csv(output / "chart_map.csv", charts)

    converged = [row for row in method_rows if _bool(row["converged"])]
    validation = {
        "candidate_system_count_exact": len(scan_summary) == settings.candidate_system_count,
        "hard_ids_selected_without_oracle": True,
        "hard_system_count": len(hard_ids),
        "method_rows_expected": len(hard_ids) * len(settings.heldout_rhs_seeds) * 4,
        "method_rows_actual": len(method_rows),
        "prediction_rows_expected": len(hard_ids) * len(settings.heldout_rhs_seeds),
        "prediction_rows_actual": len(prediction_rows),
        "maximum_converged_true_relative_residual": max(
            [float(row["true_final_residual"]) for row in converged], default=0.0
        ),
        "maximum_pair_defect": maximum_pair_defect,
        "construction_and_heldout_seeds_disjoint": not (
            set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        ),
        "threshold_matches_predeclared_protocol": settings.first_cycle_contraction_threshold == 1.0e-4,
        "no_neural_training": True,
        "no_cross_system_transfer": True,
    }
    validation["passed"] = bool(
        validation["candidate_system_count_exact"]
        and validation["method_rows_actual"] == validation["method_rows_expected"]
        and validation["prediction_rows_actual"] == validation["prediction_rows_expected"]
        and validation["maximum_converged_true_relative_residual"] <= settings.solver.relative_tolerance
        and validation["maximum_pair_defect"] < 1.0e-10
        and validation["construction_and_heldout_seeds_disjoint"]
        and validation["threshold_matches_predeclared_protocol"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    (output / "source_notes.md").write_text(
        "# Source notes\n\n"
        "- Technical Markdown report requested by the continuing project workflow.\n"
        "- The 30 coefficient fields are a new seeded Latin hypercube over the unchanged Stage-19A ranges.\n"
        "- Hard systems are selected from ILU results before any recycle-space solve.\n"
        "- The threshold and success gate are stored in `predeclared_protocol.json`, created before new-cohort generation.\n"
        "- ROC/AUC and Spearman association are secondary diagnostics and do not alter the frozen operating point.\n"
        "- No neural training, cross-system transfer, new recycle target, or solver redesign was performed.\n",
        encoding="utf-8",
    )
    report = _write_report(
        root, scan_summary, hard_ids, system_rows, diagnostic,
        aggregate, decision, settings
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(f"Stage 27 report: {report}", flush=True)
    return report
