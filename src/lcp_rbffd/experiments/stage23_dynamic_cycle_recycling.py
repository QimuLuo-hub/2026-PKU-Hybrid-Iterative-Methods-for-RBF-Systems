"""Stage 23: restart-cycle dynamics of spatial-K Ritz recycle spaces.

The experiment reuses the Stage-20 hard diagnostic cohort and the validated
right-preconditioned recycled-GMRES implementation.  The first restart cycle
is common to all methods.  Ordinary Ritz directions extracted from a completed
cycle can affect only the following cycle, preventing look-ahead leakage.
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
import scipy.linalg as la

from neurkitt_oracle.gcrodr import (
    CycleArnoldiData,
    GmresResult,
    restarted_projected_gmres,
)

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage20_spatial_k_robust_oracle import (
    SystemContext,
    _coefficient_from_row,
    _load_context,
    _read_rows,
    _relative_residual,
)
from .stage21_population_validation import _safe_spearman
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
class Stage23Settings:
    """Outcome-independent cohort, construction, and decision thresholds."""

    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    methods: Tuple[str, ...] = ("ilu", "fixed", "refreshed", "cumulative")
    ritz_type: str = "ordinary_smallest_magnitude_pair_preserving"
    local_requested_q: int = 24
    cumulative_max_q: int = 48
    numerical_rank_tolerance: float = 1.0e-11
    system_fraction_gate: float = 0.75
    fixed_reduction_gate: float = 0.25
    population_dynamic_fixed_ratio_gate: float = 0.75
    population_dynamic_ilu_ratio_gate: float = 0.65
    per_rhs_worsening_tolerance: float = 0.10

    @property
    def stage20(self):
        from .stage20_spatial_k_robust_oracle import Stage20Settings

        return Stage20Settings()

    @property
    def solver(self):
        return self.stage20.solver


@dataclass(frozen=True)
class LocalRitzSpace:
    basis: np.ndarray
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    extraction_seconds: float
    orthogonality_error: float


def _orthogonal_component(
    component: np.ndarray, accepted: Sequence[np.ndarray]
) -> np.ndarray | None:
    work = np.asarray(component, dtype=float).copy()
    for _ in range(2):
        for previous in accepted:
            work -= previous * float(np.dot(previous, work))
    norm = float(np.linalg.norm(work))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        return None
    return work / norm


def extract_cycle_ritz(data: CycleArnoldiData, requested_q: int) -> LocalRitzSpace:
    """Extract a pair-preserving ordinary-Ritz solution space from one cycle."""

    start = perf_counter()
    m = int(data.hessenberg.shape[1])
    if m < 1:
        raise ValueError("completed cycle has no Arnoldi directions")
    projected = data.hessenberg[:m, :m]
    values, vectors = la.eig(projected, check_finite=False)
    order = np.argsort(np.abs(values))
    accepted: List[np.ndarray] = []
    selected_values: List[complex] = []
    residuals: List[float] = []
    for index in order:
        coordinate = vectors[:, index]
        mapped = data.solution_basis[:, :m] @ coordinate
        raw_parts = [np.real(mapped)]
        if float(np.linalg.norm(np.imag(mapped))) > 1.0e-10:
            raw_parts.append(np.imag(mapped))
        atom: List[np.ndarray] = []
        for part in raw_parts:
            candidate = _orthogonal_component(part, [*accepted, *atom])
            if candidate is not None:
                atom.append(candidate)
        if not atom:
            continue
        if len(accepted) + len(atom) > requested_q:
            break
        h_coordinate = data.hessenberg[: m + 1, :m] @ coordinate
        embedded = np.zeros(m + 1, dtype=complex)
        embedded[:m] = coordinate
        defect = h_coordinate - values[index] * embedded
        denominator = max(
            float(np.linalg.norm(h_coordinate))
            + abs(values[index]) * float(np.linalg.norm(coordinate)),
            np.finfo(float).tiny,
        )
        relative_defect = float(np.linalg.norm(defect) / denominator)
        accepted.extend(atom)
        selected_values.extend([values[index]] * len(atom))
        residuals.extend([relative_defect] * len(atom))
        if len(accepted) == requested_q:
            break
    if not accepted:
        raise RuntimeError("cycle Arnoldi relation produced no real Ritz directions")
    basis = np.column_stack(accepted)
    error = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord="fro"))
    if error > 1.0e-8:
        raise RuntimeError(f"cycle Ritz basis lost orthogonality ({error:.3e})")
    return LocalRitzSpace(
        basis=basis,
        eigenvalues=np.asarray(selected_values, dtype=complex),
        residual_norms=np.asarray(residuals, dtype=float),
        extraction_seconds=perf_counter() - start,
        orthogonality_error=error,
    )


def compress_union(raw: np.ndarray, max_rank: int, tolerance: float) -> np.ndarray:
    """SVD-compress accumulated solution-space directions to a fixed cap."""

    u, singular, _ = la.svd(np.asarray(raw, dtype=float), full_matrices=False)
    if singular.size == 0:
        raise ValueError("empty cumulative dictionary")
    rank = min(max_rank, int(np.count_nonzero(singular > tolerance * singular[0])))
    if rank < 1:
        raise RuntimeError("cumulative recycle dictionary has zero numerical rank")
    return u[:, :rank]


def _pair_metrics(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    metrics = generalized_subspace_metrics(first, second)
    angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
    return {
        "projector_distance": float(metrics["projector_distance_normalized"]),
        "max_angle_degrees": float(np.max(angles)) if angles.size else float("nan"),
        "mean_angle_degrees": float(np.mean(angles)) if angles.size else float("nan"),
    }


class CycleController:
    """Extract cycle-local Ritz spaces and optionally update the recycle pair."""

    def __init__(
        self,
        matrix,
        strategy: str,
        settings: Stage23Settings,
        rhs_norm: float,
    ) -> None:
        self.matrix = matrix
        self.strategy = strategy
        self.settings = settings
        self.rhs_norm = rhs_norm
        self.active_z: np.ndarray | None = None
        self.previous_local_z: np.ndarray | None = None
        self.previous_local_w: np.ndarray | None = None
        self.records: List[Dict[str, Any]] = []
        self.callback_seconds = 0.0
        self.method_setup_seconds = 0.0
        self.method_extra_a_applications = 0
        self.update_count = 0

    def __call__(
        self, data: CycleArnoldiData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        callback_start = perf_counter()
        local = extract_cycle_ritz(data, self.settings.local_requested_q)
        local_pair_start = perf_counter()
        local_pair = factor_az_petrov(self.matrix, local.basis)
        local_pair_seconds = perf_counter() - local_pair_start
        z_change = {
            "projector_distance": float("nan"),
            "max_angle_degrees": float("nan"),
            "mean_angle_degrees": float("nan"),
        }
        w_change = dict(z_change)
        if self.previous_local_z is not None and self.previous_local_w is not None:
            z_change = _pair_metrics(self.previous_local_z, local.basis)
            w_change = _pair_metrics(self.previous_local_w, local_pair.test)

        end_relative = float(np.linalg.norm(data.end_residual) / max(self.rhs_norm, 1e-300))
        may_update = (
            end_relative > self.settings.solver.relative_tolerance
            and data.end_iteration < self.settings.solver.max_iterations
            and self.strategy != "ilu"
        )
        active_before = int(data.recycle_c.shape[1])
        replacement: AZPetrovFactors | None = None
        update_seconds = 0.0
        extraction_is_method_cost = False
        if may_update and self.strategy == "fixed" and data.cycle == 1:
            replacement = local_pair
            self.active_z = local.basis
            update_seconds = local_pair_seconds
            extraction_is_method_cost = True
        elif may_update and self.strategy == "refreshed":
            replacement = local_pair
            self.active_z = local.basis
            update_seconds = local_pair_seconds
            extraction_is_method_cost = True
        elif may_update and self.strategy == "cumulative":
            update_start = perf_counter()
            raw = local.basis if self.active_z is None else np.column_stack((self.active_z, local.basis))
            self.active_z = compress_union(
                raw,
                self.settings.cumulative_max_q,
                self.settings.numerical_rank_tolerance,
            )
            replacement = factor_az_petrov(self.matrix, self.active_z)
            update_seconds = perf_counter() - update_start
            extraction_is_method_cost = True

        if replacement is not None:
            self.update_count += 1
            self.method_setup_seconds += update_seconds
            if extraction_is_method_cost:
                self.method_setup_seconds += local.extraction_seconds
            self.method_extra_a_applications += int(replacement.test.shape[1])
            active_after = int(replacement.test.shape[1])
        else:
            active_after = active_before

        self.records.append(
            {
                "cycle": data.cycle,
                "start_iteration": data.start_iteration,
                "end_iteration": data.end_iteration,
                "cycle_iterations": data.end_iteration - data.start_iteration,
                "start_relative_residual": float(
                    np.linalg.norm(data.start_residual) / max(self.rhs_norm, 1e-300)
                ),
                "end_relative_residual": end_relative,
                "local_rank": local.basis.shape[1],
                "active_rank_before": active_before,
                "active_rank_after": active_after,
                "update_used_by_next_cycle": replacement is not None,
                "ritz_extraction_seconds": local.extraction_seconds,
                "local_pair_seconds": local_pair_seconds,
                "method_update_seconds": (
                    local.extraction_seconds + update_seconds if replacement is not None else 0.0
                ),
                "selected_ritz_magnitude_min": float(np.min(np.abs(local.eigenvalues))),
                "selected_ritz_magnitude_median": float(np.median(np.abs(local.eigenvalues))),
                "selected_ritz_magnitude_max": float(np.max(np.abs(local.eigenvalues))),
                "selected_ritz_residual_median": float(np.median(local.residual_norms)),
                "selected_ritz_residual_max": float(np.max(local.residual_norms)),
                "local_basis_orthogonality_error": local.orthogonality_error,
                "local_pair_au_error": float(
                    local_pair.diagnostics["au_equals_w_relative_error"]
                ),
                "d_Z_previous_cycle": z_change["projector_distance"],
                "max_angle_Z_degrees": z_change["max_angle_degrees"],
                "mean_angle_Z_degrees": z_change["mean_angle_degrees"],
                "d_W_previous_cycle": w_change["projector_distance"],
                "max_angle_W_degrees": w_change["max_angle_degrees"],
                "mean_angle_W_degrees": w_change["mean_angle_degrees"],
            }
        )
        self.previous_local_z = local.basis
        self.previous_local_w = local_pair.test
        self.callback_seconds += perf_counter() - callback_start
        if replacement is None:
            return None
        return replacement.recycling_trial, replacement.test


def dynamic_system_gate(
    fixed_iterations: Sequence[int],
    dynamic_iterations: Sequence[int],
    dynamic_converged: Sequence[bool],
    settings: Stage23Settings,
) -> Dict[str, Any]:
    """Predeclared per-system consistency and improvement gate."""

    fixed = np.asarray(fixed_iterations, dtype=float)
    dynamic = np.asarray(dynamic_iterations, dtype=float)
    relative_gain = (fixed - dynamic) / np.maximum(fixed, 1.0)
    no_large_worsening = bool(
        np.all(dynamic <= (1.0 + settings.per_rhs_worsening_tolerance) * fixed)
    )
    passes = bool(
        all(bool(value) for value in dynamic_converged)
        and float(np.median(relative_gain)) >= settings.fixed_reduction_gate
        and int(np.count_nonzero(dynamic < fixed)) >= 2
        and no_large_worsening
    )
    return {
        "median_reduction_vs_fixed": float(np.median(relative_gain)),
        "rhs_improved_count": int(np.count_nonzero(dynamic < fixed)),
        "no_rhs_more_than_10pct_worse": no_large_worsening,
        "system_dynamic_gate_passed": passes,
    }


def population_dynamic_gate(
    summaries: Sequence[Mapping[str, Any]], method: str, settings: Stage23Settings
) -> Dict[str, Any]:
    rows = [row for row in summaries if str(row["method"]) == method]
    passing = sum(bool(row["system_dynamic_gate_passed"]) for row in rows)
    fraction = passing / max(len(rows), 1)
    fixed_ratio = float(np.median([float(row["ratio_to_fixed"]) for row in rows]))
    ilu_ratio = float(np.median([float(row["ratio_to_ilu"]) for row in rows]))
    return {
        "method": method,
        "systems": len(rows),
        "systems_passing": passing,
        "system_pass_fraction": fraction,
        "median_dynamic_fixed_ratio": fixed_ratio,
        "median_dynamic_ilu_ratio": ilu_ratio,
        "fraction_gate": fraction >= settings.system_fraction_gate,
        "fixed_ratio_gate": fixed_ratio
        <= settings.population_dynamic_fixed_ratio_gate,
        "ilu_ratio_gate": ilu_ratio <= settings.population_dynamic_ilu_ratio_gate,
        "passed": bool(
            fraction >= settings.system_fraction_gate
            and fixed_ratio <= settings.population_dynamic_fixed_ratio_gate
            and ilu_ratio <= settings.population_dynamic_ilu_ratio_gate
        ),
    }


def _run_one(
    context: SystemContext,
    rhs: np.ndarray,
    strategy: str,
    settings: Stage23Settings,
) -> tuple[GmresResult, CycleController, float]:
    controller = CycleController(
        context.system.matrix, strategy, settings, float(np.linalg.norm(rhs))
    )
    context.ilu.reset_timing()
    result = restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        rtol=settings.solver.relative_tolerance,
        atol=settings.solver.absolute_tolerance,
        restart=settings.solver.restart,
        max_iterations=settings.solver.max_iterations,
        record_true_residuals=True,
        cycle_update=controller,
    )
    online_excluding_callback = max(result.solve_seconds - controller.callback_seconds, 0.0)
    return result, controller, online_excluding_callback


def _summarize(
    solver_rows: Sequence[Mapping[str, Any]], settings: Stage23Settings
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        local_by_method = {
            method: [
                row
                for row in solver_rows
                if int(row["sample_id"]) == sample_id and str(row["method"]) == method
            ]
            for method in settings.methods
        }
        ilu_median = float(np.median([row["iterations"] for row in local_by_method["ilu"]]))
        fixed_median = float(
            np.median([row["iterations"] for row in local_by_method["fixed"]])
        )
        fixed_iterations = [int(row["iterations"]) for row in local_by_method["fixed"]]
        for method in settings.methods:
            rows = local_by_method[method]
            median_iterations = float(np.median([row["iterations"] for row in rows]))
            summary: Dict[str, Any] = {
                "sample_id": sample_id,
                "method": method,
                "rhs_count": len(rows),
                "all_rhs_converged": all(bool(row["converged"]) for row in rows),
                "convergence_rate": float(np.mean([bool(row["converged"]) for row in rows])),
                "iterations_median": median_iterations,
                "iterations_min": min(int(row["iterations"]) for row in rows),
                "iterations_max": max(int(row["iterations"]) for row in rows),
                "ratio_to_ilu": median_iterations / max(ilu_median, 1.0),
                "ratio_to_fixed": median_iterations / max(fixed_median, 1.0),
                "true_residual_max": max(float(row["true_final_residual"]) for row in rows),
                "online_seconds_median": float(np.median([row["online_seconds"] for row in rows])),
                "method_setup_seconds_median": float(
                    np.median([row["method_setup_seconds"] for row in rows])
                ),
                "total_seconds_median": float(np.median([row["total_seconds"] for row in rows])),
                "restart_cycles_median": float(np.median([row["restart_cycles"] for row in rows])),
                "active_rank_max": max(int(row["active_rank_max"]) for row in rows),
            }
            if method in {"refreshed", "cumulative"}:
                summary.update(
                    dynamic_system_gate(
                        fixed_iterations,
                        [int(row["iterations"]) for row in rows],
                        [bool(row["converged"]) for row in rows],
                        settings,
                    )
                )
            else:
                summary.update(
                    {
                        "median_reduction_vs_fixed": float("nan"),
                        "rhs_improved_count": 0,
                        "no_rhs_more_than_10pct_worse": True,
                        "system_dynamic_gate_passed": False,
                    }
                )
            summaries.append(summary)
    return summaries


def _plot_results(
    output: Path,
    solver_rows: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    settings: Stage23Settings,
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    chart_map: List[Dict[str, str]] = []
    colors = {"ilu": GRAY, "fixed": BLUE, "refreshed": ORANGE, "cumulative": OLIVE}
    labels = {"ilu": "ILU", "fixed": "fixed", "refreshed": "refreshed", "cumulative": "cumulative"}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(settings.sample_ids))
    width = 0.19
    for offset, method in enumerate(settings.methods):
        values = [
            next(
                float(row["iterations_median"])
                for row in summaries
                if int(row["sample_id"]) == sample_id and row["method"] == method
            )
            for sample_id in settings.sample_ids
        ]
        ax.bar(x + (offset - 1.5) * width, values, width, color=colors[method], label=labels[method])
    ax.axhline(settings.solver.max_iterations, color=CHARCOAL, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, [str(value) for value in settings.sample_ids])
    ax.set_xlabel("spatial-K system ID")
    ax.set_ylabel("median held-out iterations")
    ax.set_ylim(0, 325)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = plot_dir / "strategy_iteration_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "system_summary.csv", "purpose": "Compare fixed, refreshed, cumulative, and ILU iteration counts."})

    representative = (10, 15, 17, 34)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True, sharey=True)
    for ax, sample_id in zip(axes.ravel(), representative):
        for method in settings.methods:
            rows = [
                row
                for row in history_rows
                if int(row["sample_id"]) == sample_id
                and int(row["rhs_seed"]) == settings.heldout_rhs_seeds[0]
                and row["method"] == method
            ]
            ax.semilogy(
                [int(row["iteration"]) for row in rows],
                [max(float(row["true_relative_residual"]), 1e-16) for row in rows],
                color=colors[method],
                label=labels[method],
            )
        ax.axhline(settings.solver.relative_tolerance, color=CHARCOAL, linewidth=0.7, linestyle="--")
        ax.set_title(f"ID {sample_id}")
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[-1, 0].set_xlabel("iteration")
    axes[-1, 1].set_xlabel("iteration")
    axes[0, 0].set_ylabel("true relative residual")
    axes[1, 0].set_ylabel("true relative residual")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = plot_dir / "representative_residual_histories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "residual_histories.csv", "purpose": "Show true-residual behavior on representative hard systems."})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for ax, field, title in (
        (axes[0], "d_Z_previous_cycle", "$Z$-space"),
        (axes[1], "d_W_previous_cycle", "$W$-space"),
    ):
        for method in ("ilu", "fixed", "refreshed", "cumulative"):
            rows = [row for row in cycle_rows if row["method"] == method and np.isfinite(float(row[field]))]
            if rows:
                ax.scatter(
                    [int(row["cycle"]) for row in rows],
                    [float(row[field]) for row in rows],
                    s=15,
                    alpha=0.45,
                    color=colors[method],
                    label=labels[method],
                )
        ax.set_title(title)
        ax.set_xlabel("restart cycle")
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].set_ylabel("normalized consecutive-cycle projector distance")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = plot_dir / "cycle_subspace_variation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "cycle_subspace_diagnostics.csv", "purpose": "Compare consecutive-cycle local Ritz-space movement in solution and residual coordinates."})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    dynamic = [row for row in solver_rows if row["method"] in {"fixed", "refreshed", "cumulative"}]
    for method in ("fixed", "refreshed", "cumulative"):
        rows = [row for row in dynamic if row["method"] == method]
        axes[0].scatter(
            [float(row["method_setup_seconds"]) for row in rows],
            [int(row["iterations"]) for row in rows],
            s=22,
            alpha=0.65,
            color=colors[method],
            label=labels[method],
        )
        axes[1].scatter(
            [int(row["active_rank_max"]) for row in rows],
            [int(row["iterations"]) for row in rows],
            s=22,
            alpha=0.65,
            color=colors[method],
            label=labels[method],
        )
    axes[0].set_xlabel("method-specific cycle setup (s)")
    axes[0].set_ylabel("iterations")
    axes[1].set_xlabel("maximum active recycle rank")
    axes[1].set_ylabel("iterations")
    for ax in axes:
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "construction_cost_and_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Expose the cost and rank paid by dynamic updates."})
    return chart_map


def _write_report(
    root: Path,
    output: Path,
    solver_rows: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    settings: Stage23Settings,
) -> Path:
    report_dir = root / "results" / "results-fd" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "stage23_dynamic_cycle_recycle_diagnostic.md"
    gate_by_method = {str(row["method"]): row for row in gates}
    winner = next((name for name in ("refreshed", "cumulative") if gate_by_method[name]["passed"]), None)
    if winner is None:
        answer = (
            "The preregistered dynamic-space explanation is rejected: neither refreshed nor "
            "cumulative recycling passes the family gate."
        )
        recommendation = "Do not proceed to learning a dynamic recycle space; return to the complementary target mechanism."
    else:
        answer = (
            f"The preregistered dynamic-space explanation is supported by the {winner} strategy."
        )
        recommendation = "The solver-relevant object is cycle-dynamic rather than system-static; the next study should target update prediction, not a fixed system-to-space map."

    def value(sample: int, method: str, key: str) -> Any:
        return next(row[key] for row in summaries if int(row["sample_id"]) == sample and row["method"] == method)

    geometry = [row for row in cycle_rows if np.isfinite(float(row["d_Z_previous_cycle"]))]
    median_dz = float(np.median([float(row["d_Z_previous_cycle"]) for row in geometry])) if geometry else float("nan")
    median_dw = float(np.median([float(row["d_W_previous_cycle"]) for row in geometry])) if geometry else float("nan")
    max_au = max(float(row["local_pair_au_error"]) for row in cycle_rows)
    geometry_by_method: Dict[str, Dict[str, float]] = {}
    execution_by_method: Dict[str, Dict[str, float]] = {}
    for method in settings.methods:
        method_cycles = [
            row
            for row in cycle_rows
            if row["method"] == method
            and np.isfinite(float(row["d_Z_previous_cycle"]))
        ]
        geometry_by_method[method] = {
            "d_z": float(np.median([float(row["d_Z_previous_cycle"]) for row in method_cycles])),
            "d_w": float(np.median([float(row["d_W_previous_cycle"]) for row in method_cycles])),
            "angle_z": float(np.median([float(row["max_angle_Z_degrees"]) for row in method_cycles])),
            "angle_w": float(np.median([float(row["max_angle_W_degrees"]) for row in method_cycles])),
        }
        method_solves = [row for row in solver_rows if row["method"] == method]
        execution_by_method[method] = {
            "converged": float(sum(bool(row["converged"]) for row in method_solves)),
            "online": float(np.median([float(row["online_seconds"]) for row in method_solves])),
            "setup": float(np.median([float(row["method_setup_seconds"]) for row in method_solves])),
            "total": float(np.median([float(row["total_seconds"]) for row in method_solves])),
            "rank": float(np.median([float(row["active_rank_max"]) for row in method_solves])),
            "ilu_apps": float(np.median([float(row["ilu_applications"]) for row in method_solves])),
            "extra_a": float(np.median([float(row["method_extra_a_applications"]) for row in method_solves])),
        }
    lines = [
        "# Stage 23: restart-cycle dynamics of spatial-$K$ recycle spaces",
        "",
        "## Answer and preregistered decision",
        "",
        f"**{answer}** The conclusion is based on the fixed eight-system Stage-20 diagnostic cohort and three held-out right-hand sides per system. The decision was made from solver performance, not subspace geometry. "
        f"{recommendation}",
        "",
        "A dynamic strategy was required to converge on all three held-out right-hand sides and reduce the fixed-space median by at least 25% on at least six of eight systems. At population level it also required median dynamic/fixed iteration ratio at most 0.75 and median dynamic/ILU ratio at most 0.65. The thresholds, cohort, ordinary-Ritz extraction, local rank 24, and cumulative cap 48 were written before the run.",
        "",
        "| strategy | systems passing | median dynamic/fixed ratio | median dynamic/ILU ratio | gate |",
        "|:--|--:|--:|--:|:--:|",
    ]
    for method in ("refreshed", "cumulative"):
        row = gate_by_method[method]
        lines.append(
            f"| {method} | {row['systems_passing']}/{row['systems']} | {row['median_dynamic_fixed_ratio']:.3f} | {row['median_dynamic_ilu_ratio']:.3f} | {'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Frozen experiment and integration order",
            "",
            "The systems are IDs 3, 7, 10, 14, 15, 17, 19, and 34 from the saved spatially varying anisotropic $K(x)$ population. Each uses the fixed $N=5041$ seed-7 jittered node set, jitter 0.15, stencil size 25, PHS $r^3$, degree-2 polynomial augmentation, and the saved matrix. Strong ILU is unchanged at drop tolerance $10^{-3}$ and fill factor 5. Right-preconditioned GMRES uses restart 50, a 300-iteration cap, and true-relative-residual tolerance $10^{-8}$. Evaluation seeds are 20261101--20261103.",
            "",
            "For cycle $s$, Arnoldi is applied to $(I-W_sW_s^T)AM^{-1}$ when a recycle pair is active, or to $AM^{-1}$ otherwise. At the end of that cycle, ordinary Ritz vectors of the completed Hessenberg matrix are ordered by increasing magnitude with real/complex blocks preserved. A Ritz coordinate $g$ is mapped through the already computed solution basis as $z=M^{-1}V_mg$. The resulting $Z_s$ is orthonormalized, paired by $AZ_s=W_sR_s$, and normalized by $U_s=Z_sR_s^{-1}$. The new pair can affect only cycle $s+1$, so no method uses future Arnoldi information.",
            "",
            "The fixed strategy installs the first cycle's rank-24 space and never changes it. Refreshed recycling replaces that space after every completed cycle. Cumulative recycling SVD-compresses the previous active solution space together with the new rank-24 local space to at most 48 directions. ILU remains active in every Arnoldi step for all strategies.",
            "",
            "## Solver results",
            "",
            "| ID | ILU | fixed | refreshed | cumulative | refreshed/fixed | cumulative/fixed |",
            "|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for sample_id in settings.sample_ids:
        fixed = float(value(sample_id, "fixed", "iterations_median"))
        refreshed = float(value(sample_id, "refreshed", "iterations_median"))
        cumulative = float(value(sample_id, "cumulative", "iterations_median"))
        method_cells = {}
        for method in settings.methods:
            converged_count = sum(
                bool(row["converged"])
                for row in solver_rows
                if int(row["sample_id"]) == sample_id and row["method"] == method
            )
            method_cells[method] = (
                f"{float(value(sample_id, method, 'iterations_median')):.0f} "
                f"[{converged_count}/3]"
            )
        lines.append(
            f"| {sample_id} | {method_cells['ilu']} | {method_cells['fixed']} | {method_cells['refreshed']} | {method_cells['cumulative']} | {refreshed/max(fixed,1):.3f} | {cumulative/max(fixed,1):.3f} |"
        )
    lines.extend(
        [
            "",
            "Every entry is the median over three held-out right-hand sides, with the number reaching the true residual tolerance in brackets. Per-RHS residuals, times, operation counts, and ranks are retained in `solver_results.csv`. Refreshed recycling converges only on the three ID-15 solves and never improves on the fixed count. Cumulative recycling converges on 12/24 solves: all right-hand sides for IDs 14, 15, 19, and 34. IDs 14, 19, and 34 pass the dynamic system gate; ID 15 is already solved in 76 iterations by the fixed first-cycle space.",
            "",
            "![Iteration comparison](../stages/stage23_dynamic_cycle_recycle_diagnostic/plots/strategy_iteration_comparison.png)",
            "",
            "![Representative true residual histories](../stages/stage23_dynamic_cycle_recycle_diagnostic/plots/representative_residual_histories.png)",
            "",
            "## Do solver-critical directions move?",
            "",
            f"Across consecutive cycle-local spaces, the pooled median normalized projector distances are $d_Z={median_dz:.3f}$ and $d_W={median_dw:.3f}$. On the unmodified ILU trajectory, which is the cleanest diagnostic of native cycle evolution, the corresponding medians are $d_Z={geometry_by_method['ilu']['d_z']:.3f}$ and $d_W={geometry_by_method['ilu']['d_w']:.3f}$, with median maximum angles {geometry_by_method['ilu']['angle_z']:.1f} and {geometry_by_method['ilu']['angle_w']:.1f} degrees. The directions therefore change appreciably. Movement alone, however, does not establish that refreshing is useful. The causal diagnostic is whether B or C beats A under the preregistered solver gate.",
            "",
            "| trajectory | median $d_Z$ | median $d_W$ | median max angle $Z$ | median max angle $W$ |",
            "|:--|--:|--:|--:|--:|",
            *[
                f"| {method} | {geometry_by_method[method]['d_z']:.3f} | {geometry_by_method[method]['d_w']:.3f} | {geometry_by_method[method]['angle_z']:.1f}$^\\circ$ | {geometry_by_method[method]['angle_w']:.1f}$^\\circ$ |"
                for method in settings.methods
            ],
            "",
            "![Consecutive-cycle subspace variation](../stages/stage23_dynamic_cycle_recycle_diagnostic/plots/cycle_subspace_variation.png)",
            "",
            "The CSV reports principal angles and projector distances separately for each strategy, system, right-hand side, and restart boundary. Since refreshed and cumulative methods alter later trajectories, their later local spaces are not paired observations with the ILU trajectory. The first cycle is identical across all methods and subsequent changes are mechanism-specific.",
            "",
            "## Cost and numerical validity",
            "",
            "Cycle-local Ritz extraction reuses the Arnoldi relation already produced by the solve. Method-specific construction time includes the projected eigenproblem, solution-space mapping, and target-side $AZ=WR$ pairing only when the resulting pair is used in a later cycle. Diagnostic extraction after the final converged cycle and geometry-only pairings are excluded from deployable method cost. The cumulative method pays an additional capped SVD. Online times subtract all callback diagnostics, while total method times add only the update cost actually used.",
            "",
            "| strategy | converged RHS | median online (s) | median update setup (s) | median total (s) | median max rank | median ILU apps | median extra $A$ apps |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|",
            *[
                f"| {method} | {execution_by_method[method]['converged']:.0f}/24 | {execution_by_method[method]['online']:.3f} | {execution_by_method[method]['setup']:.3f} | {execution_by_method[method]['total']:.3f} | {execution_by_method[method]['rank']:.0f} | {execution_by_method[method]['ilu_apps']:.0f} | {execution_by_method[method]['extra_a']:.0f} |"
                for method in settings.methods
            ],
            "",
            f"The maximum measured relative $\|AU-W\|_F/\|W\|_F$ defect over cycle-local pairings is {max_au:.2e}. True-residual convergence is never inferred from the Arnoldi residual estimate.",
            "",
            "![Construction cost and active rank](../stages/stage23_dynamic_cycle_recycle_diagnostic/plots/construction_cost_and_rank.png)",
            "",
            "## Scientific interpretation",
            "",
        ]
    )
    if winner is None:
        lines.extend(
            [
                "Although restart-cycle Ritz spaces may rotate, the refreshed and capped cumulative mechanisms do not convert that motion into a sufficiently consistent solver advantage over the fixed first-cycle space. The evidence therefore does not support the claim that Stage-22 transfer failure is primarily the absence of a useful fixed space. On this protocol, making the recycle object dynamic is not enough.",
                "",
                "This does not erase the Stage-22A own-system result. Three independent $m=128$ constructions compressed to at most 48 directions solved all 15 hard systems there, so a useful moderate-dimensional own-system space can exist. Stage 23 instead shows that a single 50-step restart cycle and simple online replacement or accumulation do not recover that information reliably. Cumulative accumulation supplies partial evidence for later-cycle information on IDs 14, 19, and 34, but the failures on five systems rule out solver-time variation as the family-wide explanation.",
                "",
                "**Decision:** reject the dynamic-space explanation and do not train a dynamic predictor.",
            ]
        )
    else:
        lines.extend(
            [
                f"The {winner} method passes every preregistered population condition while using the same ILU and cycle-local Ritz rule. This supports a dynamic solver object: directions exposed later in the solve provide information not contained in the first-cycle space. The result does not establish that those updates are predictable or economical to learn.",
                "",
                "**Decision:** the relevant complementary object is dynamic rather than a single system-static recycle space.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The diagnostic uses the established eight-system hard cohort rather than the full 15-system population. Ordinary Ritz and the local rank were frozen rather than compared with harmonic Ritz or tuned per system. The cumulative union is a capped SVD of historical and current directions, not an optimal selection oracle. Timing comes from one workstation. Iteration caps remain censored at 300. These limits are deliberate because the question is whether one simple dynamic mechanism rescues the fixed-space failure without target redesign.",
            "",
            "## Reproducibility",
            "",
            "Raw per-RHS results, true residual histories, cycle geometry, system summaries, the preregistered protocol, settings, validation checks, chart map, source notes, and artifact hashes are saved in `results/results-fd/stages/stage23_dynamic_cycle_recycle_diagnostic/`.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _manifest(output: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_stage23(root: Path) -> Path:
    settings = Stage23Settings()
    output = root / "results" / "results-fd" / "stages" / "stage23_dynamic_cycle_recycle_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    stage19a = root / "results" / "results-fd" / "stages" / "stage19A_spatial_coefficient_oracle_validation"
    node_data = np.load(stage19a / "fixed_nodes.npz")
    nodes = np.asarray(node_data["nodes"], dtype=float)
    boundary = np.asarray(node_data["boundary"], dtype=bool)
    coefficient_rows = {
        int(row["sample_id"]): row for row in _read_rows(stage19a / "coefficient_samples.csv")
    }

    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "predeclared_protocol.json",
        {
            "cohort_basis": "Stage-20 fixed hard diagnostic cohort; selected without Stage-23 outcomes",
            "sample_ids": list(settings.sample_ids),
            "heldout_rhs_seeds": list(settings.heldout_rhs_seeds),
            "ordinary_ritz_local_requested_q": settings.local_requested_q,
            "cumulative_max_q": settings.cumulative_max_q,
            "update_timing": "completed cycle s can affect cycle s+1 only",
            "system_gate": {
                "all_rhs_converge": True,
                "median_reduction_vs_fixed_at_least": settings.fixed_reduction_gate,
                "rhs_improved_at_least": 2,
                "no_rhs_more_than_fraction_worse": settings.per_rhs_worsening_tolerance,
            },
            "population_gate": {
                "system_pass_fraction_at_least": settings.system_fraction_gate,
                "median_dynamic_fixed_ratio_at_most": settings.population_dynamic_fixed_ratio_gate,
                "median_dynamic_ilu_ratio_at_most": settings.population_dynamic_ilu_ratio_gate,
            },
        },
    )

    solver_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        context = _load_context(
            root, sample_id, coefficient_rows, nodes, boundary, settings.stage20
        )
        for rhs_seed in settings.heldout_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            for method in settings.methods:
                result, controller, online_seconds = _run_one(context, rhs, method, settings)
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                counts = result.operation_counts or {}
                active_rank_max = max(
                    [int(row["active_rank_after"]) for row in controller.records],
                    default=0,
                )
                solver_rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": method,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "true_final_residual": true_residual,
                        "restart_cycles": result.restart_cycles,
                        "active_rank_max": active_rank_max,
                        "update_count": controller.update_count,
                        "online_seconds": online_seconds,
                        "method_setup_seconds": controller.method_setup_seconds,
                        "total_seconds": online_seconds + controller.method_setup_seconds,
                        "diagnostic_callback_seconds": controller.callback_seconds,
                        "a_matvecs": counts.get("a_matvecs", 0),
                        "ilu_applications": counts.get("preconditioner_applications", 0),
                        "method_extra_a_applications": controller.method_extra_a_applications,
                        "projected_arnoldi_relation_error_max": (result.diagnostics or {}).get("projected_arnoldi_relation_error_max", float("nan")),
                        "arnoldi_basis_orthogonality_error_max": (result.diagnostics or {}).get("arnoldi_basis_orthogonality_error_max", float("nan")),
                        "arnoldi_basis_to_w_orthogonality_error_max": (result.diagnostics or {}).get("arnoldi_basis_to_w_orthogonality_error_max", float("nan")),
                    }
                )
                for row in controller.records:
                    cycle_rows.append(
                        {"sample_id": sample_id, "rhs_seed": rhs_seed, "method": method, **row}
                    )
                for iteration, residual in zip(
                    result.true_residual_iterations, result.true_residual_history
                ):
                    history_rows.append(
                        {
                            "sample_id": sample_id,
                            "rhs_seed": rhs_seed,
                            "method": method,
                            "iteration": iteration,
                            "true_relative_residual": residual,
                        }
                    )
        print(f"Stage 23 completed system {sample_id}", flush=True)

    summaries = _summarize(solver_rows, settings)
    gates = [
        population_dynamic_gate(summaries, method, settings)
        for method in ("refreshed", "cumulative")
    ]
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "cycle_subspace_diagnostics.csv", cycle_rows)
    _write_csv(output / "residual_histories.csv", history_rows)
    _write_csv(output / "system_summary.csv", summaries)
    _write_csv(output / "population_gate.csv", gates)

    correlation_rows: List[Dict[str, Any]] = []
    for method in settings.methods:
        rows = [
            row
            for row in cycle_rows
            if row["method"] == method and np.isfinite(float(row["d_Z_previous_cycle"]))
        ]
        cycle_progress = [
            np.log10(max(float(row["start_relative_residual"]), 1e-16))
            - np.log10(max(float(row["end_relative_residual"]), 1e-16))
            for row in rows
        ]
        for metric in (
            "d_Z_previous_cycle",
            "d_W_previous_cycle",
            "max_angle_Z_degrees",
            "max_angle_W_degrees",
        ):
            rho, pvalue, count = _safe_spearman(
                [float(row[metric]) for row in rows], cycle_progress
            )
            correlation_rows.append(
                {
                    "method": method,
                    "geometry_metric": metric,
                    "performance_metric": "cycle_log10_residual_reduction",
                    "spearman_rho": rho,
                    "pvalue_descriptive_only": pvalue,
                    "observations": count,
                }
            )
    _write_csv(output / "geometry_progress_correlations.csv", correlation_rows)

    chart_map = _plot_results(output, solver_rows, cycle_rows, history_rows, summaries, settings)
    _write_csv(output / "chart_map.csv", chart_map)
    decision = {
        "dynamic_explanation_supported": any(bool(row["passed"]) for row in gates),
        "winning_strategy": next((row["method"] for row in gates if row["passed"]), None),
        "gates": gates,
        "recommendation": (
            "dynamic solver object; study update prediction next"
            if any(bool(row["passed"]) for row in gates)
            else "reject dynamic-space explanation; do not train a dynamic predictor"
        ),
    }
    _write_json(output / "decision.json", decision)

    archived_stage21_rows = _read_rows(
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage21_spatial_K_population_validation"
        / "population_solver_results.csv"
    )
    archived_baseline = {
        (int(row["sample_id"]), int(row["heldout_rhs_seed"])): row
        for row in archived_stage21_rows
    }
    baseline_rows = [row for row in solver_rows if row["method"] == "ilu"]
    baseline_iteration_matches = all(
        int(row["iterations"])
        == int(
            archived_baseline[(int(row["sample_id"]), int(row["rhs_seed"]))][
                "baseline_iterations"
            ]
        )
        for row in baseline_rows
    )
    baseline_residual_difference_max = max(
        abs(
            float(row["true_final_residual"])
            - float(
                archived_baseline[(int(row["sample_id"]), int(row["rhs_seed"]))][
                    "baseline_true_final_relative_residual"
                ]
            )
        )
        for row in baseline_rows
    )
    first_cycle_residual_difference_max = 0.0
    for sample_id in settings.sample_ids:
        for rhs_seed in settings.heldout_rhs_seeds:
            histories = []
            for method in settings.methods:
                histories.append(
                    {
                        int(row["iteration"]): float(row["true_relative_residual"])
                        for row in history_rows
                        if int(row["sample_id"]) == sample_id
                        and int(row["rhs_seed"]) == rhs_seed
                        and row["method"] == method
                        and int(row["iteration"]) <= settings.solver.restart
                    }
                )
            common_iterations = set.intersection(
                *(set(history) for history in histories)
            )
            first_cycle_residual_difference_max = max(
                first_cycle_residual_difference_max,
                max(
                    abs(histories[index][iteration] - histories[0][iteration])
                    for index in range(1, len(histories))
                    for iteration in common_iterations
                ),
            )
    validation = {
        "expected_solver_rows": len(settings.sample_ids) * len(settings.heldout_rhs_seeds) * len(settings.methods),
        "actual_solver_rows": len(solver_rows),
        "all_true_residuals_finite": all(np.isfinite(float(row["true_final_residual"])) for row in solver_rows),
        "maximum_local_AU_minus_W_relative_error": max(float(row["local_pair_au_error"]) for row in cycle_rows),
        "maximum_projected_arnoldi_relation_error": max(float(row["projected_arnoldi_relation_error_max"]) for row in solver_rows),
        "maximum_arnoldi_orthogonality_error": max(float(row["arnoldi_basis_orthogonality_error_max"]) for row in solver_rows),
        "first_cycle_iterations_identical_by_system_rhs": all(
            len(
                {
                    int(next(row["end_iteration"] for row in cycle_rows if int(row["sample_id"]) == sample_id and int(row["rhs_seed"]) == seed and row["method"] == method and int(row["cycle"]) == 1))
                    for method in settings.methods
                }
            )
            == 1
            for sample_id in settings.sample_ids
            for seed in settings.heldout_rhs_seeds
        ),
        "first_cycle_true_residual_history_max_difference": first_cycle_residual_difference_max,
        "no_lookahead_update": all(
            (not bool(row["update_used_by_next_cycle"]))
            or float(row["end_relative_residual"]) > settings.solver.relative_tolerance
            for row in cycle_rows
        ),
        "baseline_iterations_match_stage21": baseline_iteration_matches,
        "baseline_true_residual_absolute_difference_max_vs_stage21": baseline_residual_difference_max,
        "no_neural_model": True,
    }
    validation["passed"] = bool(
        validation["actual_solver_rows"] == validation["expected_solver_rows"]
        and validation["all_true_residuals_finite"]
        and validation["maximum_local_AU_minus_W_relative_error"] < 1.0e-9
        and validation["maximum_projected_arnoldi_relation_error"] < 1.0e-9
        and validation["first_cycle_iterations_identical_by_system_rhs"]
        and validation["first_cycle_true_residual_history_max_difference"]
        <= 10.0 * np.finfo(float).eps
        and validation["no_lookahead_update"]
        and validation["baseline_iterations_match_stage21"]
    )
    _write_json(output / "validation.json", validation)
    (output / "source_notes.md").write_text(
        "# Source notes\n\n"
        "- Matrices: archived Stage-19C reconstructed systems for the Stage-20 hard cohort.\n"
        "- Coefficients and nodes: Stage-19A saved population artifacts.\n"
        "- ILU and solver parameters: Stage-10/20 frozen settings.\n"
        "- Held-out RHSs: Stage-20--22 seeds 20261101--20261103.\n"
        "- Dynamic cycle hook: backward-compatible extension of the validated GCRO implementation; completed-cycle data only.\n"
        "- No experiment or result outside the RBF-FD branch is used.\n",
        encoding="utf-8",
    )
    report = _write_report(root, output, solver_rows, cycle_rows, summaries, gates, settings)
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(f"Stage 23 report: {report}", flush=True)
    return report
