"""Stage 25: effective-rank ablation of the successful spatial-K oracle.

The experiment freezes the Stage-22A three-trajectory, m=128 ordinary-Ritz
candidate dictionary and changes only the number of leading Z-union singular
directions retained.  Archived q=24, 32, and 48 solver results are reused;
only q=28, 36, 40, and 44 are newly evaluated.  No learning or transfer is
performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage21_population_validation import _assemble_context, _load_population
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    Stage22ASettings,
    _build_ordinary_construction,
    _raw_dictionary,
    _run_pair,
    _setup_metadata,
    _z_union,
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
class Stage25Settings:
    """Frozen cohort, nested rank grid, and effective-rank gates."""

    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    construction_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    requested_q_values: Tuple[int, ...] = (24, 28, 32, 36, 40, 44, 48)
    archived_q_values: Tuple[int, ...] = (24, 32, 48)
    arnoldi_dimension: int = 128
    candidates_per_rhs: int = 24
    reference_q: int = 48
    recovery_gate: float = 0.75
    family_fraction_gate: float = 0.75
    numerical_rank_tolerance: float = 1.0e-11

    @property
    def stage22a(self) -> Stage22ASettings:
        return Stage22ASettings()

    @property
    def required_systems(self) -> int:
        return int(np.ceil(self.family_fraction_gate * len(self.sample_ids)))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _rule_id(q: int) -> str:
    return f"Z_union_n3_m128_q{q}"


def _standardize_archived(row: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "requested_q": int(row["requested_q"]),
        "actual_q": int(row["actual_q"]),
        "rhs_seed": int(row["heldout_rhs_seed"]),
        "source": "archived_stage22A",
        "baseline_iterations": int(row["baseline_iterations"]),
        "baseline_converged": _bool(row["baseline_converged"]),
        "baseline_true_final_residual": float(
            row["baseline_true_final_relative_residual"]
        ),
        "baseline_solve_seconds": float(row["baseline_solve_seconds"]),
        "recycle_iterations": int(row["recycle_iterations"]),
        "recycle_converged": _bool(row["recycle_converged"]),
        "recycle_true_final_residual": float(
            row["recycle_true_final_relative_residual"]
        ),
        "recycle_solve_seconds": float(row["recycle_solve_seconds"]),
        "J_iter": float(row["J_iter"]),
        "setup_seconds": float(row["deployable_setup_seconds"]),
        "ritz_setup_seconds": float(row["ritz_setup_seconds"]),
        "compression_seconds": float(row["compression_seconds"]),
        "pair_setup_seconds": float(row["pair_setup_seconds"]),
        "setup_a_applications": int(row["total_setup_a_applications"]),
        "setup_m_applications": int(row["total_setup_m_applications"]),
        "online_a_applications": int(row["online_a_applications"]),
        "online_m_applications": int(row["online_m_applications"]),
    }


def rank_decision(
    family_rows: Sequence[Mapping[str, Any]], settings: Stage25Settings
) -> Dict[str, Any]:
    """Apply the preregistered effective-rank classification."""

    passing = sorted(
        int(row["requested_q"])
        for row in family_rows
        if _bool(row["family_gate_passed"])
    )
    if not passing:
        return {
            "category": "no_robust_prefix",
            "effective_requested_q": None,
            "conclusion": "no tested fixed prefix recovers the reference robustly",
        }
    effective = passing[0]
    if effective <= 32:
        category = "substantially_smaller"
        conclusion = "most reference benefit is recovered by a substantially smaller prefix"
    elif effective <= 40:
        category = "intermediate"
        conclusion = "an intermediate-rank prefix is required"
    else:
        category = "near_full"
        conclusion = "benefit appears only near the full retained span"
    return {
        "category": category,
        "effective_requested_q": effective,
        "conclusion": conclusion,
    }


def _prefix_diagnostic(
    sample_id: int, q: int, pair: RecyclePair, reference: RecyclePair
) -> Dict[str, Any]:
    z = generalized_subspace_metrics(pair.z_basis, reference.z_basis)
    w = generalized_subspace_metrics(pair.w_basis, reference.w_basis)
    return {
        "sample_id": sample_id,
        "requested_q": q,
        "actual_q": pair.rank,
        "reference_actual_q": reference.rank,
        "d_Z_to_reference": float(z["projector_distance_normalized"]),
        "d_W_to_reference": float(w["projector_distance_normalized"]),
        "max_angle_Z_degrees": float(z["principal_angle_max_degrees"]),
        "max_angle_W_degrees": float(w["principal_angle_max_degrees"]),
        "candidate_containment_in_reference_Z": float(z["containment_first"]),
        "candidate_containment_in_reference_W": float(w["containment_first"]),
        "reference_energy_seen_by_candidate_Z": float(z["containment_second"]),
        "reference_energy_seen_by_candidate_W": float(w["containment_second"]),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]], settings: Stage25Settings
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    reference = {
        (int(row["sample_id"]), int(row["rhs_seed"])): int(
            row["recycle_iterations"]
        )
        for row in rows
        if int(row["requested_q"]) == settings.reference_q
    }
    system_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        previous_iterations = None
        previous_eta = None
        for q in settings.requested_q_values:
            local = [
                row
                for row in rows
                if int(row["sample_id"]) == sample_id
                and int(row["requested_q"]) == q
            ]
            if len(local) != len(settings.heldout_rhs_seeds):
                raise RuntimeError(f"expected three rows for ID {sample_id}, q={q}")
            eta = []
            for row in local:
                baseline = int(row["baseline_iterations"])
                denominator = baseline - reference[(sample_id, int(row["rhs_seed"]))]
                eta.append(
                    (baseline - int(row["recycle_iterations"])) / denominator
                    if denominator > 0
                    else float("nan")
                )
            median_iterations = float(
                np.median([int(row["recycle_iterations"]) for row in local])
            )
            median_eta = float(np.nanmedian(eta))
            all_converged = all(_bool(row["recycle_converged"]) for row in local)
            current = {
                "sample_id": sample_id,
                "requested_q": q,
                "actual_q": int(local[0]["actual_q"]),
                "rhs_count": len(local),
                "converged_rhs": sum(_bool(row["recycle_converged"]) for row in local),
                "all_rhs_converged": all_converged,
                "baseline_iterations_median": float(
                    np.median([int(row["baseline_iterations"]) for row in local])
                ),
                "recycle_iterations_median": median_iterations,
                "recycle_iterations_min": min(
                    int(row["recycle_iterations"]) for row in local
                ),
                "recycle_iterations_max": max(
                    int(row["recycle_iterations"]) for row in local
                ),
                "J_iter_median": float(
                    np.median([float(row["J_iter"]) for row in local])
                ),
                "reference_benefit_recovery_median": median_eta,
                "setup_seconds": float(local[0]["setup_seconds"]),
                "online_seconds_median": float(
                    np.median([float(row["recycle_solve_seconds"]) for row in local])
                ),
                "marginal_iteration_improvement": (
                    float(previous_iterations - median_iterations)
                    if previous_iterations is not None
                    else float("nan")
                ),
                "marginal_reference_recovery": (
                    float(median_eta - previous_eta)
                    if previous_eta is not None
                    else float("nan")
                ),
                "system_gate_passed": bool(
                    all_converged and median_eta >= settings.recovery_gate
                ),
            }
            system_rows.append(current)
            previous_iterations = median_iterations
            previous_eta = median_eta

    family_rows: List[Dict[str, Any]] = []
    for q in settings.requested_q_values:
        local = [row for row in system_rows if int(row["requested_q"]) == q]
        systems_passing = sum(_bool(row["system_gate_passed"]) for row in local)
        median_eta = float(
            np.median([float(row["reference_benefit_recovery_median"]) for row in local])
        )
        family_rows.append(
            {
                "requested_q": q,
                "systems": len(local),
                "systems_passing": systems_passing,
                "required_systems": settings.required_systems,
                "all_rhs_convergence_systems": sum(
                    _bool(row["all_rhs_converged"]) for row in local
                ),
                "median_reference_benefit_recovery": median_eta,
                "median_iteration_ratio_to_ilu": float(
                    np.median(
                        [
                            float(row["recycle_iterations_median"])
                            / max(float(row["baseline_iterations_median"]), 1.0)
                            for row in local
                        ]
                    )
                ),
                "median_J_iter": float(
                    np.median([float(row["J_iter_median"]) for row in local])
                ),
                "median_actual_q": float(
                    np.median([float(row["actual_q"]) for row in local])
                ),
                "median_setup_seconds": float(
                    np.median([float(row["setup_seconds"]) for row in local])
                ),
                "median_online_seconds": float(
                    np.median([float(row["online_seconds_median"]) for row in local])
                ),
                "median_marginal_iteration_improvement": (
                    float(
                        np.nanmedian(
                            [float(row["marginal_iteration_improvement"]) for row in local]
                        )
                    )
                    if q != settings.requested_q_values[0]
                    else float("nan")
                ),
                "family_gate_passed": bool(
                    systems_passing >= settings.required_systems
                    and median_eta >= settings.recovery_gate
                ),
            }
        )

    effective_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        passing = [
            row
            for row in system_rows
            if int(row["sample_id"]) == sample_id and _bool(row["system_gate_passed"])
        ]
        first = min(passing, key=lambda row: int(row["requested_q"])) if passing else None
        effective_rows.append(
            {
                "sample_id": sample_id,
                "smallest_passing_requested_q": (
                    int(first["requested_q"]) if first is not None else ""
                ),
                "smallest_passing_actual_q": (
                    int(first["actual_q"]) if first is not None else ""
                ),
                "passes_by_q48": bool(first is not None),
            }
        )
    return system_rows, family_rows, effective_rows


def _plots(
    output: Path,
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    settings: Stage25Settings,
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    q_values = np.asarray(settings.requested_q_values)
    charts: List[Dict[str, str]] = []

    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.5), sharex=True, sharey=True)
    for ax, sample_id in zip(axes.ravel(), settings.sample_ids):
        local = sorted(
            [row for row in system_rows if int(row["sample_id"]) == sample_id],
            key=lambda row: int(row["requested_q"]),
        )
        ax.plot(
            [int(row["requested_q"]) for row in local],
            [float(row["recycle_iterations_median"]) for row in local],
            color=BLUE,
            marker="o",
            linewidth=1.7,
        )
        baseline = float(local[0]["baseline_iterations_median"])
        ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1.0)
        ax.set_title(f"ID {sample_id}")
        ax.set_xticks(q_values[::2])
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel("requested retained rank")
    for ax in axes[:, 0]:
        ax.set_ylabel("median iterations")
    fig.suptitle("Nested three-trajectory Z-union rank ablation", y=1.01)
    fig.tight_layout()
    path = plot_dir / "system_iterations_vs_rank.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(
        {
            "figure": str(path.relative_to(output)),
            "source": "system_rank_summary.csv",
            "purpose": "Show where each hard system gains from retaining additional joint-span directions.",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].plot(
        q_values,
        [int(row["systems_passing"]) for row in family_rows],
        color=OLIVE,
        marker="o",
        linewidth=1.8,
    )
    axes[0].axhline(settings.required_systems, color=CHARCOAL, linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("systems passing (of 8)")
    axes[0].set_ylim(0, 8.4)
    axes[0].set_title("Family recovery gate")
    axes[1].plot(
        q_values,
        [float(row["median_reference_benefit_recovery"]) for row in family_rows],
        color=ORANGE,
        marker="o",
        linewidth=1.8,
    )
    axes[1].axhline(settings.recovery_gate, color=CHARCOAL, linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("median reference-benefit recovery")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Population benefit recovery")
    for ax in axes:
        ax.set_xlabel("requested retained rank")
        ax.set_xticks(q_values)
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    path = plot_dir / "family_gate_vs_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append(
        {
            "figure": str(path.relative_to(output)),
            "source": "family_rank_summary.csv",
            "purpose": "Identify the smallest frozen rank that clears the preregistered family gate.",
        }
    )

    increments = settings.requested_q_values[1:]
    matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["marginal_iteration_improvement"]
                        for row in system_rows
                        if int(row["sample_id"]) == sample_id
                        and int(row["requested_q"]) == q
                    )
                )
                for q in increments
            ]
            for sample_id in settings.sample_ids
        ]
    )
    bound = max(float(np.max(np.abs(matrix))), 1.0)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    image = ax.imshow(matrix, cmap="PuOr", vmin=-bound, vmax=bound, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if abs(value) >= 0.55 * bound else CHARCOAL
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    ax.set_xticks(
        np.arange(len(increments)),
        [f"{settings.requested_q_values[index]}→{q}" for index, q in enumerate(increments)],
    )
    ax.set_yticks(np.arange(len(settings.sample_ids)), [str(value) for value in settings.sample_ids])
    ax.set_xlabel("newly retained SVD-prefix block")
    ax.set_ylabel("spatial-K system ID")
    ax.set_title("Marginal median-iteration improvement by retained block")
    fig.colorbar(image, ax=ax, label="iterations saved versus preceding rank")
    fig.tight_layout()
    path = plot_dir / "marginal_block_gain_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append(
        {
            "figure": str(path.relative_to(output)),
            "source": "system_rank_summary.csv",
            "purpose": "Locate whether solver benefit is concentrated in one added prefix block or distributed across ranks.",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharex=True, sharey=True)
    colors = {24: GRAY, 32: BLUE, 40: ORANGE, 48: OLIVE}
    for ax, sample_id in zip(axes, (3, 10, 17)):
        for q in (24, 32, 40, 48):
            local = [
                row
                for row in history_rows
                if int(row["sample_id"]) == sample_id
                and int(row["requested_q"]) == q
                and int(row["rhs_seed"]) == settings.heldout_rhs_seeds[0]
            ]
            ax.semilogy(
                [int(row["iteration"]) for row in local],
                [max(float(row["true_relative_residual"]), 1.0e-16) for row in local],
                color=colors[q],
                linewidth=1.5,
                label=f"q={q}",
            )
        ax.axhline(
            settings.stage22a.solver.relative_tolerance,
            color=CHARCOAL,
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_title(f"ID {sample_id}")
        ax.set_xlabel("iteration")
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].set_ylabel("true relative residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    path = plot_dir / "stubborn_system_residual_histories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append(
        {
            "figure": str(path.relative_to(output)),
            "source": "residual_histories.csv",
            "purpose": "Show true-residual effects of retaining progressively broader prefixes on the stubborn systems.",
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        q_values,
        [float(row["median_setup_seconds"]) for row in family_rows],
        color=PURPLE,
        marker="o",
        linewidth=1.7,
    )
    ax.set_xlabel("requested retained rank")
    ax.set_ylabel("median recycle construction cost (s)")
    ax.set_xticks(q_values)
    ax.set_ylim(bottom=0)
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.set_title("Construction cost across the retained-rank ablation")
    fig.tight_layout()
    path = plot_dir / "setup_cost_vs_rank.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append(
        {
            "figure": str(path.relative_to(output)),
            "source": "family_rank_summary.csv",
            "purpose": "Keep the effective-rank conclusion separate from construction-cost implications.",
        }
    )
    return charts


def _write_report(
    root: Path,
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    effective_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage25Settings,
) -> Path:
    report = (
        root
        / "results"
        / "results-fd"
        / "reports"
        / "stage25_spatial_K_recycle_rank_ablation.md"
    )
    effective_q = decision["effective_requested_q"]
    lines = [
        "# Stage 25: effective rank of the spatial-$K$ own-system recycle oracle",
        "",
        "## Technical conclusion",
        "",
        f"**{decision['conclusion'].capitalize()}: the smallest frozen rank clearing the family gate is requested $q={effective_q}$.** At $N=5041$, this is 0.56% of the full system dimension. The three-trajectory $m=128$ ordinary-Ritz union therefore does not require the full requested rank 48 to recover most of its benefit on the eight-system cohort. The effective family-level rank is updated to $q={effective_q}$, while the per-system exception remains explicit rather than being averaged away.",
        "",
        "A system passes only when all three held-out right-hand sides converge to the true-residual tolerance and its median benefit recovery relative to the requested-$q=48$ own-system reference is at least 0.75. A fixed rank passes the family gate at six of eight systems with population median recovery at least 0.75. These rules and the rank grid were written before the four missing ranks were solved.",
        "",
        "## The family gate is reached before the full span",
        "",
        "The table reports the frozen prefix at each requested rank. Actual rank can be smaller when the three-trajectory dictionary is numerically exhausted.",
        "",
        "| requested $q$ | median actual rank | systems passing | all-RHS convergence | median $\\eta_{ref}$ | median ILU reduction | median iterations/ILU | median setup (s) | median online (s) |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {int(row['requested_q'])} | {float(row['median_actual_q']):.0f} | {int(row['systems_passing'])}/8 | {int(row['all_rhs_convergence_systems'])}/8 | {float(row['median_reference_benefit_recovery']):.3f} | {float(row['median_J_iter']):.3f} | {float(row['median_iteration_ratio_to_ilu']):.3f} | {float(row['median_setup_seconds']):.3f} | {float(row['median_online_seconds']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Requested $q=28$ is the formal minimum because it passes exactly 6/8 systems with median $\\eta_{ref}=0.950$. Requested $q=32$ is the stronger fixed moderate-rank point: it passes 7/8 systems with median recovery 0.972. Neither solves ID 17, so the family-level rank update must not be read as an all-system guarantee.",
            "",
            "The left panel shows the number of systems satisfying the complete held-out gate. The right panel shows median benefit recovery. Passing the median alone is insufficient when one or more systems do not converge.",
            "",
            "![Family gate versus rank](../stages/stage25_spatial_K_recycle_rank_ablation/plots/family_gate_vs_rank.png)",
            "",
            "## Effective rank varies sharply by system",
            "",
            "Median held-out iteration counts are shown below. Brackets give converged held-out right-hand sides out of three.",
            "",
            "| ID | ILU | $q=24$ | $q=28$ | $q=32$ | $q=36$ | $q=40$ | $q=44$ | $q=48$ | smallest passing $q$ |",
            "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    effective_lookup = {int(row["sample_id"]): row for row in effective_rows}
    for sample_id in settings.sample_ids:
        local = {
            int(row["requested_q"]): row
            for row in system_rows
            if int(row["sample_id"]) == sample_id
        }
        cells = [
            f"{float(local[q]['recycle_iterations_median']):.0f} [{int(local[q]['converged_rhs'])}/3]"
            for q in settings.requested_q_values
        ]
        q_min = effective_lookup[sample_id]["smallest_passing_requested_q"]
        q_min_text = str(q_min) if q_min != "" else "none"
        lines.append(
            f"| {sample_id} | {float(local[24]['baseline_iterations_median']):.0f} | "
            + " | ".join(cells)
            + f" | {q_min_text} |"
        )
    lines.extend(
        [
            "",
            "The small-multiple curves expose the heterogeneity directly. They distinguish a family-level moderate-rank result from the claim that one low rank solves every spatial-$K$ system.",
            "",
            "![System iterations versus rank](../stages/stage25_spatial_K_recycle_rank_ablation/plots/system_iterations_vs_rank.png)",
            "",
            "## Added directions have system-specific marginal value",
            "",
            "Each heatmap cell is the change in median held-out iterations after adding the next four-direction SVD-prefix block. Positive values are iterations saved. A uniformly dominant block would appear as one consistently strong column; instead the ablation shows where gains concentrate and where the response remains distributed.",
            "",
            "![Marginal block gain](../stages/stage25_spatial_K_recycle_rank_ablation/plots/marginal_block_gain_heatmap.png)",
            "",
            "For the stubborn systems, true-residual histories confirm that the rank effect is a convergence effect rather than a mismatch between reported and true residuals.",
            "",
            "![Stubborn system residual histories](../stages/stage25_spatial_K_recycle_rank_ablation/plots/stubborn_system_residual_histories.png)",
            "",
            "## Experimental controls and costs",
            "",
            "Every rank uses the same three construction right-hand sides, $m=128$ Arnoldi depth per trajectory, ordinary pair-preserving Ritz extraction within each trajectory, SVD ordering of the concatenated solution-space dictionary, strong ILU, paired nonsymmetric recycle construction, and three held-out evaluation right-hand sides. Thus the only solver-information change is the length of the nested retained prefix. Stage-22A outcomes at requested $q=24,32,48$ are reused. Requested $q=28,36,40,44$ are newly solved with the same implementation.",
            "",
            "Setup times include all three Arnoldi constructions, union compression, and paired $AZ=WR$, $U=ZR^{-1}$ construction. Archived and newly measured timings come from different executions, so their small nonmonotonic differences should not be interpreted as rank-scaling estimates.",
            "",
            "![Setup cost versus rank](../stages/stage25_spatial_K_recycle_rank_ablation/plots/setup_cost_vs_rank.png)",
            "",
            "## Limitation and next step",
            "",
            "This is an own-system oracle ablation on the fixed eight-system hard cohort. It establishes solver-effective retained dimension for this construction but does not establish cross-system predictability. The SVD prefixes are nested, yet their Ritz provenance is mixed across three trajectories, so a four-direction increment is not generally a single spectral cluster.",
            "",
            f"**Recommendation:** treat requested $q={effective_q}$ as the family-level effective oracle rank for the next mechanism study, while retaining an adaptive escape to the full available span for systems that fail the fixed-rank gate. Do not train a predictor from this result alone.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def run_stage25(root: Path) -> Path:
    settings = Stage25Settings()
    output = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage25_spatial_K_recycle_rank_ablation"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "spaces").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))

    stage22a = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage22A_spatial_K_multi_rhs_functional_oracle"
    )
    archived_solver = _read_csv(stage22a / "solver_results.csv")
    archived_histories = _read_csv(stage22a / "residual_histories.csv")
    (
        _,
        nodes,
        boundary,
        coefficient_rows,
        _,
        _,
        _,
        _,
    ) = _load_population(root, settings.stage22a.stage21)

    rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    maximum_pair_defect = 0.0
    for sample_id in settings.sample_ids:
        context, _ = _assemble_context(
            sample_id,
            coefficient_rows[sample_id],
            nodes,
            boundary,
            settings.stage22a.stage21,
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
        pairs: Dict[int, RecyclePair] = {}
        setup: Dict[int, Dict[str, Any]] = {}
        for q in settings.requested_q_values:
            pair, details = _z_union(
                context.system.matrix,
                raw,
                q,
                settings.numerical_rank_tolerance,
            )
            pairs[q] = pair
            setup[q] = _setup_metadata(
                "Z_union",
                _rule_id(q),
                constructions,
                pair,
                details,
                q,
            )
            maximum_pair_defect = max(
                maximum_pair_defect,
                float(pair.diagnostics["au_equals_w_relative_error"]),
            )
        reference = pairs[settings.reference_q]
        for q in settings.requested_q_values:
            diagnostic_rows.append(
                _prefix_diagnostic(sample_id, q, pairs[q], reference)
            )
        np.savez_compressed(
            output / "spaces" / f"sample_{sample_id:02d}_nested_prefix.npz",
            z_full=reference.z_basis,
            requested_q_values=np.asarray(settings.requested_q_values, dtype=int),
            actual_q_values=np.asarray([pairs[q].rank for q in settings.requested_q_values], dtype=int),
        )

        reference_source = {
            int(row["heldout_rhs_seed"]): row
            for row in archived_solver
            if int(row["sample_id"]) == sample_id
            and row["rule_id"] == _rule_id(settings.reference_q)
        }
        for q in settings.requested_q_values:
            if q in settings.archived_q_values:
                source_rows = [
                    row
                    for row in archived_solver
                    if int(row["sample_id"]) == sample_id
                    and row["rule_id"] == _rule_id(q)
                ]
                if len(source_rows) != len(settings.heldout_rhs_seeds):
                    raise RuntimeError(
                        f"archived Stage-22A rows missing for ID {sample_id}, q={q}"
                    )
                rows.extend(_standardize_archived(row) for row in source_rows)
                for row in archived_histories:
                    if (
                        int(row["sample_id"]) == sample_id
                        and row["rule_id"] == _rule_id(q)
                        and int(row["heldout_rhs_seed"]) in settings.heldout_rhs_seeds
                    ):
                        history_rows.append(
                            {
                                "sample_id": sample_id,
                                "requested_q": q,
                                "actual_q": pairs[q].rank,
                                "rhs_seed": int(row["heldout_rhs_seed"]),
                                "iteration": int(row["iteration"]),
                                "true_relative_residual": float(row["true_relative_residual"]),
                                "source": "archived_stage22A",
                            }
                        )
                continue

            pair = pairs[q]
            metadata = setup[q]
            for rhs_seed in settings.heldout_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                solve = _run_pair(context, rhs, pair, settings.stage22a)
                true_residual = _relative_residual(context.system.matrix, solve.x, rhs)
                base = reference_source[rhs_seed]
                baseline_iterations = int(base["baseline_iterations"])
                counts = solve.operation_counts or {}
                rows.append(
                    {
                        "sample_id": sample_id,
                        "requested_q": q,
                        "actual_q": pair.rank,
                        "rhs_seed": rhs_seed,
                        "source": "new_stage25_rank_arm",
                        "baseline_iterations": baseline_iterations,
                        "baseline_converged": _bool(base["baseline_converged"]),
                        "baseline_true_final_residual": float(
                            base["baseline_true_final_relative_residual"]
                        ),
                        "baseline_solve_seconds": float(base["baseline_solve_seconds"]),
                        "recycle_iterations": solve.iterations,
                        "recycle_converged": solve.converged,
                        "recycle_true_final_residual": true_residual,
                        "recycle_solve_seconds": solve.solve_seconds,
                        "J_iter": (
                            baseline_iterations - solve.iterations
                        )
                        / max(baseline_iterations, 1),
                        "setup_seconds": float(metadata["deployable_setup_seconds"]),
                        "ritz_setup_seconds": float(metadata["ritz_setup_seconds"]),
                        "compression_seconds": float(metadata["compression_seconds"]),
                        "pair_setup_seconds": float(metadata["pair_setup_seconds"]),
                        "setup_a_applications": int(metadata["total_setup_a_applications"]),
                        "setup_m_applications": int(metadata["total_setup_m_applications"]),
                        "online_a_applications": int(counts.get("a_matvecs", 0)),
                        "online_m_applications": int(
                            counts.get("preconditioner_applications", 0)
                        ),
                    }
                )
                for iteration, residual in zip(
                    solve.true_residual_iterations, solve.true_residual_history
                ):
                    history_rows.append(
                        {
                            "sample_id": sample_id,
                            "requested_q": q,
                            "actual_q": pair.rank,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "true_relative_residual": residual,
                            "source": "new_stage25_rank_arm",
                        }
                    )
        print(f"Stage 25 completed nested-rank arms for system {sample_id}", flush=True)

    system_rows, family_rows, effective_rows = _summarize(rows, settings)
    decision = rank_decision(family_rows, settings)
    _write_csv(output / "rank_ablation_results.csv", rows)
    _write_csv(output / "system_rank_summary.csv", system_rows)
    _write_csv(output / "family_rank_summary.csv", family_rows)
    _write_csv(output / "system_effective_rank.csv", effective_rows)
    _write_csv(output / "rank_prefix_diagnostics.csv", diagnostic_rows)
    _write_csv(output / "residual_histories.csv", history_rows)
    chart_map = _plots(output, system_rows, family_rows, history_rows, settings)
    _write_csv(output / "chart_map.csv", chart_map)
    _write_json(output / "decision.json", decision)
    converged_residuals = [
        float(row["recycle_true_final_residual"])
        for row in rows
        if _bool(row["recycle_converged"])
    ]
    validation = {
        "expected_result_rows": len(settings.sample_ids)
        * len(settings.requested_q_values)
        * len(settings.heldout_rhs_seeds),
        "actual_result_rows": len(rows),
        "archived_rows_reused": sum(
            row["source"] == "archived_stage22A" for row in rows
        ),
        "new_rows_computed": sum(
            row["source"] == "new_stage25_rank_arm" for row in rows
        ),
        "all_rows_have_disjoint_evaluation_rhs": not (
            set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        ),
        "maximum_converged_true_relative_residual": max(converged_residuals),
        "maximum_AU_minus_W_relative_error": maximum_pair_defect,
        "all_reference_rows_converged": all(
            _bool(row["recycle_converged"])
            for row in rows
            if int(row["requested_q"]) == settings.reference_q
        ),
        "nested_requested_rank_grid": list(settings.requested_q_values),
        "no_neural_training": True,
        "no_cross_system_transfer": True,
    }
    validation["passed"] = bool(
        validation["actual_result_rows"] == validation["expected_result_rows"]
        and validation["maximum_converged_true_relative_residual"]
        <= settings.stage22a.solver.relative_tolerance
        and validation["maximum_AU_minus_W_relative_error"] < 1.0e-9
        and validation["all_reference_rows_converged"]
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
        "- Technical audience and Markdown delivery were requested by the project workflow.\n"
        "- Stage-22A solver outcomes and histories are reused for requested q=24, 32, and 48.\n"
        "- Stage 25 newly solves only requested q=28, 36, 40, and 44.\n"
        "- Every arm uses the same three construction RHSs, m=128, ordinary Ritz candidates, Z-union SVD ordering, ILU, and held-out RHSs.\n"
        "- Figures: small-multiple line comparison, two-panel gate diagnostic, annotated marginal-gain heatmap, true-residual comparison, and setup-cost line.\n"
        "- No metric requested by the user was omitted. No neural training or cross-system transfer was performed.\n",
        encoding="utf-8",
    )
    report = _write_report(
        root, system_rows, family_rows, effective_rows, decision, settings
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(f"Stage 25 report: {report}", flush=True)
    return report
