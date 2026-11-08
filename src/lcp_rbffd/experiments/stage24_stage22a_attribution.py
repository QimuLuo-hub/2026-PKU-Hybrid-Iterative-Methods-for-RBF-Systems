"""Stage 24: controlled attribution of the successful Stage-22A oracle.

All archived Stage-22A solver outcomes are reused.  The only missing solver
arm is a single construction RHS with Arnoldi depth 50 and rank 24.  Spaces
are deterministically reconstructed solely to measure overlap with the
successful three-RHS, m=128, rank-at-most-48 reference.
"""

from __future__ import annotations

import csv
import hashlib
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
    RitzConstruction,
    Stage22ASettings,
    _build_ordinary_construction,
    _raw_dictionary,
    _run_pair,
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
class Stage24Settings:
    """Frozen cohort, arms, and attribution thresholds."""

    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    construction_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    shallow_m: int = 50
    deep_m: int = 128
    individual_q: int = 24
    reference_q: int = 48
    recovery_gate: float = 0.75
    family_fraction_gate: float = 0.75
    system_ilu_reduction_gate: float = 0.40
    numerical_rank_tolerance: float = 1.0e-11

    @property
    def stage22a(self) -> Stage22ASettings:
        return Stage22ASettings()


VARIANTS: Tuple[Tuple[str, str | None, str], ...] = (
    ("single_m50_seed1_q24", None, "single $m=50$, seed 1, $q=24$"),
    (
        "single_m128_seed1_q24",
        "Z_union_n1_m128_q24",
        "single $m=128$, seed 1, $q=24$",
    ),
    (
        "single_m128_seed2_q24",
        "independent_seed20261011_m128_q24",
        "single $m=128$, seed 2, $q=24$",
    ),
    (
        "single_m128_seed3_q24",
        "independent_seed20261021_m128_q24",
        "single $m=128$, seed 3, $q=24$",
    ),
    (
        "union3_m128_q24",
        "Z_union_n3_m128_q24",
        "three $m=128$ union, $q=24$",
    ),
    (
        "reference_union3_m128_q48",
        "Z_union_n3_m128_q48",
        "reference: three $m=128$ union, $q\\leq 48$",
    ),
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _variant_label(variant_id: str) -> str:
    return next(label for key, _, label in VARIANTS if key == variant_id)


def _build_spaces(
    context,
    settings: Stage24Settings,
) -> Tuple[Dict[str, RecyclePair], Dict[str, Dict[str, Any]], Dict[int, RitzConstruction]]:
    constructions = {
        seed: _build_ordinary_construction(
            context,
            seed,
            settings.deep_m,
            settings.stage22a,
        )
        for seed in settings.construction_rhs_seeds
    }
    shallow = _build_ordinary_construction(
        context,
        settings.construction_rhs_seeds[0],
        settings.shallow_m,
        settings.stage22a,
    )
    first, second, third = (constructions[seed] for seed in settings.construction_rhs_seeds)
    raw_three = _raw_dictionary((first, second, third))
    inputs = {
        "single_m50_seed1_q24": (shallow.basis, settings.individual_q, (shallow,)),
        "single_m128_seed1_q24": (first.basis, settings.individual_q, (first,)),
        "single_m128_seed2_q24": (second.basis, settings.individual_q, (second,)),
        "single_m128_seed3_q24": (third.basis, settings.individual_q, (third,)),
        "union3_m128_q24": (raw_three, settings.individual_q, (first, second, third)),
        "reference_union3_m128_q48": (
            raw_three,
            settings.reference_q,
            (first, second, third),
        ),
    }
    pairs: Dict[str, RecyclePair] = {}
    reconstructed: Dict[str, Dict[str, Any]] = {}
    for variant_id, (raw, requested_q, sources) in inputs.items():
        pair, details = _z_union(
            context.system.matrix,
            raw,
            requested_q,
            settings.numerical_rank_tolerance,
        )
        pairs[variant_id] = pair
        reconstructed[variant_id] = {
            "sample_id": context.sample_id,
            "variant_id": variant_id,
            "construction_rhs_count": len(sources),
            "arnoldi_depth": sources[0].arnoldi_dimension,
            "requested_q": requested_q,
            "actual_q": pair.rank,
            "reconstructed_ritz_seconds": float(sum(item.setup_seconds for item in sources)),
            "reconstructed_compression_seconds": float(details["compression_seconds"]),
            "reconstructed_pair_seconds": float(details["pair_setup_seconds"]),
            "reconstructed_total_setup_seconds": float(
                sum(item.setup_seconds for item in sources)
                + details["compression_seconds"]
                + details["pair_setup_seconds"]
            ),
            "dictionary_columns": int(details["dictionary_columns"]),
            "dictionary_rank": int(details["dictionary_rank"]),
            "au_equals_w_relative_error": float(
                pair.diagnostics["au_equals_w_relative_error"]
            ),
            "test_orthogonality_error": float(
                pair.diagnostics["test_orthogonality_error"]
            ),
        }
    constructions[settings.shallow_m] = shallow
    return pairs, reconstructed, constructions


def _geometry_row(
    sample_id: int,
    variant_id: str,
    candidate: RecyclePair,
    reference: RecyclePair,
) -> Dict[str, Any]:
    z = generalized_subspace_metrics(candidate.z_basis, reference.z_basis)
    w = generalized_subspace_metrics(candidate.w_basis, reference.w_basis)
    return {
        "sample_id": sample_id,
        "variant_id": variant_id,
        "actual_q": candidate.rank,
        "reference_q": reference.rank,
        "d_Z_to_reference": float(z["projector_distance_normalized"]),
        "d_W_to_reference": float(w["projector_distance_normalized"]),
        "max_angle_Z_degrees": float(z["principal_angle_max_degrees"]),
        "mean_angle_Z_degrees": float(z["principal_angle_mean_degrees"]),
        "max_angle_W_degrees": float(w["principal_angle_max_degrees"]),
        "mean_angle_W_degrees": float(w["principal_angle_mean_degrees"]),
        "candidate_contained_in_reference_Z": float(z["containment_first"]),
        "reference_energy_seen_by_candidate_Z": float(z["containment_second"]),
        "candidate_contained_in_reference_W": float(w["containment_first"]),
        "reference_energy_seen_by_candidate_W": float(w["containment_second"]),
    }


def _standardize_archived_row(
    row: Mapping[str, str], variant_id: str
) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "variant_id": variant_id,
        "variant_label": _variant_label(variant_id),
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
        "actual_q": int(row["actual_q"]),
        "construction_seconds": float(row["deployable_setup_seconds"]),
        "archived_rule_id": row["rule_id"],
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage24Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reference = {
        (int(row["sample_id"]), int(row["rhs_seed"])): int(row["recycle_iterations"])
        for row in rows
        if row["variant_id"] == "reference_union3_m128_q48"
    }
    system_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        for variant_id, _, _ in VARIANTS:
            local = [
                row
                for row in rows
                if int(row["sample_id"]) == sample_id
                and row["variant_id"] == variant_id
            ]
            eta = []
            for row in local:
                baseline = int(row["baseline_iterations"])
                denominator = baseline - reference[(sample_id, int(row["rhs_seed"]))]
                eta.append(
                    (baseline - int(row["recycle_iterations"])) / denominator
                    if denominator > 0
                    else float("nan")
                )
            geom = next(
                row
                for row in geometry
                if int(row["sample_id"]) == sample_id
                and row["variant_id"] == variant_id
            )
            median_j = float(np.median([float(row["J_iter"]) for row in local]))
            median_eta = float(np.nanmedian(eta))
            all_converged = all(_bool(row["recycle_converged"]) for row in local)
            system_rows.append(
                {
                    "sample_id": sample_id,
                    "variant_id": variant_id,
                    "variant_label": _variant_label(variant_id),
                    "rhs_count": len(local),
                    "all_rhs_converged": all_converged,
                    "converged_rhs": sum(_bool(row["recycle_converged"]) for row in local),
                    "baseline_iterations_median": float(
                        np.median([int(row["baseline_iterations"]) for row in local])
                    ),
                    "recycle_iterations_median": float(
                        np.median([int(row["recycle_iterations"]) for row in local])
                    ),
                    "recycle_iterations_min": min(int(row["recycle_iterations"]) for row in local),
                    "recycle_iterations_max": max(int(row["recycle_iterations"]) for row in local),
                    "J_iter_median": median_j,
                    "reference_benefit_recovery_median": median_eta,
                    "actual_q": int(local[0]["actual_q"]),
                    "construction_seconds": float(local[0]["construction_seconds"]),
                    "online_seconds_median": float(
                        np.median([float(row["recycle_solve_seconds"]) for row in local])
                    ),
                    "d_Z_to_reference": float(geom["d_Z_to_reference"]),
                    "d_W_to_reference": float(geom["d_W_to_reference"]),
                    "reference_recovery_gate_passed": bool(
                        all_converged and median_eta >= settings.recovery_gate
                    ),
                    "ilu_reduction_gate_passed": bool(
                        all_converged and median_j >= settings.system_ilu_reduction_gate
                    ),
                }
            )
    family_rows: List[Dict[str, Any]] = []
    required = int(np.ceil(settings.family_fraction_gate * len(settings.sample_ids)))
    for variant_id, _, _ in VARIANTS:
        local = [row for row in system_rows if row["variant_id"] == variant_id]
        passing = sum(_bool(row["reference_recovery_gate_passed"]) for row in local)
        median_eta = float(
            np.median([float(row["reference_benefit_recovery_median"]) for row in local])
        )
        family_rows.append(
            {
                "variant_id": variant_id,
                "variant_label": _variant_label(variant_id),
                "systems": len(local),
                "systems_recovering_most_reference_benefit": passing,
                "required_systems": required,
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
                "all_rhs_convergence_systems": sum(_bool(row["all_rhs_converged"]) for row in local),
                "median_actual_q": float(np.median([float(row["actual_q"]) for row in local])),
                "median_construction_seconds": float(
                    np.median([float(row["construction_seconds"]) for row in local])
                ),
                "median_d_Z_to_reference": float(
                    np.median([float(row["d_Z_to_reference"]) for row in local])
                ),
                "median_d_W_to_reference": float(
                    np.median([float(row["d_W_to_reference"]) for row in local])
                ),
                "family_reproduction_gate_passed": bool(
                    passing >= required and median_eta >= settings.recovery_gate
                ),
            }
        )
    return system_rows, family_rows


def attribution_decision(
    family_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_id = {str(row["variant_id"]): row for row in family_rows}
    shallow = by_id["single_m50_seed1_q24"]
    deep = by_id["single_m128_seed1_q24"]
    union24 = by_id["union3_m128_q24"]
    reference = by_id["reference_union3_m128_q48"]
    individuals = [
        by_id["single_m128_seed1_q24"],
        by_id["single_m128_seed2_q24"],
        by_id["single_m128_seed3_q24"],
    ]
    depth_sufficient = bool(
        _bool(deep["family_reproduction_gate_passed"])
        and not _bool(shallow["family_reproduction_gate_passed"])
    )
    diversity_sufficient = bool(
        _bool(union24["family_reproduction_gate_passed"])
        and not _bool(deep["family_reproduction_gate_passed"])
    )
    combination_sufficient = bool(
        _bool(reference["family_reproduction_gate_passed"])
        and not _bool(union24["family_reproduction_gate_passed"])
        and all(not _bool(row["family_reproduction_gate_passed"]) for row in individuals)
    )
    if depth_sufficient:
        category = "A"
        conclusion = "deeper Krylov information is sufficient"
    elif diversity_sufficient:
        category = "B"
        conclusion = "multiple independent trajectories at fixed rank are sufficient"
    elif combination_sufficient:
        category = "C"
        conclusion = "retaining the combined multi-trajectory span through q<=48 is required"
    else:
        category = "D"
        conclusion = "no single tested factor explains the successful reference"
    return {
        "category": category,
        "conclusion": conclusion,
        "depth_sufficient": depth_sufficient,
        "multiple_rhs_at_fixed_rank_sufficient": diversity_sufficient,
        "combined_q48_sufficient": combination_sufficient,
    }


def _plots(
    output: Path,
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    settings: Stage24Settings,
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Dict[str, str]] = []
    ids = list(settings.sample_ids)
    colors = {
        "single_m50_seed1_q24": GRAY,
        "single_m128_seed1_q24": BLUE,
        "single_m128_seed2_q24": PURPLE,
        "single_m128_seed3_q24": PINK,
        "union3_m128_q24": ORANGE,
        "reference_union3_m128_q48": OLIVE,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    panels = (
        (
            axes[0],
            ("single_m50_seed1_q24", "single_m128_seed1_q24", "reference_union3_m128_q48"),
            "Depth comparison",
        ),
        (
            axes[1],
            (
                "single_m128_seed1_q24",
                "single_m128_seed2_q24",
                "single_m128_seed3_q24",
                "union3_m128_q24",
                "reference_union3_m128_q48",
            ),
            "Trajectory and retained-rank comparison",
        ),
    )
    system_positions = np.arange(len(ids))
    for ax, variants, title in panels:
        for variant in variants:
            values = [
                next(
                    float(row["recycle_iterations_median"])
                    for row in system_rows
                    if int(row["sample_id"]) == sample_id
                    and row["variant_id"] == variant
                )
                for sample_id in ids
            ]
            ax.plot(system_positions, values, marker="o", linewidth=1.6, color=colors[variant], label=_variant_label(variant))
        ax.axhline(300, color=CHARCOAL, linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("spatial-K system ID")
        ax.set_xticks(system_positions, [str(sample_id) for sample_id in ids])
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("median held-out iterations")
    fig.tight_layout()
    path = plot_dir / "controlled_iteration_attribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "system_variant_summary.csv", "purpose": "Isolate depth, trajectory diversity, and retained combined rank."})

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    positions = np.arange(len(VARIANTS))
    for index, (variant, _, label) in enumerate(VARIANTS):
        values = [
            float(row["reference_benefit_recovery_median"])
            for row in system_rows
            if row["variant_id"] == variant
        ]
        ax.scatter(np.full(len(values), index), values, color=colors[variant], s=28, alpha=0.75)
        ax.plot([index - 0.22, index + 0.22], [np.median(values)] * 2, color=CHARCOAL, linewidth=2)
    ax.axhline(settings.recovery_gate, color=CHARCOAL, linestyle="--", linewidth=0.9, label="75% recovery gate")
    ax.set_xticks(
        positions,
        (
            "1 RHS, m=50, q=24",
            "RHS 1, m=128, q=24",
            "RHS 2, m=128, q=24",
            "RHS 3, m=128, q=24",
            "3-RHS union, q=24",
            "3-RHS union, q<=48",
        ),
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("median own-oracle benefit recovery")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "reference_benefit_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "system_variant_summary.csv", "purpose": "Show which arms recover most of the successful reference benefit across systems."})

    fig, ax = plt.subplots(figsize=(6.7, 4.7))
    for variant, _, label in VARIANTS[:-1]:
        local = [row for row in system_rows if row["variant_id"] == variant]
        ax.scatter(
            [float(row["d_Z_to_reference"]) for row in local],
            [float(row["reference_benefit_recovery_median"]) for row in local],
            color=colors[variant],
            s=28,
            alpha=0.72,
            label=label,
        )
    ax.axhline(settings.recovery_gate, color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax.set_xlabel("normalized $Z$-projector distance to reference")
    ax.set_ylabel("median reference-benefit recovery")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = plot_dir / "overlap_vs_solver_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "system_variant_summary.csv", "purpose": "Relate reference-space overlap to functional recovery without treating geometry as the decision metric."})

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    annotation_style = {
        "single_m50_seed1_q24": ("m=50 single", (4, 5), "left"),
        "single_m128_seed1_q24": ("m=128 seed 1", (4, -13), "left"),
        "single_m128_seed2_q24": ("m=128 seed 2", (4, 4), "left"),
        "single_m128_seed3_q24": ("m=128 seed 3", (4, 14), "left"),
        "union3_m128_q24": ("3-RHS q=24", (4, -13), "left"),
        "reference_union3_m128_q48": ("reference q<=48", (-5, 5), "right"),
    }
    for row in family_rows:
        variant = str(row["variant_id"])
        ax.scatter(
            float(row["median_construction_seconds"]),
            float(row["median_reference_benefit_recovery"]),
            color=colors[variant],
            s=55,
        )
        annotation, offset, horizontal_alignment = annotation_style[variant]
        ax.annotate(
            annotation,
            (float(row["median_construction_seconds"]), float(row["median_reference_benefit_recovery"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha=horizontal_alignment,
        )
    ax.axhline(settings.recovery_gate, color=CHARCOAL, linestyle="--", linewidth=0.8)
    ax.set_xlabel("median construction cost (s)")
    ax.set_ylabel("median reference-benefit recovery")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    path = plot_dir / "construction_cost_vs_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "family_variant_summary.csv", "purpose": "Expose the cost paid for deeper and combined construction information."})

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
    for ax, sample_id in zip(axes, (10, 17)):
        for variant in (
            "ilu",
            "single_m50_seed1_q24",
            "single_m128_seed1_q24",
            "union3_m128_q24",
            "reference_union3_m128_q48",
        ):
            local = [
                row
                for row in history_rows
                if int(row["sample_id"]) == sample_id
                and int(row["rhs_seed"]) == settings.heldout_rhs_seeds[0]
                and row["variant_id"] == variant
            ]
            color = CHARCOAL if variant == "ilu" else colors[variant]
            linestyle = "--" if variant == "ilu" else ":" if variant == "single_m50_seed1_q24" else "-"
            ax.semilogy(
                [int(row["iteration"]) for row in local],
                [max(float(row["true_relative_residual"]), 1e-16) for row in local],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                label="ILU" if variant == "ilu" else _variant_label(variant),
            )
        ax.axhline(settings.stage22a.solver.relative_tolerance, color=CHARCOAL, linestyle="--", linewidth=0.7)
        ax.set_title(f"ID {sample_id}")
        ax.set_xlabel("iteration")
        ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].set_ylabel("true relative residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = plot_dir / "representative_residual_histories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    charts.append({"figure": str(path.relative_to(output)), "source": "residual_histories.csv", "purpose": "Show how the retained combined span repairs stubborn systems."})
    return charts


def _write_report(
    root: Path,
    output: Path,
    system_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage24Settings,
) -> Path:
    report = root / "results" / "results-fd" / "reports" / "stage24_stage22A_oracle_attribution.md"
    family = {str(row["variant_id"]): row for row in family_rows}
    label_short = {
        "single_m50_seed1_q24": "1 RHS, $m=50$, $q=24$",
        "single_m128_seed1_q24": "1 RHS, $m=128$, $q=24$",
        "single_m128_seed2_q24": "RHS 2, $m=128$, $q=24$",
        "single_m128_seed3_q24": "RHS 3, $m=128$, $q=24$",
        "union3_m128_q24": "3-RHS union, $m=128$, $q=24$",
        "reference_union3_m128_q48": "3-RHS union, $m=128$, $q\\leq48$",
    }
    lines = [
        "# Stage 24: attribution of the successful Stage-22A own-system oracle",
        "",
        "## Technical conclusion",
        "",
        f"**Decision {decision['category']}: {decision['conclusion']}.** Deeper Arnoldi information is important but is not sufficient by itself. Multiple construction trajectories compressed back to rank 24 are also not sufficient. The only arm that passes the preregistered family gate is the Stage-22A reference that retains up to 48 directions from the joint three-trajectory dictionary. The evidence therefore attributes robust success to the retained combined span, not to trajectory count or compression at a fixed rank alone.",
        "",
        "A variant reproduces most reference benefit on a system only if all three held-out right-hand sides converge and median $\\eta_{ref}\\ge0.75$, where $\\eta_{ref}=(k_{ILU}-k_v)/(k_{ILU}-k_{ref})$. A factor is sufficient only if one frozen variant passes at least six of eight systems and has population median $\\eta_{ref}\\ge0.75$. These thresholds and the factor-priority rule A, then B, then C, then D were saved before the missing $m=50$ arm ran.",
        "",
        "| controlled arm | systems recovering most benefit | systems with all RHS converged | median $\\eta_{ref}$ | median ILU reduction | median rank | median setup (s) | median $d_Z$ | median $d_W$ |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for variant, _, _ in VARIANTS:
        row = family[variant]
        lines.append(
            f"| {label_short[variant]} | {row['systems_recovering_most_reference_benefit']}/8 | {row['all_rhs_convergence_systems']}/8 | {float(row['median_reference_benefit_recovery']):.3f} | {1.0 - float(row['median_iteration_ratio_to_ilu']):.3f} | {float(row['median_actual_q']):.0f} | {float(row['median_construction_seconds']):.3f} | {float(row['median_d_Z_to_reference']):.3f} | {float(row['median_d_W_to_reference']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The per-system entries below are median held-out iterations followed by the number of converged right-hand sides out of three. The ILU column is the corresponding system-level median baseline.",
            "",
            "| ID | ILU | $m=50$ single | $m=128$ seed 1 | $m=128$ seed 2 | $m=128$ seed 3 | 3-RHS $q=24$ | reference $q\\leq48$ |",
            "|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for sample_id in settings.sample_ids:
        local = {
            str(row["variant_id"]): row
            for row in system_rows
            if int(row["sample_id"]) == sample_id
        }
        cells = []
        for variant, _, _ in VARIANTS:
            row = local[variant]
            cells.append(
                f"{float(row['recycle_iterations_median']):.0f} "
                f"[{int(row['converged_rhs'])}/3]"
            )
        ilu_median = float(local[VARIANTS[0][0]]["baseline_iterations_median"])
        lines.append(f"| {sample_id} | {ilu_median:.0f} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "![Controlled iteration attribution](../stages/stage24_stage22A_oracle_attribution/plots/controlled_iteration_attribution.png)",
            "",
            "## Depth helps, but leaves three systems below the recovery gate",
            "",
            "The depth contrast uses the same construction seed 20261001, the same ordinary smallest-magnitude pair-preserving Ritz rule, and rank 24. Only Arnoldi depth changes from 50 to 128. The deeper arm sharply improves the family result, but it recovers at least 75% of reference benefit on only five systems. IDs 3, 10, and 17 remain below the gate, with ID 17 still capped. Depth is therefore a necessary contributor on this comparison, not a sufficient explanation of the successful reference.",
            "",
            "## More trajectories do not help when the union is truncated to 24",
            "",
            "At fixed $m=128$ and final rank 24, replacing the first trajectory by a three-trajectory SVD union leaves the recovery count at five of eight. The population median recovery changes only modestly, and ID 17 remains capped. The second and third individual trajectory spaces also each pass five systems. Thus merely observing three independent trajectories, or using their leading rank-24 union, does not reproduce Stage-22A robustness.",
            "",
            "## Retaining the larger joint span repairs every stubborn system",
            "",
            "The decisive controlled change is between two spaces built from exactly the same three $m=128$ trajectory dictionaries: truncation at rank 24 versus retention of up to 48 SVD directions. The rank-48 reference converges on all 24 held-out solves and recovers its full benefit by definition. It reduces median iterations on IDs 3, 10, and 17 from 147, 186, and 300 under the rank-24 union to 33, 38, and 91. It preserves the already strong results on IDs 14, 15, 19, and 34.",
            "",
            "![Reference-benefit recovery](../stages/stage24_stage22A_oracle_attribution/plots/reference_benefit_recovery.png)",
            "",
            "![Representative residual histories](../stages/stage24_stage22A_oracle_attribution/plots/representative_residual_histories.png)",
            "",
            "This is evidence for the retained combined multi-trajectory span. It is not evidence that SVD geometry itself is the cause. At rank 24, the same SVD combination is inadequate. The existing construction also exposes only 24 candidates per individual trajectory, so this frozen experiment cannot separate a generic rank-48 effect from the need for cross-RHS diversity without changing the established target construction.",
            "",
            "## Geometry is diagnostic, not the attribution criterion",
            "",
            "Every candidate was reconstructed from the saved matrices and construction seeds to measure solution-space and target-side residual-space overlap with the reference. Unequal-rank normalized projector distance is $\\|P_1-P_2\\|_F/\\sqrt{q_1+q_2}$. The rank-24 union is nested in the reference SVD basis, yet it still fails three systems. Consequently containment or small projector error cannot replace held-out solver evaluation.",
            "",
            "![Overlap versus solver recovery](../stages/stage24_stage22A_oracle_attribution/plots/overlap_vs_solver_recovery.png)",
            "",
            "## Cost and provenance",
            "",
            "All $m=128$ solver outcomes, setup times, and residual histories are reused directly from the validated Stage-22A artifacts. Only the single-RHS $m=50$ arm is newly solved. Deterministic reconstruction of the archived spaces is used only for overlap diagnostics and is recorded separately from the archived setup cost. The reference's extra robustness therefore retains its original construction cost rather than receiving a timing advantage from this reuse.",
            "",
            "![Construction cost versus recovery](../stages/stage24_stage22A_oracle_attribution/plots/construction_cost_vs_recovery.png)",
            "",
            "## Limitations and next mechanism",
            "",
            "The conclusion is restricted to the established eight-system hard cohort, ordinary Ritz ordering, three fixed construction seeds, and the existing 24-candidate-per-trajectory rule. Fourteen of the broader Stage-22A hard baselines were capped, and seven of eight baselines here are capped, so iteration reductions use the project's censored convention. The experiment identifies sufficiency among the requested frozen arms. It does not prove spectral necessity or a unique invariant-space decomposition.",
            "",
            "**One next step:** study how the additional 24 retained directions in the successful joint span act on the three stubborn systems, using deletion/block ablation inside the existing classical oracle. Do not train a predictor before that mechanism is localized.",
            "",
            "## Saved evidence",
            "",
            "Per-RHS results, system and family summaries, reconstructed-space overlap diagnostics, combined residual histories, source provenance, validation checks, figures, and hashes are saved in `results/results-fd/stages/stage24_stage22A_oracle_attribution/`.",
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


def run_stage24(root: Path) -> Path:
    settings = Stage24Settings()
    output = root / "results" / "results-fd" / "stages" / "stage24_stage22A_oracle_attribution"
    output.mkdir(parents=True, exist_ok=True)
    (output / "spaces").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "predeclared_protocol.json",
        {
            "written_before_missing_arm_execution": True,
            "sample_ids": list(settings.sample_ids),
            "construction_rhs_seeds": list(settings.construction_rhs_seeds),
            "heldout_rhs_seeds": list(settings.heldout_rhs_seeds),
            "variants": [key for key, _, _ in VARIANTS],
            "system_reproduction_gate": {
                "all_three_heldout_rhs_converge": True,
                "median_reference_benefit_recovery_at_least": settings.recovery_gate,
            },
            "family_reproduction_gate": {
                "system_fraction_at_least": settings.family_fraction_gate,
                "required_systems": 6,
                "median_reference_benefit_recovery_at_least": settings.recovery_gate,
            },
            "attribution_priority": [
                "A if single m128 q24 passes and m50 does not",
                "B if three-RHS m128 q24 passes and single m128 q24 does not",
                "C if reference q<=48 passes while q24 union and all individuals fail",
                "D otherwise",
            ],
            "neural_training": False,
        },
    )

    stage22a_dir = root / "results" / "results-fd" / "stages" / "stage22A_spatial_K_multi_rhs_functional_oracle"
    archived_solver = _read_csv(stage22a_dir / "solver_results.csv")
    archived_setup = _read_csv(stage22a_dir / "setup_results.csv")
    archived_histories = _read_csv(stage22a_dir / "residual_histories.csv")
    stage23_histories = _read_csv(
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage23_dynamic_cycle_recycle_diagnostic"
        / "residual_histories.csv"
    )
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

    result_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []
    new_history_rows: List[Dict[str, Any]] = []
    max_space_reconstruction_defect = 0.0
    for sample_id in settings.sample_ids:
        context, _ = _assemble_context(
            sample_id,
            coefficient_rows[sample_id],
            nodes,
            boundary,
            settings.stage22a.stage21,
        )
        pairs, reconstruction, _ = _build_spaces(context, settings)
        reference = pairs["reference_union3_m128_q48"]
        np.savez_compressed(
            output / "spaces" / f"sample_{sample_id:02d}_spaces.npz",
            **{variant: pair.z_basis for variant, pair in pairs.items()},
        )
        for variant_id, pair in pairs.items():
            geometry_rows.append(_geometry_row(sample_id, variant_id, pair, reference))
            reconstruction_rows.append(reconstruction[variant_id])
            max_space_reconstruction_defect = max(
                max_space_reconstruction_defect,
                float(reconstruction[variant_id]["au_equals_w_relative_error"]),
            )

        for variant_id, archived_rule, _ in VARIANTS:
            if archived_rule is None:
                continue
            source_rows = [
                row
                for row in archived_solver
                if int(row["sample_id"]) == sample_id
                and row["rule_id"] == archived_rule
            ]
            if len(source_rows) != len(settings.heldout_rhs_seeds):
                raise RuntimeError(
                    f"archived rule {archived_rule} has {len(source_rows)} rows for ID {sample_id}"
                )
            result_rows.extend(
                _standardize_archived_row(row, variant_id) for row in source_rows
            )

        shallow_variant = "single_m50_seed1_q24"
        shallow_pair = pairs[shallow_variant]
        shallow_reconstruction = reconstruction[shallow_variant]
        reference_source = {
            int(row["heldout_rhs_seed"]): row
            for row in archived_solver
            if int(row["sample_id"]) == sample_id
            and row["rule_id"] == "Z_union_n3_m128_q48"
        }
        for rhs_seed in settings.heldout_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            solve = _run_pair(context, rhs, shallow_pair, settings.stage22a)
            true_residual = _relative_residual(context.system.matrix, solve.x, rhs)
            base = reference_source[rhs_seed]
            k_base = int(base["baseline_iterations"])
            result_rows.append(
                {
                    "sample_id": sample_id,
                    "variant_id": shallow_variant,
                    "variant_label": _variant_label(shallow_variant),
                    "rhs_seed": rhs_seed,
                    "source": "new_stage24_missing_arm",
                    "baseline_iterations": k_base,
                    "baseline_converged": _bool(base["baseline_converged"]),
                    "baseline_true_final_residual": float(
                        base["baseline_true_final_relative_residual"]
                    ),
                    "baseline_solve_seconds": float(base["baseline_solve_seconds"]),
                    "recycle_iterations": solve.iterations,
                    "recycle_converged": solve.converged,
                    "recycle_true_final_residual": true_residual,
                    "recycle_solve_seconds": solve.solve_seconds,
                    "J_iter": (k_base - solve.iterations) / max(k_base, 1),
                    "actual_q": shallow_pair.rank,
                    "construction_seconds": shallow_reconstruction[
                        "reconstructed_total_setup_seconds"
                    ],
                    "archived_rule_id": "",
                }
            )
            for iteration, residual in zip(
                solve.true_residual_iterations, solve.true_residual_history
            ):
                new_history_rows.append(
                    {
                        "sample_id": sample_id,
                        "variant_id": shallow_variant,
                        "rhs_seed": rhs_seed,
                        "iteration": iteration,
                        "true_relative_residual": residual,
                        "source": "new_stage24_missing_arm",
                    }
                )
        print(f"Stage 24 completed missing arm and overlap for system {sample_id}", flush=True)

    geometry_lookup = {
        (int(row["sample_id"]), row["variant_id"]): row for row in geometry_rows
    }
    # Archived setup values remain the reported costs for archived solver rows.
    setup_lookup = {
        (int(row["sample_id"]), row["rule_id"]): row for row in archived_setup
    }
    for row in result_rows:
        if row["source"] == "archived_stage22A":
            setup = setup_lookup[(int(row["sample_id"]), row["archived_rule_id"])]
            row["construction_seconds"] = float(setup["deployable_setup_seconds"])
        geom = geometry_lookup[(int(row["sample_id"]), row["variant_id"])]
        row["d_Z_to_reference"] = geom["d_Z_to_reference"]
        row["d_W_to_reference"] = geom["d_W_to_reference"]

    system_rows, family_rows = _summarize(result_rows, geometry_rows, settings)
    decision = attribution_decision(family_rows)
    _write_csv(output / "controlled_variant_results.csv", result_rows)
    _write_csv(output / "space_overlap_diagnostics.csv", geometry_rows)
    _write_csv(output / "space_reconstruction_diagnostics.csv", reconstruction_rows)
    _write_csv(output / "system_variant_summary.csv", system_rows)
    _write_csv(output / "family_variant_summary.csv", family_rows)

    history_rows: List[Dict[str, Any]] = list(new_history_rows)
    rule_to_variant = {
        archived_rule: variant
        for variant, archived_rule, _ in VARIANTS
        if archived_rule is not None
    }
    for row in archived_histories:
        if (
            int(row["sample_id"]) in settings.sample_ids
            and row["rule_id"] in rule_to_variant
            and int(row["heldout_rhs_seed"]) in settings.heldout_rhs_seeds
        ):
            history_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "variant_id": rule_to_variant[row["rule_id"]],
                    "rhs_seed": int(row["heldout_rhs_seed"]),
                    "iteration": int(row["iteration"]),
                    "true_relative_residual": float(row["true_relative_residual"]),
                    "source": "archived_stage22A",
                }
            )
    for row in stage23_histories:
        if (
            int(row["sample_id"]) in settings.sample_ids
            and row["method"] == "ilu"
            and int(row["rhs_seed"]) in settings.heldout_rhs_seeds
        ):
            history_rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "variant_id": "ilu",
                    "rhs_seed": int(row["rhs_seed"]),
                    "iteration": int(row["iteration"]),
                    "true_relative_residual": float(row["true_relative_residual"]),
                    "source": "archived_stage23_baseline",
                }
            )
    _write_csv(output / "residual_histories.csv", history_rows)

    factor_rows = []
    family_lookup = {row["variant_id"]: row for row in family_rows}
    for factor, before, after in (
        ("A_arnoldi_depth", "single_m50_seed1_q24", "single_m128_seed1_q24"),
        ("B_multiple_rhs_fixed_rank", "single_m128_seed1_q24", "union3_m128_q24"),
        ("C_retained_combined_span", "union3_m128_q24", "reference_union3_m128_q48"),
    ):
        first = family_lookup[before]
        second = family_lookup[after]
        factor_rows.append(
            {
                "factor": factor,
                "before_variant": before,
                "after_variant": after,
                "before_systems_recovering": first[
                    "systems_recovering_most_reference_benefit"
                ],
                "after_systems_recovering": second[
                    "systems_recovering_most_reference_benefit"
                ],
                "before_median_eta": first["median_reference_benefit_recovery"],
                "after_median_eta": second["median_reference_benefit_recovery"],
                "after_family_gate_passed": second[
                    "family_reproduction_gate_passed"
                ],
            }
        )
    _write_csv(output / "factor_attribution_summary.csv", factor_rows)
    chart_map = _plots(output, system_rows, family_rows, history_rows, settings)
    _write_csv(output / "chart_map.csv", chart_map)
    _write_json(output / "decision.json", decision)
    _write_json(
        output / "validation.json",
        {
            "expected_result_rows": len(settings.sample_ids)
            * len(settings.heldout_rhs_seeds)
            * len(VARIANTS),
            "actual_result_rows": len(result_rows),
            "archived_rows_reused": sum(
                row["source"] == "archived_stage22A" for row in result_rows
            ),
            "new_rows_computed": sum(
                row["source"] == "new_stage24_missing_arm" for row in result_rows
            ),
            "all_variants_have_three_rhs": all(
                int(row["rhs_count"]) == 3 for row in system_rows
            ),
            "all_reference_rows_converged": all(
                _bool(row["recycle_converged"])
                for row in result_rows
                if row["variant_id"] == "reference_union3_m128_q48"
            ),
            "maximum_converged_true_relative_residual": max(
                float(row["recycle_true_final_residual"])
                for row in result_rows
                if _bool(row["recycle_converged"])
            ),
            "maximum_reconstructed_AU_minus_W_relative_error": max_space_reconstruction_defect,
            "construction_and_evaluation_seeds_disjoint": not (
                set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
            ),
            "no_neural_training": True,
            "passed": bool(
                len(result_rows)
                == len(settings.sample_ids)
                * len(settings.heldout_rhs_seeds)
                * len(VARIANTS)
                and max_space_reconstruction_defect < 1.0e-9
            ),
        },
    )
    (output / "source_notes.md").write_text(
        "# Source notes\n\n"
        "- Archived solver outcomes, setup costs, and histories: Stage 22A.\n"
        "- Baseline histories: Stage 23, whose iteration counts match Stage 21.\n"
        "- New computation: only single construction RHS, m=50, q=24 on the eight fixed systems.\n"
        "- Reconstructed spaces: deterministic Stage-22A construction code and saved Stage-19A matrices/nodes, used only for overlap diagnostics.\n"
        "- No cross-system transfer and no neural model.\n",
        encoding="utf-8",
    )
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    report = _write_report(root, output, system_rows, family_rows, decision, settings)
    _write_json(output / "manifest.json", _manifest(output))
    print(f"Stage 24 report: {report}", flush=True)
    return report
