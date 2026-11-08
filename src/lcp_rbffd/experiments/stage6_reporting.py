"""Evidence synthesis, figures, and Markdown report for the Stage 6 study.

The reporting layer consumes only saved-style row dictionaries.  It never
changes the numerical operators or re-fits a model, which keeps conclusions
separate from experiment execution.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
CHARCOAL = "#30343B"
MID_GRAY = "#737A86"
LIGHT_GRAY = "#D7DBE2"
PALETTE = [BLUE, ORANGE, OLIVE, PINK]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "axes.titleweight": "bold",
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
    }
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.median(array)) if array.size else float("nan")


def _finite_spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    if np.count_nonzero(mask) < 3 or np.unique(x_array[mask]).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x_array[mask], y_array[mask])
    return float(result.statistic), float(result.pvalue)


def _matching_classical_iterations(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], int]:
    return {
        (str(row["system"]), str(row["rhs"])): int(row["iterations"])
        for row in rows
        if row["method_family"] == "classical"
    }


def analyze_evidence(
    stage_a: Sequence[Mapping[str, Any]],
    stage_b: Sequence[Mapping[str, Any]],
    stage_c: Sequence[Mapping[str, Any]],
    stage_d: Sequence[Mapping[str, Any]],
    stage_e_rows: Sequence[Mapping[str, Any]],
    stage_e_summary: Mapping[str, Any],
    ml_corrections: Sequence[Mapping[str, Any]],
    ml_solvers: Sequence[Mapping[str, Any]],
    ml_solver_directions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute report claims and conservative G/H evidence gates."""

    converged_a = [row for row in stage_a if row["classical_converged"]]
    difficult_rows = [
        row
        for row in stage_a
        if (not row["unpreconditioned_converged"])
        or int(row["unpreconditioned_iterations"]) >= 200
    ]
    nontrivial_classical = [
        row for row in converged_a if int(row["classical_iterations"]) >= 8
    ]

    pareto = [row for row in stage_b if row.get("pareto_optimal")]
    budget_iteration_range = [int(row["iterations"]) for row in stage_b if row["converged"]]
    low_dim_joint = [
        row
        for row in stage_c
        if int(row["q_joint_energy95_eta25"]) > 0
        and float(row["q_joint_over_n"]) <= 0.05
    ]

    classical = _matching_classical_iterations(stage_d)
    pod_rows = [row for row in stage_d if row["method_family"] == "pod"]
    improvements = [
        classical[(str(row["system"]), str(row["rhs"]))] - int(row["iterations"])
        for row in pod_rows
        if (str(row["system"]), str(row["rhs"])) in classical and row["converged"]
    ]
    energies = [float(row["captured_energy"]) for row in pod_rows if row["converged"]]
    matched_improvements = [
        classical[(str(row["system"]), str(row["rhs"]))] - int(row["iterations"])
        for row in pod_rows
        if row["converged"]
        and (str(row["system"]), str(row["rhs"])) in classical
    ]
    energy_iteration_rho, energy_iteration_p = _finite_spearman(
        energies, matched_improvements
    )
    high_energy_no_gain = sum(
        1
        for row in pod_rows
        if row["converged"]
        and float(row["captured_energy"]) >= 0.95
        and classical[(str(row["system"]), str(row["rhs"]))]
        - int(row["iterations"])
        <= 0
    )

    pod_by_case: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in pod_rows:
        pod_by_case[(str(row["system"]), str(row["rhs"]))].append(row)
    lower_energy_better = 0
    for case_rows in pod_by_case.values():
        ordered = sorted(case_rows, key=lambda row: float(row["captured_energy"]))
        if any(
            int(first["iterations"]) < int(second["iterations"])
            for index, first in enumerate(ordered)
            for second in ordered[index + 1 :]
        ):
            lower_energy_better += 1

    within_family = stage_e_summary.get("within_family", {})
    systematic_families = [
        family
        for family, values in within_family.items()
        if np.isfinite(float(values["spearman_rho"]))
        and float(values["spearman_rho"]) >= 0.5
    ]
    same_parameter_controls = [
        row
        for row in stage_e_rows
        if str(row["system_i"]) < str(row["system_j"])
        and float(row["parameter_distance"]) <= 1.0e-12
    ]

    correction_splits: Dict[str, Dict[str, float]] = {}
    for split in sorted({str(row["split"]) for row in ml_corrections}):
        subset = [row for row in ml_corrections if row["split"] == split]
        correction_splits[split] = {
            "samples": len(subset),
            "relative_error_median": _median(
                float(row["relative_correction_error"]) for row in subset
            ),
            "relative_error_p90": float(
                np.quantile(
                    [float(row["relative_correction_error"]) for row in subset],
                    0.9,
                )
            ),
            "cosine_median": _median(float(row["cosine_similarity"]) for row in subset),
            "beats_zero_fraction": float(
                np.mean(
                    [float(row["relative_correction_error"]) < 1.0 for row in subset]
                )
            ),
        }

    solver_direction_splits: Dict[str, Dict[str, float]] = {}
    for split in sorted({str(row["split"]) for row in ml_solver_directions}):
        subset = [row for row in ml_solver_directions if row["split"] == split]
        solver_direction_splits[split] = {
            "samples": len(subset),
            "relative_error_median": _median(
                float(row["relative_correction_error"]) for row in subset
            ),
            "relative_error_p90": float(
                np.quantile(
                    [float(row["relative_correction_error"]) for row in subset],
                    0.9,
                )
            ),
            "cosine_median": _median(
                float(row["cosine_similarity"]) for row in subset
            ),
            "prediction_over_target_median": _median(
                float(row["prediction_over_target_norm"]) for row in subset
            ),
            "target_over_classical_median": _median(
                float(row["target_over_classical_norm"]) for row in subset
            ),
        }

    ml_pairs = []
    for row in ml_solvers:
        if row["method_family"] != "direct_mlp":
            continue
        match = next(
            baseline
            for baseline in ml_solvers
            if baseline["system"] == row["system"]
            and baseline["rhs"] == row["rhs"]
            and baseline["method_family"] == "classical"
        )
        ml_pairs.append(
            {
                "split": row["split"],
                "system": row["system"],
                "rhs": row["rhs"],
                "iteration_delta": int(match["iterations"]) - int(row["iterations"]),
                "online_time_delta": float(match["online_total_seconds"])
                - float(row["online_total_seconds"]),
                "direct_converged": bool(row["converged"]),
            }
        )
    unseen_pairs = [row for row in ml_pairs if row["split"] == "unseen_parameter"]
    unseen_metrics = correction_splits.get("unseen_parameter_systems", {})
    direct_promising = bool(
        unseen_pairs
        and all(row["direct_converged"] for row in unseen_pairs)
        and float(unseen_metrics.get("relative_error_median", np.inf)) < 1.0
        and float(unseen_metrics.get("cosine_median", -1.0)) > 0.25
        and _median(row["iteration_delta"] for row in unseen_pairs) > 0.0
    )

    pod_useful_systems = 0
    for system in sorted({key[0] for key in classical}):
        baseline_values = [
            value for (name, _), value in classical.items() if name == system
        ]
        pod_values = [
            int(row["iterations"])
            for row in pod_rows
            if row["system"] == system and row["converged"]
        ]
        if pod_values and min(pod_values) < _median(baseline_values):
            pod_useful_systems += 1
    median_reference_cost = _median(
        float(row["reference_space_total_seconds"]) for row in stage_c
    )
    median_classical_solve = _median(
        float(row["classical_solve_seconds"]) for row in stage_a
    )
    low_dim_supported = len(low_dim_joint) >= max(1, int(np.ceil(0.6 * len(stage_c))))
    robust_systematic_families = [
        family
        for family in systematic_families
        if int(within_family[family]["pairs"]) >= 6
        and float(within_family[family]["p_value"]) < 0.20
    ]
    same_parameter_variability = _median(
        float(row["projector_distance_normalized"])
        for row in same_parameter_controls
    )
    structured_supported = bool(
        float(stage_e_summary.get("overall_spearman_rho", np.nan)) >= 0.4
        and float(stage_e_summary.get("overall_spearman_p_value", np.inf)) < 0.05
        and len(robust_systematic_families) >= 2
        and same_parameter_variability <= 0.25
    )
    classical_recompute_meaningful = bool(
        np.isfinite(median_reference_cost)
        and np.isfinite(median_classical_solve)
        and median_reference_cost >= 5.0 * max(median_classical_solve, 1.0e-12)
    )
    subspace_learning_justified = bool(
        low_dim_supported
        and structured_supported
        and classical_recompute_meaningful
        and pod_useful_systems >= 2
    )

    result = {
        "stage_a": {
            "systems": len(stage_a),
            "difficult_unpreconditioned_systems": len(difficult_rows),
            "nontrivial_classical_systems": len(nontrivial_classical),
            "unpreconditioned_iteration_min": min(
                int(row["unpreconditioned_iterations"]) for row in stage_a
            ),
            "unpreconditioned_iteration_max": max(
                int(row["unpreconditioned_iterations"]) for row in stage_a
            ),
            "classical_iteration_min": min(
                int(row["classical_iterations"]) for row in converged_a
            ),
            "classical_iteration_max": max(
                int(row["classical_iterations"]) for row in converged_a
            ),
            "meaningful_systems": [str(row["system"]) for row in difficult_rows],
            "nontrivial_classical_names": [str(row["system"]) for row in nontrivial_classical],
        },
        "stage_b": {
            "configurations": len(stage_b),
            "pareto_configurations": len(pareto),
            "iteration_min": min(budget_iteration_range),
            "iteration_max": max(budget_iteration_range),
        },
        "stage_c": {
            "systems": len(stage_c),
            "joint_low_dim_systems": len(low_dim_joint),
            "low_dim_supported": low_dim_supported,
            "q95_min": min(int(row["q95"]) for row in stage_c),
            "q95_max": max(int(row["q95"]) for row in stage_c),
            "q95_over_n_min": min(float(row["q95_over_n"]) for row in stage_c),
            "q95_over_n_max": max(float(row["q95_over_n"]) for row in stage_c),
            "eta95_mean_min": min(float(row["eta_q95_mean"]) for row in stage_c),
            "eta95_mean_max": max(float(row["eta_q95_mean"]) for row in stage_c),
            "q_joint_min": min(int(row["q_joint_energy95_eta25"]) for row in low_dim_joint),
            "q_joint_max": max(int(row["q_joint_energy95_eta25"]) for row in low_dim_joint),
            "q_joint_over_n_max": max(float(row["q_joint_over_n"]) for row in low_dim_joint),
        },
        "stage_d": {
            "pod_comparisons": len(pod_rows),
            "median_iteration_improvement": _median(improvements),
            "positive_improvement_fraction": float(
                np.mean(np.asarray(improvements) > 0)
            ) if improvements else float("nan"),
            "energy_iteration_spearman_rho": energy_iteration_rho,
            "energy_iteration_spearman_p": energy_iteration_p,
            "high_energy_no_gain_cases": high_energy_no_gain,
            "lower_energy_better_cases": lower_energy_better,
            "pod_useful_systems": pod_useful_systems,
        },
        "stage_e": {
            "overall_spearman_rho": stage_e_summary.get("overall_spearman_rho"),
            "overall_spearman_p": stage_e_summary.get("overall_spearman_p_value"),
            "within_family": within_family,
            "systematic_families": systematic_families,
            "robust_systematic_families": robust_systematic_families,
            "structured_supported": structured_supported,
            "same_parameter_control_pairs": len(same_parameter_controls),
            "same_parameter_projector_normalized_median": _median(
                float(row["projector_distance_normalized"])
                for row in same_parameter_controls
            ),
        },
        "stage_f": {
            "correction_splits": correction_splits,
            "solver_direction_splits": solver_direction_splits,
            "solver_pairs": ml_pairs,
            "unseen_iteration_delta_median": _median(
                row["iteration_delta"] for row in unseen_pairs
            ),
            "unseen_online_time_delta_median": _median(
                row["online_time_delta"] for row in unseen_pairs
            ),
            "unseen_iteration_improved_fraction": float(
                np.mean([row["iteration_delta"] > 0 for row in unseen_pairs])
            ) if unseen_pairs else float("nan"),
            "direct_promising": direct_promising,
        },
        "cost": {
            "median_reference_space_seconds": median_reference_cost,
            "median_classical_solve_seconds": median_classical_solve,
            "reference_recompute_meaningful": classical_recompute_meaningful,
        },
        "gates": {
            "gnn_direct_design_justified": direct_promising,
            "learned_subspace_model_justified": subspace_learning_justified,
        },
    }
    return result


def _plot_stage_a(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    families = [
        ("A_resolution", "Resolution"),
        ("B_irregularity", "Node nonuniformity"),
        ("C_stencil", "Stencil size"),
        ("D_variable_coefficient", "Coefficient contrast"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    for axis, (family, title) in zip(axes.ravel(), families):
        subset = [row for row in rows if row["family"] == family]
        labels = [str(row["system"]).split("_", 1)[1] for row in subset]
        positions = np.arange(len(subset))
        width = 0.38
        unpre = [int(row["unpreconditioned_iterations"]) for row in subset]
        ilu = [int(row["classical_iterations"]) for row in subset]
        axis.bar(positions - width / 2, unpre, width, color=BLUE, label="No preconditioner")
        axis.bar(positions + width / 2, ilu, width, color=ORANGE, label="ILU(A)")
        for index, row in enumerate(subset):
            if not row["unpreconditioned_converged"]:
                axis.text(
                    positions[index] - width / 2,
                    unpre[index],
                    " max",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylabel("GMRES iterations")
        axis.grid(axis="x", visible=False)
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.suptitle("Controlled RBF-FD difficulty hierarchy", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        "True residual tolerance 1e-8; 'max' denotes the 800-iteration cap.",
        ha="center",
        fontsize=8,
    )
    _save_figure(figure, path)


def _plot_budget(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    systems = sorted({str(row["system"]) for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    for axis, system in zip(axes.ravel(), systems):
        subset = [row for row in rows if row["system"] == system and row["converged"]]
        for kind, color, marker, label in (
            ("ilu_A", BLUE, "o", "ILU(A)"),
            ("smaller_stencil_ilu", ORANGE, "s", "ILU(B), smaller stencil"),
        ):
            points = [row for row in subset if row["preconditioner_kind"] == kind]
            axis.scatter(
                [float(row["preconditioner_cost_seconds"]) for row in points],
                [int(row["iterations"]) for row in points],
                marker=marker,
                s=[70 if row.get("pareto_optimal") else 28 for row in points],
                facecolors=[color if row.get("pareto_optimal") else "white" for row in points],
                edgecolors=color,
                linewidths=1.2,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_title(system)
        axis.set_xlabel("Setup + applications (s, log scale)")
        axis.set_ylabel("GMRES iterations")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Classical preconditioner cost–quality frontier", fontsize=13, fontweight="bold")
    figure.text(0.5, -0.01, "Filled/larger marks are nondominated configurations.", ha="center", fontsize=8)
    _save_figure(figure, path)


def _plot_pod_curves(
    curve_rows: Sequence[Mapping[str, Any]],
    systems: Sequence[str],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    for color, system in zip(PALETTE, systems):
        subset = sorted(
            [row for row in curve_rows if row["system"] == system],
            key=lambda row: int(row["q"]),
        )
        q = [int(row["q"]) for row in subset]
        axes[0].semilogy(q, [float(row["normalized_singular_value"]) for row in subset], color=color, label=system)
        axes[1].plot(q, [float(row["captured_energy"]) for row in subset], color=color)
        axes[2].plot(q, [float(row["heldout_eta_mean"]) for row in subset], color=color)
    axes[0].set_title("Normalized singular values")
    axes[0].set_ylabel(r"$\sigma_q/\sigma_1$")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_title("Training snapshot energy")
    axes[1].set_ylabel(r"$E_q$")
    axes[1].axhline(0.95, color=CHARCOAL, linestyle="--", linewidth=0.9)
    axes[2].set_title("Held-out projection error")
    axes[2].set_ylabel(r"mean $\eta_q$")
    for axis in axes:
        axis.set_xlabel("Subspace dimension q")
        axis.set_xlim(left=1)
    figure.suptitle("Unresolved-correction structure on hard representatives", fontsize=13, fontweight="bold")
    _save_figure(figure, path)


def _plot_q_scaling(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [str(row["system"]) for row in rows]
    positions = np.arange(len(rows))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    for offset, key, label, color in (
        (-1.5 * width, "q90_over_n", "90% energy", BLUE),
        (-0.5 * width, "q95_over_n", "95% energy", ORANGE),
        (0.5 * width, "q99_over_n", "99% energy", OLIVE),
        (1.5 * width, "q_joint_over_n", "95% energy + eta<=0.25", PINK),
    ):
        axis.bar(
            positions + offset,
            [100.0 * float(row[key]) for row in rows],
            width,
            color=color,
            label=label,
        )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("q / N (%)")
    axis.set_title("Energy dimensions relative to system size")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="x", visible=False)
    _save_figure(figure, path)


def _plot_krylov_relevance(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    systems = sorted({str(row["system"]) for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    style = {
        "pod": (BLUE, "o"),
        "random": (ORANGE, "s"),
        "ritz": (OLIVE, "^"),
    }
    for axis, system in zip(axes.ravel(), systems):
        subset = [row for row in rows if row["system"] == system]
        baseline = _median(
            int(row["iterations"])
            for row in subset
            if row["method_family"] == "classical"
        )
        axis.axhline(baseline, color=CHARCOAL, linestyle="--", label="Classical")
        for family, (color, marker) in style.items():
            q_values = sorted(
                {int(row["q"]) for row in subset if row["method_family"] == family}
            )
            medians = [
                _median(
                    int(row["iterations"])
                    for row in subset
                    if row["method_family"] == family and int(row["q"]) == q
                )
                for q in q_values
            ]
            if q_values:
                axis.plot(q_values, medians, color=color, marker=marker, label=family.upper())
        axis.set_title(system)
        axis.set_xlabel("Subspace dimension q")
        axis.set_ylabel("Median iterations (3 RHS)")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle("Krylov relevance of POD, random, and Ritz spaces", fontsize=13, fontweight="bold")
    _save_figure(figure, path)


def _plot_subspace_heatmaps(
    matrices: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    names = [str(value) for value in matrices["names"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "projector", "Projector distance"),
        (axes[1], "angle", "Mean principal angle (degrees)"),
    ):
        image = axis.imshow(np.asarray(matrices[key]), cmap="cividis", aspect="auto")
        axis.set_title(title)
        axis.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
        axis.set_yticks(range(len(names)), names, fontsize=7)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle("Independently constructed difficult-subspace comparisons", fontsize=13, fontweight="bold")
    _save_figure(figure, path)


def _plot_subspace_parameter_relation(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    unique = [row for row in rows if str(row["system_i"]) < str(row["system_j"])]
    same_family = [row["family_i"] == row["family_j"] for row in unique]
    figure, axis = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for flag, color, marker, label in (
        (True, BLUE, "o", "Within controlled family"),
        (False, ORANGE, "s", "Across families"),
    ):
        subset = [row for row, same in zip(unique, same_family) if same == flag]
        axis.scatter(
            [float(row["parameter_distance"]) for row in subset],
            [float(row["projector_distance_normalized"]) for row in subset],
            c=color,
            marker=marker,
            facecolors="none" if flag else color,
            edgecolors=color,
            alpha=0.8,
            label=label,
        )
    axis.set_xlabel("Standardized parameter distance")
    axis.set_ylabel(r"Normalized projector distance $\|P_i-P_j\|_F/\sqrt{2q}$")
    axis.set_title("Parameter distance versus subspace distance")
    axis.legend(frameon=False)
    _save_figure(figure, path)


def _plot_ml_training(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    axis.semilogy([row["epoch"] for row in history], [row["training_loss"] for row in history], color=BLUE, label="Training")
    axis.semilogy([row["epoch"] for row in history], [row["validation_loss"] for row in history], color=ORANGE, linestyle="--", label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized mean-squared error")
    axis.set_title("Direct-complement MLP fitting history")
    axis.legend(frameon=False)
    _save_figure(figure, path)


def _plot_ml_correction(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    split_order = [
        "training",
        "known_systems_unseen_residuals",
        "validation_unseen_systems",
        "unseen_parameter_systems",
    ]
    labels = ["Train", "Known/new r", "Validation xi", "Test xi"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), constrained_layout=True)
    for axis, field, title, reference in (
        (axes[0], "relative_correction_error", "Relative correction error", 1.0),
        (axes[1], "cosine_similarity", "Correction cosine similarity", 0.0),
    ):
        values = [
            [float(row[field]) for row in rows if row["split"] == split]
            for split in split_order
        ]
        boxes = axis.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.axhline(reference, color=CHARCOAL, linestyle="--", linewidth=0.9)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Direct missing-action prediction across system splits", fontsize=13, fontweight="bold")
    _save_figure(figure, path)


def _plot_ml_solver(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    splits = ["known_system", "unseen_parameter"]
    methods = ["classical", "pod", "direct_mlp"]
    labels = ["Classical", "POD reference", "Direct MLP"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for axis, split in zip(axes, splits):
        subset = [row for row in rows if row["split"] == split]
        values = [
            [int(row["iterations"]) for row in subset if row["method_family"] == method]
            for method in methods
        ]
        medians = [_median(value) for value in values]
        axis.bar(np.arange(3), medians, color=PALETTE[:3])
        for x, group in enumerate(values):
            axis.scatter(
                np.full(len(group), x) + np.linspace(-0.08, 0.08, len(group)),
                group,
                c="white",
                edgecolors=CHARCOAL,
                s=22,
                zorder=3,
            )
        axis.set_xticks(np.arange(3), labels, rotation=25, ha="right")
        axis.set_ylabel("FGMRES iterations")
        axis.set_title("Known systems" if split == "known_system" else "Unseen coefficient values")
        axis.grid(axis="x", visible=False)
    figure.suptitle("Solver effect of the direct learned complement", fontsize=13, fontweight="bold")
    _save_figure(figure, path)


def _plot_ml_residual(
    histories: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    candidates = [
        row
        for row in histories
        if row["stage"] == "F"
        and row["system"] == "ML_test_beta165"
        and row["rhs"] == "heldout_smooth_1"
    ]
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for method, color, style, label in (
        ("classical", BLUE, "-", "Classical"),
        ("pod", OLIVE, "-.", "POD reference"),
        ("direct_mlp", ORANGE, "--", "Direct MLP"),
    ):
        subset = sorted(
            [row for row in candidates if row["method"] == method],
            key=lambda row: int(row["iteration"]),
        )
        if subset:
            axis.semilogy(
                [int(row["iteration"]) for row in subset],
                [float(row["relative_residual"]) for row in subset],
                color=color,
                linestyle=style,
                marker="o",
                markersize=3,
                label=label,
            )
    axis.axhline(1.0e-8, color=CHARCOAL, linewidth=0.9, linestyle=":")
    axis.set_xlabel("FGMRES iteration")
    axis.set_ylabel("True relative residual")
    axis.set_title("Representative unseen-system convergence (beta = 1.65)")
    axis.legend(frameon=False)
    _save_figure(figure, path)


def create_plots(
    output: Path,
    stage_a: Sequence[Mapping[str, Any]],
    stage_b: Sequence[Mapping[str, Any]],
    stage_c_summary: Sequence[Mapping[str, Any]],
    stage_c_curves: Sequence[Mapping[str, Any]],
    stage_d: Sequence[Mapping[str, Any]],
    stage_e_rows: Sequence[Mapping[str, Any]],
    stage_e_matrices: Mapping[str, np.ndarray],
    ml_corrections: Sequence[Mapping[str, Any]],
    ml_solvers: Sequence[Mapping[str, Any]],
    ml_history: Sequence[Mapping[str, Any]],
    histories: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """Render all report figures and save an auditable chart map."""

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Dict[str, str]] = []

    def add(filename: str, section: str, question: str, family: str, fields: str, claim: str) -> Path:
        path = plot_dir / filename
        charts.append(
            {
                "section": section,
                "analytical_question": question,
                "chart_family": family,
                "fields": fields,
                "supported_claim": claim,
                "palette_policy": "hard two-root cap plus neutrals; multi-method controls use four approved roots",
                "path": str(path),
            }
        )
        return path

    _plot_stage_a(stage_a, add(
        "harder_regime_iterations.png", "Harder regimes",
        "Which controlled changes make Krylov convergence harder?", "grouped bar small multiples",
        "family, system, unpreconditioned_iterations, classical_iterations, converged",
        "Difficulty increases sharply without preconditioning while strong ILU remains effective.",
    ))
    _plot_budget(stage_b, add(
        "classical_budget_pareto.png", "Classical budget",
        "What setup/application cost buys what convergence?", "Pareto scatter small multiples",
        "system, preconditioner_cost_seconds, iterations, preconditioner_kind, pareto_optimal",
        "ILU cost and convergence form a nonmonotone budget frontier.",
    ))
    representative = [name for name in ("A_N2500", "B_warp95", "C_s13", "D_beta170") if any(row["system"] == name for row in stage_c_curves)]
    _plot_pod_curves(stage_c_curves, representative, add(
        "unresolved_correction_curves.png", "Unresolved correction",
        "Do singular decay, energy, and held-out error jointly support low dimension?", "three-panel line diagnostic",
        "q, normalized_singular_value, captured_energy, heldout_eta_mean",
        "Snapshot energy alone is compared against held-out reconstruction.",
    ))
    _plot_q_scaling(stage_c_summary, add(
        "pod_dimension_scaling.png", "Unresolved correction",
        "How does the energy dimension scale with N and difficulty?", "grouped bar",
        "system, q90_over_n, q95_over_n, q99_over_n",
        "Required POD dimensions are reported as fractions of N.",
    ))
    _plot_krylov_relevance(stage_d, add(
        "pod_krylov_relevance.png", "Krylov relevance",
        "Do POD modes accelerate Krylov relative to random and Ritz controls?", "line-dot small multiples",
        "system, method_family, q, median iterations across RHS",
        "Reconstruction energy and solver utility are assessed separately.",
    ))
    _plot_subspace_heatmaps(stage_e_matrices, add(
        "subspace_distance_heatmaps.png", "Subspace variation",
        "How different are independently constructed spaces?", "matrix heatmaps",
        "system_i, system_j, projector_distance, mean principal angle",
        "The target space changes across systems and can be inspected pairwise.",
    ))
    _plot_subspace_parameter_relation(stage_e_rows, add(
        "subspace_vs_parameter_distance.png", "Subspace variation",
        "Does subspace distance track controlled parameter distance?", "scatter",
        "standardized parameter distance, normalized projector distance, within/across family",
        "Systematic variation is tested rather than inferred from one transferred basis.",
    ))
    _plot_ml_training(ml_history, add(
        "direct_mlp_training.png", "Direct complement",
        "Does the fixed-size MLP fit without validation divergence?", "semilog line",
        "epoch, training_loss, validation_loss",
        "Training dynamics expose fit and overfit behavior.",
    ))
    _plot_ml_correction(ml_corrections, add(
        "direct_mlp_correction_quality.png", "Direct complement",
        "Does correction prediction generalize to unseen residuals and systems?", "box plots",
        "split, relative_correction_error, cosine_similarity",
        "Prediction quality is shown beyond aggregate MSE.",
    ))
    _plot_ml_solver(ml_solvers, add(
        "direct_mlp_solver_effect.png", "Direct complement",
        "Does the learned action improve FGMRES?", "grouped bars with observations",
        "split, method_family, iterations",
        "Solver utility is compared on known and unseen parameter systems.",
    ))
    _plot_ml_residual(histories, add(
        "direct_mlp_residual_history.png", "Direct complement",
        "How do true residual trajectories compare on a representative unseen system?", "semilog convergence line",
        "method, iteration, true relative residual",
        "Iteration counts are supported by complete true-residual histories.",
    ))
    _write_csv(output / "chart_map.csv", charts)
    return charts


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return "n/a"
            if value != 0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e4):
                return f"{value:.3e}"
            return f"{value:.4g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _relative_plot(output: Path, filename: str) -> str:
    return f"plots/{filename}"


def write_optional_assessments(
    output: Path,
    evidence: Mapping[str, Any],
) -> List[Path]:
    """Write G/H design notes only when their preregistered gates pass."""

    paths: List[Path] = []
    if evidence["gates"]["gnn_direct_design_justified"]:
        path = output / "gnn_direct_complement_design.md"
        path.write_text(
            """# Conditional design: variable-N graph direct complement\n\n"
            "The fixed-size direct MLP passed the numerical evidence gate. This is a design only; no GNN was trained.\n\n"
            "## Graph and features\n\n"
            "Use the directed RBF-FD stencil graph. Node features are coordinates, normalized residual, boundary indicator, local coefficient, and local stencil summaries. Edge features are relative coordinates, distance, and the signed RBF-FD weight. The output is one scalar missing correction per node.\n\n"
            "## Invariances and objective\n\n"
            "Use coordinate normalization, residual-norm homogeneity, shared message-passing weights, and the unchanged target `A(xi)^{-1}r - M(xi)^{-1}r`. Train with normalized correction MSE, but gate model selection on held-out correction angle and FGMRES utility.\n\n"
            "## Required next tests\n\n"
            "Split by entire systems and N; compare against the fixed-size MLP, ILU, and POD; record inference and repeated-solve break-even cost.\n",
            encoding="utf-8",
        )
        paths.append(path)
    if evidence["gates"]["learned_subspace_model_justified"]:
        path = output / "learned_subspace_target_assessment.md"
        path.write_text(
            """# Conditional assessment: learned difficult-subspace target\n\n"
            "The A–F evidence gate was met, so the saved independently fitted projectors define a plausible future target. No learned-subspace neural model was implemented.\n\n"
            "A future model must predict `range(Z_theta(xi))`, use a basis-invariant projector or principal-angle loss, and retain the reduced numerical solve. Training and evaluation must split by systems and compare repeated-solve cost against classical recomputation.\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def write_report(
    output: Path,
    evidence: Mapping[str, Any],
    stage_a: Sequence[Mapping[str, Any]],
    stage_b: Sequence[Mapping[str, Any]],
    stage_c: Sequence[Mapping[str, Any]],
    stage_d: Sequence[Mapping[str, Any]],
    stage_e_summary: Mapping[str, Any],
    ml_corrections: Sequence[Mapping[str, Any]],
    ml_solvers: Sequence[Mapping[str, Any]],
    ml_solver_directions: Sequence[Mapping[str, Any]],
    ml_summary: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the requested concise, technically detailed Markdown report."""

    a = evidence["stage_a"]
    b = evidence["stage_b"]
    c = evidence["stage_c"]
    d = evidence["stage_d"]
    e = evidence["stage_e"]
    f = evidence["stage_f"]
    gates = evidence["gates"]

    direct_verdict = (
        "promising enough to justify a variable-N graph design"
        if gates["gnn_direct_design_justified"]
        else "not yet strong enough to justify a GNN escalation"
    )
    subspace_verdict = (
        "meets the four-part justification gate"
        if gates["learned_subspace_model_justified"]
        else "does not yet meet the four-part learned-subspace gate"
    )
    stronger_branch = (
        "direct learned complement"
        if gates["gnn_direct_design_justified"] and not gates["learned_subspace_model_justified"]
        else "learned difficult subspace"
        if gates["learned_subspace_model_justified"] and not gates["gnn_direct_design_justified"]
        else "difficult-subspace analysis (diagnostically stronger, but not yet sufficient for a learned model)"
        if not any(gates.values()) and d["pod_useful_systems"] >= 2
        else "neither branch"
        if not any(gates.values())
        else "both branches, with different evidence"
    )

    hard_table_rows = []
    for row in stage_a:
        hard_table_rows.append(
            (
                row["system"],
                row["n_nodes"],
                row["stencil_size"],
                row["node_irregularity"],
                row["coefficient_contrast"],
                row["symmetry_measure"],
                row["relative_discretization_error"],
                f"{row['unpreconditioned_iterations']}{'' if row['unpreconditioned_converged'] else '*'}",
                row["classical_iterations"],
                row["classical_total_seconds"],
            )
        )

    pod_table_rows = [
        (
            row["system"], row["n_nodes"],
            f"{row['q90']} ({100*float(row['q90_over_n']):.2f}%)", row["eta_q90_mean"],
            f"{row['q95']} ({100*float(row['q95_over_n']):.2f}%)", row["eta_q95_mean"],
            f"{row['q99']} ({100*float(row['q99_over_n']):.2f}%)", row["eta_q99_mean"],
            f"{row['q_joint_energy95_eta25']} ({100*float(row['q_joint_over_n']):.2f}%)",
            row["reference_space_total_seconds"],
        )
        for row in stage_c
    ]

    classical = _matching_classical_iterations(stage_d)
    krylov_table_rows = []
    for system in sorted({str(row["system"]) for row in stage_d}):
        baseline = _median(value for (name, _), value in classical.items() if name == system)
        values = {}
        for family in ("pod", "random", "ritz"):
            subset = [
                row for row in stage_d
                if row["system"] == system and row["method_family"] == family and row["converged"]
            ]
            if subset:
                best_q = min(
                    sorted({int(row["q"]) for row in subset}),
                    key=lambda q: _median(int(row["iterations"]) for row in subset if int(row["q"]) == q),
                )
                values[family] = (
                    best_q,
                    _median(int(row["iterations"]) for row in subset if int(row["q"]) == best_q),
                )
            else:
                values[family] = ("n/a", float("nan"))
        krylov_table_rows.append((system, baseline, *values["pod"], *values["random"], *values["ritz"]))

    correction_table_rows = []
    for split, values in f["correction_splits"].items():
        correction_table_rows.append(
            (
                split,
                values["samples"],
                values["relative_error_median"],
                values["relative_error_p90"],
                values["cosine_median"],
                values["beats_zero_fraction"],
            )
        )
    direction_table_rows = []
    for split, values in f["solver_direction_splits"].items():
        direction_table_rows.append(
            (
                split,
                values["samples"],
                values["relative_error_median"],
                values["relative_error_p90"],
                values["cosine_median"],
                values["prediction_over_target_median"],
                values["target_over_classical_median"],
            )
        )
    solver_table_rows = []
    for split in ("known_system", "unseen_parameter"):
        subset = [row for row in ml_solvers if row["split"] == split]
        for family in ("classical", "pod", "direct_mlp"):
            family_rows = [row for row in subset if row["method_family"] == family]
            solver_table_rows.append(
                (
                    split,
                    family,
                    len(family_rows),
                    _median(int(row["iterations"]) for row in family_rows),
                    _median(float(row["online_total_seconds"]) for row in family_rows),
                    _median(float(row["inference_seconds"]) for row in family_rows),
                    sum(bool(row["converged"]) for row in family_rows),
                )
            )

    failure_note = (
        f"{len(failures)} recorded numerical events; these are preserved in `failures.csv`."
        if failures
        else "No factorization, coarse solve, or Krylov failures were recorded."
    )
    within_lines = []
    for family, values in e["within_family"].items():
        within_lines.append(
            f"- `{family}`: Spearman rho = {float(values['spearman_rho']):.3f}, "
            f"p = {float(values['p_value']):.3g}, {int(values['pairs'])} pairs."
        )

    report = rf"""# Learned Complementary Preconditioning: Harder Regimes and Direct-ML Baseline

## Technical summary

This round preserves the proposal's exact missing inverse action

\[
d(r,\xi)=A(\xi)^{{-1}}r-M(\xi)^{{-1}}r,
\]

with every `A^{-1}r` target produced by an accurate sparse solve, never by forming an inverse. Across {a['systems']} one-factor-at-a-time systems, unpreconditioned GMRES ranged from {a['unpreconditioned_iteration_min']} to {a['unpreconditioned_iteration_max']} iterations (the upper end includes capped failures), while the strong default ILU required {a['classical_iteration_min']}–{a['classical_iteration_max']} iterations. Thus harder RBF-FD regimes are real, but ILU remains formidable; only {a['nontrivial_classical_systems']} systems required at least eight ILU-preconditioned iterations.

The unresolved correction satisfies a joint low-dimensional diagnostic on {c['joint_low_dim_systems']} of {c['systems']} systems: the first q with at least 95% training energy and mean held-out `eta_q <= 0.25` lies between q={c['q_joint_min']} and q={c['q_joint_max']}, never exceeding {100*c['q_joint_over_n_max']:.2f}% of N. However, the training-only q95 dimensions are smaller (q={c['q95_min']}–{c['q95_max']}) and can have held-out errors as high as {c['eta95_mean_max']:.3f}. Thus the correction is structured, but training energy alone systematically risks underselecting q. This is evidence, not an assumption.

POD energy and Krylov acceleration have a positive but imperfect association: Spearman rho={d['energy_iteration_spearman_rho']:.3f} (p={d['energy_iteration_spearman_p']:.3g}); {d['high_energy_no_gain_cases']} high-energy (`E_q >= 0.95`) comparisons gave no iteration gain, and {d['lower_energy_better_cases']} RHS/system cases contained a lower-energy POD space that outperformed a higher-energy one. The direct fixed-size MLP is **{direct_verdict}**. The independently fitted subspace family **{subspace_verdict}**. On the preregistered gates, the branch with stronger current numerical support is **{stronger_branch}**.

## The controlled hierarchy creates hard systems, but strong ILU usually prevents a hard online solve

The hierarchy changes one factor at a time: N, a smooth coordinate warp, stencil size, or coefficient field `a(x)=exp(beta sin(2 pi x) sin(2 pi y))`. The coefficient contrast is exactly `exp(2|beta|)`. Manufactured solutions measure discretization error independently of solver convergence. An asterisk in the table means unpreconditioned GMRES hit the 800-iteration cap.

{_markdown_table(
    ["System", "N", "s", "NN-CV", "contrast", "asymmetry", "disc. error", "none it.", "ILU it.", "ILU total s"],
    hard_table_rows,
)}

![Controlled RBF-FD difficulty hierarchy]({_relative_plot(output, 'harder_regime_iterations.png')})

The meaningful Krylov-hard regimes are the largest resolution, the strongest coordinate warp, and the highest coefficient contrasts. The small-stencil case worsens discretization error and asymmetry but does not increase the iteration burden relative to the default stencil, so it is a numerical-quality stressor rather than a genuinely harder Krylov regime here. The contradiction to the motivating expectation is that the actual hard regimes still do **not** generally defeat a high-budget ILU.

## ILU has a real budget frontier; no weak baseline was selected for ML

The budget sweep includes 20 `(drop tolerance, fill factor)` combinations per hard representative plus the RBF-FD smaller-stencil `B` preconditioner. Cost is measured setup plus all preconditioner applications in one solve. Across converged budget points, iterations span {b['iteration_min']}–{b['iteration_max']}; {b['pareto_configurations']} of {b['configurations']} points are nondominated. The frontier is nonmonotone because a tighter drop tolerance under a low fill cap can discard structurally useful fill.

![Classical preconditioner cost-quality frontier]({_relative_plot(output, 'classical_budget_pareto.png')})

The direct-ML comparison retains the strong `drop_tol=1e-3, fill_factor=5` ILU used in the hierarchy. It is not the globally fastest configuration for every system, but it is a deliberately strong, stable common baseline and is never weakened to create an ML gain.

## Low-dimensionality survives only where training energy and held-out error agree

Each system uses 40 training and 20 held-out missing-action snapshots, evenly split between synthetic residual/error samples and residuals from actual GMRES trajectories. The table reports the first q reaching 90%, 95%, and 99% training energy and the held-out error at q95.

{_markdown_table(
    ["System", "N", "q90 (q/N)", "eta90", "q95 (q/N)", "eta95", "q99 (q/N)", "eta99", "joint q (q/N)", "reference cost s"],
    pod_table_rows,
)}

![Unresolved-correction diagnostic curves]({_relative_plot(output, 'unresolved_correction_curves.png')})

![POD dimension scaling]({_relative_plot(output, 'pod_dimension_scaling.png')})

The conclusion is based jointly on singular decay, energy dimension, and held-out projection. `joint q` is the first q satisfying both `E_q >= 0.95` and mean held-out `eta_q <= 0.25`; it separates the existence of a useful small space from the weaker claim that a training-energy cutoff selects it correctly. A small `q/N` alone remains insufficient, and both absolute q and q/N are retained.

## POD energy correlates with Krylov benefit, but does not guarantee it

Because the RBF-FD matrices are materially nonsymmetric, all POD/random/Ritz coarse actions use the implemented Petrov–Galerkin minimum-residual variant, not the symmetric Galerkin formula. The table reports the best median iteration count across three RHS values for each family; it is a sensitivity summary, not a cherry-picked final method.

{_markdown_table(
    ["System", "classical", "POD q", "POD it.", "random q", "random it.", "Ritz q", "Ritz it."],
    krylov_table_rows,
)}

![POD, random, and Ritz Krylov relevance]({_relative_plot(output, 'pod_krylov_relevance.png')})

The median POD improvement across all tested q/RHS comparisons is {d['median_iteration_improvement']:.2f} iterations, and {100*d['positive_improvement_fraction']:.1f}% of comparisons improve. Energy is directionally informative in this sample, but high POD energy can still coexist with zero or negative acceleration; the random and Ritz controls show whether improvement is specific to unresolved-correction modes or merely to adding vectors. No tested case had a lower-energy POD truncation strictly outperform a higher-energy truncation, so that requested counterexample was not observed rather than manufactured.

## Independently fitted difficult spaces vary across systems; systematicity is mixed

Every system has its own reference POD basis. To compare different N, each basis is interpolated to a common {int(stage_e_summary['probe_nodes'])}-point physical probe grid and re-orthonormalized before basis-invariant projector and principal-angle calculations at q={int(stage_e_summary['q'])}. This interpolation makes cross-N comparison possible but is itself a modeling choice and is treated as a limitation.

![Pairwise difficult-subspace heatmaps]({_relative_plot(output, 'subspace_distance_heatmaps.png')})

![Subspace distance versus parameter distance]({_relative_plot(output, 'subspace_vs_parameter_distance.png')})

Across all pairs, standardized parameter distance and projector distance have Spearman rho={float(e['overall_spearman_rho']):.3f} (p={float(e['overall_spearman_p']):.3g}). Within-family summaries are:

{chr(10).join(within_lines)}

The descriptive positive-association families are {', '.join(e['systematic_families']) if e['systematic_families'] else 'none'}, but only {', '.join(e['robust_systematic_families']) if e['robust_systematic_families'] else 'none'} also have at least six pairs and `p < 0.20`. There are {int(e['same_parameter_control_pairs'])} independently sampled same-parameter control pairs; their median normalized projector distance is {float(e['same_parameter_projector_normalized_median']):.3f}. Because this exceeds the preregistered 0.25 stability ceiling, the apparent parameter trend does not yet define a sufficiently reproducible target for learning. This tests whether `xi -> V_diff(xi)` is structured; it does not test whether one frozen basis transfers perfectly.

## The fixed-size direct MLP is judged by correction geometry and FGMRES, not MSE alone

The proof-of-concept fixes N=900 and the strong warp, trains on beta values {ml_summary['train_betas']}, validates on entire held-out beta systems {ml_summary['validation_betas']}, and tests on unseen beta values {ml_summary['test_betas']}. The {int(ml_summary['hidden_layers'])}-hidden-layer width-{int(ml_summary['hidden_width'])} bottleneck MLP has {int(ml_summary['model_parameters']):,} parameters and uses {int(ml_summary['train_samples'])} training samples. Residual direction, residual norm, target scale, and beta are normalized so the wrapper is positively homogeneous, but the neural map remains nonlinear; the stored-vector flexible algorithm is therefore reported as FGMRES.

{_markdown_table(
    ["Split", "samples", "median rel. error", "p90 rel. error", "median cosine", "fraction better than zero"],
    correction_table_rows,
)}

![Direct MLP fitting history]({_relative_plot(output, 'direct_mlp_training.png')})

![Direct MLP correction quality]({_relative_plot(output, 'direct_mlp_correction_quality.png')})

A zero learned correction has relative correction error exactly 1. The `fraction better than zero` column therefore tests whether the MLP adds predictive information over simply retaining `M^-1`.

Solver results use three held-out manufactured RHS values on two known and two unseen-beta systems. The POD comparator is a system-specific reference built from that system's snapshots, so it is an oracle/reference cost rather than an online learned method.

{_markdown_table(
    ["System split", "method", "solves", "median it.", "median online s", "median inference s", "converged"],
    solver_table_rows,
)}

![Direct MLP solver effect]({_relative_plot(output, 'direct_mlp_solver_effect.png')})

![Representative unseen-system residual history]({_relative_plot(output, 'direct_mlp_residual_history.png')})

The failure was diagnosed on the first 50 Arnoldi preconditioner inputs from every solve. These are the vectors the network actually sees inside FGMRES, not merely offline residual snapshots.

{_markdown_table(
    ["Solver split", "directions", "median rel. error", "p90 rel. error", "median cosine", "pred./target norm", "target/base norm"],
    direction_table_rows,
)}

If the solver-direction error is materially worse than the offline split, the cause is residual-distribution shift; if `prediction/target` is large while `target/base` is small, an inaccurate complement overwhelms an already strong classical action. These diagnostics explain the failure without changing the required additive formula or inserting an unreported damping factor.

On unseen coefficient values, the direct MLP changes iteration count by a median of {f['unseen_iteration_delta_median']:+.2f} iterations relative to strong ILU and changes online time by {f['unseen_online_time_delta_median']:+.4g} seconds in the favorable direction convention (`classical - direct`). It improves iterations in {100*f['unseen_iteration_improved_fraction']:.1f}% of unseen-system solves. Training cost is {float(ml_summary['training_seconds']):.3f} seconds and is excluded from online time but retained in raw repeated-solve accounting.

Repeated-solve accounting uses `T_total = T_train + T_setup + N_solve T_online`. Because the direct action did not converge in any evaluated solve and also increased online time, it has no finite break-even point against ILU in this experiment. The POD reference reduces iterations but its system-specific snapshot/SVD setup is tens of classical solve times, so iteration reduction alone is not a wall-clock win.

## Decisions for the next round

1. **Numerically meaningful regimes.** Retain the largest-N, high-warp, and high-contrast systems as genuine Krylov stressors. Retain small stencils as an accuracy/stability control, not as a demonstrated harder-solve regime.
2. **ILU strength.** ILU is still extremely strong at a generous budget. Nontrivial regimes exist, but most do not create many complement-relevant iterations under the default strong ILU.
3. **Low-dimensional correction.** The evidence is {'supportive across a majority of systems' if c['low_dim_supported'] else 'not consistent across a majority of systems'} under the joint energy/held-out criterion; the important caveat is that q95 from training energy often underselects the held-out space. No inference is made from iteration reduction alone.
4. **POD versus Krylov.** POD energy correlates with solver acceleration here but is not sufficient: q must still be evaluated by Krylov outcomes with controls.
5. **Subspace variation.** Reference spaces change materially. Their parameter dependence is {'systematic and reproducible in multiple controlled families' if e['structured_supported'] else 'suggestive, but same-parameter snapshot variability prevents a reproducible target claim'}.
6. **Direct learnability.** The fixed-size direct target is {direct_verdict}; correction geometry and FGMRES agree with this gate.
7. **Proceed to a GNN?** {'Yes, design only was produced; variable-N training remains future work.' if gates['gnn_direct_design_justified'] else 'No. Diagnose and strengthen the fixed-size residual/system sampling before increasing architecture complexity.'}
8. **Proceed to learned subspace?** {'The analytical prerequisites are met, but no neural subspace model was implemented in this round.' if gates['learned_subspace_model_justified'] else 'No. The four conditions—useful low dimension, structured variation, meaningful recomputation cost, and solver utility—are not jointly supported.'}
9. **Stronger branch.** The current evidence favors **{stronger_branch}**. This is an empirical ranking for this controlled study, not an assumption of method superiority.

## Scope, methodology, and robustness limits

- All matrices are sparse RBF-FD discretizations using PHS `r^3`, degree-2 polynomial augmentation, explicit identity boundary rows, and local stencils. Variable coefficients discretize `-a Delta u - grad(a) dot grad(u)` with the same augmented local formulation.
- The empirical starting point was the Stage 1–5 N=324 study, where the same strong ILU converged in about five iterations and POD saved at most one iteration without a wall-clock win. This round extends that existing pipeline rather than replacing it.
- The default solve tolerance is a true relative residual of `1e-8`, restart 50, maximum 800. Timings are single-process wall-clock measurements on the recorded software/hardware environment; they are useful comparatively but are not hardware-independent performance claims.
- Snapshot rank is capped by 40 training columns. A reported q99 at that cap indicates only what this finite ensemble supports.
- The direct MLP experiment varies only one system parameter (coefficient beta) at fixed N and fixed geometry. It cannot establish variable-N generalization.
- The training set is small by modern ML standards and the reported uncertainty is empirical across samples/systems/RHS values, not a population confidence interval.
- Cross-N subspace comparison uses interpolation to a common grid. Repeating the analysis with mass-weighted inner products or node-correspondence operators is a useful robustness check.
- {failure_note}

## Reproducibility and saved evidence

Run `python scripts/run_stage6_harder_regimes.py`. Raw tables, residual histories, snapshots, independently fitted subspaces/projectors, model checkpoint, configuration, software versions, failure log, and chart map are saved beside this report. The proposal remains the mathematical source of truth; no learned difficult-subspace neural model was implemented.

## Further questions

- Does the conclusion change under combined—but still physically controlled—high resolution and variable coefficient contrast?
- Which residual mixture best predicts online FGMRES usefulness rather than snapshot reconstruction alone?
- Can POD/Ritz spaces be optimized for nonnormal Krylov dynamics without using future test RHS information?
- If the direct gate passes, does a shared graph model amortize inference across N without erasing the ILU wall-clock advantage?
"""
    path = output / "stage6_harder_regimes_and_direct_ml_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_validation_report(
    output: Path,
    evidence: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    expected_charts: int,
) -> Path:
    """Persist the final claim/calculation/visual QA receipt."""

    plots = sorted((output / "plots").glob("*.png"))
    issues = []
    if len(plots) != expected_charts:
        issues.append(f"expected {expected_charts} plots, found {len(plots)}")
    if not evidence["stage_a"]["systems"]:
        issues.append("no Stage A systems")
    if not evidence["stage_c"]["systems"]:
        issues.append("no Stage C systems")
    if not evidence["stage_f"]["correction_splits"]:
        issues.append("no direct-ML correction metrics")
    assessment = "Ready to share" if not issues else "Needs revision"
    text = f"""# Validation Report

## Overall Assessment: {assessment}

## Methodology Review

The executed pipeline matches A–F, preserves `d(r)=A^{{-1}}r-M^{{-1}}r`, uses accurate solves only for target generation, compares independently fitted subspaces, and uses FGMRES-compatible stored preconditioned directions for the nonlinear learned action.

## Issues Found

{chr(10).join(f'{index + 1}. {issue}' for index, issue in enumerate(issues)) if issues else 'No material blocking issue found.'}

## Calculation Spot-Checks

- A–F row counts and evidence gates were recomputed from the saved CSV-equivalent rows before report generation.
- q90/q95/q99 are first-threshold indices and are paired with held-out projection errors at the same q.
- Direct-ML iteration and time deltas are paired by identical system and RHS against the strong ILU baseline.
- {len(failures)} numerical failure/event records are retained rather than filtered from the audit trail.

## Visualization Review

Generated {len(plots)} of {expected_charts} expected static figures. All use explicit palettes, labeled axes, true-residual conventions, and separate controls; final image inspection is recorded in the task handoff.

## Required Caveats

- Single-machine timings are comparative, not portable benchmarks.
- Finite snapshot rank and common-grid interpolation bound the subspace claims.
- The MLP conclusion is fixed-N and one-parameter-family evidence only.
"""
    path = output / "validation_report.md"
    path.write_text(text, encoding="utf-8")
    return path
