"""Stage 31: paired residual/action-aware compression for FD recycle spaces."""

from __future__ import annotations

import json
import platform
import shutil
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
import scipy.sparse as sp

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _orthonormalize, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import (
    Stage30Settings,
    _bool,
    _file_hash,
    _fmt,
    _manifest,
    _read_csv,
    _sci,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
GREEN = "#30805C"
CHARCOAL = "#30343B"
GRAY = "#8A9099"
LIGHT_GRAY = "#D7DBE2"


@dataclass(frozen=True)
class Stage31Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    shared_rank: int = 168
    numerical_rank_tolerance: float = 1.0e-13
    rhs_convergence_min: float = 0.90
    system_convergence_min: float = 6.0 / 7.0
    median_rho_min: float = 0.80
    iteration_ratio_to_full_max: float = 1.50
    overlap_min: float = 0.90
    dW_max: float = 0.35

    @property
    def stage30(self) -> Stage30Settings:
        return Stage30Settings()


def build_action_aware_basis(
    root: Path,
    output: Path,
    split: Mapping[str, Sequence[int]],
    settings: Stage31Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compress paired training teachers by residual-snapshot reconstruction.

    For paired blocks ``A_i U_i = W_i``, the leading right singular vectors
    of ``W=[W_i]`` define the rank-p coefficient projector minimizing
    ``||W-W V_p V_p.T||_F``.  Applying the same coefficients to ``U=[U_i]``
    maps the residual-aware compression back to a shared solution space.
    """

    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage30 = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    paired_z_blocks: List[np.ndarray] = []
    paired_w_blocks: List[np.ndarray] = []
    source_ranks: List[int] = []
    source_hashes: Dict[str, Dict[str, str]] = {}
    pair_defects: List[float] = []
    for sample_id in split["train"]:
        teacher_path = stage28 / "systems" / f"system_{sample_id:03d}.npz"
        matrix_path = stage28 / "matrices" / f"A_{sample_id:03d}.npz"
        with np.load(teacher_path) as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
        matrix = sp.load_npz(matrix_path).tocsr()
        pair = _pair_from_z(matrix, zfull)
        paired_z_blocks.append(pair.u_basis)
        paired_w_blocks.append(pair.w_basis)
        source_ranks.append(pair.rank)
        pair_defects.append(float(pair.diagnostics["au_equals_w_relative_error"]))
        source_hashes[str(sample_id)] = {
            "teacher": _file_hash(teacher_path),
            "matrix": _file_hash(matrix_path),
        }

    paired_z = np.column_stack(paired_z_blocks)
    paired_w = np.column_stack(paired_w_blocks)
    started = perf_counter()
    _, singular_values, right_t = la.svd(
        paired_w, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    right = right_t[: settings.shared_rank].T
    compressed_raw = paired_z @ right
    action_basis, compressed_singular_values, compressed_rank = _orthonormalize(
        compressed_raw, settings.numerical_rank_tolerance
    )
    build_seconds = perf_counter() - started

    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    residual_reconstruction_error = float(np.sqrt(max(0.0, 1.0 - energy[settings.shared_rank - 1])))
    w_rank = int(np.count_nonzero(singular_values > 1.0e-14 * singular_values[0]))
    with np.load(stage30 / "full_training_span.npz") as archive:
        full_basis = np.asarray(archive["psi"], dtype=float)
        full_training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()
    projection = full_basis.T @ action_basis
    full_span_residual = float(
        la.norm(action_basis - full_basis @ projection, ord="fro")
        / max(la.norm(action_basis, ord="fro"), np.finfo(float).tiny)
    )
    validation_overlap = sorted(set(split["train"]) & set(split["validation"]))
    test_overlap = sorted(set(split["train"]) & set(split["test"]))
    diagnostics: Dict[str, Any] = {
        "criterion": "paired residual POD: V_p minimizes ||W-W V_p V_p^T||_F; basis=orth(U V_p)",
        "training_ids_exactly_match_stage30": list(split["train"]) == full_training_ids,
        "training_system_count": len(split["train"]),
        "validation_teacher_overlap": validation_overlap,
        "test_teacher_overlap": test_overlap,
        "paired_snapshot_shape": list(paired_z.shape),
        "paired_W_snapshot_shape": list(paired_w.shape),
        "sum_source_ranks": int(sum(source_ranks)),
        "maximum_training_pair_defect": max(pair_defects),
        "W_numerical_rank": w_rank,
        "requested_rank": settings.shared_rank,
        "compressed_basis_rank": int(compressed_rank),
        "captured_training_W_energy": float(energy[settings.shared_rank - 1]),
        "W_energy_ranks": {
            str(int(100 * target)): int(np.searchsorted(energy, target) + 1)
            for target in (0.80, 0.90, 0.95, 0.99)
        },
        "training_W_relative_reconstruction_error": residual_reconstruction_error,
        "compressed_raw_condition": float(
            compressed_singular_values[0] / compressed_singular_values[-1]
        ),
        "action_basis_full_span_relative_residual": full_span_residual,
        "build_seconds": build_seconds,
        "source_file_sha256": source_hashes,
    }
    diagnostics["passed"] = bool(
        diagnostics["training_ids_exactly_match_stage30"]
        and not validation_overlap
        and not test_overlap
        and paired_z.shape == paired_w.shape
        and paired_z.shape[1] == sum(source_ranks) == settings.stage30.full_expected_rank
        and max(pair_defects) <= 1.0e-9
        and w_rank == settings.stage30.full_expected_rank
        and compressed_rank == settings.shared_rank
        and full_span_residual <= 1.0e-9
    )
    _write_json(output / "compression_diagnostics.json", diagnostics)
    if not diagnostics["passed"]:
        raise RuntimeError("Stage-31 action-aware compression preflight failed; no held-out solve was run")
    np.savez_compressed(
        output / "krylov_action_basis_p168.npz",
        psi=action_basis,
        residual_singular_values=singular_values,
        residual_energy=energy,
        right_coefficients=right,
        training_ids=np.asarray(split["train"], dtype=int),
        source_ranks=np.asarray(source_ranks, dtype=int),
    )
    return action_basis, diagnostics


def _archive_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage30 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "method_setup_seconds": float(row["method_setup_seconds"]),
        "online_solve_seconds": float(row["online_solve_seconds"]),
        "reduced_condition": row["reduced_condition"],
        "au_equals_w_relative_error": row["au_equals_w_relative_error"],
        "failure": row["failure"],
    }


def run_solver_experiment(
    root: Path,
    output: Path,
    action_basis: np.ndarray,
    settings: Stage31Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage30 = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    if tuple(sorted(set(split["test"]) & set(split["hard_ids"]))) != settings.hard_system_ids:
        raise RuntimeError("Stage-31 hard-system cohort differs from Stage 30")

    archived = _read_csv(stage30 / "solver_results.csv")
    archived_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    geometry30 = {
        (int(row["sample_id"]), row["method"]): row
        for row in _read_csv(stage30 / "geometry_diagnostics.csv")
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage30.stage29.stage28.stage19a)
    archived_methods = {
        "strong_ilu": "strong_ilu",
        "oracle_within_p168": "energy_p168",
        "oracle_within_full_span": "full_span",
        "teacher_full": "teacher_full",
    }

    rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []
    setup_rows: List[Dict[str, Any]] = []
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage30.stage29
        )
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
        teacher_pair = _pair_from_z(context.system.matrix, zfull)
        started = perf_counter()
        action_pair, selection = oracle_select_inside_dictionary(
            context.system.matrix, action_basis, wfull
        )
        setup_seconds = perf_counter() - started
        if action_pair.rank != teacher_pair.rank:
            raise RuntimeError(
                f"system {sample_id}: action selection rank {action_pair.rank} "
                f"does not equal teacher rank {teacher_pair.rank}"
            )

        for old_method, new_method in (
            ("oracle_within_p168", "energy_p168"),
            ("oracle_within_full_span", "full_span"),
        ):
            old = geometry30[(sample_id, old_method)]
            geometry.append(
                {
                    "sample_id": sample_id,
                    "method": new_method,
                    "candidate_span_rank": int(old["candidate_span_rank"]),
                    "actual_recycle_rank": int(old["actual_recycle_rank"]),
                    "d_Z_to_teacher": float(old["d_Z_to_teacher"]),
                    "d_W_to_teacher": float(old["d_W_to_teacher"]),
                    "maximum_angle_Z_degrees": float(old["maximum_angle_Z_degrees"]),
                    "maximum_angle_W_degrees": float(old["maximum_angle_W_degrees"]),
                    "mean_squared_principal_cosine": float(old["mean_squared_principal_cosine"]),
                    "minimum_principal_cosine": old["minimum_principal_cosine"],
                    "candidate_pair_condition": float(old["candidate_pair_condition"]),
                }
            )
        z_metrics = generalized_subspace_metrics(action_pair.z_basis, teacher_pair.z_basis)
        w_metrics = generalized_subspace_metrics(action_pair.w_basis, teacher_pair.w_basis)
        geometry.append(
            {
                "sample_id": sample_id,
                "method": "action_p168",
                "candidate_span_rank": action_basis.shape[1],
                "actual_recycle_rank": action_pair.rank,
                "d_Z_to_teacher": z_metrics["projector_distance_normalized"],
                "d_W_to_teacher": w_metrics["projector_distance_normalized"],
                "maximum_angle_Z_degrees": z_metrics["principal_angle_max_degrees"],
                "maximum_angle_W_degrees": w_metrics["principal_angle_max_degrees"],
                "mean_squared_principal_cosine": selection["mean_squared_principal_cosine"],
                "minimum_principal_cosine": selection["minimum_principal_cosine"],
                "candidate_pair_condition": selection["family_pair_condition"],
            }
        )
        setup_rows.append(
            {
                "sample_id": sample_id,
                "matrix_nnz": context.system.matrix.nnz,
                "teacher_rank": teacher_pair.rank,
                "action_candidate_rank": action_basis.shape[1],
                "action_selected_rank": action_pair.rank,
                "selection_and_pair_seconds": setup_seconds,
                "selected_pair_defect": action_pair.diagnostics["au_equals_w_relative_error"],
            }
        )

        for rhs_seed in settings.stage30.stage29.stage28.evaluation_rhs_seeds:
            for old_method, new_method in archived_methods.items():
                rows.append(
                    _archive_row(
                        archived_lookup[(sample_id, rhs_seed, old_method)], new_method
                    )
                )
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            result = _run_pair(
                context, rhs, action_pair, settings.stage30.stage29.stage28.stage22a
            )
            true_residual = _relative_residual(context.system.matrix, result.x, rhs)
            counts = result.operation_counts or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": "action_p168",
                    "source": "Stage31 new",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and true_residual <= 1.0e-8),
                    "true_final_residual": true_residual,
                    "actual_rank": action_pair.rank,
                    "method_setup_seconds": setup_seconds,
                    "online_solve_seconds": float(result.solve_seconds),
                    "reduced_condition": action_pair.diagnostics["reduced_condition"],
                    "au_equals_w_relative_error": action_pair.diagnostics["au_equals_w_relative_error"],
                    "A_applications": int(counts.get("a_matvecs", 0)),
                    "M_applications": int(counts.get("preconditioner_applications", 0)),
                    "failure": getattr(result, "failure", "") or "",
                }
            )
            baseline = archived_lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = archived_lookup[(sample_id, rhs_seed, "teacher_full")]
            full = archived_lookup[(sample_id, rhs_seed, "oracle_within_full_span")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for row in rows:
                if int(row["sample_id"]) != sample_id or int(row["rhs_seed"]) != rhs_seed:
                    continue
                row["baseline_iterations"] = int(baseline["iterations"])
                row["teacher_iterations"] = int(teacher["iterations"])
                row["full_span_iterations"] = int(full["iterations"])
                row["rho"] = (
                    (int(baseline["iterations"]) - int(row["iterations"])) / denominator
                    if denominator > 0
                    else float("nan")
                )
                row["iteration_ratio_to_ilu"] = int(row["iterations"]) / max(
                    int(baseline["iterations"]), 1
                )
                row["iteration_ratio_to_full_span"] = int(row["iterations"]) / max(
                    int(full["iterations"]), 1
                )
        print(f"[Stage31 {position}/7] system {sample_id} complete", flush=True)

    _write_csv(output / "solver_results.csv", rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    _write_csv(output / "system_setup.csv", setup_rows)
    return rows, geometry, setup_rows


def summarize(
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage31Settings,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    methods = ("strong_ilu", "energy_p168", "action_p168", "full_span", "teacher_full")
    summaries: List[Dict[str, Any]] = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        system_passes = [
            all(
                _bool(row["converged"])
                for row in local
                if int(row["sample_id"]) == sample_id
            )
            for sample_id in settings.hard_system_ids
        ]
        summaries.append(
            {
                "method": method,
                "rhs_count": len(local),
                "rhs_convergence_fraction": float(np.mean([_bool(row["converged"]) for row in local])),
                "all_rhs_system_pass_fraction": float(np.mean(system_passes)),
                "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                "median_rho": float(np.median([float(row["rho"]) for row in local])),
                "median_iteration_ratio_to_ilu": float(np.median([float(row["iteration_ratio_to_ilu"]) for row in local])),
                "median_iteration_ratio_to_full_span": float(np.median([float(row["iteration_ratio_to_full_span"]) for row in local])),
                "median_actual_rank": float(np.median([int(row["actual_rank"]) for row in local])),
                "median_setup_seconds": float(np.median([float(row["method_setup_seconds"]) for row in local])),
                "median_online_solve_seconds": float(np.median([float(row["online_solve_seconds"]) for row in local])),
            }
        )
    geometry_summary: Dict[str, Dict[str, float]] = {}
    for method in ("energy_p168", "action_p168", "full_span"):
        local = [row for row in geometry if row["method"] == method]
        geometry_summary[method] = {
            key: float(np.median([float(row[key]) for row in local]))
            for key in (
                "d_Z_to_teacher",
                "d_W_to_teacher",
                "maximum_angle_Z_degrees",
                "maximum_angle_W_degrees",
                "mean_squared_principal_cosine",
            )
        }
    return summaries, geometry_summary


def classify_result(
    action: Mapping[str, float],
    energy: Mapping[str, float],
    action_geometry: Mapping[str, float],
    energy_geometry: Mapping[str, float],
    settings: Stage31Settings,
) -> Dict[str, Any]:
    solver_gate = bool(
        action["rhs_convergence_fraction"] >= settings.rhs_convergence_min
        and action["all_rhs_system_pass_fraction"] >= settings.system_convergence_min
        and action["median_rho"] >= settings.median_rho_min
        and action["median_iteration_ratio_to_full_span"] <= settings.iteration_ratio_to_full_max
    )
    geometry_gate = bool(
        action_geometry["mean_squared_principal_cosine"] >= settings.overlap_min
        and action_geometry["d_W_to_teacher"] <= settings.dW_max
    )
    improves_both = bool(
        action["rhs_convergence_fraction"] > energy["rhs_convergence_fraction"]
        and action["median_rho"] > energy["median_rho"]
        and action_geometry["mean_squared_principal_cosine"]
        > energy_geometry["mean_squared_principal_cosine"]
        and action_geometry["d_W_to_teacher"] < energy_geometry["d_W_to_teacher"]
    )
    if solver_gate and geometry_gate:
        classification = "success"
        statement = "Rank-168 action-aware compression recovers most full-span performance."
        stage32 = (
            "Freeze the action-aware basis and train an oracle-blind predictor/selector for its "
            "target coordinates, with closed-loop GMRES recovery as the acceptance metric."
        )
    elif improves_both:
        classification = "partial"
        statement = "Action-aware compression improves the representation but rank 168 does not recover most full-span performance."
        stage32 = (
            "Keep the paired residual criterion and use validation systems to locate the smallest "
            "rank that reaches the full-span geometry and solver gates before training a predictor."
        )
    else:
        classification = "failure"
        statement = "Paired residual POD at rank 168 does not repair the representation bottleneck."
        stage32 = (
            "Replace global paired POD with an operator-conditioned or localized compression; "
            "do not train a predictor on the failed basis."
        )
    return {
        "classification": classification,
        "statement": statement,
        "solver_gate_passed": solver_gate,
        "geometry_gate_passed": geometry_gate,
        "improves_over_energy_on_solver_and_geometry": improves_both,
        "stage32_direction": stage32,
    }


def make_plots(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage31Settings,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    methods = ("energy_p168", "action_p168", "full_span", "teacher_full")
    labels = ("energy p=168", "action p=168", "full span", "own teacher")
    colors = (ORANGE, GREEN, BLUE, CHARCOAL)
    x = np.arange(len(settings.hard_system_ids), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods))
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    for offset, method, label, color in zip(offsets, methods, labels, colors):
        values = [
            np.median([
                int(row["iterations"])
                for row in rows
                if int(row["sample_id"]) == sample_id and row["method"] == method
            ])
            for sample_id in settings.hard_system_ids
        ]
        ax.scatter(x + offset, values, s=48, color=color, label=label)
    ax.axhline(300, color=GRAY, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, [str(value) for value in settings.hard_system_ids])
    ax.set(
        xlabel="hard final-test system ID",
        ylabel="median iterations over three RHSs",
        title="Stage 31 rank-168 action-aware compression",
    )
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "solver_comparison.png", dpi=180)
    plt.close(fig)

    lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    width = 0.25
    for axis, key, ylabel in (
        (axes[0], "d_W_to_teacher", "normalized W-projector distance"),
        (axes[1], "maximum_angle_W_degrees", "maximum W principal angle (degrees)"),
    ):
        for offset, method, label, color in zip(
            (-width, 0.0, width),
            ("energy_p168", "action_p168", "full_span"),
            ("energy p=168", "action p=168", "full span"),
            (ORANGE, GREEN, BLUE),
        ):
            values = [float(lookup[(sid, method)][key]) for sid in settings.hard_system_ids]
            axis.bar(x + offset, values, width, color=color, label=label)
        axis.set_xticks(x, [str(value) for value in settings.hard_system_ids])
        axis.set(xlabel="system ID", ylabel=ylabel)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "teacher_geometry.png", dpi=180)
    plt.close(fig)

    local = [row for row in rows if int(row["sample_id"]) == 63 and row["method"] in methods]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    seeds = sorted({int(row["rhs_seed"]) for row in local})
    width = 0.19
    sx = np.arange(len(seeds), dtype=float)
    for offset, method, label, color in zip(
        np.linspace(-0.285, 0.285, len(methods)), methods, labels, colors
    ):
        values = [
            int(next(row["iterations"] for row in local if row["method"] == method and int(row["rhs_seed"]) == seed))
            for seed in seeds
        ]
        ax.bar(sx + offset, values, width, color=color, label=label)
    ax.set_xticks(sx, [str(seed) for seed in seeds])
    ax.set(xlabel="RHS seed", ylabel="iterations", title="System 63")
    ax.axhline(300, color=GRAY, linestyle=":", linewidth=1.0)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "system63.png", dpi=180)
    plt.close(fig)
    _write_csv(
        output / "chart_map.csv",
        [
            {"figure": "plots/solver_comparison.png", "source": "solver_results.csv", "purpose": "Compare fixed-rank compression with both Stage-30 ceilings."},
            {"figure": "plots/teacher_geometry.png", "source": "geometry_diagnostics.csv", "purpose": "Compare target teacher residual-space geometry."},
            {"figure": "plots/system63.png", "source": "solver_results.csv", "purpose": "Show the three RHS outcomes for the difficult system 63."},
        ],
    )


def write_report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    geometry_summary: Mapping[str, Mapping[str, float]],
    decision: Mapping[str, Any],
    compression: Mapping[str, Any],
    settings: Stage31Settings,
) -> Path:
    summary = {row["method"]: row for row in summaries}
    labels = {
        "strong_ilu": "strong ILU",
        "energy_p168": "energy p=168",
        "action_p168": "action-aware p=168",
        "full_span": "full span",
        "teacher_full": "own-system teacher",
    }
    lines = [
        "# Stage 31: Krylov-aware shared-basis compression",
        "",
        "## Result",
        "",
        f"**{decision['statement']}**",
        "",
        "The new basis uses only the 43 Stage-28 training teachers. For each system, the archived teacher space is reconstructed as a paired block $A_iU_i=W_i$. The leading 168 right singular vectors of $W=[W_i]$ minimize residual-snapshot reconstruction error, and the same coefficients are mapped back through $U=[U_i]$ before orthonormalization. Held-out teachers enter only the unchanged Stage-29 target-informed ceiling selector and geometry diagnostics. No neural model was trained.",
        "",
        f"The compression preflight passed: both paired snapshot matrices have shape {tuple(compression['paired_snapshot_shape'])}, the residual snapshot rank is {compression['W_numerical_rank']}, the final basis rank is {compression['compressed_basis_rank']}, and its distance outside the verified Stage-30 full span is {_sci(compression['action_basis_full_span_relative_residual'])}. Rank 168 captures {_fmt(100.0 * compression['captured_training_W_energy'], 1)}% of training residual energy, versus 99.0% solution-space energy for the old basis.",
        "",
        "## Closed-loop solver results",
        "",
        "| method | converged RHS | all-RHS systems | median iterations | median $\\rho$ | iterations / full span | median rank |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for method in labels:
        item = summary[method]
        lines.append(
            f"| {labels[method]} | {int(round(item['rhs_convergence_fraction'] * item['rhs_count']))}/{item['rhs_count']} | {_fmt(item['all_rhs_system_pass_fraction'])} | {_fmt(item['median_iterations'], 1)} | {_fmt(item['median_rho'])} | {_fmt(item['median_iteration_ratio_to_full_span'])} | {_fmt(item['median_actual_rank'], 1)} |"
        )
    lines.extend([
        "",
        "![Solver comparison](plots/solver_comparison.png)",
        "",
        "| system | energy p=168 | action p=168 | full span | teacher | action $\\rho$ | rank |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for sample_id in settings.hard_system_ids:
        local_by_method = {
            method: [row for row in rows if int(row["sample_id"]) == sample_id and row["method"] == method]
            for method in ("energy_p168", "action_p168", "full_span", "teacher_full")
        }
        values = {}
        for method, local in local_by_method.items():
            value = int(np.median([int(row["iterations"]) for row in local]))
            values[method] = f"{value}{'' if all(_bool(row['converged']) for row in local) else ' (cap)'}"
        action = local_by_method["action_p168"]
        lines.append(
            f"| {sample_id} | {values['energy_p168']} | {values['action_p168']} | {values['full_span']} | {values['teacher_full']} | {_fmt(np.median([float(row['rho']) for row in action]))} | {int(action[0]['actual_rank'])} |"
        )
    lines.extend([
        "",
        "## Teacher-space geometry",
        "",
        "| candidate | median $d_Z$ | median $d_W$ | max angle Z | max angle W | mean squared principal cosine |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    for method in ("energy_p168", "action_p168", "full_span"):
        item = geometry_summary[method]
        lines.append(
            f"| {labels[method]} | {_fmt(item['d_Z_to_teacher'])} | {_fmt(item['d_W_to_teacher'])} | {_fmt(item['maximum_angle_Z_degrees'], 1)} degrees | {_fmt(item['maximum_angle_W_degrees'], 1)} degrees | {_fmt(item['mean_squared_principal_cosine'])} |"
        )
    lines.extend([
        "",
        f"Action-aware compression improves average residual overlap, but it does not improve the weakest retained direction: the median maximum $W$ angle is {_fmt(geometry_summary['action_p168']['maximum_angle_W_degrees'], 1)} degrees versus {_fmt(geometry_summary['energy_p168']['maximum_angle_W_degrees'], 1)} degrees for energy p=168 and {_fmt(geometry_summary['full_span']['maximum_angle_W_degrees'], 1)} degrees for the full span. This explains why better average geometry produces a large iteration improvement without full-span Krylov behavior; a single poorly represented slow direction can still control convergence.",
    ])
    system63 = {
        method: [row for row in rows if int(row["sample_id"]) == 63 and row["method"] == method]
        for method in ("energy_p168", "action_p168", "full_span", "teacher_full")
    }
    geom63 = {
        row["method"]: row
        for row in _read_csv(output / "geometry_diagnostics.csv")
        if int(row["sample_id"]) == 63
    }
    action63 = system63["action_p168"]
    lines.extend([
        "",
        "![Teacher geometry](plots/teacher_geometry.png)",
        "",
        "## System 63",
        "",
        f"System 63 uses teacher rank {int(action63[0]['actual_rank'])}. Its median iterations are {int(np.median([int(row['iterations']) for row in system63['energy_p168']]))} for energy p=168, {int(np.median([int(row['iterations']) for row in action63]))} for action p=168, {int(np.median([int(row['iterations']) for row in system63['full_span']]))} for the full span, and {int(np.median([int(row['iterations']) for row in system63['teacher_full']]))} for its own teacher. The action basis has $d_W={_fmt(geom63['action_p168']['d_W_to_teacher'])}$, maximum $W$ angle {_fmt(geom63['action_p168']['maximum_angle_W_degrees'], 1)} degrees, and median $\\rho={_fmt(np.median([float(row['rho']) for row in action63]))}$.",
        "",
        "![System 63](plots/system63.png)",
        "",
        "## Stage 32 recommendation",
        "",
        decision["stage32_direction"],
        "",
        "This remains a representation ceiling because held-out teacher spaces select the final coordinates. It does not demonstrate an oracle-blind deployable method.",
    ])
    report = output / "stage31_krylov_aware_shared_basis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    compression: Mapping[str, Any],
    decision: Mapping[str, Any],
    settings: Stage31Settings,
) -> Dict[str, Any]:
    new_rows = [row for row in rows if row["source"] == "Stage31 new"]
    converged = [row for row in rows if _bool(row["converged"])]
    validation = {
        "compression_preflight_passed_before_solves": bool(compression["passed"]),
        "solver_rows_expected": 7 * 3 * 5,
        "solver_rows_actual": len(rows),
        "new_solver_rows_expected": 7 * 3,
        "new_solver_rows_actual": len(new_rows),
        "geometry_rows_expected": 7 * 3,
        "geometry_rows_actual": len(geometry),
        "action_candidate_rank_is_168": int(compression["compressed_basis_rank"]) == settings.shared_rank,
        "selected_ranks_match_teacher": all(
            int(row["actual_rank"])
            == int(next(candidate["actual_rank"] for candidate in rows if candidate["sample_id"] == row["sample_id"] and candidate["rhs_seed"] == row["rhs_seed"] and candidate["method"] == "teacher_full"))
            for row in rows
            if row["method"] in {"energy_p168", "action_p168", "full_span"}
        ),
        "maximum_converged_true_relative_residual": max(float(row["true_final_residual"]) for row in converged),
        "maximum_new_pair_defect": max(float(row["au_equals_w_relative_error"]) for row in new_rows),
        "classification": decision["classification"],
        "no_neural_training": True,
    }
    validation["passed"] = bool(
        validation["compression_preflight_passed_before_solves"]
        and validation["solver_rows_actual"] == validation["solver_rows_expected"]
        and validation["new_solver_rows_actual"] == validation["new_solver_rows_expected"]
        and validation["geometry_rows_actual"] == validation["geometry_rows_expected"]
        and validation["action_candidate_rank_is_168"]
        and validation["selected_ranks_match_teacher"]
        and validation["maximum_converged_true_relative_residual"] <= 1.0e-8
        and validation["maximum_new_pair_defect"] <= 1.0e-9
    )
    _write_json(output / "validation.json", validation)
    return validation


def run_stage31(root: Path) -> Path:
    settings = Stage31Settings()
    output = root / "results/results-fd/stages/stage31_krylov_aware_shared_basis"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("predeclared_protocol.json must exist before Stage-31 execution")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    basis, compression = build_action_aware_basis(root, output, split, settings)
    print(
        f"[Stage31 preflight] verified paired action basis, rank={basis.shape[1]}, "
        f"W energy={compression['captured_training_W_energy']:.6f}",
        flush=True,
    )
    rows, geometry, _ = run_solver_experiment(root, output, basis, settings)
    summaries, geometry_summary = summarize(rows, geometry, settings)
    summary = {row["method"]: row for row in summaries}
    decision = classify_result(
        summary["action_p168"],
        summary["energy_p168"],
        geometry_summary["action_p168"],
        geometry_summary["energy_p168"],
        settings,
    )
    if decision["classification"] == "partial":
        ranks = compression["W_energy_ranks"]
        decision["stage32_direction"] = (
            "Use only the Stage-28 validation systems to test the same paired residual-POD "
            f"construction at ranks {ranks['80']}, {ranks['90']}, and {ranks['95']} "
            "(80%, 90%, and 95% training W energy), with rank 168 and the full span as "
            "anchors. Freeze the smallest rank meeting the Stage-31 geometry and solver gates "
            "before any further held-out test or neural training."
        )
    decision["solver_summary"] = summary
    decision["geometry_summary"] = geometry_summary
    decision["gates"] = {
        "rhs_convergence_min": settings.rhs_convergence_min,
        "system_convergence_min": settings.system_convergence_min,
        "median_rho_min": settings.median_rho_min,
        "iteration_ratio_to_full_max": settings.iteration_ratio_to_full_max,
        "overlap_min": settings.overlap_min,
        "dW_max": settings.dW_max,
    }
    _write_csv(output / "solver_summary.csv", summaries)
    _write_json(output / "decision.json", decision)
    make_plots(output, rows, geometry, settings)
    report = write_report(
        output, rows, summaries, geometry_summary, decision, compression, settings
    )
    validation = validate(output, rows, geometry, compression, decision, settings)
    _write_json(
        output / "source_notes.json",
        {
            "primary_sources": [
                "Stage-28 training zfull spaces and saved matrices",
                "Stage-30 archived energy/full-span/teacher/baseline rows",
            ],
            "only_new_solver_method": "paired residual-POD action basis at rank 168",
            "validation_assessment": "ready to share" if validation["passed"] else "needs revision",
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage31_krylov_aware_shared_basis.md"
    copied = report.read_text(encoding="utf-8").replace(
        "](plots/", "](../stages/stage31_krylov_aware_shared_basis/plots/"
    )
    report_copy.write_text(copied, encoding="utf-8")
    print("[Stage31] complete", flush=True)
    return report_copy
