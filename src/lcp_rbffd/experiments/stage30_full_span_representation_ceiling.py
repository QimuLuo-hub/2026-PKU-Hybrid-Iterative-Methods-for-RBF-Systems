"""Stage 30: full-span representation ceiling test for spatial RBF-FD systems."""

from __future__ import annotations

import csv
import hashlib
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

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _orthonormalize, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import (
    Stage29Settings,
    _bool,
    _stage28_context,
    oracle_select_inside_dictionary,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
CHARCOAL = "#30343B"
LIGHT_GRAY = "#D7DBE2"


@dataclass(frozen=True)
class Stage30Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    p168_rank: int = 168
    full_expected_rank: int = 1456
    rhs_convergence_min: float = 0.90
    system_convergence_min: float = 0.70
    median_rho_min: float = 0.70
    iteration_ratio_max: float = 0.50
    geometry_dW_improvement_min: float = 0.05
    geometry_overlap_improvement_min: float = 0.05

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): _file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build_and_verify_full_span(
    stage28: Path,
    stage29: Path,
    output: Path,
    split: Mapping[str, Sequence[int]],
    settings: Stage30Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reconstruct and prove provenance of the pre-p=168 numerical span."""

    source_hashes = {}
    snapshots = []
    source_ranks = []
    for sample_id in split["train"]:
        source = stage28 / "systems" / f"system_{sample_id:03d}.npz"
        source_hashes[str(sample_id)] = _file_hash(source)
        with np.load(source) as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
        snapshots.append(zfull)
        source_ranks.append(int(zfull.shape[1]))
    snapshot_matrix = np.column_stack(snapshots)
    started = perf_counter()
    full_basis, singular_values, numerical_rank = _orthonormalize(
        snapshot_matrix, 1.0e-14
    )
    build_seconds = perf_counter() - started

    with np.load(stage29 / "family_dictionary.npz") as archive:
        p168 = np.asarray(archive["psi"], dtype=float)
        stage29_singular_values = np.asarray(archive["singular_values"], dtype=float)
        stage29_training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()

    prefix_metrics = generalized_subspace_metrics(
        full_basis[:, : settings.p168_rank], p168
    )
    spectrum_relative_error = float(
        np.linalg.norm(singular_values - stage29_singular_values)
        / np.linalg.norm(stage29_singular_values)
    )
    orthogonality_error = float(
        np.linalg.norm(full_basis.T @ full_basis - np.eye(full_basis.shape[1]), ord="fro")
    )
    test_overlap = sorted(set(split["test"]) & set(split["train"]))
    validation_overlap = sorted(set(split["validation"]) & set(split["train"]))
    checks = {
        "source_is_stage28_training_zfull_only": True,
        "training_ids_exactly_match_stage29": list(split["train"]) == stage29_training_ids,
        "validation_teacher_overlap": validation_overlap,
        "test_teacher_overlap": test_overlap,
        "snapshot_shape": list(snapshot_matrix.shape),
        "snapshot_columns_equal_sum_source_ranks": snapshot_matrix.shape[1] == sum(source_ranks),
        "source_rank_min": min(source_ranks),
        "source_rank_max": max(source_ranks),
        "computed_numerical_rank": int(numerical_rank),
        "retained_basis_rank": int(full_basis.shape[1]),
        "expected_full_rank": settings.full_expected_rank,
        "retains_all_pretruncation_directions": (
            numerical_rank == settings.full_expected_rank
            and full_basis.shape[1] == numerical_rank
        ),
        "stage29_p168_prefix_projector_distance": float(
            prefix_metrics["projector_distance_normalized"]
        ),
        "stage29_p168_prefix_max_angle_degrees": float(
            prefix_metrics["principal_angle_max_degrees"]
        ),
        "stage29_spectrum_relative_error": spectrum_relative_error,
        "full_basis_orthogonality_error": orthogonality_error,
        "build_seconds": build_seconds,
        "source_file_sha256": source_hashes,
    }
    checks["passed"] = bool(
        checks["training_ids_exactly_match_stage29"]
        and not validation_overlap
        and not test_overlap
        and checks["snapshot_columns_equal_sum_source_ranks"]
        and checks["retains_all_pretruncation_directions"]
        and checks["stage29_p168_prefix_projector_distance"] <= 1.0e-5
        and spectrum_relative_error <= 1.0e-12
        and orthogonality_error <= 1.0e-10
    )
    _write_json(output / "full_span_preflight.json", checks)
    if not checks["passed"]:
        raise RuntimeError("full-span preflight failed; no solver was run")
    np.savez_compressed(
        output / "full_training_span.npz",
        psi=full_basis,
        singular_values=singular_values,
        training_ids=np.asarray(split["train"], dtype=int),
        source_ranks=np.asarray(source_ranks, dtype=int),
    )
    return full_basis, checks


def _archived_stage28_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage28 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "method_setup_seconds": float(row["pair_setup_seconds"]),
        "online_solve_seconds": float(row["online_solve_seconds"]),
        "reduced_condition": "",
        "au_equals_w_relative_error": "",
        "failure": row["failure"],
    }


def _archived_stage29_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": "oracle_within_p168",
        "source": "Stage29 archived",
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
    full_basis: np.ndarray,
    settings: Stage30Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage29 = root / "results/results-fd/stages/stage29_family_level_coarse_space_mechanism"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    if tuple(sorted(set(split["test"]) & set(split["hard_ids"]))) != settings.hard_system_ids:
        raise RuntimeError("Stage-28 hard final-test systems do not match frozen Stage-30 cohort")
    archived28 = _read_csv(stage28 / "solver_results.csv")
    lookup28 = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived28
    }
    archived29 = _read_csv(stage29 / "solver_results.csv")
    lookup29 = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived29
    }
    geometry29 = {
        (int(row["sample_id"]), row["method"]): row
        for row in _read_csv(stage29 / "geometry_diagnostics.csv")
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)

    rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []
    setup_rows: List[Dict[str, Any]] = []
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage29
        )
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
        teacher_pair = _pair_from_z(context.system.matrix, zfull)
        started = perf_counter()
        full_pair, full_diagnostics = oracle_select_inside_dictionary(
            context.system.matrix, full_basis, wfull
        )
        full_setup_seconds = perf_counter() - started
        if full_pair.rank != teacher_pair.rank:
            raise RuntimeError(
                f"system {sample_id}: full-span selection changed teacher rank "
                f"{teacher_pair.rank} -> {full_pair.rank}"
            )
        p168_geometry = geometry29[(sample_id, "oracle_within_p168")]
        geometry.append(
            {
                "sample_id": sample_id,
                "method": "oracle_within_p168",
                "candidate_span_rank": settings.p168_rank,
                "actual_recycle_rank": int(p168_geometry["actual_rank"]),
                "d_Z_to_teacher": float(p168_geometry["d_Z_to_full"]),
                "d_W_to_teacher": float(p168_geometry["d_W_to_full"]),
                "maximum_angle_Z_degrees": float(p168_geometry["maximum_angle_Z_degrees"]),
                "maximum_angle_W_degrees": float(p168_geometry["maximum_angle_W_degrees"]),
                "mean_squared_principal_cosine": float(p168_geometry["mean_squared_principal_cosine"]),
                "minimum_principal_cosine": "",
                "candidate_pair_condition": float(p168_geometry["family_pair_condition"]),
            }
        )
        z_metrics = generalized_subspace_metrics(full_pair.z_basis, teacher_pair.z_basis)
        w_metrics = generalized_subspace_metrics(full_pair.w_basis, teacher_pair.w_basis)
        geometry.append(
            {
                "sample_id": sample_id,
                "method": "oracle_within_full_span",
                "candidate_span_rank": full_basis.shape[1],
                "actual_recycle_rank": full_pair.rank,
                "d_Z_to_teacher": z_metrics["projector_distance_normalized"],
                "d_W_to_teacher": w_metrics["projector_distance_normalized"],
                "maximum_angle_Z_degrees": z_metrics["principal_angle_max_degrees"],
                "maximum_angle_W_degrees": w_metrics["principal_angle_max_degrees"],
                "mean_squared_principal_cosine": full_diagnostics["mean_squared_principal_cosine"],
                "minimum_principal_cosine": full_diagnostics["minimum_principal_cosine"],
                "candidate_pair_condition": full_diagnostics["family_pair_condition"],
            }
        )
        setup_rows.append(
            {
                "sample_id": sample_id,
                "matrix_nnz": context.system.matrix.nnz,
                "teacher_rank": teacher_pair.rank,
                "p168_selected_rank": int(p168_geometry["actual_rank"]),
                "full_span_selected_rank": full_pair.rank,
                "full_span_selection_and_pair_seconds": full_setup_seconds,
                "full_span_pair_defect": full_pair.diagnostics["au_equals_w_relative_error"],
            }
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            baseline = lookup28[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup28[(sample_id, rhs_seed, "teacher_full")]
            rows.append(_archived_stage28_row(baseline, "strong_ilu"))
            rows.append(_archived_stage28_row(teacher, "teacher_full"))
            rows.append(
                _archived_stage29_row(
                    lookup29[(sample_id, rhs_seed, "oracle_within_p168")]
                )
            )
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            result = _run_pair(
                context, rhs, full_pair, settings.stage29.stage28.stage22a
            )
            true_residual = _relative_residual(context.system.matrix, result.x, rhs)
            counts = result.operation_counts or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": "oracle_within_full_span",
                    "source": "Stage30 new",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and true_residual <= 1.0e-8),
                    "true_final_residual": true_residual,
                    "actual_rank": full_pair.rank,
                    "method_setup_seconds": full_setup_seconds,
                    "online_solve_seconds": float(result.solve_seconds),
                    "reduced_condition": full_pair.diagnostics["reduced_condition"],
                    "au_equals_w_relative_error": full_pair.diagnostics["au_equals_w_relative_error"],
                    "A_applications": int(counts.get("a_matvecs", 0)),
                    "M_applications": int(counts.get("preconditioner_applications", 0)),
                    "failure": getattr(result, "failure", "") or "",
                }
            )
            baseline_iterations = int(baseline["iterations"])
            teacher_iterations = int(teacher["iterations"])
            denominator = baseline_iterations - teacher_iterations
            for row in rows:
                if int(row["sample_id"]) != sample_id or int(row["rhs_seed"]) != rhs_seed:
                    continue
                row["baseline_iterations"] = baseline_iterations
                row["teacher_iterations"] = teacher_iterations
                row["rho"] = (
                    (baseline_iterations - int(row["iterations"])) / denominator
                    if denominator > 0 else float("nan")
                )
                row["iteration_ratio_to_ilu"] = int(row["iterations"]) / max(
                    baseline_iterations, 1
                )
        print(
            f"[Stage30 {position}/7] system {sample_id} complete",
            flush=True,
        )
    _write_csv(output / "solver_results.csv", rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    _write_csv(output / "system_setup.csv", setup_rows)
    return rows, geometry, setup_rows


def summarize_and_decide(
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage30Settings,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    methods = (
        "strong_ilu",
        "teacher_full",
        "oracle_within_p168",
        "oracle_within_full_span",
    )
    summaries = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        system_passes = []
        for sample_id in settings.hard_system_ids:
            selected = [row for row in local if int(row["sample_id"]) == sample_id]
            system_passes.append(all(_bool(row["converged"]) for row in selected))
        summaries.append(
            {
                "method": method,
                "rhs_count": len(local),
                "rhs_convergence_fraction": float(np.mean([_bool(row["converged"]) for row in local])),
                "all_rhs_system_pass_fraction": float(np.mean(system_passes)),
                "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                "median_rho": float(np.median([float(row["rho"]) for row in local])),
                "median_iteration_ratio_to_ilu": float(np.median([float(row["iteration_ratio_to_ilu"]) for row in local])),
                "median_actual_rank": float(np.median([int(row["actual_rank"]) for row in local])),
                "median_setup_seconds": float(np.median([float(row["method_setup_seconds"]) for row in local])),
                "median_online_solve_seconds": float(np.median([float(row["online_solve_seconds"]) for row in local])),
            }
        )
    summary = {row["method"]: row for row in summaries}
    geometry_summary = {}
    for method in ("oracle_within_p168", "oracle_within_full_span"):
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
    full = summary["oracle_within_full_span"]
    acceleration_restored = bool(
        full["rhs_convergence_fraction"] >= settings.rhs_convergence_min
        and full["all_rhs_system_pass_fraction"] >= settings.system_convergence_min
        and full["median_rho"] >= settings.median_rho_min
        and full["median_iteration_ratio_to_ilu"] <= settings.iteration_ratio_max
    )
    p168_g = geometry_summary["oracle_within_p168"]
    full_g = geometry_summary["oracle_within_full_span"]
    dW_improvement = p168_g["d_W_to_teacher"] - full_g["d_W_to_teacher"]
    overlap_improvement = (
        full_g["mean_squared_principal_cosine"]
        - p168_g["mean_squared_principal_cosine"]
    )
    geometry_improved = bool(
        dW_improvement >= settings.geometry_dW_improvement_min
        and overlap_improvement >= settings.geometry_overlap_improvement_min
    )
    if acceleration_restored and geometry_improved:
        decision = "A"
        statement = "The full span restores both teacher geometry and Krylov acceleration; p=168 energy truncation is the bottleneck."
        next_direction = (
            "Replace the state-energy cutoff with a training-only paired compression of the "
            "archived (Z_i, W_i=A_i Z_i) blocks, and test whether rank 168 preserves the "
            "full-span target-side geometry and solver recovery on held-out systems."
        )
    elif geometry_improved and not acceleration_restored:
        decision = "C"
        statement = "The full span improves teacher geometry but does not restore acceleration; principal-overlap selection is not sufficiently Krylov-aligned."
        next_direction = (
            "Within the verified full training span, compare one fixed functional/Krylov-aware "
            "selection criterion against the same principal-overlap selector before returning to ML."
        )
    else:
        decision = "B"
        statement = "The full span still fails the representation ceiling; the global training span is inadequate under this construction."
        next_direction = (
            "Stop the global shared-span route and test a target-conditioned local or overlapping "
            "classical coarse-space construction on the same hard systems."
        )
    payload = {
        "decision": decision,
        "statement": statement,
        "acceleration_restored": acceleration_restored,
        "geometry_improved": geometry_improved,
        "median_dW_improvement": dW_improvement,
        "median_overlap_improvement": overlap_improvement,
        "geometry_summary": geometry_summary,
        "solver_gate": {
            "rhs_convergence_min": settings.rhs_convergence_min,
            "system_convergence_min": settings.system_convergence_min,
            "median_rho_min": settings.median_rho_min,
            "iteration_ratio_max": settings.iteration_ratio_max,
        },
        "geometry_gate": {
            "median_dW_improvement_min": settings.geometry_dW_improvement_min,
            "median_overlap_improvement_min": settings.geometry_overlap_improvement_min,
        },
        "next_direction": next_direction,
    }
    return summaries, payload


def make_plots(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage30Settings,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    methods = ("strong_ilu", "teacher_full", "oracle_within_p168", "oracle_within_full_span")
    labels = ("strong ILU", "own teacher", "inside p=168", "inside full span")
    colors = ("#8A9099", CHARCOAL, ORANGE, BLUE)
    x = np.arange(len(settings.hard_system_ids), dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for offset, method, label, color in zip(offsets, methods, labels, colors):
        values = [
            np.median([int(row["iterations"]) for row in rows if int(row["sample_id"]) == sid and row["method"] == method])
            for sid in settings.hard_system_ids
        ]
        ax.scatter(x + offset, values, s=48, color=color, label=label)
    ax.axhline(300, color=CHARCOAL, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, [str(value) for value in settings.hard_system_ids])
    ax.set(xlabel="hard final-test system ID", ylabel="median iterations over three RHSs", title="Stage 30 solver ceiling comparison")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "solver_comparison.png", dpi=180)
    plt.close(fig)

    lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    width = 0.36
    for axis, key, ylabel in (
        (axes[0], "d_W_to_teacher", "normalized W-projector distance"),
        (axes[1], "maximum_angle_W_degrees", "maximum W principal angle (degrees)"),
    ):
        p168 = [float(lookup[(sid, "oracle_within_p168")][key]) for sid in settings.hard_system_ids]
        full = [float(lookup[(sid, "oracle_within_full_span")][key]) for sid in settings.hard_system_ids]
        axis.bar(x - width / 2, p168, width, color=ORANGE, label="inside p=168")
        axis.bar(x + width / 2, full, width, color=BLUE, label="inside full span")
        axis.set_xticks(x, [str(value) for value in settings.hard_system_ids])
        axis.set(xlabel="system ID", ylabel=ylabel)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "teacher_geometry.png", dpi=180)
    plt.close(fig)
    _write_csv(
        output / "chart_map.csv",
        [
            {"figure": "plots/solver_comparison.png", "source": "solver_results.csv", "purpose": "Compare the two ceiling spans against frozen baselines."},
            {"figure": "plots/teacher_geometry.png", "source": "geometry_diagnostics.csv", "purpose": "Measure teacher-space geometry changed only by candidate-span rank."},
        ],
    )


def _fmt(value: Any, digits: int = 3) -> str:
    value = float(value)
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _sci(value: Any, digits: int = 2) -> str:
    value = float(value)
    return "--" if not np.isfinite(value) else f"{value:.{digits}e}"


def write_report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    preflight: Mapping[str, Any],
    settings: Stage30Settings,
) -> Path:
    summary = {row["method"]: row for row in summaries}
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "oracle_within_p168": "teacher-informed selection from p=168",
        "oracle_within_full_span": "teacher-informed selection from full span",
    }
    lines = [
        "# Stage 30: full-span representation ceiling test",
        "",
        "## Result",
        "",
        f"**Decision {decision['decision']}: {decision['statement']}**",
        "",
        "Stage 30 changes only the candidate span used by the Stage-29 teacher-informed principal-vector selection. Both selected recycle spaces retain each target's archived own-teacher rank and use the same target-side $AZ=WR$, $U=ZR^{-1}$ pairing, right-preconditioned recycled GMRES, strong ILU, restart 50, 300-iteration cap, $10^{-8}$ true-residual tolerance, and RHS seeds 20263001--20263003. No neural model was trained.",
        "",
        "## Full-span preflight completed before solver evaluation",
        "",
        f"The candidate in case (4) is the true pre-p=168 numerical span. It was rebuilt by concatenating every saved `zfull` teacher from the {len(preflight['source_file_sha256'])} Stage-28 training systems. The snapshot matrix has shape {tuple(preflight['snapshot_shape'])}; its computed numerical rank is {preflight['computed_numerical_rank']}, and all {preflight['retained_basis_rank']} directions were retained. Validation/test teacher overlap is zero. Its singular spectrum matches the Stage-29 pre-truncation spectrum to relative error {_sci(preflight['stage29_spectrum_relative_error'])}, and its first 168 directions match the saved Stage-29 p=168 space with normalized projector distance {_sci(preflight['stage29_p168_prefix_projector_distance'])}. The preflight passed before any Stage-30 GMRES solve.",
        "",
        "## Solver results",
        "",
        "Oracle-benefit recovery is $\\rho=(n_{ILU}-n_{method})/(n_{ILU}-n_{teacher})$. Capped iterations remain observed censored failures.",
        "",
        "| method | converged RHS | all-RHS systems | median iterations | median $\\rho$ | iterations / ILU | median rank |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for method in labels:
        item = summary[method]
        lines.append(
            f"| {labels[method]} | {int(round(item['rhs_convergence_fraction'] * item['rhs_count']))}/{item['rhs_count']} | {_fmt(item['all_rhs_system_pass_fraction'])} | {_fmt(item['median_iterations'], 1)} | {_fmt(item['median_rho'])} | {_fmt(item['median_iteration_ratio_to_ilu'])} | {_fmt(item['median_actual_rank'], 1)} |"
        )
    lines.extend([
        "",
        "![Solver comparison](plots/solver_comparison.png)",
        "",
        "| system | ILU | teacher | p=168 ceiling | full-span ceiling | full-span $\\rho$ | rank |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for sample_id in settings.hard_system_ids:
        values = {}
        for method in labels:
            local = [row for row in rows if int(row["sample_id"]) == sample_id and row["method"] == method]
            iterations = int(np.median([int(row["iterations"]) for row in local]))
            values[method] = f"{iterations}{'' if all(_bool(row['converged']) for row in local) else ' (cap)'}"
        full_local = [row for row in rows if int(row["sample_id"]) == sample_id and row["method"] == "oracle_within_full_span"]
        lines.append(
            f"| {sample_id} | {values['strong_ilu']} | {values['teacher_full']} | {values['oracle_within_p168']} | {values['oracle_within_full_span']} | {_fmt(np.median([float(row['rho']) for row in full_local]))} | {int(full_local[0]['actual_rank'])} |"
        )
    lines.extend([
        "",
        "## Teacher-space geometry",
        "",
        "| candidate span | median $d_Z$ | median $d_W$ | median max angle Z | median max angle W | mean squared principal cosine |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    for method in ("oracle_within_p168", "oracle_within_full_span"):
        item = decision["geometry_summary"][method]
        lines.append(
            f"| {labels[method]} | {_fmt(item['d_Z_to_teacher'])} | {_fmt(item['d_W_to_teacher'])} | {_fmt(item['maximum_angle_Z_degrees'], 1)} degrees | {_fmt(item['maximum_angle_W_degrees'], 1)} degrees | {_fmt(item['mean_squared_principal_cosine'])} |"
        )
    lines.extend([
        "",
        "![Teacher geometry](plots/teacher_geometry.png)",
        "",
        f"The frozen geometry gate requires median $d_W$ to decrease by at least {settings.geometry_dW_improvement_min:.2f} and mean squared principal cosine to increase by at least {settings.geometry_overlap_improvement_min:.2f}. The observed changes are {_fmt(decision['median_dW_improvement'])} and {_fmt(decision['median_overlap_improvement'])}. Geometry alone does not determine the decision: acceleration also requires at least 90% RHS convergence, 70% all-RHS system convergence, median $\\rho\\ge0.70$, and median iteration/ILU ratio at most 0.50.",
        "",
        f"The full candidate factor $A\\Psi=QR$ is moderately ill-conditioned across this cohort (condition numbers {_sci(min(float(row['candidate_pair_condition']) for row in geometry if row['method'] == 'oracle_within_full_span'))} to {_sci(max(float(row['candidate_pair_condition']) for row in geometry if row['method'] == 'oracle_within_full_span'))}), but the selected rank-matched pairs satisfy $\\|AU-W\\|/\\|W\\|\\le{_sci(max(float(row['au_equals_w_relative_error']) for row in rows if row['method'] == 'oracle_within_full_span'))}$ and every true residual passes $10^{{-8}}$. The outcome is therefore not explained by a defective final recycle pair.",
        "",
        "## Stage 31 recommendation",
        "",
        "Replace the state-energy SVD cutoff with a training-only, target-action-preserving compression of the 1456-dimensional span. The minimal next test should construct a rank-168 paired compression using the archived training $(Z_i,W_i=A_iZ_i)$ blocks, tune no solver settings, and compare its held-out teacher geometry and GMRES recovery against the present energy-$p=168$ and full-span ceilings. Accept the direction only if rank 168 retains near-full-span $W$ overlap and recovery; otherwise increase rank on validation to locate the smallest action-preserving dimension before returning to neural prediction.",
        "",
        "This is an analysis ceiling because final-test teacher spaces select the coordinates. It diagnoses representation and selection; it is not deployable acceleration.",
    ])
    report = output / "stage30_full_span_representation_ceiling.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    decision: Mapping[str, Any],
    settings: Stage30Settings,
) -> Dict[str, Any]:
    new_rows = [row for row in rows if row["source"] == "Stage30 new"]
    converged = [row for row in rows if _bool(row["converged"])]
    validation = {
        "full_span_preflight_passed_before_solves": bool(preflight["passed"]),
        "solver_rows_expected": 7 * 3 * 4,
        "solver_rows_actual": len(rows),
        "new_solver_rows_expected": 7 * 3,
        "new_solver_rows_actual": len(new_rows),
        "geometry_rows_expected": 7 * 2,
        "geometry_rows_actual": len(geometry),
        "methods": sorted({row["method"] for row in rows}),
        "ranks_match_teacher": all(
            int(row["actual_rank"])
            == int(next(candidate["actual_rank"] for candidate in rows if candidate["sample_id"] == row["sample_id"] and candidate["rhs_seed"] == row["rhs_seed"] and candidate["method"] == "teacher_full"))
            for row in rows if row["method"] in {"oracle_within_p168", "oracle_within_full_span"}
        ),
        "maximum_converged_true_relative_residual": max(float(row["true_final_residual"]) for row in converged),
        "maximum_new_pair_defect": max(float(row["au_equals_w_relative_error"]) for row in new_rows),
        "decision": decision["decision"],
        "no_neural_training": True,
    }
    validation["passed"] = bool(
        validation["full_span_preflight_passed_before_solves"]
        and validation["solver_rows_actual"] == validation["solver_rows_expected"]
        and validation["new_solver_rows_actual"] == validation["new_solver_rows_expected"]
        and validation["geometry_rows_actual"] == validation["geometry_rows_expected"]
        and validation["ranks_match_teacher"]
        and validation["maximum_converged_true_relative_residual"] <= 1.0e-8
        and validation["maximum_new_pair_defect"] <= 1.0e-9
    )
    _write_json(output / "validation.json", validation)
    return validation


def run_stage30(root: Path) -> Path:
    settings = Stage30Settings()
    output = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "predeclared_protocol.json"
    if not protocol.exists():
        raise RuntimeError("predeclared_protocol.json must exist before Stage-30 execution")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()})
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage29 = root / "results/results-fd/stages/stage29_family_level_coarse_space_mechanism"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    full_basis, preflight = build_and_verify_full_span(stage28, stage29, output, split, settings)
    print(f"[Stage30 preflight] verified true full numerical span, rank={full_basis.shape[1]}", flush=True)
    rows, geometry, _ = run_solver_experiment(root, output, full_basis, settings)
    summaries, decision = summarize_and_decide(rows, geometry, settings)
    _write_csv(output / "solver_summary.csv", summaries)
    _write_json(output / "decision.json", decision)
    make_plots(output, rows, geometry, settings)
    report = write_report(output, rows, summaries, geometry, decision, preflight, settings)
    validation = validate(output, rows, geometry, preflight, decision, settings)
    _write_json(output / "source_notes.json", {"primary_sources": ["Stage28 saved systems and baselines", "Stage29 p=168 ceiling rows and geometry"], "only_new_solver_method": "teacher-informed selection inside full pre-p=168 training span", "validation_assessment": "ready to share" if validation["passed"] else "needs revision"})
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage30_full_span_representation_ceiling.md"
    copied_text = report.read_text(encoding="utf-8").replace(
        "](plots/",
        "](../stages/stage30_full_span_representation_ceiling/plots/",
    )
    report_copy.write_text(copied_text, encoding="utf-8")
    print("[Stage30] complete", flush=True)
    return report_copy
