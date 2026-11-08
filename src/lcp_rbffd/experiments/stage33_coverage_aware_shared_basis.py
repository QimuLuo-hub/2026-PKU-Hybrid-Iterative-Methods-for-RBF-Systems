"""Stage 33: greedy worst-covered residual-space compression."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _orthonormalize, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage32_action_aware_rank_sweep import Stage32Settings
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage33Settings:
    ranks: Tuple[int, ...] = (168, 270)
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    worst_angle_improvement_min: float = 5.0
    convergence_improvement_min: int = 1
    median_rho_tolerance: float = 0.02

    @property
    def stage32(self) -> Stage32Settings:
        return Stage32Settings()


def build_coverage_bases(
    root: Path,
    output: Path,
    split: Mapping[str, Sequence[int]],
    settings: Stage33Settings,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Greedily add the least-covered direction of the worst-covered W block."""

    stage28 = root / "results/results-fd/stage28_operator_learning"
    sample_ids = list(split["train"])
    u_blocks: List[np.ndarray] = []
    w_blocks: List[np.ndarray] = []
    pair_defects: List[float] = []
    slices: List[slice] = []
    offset = 0
    for sample_id in sample_ids:
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        pair = _pair_from_z(matrix, zfull)
        u_blocks.append(pair.u_basis)
        w_blocks.append(pair.w_basis)
        pair_defects.append(float(pair.diagnostics["au_equals_w_relative_error"]))
        slices.append(slice(offset, offset + pair.rank))
        offset += pair.rank

    all_w = np.column_stack(w_blocks)
    max_rank = max(settings.ranks)
    q_basis = np.empty((all_w.shape[0], max_rank), dtype=float)
    residual_grams = [np.eye(block.shape[1]) for block in w_blocks]
    selected_u: List[np.ndarray] = []
    selection_rows: List[Dict[str, Any]] = []
    milestone_rows: List[Dict[str, Any]] = []
    bases: Dict[int, np.ndarray] = {}
    started = perf_counter()

    for step in range(max_rank):
        scores = [
            max(0.0, float(np.trace(gram)) / gram.shape[0])
            for gram in residual_grams
        ]
        block_index = max(
            range(len(sample_ids)), key=lambda index: (scores[index], -sample_ids[index])
        )
        gram = 0.5 * (residual_grams[block_index] + residual_grams[block_index].T)
        eigenvalues, eigenvectors = la.eigh(gram, check_finite=False)
        direction = eigenvectors[:, -1]
        raw_w = w_blocks[block_index] @ direction
        raw_u = u_blocks[block_index] @ direction
        residual = raw_w.copy()
        if step:
            for _ in range(2):
                coefficients = q_basis[:, :step].T @ residual
                residual -= q_basis[:, :step] @ coefficients
        norm = float(la.norm(residual))
        if norm <= 1.0e-12:
            raise RuntimeError(f"coverage direction became dependent at step {step + 1}")
        new_q = residual / norm
        q_basis[:, step] = new_q
        selected_u.append(raw_u)
        overlaps = new_q @ all_w
        for index, block_slice in enumerate(slices):
            local = overlaps[block_slice]
            residual_grams[index] -= np.outer(local, local)
            residual_grams[index] = 0.5 * (
                residual_grams[index] + residual_grams[index].T
            )
        selection_rows.append(
            {
                "step": step + 1,
                "selected_training_system": sample_ids[block_index],
                "preselection_projector_defect": float(np.sqrt(scores[block_index])),
                "preselection_worst_angle_degrees": float(
                    np.degrees(np.arcsin(np.sqrt(np.clip(eigenvalues[-1], 0.0, 1.0))))
                ),
                "new_residual_norm": norm,
            }
        )

        rank = step + 1
        if rank in settings.ranks:
            raw_basis = np.column_stack(selected_u)
            basis, singular_values, actual_rank = _orthonormalize(raw_basis, 1.0e-13)
            if actual_rank != rank:
                raise RuntimeError(f"coverage p={rank} has numerical rank {actual_rank}")
            bases[rank] = basis
            block_d = []
            block_angles = []
            for local_gram in residual_grams:
                local_gram = 0.5 * (local_gram + local_gram.T)
                block_d.append(
                    float(np.sqrt(max(0.0, np.trace(local_gram) / local_gram.shape[0])))
                )
                largest = float(la.eigvalsh(local_gram, check_finite=False)[-1])
                block_angles.append(
                    float(np.degrees(np.arcsin(np.sqrt(np.clip(largest, 0.0, 1.0)))))
                )
            milestone_rows.append(
                {
                    "requested_rank": rank,
                    "actual_rank": actual_rank,
                    "training_median_projector_defect": float(np.median(block_d)),
                    "training_worst_projector_defect": float(np.max(block_d)),
                    "training_median_worst_angle_degrees": float(np.median(block_angles)),
                    "training_worst_angle_degrees": float(np.max(block_angles)),
                    "raw_solution_basis_condition": float(singular_values[0] / singular_values[-1]),
                }
            )
            np.savez_compressed(
                output / f"coverage_basis_p{rank}.npz", psi=basis, rank=np.asarray(rank)
            )

    checks = {
        "criterion": "max normalized residual projector defect, then least-covered principal direction",
        "training_system_count": len(sample_ids),
        "validation_teacher_overlap": sorted(set(split["train"]) & set(split["validation"])),
        "test_teacher_overlap": sorted(set(split["train"]) & set(split["test"])),
        "requested_ranks": list(settings.ranks),
        "actual_ranks": [int(bases[rank].shape[1]) for rank in settings.ranks],
        "nested_greedy_sequence": True,
        "maximum_training_pair_defect": max(pair_defects),
        "build_seconds": perf_counter() - started,
    }
    checks["passed"] = bool(
        checks["requested_ranks"] == checks["actual_ranks"]
        and not checks["validation_teacher_overlap"]
        and not checks["test_teacher_overlap"]
        and max(pair_defects) <= 1.0e-9
        and len(selection_rows) == max_rank
    )
    _write_csv(output / "coverage_selection.csv", selection_rows)
    _write_csv(output / "basis_metadata.csv", milestone_rows)
    _write_json(output / "preflight.json", checks)
    if not checks["passed"]:
        raise RuntimeError("Stage-33 coverage preflight failed; no held-out solve was run")
    return bases, milestone_rows, checks


def _archive_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage32 archived",
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


def run_comparison(
    root: Path,
    output: Path,
    bases: Mapping[int, np.ndarray],
    settings: Stage33Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage32 = root / "results/results-fd/stages/stage32_action_aware_rank_sweep"
    archived = _read_csv(stage32 / "solver_results.csv")
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    geometry32 = {
        (int(row["sample_id"]), row["method"]): row
        for row in _read_csv(stage32 / "geometry_diagnostics.csv")
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage32.stage31.stage30.stage29.stage28.stage19a)
    rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []

    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients,
            settings.stage32.stage31.stage30.stage29,
        )
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
        teacher_pair = _pair_from_z(context.system.matrix, zfull)
        selected = {}
        for rank in settings.ranks:
            started = perf_counter()
            pair, diagnostics = oracle_select_inside_dictionary(
                context.system.matrix, bases[rank], wfull
            )
            setup_seconds = perf_counter() - started
            if pair.rank != teacher_pair.rank:
                raise RuntimeError(
                    f"system {sample_id}, coverage p={rank}: selected rank mismatch"
                )
            selected[rank] = (pair, setup_seconds)
            metrics = generalized_subspace_metrics(pair.w_basis, teacher_pair.w_basis)
            geometry.append(
                {
                    "sample_id": sample_id,
                    "method": f"coverage_p{rank}",
                    "candidate_rank": rank,
                    "selected_rank": pair.rank,
                    "d_W": metrics["projector_distance_normalized"],
                    "maximum_W_angle_degrees": metrics["principal_angle_max_degrees"],
                    "mean_squared_principal_cosine": diagnostics["mean_squared_principal_cosine"],
                }
            )
            old = geometry32[(sample_id, f"action_p{rank}")]
            geometry.append(
                {
                    "sample_id": sample_id,
                    "method": f"action_p{rank}",
                    "candidate_rank": rank,
                    "selected_rank": int(old["selected_rank"]),
                    "d_W": float(old["d_W"]),
                    "maximum_W_angle_degrees": float(old["maximum_W_angle_degrees"]),
                    "mean_squared_principal_cosine": float(old["mean_squared_principal_cosine"]),
                }
            )
        old_full = geometry32[(sample_id, "full_span")]
        geometry.append(
            {
                "sample_id": sample_id,
                "method": "full_span",
                "candidate_rank": 1456,
                "selected_rank": int(old_full["selected_rank"]),
                "d_W": float(old_full["d_W"]),
                "maximum_W_angle_degrees": float(old_full["maximum_W_angle_degrees"]),
                "mean_squared_principal_cosine": float(old_full["mean_squared_principal_cosine"]),
            }
        )

        for rhs_seed in settings.stage32.stage31.stage30.stage29.stage28.evaluation_rhs_seeds:
            for method in (
                "strong_ilu", "action_p168", "action_p270", "full_span", "teacher_full"
            ):
                rows.append(_archive_row(lookup[(sample_id, rhs_seed, method)], method))
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            for rank in settings.ranks:
                pair, setup_seconds = selected[rank]
                result = _run_pair(
                    context, rhs, pair,
                    settings.stage32.stage31.stage30.stage29.stage28.stage22a,
                )
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": f"coverage_p{rank}",
                        "source": "Stage33 new",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and true_residual <= 1.0e-8),
                        "true_final_residual": true_residual,
                        "actual_rank": pair.rank,
                        "method_setup_seconds": setup_seconds,
                        "online_solve_seconds": float(result.solve_seconds),
                        "reduced_condition": pair.diagnostics["reduced_condition"],
                        "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                        "failure": getattr(result, "failure", "") or "",
                    }
                )
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for row in rows:
                if int(row["sample_id"]) == sample_id and int(row["rhs_seed"]) == rhs_seed:
                    row["rho"] = (
                        (int(baseline["iterations"]) - int(row["iterations"])) / denominator
                        if denominator > 0 else float("nan")
                    )
        print(f"[Stage33 {position}/7] system {sample_id} complete", flush=True)

    _write_csv(output / "solver_results.csv", rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    return rows, geometry


def summarize(
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    settings: Stage33Settings,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
]:
    methods = [
        *(f"action_p{rank}" for rank in settings.ranks),
        *(f"coverage_p{rank}" for rank in settings.ranks),
        "full_span", "teacher_full",
    ]
    solver_summary = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        solver_summary.append(
            {
                "method": method,
                "converged_rhs": int(sum(_bool(row["converged"]) for row in local)),
                "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                "median_rho": float(np.median([float(row["rho"]) for row in local])),
            }
        )
    geometry_summary = []
    for method in [
        *(f"action_p{rank}" for rank in settings.ranks),
        *(f"coverage_p{rank}" for rank in settings.ranks),
        "full_span",
    ]:
        local = [row for row in geometry if row["method"] == method]
        geometry_summary.append(
            {
                "method": method,
                "median_dW": float(np.median([float(row["d_W"]) for row in local])),
                "median_maximum_W_angle_degrees": float(np.median([float(row["maximum_W_angle_degrees"]) for row in local])),
                "worst_maximum_W_angle_degrees": float(np.max([float(row["maximum_W_angle_degrees"]) for row in local])),
            }
        )
    solver_lookup = {row["method"]: row for row in solver_summary}
    geometry_lookup = {row["method"]: row for row in geometry_summary}
    action = {rank: solver_lookup[f"action_p{rank}"] for rank in settings.ranks}
    coverage = {rank: solver_lookup[f"coverage_p{rank}"] for rank in settings.ranks}
    action_geometry = {
        rank: {"worst_angle": geometry_lookup[f"action_p{rank}"]["worst_maximum_W_angle_degrees"]}
        for rank in settings.ranks
    }
    coverage_geometry = {
        rank: {"worst_angle": geometry_lookup[f"coverage_p{rank}"]["worst_maximum_W_angle_degrees"]}
        for rank in settings.ranks
    }
    return solver_summary, geometry_summary, action, coverage, action_geometry, coverage_geometry


def classify_coverage(
    action: Mapping[int, Mapping[str, float]],
    coverage: Mapping[int, Mapping[str, float]],
    action_geometry: Mapping[int, Mapping[str, float]],
    coverage_geometry: Mapping[int, Mapping[str, float]],
) -> Dict[str, Any]:
    successful_ranks = []
    for rank in action:
        angle_gain = action_geometry[rank]["worst_angle"] - coverage_geometry[rank]["worst_angle"]
        convergence_gain = coverage[rank]["converged_rhs"] - action[rank]["converged_rhs"]
        rho_ok = coverage[rank]["median_rho"] >= action[rank]["median_rho"] - 0.02
        if angle_gain >= 5.0 and convergence_gain >= 1 and rho_ok:
            successful_ranks.append(rank)
    if successful_ranks:
        best = max(successful_ranks, key=lambda rank: (coverage[rank]["converged_rhs"], coverage[rank]["median_rho"], -rank))
        return {
            "conclusion": "coverage_succeeds",
            "statement": "Coverage-aware compression improves worst-direction geometry and hard-system convergence at matched rank.",
            "successful_ranks": successful_ranks,
            "recommended_basis_rank": best,
            "stage34_direction": f"Use the frozen coverage p={best} basis for the next oracle-blind coordinate-learning stage.",
        }
    return {
        "conclusion": "global_basis_insufficient",
        "statement": "Coverage-aware compression does not improve both worst-direction geometry and hard-system convergence at matched rank.",
        "successful_ranks": [],
        "recommended_basis_rank": None,
        "stage34_direction": "Move to a target-conditioned or local coarse-space construction instead of another single global shared basis.",
    }


def write_report(
    output: Path,
    solver_summary: Sequence[Mapping[str, Any]],
    geometry_summary: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage33Settings,
) -> Path:
    solver = {row["method"]: row for row in solver_summary}
    geom = {row["method"]: row for row in geometry_summary}
    lines = [
        "# Stage 33: coverage-aware shared-basis compression",
        "",
        "## Result",
        "",
        f"**{decision['statement']}**",
        "",
        "The new nested basis greedily targets the training teacher block with the largest residual projector defect and adds its least-covered principal direction. All target-side selection and solver settings are unchanged from Stage 32.",
        "",
        "| basis | converged RHS | median iterations | median $\\rho$ | median $d_W$ | median max angle | worst max angle |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for rank in settings.ranks:
        for prefix, label in (("action", "action-energy"), ("coverage", "coverage")):
            method = f"{prefix}_p{rank}"
            s, g = solver[method], geom[method]
            lines.append(
                f"| {label} p={rank} | {s['converged_rhs']}/21 | {_fmt(s['median_iterations'], 1)} | {_fmt(s['median_rho'])} | {_fmt(g['median_dW'])} | {_fmt(g['median_maximum_W_angle_degrees'], 1)} degrees | {_fmt(g['worst_maximum_W_angle_degrees'], 1)} degrees |"
            )
    full_s, full_g = solver["full_span"], geom["full_span"]
    teacher = solver["teacher_full"]
    lines.extend([
        f"| full span | {full_s['converged_rhs']}/21 | {_fmt(full_s['median_iterations'], 1)} | {_fmt(full_s['median_rho'])} | {_fmt(full_g['median_dW'])} | {_fmt(full_g['median_maximum_W_angle_degrees'], 1)} degrees | {_fmt(full_g['worst_maximum_W_angle_degrees'], 1)} degrees |",
        f"| own teacher | {teacher['converged_rhs']}/21 | {_fmt(teacher['median_iterations'], 1)} | {_fmt(teacher['median_rho'])} | 0.000 | 0.0 degrees | 0.0 degrees |",
        "",
        "## System 63",
        "",
        "| basis | converged RHS | median iterations | median $\\rho$ | $d_W$ | max angle |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry}
    for rank in settings.ranks:
        for prefix, label in (("action", "action-energy"), ("coverage", "coverage")):
            method = f"{prefix}_p{rank}"
            local = [row for row in rows if int(row["sample_id"]) == 63 and row["method"] == method]
            g = geometry_lookup[(63, method)]
            lines.append(
                f"| {label} p={rank} | {sum(_bool(row['converged']) for row in local)}/3 | {_fmt(np.median([int(row['iterations']) for row in local]), 1)} | {_fmt(np.median([float(row['rho']) for row in local]))} | {_fmt(g['d_W'])} | {_fmt(g['maximum_W_angle_degrees'], 1)} degrees |"
            )
    for method, label in (("full_span", "full span"), ("teacher_full", "own teacher")):
        local = [row for row in rows if int(row["sample_id"]) == 63 and row["method"] == method]
        if method == "full_span":
            g = geometry_lookup[(63, method)]
            dw, angle = _fmt(g["d_W"]), f"{_fmt(g['maximum_W_angle_degrees'], 1)} degrees"
        else:
            dw, angle = "0.000", "0.0 degrees"
        lines.append(
            f"| {label} | {sum(_bool(row['converged']) for row in local)}/3 | {_fmt(np.median([int(row['iterations']) for row in local]), 1)} | {_fmt(np.median([float(row['rho']) for row in local]))} | {dw} | {angle} |"
        )
    lines.extend([
        "",
        "## Conclusion and Stage 34",
        "",
        f"**Supported conclusion:** {decision['statement']}",
        "",
        f"**Stage 34:** {decision['stage34_direction']}",
    ])
    report = output / "stage33_coverage_aware_shared_basis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage33(root: Path) -> Path:
    settings = Stage33Settings()
    output = root / "results/results-fd/stages/stage33_coverage_aware_shared_basis"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-33 protocol must exist before new solves")
    _write_json(output / "settings.json", {
        "ranks": list(settings.ranks),
        "hard_system_ids": list(settings.hard_system_ids),
        "solver_source": "Stage32Settings -> Stage31 -> Stage30",
    })
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()
    })
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    bases, _, preflight = build_coverage_bases(root, output, split, settings)
    print(f"[Stage33 preflight] exact coverage ranks verified: {preflight['actual_ranks']}", flush=True)
    rows, geometry = run_comparison(root, output, bases, settings)
    solver_summary, geometry_summary, action, coverage, action_geometry, coverage_geometry = summarize(rows, geometry, settings)
    decision = classify_coverage(action, coverage, action_geometry, coverage_geometry)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(output, solver_summary, geometry_summary, rows, geometry, decision, settings)
    new_rows = [row for row in rows if row["source"] == "Stage33 new"]
    validation = {
        "preflight_passed": bool(preflight["passed"]),
        "requested_ranks": list(settings.ranks),
        "actual_ranks": preflight["actual_ranks"],
        "new_solver_rows": len(new_rows),
        "expected_new_solver_rows": 7 * 3 * 2,
        "maximum_new_pair_defect": max(float(row["au_equals_w_relative_error"]) for row in new_rows),
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in rows if _bool(row["converged"])),
        "no_neural_training": True,
    }
    validation["passed"] = bool(
        validation["preflight_passed"]
        and validation["requested_ranks"] == validation["actual_ranks"]
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["maximum_new_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage33_coverage_aware_shared_basis.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage33] complete", flush=True)
    return report_copy
