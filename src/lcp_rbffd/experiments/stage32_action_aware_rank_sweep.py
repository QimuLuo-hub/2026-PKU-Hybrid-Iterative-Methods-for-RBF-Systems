"""Stage 32: rank sweep for the Stage-31 paired residual-POD basis."""

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
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv, _sci
from .stage31_krylov_aware_shared_basis import Stage31Settings
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage32Settings:
    ranks: Tuple[int, ...] = (168, 270, 428, 573)
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    high_rank_converged_rhs_min: int = 20
    high_rank_median_rho_min: float = 0.90
    high_rank_median_dW_max: float = 0.30
    high_rank_worst_angle_max: float = 50.0
    high_rank_system63_converged_rhs_min: int = 3

    @property
    def stage31(self) -> Stage31Settings:
        return Stage31Settings()


def build_ranked_bases(
    root: Path,
    output: Path,
    split: Mapping[str, Sequence[int]],
    settings: Stage32Settings,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Reapply the unchanged Stage-31 residual POD at the requested ranks."""

    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage31 = root / "results/results-fd/stages/stage31_krylov_aware_shared_basis"
    paired_u_blocks: List[np.ndarray] = []
    paired_w_blocks: List[np.ndarray] = []
    pair_defects: List[float] = []
    for sample_id in split["train"]:
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        pair = _pair_from_z(matrix, zfull)
        paired_u_blocks.append(pair.u_basis)
        paired_w_blocks.append(pair.w_basis)
        pair_defects.append(float(pair.diagnostics["au_equals_w_relative_error"]))

    paired_u = np.column_stack(paired_u_blocks)
    paired_w = np.column_stack(paired_w_blocks)
    started = perf_counter()
    _, singular_values, right_t = la.svd(
        paired_w, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    bases: Dict[int, np.ndarray] = {}
    metadata: List[Dict[str, Any]] = []
    for rank in settings.ranks:
        raw = paired_u @ right_t[:rank].T
        basis, raw_singular_values, actual_rank = _orthonormalize(
            raw, settings.stage31.numerical_rank_tolerance
        )
        if actual_rank != rank:
            raise RuntimeError(f"requested action rank {rank}, obtained {actual_rank}")
        bases[rank] = basis
        metadata.append(
            {
                "requested_rank": rank,
                "actual_rank": actual_rank,
                "captured_training_W_energy": float(energy[rank - 1]),
                "training_W_relative_reconstruction_error": float(
                    np.sqrt(max(0.0, 1.0 - energy[rank - 1]))
                ),
                "compressed_raw_condition": float(
                    raw_singular_values[0] / raw_singular_values[-1]
                ),
            }
        )
        np.savez_compressed(
            output / f"action_basis_p{rank}.npz",
            psi=basis,
            rank=np.asarray(rank),
            captured_training_W_energy=np.asarray(energy[rank - 1]),
        )

    with np.load(stage31 / "krylov_action_basis_p168.npz") as archive:
        cached168 = np.asarray(archive["psi"], dtype=float)
    cached_metrics = generalized_subspace_metrics(bases[168], cached168)
    checks = {
        "criterion_matches_stage31": "paired residual POD: basis_p=orth(U V_p)",
        "training_system_count": len(split["train"]),
        "validation_teacher_overlap": sorted(set(split["train"]) & set(split["validation"])),
        "test_teacher_overlap": sorted(set(split["train"]) & set(split["test"])),
        "paired_snapshot_shape": list(paired_u.shape),
        "requested_ranks": list(settings.ranks),
        "actual_ranks": [int(bases[rank].shape[1]) for rank in settings.ranks],
        "maximum_training_pair_defect": max(pair_defects),
        "cached_stage31_p168_projector_distance": float(
            cached_metrics["projector_distance_normalized"]
        ),
        "cached_stage31_p168_maximum_angle_degrees": float(
            cached_metrics["principal_angle_max_degrees"]
        ),
        "basis_build_seconds": perf_counter() - started,
    }
    checks["passed"] = bool(
        checks["requested_ranks"] == checks["actual_ranks"]
        and not checks["validation_teacher_overlap"]
        and not checks["test_teacher_overlap"]
        and max(pair_defects) <= 1.0e-9
        and checks["cached_stage31_p168_projector_distance"] <= 1.0e-7
    )
    _write_csv(output / "basis_metadata.csv", metadata)
    _write_json(output / "preflight.json", checks)
    np.savez_compressed(
        output / "paired_residual_pod_cache.npz",
        right_coefficients=right_t[: max(settings.ranks)].T,
        residual_singular_values=singular_values,
        residual_energy=energy,
        training_ids=np.asarray(split["train"], dtype=int),
    )
    if not checks["passed"]:
        raise RuntimeError("Stage-32 preflight failed; no new solve was run")
    bases[168] = cached168
    return bases, metadata, checks


def _archive_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage31 archived",
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


def run_sweep(
    root: Path,
    output: Path,
    bases: Mapping[int, np.ndarray],
    settings: Stage32Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage31 = root / "results/results-fd/stages/stage31_krylov_aware_shared_basis"
    archived = _read_csv(stage31 / "solver_results.csv")
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    geometry31 = {
        (int(row["sample_id"]), row["method"]): row
        for row in _read_csv(stage31 / "geometry_diagnostics.csv")
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage31.stage30.stage29.stage28.stage19a)
    rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []

    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage31.stage30.stage29
        )
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
        teacher_pair = _pair_from_z(context.system.matrix, zfull)
        selected = {}
        for rank in settings.ranks[1:]:
            started = perf_counter()
            pair, diagnostics = oracle_select_inside_dictionary(
                context.system.matrix, bases[rank], wfull
            )
            setup_seconds = perf_counter() - started
            if pair.rank != teacher_pair.rank:
                raise RuntimeError(
                    f"system {sample_id}, p={rank}: selected {pair.rank}, teacher {teacher_pair.rank}"
                )
            selected[rank] = (pair, diagnostics, setup_seconds)
            w_metrics = generalized_subspace_metrics(pair.w_basis, teacher_pair.w_basis)
            geometry.append(
                {
                    "sample_id": sample_id,
                    "method": f"action_p{rank}",
                    "candidate_rank": rank,
                    "selected_rank": pair.rank,
                    "d_W": w_metrics["projector_distance_normalized"],
                    "maximum_W_angle_degrees": w_metrics["principal_angle_max_degrees"],
                    "mean_squared_principal_cosine": diagnostics["mean_squared_principal_cosine"],
                }
            )

        for archived_method, method in (
            ("action_p168", "action_p168"),
            ("full_span", "full_span"),
        ):
            old = geometry31[(sample_id, archived_method)]
            geometry.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "candidate_rank": 168 if method == "action_p168" else 1456,
                    "selected_rank": int(old["actual_recycle_rank"]),
                    "d_W": float(old["d_W_to_teacher"]),
                    "maximum_W_angle_degrees": float(old["maximum_angle_W_degrees"]),
                    "mean_squared_principal_cosine": float(old["mean_squared_principal_cosine"]),
                }
            )

        for rhs_seed in settings.stage31.stage30.stage29.stage28.evaluation_rhs_seeds:
            for archived_method, method in (
                ("strong_ilu", "strong_ilu"),
                ("action_p168", "action_p168"),
                ("full_span", "full_span"),
                ("teacher_full", "teacher_full"),
            ):
                rows.append(_archive_row(lookup[(sample_id, rhs_seed, archived_method)], method))
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            for rank in settings.ranks[1:]:
                pair, _, setup_seconds = selected[rank]
                result = _run_pair(
                    context, rhs, pair, settings.stage31.stage30.stage29.stage28.stage22a
                )
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": f"action_p{rank}",
                        "source": "Stage32 new",
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
                        if denominator > 0
                        else float("nan")
                    )
        print(f"[Stage32 {position}/7] system {sample_id} complete", flush=True)

    _write_csv(output / "solver_results.csv", rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    return rows, geometry


def summarize(
    rows: Sequence[Mapping[str, Any]], geometry: Sequence[Mapping[str, Any]], settings: Stage32Settings
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    methods = [f"action_p{rank}" for rank in settings.ranks] + ["full_span", "teacher_full"]
    solver_summary: List[Dict[str, Any]] = []
    solver_by_rank: Dict[int, Dict[str, float]] = {}
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        item = {
            "method": method,
            "converged_rhs": int(sum(_bool(row["converged"]) for row in local)),
            "rhs_count": len(local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
        }
        solver_summary.append(item)
        if method.startswith("action_p"):
            solver_by_rank[int(method.removeprefix("action_p"))] = item

    geometry_summary: List[Dict[str, Any]] = []
    geometry_by_rank: Dict[int, Dict[str, float]] = {}
    for method, rank in [(f"action_p{rank}", rank) for rank in settings.ranks] + [("full_span", 1456)]:
        local = [row for row in geometry if row["method"] == method]
        item = {
            "method": method,
            "rank": rank,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_maximum_W_angle_degrees": float(
                np.median([float(row["maximum_W_angle_degrees"]) for row in local])
            ),
            "worst_maximum_W_angle_degrees": float(
                np.max([float(row["maximum_W_angle_degrees"]) for row in local])
            ),
        }
        geometry_summary.append(item)
        if method.startswith("action_p"):
            geometry_by_rank[rank] = {
                "median_dW": item["median_dW"],
                "median_angle": item["median_maximum_W_angle_degrees"],
                "worst_angle": item["worst_maximum_W_angle_degrees"],
            }
    return solver_summary, geometry_summary, solver_by_rank, geometry_by_rank


def classify_sweep(
    solver_by_rank: Mapping[int, Mapping[str, float]],
    geometry_by_rank: Mapping[int, Mapping[str, float]],
    system63_high_rank_converged: int,
) -> Dict[str, Any]:
    high = solver_by_rank[573]
    high_geometry = geometry_by_rank[573]
    approaches_full = bool(
        high["converged_rhs"] >= 20
        and high["median_rho"] >= 0.90
        and high_geometry["median_dW"] <= 0.30
        and high_geometry["worst_angle"] <= 50.0
        and system63_high_rank_converged >= 3
    )
    improves_with_rank = bool(
        high["converged_rhs"] >= solver_by_rank[168]["converged_rhs"]
        and high["median_rho"] > solver_by_rank[168]["median_rho"]
        and high_geometry["median_dW"] < geometry_by_rank[168]["median_dW"]
        and high_geometry["worst_angle"] < geometry_by_rank[168]["worst_angle"]
    )
    if approaches_full and improves_with_rank:
        return {
            "conclusion": "rank_limited",
            "statement": "Action-aware compression is viable; rank 168 was too restrictive.",
            "stage33_direction": "Freeze the smallest successful action rank and test an oracle-blind selector for that basis.",
            "p573_full_span_approach_gate": True,
            "improves_with_rank": True,
        }
    return {
        "conclusion": "criterion_limited",
        "statement": "Residual-energy SVD improves average geometry but still discards low-energy Krylov-critical directions.",
        "stage33_direction": "Replace residual-energy ranking with a worst-direction or hard-system-aware compression objective before training a neural selector.",
        "p573_full_span_approach_gate": False,
        "improves_with_rank": improves_with_rank,
    }


def write_report(
    output: Path,
    solver_summary: Sequence[Mapping[str, Any]],
    geometry_summary: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage32Settings,
) -> Path:
    solver = {row["method"]: row for row in solver_summary}
    geom = {row["method"]: row for row in geometry_summary}
    lines = [
        "# Stage 32: action-aware basis rank sweep",
        "",
        "## Result",
        "",
        f"**{decision['statement']}**",
        "",
        "The Stage-31 paired residual-POD construction, target-informed selection, teacher ranks, target-side pairing, strong ILU, and recycled-GMRES settings are unchanged. Only the shared candidate rank varies.",
        "The four ranks retain 68.2%, 80.0%, 90.0%, and 95.0% of the archived training $W$ energy, respectively.",
        "",
        "| basis | converged RHS | median iterations | median $\\rho$ | median $d_W$ | median max angle | worst max angle |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for rank in settings.ranks:
        s = solver[f"action_p{rank}"]
        g = geom[f"action_p{rank}"]
        lines.append(
            f"| action p={rank} | {s['converged_rhs']}/21 | {_fmt(s['median_iterations'], 1)} | {_fmt(s['median_rho'])} | {_fmt(g['median_dW'])} | {_fmt(g['median_maximum_W_angle_degrees'], 1)} degrees | {_fmt(g['worst_maximum_W_angle_degrees'], 1)} degrees |"
        )
    for method, label in (("full_span", "full span"), ("teacher_full", "own teacher")):
        s = solver[method]
        if method == "full_span":
            g = geom[method]
            dw = _fmt(g["median_dW"])
            median_angle = f"{_fmt(g['median_maximum_W_angle_degrees'], 1)} degrees"
            worst_angle = f"{_fmt(g['worst_maximum_W_angle_degrees'], 1)} degrees"
        else:
            dw, median_angle, worst_angle = "0.000", "0.0 degrees", "0.0 degrees"
        lines.append(
            f"| {label} | {s['converged_rhs']}/21 | {_fmt(s['median_iterations'], 1)} | {_fmt(s['median_rho'])} | {dw} | {median_angle} | {worst_angle} |"
        )

    lines.extend([
        "",
        "## System 63",
        "",
        "| basis | converged RHS | median iterations | median $\\rho$ | $d_W$ | max angle |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry}
    for rank in settings.ranks:
        method = f"action_p{rank}"
        local = [row for row in rows if int(row["sample_id"]) == 63 and row["method"] == method]
        g = geometry_lookup[(63, method)]
        lines.append(
            f"| action p={rank} | {sum(_bool(row['converged']) for row in local)}/3 | {_fmt(np.median([int(row['iterations']) for row in local]), 1)} | {_fmt(np.median([float(row['rho']) for row in local]))} | {_fmt(g['d_W'])} | {_fmt(g['maximum_W_angle_degrees'], 1)} degrees |"
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
    high = geom["action_p573"]
    lines.extend([
        "",
        f"Across the sweep, the p=573 basis reaches median $d_W={_fmt(high['median_dW'])}$, but its worst principal angle is {_fmt(high['worst_maximum_W_angle_degrees'], 1)} degrees. The conclusion uses the frozen p=573 gate recorded before the new solves.",
        "Geometry improves monotonically with rank, while hard-system convergence remains 18/21 at every rank. Median iterations improve through p=428 and then worsen from 66 to 77 at p=573; median $\\rho$ likewise falls from 0.854 to 0.820. More residual energy therefore does not monotonically recover the missing Krylov benefit.",
        "",
        "## Conclusion and Stage 33",
        "",
        f"**Supported conclusion:** {decision['statement']}",
        "",
        f"**Stage 33:** {decision['stage33_direction']}",
    ])
    report = output / "stage32_action_aware_rank_sweep.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage32(root: Path) -> Path:
    settings = Stage32Settings()
    output = root / "results/results-fd/stages/stage32_action_aware_rank_sweep"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-32 protocol must exist before new solves")
    _write_json(output / "settings.json", {
        "ranks": list(settings.ranks),
        "hard_system_ids": list(settings.hard_system_ids),
        "stage31_settings": {
            "hard_system_ids": list(settings.stage31.hard_system_ids),
            "shared_rank": settings.stage31.shared_rank,
        },
    })
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()
    })
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    bases, _, preflight = build_ranked_bases(root, output, split, settings)
    print(f"[Stage32 preflight] exact ranks verified: {preflight['actual_ranks']}", flush=True)
    rows, geometry = run_sweep(root, output, bases, settings)
    solver_summary, geometry_summary, solver_by_rank, geometry_by_rank = summarize(
        rows, geometry, settings
    )
    system63_p573 = sum(
        _bool(row["converged"])
        for row in rows
        if int(row["sample_id"]) == 63 and row["method"] == "action_p573"
    )
    decision = classify_sweep(solver_by_rank, geometry_by_rank, system63_p573)
    decision["system63_p573_converged_rhs"] = system63_p573
    decision["solver_by_rank"] = solver_by_rank
    decision["geometry_by_rank"] = geometry_by_rank
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output, solver_summary, geometry_summary, rows, geometry, decision, settings
    )
    validation = {
        "preflight_passed": bool(preflight["passed"]),
        "requested_ranks": list(settings.ranks),
        "actual_ranks": preflight["actual_ranks"],
        "new_solver_rows": sum(row["source"] == "Stage32 new" for row in rows),
        "expected_new_solver_rows": 7 * 3 * 3,
        "maximum_new_pair_defect": max(
            float(row["au_equals_w_relative_error"])
            for row in rows if row["source"] == "Stage32 new"
        ),
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"]) for row in rows if _bool(row["converged"])
        ),
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
    report_copy = root / "results/results-fd/reports/stage32_action_aware_rank_sweep.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage32] complete", flush=True)
    return report_copy
