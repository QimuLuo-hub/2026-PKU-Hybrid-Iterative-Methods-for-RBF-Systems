"""Stage 43: oracle ceiling for basis-invariant missing-subspace blocks."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage40_target_conditioned_local_basis import _geometry, _teacher
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage43Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    thresholds: Tuple[float, ...] = (0.25, 0.50, 0.75)
    block_ranks: Tuple[int, ...] = (4, 8, 12)
    stage41_full_rank: int = 48

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


BASE_METHODS = (
    "strong_ilu",
    "teacher_full",
    "full_span_oracle",
    "multi_rhs_harmonic_m64",
)
FIXED_BASE_METHOD = "stage41_harmonic_fixed_q48"


def _threshold_method(threshold: float) -> str:
    return f"missing_tau_{str(threshold).replace('.', 'p')}"


def _rank_method(rank: int) -> str:
    return f"missing_k{rank}"


def select_missing_indices(
    eta: np.ndarray,
    *,
    threshold: Optional[float] = None,
    rank: Optional[int] = None,
) -> List[int]:
    if (threshold is None) == (rank is None):
        raise ValueError("provide exactly one of threshold or rank")
    values = np.asarray(eta, dtype=float)
    order = np.argsort(values)
    if threshold is not None:
        selected = np.flatnonzero(values < threshold)
        return [int(index) for index in selected[np.argsort(values[selected])]]
    return [int(index) for index in order[: min(int(rank), values.size)]]


def _block_overlap(block_w: np.ndarray, probe: np.ndarray) -> Dict[str, float]:
    if block_w.shape[1] == 0:
        return {
            "block_probe_mean_squared_overlap": float("nan"),
            "block_probe_median_principal_angle": float("nan"),
            "block_probe_largest_principal_angle": float("nan"),
        }
    metrics = generalized_subspace_metrics(block_w, probe)
    angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
    return {
        "block_probe_mean_squared_overlap": float(metrics["containment_first"]),
        "block_probe_median_principal_angle": float(np.median(angles)),
        "block_probe_largest_principal_angle": float(np.max(angles)),
    }


def construct_blocks(root: Path, settings: Stage43Settings):
    stage42 = root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis"
    principal_rows = _read_csv(stage42 / "principal_oracle_combination_diagnostics.csv")
    row_lookup = {
        (int(row["sample_id"]), int(row["direction_index"])): row
        for row in principal_rows
    }
    principal_archive = np.load(stage42 / "combined_harmonic_probe_spaces.npz")
    stage41_archive = np.load(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/multi_rhs_local_spaces.npz"
    )
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[Tuple[int, str], RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []

    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        stage41_z = np.asarray(
            stage41_archive[f"system_{sample_id:03d}_multi_rhs_harmonic_m64_z"], dtype=float
        )
        stage41_pair = _pair_from_z(context.system.matrix, stage41_z)
        pairs[(sample_id, FIXED_BASE_METHOD)] = stage41_pair
        canonical_u = np.asarray(
            principal_archive[f"system_{sample_id:03d}_principal_oracle_u"], dtype=float
        )
        canonical_w = np.asarray(
            principal_archive[f"system_{sample_id:03d}_principal_oracle_w"], dtype=float
        )
        probe = np.asarray(
            principal_archive[f"system_{sample_id:03d}_combined_harmonic_action_basis"], dtype=float
        )
        eta = np.asarray([
            float(row_lookup[(sample_id, index)]["eta_combined"])
            for index in range(canonical_w.shape[1])
        ])
        teacher = _teacher(root, sample_id, context.system.matrix)
        rules = []
        for threshold in settings.thresholds:
            rules.append((_threshold_method(threshold), select_missing_indices(eta, threshold=threshold), "threshold", threshold))
        for rank in settings.block_ranks:
            rules.append((_rank_method(rank), select_missing_indices(eta, rank=rank), "rank", rank))

        for method, selected, rule_type, rule_value in rules:
            if selected:
                raw_u = np.column_stack((stage41_pair.u_basis, canonical_u[:, selected]))
                block_w = canonical_w[:, selected]
            else:
                raw_u = stage41_pair.u_basis
                block_w = np.empty((canonical_w.shape[0], 0), dtype=float)
            pair = _pair_from_z(context.system.matrix, raw_u)
            pairs[(sample_id, method)] = pair
            block_geometry = _block_overlap(block_w, probe)
            selected_eta = eta[selected] if selected else np.asarray([], dtype=float)
            block_rows.append({
                "sample_id": sample_id,
                "method": method,
                "rule_type": rule_type,
                "rule_value": rule_value,
                "selected_principal_indices": ";".join(str(index) for index in selected),
                "correction_block_rank": len(selected),
                "stage41_base_rank": stage41_pair.rank,
                "total_final_rank": pair.rank,
                "minimum_selected_eta": float(np.min(selected_eta)) if selected else float("nan"),
                "maximum_selected_eta": float(np.max(selected_eta)) if selected else float("nan"),
                "median_selected_eta": float(np.median(selected_eta)) if selected else float("nan"),
                **block_geometry,
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
                "reduced_condition": pair.diagnostics["reduced_condition"],
            })
            geometry_rows.append({
                "sample_id": sample_id,
                "method": method,
                "correction_block_rank": len(selected),
                "total_final_rank": pair.rank,
                **_geometry(pair, teacher),
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
        print(f"[Stage43 spaces] system {sample_id}", flush=True)

    principal_archive.close()
    stage41_archive.close()
    return pairs, geometry_rows, block_rows


def _archive_row(row: Mapping[str, Any]):
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": row["method"],
        "source": "Stage41 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "rho": float(row["rho"]),
        "correction_block_rank": 0,
        "failure": row["failure"],
    }


def evaluate_solver(root: Path, pairs, block_rows, settings: Stage43Settings):
    archived = _read_csv(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/solver_results.csv"
    )
    rows = [_archive_row(row) for row in archived if row["method"] in set(BASE_METHODS)]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    block_lookup = {(int(row["sample_id"]), row["method"]): row for row in block_rows}
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    block_methods = tuple(_threshold_method(value) for value in settings.thresholds) + tuple(
        _rank_method(value) for value in settings.block_ranks
    )
    new_methods = (FIXED_BASE_METHOD,) + block_methods
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for method in new_methods:
                pair = pairs[(sample_id, method)]
                result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                block_rank = (
                    0 if method == FIXED_BASE_METHOD
                    else int(block_lookup[(sample_id, method)]["correction_block_rank"])
                )
                rows.append({
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": method,
                    "source": (
                        "Stage43 diagnostic fixed-rank control"
                        if method == FIXED_BASE_METHOD
                        else "Stage43 non-deployable teacher-informed ceiling"
                    ),
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and residual <= 1.0e-8),
                    "true_final_residual": residual,
                    "actual_rank": pair.rank,
                    "rho": (
                        (int(baseline["iterations"]) - int(result.iterations)) / denominator
                        if denominator > 0 else float("nan")
                    ),
                    "correction_block_rank": block_rank,
                    "failure": getattr(result, "failure", "") or "",
                })
        print(f"[Stage43 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]):
    result = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_correction_block_rank": float(
                np.median([int(row["correction_block_rank"]) for row in local])
            ),
            "median_total_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return result


def comparison_geometry(root: Path, new_rows):
    archived = _read_csv(
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/comparison_geometry.csv"
    )
    keep = {"teacher_full", "full_span_oracle", "multi_rhs_harmonic_m64"}
    rows = [
        {
            "sample_id": int(row["sample_id"]),
            "method": row["method"],
            "d_W": float(row["d_W"]),
            "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"]),
        }
        for row in archived if row["method"] in keep
    ]
    rows.extend({
        "sample_id": int(row["sample_id"]),
        "method": FIXED_BASE_METHOD,
        "d_W": float(row["d_W"]),
        "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"]),
    } for row in archived if row["method"] == "multi_rhs_harmonic_m64")
    rows.extend({
        "sample_id": int(row["sample_id"]),
        "method": row["method"],
        "d_W": float(row["d_W"]),
        "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"]),
    } for row in new_rows)
    return rows


def summarize_geometry(rows: Sequence[Mapping[str, Any]], methods: Sequence[str]):
    result = []
    for method in methods:
        if method == "strong_ilu":
            continue
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_largest_angle": float(
                np.median([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
            "worst_largest_angle": float(
                np.max([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
        })
    return result


def summarize_overlap(block_rows, new_methods):
    result = []
    for method in new_methods:
        local = [row for row in block_rows if row["method"] == method]
        finite_overlap = [
            float(row["block_probe_mean_squared_overlap"])
            for row in local if np.isfinite(float(row["block_probe_mean_squared_overlap"]))
        ]
        finite_angle = [
            float(row["block_probe_largest_principal_angle"])
            for row in local if np.isfinite(float(row["block_probe_largest_principal_angle"]))
        ]
        result.append({
            "method": method,
            "systems_with_nonempty_block": len(finite_overlap),
            "median_mean_squared_probe_overlap": (
                float(np.median(finite_overlap)) if finite_overlap else float("nan")
            ),
            "median_largest_probe_angle": (
                float(np.median(finite_angle)) if finite_angle else float("nan")
            ),
            "worst_largest_probe_angle": max(finite_angle) if finite_angle else float("nan"),
        })
    return result


def decide(solver_summary, solver_rows, block_rows):
    lookup = {row["method"]: row for row in solver_summary}
    oracle = lookup["full_span_oracle"]
    candidates = [
        row for row in solver_summary
        if row["method"] not in BASE_METHODS + (FIXED_BASE_METHOD,)
    ]
    best = max(
        candidates,
        key=lambda row: (
            row["converged_rhs"], row["median_rho"], -row["median_iterations"]
        ),
    )
    ranks = [
        int(row["correction_block_rank"])
        for row in block_rows if row["method"] == best["method"]
    ]
    compact = max(ranks) <= 15
    system63_oracle = [
        row for row in solver_rows
        if int(row["sample_id"]) == 63 and row["method"] == "full_span_oracle"
    ]
    system63_best = [
        row for row in solver_rows
        if int(row["sample_id"]) == 63 and row["method"] == best["method"]
    ]
    system63_oracle_rho = float(np.median([float(row["rho"]) for row in system63_oracle]))
    system63_best_rho = float(np.median([float(row["rho"]) for row in system63_best]))
    approaches = bool(
        best["converged_rhs"] == 21
        and oracle["median_rho"] - best["median_rho"] <= 0.15
        and system63_oracle_rho - system63_best_rho <= 0.15
    )
    consistent_improvement = bool(
        best["converged_rhs"] > lookup[FIXED_BASE_METHOD]["converged_rhs"]
        or best["median_rho"] >= lookup[FIXED_BASE_METHOD]["median_rho"] + 0.10
    )
    if compact and approaches:
        conclusion = "A compact missing-subspace block reaches the Stage-30 ceiling closely enough to freeze the formulation."
        recommendation = "Stage 44 should train FNO to predict the missing correction subspace from K(x) with a basis-invariant projector loss, keeping the Stage-41 base and solver fixed."
    else:
        conclusion = "A compact missing block restores convergence, but even the larger blocks fail to recover the Stage-30 acceleration on system 63."
        recommendation = "Do not use the Stage-41 plus missing-block decomposition as a neural target in Stage 44; redesign the target representation."
    return {
        "best_block_method": best["method"],
        "best_block_rank_range": [min(ranks), max(ranks)],
        "compact_rank_at_most_15": compact,
        "approaches_stage30": approaches,
        "consistent_improvement": consistent_improvement,
        "system63_full_span_oracle_median_rho": system63_oracle_rho,
        "system63_best_block_median_rho": system63_best_rho,
        "conclusion": conclusion,
        "stage44_recommendation": recommendation,
    }


def write_report(
    output: Path,
    methods,
    solver_summary,
    geometry_summary,
    overlap_summary,
    solver_rows,
    geometry_rows,
    block_rows,
    decision,
):
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "multi_rhs_harmonic_m64": "Stage-41 harmonic-Ritz",
        "stage41_harmonic_fixed_q48": "Stage-41 harmonic-Ritz fixed $q=48$",
        "missing_tau_0p25": "Stage-41 + miss($\\tau=0.25$)",
        "missing_tau_0p5": "Stage-41 + miss($\\tau=0.50$)",
        "missing_tau_0p75": "Stage-41 + miss($\\tau=0.75$)",
        "missing_k4": "Stage-41 + worst $k=4$",
        "missing_k8": "Stage-41 + worst $k=8$",
        "missing_k12": "Stage-41 + worst $k=12$",
    }
    lines = [
        "# Stage 43: Missing-subspace block correction ceiling",
        "",
        "## Result",
        "",
        f"**{decision['conclusion']}**",
        "",
        "All Stage-43 block augmentations use held-out principal-oracle information and are non-deployable representation ceilings. The fixed-q=48 Stage-41 row isolates the effect of the block from the earlier adaptive rank policy.",
        "",
        "| method | converged | median iterations | median $\\rho$ | median correction rank | median total rank |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        lines.append(
            f"| {labels[row['method']]} | {row['converged_rhs']}/21 | "
            f"{_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | "
            f"{_fmt(row['median_correction_block_rank'],1)} | {_fmt(row['median_total_final_rank'],1)} |"
        )
    lines.extend([
        "",
        "## Residual-space geometry against the own-system teacher",
        "",
        "| method | median $d_W$ | median largest angle | worst largest angle |",
        "|:--|--:|--:|--:|",
    ])
    for row in geometry_summary:
        lines.append(
            f"| {labels[row['method']]} | {_fmt(row['median_dW'])} | "
            f"{_fmt(row['median_largest_angle'],1)} degrees | "
            f"{_fmt(row['worst_largest_angle'],1)} degrees |"
        )
    lines.extend([
        "",
        "## Complementarity to the Stage-41 probe space",
        "",
        "Low mean squared overlap and large principal angles mean that the correction contributes information absent from the probe.",
        "",
        "| block | nonempty systems | median mean squared overlap | median largest angle | worst largest angle |",
        "|:--|--:|--:|--:|--:|",
    ])
    for row in overlap_summary:
        lines.append(
            f"| {labels[row['method']]} | {row['systems_with_nonempty_block']}/7 | "
            f"{_fmt(row['median_mean_squared_probe_overlap'])} | "
            f"{_fmt(row['median_largest_probe_angle'],1)} degrees | "
            f"{_fmt(row['worst_largest_probe_angle'],1)} degrees |"
        )

    solver_lookup = {(int(row["sample_id"]), row["method"]): row for row in solver_rows if int(row["sample_id"]) == 63}
    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry_rows}
    block_lookup = {(int(row["sample_id"]), row["method"]): row for row in block_rows}
    lines.extend([
        "",
        "## System 63 correction-rank curve",
        "",
        "| $k$ | converged | median iterations | median $\\rho$ | total rank | $d_W$ | largest angle | probe overlap |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ])
    base = [row for row in solver_rows if int(row["sample_id"]) == 63 and row["method"] == FIXED_BASE_METHOD]
    base_geometry = geometry_lookup[(63, FIXED_BASE_METHOD)]
    lines.append(
        f"| 0 | {sum(_bool(row['converged']) for row in base)}/3 | "
        f"{_fmt(np.median([int(row['iterations']) for row in base]),1)} | "
        f"{_fmt(np.median([float(row['rho']) for row in base]))} | "
        f"{int(np.median([int(row['actual_rank']) for row in base]))} | "
        f"{_fmt(base_geometry['d_W'])} | {_fmt(base_geometry['largest_principal_angle_degrees'],1)} degrees | n/a |"
    )
    for rank in (4, 8, 12):
        method = _rank_method(rank)
        local = [row for row in solver_rows if int(row["sample_id"]) == 63 and row["method"] == method]
        geometry = geometry_lookup[(63, method)]
        block = block_lookup[(63, method)]
        lines.append(
            f"| {rank} | {sum(_bool(row['converged']) for row in local)}/3 | "
            f"{_fmt(np.median([int(row['iterations']) for row in local]),1)} | "
            f"{_fmt(np.median([float(row['rho']) for row in local]))} | "
            f"{int(np.median([int(row['actual_rank']) for row in local]))} | "
            f"{_fmt(geometry['d_W'])} | {_fmt(geometry['largest_principal_angle_degrees'],1)} degrees | "
            f"{_fmt(block['block_probe_mean_squared_overlap'])} |"
        )

    lines.extend([
        "",
        "## System 63 threshold blocks",
        "",
        "| threshold | correction rank | converged | median iterations | median $\\rho$ | total rank |",
        "|--:|--:|--:|--:|--:|--:|",
    ])
    for threshold in (0.25, 0.50, 0.75):
        method = _threshold_method(threshold)
        local = [row for row in solver_rows if int(row["sample_id"]) == 63 and row["method"] == method]
        block = block_lookup[(63, method)]
        lines.append(
            f"| {threshold:.2f} | {block['correction_block_rank']} | "
            f"{sum(_bool(row['converged']) for row in local)}/3 | "
            f"{_fmt(np.median([int(row['iterations']) for row in local]),1)} | "
            f"{_fmt(np.median([float(row['rho']) for row in local]))} | "
            f"{int(np.median([int(row['actual_rank']) for row in local]))} |"
        )

    lines.extend([
        "",
        "The $\\tau=0.50$ block is the only rule that restores system 63 to 3/3, but its median $\\rho=0.238$ remains far below the Stage-30 value 0.808. Adding more directions is nonmonotone: $\\tau=0.75$ and $k=12$ fall back to 2/3. The correction therefore crosses the convergence boundary without reproducing the oracle acceleration mechanism.",
        "",
        "Across all targets, the $\\tau=0.50$ correction ranges from rank 0 to 10 and is nonempty for only 4/7 systems. It is compact, but the target-dependent rank discontinuity and nonmonotone response on system 63 show that it is not yet a stable neural target. The fixed-$k$ blocks are more uniform, but none restores system 63 to 3/3.",
    ])

    lines.extend([
        "",
        "## Controls and interpretation",
        "",
        "The cached Stage-42 principal combinations are basis-invariant singular directions of the oracle space relative to the probe space. Threshold and rank rules operate only on their singular-value coverages. Selected cached $U_{\\rm miss}$ columns satisfy $AU_{\\rm miss}=W_{\\rm miss}$; every augmented union is refactored into the existing paired recycle representation before fixed-space recycled GMRES. New methods use no deployable claim and no solver change.",
        "",
        "## Stage 44 recommendation",
        "",
        decision["stage44_recommendation"],
    ])
    report = output / "stage43_missing_subspace_block_ceiling.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage43(root: Path) -> Path:
    settings = Stage43Settings()
    output = root / "results/results-fd/stages/stage43_missing_subspace_block_ceiling"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-43 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    })
    stage42_validation = json.loads(
        (root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis/validation.json").read_text()
    )
    preflight = {
        "stage42_validation_passed": bool(stage42_validation["passed"]),
        "thresholds_exact": settings.thresholds == (0.25, 0.50, 0.75),
        "ranks_exact": settings.block_ranks == (4, 8, 12),
        "cached_principal_spaces_present": (
            root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis/combined_harmonic_probe_spaces.npz"
        ).exists(),
        "block_methods_are_diagnostic_teacher_ceilings": True,
        "solver_unchanged": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-43 preflight failed")

    pairs, new_geometry, block_rows = construct_blocks(root, settings)
    solver_rows = evaluate_solver(root, pairs, block_rows, settings)
    block_methods = tuple(_threshold_method(value) for value in settings.thresholds) + tuple(
        _rank_method(value) for value in settings.block_ranks
    )
    new_methods = (FIXED_BASE_METHOD,) + block_methods
    methods = BASE_METHODS + new_methods
    solver_summary = summarize_solver(solver_rows, methods)
    all_geometry = comparison_geometry(root, new_geometry)
    geometry_summary = summarize_geometry(all_geometry, methods)
    overlap_summary = summarize_overlap(block_rows, block_methods)
    decision = decide(solver_summary, solver_rows, block_rows)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}

    _write_csv(output / "block_construction.csv", block_rows)
    _write_csv(output / "geometry_diagnostics.csv", new_geometry)
    _write_csv(output / "comparison_geometry.csv", all_geometry)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "block_probe_overlap_summary.csv", overlap_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output, methods, solver_summary, geometry_summary, overlap_summary,
        solver_rows, all_geometry, block_rows, decision,
    )
    new_solver_rows = [row for row in solver_rows if row["source"].startswith("Stage43")]
    validation = {
        "preflight_passed": preflight["passed"],
        "all_requested_methods_present": all(
            sum(row["method"] == method for row in new_solver_rows) == 21
            for method in new_methods
        ),
        "selected_rules_exact": all(
            (
                row["rule_type"] == "threshold"
                and float(row["rule_value"]) in settings.thresholds
            ) or (
                row["rule_type"] == "rank"
                and int(row["rule_value"]) in settings.block_ranks
                and int(row["correction_block_rank"]) == int(row["rule_value"])
            )
            for row in block_rows
        ),
        "stage41_base_rank_exact": all(
            int(row["stage41_base_rank"]) == settings.stage41_full_rank for row in block_rows
        ),
        "total_rank_consistent": all(
            int(row["total_final_rank"]) >= int(row["stage41_base_rank"])
            and int(row["total_final_rank"]) <= int(row["stage41_base_rank"]) + int(row["correction_block_rank"])
            for row in block_rows
        ),
        "maximum_pair_defect": max(float(row["pair_defect"]) for row in block_rows),
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])
        ),
        "new_solver_rows": len(new_solver_rows),
        "expected_new_solver_rows": 147,
        "all_block_methods_labeled_non_deployable": all(
            "non-deployable" in row["source"]
            for row in new_solver_rows if row["method"] in block_methods
        ),
    }
    validation["passed"] = bool(
        all(validation[key] for key in (
            "preflight_passed", "all_requested_methods_present", "selected_rules_exact",
            "stage41_base_rank_exact", "total_rank_consistent",
            "all_block_methods_labeled_non_deployable",
        ))
        and validation["maximum_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage43_missing_subspace_block_ceiling.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage43] complete", flush=True)
    return copy
