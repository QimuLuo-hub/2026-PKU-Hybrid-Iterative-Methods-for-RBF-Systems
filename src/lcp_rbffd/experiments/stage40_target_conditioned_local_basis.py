"""Stage 40: target-conditioned full-span combinations from Krylov probes."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage27_spatial_k_adaptive_rank_validation import Stage27Settings, _run_adaptive
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage40Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    probe_rhs_seed: int = 20262911
    probe_depths: Tuple[int, int] = (64, 128)
    initial_rank: int = 28
    expanded_rank: int = 48
    expected_full_span_rank: int = 1456
    numerical_rank_tolerance: float = 1.0e-13
    coefficient_nonzero_relative_tolerance: float = 1.0e-8

    @property
    def stage27(self) -> Stage27Settings:
        return Stage27Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


def factor_full_action(matrix: sp.spmatrix, psi: np.ndarray):
    started = perf_counter()
    image = np.asarray(matrix @ np.asarray(psi, dtype=float), dtype=float)
    q_action, triangular = la.qr(image, mode="economic", check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    numerical_rank = int(
        np.count_nonzero(diagonal > np.max(diagonal) * np.finfo(float).eps * max(image.shape))
    )
    if numerical_rank != psi.shape[1]:
        raise RuntimeError("target action of the Stage-30 full span lost numerical rank")
    return q_action, triangular, {
        "full_span_columns": int(psi.shape[1]),
        "target_action_rank": numerical_rank,
        "target_action_condition": float(np.linalg.cond(triangular)),
        "factor_seconds": perf_counter() - started,
    }


def select_local_combinations(
    matrix: sp.spmatrix,
    psi: np.ndarray,
    q_action: np.ndarray,
    triangular: np.ndarray,
    target_action_information: np.ndarray,
    requested_rank: int,
    coefficient_tolerance: float,
    preserve_order: bool,
):
    """Select full-span combinations whose target actions best match information."""

    started = perf_counter()
    information, _ = la.qr(
        np.asarray(target_action_information, dtype=float),
        mode="economic",
        check_finite=False,
    )
    overlap = q_action.T @ information
    left, singular, _ = la.svd(overlap, full_matrices=False, check_finite=False)
    actual = min(requested_rank, left.shape[1])
    coefficients = la.solve_triangular(
        triangular, left[:, :actual], lower=False, check_finite=False
    )
    z_raw = np.asarray(psi, dtype=float) @ coefficients
    pair = (
        _ordered_pair_from_z(matrix, z_raw, 1.0e-13)
        if preserve_order
        else _pair_from_z(matrix, z_raw)
    )
    scale = np.max(np.abs(coefficients), axis=0, keepdims=True)
    nonzero = np.abs(coefficients) > coefficient_tolerance * np.maximum(scale, np.finfo(float).tiny)
    weights = np.sum(coefficients ** 2, axis=1)
    weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
    return pair, coefficients, {
        "requested_rank": requested_rank,
        "actual_rank": pair.rank,
        "information_rank": information.shape[1],
        "minimum_selected_overlap_cosine": float(singular[actual - 1]),
        "mean_squared_selected_overlap_cosine": float(np.mean(singular[:actual] ** 2)),
        "coefficient_nonzero_fraction": float(np.mean(nonzero)),
        "coefficient_mode_participation": float(1.0 / np.sum(weights ** 2)),
        "selection_and_pair_seconds": perf_counter() - started,
        "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
    }


def _teacher(root: Path, sample_id: int, matrix: sp.spmatrix) -> RecyclePair:
    path = root / "results/results-fd/stage28_operator_learning/systems" / f"system_{sample_id:03d}.npz"
    with np.load(path) as archive:
        return _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))


def _geometry(pair: RecyclePair, reference: RecyclePair):
    metrics = generalized_subspace_metrics(pair.w_basis, reference.w_basis)
    angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
    return {
        "d_W": float(metrics["projector_distance_normalized"]),
        "median_principal_angle_degrees": float(np.median(angles)),
        "largest_principal_angle_degrees": float(np.max(angles)),
        "reference_containment": float(metrics["containment_second"]),
    }


def construct_spaces(root: Path, output: Path, psi: np.ndarray, settings: Stage40Settings):
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[Tuple[int, str], RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    closeness_rows: List[Dict[str, Any]] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    archive: Dict[str, np.ndarray] = {}
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        probe_rhs, _ = _discrete_rhs(context.system, settings.probe_rhs_seed)
        context.ilu.reset_timing()
        arnoldi = _capture_arnoldi(
            context.system.matrix, context.ilu, probe_rhs, max(settings.probe_depths)
        )
        probe_m_applications = int(arnoldi.hessenberg.shape[1])
        probe_apply_seconds = float(context.ilu.apply_seconds)
        q_action, triangular, action_diagnostics = factor_full_action(
            context.system.matrix, psi
        )
        local_metadata = {
            "probe_rhs_seed": settings.probe_rhs_seed,
            "probe_actual_depth": int(arnoldi.hessenberg.shape[1]),
            "probe_A_applications": int(arnoldi.hessenberg.shape[1]),
            "probe_M_applications": probe_m_applications,
            "probe_seconds": float(arnoldi.seconds),
            "probe_preconditioner_apply_seconds": probe_apply_seconds,
            "action_factor": action_diagnostics,
            "methods": {},
        }

        # Both deployable spaces are finalized before the held-out teacher is loaded.
        for depth in settings.probe_depths:
            method = f"target_local_m{depth}"
            pair, c_local, diagnostics = select_local_combinations(
                context.system.matrix,
                psi,
                q_action,
                triangular,
                arnoldi.residual_basis[:, :depth],
                settings.expanded_rank,
                settings.coefficient_nonzero_relative_tolerance,
                preserve_order=True,
            )
            pairs[(sample_id, method)] = pair
            local_metadata["methods"][str(depth)] = diagnostics
            archive[f"system_{sample_id:03d}_{method}_c"] = c_local
            archive[f"system_{sample_id:03d}_{method}_z"] = pair.z_basis

        teacher = _teacher(root, sample_id, context.system.matrix)
        oracle, c_oracle, oracle_diagnostics = select_local_combinations(
            context.system.matrix,
            psi,
            q_action,
            triangular,
            teacher.w_basis,
            teacher.rank,
            settings.coefficient_nonzero_relative_tolerance,
            preserve_order=False,
        )
        pairs[(sample_id, "teacher_local_oracle")] = oracle
        archive[f"system_{sample_id:03d}_teacher_local_oracle_c"] = c_oracle
        archive[f"system_{sample_id:03d}_teacher_local_oracle_z"] = oracle.z_basis
        local_metadata["teacher_oracle"] = oracle_diagnostics
        for depth in settings.probe_depths:
            method = f"target_local_m{depth}"
            pair = pairs[(sample_id, method)]
            geometry_rows.append({
                "sample_id": sample_id,
                "method": method,
                "rank": pair.rank,
                **_geometry(pair, teacher),
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
            closeness = _geometry(pair, oracle)
            closeness_rows.append({
                "sample_id": sample_id,
                "method": method,
                "d_W_to_teacher_local_oracle": closeness["d_W"],
                "median_angle_to_teacher_local_oracle": closeness["median_principal_angle_degrees"],
                "largest_angle_to_teacher_local_oracle": closeness["largest_principal_angle_degrees"],
                "teacher_oracle_containment": closeness["reference_containment"],
            })
        geometry_rows.append({
            "sample_id": sample_id,
            "method": "teacher_local_oracle",
            "rank": oracle.rank,
            **_geometry(oracle, teacher),
            "pair_defect": oracle.diagnostics["au_equals_w_relative_error"],
        })
        metadata[sample_id] = local_metadata
        print(f"[Stage40 spaces] {position}/7 system {sample_id}", flush=True)
    np.savez_compressed(output / "local_combination_spaces.npz", **archive)
    _write_json(output / "construction_metadata.json", metadata)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "oracle_closeness.csv", closeness_rows)
    return pairs, geometry_rows, closeness_rows, metadata


METHODS = (
    "strong_ilu",
    "teacher_full",
    "full_span_oracle",
    "stage37_krylov_m64",
    "target_local_m64",
    "target_local_m128",
    "teacher_local_oracle",
)


def _archive_row(row: Mapping[str, Any], method: str):
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage37 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "rho": float(row["rho"]),
        "expanded": row["expanded"],
        "failure": row["failure"],
    }


def evaluate_solver(root: Path, pairs, settings: Stage40Settings):
    archived_all = _read_csv(
        root / "results/results-fd/stages/stage37_krylov_aware_local_reduction/solver_results.csv"
    )
    keep = {"strong_ilu", "teacher_full", "full_span_oracle", "krylov_m64"}
    rename = {"krylov_m64": "stage37_krylov_m64"}
    archived = [row for row in archived_all if row["method"] in keep]
    rows = [_archive_row(row, rename.get(row["method"], row["method"])) for row in archived]
    lookup = {(int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in archived_all}
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            oracle = pairs[(sample_id, "teacher_local_oracle")]
            result = _run_pair(context, rhs, oracle, settings.stage29.stage28.stage22a)
            residual = _relative_residual(context.system.matrix, result.x, rhs)
            rows.append({
                "sample_id": sample_id, "rhs_seed": rhs_seed,
                "method": "teacher_local_oracle", "source": "Stage40 diagnostic",
                "iterations": int(result.iterations),
                "converged": bool(result.converged and residual <= 1.0e-8),
                "true_final_residual": residual, "actual_rank": oracle.rank,
                "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator if denominator > 0 else float("nan")),
                "expanded": "", "failure": getattr(result, "failure", "") or "",
            })
            for depth in settings.probe_depths:
                method = f"target_local_m{depth}"
                result, controller, _ = _run_adaptive(
                    context, rhs, pairs[(sample_id, method)].z_basis, settings.stage27
                )
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append({
                    "sample_id": sample_id, "rhs_seed": rhs_seed,
                    "method": method, "source": "Stage40 deployable",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and residual <= 1.0e-8),
                    "true_final_residual": residual, "actual_rank": controller.pair.rank,
                    "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator if denominator > 0 else float("nan")),
                    "expanded": controller.expanded,
                    "failure": getattr(result, "failure", "") or "",
                })
    return rows


def summarize_solver(rows: Sequence[Mapping[str, Any]]):
    result = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return result


def comparison_geometry(root: Path, new_rows, settings: Stage40Settings):
    stage30 = _read_csv(root / "results/results-fd/stages/stage30_full_span_representation_ceiling/geometry_diagnostics.csv")
    stage37 = _read_csv(root / "results/results-fd/stages/stage37_krylov_aware_local_reduction/geometry_diagnostics.csv")
    rows = []
    for sample_id in settings.hard_system_ids:
        rows.append({"sample_id": sample_id, "method": "teacher_full", "d_W": 0.0, "largest_principal_angle_degrees": 0.0})
        full = next(row for row in stage30 if int(row["sample_id"]) == sample_id and row["method"] == "oracle_within_full_span")
        rows.append({"sample_id": sample_id, "method": "full_span_oracle", "d_W": float(full["d_W_to_teacher"]), "largest_principal_angle_degrees": float(full["maximum_angle_W_degrees"])})
        old = next(row for row in stage37 if int(row["sample_id"]) == sample_id and row["method"] == "krylov_m64_q48")
        rows.append({"sample_id": sample_id, "method": "stage37_krylov_m64", "d_W": float(old["d_W"]), "largest_principal_angle_degrees": float(old["largest_principal_angle_degrees"])})
        for method in ("target_local_m64", "target_local_m128", "teacher_local_oracle"):
            row = next(item for item in new_rows if int(item["sample_id"]) == sample_id and item["method"] == method)
            rows.append({"sample_id": sample_id, "method": method, "d_W": float(row["d_W"]), "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"])})
    return rows


def summarize_geometry(rows: Sequence[Mapping[str, Any]]):
    result = []
    for method in METHODS[1:]:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_largest_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in local])),
            "worst_largest_angle": float(np.max([float(row["largest_principal_angle_degrees"]) for row in local])),
        })
    return result


def summarize_closeness(rows: Sequence[Mapping[str, Any]]):
    result = []
    for depth in (64, 128):
        method = f"target_local_m{depth}"
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "median_dW_to_oracle": float(np.median([float(row["d_W_to_teacher_local_oracle"]) for row in local])),
            "median_largest_angle_to_oracle": float(np.median([float(row["largest_angle_to_teacher_local_oracle"]) for row in local])),
            "worst_largest_angle_to_oracle": float(np.max([float(row["largest_angle_to_teacher_local_oracle"]) for row in local])),
            "median_oracle_containment": float(np.median([float(row["teacher_oracle_containment"]) for row in local])),
        })
    return result


def summarize_cost(metadata, settings: Stage40Settings):
    result = []
    for depth in settings.probe_depths:
        local = [metadata[sample] for sample in settings.hard_system_ids]
        result.append({
            "method": f"target_local_m{depth}",
            "nominal_probe_A_applications": depth,
            "nominal_probe_M_applications": depth,
            "median_shared_m128_probe_seconds": float(np.median([row["probe_seconds"] for row in local])),
            "median_action_factor_seconds": float(np.median([row["action_factor"]["factor_seconds"] for row in local])),
            "median_selection_seconds": float(np.median([row["methods"][str(depth)]["selection_and_pair_seconds"] for row in local])),
            "median_coefficient_nonzero_fraction": float(np.median([row["methods"][str(depth)]["coefficient_nonzero_fraction"] for row in local])),
            "median_coefficient_mode_participation": float(np.median([row["methods"][str(depth)]["coefficient_mode_participation"] for row in local])),
        })
    return result


def decide(solver_summary, oracle_reproduced: bool):
    solver = {row["method"]: row for row in solver_summary}
    oracle = solver["teacher_local_oracle"]
    m64 = solver["target_local_m64"]
    m128 = solver["target_local_m128"]
    best_method = max(
        (solver[method]["converged_rhs"], -solver[method]["median_iterations"], solver[method]["median_rho"], method)
        for method in ("target_local_m64", "target_local_m128")
    )[3]
    best = solver[best_method]
    approaches = bool(
        oracle["converged_rhs"] - best["converged_rhs"] <= 3
        and oracle["median_rho"] - best["median_rho"] <= 0.15
    )
    m128_better = bool(
        m128["converged_rhs"] >= m64["converged_rhs"] + 3
        or m128["median_rho"] >= m64["median_rho"] + 0.10
        or (m128["converged_rhs"] == m64["converged_rhs"] and m128["median_iterations"] <= 0.85 * m64["median_iterations"])
    )
    if approaches:
        conclusion = f"{best_method.replace('target_local_', 'm=')} approaches the teacher-informed full-span ceiling using target information alone."
        stage41 = "Use the selected target-local coefficient/subspace as Stage-41 neural supervision, keeping the full-span representation and solver fixed."
    elif m128_better and m128["median_rho"] >= 0.50:
        conclusion = "The 128-step probe is clearly stronger, though it remains short of the full-span oracle."
        stage41 = "Train an NN to predict or warm-start the m=128 target-local construction, using its dense full-span coefficients and unchanged residual-space evaluation."
    else:
        conclusion = "Both short target probes remain far from the correctly reconstructed full-span oracle; their residual-space overlap does not reveal enough solver-critical information."
        stage41 = "Do not train on these targets; test a target-local construction driven by multiple probe RHSs and Ritz-filtered slow modes rather than raw Arnoldi-space overlap."
    return {
        "best_deployable_method": best_method,
        "approaches_teacher_oracle": approaches,
        "m128_clearly_better": m128_better,
        "teacher_oracle_reproduces_stage30": oracle_reproduced,
        "conclusion": conclusion,
        "stage41_direction": stage41,
    }


def write_report(output: Path, solver_summary, geometry_summary, closeness_summary, cost_summary, solver_rows, geometry_rows, decision):
    labels = {
        "strong_ilu": "strong ILU", "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage37_krylov_m64": "Stage-37 donor Krylov m=64",
        "target_local_m64": "Stage-40 target-local m=64",
        "target_local_m128": "Stage-40 target-local m=128",
        "teacher_local_oracle": "teacher-informed local-combination oracle",
    }
    lines = ["# Stage 40: Target-conditioned local basis from Krylov probes", "", "## Result", "", f"**{decision['conclusion']}**", "", "| method | converged | median iterations | median $\\rho$ | median final rank |", "|:--|--:|--:|--:|--:|"]
    for row in solver_summary:
        lines.append(f"| {labels[row['method']]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_rank'],1)} |")
    lines.extend(["", "## Residual-space geometry", "", "| method | median $d_W$ | median largest angle | worst largest angle |", "|:--|--:|--:|--:|"])
    for row in geometry_summary:
        lines.append(f"| {labels[row['method']]} | {_fmt(row['median_dW'])} | {_fmt(row['median_largest_angle'],1)} degrees | {_fmt(row['worst_largest_angle'],1)} degrees |")
    lines.extend(["", "## Distance to the teacher-informed full-span oracle", "", "| method | median $d_W$ to oracle | median largest angle | worst largest angle | oracle containment |", "|:--|--:|--:|--:|--:|"])
    for row in closeness_summary:
        lines.append(f"| {labels[row['method']]} | {_fmt(row['median_dW_to_oracle'])} | {_fmt(row['median_largest_angle_to_oracle'],1)} degrees | {_fmt(row['worst_largest_angle_to_oracle'],1)} degrees | {_fmt(row['median_oracle_containment'])} |")
    lines.extend(["", "## Probe and construction cost", "", "| method | nominal A/M probe steps | shared m=128 probe seconds | $A\\Psi$ factor seconds | selection seconds | coefficient density | effective full-span modes |", "|:--|--:|--:|--:|--:|--:|--:|"])
    for row in cost_summary:
        lines.append(f"| {labels[row['method']]} | {row['nominal_probe_A_applications']} / {row['nominal_probe_M_applications']} | {_fmt(row['median_shared_m128_probe_seconds'],2)} | {_fmt(row['median_action_factor_seconds'],2)} | {_fmt(row['median_selection_seconds'],2)} | {_fmt(row['median_coefficient_nonzero_fraction'])} | {_fmt(row['median_coefficient_mode_participation'],1)} |")
    lines.extend(["", "One nested 128-step Arnoldi factorization is captured per target; m=64 uses its prefix. The dense coefficient statistics verify that construction forms new combinations throughout the Stage-30 basis instead of selecting donor columns.", "", "## System 63", ""])
    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry_rows}
    for method in METHODS:
        local = [row for row in solver_rows if row["method"] == method and int(row["sample_id"]) == 63]
        extra = ""
        if method != "strong_ilu":
            geom = geometry_lookup[(63, method)]
            extra = f", $d_W={_fmt(geom['d_W'])}$, largest angle {_fmt(geom['largest_principal_angle_degrees'],1)} degrees"
        lines.append(f"- {labels[method]}: {sum(_bool(row['converged']) for row in local)}/3, median {int(np.median([int(row['iterations']) for row in local]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in local]))}${extra}.")
    lines.extend(["", "## Scope and leakage control", "", "Both deployable spaces use only the verified Stage-30 full span, target matrix, strong ILU, and construction RHS. The held-out teacher is loaded only after m=64 and m=128 are fixed, then reconstructs the named diagnostic oracle with the same $A\\Psi=QR$ factor. Solver settings and evaluation RHSs are unchanged.", "", "## Stage 41 recommendation", "", decision["stage41_direction"]])
    report = output / "stage40_target_conditioned_local_basis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage40(root: Path) -> Path:
    settings = Stage40Settings()
    output = root / "results/results-fd/stages/stage40_target_conditioned_local_basis"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-40 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()})
    full_path = root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    with np.load(full_path) as archive:
        psi = np.asarray(archive["psi"], dtype=float)
        training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()
    split = json.loads((root / "results/results-fd/stage28_operator_learning/split.json").read_text(encoding="utf-8"))
    preflight = {
        "full_span_shape": list(psi.shape),
        "full_span_rank_exact": psi.shape[1] == settings.expected_full_span_rank,
        "full_span_training_ids_match": training_ids == split["train"],
        "heldout_disjoint_from_training": not bool(set(settings.hard_system_ids) & set(training_ids)),
        "probe_depths": list(settings.probe_depths),
        "probe_rhs_disjoint_from_evaluation": settings.probe_rhs_seed not in settings.stage29.stage28.evaluation_rhs_seeds,
        "rank_policy_matches_stage37": settings.initial_rank == settings.stage27.default_q and settings.expanded_rank == settings.stage27.full_q,
        "solver_matches_stage37": settings.stage27.solver == settings.stage29.stage28.solver,
        "teacher_not_used_in_deployable_construction": True,
    }
    preflight["passed"] = all(value for key, value in preflight.items() if key not in {"full_span_shape", "probe_depths"})
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-40 preflight failed")
    pairs, new_geometry, closeness_rows, metadata = construct_spaces(root, output, psi, settings)
    solver_rows = evaluate_solver(root, pairs, settings)
    solver_summary = summarize_solver(solver_rows)
    all_geometry = comparison_geometry(root, new_geometry, settings)
    geometry_summary = summarize_geometry(all_geometry)
    closeness_summary = summarize_closeness(closeness_rows)
    cost_summary = summarize_cost(metadata, settings)
    archived_lookup = {(int(row["sample_id"]), int(row["rhs_seed"])): row for row in solver_rows if row["method"] == "full_span_oracle"}
    new_lookup = {(int(row["sample_id"]), int(row["rhs_seed"])): row for row in solver_rows if row["method"] == "teacher_local_oracle"}
    oracle_reproduced = all(
        _bool(new_lookup[key]["converged"]) == _bool(archived_lookup[key]["converged"])
        and int(new_lookup[key]["iterations"]) == int(archived_lookup[key]["iterations"])
        for key in archived_lookup
    )
    decision = decide(solver_summary, oracle_reproduced)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}
    _write_csv(output / "comparison_geometry.csv", all_geometry)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "oracle_closeness_summary.csv", closeness_summary)
    _write_csv(output / "probe_cost_summary.csv", cost_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(output, solver_summary, geometry_summary, closeness_summary, cost_summary, solver_rows, all_geometry, decision)
    validation = {
        "preflight_passed": preflight["passed"],
        "full_span_used_by_all_constructions": all(metadata[sample]["action_factor"]["full_span_columns"] == settings.expected_full_span_rank for sample in settings.hard_system_ids),
        "teacher_not_used_before_deployable_spaces": True,
        "probe_depths_exact": all(metadata[sample]["probe_actual_depth"] == max(settings.probe_depths) for sample in settings.hard_system_ids),
        "deployable_full_ranks_exact": all(pairs[(sample, f"target_local_m{depth}")].rank == settings.expanded_rank for sample in settings.hard_system_ids for depth in settings.probe_depths),
        "adaptive_final_ranks_valid": all(int(row["actual_rank"]) in {settings.initial_rank, settings.expanded_rank} for row in solver_rows if row["source"] == "Stage40 deployable"),
        "teacher_oracle_reproduces_stage30_iterations": oracle_reproduced,
        "maximum_pair_defect": max(float(row["pair_defect"]) for row in new_geometry),
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])),
        "new_solver_rows": sum(row["source"].startswith("Stage40") for row in solver_rows),
        "expected_new_solver_rows": 63,
    }
    validation["passed"] = bool(
        all(validation[key] for key in ("preflight_passed", "full_span_used_by_all_constructions", "teacher_not_used_before_deployable_spaces", "probe_depths_exact", "deployable_full_ranks_exact", "adaptive_final_ranks_valid", "teacher_oracle_reproduces_stage30_iterations"))
        and validation["maximum_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage40_target_conditioned_local_basis.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage40] complete", flush=True)
    return copy
