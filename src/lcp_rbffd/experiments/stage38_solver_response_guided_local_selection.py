"""Stage 38: solver-response-guided reduction inside frozen MLP donor unions."""

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

from neurkitt_oracle.gcrodr import restarted_projected_gmres

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage27_spatial_k_adaptive_rank_validation import _run_adaptive
from .stage29_family_level_coarse_space import _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage36_neural_donor_mixture import load_donors
from .stage37_krylov_aware_local_reduction import Stage37Settings, frozen_mlp_selections
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage38Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    probe_rhs_seed: int = 20262911
    probe_budgets: Tuple[int, int] = (8, 16)
    candidate_block_width: int = 4
    initial_rank: int = 28
    expanded_rank: int = 48
    numerical_rank_tolerance: float = 1.0e-13
    score_floor: float = 1.0e-15

    @property
    def stage37(self) -> Stage37Settings:
        return Stage37Settings()

    @property
    def stage27(self):
        return self.stage37.stage27

    @property
    def stage29(self):
        return self.stage37.stage29


def _ordered_pair_from_z(matrix: sp.spmatrix, raw: np.ndarray, tolerance: float) -> RecyclePair:
    """Build AZ=WR while preserving the greedy column-prefix ordering."""

    started = perf_counter()
    array = np.asarray(raw, dtype=float)
    z, triangular_z = la.qr(array, mode="economic", check_finite=False)
    diagonal = np.abs(np.diag(triangular_z))
    scale = max(float(la.norm(triangular_z, ord=2)), 1.0)
    rank = int(np.count_nonzero(diagonal > tolerance * scale))
    if rank != array.shape[1]:
        raise RuntimeError("greedy direction set lost solution-space rank")
    signs = np.where(np.diag(triangular_z) < 0.0, -1.0, 1.0)
    z = z * signs
    az = np.asarray(matrix @ z, dtype=float)
    w, triangular = la.qr(az, mode="economic", check_finite=False)
    action_diagonal = np.abs(np.diag(triangular))
    action_scale = max(float(la.norm(triangular, ord=2)), 1.0)
    if int(np.count_nonzero(action_diagonal > tolerance * action_scale)) != rank:
        raise RuntimeError("greedy direction set lost target-action rank")
    u = la.solve_triangular(triangular.T, z.T, lower=True, check_finite=False).T
    defect = float(
        la.norm(np.asarray(matrix @ u, dtype=float) - w, ord="fro")
        / max(la.norm(w, ord="fro"), np.finfo(float).tiny)
    )
    return RecyclePair(
        z_basis=z,
        u_basis=u,
        w_basis=w,
        setup_seconds=perf_counter() - started,
        diagnostics={
            "au_equals_w_relative_error": defect,
            "test_orthogonality_error": float(la.norm(w.T @ w - np.eye(rank), ord="fro")),
            "reduced_condition": float(np.linalg.cond(triangular)),
            "dictionary_rank": float(rank),
        },
    )


def _next_donor_block(
    donor_pair: RecyclePair,
    selected_pair: RecyclePair | None,
    width: int,
    tolerance: float,
) -> Tuple[np.ndarray, Dict[str, float]] | None:
    """Return a small donor block with maximal action novelty to the current space."""

    residual_action = donor_pair.w_basis.copy()
    if selected_pair is not None:
        residual_action -= selected_pair.w_basis @ (selected_pair.w_basis.T @ residual_action)
    gram = residual_action.T @ residual_action
    eigenvalues, eigenvectors = la.eigh(gram, check_finite=False)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    if eigenvalues.size == 0 or eigenvalues[0] <= tolerance:
        return None
    rank = int(np.count_nonzero(eigenvalues > tolerance * max(eigenvalues[0], 1.0)))
    actual = min(width, rank)
    if actual == 0:
        return None
    coefficients = eigenvectors[:, :actual]
    block = donor_pair.u_basis @ coefficients
    return block, {
        "block_rank": actual,
        "maximum_action_novelty": float(np.sqrt(eigenvalues[0])),
        "minimum_selected_action_novelty": float(np.sqrt(eigenvalues[actual - 1])),
    }


def _short_solver_response(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    apply_preconditioner,
    pair: RecyclePair | None,
    budget: int,
    solver,
) -> Tuple[float, Dict[str, Any]]:
    started = perf_counter()
    kwargs: Dict[str, Any] = {}
    if pair is not None:
        kwargs = {"u_basis": pair.u_basis, "c_basis": pair.w_basis}
    result = restarted_projected_gmres(
        matrix,
        rhs,
        apply_preconditioner,
        rtol=solver.relative_tolerance,
        atol=solver.absolute_tolerance,
        restart=solver.restart,
        max_iterations=budget,
        record_true_residuals=True,
        record_projection_diagnostics=False,
        **kwargs,
    )
    return float(result.relative_residual), {
        "iterations": int(result.iterations),
        "seconds": perf_counter() - started,
        "a_matvecs": int((result.operation_counts or {}).get("a_matvecs", 0)),
        "preconditioner_applications": int(
            (result.operation_counts or {}).get("preconditioner_applications", 0)
        ),
    }


def greedy_solver_response_reduction(
    matrix: sp.spmatrix,
    donor_spaces: Mapping[int, np.ndarray],
    rhs: np.ndarray,
    apply_preconditioner,
    budget: int,
    requested_rank: int,
    block_width: int,
    tolerance: float,
    score_floor: float,
    solver,
) -> Tuple[RecyclePair, List[Dict[str, Any]], Dict[str, Any]]:
    """Greedily add the donor novelty block with best short-GMRES response."""

    started = perf_counter()
    donor_pairs = {
        donor_id: _pair_from_z(matrix, basis)
        for donor_id, basis in donor_spaces.items()
    }
    current_pair: RecyclePair | None = None
    selected_raw: np.ndarray | None = None
    current_residual, baseline_cost = _short_solver_response(
        matrix, rhs, apply_preconditioner, None, budget, solver
    )
    evaluations = 1
    short_iterations = baseline_cost["iterations"]
    short_seconds = baseline_cost["seconds"]
    a_matvecs = baseline_cost["a_matvecs"]
    preconditioner_applications = baseline_cost["preconditioner_applications"]
    trace: List[Dict[str, Any]] = []
    round_index = 0
    selected_donor_blocks: List[int] = []
    while current_pair is None or current_pair.rank < requested_rank:
        round_index += 1
        current_rank = 0 if current_pair is None else current_pair.rank
        width = min(block_width, requested_rank - current_rank)
        candidates = []
        for donor_id in sorted(donor_pairs):
            proposed = _next_donor_block(
                donor_pairs[donor_id], current_pair, width, tolerance
            )
            if proposed is None:
                continue
            block, novelty = proposed
            combined = block if selected_raw is None else np.column_stack((selected_raw, block))
            try:
                pair = _ordered_pair_from_z(matrix, combined, tolerance)
            except (RuntimeError, la.LinAlgError):
                continue
            residual, cost = _short_solver_response(
                matrix, rhs, apply_preconditioner, pair, budget, solver
            )
            evaluations += 1
            short_iterations += cost["iterations"]
            short_seconds += cost["seconds"]
            a_matvecs += cost["a_matvecs"]
            preconditioner_applications += cost["preconditioner_applications"]
            score = float(
                np.log(max(current_residual, score_floor))
                - np.log(max(residual, score_floor))
            )
            record = {
                "round": round_index,
                "rank_before": current_rank,
                "rank_after": pair.rank,
                "donor_id": donor_id,
                "score": score,
                "current_residual_ratio": current_residual,
                "candidate_residual_ratio": residual,
                **novelty,
                **{f"probe_{key}": value for key, value in cost.items()},
            }
            candidates.append((score, -donor_id, pair, combined, residual, record))
        if not candidates:
            raise RuntimeError("no full-rank donor novelty block remains")
        winner = max(candidates, key=lambda item: (item[0], item[1]))
        _, _, current_pair, selected_raw, current_residual, winning_record = winner
        selected_donor_blocks.append(int(winning_record["donor_id"]))
        for candidate in candidates:
            record = dict(candidate[5])
            record["selected"] = record["donor_id"] == winning_record["donor_id"]
            trace.append(record)
    if current_pair is None or current_pair.rank != requested_rank:
        raise RuntimeError("solver-response greedy selection did not reach requested rank")
    return current_pair, trace, {
        "probe_budget": budget,
        "candidate_block_width": block_width,
        "requested_rank": requested_rank,
        "actual_rank": current_pair.rank,
        "short_gmres_evaluations": evaluations,
        "short_gmres_iterations": short_iterations,
        "short_gmres_seconds": short_seconds,
        "probe_a_matvecs": a_matvecs,
        "probe_preconditioner_applications": preconditioner_applications,
        "selected_donor_blocks": selected_donor_blocks,
        "final_probe_residual_ratio": current_residual,
        "setup_seconds": perf_counter() - started,
        "pair_defect": current_pair.diagnostics["au_equals_w_relative_error"],
    }


def _teacher(root: Path, sample_id: int, matrix: sp.spmatrix) -> RecyclePair:
    path = root / "results/results-fd/stage28_operator_learning/systems" / f"system_{sample_id:03d}.npz"
    with np.load(path) as archive:
        return _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))


def _geometry(pair: RecyclePair, reference: RecyclePair) -> Dict[str, float]:
    metrics = generalized_subspace_metrics(pair.w_basis, reference.w_basis)
    angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
    return {
        "d_W": float(metrics["projector_distance_normalized"]),
        "median_principal_angle_degrees": float(np.median(angles)),
        "largest_principal_angle_degrees": float(np.max(angles)),
        "reference_containment": float(metrics["containment_second"]),
    }


def construct_spaces(
    root: Path,
    output: Path,
    donors: Mapping[int, np.ndarray],
    selections: Mapping[int, Sequence[int]],
    settings: Stage38Settings,
):
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[Tuple[int, str], RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    basis_archive: Dict[str, np.ndarray] = {}
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        selected_spaces = {donor: donors[donor] for donor in selections[sample_id]}
        raw_union = np.column_stack(list(selected_spaces.values()))
        probe_rhs, _ = _discrete_rhs(context.system, settings.probe_rhs_seed)
        local_metadata = {"selected_donors": list(selections[sample_id]), "union_columns": raw_union.shape[1], "budgets": {}}

        # Build every deployable space before loading the held-out teacher.
        for budget in settings.probe_budgets:
            label = f"solver_response_t{budget}"
            context.ilu.reset_timing()
            pair, trace, diagnostics = greedy_solver_response_reduction(
                context.system.matrix,
                selected_spaces,
                probe_rhs,
                context.ilu.apply,
                budget,
                settings.expanded_rank,
                settings.candidate_block_width,
                settings.numerical_rank_tolerance,
                settings.score_floor,
                settings.stage27.solver,
            )
            pairs[(sample_id, label)] = pair
            basis_archive[f"system_{sample_id:03d}_{label}_zfull"] = pair.z_basis
            for row in trace:
                trace_rows.append({"sample_id": sample_id, "probe_budget": budget, **row})
            local_metadata["budgets"][str(budget)] = diagnostics

        teacher = _teacher(root, sample_id, context.system.matrix)
        teacher_reduction, _ = oracle_select_inside_dictionary(
            context.system.matrix, raw_union, teacher.w_basis
        )
        pairs[(sample_id, "teacher_reduction")] = teacher_reduction
        for budget in settings.probe_budgets:
            label = f"solver_response_t{budget}"
            pair = pairs[(sample_id, label)]
            geometry_rows.append({
                "sample_id": sample_id,
                "method": label,
                "rank": pair.rank,
                **_geometry(pair, teacher),
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
            overlap = _geometry(pair, teacher_reduction)
            overlap_rows.append({
                "sample_id": sample_id,
                "method": label,
                "teacher_reduction_rank": teacher_reduction.rank,
                "selected_rank": pair.rank,
                "d_W_to_teacher_reduction": overlap["d_W"],
                "median_angle_to_teacher_reduction": overlap["median_principal_angle_degrees"],
                "largest_angle_to_teacher_reduction": overlap["largest_principal_angle_degrees"],
                "teacher_reduction_containment": overlap["reference_containment"],
            })
        metadata[sample_id] = local_metadata
        print(f"[Stage38 spaces] {position}/7 system {sample_id}", flush=True)
    np.savez_compressed(output / "selected_spaces.npz", **basis_archive)
    _write_csv(output / "greedy_trace.csv", trace_rows)
    _write_json(output / "probe_cost_and_selection_metadata.json", metadata)
    return geometry_rows, overlap_rows, pairs, metadata


def _archive_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
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


METHODS = (
    "strong_ilu",
    "teacher_full",
    "full_span_oracle",
    "stage37_teacher_reduction_oracle",
    "stage37_krylov_m64",
    "solver_response_t8",
    "solver_response_t16",
)


def evaluate_solver(root: Path, pairs, settings: Stage38Settings) -> List[Dict[str, Any]]:
    stage37 = root / "results/results-fd/stages/stage37_krylov_aware_local_reduction"
    archived_all = _read_csv(stage37 / "solver_results.csv")
    rename = {
        "selected_union_teacher_oracle": "stage37_teacher_reduction_oracle",
        "krylov_m64": "stage37_krylov_m64",
    }
    keep = {"strong_ilu", "teacher_full", "full_span_oracle", "selected_union_teacher_oracle", "krylov_m64"}
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
            for budget in settings.probe_budgets:
                method = f"solver_response_t{budget}"
                pair = pairs[(sample_id, method)]
                result, controller, _ = _run_adaptive(context, rhs, pair.z_basis, settings.stage27)
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append({
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": method,
                    "source": "Stage38 new",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and residual <= 1.0e-8),
                    "true_final_residual": residual,
                    "actual_rank": controller.pair.rank,
                    "rho": (
                        (int(baseline["iterations"]) - int(result.iterations)) / denominator
                        if denominator > 0 else float("nan")
                    ),
                    "expanded": controller.expanded,
                    "failure": getattr(result, "failure", "") or "",
                })
    return rows


def summarize_solver(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return summary


def summarize_geometry(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for budget in (8, 16):
        method = f"solver_response_t{budget}"
        local = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_principal_angle": float(np.median([float(row["median_principal_angle_degrees"]) for row in local])),
            "median_largest_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in local])),
            "worst_largest_angle": float(np.max([float(row["largest_principal_angle_degrees"]) for row in local])),
        })
    return summary


def summarize_cost(metadata: Mapping[int, Mapping[str, Any]], settings: Stage38Settings):
    rows = []
    for budget in settings.probe_budgets:
        local = [metadata[sample]["budgets"][str(budget)] for sample in settings.hard_system_ids]
        rows.append({
            "method": f"solver_response_t{budget}",
            "median_short_gmres_evaluations": float(np.median([row["short_gmres_evaluations"] for row in local])),
            "total_short_gmres_evaluations": int(sum(row["short_gmres_evaluations"] for row in local)),
            "median_short_gmres_iterations": float(np.median([row["short_gmres_iterations"] for row in local])),
            "total_short_gmres_iterations": int(sum(row["short_gmres_iterations"] for row in local)),
            "median_probe_seconds": float(np.median([row["short_gmres_seconds"] for row in local])),
            "total_probe_seconds": float(sum(row["short_gmres_seconds"] for row in local)),
        })
    return rows


def decide(solver_summary: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    solver = {row["method"]: row for row in solver_summary}
    best = max(
        (solver[f"solver_response_t{budget}"]["converged_rhs"],
         -solver[f"solver_response_t{budget}"]["median_iterations"],
         solver[f"solver_response_t{budget}"]["median_rho"], budget)
        for budget in (8, 16)
    )[3]
    response = solver[f"solver_response_t{best}"]
    old = solver["stage37_krylov_m64"]
    oracle = solver["stage37_teacher_reduction_oracle"]
    material = bool(
        response["converged_rhs"] > old["converged_rhs"]
        or (response["converged_rhs"] == old["converged_rhs"]
            and response["median_iterations"] <= 0.9 * old["median_iterations"])
        or response["median_rho"] >= old["median_rho"] + 0.10
    )
    approaches = bool(
        oracle["converged_rhs"] - response["converged_rhs"] <= 3
        and oracle["median_rho"] - response["median_rho"] <= 0.15
    )
    if material and approaches:
        conclusion = "Direct short-horizon solver response materially improves the frozen MLP hybrid and approaches the selected-union teacher oracle."
        stage39 = "Keep the frozen MLP plus solver-response pipeline and validate the selected probe budget on a fresh held-out hard-system cohort."
    elif material:
        conclusion = "Direct solver-response selection improves Stage 37 but remains materially below the selected-union teacher oracle."
        stage39 = "Keep the frozen MLP union and test whether a richer local candidate pool closes the remaining gap, with special attention to system 63."
    else:
        conclusion = "Solver-response selection does not materially improve the Stage-37 Krylov reducer."
        stage39 = "Stop refining local selection scores and replace the coarse whole-teacher donor representation with direction-level or target-local candidates."
    return {
        "best_probe_budget": best,
        "material_improvement_over_stage37": material,
        "approaches_teacher_reduction_oracle": approaches,
        "conclusion": conclusion,
        "stage39_direction": stage39,
    }


def write_report(output: Path, solver_summary, geometry_summary, overlap_rows, cost_summary, solver_rows, geometry_rows, decision):
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage37_teacher_reduction_oracle": "Stage-37 selected-union teacher oracle",
        "stage37_krylov_m64": "Stage-37 Krylov m=64",
        "solver_response_t8": "Stage-38 solver response t=8",
        "solver_response_t16": "Stage-38 solver response t=16",
    }
    lines = [
        "# Stage 38: Solver-response-guided local direction selection", "",
        "## Result", "", f"**{decision['conclusion']}**", "",
        "| method | converged | median iterations | median $\\rho$ | median final rank |",
        "|:--|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        lines.append(f"| {labels[row['method']]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_rank'],1)} |")
    lines.extend(["", "## Residual-space geometry", "", "| method | median $d_W$ | median angle | median largest angle | worst largest angle |", "|:--|--:|--:|--:|--:|"])
    stage37_geometry = {row["method"]: row for row in _read_csv(output.parent / "stage37_krylov_aware_local_reduction" / "geometry_summary.csv")}
    for method, source in (("stage37_krylov_m64", stage37_geometry["krylov_m64_q48"]),):
        lines.append(f"| {labels[method]} | {_fmt(source['median_dW'])} | {_fmt(source['median_principal_angle'],1)} degrees | {_fmt(source['median_largest_angle'],1)} degrees | {_fmt(source['worst_largest_angle'],1)} degrees |")
    for row in geometry_summary:
        lines.append(f"| {labels[row['method']]} | {_fmt(row['median_dW'])} | {_fmt(row['median_principal_angle'],1)} degrees | {_fmt(row['median_largest_angle'],1)} degrees | {_fmt(row['worst_largest_angle'],1)} degrees |")
    lines.extend(["", "## Probe cost", "", "| method | short evaluations / target | total evaluations | median short iterations / target | total short iterations | median probe seconds / target |", "|:--|--:|--:|--:|--:|--:|"])
    for row in cost_summary:
        lines.append(f"| {labels[row['method']]} | {_fmt(row['median_short_gmres_evaluations'],1)} | {row['total_short_gmres_evaluations']} | {_fmt(row['median_short_gmres_iterations'],1)} | {row['total_short_gmres_iterations']} | {_fmt(row['median_probe_seconds'],2)} |")
    lines.extend(["", "The scorer uses one construction RHS (seed 20262911), disjoint from evaluation seeds 20263001–20263003. Each round tests at most five rank-4 donor-novelty blocks, so it reaches ordered rank 48 in 12 rounds without scoring roughly 200 individual directions.", "", "## Overlap with the teacher-informed selection", ""])
    for budget in (8, 16):
        local = [row for row in overlap_rows if row["method"] == f"solver_response_t{budget}"]
        lines.append(f"- t={budget}: median containment of the Stage-37 teacher-selected action space is {_fmt(np.median([float(row['teacher_reduction_containment']) for row in local]))}; median/largest-across-systems principal angles are {_fmt(np.median([float(row['median_angle_to_teacher_reduction']) for row in local]),1)} / {_fmt(np.max([float(row['largest_angle_to_teacher_reduction']) for row in local]),1)} degrees.")
    lines.extend(["", "This overlap is diagnostic only; selection and final construction never access the held-out teacher.", "", "## System 63", ""])
    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry_rows}
    for method in ("stage37_teacher_reduction_oracle", "stage37_krylov_m64", "solver_response_t8", "solver_response_t16"):
        local = [row for row in solver_rows if row["method"] == method and int(row["sample_id"]) == 63]
        extra = ""
        if method.startswith("solver_response"):
            geom = geometry_lookup[(63, method)]
            extra = f", $d_W={_fmt(geom['d_W'])}$, largest angle {_fmt(geom['largest_principal_angle_degrees'],1)} degrees"
        lines.append(f"- {labels[method]}: {sum(_bool(row['converged']) for row in local)}/3, median {int(np.median([int(row['iterations']) for row in local]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in local]))}${extra}.")
    best = decision["best_probe_budget"]
    solver = {row["method"]: row for row in solver_summary}
    cost = next(row for row in cost_summary if row["method"] == f"solver_response_t{best}")
    saved = 3.0 * (solver["stage37_krylov_m64"]["median_iterations"] - solver[f"solver_response_t{best}"]["median_iterations"])
    lines.extend(["", "## Cost interpretation", "", f"At the selected t={best} budget, the median target spends {_fmt(cost['median_short_gmres_iterations'],1)} short-GMRES iterations during selection. The rough three-RHS median final-solve saving relative to Stage 37 is {_fmt(saved,1)} iterations; this excludes reusable amortization over additional RHSs and should be read only as an iteration-equivalent cost check.", "", "## Leakage and frozen components", "", "The Stage-36 MLP checkpoint and exact Stage-37 top-5 donor selections are unchanged. The response score uses only the target matrix, strong ILU, fixed donor union, and construction RHS. The own-system teacher is loaded only after both deployable spaces have been finalized, for geometry and the named diagnostic oracle.", "", "## Stage 39 recommendation", "", decision["stage39_direction"]])
    report = output / "stage38_solver_response_guided_local_selection.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage38(root: Path) -> Path:
    settings = Stage38Settings()
    output = root / "results/results-fd/stages/stage38_solver_response_guided_local_selection"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-38 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()})
    donor_ids, selections = frozen_mlp_selections(root)
    donors, _ = load_donors(root, donor_ids)
    checkpoint = root / "results/results-fd/stages/stage36_neural_donor_mixture/models/selected_mlp_donor.pt"
    checkpoint_mtime = checkpoint.stat().st_mtime
    stage37_metadata = json.loads((root / "results/results-fd/stages/stage37_krylov_aware_local_reduction/probe_and_reduction_metadata.json").read_text(encoding="utf-8"))
    exact_reuse = all(stage37_metadata[str(target)]["selected_donors"] == selections[target] for target in settings.hard_system_ids)
    preflight = {
        "frozen_checkpoint_exists": checkpoint.exists(),
        "checkpoint_not_modified": True,
        "exact_stage37_selections_reused": exact_reuse,
        "five_unique_donors_per_target": all(len(values) == 5 and len(set(values)) == 5 for values in selections.values()),
        "probe_budgets": list(settings.probe_budgets),
        "probe_disjoint_from_evaluation": settings.probe_rhs_seed not in settings.stage29.stage28.evaluation_rhs_seeds,
        "solver_matches_stage37": settings.stage27.solver == settings.stage37.stage27.solver,
        "rank_policy_matches_stage37": settings.initial_rank == settings.stage27.default_q and settings.expanded_rank == settings.stage27.full_q,
    }
    preflight["passed"] = all(value for key, value in preflight.items() if key != "probe_budgets")
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-38 preflight failed")
    geometry_rows, overlap_rows, pairs, metadata = construct_spaces(root, output, donors, selections, settings)
    if checkpoint.stat().st_mtime != checkpoint_mtime:
        raise RuntimeError("frozen MLP checkpoint was modified")
    solver_rows = evaluate_solver(root, pairs, settings)
    solver_summary = summarize_solver(solver_rows)
    geometry_summary = summarize_geometry(geometry_rows)
    cost_summary = summarize_cost(metadata, settings)
    decision = decide(solver_summary)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "teacher_overlap_diagnostics.csv", overlap_rows)
    _write_csv(output / "probe_cost_summary.csv", cost_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(output, solver_summary, geometry_summary, overlap_rows, cost_summary, solver_rows, geometry_rows, decision)
    trace = _read_csv(output / "greedy_trace.csv")
    selected_per_round = sum(_bool(row["selected"]) for row in trace)
    expected_rounds = len(settings.hard_system_ids) * len(settings.probe_budgets) * (settings.expanded_rank // settings.candidate_block_width)
    validation = {
        "preflight_passed": preflight["passed"],
        "frozen_checkpoint_unchanged": checkpoint.stat().st_mtime == checkpoint_mtime,
        "exact_stage37_unions_reused": exact_reuse,
        "teacher_loaded_after_deployable_construction": True,
        "probe_budgets_exact": all(int(metadata[s]["budgets"][str(b)]["probe_budget"]) == b for s in settings.hard_system_ids for b in settings.probe_budgets),
        "greedy_selected_rounds": selected_per_round,
        "expected_greedy_selected_rounds": expected_rounds,
        "full_space_ranks_exact": all(pairs[(s, f"solver_response_t{b}")].rank == settings.expanded_rank for s in settings.hard_system_ids for b in settings.probe_budgets),
        "adaptive_ranks_valid": all(int(row["actual_rank"]) in {settings.initial_rank, settings.expanded_rank} for row in solver_rows if row["source"] == "Stage38 new"),
        "maximum_pair_defect": max(float(row["pair_defect"]) for row in geometry_rows),
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])),
    }
    validation["passed"] = bool(
        all(validation[key] for key in ("preflight_passed", "frozen_checkpoint_unchanged", "exact_stage37_unions_reused", "teacher_loaded_after_deployable_construction", "probe_budgets_exact", "full_space_ranks_exact", "adaptive_ranks_valid"))
        and selected_per_round == expected_rounds
        and validation["maximum_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage38_solver_response_guided_local_selection.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage38] complete", flush=True)
    return copy
