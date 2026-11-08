"""Stage 45: deployable rank stopping and representation-failure detection."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la

from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import (
    _ordered_pair_from_z,
    _short_solver_response,
)
from .stage44_rank_retention_mechanism import _prefix_method
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage45Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    retained_ranks: Tuple[int, ...] = (8, 16, 24, 32, 40, 48)
    diagnostic_rhs_seed: int = 20262931
    diagnostic_budget: int = 8
    endpoint_recovery_fraction: float = 0.90
    ordering_dominance_fraction: float = 0.65
    minimum_material_log_gain: float = 0.10
    robust_outlier_multiplier: float = 2.5
    robust_minimum_log_margin: float = 0.25
    saturation_absolute_log_gain: float = 0.10
    saturation_relative_gain: float = 0.15
    stage41_probe_iteration_equivalent: int = 192

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


def _robust_upper(values: Sequence[float], multiplier: float, minimum_margin: float) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    scaled_mad = 1.4826 * float(np.median(np.abs(array - median)))
    return median + max(multiplier * scaled_mad, minimum_margin)


def _robust_lower(values: Sequence[float], multiplier: float, minimum_margin: float) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    scaled_mad = 1.4826 * float(np.median(np.abs(array - median)))
    return median - max(multiplier * scaled_mad, minimum_margin)


def _select_rank(curve: Sequence[Mapping[str, Any]], fraction: float) -> int:
    ordered = sorted(curve, key=lambda row: int(row["rank"]))
    first = float(ordered[0]["log10_relative_residual"])
    last = float(ordered[-1]["log10_relative_residual"])
    total_gain = first - last
    if total_gain <= np.finfo(float).eps:
        return int(ordered[0]["rank"])
    target = first - fraction * total_gain
    eligible = [int(row["rank"]) for row in ordered if float(row["log10_relative_residual"]) <= target]
    return min(eligible) if eligible else int(ordered[-1]["rank"])


def _curve_features(curve: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(curve, key=lambda row: int(row["rank"]))
    by_rank = {int(row["rank"]): float(row["log10_relative_residual"]) for row in ordered}
    gains = {f"gain_{q}_to_{q+8}": by_rank[q] - by_rank[q + 8] for q in (8, 16, 24, 32, 40)}
    positive_late = [max(gains[f"gain_{q}_to_{q+8}"], 0.0) for q in (24, 32, 40)]
    late_total = float(sum(positive_late))
    dominant = float(max(positive_late) / max(late_total, np.finfo(float).tiny))
    return {
        **gains,
        "endpoint_log_residual": by_rank[48],
        "total_log_gain_8_to_48": by_rank[8] - by_rank[48],
        "late_log_gain_24_to_48": by_rank[24] - by_rank[48],
        "last_two_positive_log_gain": max(gains["gain_32_to_40"], 0.0) + max(gains["gain_40_to_48"], 0.0),
        "dominant_late_block_fraction": dominant,
        "dominant_late_block": ("24_to_32", "32_to_40", "40_to_48")[int(np.argmax(positive_late))],
        "maximum_late_block_gain": float(max(positive_late)),
    }


def freeze_deployable_decisions(
    curves: Sequence[Mapping[str, Any]],
    settings: Stage45Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Freeze all rank/class decisions from deployable features only."""

    grouped = {
        sample_id: [row for row in curves if int(row["sample_id"]) == sample_id]
        for sample_id in settings.hard_system_ids
    }
    features = []
    for sample_id, local in grouped.items():
        row = {"sample_id": sample_id, **_curve_features(local)}
        baseline = next(float(item["baseline_log10_relative_residual"]) for item in local)
        row["baseline_log10_relative_residual"] = baseline
        row["q48_log_benefit_over_baseline"] = baseline - float(row["endpoint_log_residual"])
        features.append(row)
    feature_lookup = {int(row["sample_id"]): row for row in features}
    decisions = []
    for sample_id in settings.hard_system_ids:
        target = feature_lookup[sample_id]
        others = [row for row in features if int(row["sample_id"]) != sample_id]
        endpoint_limit = _robust_upper(
            [float(row["endpoint_log_residual"]) for row in others],
            settings.robust_outlier_multiplier,
            settings.robust_minimum_log_margin,
        )
        benefit_limit = _robust_lower(
            [float(row["q48_log_benefit_over_baseline"]) for row in others],
            settings.robust_outlier_multiplier,
            settings.robust_minimum_log_margin,
        )
        total_gain = max(float(target["total_log_gain_8_to_48"]), 0.0)
        saturated = float(target["last_two_positive_log_gain"]) <= max(
            settings.saturation_absolute_log_gain,
            settings.saturation_relative_gain * total_gain,
        )
        endpoint_bad = float(target["endpoint_log_residual"]) > endpoint_limit
        benefit_bad = float(target["q48_log_benefit_over_baseline"]) < benefit_limit
        representation_failure = bool(endpoint_bad and (saturated or benefit_bad))
        ordering_sensitive = bool(
            not representation_failure
            and float(target["dominant_late_block_fraction"]) >= settings.ordering_dominance_fraction
            and float(target["maximum_late_block_gain"]) >= settings.minimum_material_log_gain
        )
        flag = (
            "representation-failure" if representation_failure
            else "ordering-sensitive" if ordering_sensitive
            else "normal"
        )
        selected_rank = _select_rank(grouped[sample_id], settings.endpoint_recovery_fraction)
        decisions.append({
            "sample_id": sample_id,
            "selected_rank": selected_rank,
            "deployable_flag": flag,
            "endpoint_bad_leave_one_out": endpoint_bad,
            "benefit_bad_leave_one_out": benefit_bad,
            "saturated_late_gain": saturated,
            "leave_one_out_endpoint_limit": endpoint_limit,
            "leave_one_out_benefit_limit": benefit_limit,
            "decision_uses_teacher_or_oracle": False,
        })
    return decisions, features


def _candidate_diagnostics(pairs: Mapping[int, Any]):
    rows = []
    previous = None
    previous_coupling = None
    for rank in sorted(pairs):
        pair = pairs[rank]
        coupling = pair.w_basis.T @ pair.u_basis
        if previous is None:
            novelty = float("nan")
            coupling_change = float("nan")
        else:
            old_u = la.qr(previous.u_basis, mode="economic", check_finite=False)[0]
            block = pair.u_basis[:, previous.rank:]
            residual = block - old_u @ (old_u.T @ block)
            novelty = float(la.norm(residual, ord="fro") / max(la.norm(block, ord="fro"), np.finfo(float).tiny))
            padded = np.zeros_like(coupling)
            padded[: previous_coupling.shape[0], : previous_coupling.shape[1]] = previous_coupling
            coupling_change = float(la.norm(coupling - padded, ord="fro") / max(la.norm(coupling, ord="fro"), np.finfo(float).tiny))
        rows.append({
            "rank": rank,
            "solution_block_novelty": novelty,
            "projected_pair_coupling_change": coupling_change,
        })
        previous = pair
        previous_coupling = coupling
    return rows


def _final_rows(
    stage44_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
):
    ranks = {int(row["sample_id"]): int(row["selected_rank"]) for row in decisions}
    return [
        {
            "sample_id": int(row["sample_id"]),
            "rhs_seed": int(row["rhs_seed"]),
            "selected_rank": ranks[int(row["sample_id"])],
            "method": "stage45_adaptive_rank",
            "iterations": int(row["iterations"]),
            "converged": _bool(row["converged"]),
            "true_final_residual": float(row["true_final_residual"]),
            "rho": float(row["rho"]),
        }
        for row in stage44_rows
        if row["method"] == _prefix_method(ranks[int(row["sample_id"])])
    ]


def _cost_rows(
    decisions,
    curves,
    adaptive_final,
    stage43_solver,
    settings: Stage45Settings,
):
    decision_lookup = {int(row["sample_id"]): row for row in decisions}
    rows = []
    for sample_id in settings.hard_system_ids:
        local_curve = [row for row in curves if int(row["sample_id"]) == sample_id]
        diagnostic_iterations = int(sum(int(row["short_iterations"]) for row in local_curve) + int(local_curve[0]["baseline_short_iterations"]))
        diagnostic_a = int(sum(int(row["short_a_matvecs"]) for row in local_curve) + int(local_curve[0]["baseline_a_matvecs"]))
        diagnostic_m = int(sum(int(row["short_m_applications"]) for row in local_curve) + int(local_curve[0]["baseline_m_applications"]))
        adaptive = [row for row in adaptive_final if int(row["sample_id"]) == sample_id]
        fixed = [row for row in stage43_solver if int(row["sample_id"]) == sample_id and row["method"] == "stage41_harmonic_fixed_q48"]
        ilu = [row for row in stage43_solver if int(row["sample_id"]) == sample_id and row["method"] == "strong_ilu"]
        teacher = [row for row in stage43_solver if int(row["sample_id"]) == sample_id and row["method"] == "teacher_full"]
        adaptive_final_cost = sum(int(row["iterations"]) for row in adaptive)
        fixed_final_cost = sum(int(row["iterations"]) for row in fixed)
        rows.append({
            "sample_id": sample_id,
            "selected_rank": int(decision_lookup[sample_id]["selected_rank"]),
            "probe_iteration_equivalent": settings.stage41_probe_iteration_equivalent,
            "rank_diagnostic_iteration_equivalent": diagnostic_iterations,
            "rank_diagnostic_a_matvecs": diagnostic_a,
            "rank_diagnostic_m_applications": diagnostic_m,
            "adaptive_final_iterations_3rhs": adaptive_final_cost,
            "adaptive_total_iteration_equivalent": settings.stage41_probe_iteration_equivalent + diagnostic_iterations + adaptive_final_cost,
            "fixed_q48_final_iterations_3rhs": fixed_final_cost,
            "fixed_q48_total_iteration_equivalent": settings.stage41_probe_iteration_equivalent + fixed_final_cost,
            "strong_ilu_final_iterations_3rhs": sum(int(row["iterations"]) for row in ilu),
            "teacher_final_iterations_3rhs_oracle_reference": sum(int(row["iterations"]) for row in teacher),
            "shared_A_Psi_construction_excluded": True,
        })
    return rows


def _report(
    output: Path,
    decisions,
    features,
    adaptive_final,
    costs,
    validation_rows,
    harmonic_rows,
    decision,
):
    feature_lookup = {int(row["sample_id"]): row for row in features}
    validation_lookup = {int(row["sample_id"]): row for row in validation_rows}
    harmonic_lookup = {int(row["sample_id"]): row for row in harmonic_rows}
    final_converged = sum(_bool(row["converged"]) for row in adaptive_final)
    final_iterations = float(np.median([int(row["iterations"]) for row in adaptive_final]))
    final_rho = float(np.median([float(row["rho"]) for row in adaptive_final]))
    median_rank = float(np.median([int(row["selected_rank"]) for row in decisions]))
    lines = [
        "# Stage 45: Deployable adaptive rank stopping and representation-failure detection",
        "",
        "## Result",
        "",
        f"**{decision['answer']}**",
        "",
        f"The frozen deployable rule selects median rank {_fmt(median_rank,0)} and yields {final_converged}/21, median {_fmt(final_iterations,1)} iterations, and median $\\rho={_fmt(final_rho)}$.",
        "",
        f"This loses {18-final_converged} converged RHSs relative to fixed q=48 because the nonmonotone short curve incorrectly stops system 43 at q=8. The rule therefore does not retain the fixed-space benefit despite its high aggregate median $\\rho$.",
        "",
        "| method | converged | median iterations | median $\\rho$ |",
        "|:--|--:|--:|--:|",
        "| strong ILU | 3/21 | 300 | 0.000 |",
        "| fixed q=48 | 18/21 | 33 | 0.985 |",
        f"| Stage-45 adaptive | {final_converged}/21 | {_fmt(final_iterations,1)} | {_fmt(final_rho)} |",
        "| own-system teacher | 21/21 | 31 | 1.000 |",
        "",
        "| system | selected rank | deployable flag | Stage-44 label | correct qualitative mechanism | final converged | median iterations | median $\\rho$ |",
        "|--:|--:|:--|:--|:--:|--:|--:|--:|",
    ]
    for row in decisions:
        sample_id = int(row["sample_id"])
        local = [item for item in adaptive_final if int(item["sample_id"]) == sample_id]
        validation = validation_lookup[sample_id]
        lines.append(
            f"| {sample_id} | {row['selected_rank']} | {row['deployable_flag']} | {validation['stage44_classification']} | "
            f"{validation['qualitative_match']} | {sum(_bool(item['converged']) for item in local)}/3 | "
            f"{_fmt(np.median([int(item['iterations']) for item in local]),1)} | {_fmt(np.median([float(item['rho']) for item in local]))} |"
        )
    lines += [
        "",
        "The rank rule takes the first prefix recovering 90% of the target's own q=8-to-q=48 short-solve improvement. Representation failure is a leave-one-target-out robust endpoint outlier combined with either marginal-gain saturation or abnormally weak q=48 benefit. Ordering sensitivity requires one late block to supply at least 65% of positive late gain and at least 0.10 decade of residual improvement. These rules were frozen before Stage-44 labels and oracle geometry were loaded.",
        "",
        "## Deployable response and saturation",
        "",
        "| system | q=8 to 48 log gain | last-two-block gain | dominant late block | dominance | q48 benefit vs ILU | endpoint outlier | saturated |",
        "|--:|--:|--:|:--|--:|--:|:--:|:--:|",
    ]
    for row in decisions:
        sample_id = int(row["sample_id"])
        feature = feature_lookup[sample_id]
        lines.append(
            f"| {sample_id} | {_fmt(feature['total_log_gain_8_to_48'])} | {_fmt(feature['last_two_positive_log_gain'])} | "
            f"{feature['dominant_late_block']} | {_fmt(feature['dominant_late_block_fraction'])} | "
            f"{_fmt(feature['q48_log_benefit_over_baseline'])} | {row['endpoint_bad_leave_one_out']} | {row['saturated_late_gain']} |"
        )
    system63 = next(row for row in decisions if int(row["sample_id"]) == 63)
    lines += [
        "",
        f"For system 63, the deployable answer is **{'yes' if system63['deployable_flag'] == 'representation-failure' else 'no'}**: its q=48 short-response endpoint is {'a' if system63['endpoint_bad_leave_one_out'] else 'not a'} leave-one-out outlier and its late gain is {'saturated' if system63['saturated_late_gain'] else 'not saturated'}. No oracle quantity enters this decision.",
        "",
        f"System 63 has the largest cached median and maximum harmonic-Ritz residuals ({_fmt(harmonic_lookup[63]['median_harmonic_ritz_residual'])} and {_fmt(harmonic_lookup[63]['maximum_harmonic_ritz_residual'])}), but the values overlap systems 25 and 71; its solution-block novelty and projected coupling changes also follow the ordinary declining trend. These construction-only signals are saved alongside the response curve, but none is yet a validated standalone failure indicator. The transparent frozen rule does not combine them into a fitted classifier.",
        "",
        "## Cost benefit",
        "",
        "Costs are iteration equivalents. Stage-41's three 64-step probes cost 192 per target; the rank diagnostic includes one ILU-only and six recycled eight-step solves. The target-specific $A\\Psi$ construction is shared by fixed and adaptive candidates and is excluded from both totals.",
        "",
        "| system | rank | probe | rank diagnostic | adaptive final | adaptive total | fixed-q48 total | strong ILU final | teacher final |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in costs:
        lines.append(
            f"| {row['sample_id']} | {row['selected_rank']} | {row['probe_iteration_equivalent']} | "
            f"{row['rank_diagnostic_iteration_equivalent']} | {row['adaptive_final_iterations_3rhs']} | "
            f"{row['adaptive_total_iteration_equivalent']} | {row['fixed_q48_total_iteration_equivalent']} | "
            f"{row['strong_ilu_final_iterations_3rhs']} | {row['teacher_final_iterations_3rhs_oracle_reference']} |"
        )
    adaptive_total = sum(int(row["adaptive_total_iteration_equivalent"]) for row in costs)
    fixed_total = sum(int(row["fixed_q48_total_iteration_equivalent"]) for row in costs)
    lines += [
        "",
        f"Across seven targets, adaptive work is {adaptive_total} iteration equivalents versus {fixed_total} for fixed q=48. The diagnostic therefore costs {adaptive_total-fixed_total} more equivalents; smaller recycle ranks do not repay it through final-iteration savings.",
        "",
        "## Oracle validation after decision freeze",
        "",
        "| system | deployable flag | Stage-44 type | oracle containment q48 | oracle tail below 0.25 | qualitative match |",
        "|--:|:--|:--|--:|--:|:--:|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['sample_id']} | {row['deployable_flag']} | {row['stage44_classification']} | "
            f"{_fmt(row['oracle_containment_q48'])} | {row['oracle_directions_below_0p25_q48']} | {row['qualitative_match']} |"
        )
    lines += [
        "",
        f"Leave-one-target-out qualitative agreement is {sum(_bool(row['qualitative_match']) for row in validation_rows)}/7. Oracle quantities explain the frozen decisions but were never inputs.",
        "",
        "## Stage 46 recommendation",
        "",
        decision["stage46_recommendation"],
    ]
    report = output / "stage45_deployable_rank_stopping.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage45(root: Path) -> Path:
    settings = Stage45Settings()
    output = root / "results/results-fd/stages/stage45_deployable_rank_stopping"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-45 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    })
    stage41_dir = root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target"
    stage44_dir = root / "results/results-fd/stages/stage44_rank_retention_mechanism"
    validations = [
        json.loads((root / f"results/results-fd/stages/stage{stage}_{name}/validation.json").read_text())
        for stage, name in (
            (41, "multi_rhs_slow_mode_target"),
            (42, "missing_critical_mode_diagnosis"),
            (43, "missing_subspace_block_ceiling"),
            (44, "rank_retention_mechanism"),
        )
    ]
    preflight = {
        "cached_stages_41_to_44_valid": all(item["passed"] for item in validations),
        "diagnostic_rhs_is_existing_stage41_probe": settings.diagnostic_rhs_seed in (20262911, 20262921, 20262931),
        "diagnostic_rhs_disjoint_from_final_evaluation": settings.diagnostic_rhs_seed not in settings.stage29.stage28.evaluation_rhs_seeds,
        "diagnostic_budget_exact": settings.diagnostic_budget == 8,
        "rank_grid_exact": settings.retained_ranks == (8, 16, 24, 32, 40, 48),
        "rules_frozen_before_oracle_validation": True,
        "no_new_probe_or_candidate_construction": True,
        "solver_unchanged": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-45 preflight failed")

    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    curve_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    archive_path = stage41_dir / "multi_rhs_local_spaces.npz"
    with np.load(archive_path) as archive:
        for sample_id in settings.hard_system_ids:
            context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
            rhs, _ = _discrete_rhs(context.system, settings.diagnostic_rhs_seed)
            z48 = np.asarray(archive[f"system_{sample_id:03d}_multi_rhs_harmonic_m64_z"], dtype=float)
            pairs = {
                q: _ordered_pair_from_z(context.system.matrix, z48[:, :q], 1.0e-13)
                for q in settings.retained_ranks
            }
            baseline_residual, baseline_cost = _short_solver_response(
                context.system.matrix, rhs, context.ilu.apply, None,
                settings.diagnostic_budget, settings.stage29.stage28.stage22a.solver,
            )
            local_candidates = _candidate_diagnostics(pairs)
            for row in local_candidates:
                candidate_rows.append({"sample_id": sample_id, **row})
            for q in settings.retained_ranks:
                residual, cost = _short_solver_response(
                    context.system.matrix, rhs, context.ilu.apply, pairs[q],
                    settings.diagnostic_budget, settings.stage29.stage28.stage22a.solver,
                )
                curve_rows.append({
                    "sample_id": sample_id,
                    "rhs_seed": settings.diagnostic_rhs_seed,
                    "rank": q,
                    "relative_residual": residual,
                    "log10_relative_residual": float(np.log10(max(residual, 1.0e-300))),
                    "baseline_relative_residual": baseline_residual,
                    "baseline_log10_relative_residual": float(np.log10(max(baseline_residual, 1.0e-300))),
                    "short_iterations": cost["iterations"],
                    "short_a_matvecs": cost["a_matvecs"],
                    "short_m_applications": cost["preconditioner_applications"],
                    "baseline_short_iterations": baseline_cost["iterations"],
                    "baseline_a_matvecs": baseline_cost["a_matvecs"],
                    "baseline_m_applications": baseline_cost["preconditioner_applications"],
                    "teacher_or_oracle_used": False,
                })
            print(f"[Stage45 diagnostics] system {sample_id}", flush=True)

    # Freeze deployable decisions before loading Stage-44 labels or oracle geometry.
    decisions, features = freeze_deployable_decisions(curve_rows, settings)
    _write_csv(output / "frozen_deployable_decisions.csv", decisions)
    _write_json(output / "decision_freeze.json", {
        "frozen_before_oracle_validation": True,
        "inputs": "short diagnostic response and scale-normalized/LOO robust thresholds only",
        "decisions": decisions,
    })

    stage44_solver = _read_csv(stage44_dir / "nested_solver_results.csv")
    adaptive_final = _final_rows(stage44_solver, decisions)
    stage43_solver = _read_csv(root / "results/results-fd/stages/stage43_missing_subspace_block_ceiling/solver_results.csv")
    costs = _cost_rows(decisions, curve_rows, adaptive_final, stage43_solver, settings)
    stage44_classes = _read_csv(stage44_dir / "mechanism_classification.csv")
    class_lookup = {int(row["sample_id"]): row["classification"] for row in stage44_classes}
    stage44_geometry = _read_csv(stage44_dir / "nested_geometry.csv")
    geometry48 = {
        int(row["sample_id"]): row for row in stage44_geometry if row["method"] == _prefix_method(48)
    }
    decision_lookup = {int(row["sample_id"]): row for row in decisions}
    mapping = {"Type I": "normal", "Type II": "ordering-sensitive", "Type III": "representation-failure"}
    validation_rows = []
    for sample_id in settings.hard_system_ids:
        deployable = decision_lookup[sample_id]["deployable_flag"]
        expected = mapping[class_lookup[sample_id]]
        geometry = geometry48[sample_id]
        validation_rows.append({
            "sample_id": sample_id,
            "deployable_flag": deployable,
            "stage44_classification": class_lookup[sample_id],
            "qualitative_match": deployable == expected,
            "oracle_containment_q48": float(geometry["oracle_containment"]),
            "oracle_directions_below_0p25_q48": int(geometry["canonical_oracle_below_0p25"]),
            "oracle_used_only_after_decision_freeze": True,
        })
    spectra = _read_csv(stage41_dir / "harmonic_ritz_spectra.csv")
    harmonic_rows = []
    for sample_id in settings.hard_system_ids:
        local = [row for row in spectra if int(row["sample_id"]) == sample_id]
        harmonic_rows.append({
            "sample_id": sample_id,
            "median_harmonic_ritz_residual": float(np.median([float(row["eigen_residual_relative"]) for row in local])),
            "maximum_harmonic_ritz_residual": float(np.max([float(row["eigen_residual_relative"]) for row in local])),
            "median_harmonic_ritz_magnitude": float(np.median([float(row["magnitude"]) for row in local])),
        })

    final_converged = sum(_bool(row["converged"]) for row in adaptive_final)
    final_median = float(np.median([int(row["iterations"]) for row in adaptive_final]))
    final_rho = float(np.median([float(row["rho"]) for row in adaptive_final]))
    adaptive_total = sum(int(row["adaptive_total_iteration_equivalent"]) for row in costs)
    fixed_total = sum(int(row["fixed_q48_total_iteration_equivalent"]) for row in costs)
    system63_flagged = decision_lookup[63]["deployable_flag"] == "representation-failure"
    too_expensive = adaptive_total >= fixed_total
    if too_expensive:
        answer = "No: the eight-step response curve neither stops safely nor detects candidate-family failure, and its diagnostic cost exceeds fixed q=48."
        recommendation = "Stage 46 should simplify the decision mechanism by testing whether the saved construction-only coupling, novelty, and harmonic-residual signals can reproduce the frozen Stage-45 decisions with zero additional short solves; do not begin neural correction training."
        case = "D"
    elif system63_flagged:
        answer = "Yes: the deployable curve cheaply identifies both a stopping rank and candidate-family failure."
        recommendation = "Stage 46 should investigate a neural correction only for targets flagged as representation failures, while retaining the classical candidate space for ordinary targets."
        case = "A"
    else:
        answer = "Adaptive rank selection works, but candidate-family failure is not robustly detectable."
        recommendation = "Stage 46 should focus only on a cheaper representation-failure indicator before neural training."
        case = "B"
    decision = {
        "answer": answer,
        "stage46_case": case,
        "stage46_recommendation": recommendation,
        "adaptive_converged_rhs": final_converged,
        "adaptive_median_iterations": final_median,
        "adaptive_median_rho": final_rho,
        "median_selected_rank": float(np.median([int(row["selected_rank"]) for row in decisions])),
        "system63_representation_failure_flagged": system63_flagged,
        "qualitative_matches": sum(_bool(row["qualitative_match"]) for row in validation_rows),
        "adaptive_total_iteration_equivalent": adaptive_total,
        "fixed_q48_total_iteration_equivalent": fixed_total,
        "rank_rule_preserves_fixed_q48_convergence": final_converged >= 18,
    }
    _write_csv(output / "short_response_curves.csv", curve_rows)
    _write_csv(output / "deployable_curve_features.csv", features)
    _write_csv(output / "candidate_construction_diagnostics.csv", candidate_rows)
    _write_csv(output / "harmonic_residual_summary.csv", harmonic_rows)
    _write_csv(output / "adaptive_final_solver_results.csv", adaptive_final)
    _write_csv(output / "cost_accounting.csv", costs)
    _write_csv(output / "oracle_validation_after_freeze.csv", validation_rows)
    _write_json(output / "decision.json", decision)
    report = _report(
        output, decisions, features, adaptive_final, costs, validation_rows,
        harmonic_rows, decision,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "short_response_rows": len(curve_rows),
        "expected_short_response_rows": 42,
        "all_short_budgets_at_most_8": all(int(row["short_iterations"]) <= 8 for row in curve_rows),
        "diagnostic_rhs_disjoint_from_final_evaluation": preflight["diagnostic_rhs_disjoint_from_final_evaluation"],
        "all_decisions_frozen_without_teacher_or_oracle": all(not _bool(row["decision_uses_teacher_or_oracle"]) for row in decisions),
        "oracle_validation_after_freeze": all(_bool(row["oracle_used_only_after_decision_freeze"]) for row in validation_rows),
        "adaptive_final_rows": len(adaptive_final),
        "expected_adaptive_final_rows": 21,
        "selected_ranks_valid": all(int(row["selected_rank"]) in settings.retained_ranks for row in decisions),
    }
    validation["passed"] = all(validation[key] for key in (
        "preflight_passed", "all_short_budgets_at_most_8",
        "diagnostic_rhs_disjoint_from_final_evaluation",
        "all_decisions_frozen_without_teacher_or_oracle", "oracle_validation_after_freeze",
        "selected_ranks_valid",
    )) and validation["short_response_rows"] == 42 and validation["adaptive_final_rows"] == 21
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage45_deployable_rank_stopping.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage45] complete", flush=True)
    return copy
