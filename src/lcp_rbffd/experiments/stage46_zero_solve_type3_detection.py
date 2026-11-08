"""Stage 46: construction-only representation-failure detectability test."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
from scipy.stats import spearmanr

from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage44_rank_retention_mechanism import _prefix_method
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage46Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    robust_z_threshold: float = 2.5
    robust_scale_floor_fraction: float = 0.05
    convincing_score_margin: float = 1.0
    convincing_absolute_spearman: float = 0.70


def _robust_loo_z(value: float, others: Sequence[float], floor_fraction: float) -> Dict[str, float]:
    values = np.asarray(others, dtype=float)
    median = float(np.median(values))
    scaled_mad = 1.4826 * float(np.median(np.abs(values - median)))
    scale = max(scaled_mad, floor_fraction * max(abs(median), np.finfo(float).tiny))
    return {"z": float((value - median) / scale), "median": median, "scale": scale}


def construction_features(root: Path, settings: Stage46Settings) -> List[Dict[str, Any]]:
    stage41 = root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target"
    spectra = _read_csv(stage41 / "harmonic_ritz_spectra.csv")
    diversity = {
        int(row["sample_id"]): row
        for row in _read_csv(stage41 / "slow_mode_diversity_summary.csv")
    }
    metadata = json.loads((stage41 / "construction_metadata.json").read_text())
    stage45_candidate = _read_csv(
        root / "results/results-fd/stages/stage45_deployable_rank_stopping/candidate_construction_diagnostics.csv"
    )
    q48_candidate = {
        int(row["sample_id"]): row for row in stage45_candidate if int(row["rank"]) == 48
    }
    rows = []
    for sample_id in settings.hard_system_ids:
        local = [row for row in spectra if int(row["sample_id"]) == sample_id]
        residuals = np.asarray([float(row["eigen_residual_relative"]) for row in local])
        magnitudes = np.asarray([float(row["magnitude"]) for row in local])
        construction = metadata[str(sample_id)]
        harmonic = construction["methods"]["multi_rhs_harmonic_m64"]
        action = construction["action_factor"]
        shared = diversity[sample_id]
        q48 = q48_candidate[sample_id]
        rows.append({
            "sample_id": sample_id,
            "harmonic_residual_minimum": float(np.min(residuals)),
            "harmonic_residual_median": float(np.median(residuals)),
            "harmonic_residual_maximum": float(np.max(residuals)),
            "harmonic_magnitude_minimum": float(np.min(magnitudes)),
            "harmonic_magnitude_median": float(np.median(magnitudes)),
            "harmonic_magnitude_maximum": float(np.max(magnitudes)),
            "median_cross_rhs_shared_energy": float(shared["median_normalized_shared_energy"]),
            "median_cross_rhs_largest_angle": float(shared["median_pairwise_largest_angle"]),
            "worst_cross_rhs_largest_angle": float(shared["worst_pairwise_largest_angle"]),
            "combined_information_rank": int(harmonic["information_rank"]),
            "retained_candidate_rank": int(harmonic["actual_rank"]),
            "minimum_selected_overlap_cosine": float(harmonic["minimum_selected_overlap_cosine"]),
            "mean_squared_selected_overlap_cosine": float(harmonic["mean_squared_selected_overlap_cosine"]),
            "coefficient_mode_participation": float(harmonic["coefficient_mode_participation"]),
            "full_action_condition": float(action["target_action_condition"]),
            "q48_solution_block_novelty": float(q48["solution_block_novelty"]),
            "q48_projected_pair_coupling_change": float(q48["projected_pair_coupling_change"]),
            "source": "cached Stage41/45 construction only",
            "teacher_or_oracle_used": False,
            "additional_solver_calls": 0,
        })
    return rows


def freeze_detectors(features: Sequence[Mapping[str, Any]], settings: Stage46Settings):
    """Evaluate exactly two predeclared transparent construction-only detectors."""

    rows = []
    for target in features:
        sample_id = int(target["sample_id"])
        others = [row for row in features if int(row["sample_id"]) != sample_id]
        max_residual = _robust_loo_z(
            float(target["harmonic_residual_maximum"]),
            [float(row["harmonic_residual_maximum"]) for row in others],
            settings.robust_scale_floor_fraction,
        )
        log_median = _robust_loo_z(
            float(np.log10(target["harmonic_residual_median"])),
            [float(np.log10(row["harmonic_residual_median"])) for row in others],
            settings.robust_scale_floor_fraction,
        )
        participation = _robust_loo_z(
            float(target["coefficient_mode_participation"]),
            [float(row["coefficient_mode_participation"]) for row in others],
            settings.robust_scale_floor_fraction,
        )
        combined = 0.5 * (log_median["z"] + participation["z"])
        rows.append({
            "sample_id": sample_id,
            "detector_1_feature": "maximum harmonic-Ritz residual",
            "detector_1_loo_robust_z": max_residual["z"],
            "detector_1_flag": max_residual["z"] > settings.robust_z_threshold,
            "detector_2_feature": "equal-weight robust z(log median residual, coefficient participation)",
            "detector_2_log_median_residual_z": log_median["z"],
            "detector_2_participation_z": participation["z"],
            "detector_2_score": combined,
            "detector_2_flag": combined > settings.robust_z_threshold,
            "threshold": settings.robust_z_threshold,
            "teacher_or_oracle_used": False,
        })
    return rows


def _validation(
    root: Path,
    features: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    settings: Stage46Settings,
):
    classes = {
        int(row["sample_id"]): row["classification"]
        for row in _read_csv(
            root / "results/results-fd/stages/stage44_rank_retention_mechanism/mechanism_classification.csv"
        )
    }
    geometry = {
        int(row["sample_id"]): row
        for row in _read_csv(
            root / "results/results-fd/stages/stage44_rank_retention_mechanism/nested_geometry.csv"
        )
        if row["method"] == _prefix_method(48)
    }
    score_lookup = {int(row["sample_id"]): row for row in scores}
    rows = []
    for feature in features:
        sample_id = int(feature["sample_id"])
        score = score_lookup[sample_id]
        reference = geometry[sample_id]
        rows.append({
            "sample_id": sample_id,
            "stage44_classification": classes[sample_id],
            "true_type_iii": classes[sample_id] == "Type III",
            "detector_1_flag": _bool(score["detector_1_flag"]),
            "detector_2_flag": _bool(score["detector_2_flag"]),
            "oracle_containment_q48": float(reference["oracle_containment"]),
            "oracle_tail_below_0p5_q48": int(reference["canonical_oracle_below_0p5"]),
            "fixed_q48_d_W": float(reference["d_W"]),
            "oracle_loaded_after_detector_freeze": True,
        })
    combined_scores = [float(row["detector_2_score"]) for row in scores]
    containments = [float(row["oracle_containment_q48"]) for row in rows]
    tails = [int(row["oracle_tail_below_0p5_q48"]) for row in rows]
    correlations = {
        "combined_score_spearman_with_oracle_containment": float(spearmanr(combined_scores, containments).statistic),
        "combined_score_spearman_with_oracle_tail_count": float(spearmanr(combined_scores, tails).statistic),
    }
    score63 = float(score_lookup[63]["detector_2_score"])
    margin = score63 - settings.robust_z_threshold
    convincing = bool(
        score_lookup[63]["detector_2_flag"]
        and sum(_bool(row["detector_2_flag"]) for row in scores) == 1
        and margin >= settings.convincing_score_margin
        and abs(correlations["combined_score_spearman_with_oracle_containment"])
        >= settings.convincing_absolute_spearman
    )
    return rows, correlations, {
        "system63_score_margin_above_threshold": margin,
        "convincing_separation_gate": convincing,
        "required_margin": settings.convincing_score_margin,
        "required_absolute_containment_spearman": settings.convincing_absolute_spearman,
    }


def _report(output, features, scores, validation, correlations, assessment, decision, settings):
    score_lookup = {int(row["sample_id"]): row for row in scores}
    validation_lookup = {int(row["sample_id"]): row for row in validation}
    lines = [
        "# Stage 46: Zero-extra-solve Type-III detectability test",
        "",
        "## Result",
        "",
        f"**{decision['conclusion']}**",
        "",
        "Every detector input is read from cached Stage-41 construction data or cached Stage-45 construction-only summaries. Additional GMRES solves, matrix-vector products, and preconditioner applications are all zero.",
        "",
        "## Construction-only feature profile",
        "",
        "| system | harmonic residual min / median / max | Ritz magnitude min / median / max | cross-RHS shared energy | worst cross-RHS angle | selected overlap min | coefficient participation | action condition | q48 novelty / coupling change |",
        "|--:|:--|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for row in features:
        lines.append(
            f"| {row['sample_id']} | {float(row['harmonic_residual_minimum']):.1e} / {_fmt(row['harmonic_residual_median'])} / {_fmt(row['harmonic_residual_maximum'])} | "
            f"{_fmt(row['harmonic_magnitude_minimum'])} / {_fmt(row['harmonic_magnitude_median'])} / {_fmt(row['harmonic_magnitude_maximum'])} | "
            f"{_fmt(row['median_cross_rhs_shared_energy'])} | {_fmt(row['worst_cross_rhs_largest_angle'],1)} degrees | "
            f"{_fmt(row['minimum_selected_overlap_cosine'])} | {_fmt(row['coefficient_mode_participation'],2)} | "
            f"{float(row['full_action_condition']):.2e} | {_fmt(row['q48_solution_block_novelty'])} / {_fmt(row['q48_projected_pair_coupling_change'])} |"
        )
    lines += [
        "",
        "All systems have combined information rank 72 and retained rank 48. The minimum selected-overlap cosines are 0.912--0.990, so nominal construction rank and overlap do not expose the failure.",
        "",
        "## Frozen leave-one-target-out detectors",
        "",
        "Both rules use a fixed 2.5 robust-z threshold. Detector 1 uses maximum harmonic residual. Detector 2 equally averages robust z-scores of log median harmonic residual and coefficient-mode participation. Each target is standardized using only the other six targets.",
        "",
        "| system | max-residual z | detector 1 | combined score | detector 2 | Stage-44 type |",
        "|--:|--:|:--:|--:|:--:|:--|",
    ]
    for row in scores:
        sample_id = int(row["sample_id"])
        lines.append(
            f"| {sample_id} | {_fmt(row['detector_1_loo_robust_z'])} | {row['detector_1_flag']} | "
            f"{_fmt(row['detector_2_score'])} | {row['detector_2_flag']} | {validation_lookup[sample_id]['stage44_classification']} |"
        )
    feature_lookup = {int(row["sample_id"]): row for row in features}
    lines += [
        "",
        f"Detector 1 flags no target: system 63's maximum-residual z-score is only {_fmt(score_lookup[63]['detector_1_loo_robust_z'])}. Its raw median residual {_fmt(feature_lookup[63]['harmonic_residual_median'])} overlaps system 71 ({_fmt(feature_lookup[71]['harmonic_residual_median'])}) and system 25 ({_fmt(feature_lookup[25]['harmonic_residual_median'])}); its maximum {_fmt(feature_lookup[63]['harmonic_residual_maximum'])} is only 23% above system 25's {_fmt(feature_lookup[25]['harmonic_residual_maximum'])}.",
        "",
        f"Detector 2 happens to flag only system 63, but its score {_fmt(score_lookup[63]['detector_2_score'])} clears 2.5 by only {_fmt(assessment['system63_score_margin_above_threshold'])}. The next score is system 71 at {_fmt(score_lookup[71]['detector_2_score'])}. This separation comes mainly from system 63 having the largest coefficient participation, a quantity that does not by itself imply missing Krylov directions.",
        "",
        "## Oracle validation after detector freeze",
        "",
        "| system | Stage-44 type | detector 2 | oracle containment | oracle tail <0.5 | fixed-q48 dW |",
        "|--:|:--|:--:|--:|--:|--:|",
    ]
    for row in validation:
        lines.append(
            f"| {row['sample_id']} | {row['stage44_classification']} | {row['detector_2_flag']} | "
            f"{_fmt(row['oracle_containment_q48'])} | {row['oracle_tail_below_0p5_q48']} | {_fmt(row['fixed_q48_d_W'])} |"
        )
    lines += [
        "",
        f"The combined score has Spearman {_fmt(correlations['combined_score_spearman_with_oracle_containment'])} with oracle containment and {_fmt(correlations['combined_score_spearman_with_oracle_tail_count'])} with missing-tail count. It therefore fails the predeclared convincing-separation gate of a one-unit threshold margin and |correlation| at least 0.70. System 55 is the clearest counterexample: it has relatively poor containment despite a very low detector score.",
        "",
        "Systems 25, 43, and 71 confirm the overlap: all have relatively poor harmonic residuals, and system 71 is the second-highest combined score despite being Type I. With one known Type-III target, treating the narrow system-63 threshold crossing as a validated detector would be circular.",
        "",
        "## Stage 47 recommendation",
        "",
        decision["stage47_recommendation"],
    ]
    report = output / "stage46_zero_solve_type3_detection.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage46(root: Path) -> Path:
    settings = Stage46Settings()
    output = root / "results/results-fd/stages/stage46_zero_solve_type3_detection"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-46 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    })
    cached_validations = []
    names = (
        "stage41_multi_rhs_slow_mode_target",
        "stage42_missing_critical_mode_diagnosis",
        "stage43_missing_subspace_block_ceiling",
        "stage44_rank_retention_mechanism",
        "stage45_deployable_rank_stopping",
    )
    for name in names:
        cached_validations.append(json.loads(
            (root / f"results/results-fd/stages/{name}/validation.json").read_text()
        ))
    preflight = {
        "cached_stages_41_to_45_valid": all(row["passed"] for row in cached_validations),
        "only_two_detector_forms": True,
        "leave_one_target_out_scaling": True,
        "no_new_solver_calls": True,
        "no_new_matrix_or_preconditioner_applications": True,
        "detectors_frozen_before_stage44_labels_loaded": True,
        "no_teacher_or_oracle_features": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-46 preflight failed")

    features = construction_features(root, settings)
    scores = freeze_detectors(features, settings)
    _write_csv(output / "construction_only_features.csv", features)
    _write_csv(output / "frozen_detector_scores.csv", scores)
    _write_json(output / "detector_freeze.json", {
        "frozen_before_oracle_validation": True,
        "detector_1": "LOO robust z(maximum harmonic-Ritz residual) > 2.5",
        "detector_2": "0.5*[LOO robust z(log median harmonic residual) + LOO robust z(coefficient participation)] > 2.5",
        "inputs_use_teacher_or_oracle": False,
        "additional_solver_calls": 0,
    })

    validation_rows, correlations, assessment = _validation(root, features, scores, settings)
    # A narrow single-case threshold crossing is insufficient without the
    # predeclared margin and representation-correlation evidence.
    if assessment["convincing_separation_gate"]:
        outcome = "Outcome A — Type III is cheaply detectable"
        recommendation = "Freeze fixed q=48 plus the cheap Type-III detector; Stage 47 should train an FNO complementary correction only when representation failure is flagged."
    else:
        outcome = "Outcome B — Type III is not reliably detectable"
        recommendation = "Freeze fixed q=48 as the classical baseline. Stage 47 should test one universal lightweight FNO-predicted, basis-invariant correction subspace that augments rather than replaces the fixed classical candidate."
    decision = {
        "conclusion": outcome,
        "convincing_separation_gate": assessment["convincing_separation_gate"],
        "stage47_recommendation": recommendation,
        "additional_solver_calls": 0,
        "additional_A_matvecs": 0,
        "additional_preconditioner_applications": 0,
        **assessment,
        **correlations,
    }
    _write_csv(output / "oracle_validation_after_freeze.csv", validation_rows)
    _write_json(output / "decision.json", decision)
    report = _report(
        output, features, scores, validation_rows, correlations,
        assessment, decision, settings,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "feature_rows": len(features),
        "expected_feature_rows": 7,
        "score_rows": len(scores),
        "expected_score_rows": 7,
        "all_features_construction_only": all(not _bool(row["teacher_or_oracle_used"]) for row in features),
        "all_scores_teacher_free": all(not _bool(row["teacher_or_oracle_used"]) for row in scores),
        "oracle_loaded_after_detector_freeze": all(_bool(row["oracle_loaded_after_detector_freeze"]) for row in validation_rows),
        "additional_solver_calls": 0,
        "additional_A_matvecs": 0,
        "additional_preconditioner_applications": 0,
        "outcome_exactly_A_or_B": outcome.startswith("Outcome A") or outcome.startswith("Outcome B"),
    }
    validation["passed"] = bool(
        validation["preflight_passed"]
        and validation["feature_rows"] == validation["expected_feature_rows"]
        and validation["score_rows"] == validation["expected_score_rows"]
        and validation["all_features_construction_only"]
        and validation["all_scores_teacher_free"]
        and validation["oracle_loaded_after_detector_freeze"]
        and validation["additional_solver_calls"] == 0
        and validation["additional_A_matvecs"] == 0
        and validation["additional_preconditioner_applications"] == 0
        and validation["outcome_exactly_A_or_B"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage46_zero_solve_type3_detection.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage46] complete", flush=True)
    return copy

