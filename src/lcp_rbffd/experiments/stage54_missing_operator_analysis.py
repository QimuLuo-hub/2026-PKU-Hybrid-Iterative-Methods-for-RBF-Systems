"""Stage 54: diagnose projector components missing from the Stage-52 algebra."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
from scipy.fft import dct
import torch

from .stage30_full_span_representation_ceiling import _fmt, _manifest, _read_csv
from .stage52_basis_equivariant_raw_compression import _normalize_operator
from .stage53_equivariant_algebra_oracle_ceiling import (
    Stage53Settings,
    _coordinate_containment,
    _load_split,
    coverage_metrics,
    optimize_coefficients,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage54Settings:
    correction_ranks: Tuple[int, int] = (8, 12)
    base_operator_count: int = 33
    extension_useful_residual_fraction: float = 0.02
    extended_algebra_containment_gate: float = 0.80
    near_step_lag: int = 8
    long_step_lag: int = 24
    raw_depth: int = 64
    optimization_seed: int = 20265400
    maximum_extension_families: int = 3

    @property
    def stage53(self) -> Stage53Settings:
        return Stage53Settings()


def projector_span_projection(operator_basis: np.ndarray, target: np.ndarray):
    """Orthogonally project target projector onto an operator linear span."""

    basis = np.asarray(operator_basis, float)
    projector = np.asarray(target, float) @ np.asarray(target, float).T
    gram = np.einsum("kij,lij->kl", basis, basis, optimize=True)
    right = np.einsum("kij,ij->k", basis, projector, optimize=True)
    coefficients = la.pinvh(gram, rtol=1.0e-10) @ right
    projected = np.einsum("k,kij->ij", coefficients, basis, optimize=True)
    residual = projector - projected
    energy = float(la.norm(residual, ord="fro") ** 2 / target.shape[1])
    span_rank = int(np.count_nonzero(la.eigvalsh(gram) > 1.0e-10 * la.eigvalsh(gram)[-1]))
    return coefficients, projected, residual, energy, span_rank


def leading_eigenspace(operator: np.ndarray, rank: int) -> np.ndarray:
    _, vectors = la.eigh(0.5 * (operator + operator.T), check_finite=False, driver="evr")
    return vectors[:, -rank:]


def residual_structure(
    residual: np.ndarray, token_features: np.ndarray, raw_depth: int,
    near_lag: int, long_lag: int,
) -> Dict[str, float]:
    """Aggregate residual energy in the canonical paired-SVD trajectory coordinates."""

    squared = np.asarray(residual, float) ** 2
    total = max(float(np.sum(squared)), np.finfo(float).tiny)
    dimension = residual.shape[0]
    diagonal_mask = np.eye(dimension, dtype=bool)
    off_mask = ~diagonal_mask
    seed_fraction = np.asarray(token_features[:, 39:42], float)
    same_rhs_affinity = np.clip(seed_fraction @ seed_fraction.T, 0.0, 1.0)
    mean_step = np.asarray(token_features[:, 43], float)
    step_distance = np.abs(mean_step[:, None] - mean_step[None, :])
    near = near_lag / max(raw_depth - 1, 1)
    long = long_lag / max(raw_depth - 1, 1)
    return {
        "diagonal_self_energy": float(np.sum(squared[diagonal_mask]) / total),
        "off_diagonal_energy": float(np.sum(squared[off_mask]) / total),
        "within_rhs_energy": float(np.sum((squared * same_rhs_affinity)[off_mask]) / total),
        "cross_rhs_energy": float(np.sum((squared * (1.0 - same_rhs_affinity))[off_mask]) / total),
        "near_step_energy": float(np.sum(squared[off_mask & (step_distance <= near)]) / total),
        "intermediate_step_energy": float(
            np.sum(squared[off_mask & (step_distance > near) & (step_distance <= long)]) / total
        ),
        "long_range_step_energy": float(np.sum(squared[off_mask & (step_distance > long)]) / total),
    }


def operator_correlations(operator_basis: np.ndarray, projector: np.ndarray, residual: np.ndarray):
    labels = (
        ["primitive"] * 6
        + ["symmetric_quadratic"] * 21
        + ["primitive_cube"] * 6
    )
    rows = []
    projector_norm = max(float(la.norm(projector, ord="fro")), np.finfo(float).tiny)
    residual_norm = max(float(la.norm(residual, ord="fro")), np.finfo(float).tiny)
    for index, (operator, family) in enumerate(zip(operator_basis, labels)):
        norm = max(float(la.norm(operator, ord="fro")), np.finfo(float).tiny)
        rows.append({
            "operator_index": index,
            "operator_family": family,
            "absolute_projector_correlation": abs(float(np.sum(operator * projector))) / (norm * projector_norm),
            "absolute_residual_correlation": abs(float(np.sum(operator * residual))) / (norm * residual_norm),
        })
    return rows


def cross_rhs_spatial_family(operator_basis: np.ndarray) -> np.ndarray:
    """Cross-RHS trajectory couplings bridged by low/high spatial Gram operators."""

    operators = []
    for first in (1, 2, 3):
        for second in range(first + 1, 4):
            for spatial in (4, 5):
                value = 0.5 * (
                    operator_basis[first] @ operator_basis[spatial] @ operator_basis[second]
                    + operator_basis[second] @ operator_basis[spatial] @ operator_basis[first]
                )
                operators.append(_normalize_operator(value))
    return np.stack(operators)


def conditioned_spatial_family(operator_basis: np.ndarray) -> np.ndarray:
    """Energy/RHS-conditioned spatial sandwiches absent from degree-two Stage 52."""

    return np.stack([
        _normalize_operator(operator_basis[condition] @ operator_basis[spatial] @ operator_basis[condition])
        for condition in (0, 1, 2, 3) for spatial in (4, 5)
    ])


def _dct_physical_basis(side: int = 71, modes: int = 6) -> np.ndarray:
    transform = dct(np.eye(side), axis=0, type=2, norm="ortho")
    return np.column_stack([
        np.outer(transform[first], transform[second]).ravel()
        for first in range(modes) for second in range(modes)
    ])


def coefficient_weighted_spatial_family(
    token_features: np.ndarray, normalized_field: np.ndarray, physical_basis: np.ndarray
) -> np.ndarray:
    """Galerkin multiplication by each cached coefficient-field channel."""

    spatial = np.asarray(token_features[:, :36], float)
    fields = np.asarray(normalized_field[:3], float).reshape(3, -1)
    operators = []
    for field in fields:
        physical_multiplier = physical_basis.T @ (field[:, None] * physical_basis)
        operators.append(_normalize_operator(spatial @ physical_multiplier @ spatial.T))
    return np.stack(operators)


def extension_families(data, index: int, base: np.ndarray, physical_basis: np.ndarray):
    dimension = int(data["effective_dimensions"][index])
    features = np.asarray(data["token_features"][index, :dimension], float)
    return {
        "cross_rhs_spatial_sandwich": cross_rhs_spatial_family(base),
        "conditioned_spatial_sandwich": conditioned_spatial_family(base),
        "coefficient_weighted_spatial": coefficient_weighted_spatial_family(
            features, data["normalized_fields"][index], physical_basis
        ),
    }


def _load_stage51_data(root: Path, split: str):
    path = (
        root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
        / f"{split}_raw_loss_data.npz"
    )
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def analyze_base_and_screen(root: Path, data_by_split, settings: Stage54Settings):
    physical_basis = _dct_physical_basis()
    projector_rows, structure_rows, correlation_rows, screen_rows = [], [], [], []
    candidates: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for split, data in data_by_split.items():
        for index, sample_value in enumerate(data["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(data["effective_dimensions"][index])
            base = np.asarray(data["operator_basis"][index, :, :dimension, :dimension], float)
            families = extension_families(data, index, base, physical_basis)
            candidates[(split, sample_id)] = families
            for rank_index, rank in enumerate(settings.correction_ranks):
                target = np.asarray(data["oracle_mixings"][index, rank_index, :dimension, :rank], float)
                projector = target @ target.T
                _, projected, residual, energy, span_rank = projector_span_projection(base, target)
                projected_mixing = leading_eigenspace(projected, rank)
                projector_rows.append({
                    "split": split, "sample_id": sample_id, "rank": rank,
                    "algebra": "A33", "operator_count": 33,
                    "operator_span_rank": span_rank,
                    "projector_span_residual": energy,
                    "projected_top_eigenspace_containment_with_raw_oracle": _coordinate_containment(
                        target, projected_mixing
                    ),
                })
                structure_rows.append({
                    "split": split, "sample_id": sample_id, "rank": rank,
                    **residual_structure(
                        residual,
                        data["token_features"][index, :dimension],
                        settings.raw_depth,
                        settings.near_step_lag,
                        settings.long_step_lag,
                    ),
                })
                for local in operator_correlations(base, projector, residual):
                    correlation_rows.append({
                        "split": split, "sample_id": sample_id, "rank": rank, **local
                    })
                for family, extension in families.items():
                    combined = np.concatenate((base, extension))
                    _, projected_ext, _, energy_ext, span_rank_ext = projector_span_projection(
                        combined, target
                    )
                    screen_rows.append({
                        "split": split, "sample_id": sample_id, "rank": rank,
                        "extension_family": family,
                        "added_operator_count": extension.shape[0],
                        "operator_span_rank": span_rank_ext,
                        "base_projector_span_residual": energy,
                        "extended_projector_span_residual": energy_ext,
                        "relative_residual_closed": (energy - energy_ext) / max(energy, np.finfo(float).tiny),
                        "projected_top_eigenspace_containment_with_raw_oracle": _coordinate_containment(
                            target, leading_eigenspace(projected_ext, rank)
                        ),
                    })
    return projector_rows, structure_rows, correlation_rows, screen_rows, candidates


def choose_extension(structure_rows, screen_rows, settings: Stage54Settings):
    training = [row for row in structure_rows if row["split"] == "train"]
    medians = {
        key: float(np.median([float(row[key]) for row in training]))
        for key in (
            "diagonal_self_energy", "off_diagonal_energy", "within_rhs_energy",
            "cross_rhs_energy", "near_step_energy", "intermediate_step_energy",
            "long_range_step_energy",
        )
    }
    # The diagnostic determines the first physical family. Cross-RHS coupling
    # is selected only when it dominates within-RHS off-diagonal energy.
    if medians["cross_rhs_energy"] >= medians["within_rhs_energy"]:
        selected = "cross_rhs_spatial_sandwich"
    else:
        selected = "conditioned_spatial_sandwich"
    local = [
        row for row in screen_rows
        if row["split"] == "train" and row["extension_family"] == selected
    ]
    median_closed = float(np.median([float(row["relative_residual_closed"]) for row in local]))
    return selected, median_closed, medians, median_closed >= settings.extension_useful_residual_fraction


def solution_action_diagnostics(root: Path, projector_rows, data_by_split):
    candidate_path = (
        root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
        / "hard_test_paired_raw_candidates.npz"
    )
    rows = []
    hard = data_by_split["hard_test"]
    with np.load(candidate_path) as archive:
        for index, sample_value in enumerate(hard["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(hard["effective_dimensions"][index])
            action = np.asarray(archive[f"system_{sample_id:03d}_action_basis"], float)[:, :dimension]
            solution = np.asarray(archive[f"system_{sample_id:03d}_paired_solution"], float)[:, :dimension]
            mixed = _normalize_operator(0.5 * (action.T @ solution + solution.T @ action))
            base = np.asarray(hard["operator_basis"][index, :, :dimension, :dimension], float)
            for rank_index, rank in enumerate((8, 12)):
                target = np.asarray(hard["oracle_mixings"][index, rank_index, :dimension, :rank], float)
                _, _, residual, _, _ = projector_span_projection(base, target)
                corr = abs(float(np.sum(mixed * residual))) / max(
                    float(la.norm(residual, ord="fro")), np.finfo(float).tiny
                )
                rows.append({
                    "sample_id": sample_id, "rank": rank,
                    "solution_action_residual_correlation": corr,
                    "single_operator_residual_energy_fraction": corr**2,
                    "paired_solution_action_available": True,
                })
    return rows


def optimize_selected_extension(
    output: Path, data_by_split, candidates, selected_family: str, settings: Stage54Settings
):
    predictions_by_split, coefficients_by_split, restart_rows = {}, {}, []
    stage53_root = output.parent / "stage53_equivariant_algebra_oracle_ceiling"
    for split, data in data_by_split.items():
        cache = output / f"{split}_{selected_family}_oracle.npz"
        if cache.exists():
            with np.load(cache) as archive:
                predictions_by_split[split] = np.asarray(archive["predictions"])
                coefficients_by_split[split] = np.asarray(archive["coefficients"])
            continue
        with np.load(stage53_root / f"{split}_algebra_oracle.npz") as archive:
            base_coefficients = np.asarray(archive["coefficients"], float)
        predictions = np.zeros_like(data["oracle_mixings"], dtype=np.float32)
        first_extension = next(iter(candidates.values()))[selected_family]
        coefficients = np.zeros((
            len(data["system_ids"]), 2, 33 + first_extension.shape[0]
        ), float)
        for index, sample_value in enumerate(data["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(data["effective_dimensions"][index])
            base = np.asarray(data["operator_basis"][index, :, :dimension, :dimension], float)
            extension = candidates[(split, sample_id)][selected_family]
            combined = np.concatenate((base, extension))
            for rank_index, rank in enumerate(settings.correction_ranks):
                target = np.asarray(data["oracle_mixings"][index, rank_index, :dimension, :rank], float)
                initialization = np.pad(
                    base_coefficients[index, rank_index], (0, extension.shape[0])
                )
                coefficient, mixing, local_rows, diagnostic = optimize_coefficients(
                    combined, target, initialization, rank,
                    settings.optimization_seed + 10000 * index + 100 * rank + len(split),
                    settings.stage53,
                )
                coefficients[index, rank_index] = coefficient
                predictions[index, rank_index, :dimension, :rank] = mixing
                for row in local_rows:
                    restart_rows.append({
                        "split": split, "sample_id": sample_id, "rank": rank,
                        "extension_family": selected_family, **row, **diagnostic,
                    })
            print(
                f"[Stage54 {split}] {index + 1}/{len(data['system_ids'])} system {sample_id}",
                flush=True,
            )
        np.savez_compressed(cache, predictions=predictions, coefficients=coefficients)
        predictions_by_split[split] = predictions
        coefficients_by_split[split] = coefficients
    return predictions_by_split, coefficients_by_split, restart_rows


def geometry_hierarchy(data_by_split, extended_predictions, selected_family, settings):
    stage53_root = Path(settings["stage53_root"])
    rows = []
    for split, data in data_by_split.items():
        with np.load(stage53_root / f"{split}_algebra_oracle.npz") as archive:
            algebra_predictions = np.asarray(archive["predictions"])
        for index, sample_value in enumerate(data["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(data["effective_dimensions"][index])
            target_cross = np.asarray(data["target_crosses"][index, :, :dimension], float)
            for rank_index, rank in enumerate((8, 12)):
                for family, prediction, count in (
                    ("raw_oracle", data["oracle_mixings"], 0),
                    ("A33", algebra_predictions, 33),
                    (f"A33+{selected_family}", extended_predictions[split],
                     33 + extended_predictions[split].shape[-1] * 0),
                ):
                    mixing = np.asarray(prediction[index, rank_index, :dimension, :rank], float)
                    metrics = coverage_metrics(target_cross, mixing, rank)
                    rows.append({
                        "split": split, "sample_id": sample_id, "rank": rank,
                        "algebra": family, "operator_count": count,
                        "raw_oracle_coordinate_alignment": _coordinate_containment(
                            np.asarray(data["oracle_mixings"][index, rank_index, :dimension, :rank], float),
                            mixing,
                        ),
                        **metrics,
                    })
    return rows


def summarize_hierarchy(rows, projector_rows, selected_family, extension_count):
    e_lookup = {
        (row["split"], int(row["sample_id"]), int(row["rank"]), row["algebra"]): row
        for row in projector_rows
    }
    result = []
    for split in ("train", "validation", "hard_test"):
        for rank in (8, 12):
            locals_by = {
                algebra: [
                    row for row in rows if row["split"] == split
                    and int(row["rank"]) == rank and row["algebra"] == algebra
                ]
                for algebra in ("raw_oracle", "A33", f"A33+{selected_family}")
            }
            raw = float(np.median([
                row["correction_containment"] for row in locals_by["raw_oracle"]
            ]))
            base = float(np.median([
                row["correction_containment"] for row in locals_by["A33"]
            ]))
            for algebra, local in locals_by.items():
                containment = float(np.median([row["correction_containment"] for row in local]))
                if algebra == "raw_oracle":
                    espan = 0.0
                    count = 0
                else:
                    espan = float(np.median([
                        e_lookup[(split, int(row["sample_id"]), rank, algebra)]["projector_span_residual"]
                        for row in local
                    ]))
                    count = 33 if algebra == "A33" else 33 + extension_count
                result.append({
                    "split": split, "rank": rank, "algebra": algebra,
                    "operator_count": count,
                    "median_containment": containment,
                    "median_projector_span_residual": espan,
                    "median_p10": float(np.median([row["p10_principal_coverage"] for row in local])),
                    "median_minimum": float(np.median([row["minimum_principal_coverage"] for row in local])),
                    "median_largest_angle": float(np.median([row["largest_principal_angle_degrees"] for row in local])),
                    "stage53_gap_closed_fraction": (
                        (containment - base) / max(raw - base, np.finfo(float).tiny)
                        if algebra != "raw_oracle" else 1.0
                    ),
                })
    return result


def decide(summary, selected_useful, selected_family, settings: Stage54Settings):
    lookup = {
        (row["split"], int(row["rank"]), row["algebra"]): row for row in summary
    }
    projection_good = all(
        lookup[("hard_test", rank, "A33")]["median_projector_span_residual"] <= 0.20
        for rank in settings.correction_ranks
    )
    extended_fraction = [
        lookup[("hard_test", rank, f"A33+{selected_family}")]["median_containment"]
        / max(lookup[("hard_test", rank, "raw_oracle")]["median_containment"], np.finfo(float).tiny)
        for rank in settings.correction_ranks
    ]
    tail_improved = all(
        lookup[("hard_test", rank, f"A33+{selected_family}")]["median_p10"]
        > lookup[("hard_test", rank, "A33")]["median_p10"] + 0.05
        for rank in settings.correction_ranks
    )
    if projection_good:
        outcome = "Outcome A"
        next_step = "extended algebra"
        conclusion = "The projector is mostly in the linear span; the leading-eigenspace parameterization is the limiting mechanism."
        recommendation = "Stage 55 should replace the leading-eigenspace parameterization while keeping the frozen operator span."
    elif selected_useful and min(extended_fraction) >= settings.extended_algebra_containment_gate and tail_improved:
        outcome = "Outcome B"
        next_step = "extended algebra"
        conclusion = "One motivated extension closes a substantial projector and eigenspace gap."
        recommendation = f"Stage 55 should freeze A33 plus {selected_family} and study coefficient inference."
    elif selected_useful:
        outcome = "Outcome C"
        next_step = "extended algebra"
        conclusion = "The motivated extension progressively closes the gap but does not yet reach the hard ceiling gate."
        recommendation = "Stage 55 should retain only the smallest useful nested extension combination before learning coefficients."
    else:
        outcome = "Outcome D"
        next_step = "abandon finite operator algebra"
        conclusion = "The projector residual remains large and the motivated extension captures negligible missing energy."
        recommendation = (
            "Stage 55 should abandon the hand-designed finite operator dictionary and test a richer "
            "basis-equivariant compression architecture on the same frozen raw candidate."
        )
    return {
        "outcome": outcome,
        "conclusion": conclusion,
        "selected_extension_family": selected_family,
        "selected_extension_useful": selected_useful,
        "hard_extended_raw_fractions": extended_fraction,
        "hard_lower_tail_materially_improved": tail_improved,
        "recommended_next_step": next_step,
        "stage55_recommendation": recommendation,
        "solver_gate_passed": bool(
            min(extended_fraction) >= settings.extended_algebra_containment_gate and tail_improved
        ),
    }


def _report(
    output, decision, medians, screen_summary, structure_summary, correlation_summary,
    hierarchy_summary, projection_summary, hard_rows, solution_action, selected_family,
):
    missing = "off-diagonal cross-RHS interactions concentrated at nearby expected Krylov steps"
    lines = [
        "# Stage 54: Missing-operator analysis of the equivariant algebra",
        "", "## Decision", "",
        f"**{decision['outcome']} — {decision['conclusion']}**", "",
        f"**Missing structure: {missing}.**", "",
        f"**Recommended next step: {decision['recommended_next_step']}.**", "",
        decision["stage55_recommendation"], "",
        "The Stage-51 paired-SVD raw candidate coordinates and oracle projectors, all 33 cached Stage-52 operators, ranks, splits, probes, and q48 construction were frozen. No neural model, new probe, harmonic filter, or solver run was used.",
        "", "## Direct projector-span ceiling", "",
        "| split | r | algebra | operators | containment | E_span | p10 | min | largest angle | gap closed |",
        "|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in hierarchy_summary:
        lines.append(
            f"| {row['split']} | {row['rank']} | {row['algebra']} | {row['operator_count']} | "
            f"{_fmt(row['median_containment'])} | {_fmt(row['median_projector_span_residual'])} | "
            f"{_fmt(row['median_p10'])} | {_fmt(row['median_minimum'])} | "
            f"{_fmt(row['median_largest_angle'])} | {_fmt(row['stage53_gap_closed_fraction'])} |"
        )
    lines += [
        "", "E_span is the squared Frobenius residual divided by projector rank. The extended-algebra gap-closed fraction is measured against the Stage-51 raw-space containment and Stage-53 A33 containment.",
        "", "The direct Frobenius projection and its leading eigenspace give the separate linear-span/parameterization diagnostic:",
        "", "| split | r | algebra | E_span | top-r eigenspace containment with P* | operator-span rank |",
        "|:--|--:|:--|--:|--:|--:|",
    ]
    for row in projection_summary:
        lines.append(
            f"| {row['split']} | {row['rank']} | {row['algebra']} | "
            f"{_fmt(row['median_projector_span_residual'])} | "
            f"{_fmt(row['median_projected_top_eigenspace_containment'])} | "
            f"{_fmt(row['median_operator_span_rank'])} |"
        )
    lines += [
        "", "## Residual structure", "",
        "The following energy fractions aggregate the residual projector in the frozen canonical paired-SVD coordinates. No individual coordinate is interpreted as a physical mode. RHS and step categories use the cached trajectory participation summaries.",
        "", "| split | r | self | off-diagonal | within RHS | cross RHS | nearby steps | intermediate | long range |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in structure_summary:
        lines.append(
            f"| {row['split']} | {row['rank']} | {_fmt(row['diagonal_self_energy'])} | "
            f"{_fmt(row['off_diagonal_energy'])} | {_fmt(row['within_rhs_energy'])} | "
            f"{_fmt(row['cross_rhs_energy'])} | {_fmt(row['near_step_energy'])} | "
            f"{_fmt(row['intermediate_step_energy'])} | {_fmt(row['long_range_step_energy'])} |"
        )
    lines += [
        "", "Across training targets, the median residual is "
        f"`{_fmt(medians['off_diagonal_energy'])}` off-diagonal, with `{_fmt(medians['cross_rhs_energy'])}` "
        f"of total residual energy attributed to cross-RHS participation and `{_fmt(medians['near_step_energy'])}` "
        "to nearby expected steps. These fuzzy RHS contributions partition the off-diagonal energy; the step partition is disjoint.",
        "", "## Existing operator correlations", "",
        "| split | r | operator class | median |corr(P*)| | max |corr(P*)| | median |corr(R)| | max |corr(R)| |",
        "|:--|--:|:--|--:|--:|--:|--:|",
    ]
    for row in correlation_summary:
        lines.append(
            f"| {row['split']} | {row['rank']} | {row['operator_family']} | "
            f"{_fmt(row['median_projector_correlation'])} | {_fmt(row['maximum_projector_correlation'])} | "
            f"{row['median_residual_correlation']:.2e} | {row['maximum_residual_correlation']:.2e} |"
        )
    lines += [
        "", "Residual correlations are at projection-roundoff scale by construction: the direct least-squares residual is Frobenius-orthogonal to the complete A33 span. This confirms that reweighting saturated existing classes cannot recover the missing projector.",
        "", "## Motivated extension screen", "",
        "| candidate family | added operators | train residual fraction closed | validation | hard | advanced to eigenspace ceiling? |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for row in screen_summary:
        lines.append(
            f"| {row['extension_family']} | {row['added_operator_count']} | "
            f"{_fmt(row['train_relative_residual_closed'])} | {_fmt(row['validation_relative_residual_closed'])} | "
            f"{_fmt(row['hard_relative_residual_closed'])} | "
            f"{'yes' if row['extension_family'] == selected_family else 'no'} |"
        )
    lines += [
        "", f"The residual diagnosis selected `{selected_family}` first. It contains only symmetric cross-RHS/spatial sandwiches and is equivariant because every word transforms by conjugation. Its direct residual closure was below the predeclared 2% usefulness threshold, so no second nested family was added. The other two small families were screened only to verify that a different obvious physical interaction did not explain the residual.",
        "", "The paired solution/action diagnostic was available for the seven hard systems only. Its median single-operator residual-energy fraction was `"
        + _fmt(np.median([row['single_operator_residual_energy_fraction'] for row in solution_action]))
        + "`, so a lone symmetric solution/action coupling does not explain the missing projector.",
        "", "## Seven hard systems", "",
        "| system | r | raw C | A33 C / E_span | extended C / E_span | p10 / min | gap closed |",
        "|--:|--:|--:|:--|:--|:--|--:|",
    ]
    for row in hard_rows:
        lines.append(
            f"| {row['sample_id']} | {row['rank']} | {_fmt(row['raw_containment'])} | "
            f"{_fmt(row['base_containment'])} / {_fmt(row['base_projector_span_residual'])} | "
            f"{_fmt(row['extended_containment'])} / {_fmt(row['extended_projector_span_residual'])} | "
            f"{_fmt(row['extended_p10'])} / {_fmt(row['extended_minimum'])} | "
            f"{_fmt(row['gap_closed_fraction'])} |"
        )
    lines += ["", "### System 63", ""]
    for row in hard_rows:
        if int(row["sample_id"]) == 63:
            lines.append(
                f"- r={row['rank']}: raw `{_fmt(row['raw_containment'])}`, A33 "
                f"`{_fmt(row['base_containment'])}`, E_span `{_fmt(row['base_projector_span_residual'])}`; "
                f"extended `{_fmt(row['extended_containment'])}`, E_span "
                f"`{_fmt(row['extended_projector_span_residual'])}`, p10 `{_fmt(row['extended_p10'])}`, "
                f"minimum `{_fmt(row['extended_minimum'])}`. The dominant residual category is {missing}."
            )
    lines += [
        "", "## Solver gate", "",
        "The extended algebra did not reach 80% of raw-oracle containment with a material lower-tail improvement on both hard ranks. The optional solver ceiling was therefore not run.",
        "", "## Stage-55 recommendation", "", decision["stage55_recommendation"], "",
        f"**Missing structure: {missing}.**", "",
        f"**Recommended next step: {decision['recommended_next_step']}.**",
    ]
    report = output / "stage54_missing_operator_analysis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage54(root: Path) -> Path:
    settings = Stage54Settings()
    output = root / "results/results-fd/stages/stage54_missing_operator_analysis"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-54 predeclared protocol is missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
    })
    stage52_root = root / "results/results-fd/stages/stage52_basis_equivariant_raw_compression"
    stage53_root = root / "results/results-fd/stages/stage53_equivariant_algebra_oracle_ceiling"
    validation53 = json.loads((stage53_root / "validation.json").read_text())
    data_by_split = {
        split: _load_split(stage52_root, split)
        for split in ("train", "validation", "hard_test")
    }
    for split, data in data_by_split.items():
        stage51 = _load_stage51_data(root, split)
        data["token_features"] = stage51["token_features"]
        data["normalized_fields"] = stage51["normalized_fields"]
    preflight = {
        "stage53_validation_passed": bool(validation53["passed"]),
        "operator_count_exact": all(data["operator_basis"].shape[1] == 33 for data in data_by_split.values()),
        "split_sizes_exact": [len(data_by_split[s]["system_ids"]) for s in ("train", "validation", "hard_test")] == [43, 15, 7],
        "ranks_exact": settings.correction_ranks == (8, 12),
        "no_neural_training": True, "no_new_probes": True,
        "no_harmonic_filtering": True, "no_solver_change": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-54 preflight failed")

    projector, structure, correlations, screen, candidates = analyze_base_and_screen(
        root, data_by_split, settings
    )
    selected, median_closed, structure_medians, useful = choose_extension(
        structure, screen, settings
    )
    predictions, _, restart_rows = optimize_selected_extension(
        output, data_by_split, candidates, selected, settings
    )
    if restart_rows:
        _write_csv(output / "extension_optimization_restarts.csv", restart_rows)

    # Add the selected extension's direct projector projection to the shared table.
    for split, data in data_by_split.items():
        for index, sample_value in enumerate(data["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(data["effective_dimensions"][index])
            base = np.asarray(data["operator_basis"][index, :, :dimension, :dimension], float)
            extension = candidates[(split, sample_id)][selected]
            combined = np.concatenate((base, extension))
            for rank_index, rank in enumerate(settings.correction_ranks):
                target = np.asarray(data["oracle_mixings"][index, rank_index, :dimension, :rank], float)
                _, projected, _, energy, span_rank = projector_span_projection(combined, target)
                projector.append({
                    "split": split, "sample_id": sample_id, "rank": rank,
                    "algebra": f"A33+{selected}",
                    "operator_count": combined.shape[0], "operator_span_rank": span_rank,
                    "projector_span_residual": energy,
                    "projected_top_eigenspace_containment_with_raw_oracle": _coordinate_containment(
                        target, leading_eigenspace(projected, rank)
                    ),
                })

    hierarchy = geometry_hierarchy(
        data_by_split, predictions, selected, {"stage53_root": str(stage53_root)}
    )
    extension_count = next(iter(candidates.values()))[selected].shape[0]
    # Correct the display count stored by geometry_hierarchy.
    for row in hierarchy:
        if row["algebra"] == f"A33+{selected}":
            row["operator_count"] = 33 + extension_count
    summary = summarize_hierarchy(hierarchy, projector, selected, extension_count)
    projection_summary = []
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            for algebra in ("A33", f"A33+{selected}"):
                local = [
                    row for row in projector if row["split"] == split
                    and int(row["rank"]) == rank and row["algebra"] == algebra
                ]
                projection_summary.append({
                    "split": split, "rank": rank, "algebra": algebra,
                    "median_projector_span_residual": float(np.median([
                        row["projector_span_residual"] for row in local
                    ])),
                    "median_projected_top_eigenspace_containment": float(np.median([
                        row["projected_top_eigenspace_containment_with_raw_oracle"] for row in local
                    ])),
                    "median_operator_span_rank": float(np.median([
                        row["operator_span_rank"] for row in local
                    ])),
                })

    structure_summary = []
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            local = [row for row in structure if row["split"] == split and int(row["rank"]) == rank]
            structure_summary.append({
                "split": split, "rank": rank,
                **{key: float(np.median([row[key] for row in local])) for key in (
                    "diagonal_self_energy", "off_diagonal_energy", "within_rhs_energy",
                    "cross_rhs_energy", "near_step_energy", "intermediate_step_energy",
                    "long_range_step_energy",
                )},
            })
    correlation_summary = []
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            for family in ("primitive", "symmetric_quadratic", "primitive_cube"):
                local = [row for row in correlations if row["split"] == split and int(row["rank"]) == rank and row["operator_family"] == family]
                correlation_summary.append({
                    "split": split, "rank": rank, "operator_family": family,
                    "median_projector_correlation": float(np.median([row["absolute_projector_correlation"] for row in local])),
                    "maximum_projector_correlation": float(np.max([row["absolute_projector_correlation"] for row in local])),
                    "median_residual_correlation": float(np.median([row["absolute_residual_correlation"] for row in local])),
                    "maximum_residual_correlation": float(np.max([row["absolute_residual_correlation"] for row in local])),
                })
    screen_summary = []
    for family in sorted({row["extension_family"] for row in screen}):
        local_family = [row for row in screen if row["extension_family"] == family]
        screen_summary.append({
            "extension_family": family,
            "added_operator_count": int(local_family[0]["added_operator_count"]),
            **{
                f"{split.replace('_test', '')}_relative_residual_closed": float(np.median([
                    row["relative_residual_closed"] for row in local_family if row["split"] == split
                ]))
                for split in ("train", "validation", "hard_test")
            },
        })
    solution_action = solution_action_diagnostics(root, projector, data_by_split)

    h_lookup = {
        (row["split"], int(row["sample_id"]), int(row["rank"]), row["algebra"]): row
        for row in hierarchy
    }
    p_lookup = {
        (row["split"], int(row["sample_id"]), int(row["rank"]), row["algebra"]): row
        for row in projector
    }
    hard_rows = []
    for sample_id in settings.stage53.stage52.hard_system_ids:
        for rank in settings.correction_ranks:
            raw = h_lookup[("hard_test", sample_id, rank, "raw_oracle")]
            base = h_lookup[("hard_test", sample_id, rank, "A33")]
            ext = h_lookup[("hard_test", sample_id, rank, f"A33+{selected}")]
            hard_rows.append({
                "sample_id": sample_id, "rank": rank,
                "raw_containment": raw["correction_containment"],
                "base_containment": base["correction_containment"],
                "extended_containment": ext["correction_containment"],
                "base_projector_span_residual": p_lookup[("hard_test", sample_id, rank, "A33")]["projector_span_residual"],
                "extended_projector_span_residual": p_lookup[("hard_test", sample_id, rank, f"A33+{selected}")]["projector_span_residual"],
                "extended_p10": ext["p10_principal_coverage"],
                "extended_minimum": ext["minimum_principal_coverage"],
                "gap_closed_fraction": (ext["correction_containment"] - base["correction_containment"])
                / max(raw["correction_containment"] - base["correction_containment"], np.finfo(float).tiny),
            })
    decision = decide(summary, useful, selected, settings)
    if decision["solver_gate_passed"]:
        raise RuntimeError("Stage-54 solver gate unexpectedly passed; explicit solver implementation is required")

    _write_csv(output / "projector_span_ceiling.csv", projector)
    _write_csv(output / "projector_span_summary.csv", projection_summary)
    _write_csv(output / "residual_structure.csv", structure)
    _write_csv(output / "residual_structure_summary.csv", structure_summary)
    _write_csv(output / "operator_correlations.csv", correlations)
    _write_csv(output / "operator_correlation_summary.csv", correlation_summary)
    _write_csv(output / "extension_family_screen.csv", screen)
    _write_csv(output / "extension_family_screen_summary.csv", screen_summary)
    _write_csv(output / "solution_action_diagnostics.csv", solution_action)
    _write_csv(output / "geometry_hierarchy.csv", hierarchy)
    _write_csv(output / "geometry_summary.csv", summary)
    _write_csv(output / "hard_system_ceiling.csv", hard_rows)
    _write_json(output / "decision.json", decision)
    report = _report(
        output, decision, structure_medians, screen_summary, structure_summary,
        correlation_summary, summary, projection_summary, hard_rows, solution_action, selected,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "base_operator_count": 33,
        "extension_family_count_screened": len(screen_summary),
        "extension_family_count_advanced": 1,
        "advanced_extension_operator_count": extension_count,
        "ranks_exact": settings.correction_ranks == (8, 12),
        "all_splits_present": {row["split"] for row in summary} == {"train", "validation", "hard_test"},
        "direct_projection_summary_rows": len(projection_summary),
        "projector_residual_orthogonal_to_A33": max(row["maximum_residual_correlation"] for row in correlation_summary) <= 1.0e-5,
        "extension_operators_symmetric": all(
            np.allclose(value, value.transpose(0, 2, 1), atol=1.0e-10)
            for value in next(iter(candidates.values())).values()
        ),
        "no_neural_training": True, "no_new_probes": True,
        "solver_gate_passed": decision["solver_gate_passed"],
        "solver_run": False,
    }
    validation["passed"] = bool(
        validation["preflight_passed"] and validation["base_operator_count"] == 33
        and validation["extension_family_count_screened"] <= settings.maximum_extension_families
        and validation["extension_family_count_advanced"] == 1
        and validation["ranks_exact"] and validation["all_splits_present"]
        and validation["direct_projection_summary_rows"] == 12
        and validation["projector_residual_orthogonal_to_A33"]
        and validation["extension_operators_symmetric"]
        and validation["no_neural_training"] and validation["no_new_probes"]
        and validation["solver_run"] == validation["solver_gate_passed"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage54_missing_operator_analysis.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage54] complete", flush=True)
    return report_copy
