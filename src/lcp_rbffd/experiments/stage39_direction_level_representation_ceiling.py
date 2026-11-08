"""Stage 39: direction/local-block representation ceiling for FD recycling."""

from __future__ import annotations

import json
import platform
from collections import Counter
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
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage39Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    block_width: int = 3
    numerical_rank_tolerance: float = 1.0e-13
    expected_training_systems: int = 43
    expected_full_span_rank: int = 1456

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


@dataclass(frozen=True)
class PairedBlock:
    block_id: int
    donor_id: int
    start: int
    stop: int
    z_basis: np.ndarray
    w_basis: np.ndarray
    pair_defect: float

    @property
    def rank(self) -> int:
        return int(self.z_basis.shape[1])


def build_block_library(root: Path, settings: Stage39Settings):
    """Partition every cached training teacher and preserve exact own-A pairing."""

    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    donor_ids = [int(value) for value in split["train"]]
    blocks: List[PairedBlock] = []
    rows: List[Dict[str, Any]] = []
    source_ranks: Dict[int, int] = {}
    block_id = 0
    for donor_id in donor_ids:
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{donor_id:03d}.npz").tocsr()
        with np.load(stage28 / "systems" / f"system_{donor_id:03d}.npz") as archive:
            teacher_z = np.asarray(archive["zfull"], dtype=float)
        source_ranks[donor_id] = teacher_z.shape[1]
        for start in range(0, teacher_z.shape[1], settings.block_width):
            stop = min(start + settings.block_width, teacher_z.shape[1])
            pair = _ordered_pair_from_z(
                matrix, teacher_z[:, start:stop], settings.numerical_rank_tolerance
            )
            block = PairedBlock(
                block_id=block_id,
                donor_id=donor_id,
                start=start,
                stop=stop,
                z_basis=pair.u_basis,
                w_basis=pair.w_basis,
                pair_defect=float(pair.diagnostics["au_equals_w_relative_error"]),
            )
            blocks.append(block)
            rows.append({
                "block_id": block_id,
                "donor_id": donor_id,
                "source_start": start,
                "source_stop": stop,
                "rank": block.rank,
                "own_system_pair_defect": block.pair_defect,
                "ordering": "cached Stage28 zfull column order",
            })
            block_id += 1
    return donor_ids, source_ranks, blocks, rows


def _best_addition(
    block_pair: RecyclePair,
    selected_pair: RecyclePair | None,
    teacher_w: np.ndarray,
    width: int,
    tolerance: float,
):
    """Return the within-block directions giving maximal new teacher coverage."""

    w_block = block_pair.w_basis
    if selected_pair is None:
        coupling = np.empty((w_block.shape[1], 0), dtype=float)
        selected_teacher = np.empty((0, teacher_w.shape[1]), dtype=float)
    else:
        coupling = w_block.T @ selected_pair.w_basis
        selected_teacher = selected_pair.w_basis.T @ teacher_w
    gram = np.eye(w_block.shape[1]) - coupling @ coupling.T
    gram = 0.5 * (gram + gram.T)
    values, vectors = la.eigh(gram, check_finite=False)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    if values.size == 0 or values[0] <= tolerance:
        return None
    rank = int(np.count_nonzero(values > tolerance * max(values[0], 1.0)))
    if rank == 0:
        return None
    vectors = vectors[:, :rank]
    values = values[:rank]
    cross = w_block.T @ teacher_w - coupling @ selected_teacher
    overlap = (vectors.T @ cross) / np.sqrt(values)[:, None]
    left, singular, _ = la.svd(overlap, full_matrices=False, check_finite=False)
    actual = min(width, rank, left.shape[1])
    if actual == 0:
        return None
    coefficients = (vectors / np.sqrt(values)[None, :]) @ left[:, :actual]
    z_add = block_pair.u_basis @ coefficients
    return z_add, float(np.sum(singular[:actual] ** 2)), actual, singular[:actual]


def greedy_block_oracle(
    matrix: sp.spmatrix,
    blocks: Sequence[PairedBlock],
    teacher: RecyclePair,
    candidate_direction_budget: int,
    final_rank: int,
    tolerance: float,
):
    """Greedy held-out-teacher oracle with a whole-donor-matched raw budget."""

    started = perf_counter()
    target_pairs = {block.block_id: _pair_from_z(matrix, block.z_basis) for block in blocks}
    offsets: Dict[int, np.ndarray] = {}
    w_parts = []
    offset = 0
    for block in blocks:
        pair = target_pairs[block.block_id]
        offsets[block.block_id] = np.arange(offset, offset + pair.rank)
        w_parts.append(pair.w_basis)
        offset += pair.rank
    w_all = np.column_stack(w_parts)
    gram = w_all.T @ w_all
    gram = 0.5 * (gram + gram.T)
    teacher_cross = w_all.T @ teacher.w_basis
    remaining = {block.block_id: block for block in blocks}
    selected_coordinates = np.empty((w_all.shape[1], 0), dtype=float)
    selected_raw: np.ndarray | None = None
    trace: List[Dict[str, Any]] = []
    direction_counts: Counter[int] = Counter()
    selected_blocks: List[int] = []
    round_index = 0
    selected_rank = 0
    while selected_rank < candidate_direction_budget:
        round_index += 1
        rank_before = selected_rank
        width = min(3, candidate_direction_budget - rank_before)
        selected_teacher = selected_coordinates.T @ teacher_cross
        choices = []
        for block_id, block in remaining.items():
            indices = offsets[block_id]
            coupling = gram[indices, :] @ selected_coordinates
            local_gram = gram[np.ix_(indices, indices)] - coupling @ coupling.T
            local_gram = 0.5 * (local_gram + local_gram.T)
            values, vectors = la.eigh(local_gram, check_finite=False)
            order = np.argsort(values)[::-1]
            values = np.maximum(values[order], 0.0)
            vectors = vectors[:, order]
            if values.size == 0 or values[0] <= tolerance:
                continue
            local_rank = int(np.count_nonzero(values > tolerance * max(values[0], 1.0)))
            if local_rank == 0:
                continue
            vectors = vectors[:, :local_rank]
            values = values[:local_rank]
            cross = teacher_cross[indices, :] - coupling @ selected_teacher
            overlap = (vectors.T @ cross) / np.sqrt(values)[:, None]
            left, singular, _ = la.svd(overlap, full_matrices=False, check_finite=False)
            actual = min(width, local_rank, left.shape[1])
            local_coefficients = (vectors / np.sqrt(values)[None, :]) @ left[:, :actual]
            global_coefficients = np.zeros((w_all.shape[1], actual), dtype=float)
            global_coefficients[indices, :] = local_coefficients
            residual_coordinates = global_coefficients - selected_coordinates @ (
                coupling.T @ local_coefficients
            )
            z_add = target_pairs[block_id].u_basis @ local_coefficients
            gain = float(np.sum(singular[:actual] ** 2))
            choices.append((gain, -block_id, block, z_add, actual, singular, residual_coordinates))
        if not choices:
            raise RuntimeError("block oracle exhausted the library before reaching target rank")
        gain, _, block, z_add, actual, singular, normalized_coordinates = max(choices, key=lambda value: (value[0], value[1]))
        for _ in range(2):
            normalized_coordinates -= selected_coordinates @ (
                selected_coordinates.T @ gram @ normalized_coordinates
            )
        small = normalized_coordinates.T @ gram @ normalized_coordinates
        small = 0.5 * (small + small.T)
        small_values, small_vectors = la.eigh(small, check_finite=False)
        if np.min(small_values) <= tolerance:
            raise RuntimeError("selected block lost action rank")
        normalized_coordinates = normalized_coordinates @ (
            small_vectors @ np.diag(1.0 / np.sqrt(small_values)) @ small_vectors.T
        )
        combined = z_add if selected_raw is None else np.column_stack((selected_raw, z_add))
        selected_raw = combined
        selected_coordinates = np.column_stack((selected_coordinates, normalized_coordinates))
        selected_rank += actual
        remaining.pop(block.block_id)
        selected_blocks.append(block.block_id)
        direction_counts[block.donor_id] += actual
        coverage = float(la.norm(selected_coordinates.T @ teacher_cross, ord="fro") ** 2 / teacher.rank)
        trace.append({
            "round": round_index,
            "block_id": block.block_id,
            "donor_id": block.donor_id,
            "source_start": block.start,
            "source_stop": block.stop,
            "block_rank": block.rank,
            "selected_directions": actual,
            "rank_before": rank_before,
            "rank_after": selected_rank,
            "incremental_teacher_coverage": gain / teacher.rank,
            "cumulative_teacher_coverage": coverage,
            "minimum_selected_local_cosine": float(singular[actual - 1]),
        })
    if selected_raw is None or selected_rank != candidate_direction_budget:
        raise RuntimeError("block oracle changed the matched candidate budget")
    selected_pair = _ordered_pair_from_z(matrix, selected_raw, tolerance)
    final_pair, reduction = oracle_select_inside_dictionary(
        matrix, selected_pair.z_basis, teacher.w_basis
    )
    if final_pair.rank != final_rank:
        raise RuntimeError("block oracle changed the requested final recycle rank")
    distinct = len(direction_counts)
    dominant = max(direction_counts.values()) / candidate_direction_budget
    if dominant >= 0.5:
        category = "many directions from one donor"
    elif distinct >= 9:
        category = "widely distributed across many training systems"
    else:
        category = "a few directions from several donors"
    diagnostics = {
        "candidate_direction_budget": candidate_direction_budget,
        "candidate_union_rank": selected_pair.rank,
        "requested_rank": final_rank,
        "actual_rank": final_pair.rank,
        "selected_blocks": len(selected_blocks),
        "selected_directions": int(sum(direction_counts.values())),
        "distinct_source_systems": distinct,
        "dominant_source_fraction": float(dominant),
        "assembly_category": category,
        "directions_by_source": {str(key): value for key, value in sorted(direction_counts.items())},
        "selected_block_ids": selected_blocks,
        "selection_seconds": perf_counter() - started,
        "candidate_pair_defect": selected_pair.diagnostics["au_equals_w_relative_error"],
        "pair_defect": final_pair.diagnostics["au_equals_w_relative_error"],
        "final_mean_squared_principal_cosine": reduction["mean_squared_principal_cosine"],
    }
    return final_pair, trace, diagnostics


def _teacher(root: Path, sample_id: int, matrix: sp.spmatrix) -> RecyclePair:
    path = root / "results/results-fd/stage28_operator_learning/systems" / f"system_{sample_id:03d}.npz"
    with np.load(path) as archive:
        return _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))


def _geometry(pair: RecyclePair, teacher: RecyclePair):
    metrics = generalized_subspace_metrics(pair.w_basis, teacher.w_basis)
    angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
    return {
        "d_W": float(metrics["projector_distance_normalized"]),
        "median_principal_angle_degrees": float(np.median(angles)),
        "largest_principal_angle_degrees": float(np.max(angles)),
        "teacher_containment": float(metrics["containment_second"]),
    }


def construct_oracles(root: Path, output: Path, blocks, candidate_budgets, settings: Stage39Settings):
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[int, RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    basis_archive: Dict[str, np.ndarray] = {}
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        teacher = _teacher(root, sample_id, context.system.matrix)
        pair, trace, diagnostics = greedy_block_oracle(
            context.system.matrix, blocks, teacher, candidate_budgets[sample_id],
            teacher.rank,
            settings.numerical_rank_tolerance,
        )
        pairs[sample_id] = pair
        metadata[sample_id] = diagnostics
        basis_archive[f"system_{sample_id:03d}_z"] = pair.z_basis
        for row in trace:
            trace_rows.append({"sample_id": sample_id, **row})
        geometry_rows.append({
            "sample_id": sample_id,
            "method": "block_level_oracle",
            "rank": pair.rank,
            **_geometry(pair, teacher),
            "selected_blocks": diagnostics["selected_blocks"],
            "selected_directions": diagnostics["selected_directions"],
            "distinct_source_systems": diagnostics["distinct_source_systems"],
            "dominant_source_fraction": diagnostics["dominant_source_fraction"],
            "assembly_category": diagnostics["assembly_category"],
            "pair_defect": diagnostics["pair_defect"],
        })
        print(f"[Stage39 oracle] {position}/7 system {sample_id}", flush=True)
    np.savez_compressed(output / "selected_block_oracle_spaces.npz", **basis_archive)
    _write_csv(output / "block_selection_trace.csv", trace_rows)
    _write_json(output / "block_selection_metadata.json", metadata)
    return pairs, geometry_rows, metadata


def whole_donor_candidate_budgets(
    root: Path,
    source_ranks: Mapping[int, int],
    settings: Stage39Settings,
):
    """Match each target's Stage-36 five-donor oracle raw-column budget."""

    rows = _read_csv(
        root
        / "results/results-fd/stages/stage36_neural_donor_mixture/selection_diagnostics.csv"
    )
    budgets: Dict[int, int] = {}
    donors_by_target: Dict[int, List[int]] = {}
    for sample_id in settings.hard_system_ids:
        row = next(
            item
            for item in rows
            if item["model"] == "mlp_donor"
            and item["split"] == "hard_test"
            and int(item["target_id"]) == sample_id
            and int(item["k"]) == 5
        )
        selected = [int(value) for value in row["greedy_oracle_donors"].split(";")]
        donors_by_target[sample_id] = selected
        budgets[sample_id] = int(sum(source_ranks[donor] for donor in selected))
    return budgets, donors_by_target


METHODS = (
    "teacher_full",
    "full_span_oracle",
    "stage37_selected_union_teacher_oracle",
    "whole_donor_oracle",
    "block_level_oracle",
)


def _archive_row(row: Mapping[str, Any], method: str):
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "rho": float(row["rho"]),
        "failure": row["failure"],
    }


def evaluate_solver(root: Path, pairs: Mapping[int, RecyclePair], settings: Stage39Settings):
    stage37_rows = _read_csv(root / "results/results-fd/stages/stage37_krylov_aware_local_reduction/solver_results.csv")
    rename = {
        "selected_union_teacher_oracle": "stage37_selected_union_teacher_oracle",
        "donor_oracle": "whole_donor_oracle",
    }
    keep = {"teacher_full", "full_span_oracle", "selected_union_teacher_oracle", "donor_oracle"}
    rows = [_archive_row(row, rename.get(row["method"], row["method"])) for row in stage37_rows if row["method"] in keep]
    lookup = {(int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in stage37_rows}
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        pair = pairs[sample_id]
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
            residual = _relative_residual(context.system.matrix, result.x, rhs)
            rows.append({
                "sample_id": sample_id,
                "rhs_seed": rhs_seed,
                "method": "block_level_oracle",
                "source": "Stage39 diagnostic oracle",
                "iterations": int(result.iterations),
                "converged": bool(result.converged and residual <= 1.0e-8),
                "true_final_residual": residual,
                "actual_rank": pair.rank,
                "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator if denominator > 0 else float("nan")),
                "failure": getattr(result, "failure", "") or "",
            })
    return rows


def comparison_geometry(root: Path, new_rows: Sequence[Mapping[str, Any]], settings: Stage39Settings):
    stage30 = _read_csv(root / "results/results-fd/stages/stage30_full_span_representation_ceiling/geometry_diagnostics.csv")
    stage37 = _read_csv(root / "results/results-fd/stages/stage37_krylov_aware_local_reduction/geometry_diagnostics.csv")
    stage36_solver = _read_csv(root / "results/results-fd/stages/stage36_neural_donor_mixture/solver_results.csv")
    rows = []
    for sample_id in settings.hard_system_ids:
        rows.append({"sample_id": sample_id, "method": "teacher_full", "d_W": 0.0, "largest_principal_angle_degrees": 0.0})
        full = next(row for row in stage30 if int(row["sample_id"]) == sample_id and row["method"] == "oracle_within_full_span")
        rows.append({"sample_id": sample_id, "method": "full_span_oracle", "d_W": float(full["d_W_to_teacher"]), "largest_principal_angle_degrees": float(full["maximum_angle_W_degrees"])})
        selected = next(row for row in stage37 if int(row["sample_id"]) == sample_id and row["method"] == "teacher_reduction")
        rows.append({"sample_id": sample_id, "method": "stage37_selected_union_teacher_oracle", "d_W": float(selected["d_W"]), "largest_principal_angle_degrees": float(selected["largest_principal_angle_degrees"])})
        whole = next(row for row in stage36_solver if int(row["sample_id"]) == sample_id and row["method"] == "donor_oracle")
        rows.append({"sample_id": sample_id, "method": "whole_donor_oracle", "d_W": float(whole["final_dW"]), "largest_principal_angle_degrees": float(whole["final_largest_angle_degrees"])})
        new = next(row for row in new_rows if int(row["sample_id"]) == sample_id)
        rows.append({"sample_id": sample_id, "method": "block_level_oracle", "d_W": float(new["d_W"]), "largest_principal_angle_degrees": float(new["largest_principal_angle_degrees"])})
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


def summarize_geometry(rows: Sequence[Mapping[str, Any]]):
    result = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_largest_principal_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in local])),
            "worst_largest_principal_angle": float(np.max([float(row["largest_principal_angle_degrees"]) for row in local])),
        })
    return result


def decide(solver_summary, geometry_summary):
    solver = {row["method"]: row for row in solver_summary}
    geometry = {row["method"]: row for row in geometry_summary}
    block = solver["block_level_oracle"]
    full = solver["full_span_oracle"]
    approaches = bool(
        full["converged_rhs"] - block["converged_rhs"] <= 3
        and full["median_rho"] - block["median_rho"] <= 0.15
        and geometry["block_level_oracle"]["median_dW"] - geometry["full_span_oracle"]["median_dW"] <= 0.15
    )
    if approaches:
        conclusion = "Fine-grained cross-donor block selection approaches the Stage-30 full-span ceiling; whole-donor granularity was the main representation bottleneck."
        stage40 = "Train one compact MLP/FNO comparison to predict rank-3 block relevance, using the Stage-39 oracle labels and the unchanged target-side pairing and solver."
    else:
        conclusion = "Fine-grained block selection remains materially below the Stage-30 full-span ceiling; fixed training-teacher pieces are still too restrictive."
        stage40 = "Do not train another selector; construct the candidate basis locally from target operator and probe information instead of selecting fixed training blocks."
    return {"approaches_full_span": approaches, "conclusion": conclusion, "stage40_direction": stage40}


def write_report(output: Path, solver_summary, geometry_summary, geometry_rows, metadata, decision):
    labels = {
        "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage37_selected_union_teacher_oracle": "Stage-37 selected-union teacher oracle",
        "whole_donor_oracle": "Stage-36 whole-donor oracle",
        "block_level_oracle": "Stage-39 direction/block oracle",
    }
    lines = [
        "# Stage 39: Direction-level / local-block representation ceiling", "",
        "## Result", "", f"**{decision['conclusion']}**", "",
        "| method | converged | median iterations | median $\\rho$ | median rank | median $d_W$ | median largest angle | worst largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    solver = {row["method"]: row for row in solver_summary}
    geometry = {row["method"]: row for row in geometry_summary}
    for method in METHODS:
        s, g = solver[method], geometry[method]
        lines.append(f"| {labels[method]} | {s['converged_rhs']}/21 | {_fmt(s['median_iterations'],1)} | {_fmt(s['median_rho'])} | {_fmt(s['median_rank'],1)} | {_fmt(g['median_dW'])} | {_fmt(g['median_largest_principal_angle'],1)} degrees | {_fmt(g['worst_largest_principal_angle'],1)} degrees |")
    lines.extend(["", "## Block assembly by target", "", "| target | final rank | candidate directions | selected blocks | distinct source systems | dominant source share | assembly pattern | directions by source |", "|--:|--:|--:|--:|--:|--:|:--|:--|"])
    for sample_id in sorted(metadata):
        row = metadata[sample_id]
        sources = ", ".join(f"{source}:{count}" for source, count in row["directions_by_source"].items())
        lines.append(f"| {sample_id} | {row['actual_rank']} | {row['selected_directions']} | {row['selected_blocks']} | {row['distinct_source_systems']} | {_fmt(row['dominant_source_fraction'])} | {row['assembly_category']} | {sources} |")
    categories = Counter(row["assembly_category"] for row in metadata.values())
    dominant_category, dominant_count = categories.most_common(1)[0]
    lines.extend(["", f"The typical pattern is **{dominant_category}** ({dominant_count}/7 targets). Blocks have rank at most 3; the candidate-direction budget exactly matches the corresponding Stage-36 five-donor oracle before the final teacher-informed rank-q reduction.", "", "## System 63", ""])
    for method in METHODS:
        local_solver = [row for row in _read_csv(output / "solver_results.csv") if row["method"] == method and int(row["sample_id"]) == 63]
        geom = next(row for row in geometry_rows if row["method"] == method and int(row["sample_id"]) == 63)
        lines.append(f"- {labels[method]}: {sum(_bool(row['converged']) for row in local_solver)}/3, median {int(np.median([int(row['iterations']) for row in local_solver]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in local_solver]))}$, $d_W={_fmt(geom['d_W'])}$, largest angle {_fmt(geom['largest_principal_angle_degrees'],1)} degrees.")
    system63 = metadata[63]
    lines.append(f"System 63 uses {system63['selected_blocks']} blocks and {system63['distinct_source_systems']} distinct contributing training systems; its dominant source supplies {_fmt(system63['dominant_source_fraction'])} of the {system63['selected_directions']} candidate directions before rank-{system63['actual_rank']} reduction.")
    lines.extend(["", "## Construction and scope", "", "The library partitions all 1,456 cached columns from the 43 Stage-28 training teachers. Each block rebuilds and verifies its own-system paired action relation before transfer. Held-out teachers are used only by the named Stage-39 representation oracle and subsequent geometry; no neural model or deployable selector is trained.", "", "## Stage 40 recommendation", "", decision["stage40_direction"]])
    report = output / "stage39_direction_level_representation_ceiling.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage39(root: Path) -> Path:
    settings = Stage39Settings()
    output = root / "results/results-fd/stages/stage39_direction_level_representation_ceiling"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-39 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()})
    donor_ids, source_ranks, blocks, block_rows = build_block_library(root, settings)
    candidate_budgets, whole_donors = whole_donor_candidate_budgets(
        root, source_ranks, settings
    )
    stage30 = np.load(root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz")
    stage30_ids = np.asarray(stage30["training_ids"], dtype=int).tolist()
    stage30_ranks = np.asarray(stage30["source_ranks"], dtype=int).tolist()
    preflight = {
        "training_system_count": len(donor_ids),
        "training_ids_match_stage30": donor_ids == stage30_ids,
        "source_ranks_match_stage30": [source_ranks[item] for item in donor_ids] == stage30_ranks,
        "block_count": len(blocks),
        "block_widths_valid": all(1 <= block.rank <= settings.block_width for block in blocks),
        "block_columns": sum(block.rank for block in blocks),
        "full_span_columns_preserved": sum(block.rank for block in blocks) == settings.expected_full_span_rank,
        "matched_candidate_budgets": {str(key): value for key, value in candidate_budgets.items()},
        "stage36_whole_donors": {str(key): value for key, value in whole_donors.items()},
        "candidate_budgets_valid": all(settings.block_width < value < settings.expected_full_span_rank for value in candidate_budgets.values()),
        "maximum_own_system_pair_defect": max(block.pair_defect for block in blocks),
        "heldout_disjoint_from_training": not bool(set(settings.hard_system_ids) & set(donor_ids)),
        "teacher_use_is_diagnostic_only": True,
        "solver_matches_stage30": settings.stage29.stage28.solver == settings.stage29.stage28.stage22a.solver,
    }
    preflight["passed"] = bool(
        len(donor_ids) == settings.expected_training_systems
        and preflight["training_ids_match_stage30"]
        and preflight["source_ranks_match_stage30"]
        and preflight["block_widths_valid"]
        and preflight["full_span_columns_preserved"]
        and preflight["candidate_budgets_valid"]
        and preflight["maximum_own_system_pair_defect"] <= 1.0e-9
        and preflight["heldout_disjoint_from_training"]
        and preflight["teacher_use_is_diagnostic_only"]
        and preflight["solver_matches_stage30"]
    )
    _write_csv(output / "block_library.csv", block_rows)
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-39 preflight failed")
    pairs, new_geometry, metadata = construct_oracles(
        root, output, blocks, candidate_budgets, settings
    )
    solver_rows = evaluate_solver(root, pairs, settings)
    solver_summary = summarize_solver(solver_rows)
    all_geometry = comparison_geometry(root, new_geometry, settings)
    geometry_summary = summarize_geometry(all_geometry)
    decision = decide(solver_summary, geometry_summary)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}
    _write_csv(output / "geometry_diagnostics.csv", all_geometry)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(output, solver_summary, geometry_summary, all_geometry, metadata, decision)
    validation = {
        "preflight_passed": preflight["passed"],
        "all_new_methods_are_teacher_diagnostic_oracles": True,
        "block_pairing_verified": preflight["maximum_own_system_pair_defect"] <= 1.0e-9,
        "selected_ranks_match_teacher": all(pairs[sample].rank == _teacher(root, sample, sp.load_npz(root / "results/results-fd/stage28_operator_learning/matrices" / f"A_{sample:03d}.npz")).rank for sample in settings.hard_system_ids),
        "selected_direction_counts_match_budget": all(metadata[sample]["selected_directions"] == candidate_budgets[sample] for sample in settings.hard_system_ids),
        "maximum_candidate_union_pair_defect": max(float(metadata[sample]["candidate_pair_defect"]) for sample in settings.hard_system_ids),
        "maximum_target_pair_defect": max(float(row["pair_defect"]) for row in new_geometry),
        "new_solver_rows": sum(row["source"] == "Stage39 diagnostic oracle" for row in solver_rows),
        "expected_new_solver_rows": 21,
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])),
    }
    validation["passed"] = bool(
        all(validation[key] for key in ("preflight_passed", "all_new_methods_are_teacher_diagnostic_oracles", "block_pairing_verified", "selected_ranks_match_teacher", "selected_direction_counts_match_budget"))
        and validation["maximum_candidate_union_pair_defect"] <= 1.0e-9
        and validation["maximum_target_pair_defect"] <= 1.0e-9
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage39_direction_level_representation_ceiling.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage39] complete", flush=True)
    return copy
