"""Stage 22: frozen cross-system transfer gate for spatially varying K(x).

This experiment reuses the Stage-22A hard-system population and freezes its
validated own-system construction: three independent construction right-hand
sides, 128 right-preconditioned Arnoldi steps per right-hand side, pair-
preserving ordinary Ritz extraction, Z-union compression, and requested rank
48.  No model is trained and no transfer-dependent tuning is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from ..pde import SpatialAnisotropicDiffusion
from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import Stage19ASettings, _discrete_rhs
from .stage21_population_validation import (
    _archived_median,
    _assemble_context,
    _bool,
    _load_population,
    classify_difficulty,
)
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    Stage22ASettings,
    _baseline_map,
    _build_ordinary_construction,
    _load_stage21_baselines,
    _raw_dictionary,
    _run_pair,
    _z_union,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PURPLE = "#7553A6"
CHARCOAL = "#30343B"
GRAY = "#8A9099"
LIGHT_GRAY = "#D7DBE2"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
    }
)


@dataclass(frozen=True)
class Stage22TransferSettings:
    """Frozen construction, field descriptors, controls, and transfer gate."""

    construction_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    arnoldi_dimension: int = 128
    candidates_per_rhs: int = 24
    requested_q: int = 48
    numerical_rank_tolerance: float = 1.0e-11
    pair_rank_tolerance: float = 1.0e-12
    field_grid_size: int = 81
    random_seed: int = 20262270
    nearest_all_rhs_convergence_fraction_gate: float = 0.70
    median_eta_gate: float = 0.50
    meaningful_iteration_ratio_gate: float = 0.75
    random_win_fraction_gate: float = 0.70
    high_recovery_eta: float = 0.75

    @property
    def stage22a(self) -> Stage22ASettings:
        return Stage22ASettings()

    @property
    def stage21(self):
        return self.stage22a.stage21

    @property
    def solver(self):
        return self.stage22a.solver


PARAMETER_NAMES = (
    "a0",
    "a1",
    "a2",
    "a3",
    "theta0_degrees",
    "b1_degrees",
    "b2_degrees",
)


def transfer_eta(k_ilu: int, k_transfer: int, k_own: int) -> float:
    """Return the fraction of positive own-oracle iteration benefit recovered."""

    denominator = int(k_ilu) - int(k_own)
    if denominator <= 0:
        return float("nan")
    return float((int(k_ilu) - int(k_transfer)) / denominator)


def _coefficient(row: Mapping[str, Any]) -> SpatialAnisotropicDiffusion:
    return SpatialAnisotropicDiffusion(
        *[float(row[name]) for name in PARAMETER_NAMES]
    )


def _field_grid(size: int) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, size)
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    return np.column_stack((x.ravel(), y.ravel()))


def _field_descriptor(
    sample_id: int,
    row: Mapping[str, Any],
    points: np.ndarray,
) -> Dict[str, Any]:
    coefficient = _coefficient(row)
    theta = coefficient.angle_radians(points)
    return {
        "sample_id": int(sample_id),
        "tensor": coefficient.tensor_values(points),
        "log_ratio": coefficient.log_ratio(points),
        "orientation": np.column_stack((np.cos(2.0 * theta), np.sin(2.0 * theta))),
        "parameters": np.asarray([float(row[name]) for name in PARAMETER_NAMES]),
    }


def physical_distance_rows(
    sample_ids: Sequence[int],
    coefficient_rows: Mapping[int, Mapping[str, Any]],
    settings: Stage22TransferSettings,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Compute solver-independent symmetric distances between coefficient fields."""

    points = _field_grid(settings.field_grid_size)
    descriptors = {
        sample_id: _field_descriptor(sample_id, coefficient_rows[sample_id], points)
        for sample_id in sample_ids
    }
    count = len(sample_ids)
    k_distance = np.zeros((count, count), dtype=float)
    log_distance = np.zeros_like(k_distance)
    orientation_distance = np.zeros_like(k_distance)
    parameter_distance = np.zeros_like(k_distance)
    parameter_ranges = np.asarray(Stage19ASettings().parameter_ranges, dtype=float)
    parameter_scales = parameter_ranges[:, 1] - parameter_ranges[:, 0]
    rows: List[Dict[str, Any]] = []
    for i, first_id in enumerate(sample_ids):
        first = descriptors[first_id]
        for j, second_id in enumerate(sample_ids):
            second = descriptors[second_id]
            tensor_delta = np.linalg.norm(first["tensor"] - second["tensor"])
            tensor_scale = 0.5 * (
                np.linalg.norm(first["tensor"]) + np.linalg.norm(second["tensor"])
            )
            relative_tensor = float(tensor_delta / max(tensor_scale, np.finfo(float).tiny))
            log_rms = float(np.sqrt(np.mean((first["log_ratio"] - second["log_ratio"]) ** 2)))
            orientation_rms = float(
                np.sqrt(np.mean(np.sum((first["orientation"] - second["orientation"]) ** 2, axis=1)))
            )
            scaled_parameter_delta = (
                first["parameters"] - second["parameters"]
            ) / parameter_scales
            parameter = float(np.linalg.norm(scaled_parameter_delta) / np.sqrt(len(PARAMETER_NAMES)))
            k_distance[i, j] = relative_tensor
            log_distance[i, j] = log_rms
            orientation_distance[i, j] = orientation_rms
            parameter_distance[i, j] = parameter
            rows.append(
                {
                    "donor_id": first_id,
                    "target_id": second_id,
                    "relative_K_field_distance": relative_tensor,
                    "log_R_field_rms_distance": log_rms,
                    "orientation_field_rms_distance": orientation_rms,
                    "generation_parameter_distance": parameter,
                    "field_grid_size": settings.field_grid_size,
                    **{
                        f"donor_{name}": float(coefficient_rows[first_id][name])
                        for name in PARAMETER_NAMES
                    },
                    **{
                        f"target_{name}": float(coefficient_rows[second_id][name])
                        for name in PARAMETER_NAMES
                    },
                }
            )
    return rows, {
        "relative_K": k_distance,
        "log_R": log_distance,
        "orientation": orientation_distance,
        "parameters": parameter_distance,
    }


def _target_pair_from_donor_z(
    matrix,
    donor_z: np.ndarray,
    tolerance: float,
) -> RecyclePair:
    """Pair a transferred right space to a target operator without donor W reuse.

    The donor basis is kept unchanged unless ``A Z`` loses numerical rank.  In
    that exceptional case an SVD rotation inside the donor range removes only
    numerically null target-action directions, and the actual rank is reported.
    """

    start = perf_counter()
    z = np.asarray(donor_z, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0:
        raise ValueError("donor_z must be a nonempty matrix")
    orthogonality = np.linalg.norm(z.T @ z - np.eye(z.shape[1]), ord="fro")
    if orthogonality > 1.0e-9:
        raise ValueError("transferred donor Z must be orthonormal")
    az = np.asarray(matrix @ z, dtype=float)
    singular = la.svdvals(az, check_finite=False)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank == 0:
        raise RuntimeError("target A maps the transferred space to numerical zero")
    compressed = rank < z.shape[1]
    if compressed:
        _, _, vh = la.svd(az, full_matrices=False, check_finite=False)
        z = z @ vh[:rank, :].T
        az = np.asarray(matrix @ z, dtype=float)
    w, triangular = la.qr(az, mode="economic", check_finite=False)
    u = la.solve_triangular(
        triangular.T, z.T, lower=True, check_finite=False
    ).T
    au_error = float(
        np.linalg.norm(np.asarray(matrix @ u, dtype=float) - w, ord="fro")
        / max(np.linalg.norm(w, ord="fro"), np.finfo(float).tiny)
    )
    w_error = float(np.linalg.norm(w.T @ w - np.eye(rank), ord="fro"))
    return RecyclePair(
        z_basis=z,
        u_basis=u,
        w_basis=w,
        setup_seconds=perf_counter() - start,
        diagnostics={
            "au_equals_w_relative_error": au_error,
            "test_orthogonality_error": w_error,
            "reduced_condition": float(np.linalg.cond(triangular)),
            "dictionary_rank": float(rank),
            "input_rank": float(donor_z.shape[1]),
            "target_rank_compressed": float(compressed),
            "minimum_AZ_singular_value": float(singular[-1]),
            "maximum_AZ_singular_value": float(singular[0]),
        },
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_frozen_oracle_spaces(
    root: Path,
    output: Path,
    sample_ids: Sequence[int],
    coefficient_rows: Mapping[int, Mapping[str, Any]],
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage22TransferSettings,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    spaces: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    space_dir = output / "oracle_spaces"
    space_dir.mkdir(parents=True, exist_ok=True)
    archived_setup = _read_csv(
        root / "results" / "results-fd" / "stages"
        / "stage22A_spatial_K_multi_rhs_functional_oracle" / "setup_results.csv"
    )
    archived = {
        int(row["sample_id"]): row
        for row in archived_setup
        if row["rule_id"] == "Z_union_n3_m128_q48"
    }
    for ordinal, sample_id in enumerate(sample_ids, start=1):
        path = space_dir / f"sample_{sample_id:02d}_Z_union_n3_m128_q48.npz"
        if path.exists():
            data = np.load(path)
            z = np.asarray(data["z_basis"], dtype=float)
            measured_setup = float(data["measured_setup_seconds"])
            construction_seconds = float(data["ritz_setup_seconds"])
            pair_seconds = float(data["pair_setup_seconds"])
            compression_seconds = float(data["compression_seconds"])
            a_applications = int(data["construction_a_applications"])
            m_applications = int(data["construction_m_applications"])
            print(f"[Stage22 space {ordinal:02d}/{len(sample_ids)}] ID {sample_id:02d} cached", flush=True)
        else:
            print(f"[Stage22 space {ordinal:02d}/{len(sample_ids)}] ID {sample_id:02d} construct", flush=True)
            context, _ = _assemble_context(
                sample_id, coefficient_rows[sample_id], nodes, boundary, settings.stage21
            )
            constructions = [
                _build_ordinary_construction(
                    context, seed, settings.arnoldi_dimension, settings.stage22a
                )
                for seed in settings.construction_rhs_seeds
            ]
            raw = _raw_dictionary(constructions)
            pair, details = _z_union(
                context.system.matrix,
                raw,
                settings.requested_q,
                settings.numerical_rank_tolerance,
            )
            z = pair.z_basis
            construction_seconds = float(sum(item.setup_seconds for item in constructions))
            pair_seconds = float(details["pair_setup_seconds"])
            compression_seconds = float(details["compression_seconds"])
            measured_setup = construction_seconds + pair_seconds + compression_seconds
            a_applications = int(sum(item.a_applications for item in constructions))
            m_applications = int(sum(item.m_applications for item in constructions))
            np.savez_compressed(
                path,
                z_basis=z,
                measured_setup_seconds=measured_setup,
                ritz_setup_seconds=construction_seconds,
                pair_setup_seconds=pair_seconds,
                compression_seconds=compression_seconds,
                construction_a_applications=a_applications,
                construction_m_applications=m_applications,
            )
        archived_row = archived[sample_id]
        expected_rank = int(archived_row["actual_q"])
        if z.shape[1] != expected_rank:
            raise RuntimeError(
                f"ID {sample_id} reconstructed rank {z.shape[1]} != archived {expected_rank}"
            )
        spaces[sample_id] = {
            "z_basis": z,
            "measured_setup_seconds": measured_setup,
            "archived_setup_seconds": float(archived_row["deployable_setup_seconds"]),
        }
        rows.append(
            {
                "sample_id": sample_id,
                "construction_rhs_count": len(settings.construction_rhs_seeds),
                "arnoldi_dimension_per_rhs": settings.arnoldi_dimension,
                "requested_q": settings.requested_q,
                "actual_q": z.shape[1],
                "ritz_setup_seconds_current": construction_seconds,
                "compression_seconds_current": compression_seconds,
                "pair_setup_seconds_current": pair_seconds,
                "total_setup_seconds_current": measured_setup,
                "stage22a_archived_setup_seconds": float(archived_row["deployable_setup_seconds"]),
                "construction_a_applications": a_applications,
                "construction_m_applications": m_applications,
                "z_orthogonality_error": float(
                    np.linalg.norm(z.T @ z - np.eye(z.shape[1]), ord="fro")
                ),
            }
        )
    return spaces, rows


def _own_and_baseline_maps(
    root: Path,
    sample_ids: Sequence[int],
    settings: Stage22TransferSettings,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    stage = root / "results" / "results-fd" / "stages"
    source = _read_csv(
        stage / "stage22A_spatial_K_multi_rhs_functional_oracle" / "solver_results.csv"
    )
    own: Dict[Tuple[int, int], Dict[str, Any]] = {}
    baseline: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in source:
        sample_id = int(row["sample_id"])
        rhs_seed = int(row["heldout_rhs_seed"])
        if (
            sample_id not in sample_ids
            or rhs_seed not in settings.heldout_rhs_seeds
            or row["rule_id"] != "Z_union_n3_m128_q48"
        ):
            continue
        key = (sample_id, rhs_seed)
        own[key] = {
            "iterations": int(row["recycle_iterations"]),
            "converged": _bool(row["recycle_converged"]),
            "true_residual": float(row["recycle_true_final_relative_residual"]),
            "solve_seconds": float(row["recycle_solve_seconds"]),
            "actual_q": int(row["actual_q"]),
        }
        baseline[key] = {
            "iterations": int(row["baseline_iterations"]),
            "converged": _bool(row["baseline_converged"]),
            "true_residual": float(row["baseline_true_final_relative_residual"]),
            "solve_seconds": float(row["baseline_solve_seconds"]),
        }
    expected = len(sample_ids) * len(settings.heldout_rhs_seeds)
    if len(own) != expected or len(baseline) != expected:
        raise RuntimeError("Stage-22A archived own/baseline rows are incomplete")
    return own, baseline


def _random_pair(matrix, n: int, rank: int, seed: int, tolerance: float) -> RecyclePair:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, rank))
    z, _ = la.qr(raw, mode="economic", check_finite=False)
    return _target_pair_from_donor_z(matrix, z, tolerance)


def _safe_spearman(first: Iterable[float], second: Iterable[float]) -> Dict[str, Any]:
    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(finite))
    if count < 3 or np.ptp(x[finite]) == 0.0 or np.ptp(y[finite]) == 0.0:
        return {"coefficient": float("nan"), "pvalue": float("nan"), "count": count}
    result = stats.spearmanr(x[finite], y[finite])
    return {
        "coefficient": float(result.statistic),
        "pvalue": float(result.pvalue),
        "count": count,
    }


def _matrix_csv(path: Path, ids: Sequence[int], matrix: np.ndarray) -> None:
    rows = []
    for index, sample_id in enumerate(ids):
        row: Dict[str, Any] = {"target_id": sample_id}
        row.update({f"donor_{donor:02d}": float(matrix[index, j]) for j, donor in enumerate(ids)})
        rows.append(row)
    _write_csv(path, rows)


def _distance_rank_map(
    sample_ids: Sequence[int], physical: np.ndarray
) -> Dict[Tuple[int, int], int]:
    ranks: Dict[Tuple[int, int], int] = {}
    for target_index, target in enumerate(sample_ids):
        donors = [
            (float(physical[donor_index, target_index]), donor)
            for donor_index, donor in enumerate(sample_ids)
            if donor != target
        ]
        donors.sort()
        for rank, (_, donor) in enumerate(donors, start=1):
            ranks[(donor, target)] = rank
    return ranks


def _distance_level(rank: int, donor_count: int) -> str:
    moderate_rank = max(3, int(math.ceil(donor_count / 2)))
    if rank == 1:
        return "nearest"
    if rank == 2:
        return "second_nearest"
    if rank == moderate_rank:
        return "moderate"
    if rank == donor_count:
        return "far"
    return "other"


def _run_transfer_population(
    root: Path,
    output: Path,
    sample_ids: Sequence[int],
    coefficient_rows: Mapping[int, Mapping[str, Any]],
    archived_summaries: Mapping[int, Mapping[str, Any]],
    nodes: np.ndarray,
    boundary: np.ndarray,
    spaces: Mapping[int, Mapping[str, Any]],
    field_rows: Sequence[Mapping[str, Any]],
    field_matrices: Mapping[str, np.ndarray],
    own: Mapping[Tuple[int, int], Mapping[str, Any]],
    baseline: Mapping[Tuple[int, int], Mapping[str, Any]],
    settings: Stage22TransferSettings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    distance_lookup = {
        (int(row["donor_id"]), int(row["target_id"])): row for row in field_rows
    }
    id_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    ranks = _distance_rank_map(sample_ids, field_matrices["relative_K"])
    pair_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    random_rows: List[Dict[str, Any]] = []
    random_cache: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = {}
    for target_ordinal, target_id in enumerate(sample_ids, start=1):
        print(f"[Stage22 target {target_ordinal:02d}/{len(sample_ids)}] ID {target_id:02d}", flush=True)
        context, assembly_seconds = _assemble_context(
            target_id, coefficient_rows[target_id], nodes, boundary, settings.stage21
        )
        target_own_pair = _target_pair_from_donor_z(
            context.system.matrix,
            np.asarray(spaces[target_id]["z_basis"]),
            settings.pair_rank_tolerance,
        )
        difficulty = classify_difficulty(
            _archived_median(archived_summaries[target_id]), settings.stage21
        )
        for donor_id in sample_ids:
            if donor_id == target_id:
                continue
            donor_z = np.asarray(spaces[donor_id]["z_basis"])
            transfer_pair = _target_pair_from_donor_z(
                context.system.matrix, donor_z, settings.pair_rank_tolerance
            )
            actual_rank = transfer_pair.rank
            cache_key = (target_id, actual_rank)
            if cache_key not in random_cache:
                random_pair = _random_pair(
                    context.system.matrix,
                    context.system.matrix.shape[0],
                    actual_rank,
                    settings.random_seed + 1000 * target_id + actual_rank,
                    settings.pair_rank_tolerance,
                )
                random_cache[cache_key] = {}
                for rhs_seed in settings.heldout_rhs_seeds:
                    rhs, _ = _discrete_rhs(context.system, rhs_seed)
                    result = _run_pair(context, rhs, random_pair, settings.stage22a)
                    true_residual = float(
                        np.linalg.norm(context.system.matrix @ result.x - rhs)
                        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
                    )
                    key = (target_id, rhs_seed)
                    item = {
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged),
                        "true_residual": true_residual,
                        "solve_seconds": float(result.solve_seconds),
                        "eta": transfer_eta(
                            int(baseline[key]["iterations"]),
                            int(result.iterations),
                            int(own[key]["iterations"]),
                        ),
                    }
                    random_cache[cache_key][rhs_seed] = item
                    random_rows.append(
                        {
                            "target_id": target_id,
                            "actual_rank": actual_rank,
                            "heldout_rhs_seed": rhs_seed,
                            **item,
                            "pair_setup_seconds": random_pair.setup_seconds,
                            "au_equals_w_relative_error": random_pair.diagnostics[
                                "au_equals_w_relative_error"
                            ],
                        }
                    )
            z_metrics = generalized_subspace_metrics(donor_z, target_own_pair.z_basis)
            w_metrics = generalized_subspace_metrics(
                transfer_pair.w_basis, target_own_pair.w_basis
            )
            physical = distance_lookup[(donor_id, target_id)]
            physical_rank = ranks[(donor_id, target_id)]
            geometry_rows.append(
                {
                    "donor_id": donor_id,
                    "target_id": target_id,
                    **dict(physical),
                    "physical_distance_rank_for_target": physical_rank,
                    "distance_level": _distance_level(physical_rank, len(sample_ids) - 1),
                    "donor_rank": donor_z.shape[1],
                    "target_own_rank": target_own_pair.rank,
                    "transferred_actual_rank": actual_rank,
                    "d_Z": z_metrics["projector_distance_normalized"],
                    "d_W_target_operator": w_metrics["projector_distance_normalized"],
                    "Z_principal_angle_mean_degrees": z_metrics[
                        "principal_angle_mean_degrees"
                    ],
                    "Z_principal_angle_max_degrees": z_metrics[
                        "principal_angle_max_degrees"
                    ],
                    "W_principal_angle_mean_degrees": w_metrics[
                        "principal_angle_mean_degrees"
                    ],
                    "W_principal_angle_max_degrees": w_metrics[
                        "principal_angle_max_degrees"
                    ],
                    "Z_containment_in_target": z_metrics["containment_first"],
                    "target_containment_in_Z": z_metrics["containment_second"],
                    "W_containment_in_target": w_metrics["containment_first"],
                    "target_containment_in_W": w_metrics["containment_second"],
                    "target_pair_setup_seconds": transfer_pair.setup_seconds,
                    "target_rank_compressed": bool(
                        transfer_pair.diagnostics["target_rank_compressed"]
                    ),
                    "au_equals_w_relative_error": transfer_pair.diagnostics[
                        "au_equals_w_relative_error"
                    ],
                    "test_orthogonality_error": transfer_pair.diagnostics[
                        "test_orthogonality_error"
                    ],
                    "reduced_condition": transfer_pair.diagnostics["reduced_condition"],
                    "target_assembly_seconds": assembly_seconds,
                    "difficulty_class": difficulty,
                }
            )
            for rhs_seed in settings.heldout_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                result = _run_pair(context, rhs, transfer_pair, settings.stage22a)
                key = (target_id, rhs_seed)
                base = baseline[key]
                own_item = own[key]
                random_item = random_cache[cache_key][rhs_seed]
                true_residual = float(
                    np.linalg.norm(context.system.matrix @ result.x - rhs)
                    / max(np.linalg.norm(rhs), np.finfo(float).tiny)
                )
                counts = result.operation_counts or {}
                pair_rows.append(
                    {
                        "donor_id": donor_id,
                        "target_id": target_id,
                        "heldout_rhs_seed": rhs_seed,
                        "difficulty_class": difficulty,
                        **dict(physical),
                        "physical_distance_rank_for_target": physical_rank,
                        "distance_level": _distance_level(physical_rank, len(sample_ids) - 1),
                        "d_Z": z_metrics["projector_distance_normalized"],
                        "d_W_target_operator": w_metrics[
                            "projector_distance_normalized"
                        ],
                        "Z_principal_angle_mean_degrees": z_metrics[
                            "principal_angle_mean_degrees"
                        ],
                        "Z_principal_angle_max_degrees": z_metrics[
                            "principal_angle_max_degrees"
                        ],
                        "W_principal_angle_mean_degrees": w_metrics[
                            "principal_angle_mean_degrees"
                        ],
                        "W_principal_angle_max_degrees": w_metrics[
                            "principal_angle_max_degrees"
                        ],
                        "donor_oracle_rank": donor_z.shape[1],
                        "transferred_actual_rank": actual_rank,
                        "target_own_rank": own_item["actual_q"],
                        "baseline_iterations": base["iterations"],
                        "baseline_converged": base["converged"],
                        "baseline_true_final_relative_residual": base["true_residual"],
                        "baseline_solve_seconds": base["solve_seconds"],
                        "own_iterations": own_item["iterations"],
                        "own_converged": own_item["converged"],
                        "own_true_final_relative_residual": own_item["true_residual"],
                        "own_solve_seconds": own_item["solve_seconds"],
                        "transfer_iterations": int(result.iterations),
                        "transfer_converged": bool(result.converged),
                        "transfer_true_final_relative_residual": true_residual,
                        "transfer_solve_seconds": float(result.solve_seconds),
                        "transfer_over_ILU_iteration_ratio": float(
                            result.iterations / max(int(base["iterations"]), 1)
                        ),
                        "transfer_over_own_iteration_ratio": float(
                            result.iterations / max(int(own_item["iterations"]), 1)
                        ),
                        "eta": transfer_eta(
                            int(base["iterations"]),
                            int(result.iterations),
                            int(own_item["iterations"]),
                        ),
                        "random_iterations": random_item["iterations"],
                        "random_converged": random_item["converged"],
                        "random_true_final_relative_residual": random_item["true_residual"],
                        "random_solve_seconds": random_item["solve_seconds"],
                        "random_eta": random_item["eta"],
                        "target_pair_setup_seconds": transfer_pair.setup_seconds,
                        "donor_oracle_setup_seconds_current": spaces[donor_id][
                            "measured_setup_seconds"
                        ],
                        "donor_oracle_setup_seconds_stage22a": spaces[donor_id][
                            "archived_setup_seconds"
                        ],
                        "online_a_applications": counts.get("a_matvecs", 0),
                        "online_m_applications": counts.get(
                            "preconditioner_applications", 0
                        ),
                    }
                )
    return pair_rows, geometry_rows, random_rows


def _summarize_pairs(
    pair_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    geometry = {
        (int(row["donor_id"]), int(row["target_id"])): row for row in geometry_rows
    }
    grouped: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in pair_rows:
        grouped.setdefault((int(row["donor_id"]), int(row["target_id"])), []).append(row)
    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        first = rows[0]
        g = geometry[key]
        transfer_iterations = [int(row["transfer_iterations"]) for row in rows]
        summaries.append(
            {
                **dict(g),
                "baseline_iterations_median": float(
                    np.median([int(row["baseline_iterations"]) for row in rows])
                ),
                "own_iterations_median": float(
                    np.median([int(row["own_iterations"]) for row in rows])
                ),
                "transfer_iterations_median": float(np.median(transfer_iterations)),
                "transfer_iterations_min": min(transfer_iterations),
                "transfer_iterations_max": max(transfer_iterations),
                "transfer_all_rhs_converged": bool(
                    all(_bool(row["transfer_converged"]) for row in rows)
                ),
                "transfer_convergence_fraction": float(
                    np.mean([_bool(row["transfer_converged"]) for row in rows])
                ),
                "eta_median": float(np.median([float(row["eta"]) for row in rows])),
                "transfer_over_ILU_ratio_median": float(
                    np.median([float(row["transfer_over_ILU_iteration_ratio"]) for row in rows])
                ),
                "transfer_over_own_ratio_median": float(
                    np.median([float(row["transfer_over_own_iteration_ratio"]) for row in rows])
                ),
                "random_iterations_median": float(
                    np.median([int(row["random_iterations"]) for row in rows])
                ),
                "random_all_rhs_converged": bool(
                    all(_bool(row["random_converged"]) for row in rows)
                ),
                "random_eta_median": float(
                    np.median([float(row["random_eta"]) for row in rows])
                ),
                "transfer_beats_matched_random": bool(
                    np.median(transfer_iterations)
                    < np.median([int(row["random_iterations"]) for row in rows])
                    and sum(_bool(row["transfer_converged"]) for row in rows)
                    >= sum(_bool(row["random_converged"]) for row in rows)
                ),
                "transfer_solve_seconds_median": float(
                    np.median([float(row["transfer_solve_seconds"]) for row in rows])
                ),
                "random_solve_seconds_median": float(
                    np.median([float(row["random_solve_seconds"]) for row in rows])
                ),
                "target_pair_setup_seconds": float(first["target_pair_setup_seconds"]),
                "donor_oracle_setup_seconds_current": float(
                    first["donor_oracle_setup_seconds_current"]
                ),
            }
        )
    return summaries


def _detailed_principal_angles(
    sample_ids: Sequence[int],
    coefficient_rows: Mapping[int, Mapping[str, Any]],
    nodes: np.ndarray,
    boundary: np.ndarray,
    spaces: Mapping[int, Mapping[str, Any]],
    settings: Stage22TransferSettings,
) -> List[Dict[str, Any]]:
    """Retain every solution- and target-residual-space principal angle."""

    rows: List[Dict[str, Any]] = []
    for target_id in sample_ids:
        context, _ = _assemble_context(
            target_id, coefficient_rows[target_id], nodes, boundary, settings.stage21
        )
        target_pair = _target_pair_from_donor_z(
            context.system.matrix,
            np.asarray(spaces[target_id]["z_basis"]),
            settings.pair_rank_tolerance,
        )
        for donor_id in sample_ids:
            if donor_id == target_id:
                continue
            donor_z = np.asarray(spaces[donor_id]["z_basis"])
            transfer_pair = _target_pair_from_donor_z(
                context.system.matrix, donor_z, settings.pair_rank_tolerance
            )
            z_angles = np.degrees(la.subspace_angles(donor_z, target_pair.z_basis))
            w_angles = np.degrees(
                la.subspace_angles(transfer_pair.w_basis, target_pair.w_basis)
            )
            if z_angles.size != w_angles.size:
                raise RuntimeError("Z and W principal-angle counts must agree")
            for index, (z_angle, w_angle) in enumerate(
                zip(z_angles, w_angles), start=1
            ):
                rows.append(
                    {
                        "donor_id": donor_id,
                        "target_id": target_id,
                        "angle_index_largest_first": index,
                        "Z_principal_angle_degrees": float(z_angle),
                        "W_principal_angle_degrees_target_operator": float(w_angle),
                        "shared_angle_count": z_angles.size,
                    }
                )
    return rows


def _select_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            not _bool(row["transfer_all_rhs_converged"]),
            float(row["transfer_iterations_median"]),
            -float(row["eta_median"]),
            int(row["donor_id"]),
        ),
    )


def _target_summary(
    sample_ids: Sequence[int],
    pair_summaries: Sequence[Mapping[str, Any]],
    settings: Stage22TransferSettings,
) -> List[Dict[str, Any]]:
    result = []
    for target_id in sample_ids:
        rows = [row for row in pair_summaries if int(row["target_id"]) == target_id]
        physical = min(rows, key=lambda row: float(row["relative_K_field_distance"]))
        parameter = min(rows, key=lambda row: float(row["generation_parameter_distance"]))
        best = _select_best(rows)
        second = min(
            (row for row in rows if int(row["physical_distance_rank_for_target"]) == 2),
            key=lambda row: float(row["relative_K_field_distance"]),
        )
        moderate_rank = max(3, int(math.ceil(len(rows) / 2)))
        moderate = next(
            row for row in rows if int(row["physical_distance_rank_for_target"]) == moderate_rank
        )
        far = next(
            row for row in rows if int(row["physical_distance_rank_for_target"]) == len(rows)
        )
        result.append(
            {
                "target_id": target_id,
                "difficulty_class": physical["difficulty_class"],
                "baseline_iterations_median": physical["baseline_iterations_median"],
                "own_iterations_median": physical["own_iterations_median"],
                "own_rank": physical["target_own_rank"],
                "physical_nearest_donor_id": physical["donor_id"],
                "physical_nearest_distance": physical["relative_K_field_distance"],
                "physical_nearest_transfer_iterations_median": physical[
                    "transfer_iterations_median"
                ],
                "physical_nearest_all_rhs_converged": physical[
                    "transfer_all_rhs_converged"
                ],
                "physical_nearest_eta_median": physical["eta_median"],
                "physical_nearest_transfer_over_ILU_ratio_median": physical[
                    "transfer_over_ILU_ratio_median"
                ],
                "physical_nearest_transfer_over_own_ratio_median": physical[
                    "transfer_over_own_ratio_median"
                ],
                "physical_nearest_random_iterations_median": physical[
                    "random_iterations_median"
                ],
                "physical_nearest_random_eta_median": physical["random_eta_median"],
                "physical_nearest_beats_random": physical[
                    "transfer_beats_matched_random"
                ],
                "parameter_nearest_donor_id": parameter["donor_id"],
                "parameter_nearest_distance": parameter["generation_parameter_distance"],
                "parameter_nearest_transfer_iterations_median": parameter[
                    "transfer_iterations_median"
                ],
                "parameter_nearest_all_rhs_converged": parameter[
                    "transfer_all_rhs_converged"
                ],
                "parameter_nearest_eta_median": parameter["eta_median"],
                "best_donor_id_diagnostic_only": best["donor_id"],
                "best_transfer_iterations_median_diagnostic_only": best[
                    "transfer_iterations_median"
                ],
                "best_eta_median_diagnostic_only": best["eta_median"],
                "best_all_rhs_converged_diagnostic_only": best[
                    "transfer_all_rhs_converged"
                ],
                "best_minus_physical_nearest_eta_gap": float(best["eta_median"])
                - float(physical["eta_median"]),
                "second_nearest_donor_id": second["donor_id"],
                "second_nearest_eta_median": second["eta_median"],
                "moderate_donor_id": moderate["donor_id"],
                "moderate_eta_median": moderate["eta_median"],
                "far_donor_id": far["donor_id"],
                "far_eta_median": far["eta_median"],
                "physical_nearest_eta_at_least_0_75": bool(
                    float(physical["eta_median"]) >= settings.high_recovery_eta
                ),
            }
        )
    return result


def _gate(
    target_rows: Sequence[Mapping[str, Any]],
    settings: Stage22TransferSettings,
) -> Dict[str, Any]:
    count = len(target_rows)
    convergence_fraction = float(
        np.mean([_bool(row["physical_nearest_all_rhs_converged"]) for row in target_rows])
    )
    median_eta = float(
        np.median([float(row["physical_nearest_eta_median"]) for row in target_rows])
    )
    median_ratio = float(
        np.median(
            [float(row["physical_nearest_transfer_over_ILU_ratio_median"]) for row in target_rows]
        )
    )
    random_win_fraction = float(
        np.mean([_bool(row["physical_nearest_beats_random"]) for row in target_rows])
    )
    high_recovery_fraction = float(
        np.mean([_bool(row["physical_nearest_eta_at_least_0_75"]) for row in target_rows])
    )
    checks = {
        "all_rhs_convergence_fraction": convergence_fraction,
        "all_rhs_convergence_fraction_required": settings.nearest_all_rhs_convergence_fraction_gate,
        "convergence_gate_passed": convergence_fraction
        >= settings.nearest_all_rhs_convergence_fraction_gate,
        "median_eta": median_eta,
        "median_eta_required": settings.median_eta_gate,
        "eta_gate_passed": median_eta >= settings.median_eta_gate,
        "median_transfer_over_ILU_iteration_ratio": median_ratio,
        "meaningful_iteration_ratio_maximum": settings.meaningful_iteration_ratio_gate,
        "meaningful_improvement_gate_passed": median_ratio
        <= settings.meaningful_iteration_ratio_gate,
        "matched_random_win_fraction": random_win_fraction,
        "matched_random_win_fraction_required": settings.random_win_fraction_gate,
        "random_control_gate_passed": random_win_fraction
        >= settings.random_win_fraction_gate,
        "fraction_eta_at_least_0_75": high_recovery_fraction,
        "targets": count,
    }
    checks["transfer_gate_passed"] = bool(
        checks["convergence_gate_passed"]
        and checks["eta_gate_passed"]
        and checks["meaningful_improvement_gate_passed"]
        and checks["random_control_gate_passed"]
    )
    return checks


def _correlations(pair_summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    predictors = (
        "relative_K_field_distance",
        "log_R_field_rms_distance",
        "orientation_field_rms_distance",
        "generation_parameter_distance",
        "d_Z",
        "d_W_target_operator",
        "Z_principal_angle_max_degrees",
        "W_principal_angle_max_degrees",
    )
    outcomes = (
        "eta_median",
        "transfer_over_ILU_ratio_median",
        "transfer_over_own_ratio_median",
        "transfer_all_rhs_converged",
    )
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            x = [float(row[predictor]) for row in pair_summaries]
            if outcome == "transfer_all_rhs_converged":
                y = [float(_bool(row[outcome])) for row in pair_summaries]
            else:
                y = [float(row[outcome]) for row in pair_summaries]
            result = _safe_spearman(x, y)
            rows.append(
                {
                    "scope": "all_ordered_hard_system_pairs",
                    "predictor": predictor,
                    "outcome": outcome,
                    "spearman_coefficient": result["coefficient"],
                    "spearman_pvalue_descriptive": result["pvalue"],
                    "pair_count": result["count"],
                    "dependent_pair_observations": True,
                }
            )
    return rows


def _locality_summary(
    sample_ids: Sequence[int], pair_summaries: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for target_id in sample_ids:
        group = [row for row in pair_summaries if int(row["target_id"]) == target_id]
        result = _safe_spearman(
            [float(row["relative_K_field_distance"]) for row in group],
            [float(row["eta_median"]) for row in group],
        )
        rows.append(
            {
                "target_id": target_id,
                "spearman_K_distance_vs_eta": result["coefficient"],
                "spearman_pvalue_descriptive": result["pvalue"],
                "donor_count": result["count"],
            }
        )
    return rows


def _heatmap(
    path: Path,
    matrix: np.ndarray,
    ids: Sequence[int],
    title: str,
    subtitle: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 6.7))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(ids)), labels=ids, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), labels=ids)
    ax.set_xlabel("Donor system ID")
    ax.set_ylabel("Target system ID")
    fig.suptitle(title, x=0.125, y=0.985, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.125, 0.948, subtitle, ha="left", fontsize=9, color=GRAY)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scatter(
    path: Path,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    subtitle: str,
    xlabel: str,
    rho: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.scatter(x, y, s=25, alpha=0.68, facecolor=BLUE, edgecolor="white", linewidth=0.35)
    ax.axhline(0.0, color=CHARCOAL, linewidth=0.9, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Oracle-benefit recovery $\eta$ (median over RHSs)")
    fig.suptitle(title, x=0.125, y=0.985, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.125,
        0.948,
        f"{subtitle}; Spearman $\\rho$={rho:.3f}",
        ha="left",
        fontsize=9,
        color=GRAY,
    )
    ax.grid(axis="both", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_plots(
    output: Path,
    sample_ids: Sequence[int],
    matrices: Mapping[str, np.ndarray],
    pair_summaries: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _heatmap(
        plot_dir / "physical_K_distance.png",
        matrices["physical_K"], sample_ids,
        "Physical spatial-$K$ distance", "Symmetric relative tensor-field $L^2$ distance", "Blues"
    )
    _heatmap(
        plot_dir / "d_Z_matrix.png", matrices["d_Z"], sample_ids,
        "$Z$-space distance", "Normalized projector distance; unequal ranks retained", "Purples"
    )
    _heatmap(
        plot_dir / "target_side_d_W_matrix.png", matrices["d_W"], sample_ids,
        "Target-side $W$-space distance", r"Both spaces formed with target $A_j$", "YlOrBr"
    )
    eta_plot = matrices["eta"].copy()
    np.fill_diagonal(eta_plot, np.nan)
    _heatmap(
        plot_dir / "eta_matrix.png", eta_plot, sample_ids,
        "Cross-system oracle-benefit recovery", r"Median $\eta$ over three held-out target RHSs", "coolwarm", -0.5, 1.1
    )
    correlation_lookup = {
        (row["predictor"], row["outcome"]): float(row["spearman_coefficient"])
        for row in correlations
    }
    for predictor, filename, title, xlabel in (
        ("relative_K_field_distance", "K_distance_vs_eta.png", "Physical distance versus transfer recovery", "Relative spatial-$K$ field distance"),
        ("d_Z", "d_Z_vs_eta.png", "$Z$ geometry versus transfer recovery", "Normalized $Z$ projector distance"),
        ("d_W_target_operator", "d_W_vs_eta.png", "Target-side $W$ geometry versus transfer recovery", "Normalized target-side $W$ projector distance"),
    ):
        _scatter(
            plot_dir / filename,
            [float(row[predictor]) for row in pair_summaries],
            [float(row["eta_median"]) for row in pair_summaries],
            title,
            "210 ordered donor-target pairs; descriptive dependent observations",
            xlabel,
            correlation_lookup[(predictor, "eta_median")],
        )
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ordered = sorted(target_rows, key=lambda row: int(row["target_id"]))
    x = np.arange(len(ordered))
    nearest = [float(row["physical_nearest_eta_median"]) for row in ordered]
    best = [float(row["best_eta_median_diagnostic_only"]) for row in ordered]
    width = 0.38
    ax.bar(x - width / 2, nearest, width, color=BLUE, edgecolor=CHARCOAL, linewidth=0.4, label="Physical-nearest")
    ax.bar(x + width / 2, best, width, facecolor="white", edgecolor=ORANGE, linewidth=1.2, label="Best donor (diagnostic)")
    ax.axhline(0.5, color=CHARCOAL, linewidth=0.9, linestyle="--", label=r"$\eta=0.50$ gate")
    ax.set_xticks(x, labels=[int(row["target_id"]) for row in ordered])
    ax.set_xlabel("Target system ID")
    ax.set_ylabel(r"Median oracle-benefit recovery $\eta$")
    fig.suptitle(
        "Physical-nearest versus best-transfer donor",
        x=0.125,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.125,
        0.948,
        "Best donor uses outcome information and is not deployment-valid",
        ha="left",
        fontsize=9,
        color=GRAY,
    )
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(plot_dir / "nearest_vs_best_donor_eta.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    _write_json(output / "manifest.json", {"files": files})


def _write_protocol(output: Path, sample_ids: Sequence[int], settings: Stage22TransferSettings) -> None:
    _write_json(
        output / "predeclared_protocol.json",
        {
            "written_before_transfer_execution": True,
            "stage": "22_spatial_K_transfer_gate",
            "neural_training_performed": False,
            "oracle_construction_frozen_from_stage22A": {
                "construction_rhs_seeds": list(settings.construction_rhs_seeds),
                "arnoldi_dimension_per_rhs": settings.arnoldi_dimension,
                "ordinary_ritz_pair_preserving": True,
                "candidates_per_rhs": settings.candidates_per_rhs,
                "compression": "Z_union_leading_left_singular_vectors",
                "requested_q": settings.requested_q,
            },
            "population": {
                "hard_transition_ids": list(sample_ids),
                "transition_count": 0,
                "donor_target_pairs": len(sample_ids) * (len(sample_ids) - 1),
                "heldout_rhs_seeds": list(settings.heldout_rhs_seeds),
            },
            "deployment_selection": {
                "primary": "minimum symmetric relative K-field L2 distance",
                "secondary": "minimum normalized generation-parameter distance",
                "geometry_not_used_for_selection": True,
                "best_transfer_donor_is_diagnostic_only": True,
            },
            "gate": {
                "nearest_all_rhs_convergence_fraction_at_least": settings.nearest_all_rhs_convergence_fraction_gate,
                "median_eta_at_least": settings.median_eta_gate,
                "median_transfer_over_ILU_iteration_ratio_at_most": settings.meaningful_iteration_ratio_gate,
                "fraction_beating_matched_random_at_least": settings.random_win_fraction_gate,
                "high_recovery_eta_threshold": settings.high_recovery_eta,
            },
        },
    )
    _write_json(output / "settings.json", asdict(settings))


def _validation(
    output: Path,
    sample_ids: Sequence[int],
    pair_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    field_matrices: Mapping[str, np.ndarray],
    gate: Mapping[str, Any],
    settings: Stage22TransferSettings,
) -> Dict[str, Any]:
    expected_pairs = len(sample_ids) * (len(sample_ids) - 1)
    expected_rows = expected_pairs * len(settings.heldout_rhs_seeds)
    checks = {
        "pairwise_rhs_row_count": len(pair_rows) == expected_rows,
        "geometry_pair_count": len(geometry_rows) == expected_pairs,
        "target_count": len(target_rows) == len(sample_ids),
        "no_self_transfer_rows": all(int(row["donor_id"]) != int(row["target_id"]) for row in pair_rows),
        "field_distance_symmetric": bool(
            np.allclose(field_matrices["relative_K"], field_matrices["relative_K"].T, atol=1.0e-13)
        ),
        "field_distance_zero_diagonal": bool(
            np.allclose(np.diag(field_matrices["relative_K"]), 0.0, atol=1.0e-13)
        ),
        "paired_AU_identity_max_below_1e_9": max(
            float(row["au_equals_w_relative_error"]) for row in geometry_rows
        ) < 1.0e-9,
        "paired_W_orthogonality_max_below_1e_9": max(
            float(row["test_orthogonality_error"]) for row in geometry_rows
        ) < 1.0e-9,
        "target_rank_compression_count": int(sum(
            _bool(row["target_rank_compressed"]) for row in geometry_rows
        )),
        "all_eta_denominators_positive": all(np.isfinite(float(row["eta"])) for row in pair_rows),
        "nearest_is_physical_rank_one": all(
            int(row["physical_nearest_donor_id"])
            == int(
                min(
                    (g for g in geometry_rows if int(g["target_id"]) == int(row["target_id"])),
                    key=lambda value: float(value["relative_K_field_distance"]),
                )["donor_id"]
            )
            for row in target_rows
        ),
        "best_donor_excluded_from_gate": True,
        "gate_recomputed": dict(gate),
    }
    required_boolean_checks = (
        "pairwise_rhs_row_count",
        "geometry_pair_count",
        "target_count",
        "no_self_transfer_rows",
        "field_distance_symmetric",
        "field_distance_zero_diagonal",
        "paired_AU_identity_max_below_1e_9",
        "paired_W_orthogonality_max_below_1e_9",
        "all_eta_denominators_positive",
        "nearest_is_physical_rank_one",
        "best_donor_excluded_from_gate",
    )
    checks["ready_to_report"] = bool(
        all(bool(checks[name]) for name in required_boolean_checks)
    )
    _write_json(output / "validation.json", checks)
    return checks


def _write_source_notes(output: Path) -> None:
    content = """# Stage 22 source and QA notes

## Decision frame

- Technical question: does a frozen Stage-22A right recycle space transfer between physically nearby hard spatial-K systems?
- Decision cohort: the 15 archived hard systems used in Stage 22A. The archived population has no transition systems.
- Primary deployment rule: minimum symmetric relative tensor-field distance, computed on a fixed 81 by 81 grid without solver outcomes.
- The best-transfer donor is retained only as an unattainable diagnostic upper bound.

## Evidence lineage

- Stage-19A coefficient rows define the seven-parameter spatial fields.
- Stage-22A archived solver rows supply identical ILU and own-oracle comparisons for held-out RHS seeds 20261101--20261103.
- Frozen donor Z spaces are deterministically reconstructed with the Stage-22A code and saved under `oracle_spaces/`.
- `pairwise_transfer.csv` is at donor-target-RHS grain. `pairwise_transfer_summary.csv` is at ordered-pair grain.

## Metric cautions

- Fourteen of 15 ILU baselines hit the 300-iteration cap. Eta and iteration ratios therefore use the project's established censored convention.
- Ordered donor-target pairs and RHSs are dependent observations. Spearman p-values are descriptive only.
- Random controls are cached per target and actual rank, so pairs sharing target and rank intentionally share the same random control.
- Target-side W geometry is unavailable for deployment and is used only diagnostically.

## Chart map and QA

- Matrix heatmaps show physical K distance, Z distance, target-side W distance, and median eta at all 15 by 15 system pairs.
- Relationship scatters use all 210 ordered pairs and explicitly label the dependent-observation scope.
- The nearest-versus-best bar chart separates deployment-valid and diagnostic donor selection.
- Static PNGs use restrained blue/orange roots, neutral references, labeled axes, and no truncated absolute-magnitude bar scale.
"""
    (output / "source_notes.md").write_text(content, encoding="utf-8")


def run_stage22_transfer(root: Path) -> Path:
    """Run the frozen hard-population transfer gate and preserve raw evidence."""

    settings = Stage22TransferSettings()
    output = (
        root / "results" / "results-fd" / "stages"
        / "stage22_spatial_K_transfer_gate"
    )
    output.mkdir(parents=True, exist_ok=True)
    (
        _, nodes, boundary, coefficient_rows, archived_summaries, _, valid_ids, _
    ) = _load_population(root, settings.stage21)
    sample_ids = [
        sample_id for sample_id in valid_ids
        if classify_difficulty(
            _archived_median(archived_summaries[sample_id]), settings.stage21
        ) in {"transition", "hard"}
    ]
    _write_protocol(output, sample_ids, settings)
    field_rows, field_matrices = physical_distance_rows(
        sample_ids, coefficient_rows, settings
    )
    _write_csv(output / "physical_field_distances.csv", field_rows)
    descriptor_rows = []
    for sample_id in sample_ids:
        source = coefficient_rows[sample_id]
        descriptor_rows.append(
            {
                "sample_id": sample_id,
                **{name: float(source[name]) for name in PARAMETER_NAMES},
                "ratio_min_nodes": float(source["ratio_min_nodes"]),
                "ratio_max_nodes": float(source["ratio_max_nodes"]),
                "ratio_mean_nodes": float(source["ratio_mean_nodes"]),
                "log_ratio_std_nodes": float(source["log_ratio_std_nodes"]),
                "theta_min_degrees_nodes": float(source["theta_min_degrees_nodes"]),
                "theta_max_degrees_nodes": float(source["theta_max_degrees_nodes"]),
                "theta_mean_degrees_nodes": float(source["theta_mean_degrees_nodes"]),
                "theta_std_degrees_nodes": float(source["theta_std_degrees_nodes"]),
            }
        )
    _write_csv(output / "spatial_K_system_descriptors.csv", descriptor_rows)
    spaces, construction_rows = _build_frozen_oracle_spaces(
        root, output, sample_ids, coefficient_rows, nodes, boundary, settings
    )
    _write_csv(output / "oracle_construction_results.csv", construction_rows)
    own, baseline = _own_and_baseline_maps(root, sample_ids, settings)
    pair_rows, geometry_rows, random_rows = _run_transfer_population(
        root, output, sample_ids, coefficient_rows, archived_summaries,
        nodes, boundary, spaces, field_rows, field_matrices, own, baseline, settings
    )
    pair_summaries = _summarize_pairs(pair_rows, geometry_rows)
    angle_rows = _detailed_principal_angles(
        sample_ids, coefficient_rows, nodes, boundary, spaces, settings
    )
    target_rows = _target_summary(sample_ids, pair_summaries, settings)
    gate = _gate(target_rows, settings)
    correlation_rows = _correlations(pair_summaries)
    locality_rows = _locality_summary(sample_ids, pair_summaries)
    _write_csv(output / "pairwise_transfer.csv", pair_rows)
    _write_csv(output / "pairwise_transfer_geometry.csv", geometry_rows)
    _write_csv(output / "pairwise_principal_angles.csv", angle_rows)
    _write_csv(output / "pairwise_transfer_summary.csv", pair_summaries)
    _write_csv(output / "target_transfer_summary.csv", target_rows)
    _write_csv(output / "random_control_results.csv", random_rows)
    _write_csv(output / "transfer_correlations.csv", correlation_rows)
    _write_csv(output / "transfer_locality_by_target.csv", locality_rows)
    _write_json(output / "transfer_gate.json", gate)
    count = len(sample_ids)
    id_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    dz = np.zeros((count, count), dtype=float)
    dw = np.zeros((count, count), dtype=float)
    eta = np.ones((count, count), dtype=float)
    for row in pair_summaries:
        target_index = id_index[int(row["target_id"])]
        donor_index = id_index[int(row["donor_id"])]
        dz[target_index, donor_index] = float(row["d_Z"])
        dw[target_index, donor_index] = float(row["d_W_target_operator"])
        eta[target_index, donor_index] = float(row["eta_median"])
    matrices = {
        "physical_K": field_matrices["relative_K"].T,
        "d_Z": dz,
        "d_W": dw,
        "eta": eta,
    }
    _matrix_csv(output / "physical_K_distance_matrix.csv", sample_ids, matrices["physical_K"])
    _matrix_csv(output / "d_Z_matrix.csv", sample_ids, dz)
    _matrix_csv(output / "target_side_d_W_matrix.csv", sample_ids, dw)
    _matrix_csv(output / "eta_matrix.csv", sample_ids, eta)
    np.savez_compressed(
        output / "transfer_matrices.npz",
        sample_ids=np.asarray(sample_ids, dtype=int),
        physical_K_distance=matrices["physical_K"],
        log_R_distance=field_matrices["log_R"].T,
        orientation_distance=field_matrices["orientation"].T,
        generation_parameter_distance=field_matrices["parameters"].T,
        d_Z=dz,
        target_side_d_W=dw,
        eta=eta,
    )
    _make_plots(output, sample_ids, matrices, pair_summaries, target_rows, correlation_rows)
    validation = _validation(
        output, sample_ids, pair_rows, geometry_rows, target_rows,
        field_matrices, gate, settings
    )
    _write_json(
        output / "aggregate_results.json",
        {
            "population": {
                "hard_transition_systems": len(sample_ids),
                "sample_ids": sample_ids,
                "ordered_pairs": len(pair_summaries),
                "per_rhs_rows": len(pair_rows),
                "principal_angle_rows": len(angle_rows),
            },
            "gate": gate,
            "correlations": correlation_rows,
            "locality": locality_rows,
            "validation_ready_to_report": validation["ready_to_report"],
            "no_neural_training": True,
        },
    )
    _write_source_notes(output)
    _manifest(output)
    return output


if __name__ == "__main__":
    print(run_stage22_transfer(Path(__file__).resolve().parents[3]))
