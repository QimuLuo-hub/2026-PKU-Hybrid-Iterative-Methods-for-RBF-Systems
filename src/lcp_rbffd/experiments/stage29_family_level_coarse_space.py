"""Stage 29: family-level coarse-space mechanism test.

The experiment uses only Stage-28 training-system full teacher spaces to form
one family dictionary.  It applies frozen energy-prefix spaces to untouched
Stage-28 final-test systems with target-side nonsymmetric pairing.  No model is
trained and no target-system teacher information enters deployable methods.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp

from ..preconditioners import ilu_preconditioner
from ..rbffd import RBFSystem
from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import SystemContext, _relative_residual
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    _orthonormalize,
    _pair_from_z,
    _run_pair,
)
from .stage28_spatial_k_operator_learning import Stage28Settings
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
class Stage29Settings:
    """Frozen Stage-29 controls, matching ``predeclared_protocol.json``."""

    rank_labels: Tuple[str, ...] = ("energy90", "energy95", "energy99")
    ranks: Tuple[int, ...] = (80, 105, 168)
    energies: Tuple[float, ...] = (
        0.9004603656147988,
        0.9513278889426774,
        0.9901591340888743,
    )
    random_seed: int = 20262999
    a_rhs_convergence_min: float = 0.90
    a_system_convergence_min: float = 0.70
    a_median_eta_min: float = 0.70
    a_iteration_ratio_max: float = 0.50
    a_random_win_fraction_min: float = 0.70
    b_rhs_convergence_min: float = 0.90
    b_system_convergence_min: float = 0.70
    b_median_eta_min: float = 0.70
    b_iteration_ratio_max: float = 0.50
    ceiling_label: str = "oracle_within_p168"

    @property
    def stage28(self) -> Stage28Settings:
        return Stage28Settings()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): _file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _stage28_context(
    root: Path,
    sample_id: int,
    nodes: np.ndarray,
    boundary: np.ndarray,
    coefficients: Sequence[Any],
    settings: Stage29Settings,
) -> SystemContext:
    """Create a context around the exact saved Stage-28 sparse matrix."""

    source = root / "results/results-fd/stage28_operator_learning"
    matrix = sp.load_npz(source / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
    system = RBFSystem(
        matrix=matrix,
        nodes=nodes,
        boundary=boundary,
        interior=~boundary,
        diagnostics={},
        config=settings.stage28.stage19a.config,
        coefficient=coefficients[sample_id],
    )
    ilu = ilu_preconditioner(
        matrix,
        settings.stage28.stage19a.stage10.classical_ilu,
        name="strong_ilu",
    )
    return SystemContext(sample_id, system, ilu)


def build_family_dictionary(
    stage28: Path, output: Path, split: Mapping[str, Sequence[int]], settings: Stage29Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build or validate the training-only dictionary from full teachers."""

    path = output / "family_dictionary.npz"
    metadata_path = output / "family_dictionary_metadata.json"
    if path.exists() and metadata_path.exists():
        with np.load(path) as archive:
            psi = np.asarray(archive["psi"], dtype=float)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        snapshots = []
        for sample_id in split["train"]:
            with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
                snapshots.append(np.asarray(archive["zfull"], dtype=float))
        snapshot_matrix = np.column_stack(snapshots)
        started = perf_counter()
        dictionary, singular_values, numerical_rank = _orthonormalize(
            snapshot_matrix, 1.0e-14
        )
        energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        pmax = max(settings.ranks)
        psi = dictionary[:, :pmax]
        np.savez_compressed(
            path,
            psi=psi.astype(np.float32),
            singular_values=singular_values,
            energy=energy,
            training_ids=np.asarray(split["train"], dtype=int),
        )
        metadata = {
            "source": "Stage-28 training zfull spaces only",
            "training_system_count": len(split["train"]),
            "snapshot_columns": snapshot_matrix.shape[1],
            "numerical_rank": numerical_rank,
            "ranks": {
                str(int(100 * target)): int(np.searchsorted(energy, target) + 1)
                for target in (0.90, 0.95, 0.99)
            },
            "captured_energy": {
                str(int(100 * target)): float(
                    energy[int(np.searchsorted(energy, target))]
                )
                for target in (0.90, 0.95, 0.99)
            },
            "svd_seconds": perf_counter() - started,
        }
        _write_json(metadata_path, metadata)
    expected = dict(zip(settings.ranks, settings.energies))
    if psi.shape[1] != max(settings.ranks):
        raise RuntimeError("saved family dictionary does not match frozen p_max")
    with np.load(path) as archive:
        energy = np.asarray(archive["energy"], dtype=float)
        singular_values = np.asarray(archive["singular_values"], dtype=float)
        training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()
    if training_ids != list(split["train"]):
        raise RuntimeError("family dictionary training IDs do not match frozen split")
    for rank, target in expected.items():
        if not np.isclose(energy[rank - 1], target, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"energy mismatch at frozen rank {rank}")
    rows = [
        {
            "mode": index + 1,
            "singular_value": float(value),
            "normalized_singular_value": float(value / singular_values[0]),
            "captured_energy": float(energy[index]),
            "selected": (index + 1) in settings.ranks,
        }
        for index, value in enumerate(singular_values)
    ]
    _write_csv(output / "family_dictionary_spectrum.csv", rows)
    return psi, metadata


def oracle_select_inside_dictionary(
    matrix: sp.spmatrix,
    psi: np.ndarray,
    target_w: np.ndarray,
) -> Tuple[RecyclePair, Dict[str, float]]:
    """Residual-space-optimal target-informed selection inside ``range(Psi)``.

    If ``A Psi = Q R``, the leading left singular vectors of ``Q.T W_*``
    define the subspace of ``range(A Psi)`` with maximal projector overlap
    with the target full-teacher residual space.  A triangular solve maps those
    residual directions back into ``range(Psi)``.  This is diagnostic only.
    """

    psi_q, _ = la.qr(np.asarray(psi, dtype=float), mode="economic", check_finite=False)
    image = np.asarray(matrix @ psi_q, dtype=float)
    family_w, triangular = la.qr(image, mode="economic", check_finite=False)
    target_q, _ = la.qr(np.asarray(target_w, dtype=float), mode="economic", check_finite=False)
    overlap = family_w.T @ target_q
    left, singular_values, _ = la.svd(overlap, full_matrices=False, check_finite=False)
    selected_rank = min(target_q.shape[1], left.shape[1])
    coordinates = la.solve_triangular(
        triangular, left[:, :selected_rank], lower=False, check_finite=False
    )
    z_raw = psi_q @ coordinates
    pair = _pair_from_z(matrix, z_raw)
    return pair, {
        "selected_rank": float(pair.rank),
        "family_pair_condition": float(np.linalg.cond(triangular)),
        "mean_squared_principal_cosine": float(
            np.sum(singular_values[:selected_rank] ** 2) / max(selected_rank, 1)
        ),
        "minimum_principal_cosine": float(singular_values[selected_rank - 1]),
    }


def _archived_row(
    row: Mapping[str, Any], method: str, hard: bool,
) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "hard": hard,
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage28 archived",
        "energy_label": "",
        "captured_training_energy": "",
        "requested_rank": int(row["actual_rank"]),
        "actual_rank": int(row["actual_rank"]),
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "ilu_setup_seconds": 0.0,
        "method_setup_seconds": float(row["pair_setup_seconds"]),
        "online_solve_seconds": float(row["online_solve_seconds"]),
        "total_target_online_seconds": float(row["online_total_seconds"]),
        "offline_family_setup_seconds": 0.0,
        "own_teacher_setup_seconds": float(row["teacher_setup_seconds_charged"]),
        "A_applications": int(row["A_applications"]),
        "M_applications": int(row["M_applications"]),
        "reduced_condition": "",
        "au_equals_w_relative_error": "",
        "failure": row["failure"],
    }


def _new_row(
    sample_id: int,
    hard: bool,
    rhs_seed: int,
    method: str,
    pair: RecyclePair,
    result: Any,
    true_residual: float,
    ilu_setup_seconds: float,
    energy_label: str = "",
    captured_energy: Any = "",
    requested_rank: int = 0,
) -> Dict[str, Any]:
    counts = result.operation_counts or {}
    return {
        "sample_id": sample_id,
        "hard": hard,
        "rhs_seed": rhs_seed,
        "method": method,
        "source": "Stage29 new",
        "energy_label": energy_label,
        "captured_training_energy": captured_energy,
        "requested_rank": requested_rank,
        "actual_rank": pair.rank,
        "iterations": int(result.iterations),
        "converged": bool(result.converged and true_residual <= 1.0e-8),
        "true_final_residual": true_residual,
        "ilu_setup_seconds": ilu_setup_seconds,
        "method_setup_seconds": pair.setup_seconds,
        "online_solve_seconds": float(result.solve_seconds),
        "total_target_online_seconds": pair.setup_seconds + float(result.solve_seconds),
        "offline_family_setup_seconds": 0.0,
        "own_teacher_setup_seconds": 0.0,
        "A_applications": int(counts.get("a_matvecs", 0)),
        "M_applications": int(counts.get("preconditioner_applications", 0)),
        "reduced_condition": pair.diagnostics["reduced_condition"],
        "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
        "failure": getattr(result, "failure", "") or "",
    }


def run_solver_experiment(
    root: Path,
    output: Path,
    psi: np.ndarray,
    dictionary_metadata: Mapping[str, Any],
    settings: Stage29Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    hard_ids = set(int(value) for value in split["hard_ids"])
    archived = _read_csv(stage28 / "solver_results.csv")
    archived_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage28.stage19a)
    rng = np.random.default_rng(settings.random_seed)
    random_raw = rng.normal(size=(nodes.shape[0], max(settings.ranks)))
    random_basis, _ = la.qr(random_raw, mode="economic", check_finite=False)

    rows: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    system_rows: List[Dict[str, Any]] = []
    baseline_methods = ("strong_ilu", "nearest_reuse", "teacher_q28", "teacher_full")
    for position, sample_id in enumerate(split["test"], start=1):
        hard = sample_id in hard_ids
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings
        )
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
        full_pair = _pair_from_z(context.system.matrix, zfull)
        pair_map: Dict[str, RecyclePair] = {}
        setup_started = perf_counter()
        for label, rank, energy in zip(settings.rank_labels, settings.ranks, settings.energies):
            method = f"shared_p{rank}"
            pair_map[method] = _pair_from_z(context.system.matrix, psi[:, :rank])
            random_method = f"random_p{rank}"
            pair_map[random_method] = _pair_from_z(
                context.system.matrix, random_basis[:, :rank]
            )
            shared_z_metrics = generalized_subspace_metrics(
                pair_map[method].z_basis, full_pair.z_basis
            )
            shared_w_metrics = generalized_subspace_metrics(
                pair_map[method].w_basis, full_pair.w_basis
            )
            geometry.append(
                {
                    "sample_id": sample_id,
                    "hard": hard,
                    "method": method,
                    "requested_rank": rank,
                    "actual_rank": pair_map[method].rank,
                    "captured_training_energy": energy,
                    "d_Z_to_full": shared_z_metrics["projector_distance_normalized"],
                    "d_W_to_full": shared_w_metrics["projector_distance_normalized"],
                    "maximum_angle_Z_degrees": shared_z_metrics["principal_angle_max_degrees"],
                    "maximum_angle_W_degrees": shared_w_metrics["principal_angle_max_degrees"],
                    "mean_squared_principal_cosine": "",
                    "family_pair_condition": pair_map[method].diagnostics["reduced_condition"],
                }
            )
        ceiling_pair, ceiling_diagnostics = oracle_select_inside_dictionary(
            context.system.matrix, psi[:, : max(settings.ranks)], wfull
        )
        pair_map[settings.ceiling_label] = ceiling_pair
        ceiling_z_metrics = generalized_subspace_metrics(
            ceiling_pair.z_basis, full_pair.z_basis
        )
        ceiling_w_metrics = generalized_subspace_metrics(
            ceiling_pair.w_basis, full_pair.w_basis
        )
        geometry.append(
            {
                "sample_id": sample_id,
                "hard": hard,
                "method": settings.ceiling_label,
                "requested_rank": max(settings.ranks),
                "actual_rank": ceiling_pair.rank,
                "captured_training_energy": settings.energies[-1],
                "d_Z_to_full": ceiling_z_metrics["projector_distance_normalized"],
                "d_W_to_full": ceiling_w_metrics["projector_distance_normalized"],
                "maximum_angle_Z_degrees": ceiling_z_metrics["principal_angle_max_degrees"],
                "maximum_angle_W_degrees": ceiling_w_metrics["principal_angle_max_degrees"],
                "mean_squared_principal_cosine": ceiling_diagnostics["mean_squared_principal_cosine"],
                "family_pair_condition": ceiling_diagnostics["family_pair_condition"],
            }
        )
        pair_construction_seconds = perf_counter() - setup_started
        system_rows.append(
            {
                "sample_id": sample_id,
                "hard": hard,
                "matrix_nnz": context.system.matrix.nnz,
                "ilu_setup_seconds": context.ilu.setup_seconds,
                "target_pair_suite_seconds": pair_construction_seconds,
                "full_teacher_actual_rank": full_pair.rank,
                "full_teacher_pair_defect": full_pair.diagnostics["au_equals_w_relative_error"],
            }
        )
        for rhs_seed in settings.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = archived_lookup[(sample_id, rhs_seed, "strong_ilu")]
            full = archived_lookup[(sample_id, rhs_seed, "teacher_full")]
            for method in baseline_methods:
                rows.append(
                    _archived_row(
                        archived_lookup[(sample_id, rhs_seed, method)], method, hard
                    )
                )
            for label, rank, energy in zip(settings.rank_labels, settings.ranks, settings.energies):
                for method in (f"shared_p{rank}", f"random_p{rank}"):
                    result = _run_pair(context, rhs, pair_map[method], settings.stage28.stage22a)
                    true_residual = _relative_residual(
                        context.system.matrix, result.x, rhs
                    )
                    rows.append(
                        _new_row(
                            sample_id, hard, rhs_seed, method, pair_map[method], result,
                            true_residual, context.ilu.setup_seconds,
                            energy_label=label, captured_energy=energy,
                            requested_rank=rank,
                        )
                    )
                    for iteration, residual in zip(
                        result.true_residual_iterations, result.true_residual_history
                    ):
                        histories.append(
                            {
                                "sample_id": sample_id,
                                "hard": hard,
                                "rhs_seed": rhs_seed,
                                "method": method,
                                "iteration": iteration,
                                "true_relative_residual": residual,
                            }
                        )
            result = _run_pair(
                context, rhs, ceiling_pair, settings.stage28.stage22a
            )
            true_residual = _relative_residual(context.system.matrix, result.x, rhs)
            rows.append(
                _new_row(
                    sample_id, hard, rhs_seed, settings.ceiling_label,
                    ceiling_pair, result, true_residual, context.ilu.setup_seconds,
                    energy_label="energy99 target-informed diagnostic",
                    captured_energy=settings.energies[-1],
                    requested_rank=max(settings.ranks),
                )
            )
            for iteration, residual in zip(
                result.true_residual_iterations, result.true_residual_history
            ):
                histories.append(
                    {
                        "sample_id": sample_id,
                        "hard": hard,
                        "rhs_seed": rhs_seed,
                        "method": settings.ceiling_label,
                        "iteration": iteration,
                        "true_relative_residual": residual,
                    }
                )
            baseline_iterations = int(baseline["iterations"])
            full_iterations = int(full["iterations"])
            denominator = baseline_iterations - full_iterations
            for row in rows:
                if int(row["sample_id"]) != sample_id or int(row["rhs_seed"]) != rhs_seed:
                    continue
                row["baseline_iterations"] = baseline_iterations
                row["full_teacher_iterations"] = full_iterations
                row["eta_full"] = (
                    (baseline_iterations - int(row["iterations"])) / denominator
                    if denominator > 0 else float("nan")
                )
                row["iteration_ratio_to_ilu"] = int(row["iterations"]) / max(
                    baseline_iterations, 1
                )
        print(
            f"[Stage29 {position:02d}/{len(split['test'])}] system {sample_id}",
            flush=True,
        )
    _write_csv(output / "solver_results.csv", rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    _write_csv(output / "residual_histories.csv", histories)
    _write_csv(output / "system_setup.csv", system_rows)
    return rows, geometry, histories, system_rows


def summarize(
    rows: Sequence[Mapping[str, Any]], settings: Stage29Settings,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    summaries: List[Dict[str, Any]] = []
    for cohort, predicate in (
        ("all_test", lambda row: True),
        ("hard_test", lambda row: _bool(row["hard"])),
    ):
        cohort_rows = [row for row in rows if predicate(row)]
        system_ids = sorted({int(row["sample_id"]) for row in cohort_rows})
        for method in methods:
            local = [row for row in cohort_rows if row["method"] == method]
            if not local:
                continue
            system_passes = {
                sample_id: all(
                    _bool(row["converged"])
                    for row in local if int(row["sample_id"]) == sample_id
                )
                for sample_id in system_ids
            }
            summaries.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "systems": len(system_ids),
                    "rhs_count": len(local),
                    "rhs_convergence_fraction": float(
                        np.mean([_bool(row["converged"]) for row in local])
                    ),
                    "all_rhs_system_pass_fraction": float(
                        np.mean(list(system_passes.values()))
                    ),
                    "median_iterations": float(
                        np.median([int(row["iterations"]) for row in local])
                    ),
                    "median_system_iterations": float(
                        np.median(
                            [
                                np.median(
                                    [
                                        int(row["iterations"])
                                        for row in local
                                        if int(row["sample_id"]) == sample_id
                                    ]
                                )
                                for sample_id in system_ids
                            ]
                        )
                    ),
                    "median_eta_full": _finite_median(
                        [float(row["eta_full"]) for row in local]
                    ),
                    "median_iteration_ratio_to_ilu": _finite_median(
                        [float(row["iteration_ratio_to_ilu"]) for row in local]
                    ),
                    "median_actual_rank": float(
                        np.median([int(row["actual_rank"]) for row in local])
                    ),
                    "median_method_setup_seconds": float(
                        np.median([float(row["method_setup_seconds"]) for row in local])
                    ),
                    "median_online_solve_seconds": float(
                        np.median([float(row["online_solve_seconds"]) for row in local])
                    ),
                    "median_target_online_seconds": float(
                        np.median([float(row["total_target_online_seconds"]) for row in local])
                    ),
                }
            )
    summary_lookup = {
        (row["cohort"], row["method"]): row for row in summaries
    }
    hard_rows = [row for row in rows if _bool(row["hard"])]
    hard_ids = sorted({int(row["sample_id"]) for row in hard_rows})
    a_candidates = []
    for rank in settings.ranks:
        method = f"shared_p{rank}"
        random_method = f"random_p{rank}"
        summary = summary_lookup[("hard_test", method)]
        wins = []
        for sample_id in hard_ids:
            shared_local = [
                row for row in hard_rows
                if row["method"] == method and int(row["sample_id"]) == sample_id
            ]
            random_local = [
                row for row in hard_rows
                if row["method"] == random_method and int(row["sample_id"]) == sample_id
            ]
            shared_convergence = sum(_bool(row["converged"]) for row in shared_local)
            random_convergence = sum(_bool(row["converged"]) for row in random_local)
            wins.append(
                shared_convergence > random_convergence
                or (
                    shared_convergence == random_convergence
                    and np.median([int(row["iterations"]) for row in shared_local])
                    < np.median([int(row["iterations"]) for row in random_local])
                )
            )
        random_win_fraction = float(np.mean(wins))
        passes = bool(
            float(summary["rhs_convergence_fraction"]) >= settings.a_rhs_convergence_min
            and float(summary["all_rhs_system_pass_fraction"]) >= settings.a_system_convergence_min
            and float(summary["median_eta_full"]) >= settings.a_median_eta_min
            and float(summary["median_iteration_ratio_to_ilu"]) <= settings.a_iteration_ratio_max
            and random_win_fraction >= settings.a_random_win_fraction_min
        )
        a_candidates.append(
            {
                "method": method,
                "rank": rank,
                "random_win_fraction": random_win_fraction,
                "passes_A": passes,
            }
        )
    passing_a = [candidate for candidate in a_candidates if candidate["passes_A"]]
    ceiling = summary_lookup[("hard_test", settings.ceiling_label)]
    passes_b = bool(
        float(ceiling["rhs_convergence_fraction"]) >= settings.b_rhs_convergence_min
        and float(ceiling["all_rhs_system_pass_fraction"]) >= settings.b_system_convergence_min
        and float(ceiling["median_eta_full"]) >= settings.b_median_eta_min
        and float(ceiling["median_iteration_ratio_to_ilu"]) <= settings.b_iteration_ratio_max
    )
    if passing_a:
        selected = min(passing_a, key=lambda candidate: candidate["rank"])
        decision = "A"
        statement = "A reusable family-level coarse space exists under the frozen gate."
        next_direction = (
            "Measure repeated-solve economics and compress the fixed shared-space application "
            "without introducing target-specific prediction."
        )
    elif passes_b:
        selected = None
        decision = "B"
        statement = "The training span contains useful information, but target-specific selection is missing."
        next_direction = (
            "Test one reduced target-specific gating/coordinate selector inside the frozen family dictionary."
        )
    else:
        selected = None
        decision = "C"
        statement = "Even target-informed selection inside the training span does not robustly recover the own-system benefit."
        next_direction = (
            "Stop the shared-space ML route and investigate a different classical complementary-preconditioning mechanism."
        )
    payload = {
        "decision": decision,
        "statement": statement,
        "selected_shared_method": None if selected is None else selected["method"],
        "A_candidates": a_candidates,
        "B_ceiling_passes": passes_b,
        "next_direction": next_direction,
        "hard_test_systems": hard_ids,
        "hard_test_system_count": len(hard_ids),
    }
    return summaries, payload


def make_plots(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    settings: Stage29Settings,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    chart_map: List[Dict[str, str]] = []
    with np.load(output / "family_dictionary.npz") as archive:
        energy = np.asarray(archive["energy"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    modes = np.arange(1, len(energy) + 1)
    ax.plot(modes, energy, color=BLUE, linewidth=2.0)
    for rank, value, color in zip(settings.ranks, settings.energies, (OLIVE, PURPLE, ORANGE)):
        ax.axvline(rank, color=color, linestyle="--", linewidth=1.3, label=f"p={rank}, E={value:.3f}")
    ax.set(
        xlabel="family dictionary rank p",
        ylabel="captured training snapshot energy",
        ylim=(0.0, 1.01),
        title="Training full-teacher family dictionary",
    )
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "family_dictionary_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "family_dictionary_spectrum.csv", "purpose": "Document the frozen 90/95/99% training-energy rank ladder."})

    hard_ids = sorted({int(row["sample_id"]) for row in rows if _bool(row["hard"])})
    methods = ("strong_ilu", "nearest_reuse", "shared_p80", "shared_p105", "shared_p168", settings.ceiling_label, "teacher_full")
    labels = ("strong ILU", "nearest reuse", "shared p=80", "shared p=105", "shared p=168", "oracle inside p=168", "own full")
    colors = (GRAY, PURPLE, BLUE, OLIVE, ORANGE, "#B14D75", CHARCOAL)
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    x = np.arange(len(hard_ids), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for offset, method, label, color in zip(offsets, methods, labels, colors):
        values = [
            np.median(
                [int(row["iterations"]) for row in rows if int(row["sample_id"]) == sid and row["method"] == method]
            )
            for sid in hard_ids
        ]
        ax.scatter(x + offset, values, s=42, color=color, label=label)
    ax.axhline(300, color=CHARCOAL, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, [str(value) for value in hard_ids])
    ax.set(
        xlabel="untouched hard final-test system ID",
        ylabel="median GMRES iterations over three RHSs",
        title="Family-level coarse-space solver comparison",
    )
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    path = plot_dir / "hard_system_iterations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Compare shared spaces, target-informed ceiling, and exact baselines on the hard cohort."})

    summary_lookup = {(row["cohort"], row["method"]): row for row in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    ranks = np.asarray(settings.ranks)
    convergence = [float(summary_lookup[("hard_test", f"shared_p{rank}")]["rhs_convergence_fraction"]) for rank in ranks]
    eta = [float(summary_lookup[("hard_test", f"shared_p{rank}")]["median_eta_full"]) for rank in ranks]
    axes[0].plot(ranks, convergence, color=BLUE, marker="o")
    axes[0].axhline(settings.a_rhs_convergence_min, color=CHARCOAL, linestyle="--")
    axes[0].set(xlabel="shared rank p", ylabel="hard-RHS convergence fraction", ylim=(0, 1.05), title="Convergence across the rank ladder")
    axes[1].plot(ranks, eta, color=ORANGE, marker="s")
    axes[1].axhline(settings.a_median_eta_min, color=CHARCOAL, linestyle="--")
    axes[1].set(xlabel="shared rank p", ylabel=r"median $\eta_{full}$", title="Own-oracle benefit recovery")
    for axis in axes:
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    path = plot_dir / "shared_rank_gate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_summary.csv", "purpose": "Apply the preregistered shared-space gate to the three training-energy ranks."})

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    shared_values = []
    ceiling_values = []
    for sample_id in hard_ids:
        shared_values.append(_finite_median([float(row["eta_full"]) for row in rows if int(row["sample_id"]) == sample_id and row["method"] == "shared_p168"]))
        ceiling_values.append(_finite_median([float(row["eta_full"]) for row in rows if int(row["sample_id"]) == sample_id and row["method"] == settings.ceiling_label]))
    width = 0.36
    ax.bar(np.arange(len(hard_ids)) - width / 2, shared_values, width, color=ORANGE, edgecolor=CHARCOAL, label="shared p=168")
    ax.bar(np.arange(len(hard_ids)) + width / 2, ceiling_values, width, color=BLUE, edgecolor=CHARCOAL, label="oracle selection inside p=168")
    ax.axhline(0.7, color=CHARCOAL, linestyle="--", linewidth=1.1)
    ax.set_xticks(np.arange(len(hard_ids)), [str(value) for value in hard_ids])
    ax.set(xlabel="hard final-test system ID", ylabel=r"median $\eta_{full}$", title="Representation ceiling inside the training span")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "representation_ceiling.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Separate an inadequate shared space from missing target-specific selection."})

    geometry_lookup = {(int(row["sample_id"]), row["method"]): row for row in geometry}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for method, marker, color, label in (("shared_p168", "o", ORANGE, "shared p=168"), (settings.ceiling_label, "s", BLUE, "oracle selection")):
        x_values, y_values = [], []
        for sample_id in hard_ids:
            x_values.append(float(geometry_lookup[(sample_id, method)]["d_W_to_full"]))
            y_values.append(_finite_median([float(row["eta_full"]) for row in rows if int(row["sample_id"]) == sample_id and row["method"] == method]))
        ax.scatter(x_values, y_values, marker=marker, color=color, s=52, label=label)
    ax.axhline(0.7, color=CHARCOAL, linestyle="--", linewidth=1.0)
    ax.set(xlabel="target-side normalized W-projector distance", ylabel=r"median $\eta_{full}$", title="Residual-space geometry and solver recovery")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "geometry_vs_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "geometry_diagnostics.csv and solver_results.csv", "purpose": "Show whether the target-informed ceiling improves both W overlap and functional recovery."})

    _write_csv(output / "chart_map.csv", chart_map)


def _fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def write_report(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    dictionary: Mapping[str, Any],
    settings: Stage29Settings,
) -> Path:
    summary = {(row["cohort"], row["method"]): row for row in summaries}
    hard_ids = decision["hard_test_systems"]
    labels = {
        "strong_ilu": "strong ILU",
        "nearest_reuse": "nearest training-space reuse",
        "shared_p80": "shared $p=80$ (90% energy)",
        "shared_p105": "shared $p=105$ (95% energy)",
        "shared_p168": "shared $p=168$ (99% energy)",
        "oracle_within_p168": "target-informed selection inside $p=168$",
        "random_p168": "matched random $p=168$",
        "teacher_q28": "own-system q=28",
        "teacher_full": "own-system full",
    }
    primary_methods = (
        "strong_ilu", "nearest_reuse", "shared_p80", "shared_p105",
        "shared_p168", "oracle_within_p168", "random_p168",
        "teacher_q28", "teacher_full",
    )
    lines = [
        "# Stage 29: family-level coarse-space mechanism test",
        "",
        "## Technical conclusion",
        "",
        f"**Decision {decision['decision']}: {decision['statement']}** The decision is based on the seven untouched Stage-28 hard/transition final-test systems and the preregistered solver gate. {decision['next_direction']}",
        "",
        "This experiment is not single-donor transfer. It concatenates the successful full own-system teacher spaces from all 43 Stage-28 training systems, extracts one training-only family basis, and then pairs each frozen prefix with the current target matrix. Final-test teachers are used only as solver baselines and in the explicitly nondeployable representation-ceiling diagnostic.",
        "",
        "## Scope and metrics",
        "",
        "The numerical problem is unchanged: $N=5041$ fixed jittered-grid nodes (seed 7, jitter 0.15), stencil size 25, PHS $r^3$ with degree-2 polynomial augmentation, full Dirichlet rows, and the saved spatially varying SPD tensor fields. Strong ILU uses drop tolerance $10^{-3}$, fill factor 5, and COLAMD. Right-preconditioned recycled GMRES uses restart 50, cap 300, and true-residual tolerance $10^{-8}$. Evaluation uses the three Stage-28 held-out RHS seeds 20263001--20263003.",
        "",
        "For each target and shared basis $\\Psi_p$, thin QR gives $A_j\\Psi_p=W_jR_j$, followed by the triangular solve $U_j=\\Psi_pR_j^{-1}$. Thus $A_jU_j=W_j$, and Arnoldi acts on the existing projected right-preconditioned problem. Oracle recovery is $\\eta_{full}=(k_{ILU}-k)/(k_{ILU}-k_{full})$ whenever the denominator is positive. A system converges only when all three true residuals meet tolerance.",
        "",
        "## The family basis is larger than the earlier system-specific target",
        "",
        f"The training snapshot matrix contains {dictionary['snapshot_columns']} columns from the 43 training full-teacher spaces. The frozen ranks $p=80$, 105, and 168 capture 90.05%, 95.13%, and 99.02% of training energy. They are 1.59%, 2.08%, and 3.33% of $N$, respectively. No final-test space contributes to this SVD.",
        "",
        "![Family dictionary energy](plots/family_dictionary_energy.png)",
        "",
        "The figure documents the only shared-rank ladder tested. It is energy-based and was frozen before any Stage-29 final-test solve, so solver outcomes did not choose the ranks.",
        "",
        "## Shared-space and representation-ceiling results",
        "",
        "The aggregate table uses only the hard/transition final-test cohort. Capped failures remain at 300 iterations.",
        "",
        "| method | RHS convergence | all-RHS systems | median iterations | median $\\eta_{full}$ | iteration/ILU | rank | pair setup (s) | solve (s) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for method in primary_methods:
        row = summary[("hard_test", method)]
        lines.append(
            f"| {labels[method]} | {_fmt(row['rhs_convergence_fraction'])} | {_fmt(row['all_rhs_system_pass_fraction'])} | {_fmt(row['median_iterations'], 1)} | {_fmt(row['median_eta_full'])} | {_fmt(row['median_iteration_ratio_to_ilu'])} | {_fmt(row['median_actual_rank'], 1)} | {_fmt(row['median_method_setup_seconds'], 4)} | {_fmt(row['median_online_solve_seconds'], 4)} |"
        )
    lines.extend([
        "",
        "![Hard-system iterations](plots/hard_system_iterations.png)",
        "",
        "The system-level comparison shows whether any population median is driven by a small subset. The shared basis is applied unchanged, while every method receives the same target-side nonsymmetric pairing and held-out RHSs.",
        "",
        "| ID | ILU | nearest | shared 80 | shared 105 | shared 168 | ceiling | random 168 | own full |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for sample_id in hard_ids:
        values = {}
        for method in ("strong_ilu", "nearest_reuse", "shared_p80", "shared_p105", "shared_p168", "oracle_within_p168", "random_p168", "teacher_full"):
            local = [row for row in rows if int(row["sample_id"]) == int(sample_id) and row["method"] == method]
            value = int(np.median([int(row["iterations"]) for row in local]))
            converged = all(_bool(row["converged"]) for row in local)
            values[method] = f"{value}{'' if converged else ' (cap)'}"
        lines.append(
            f"| {sample_id} | {values['strong_ilu']} | {values['nearest_reuse']} | {values['shared_p80']} | {values['shared_p105']} | {values['shared_p168']} | {values['oracle_within_p168']} | {values['random_p168']} | {values['teacher_full']} |"
        )
    lines.extend([
        "",
        "![Shared-rank gate](plots/shared_rank_gate.png)",
        "",
        "The left panel applies the convergence component of the preregistered gate. The right panel shows benefit recovery relative to each target's expensive full teacher. Passing one scalar median is insufficient when held-out systems still fail.",
        "",
        "## Target-informed selection tests the representation ceiling",
        "",
        "The diagnostic first forms $A_j\\Psi_{168}=Q_jR_j$. Using the target full-teacher residual basis $W_*$ only for analysis, the leading principal vectors of $Q_j^TW_*$ select the most aligned residual block available inside the training-derived span. A triangular solve maps that block back into $\\operatorname{range}(\\Psi_{168})$. This is an optimistic ceiling, not a deployable selector.",
        "",
        "![Representation ceiling](plots/representation_ceiling.png)",
        "",
        "The comparison separates two failure modes. If the ceiling succeeds while the full shared prefix fails, family information is present but target-specific coordinates are missing. If the ceiling also fails, the training span itself lacks enough target solver information under this construction.",
        "",
        "![Geometry and recovery](plots/geometry_vs_recovery.png)",
        "",
        "Residual-space geometry is diagnostic only. The decision remains based on closed-loop convergence and iterations, not projector overlap.",
        "",
        "## Costs, limitations, and robustness",
        "",
        f"The one-time family SVD took {_fmt(dictionary['svd_seconds'], 3)} s after the training teachers already existed. It does not include the substantially larger offline cost of constructing those 43 own-system teachers. Per-target pairing and online costs are reported separately above. The shared prefixes are larger than the own-system references, so a solver benefit must be judged against their additional projection and QR costs rather than iteration count alone.",
        "",
        "The experiment reuses a single fixed 72-system sample and its existing final-test split. The seven-system hard cohort is adequate for the preregistered mechanism gate but not for broad statistical claims. Most hard ILU baselines hit the 300-step cap, so recovery uses the project's censored convention. The target-informed ceiling is deliberately optimistic and unavailable at deployment. The Gaussian control tests whether teacher provenance matters, but it is not an exhaustive generic-basis study.",
        "",
        "## Final decision and one next direction",
        "",
        f"**{decision['decision']}. {decision['statement']}**",
        "",
        f"**Next direction:** {decision['next_direction']}",
        "",
        "No neural model was trained, and Stage 30 was not started.",
    ])
    report = output / "stage29_family_level_coarse_space_mechanism.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate(
    root: Path,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage29Settings,
) -> Dict[str, Any]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    expected_methods = 4 + 2 * len(settings.ranks) + 1
    expected_rows = len(split["test"]) * len(settings.stage28.evaluation_rhs_seeds) * expected_methods
    new_rows = [row for row in rows if row["source"] == "Stage29 new"]
    converged = [row for row in rows if _bool(row["converged"])]
    nonfinite_core = []
    for index, row in enumerate(rows):
        for column in ("iterations", "true_final_residual", "method_setup_seconds", "online_solve_seconds"):
            if not np.isfinite(float(row[column])):
                nonfinite_core.append((index, column))
    maximum_pair_defect = max(
        float(row["au_equals_w_relative_error"])
        for row in new_rows
        if str(row["au_equals_w_relative_error"]) != ""
    )
    training_ids = np.load(output / "family_dictionary.npz")["training_ids"].astype(int).tolist()
    validation = {
        "predeclared_protocol_predates_solver_results": (output / "predeclared_protocol.json").stat().st_mtime <= (output / "solver_results.csv").stat().st_mtime,
        "training_ids_exactly_match_stage28": training_ids == split["train"],
        "test_ids_used_in_family_dictionary": len(set(training_ids) & set(split["test"])),
        "solver_rows_expected": expected_rows,
        "solver_rows_actual": len(rows),
        "geometry_rows_expected": len(split["test"]) * (len(settings.ranks) + 1),
        "geometry_rows_actual": len(geometry),
        "nonfinite_core_values": nonfinite_core,
        "maximum_converged_true_relative_residual": max(float(row["true_final_residual"]) for row in converged),
        "maximum_pair_defect": maximum_pair_defect,
        "decision": decision["decision"],
        "no_neural_training": True,
        "no_stage30": True,
    }
    validation["passed"] = bool(
        validation["predeclared_protocol_predates_solver_results"]
        and validation["training_ids_exactly_match_stage28"]
        and validation["test_ids_used_in_family_dictionary"] == 0
        and validation["solver_rows_actual"] == expected_rows
        and validation["geometry_rows_actual"] == validation["geometry_rows_expected"]
        and not nonfinite_core
        and validation["maximum_converged_true_relative_residual"] <= 1.0e-8
        and maximum_pair_defect <= 1.0e-9
    )
    _write_json(output / "validation.json", validation)
    return validation


def run_stage29(root: Path) -> Path:
    settings = Stage29Settings()
    output = root / "results/results-fd/stages/stage29_family_level_coarse_space_mechanism"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("predeclared_protocol.json must exist before solver evaluation")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    )
    stage28 = root / "results/results-fd/stage28_operator_learning"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    psi, dictionary = build_family_dictionary(stage28, output, split, settings)
    print("[1/4] frozen family dictionary verified", flush=True)
    rows, geometry, histories, system_rows = run_solver_experiment(
        root, output, psi, dictionary, settings
    )
    print("[2/4] final-test solves complete", flush=True)
    summaries, decision = summarize(rows, settings)
    _write_csv(output / "solver_summary.csv", summaries)
    _write_json(output / "decision.json", decision)
    make_plots(output, rows, geometry, summaries, settings)
    report = write_report(output, rows, summaries, decision, dictionary, settings)
    print("[3/4] report and figures complete", flush=True)
    validation = validate(root, output, rows, geometry, decision, settings)
    _write_json(
        output / "source_notes.json",
        {
            "audience": "technical",
            "delivery_surface": "project-requested Markdown",
            "source_reports_read": ["Stage22A", "Stage22", "Stage23", "Stage24", "Stage25", "Stage26", "Stage27", "Stage28"],
            "primary_source": "Stage-28 training/test systems and archived baselines",
            "rank_ladder_source": "training full-teacher snapshot energy only",
            "chart_policy": "reproducible static Matplotlib figures in the requested stage artifact",
            "validation_assessment": "ready to share" if validation["passed"] else "needs revision",
            "omitted_requested_metrics": [],
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage29_family_level_coarse_space_mechanism.md"
    shutil.copyfile(report, report_copy)
    print("[4/4] Stage 29 complete", flush=True)
    return report_copy
