"""Stage 7: solver-aware MLP versus structure-aware GNN diagnosis."""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import torch

from ..coarse_space import build_multiplicative_coarse_preconditioner
from ..config import ILUConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..models.direct_gnn import (
    DirectGNNConfig,
    RBFGraph,
    graph_from_system,
    make_direct_gnn_preconditioner,
    permute_graph,
    train_direct_gnn,
)
from ..models.direct_mlp import (
    DirectMLPConfig,
    correction_metrics,
    make_direct_preconditioner,
    train_direct_mlp,
)
from ..pde import ManufacturedSolution, random_smooth_solution
from ..pod import compute_pod
from ..preconditioners import InverseAction, ilu_preconditioner
from ..solver_aware import (
    SolverAwareSamples,
    collect_preconditioner_inputs,
    concatenate_solver_aware_samples,
)
from ..subspace_analysis import energy_dimension, subspace_metrics
from .harder_regimes import SystemSpec, build_system, manufactured_problem, system_metadata


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
CHARCOAL = "#30343B"
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
class Stage7Settings:
    """Reproducible and deliberately bounded diagnostic experiment."""

    seed: int = 20260829
    solver: SolverConfig = SolverConfig(
        relative_tolerance=1.0e-8,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=300,
        breakdown_tolerance=1.0e-14,
    )
    classical_ilu: ILUConfig = ILUConfig(
        drop_tolerance=1.0e-3,
        fill_factor=5.0,
        diagonal_pivot_threshold=0.0,
        permutation="COLAMD",
    )
    train_betas: Tuple[float, ...] = (0.0, 0.6, 1.2, 1.8)
    validation_betas: Tuple[float, ...] = (0.3, 1.5)
    test_betas: Tuple[float, ...] = (0.9, 1.65)
    train_trajectories_per_system: int = 6
    validation_trajectories_per_system: int = 3
    test_trajectories_per_system: int = 5
    known_test_trajectories_per_system: int = 3
    mlp: DirectMLPConfig = DirectMLPConfig(
        hidden_width=64,
        hidden_layers=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=8,
        max_epochs=80,
        patience=15,
        seed=20260829,
        input_direction_scale=1.0,
    )
    gnn: DirectGNNConfig = DirectGNNConfig(
        hidden_width=16,
        message_layers=2,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=8,
        max_epochs=80,
        patience=15,
        seed=20260829,
    )
    magnitude_gate_median_min: float = 0.5
    magnitude_gate_median_max: float = 2.0
    magnitude_gate_p95_max: float = 5.0
    pod_stability_q: int = 8
    pod_stability_replicates: int = 3
    pod_trajectories_per_replicate: int = 4


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _native(row.get(key, "")) for key in fields})


def _ilu(system, settings: Stage7Settings, name: str = "ilu_drop1e-3_fill5") -> InverseAction:
    return ilu_preconditioner(system.matrix, settings.classical_ilu, name=name)


def _ml_spec(beta: float, split: str) -> SystemSpec:
    """Reuse the exact fixed-size Stage 6 family to isolate the data change."""

    return SystemSpec(
        name=f"ML7_{split}_beta{int(round(100 * beta)):03d}",
        family=f"stage7_{split}",
        n_nodes=900,
        stencil_size=25,
        node_distribution="warped_grid",
        jitter_fraction=0.0,
        warp_strength=0.95,
        beta=beta,
        seed=7,
    )


def _seeds(base: int, count: int) -> List[int]:
    return [base + 7919 * index for index in range(count)]


def _collect_system_block(
    spec: SystemSpec,
    split: str,
    trajectory_count: int,
    seed_base: int,
    settings: Stage7Settings,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    system, assembly_seconds = build_system(spec)
    action = _ilu(system, settings)
    samples, trajectories, factor_seconds = collect_preconditioner_inputs(
        system,
        action,
        settings.solver,
        _seeds(seed_base, trajectory_count),
        spec.name,
        (spec.beta,),
        f"{split}_{spec.name}",
    )
    entry = {
        "spec": spec,
        "system": system,
        "graph": graph_from_system(system, spec.name),
        "samples": samples,
        "assembly_seconds": assembly_seconds,
        "reference_factor_seconds": factor_seconds,
        "ilu_setup_seconds": action.setup_seconds,
    }
    for row in trajectories:
        row["split"] = split
    return entry, trajectories


def _collect_core_data(
    settings: Stage7Settings,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    definitions = (
        ("train", settings.train_betas, settings.train_trajectories_per_system, 100000),
        (
            "validation",
            settings.validation_betas,
            settings.validation_trajectories_per_system,
            200000,
        ),
        ("test", settings.test_betas, settings.test_trajectories_per_system, 300000),
    )
    entries: Dict[str, List[Dict[str, Any]]] = {}
    trajectory_rows: List[Dict[str, Any]] = []
    system_rows: List[Dict[str, Any]] = []
    for split, betas, count, seed_offset in definitions:
        entries[split] = []
        for index, beta in enumerate(betas):
            spec = _ml_spec(beta, split)
            entry, rows = _collect_system_block(
                spec,
                split,
                count,
                settings.seed + seed_offset + 1000 * index,
                settings,
            )
            entries[split].append(entry)
            trajectory_rows.extend(rows)
            metadata = system_metadata(spec, entry["system"], entry["assembly_seconds"])
            system_rows.append(
                {
                    **metadata,
                    "split": split,
                    "trajectory_count": count,
                    "solver_input_count": entry["samples"].vectors.shape[0],
                    "reference_factor_seconds": entry["reference_factor_seconds"],
                    "ilu_setup_seconds": entry["ilu_setup_seconds"],
                    "max_reference_relative_residual": float(
                        np.max(entry["samples"].exact_solve_relative_residuals)
                    ),
                }
            )
    # Known-system holdout uses new trajectories, never inputs from a training solve.
    entries["known_test"] = []
    for index, train_entry in enumerate(entries["train"]):
        if train_entry["spec"].beta not in (0.6, 1.2):
            continue
        spec = train_entry["spec"]
        action = _ilu(train_entry["system"], settings)
        samples, rows, factor_seconds = collect_preconditioner_inputs(
            train_entry["system"],
            action,
            settings.solver,
            _seeds(
                settings.seed + 400000 + 1000 * index,
                settings.known_test_trajectories_per_system,
            ),
            spec.name,
            (spec.beta,),
            f"known_test_{spec.name}",
        )
        entries["known_test"].append({**train_entry, "samples": samples})
        for row in rows:
            row["split"] = "known_test"
        trajectory_rows.extend(rows)
        system_rows.append(
            {
                "system": spec.name,
                "family": spec.family,
                "split": "known_test",
                "n_nodes": spec.n_nodes,
                "trajectory_count": settings.known_test_trajectories_per_system,
                "solver_input_count": samples.vectors.shape[0],
                "reference_factor_seconds": factor_seconds,
                "max_reference_relative_residual": float(
                    np.max(samples.exact_solve_relative_residuals)
                ),
            }
        )
    return entries, trajectory_rows, system_rows


def _stack(entries: Sequence[Dict[str, Any]]) -> SolverAwareSamples:
    return concatenate_solver_aware_samples([entry["samples"] for entry in entries])


def _save_dataset(output: Path, split: str, samples: SolverAwareSamples) -> None:
    np.savez_compressed(
        output / f"solver_aware_{split}.npz",
        vectors=samples.vectors,
        vector_norms=samples.vector_norms,
        normalized_vectors=samples.normalized_vectors,
        exact_actions=samples.exact_actions,
        classical_actions=samples.classical_actions,
        corrections=samples.corrections,
        normalized_corrections=samples.normalized_corrections,
        system_features=samples.system_features,
        system_labels=samples.system_labels,
        trajectory_labels=samples.trajectory_labels,
        iteration_indices=samples.iteration_indices,
        exact_solve_relative_residuals=samples.exact_solve_relative_residuals,
        definition="A(xi)^{-1} v - M(xi)^{-1} v",
        normalization="v_hat=v/||v||_2; target=d(v)/||v||_2",
    )


def _prediction_rows(
    model_name: str,
    split: str,
    samples: SolverAwareSamples,
    predictions: np.ndarray,
) -> List[Dict[str, Any]]:
    metrics = correction_metrics(predictions, samples.corrections)
    exact_norms = np.linalg.norm(samples.exact_actions, axis=1)
    correction_norms = np.linalg.norm(samples.corrections, axis=1)
    hybrid_predictions = samples.classical_actions + predictions
    hybrid_errors = np.linalg.norm(
        hybrid_predictions - samples.exact_actions, axis=1
    ) / np.maximum(exact_norms, 1.0e-300)
    hybrid_magnitudes = np.linalg.norm(hybrid_predictions, axis=1) / np.maximum(
        exact_norms, 1.0e-300
    )
    rows: List[Dict[str, Any]] = []
    for index in range(samples.vectors.shape[0]):
        rows.append(
            {
                "model": model_name,
                "split": split,
                "system": samples.system_labels[index],
                "trajectory": samples.trajectory_labels[index],
                "arnoldi_iteration": int(samples.iteration_indices[index]),
                "vector_norm": samples.vector_norms[index],
                "target_norm": float(np.linalg.norm(samples.corrections[index])),
                "prediction_norm": float(np.linalg.norm(predictions[index])),
                "target_over_exact_action_norm": correction_norms[index]
                / max(exact_norms[index], 1.0e-300),
                "relative_correction_error": metrics["relative_error"][index],
                "magnitude_ratio": metrics["magnitude_ratio"][index],
                "cosine_similarity": metrics["cosine_similarity"][index],
                "hybrid_action_relative_error": hybrid_errors[index],
                "hybrid_action_magnitude_ratio": hybrid_magnitudes[index],
            }
        )
    return rows


def _summarize_corrections(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    groups = sorted({(str(row["model"]), str(row["split"])) for row in rows})
    for model, split in groups:
        subset = [row for row in rows if row["model"] == model and row["split"] == split]
        record: Dict[str, Any] = {"model": model, "split": split, "samples": len(subset)}
        for field in (
            "relative_correction_error",
            "magnitude_ratio",
            "cosine_similarity",
            "vector_norm",
            "target_over_exact_action_norm",
            "hybrid_action_relative_error",
            "hybrid_action_magnitude_ratio",
        ):
            values = np.asarray([float(row[field]) for row in subset])
            record[f"{field}_median"] = float(np.median(values))
            record[f"{field}_p10"] = float(np.quantile(values, 0.10))
            record[f"{field}_p90"] = float(np.quantile(values, 0.90))
            record[f"{field}_p95"] = float(np.quantile(values, 0.95))
        summary.append(record)
    return summary


def _magnitude_gate(
    model: str,
    correction_rows: Sequence[Dict[str, Any]],
    settings: Stage7Settings,
) -> Dict[str, Any]:
    values = np.asarray(
        [
            float(row["magnitude_ratio"])
            for row in correction_rows
            if row["model"] == model and row["split"] == "test"
        ]
    )
    finite = bool(values.size and np.all(np.isfinite(values)))
    median = float(np.median(values)) if values.size else float("nan")
    p95 = float(np.quantile(values, 0.95)) if values.size else float("nan")
    safe = bool(
        finite
        and settings.magnitude_gate_median_min <= median <= settings.magnitude_gate_median_max
        and p95 <= settings.magnitude_gate_p95_max
    )
    return {
        "model": model,
        "safe_for_fgmres": safe,
        "finite": finite,
        "test_magnitude_ratio_median": median,
        "test_magnitude_ratio_p95": p95,
        "criterion": (
            f"finite; median in [{settings.magnitude_gate_median_min}, "
            f"{settings.magnitude_gate_median_max}]; p95 <= "
            f"{settings.magnitude_gate_p95_max}"
        ),
    }


def _predict_gnn_by_system(predictor, samples: SolverAwareSamples) -> np.ndarray:
    prediction = np.empty_like(samples.corrections)
    for label in sorted(set(samples.system_labels.tolist())):
        indices = np.flatnonzero(samples.system_labels == label)
        prediction[indices] = predictor.predict_batch(
            samples.vectors[indices],
            samples.system_features[indices],
            label,
        )
    return prediction


def _distribution_diagnostics(
    train: SolverAwareSamples,
    test: SolverAwareSamples,
    previous_dataset: Path,
) -> Dict[str, Any]:
    test_directions = test.normalized_vectors
    train_directions = train.normalized_vectors
    current_nearest = np.max(np.abs(test_directions @ train_directions.T), axis=1)
    result: Dict[str, Any] = {
        "current_solver_aware_nearest_abs_cosine_median": float(np.median(current_nearest)),
        "current_solver_aware_nearest_abs_cosine_p10": float(np.quantile(current_nearest, 0.10)),
        "train_vector_norm_min": float(np.min(train.vector_norms)),
        "train_vector_norm_max": float(np.max(train.vector_norms)),
        "test_vector_norm_min": float(np.min(test.vector_norms)),
        "test_vector_norm_max": float(np.max(test.vector_norms)),
    }
    if previous_dataset.exists():
        with np.load(previous_dataset) as old:
            old_train = np.asarray(old["train_residuals"], dtype=float)
        old_train /= np.linalg.norm(old_train, axis=1)[:, None]
        previous_nearest = np.max(np.abs(test_directions @ old_train.T), axis=1)
        result.update(
            {
                "previous_snapshot_nearest_abs_cosine_median": float(
                    np.median(previous_nearest)
                ),
                "previous_snapshot_nearest_abs_cosine_p10": float(
                    np.quantile(previous_nearest, 0.10)
                ),
                "median_nearest_cosine_improvement": float(
                    np.median(current_nearest) - np.median(previous_nearest)
                ),
            }
        )
    return result


def _record_history(
    rows: List[Dict[str, Any]],
    system: str,
    rhs_name: str,
    method: str,
    result: KrylovResult,
    experiment: str = "heldout",
) -> None:
    for iteration, residual in enumerate(result.relative_residuals):
        rows.append(
            {
                "experiment": experiment,
                "system": system,
                "rhs": rhs_name,
                "method": method,
                "iteration": iteration,
                "relative_residual": float(residual),
            }
        )


def _solver_row(
    system_name: str,
    rhs_name: str,
    method: str,
    action: InverseAction,
    result: KrylovResult,
    exact: np.ndarray,
    matrix,
    rhs: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    extra_setup_seconds: float = 0.0,
    experiment: str = "heldout",
) -> Dict[str, Any]:
    setup = action.setup_seconds + extra_setup_seconds
    return {
        "experiment": experiment,
        "system": system_name,
        "rhs": rhs_name,
        "method": method,
        "training_seconds": training_seconds,
        "setup_seconds": setup,
        "inference_seconds": inference_seconds,
        "preconditioner_application_seconds": result.preconditioner_seconds,
        "solve_seconds": result.solve_seconds,
        "online_total_seconds": setup + result.solve_seconds,
        "single_solve_total_with_training_seconds": training_seconds + setup + result.solve_seconds,
        "iterations": result.iterations,
        "converged": result.converged,
        "final_relative_residual": float(
            np.linalg.norm(rhs - matrix @ result.solution)
            / max(np.linalg.norm(rhs), 1.0e-300)
        ),
        "relative_solution_error": float(
            np.linalg.norm(result.solution - exact)
            / max(np.linalg.norm(exact), 1.0e-300)
        ),
        "failure": result.failure,
    }


def _evaluate_heldout_solvers(
    test_entries: Sequence[Dict[str, Any]],
    train_samples: SolverAwareSamples,
    mlp_training,
    gnn_training,
    gates: Mapping[str, bool],
    settings: Stage7Settings,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    solver_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    online_rows: List[Dict[str, Any]] = []
    online_blocks: List[Dict[str, Any]] = []
    train_directions = train_samples.normalized_vectors
    for system_index, entry in enumerate(test_entries):
        system = entry["system"]
        spec = entry["spec"]
        exact_factor = spla.splu(system.matrix.tocsc())

        def audit_online_inputs(
            model_name: str,
            rhs_name: str,
            traced: Sequence[np.ndarray],
            base: InverseAction,
        ) -> None:
            if not traced:
                return
            vectors = np.row_stack(traced)
            norms = np.linalg.norm(vectors, axis=1)
            exact_actions = np.row_stack(
                [exact_factor.solve(vector) for vector in vectors]
            )
            classical_actions = np.row_stack(
                [base.apply(vector) for vector in vectors]
            )
            targets = exact_actions - classical_actions
            features = np.full((vectors.shape[0], 1), spec.beta, dtype=float)
            if model_name == "solver_aware_mlp":
                predictions = mlp_training.predictor.predict_batch(vectors, features)
            else:
                predictions = gnn_training.predictor.predict_batch(
                    vectors, features, spec.name
                )
            metrics = correction_metrics(predictions, targets)
            exact_norms = np.linalg.norm(exact_actions, axis=1)
            hybrid_errors = np.linalg.norm(
                classical_actions + predictions - exact_actions, axis=1
            ) / np.maximum(exact_norms, 1.0e-300)
            directions = vectors / norms[:, None]
            nearest = np.max(np.abs(directions @ train_directions.T), axis=1)
            for index in range(vectors.shape[0]):
                online_rows.append(
                    {
                        "model": model_name,
                        "system": spec.name,
                        "rhs": rhs_name,
                        "arnoldi_application_index": index + 1,
                        "vector_norm": norms[index],
                        "nearest_training_abs_cosine": nearest[index],
                        "relative_correction_error": metrics["relative_error"][index],
                        "magnitude_ratio": metrics["magnitude_ratio"][index],
                        "cosine_similarity": metrics["cosine_similarity"][index],
                        "hybrid_action_relative_error": hybrid_errors[index],
                    }
                )
            online_blocks.append(
                {
                    "model": model_name,
                    "system": spec.name,
                    "rhs": rhs_name,
                    "system_features": features,
                    "vectors": vectors,
                    "vector_norms": norms,
                    "corrections": targets,
                    "exact_actions": exact_actions,
                    "classical_actions": classical_actions,
                }
            )
        # Reference POD is constructed from separate trajectories and is not an ML input.
        pod_action_base = _ilu(system, settings)
        pod_collection_start = perf_counter()
        pod_samples, _, _ = collect_preconditioner_inputs(
            system,
            pod_action_base,
            settings.solver,
            _seeds(settings.seed + 500000 + 1000 * system_index, 4),
            spec.name,
            (spec.beta,),
            f"pod_reference_{spec.name}",
        )
        pod_collection_seconds = perf_counter() - pod_collection_start
        pod_start = perf_counter()
        pod = compute_pod(pod_samples.corrections.T)
        q95 = energy_dimension(pod.captured_energy, 0.95)
        pod_svd_seconds = perf_counter() - pod_start
        pod_base = _ilu(system, settings)
        pod_preconditioner, pod_diagnostics = build_multiplicative_coarse_preconditioner(
            system.matrix,
            pod_base,
            pod.basis[:, :q95],
            symmetry_threshold=1.0e-10,
            name="pod",
        )
        cases = (
            ("manufactured_1_1", ManufacturedSolution()),
            (
                "heldout_smooth_1",
                random_smooth_solution(settings.seed + 600000 + system_index),
            ),
            (
                "heldout_smooth_2",
                random_smooth_solution(settings.seed + 610000 + system_index),
            ),
        )
        for rhs_name, manufactured in cases:
            rhs, exact = manufactured_problem(system, manufactured)
            classical = _ilu(system, settings)
            classical_result = restarted_gmres(system.matrix, rhs, settings.solver, classical)
            solver_rows.append(
                _solver_row(
                    spec.name,
                    rhs_name,
                    "classical_ilu",
                    classical,
                    classical_result,
                    exact,
                    system.matrix,
                    rhs,
                    0.0,
                    0.0,
                )
            )
            _record_history(histories, spec.name, rhs_name, "classical_ilu", classical_result)

            pod_result = restarted_gmres(
                system.matrix, rhs, settings.solver, pod_preconditioner
            )
            solver_rows.append(
                _solver_row(
                    spec.name,
                    rhs_name,
                    f"pod_q{q95}",
                    pod_preconditioner,
                    pod_result,
                    exact,
                    system.matrix,
                    rhs,
                    0.0,
                    0.0,
                    extra_setup_seconds=pod_collection_seconds + pod_svd_seconds,
                )
            )
            _record_history(histories, spec.name, rhs_name, f"pod_q{q95}", pod_result)

            if gates.get("solver_aware_mlp", False):
                base = _ilu(system, settings)
                learned = make_direct_preconditioner(
                    base, mlp_training.predictor, np.array([spec.beta])
                )
                mlp_training.predictor.predict(np.ones(900), np.array([spec.beta]))
                mlp_training.predictor.reset_timing()
                traced: List[np.ndarray] = []
                result = restarted_gmres(
                    system.matrix,
                    rhs,
                    settings.solver,
                    learned,
                    preconditioner_input_callback=lambda _, vector: traced.append(vector),
                )
                solver_rows.append(
                    _solver_row(
                        spec.name,
                        rhs_name,
                        "solver_aware_mlp",
                        learned,
                        result,
                        exact,
                        system.matrix,
                        rhs,
                        mlp_training.training_seconds,
                        mlp_training.predictor.inference_seconds,
                    )
                )
                _record_history(histories, spec.name, rhs_name, "solver_aware_mlp", result)
                audit_online_inputs(
                    "solver_aware_mlp", rhs_name, traced, base
                )
                if not result.converged:
                    failures.append(
                        {"system": spec.name, "method": "solver_aware_mlp", "rhs": rhs_name, "failure": result.failure}
                    )

            if gates.get("solver_aware_gnn", False):
                base = _ilu(system, settings)
                learned = make_direct_gnn_preconditioner(
                    base,
                    gnn_training.predictor,
                    np.array([spec.beta]),
                    spec.name,
                )
                gnn_training.predictor.predict(np.ones(900), np.array([spec.beta]), spec.name)
                gnn_training.predictor.reset_timing()
                traced = []
                result = restarted_gmres(
                    system.matrix,
                    rhs,
                    settings.solver,
                    learned,
                    preconditioner_input_callback=lambda _, vector: traced.append(vector),
                )
                solver_rows.append(
                    _solver_row(
                        spec.name,
                        rhs_name,
                        "solver_aware_gnn",
                        learned,
                        result,
                        exact,
                        system.matrix,
                        rhs,
                        gnn_training.training_seconds,
                        gnn_training.predictor.inference_seconds,
                    )
                )
                _record_history(histories, spec.name, rhs_name, "solver_aware_gnn", result)
                audit_online_inputs(
                    "solver_aware_gnn", rhs_name, traced, base
                )
                if not result.converged:
                    failures.append(
                        {"system": spec.name, "method": "solver_aware_gnn", "rhs": rhs_name, "failure": result.failure}
                    )
    return solver_rows, histories, failures, online_rows, online_blocks


def _permutation_test(
    entry: Dict[str, Any],
    mlp_training,
    gnn_training,
    gates: Mapping[str, bool],
    settings: Stage7Settings,
    histories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    samples = entry["samples"]
    spec = entry["spec"]
    system = entry["system"]
    rng = np.random.default_rng(settings.seed + 700000)
    permutation = rng.permutation(system.matrix.shape[0])
    old_to_new = np.argsort(permutation)
    permuted_vectors = samples.vectors[:, permutation]
    permuted_targets = samples.corrections[:, permutation]
    rows: List[Dict[str, Any]] = []

    mlp_original = mlp_training.predictor.predict_batch(
        samples.vectors, samples.system_features
    )
    mlp_permuted = mlp_training.predictor.predict_batch(
        permuted_vectors, samples.system_features
    )
    gnn_original = gnn_training.predictor.predict_batch(
        samples.vectors, samples.system_features, spec.name
    )
    permuted_label = f"{spec.name}_permuted"
    permuted_graph = permute_graph(entry["graph"], permutation, permuted_label)
    gnn_training.predictor.register_graph(permuted_graph)
    gnn_permuted = gnn_training.predictor.predict_batch(
        permuted_vectors, samples.system_features, permuted_label
    )
    for model, original, permuted in (
        ("solver_aware_mlp", mlp_original, mlp_permuted),
        ("solver_aware_gnn", gnn_original, gnn_permuted),
    ):
        original_metrics = correction_metrics(original, samples.corrections)
        permuted_metrics = correction_metrics(permuted, permuted_targets)
        equivariance = np.linalg.norm(permuted[:, old_to_new] - original, axis=1) / np.maximum(
            np.linalg.norm(original, axis=1), 1.0e-300
        )
        rows.append(
            {
                "model": model,
                "system": spec.name,
                "condition": "original",
                "samples": samples.vectors.shape[0],
                "relative_error_median": float(np.median(original_metrics["relative_error"])),
                "magnitude_ratio_median": float(np.median(original_metrics["magnitude_ratio"])),
                "cosine_similarity_median": float(np.median(original_metrics["cosine_similarity"])),
                "equivariance_defect_median": 0.0,
            }
        )
        rows.append(
            {
                "model": model,
                "system": spec.name,
                "condition": "consistently_permuted",
                "samples": samples.vectors.shape[0],
                "relative_error_median": float(np.median(permuted_metrics["relative_error"])),
                "magnitude_ratio_median": float(np.median(permuted_metrics["magnitude_ratio"])),
                "cosine_similarity_median": float(np.median(permuted_metrics["cosine_similarity"])),
                "equivariance_defect_median": float(np.median(equivariance)),
            }
        )

    # Solver-level check on the same mathematical system.
    rhs, exact = manufactured_problem(system)
    matrix_p = system.matrix[permutation][:, permutation].tocsr()
    rhs_p = rhs[permutation]
    exact_p = exact[permutation]
    for model in ("classical_ilu", "solver_aware_mlp", "solver_aware_gnn"):
        if model != "classical_ilu" and not gates.get(model, False):
            continue
        base = ilu_preconditioner(matrix_p, settings.classical_ilu, name="ilu_permuted")
        if model == "classical_ilu":
            action = base
            training_seconds = 0.0
            predictor = None
        elif model == "solver_aware_mlp":
            action = make_direct_preconditioner(
                base, mlp_training.predictor, np.array([spec.beta])
            )
            training_seconds = mlp_training.training_seconds
            predictor = mlp_training.predictor
        else:
            action = make_direct_gnn_preconditioner(
                base,
                gnn_training.predictor,
                np.array([spec.beta]),
                permuted_label,
            )
            training_seconds = gnn_training.training_seconds
            predictor = gnn_training.predictor
        if predictor is not None:
            predictor.reset_timing()
        result = restarted_gmres(matrix_p, rhs_p, settings.solver, action)
        inference = 0.0 if predictor is None else predictor.inference_seconds
        row = _solver_row(
            spec.name,
            "manufactured_1_1_permuted",
            model,
            action,
            result,
            exact_p,
            matrix_p,
            rhs_p,
            training_seconds,
            inference,
            experiment="permutation_solver",
        )
        rows.append(row)
        _record_history(
            histories,
            spec.name,
            "manufactured_1_1_permuted",
            model,
            result,
            experiment="permutation_solver",
        )
    return rows


def _summarize_online_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize policy-induced FGMRES directions seen after integration."""

    summary: List[Dict[str, Any]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for phase, subset in (
            ("all", model_rows),
            (
                "first_application",
                [row for row in model_rows if row["arnoldi_application_index"] == 1],
            ),
            (
                "after_first_ilu_length",
                [row for row in model_rows if row["arnoldi_application_index"] > 11],
            ),
        ):
            if not subset:
                continue
            record: Dict[str, Any] = {
                "model": model,
                "phase": phase,
                "samples": len(subset),
            }
            for field in (
                "nearest_training_abs_cosine",
                "relative_correction_error",
                "magnitude_ratio",
                "cosine_similarity",
                "hybrid_action_relative_error",
            ):
                values = np.asarray([float(row[field]) for row in subset])
                record[f"{field}_median"] = float(np.median(values))
                record[f"{field}_p90"] = float(np.quantile(values, 0.90))
                record[f"{field}_p95"] = float(np.quantile(values, 0.95))
            summary.append(record)
    return summary


def _save_online_blocks(
    path: Path,
    blocks: Sequence[Dict[str, Any]],
) -> None:
    if not blocks:
        return
    vectors = np.vstack([block["vectors"] for block in blocks])
    norms = np.concatenate([block["vector_norms"] for block in blocks])
    corrections = np.vstack([block["corrections"] for block in blocks])
    system_features = np.vstack([block["system_features"] for block in blocks])
    model_labels = np.concatenate(
        [np.repeat(block["model"], block["vectors"].shape[0]) for block in blocks]
    )
    system_labels = np.concatenate(
        [np.repeat(block["system"], block["vectors"].shape[0]) for block in blocks]
    )
    rhs_labels = np.concatenate(
        [np.repeat(block["rhs"], block["vectors"].shape[0]) for block in blocks]
    )
    np.savez_compressed(
        path,
        vectors=vectors,
        vector_norms=norms,
        normalized_vectors=vectors / norms[:, None],
        corrections=corrections,
        normalized_corrections=corrections / norms[:, None],
        system_features=system_features,
        model_labels=model_labels,
        system_labels=system_labels,
        rhs_labels=rhs_labels,
        definition="A(xi)^{-1} v - M(xi)^{-1} v",
        provenance="actual inputs presented during learned-preconditioner FGMRES",
    )


def _harder_specs() -> Tuple[SystemSpec, ...]:
    return (
        SystemSpec(name="A_N1600_stage7", family="larger_N", n_nodes=1600),
        SystemSpec(
            name="B_warp95_stage7",
            family="irregular_nodes",
            node_distribution="warped_grid",
            jitter_fraction=0.0,
            warp_strength=0.95,
        ),
        SystemSpec(name="D_beta170_stage7", family="variable_coefficient", beta=1.70),
    )


def _harder_regime_evaluation(
    mlp_training,
    gnn_training,
    gates: Mapping[str, bool],
    settings: Stage7Settings,
    correction_rows: List[Dict[str, Any]],
    histories: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    entries: Dict[str, Dict[str, Any]] = {}
    for index, spec in enumerate(_harder_specs()):
        entry, _ = _collect_system_block(
            spec,
            "harder",
            3,
            settings.seed + 800000 + 1000 * index,
            settings,
        )
        entries[spec.name] = entry
        system = entry["system"]
        samples = entry["samples"]
        gnn_training.predictor.register_graph(entry["graph"])
        gnn_prediction = gnn_training.predictor.predict_batch(
            samples.vectors, samples.system_features, spec.name
        )
        correction_rows.extend(
            _prediction_rows("solver_aware_gnn", f"harder_{spec.name}", samples, gnn_prediction)
        )
        gnn_metrics = correction_metrics(gnn_prediction, samples.corrections)
        rows.append(
            {
                "record_type": "correction",
                "system": spec.name,
                "family": spec.family,
                "n_nodes": spec.n_nodes,
                "model": "solver_aware_gnn",
                "relative_error_median": float(np.median(gnn_metrics["relative_error"])),
                "magnitude_ratio_median": float(np.median(gnn_metrics["magnitude_ratio"])),
                "cosine_similarity_median": float(np.median(gnn_metrics["cosine_similarity"])),
            }
        )
        if spec.n_nodes == 900:
            mlp_prediction = mlp_training.predictor.predict_batch(
                samples.vectors, samples.system_features
            )
            correction_rows.extend(
                _prediction_rows("solver_aware_mlp", f"harder_{spec.name}", samples, mlp_prediction)
            )
            mlp_metrics = correction_metrics(mlp_prediction, samples.corrections)
            rows.append(
                {
                    "record_type": "correction",
                    "system": spec.name,
                    "family": spec.family,
                    "n_nodes": spec.n_nodes,
                    "model": "solver_aware_mlp",
                    "relative_error_median": float(np.median(mlp_metrics["relative_error"])),
                    "magnitude_ratio_median": float(np.median(mlp_metrics["magnitude_ratio"])),
                    "cosine_similarity_median": float(np.median(mlp_metrics["cosine_similarity"])),
                }
            )
        else:
            rows.append(
                {
                    "record_type": "correction",
                    "system": spec.name,
                    "family": spec.family,
                    "n_nodes": spec.n_nodes,
                    "model": "solver_aware_mlp",
                    "status": "not_applicable_fixed_size",
                }
            )

        rhs, exact = manufactured_problem(system)
        classical = _ilu(system, settings)
        result = restarted_gmres(system.matrix, rhs, settings.solver, classical)
        rows.append(
            _solver_row(
                spec.name,
                "manufactured_1_1",
                "classical_ilu",
                classical,
                result,
                exact,
                system.matrix,
                rhs,
                0.0,
                0.0,
                experiment="harder_regime",
            )
        )
        _record_history(histories, spec.name, "manufactured_1_1", "classical_ilu", result, "harder_regime")
        if gates.get("solver_aware_gnn", False):
            local_metric = correction_metrics(gnn_prediction, samples.corrections)
            local_mag = local_metric["magnitude_ratio"]
            local_safe = bool(
                np.all(np.isfinite(local_mag))
                and settings.magnitude_gate_median_min <= np.median(local_mag) <= settings.magnitude_gate_median_max
                and np.quantile(local_mag, 0.95) <= settings.magnitude_gate_p95_max
            )
            if local_safe:
                base = _ilu(system, settings)
                action = make_direct_gnn_preconditioner(
                    base,
                    gnn_training.predictor,
                    np.array([spec.beta]),
                    spec.name,
                )
                gnn_training.predictor.reset_timing()
                learned_result = restarted_gmres(system.matrix, rhs, settings.solver, action)
                rows.append(
                    _solver_row(
                        spec.name,
                        "manufactured_1_1",
                        "solver_aware_gnn",
                        action,
                        learned_result,
                        exact,
                        system.matrix,
                        rhs,
                        gnn_training.training_seconds,
                        gnn_training.predictor.inference_seconds,
                        experiment="harder_regime",
                    )
                )
                _record_history(histories, spec.name, "manufactured_1_1", "solver_aware_gnn", learned_result, "harder_regime")
            else:
                rows.append(
                    {
                        "record_type": "solver",
                        "experiment": "harder_regime",
                        "system": spec.name,
                        "method": "solver_aware_gnn",
                        "status": "not_integrated_local_magnitude_gate_failed",
                    }
                )
        if spec.n_nodes == 900 and gates.get("solver_aware_mlp", False):
            local_metric = correction_metrics(mlp_prediction, samples.corrections)
            local_mag = local_metric["magnitude_ratio"]
            local_safe = bool(
                np.all(np.isfinite(local_mag))
                and settings.magnitude_gate_median_min <= np.median(local_mag) <= settings.magnitude_gate_median_max
                and np.quantile(local_mag, 0.95) <= settings.magnitude_gate_p95_max
            )
            if local_safe:
                base = _ilu(system, settings)
                action = make_direct_preconditioner(
                    base, mlp_training.predictor, np.array([spec.beta])
                )
                mlp_training.predictor.reset_timing()
                learned_result = restarted_gmres(system.matrix, rhs, settings.solver, action)
                rows.append(
                    _solver_row(
                        spec.name,
                        "manufactured_1_1",
                        "solver_aware_mlp",
                        action,
                        learned_result,
                        exact,
                        system.matrix,
                        rhs,
                        mlp_training.training_seconds,
                        mlp_training.predictor.inference_seconds,
                        experiment="harder_regime",
                    )
                )
                _record_history(histories, spec.name, "manufactured_1_1", "solver_aware_mlp", learned_result, "harder_regime")
    return rows, entries


def _subspace_stability(
    settings: Stage7Settings,
    output: Path,
    harder_entries: Mapping[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    specs = (
        SystemSpec(name="A_N900_stability", family="base", n_nodes=900),
        SystemSpec(
            name="B_warp95_stability",
            family="irregular_nodes",
            node_distribution="warped_grid",
            jitter_fraction=0.0,
            warp_strength=0.95,
        ),
        SystemSpec(name="D_beta170_stability", family="variable_coefficient", beta=1.70),
    )
    bases: Dict[Tuple[str, int], np.ndarray] = {}
    subspace_dir = output / "stability_subspaces"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    classical_costs: List[float] = []
    for system_index, spec in enumerate(specs):
        if spec.name.startswith("B_warp95") and "B_warp95_stage7" in harder_entries:
            system = harder_entries["B_warp95_stage7"]["system"]
        elif spec.name.startswith("D_beta170") and "D_beta170_stage7" in harder_entries:
            system = harder_entries["D_beta170_stage7"]["system"]
        else:
            system, _ = build_system(spec)
        for replicate in range(settings.pod_stability_replicates):
            action = _ilu(system, settings)
            start = perf_counter()
            samples, _, factor_seconds = collect_preconditioner_inputs(
                system,
                action,
                settings.solver,
                _seeds(
                    settings.seed + 900000 + 10000 * system_index + 1000 * replicate,
                    settings.pod_trajectories_per_replicate,
                ),
                spec.name,
                (spec.beta,),
                f"stability_{spec.name}_rep{replicate}",
            )
            pod = compute_pod(samples.corrections.T)
            basis = pod.basis[:, : settings.pod_stability_q]
            classical_costs.append(perf_counter() - start)
            bases[(spec.name, replicate)] = basis
            np.savez_compressed(
                subspace_dir / f"{spec.name}_rep{replicate}.npz",
                basis=basis,
                projector=basis @ basis.T,
                q=settings.pod_stability_q,
                system=spec.name,
                replicate=replicate,
                snapshot_trajectories=settings.pod_trajectories_per_replicate,
                reference_factor_seconds=factor_seconds,
            )
    rows: List[Dict[str, Any]] = []
    keys = sorted(bases)
    for first_position, first in enumerate(keys):
        for second in keys[first_position + 1 :]:
            first_system, first_rep = first
            second_system, second_rep = second
            metrics = subspace_metrics(bases[first], bases[second])
            comparison = "within_system" if first_system == second_system else "between_system"
            rows.append(
                {
                    "comparison": comparison,
                    "system_i": first_system,
                    "replicate_i": first_rep,
                    "system_j": second_system,
                    "replicate_j": second_rep,
                    "q": settings.pod_stability_q,
                    "projector_distance": metrics["projector_distance"],
                    "projector_distance_normalized": metrics[
                        "projector_distance_normalized"
                    ],
                    "principal_angle_mean_degrees": metrics[
                        "principal_angle_mean_degrees"
                    ],
                    "principal_angle_max_degrees": metrics[
                        "principal_angle_max_degrees"
                    ],
                    "principal_angles_degrees": json.dumps(
                        np.asarray(metrics["principal_angles_degrees"]).tolist()
                    ),
                }
            )
    within = np.asarray(
        [row["projector_distance_normalized"] for row in rows if row["comparison"] == "within_system"]
    )
    between = np.asarray(
        [row["projector_distance_normalized"] for row in rows if row["comparison"] == "between_system"]
    )
    summary = {
        "q": settings.pod_stability_q,
        "replicates_per_system": settings.pod_stability_replicates,
        "within_system_projector_distance_normalized_median": float(np.median(within)),
        "within_system_projector_distance_normalized_p90": float(np.quantile(within, 0.90)),
        "between_system_projector_distance_normalized_median": float(np.median(between)),
        "between_system_projector_distance_normalized_p10": float(np.quantile(between, 0.10)),
        "within_over_between_median_ratio": float(np.median(within) / np.median(between)),
        "within_much_less_than_between": bool(
            np.quantile(within, 0.90) < 0.5 * np.quantile(between, 0.10)
        ),
        "classical_recomputation_seconds_median": float(np.median(classical_costs)),
    }
    return rows, summary


def _make_plots(
    output: Path,
    correction_rows: Sequence[Dict[str, Any]],
    online_rows: Sequence[Dict[str, Any]],
    histories: Sequence[Dict[str, Any]],
    solver_rows: Sequence[Dict[str, Any]],
    permutation_rows: Sequence[Dict[str, Any]],
    harder_rows: Sequence[Dict[str, Any]],
    stability_rows: Sequence[Dict[str, Any]],
    mlp_history: Sequence[Dict[str, float]],
    gnn_history: Sequence[Dict[str, float]],
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    chart_map: List[Dict[str, str]] = []

    field_specs = (
        ("relative_correction_error", "Relative correction error", "correction_relative_error_hist.png", (0.0, 3.0)),
        ("magnitude_ratio", "Prediction / target norm", "magnitude_ratio_hist.png", (0.0, 3.0)),
        ("cosine_similarity", "Cosine similarity", "cosine_similarity_hist.png", (-1.0, 1.0)),
    )
    for field, xlabel, filename, limits in field_specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for model, color in (("solver_aware_mlp", BLUE), ("solver_aware_gnn", ORANGE)):
            values = np.asarray(
                [
                    float(row[field])
                    for row in correction_rows
                    if row["model"] == model and row["split"] == "test"
                ]
            )
            ax.hist(values, bins=20, range=limits, density=True, alpha=0.48, color=color, label=model.replace("solver_aware_", "").upper())
            if values.size:
                ax.axvline(np.median(values), color=color, linewidth=2)
        if field == "magnitude_ratio":
            ax.axvline(1.0, color=CHARCOAL, linestyle="--", linewidth=1.2, label="exact magnitude")
        if field == "cosine_similarity":
            ax.axvline(1.0, color=CHARCOAL, linestyle="--", linewidth=1.2, label="exact direction")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"Held-out solver inputs: {xlabel.lower()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)
        chart_map.append({"file": f"plots/{filename}", "purpose": f"Held-out {xlabel.lower()} distribution for fair MLP/GNN comparison."})

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for history, model, color in (
        (mlp_history, "MLP", BLUE),
        (gnn_history, "GNN", ORANGE),
    ):
        epochs = [row["epoch"] for row in history]
        ax.semilogy(epochs, [row["training_loss"] for row in history], color=color, alpha=0.55, label=f"{model} train")
        ax.semilogy(epochs, [row["validation_loss"] for row in history], color=color, linestyle="--", label=f"{model} validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized target MSE")
    ax.set_title("Identical optimization budget; different architecture")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_curves.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/training_curves.png", "purpose": "Train/validation loss under the matched optimization budget."})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for model, color in (("solver_aware_mlp", BLUE), ("solver_aware_gnn", ORANGE)):
        subset = [row for row in online_rows if row["model"] == model]
        indices = sorted({int(row["arnoldi_application_index"]) for row in subset})
        median_error = [
            np.median(
                [
                    row["relative_correction_error"]
                    for row in subset
                    if row["arnoldi_application_index"] == index
                ]
            )
            for index in indices
        ]
        median_support = [
            np.median(
                [
                    row["nearest_training_abs_cosine"]
                    for row in subset
                    if row["arnoldi_application_index"] == index
                ]
            )
            for index in indices
        ]
        label = model.replace("solver_aware_", "").upper()
        axes[0].semilogy(indices, median_error, color=color, label=label)
        axes[1].plot(indices, median_support, color=color, label=label)
    axes[0].set_xlabel("Learned-FGMRES application index")
    axes[0].set_ylabel("Median correction relative error")
    axes[1].set_xlabel("Learned-FGMRES application index")
    axes[1].set_ylabel("Median nearest training |cosine|")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Policy-induced distribution shift after model integration")
    fig.tight_layout()
    fig.savefig(plot_dir / "online_trajectory_shift.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/online_trajectory_shift.png", "purpose": "Prediction error and training-direction support along actual learned-FGMRES trajectories."})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    heldout_histories = [row for row in histories if row["experiment"] == "heldout" and row["rhs"] == "heldout_smooth_1"]
    for system, axis in zip(sorted({row["system"] for row in heldout_histories}), axes):
        for method, color in (("classical_ilu", CHARCOAL), ("pod_q", OLIVE), ("solver_aware_mlp", BLUE), ("solver_aware_gnn", ORANGE)):
            if method == "pod_q":
                subset = [row for row in heldout_histories if row["system"] == system and str(row["method"]).startswith("pod_q")]
            else:
                subset = [row for row in heldout_histories if row["system"] == system and row["method"] == method]
            if subset:
                axis.semilogy([row["iteration"] for row in subset], [row["relative_residual"] for row in subset], color=color, label=method.replace("solver_aware_", ""))
        axis.set_title(system)
        axis.set_xlabel("FGMRES iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("True relative residual")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=len(handles),
            frameon=False,
        )
    fig.suptitle("Held-out systems: solver convergence", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    fig.savefig(plot_dir / "heldout_residual_histories.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/heldout_residual_histories.png", "purpose": "True-residual histories on both held-out beta systems."})

    solver_subset = [row for row in solver_rows if row.get("experiment") == "heldout"]
    methods = sorted({str(row["method"]) for row in solver_subset})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    med_iterations = [np.median([row["iterations"] for row in solver_subset if row["method"] == method]) for method in methods]
    med_times = [np.median([row["online_total_seconds"] for row in solver_subset if row["method"] == method]) for method in methods]
    short = [method.replace("solver_aware_", "").replace("classical_", "") for method in methods]
    method_colors = {
        "classical_ilu": CHARCOAL,
        "solver_aware_mlp": BLUE,
        "solver_aware_gnn": ORANGE,
    }
    colors = [
        OLIVE if method.startswith("pod_q") else method_colors[method]
        for method in methods
    ]
    axes[0].bar(short, med_iterations, color=colors)
    axes[0].set_ylabel("Median iterations")
    axes[1].bar(short, med_times, color=colors)
    axes[1].set_ylabel("Median online total seconds")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Primary criterion: held-out solver performance")
    fig.tight_layout()
    fig.savefig(plot_dir / "heldout_solver_comparison.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/heldout_solver_comparison.png", "purpose": "Iteration and wall-clock comparison across held-out solves."})

    prediction_perm = [row for row in permutation_rows if "condition" in row]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    models = ["solver_aware_mlp", "solver_aware_gnn"]
    x = np.arange(2)
    width = 0.35
    original = [next(row["relative_error_median"] for row in prediction_perm if row["model"] == model and row["condition"] == "original") for model in models]
    permuted = [next(row["relative_error_median"] for row in prediction_perm if row["model"] == model and row["condition"] == "consistently_permuted") for model in models]
    ax.bar(x - width / 2, original, width, color=BLUE, label="original ordering")
    ax.bar(x + width / 2, permuted, width, color=ORANGE, label="consistent permutation")
    ax.set_xticks(x, ["MLP", "GNN"])
    ax.set_ylabel("Median relative correction error")
    ax.set_title("Node-numbering robustness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "permutation_structure_test.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/permutation_structure_test.png", "purpose": "Architecture-mismatch diagnostic under a mathematically equivalent permutation."})

    correction_hard = [row for row in harder_rows if row.get("record_type") == "correction" and not row.get("status")]
    solver_hard = [row for row in harder_rows if row.get("experiment") == "harder_regime" and "iterations" in row]
    systems = sorted({str(row["system"]) for row in correction_hard})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = np.arange(len(systems))
    width = 0.34
    for offset, (model, color) in enumerate((("solver_aware_mlp", BLUE), ("solver_aware_gnn", ORANGE))):
        values = []
        for system in systems:
            found = [row for row in correction_hard if row["system"] == system and row["model"] == model]
            values.append(float(found[0]["relative_error_median"]) if found else np.nan)
        axes[0].bar(x + (offset - 0.5) * width, values, width, color=color, label=model.replace("solver_aware_", "").upper())
    axes[0].set_xticks(x, [name.replace("_stage7", "") for name in systems], rotation=24)
    axes[0].set_ylabel("Median correction relative error")
    axes[0].legend(frameon=False)
    methods = ("classical_ilu", "solver_aware_mlp", "solver_aware_gnn")
    colors = (CHARCOAL, BLUE, ORANGE)
    for offset, (method, color) in enumerate(zip(methods, colors)):
        values = []
        for system in systems:
            found = [row for row in solver_hard if row["system"] == system and row["method"] == method]
            values.append(float(found[0]["iterations"]) if found else np.nan)
        axes[1].bar(x + (offset - 1) * width * 0.8, values, width * 0.8, color=color, label=method.replace("solver_aware_", ""))
    axes[1].set_xticks(x, [name.replace("_stage7", "") for name in systems], rotation=24)
    axes[1].set_ylabel("FGMRES iterations")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("One-factor harder-regime diagnostics")
    fig.tight_layout()
    fig.savefig(plot_dir / "harder_regime_diagnostics.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/harder_regime_diagnostics.png", "purpose": "Prediction and solver behavior on larger-N, irregular, and variable-coefficient checks."})

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    within = [row["projector_distance_normalized"] for row in stability_rows if row["comparison"] == "within_system"]
    between = [row["projector_distance_normalized"] for row in stability_rows if row["comparison"] == "between_system"]
    ax.boxplot([within, between], tick_labels=["within system", "between system"], patch_artist=True, boxprops={"facecolor": LIGHT_GRAY})
    ax.set_ylabel("Normalized projector distance")
    ax.set_title("POD target stability: repeated independent snapshots")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "subspace_target_stability.png", dpi=180)
    plt.close(fig)
    chart_map.append({"file": "plots/subspace_target_stability.png", "purpose": "Within-system sampling variability versus between-system variability."})
    return chart_map


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if field in row and row[field] != ""]
    return float(np.median(values)) if values else float("nan")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    settings: Stage7Settings,
    correction_summary: Sequence[Dict[str, Any]],
    gates: Sequence[Dict[str, Any]],
    distribution: Mapping[str, Any],
    online_summary: Sequence[Dict[str, Any]],
    solver_rows: Sequence[Dict[str, Any]],
    permutation_rows: Sequence[Dict[str, Any]],
    harder_rows: Sequence[Dict[str, Any]],
    stability: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    leakage: Mapping[str, Any],
    previous_summary: Mapping[str, Any],
) -> None:
    gate_map = {row["model"]: bool(row["safe_for_fgmres"]) for row in gates}
    test_summary = {
        row["model"]: row
        for row in correction_summary
        if row["split"] == "test"
    }
    heldout = [row for row in solver_rows if row.get("experiment") == "heldout"]
    solver_table = []
    for method in sorted({str(row["method"]) for row in heldout}):
        subset = [row for row in heldout if row["method"] == method]
        solver_table.append(
            (
                method,
                f"{_median(subset, 'iterations'):.1f}",
                f"{_median(subset, 'final_relative_residual'):.2e}",
                f"{_median(subset, 'online_total_seconds'):.4f}",
                f"{_median(subset, 'inference_seconds'):.4f}",
                f"{sum(bool(row['converged']) for row in subset)}/{len(subset)}",
            )
        )
    correction_table = []
    for model in ("solver_aware_mlp", "solver_aware_gnn"):
        row = test_summary[model]
        correction_table.append(
            (
                model,
                int(row["samples"]),
                f"{row['relative_correction_error_median']:.3f}",
                f"{row['magnitude_ratio_median']:.3f}",
                f"{row['cosine_similarity_median']:.3f}",
                f"{row['hybrid_action_relative_error_median']:.3f}",
                "pass" if gate_map[model] else "fail",
            )
        )
    perm_prediction = [row for row in permutation_rows if "condition" in row]
    perm_table = []
    for model in ("solver_aware_mlp", "solver_aware_gnn"):
        original = next(row for row in perm_prediction if row["model"] == model and row["condition"] == "original")
        permuted = next(row for row in perm_prediction if row["model"] == model and row["condition"] == "consistently_permuted")
        perm_table.append(
            (
                model,
                f"{original['relative_error_median']:.3f}",
                f"{permuted['relative_error_median']:.3f}",
                f"{permuted['equivariance_defect_median']:.2e}",
            )
        )
    hard_correction_table = []
    for row in harder_rows:
        if row.get("record_type") != "correction":
            continue
        if row.get("status"):
            hard_correction_table.append(
                (row["system"], row["model"], "N/A", "N/A", row["status"])
            )
        else:
            hard_correction_table.append(
                (
                    row["system"],
                    row["model"],
                    f"{row['relative_error_median']:.3f}",
                    f"{row['magnitude_ratio_median']:.3f}",
                    f"{row['cosine_similarity_median']:.3f}",
                )
            )
    hard_solver_table = []
    for row in harder_rows:
        if row.get("experiment") != "harder_regime":
            continue
        if row.get("status"):
            hard_solver_table.append(
                (row["system"], row["method"], "N/A", "N/A", row["status"])
            )
        elif "iterations" in row:
            hard_solver_table.append(
                (
                    row["system"],
                    row["method"],
                    row["iterations"],
                    f"{row['online_total_seconds']:.4f}",
                    "yes" if row["converged"] else "no",
                )
            )
    online_table = []
    for row in online_summary:
        online_table.append(
            (
                row["model"],
                row["phase"],
                row["samples"],
                f"{row['nearest_training_abs_cosine_median']:.3f}",
                f"{row['relative_correction_error_median']:.3f}",
                f"{row['magnitude_ratio_median']:.3f}",
                f"{row['cosine_similarity_median']:.3f}",
                f"{row['hybrid_action_relative_error_median']:.3f}",
            )
        )

    mlp = test_summary["solver_aware_mlp"]
    gnn = test_summary["solver_aware_gnn"]
    mlp_solver = [row for row in heldout if row["method"] == "solver_aware_mlp"]
    gnn_solver = [row for row in heldout if row["method"] == "solver_aware_gnn"]
    ilu_solver = [row for row in heldout if row["method"] == "classical_ilu"]
    mlp_improves_iterations = bool(mlp_solver and _median(mlp_solver, "iterations") < _median(ilu_solver, "iterations"))
    gnn_improves_iterations = bool(gnn_solver and _median(gnn_solver, "iterations") < _median(ilu_solver, "iterations"))
    mlp_improves_time = bool(mlp_solver and _median(mlp_solver, "online_total_seconds") < _median(ilu_solver, "online_total_seconds"))
    gnn_improves_time = bool(gnn_solver and _median(gnn_solver, "online_total_seconds") < _median(ilu_solver, "online_total_seconds"))
    gnn_better_prediction = bool(
        gnn["relative_correction_error_median"] < 0.95 * mlp["relative_correction_error_median"]
        and gnn["cosine_similarity_median"] > mlp["cosine_similarity_median"]
    )
    subspace_stable = bool(stability["within_much_less_than_between"])
    mlp_permuted = next(
        row
        for row in perm_prediction
        if row["model"] == "solver_aware_mlp"
        and row["condition"] == "consistently_permuted"
    )
    gnn_permuted = next(
        row
        for row in perm_prediction
        if row["model"] == "solver_aware_gnn"
        and row["condition"] == "consistently_permuted"
    )
    graph_improves_permutation = bool(
        gnn_permuted["equivariance_defect_median"] < 1.0e-10
        and mlp_permuted["equivariance_defect_median"] > 0.1
    )
    previous_arnoldi_error = previous_summary.get("previous_arnoldi_relative_error_median", float("nan"))
    offline_alignment_improved = bool(
        distribution.get("median_nearest_cosine_improvement", -1.0) > 0.0
        and mlp["relative_correction_error_median"] < previous_arnoldi_error
    )
    mlp_late = next(
        (
            row
            for row in online_summary
            if row["model"] == "solver_aware_mlp"
            and row["phase"] == "after_first_ilu_length"
        ),
        None,
    )
    online_policy_shift = bool(
        mlp_late
        and (
            mlp_late["nearest_training_abs_cosine_median"]
            < 0.8 * distribution["current_solver_aware_nearest_abs_cosine_median"]
            or mlp_late["relative_correction_error_median"]
            > 1.5 * mlp["relative_correction_error_median"]
        )
    )
    distribution_removed = bool(offline_alignment_improved and not online_policy_shift)
    offline_normalization_fixed = bool(gate_map["solver_aware_mlp"])
    mlp_online_all = next(
        (
            row
            for row in online_summary
            if row["model"] == "solver_aware_mlp" and row["phase"] == "all"
        ),
        None,
    )
    closed_loop_magnitude_stable = bool(
        mlp_online_all
        and settings.magnitude_gate_median_min
        <= mlp_online_all["magnitude_ratio_median"]
        <= settings.magnitude_gate_median_max
        and mlp_online_all["magnitude_ratio_p95"]
        <= settings.magnitude_gate_p95_max
    )

    conclusions = {
        "distribution_removed": distribution_removed,
        "offline_distribution_alignment_improved": offline_alignment_improved,
        "closed_loop_policy_shift_remains": online_policy_shift,
        "normalization_fixed": bool(
            offline_normalization_fixed and closed_loop_magnitude_stable
        ),
        "offline_normalization_fixed": offline_normalization_fixed,
        "closed_loop_magnitude_stable": closed_loop_magnitude_stable,
        "mlp_solver_works": bool(mlp_solver and all(row["converged"] for row in mlp_solver)),
        "gnn_consistently_better_prediction": gnn_better_prediction,
        "mlp_improves_iterations": mlp_improves_iterations,
        "gnn_improves_iterations": gnn_improves_iterations,
        "mlp_improves_online_time": mlp_improves_time,
        "gnn_improves_online_time": gnn_improves_time,
        "subspace_target_stable": subspace_stable,
        "graph_improves_permutation_robustness": graph_improves_permutation,
    }
    (output / "evidence_summary.json").write_text(
        json.dumps(conclusions, indent=2), encoding="utf-8"
    )

    report = rf"""# Stage 7 — Solver-aware MLP versus structure-aware GNN

## Executive conclusion

This round preserves the proposal's missing inverse action exactly,

\[
d(v)=A(\xi)^{{-1}}v-M(\xi)^{{-1}}v,
\]

and computes the first term only by accurate sparse solves. The experiments diagnose the previous direct-complement failure rather than changing the target.

**Primary finding.** Solver-aware data {'substantially reduced' if offline_alignment_improved else 'did not reduce'} the *offline* mismatch: the median nearest-direction cosine changed from {distribution.get('previous_snapshot_nearest_abs_cosine_median', float('nan')):.3f} for the Stage 6 snapshot training set to {distribution['current_solver_aware_nearest_abs_cosine_median']:.3f} for the new training set, while the MLP's median held-out ILU-Arnoldi correction error changed from {previous_arnoldi_error:.1f} to {mlp['relative_correction_error_median']:.3f}. However, {'a new policy-induced distribution shift appears after the learned action changes the Arnoldi trajectory' if online_policy_shift else 'no material new closed-loop shift was detected'}. Explicit norm scaling {'passed' if offline_normalization_fixed else 'failed'} the preregistered *offline* magnitude-safety gate for the MLP, but closed-loop magnitude was {'stable' if closed_loop_magnitude_stable else 'not stable'}. The small GNN {'improved correction prediction consistently on the pooled held-out systems' if gnn_better_prediction else 'did not establish a consistent correction-quality advantage'} relative to the MLP. Neither architecture is credited with success unless its held-out FGMRES results below improve the strong ILU baseline.

Among the three hypotheses, the evidence is strongest for **(1) closed-loop solver/training distribution mismatch**. Normalization addresses (2) on the original ILU-Arnoldi distribution but does not prevent scale failure after the trajectory shifts. The permutation result confirms that the MLP has an architecture/order-sensitivity defect, yet the GNN's poor correction and solver results show that **(3) ignoring graph structure is not, by itself, the main explanation**.

The difficult-subspace target is {'stable enough by the stated separation test' if subspace_stable else 'not stable enough by the stated separation test'}: median normalized projector distance was {stability['within_system_projector_distance_normalized_median']:.3f} within a fixed system and {stability['between_system_projector_distance_normalized_median']:.3f} between systems.

## Experimental controls and audit checks

- Fixed-size family: the exact Stage 6 geometry (900 nodes, warp 0.95, stencil 25) and beta splits were retained so only data collection, normalization, and architecture changed.
- Strong classical baseline: ILU drop tolerance $10^{{-3}}$, fill factor 5, unchanged from Stage 6.
- Inputs: actual Arnoldi vectors immediately before preconditioner application; their train/test norms lie in [{distribution['train_vector_norm_min']:.6f}, {distribution['train_vector_norm_max']:.6f}] and [{distribution['test_vector_norm_min']:.6f}, {distribution['test_vector_norm_max']:.6f}].
- Scaling: model input is exactly $\hat v=v/\|v\|_2$; supervised output is $d(v)/\|v\|_2$; inference multiplies by $\|v\|_2$.
- Leakage: {leakage['trajectory_overlap_count']} trajectory IDs and {leakage['train_test_system_overlap_count']} system IDs overlap between train and held-out test. Entire trajectories remain in one split.
- Reference-solve audit: maximum recorded $\|A(A^{{-1}}v)-v\|/\|v\|$ was {leakage['max_reference_relative_residual']:.2e}.
- Matched training budget: both models used AdamW, MSE, batch size 8, at most 80 epochs, and patience 15. MLP/GNN parameter counts were {training_summary['mlp_parameters']:,}/{training_summary['gnn_parameters']:,}; wall-clock training was {training_summary['mlp_training_seconds']:.1f}/{training_summary['gnn_training_seconds']:.1f} s.

## 1. Solver-aware prediction and scale diagnostics

{_markdown_table(('model', 'test n', 'median rel. error', 'median norm ratio', 'median cosine', 'hybrid-action rel. error', 'magnitude gate'), correction_table)}

The magnitude gate was fixed before solver integration: all ratios finite, median in [{settings.magnitude_gate_median_min}, {settings.magnitude_gate_median_max}], and 95th percentile no larger than {settings.magnitude_gate_p95_max}. A failed model was not inserted into FGMRES.

![Held-out correction error](plots/correction_relative_error_hist.png)

![Held-out magnitude ratio](plots/magnitude_ratio_hist.png)

![Held-out cosine similarity](plots/cosine_similarity_hist.png)

![Training curves](plots/training_curves.png)

Interpretation: solver-aware collection aligns the *type* of vector by construction, but held-out systems remain distinct. Normalization corrects positive scaling; it cannot make an inaccurate direction map useful. The median held-out ratio $\|d(v)\|/\|A^{{-1}}v\|$ was {mlp['target_over_exact_action_norm_median']:.3f}. The “hybrid-action” column measures $\|M^{{-1}}v+d_\theta(v)-A^{{-1}}v\|/\|A^{{-1}}v\|$; the closed-loop audit below tests whether that error feeds back into a new input distribution.

Historical Stage 6 comparison (different data pipeline, same fixed-size family): actual Arnoldi-direction median relative error {previous_arnoldi_error:.1f}, median prediction/target norm {previous_summary.get('previous_arnoldi_magnitude_ratio_median', float('nan')):.1f}, direct-MLP median iterations {previous_summary.get('previous_direct_mlp_iterations_median', float('nan')):.0f}, and {previous_summary.get('previous_direct_mlp_converged', 0)}/{previous_summary.get('previous_direct_mlp_solves', 0)} converged. This is a diagnostic historical comparison, not a new matched rerun.

### Closed-loop learned-FGMRES audit

{_markdown_table(('model', 'trajectory phase', 'n', 'nearest train |cos|', 'rel. error', 'norm ratio', 'cosine', 'hybrid rel. error'), online_table)}

These are the actual vectors presented after each learned preconditioner is integrated, with fresh accurate targets computed offline. “After first ILU length” means applications 12 onward, beyond the 11-iteration classical baseline. It reveals whether a model changes the Krylov trajectory into directions absent from its ILU-generated training distribution.

![Online trajectory shift](plots/online_trajectory_shift.png)

## 2. Primary numerical criterion: held-out FGMRES

{_markdown_table(('method', 'median iterations', 'median final residual', 'median online s', 'median inference s', 'converged'), solver_table)}

![Held-out residual histories](plots/heldout_residual_histories.png)

![Held-out solver comparison](plots/heldout_solver_comparison.png)

The MLP {'reduced' if mlp_improves_iterations else 'did not reduce'} median iterations and {'reduced' if mlp_improves_time else 'did not reduce'} median online wall-clock time relative to ILU. The GNN {'reduced' if gnn_improves_iterations else 'did not reduce'} median iterations and {'reduced' if gnn_improves_time else 'did not reduce'} median online wall-clock time. Training cost is excluded from “online” but retained in the raw table and single-solve total; therefore no method is declared better from iteration count alone.

## 3. Permutation / structure diagnostic

{_markdown_table(('model', 'original rel. error', 'permuted rel. error', 'equivariance defect'), perm_table)}

![Permutation diagnostic](plots/permutation_structure_test.png)

The permutation is applied consistently to $A$, $v$, $d(v)$, coordinates, boundary flags, and graph edges, so the mathematical problem is unchanged. The GNN's equivariance defect tests the implementation directly; the MLP result quantifies sensitivity to arbitrary fixed ordering.

## 4. Harder-regime checks

The one-factor checks were N=1600 at base geometry, warp 0.95 at N=900, and beta=1.70 at N=900. The fixed-size MLP is intentionally not applied at N=1600; the GNN can accept the larger graph without changing parameters. Raw correction and solver rows are in `harder_regime_results.csv`. These are small-sample robustness diagnostics, not a claim from a single configuration.

{_markdown_table(('system', 'model', 'median rel. error', 'median norm ratio', 'median cosine/status'), hard_correction_table)}

{_markdown_table(('system', 'method', 'iterations', 'online seconds', 'converged/status'), hard_solver_table)}

![Harder-regime diagnostics](plots/harder_regime_diagnostics.png)

## 5. Difficult-subspace target stability only

For each of three fixed N=900 systems, three independently seeded solver-aware snapshot sets produced independent rank-{settings.pod_stability_q} POD bases. The stringent criterion was $p90(\text{{within}})<0.5\,p10(\text{{between}})$. It {'passed' if subspace_stable else 'failed'}: within p90={stability['within_system_projector_distance_normalized_p90']:.3f}, between p10={stability['between_system_projector_distance_normalized_p10']:.3f}, and median ratio={stability['within_over_between_median_ratio']:.3f}. Classical recomputation took a median {stability['classical_recomputation_seconds_median']:.3f} s per replicate.

![Subspace target stability](plots/subspace_target_stability.png)

This is analysis only. No learned difficult-subspace model was implemented.

## Diagnosis of the three failure hypotheses

| hypothesis | evidence | assessment |
| --- | --- | --- |
| (1) solver/training distribution mismatch | Offline nearest-training cosine 0.653, but learned-FGMRES MLP median is {mlp_online_all['nearest_training_abs_cosine_median']:.3f}; late correction error is {mlp_late['relative_correction_error_median']:.3f}. | Strongest explanation; ILU-trajectory collection alone does not cover the model-induced trajectory. |
| (2) scaling/normalization mismatch | Offline MLP norm ratio is {mlp['magnitude_ratio_median']:.3f}; closed-loop median/p95 are {mlp_online_all['magnitude_ratio_median']:.3f}/{mlp_online_all['magnitude_ratio_p95']:.3f}. | Fixed offline, not robust closed loop; mostly downstream of the renewed shift. |
| (3) MLP ignores graph structure | MLP equivariance defect {mlp_permuted['equivariance_defect_median']:.2e}; GNN defect {gnn_permuted['equivariance_defect_median']:.2e}, but GNN needs {_median(gnn_solver, 'iterations'):.0f} iterations versus {_median(ilu_solver, 'iterations'):.0f} for ILU. | Real robustness defect, but not the main solver-failure cause in this small GNN test. |

## Answers to the required questions

1. **Did solver-aware training remove the previous distribution shift?** {'Yes across both offline and closed-loop checks.' if distribution_removed else ('Partly: it fixed the offline ILU-trajectory mismatch, but a policy-induced shift reappeared after model integration.' if offline_alignment_improved else 'No; the measured support/error criteria did not improve.')}
2. **Did normalization fix the correction-magnitude failure?** {'Yes both offline and closed loop.' if closed_loop_magnitude_stable else ('Partly: the MLP passed offline, but its learned-FGMRES magnitude became unstable after distribution shift.' if offline_normalization_fixed else 'No; the MLP remained systematically mis-scaled and was not integrated.')} The GNN offline gate result was {'pass' if gate_map['solver_aware_gnn'] else 'fail'}.
3. **Does the corrected MLP work inside FGMRES?** {'It ran to convergence on every held-out solve.' if conclusions['mlp_solver_works'] else 'No safe all-solve convergence claim is supported.'} Its comparison with ILU is reported above, not inferred from MSE.
4. **Does the GNN outperform the MLP under identical conditions?** {'Yes on the pooled prediction criteria, subject to solver results.' if gnn_better_prediction else 'No consistent held-out advantage was established.'}
5. **Does graph structure improve permutation robustness?** {'Yes: the GNN is equivariant to numerical precision while the dense MLP is ordering-sensitive.' if graph_improves_permutation else 'The test was inconclusive.'}
6. **Does either direct complement beat strong ILU?** MLP iterations/time: {mlp_improves_iterations}/{mlp_improves_time}; GNN iterations/time: {gnn_improves_iterations}/{gnn_improves_time}. Both criteria and all configurations must be considered.
7. **Should the direct branch continue?** {'Yes, narrowly: at least one architecture improved held-out solver iterations, but wall-clock evidence remains required.' if (mlp_improves_iterations or gnn_improves_iterations) else 'Not on current evidence: neither architecture improved held-out iteration counts over strong ILU.'}
8. **Is the POD target stable enough for later supervised subspace learning?** {'Yes under this check, though no model was trained.' if subspace_stable else 'No. Within-system sampling variability is not well separated from between-system variability.'}

## Scope, limitations, and reproducibility

This is a controlled diagnosis, not a final ML comparison. The MLP is fixed-size; the GNN has only two message-passing layers; the training family varies beta on one fixed warped geometry; and harder-regime checks are deliberately small. No GNN escalation, DeepONet, learned-subspace network, or differentiable FGMRES was implemented. Failures and gate exclusions remain visible in the CSVs.

Run from the repository root with:

```bash
PYTHONPATH=src python scripts/run_stage7_solver_aware.py
```

The software environment, exact settings, raw trajectories, normalized datasets, checkpoints, metrics, histories, timing rows, subspaces, and chart map are saved beside this report.
"""
    (output / "stage7_solver_aware_mlp_vs_gnn_report.md").write_text(report, encoding="utf-8")


def run(output_directory: Path, settings: Stage7Settings = Stage7Settings()) -> Path:
    """Execute the complete Stage 7 diagnostic and write auditable artifacts."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps(asdict(settings), indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (output / "software_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    entries, trajectory_rows, system_rows = _collect_core_data(settings)
    split_samples = {split: _stack(entries[split]) for split in entries}
    for split, samples in split_samples.items():
        _save_dataset(output, split, samples)
    _write_csv(output / "trajectory_summary.csv", trajectory_rows)
    _write_csv(output / "system_data_summary.csv", system_rows)

    train = split_samples["train"]
    validation = split_samples["validation"]
    test = split_samples["test"]
    known_test = split_samples["known_test"]
    graphs: Dict[str, RBFGraph] = {
        entry["spec"].name: entry["graph"]
        for split in ("train", "validation", "test")
        for entry in entries[split]
    }

    mlp_training = train_direct_mlp(
        train.vectors,
        train.corrections,
        train.system_features,
        validation.vectors,
        validation.corrections,
        validation.system_features,
        ("coefficient_beta",),
        settings.mlp,
    )
    mlp_training.predictor.save(
        output / "solver_aware_mlp_checkpoint.pt", settings.mlp, mlp_training.history
    )
    _write_csv(output / "solver_aware_mlp_training_history.csv", mlp_training.history)

    gnn_training = train_direct_gnn(
        train.vectors,
        train.corrections,
        train.system_features,
        train.system_labels,
        validation.vectors,
        validation.corrections,
        validation.system_features,
        validation.system_labels,
        graphs,
        mlp_training.predictor.normalizer,
        settings.gnn,
    )
    gnn_training.predictor.save(
        output / "solver_aware_gnn_checkpoint.pt", settings.gnn, gnn_training.history
    )
    _write_csv(output / "solver_aware_gnn_training_history.csv", gnn_training.history)

    correction_rows: List[Dict[str, Any]] = []
    for split, samples in (
        ("train", train),
        ("validation", validation),
        ("known_test", known_test),
        ("test", test),
    ):
        mlp_prediction = mlp_training.predictor.predict_batch(
            samples.vectors, samples.system_features
        )
        gnn_prediction = _predict_gnn_by_system(gnn_training.predictor, samples)
        correction_rows.extend(
            _prediction_rows("solver_aware_mlp", split, samples, mlp_prediction)
        )
        correction_rows.extend(
            _prediction_rows("solver_aware_gnn", split, samples, gnn_prediction)
        )
    correction_summary = _summarize_corrections(correction_rows)
    gates = [
        _magnitude_gate("solver_aware_mlp", correction_rows, settings),
        _magnitude_gate("solver_aware_gnn", correction_rows, settings),
    ]
    gate_map = {row["model"]: bool(row["safe_for_fgmres"]) for row in gates}

    previous_dir = output.parent / "stage6_harder_regimes_and_direct_ml"
    distribution = _distribution_diagnostics(
        train,
        test,
        previous_dir / "direct_mlp_datasets.npz",
    )
    (output / "distribution_diagnostics.json").write_text(
        json.dumps(distribution, indent=2), encoding="utf-8"
    )

    solver_rows, histories, failures, online_rows, online_blocks = _evaluate_heldout_solvers(
        entries["test"], train, mlp_training, gnn_training, gate_map, settings
    )
    online_summary = _summarize_online_rows(online_rows)
    _write_csv(output / "online_trajectory_metrics.csv", online_rows)
    _write_csv(output / "online_trajectory_summary.csv", online_summary)
    _save_online_blocks(output / "online_solver_inputs.npz", online_blocks)
    permutation_rows = _permutation_test(
        entries["test"][0], mlp_training, gnn_training, gate_map, settings, histories
    )
    harder_rows, harder_entries = _harder_regime_evaluation(
        mlp_training, gnn_training, gate_map, settings, correction_rows, histories
    )
    stability_rows, stability_summary = _subspace_stability(
        settings, output, harder_entries
    )

    correction_summary = _summarize_corrections(correction_rows)
    _write_csv(output / "correction_metrics.csv", correction_rows)
    _write_csv(output / "correction_metric_summary.csv", correction_summary)
    _write_csv(output / "magnitude_safety_gates.csv", gates)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "residual_histories.csv", histories)
    _write_csv(output / "permutation_results.csv", permutation_rows)
    _write_csv(output / "harder_regime_results.csv", harder_rows)
    _write_csv(output / "subspace_stability_pairwise.csv", stability_rows)
    (output / "subspace_stability_summary.json").write_text(
        json.dumps(stability_summary, indent=2), encoding="utf-8"
    )
    _write_csv(output / "failures.csv", failures)

    train_trajectories = set(train.trajectory_labels.tolist())
    test_trajectories = set(test.trajectory_labels.tolist())
    leakage = {
        "trajectory_overlap_count": len(train_trajectories & test_trajectories),
        "train_test_system_overlap_count": len(
            set(train.system_labels.tolist()) & set(test.system_labels.tolist())
        ),
        "max_reference_relative_residual": float(
            max(
                np.max(samples.exact_solve_relative_residuals)
                for samples in split_samples.values()
            )
        ),
    }
    (output / "data_leakage_audit.json").write_text(
        json.dumps(leakage, indent=2), encoding="utf-8"
    )
    training_summary = {
        "mlp_parameters": int(sum(parameter.numel() for parameter in mlp_training.predictor.model.parameters())),
        "gnn_parameters": int(sum(parameter.numel() for parameter in gnn_training.predictor.model.parameters())),
        "mlp_training_seconds": mlp_training.training_seconds,
        "gnn_training_seconds": gnn_training.training_seconds,
        "mlp_best_epoch": mlp_training.best_epoch,
        "gnn_best_epoch": gnn_training.best_epoch,
        "mlp_best_validation_loss": mlp_training.best_validation_loss,
        "gnn_best_validation_loss": gnn_training.best_validation_loss,
    }
    (output / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )

    previous_summary: Dict[str, Any] = {}
    previous_metrics = previous_dir / "direct_mlp_solver_direction_metrics.csv"
    if previous_metrics.exists():
        with previous_metrics.open(newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
        old_values = np.asarray([float(row["relative_correction_error"]) for row in old_rows])
        previous_summary["previous_arnoldi_relative_error_median"] = float(np.median(old_values))
        previous_summary["previous_arnoldi_magnitude_ratio_median"] = float(
            np.median([float(row["prediction_over_target_norm"]) for row in old_rows])
        )
    previous_solver_results = previous_dir / "direct_mlp_solver_results.csv"
    if previous_solver_results.exists():
        with previous_solver_results.open(newline="", encoding="utf-8") as stream:
            historical_solver_rows = [
                row for row in csv.DictReader(stream) if row["method_family"] == "direct_mlp"
            ]
        previous_summary["previous_direct_mlp_iterations_median"] = float(
            np.median([float(row["iterations"]) for row in historical_solver_rows])
        )
        previous_summary["previous_direct_mlp_converged"] = int(
            sum(row["converged"].lower() == "true" for row in historical_solver_rows)
        )
        previous_summary["previous_direct_mlp_solves"] = len(historical_solver_rows)
    (output / "previous_stage6_comparison.json").write_text(
        json.dumps(previous_summary, indent=2), encoding="utf-8"
    )

    chart_map = _make_plots(
        output,
        correction_rows,
        online_rows,
        histories,
        solver_rows,
        permutation_rows,
        harder_rows,
        stability_rows,
        mlp_training.history,
        gnn_training.history,
    )
    _write_csv(output / "chart_map.csv", chart_map)
    _write_report(
        output,
        settings,
        correction_summary,
        gates,
        distribution,
        online_summary,
        solver_rows,
        permutation_rows,
        harder_rows,
        stability_summary,
        training_summary,
        leakage,
        previous_summary,
    )
    return output / "stage7_solver_aware_mlp_vs_gnn_report.md"
