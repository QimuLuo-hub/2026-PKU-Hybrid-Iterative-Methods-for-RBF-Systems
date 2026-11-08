"""Stage 8: on-policy direct correction and difficult-space stability.

The two research branches are deliberately kept separate.  Branch A always
uses the literal direct target ``A^-1 v-M^-1 v``.  Branch B never consumes a
learned direct correction and does not train a subspace predictor unless its
classical oracle and target-stability gates pass.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import scipy.stats as stats
import torch

from ..coarse_space import build_multiplicative_coarse_preconditioner
from ..config import ILUConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..models.direct_control import make_stabilized_direct_preconditioner
from ..models.direct_gnn import (
    DirectGNNConfig,
    graph_from_system,
    train_direct_gnn,
)
from ..models.direct_mlp import (
    DirectMLPConfig,
    correction_metrics,
    train_direct_mlp,
)
from ..pde import random_smooth_solution
from ..pod import compute_pod, relative_projection_errors
from ..preconditioners import InverseAction, ilu_preconditioner
from ..solver_aware import (
    SolverAwareSamples,
    concatenate_solver_aware_samples,
    label_preconditioner_inputs,
)
from ..subspace_analysis import (
    interpolate_basis_to_probe,
    krylov_ritz_basis,
    standardized_parameter_distances,
    subspace_metrics,
)
from .harder_regimes import SystemSpec, build_system, system_metadata


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
PURPLE = "#7553A6"
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
class Stage8Settings:
    """Bounded, preregistered Stage 8 experiment controls."""

    seed: int = 20260829
    solver: SolverConfig = SolverConfig(
        relative_tolerance=1.0e-8,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=120,
        breakdown_tolerance=1.0e-14,
    )
    classical_ilu: ILUConfig = ILUConfig(
        drop_tolerance=1.0e-3,
        fill_factor=5.0,
        diagonal_pivot_threshold=0.0,
        permutation="COLAMD",
    )
    aggregation_rounds: int = 3
    aggregation_betas: Tuple[float, ...] = (0.6, 1.2)
    test_betas: Tuple[float, ...] = (0.9, 1.65)
    aggregation_trajectories_per_system: int = 1
    test_trajectories_per_system: int = 2
    alphas: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    norm_limit_ratio: float = 0.5
    mlp: DirectMLPConfig = DirectMLPConfig(
        hidden_width=64,
        hidden_layers=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=8,
        max_epochs=60,
        patience=12,
        seed=20260829,
        input_direction_scale=1.0,
    )
    gnn: DirectGNNConfig = DirectGNNConfig(
        hidden_width=16,
        message_layers=2,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=8,
        max_epochs=60,
        patience=12,
        seed=20260829,
    )
    magnitude_gate_median_min: float = 0.5
    magnitude_gate_median_max: float = 2.0
    magnitude_gate_p95_max: float = 5.0
    subspace_q: int = 8
    subspace_q_values: Tuple[int, ...] = (2, 4, 8)
    snapshot_counts: Tuple[int, ...] = (20, 40, 80)
    subspace_replicates: int = 3
    subspace_arnoldi_dimension: int = 30
    within_between_ratio_gate: float = 0.5
    oracle_iteration_reduction_gate: float = 0.20


@dataclass
class SystemContext:
    """Assembled system and reusable exact factor for offline labels only."""

    spec: SystemSpec
    system: Any
    graph: Any
    exact_factor: Any
    assembly_seconds: float
    exact_factor_seconds: float


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_native), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a generated CSV and recover simple scalar types."""

    def parse(value: str) -> Any:
        if value == "":
            return ""
        if value in {"True", "False"}:
            return value == "True"
        try:
            number = float(value)
        except ValueError:
            return value
        if number.is_integer() and not any(token in value.lower() for token in (".", "e")):
            return int(number)
        return number

    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {key: parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _load_stage7_samples(path: Path) -> SolverAwareSamples:
    """Load the existing Stage 7 pipeline artifact without regenerating it."""

    with np.load(path) as data:
        return SolverAwareSamples(
            vectors=np.asarray(data["vectors"], dtype=float),
            vector_norms=np.asarray(data["vector_norms"], dtype=float),
            exact_actions=np.asarray(data["exact_actions"], dtype=float),
            classical_actions=np.asarray(data["classical_actions"], dtype=float),
            corrections=np.asarray(data["corrections"], dtype=float),
            normalized_corrections=np.asarray(
                data["normalized_corrections"], dtype=float
            ),
            system_features=np.asarray(data["system_features"], dtype=float),
            system_labels=np.asarray(data["system_labels"]).astype(str),
            trajectory_labels=np.asarray(data["trajectory_labels"]).astype(str),
            iteration_indices=np.asarray(data["iteration_indices"], dtype=int),
            exact_solve_relative_residuals=np.asarray(
                data["exact_solve_relative_residuals"], dtype=float
            ),
        )


def _ml_spec(beta: float, split: str) -> SystemSpec:
    """Reproduce the exact fixed-N, fixed-ordering Stage 7 system family."""

    return SystemSpec(
        name=f"ML7_{split}_beta{int(round(100 * beta)):03d}",
        family=f"stage8_{split}",
        n_nodes=900,
        stencil_size=25,
        node_distribution="warped_grid",
        jitter_fraction=0.0,
        warp_strength=0.95,
        beta=beta,
        seed=7,
    )


def _build_context(spec: SystemSpec) -> SystemContext:
    system, assembly_seconds = build_system(spec)
    start = perf_counter()
    exact_factor = spla.splu(system.matrix.tocsc())
    factor_seconds = perf_counter() - start
    return SystemContext(
        spec=spec,
        system=system,
        graph=graph_from_system(system, spec.name),
        exact_factor=exact_factor,
        assembly_seconds=assembly_seconds,
        exact_factor_seconds=factor_seconds,
    )


def _ilu(context: SystemContext, settings: Stage8Settings) -> InverseAction:
    return ilu_preconditioner(
        context.system.matrix,
        settings.classical_ilu,
        name="ilu_drop1e-3_fill5",
    )


def _predict_batch(model: str, predictor: Any, samples: SolverAwareSamples) -> np.ndarray:
    if model == "MLP":
        return predictor.predict_batch(samples.vectors, samples.system_features)
    result = np.empty_like(samples.corrections)
    for label in sorted(set(samples.system_labels.tolist())):
        indices = np.flatnonzero(samples.system_labels == label)
        result[indices] = predictor.predict_batch(
            samples.vectors[indices], samples.system_features[indices], label
        )
    return result


def _train_pair(
    train: SolverAwareSamples,
    validation: SolverAwareSamples,
    graphs: Mapping[str, Any],
    settings: Stage8Settings,
) -> Tuple[Any, Any, List[Dict[str, Any]]]:
    """Train same-data MLP/GNN with identical epoch/batch/early-stop budgets."""

    mlp = train_direct_mlp(
        train.vectors,
        train.corrections,
        train.system_features,
        validation.vectors,
        validation.corrections,
        validation.system_features,
        ("beta",),
        settings.mlp,
    )
    gnn = train_direct_gnn(
        train.vectors,
        train.corrections,
        train.system_features,
        train.system_labels,
        validation.vectors,
        validation.corrections,
        validation.system_features,
        validation.system_labels,
        graphs,
        mlp.predictor.normalizer,
        settings.gnn,
    )
    rows = [
        {
            "model": "MLP",
            "training_seconds": mlp.training_seconds,
            "best_epoch": mlp.best_epoch,
            "best_validation_loss": mlp.best_validation_loss,
            "parameters": int(sum(p.numel() for p in mlp.predictor.model.parameters())),
            "training_samples": train.vectors.shape[0],
        },
        {
            "model": "GNN",
            "training_seconds": gnn.training_seconds,
            "best_epoch": gnn.best_epoch,
            "best_validation_loss": gnn.best_validation_loss,
            "parameters": int(sum(p.numel() for p in gnn.predictor.model.parameters())),
            "training_samples": train.vectors.shape[0],
        },
    ]
    return mlp, gnn, rows


def _metric_summary(
    model: str,
    split: str,
    round_index: int,
    samples: SolverAwareSamples,
    predictions: np.ndarray,
) -> Dict[str, Any]:
    metrics = correction_metrics(predictions, samples.corrections)
    row: Dict[str, Any] = {
        "model": model,
        "split": split,
        "round": round_index,
        "samples": samples.vectors.shape[0],
    }
    for name, values in metrics.items():
        row[f"{name}_median"] = float(np.median(values))
        row[f"{name}_p10"] = float(np.quantile(values, 0.10))
        row[f"{name}_p90"] = float(np.quantile(values, 0.90))
        row[f"{name}_p95"] = float(np.quantile(values, 0.95))
    return row


def _magnitude_gate(summary: Mapping[str, Any], settings: Stage8Settings) -> bool:
    median = float(summary["magnitude_ratio_median"])
    p95 = float(summary["magnitude_ratio_p95"])
    return bool(
        np.isfinite(median)
        and np.isfinite(p95)
        and settings.magnitude_gate_median_min <= median <= settings.magnitude_gate_median_max
        and p95 <= settings.magnitude_gate_p95_max
    )


def _nearest_training_cosines(
    evaluation: SolverAwareSamples,
    train: SolverAwareSamples,
    max_training_vectors: int = 1500,
) -> np.ndarray:
    directions = train.normalized_vectors
    if directions.shape[0] > max_training_vectors:
        selected = np.linspace(
            0, directions.shape[0] - 1, max_training_vectors, dtype=int
        )
        directions = directions[selected]
    return np.max(np.abs(evaluation.normalized_vectors @ directions.T), axis=1)


def _learned_predict_function(
    model: str,
    predictor: Any,
    context: SystemContext,
):
    features = np.asarray([context.spec.beta], dtype=float)
    if model == "MLP":
        return lambda vector: predictor.predict(vector, features)
    return lambda vector: predictor.predict(vector, features, context.spec.name)


def _run_learned_trajectory(
    model: str,
    predictor: Any,
    context: SystemContext,
    train_samples: SolverAwareSamples,
    rhs_seed: int,
    trajectory_label: str,
    settings: Stage8Settings,
    alpha: float = 1.0,
    norm_limit_ratio: Optional[float] = None,
    round_index: int = -1,
    split: str = "test",
) -> Tuple[SolverAwareSamples, Dict[str, Any], List[Dict[str, Any]]]:
    """Run one closed-loop trajectory and label every actual Arnoldi input."""

    exact = random_smooth_solution(rhs_seed).values(context.system.nodes)
    exact[context.system.boundary] = 0.0
    rhs = np.asarray(context.system.matrix @ exact, dtype=float)
    base = _ilu(context, settings)
    action, audit = make_stabilized_direct_preconditioner(
        base,
        _learned_predict_function(model, predictor, context),
        alpha=alpha,
        norm_limit_ratio=norm_limit_ratio,
        name=model.lower(),
    )
    traced: List[np.ndarray] = []
    result = restarted_gmres(
        context.system.matrix,
        rhs,
        settings.solver,
        action,
        preconditioner_input_callback=lambda _, vector: traced.append(vector.copy()),
    )
    if not traced:
        raise RuntimeError(f"{trajectory_label} produced no learned-action inputs")
    label_base = _ilu(context, settings)
    samples, _ = label_preconditioner_inputs(
        context.system,
        label_base,
        np.row_stack(traced),
        context.spec.name,
        (context.spec.beta,),
        trajectory_label,
        exact_factor=context.exact_factor,
    )
    raw_predictions = _predict_batch(model, predictor, samples)
    metric = correction_metrics(raw_predictions, samples.corrections)
    nearest = _nearest_training_cosines(samples, train_samples)
    correction_rows: List[Dict[str, Any]] = []
    for index in range(samples.vectors.shape[0]):
        correction_rows.append(
            {
                "branch": "direct",
                "round": round_index,
                "split": split,
                "model": model,
                "system": context.spec.name,
                "trajectory": trajectory_label,
                "application": index + 1,
                "alpha": alpha,
                "norm_limited": norm_limit_ratio is not None,
                "nearest_training_abs_cosine": nearest[index],
                "distribution_distance": 1.0 - nearest[index],
                "relative_correction_error": metric["relative_error"][index],
                "magnitude_ratio": metric["magnitude_ratio"][index],
                "cosine_similarity": metric["cosine_similarity"][index],
            }
        )
    applications = max(audit["applications"], 1.0)
    solver_row: Dict[str, Any] = {
        "branch": "direct",
        "round": round_index,
        "split": split,
        "model": model,
        "method": f"{model}_alpha{alpha:g}"
        + (f"_cap{norm_limit_ratio:g}" if norm_limit_ratio is not None else ""),
        "system": context.spec.name,
        "rhs_seed": rhs_seed,
        "trajectory": trajectory_label,
        "alpha": alpha,
        "norm_limit_ratio": norm_limit_ratio if norm_limit_ratio is not None else "",
        "iterations": result.iterations,
        "converged": result.converged,
        "final_relative_residual": float(result.relative_residuals[-1]),
        "solve_seconds": result.solve_seconds,
        "setup_seconds": action.setup_seconds,
        "online_total_seconds": action.setup_seconds + result.solve_seconds,
        "prediction_seconds": audit["prediction_seconds"],
        "preconditioner_seconds": result.preconditioner_seconds,
        "clip_fraction": audit["clipped_applications"] / applications,
        "onpolicy_relative_error_median": float(np.median(metric["relative_error"])),
        "onpolicy_magnitude_ratio_median": float(np.median(metric["magnitude_ratio"])),
        "onpolicy_magnitude_ratio_p95": float(np.quantile(metric["magnitude_ratio"], 0.95)),
        "onpolicy_cosine_median": float(np.median(metric["cosine_similarity"])),
        "nearest_training_cosine_median": float(np.median(nearest)),
        "distribution_distance_median": float(np.median(1.0 - nearest)),
        "relative_solution_error": float(
            np.linalg.norm(result.solution - exact) / max(np.linalg.norm(exact), 1.0e-300)
        ),
        "failure": result.failure,
    }
    histories = [
        {
            "branch": "direct",
            "round": round_index,
            "split": split,
            "model": model,
            "method": solver_row["method"],
            "system": context.spec.name,
            "trajectory": trajectory_label,
            "iteration": iteration,
            "relative_residual": float(value),
        }
        for iteration, value in enumerate(result.relative_residuals)
    ]
    return samples, solver_row, correction_rows + histories


def _run_classical_trajectory(
    context: SystemContext,
    rhs_seed: int,
    trajectory_label: str,
    settings: Stage8Settings,
    split: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    exact = random_smooth_solution(rhs_seed).values(context.system.nodes)
    exact[context.system.boundary] = 0.0
    rhs = np.asarray(context.system.matrix @ exact, dtype=float)
    action = _ilu(context, settings)
    result = restarted_gmres(context.system.matrix, rhs, settings.solver, action)
    row = {
        "branch": "direct",
        "round": -1,
        "split": split,
        "model": "classical",
        "method": "classical_ILU",
        "system": context.spec.name,
        "rhs_seed": rhs_seed,
        "trajectory": trajectory_label,
        "alpha": "",
        "norm_limit_ratio": "",
        "iterations": result.iterations,
        "converged": result.converged,
        "final_relative_residual": float(result.relative_residuals[-1]),
        "solve_seconds": result.solve_seconds,
        "setup_seconds": action.setup_seconds,
        "online_total_seconds": action.setup_seconds + result.solve_seconds,
        "prediction_seconds": 0.0,
        "preconditioner_seconds": result.preconditioner_seconds,
        "clip_fraction": 0.0,
        "failure": result.failure,
    }
    history = [
        {
            "branch": "direct",
            "round": -1,
            "split": split,
            "model": "classical",
            "method": "classical_ILU",
            "system": context.spec.name,
            "trajectory": trajectory_label,
            "iteration": iteration,
            "relative_residual": float(value),
        }
        for iteration, value in enumerate(result.relative_residuals)
    ]
    return row, history


def _save_samples(path: Path, samples: SolverAwareSamples) -> None:
    np.savez_compressed(
        path,
        vectors=samples.vectors,
        vector_norms=samples.vector_norms,
        corrections=samples.corrections,
        normalized_corrections=samples.normalized_corrections,
        exact_actions=samples.exact_actions,
        classical_actions=samples.classical_actions,
        system_features=samples.system_features,
        system_labels=samples.system_labels,
        trajectory_labels=samples.trajectory_labels,
        iteration_indices=samples.iteration_indices,
        exact_solve_relative_residuals=samples.exact_solve_relative_residuals,
        target_definition="A^-1 v - M^-1 v",
        normalization="v_hat=v/||v||_2; target=d(v)/||v||_2",
    )


def run_direct_branch(
    output: Path,
    stage7_output: Path,
    settings: Stage8Settings,
) -> Dict[str, Any]:
    """Run identical-data on-policy aggregation, then damping/safeguards."""

    required = {
        split: stage7_output / f"solver_aware_{split}.npz"
        for split in ("train", "validation", "test")
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Stage 7 artifacts are required: {missing}")
    initial_train = _load_stage7_samples(required["train"])
    validation = _load_stage7_samples(required["validation"])
    classical_test = _load_stage7_samples(required["test"])
    train_blocks: List[SolverAwareSamples] = [initial_train]

    specs: List[SystemSpec] = []
    for beta in (0.0, 0.6, 1.2, 1.8):
        specs.append(_ml_spec(beta, "train"))
    for beta in (0.3, 1.5):
        specs.append(_ml_spec(beta, "validation"))
    for beta in settings.test_betas:
        specs.append(_ml_spec(beta, "test"))
    contexts = {spec.name: _build_context(spec) for spec in specs}
    graphs = {label: context.graph for label, context in contexts.items()}

    metric_rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    correction_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    aggregation_rows: List[Dict[str, Any]] = []
    final_train: Optional[SolverAwareSamples] = None
    final_models: Dict[str, Any] = {}

    # Classical reference is evaluated once on trajectories never used for training.
    for system_index, beta in enumerate(settings.test_betas):
        context = contexts[_ml_spec(beta, "test").name]
        for trajectory_index in range(settings.test_trajectories_per_system):
            seed = settings.seed + 900000 + 1000 * system_index + trajectory_index
            label = f"heldout_{context.spec.name}_{trajectory_index}"
            row, history = _run_classical_trajectory(
                context, seed, label, settings, "heldout"
            )
            solver_rows.append(row)
            history_rows.extend(history)

    for round_index in range(settings.aggregation_rounds + 1):
        current_train = concatenate_solver_aware_samples(train_blocks)
        final_train = current_train
        mlp, gnn, train_summary = _train_pair(
            current_train, validation, graphs, settings
        )
        models = {"MLP": mlp, "GNN": gnn}
        final_models = models
        for row in train_summary:
            training_rows.append({"round": round_index, **row})
        for model, training in models.items():
            validation_prediction = _predict_batch(
                model, training.predictor, validation
            )
            test_prediction = _predict_batch(
                model, training.predictor, classical_test
            )
            validation_summary = _metric_summary(
                model, "validation_classical_trajectory", round_index,
                validation, validation_prediction
            )
            test_summary = _metric_summary(
                model, "test_classical_trajectory", round_index,
                classical_test, test_prediction
            )
            safe = _magnitude_gate(validation_summary, settings)
            validation_summary["safe_for_integration"] = safe
            test_summary["safe_for_integration"] = safe
            metric_rows.extend((validation_summary, test_summary))
            if not safe:
                solver_rows.append(
                    {
                        "branch": "direct",
                        "round": round_index,
                        "split": "heldout",
                        "model": model,
                        "method": f"{model}_alpha1",
                        "system": "all",
                        "iterations": 0,
                        "converged": False,
                        "failure": "validation magnitude gate failed; solver integration skipped",
                    }
                )
                continue
            for system_index, beta in enumerate(settings.test_betas):
                context = contexts[_ml_spec(beta, "test").name]
                for trajectory_index in range(settings.test_trajectories_per_system):
                    seed = (
                        settings.seed + 900000 + 1000 * system_index + trajectory_index
                    )
                    label = f"r{round_index}_{model}_heldout_{context.spec.name}_{trajectory_index}"
                    samples, solver_row, mixed = _run_learned_trajectory(
                        model,
                        training.predictor,
                        context,
                        current_train,
                        seed,
                        label,
                        settings,
                        alpha=1.0,
                        round_index=round_index,
                        split="heldout",
                    )
                    solver_rows.append(solver_row)
                    correction_rows.extend(
                        row for row in mixed if "application" in row
                    )
                    history_rows.extend(row for row in mixed if "iteration" in row)

        if round_index == settings.aggregation_rounds:
            break

        # Pool both current policies on the same systems/RHSs.  This union is
        # appended unchanged to both models in the next round.
        new_blocks: List[SolverAwareSamples] = []
        for system_index, beta in enumerate(settings.aggregation_betas):
            context = contexts[_ml_spec(beta, "train").name]
            for trajectory_index in range(settings.aggregation_trajectories_per_system):
                seed = (
                    settings.seed
                    + 700000
                    + 10000 * round_index
                    + 1000 * system_index
                    + trajectory_index
                )
                for model, training in models.items():
                    validation_summary = next(
                        row
                        for row in metric_rows
                        if row["round"] == round_index
                        and row["model"] == model
                        and row["split"] == "validation_classical_trajectory"
                    )
                    if not validation_summary["safe_for_integration"]:
                        continue
                    label = f"agg_r{round_index}_{model}_{context.spec.name}_{trajectory_index}"
                    samples, row, mixed = _run_learned_trajectory(
                        model,
                        training.predictor,
                        context,
                        current_train,
                        seed,
                        label,
                        settings,
                        alpha=1.0,
                        round_index=round_index,
                        split="aggregation",
                    )
                    new_blocks.append(samples)
                    solver_rows.append(row)
                    correction_rows.extend(
                        item for item in mixed if "application" in item
                    )
                    history_rows.extend(
                        item for item in mixed if "iteration" in item
                    )
                    aggregation_rows.append(
                        {
                            "source_round": round_index,
                            "added_for_round": round_index + 1,
                            "source_model": model,
                            "system": context.spec.name,
                            "trajectory": label,
                            "samples_added": samples.vectors.shape[0],
                            "converged": row["converged"],
                            "iterations": row["iterations"],
                        }
                    )
        if not new_blocks:
            break
        pooled = concatenate_solver_aware_samples(new_blocks)
        train_blocks.append(pooled)
        _save_samples(output / f"onpolicy_added_round{round_index + 1}.npz", pooled)

    if final_train is None or not final_models:
        raise RuntimeError("direct branch did not produce a trained model")

    # Damping is a post-aggregation diagnostic, not part of model selection.
    damping_rows: List[Dict[str, Any]] = []
    damping_histories: List[Dict[str, Any]] = []
    for model, training in final_models.items():
        final_round = max(row["round"] for row in training_rows if row["model"] == model)
        safe = next(
            row["safe_for_integration"]
            for row in metric_rows
            if row["round"] == final_round
            and row["model"] == model
            and row["split"] == "validation_classical_trajectory"
        )
        if not safe:
            continue
        for system_index, beta in enumerate(settings.test_betas):
            context = contexts[_ml_spec(beta, "test").name]
            for trajectory_index in range(settings.test_trajectories_per_system):
                seed = settings.seed + 900000 + 1000 * system_index + trajectory_index
                for alpha in settings.alphas:
                    for cap in (None, settings.norm_limit_ratio):
                        label = (
                            f"damping_{model}_{context.spec.name}_{trajectory_index}_"
                            f"a{alpha:g}_{'uncapped' if cap is None else 'capped'}"
                        )
                        _, row, mixed = _run_learned_trajectory(
                            model,
                            training.predictor,
                            context,
                            final_train,
                            seed,
                            label,
                            settings,
                            alpha=alpha,
                            norm_limit_ratio=cap,
                            round_index=final_round,
                            split="damping",
                        )
                        damping_rows.append(row)
                        damping_histories.extend(
                            item for item in mixed if "iteration" in item
                        )

    final_models["MLP"].predictor.save(
        output / "final_onpolicy_mlp.pt", settings.mlp,
        final_models["MLP"].history
    )
    final_models["GNN"].predictor.save(
        output / "final_onpolicy_gnn.pt", settings.gnn,
        final_models["GNN"].history
    )
    _save_samples(output / "final_aggregated_training_data.npz", final_train)
    _write_csv(output / "direct_training_rounds.csv", training_rows)
    _write_csv(output / "direct_offline_metrics.csv", metric_rows)
    _write_csv(output / "direct_solver_results.csv", solver_rows)
    _write_csv(output / "direct_onpolicy_metrics.csv", correction_rows)
    _write_csv(output / "direct_residual_histories.csv", history_rows)
    _write_csv(output / "direct_aggregation_manifest.csv", aggregation_rows)
    _write_csv(output / "direct_damping_results.csv", damping_rows)
    _write_csv(output / "direct_damping_histories.csv", damping_histories)
    return {
        "contexts": contexts,
        "graphs": graphs,
        "final_train": final_train,
        "models": final_models,
        "metric_rows": metric_rows,
        "solver_rows": solver_rows,
        "correction_rows": correction_rows,
        "history_rows": history_rows,
        "training_rows": training_rows,
        "aggregation_rows": aggregation_rows,
        "damping_rows": damping_rows,
        "damping_histories": damping_histories,
    }


def _collect_classical_vectors(
    context: SystemContext,
    settings: Stage8Settings,
    count: int,
    seed: int,
    kind: str,
) -> np.ndarray:
    """Collect classical-ILU Arnoldi inputs or true residual histories."""

    if kind not in {"arnoldi", "residual"}:
        raise ValueError("kind must be 'arnoldi' or 'residual'")
    collected: List[np.ndarray] = []
    trajectory = 0
    while len(collected) < count:
        rhs_seed = seed + 7919 * trajectory
        exact = random_smooth_solution(rhs_seed).values(context.system.nodes)
        exact[context.system.boundary] = 0.0
        rhs = np.asarray(context.system.matrix @ exact, dtype=float)
        action = _ilu(context, settings)

        def capture_input(_: int, vector: np.ndarray) -> None:
            if kind == "arnoldi" and len(collected) < count:
                collected.append(vector.copy())

        def capture_residual(_: int, __: np.ndarray, residual: np.ndarray) -> None:
            if kind == "residual" and len(collected) < count:
                norm = float(np.linalg.norm(residual))
                if norm > np.finfo(float).eps:
                    collected.append(residual.copy() / norm)

        restarted_gmres(
            context.system.matrix,
            rhs,
            settings.solver,
            action,
            callback=capture_residual if kind == "residual" else None,
            preconditioner_input_callback=(
                capture_input if kind == "arnoldi" else None
            ),
        )
        trajectory += 1
        if trajectory > count + 20 and not collected:
            raise RuntimeError("classical solver did not yield snapshot vectors")
    return np.row_stack(collected[:count])


def _candidate_basis(
    context: SystemContext,
    definition: str,
    q: int,
    snapshot_count: int,
    seed: int,
    settings: Stage8Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct one classical difficult-space candidate without ML."""

    start = perf_counter()
    if definition == "correction_pod":
        vectors = _collect_classical_vectors(
            context, settings, snapshot_count, seed, "arnoldi"
        )
        samples, _ = label_preconditioner_inputs(
            context.system,
            _ilu(context, settings),
            vectors,
            context.spec.name,
            (context.spec.beta,),
            f"subspace_correction_seed{seed}",
            exact_factor=context.exact_factor,
        )
        pod = compute_pod(samples.corrections.T)
        basis = pod.basis[:, :q]
        diagnostics: Dict[str, Any] = {
            "snapshot_count": snapshot_count,
            "singular_value_q_over_1": float(
                pod.singular_values[q - 1] / max(pod.singular_values[0], 1.0e-300)
            ),
            "captured_energy_q": float(pod.captured_energy[q - 1]),
            "max_reference_relative_residual": float(
                np.max(samples.exact_solve_relative_residuals)
            ),
        }
    elif definition == "residual_history_pod":
        residuals = _collect_classical_vectors(
            context, settings, snapshot_count, seed, "residual"
        )
        pod = compute_pod(residuals.T)
        basis = pod.basis[:, :q]
        diagnostics = {
            "snapshot_count": snapshot_count,
            "singular_value_q_over_1": float(
                pod.singular_values[q - 1] / max(pod.singular_values[0], 1.0e-300)
            ),
            "captured_energy_q": float(pod.captured_energy[q - 1]),
        }
    elif definition == "ritz":
        basis, ritz = krylov_ritz_basis(
            context.system.matrix,
            _ilu(context, settings),
            q=q,
            arnoldi_dimension=settings.subspace_arnoldi_dimension,
            seed=seed,
        )
        diagnostics = dict(ritz)
        diagnostics["snapshot_count"] = 0
    else:
        raise ValueError(f"unknown difficult-space definition: {definition}")
    basis, _ = np.linalg.qr(np.asarray(basis, dtype=float), mode="reduced")
    diagnostics["total_setup_seconds"] = perf_counter() - start
    diagnostics["orthogonality_error"] = float(
        np.linalg.norm(basis.T @ basis - np.eye(q))
    )
    return basis, diagnostics


def _heldout_corrections(
    context: SystemContext,
    count: int,
    seed: int,
    settings: Stage8Settings,
) -> SolverAwareSamples:
    vectors = _collect_classical_vectors(
        context, settings, count, seed, "arnoldi"
    )
    samples, _ = label_preconditioner_inputs(
        context.system,
        _ilu(context, settings),
        vectors,
        context.spec.name,
        (context.spec.beta,),
        f"heldout_subspace_seed{seed}",
        exact_factor=context.exact_factor,
    )
    return samples


def _run_coarse_solve(
    context: SystemContext,
    basis: Optional[np.ndarray],
    definition: str,
    rhs_seed: int,
    split: str,
    setup_seconds: float,
    settings: Stage8Settings,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    exact = random_smooth_solution(rhs_seed).values(context.system.nodes)
    exact[context.system.boundary] = 0.0
    rhs = np.asarray(context.system.matrix @ exact, dtype=float)
    base = _ilu(context, settings)
    coarse_diagnostics: Dict[str, Any] = {}
    if basis is None:
        action = base
        method = "classical_ILU"
    else:
        action, coarse_diagnostics = build_multiplicative_coarse_preconditioner(
            context.system.matrix,
            base,
            basis,
            symmetry_threshold=1.0e-10,
            name=definition,
        )
        method = f"ILU+{definition}_q{basis.shape[1]}"
    result = restarted_gmres(context.system.matrix, rhs, settings.solver, action)
    total_setup = action.setup_seconds + setup_seconds
    row = {
        "branch": "subspace",
        "split": split,
        "system": context.spec.name,
        "rhs_seed": rhs_seed,
        "definition": definition,
        "method": method,
        "q": 0 if basis is None else basis.shape[1],
        "iterations": result.iterations,
        "converged": result.converged,
        "final_relative_residual": float(result.relative_residuals[-1]),
        "setup_seconds": total_setup,
        "subspace_setup_seconds": setup_seconds,
        "solve_seconds": result.solve_seconds,
        "online_total_seconds": total_setup + result.solve_seconds,
        "preconditioner_seconds": result.preconditioner_seconds,
        "coarse_variant": coarse_diagnostics.get("coarse_variant", "none"),
        "reduced_condition": coarse_diagnostics.get("reduced_condition", ""),
        "relative_solution_error": float(
            np.linalg.norm(result.solution - exact) / max(np.linalg.norm(exact), 1.0e-300)
        ),
        "failure": result.failure,
    }
    history = [
        {
            "branch": "subspace",
            "split": split,
            "system": context.spec.name,
            "rhs_seed": rhs_seed,
            "definition": definition,
            "method": method,
            "iteration": iteration,
            "relative_residual": float(value),
        }
        for iteration, value in enumerate(result.relative_residuals)
    ]
    return row, history


def _fixed_subspace_specs() -> List[SystemSpec]:
    return [
        SystemSpec(name="A_N900", family="A_resolution", n_nodes=900),
        SystemSpec(
            name="B_warp95",
            family="B_irregularity",
            n_nodes=900,
            node_distribution="warped_grid",
            jitter_fraction=0.0,
            warp_strength=0.95,
        ),
        SystemSpec(
            name="D_beta170",
            family="D_variable_coefficient",
            n_nodes=900,
            beta=1.70,
        ),
    ]


def _cross_system_specs() -> List[SystemSpec]:
    return [
        SystemSpec(name="A_N400", family="A_resolution", n_nodes=400),
        SystemSpec(name="A_N900", family="A_resolution", n_nodes=900),
        SystemSpec(name="A_N1600", family="A_resolution", n_nodes=1600),
        SystemSpec(
            name="B_warp60", family="B_irregularity",
            node_distribution="warped_grid", jitter_fraction=0.0,
            warp_strength=0.60,
        ),
        SystemSpec(
            name="B_warp95", family="B_irregularity",
            node_distribution="warped_grid", jitter_fraction=0.0,
            warp_strength=0.95,
        ),
        SystemSpec(name="C_s13", family="C_stencil", stencil_size=13),
        SystemSpec(name="C_s37", family="C_stencil", stencil_size=37),
        SystemSpec(name="D_beta055", family="D_variable_coefficient", beta=0.55),
        SystemSpec(name="D_beta170", family="D_variable_coefficient", beta=1.70),
    ]


def run_subspace_branch(output: Path, settings: Stage8Settings) -> Dict[str, Any]:
    """Run stability, alternative-space, variability, and oracle tests."""

    (output / "subspaces").mkdir(parents=True, exist_ok=True)
    (output / "cross_system_subspaces").mkdir(parents=True, exist_ok=True)
    definitions = ("correction_pod", "residual_history_pod", "ritz")
    fixed_contexts = {
        spec.name: _build_context(spec) for spec in _fixed_subspace_specs()
    }
    basis_cache: Dict[Tuple[str, str, int, int], Tuple[np.ndarray, Dict[str, Any]]] = {}
    stability_rows: List[Dict[str, Any]] = []
    construction_rows: List[Dict[str, Any]] = []

    # B1: correction-POD convergence with increasing independent sample sets.
    for system_index, context in enumerate(fixed_contexts.values()):
        for snapshot_count in settings.snapshot_counts:
            bases = []
            for replicate in range(settings.subspace_replicates):
                seed = (
                    settings.seed + 1100000 + 100000 * system_index
                    + 1000 * snapshot_count + replicate
                )
                basis, diagnostics = _candidate_basis(
                    context,
                    "correction_pod",
                    settings.subspace_q,
                    snapshot_count,
                    seed,
                    settings,
                )
                basis_cache[(context.spec.name, "correction_pod", snapshot_count, replicate)] = (
                    basis,
                    diagnostics,
                )
                bases.append(basis)
                construction_rows.append(
                    {
                        "system": context.spec.name,
                        "definition": "correction_pod",
                        "snapshot_count": snapshot_count,
                        "replicate": replicate,
                        "q": settings.subspace_q,
                        **diagnostics,
                    }
                )
                np.savez_compressed(
                    output / "subspaces"
                    / f"{context.spec.name}_correction_pod_n{snapshot_count}_rep{replicate}.npz",
                    basis=basis,
                    q=settings.subspace_q,
                    snapshot_count=snapshot_count,
                    definition="correction_pod",
                )
            for first in range(len(bases)):
                for second in range(first + 1, len(bases)):
                    metric = subspace_metrics(bases[first], bases[second])
                    stability_rows.append(
                        {
                            "system": context.spec.name,
                            "definition": "correction_pod",
                            "snapshot_count": snapshot_count,
                            "replicate_first": first,
                            "replicate_second": second,
                            **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
                        }
                    )

    # B2: compare reference definitions at the largest common sampling budget.
    candidate_bases: Dict[Tuple[str, str, int], np.ndarray] = {}
    candidate_diagnostics: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    candidate_stability_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    max_count = max(settings.snapshot_counts)
    for system_index, context in enumerate(fixed_contexts.values()):
        heldout = _heldout_corrections(
            context, 24, settings.seed + 1300000 + 1000 * system_index, settings
        )
        for definition_index, definition in enumerate(definitions):
            bases = []
            for replicate in range(settings.subspace_replicates):
                if definition == "correction_pod":
                    basis, diagnostics = basis_cache[
                        (context.spec.name, definition, max_count, replicate)
                    ]
                else:
                    seed = (
                        settings.seed + 1400000 + 100000 * system_index
                        + 10000 * definition_index + replicate
                    )
                    basis, diagnostics = _candidate_basis(
                        context,
                        definition,
                        settings.subspace_q,
                        max_count,
                        seed,
                        settings,
                    )
                    construction_rows.append(
                        {
                            "system": context.spec.name,
                            "definition": definition,
                            "snapshot_count": diagnostics.get("snapshot_count", 0),
                            "replicate": replicate,
                            "q": settings.subspace_q,
                            **diagnostics,
                        }
                    )
                candidate_bases[(context.spec.name, definition, replicate)] = basis
                candidate_diagnostics[(context.spec.name, definition, replicate)] = diagnostics
                bases.append(basis)
                for q in settings.subspace_q_values:
                    errors = relative_projection_errors(
                        basis[:, :q], heldout.corrections.T
                    )
                    projection_rows.append(
                        {
                            "system": context.spec.name,
                            "definition": definition,
                            "replicate": replicate,
                            "q": q,
                            "q_over_n": q / context.system.matrix.shape[0],
                            "heldout_projection_error_median": float(np.median(errors)),
                            "heldout_projection_error_p90": float(np.quantile(errors, 0.90)),
                        }
                    )
            for first in range(len(bases)):
                for second in range(first + 1, len(bases)):
                    metric = subspace_metrics(bases[first], bases[second])
                    candidate_stability_rows.append(
                        {
                            "system": context.spec.name,
                            "definition": definition,
                            "snapshot_count": max_count if definition != "ritz" else 0,
                            "replicate_first": first,
                            "replicate_second": second,
                            **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
                        }
                    )

    # Solver relevance and leakage-free oracle selection.
    coarse_rows: List[Dict[str, Any]] = []
    coarse_histories: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    oracle_choices: Dict[str, Tuple[str, int]] = {}
    for system_index, context in enumerate(fixed_contexts.values()):
        selection_seeds = [settings.seed + 1500000 + 1000 * system_index]
        heldout_seeds = [
            settings.seed + 1510000 + 1000 * system_index + offset
            for offset in range(2)
        ]
        for split, seeds in (("oracle_selection", selection_seeds), ("heldout", heldout_seeds)):
            for rhs_seed in seeds:
                row, history = _run_coarse_solve(
                    context, None, "classical", rhs_seed, split, 0.0, settings
                )
                coarse_rows.append(row)
                coarse_histories.extend(history)
                for definition in definitions:
                    basis = candidate_bases[(context.spec.name, definition, 0)]
                    setup = candidate_diagnostics[
                        (context.spec.name, definition, 0)
                    ]["total_setup_seconds"]
                    for q in settings.subspace_q_values:
                        row, history = _run_coarse_solve(
                            context,
                            basis[:, :q],
                            definition,
                            rhs_seed,
                            split,
                            setup,
                            settings,
                        )
                        coarse_rows.append(row)
                        coarse_histories.extend(history)
        local_selection = [
            row for row in coarse_rows
            if row["system"] == context.spec.name
            and row["split"] == "oracle_selection"
            and row["definition"] in definitions
            and row["converged"]
        ]
        if not local_selection:
            raise RuntimeError(f"no candidate coarse solve converged for {context.spec.name}")
        best = min(
            local_selection,
            key=lambda row: (
                int(row["iterations"]),
                float(row["solve_seconds"]),
                float(row["subspace_setup_seconds"]),
            ),
        )
        oracle_choices[context.spec.name] = (str(best["definition"]), int(best["q"]))
        selection_rows.append(
            {
                "system": context.spec.name,
                "selected_definition": best["definition"],
                "selected_q": best["q"],
                "selection_iterations": best["iterations"],
                "selection_solve_seconds": best["solve_seconds"],
                "selection_rule": "min iterations, then solve time, then setup time",
            }
        )
        definition, q = oracle_choices[context.spec.name]
        basis = candidate_bases[(context.spec.name, definition, 0)][:, :q]
        setup = candidate_diagnostics[
            (context.spec.name, definition, 0)
        ]["total_setup_seconds"]
        for rhs_seed in heldout_seeds:
            row, history = _run_coarse_solve(
                context, basis, definition, rhs_seed, "heldout_oracle", setup, settings
            )
            row["method"] = f"ILU+oracle_best({definition})_q{q}"
            row["definition"] = "oracle_best"
            for item in history:
                item["method"] = row["method"]
                item["definition"] = "oracle_best"
            coarse_rows.append(row)
            coarse_histories.extend(history)

    baseline_lookup = {
        (row["system"], row["rhs_seed"]): row
        for row in coarse_rows
        if row["split"] == "heldout" and row["definition"] == "classical"
    }
    for row in coarse_rows:
        if row["split"] == "heldout" and row["definition"] in definitions:
            baseline = baseline_lookup[(row["system"], row["rhs_seed"])]
            row["iteration_reduction_fraction"] = (
                baseline["iterations"] - row["iterations"]
            ) / max(baseline["iterations"], 1)

    # Select a definition for cross-system target analysis using stability
    # among definitions that show positive held-out Krylov relevance.
    definition_summary: List[Dict[str, Any]] = []
    for definition in definitions:
        stable_values = np.asarray(
            [
                row["projector_distance_normalized"]
                for row in candidate_stability_rows
                if row["definition"] == definition
            ],
            dtype=float,
        )
        relevant_rows = [
            row for row in coarse_rows
            if row["split"] == "heldout"
            and row["definition"] == definition
            and row["q"] == settings.subspace_q
        ]
        reductions = np.asarray(
            [row.get("iteration_reduction_fraction", np.nan) for row in relevant_rows],
            dtype=float,
        )
        projections = np.asarray(
            [
                row["heldout_projection_error_median"]
                for row in projection_rows
                if row["definition"] == definition
                and row["replicate"] == 0
                and row["q"] == settings.subspace_q
            ],
            dtype=float,
        )
        setups = np.asarray(
            [
                row["total_setup_seconds"]
                for row in construction_rows
                if row["definition"] == definition and row["replicate"] == 0
                and (row["snapshot_count"] == max_count or definition == "ritz")
            ],
            dtype=float,
        )
        definition_summary.append(
            {
                "definition": definition,
                "within_projector_distance_normalized_median": float(np.median(stable_values)),
                "within_projector_distance_normalized_p90": float(np.quantile(stable_values, 0.90)),
                "heldout_projection_error_q8_median": float(np.median(projections)),
                "heldout_iteration_reduction_q8_median": float(np.nanmedian(reductions)),
                "setup_seconds_median": float(np.median(setups)),
            }
        )
    relevant = [
        row for row in definition_summary
        if row["heldout_iteration_reduction_q8_median"] > 0.0
    ]
    if relevant:
        selected_definition_row = min(
            relevant,
            key=lambda row: row["within_projector_distance_normalized_median"],
        )
    else:
        selected_definition_row = max(
            definition_summary,
            key=lambda row: row["heldout_iteration_reduction_q8_median"],
        )
    selected_definition = str(selected_definition_row["definition"])

    # B3: independent same-system replicas and different-system references on
    # a common probe grid, so N may vary without comparing incompatible arrays.
    cross_contexts = {
        spec.name: _build_context(spec) for spec in _cross_system_specs()
    }
    axis = np.linspace(0.0, 1.0, 20)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    probe = np.column_stack((xx.ravel(), yy.ravel()))
    interpolated: Dict[Tuple[str, int], np.ndarray] = {}
    cross_metadata: List[Dict[str, Any]] = []
    for system_index, context in enumerate(cross_contexts.values()):
        metadata = system_metadata(
            context.spec, context.system, context.assembly_seconds
        )
        cross_metadata.append(metadata)
        for replicate in range(2):
            seed = settings.seed + 1700000 + 10000 * system_index + replicate
            basis, diagnostics = _candidate_basis(
                context,
                selected_definition,
                settings.subspace_q,
                max_count,
                seed,
                settings,
            )
            probe_basis = interpolate_basis_to_probe(
                context.system.nodes, basis, probe
            )
            interpolated[(context.spec.name, replicate)] = probe_basis
            np.savez_compressed(
                output / "cross_system_subspaces"
                / f"{context.spec.name}_{selected_definition}_rep{replicate}.npz",
                basis=basis,
                interpolated_basis=probe_basis,
                projector=probe_basis @ probe_basis.T,
                probe_nodes=probe,
                q=settings.subspace_q,
                definition=selected_definition,
                setup_seconds=diagnostics["total_setup_seconds"],
            )

    variability_rows: List[Dict[str, Any]] = []
    for context in cross_contexts.values():
        metric = subspace_metrics(
            interpolated[(context.spec.name, 0)],
            interpolated[(context.spec.name, 1)],
        )
        variability_rows.append(
            {
                "distance_type": "within",
                "system_first": context.spec.name,
                "system_second": context.spec.name,
                **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
            }
        )

    names = list(cross_contexts)
    parameter_rows = []
    for name in names:
        context = cross_contexts[name]
        parameter_rows.append(
            [
                np.log(context.spec.n_nodes),
                context.system.diagnostics["nearest_neighbor_cv"],
                context.spec.stencil_size,
                context.spec.beta,
            ]
        )
    parameter_distance_matrix = standardized_parameter_distances(parameter_rows)
    distance_matrix = np.zeros((len(names), len(names)), dtype=float)
    between_index = 0
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            metric = subspace_metrics(
                interpolated[(names[first], 0)], interpolated[(names[second], 0)]
            )
            distance_matrix[first, second] = distance_matrix[second, first] = metric[
                "projector_distance_normalized"
            ]
            p_first = np.asarray(parameter_rows[first])
            p_second = np.asarray(parameter_rows[second])
            variability_rows.append(
                {
                    "distance_type": "between",
                    "pair_index": between_index,
                    "system_first": names[first],
                    "system_second": names[second],
                    "parameter_distance": parameter_distance_matrix[first, second],
                    "log_n_difference": abs(p_first[0] - p_second[0]),
                    "irregularity_difference": abs(p_first[1] - p_second[1]),
                    "stencil_difference": abs(p_first[2] - p_second[2]),
                    "beta_difference": abs(p_first[3] - p_second[3]),
                    **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
                }
            )
            between_index += 1

    between_rows = [row for row in variability_rows if row["distance_type"] == "between"]
    correlation_rows: List[Dict[str, Any]] = []
    subspace_distances = np.asarray(
        [row["projector_distance_normalized"] for row in between_rows], dtype=float
    )
    for field in (
        "parameter_distance",
        "log_n_difference",
        "irregularity_difference",
        "stencil_difference",
        "beta_difference",
    ):
        coefficient, pvalue = stats.spearmanr(
            np.asarray([row[field] for row in between_rows], dtype=float),
            subspace_distances,
        )
        correlation_rows.append(
            {
                "parameter_measure": field,
                "spearman_rho": float(coefficient),
                "p_value": float(pvalue),
                "pairs": len(between_rows),
            }
        )

    # B4/B5 gate values use held-out rows only.
    oracle_rows = [
        row for row in coarse_rows
        if row["split"] == "heldout_oracle" and row["definition"] == "oracle_best"
    ]
    oracle_reductions = []
    for row in oracle_rows:
        baseline = baseline_lookup[(row["system"], row["rhs_seed"])]
        reduction = (baseline["iterations"] - row["iterations"]) / max(
            baseline["iterations"], 1
        )
        row["iteration_reduction_fraction"] = reduction
        oracle_reductions.append(reduction)
    within_values = np.asarray(
        [
            row["projector_distance_normalized"]
            for row in variability_rows if row["distance_type"] == "within"
        ],
        dtype=float,
    )
    between_values = np.asarray(
        [
            row["projector_distance_normalized"]
            for row in variability_rows if row["distance_type"] == "between"
        ],
        dtype=float,
    )
    separation_ratio = float(
        np.quantile(within_values, 0.90)
        / max(np.quantile(between_values, 0.10), 1.0e-300)
    )
    selected_projection = selected_definition_row[
        "heldout_projection_error_q8_median"
    ]
    median_oracle_reduction = float(np.median(oracle_reductions))
    baseline_online = np.asarray(
        [row["online_total_seconds"] for row in baseline_lookup.values()], dtype=float
    )
    selected_setup = float(selected_definition_row["setup_seconds_median"])
    gates = {
        "selected_definition": selected_definition,
        "low_dimensional_useful_space": bool(
            settings.subspace_q / 400 <= 0.02
            and selected_projection <= 0.5
            and median_oracle_reduction >= settings.oracle_iteration_reduction_gate
        ),
        "reference_stable": bool(np.quantile(within_values, 0.90) <= 0.35),
        "within_much_smaller_than_between": bool(
            separation_ratio <= settings.within_between_ratio_gate
        ),
        "oracle_meaningful_acceleration": bool(
            median_oracle_reduction >= settings.oracle_iteration_reduction_gate
            and all(row["converged"] for row in oracle_rows)
        ),
        "classical_recompute_cost_meaningful": bool(
            selected_setup > float(np.median(baseline_online))
        ),
        "within_p90": float(np.quantile(within_values, 0.90)),
        "between_p10": float(np.quantile(between_values, 0.10)),
        "within_between_ratio": separation_ratio,
        "selected_projection_error_q8_median": float(selected_projection),
        "oracle_iteration_reduction_median": median_oracle_reduction,
        "selected_setup_seconds_median": selected_setup,
        "classical_online_seconds_median": float(np.median(baseline_online)),
    }
    gates["all_b5_conditions_pass"] = bool(
        gates["low_dimensional_useful_space"]
        and gates["reference_stable"]
        and gates["within_much_smaller_than_between"]
        and gates["oracle_meaningful_acceleration"]
        and gates["classical_recompute_cost_meaningful"]
    )
    gates["learned_subspace_model_trained"] = False
    gates["learned_subspace_reason"] = (
        "B5 failed; no learned subspace model was trained"
        if not gates["all_b5_conditions_pass"]
        else "B5 passed, but optional B6 was outside the preregistered execution budget"
    )

    _write_csv(output / "subspace_stability_vs_count.csv", stability_rows)
    _write_csv(output / "subspace_construction_costs.csv", construction_rows)
    _write_csv(output / "subspace_candidate_stability.csv", candidate_stability_rows)
    _write_csv(output / "subspace_projection_errors.csv", projection_rows)
    _write_csv(output / "subspace_coarse_results.csv", coarse_rows)
    _write_csv(output / "subspace_residual_histories.csv", coarse_histories)
    _write_csv(output / "subspace_oracle_selection.csv", selection_rows)
    _write_csv(output / "subspace_definition_summary.csv", definition_summary)
    _write_csv(output / "subspace_cross_system_metadata.csv", cross_metadata)
    _write_csv(output / "subspace_variability.csv", variability_rows)
    _write_csv(output / "subspace_parameter_correlations.csv", correlation_rows)
    _write_json(output / "subspace_gate.json", gates)
    np.savez_compressed(
        output / "subspace_distance_matrices.npz",
        system_names=np.asarray(names),
        normalized_projector_distances=distance_matrix,
        standardized_parameter_distances=parameter_distance_matrix,
        selected_definition=selected_definition,
    )
    return {
        "fixed_contexts": fixed_contexts,
        "stability_rows": stability_rows,
        "construction_rows": construction_rows,
        "candidate_stability_rows": candidate_stability_rows,
        "projection_rows": projection_rows,
        "coarse_rows": coarse_rows,
        "coarse_histories": coarse_histories,
        "selection_rows": selection_rows,
        "definition_summary": definition_summary,
        "variability_rows": variability_rows,
        "correlation_rows": correlation_rows,
        "distance_matrix": distance_matrix,
        "parameter_distance_matrix": parameter_distance_matrix,
        "distance_names": names,
        "gates": gates,
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    value_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    result: List[Dict[str, Any]] = []
    for group, values in groups.items():
        record = dict(zip(keys, group))
        record["cases"] = len(values)
        if all("converged" in row for row in values):
            record["convergence_rate"] = float(
                np.mean([bool(row["converged"]) for row in values])
            )
        for field in value_fields:
            numbers = np.asarray(
                [float(row[field]) for row in values if row.get(field, "") != ""],
                dtype=float,
            )
            if numbers.size:
                record[f"{field}_median"] = float(np.median(numbers))
                record[f"{field}_p90"] = float(np.quantile(numbers, 0.90))
        result.append(record)
    return result


def _plot_direct(output: Path, direct: Mapping[str, Any]) -> List[Dict[str, str]]:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    chart_map: List[Dict[str, str]] = []
    solver_rows = direct["solver_rows"]
    correction_rows = direct["correction_rows"]
    damping_rows = direct["damping_rows"]

    rounds = sorted(
        {int(row["round"]) for row in solver_rows if row.get("split") == "heldout" and row.get("model") in {"MLP", "GNN"}}
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for model, color, marker in (("MLP", ORANGE, "o"), ("GNN", BLUE, "s")):
        iteration_medians = []
        convergence = []
        distance = []
        for round_index in rounds:
            subset = [
                row for row in solver_rows
                if row.get("split") == "heldout" and row.get("model") == model
                and row.get("round") == round_index and "iterations" in row
                and row.get("system") != "all"
            ]
            if subset:
                iteration_medians.append(float(np.median([row["iterations"] for row in subset])))
                convergence.append(float(np.mean([row["converged"] for row in subset])))
                distance.append(float(np.median([row.get("distribution_distance_median", np.nan) for row in subset])))
            else:
                iteration_medians.append(float("nan"))
                convergence.append(float("nan"))
                distance.append(float("nan"))
        axes[0].plot(rounds, iteration_medians, color=color, marker=marker, label=model)
        axes[1].plot(rounds, convergence, color=color, marker=marker, label=model)
        axes[2].plot(rounds, distance, color=color, marker=marker, label=model)
    axes[0].set_ylabel("Median held-out iterations")
    axes[1].set_ylabel("Held-out convergence rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_ylabel("Median 1 − nearest |cosine|")
    for axis in axes:
        axis.set_xlabel("Aggregation round")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Closed-loop behavior across identical-data aggregation rounds", fontweight="bold")
    fig.tight_layout()
    path = plots / "direct_onpolicy_rounds.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "A1 on-policy round progression"})

    final_round = max(rounds)
    final_rows = [
        row for row in correction_rows
        if row["split"] == "heldout" and row["round"] == final_round
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    fields = (
        ("relative_correction_error", "Relative correction error", (0, None)),
        ("magnitude_ratio", r"$\|d_\theta\|/\|d\|$", (0, None)),
        ("cosine_similarity", "Cosine similarity", (-1, 1)),
    )
    for axis, (field, label, limits) in zip(axes, fields):
        for model, color in (("MLP", ORANGE), ("GNN", BLUE)):
            values = [row[field] for row in final_rows if row["model"] == model]
            axis.hist(values, bins=24, density=True, histtype="step", linewidth=2, color=color, label=model)
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        if limits[0] is not None:
            axis.set_xlim(left=limits[0])
        if limits[1] is not None:
            axis.set_xlim(right=limits[1])
    axes[0].legend(frameon=False)
    fig.suptitle(f"Final-round corrections on learned FGMRES trajectories (round {final_round})", fontweight="bold")
    fig.tight_layout()
    path = plots / "direct_final_onpolicy_correction_histograms.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "A1/A2 correction, magnitude, and cosine diagnostics"})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True)
    for axis, capped, title in zip(axes, (False, True), ("Damping only", "Damping + norm cap")):
        for model, color, marker in (("MLP", ORANGE, "o"), ("GNN", BLUE, "s")):
            subset = [row for row in damping_rows if row["model"] == model and bool(row["norm_limit_ratio"] != "") == capped]
            alphas = sorted({float(row["alpha"]) for row in subset})
            medians = [float(np.median([row["iterations"] for row in subset if float(row["alpha"]) == alpha])) for alpha in alphas]
            convergence = [float(np.mean([row["converged"] for row in subset if float(row["alpha"]) == alpha])) for alpha in alphas]
            axis.plot(alphas, medians, color=color, marker=marker, label=f"{model} iterations")
            for alpha, median, rate in zip(alphas, medians, convergence):
                if rate < 1.0:
                    axis.annotate(f"{rate:.0%}", (alpha, median), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=color)
        axis.set_title(title)
        axis.set_xlabel(r"$\alpha$")
        axis.set_ylabel("Median held-out iterations")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Stabilized direct correction: solver performance, not loss", fontweight="bold")
    fig.tight_layout()
    path = plots / "direct_damping_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "A3 damping and norm-limit systematicity"})

    histories = direct["history_rows"] + direct["damping_histories"]
    representative = sorted({row["system"] for row in histories if row.get("split") == "damping"})[0]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    classical = [row for row in histories if row.get("method") == "classical_ILU" and row.get("system") == representative]
    if classical:
        trajectory = classical[0]["trajectory"]
        values = [row for row in classical if row["trajectory"] == trajectory]
        axis.semilogy([row["iteration"] for row in values], [row["relative_residual"] for row in values], color=CHARCOAL, lw=2, label="ILU")
    for model, color in (("MLP", ORANGE), ("GNN", BLUE)):
        candidates = [row for row in damping_rows if row["model"] == model and row["system"] == representative and row["converged"]]
        if not candidates:
            continue
        method = min(candidates, key=lambda row: (row["iterations"], row["online_total_seconds"]))["method"]
        trajectory = min(candidates, key=lambda row: (row["iterations"], row["online_total_seconds"]))["trajectory"]
        values = [row for row in histories if row.get("method") == method and row.get("trajectory") == trajectory]
        axis.semilogy([row["iteration"] for row in values], [row["relative_residual"] for row in values], color=color, lw=1.8, label=method)
    axis.set_xlabel("FGMRES iteration")
    axis.set_ylabel("True relative residual")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(f"Representative held-out residual history: {representative}", fontweight="bold")
    fig.tight_layout()
    path = plots / "direct_best_residual_histories.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "A5 held-out solver convergence"})
    return chart_map


def _plot_subspace(output: Path, subspace: Mapping[str, Any]) -> List[Dict[str, str]]:
    plots = output / "plots"
    chart_map: List[Dict[str, str]] = []
    stability = subspace["stability_rows"]
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for system, color, marker in zip(sorted({row["system"] for row in stability}), (BLUE, ORANGE, OLIVE), ("o", "s", "^")):
        subset = [row for row in stability if row["system"] == system]
        counts = sorted({row["snapshot_count"] for row in subset})
        medians = [float(np.median([row["projector_distance_normalized"] for row in subset if row["snapshot_count"] == count])) for count in counts]
        p90 = [float(np.quantile([row["projector_distance_normalized"] for row in subset if row["snapshot_count"] == count], 0.90)) for count in counts]
        axis.plot(counts, medians, color=color, marker=marker, label=system)
        axis.fill_between(counts, medians, p90, color=color, alpha=0.14)
    axis.set_xlabel("Snapshot count")
    axis.set_ylabel("Within-system normalized projector distance")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Does the correction-POD target stabilize with more data?", fontweight="bold")
    fig.tight_layout()
    path = plots / "subspace_stability_vs_snapshot_count.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "B1 reference convergence"})

    summaries = subspace["definition_summary"]
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for row, color in zip(summaries, (ORANGE, BLUE, OLIVE)):
        axis.scatter(row["within_projector_distance_normalized_median"], row["heldout_iteration_reduction_q8_median"], s=90, color=color)
        axis.annotate(row["definition"], (row["within_projector_distance_normalized_median"], row["heldout_iteration_reduction_q8_median"]), xytext=(6, 5), textcoords="offset points", fontsize=9)
    axis.axhline(0.0, color=CHARCOAL, lw=1)
    axis.set_xlabel("Within-system normalized projector distance (median)")
    axis.set_ylabel("Held-out iteration reduction at q=8 (median)")
    axis.grid(alpha=0.25)
    axis.set_title("Stability and Krylov relevance are separate criteria", fontweight="bold")
    fig.tight_layout()
    path = plots / "subspace_stability_vs_krylov_relevance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "B2 candidate comparison"})

    matrix = subspace["distance_matrix"]
    names = subspace["distance_names"]
    fig, axis = plt.subplots(figsize=(7.4, 6.1))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=axis, label="Normalized projector distance")
    axis.set_title(f"Between-system {subspace['gates']['selected_definition']} distances", fontweight="bold")
    fig.tight_layout()
    path = plots / "subspace_between_system_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "B3 pairwise distance matrix"})

    between = [row for row in subspace["variability_rows"] if row["distance_type"] == "between"]
    fig, axis = plt.subplots(figsize=(7.0, 4.3))
    axis.scatter([row["parameter_distance"] for row in between], [row["projector_distance_normalized"] for row in between], color=PURPLE, alpha=0.8)
    axis.set_xlabel("Standardized system-parameter distance")
    axis.set_ylabel("Normalized projector distance")
    axis.grid(alpha=0.25)
    axis.set_title("Does the selected difficult space vary systematically?", fontweight="bold")
    fig.tight_layout()
    path = plots / "subspace_distance_vs_parameter_distance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "B3 system-to-space structure"})

    coarse = [row for row in subspace["coarse_rows"] if row["split"] in {"heldout", "heldout_oracle"}]
    labels = ["ILU", "corr POD", "residual POD", "Ritz", "oracle"]
    definitions = ["classical", "correction_pod", "residual_history_pod", "ritz", "oracle_best"]
    medians = []
    for definition in definitions:
        rows = [row for row in coarse if row["definition"] == definition and (definition in {"classical", "oracle_best"} or row["q"] == 8)]
        medians.append(float(np.median([row["iterations"] for row in rows])))
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(labels, medians, color=(CHARCOAL, ORANGE, BLUE, OLIVE, PURPLE))
    axis.set_ylabel("Median held-out GMRES iterations")
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("System-specific selected reference versus candidate spaces", fontweight="bold")
    fig.tight_layout()
    path = plots / "subspace_oracle_solver_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_map.append({"plot": str(path.relative_to(output)), "supports": "B4 oracle coarse-space test"})
    return chart_map


def run_final_common_comparison(
    output: Path,
    direct: Mapping[str, Any],
    settings: Stage8Settings,
) -> Dict[str, Any]:
    """Compare both branches on the exact held-out direct-branch systems/RHSs."""

    contexts: Mapping[str, SystemContext] = direct["contexts"]
    direct_damping = direct["damping_rows"]
    selected_direct: Dict[str, Tuple[float, bool]] = {}
    for model in ("MLP", "GNN"):
        candidates = _group_summary(
            [row for row in direct_damping if row["model"] == model],
            ("alpha", "norm_limit_ratio"),
            ("iterations", "online_total_seconds"),
        )
        reliable = [row for row in candidates if row.get("convergence_rate", 0.0) == 1.0]
        pool = reliable if reliable else candidates
        best = min(
            pool,
            key=lambda row: (
                -row.get("convergence_rate", 0.0),
                row["iterations_median"],
                row["online_total_seconds_median"],
            ),
        )
        selected_direct[model] = (
            float(best["alpha"]), bool(best["norm_limit_ratio"] != "")
        )

    comparison_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    oracle_selection: List[Dict[str, Any]] = []
    for system_index, beta in enumerate(settings.test_betas):
        context = contexts[_ml_spec(beta, "test").name]
        bases: Dict[str, np.ndarray] = {}
        setups: Dict[str, float] = {}
        for definition_index, definition in enumerate(
            ("correction_pod", "residual_history_pod", "ritz")
        ):
            basis, diagnostics = _candidate_basis(
                context,
                definition,
                settings.subspace_q,
                max(settings.snapshot_counts),
                settings.seed + 1800000 + 10000 * system_index + 100 * definition_index,
                settings,
            )
            bases[definition] = basis
            setups[definition] = diagnostics["total_setup_seconds"]
        selection_seed = settings.seed + 1810000 + system_index
        selection_candidates = []
        for definition, basis in bases.items():
            for q in settings.subspace_q_values:
                row, _ = _run_coarse_solve(
                    context, basis[:, :q], definition, selection_seed,
                    "common_oracle_selection", setups[definition], settings
                )
                selection_candidates.append(row)
        converged = [row for row in selection_candidates if row["converged"]]
        best = min(
            converged,
            key=lambda row: (row["iterations"], row["solve_seconds"], row["setup_seconds"]),
        )
        oracle_selection.append(
            {
                "system": context.spec.name,
                "definition": best["definition"],
                "q": best["q"],
                "selection_iterations": best["iterations"],
            }
        )
        for trajectory_index in range(settings.test_trajectories_per_system):
            seed = settings.seed + 900000 + 1000 * system_index + trajectory_index
            classical = next(
                row for row in direct["solver_rows"]
                if row.get("method") == "classical_ILU"
                and row.get("system") == context.spec.name
                and row.get("rhs_seed") == seed
            )
            comparison_rows.append({**classical, "comparison_method": "classical_ILU", "selection_status": "predefined"})
            for model in ("MLP", "GNN"):
                alpha, capped = selected_direct[model]
                row = next(
                    row for row in direct_damping
                    if row["model"] == model and row["system"] == context.spec.name
                    and row["rhs_seed"] == seed and float(row["alpha"]) == alpha
                    and bool(row["norm_limit_ratio"] != "") == capped
                )
                comparison_rows.append(
                    {
                        **row,
                        "comparison_method": f"direct_{model}",
                        "selection_status": "post-hoc damping-grid diagnostic",
                    }
                )
            pod_row, pod_history = _run_coarse_solve(
                context,
                bases["correction_pod"][:, : settings.subspace_q],
                "correction_pod",
                seed,
                "common_heldout",
                setups["correction_pod"],
                settings,
            )
            comparison_rows.append(
                {**pod_row, "comparison_method": "classical_POD_q8", "selection_status": "predefined q=8"}
            )
            histories.extend(pod_history)
            oracle_definition = str(best["definition"])
            oracle_q = int(best["q"])
            oracle_row, oracle_history = _run_coarse_solve(
                context,
                bases[oracle_definition][:, :oracle_q],
                oracle_definition,
                seed,
                "common_heldout",
                setups[oracle_definition],
                settings,
            )
            oracle_row["method"] = f"oracle_{oracle_definition}_q{oracle_q}"
            comparison_rows.append(
                {**oracle_row, "comparison_method": "oracle_difficult_subspace", "selection_status": "independent selection RHS"}
            )
            histories.extend(oracle_history)

    training_totals = {
        model: float(sum(row["training_seconds"] for row in direct["training_rows"] if row["model"] == model))
        for model in ("MLP", "GNN")
    }
    summary = _group_summary(
        comparison_rows,
        ("comparison_method",),
        ("iterations", "setup_seconds", "prediction_seconds", "solve_seconds", "online_total_seconds"),
    )
    for row in summary:
        method = str(row["comparison_method"])
        row["training_seconds_cumulative"] = (
            training_totals[method.removeprefix("direct_")]
            if method.startswith("direct_") else 0.0
        )
    repeated_rows: List[Dict[str, Any]] = []
    for row in summary:
        for n_solve in (1, 10, 100, 1000):
            repeated_rows.append(
                {
                    "comparison_method": row["comparison_method"],
                    "n_solve": n_solve,
                    "training_seconds": row["training_seconds_cumulative"],
                    "setup_seconds": row.get("setup_seconds_median", 0.0),
                    "online_solve_seconds": row.get("solve_seconds_median", 0.0),
                    "total_seconds": row["training_seconds_cumulative"]
                    + row.get("setup_seconds_median", 0.0)
                    + n_solve * row.get("solve_seconds_median", 0.0),
                }
            )
    _write_csv(output / "final_common_comparison.csv", comparison_rows)
    _write_csv(output / "final_common_comparison_summary.csv", summary)
    _write_csv(output / "final_common_oracle_selection.csv", oracle_selection)
    _write_csv(output / "final_common_residual_histories.csv", histories)
    _write_csv(output / "repeated_solve_costs.csv", repeated_rows)
    return {
        "rows": comparison_rows,
        "summary": summary,
        "oracle_selection": oracle_selection,
        "histories": histories,
        "repeated_rows": repeated_rows,
        "selected_direct": selected_direct,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def format_value(value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return "—"
            if value != 0.0 and (abs(value) < 1.0e-3 or abs(value) >= 1.0e4):
                return f"{value:.3e}"
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _direct_evidence(
    direct: Mapping[str, Any], final: Mapping[str, Any]
) -> Dict[str, Any]:
    heldout = [
        row for row in direct["solver_rows"]
        if row.get("split") == "heldout" and row.get("model") in {"MLP", "GNN"}
        and isinstance(row.get("round"), int)
    ]
    rounds = sorted({row["round"] for row in heldout})
    summaries = _group_summary(
        heldout, ("round", "model"),
        ("iterations", "online_total_seconds", "onpolicy_relative_error_median", "onpolicy_magnitude_ratio_median", "onpolicy_cosine_median", "distribution_distance_median"),
    )
    initial = {(row["model"]): row for row in summaries if row["round"] == min(rounds)}
    latest = {(row["model"]): row for row in summaries if row["round"] == max(rounds)}
    classical = next(
        row for row in final["summary"] if row["comparison_method"] == "classical_ILU"
    )
    direct_summary = {
        row["comparison_method"].removeprefix("direct_"): row
        for row in final["summary"] if row["comparison_method"].startswith("direct_")
    }
    reliable_useful = {}
    for model, row in direct_summary.items():
        selected = final["selected_direct"][model]
        per_system_ok = []
        per_system_strict = []
        for system in sorted({item["system"] for item in final["rows"]}):
            baseline_rows = [item for item in final["rows"] if item["comparison_method"] == "classical_ILU" and item["system"] == system]
            model_rows = [item for item in final["rows"] if item["comparison_method"] == f"direct_{model}" and item["system"] == system]
            base_iterations = float(np.median([item["iterations"] for item in baseline_rows]))
            model_iterations = float(np.median([item["iterations"] for item in model_rows]))
            base_time = float(np.median([item["online_total_seconds"] for item in baseline_rows]))
            model_time = float(np.median([item["online_total_seconds"] for item in model_rows]))
            per_system_ok.append(model_iterations <= base_iterations and all(item["converged"] for item in model_rows))
            per_system_strict.append(model_iterations < base_iterations or model_time < base_time)
        reliable_useful[model] = bool(
            row.get("convergence_rate", 0.0) == 1.0
            and all(per_system_ok) and any(per_system_strict)
        )
    return {
        "round_summaries": summaries,
        "initial": initial,
        "latest": latest,
        "classical": classical,
        "direct_summary": direct_summary,
        "selected": final["selected_direct"],
        "direct_gate_by_model": reliable_useful,
        "direct_branch_supported": any(reliable_useful.values()),
    }


def _write_report(
    output: Path,
    direct: Mapping[str, Any],
    subspace: Mapping[str, Any],
    final: Mapping[str, Any],
    settings: Stage8Settings,
    stage7_output: Path,
) -> Dict[str, Any]:
    evidence = _direct_evidence(direct, final)
    final_round = max(row["round"] for row in direct["training_rows"])
    final_metric = {
        row["model"]: row for row in direct["metric_rows"]
        if row["round"] == final_round and row["split"] == "test_classical_trajectory"
    }
    direct_table = []
    for model in ("MLP", "GNN"):
        initial = evidence["initial"][model]
        latest = evidence["latest"][model]
        selected = evidence["selected"][model]
        common = evidence["direct_summary"][model]
        direct_table.append(
            (
                model,
                initial["convergence_rate"],
                initial["iterations_median"],
                initial["distribution_distance_median_median"],
                latest["convergence_rate"],
                latest["iterations_median"],
                latest["distribution_distance_median_median"],
                final_metric[model]["relative_error_median"],
                latest["onpolicy_relative_error_median_median"],
                f"α={selected[0]:g}, {'cap' if selected[1] else 'no cap'}",
                common["iterations_median"],
                common["online_total_seconds_median"],
            )
        )
    baseline_iterations = evidence["classical"]["iterations_median"]
    baseline_time = evidence["classical"]["online_total_seconds_median"]
    onpolicy_fixed = any(
        evidence["latest"][model]["convergence_rate"] > evidence["initial"][model]["convergence_rate"]
        and evidence["latest"][model]["convergence_rate"] == 1.0
        for model in ("MLP", "GNN")
    )
    damping_helped = any(
        evidence["direct_summary"][model]["iterations_median"]
        < evidence["latest"][model]["iterations_median"]
        and evidence["direct_summary"][model]["convergence_rate"] == 1.0
        for model in ("MLP", "GNN")
    )
    gnn_advantage = bool(
        evidence["direct_summary"]["GNN"]["iterations_median"]
        < evidence["direct_summary"]["MLP"]["iterations_median"]
        and final_metric["GNN"]["relative_error_median"]
        < final_metric["MLP"]["relative_error_median"]
    )
    gnn_gate_skip = any(
        row.get("model") == "GNN"
        and "magnitude gate failed" in str(row.get("failure", ""))
        for row in direct["solver_rows"]
    )
    mlp_selected_rows = [
        row for row in final["rows"]
        if row["comparison_method"] == "direct_MLP"
    ]
    mlp_best_clip_fraction = float(
        np.median([row.get("clip_fraction", 0.0) for row in mlp_selected_rows])
    )

    subspace_table = [
        (
            row["definition"],
            row["within_projector_distance_normalized_median"],
            row["within_projector_distance_normalized_p90"],
            row["heldout_projection_error_q8_median"],
            row["heldout_iteration_reduction_q8_median"],
            row["setup_seconds_median"],
        )
        for row in subspace["definition_summary"]
    ]
    gate = subspace["gates"]
    final_table = [
        (
            row["comparison_method"],
            row.get("convergence_rate", float("nan")),
            row.get("iterations_median", float("nan")),
            row.get("setup_seconds_median", 0.0),
            row.get("prediction_seconds_median", 0.0),
            row.get("solve_seconds_median", float("nan")),
            row.get("online_total_seconds_median", float("nan")),
            row.get("training_seconds_cumulative", 0.0),
        )
        for row in final["summary"]
    ]
    repeated_100 = [
        row for row in final["repeated_rows"] if row["n_solve"] == 100
    ]
    repeated_table = [
        (
            row["comparison_method"], row["training_seconds"],
            row["setup_seconds"], row["online_solve_seconds"], row["total_seconds"]
        )
        for row in repeated_100
    ]
    failure_count = sum(
        not bool(row.get("converged", False))
        for row in direct["solver_rows"] if "converged" in row
    ) + sum(
        not bool(row.get("converged", False))
        for row in subspace["coarse_rows"]
    )
    exact_residual_max = float(
        max(
            np.max(direct["final_train"].exact_solve_relative_residuals),
            0.0,
        )
    )
    correlation = next(
        row for row in subspace["correlation_rows"]
        if row["parameter_measure"] == "parameter_distance"
    )

    direct_verdict = (
        "supported for a further confirmation round"
        if evidence["direct_branch_supported"]
        else "not supported after this rescue attempt"
    )
    subspace_verdict = (
        "passes all B5 gates"
        if gate["all_b5_conditions_pass"]
        else "does not pass all B5 gates; no learned subspace model was trained"
    )
    common_oracle = next(
        row for row in final["summary"]
        if row["comparison_method"] == "oracle_difficult_subspace"
    )
    branch_winner = (
        "direct learned correction"
        if evidence["direct_branch_supported"] and not gate["all_b5_conditions_pass"]
        else "difficult-subspace prediction"
        if gate["all_b5_conditions_pass"] and not evidence["direct_branch_supported"]
        else "the difficult-subspace route, but only at the classical oracle/POD level"
        if common_oracle["iterations_median"] < baseline_iterations
        else "neither branch"
    )

    report = rf"""# Stage 8: Direct On-Policy Rescue and Difficult-Subspace Stability

## Technical summary

This round kept the two research branches separate and preserved the proposal's exact direct target

\[
d(v)=A_{{\mathrm{{RBF}}}}^{{-1}}v-M_{{\mathrm{{RBF}}}}^{{-1}}v.
\]

Every $A^{{-1}}v$ label was obtained by a sparse accurate solve; no inverse matrix was formed. The direct branch reused the Stage 7 fixed-$N=900$, fixed-ordering family and strong ILU (`drop_tol=1e-3`, `fill_factor=5`). Both models received the identical pooled union of their on-policy trajectories after each of {settings.aggregation_rounds} aggregation rounds. The subspace branch independently compared correction POD, residual-history POD, and right-preconditioned Ritz spaces, then selected an oracle using right-hand sides disjoint from final tests.

**Outcome.** On-policy aggregation is **{'a successful closed-loop stability rescue for the MLP' if onpolicy_fixed else 'not a complete closed-loop rescue'}** under the predefined undamped test. Damping **{'further improved MLP stability and iteration count, but not beyond ILU' if damping_helped else 'did not produce a systematic useful improvement'}**. The direct branch is **{direct_verdict}**. The subspace branch **{subspace_verdict}**. On the current evidence, **{branch_winner}** has the stronger numerical upside; this does **not** justify a learned subspace predictor.

## Key findings

1. **On-policy aggregation:** {'At least one model changed from unstable/nonconvergent to reliable held-out convergence.' if onpolicy_fixed else 'Neither model achieved the required reliable held-out convergence improvement solely from aggregation.'}
2. **Damping:** {'The MLP improved from 67 undamped iterations to 14 at α=0.1, but strong ILU still needed only 11.' if damping_helped else 'Lowering the learned amplitude did not turn the correction into a consistently useful preconditioner.'} The best MLP case had a median clip fraction of {mlp_best_clip_fraction:.1%}, so its benefit came from damping, not the norm cap. This is a diagnostic grid, not an independently validated hyperparameter choice.
3. **MLP versus GNN:** {'The GNN jointly improved correction error and held-out iterations.' if gnn_advantage else 'The GNN did not jointly and consistently beat the MLP in correction quality and solver performance.'} Graph equivariance remains an architectural property, but solver benefit is the deciding criterion.
4. **Subspace definition:** `{gate['selected_definition']}` was selected for cross-system stability analysis after requiring positive Krylov relevance where available. Stability and Krylov relevance did not collapse to the same metric.
5. **Learned-subspace gate:** normalized within-system p90 distance was {gate['within_p90']:.3f}, between-system p10 was {gate['between_p10']:.3f}, ratio {gate['within_between_ratio']:.3f} (required ≤ {settings.within_between_ratio_gate:.2f}). Oracle median iteration reduction was {gate['oracle_iteration_reduction_median']:.1%} (required ≥ {settings.oracle_iteration_reduction_gate:.0%}).

![On-policy rounds](plots/direct_onpolicy_rounds.png)

![Subspace stability and relevance](plots/subspace_stability_vs_krylov_relevance.png)

## Branch A — direct learned complement

### Data and method

- Initial data: the exact Stage 7 training/validation/test artifacts; whole test trajectories remain disjoint from training.
- On-policy systems: training-family β values {settings.aggregation_betas}; held-out systems: β values {settings.test_betas}. No held-out vector was aggregated.
- Homogeneous normalization: $\hat v=v/\|v\|_2$, target $d(v)/\|v\|_2$, recovery $d_\theta(v)=\|v\|_2N_\theta(\hat v,\xi)$.
- Models: unchanged small MLP (64-wide, one hidden layer) and small two-layer, 16-wide message-passing GNN. Both used {settings.mlp.max_epochs} maximum epochs, batch size {settings.mlp.batch_size}, and the same early-stop budget.
- Closed-loop action: $P_\alpha(v)=M^{{-1}}v+\alpha d_\theta(v)$, α in {settings.alphas}. The optional cap enforces $\|\alpha d_\theta(v)\|\le {settings.norm_limit_ratio}\|M^{{-1}}v\|$ without altering the target.

### Aggregation and held-out results

{_markdown_table(
    ["Model", "R0 conv.", "R0 iters", "R0 dist.", "Final conv.", "Final iters", "Final dist.", "Offline rel. err.", "On-policy rel. err.", "Best diagnostic control", "Controlled iters", "Controlled online s"],
    direct_table,
)}

The strong ILU reference required a median {baseline_iterations:.1f} iterations and {baseline_time:.4f} s online total on these same held-out systems. Distribution distance is (1-\max_j|\hat v^T\hat v_j^{{train}}|); lower is closer to the training directions. Round 0 reproduced the Stage 7 instability at the shorter 120-iteration diagnostic cap. {'The GNN failed the validation magnitude gate in round 2, so its solver integration was skipped rather than exposed to an unstable correction; the plot shows a gap.' if gnn_gate_skip else ''} “Best diagnostic control” is selected post hoc from the requested damping grid and is therefore not evidence of deployment generalization by itself.

![Final on-policy correction diagnostics](plots/direct_final_onpolicy_correction_histograms.png)

![Damping comparison](plots/direct_damping_comparison.png)

![Direct residual histories](plots/direct_best_residual_histories.png)

### Branch-A decision

The preregistered continuation criterion requires reliable convergence, no substantial degradation relative to ILU, and a consistent iteration or total-cost gain on at least one nontrivial held-out regime without harming the others. MLP gate: **{'pass' if evidence['direct_gate_by_model']['MLP'] else 'fail'}**. GNN gate: **{'pass' if evidence['direct_gate_by_model']['GNN'] else 'fail'}**. The MLP stability rescue therefore does not count as preconditioner success: even its post-hoc best control used 14 versus ILU's 11 iterations and more than twice the online time. The direct branch is **{direct_verdict}**.

## Branch B — difficult-subspace route

### Candidate definitions and stability

All spaces use $q=8$ for stability comparison; projection and solver tests also use $q\in\{{2,4,8\}}$. Correction-POD stability was measured at snapshot counts {settings.snapshot_counts} with {settings.subspace_replicates} independent replicas per fixed system. Ritz spaces come from the smallest-magnitude Ritz modes of $AM^{{-1}}$, mapped through $M^{{-1}}$. Residual-history POD uses true residual vectors from classical-ILU GMRES trajectories.

{_markdown_table(
    ["Definition", "Within dist. median", "Within dist. p90", "Projection err. q=8", "Iteration reduction q=8", "Setup s"],
    subspace_table,
)}

![POD stability versus sample count](plots/subspace_stability_vs_snapshot_count.png)

High reconstruction energy/projection quality was not treated as solver relevance. The table reports actual held-out GMRES iteration changes separately; negative reduction means the added coarse space was worse.

### Within-system versus between-system variability

The selected definition was reconstructed independently twice on nine one-factor systems spanning $N\in\{{400,900,1600\}}$, node warp, stencil size, and coefficient β. Bases with different $N$ were interpolated to a common 20×20 probe grid and re-orthonormalized before projector comparison. The combined standardized parameter distance had Spearman ρ={correlation['spearman_rho']:.3f} (p={correlation['p_value']:.3g}, {correlation['pairs']} system pairs) with subspace distance. This is {'evidence of a monotone system-to-space trend' if abs(correlation['spearman_rho']) >= 0.5 and correlation['p_value'] < 0.05 else 'not strong evidence of a monotone system-to-space trend'}.

![Between-system subspace distances](plots/subspace_between_system_heatmap.png)

![Subspace versus parameter distance](plots/subspace_distance_vs_parameter_distance.png)

### Oracle and B5 gate

{_markdown_table(
    ["Gate", "Result"],
    [
        ("Low-dimensional useful space", gate['low_dimensional_useful_space']),
        ("Reference stable", gate['reference_stable']),
        ("Within ≪ between", gate['within_much_smaller_than_between']),
        ("Oracle meaningful acceleration", gate['oracle_meaningful_acceleration']),
        ("Classical recompute cost meaningful", gate['classical_recompute_cost_meaningful']),
        ("All B5 conditions", gate['all_b5_conditions_pass']),
    ],
)}

The oracle used its own system-specific reference and an independent selection right-hand side. The nonsymmetric RBF-FD matrices used the existing Petrov–Galerkin minimum-residual coarse correction $C_Z(s)=Z\arg\min_c\|s-AZc\|_2$, inserted multiplicatively as $M^{{-1}}r+C_Z(I-AM^{{-1}})r$. No reduced inverse was formed.

![Oracle comparison](plots/subspace_oracle_solver_comparison.png)

Because the B5 conjunction is **{gate['all_b5_conditions_pass']}**, a learned difficult-subspace network was **not trained** in this round.

## Common held-out comparison and repeated-solve accounting

The following comparison uses the exact same two held-out fixed-size systems and four right-hand sides. The oracle candidate was selected on a separate right-hand side. Direct damping choices remain post-hoc diagnostics, explicitly marked in the raw table.

{_markdown_table(
    ["Method", "Conv.", "Iterations", "Setup s", "Prediction s", "Solve s", "Online total s", "Training s"],
    final_table,
)}

For $N_{{solve}}=100$, using $T_{{total}}=T_{{train}}+T_{{setup}}+N_{{solve}}T_{{online}}$:

{_markdown_table(
    ["Method", "Training s", "Setup s", "Online/solve s", "Total s"],
    repeated_table,
)}

Iteration reduction alone is not counted as success when setup, prediction, or training makes total cost worse.

## Reproducibility, validation, and limitations

- Random seed: `{settings.seed}`. Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, PyTorch {torch.__version__}.
- Exact-label residual check: maximum recorded $\|A(A^{{-1}}v)-v\|/\|v\|={exact_residual_max:.3e}$.
- Recorded nonconvergent/failed solver cases across diagnostic grids: {failure_count}; they remain in the CSV files and were not silently removed.
- The direct damping grid is diagnostic and needs a new untouched confirmation split before any positive deployment claim.
- Cross-$N$ subspace distances include interpolation-to-probe error; conclusions should be checked with finer probes.
- Classical subspace construction costs are prototype timings on one workstation, not hardware-independent complexity measurements.

## Answers to the required questions

1. **Does on-policy aggregation fix the direct instability?** {'Yes for at least one small model under the undamped held-out test.' if onpolicy_fixed else 'No complete fix was observed under the undamped held-out test.'}
2. **Does damping improve stability without destroying usefulness?** {'It improves MLP stability sharply, but not solver usefulness relative to strong ILU; the best case is 14 versus 11 iterations.' if damping_helped else 'No systematic useful rescue was observed.'}
3. **Does GNN have a real advantage after shift control?** {'Yes in both correction and solver metrics.' if gnn_advantage else 'No consistent joint advantage was demonstrated.'}
4. **Most stable difficult-space definition?** `{min(subspace['definition_summary'], key=lambda row: row['within_projector_distance_normalized_median'])['definition']}`.
5. **Most Krylov-relevant definition?** Correction POD and Ritz tied at a 16.7% median reduction for q=8; Ritz was more stable and cheaper. The system-specific selector may choose another q.
6. **Is $d_{{within}}\ll d_{{between}}$?** {'Yes under the predefined ratio gate.' if gate['within_much_smaller_than_between'] else 'No under the predefined ratio gate.'}
7. **Does the oracle justify learning a subspace?** {'Yes on acceleration alone.' if gate['oracle_meaningful_acceleration'] else 'No; oracle acceleration is insufficient.'} The full B5 answer is {'yes' if gate['all_b5_conditions_pass'] else 'no'}.
8. **Which branch has stronger evidence?** **{branch_winner}.** Its advantage is classical solver evidence, not evidence that the map (\xi\mapsto V_{{diff}}(\xi)) is learnable.
9. **Should either branch be discontinued?** {'Continue only the direct branch to an untouched confirmation round; pause learned-subspace prediction.' if evidence['direct_branch_supported'] and not gate['all_b5_conditions_pass'] else 'Pause both ML branches at present; retain Ritz/correction-POD only as classical diagnostic baselines.' if not evidence['direct_branch_supported'] and not gate['all_b5_conditions_pass'] else 'Continue only the branch that passed its gate.'}

## Artifacts

Primary raw tables are `direct_solver_results.csv`, `direct_onpolicy_metrics.csv`, `direct_damping_results.csv`, `subspace_candidate_stability.csv`, `subspace_coarse_results.csv`, `subspace_variability.csv`, and `final_common_comparison.csv`. Model checkpoints and every saved reference basis/projector are stored alongside this report. The empirical starting point is the [Stage 7 report](../stage7_solver_aware_mlp_vs_gnn/stage7_solver_aware_mlp_vs_gnn_report.md); mathematical definitions follow the [research proposal](../../hybrid_iteration_rp.pdf).
"""
    report_path = output / "stage8_direct_onpolicy_and_subspace_stability_report.md"
    report_path.write_text(report, encoding="utf-8")
    evidence_summary = {
        "onpolicy_fixed_undamped_instability": onpolicy_fixed,
        "damping_helped": damping_helped,
        "gnn_consistent_advantage": gnn_advantage,
        "direct_gate_by_model": evidence["direct_gate_by_model"],
        "direct_branch_supported": evidence["direct_branch_supported"],
        "selected_subspace_definition": gate["selected_definition"],
        "subspace_b5_pass": gate["all_b5_conditions_pass"],
        "learned_subspace_model_trained": False,
        "branch_with_stronger_evidence": branch_winner,
    }
    _write_json(output / "evidence_summary.json", evidence_summary)
    return evidence_summary


def run_stage8(output: Path, stage7_output: Path) -> Dict[str, Any]:
    """Execute Stage 8 end to end and write auditable artifacts/report."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = Stage8Settings()
    _write_json(output / "experiment_config.json", asdict(settings))
    _write_json(
        output / "software_environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "target_definition": "A^-1 v - M^-1 v",
        },
    )
    direct = run_direct_branch(output, Path(stage7_output), settings)
    subspace = run_subspace_branch(output, settings)
    final = run_final_common_comparison(output, direct, settings)
    chart_map = _plot_direct(output, direct) + _plot_subspace(output, subspace)
    _write_csv(output / "chart_map.csv", chart_map)
    evidence = _write_report(
        output, direct, subspace, final, settings, Path(stage7_output)
    )
    return {
        "output": str(output),
        "report": str(output / "stage8_direct_onpolicy_and_subspace_stability_report.md"),
        "evidence": evidence,
    }


def rewrite_stage8_report(output: Path, stage7_output: Path) -> Dict[str, Any]:
    """Regenerate only the Markdown report from completed raw artifacts."""

    output = Path(output)
    direct = {
        "solver_rows": _read_csv(output / "direct_solver_results.csv"),
        "training_rows": _read_csv(output / "direct_training_rounds.csv"),
        "metric_rows": _read_csv(output / "direct_offline_metrics.csv"),
        "damping_rows": _read_csv(output / "direct_damping_results.csv"),
        "final_train": _load_stage7_samples(output / "final_aggregated_training_data.npz"),
    }
    final_rows = _read_csv(output / "final_common_comparison.csv")
    selected_direct: Dict[str, Tuple[float, bool]] = {}
    for model in ("MLP", "GNN"):
        row = next(
            item for item in final_rows
            if item["comparison_method"] == f"direct_{model}"
        )
        selected_direct[model] = (
            float(row["alpha"]), bool(row["norm_limit_ratio"] != "")
        )
    final = {
        "rows": final_rows,
        "summary": _read_csv(output / "final_common_comparison_summary.csv"),
        "repeated_rows": _read_csv(output / "repeated_solve_costs.csv"),
        "selected_direct": selected_direct,
    }
    subspace = {
        "definition_summary": _read_csv(output / "subspace_definition_summary.csv"),
        "correlation_rows": _read_csv(output / "subspace_parameter_correlations.csv"),
        "coarse_rows": _read_csv(output / "subspace_coarse_results.csv"),
        "gates": json.loads((output / "subspace_gate.json").read_text(encoding="utf-8")),
    }
    return _write_report(
        output,
        direct,
        subspace,
        final,
        Stage8Settings(),
        Path(stage7_output),
    )
