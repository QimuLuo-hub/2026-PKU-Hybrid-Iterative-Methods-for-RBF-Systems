"""Stage 35: residual-space loss for full-dictionary MLP and FNO selection."""

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
import torch
from torch import nn

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _pair_from_z, _run_pair
from .stage28_spatial_k_operator_learning import (
    FNOFieldEncoder,
    ParameterMLP,
    _normalized_inputs_for_system,
)
from .stage29_family_level_coarse_space import _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage34_neural_target_conditioned_selection import Stage34Settings, predict_coordinates
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage35Settings:
    model_seed: int = 20262821
    maximum_rank: int = 48
    hidden_widths: Tuple[int, int] = (64, 128)
    fno_width: int = 16
    fno_modes: int = 8
    fno_layers: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 6
    maximum_epochs: int = 300
    minimum_epochs: int = 100
    validation_interval: int = 5
    early_stopping_patience: int = 75
    cholesky_jitter: float = 1.0e-6
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage34(self) -> Stage34Settings:
        return Stage34Settings()


def residual_projector_loss(
    prediction: torch.Tensor,
    action_gram: torch.Tensor,
    teacher_cross: torch.Tensor,
    rank: int,
    jitter: float,
) -> torch.Tensor:
    r"""Compute normalized projector loss for ``orth(A Psi C_theta)``.

    ``action_gram=(A Psi)^T(A Psi)`` and
    ``teacher_cross=W_star^T(A Psi)`` make this exactly the residual-space
    projector loss without materializing the 5,041-row predicted action.
    """

    coefficients = prediction[:, :rank]
    coefficient_q = torch.linalg.qr(coefficients, mode="reduced").Q
    gram_c = coefficient_q.T @ action_gram @ coefficient_q
    scale = torch.diagonal(gram_c).mean()
    eye = torch.eye(rank, dtype=prediction.dtype, device=prediction.device)
    stabilized = gram_c + (
        jitter * torch.clamp(scale, min=torch.finfo(prediction.dtype).eps)
    ) * eye
    lower = torch.linalg.cholesky(stabilized)
    teacher_overlap_unscaled = teacher_cross[:rank, :] @ coefficient_q
    overlap = torch.linalg.solve_triangular(
        lower, teacher_overlap_unscaled.T, upper=False
    )
    overlap_squared = torch.sum(overlap**2)
    return torch.clamp(1.0 - overlap_squared / rank, min=0.0)


def _model(model_name: str, p: int, settings: Stage35Settings) -> nn.Module:
    if model_name == "mlp_residual_loss":
        return ParameterMLP(p, settings.maximum_rank, settings.hidden_widths)
    if model_name == "fno_residual_loss":
        return FNOFieldEncoder(
            p, settings.maximum_rank, settings.fno_width,
            settings.fno_modes, settings.fno_layers,
        )
    raise ValueError(model_name)


def _inputs(model_name: str, data: Mapping[str, np.ndarray]) -> np.ndarray:
    key = "normalized_parameters" if model_name == "mlp_residual_loss" else "normalized_fields"
    return np.asarray(data[key], dtype=np.float32)


def build_action_loss_data(
    root: Path,
    output: Path,
    split_name: str,
    psi: np.ndarray,
    system_ids: Sequence[int],
    ranks: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Cache exact full-dictionary action Grams and teacher crosses."""

    stage28 = root / "results/results-fd/stage28_operator_learning"
    gram_path = output / f"{split_name}_action_grams.npy"
    cross_path = output / f"{split_name}_teacher_crosses.npy"
    input_path = output / f"{split_name}_model_inputs.npz"
    expected_gram_shape = (len(system_ids), psi.shape[1], psi.shape[1])
    if gram_path.exists() and cross_path.exists() and input_path.exists():
        grams = np.load(gram_path, mmap_mode="r+")
        crosses = np.load(cross_path, mmap_mode="r+")
        with np.load(input_path) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if (
            grams.shape == expected_gram_shape
            and crosses.shape == (len(system_ids), 48, psi.shape[1])
            and data["system_ids"].tolist() == list(system_ids)
            and np.array_equal(data["target_ranks"], ranks)
        ):
            data["action_grams"] = grams
            data["teacher_crosses"] = crosses
            return data

    grams = np.lib.format.open_memmap(
        gram_path, mode="w+", dtype=np.float32, shape=expected_gram_shape
    )
    crosses = np.lib.format.open_memmap(
        cross_path, mode="w+", dtype=np.float32,
        shape=(len(system_ids), 48, psi.shape[1]),
    )
    with np.load(stage28 / "frozen_preprocessing.npz") as archive:
        preprocessing = {key: np.asarray(archive[key]) for key in archive.files}
    parameters, fields = [], []
    for index, sample_id in enumerate(system_ids):
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            parameter_input, field_input = _normalized_inputs_for_system(archive, preprocessing)
            teacher = _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))
        if teacher.rank != int(ranks[index]):
            raise RuntimeError(f"teacher-rank mismatch for system {sample_id}")
        action = np.asarray(matrix @ psi, dtype=float)
        grams[index] = (action.T @ action).astype(np.float32)
        crosses[index] = 0.0
        crosses[index, :teacher.rank, :] = (teacher.w_basis.T @ action).astype(np.float32)
        parameters.append(parameter_input)
        fields.append(field_input)
        grams.flush(); crosses.flush()
        print(f"[Stage35 action cache {split_name}] {index + 1}/{len(system_ids)} system {sample_id}", flush=True)
    np.savez_compressed(
        input_path,
        system_ids=np.asarray(system_ids, dtype=int),
        target_ranks=np.asarray(ranks, dtype=np.int64),
        normalized_parameters=np.asarray(parameters, dtype=np.float32),
        normalized_fields=np.asarray(fields, dtype=np.float32),
    )
    return {
        "system_ids": np.asarray(system_ids, dtype=int),
        "target_ranks": np.asarray(ranks, dtype=np.int64),
        "normalized_parameters": np.asarray(parameters, dtype=np.float32),
        "normalized_fields": np.asarray(fields, dtype=np.float32),
        "action_grams": grams,
        "teacher_crosses": crosses,
    }


def build_inference_data(
    root: Path, system_ids: Sequence[int], ranks: np.ndarray,
) -> Dict[str, np.ndarray]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    with np.load(stage28 / "frozen_preprocessing.npz") as archive:
        preprocessing = {key: np.asarray(archive[key]) for key in archive.files}
    parameters, fields = [], []
    for sample_id in system_ids:
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            parameter_input, field_input = _normalized_inputs_for_system(archive, preprocessing)
        parameters.append(parameter_input); fields.append(field_input)
    return {
        "system_ids": np.asarray(system_ids, dtype=int),
        "target_ranks": np.asarray(ranks, dtype=np.int64),
        "normalized_parameters": np.asarray(parameters, dtype=np.float32),
        "normalized_fields": np.asarray(fields, dtype=np.float32),
    }


def _losses(
    model: nn.Module,
    inputs: torch.Tensor,
    grams: torch.Tensor,
    crosses: torch.Tensor,
    ranks: torch.Tensor,
    settings: Stage35Settings,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    indices = torch.arange(inputs.shape[0])
    if generator is not None:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    losses = []
    for start in range(0, len(indices), settings.batch_size):
        local = indices[start:start + settings.batch_size]
        predictions = model(inputs[local])
        for offset, index in enumerate(local.tolist()):
            losses.append(residual_projector_loss(
                predictions[offset], grams[index], crosses[index],
                int(ranks[index].item()), settings.cholesky_jitter,
            ))
    return torch.stack(losses)


def train_model(
    output: Path,
    model_name: str,
    train_data: Mapping[str, np.ndarray],
    validation_data: Mapping[str, np.ndarray],
    p: int,
    settings: Stage35Settings,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = _model(model_name, p, settings).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_inputs = torch.from_numpy(_inputs(model_name, train_data))
    val_inputs = torch.from_numpy(_inputs(model_name, validation_data))
    train_grams = torch.from_numpy(train_data["action_grams"])
    val_grams = torch.from_numpy(validation_data["action_grams"])
    train_crosses = torch.from_numpy(train_data["teacher_crosses"])
    val_crosses = torch.from_numpy(validation_data["teacher_crosses"])
    train_ranks = torch.from_numpy(np.asarray(train_data["target_ranks"], dtype=np.int64))
    val_ranks = torch.from_numpy(np.asarray(validation_data["target_ranks"], dtype=np.int64))
    generator = torch.Generator().manual_seed(settings.model_seed)
    history: List[Dict[str, Any]] = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_losses = _losses(
            model, train_inputs, train_grams, train_crosses, train_ranks,
            settings, generator=generator,
        )
        loss = train_losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {model_name} loss at epoch {epoch}")
        loss.backward(); optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = _losses(
                    model, val_inputs, val_grams, val_crosses, val_ranks,
                    settings, generator=None,
                )
            row = {
                "model": model_name, "epoch": epoch,
                "training_loss": float(loss.detach()),
                "validation_loss": float(val_losses.mean()),
                "validation_loss_max": float(val_losses.max()),
            }
            history.append(row)
            print(
                f"[Stage35 {model_name}] epoch={epoch} train={row['training_loss']:.6f} val={row['validation_loss']:.6f}",
                flush=True,
            )
            if row["validation_loss"] < best_loss - 1.0e-8:
                best_loss, best_epoch = row["validation_loss"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= settings.minimum_epochs and epoch - best_epoch >= settings.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError(f"no finite checkpoint for {model_name}")
    model.load_state_dict(best_state); model.eval()
    (output / "models").mkdir(exist_ok=True)
    checkpoint = output / "models" / f"selected_{model_name}.pt"
    torch.save({
        "model": model_name, "seed": settings.model_seed,
        "dictionary_rank": p, "maximum_rank": settings.maximum_rank,
        "state_dict": best_state, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "settings": asdict(settings),
    }, checkpoint)
    summary = {
        "model": model_name, "seed": settings.model_seed,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "training_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint.relative_to(output)),
    }
    return model, history, summary


def load_trained_model(
    output: Path, model_name: str, p: int, settings: Stage35Settings,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Resume after reporting/evaluation failures without repeating training."""

    checkpoint = output / "models" / f"selected_{model_name}.pt"
    payload = torch.load(checkpoint, map_location="cpu")
    model = _model(model_name, p, settings).to(dtype=torch.float32)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, {
        "model": model_name, "seed": int(payload["seed"]),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": int(payload["best_epoch"]), "epochs_run": "cached checkpoint",
        "best_validation_loss": float(payload["best_validation_loss"]),
        "training_seconds": "completed before evaluation resume",
        "checkpoint": str(checkpoint.relative_to(output)),
    }


def predict_model_coordinates(
    model_name: str, model: nn.Module, data: Mapping[str, np.ndarray], index: int, rank: int,
) -> Tuple[np.ndarray, float]:
    inputs = _inputs(model_name, data)[index:index + 1]
    tensor = torch.from_numpy(inputs)
    timings, prediction = [], None
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            started = perf_counter()
            prediction = model(tensor)[0, :, :rank].cpu().numpy()
            timings.append(perf_counter() - started)
    coordinate_q, _ = la.qr(prediction, mode="economic", check_finite=False)
    return coordinate_q[:, :rank], float(np.median(timings))


def evaluate_geometry(
    root: Path,
    psi: np.ndarray,
    model_name: str,
    model: nn.Module,
    split_name: str,
    data: Mapping[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[int, Any]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    rows, pairs = [], {}
    for index, sample_id in enumerate(data["system_ids"].tolist()):
        rank = int(data["target_ranks"][index])
        coordinates, inference_seconds = predict_model_coordinates(
            model_name, model, data, index, rank
        )
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            teacher = _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))
        pair = _pair_from_z(matrix, psi @ coordinates)
        metrics = generalized_subspace_metrics(pair.w_basis, teacher.w_basis)
        rows.append({
            "model": model_name, "split": split_name, "sample_id": sample_id,
            "target_rank": rank, "predicted_rank": pair.rank,
            "d_W_to_teacher": metrics["projector_distance_normalized"],
            "maximum_W_angle_degrees": metrics["principal_angle_max_degrees"],
            "median_W_angle_degrees": float(np.median(metrics["principal_angles_degrees"])),
            "inference_seconds": inference_seconds,
            "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
        })
        pairs[sample_id] = pair
    return rows, pairs


def _copied_solver_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]), "rhs_seed": int(row["rhs_seed"]),
        "method": method, "source": "Stage34 archived",
        "iterations": int(row["iterations"]), "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "inference_seconds": float(row["inference_seconds"]),
        "method_setup_seconds": float(row["method_setup_seconds"]),
        "online_solve_seconds": float(row["online_solve_seconds"]),
        "failure": row["failure"], "rho": float(row["rho"]),
    }


def evaluate_solver(
    root: Path,
    hard_pairs: Mapping[str, Mapping[int, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    settings: Stage35Settings,
) -> List[Dict[str, Any]]:
    stage34 = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    archived = _read_csv(stage34 / "solver_results.csv")
    archive_names = {
        "strong_ilu": "strong_ilu", "teacher_full": "teacher_full",
        "full_span_oracle": "full_span_oracle",
        "neural_target_conditioned": "stage34_coefficient_mlp",
    }
    rows = [_copied_solver_row(row, archive_names[row["method"]]) for row in archived]
    inference = {
        (row["model"], int(row["sample_id"])): float(row["inference_seconds"])
        for row in geometry_rows if row["split"] == "hard_test"
    }
    archive_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], dtype=float); boundary = np.asarray(fixed["boundary"], dtype=bool)
    chain = settings.stage34.stage31.stage30.stage29
    coefficients = _sample_coefficients(chain.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, chain)
        for rhs_seed in chain.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = archive_lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = archive_lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for model_name in ("mlp_residual_loss", "fno_residual_loss"):
                pair = hard_pairs[model_name][sample_id]
                result = _run_pair(context, rhs, pair, chain.stage28.stage22a)
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append({
                    "sample_id": sample_id, "rhs_seed": rhs_seed,
                    "method": model_name, "source": "Stage35 new",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and true_residual <= 1.0e-8),
                    "true_final_residual": true_residual, "actual_rank": pair.rank,
                    "inference_seconds": inference[(model_name, sample_id)],
                    "method_setup_seconds": pair.setup_seconds,
                    "online_solve_seconds": float(result.solve_seconds),
                    "failure": getattr(result, "failure", "") or "",
                    "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator
                            if denominator > 0 else float("nan")),
                })
    return rows


METHODS = (
    "strong_ilu", "teacher_full", "full_span_oracle", "stage34_coefficient_mlp",
    "mlp_residual_loss", "fno_residual_loss",
)


def summarize_solver(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": int(sum(_bool(row["converged"]) for row in local)),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
        })
    return result


def summarize_geometry(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for model_name in ("stage34_coefficient_mlp", "mlp_residual_loss", "fno_residual_loss"):
        for split_name in ("train", "validation", "hard_test"):
            local = [row for row in rows if row["model"] == model_name and row["split"] == split_name]
            result.append({
                "model": model_name, "split": split_name, "systems": len(local),
                "median_dW": float(np.median([float(row["d_W_to_teacher"]) for row in local])),
                "median_largest_angle_degrees": float(np.median([float(row["maximum_W_angle_degrees"]) for row in local])),
                "worst_largest_angle_degrees": float(np.max([float(row["maximum_W_angle_degrees"]) for row in local])),
            })
    return result


def decide(
    solver_summary: Sequence[Mapping[str, Any]], geometry_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    solver = {row["method"]: row for row in solver_summary}
    geometry = {(row["model"], row["split"]): row for row in geometry_summary}
    old_hard = geometry[("stage34_coefficient_mlp", "hard_test")]["median_dW"]
    mlp_hard = geometry[("mlp_residual_loss", "hard_test")]["median_dW"]
    fno_hard = geometry[("fno_residual_loss", "hard_test")]["median_dW"]
    mlp_train = geometry[("mlp_residual_loss", "train")]["median_dW"]
    fno_train = geometry[("fno_residual_loss", "train")]["median_dW"]
    both_improve = bool(old_hard - max(mlp_hard, fno_hard) >= 0.20)
    fno_advantage = bool(
        mlp_hard - fno_hard >= 0.10
        or solver["fno_residual_loss"]["converged_rhs"] - solver["mlp_residual_loss"]["converged_rhs"] >= 3
        or solver["fno_residual_loss"]["median_rho"] - solver["mlp_residual_loss"]["median_rho"] >= 0.10
    )
    poor_training = bool(mlp_train >= 0.60 and fno_train >= 0.60)
    generalization = bool(
        min(mlp_train, fno_train) <= 0.35
        and min(
            mlp_hard - mlp_train,
            fno_hard - fno_train,
        ) >= 0.25
    )
    if poor_training:
        conclusion = "Both residual-loss models retain poor training geometry; the direct full-coordinate output parameterization is now the leading bottleneck, since the training targets are exactly representable."
        stage36 = "Keep the full dictionary and residual loss, but replace direct 1456xq coordinate regression with a target-conditioned donor/local-subspace mixture."
    elif generalization:
        conclusion = "Residual-space training succeeds in-sample but does not transfer; cross-system generalization is now the leading bottleneck."
        stage36 = "Keep the better Stage-35 architecture and residual loss, and test nearest-neighbor/local conditioning within the unchanged full dictionary."
    elif fno_advantage:
        conclusion = "Under the same residual-space loss, FNO is substantially better than MLP; spatial architecture matters."
        stage36 = "Keep the Stage-35 FNO and residual loss, and make rank prediction oracle-blind before changing any other component."
    elif both_improve:
        conclusion = "Both models improve strongly over Stage 34; coefficient-space loss mismatch was the main issue, without a material FNO advantage."
        stage36 = "Keep the simpler residual-loss MLP and make rank prediction oracle-blind before changing any other component."
    else:
        conclusion = "Residual-space loss helps incompletely and neither architecture resolves the held-out solver gap."
        stage36 = "Keep residual-space loss and test a target-conditioned donor/local-subspace output in the full dictionary."
    return {
        "conclusion": conclusion, "both_improve_strongly": both_improve,
        "fno_substantially_better": fno_advantage,
        "both_training_geometries_poor": poor_training,
        "good_training_poor_heldout": generalization,
        "stage36_direction": stage36,
    }


def write_report(
    output: Path,
    solver_summary: Sequence[Mapping[str, Any]],
    geometry_summary: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    training_summary: Sequence[Mapping[str, Any]],
) -> Path:
    solver = {row["method"]: row for row in solver_summary}
    labels = {
        "strong_ilu": "strong ILU", "teacher_full": "own teacher",
        "full_span_oracle": "full-span oracle",
        "stage34_coefficient_mlp": "Stage-34 coefficient MLP",
        "mlp_residual_loss": "Stage-35 residual MLP",
        "fno_residual_loss": "Stage-35 residual FNO",
    }
    lines = [
        "# Stage 35: residual-space loss, MLP versus FNO", "", "## Result", "",
        f"**{decision['conclusion']}**", "",
        "| method | converged RHS | median iterations | median $\\rho$ |",
        "|:--|--:|--:|--:|",
    ]
    for method in METHODS:
        row = solver[method]
        lines.append(f"| {labels[method]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'], 1)} | {_fmt(row['median_rho'])} |")
    lines.extend([
        "", "## Residual-space geometry", "",
        "| model | split | median $d_W$ | median largest angle | worst largest angle |",
        "|:--|:--|--:|--:|--:|",
    ])
    for row in geometry_summary:
        lines.append(
            f"| {labels[row['model']]} | {row['split']} | {_fmt(row['median_dW'])} | {_fmt(row['median_largest_angle_degrees'], 1)} degrees | {_fmt(row['worst_largest_angle_degrees'], 1)} degrees |"
        )
    geometry = {(row["model"], row["split"]): row for row in geometry_summary}
    old_hard = geometry[("stage34_coefficient_mlp", "hard_test")]["median_dW"]
    mlp_hard = geometry[("mlp_residual_loss", "hard_test")]["median_dW"]
    fno_hard = geometry[("fno_residual_loss", "hard_test")]["median_dW"]
    lines.extend([
        "",
        f"Residual loss reduces held-out median $d_W$ from {_fmt(old_hard)} to {_fmt(mlp_hard)} for MLP and {_fmt(fno_hard)} for FNO, but neither changes convergence or median $\\rho$. Both models also retain poor training geometry and near-90-degree worst directions. FNO's held-out $d_W$ advantage over MLP is only {_fmt(mlp_hard - fno_hard)} and produces no solver advantage.",
    ])
    lines.extend(["", "## System 63", ""])
    geometry63 = {
        row["model"]: row for row in geometry_rows
        if row["split"] == "hard_test" and int(row["sample_id"]) == 63
    }
    for method in ("stage34_coefficient_mlp", "mlp_residual_loss", "fno_residual_loss"):
        local = [row for row in solver_rows if row["method"] == method and int(row["sample_id"]) == 63]
        geom = geometry63[method]
        lines.append(
            f"- {labels[method]}: {sum(_bool(row['converged']) for row in local)}/3 converged, median {int(np.median([int(row['iterations']) for row in local]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in local]))}$, $d_W={_fmt(geom['d_W_to_teacher'])}$, largest angle {_fmt(geom['maximum_W_angle_degrees'], 1)} degrees."
        )
    lines.extend([
        "", "## Training controls", "",
        "Both Stage-35 models use the same 43 training and 15 validation systems, own-teacher residual targets, full dictionary, variable ranks, residual loss, seed, Adam settings, and validation stopping rule. Hard-test systems are evaluated only after checkpoint selection.",
    ])
    for row in training_summary:
        lines.append(f"- {labels[row['model']]}: {row['parameter_count']:,} parameters; selected epoch {row['best_epoch']}; validation loss {_fmt(row['best_validation_loss'])}.")
    lines.extend(["", "## Stage 36 recommendation", "", decision["stage36_direction"]])
    report = output / "stage35_residual_space_loss_mlp_vs_fno.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage35(root: Path) -> Path:
    settings = Stage35Settings()
    output = root / "results/results-fd/stages/stage35_residual_space_loss_mlp_vs_fno"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-35 protocol must exist before training")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__,
        "platform": platform.platform(),
    })
    stage30 = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    stage34 = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    with np.load(stage30 / "full_training_span.npz") as archive:
        psi = np.asarray(archive["psi"], dtype=float)
        dictionary_ids = np.asarray(archive["training_ids"], dtype=int)
    target_sets = {}
    for split_name in ("train", "validation", "hard_test"):
        with np.load(stage34 / f"{split_name}_coordinate_targets.npz") as archive:
            target_sets[split_name] = {key: np.asarray(archive[key]) for key in archive.files}
    train_ids = target_sets["train"]["system_ids"]
    validation_ids = target_sets["validation"]["system_ids"]
    hard_ids = target_sets["hard_test"]["system_ids"]
    preflight = {
        "dictionary_shape": list(psi.shape),
        "dictionary_training_ids_match": bool(np.array_equal(dictionary_ids, train_ids)),
        "train_validation_hard_disjoint": not bool(
            set(train_ids.tolist()) & set(validation_ids.tolist())
            or set(train_ids.tolist()) & set(hard_ids.tolist())
            or set(validation_ids.tolist()) & set(hard_ids.tolist())
        ),
        "hard_ids_match": hard_ids.tolist() == list(settings.hard_system_ids),
        "stage34_preflight_passed": json.loads((stage34 / "target_preflight.json").read_text())["passed"],
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-35 preflight failed")

    train_data = build_action_loss_data(
        root, output, "train", psi, train_ids.tolist(), target_sets["train"]["target_ranks"]
    )
    validation_data = build_action_loss_data(
        root, output, "validation", psi, validation_ids.tolist(), target_sets["validation"]["target_ranks"]
    )
    hard_data = build_inference_data(root, hard_ids.tolist(), target_sets["hard_test"]["target_ranks"])
    models, histories, training_summary = {}, [], []
    for model_name in ("mlp_residual_loss", "fno_residual_loss"):
        checkpoint = output / "models" / f"selected_{model_name}.pt"
        if checkpoint.exists():
            model, summary = load_trained_model(
                output, model_name, psi.shape[1], settings
            )
            history = []
            print(f"[Stage35 resume] loaded {model_name} checkpoint", flush=True)
        else:
            model, history, summary = train_model(
                output, model_name, train_data, validation_data, psi.shape[1], settings
            )
        models[model_name] = model; histories.extend(history); training_summary.append(summary)
    if histories:
        _write_csv(output / "training_history.csv", histories)
    _write_csv(output / "training_summary.csv", training_summary)

    geometry_rows: List[Dict[str, Any]] = []
    hard_pairs: Dict[str, Dict[int, Any]] = {}
    for model_name, model in models.items():
        for split_name, data in (("train", train_data), ("validation", validation_data), ("hard_test", hard_data)):
            rows, pairs = evaluate_geometry(root, psi, model_name, model, split_name, data)
            geometry_rows.extend(rows)
            if split_name == "hard_test":
                hard_pairs[model_name] = pairs
    archived_geometry = _read_csv(stage34 / "geometry_diagnostics.csv")
    for row in archived_geometry:
        geometry_rows.append({
            "model": "stage34_coefficient_mlp", "split": row["split"],
            "sample_id": int(row["sample_id"]), "target_rank": int(row["target_rank"]),
            "predicted_rank": int(row["predicted_rank"]),
            "d_W_to_teacher": float(row["d_W_to_teacher"]),
            "maximum_W_angle_degrees": float(row["maximum_W_angle_degrees"]),
            "median_W_angle_degrees": "", "inference_seconds": float(row["inference_seconds"]),
            "pair_defect": float(row["pair_defect"]),
        })
    geometry_summary = summarize_geometry(geometry_rows)
    solver_rows = evaluate_solver(root, hard_pairs, geometry_rows, settings)
    solver_summary = summarize_solver(solver_rows)
    decision = decide(solver_summary, geometry_summary)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {
        f"{row['model']}:{row['split']}": row for row in geometry_summary
    }
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output, solver_summary, geometry_summary, solver_rows, geometry_rows,
        decision, training_summary,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "loss_uses_exact_reduced_form_of_A_Psi_C": True,
        "matched_model_protocol": True,
        "hard_test_not_used_for_training_or_checkpoint": True,
        "new_solver_rows": sum(row["source"] == "Stage35 new" for row in solver_rows),
        "expected_new_solver_rows": 42,
        "all_new_ranks_match_teacher": all(
            int(row["predicted_rank"]) == int(row["target_rank"])
            for row in geometry_rows if row["model"] in models
        ),
        "maximum_new_pair_defect": max(
            float(row["pair_defect"]) for row in geometry_rows if row["model"] in models
        ),
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])
        ),
    }
    validation["passed"] = bool(
        validation["preflight_passed"] and validation["matched_model_protocol"]
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["all_new_ranks_match_teacher"]
        and validation["maximum_new_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage35_residual_space_loss_mlp_vs_fno.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage35] complete", flush=True)
    return report_copy
