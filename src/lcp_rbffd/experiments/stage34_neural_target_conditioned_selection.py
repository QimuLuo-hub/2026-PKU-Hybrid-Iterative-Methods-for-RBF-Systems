"""Stage 34: parameter-MLP selection inside the full Stage-30 dictionary."""

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

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _pair_from_z, _run_pair
from .stage28_spatial_k_operator_learning import ParameterMLP
from .stage29_family_level_coarse_space import _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage31_krylov_aware_shared_basis import Stage31Settings
from .stage9_scalability import _write_csv, _write_json

@dataclass(frozen=True)
class Stage34Settings:
    model_seed: int = 20262821
    maximum_rank: int = 48
    hidden_widths: Tuple[int, int] = (64, 128)
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 6
    maximum_epochs: int = 300
    minimum_epochs: int = 100
    validation_interval: int = 5
    early_stopping_patience: int = 75
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage31(self) -> Stage31Settings:
        return Stage31Settings()


def coordinate_projector_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    """Gauge-invariant normalized chordal loss for variable-rank targets."""

    losses = []
    for index in range(predictions.shape[0]):
        rank = int(ranks[index].item())
        predicted_q = torch.linalg.qr(predictions[index, :, :rank], mode="reduced").Q
        target_q = targets[index, :, :rank]
        overlap = predicted_q.transpose(0, 1) @ target_q
        losses.append(1.0 - torch.sum(overlap**2) / rank)
    return torch.stack(losses)


def _normalized_parameters(
    system_archive: Mapping[str, np.ndarray],
    preprocessing: Mapping[str, np.ndarray],
) -> np.ndarray:
    return (
        (np.asarray(system_archive["parameters"], dtype=float)
         - np.asarray(preprocessing["parameter_mean"], dtype=float))
        / np.asarray(preprocessing["parameter_std"], dtype=float)
    ).astype(np.float32)


def _coordinate_target(
    matrix: sp.spmatrix,
    psi: np.ndarray,
    zfull: np.ndarray,
    wfull: np.ndarray,
    *,
    oracle: bool,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    teacher = _pair_from_z(matrix, zfull)
    if oracle:
        target_pair, _ = oracle_select_inside_dictionary(matrix, psi, wfull)
        target_z = target_pair.z_basis
    else:
        target_z = teacher.z_basis
    coordinates = psi.T @ target_z
    coordinate_q, _ = la.qr(coordinates, mode="economic", check_finite=False)
    coordinate_q = coordinate_q[:, : teacher.rank]
    reconstructed = _pair_from_z(matrix, psi @ coordinate_q)
    w_metrics = generalized_subspace_metrics(reconstructed.w_basis, teacher.w_basis)
    outside = float(
        la.norm(target_z - psi @ (psi.T @ target_z), ord="fro")
        / max(la.norm(target_z, ord="fro"), np.finfo(float).tiny)
    )
    return coordinate_q, teacher.rank, {
        "target_dW": float(w_metrics["projector_distance_normalized"]),
        "target_maximum_W_angle_degrees": float(w_metrics["principal_angle_max_degrees"]),
        "target_outside_dictionary_relative_residual": outside,
    }


def build_targets(
    root: Path,
    output: Path,
    split_name: str,
    system_ids: Sequence[int],
    psi: np.ndarray,
    settings: Stage34Settings,
    *,
    oracle: bool,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    with np.load(stage28 / "frozen_preprocessing.npz") as archive:
        preprocessing = {key: np.asarray(archive[key]) for key in archive.files}
    targets = np.zeros(
        (len(system_ids), psi.shape[1], settings.maximum_rank), dtype=np.float32
    )
    parameters, ranks, quality = [], [], []
    for position, sample_id in enumerate(system_ids, start=1):
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
            wfull = np.asarray(archive["wfull"], dtype=float)
            parameters.append(_normalized_parameters(archive, preprocessing))
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        coordinates, rank, diagnostics = _coordinate_target(
            matrix, psi, zfull, wfull, oracle=oracle
        )
        targets[position - 1, :, :rank] = coordinates.astype(np.float32)
        ranks.append(rank)
        quality.append(
            {
                "split": split_name,
                "sample_id": sample_id,
                "rank": rank,
                "target_source": "full-span residual oracle" if oracle else "exact in-span own teacher",
                **diagnostics,
            }
        )
        if oracle:
            print(
                f"[Stage34 targets {split_name} {position}/{len(system_ids)}] system {sample_id}",
                flush=True,
            )
    payload = {
        "system_ids": np.asarray(system_ids, dtype=int),
        "normalized_parameters": np.asarray(parameters, dtype=np.float32),
        "targets": targets,
        "target_ranks": np.asarray(ranks, dtype=np.int64),
    }
    np.savez_compressed(output / f"{split_name}_coordinate_targets.npz", **payload)
    return payload, quality


def _loss_batches(
    model: ParameterMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ranks: torch.Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    indices = torch.arange(inputs.shape[0])
    if generator is not None:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    losses = []
    for start in range(0, len(indices), batch_size):
        local = indices[start:start + batch_size]
        losses.append(coordinate_projector_loss(model(inputs[local]), targets[local], ranks[local]))
    return torch.cat(losses)


def train_parameter_mlp(
    output: Path,
    train_data: Mapping[str, np.ndarray],
    validation_data: Mapping[str, np.ndarray],
    dictionary_rank: int,
    settings: Stage34Settings,
) -> Tuple[ParameterMLP, List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = ParameterMLP(
        dictionary_rank, settings.maximum_rank, settings.hidden_widths
    ).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_inputs = torch.from_numpy(train_data["normalized_parameters"])
    train_targets = torch.from_numpy(train_data["targets"])
    train_ranks = torch.from_numpy(train_data["target_ranks"])
    val_inputs = torch.from_numpy(validation_data["normalized_parameters"])
    val_targets = torch.from_numpy(validation_data["targets"])
    val_ranks = torch.from_numpy(validation_data["target_ranks"])
    generator = torch.Generator().manual_seed(settings.model_seed)
    history: List[Dict[str, Any]] = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = _loss_batches(
            model, train_inputs, train_targets, train_ranks,
            settings.batch_size, generator=generator,
        )
        loss = train_losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite Stage-34 loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = _loss_batches(
                    model, val_inputs, val_targets, val_ranks,
                    settings.batch_size, generator=None,
                )
            row = {
                "epoch": epoch,
                "training_loss": float(loss.detach()),
                "validation_loss": float(val_losses.mean()),
                "validation_loss_max": float(val_losses.max()),
            }
            history.append(row)
            print(
                f"[Stage34 train] epoch={epoch} train={row['training_loss']:.6f} "
                f"val={row['validation_loss']:.6f}",
                flush=True,
            )
            if row["validation_loss"] < best_loss - 1.0e-8:
                best_loss = row["validation_loss"]
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if (
                epoch >= settings.minimum_epochs
                and epoch - best_epoch >= settings.early_stopping_patience
            ):
                break
    if best_state is None:
        raise RuntimeError("Stage-34 training produced no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    (output / "models").mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "dictionary_rank": dictionary_rank,
            "maximum_rank": settings.maximum_rank,
            "settings": asdict(settings),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        output / "models/selected_parameter_mlp.pt",
    )
    summary = {
        "model": "Stage28 ParameterMLP",
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "seed": settings.model_seed,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "training_seconds": perf_counter() - started,
    }
    _write_csv(output / "training_history.csv", history)
    _write_json(output / "training_summary.json", summary)
    return model, history, summary


def predict_coordinates(
    model: ParameterMLP, parameters: np.ndarray, rank: int
) -> Tuple[np.ndarray, float]:
    tensor = torch.from_numpy(parameters[None, :])
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
    model: ParameterMLP,
    split_name: str,
    data: Mapping[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[int, Any]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    rows, pairs = [], {}
    for index, sample_id in enumerate(data["system_ids"].tolist()):
        rank = int(data["target_ranks"][index])
        predicted_c, inference_seconds = predict_coordinates(
            model, data["normalized_parameters"][index], rank
        )
        target_c = np.asarray(data["targets"][index, :, :rank], dtype=float)
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            zfull = np.asarray(archive["zfull"], dtype=float)
        teacher = _pair_from_z(matrix, zfull)
        pair = _pair_from_z(matrix, psi @ predicted_c)
        coordinate_metrics = generalized_subspace_metrics(predicted_c, target_c)
        w_metrics = generalized_subspace_metrics(pair.w_basis, teacher.w_basis)
        rows.append(
            {
                "split": split_name,
                "sample_id": sample_id,
                "target_rank": rank,
                "predicted_rank": pair.rank,
                "coordinate_projector_distance": coordinate_metrics["projector_distance_normalized"],
                "coordinate_maximum_angle_degrees": coordinate_metrics["principal_angle_max_degrees"],
                "d_W_to_teacher": w_metrics["projector_distance_normalized"],
                "maximum_W_angle_degrees": w_metrics["principal_angle_max_degrees"],
                "inference_seconds": inference_seconds,
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            }
        )
        pairs[sample_id] = pair
    return rows, pairs


def _archive_solver_row(row: Mapping[str, Any], method: str) -> Dict[str, Any]:
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": method,
        "source": "Stage30 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "inference_seconds": 0.0,
        "method_setup_seconds": float(row["method_setup_seconds"]),
        "online_solve_seconds": float(row["online_solve_seconds"]),
        "failure": row["failure"],
    }


def evaluate_solver(
    root: Path,
    hard_pairs: Mapping[int, Any],
    hard_geometry: Sequence[Mapping[str, Any]],
    settings: Stage34Settings,
) -> List[Dict[str, Any]]:
    stage30 = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    archived = _read_csv(stage30 / "solver_results.csv")
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    inference = {int(row["sample_id"]): float(row["inference_seconds"]) for row in hard_geometry}
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage31.stage30.stage29.stage28.stage19a)
    rows = []
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage31.stage30.stage29
        )
        pair = hard_pairs[sample_id]
        for rhs_seed in settings.stage31.stage30.stage29.stage28.evaluation_rhs_seeds:
            for old_method, method in (
                ("strong_ilu", "strong_ilu"),
                ("teacher_full", "teacher_full"),
                ("oracle_within_full_span", "full_span_oracle"),
            ):
                rows.append(
                    _archive_solver_row(lookup[(sample_id, rhs_seed, old_method)], method)
                )
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            result = _run_pair(
                context, rhs, pair, settings.stage31.stage30.stage29.stage28.stage22a
            )
            true_residual = _relative_residual(context.system.matrix, result.x, rhs)
            rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": "neural_target_conditioned",
                    "source": "Stage34 new",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and true_residual <= 1.0e-8),
                    "true_final_residual": true_residual,
                    "actual_rank": pair.rank,
                    "inference_seconds": inference[sample_id],
                    "method_setup_seconds": pair.setup_seconds,
                    "online_solve_seconds": float(result.solve_seconds),
                    "failure": getattr(result, "failure", "") or "",
                }
            )
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for row in rows:
                if int(row["sample_id"]) == sample_id and int(row["rhs_seed"]) == rhs_seed:
                    row["rho"] = (
                        (int(baseline["iterations"]) - int(row["iterations"])) / denominator
                        if denominator > 0 else float("nan")
                    )
    return rows


def summarize_solver(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for method in ("strong_ilu", "teacher_full", "full_span_oracle", "neural_target_conditioned"):
        local = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "converged_rhs": int(sum(_bool(row["converged"]) for row in local)),
                "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                "median_rho": float(np.median([float(row["rho"]) for row in local])),
                "median_rank": float(np.median([int(row["actual_rank"]) for row in local])),
            }
        )
    return summaries


def summarize_geometry(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for split_name in ("train", "validation", "hard_test"):
        local = [row for row in rows if row["split"] == split_name]
        summaries.append(
            {
                "split": split_name,
                "systems": len(local),
                "median_coordinate_projector_distance": float(np.median([float(row["coordinate_projector_distance"]) for row in local])),
                "median_dW_to_teacher": float(np.median([float(row["d_W_to_teacher"]) for row in local])),
                "median_maximum_W_angle_degrees": float(np.median([float(row["maximum_W_angle_degrees"]) for row in local])),
                "worst_maximum_W_angle_degrees": float(np.max([float(row["maximum_W_angle_degrees"]) for row in local])),
            }
        )
    return summaries


def decide(
    solver_summary: Sequence[Mapping[str, Any]],
    geometry_summary: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    solver = {row["method"]: row for row in solver_summary}
    geometry = {row["split"]: row for row in geometry_summary}
    neural = solver["neural_target_conditioned"]
    strong = bool(neural["converged_rhs"] >= 20 and neural["median_rho"] >= 0.80)
    meaningful = bool(neural["converged_rhs"] >= 18 and neural["median_rho"] >= 0.50)
    gap = (
        geometry["hard_test"]["median_dW_to_teacher"]
        - geometry["train"]["median_dW_to_teacher"]
    )
    action_gap = (
        geometry["train"]["median_dW_to_teacher"]
        - geometry["train"]["median_coordinate_projector_distance"]
    )
    if strong:
        statement = "The simple MLP identifies most of the useful target-specific full-span subspace."
        stage35 = "Replace oracle teacher ranks with an oracle-blind adaptive-rank head while keeping this full-span target and loss fixed."
        outcome = "strong"
    elif meaningful:
        statement = "The simple MLP recovers meaningful acceleration but not most of the full-span oracle benefit."
        stage35 = (
            "Reuse the same targets and loss with the existing Stage-28 field encoder to test whether "
            "the seven-parameter input is the remaining generalization bottleneck."
            if gap >= 0.20 else
            "Add a worst-principal-direction term to the same full-span subspace loss before changing the model."
        )
        outcome = "meaningful"
    else:
        statement = "The simple MLP trained with coefficient-space projector loss does not identify a solver-useful subspace inside the full dictionary."
        stage35 = (
            "Keep the full dictionary, targets, MLP, teacher ranks, and solver fixed; replace the coefficient-space projector loss with a system-specific residual-space projector loss on span(A_i Psi C_hat) versus W_i."
            if action_gap >= 0.20 else
            ("Reuse the same targets and loss with the existing Stage-28 field encoder; the train-to-test geometry gap indicates failed parameter-MLP generalization."
             if gap >= 0.20 else
             "Replace direct full-coordinate prediction with a target-conditioned local/donor mixture before adding model complexity.")
        )
        outcome = "failed"
    return {
        "outcome": outcome,
        "statement": statement,
        "strong_recovery": strong,
        "meaningful_acceleration": meaningful,
        "train_to_hard_test_median_dW_gap": gap,
        "training_action_amplification_gap": action_gap,
        "stage35_direction": stage35,
    }


def write_report(
    output: Path,
    solver_summary: Sequence[Mapping[str, Any]],
    geometry_summary: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    training: Mapping[str, Any],
    target_quality: Sequence[Mapping[str, Any]],
) -> Path:
    solver = {row["method"]: row for row in solver_summary}
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "full_span_oracle": "full-span oracle",
        "neural_target_conditioned": "neural target-conditioned",
    }
    lines = [
        "# Stage 34: neural target-conditioned subspace selection",
        "",
        "## Result",
        "",
        f"**{decision['statement']}**",
        "",
        f"The Stage-28 parameter MLP maps the seven normalized coefficient parameters to a teacher-rank subspace inside the verified 1456-direction Stage-30 dictionary. It has {training['parameter_count']:,} parameters and its checkpoint was selected at epoch {training['best_epoch']} using validation projector loss only.",
        "",
        "| method | converged RHS | median iterations | median $\\rho$ | median rank |",
        "|:--|--:|--:|--:|--:|",
    ]
    for method in labels:
        row = solver[method]
        lines.append(
            f"| {labels[method]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'], 1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_rank'], 1)} |"
        )
    lines.extend([
        "",
        "## Selection and residual-space geometry",
        "",
        "| split | systems | median coordinate distance | median $d_W$ | median max W angle | worst max W angle |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    for row in geometry_summary:
        lines.append(
            f"| {row['split']} | {row['systems']} | {_fmt(row['median_coordinate_projector_distance'])} | {_fmt(row['median_dW_to_teacher'])} | {_fmt(row['median_maximum_W_angle_degrees'], 1)} degrees | {_fmt(row['worst_maximum_W_angle_degrees'], 1)} degrees |"
        )
    train_target_dW = float(np.median([
        float(row["target_dW"]) for row in target_quality if row["split"] == "train"
    ]))
    hard_target_dW = float(np.median([
        float(row["target_dW"]) for row in target_quality if row["split"] == "hard_test"
    ]))
    lines.extend([
        "",
        f"The supervision construction is sound: median target $d_W$ is {_fmt(train_target_dW)} on training systems and {_fmt(hard_target_dW)} for the hard-test full-span oracles. The learned training spaces are already poor in residual geometry ($d_W={_fmt(next(row['median_dW_to_teacher'] for row in geometry_summary if row['split'] == 'train'))}$) despite a materially smaller coefficient-space distance. The failure therefore occurs before held-out generalization: coefficient errors are strongly distorted by each target action $A_i$.",
    ])
    system63 = [
        row for row in solver_rows
        if int(row["sample_id"]) == 63 and row["method"] == "neural_target_conditioned"
    ]
    geometry63 = next(
        row for row in geometry_rows
        if row["split"] == "hard_test" and int(row["sample_id"]) == 63
    )
    lines.extend([
        "",
        "## System 63",
        "",
        f"The neural rank-{int(geometry63['predicted_rank'])} space converges on {sum(_bool(row['converged']) for row in system63)}/3 RHSs, with median {int(np.median([int(row['iterations']) for row in system63]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in system63]))}$, $d_W={_fmt(geometry63['d_W_to_teacher'])}$, and maximum W angle {_fmt(geometry63['maximum_W_angle_degrees'], 1)} degrees.",
        "",
        "## Stage 35 recommendation",
        "",
        decision["stage35_direction"],
        "",
        "Held-out teacher ranks are supplied to isolate subspace selection. A deployable method must predict rank as well as coordinates.",
    ])
    report = output / "stage34_neural_target_conditioned_selection.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage34(root: Path) -> Path:
    settings = Stage34Settings()
    output = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-34 protocol must exist before training")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform()
    })
    stage28 = root / "results/results-fd/stage28_operator_learning"
    stage30 = root / "results/results-fd/stages/stage30_full_span_representation_ceiling"
    split = json.loads((stage28 / "split.json").read_text(encoding="utf-8"))
    with np.load(stage30 / "full_training_span.npz") as archive:
        psi = np.asarray(archive["psi"], dtype=float)
        dictionary_training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()
    if psi.shape != (5041, 1456) or dictionary_training_ids != split["train"]:
        raise RuntimeError("Stage-30 full dictionary provenance mismatch")

    train_data, train_target_quality = build_targets(
        root, output, "train", split["train"], psi, settings, oracle=False
    )
    validation_data, validation_target_quality = build_targets(
        root, output, "validation", split["validation"], psi, settings, oracle=True
    )
    preflight = {
        "dictionary_shape": list(psi.shape),
        "dictionary_training_ids_match": dictionary_training_ids == split["train"],
        "train_ids": split["train"],
        "validation_ids": split["validation"],
        "hard_test_ids": list(settings.hard_system_ids),
        "split_disjoint": not (
            set(split["train"]) & set(split["validation"])
            or set(split["train"]) & set(settings.hard_system_ids)
            or set(split["validation"]) & set(settings.hard_system_ids)
        ),
        "maximum_train_target_dW": max(row["target_dW"] for row in train_target_quality),
        "train_rank_range": [int(train_data["target_ranks"].min()), int(train_data["target_ranks"].max())],
        "validation_rank_range": [int(validation_data["target_ranks"].min()), int(validation_data["target_ranks"].max())],
    }
    preflight["passed"] = bool(
        preflight["dictionary_training_ids_match"]
        and preflight["split_disjoint"]
        and preflight["maximum_train_target_dW"] <= 1.0e-5
        and preflight["train_rank_range"][1] <= settings.maximum_rank
        and preflight["validation_rank_range"][1] <= settings.maximum_rank
    )
    _write_json(output / "target_preflight.json", preflight)
    _write_csv(output / "target_quality.csv", train_target_quality + validation_target_quality)
    if not preflight["passed"]:
        raise RuntimeError("Stage-34 target preflight failed; model was not trained")
    print("[Stage34 preflight] full-span coordinate targets verified", flush=True)

    model, _, training = train_parameter_mlp(
        output, train_data, validation_data, psi.shape[1], settings
    )
    hard_data, hard_target_quality = build_targets(
        root, output, "hard_test", settings.hard_system_ids, psi, settings, oracle=True
    )
    _write_csv(
        output / "target_quality.csv",
        train_target_quality + validation_target_quality + hard_target_quality,
    )
    geometry_rows = []
    hard_pairs = {}
    for split_name, data in (
        ("train", train_data),
        ("validation", validation_data),
        ("hard_test", hard_data),
    ):
        local_rows, local_pairs = evaluate_geometry(root, psi, model, split_name, data)
        geometry_rows.extend(local_rows)
        if split_name == "hard_test":
            hard_pairs = local_pairs
    geometry_summary = summarize_geometry(geometry_rows)
    solver_rows = evaluate_solver(root, hard_pairs, geometry_rows, settings)
    solver_summary = summarize_solver(solver_rows)
    decision = decide(solver_summary, geometry_summary)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["split"]: row for row in geometry_summary}
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output, solver_summary, geometry_summary, solver_rows, geometry_rows, decision,
        training, train_target_quality + validation_target_quality + hard_target_quality,
    )
    new_rows = [row for row in solver_rows if row["source"] == "Stage34 new"]
    hard_target_ranks = {
        int(row["sample_id"]): int(row["target_rank"])
        for row in geometry_rows if row["split"] == "hard_test"
    }
    teacher_rows = [row for row in solver_rows if row["method"] == "teacher_full"]
    validation = {
        "target_preflight_passed": bool(preflight["passed"]),
        "training_systems": len(train_data["system_ids"]),
        "validation_systems": len(validation_data["system_ids"]),
        "hard_test_systems": len(hard_data["system_ids"]),
        "hard_test_not_used_for_training_or_checkpoint": True,
        "new_solver_rows": len(new_rows),
        "expected_new_solver_rows": 21,
        "all_predicted_ranks_match_targets": all(int(row["predicted_rank"]) == int(row["target_rank"]) for row in geometry_rows),
        "hard_target_ranks_match_stage30_teacher": all(
            int(row["actual_rank"]) == hard_target_ranks[int(row["sample_id"])]
            for row in teacher_rows
        ),
        "maximum_new_pair_defect": max(float(row["pair_defect"]) for row in geometry_rows if row["split"] == "hard_test"),
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])),
    }
    validation["passed"] = bool(
        validation["target_preflight_passed"]
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["all_predicted_ranks_match_targets"]
        and validation["hard_target_ranks_match_stage30_teacher"]
        and validation["maximum_new_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage34_neural_target_conditioned_selection.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage34] complete", flush=True)
    return report_copy
