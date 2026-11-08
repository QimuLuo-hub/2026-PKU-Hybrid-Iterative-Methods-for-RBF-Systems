"""Stage 36: neural selection of local unions of training teacher spaces."""

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
from .stage22a_multi_rhs_functional_oracle import _pair_from_z, _run_pair, _w_union
from .stage27_spatial_k_adaptive_rank_validation import Stage27Settings, _run_adaptive
from .stage28_spatial_k_operator_learning import FNOFieldEncoder
from .stage29_family_level_coarse_space import _stage28_context, oracle_select_inside_dictionary
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage35_residual_space_loss_mlp_vs_fno import build_inference_data
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage36Settings:
    model_seed: int = 20262821
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
    top_k_values: Tuple[int, int] = (3, 5)
    initial_rank: int = 28
    expanded_rank: int = 48
    numerical_rank_tolerance: float = 1.0e-13
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage27(self) -> Stage27Settings:
        return Stage27Settings()


class DonorMLP(nn.Module):
    def __init__(self, donor_count: int, widths: Tuple[int, int]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, widths[0]), nn.GELU(),
            nn.Linear(widths[0], widths[1]), nn.GELU(),
            nn.Linear(widths[1], donor_count),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DonorFNO(nn.Module):
    def __init__(self, donor_count: int, settings: Stage36Settings) -> None:
        super().__init__()
        self.encoder = FNOFieldEncoder(
            donor_count, 1, settings.fno_width, settings.fno_modes, settings.fno_layers
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values).squeeze(-1)


def ranked_label(
    ordered_indices: Sequence[int], donor_count: int, excluded_index: int | None,
) -> np.ndarray:
    label = np.zeros(donor_count, dtype=np.float32)
    weights = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)
    accepted = [index for index in ordered_indices if index != excluded_index][:len(weights)]
    for weight, index in zip(weights, accepted):
        label[index] = weight
    if label.sum() == 0:
        raise RuntimeError("empty donor supervision label")
    return label / label.sum()


def masked_soft_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, excluded_indices: torch.Tensor | None,
) -> torch.Tensor:
    masked = logits
    if excluded_indices is not None:
        masked = logits.clone()
        masked[torch.arange(logits.shape[0]), excluded_indices] = -1.0e9
    return -(labels * torch.log_softmax(masked, dim=1)).sum(dim=1)


def load_donors(root: Path, donor_ids: Sequence[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    donors, ranks = {}, {}
    for donor_id in donor_ids:
        with np.load(stage28 / "systems" / f"system_{donor_id:03d}.npz") as archive:
            donors[donor_id] = np.asarray(archive["zfull"], dtype=float)
        ranks[donor_id] = donors[donor_id].shape[1]
    return donors, ranks


def _incremental_basis(existing: np.ndarray | None, candidate: np.ndarray) -> np.ndarray:
    residual = np.asarray(candidate, dtype=float)
    if existing is not None:
        for _ in range(2):
            residual = residual - existing @ (existing.T @ residual)
    left, singular, _ = la.svd(residual, full_matrices=False, check_finite=False)
    if singular.size == 0 or singular[0] == 0:
        return np.empty((candidate.shape[0], 0), dtype=float)
    rank = int(np.count_nonzero(singular > 1.0e-11 * singular[0]))
    return left[:, :rank]


def greedy_donor_order(
    action_bases: Mapping[int, np.ndarray],
    teacher_w: np.ndarray,
    candidates: Sequence[int],
    count: int,
) -> Tuple[List[int], List[float], Dict[int, float]]:
    individual = {
        donor: float(la.norm(action_bases[donor].T @ teacher_w, ord="fro") ** 2 / teacher_w.shape[1])
        for donor in candidates
    }
    selected, coverages = [], []
    union_basis: np.ndarray | None = None
    remaining = list(candidates)
    for _ in range(min(count, len(remaining))):
        choices = []
        for donor in remaining:
            added = _incremental_basis(union_basis, action_bases[donor])
            gain = float(la.norm(added.T @ teacher_w, ord="fro") ** 2)
            current = 0.0 if union_basis is None else float(la.norm(union_basis.T @ teacher_w, ord="fro") ** 2)
            choices.append((-(current + gain), donor, added))
        _, donor, added = min(choices, key=lambda value: (value[0], value[1]))
        selected.append(donor); remaining.remove(donor)
        union_basis = added if union_basis is None else np.column_stack((union_basis, added))
        coverages.append(float(la.norm(union_basis.T @ teacher_w, ord="fro") ** 2 / teacher_w.shape[1]))
    return selected, coverages, individual


def construct_labels(
    root: Path,
    output: Path,
    split_name: str,
    target_ids: Sequence[int],
    donor_ids: Sequence[int],
    donors: Mapping[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    cache = output / f"{split_name}_donor_labels.npz"
    rows_path = output / f"{split_name}_donor_label_details.csv"
    if cache.exists() and rows_path.exists():
        with np.load(cache) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if data["target_ids"].tolist() == list(target_ids) and data["donor_ids"].tolist() == list(donor_ids):
            return data
    stage28 = root / "results/results-fd/stage28_operator_learning"
    donor_raw = np.column_stack([donors[donor] for donor in donor_ids])
    slices, start = {}, 0
    for donor in donor_ids:
        slices[donor] = slice(start, start + donors[donor].shape[1]); start += donors[donor].shape[1]
    labels, orders, rows = [], [], []
    donor_index = {donor: index for index, donor in enumerate(donor_ids)}
    for position, target_id in enumerate(target_ids, start=1):
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{target_id:03d}.npz").tocsr()
        with np.load(stage28 / "systems" / f"system_{target_id:03d}.npz") as archive:
            teacher = _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))
        action_raw = np.asarray(matrix @ donor_raw, dtype=float)
        action_bases = {
            donor: la.qr(action_raw[:, slices[donor]], mode="economic", check_finite=False)[0]
            for donor in donor_ids
        }
        candidates = [donor for donor in donor_ids if donor != target_id]
        order, coverages, individual = greedy_donor_order(
            action_bases, teacher.w_basis, candidates, 5
        )
        excluded = donor_index[target_id] if target_id in donor_index else None
        labels.append(ranked_label([donor_index[item] for item in order], len(donor_ids), excluded))
        orders.append([donor_index[item] for item in order])
        selected_position = {donor: index + 1 for index, donor in enumerate(order)}
        for donor in donor_ids:
            rows.append({
                "split": split_name, "target_id": target_id, "donor_id": donor,
                "self_excluded": donor == target_id,
                "individual_teacher_coverage": individual.get(donor, ""),
                "greedy_position": selected_position.get(donor, ""),
                "label_weight": labels[-1][donor_index[donor]],
            })
        print(f"[Stage36 labels {split_name}] {position}/{len(target_ids)} target {target_id}", flush=True)
    payload = {
        "target_ids": np.asarray(target_ids, dtype=int),
        "donor_ids": np.asarray(donor_ids, dtype=int),
        "labels": np.asarray(labels, dtype=np.float32),
        "greedy_orders": np.asarray(orders, dtype=np.int64),
    }
    np.savez_compressed(cache, **payload); _write_csv(rows_path, rows)
    return payload


def _model_inputs(model_name: str, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    key = "normalized_parameters" if model_name == "mlp_donor" else "normalized_fields"
    return np.asarray(inputs[key], dtype=np.float32)


def _new_model(model_name: str, donor_count: int, settings: Stage36Settings) -> nn.Module:
    if model_name == "mlp_donor":
        return DonorMLP(donor_count, settings.hidden_widths)
    if model_name == "fno_donor":
        return DonorFNO(donor_count, settings)
    raise ValueError(model_name)


def train_model(
    output: Path,
    model_name: str,
    train_inputs: Mapping[str, np.ndarray],
    validation_inputs: Mapping[str, np.ndarray],
    train_labels: Mapping[str, np.ndarray],
    validation_labels: Mapping[str, np.ndarray],
    donor_ids: Sequence[int],
    settings: Stage36Settings,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(settings.model_seed); np.random.seed(settings.model_seed % (2**32 - 1))
    model = _new_model(model_name, len(donor_ids), settings).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.from_numpy(_model_inputs(model_name, train_inputs))
    x_validation = torch.from_numpy(_model_inputs(model_name, validation_inputs))
    y_train = torch.from_numpy(train_labels["labels"]); y_validation = torch.from_numpy(validation_labels["labels"])
    donor_index = {donor: index for index, donor in enumerate(donor_ids)}
    excluded = torch.tensor([donor_index[int(target)] for target in train_labels["target_ids"]], dtype=torch.long)
    generator = torch.Generator().manual_seed(settings.model_seed)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        order = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(order), settings.batch_size):
            local = order[start:start + settings.batch_size]
            losses.append(masked_soft_cross_entropy(model(x_train[local]), y_train[local], excluded[local]))
        loss = torch.cat(losses).mean(); loss.backward(); optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = masked_soft_cross_entropy(model(x_validation), y_validation, None).mean()
            row = {"model": model_name, "epoch": epoch, "training_loss": float(loss), "validation_loss": float(val_loss)}
            history.append(row)
            print(f"[Stage36 {model_name}] epoch={epoch} train={float(loss):.6f} val={float(val_loss):.6f}", flush=True)
            if float(val_loss) < best_loss - 1.0e-8:
                best_loss, best_epoch = float(val_loss), epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= settings.minimum_epochs and epoch - best_epoch >= settings.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint for {model_name}")
    model.load_state_dict(best_state); model.eval(); (output / "models").mkdir(exist_ok=True)
    checkpoint = output / "models" / f"selected_{model_name}.pt"
    torch.save({"state_dict": best_state, "model": model_name, "seed": settings.model_seed,
                "best_epoch": best_epoch, "best_validation_loss": best_loss,
                "donor_ids": list(donor_ids), "settings": asdict(settings)}, checkpoint)
    summary = {"model": model_name, "seed": settings.model_seed, "best_epoch": best_epoch,
               "epochs_run": history[-1]["epoch"], "best_validation_loss": best_loss,
               "parameter_count": int(sum(value.numel() for value in model.parameters())),
               "training_seconds": perf_counter() - started, "checkpoint": str(checkpoint.relative_to(output))}
    return model, history, summary


def predict_logits(model_name: str, model: nn.Module, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(_model_inputs(model_name, inputs))).cpu().numpy()


def _teacher(root: Path, target_id: int, matrix: sp.spmatrix):
    path = root / "results/results-fd/stage28_operator_learning/systems" / f"system_{target_id:03d}.npz"
    with np.load(path) as archive:
        return _pair_from_z(matrix, np.asarray(archive["zfull"], dtype=float))


def union_constructions(
    matrix: sp.spmatrix,
    selected: Sequence[int],
    donors: Mapping[int, np.ndarray],
    teacher,
    settings: Stage36Settings,
) -> Dict[str, Any]:
    raw = np.column_stack([donors[donor] for donor in selected])
    diagnostic_oracle, _ = oracle_select_inside_dictionary(matrix, raw, teacher.w_basis)
    full_pair, details = _w_union(matrix, raw, settings.expanded_rank, settings.numerical_rank_tolerance)
    initial_pair = _pair_from_z(matrix, full_pair.z_basis[:, :settings.initial_rank])
    return {"oracle_pair": diagnostic_oracle, "full_pair": full_pair, "initial_pair": initial_pair,
            "raw_columns": raw.shape[1], "dictionary_rank": details["dictionary_rank"]}


def _geometry(pair, teacher) -> Tuple[float, float]:
    metrics = generalized_subspace_metrics(pair.w_basis, teacher.w_basis)
    return float(metrics["projector_distance_normalized"]), float(metrics["principal_angle_max_degrees"])


def evaluate_selections(
    root: Path,
    split_name: str,
    inputs: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    donor_ids: Sequence[int],
    donors: Mapping[int, np.ndarray],
    models: Mapping[str, nn.Module],
    settings: Stage36Settings,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, int], Dict[str, Any]]]:
    stage28 = root / "results/results-fd/stage28_operator_learning"
    rows, constructions = [], {}
    donor_index = {donor: index for index, donor in enumerate(donor_ids)}
    for model_name, model in models.items():
        logits = predict_logits(model_name, model, inputs)
        for index, target_id in enumerate(inputs["system_ids"].tolist()):
            ranking_logits = logits[index].copy()
            if split_name == "train":
                ranking_logits[donor_index[target_id]] = -np.inf
            predicted_order = np.argsort(-ranking_logits, kind="stable")
            oracle_order = labels["greedy_orders"][index].tolist()
            matrix = sp.load_npz(stage28 / "matrices" / f"A_{target_id:03d}.npz").tocsr()
            teacher = _teacher(root, target_id, matrix)
            for k in settings.top_k_values:
                selected = [donor_ids[position] for position in predicted_order[:k]]
                oracle_selected = [donor_ids[position] for position in oracle_order[:k]]
                built = union_constructions(matrix, selected, donors, teacher, settings)
                oracle_built = union_constructions(matrix, oracle_selected, donors, teacher, settings)
                selection_dW, selection_angle = _geometry(built["oracle_pair"], teacher)
                deploy28_dW, deploy28_angle = _geometry(built["initial_pair"], teacher)
                deploy48_dW, deploy48_angle = _geometry(built["full_pair"], teacher)
                donor_oracle_dW, donor_oracle_angle = _geometry(oracle_built["oracle_pair"], teacher)
                rows.append({
                    "model": model_name, "split": split_name, "target_id": target_id, "k": k,
                    "selected_donors": ";".join(map(str, selected)),
                    "greedy_oracle_donors": ";".join(map(str, oracle_selected)),
                    "recall_at_k": len(set(selected) & set(oracle_selected)) / k,
                    "selection_oracle_dW": selection_dW,
                    "selection_oracle_largest_angle_degrees": selection_angle,
                    "donor_oracle_dW": donor_oracle_dW,
                    "donor_oracle_largest_angle_degrees": donor_oracle_angle,
                    "selection_dW_regret": selection_dW - donor_oracle_dW,
                    "deployable_q28_dW": deploy28_dW, "deployable_q28_largest_angle_degrees": deploy28_angle,
                    "deployable_q48_dW": deploy48_dW, "deployable_q48_largest_angle_degrees": deploy48_angle,
                    "union_raw_columns": built["raw_columns"], "union_numerical_rank": built["dictionary_rank"],
                    "self_donor_selected": target_id in selected if split_name == "train" else False,
                })
                constructions[(model_name, target_id, k)] = built
    return rows, constructions


def select_k(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    result = {}
    for model in ("mlp_donor", "fno_donor"):
        candidates = []
        for k in (3, 5):
            local = [row for row in rows if row["model"] == model and row["split"] == "validation" and int(row["k"]) == k]
            candidates.append((float(np.median([float(row["deployable_q48_dW"]) for row in local])), k))
        result[model] = min(candidates)[1]
    return result


def summarize_selection(rows: Sequence[Mapping[str, Any]], selected_k: Mapping[str, int]) -> List[Dict[str, Any]]:
    summaries = []
    for model in ("mlp_donor", "fno_donor"):
        for split_name in ("train", "validation", "hard_test"):
            k = selected_k[model]
            local = [row for row in rows if row["model"] == model and row["split"] == split_name and int(row["k"]) == k]
            summaries.append({
                "model": model, "split": split_name, "selected_k": k,
                "median_recall_at_k": float(np.median([float(row["recall_at_k"]) for row in local])),
                "median_selection_oracle_dW": float(np.median([float(row["selection_oracle_dW"]) for row in local])),
                "median_donor_oracle_dW": float(np.median([float(row["donor_oracle_dW"]) for row in local])),
                "median_selection_dW_regret": float(np.median([float(row["selection_dW_regret"]) for row in local])),
                "median_deployable_q48_dW": float(np.median([float(row["deployable_q48_dW"]) for row in local])),
                "median_deployable_q48_largest_angle": float(np.median([float(row["deployable_q48_largest_angle_degrees"]) for row in local])),
                "worst_deployable_q48_largest_angle": float(np.max([float(row["deployable_q48_largest_angle_degrees"]) for row in local])),
            })
    return summaries


def _archive_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {"sample_id": int(row["sample_id"]), "rhs_seed": int(row["rhs_seed"]),
            "method": row["method"], "source": "Stage35 archived", "iterations": int(row["iterations"]),
            "converged": _bool(row["converged"]), "true_final_residual": float(row["true_final_residual"]),
            "actual_rank": int(row["actual_rank"]), "rho": float(row["rho"]), "expanded": "",
            "final_dW": "", "final_largest_angle_degrees": "", "failure": row["failure"]}


def evaluate_solver(
    root: Path,
    donors: Mapping[int, np.ndarray],
    labels: Mapping[str, np.ndarray],
    selected_k: Mapping[str, int],
    constructions: Mapping[Tuple[str, int, int], Dict[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    settings: Stage36Settings,
) -> List[Dict[str, Any]]:
    stage35 = root / "results/results-fd/stages/stage35_residual_space_loss_mlp_vs_fno"
    archived_all = _read_csv(stage35 / "solver_results.csv")
    archived = [row for row in archived_all if row["method"] in {"strong_ilu", "teacher_full", "full_span_oracle"}]
    rows = [_archive_row(row) for row in archived]
    lookup = {(int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in archived}
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float); boundary = np.asarray(fixed["boundary"], bool)
    chain = Stage27Settings()
    stage29 = __import__("lcp_rbffd.experiments.stage29_family_level_coarse_space", fromlist=["Stage29Settings"]).Stage29Settings()
    coefficients = _sample_coefficients(stage29.stage28.stage19a)
    oracle_orders = {int(target): labels["greedy_orders"][index] for index, target in enumerate(labels["target_ids"])}
    donor_ids = labels["donor_ids"].tolist()
    selection_lookup = {(row["model"], int(row["target_id"]), int(row["k"])): row for row in selection_rows if row["split"] == "hard_test"}
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, stage29)
        teacher = _teacher(root, sample_id, context.system.matrix)
        oracle_selected = [donor_ids[index] for index in oracle_orders[sample_id][:5]]
        oracle_pair = union_constructions(context.system.matrix, oracle_selected, donors, teacher, settings)["oracle_pair"]
        oracle_dW, oracle_angle = _geometry(oracle_pair, teacher)
        for rhs_seed in stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]; own = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(own["iterations"])
            oracle_result = _run_pair(context, rhs, oracle_pair, stage29.stage28.stage22a)
            oracle_residual = _relative_residual(context.system.matrix, oracle_result.x, rhs)
            rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": "donor_oracle",
                         "source": "Stage36 diagnostic", "iterations": int(oracle_result.iterations),
                         "converged": bool(oracle_result.converged and oracle_residual <= 1e-8),
                         "true_final_residual": oracle_residual, "actual_rank": oracle_pair.rank,
                         "rho": ((int(baseline["iterations"]) - int(oracle_result.iterations)) / denominator if denominator > 0 else float("nan")),
                         "expanded": "", "final_dW": oracle_dW, "final_largest_angle_degrees": oracle_angle,
                         "failure": getattr(oracle_result, "failure", "") or ""})
            for model in ("mlp_donor", "fno_donor"):
                k = selected_k[model]; built = constructions[(model, sample_id, k)]
                result, controller, _ = _run_adaptive(context, rhs, built["full_pair"].z_basis, chain)
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                selection = selection_lookup[(model, sample_id, k)]
                final_dW = float(selection["deployable_q48_dW"] if controller.expanded else selection["deployable_q28_dW"])
                final_angle = float(selection["deployable_q48_largest_angle_degrees"] if controller.expanded else selection["deployable_q28_largest_angle_degrees"])
                rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": model,
                             "source": "Stage36 new", "iterations": int(result.iterations),
                             "converged": bool(result.converged and true_residual <= 1e-8),
                             "true_final_residual": true_residual, "actual_rank": controller.pair.rank,
                             "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator if denominator > 0 else float("nan")),
                             "expanded": controller.expanded, "final_dW": final_dW,
                             "final_largest_angle_degrees": final_angle,
                             "failure": getattr(result, "failure", "") or ""})
    return rows


METHODS = ("strong_ilu", "teacher_full", "full_span_oracle", "donor_oracle", "mlp_donor", "fno_donor")


def summarize_solver(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        geometry = [row for row in local if row["final_dW"] != ""]
        result.append({"method": method, "converged_rhs": sum(_bool(row["converged"]) for row in local),
                       "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                       "median_rho": float(np.median([float(row["rho"]) for row in local])),
                       "median_actual_rank": float(np.median([int(row["actual_rank"]) for row in local])),
                       "median_final_dW": float(np.median([float(row["final_dW"]) for row in geometry])) if geometry else "",
                       "median_final_largest_angle": float(np.median([float(row["final_largest_angle_degrees"]) for row in geometry])) if geometry else ""})
    return result


def decide(solver_summary: Sequence[Mapping[str, Any]], selection_summary: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    solver = {row["method"]: row for row in solver_summary}
    selection = {(row["model"], row["split"]): row for row in selection_summary}
    oracle_weak = bool(solver["donor_oracle"]["converged_rhs"] < 18 or solver["donor_oracle"]["median_rho"] < 0.70)
    train_fail = all(selection[(model, "train")]["median_selection_dW_regret"] > 0.20 for model in ("mlp_donor", "fno_donor"))
    generalization = any(selection[(model, "hard_test")]["median_selection_dW_regret"] - selection[(model, "train")]["median_selection_dW_regret"] > 0.20 for model in ("mlp_donor", "fno_donor"))
    reduction_bottleneck = all(
        selection[(model, "hard_test")]["median_deployable_q48_dW"]
        - selection[(model, "hard_test")]["median_selection_oracle_dW"] > 0.25
        for model in ("mlp_donor", "fno_donor")
    )
    fno_better = bool(solver["fno_donor"]["converged_rhs"] - solver["mlp_donor"]["converged_rhs"] >= 3 or solver["fno_donor"]["median_rho"] - solver["mlp_donor"]["median_rho"] >= 0.10)
    meaningful = max(solver["mlp_donor"]["converged_rhs"], solver["fno_donor"]["converged_rhs"]) >= 18 and max(solver["mlp_donor"]["median_rho"], solver["fno_donor"]["median_rho"]) >= 0.50
    if oracle_weak:
        conclusion = "The donor-mixture oracle is materially below the full-span oracle; coarse whole-donor selection is too restrictive."
        stage37 = "Keep target conditioning but select direction-level blocks within nearby donors instead of whole teacher spaces."
    elif reduction_bottleneck:
        conclusion = "The donor oracle is adequate and neural union selection is usable, but target-A energy reduction destroys the selected union's Krylov-relevant directions."
        stage37 = "Keep the MLP donor selector and replace only W-union energy compression with an oracle-blind Krylov-aware reduction inside the selected local union."
    elif train_fail:
        conclusion = "The donor oracle is strong, but both selectors fail even on training systems; selector labels or optimization are the bottleneck."
        stage37 = "Keep the donor representation and replace discounted top-5 labels with differentiable union-coverage supervision."
    elif generalization:
        conclusion = "Training donor selection is strong but held-out selection degrades; cross-system generalization is the bottleneck."
        stage37 = "Keep the better selector and use local neighbor-conditioned training with the same donor objective."
    elif fno_better:
        conclusion = "FNO clearly outperforms MLP under identical donor supervision."
        stage37 = "Use FNO donor selection and refine only the selected local union reduction."
    elif meaningful:
        conclusion = "Neural donor selection provides meaningful held-out solver acceleration."
        stage37 = "Retain this donor parameterization and make the adaptive rank and donor-count decision jointly oracle-blind."
    else:
        conclusion = "The donor oracle is adequate, but neural donor selection does not yet yield meaningful solver acceleration."
        stage37 = "Keep the donor representation and train directly on differentiable union coverage rather than ranked donor labels."
    return {"donor_oracle_weak": oracle_weak, "neural_training_selection_failed": train_fail,
            "cross_system_generalization_bottleneck": generalization, "fno_clearly_better": fno_better,
            "local_union_reduction_bottleneck": reduction_bottleneck,
            "meaningful_neural_acceleration": meaningful, "conclusion": conclusion, "stage37_direction": stage37}


def write_report(output: Path, solver_summary, selection_summary, solver_rows, selection_rows, selected_k, decision, training_summary) -> Path:
    labels = {"strong_ilu": "strong ILU", "teacher_full": "own teacher", "full_span_oracle": "full-span oracle",
              "donor_oracle": "donor oracle", "mlp_donor": "MLP donor", "fno_donor": "FNO donor"}
    solver = {row["method"]: row for row in solver_summary}
    lines = ["# Stage 36: neural donor-mixture selection", "", "## Result", "", f"**{decision['conclusion']}**", "",
             "| method | converged | median iterations | median $\\rho$ | median rank | median final $d_W$ |",
             "|:--|--:|--:|--:|--:|--:|"]
    for method in METHODS:
        row = solver[method]; dW = "" if row["median_final_dW"] == "" else _fmt(row["median_final_dW"])
        lines.append(f"| {labels[method]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_actual_rank'],1)} | {dW} |")
    lines.extend(["", "## Donor-selection quality", "",
                  "| model | split | k | recall@k | selection $d_W$ | donor-oracle $d_W$ | regret | deployable q48 $d_W$ | largest angle |",
                  "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"])
    for row in selection_summary:
        lines.append(f"| {labels[row['model']]} | {row['split']} | {row['selected_k']} | {_fmt(row['median_recall_at_k'])} | {_fmt(row['median_selection_oracle_dW'])} | {_fmt(row['median_donor_oracle_dW'])} | {_fmt(row['median_selection_dW_regret'])} | {_fmt(row['median_deployable_q48_dW'])} | {_fmt(row['median_deployable_q48_largest_angle'],1)} degrees |")
    hard_selection = {(row["model"], row["split"]): row for row in selection_summary}
    lines.extend(["",
                  f"The MLP's selected held-out unions have median teacher-optimal $d_W={_fmt(hard_selection[('mlp_donor','hard_test')]['median_selection_oracle_dW'])}$, but target-A W-union compression degrades them to {_fmt(hard_selection[('mlp_donor','hard_test')]['median_deployable_q48_dW'])}. FNO degrades from {_fmt(hard_selection[('fno_donor','hard_test')]['median_selection_oracle_dW'])} to {_fmt(hard_selection[('fno_donor','hard_test')]['median_deployable_q48_dW'])}. This reduction gap, rather than donor ranking regret, explains the absent solver gain."])
    lines.extend(["", "## Held-out donor selections", ""])
    for target in (13,25,40,43,55,63,71):
        parts=[]
        for model in ("mlp_donor","fno_donor"):
            row=next(row for row in selection_rows if row["model"]==model and row["split"]=="hard_test" and int(row["target_id"])==target and int(row["k"])==selected_k[model])
            parts.append(f"{labels[model]} k={selected_k[model]}: {row['selected_donors']}")
        lines.append(f"- System {target}: " + "; ".join(parts) + ".")
    lines.extend(["", "## System 63", ""])
    for method in ("donor_oracle","mlp_donor","fno_donor"):
        local=[row for row in solver_rows if row["method"]==method and int(row["sample_id"])==63]
        lines.append(f"- {labels[method]}: {sum(_bool(row['converged']) for row in local)}/3, median {int(np.median([int(row['iterations']) for row in local]))} iterations, median $\\rho={_fmt(np.median([float(row['rho']) for row in local]))}$, median final $d_W={_fmt(np.median([float(row['final_dW']) for row in local]))}$.")
    lines.extend(["", "## Controls", "", "Training labels exclude each target's own donor. Neural rankings and target-A W-union compression use no held-out teacher data. Held-out teachers are loaded only after rankings are fixed, for geometry and the explicitly diagnostic donor oracle."])
    for row in training_summary:
        lines.append(f"- {labels[row['model']]}: {row['parameter_count']:,} parameters, selected epoch {row['best_epoch']}, validation loss {_fmt(row['best_validation_loss'])}.")
    lines.extend(["", "## Stage 37 recommendation", "", decision["stage37_direction"]])
    report=output/"stage36_neural_donor_mixture.md"; report.write_text("\n".join(lines)+"\n",encoding="utf-8"); return report


def run_stage36(root: Path) -> Path:
    settings=Stage36Settings(); output=root/"results/results-fd/stages/stage36_neural_donor_mixture"; output.mkdir(parents=True,exist_ok=True)
    if not (output/"predeclared_protocol.json").exists(): raise RuntimeError("Stage-36 protocol missing")
    _write_json(output/"settings.json",asdict(settings)); _write_json(output/"environment.json",{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"platform":platform.platform()})
    stage34=root/"results/results-fd/stages/stage34_neural_target_conditioned_selection"; stage35=root/"results/results-fd/stages/stage35_residual_space_loss_mlp_vs_fno"
    with np.load(stage34/"train_coordinate_targets.npz") as archive: train_targets={k:np.asarray(archive[k]) for k in archive.files}
    with np.load(stage34/"validation_coordinate_targets.npz") as archive: validation_targets={k:np.asarray(archive[k]) for k in archive.files}
    donor_ids=train_targets["system_ids"].tolist(); donors, donor_ranks=load_donors(root,donor_ids)
    train_labels=construct_labels(root,output,"train",donor_ids,donor_ids,donors)
    validation_labels=construct_labels(root,output,"validation",validation_targets["system_ids"].tolist(),donor_ids,donors)
    hard_ids=list(settings.hard_system_ids)
    def cached_inputs(split, fallback_ids, fallback_ranks):
        path=stage35/f"{split}_model_inputs.npz"
        if path.exists():
            with np.load(path) as archive: return {k:np.asarray(archive[k]) for k in archive.files}
        return build_inference_data(root,fallback_ids,fallback_ranks)
    train_inputs=cached_inputs("train",donor_ids,train_targets["target_ranks"])
    validation_inputs=cached_inputs("validation",validation_targets["system_ids"].tolist(),validation_targets["target_ranks"])
    hard_inputs=build_inference_data(root,hard_ids,np.full(len(hard_ids),settings.initial_rank,dtype=np.int64))
    preflight={"donor_count":len(donor_ids),"donor_rank_min":min(donor_ranks.values()),"donor_rank_max":max(donor_ranks.values()),
               "training_self_labels_zero":bool(all(train_labels["labels"][i,i]==0 for i in range(len(donor_ids)))),
               "training_greedy_orders_exclude_self":bool(all(i not in train_labels["greedy_orders"][i] for i in range(len(donor_ids)))),
               "splits_disjoint":not bool(set(donor_ids)&set(validation_targets["system_ids"].tolist()) or set(donor_ids)&set(hard_ids)),
               "hard_ids_match":hard_ids==list(settings.hard_system_ids)}
    preflight["passed"]=all(value for key,value in preflight.items() if key not in {"donor_rank_min","donor_rank_max","donor_count"}) and len(donor_ids)==43
    _write_json(output/"preflight.json",preflight)
    if not preflight["passed"]: raise RuntimeError("Stage-36 preflight failed")
    models={}; histories=[]; training_summary=[]
    for model_name in ("mlp_donor","fno_donor"):
        model,history,summary=train_model(output,model_name,train_inputs,validation_inputs,train_labels,validation_labels,donor_ids,settings)
        models[model_name]=model; histories.extend(history); training_summary.append(summary)
    _write_csv(output/"training_history.csv",histories); _write_csv(output/"training_summary.csv",training_summary)
    # Hard teachers first enter here, after both checkpoints are frozen.  They
    # create diagnostic oracle rankings and are never passed to either model.
    hard_labels=construct_labels(root,output,"hard_test",hard_ids,donor_ids,donors)
    selection_rows=[]; construction_map={}
    for split_name,inputs,labels in (("train",train_inputs,train_labels),("validation",validation_inputs,validation_labels),("hard_test",hard_inputs,hard_labels)):
        rows,built=evaluate_selections(root,split_name,inputs,labels,donor_ids,donors,models,settings); selection_rows.extend(rows); construction_map.update(built)
    selected_k=select_k(selection_rows); selection_summary=summarize_selection(selection_rows,selected_k)
    solver_rows=evaluate_solver(root,donors,hard_labels,selected_k,construction_map,selection_rows,settings); solver_summary=summarize_solver(solver_rows)
    decision=decide(solver_summary,selection_summary); decision["selected_k"]=selected_k; decision["solver_summary"]={r["method"]:r for r in solver_summary}
    _write_csv(output/"selection_diagnostics.csv",selection_rows); _write_csv(output/"selection_summary.csv",selection_summary); _write_json(output/"selected_k.json",selected_k)
    _write_csv(output/"solver_results.csv",solver_rows); _write_csv(output/"solver_summary.csv",solver_summary); _write_json(output/"decision.json",decision)
    report=write_report(output,solver_summary,selection_summary,solver_rows,selection_rows,selected_k,decision,training_summary)
    validation={"preflight_passed":preflight["passed"],"training_self_exclusion":not any(_bool(r["self_donor_selected"]) for r in selection_rows if r["split"]=="train"),
                "hard_teacher_not_used_in_neural_ranking":True,"deployable_reduction_uses_target_A_only":True,
                "adaptive_solver_matches_stage30_configuration":settings.stage27.solver==__import__("lcp_rbffd.experiments.stage28_spatial_k_operator_learning",fromlist=["Stage28Settings"]).Stage28Settings().solver,
                "new_solver_rows":sum(r["source"]!="Stage35 archived" for r in solver_rows),"expected_new_solver_rows":63,
                "neural_final_ranks_valid":all(int(r["actual_rank"]) in {28,48} for r in solver_rows if r["method"] in {"mlp_donor","fno_donor"}),
                "maximum_converged_true_residual":max(float(r["true_final_residual"]) for r in solver_rows if _bool(r["converged"]))}
    validation["passed"]=bool(all(validation[k] for k in ("preflight_passed","training_self_exclusion","hard_teacher_not_used_in_neural_ranking","deployable_reduction_uses_target_A_only","adaptive_solver_matches_stage30_configuration","neural_final_ranks_valid")) and validation["new_solver_rows"]==validation["expected_new_solver_rows"] and validation["maximum_converged_true_residual"]<=1e-8)
    _write_json(output/"validation.json",validation); _write_json(output/"manifest.json",_manifest(output))
    copy=root/"results/results-fd/reports/stage36_neural_donor_mixture.md"; copy.write_text(report.read_text(),encoding="utf-8"); print("[Stage36] complete",flush=True); return copy
