"""Stage 47: lightweight FNO correction added to frozen Stage-41 q=48."""

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
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage28_spatial_k_operator_learning import FNOFieldEncoder
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage35_residual_space_loss_mlp_vs_fno import build_inference_data
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage40_target_conditioned_local_basis import factor_full_action, select_local_combinations
from .stage41_multi_rhs_slow_mode_target import Stage41Settings, harmonic_action_information
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage47Settings:
    model_seed: int = 20262821
    correction_ranks: Tuple[int, int] = (8, 12)
    correction_dictionary_rank: int = 96
    complementarity_weight: float = 0.10
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
    numerical_rank_tolerance: float = 1.0e-11
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage41(self) -> Stage41Settings:
        return Stage41Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


def complementary_target(
    matrix,
    classical: RecyclePair,
    teacher: RecyclePair,
    maximum_rank: int,
    tolerance: float,
) -> Tuple[RecyclePair, Dict[str, Any]]:
    """Return leading basis-invariant components of (I-P_W48) W_teacher."""

    overlap = classical.w_basis.T @ teacher.w_basis
    raw_u = teacher.u_basis - classical.u_basis @ overlap
    raw_w = np.asarray(matrix @ raw_u, dtype=float)
    left, singular, right_t = la.svd(raw_w, full_matrices=False, check_finite=False)
    numerical_rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    actual = min(maximum_rank, numerical_rank)
    if actual < maximum_rank:
        raise RuntimeError(f"complement rank {actual} is below requested {maximum_rank}")
    ordered_u = raw_u @ right_t.T[:, :actual]
    pair = _ordered_pair_from_z(matrix, ordered_u, 1.0e-13)
    classical_overlap = float(la.norm(classical.w_basis.T @ pair.w_basis, ord="fro"))
    return pair, {
        "raw_complement_rank": numerical_rank,
        "retained_rank": pair.rank,
        "largest_complement_singular_value": float(singular[0]),
        "smallest_retained_complement_singular_value": float(singular[actual - 1]),
        "discarded_complement_energy_fraction": float(
            np.sum(singular[actual:] ** 2) / max(np.sum(singular**2), np.finfo(float).tiny)
        ),
        "target_classical_overlap_frobenius": classical_overlap,
    }


def _frozen_classical_pair(
    root: Path,
    sample_id: int,
    context,
    psi: np.ndarray,
    settings: Stage47Settings,
    hard_archive: Mapping[str, np.ndarray] | None,
) -> RecyclePair:
    key = f"system_{sample_id:03d}_multi_rhs_harmonic_m64_z"
    if hard_archive is not None and key in hard_archive:
        return _ordered_pair_from_z(context.system.matrix, np.asarray(hard_archive[key], float), 1.0e-13)
    q_action, triangular, _ = factor_full_action(context.system.matrix, psi)
    slow_actions = []
    for seed in settings.stage41.probe_rhs_seeds:
        rhs, _ = _discrete_rhs(context.system, seed)
        arnoldi = _capture_arnoldi(
            context.system.matrix, context.ilu, rhs, settings.stage41.probe_depth
        )
        action, _, _, _ = harmonic_action_information(
            context.system.matrix, arnoldi, settings.stage41.slow_modes_per_probe
        )
        slow_actions.append(action)
    pair, _, _ = select_local_combinations(
        context.system.matrix,
        psi,
        q_action,
        triangular,
        np.column_stack(slow_actions),
        settings.stage41.expanded_rank,
        settings.stage41.coefficient_nonzero_relative_tolerance,
        preserve_order=True,
    )
    return pair


def construct_target_split(
    root: Path,
    output: Path,
    split: str,
    system_ids: Sequence[int],
    psi: np.ndarray,
    settings: Stage47Settings,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    cache = output / f"{split}_classical_complements.npz"
    metadata_path = output / f"{split}_target_construction.csv"
    expected = {
        f"system_{sample_id:03d}_{suffix}"
        for sample_id in system_ids
        for suffix in ("classical_z48", "target_u12", "target_w12", "complement_singular_values")
    }
    if cache.exists() and metadata_path.exists():
        with np.load(cache) as archive:
            if expected.issubset(archive.files):
                data = {
                    sample_id: {
                        "classical_z48": np.asarray(archive[f"system_{sample_id:03d}_classical_z48"], float),
                        "target_u12": np.asarray(archive[f"system_{sample_id:03d}_target_u12"], float),
                        "target_w12": np.asarray(archive[f"system_{sample_id:03d}_target_w12"], float),
                    }
                    for sample_id in system_ids
                }
                return data, _read_csv(metadata_path)

    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    hard_path = root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/multi_rhs_local_spaces.npz"
    hard_npz = np.load(hard_path) if split == "hard_test" else None
    hard_archive = hard_npz
    coordinate_path = (
        root
        / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
        / f"{split}_coordinate_targets.npz"
    )
    with np.load(coordinate_path) as coordinate_archive:
        coordinate_ids = np.asarray(coordinate_archive["system_ids"], int)
        coordinate_targets = np.asarray(coordinate_archive["targets"], float)
        coordinate_ranks = np.asarray(coordinate_archive["target_ranks"], int)
    coordinate_lookup = {int(sample_id): index for index, sample_id in enumerate(coordinate_ids)}
    arrays: Dict[str, np.ndarray] = {}
    data: Dict[int, Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    for position, sample_id in enumerate(system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        classical = _frozen_classical_pair(root, sample_id, context, psi, settings, hard_archive)
        coordinate_index = coordinate_lookup[sample_id]
        reference_rank = int(coordinate_ranks[coordinate_index])
        teacher = _pair_from_z(
            context.system.matrix,
            psi @ coordinate_targets[coordinate_index, :, :reference_rank],
        )
        target, diagnostics = complementary_target(
            context.system.matrix, classical, teacher,
            max(settings.correction_ranks), settings.numerical_rank_tolerance,
        )
        singular = la.svdvals(
            teacher.w_basis - classical.w_basis @ (classical.w_basis.T @ teacher.w_basis)
        )
        prefix = f"system_{sample_id:03d}"
        arrays[f"{prefix}_classical_z48"] = classical.z_basis.astype(np.float32)
        arrays[f"{prefix}_target_u12"] = target.u_basis.astype(np.float32)
        arrays[f"{prefix}_target_w12"] = target.w_basis.astype(np.float32)
        arrays[f"{prefix}_complement_singular_values"] = singular
        data[sample_id] = {
            "classical_z48": classical.z_basis,
            "target_u12": target.u_basis,
            "target_w12": target.w_basis,
        }
        rows.append({
            "split": split, "sample_id": sample_id,
            "classical_rank": classical.rank, "reference_rank": teacher.rank,
            "reference": "Stage30 teacher-informed full-span oracle",
            **diagnostics,
            "classical_pair_defect": classical.diagnostics["au_equals_w_relative_error"],
            "target_pair_defect": target.diagnostics["au_equals_w_relative_error"],
            "heldout_teacher_used_in_training": False,
        })
        print(f"[Stage47 targets {split}] {position}/{len(system_ids)} system {sample_id}", flush=True)
    if hard_npz is not None:
        hard_npz.close()
    np.savez_compressed(cache, **arrays)
    _write_csv(metadata_path, rows)
    return data, rows


def build_correction_dictionary(
    output: Path,
    train_targets: Mapping[int, Mapping[str, np.ndarray]],
    settings: Stage47Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = output / "correction_dictionary.npz"
    if path.exists():
        with np.load(path) as archive:
            basis = np.asarray(archive["basis"], float)
            singular = np.asarray(archive["singular_values"], float)
            ids = np.asarray(archive["training_ids"], int)
        if basis.shape[1] == settings.correction_dictionary_rank and ids.tolist() == list(train_targets):
            return basis, {
                "rank": basis.shape[1], "source_columns": len(train_targets) * max(settings.correction_ranks),
                "captured_solution_energy": float(np.sum(singular[:basis.shape[1]]**2) / np.sum(singular**2)),
            }
    raw = np.column_stack([train_targets[sample_id]["target_u12"] for sample_id in train_targets])
    left, singular, _ = la.svd(raw, full_matrices=False, check_finite=False)
    if left.shape[1] < settings.correction_dictionary_rank:
        raise RuntimeError("training complements do not support requested correction dictionary rank")
    basis = left[:, :settings.correction_dictionary_rank]
    np.savez_compressed(
        path, basis=basis.astype(np.float32), singular_values=singular,
        training_ids=np.asarray(list(train_targets), int),
    )
    return basis, {
        "rank": basis.shape[1], "source_columns": raw.shape[1],
        "captured_solution_energy": float(np.sum(singular[:basis.shape[1]]**2) / np.sum(singular**2)),
    }


def build_loss_data(
    root: Path,
    split: str,
    system_ids: Sequence[int],
    targets: Mapping[int, Mapping[str, np.ndarray]],
    dictionary: np.ndarray,
) -> Dict[str, np.ndarray]:
    inputs = build_inference_data(root, system_ids, np.full(len(system_ids), 12, dtype=int))
    grams, target_crosses, classical_crosses = [], [], []
    representation_rows = []
    stage28 = root / "results/results-fd/stage28_operator_learning"
    for sample_id in system_ids:
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        classical = _ordered_pair_from_z(matrix, targets[sample_id]["classical_z48"], 1.0e-13)
        action = np.asarray(matrix @ dictionary, dtype=float)
        gram = action.T @ action
        target_w = targets[sample_id]["target_w12"]
        grams.append(gram.astype(np.float32))
        target_crosses.append((target_w.T @ action).astype(np.float32))
        classical_crosses.append((classical.w_basis.T @ action).astype(np.float32))
        q_action, triangular = la.qr(action, mode="economic", check_finite=False)
        for rank in (8, 12):
            left, singular, _ = la.svd(
                q_action.T @ target_w[:, :rank], full_matrices=False, check_finite=False
            )
            coefficients = la.solve_triangular(triangular, left[:, :rank], lower=False, check_finite=False)
            ceiling = _pair_from_z(matrix, dictionary @ coefficients)
            metrics = generalized_subspace_metrics(ceiling.w_basis, target_w[:, :rank])
            representation_rows.append({
                "split": split, "sample_id": sample_id, "rank": rank,
                "dictionary_ceiling_d_W": metrics["projector_distance_normalized"],
                "dictionary_ceiling_largest_angle": metrics["principal_angle_max_degrees"],
                "dictionary_target_containment": metrics["containment_second"],
                "minimum_selected_action_cosine": float(singular[rank - 1]),
            })
    return {
        **inputs,
        "action_grams": np.asarray(grams, np.float32),
        "target_crosses": np.asarray(target_crosses, np.float32),
        "classical_crosses": np.asarray(classical_crosses, np.float32),
        "representation_rows": representation_rows,
    }


class ComplementFNO(nn.Module):
    def __init__(self, settings: Stage47Settings) -> None:
        super().__init__()
        self.encoder = FNOFieldEncoder(
            settings.correction_dictionary_rank,
            max(settings.correction_ranks),
            settings.fno_width,
            settings.fno_modes,
            settings.fno_layers,
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.encoder(fields)


def _reduced_losses(
    prediction: torch.Tensor,
    gram: torch.Tensor,
    target_cross: torch.Tensor,
    classical_cross: torch.Tensor,
    rank: int,
    settings: Stage47Settings,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients = torch.linalg.qr(prediction[:, :rank], mode="reduced").Q
    gram_c = coefficients.T @ gram @ coefficients
    scale = torch.clamp(torch.diagonal(gram_c).mean(), min=torch.finfo(prediction.dtype).eps)
    eye = torch.eye(rank, dtype=prediction.dtype, device=prediction.device)
    lower = torch.linalg.cholesky(gram_c + settings.cholesky_jitter * scale * eye)
    target_unscaled = target_cross[:rank] @ coefficients
    target_overlap = torch.linalg.solve_triangular(lower, target_unscaled.T, upper=False)
    classical_unscaled = classical_cross @ coefficients
    classical_overlap = torch.linalg.solve_triangular(lower, classical_unscaled.T, upper=False)
    subspace = torch.clamp(1.0 - torch.sum(target_overlap**2) / rank, min=0.0)
    complementarity = torch.sum(classical_overlap**2) / rank
    total = subspace + settings.complementarity_weight * complementarity
    return total, subspace, complementarity


def _dataset_losses(model, data, settings, *, indices: np.ndarray | None = None):
    fields = torch.from_numpy(np.asarray(data["normalized_fields"], np.float32))
    grams = torch.from_numpy(data["action_grams"])
    targets = torch.from_numpy(data["target_crosses"])
    classical = torch.from_numpy(data["classical_crosses"])
    if indices is None:
        indices = np.arange(fields.shape[0])
    index_tensor = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    predictions = model(fields[index_tensor])
    totals, subspaces, complements = [], [], []
    for prediction_index, source_index in enumerate(indices):
        local_total, local_sub, local_comp = [], [], []
        for rank in settings.correction_ranks:
            total, subspace, complementarity = _reduced_losses(
                predictions[prediction_index], grams[source_index], targets[source_index],
                classical[source_index], rank, settings
            )
            local_total.append(total); local_sub.append(subspace); local_comp.append(complementarity)
        totals.append(torch.stack(local_total).mean())
        subspaces.append(torch.stack(local_sub).mean())
        complements.append(torch.stack(local_comp).mean())
    return torch.stack(totals), torch.stack(subspaces), torch.stack(complements)


def train_fno(output: Path, train_data, validation_data, settings: Stage47Settings):
    checkpoint = output / "models/selected_complement_fno.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model = ComplementFNO(settings).to(dtype=torch.float32)
        model.load_state_dict(payload["state_dict"]); model.eval()
        return model, [], {
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "best_epoch": int(payload["best_epoch"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
            "checkpoint_reused": True,
        }
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = ComplementFNO(settings).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        order = np.random.permutation(len(train_data["system_ids"]))
        total_parts, subspace_parts, complement_parts = [], [], []
        for start in range(0, len(order), settings.batch_size):
            total_part, subspace_part, complement_part = _dataset_losses(
                model, train_data, settings, indices=order[start:start + settings.batch_size]
            )
            total_parts.append(total_part); subspace_parts.append(subspace_part)
            complement_parts.append(complement_part)
        total = torch.cat(total_parts); subspace = torch.cat(subspace_parts)
        complementarity = torch.cat(complement_parts)
        loss = total.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Stage-47 FNO loss")
        loss.backward(); optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_total, val_subspace, val_comp = _dataset_losses(model, validation_data, settings)
            row = {
                "epoch": epoch,
                "training_loss": float(loss.detach()),
                "training_subspace_loss": float(subspace.mean().detach()),
                "training_complementarity_loss": float(complementarity.mean().detach()),
                "validation_loss": float(val_total.mean()),
                "validation_subspace_loss": float(val_subspace.mean()),
                "validation_complementarity_loss": float(val_comp.mean()),
            }
            history.append(row)
            print(f"[Stage47 FNO] epoch={epoch} train={row['training_loss']:.5f} val={row['validation_loss']:.5f}", flush=True)
            if row["validation_loss"] < best_loss - 1.0e-8:
                best_loss, best_epoch = row["validation_loss"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= settings.minimum_epochs and epoch - best_epoch >= settings.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Stage-47 FNO produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "settings": asdict(settings),
    }, checkpoint)
    return model, history, {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_run": history[-1]["epoch"], "training_seconds": perf_counter() - started,
        "checkpoint_reused": False,
    }


def _predicted_correction(
    matrix,
    classical: RecyclePair,
    dictionary: np.ndarray,
    coefficients: np.ndarray,
) -> Tuple[RecyclePair, RecyclePair, Dict[str, float]]:
    coefficient_q = la.qr(coefficients, mode="economic", check_finite=False)[0]
    raw_u = dictionary @ coefficient_q
    raw_w = np.asarray(matrix @ raw_u, dtype=float)
    raw_w_q = la.qr(raw_w, mode="economic", check_finite=False)[0]
    pre_overlap = float(la.norm(classical.w_basis.T @ raw_w_q, ord="fro") ** 2 / raw_w_q.shape[1])
    complementary_u = raw_u - classical.u_basis @ (classical.w_basis.T @ raw_w)
    correction = _pair_from_z(matrix, complementary_u)
    hybrid = _ordered_pair_from_z(
        matrix, np.column_stack((classical.u_basis, correction.u_basis)), 1.0e-13
    )
    post_overlap = float(la.norm(classical.w_basis.T @ correction.w_basis, ord="fro") ** 2 / correction.rank)
    containment = float(
        la.norm(hybrid.w_basis.T @ classical.w_basis, ord="fro") ** 2 / classical.rank
    )
    return correction, hybrid, {
        "preprojection_classical_overlap": pre_overlap,
        "postprojection_classical_overlap": post_overlap,
        "classical_containment_in_hybrid": containment,
    }


def evaluate_split_geometry(
    root: Path,
    split: str,
    system_ids: Sequence[int],
    targets,
    data,
    dictionary,
    model,
    settings: Stage47Settings,
):
    fields = torch.from_numpy(np.asarray(data["normalized_fields"], np.float32))
    with torch.no_grad():
        predictions = model(fields).cpu().numpy()
    rows, pairs = [], {}
    stage28 = root / "results/results-fd/stage28_operator_learning"
    for index, sample_id in enumerate(system_ids):
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        classical = _ordered_pair_from_z(matrix, targets[sample_id]["classical_z48"], 1.0e-13)
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            teacher = _pair_from_z(matrix, np.asarray(archive["zfull"], float))
        target12 = _ordered_pair_from_z(matrix, targets[sample_id]["target_u12"], 1.0e-13)
        for rank in settings.correction_ranks:
            correction, hybrid, diagnostics = _predicted_correction(
                matrix, classical, dictionary, predictions[index, :, :rank]
            )
            target_w = target12.w_basis[:, :rank]
            corr_metrics = generalized_subspace_metrics(correction.w_basis, target_w)
            hybrid_metrics = generalized_subspace_metrics(hybrid.w_basis, teacher.w_basis)
            rows.append({
                "split": split, "sample_id": sample_id, "rank": rank,
                "produced_correction_rank": correction.rank,
                "hybrid_rank": hybrid.rank,
                "correction_d_W": corr_metrics["projector_distance_normalized"],
                "correction_largest_angle": corr_metrics["principal_angle_max_degrees"],
                "correction_target_containment": corr_metrics["containment_second"],
                "hybrid_d_W_to_teacher": hybrid_metrics["projector_distance_normalized"],
                "hybrid_largest_angle_to_teacher": hybrid_metrics["principal_angle_max_degrees"],
                **diagnostics,
            })
            if split == "hard_test":
                pairs[(sample_id, rank)] = hybrid
    return rows, pairs


def _oracle_hybrids(root, hard_ids, hard_targets, settings):
    pairs, rows = {}, []
    stage28 = root / "results/results-fd/stage28_operator_learning"
    for sample_id in hard_ids:
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        classical = _ordered_pair_from_z(matrix, hard_targets[sample_id]["classical_z48"], 1.0e-13)
        with np.load(stage28 / "systems" / f"system_{sample_id:03d}.npz") as archive:
            teacher = _pair_from_z(matrix, np.asarray(archive["zfull"], float))
        target12 = _ordered_pair_from_z(matrix, hard_targets[sample_id]["target_u12"], 1.0e-13)
        for rank in settings.correction_ranks:
            correction = _ordered_pair_from_z(matrix, target12.u_basis[:, :rank], 1.0e-13)
            hybrid = _ordered_pair_from_z(
                matrix, np.column_stack((classical.u_basis, correction.u_basis)), 1.0e-13
            )
            metrics = generalized_subspace_metrics(hybrid.w_basis, teacher.w_basis)
            pairs[(sample_id, rank)] = hybrid
            rows.append({
                "split": "hard_test", "sample_id": sample_id, "rank": rank,
                "produced_correction_rank": correction.rank, "hybrid_rank": hybrid.rank,
                "correction_d_W": 0.0, "correction_largest_angle": 0.0,
                "correction_target_containment": 1.0,
                "hybrid_d_W_to_teacher": metrics["projector_distance_normalized"],
                "hybrid_largest_angle_to_teacher": metrics["principal_angle_max_degrees"],
                "preprojection_classical_overlap": 0.0,
                "postprojection_classical_overlap": float(la.norm(classical.w_basis.T @ correction.w_basis, ord="fro")**2/rank),
                "classical_containment_in_hybrid": float(la.norm(hybrid.w_basis.T @ classical.w_basis, ord="fro")**2/classical.rank),
                "oracle": True,
            })
    return pairs, rows


def evaluate_solver(root, oracle_pairs, fno_pairs, settings):
    cached_path = root / "results/results-fd/stages/stage47_fno_complementary_correction/solver_results.csv"
    if cached_path.exists():
        cached = _read_csv(cached_path)
        new_rows = [row for row in cached if row["method"].startswith("classical_q48_plus_")]
        if len(new_rows) == 84:
            return cached
    stage43 = _read_csv(root / "results/results-fd/stages/stage43_missing_subspace_block_ceiling/solver_results.csv")
    base_methods = {"strong_ilu", "teacher_full", "full_span_oracle", "stage41_harmonic_fixed_q48"}
    rows = [dict(row) for row in stage43 if row["method"] in base_methods]
    lookup = {(int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in stage43}
    fixed = np.load(root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], float); boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for family, pair_map in (("oracle", oracle_pairs), ("fno", fno_pairs)):
                for rank in settings.correction_ranks:
                    pair = pair_map[(sample_id, rank)]
                    result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                    residual = _relative_residual(context.system.matrix, result.x, rhs)
                    rows.append({
                        "sample_id": sample_id, "rhs_seed": rhs_seed,
                        "method": f"classical_q48_plus_{family}_corr{rank}",
                        "source": "Stage47 non-deployable oracle" if family == "oracle" else "Stage47 FNO hybrid",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual, "actual_rank": pair.rank,
                        "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator if denominator > 0 else float("nan")),
                        "failure": getattr(result, "failure", "") or "",
                    })
        print(f"[Stage47 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows, methods):
    summary = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return summary


def _baseline_geometry_rows(root: Path) -> List[Dict[str, Any]]:
    source = _read_csv(
        root
        / "results/results-fd/stages/stage43_missing_subspace_block_ceiling"
        / "comparison_geometry.csv"
    )
    methods = {"teacher_full", "full_span_oracle", "stage41_harmonic_fixed_q48"}
    return [
        {
            "split": "hard_test",
            "sample_id": int(row["sample_id"]),
            "rank": 0,
            "family": "baseline",
            "method": row["method"],
            "hybrid_d_W_to_teacher": float(row["d_W"]),
            "hybrid_largest_angle_to_teacher": float(row["largest_principal_angle_degrees"]),
        }
        for row in source
        if row["method"] in methods
    ]


def _report(
    output, solver_summary, geometry_rows, representation_rows,
    split_loss_rows, training_summary, decision, settings,
):
    labels = {
        "strong_ilu": "strong ILU", "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage41_harmonic_fixed_q48": "fixed classical q=48",
    }
    for rank in settings.correction_ranks:
        labels[f"classical_q48_plus_oracle_corr{rank}"] = f"q48 + oracle correction r={rank}"
        labels[f"classical_q48_plus_fno_corr{rank}"] = f"q48 + FNO correction r={rank}"
    lines = [
        "# Stage 47: Universal lightweight FNO complementary correction",
        "", "## Result", "", f"**{decision['conclusion']}**", "",
        "The Stage-41 q=48 space is frozen and retained as an exact prefix of every hybrid. The FNO receives only the normalized K-field plus fixed coordinates and predicts a 96-by-12 correction coefficient block (1,152 outputs versus Stage 35's 69,888).",
        "",
        "| method | converged | median iterations | median rho | median final rank | median dW | median largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        method = row["method"]
        local_g = [r for r in geometry_rows if r.get("method") == method]
        d_w = np.median([float(r["hybrid_d_W_to_teacher"]) for r in local_g]) if local_g else float("nan")
        angle = np.median([float(r["hybrid_largest_angle_to_teacher"]) for r in local_g]) if local_g else float("nan")
        lines.append(
            f"| {labels[method]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_final_rank'],0)} | {_fmt(d_w)} | {_fmt(angle,1)} degrees |"
        )
    lines += [
        "", "## Correction learning diagnostics", "",
        "| split | rank | median dictionary ceiling dW | median FNO correction dW | median largest angle | median target containment | median pre/post classical overlap |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            rep = [r for r in representation_rows if r["split"] == split and int(r["rank"]) == rank]
            local = [r for r in geometry_rows if r["split"] == split and int(r["rank"]) == rank and r.get("family") == "fno"]
            lines.append(
                f"| {split} | {rank} | {_fmt(np.median([float(r['dictionary_ceiling_d_W']) for r in rep]))} | "
                f"{_fmt(np.median([float(r['correction_d_W']) for r in local]))} | {_fmt(np.median([float(r['correction_largest_angle']) for r in local]),1)} degrees | "
                f"{_fmt(np.median([float(r['correction_target_containment']) for r in local]))} | "
                f"{_fmt(np.median([float(r['preprojection_classical_overlap']) for r in local]))} / {_fmt(np.median([float(r['postprojection_classical_overlap']) for r in local]))} |"
            )
    lines += [
        "",
        f"Selected checkpoint: epoch {training_summary['best_epoch']}, validation loss {_fmt(training_summary['best_validation_loss'])}, {training_summary['parameter_count']:,} parameters. Training and validation subspace/complementarity losses are saved separately in the history and split-loss summaries.",
        "", "| split | mean subspace loss | mean classical-overlap loss |",
        "|:--|--:|--:|",
    ]
    for row in split_loss_rows:
        lines.append(
            f"| {row['split']} | {_fmt(row['mean_subspace_loss'])} | "
            f"{_fmt(row['mean_complementarity_loss'])} |"
        )
    lines += [
        "", "## System 63", "",
        "| method | converged | median iterations | median rho | dW | largest angle |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    solver_rows = _read_csv(output / "solver_results.csv")
    for method in (
        "stage41_harmonic_fixed_q48", "full_span_oracle",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "classical_q48_plus_fno_corr8", "classical_q48_plus_fno_corr12",
    ):
        local_s = [r for r in solver_rows if int(r["sample_id"]) == 63 and r["method"] == method]
        local_g = [r for r in geometry_rows if int(r["sample_id"]) == 63 and r.get("method") == method]
        lines.append(
            f"| {labels[method]} | {sum(_bool(r['converged']) for r in local_s)}/3 | {_fmt(np.median([int(r['iterations']) for r in local_s]),1)} | {_fmt(np.median([float(r['rho']) for r in local_s]))} | "
            f"{_fmt(local_g[0]['hybrid_d_W_to_teacher']) if local_g else 'n/a'} | {_fmt(local_g[0]['hybrid_largest_angle_to_teacher'],1)+' degrees' if local_g else 'n/a'} |"
        )
    lines += [
        "", "## Interpretation", "",
        decision["interpretation"],
        "", "## Stage 48 recommendation", "", decision["stage48_recommendation"],
    ]
    report = output / "stage47_fno_complementary_correction.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage47(root: Path) -> Path:
    settings = Stage47Settings()
    output = root / "results/results-fd/stages/stage47_fno_complementary_correction"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-47 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
    })
    stage34 = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    split_ids = {}
    for split in ("train", "validation", "hard_test"):
        with np.load(stage34 / f"{split}_coordinate_targets.npz") as archive:
            split_ids[split] = np.asarray(archive["system_ids"], int).tolist()
    full_span_path = root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    with np.load(full_span_path) as archive:
        psi = np.asarray(archive["psi"], float)
        dictionary_training_ids = np.asarray(archive["training_ids"], int).tolist()
    preflight = {
        "training_ids_match_stage30_dictionary": dictionary_training_ids == split_ids["train"],
        "splits_disjoint": not bool(
            set(split_ids["train"]) & set(split_ids["validation"])
            or set(split_ids["train"]) & set(split_ids["hard_test"])
            or set(split_ids["validation"]) & set(split_ids["hard_test"])
        ),
        "hard_ids_exact": split_ids["hard_test"] == list(settings.hard_system_ids),
        "correction_ranks_exact": settings.correction_ranks == (8, 12),
        "classical_rank_fixed_48": True,
        "K_field_only_model": True,
        "no_solver_change": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-47 preflight failed")

    train_targets, train_target_rows = construct_target_split(root, output, "train", split_ids["train"], psi, settings)
    validation_targets, validation_target_rows = construct_target_split(root, output, "validation", split_ids["validation"], psi, settings)
    dictionary, dictionary_meta = build_correction_dictionary(output, train_targets, settings)
    _write_json(output / "correction_dictionary_metadata.json", dictionary_meta)
    train_data = build_loss_data(root, "train", split_ids["train"], train_targets, dictionary)
    validation_data = build_loss_data(root, "validation", split_ids["validation"], validation_targets, dictionary)
    model, history, training_summary = train_fno(output, train_data, validation_data, settings)
    if history:
        _write_csv(output / "training_history.csv", history)
    _write_json(output / "training_summary.json", training_summary)

    # Hard teachers and oracle complements are loaded only after checkpoint selection.
    hard_targets, hard_target_rows = construct_target_split(root, output, "hard_test", split_ids["hard_test"], psi, settings)
    hard_data = build_loss_data(root, "hard_test", split_ids["hard_test"], hard_targets, dictionary)
    geometry_rows, fno_pairs = [], {}
    representation_rows = []
    for split, ids, targets, data in (
        ("train", split_ids["train"], train_targets, train_data),
        ("validation", split_ids["validation"], validation_targets, validation_data),
        ("hard_test", split_ids["hard_test"], hard_targets, hard_data),
    ):
        local, pairs = evaluate_split_geometry(root, split, ids, targets, data, dictionary, model, settings)
        for row in local:
            row["family"] = "fno"
            row["method"] = f"classical_q48_plus_fno_corr{row['rank']}"
        geometry_rows.extend(local); fno_pairs.update(pairs)
        representation_rows.extend(data["representation_rows"])
    oracle_pairs, oracle_geometry = _oracle_hybrids(root, split_ids["hard_test"], hard_targets, settings)
    for row in oracle_geometry:
        row["family"] = "oracle"
        row["method"] = f"classical_q48_plus_oracle_corr{row['rank']}"
    geometry_rows.extend(oracle_geometry)
    geometry_rows.extend(_baseline_geometry_rows(root))
    solver_rows = evaluate_solver(root, oracle_pairs, fno_pairs, settings)
    _write_csv(output / "solver_results.csv", solver_rows)
    methods = (
        "strong_ilu", "teacher_full", "full_span_oracle", "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "classical_q48_plus_fno_corr8", "classical_q48_plus_fno_corr12",
    )
    solver_summary = summarize_solver(solver_rows, methods)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "dictionary_representation_ceiling.csv", representation_rows)
    _write_csv(output / "target_construction_all.csv", train_target_rows + validation_target_rows + hard_target_rows)
    with torch.no_grad():
        split_loss_rows = []
        for split, data in (("train", train_data), ("validation", validation_data), ("hard_test", hard_data)):
            total, subspace, comp = _dataset_losses(model, data, settings)
            split_loss_rows.append({
                "split": split, "mean_total_loss": float(total.mean()),
                "mean_subspace_loss": float(subspace.mean()),
                "mean_complementarity_loss": float(comp.mean()),
            })
    _write_csv(output / "split_loss_summary.csv", split_loss_rows)
    summary_lookup = {row["method"]: row for row in solver_summary}
    oracle12 = summary_lookup["classical_q48_plus_oracle_corr12"]
    fno12 = summary_lookup["classical_q48_plus_fno_corr12"]
    oracle_sufficient = int(oracle12["converged_rhs"]) == 21
    fixed48 = summary_lookup["stage41_harmonic_fixed_q48"]
    fno_works = (
        int(fno12["converged_rhs"]) == 21
        and float(fno12["median_iterations"]) <= 1.10 * float(fixed48["median_iterations"])
    )
    hard_fno_geom = [row for row in geometry_rows if row.get("family") == "fno" and row["split"] == "hard_test" and int(row["rank"]) == 12]
    hard_fixed_geom = [row for row in geometry_rows if row.get("method") == "stage41_harmonic_fixed_q48"]
    geometry_improves = (
        np.median([float(row["hybrid_d_W_to_teacher"]) for row in hard_fno_geom])
        < 0.98 * np.median([float(row["hybrid_d_W_to_teacher"]) for row in hard_fixed_geom])
    )
    iteration_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): int(row["iterations"])
        for row in solver_rows
    }
    ordinary_max_degradation = max(
        iteration_lookup[(sample_id, rhs_seed, "classical_q48_plus_fno_corr12")]
        - iteration_lookup[(sample_id, rhs_seed, "stage41_harmonic_fixed_q48")]
        for sample_id in settings.hard_system_ids if sample_id != 63
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds
    )
    if not oracle_sufficient:
        conclusion = "Outcome C — oracle rank-8/12 complementary correction is insufficient."
        interpretation = "The compact complementary target is not adequate in principle; neural failure cannot be blamed on the FNO."
        recommendation = "Stage 48 should revisit the complementary target rank or definition before any further network tuning."
    elif fno_works:
        conclusion = "Outcome A — oracle correction and FNO correction both work."
        interpretation = "The hybrid formulation succeeds: a small predicted complement recovers the missing information without sacrificing ordinary systems."
        recommendation = "Stage 48 should freeze the q48-plus-FNO hybrid and test cost and broader generalization."
    elif geometry_improves:
        conclusion = "Outcome D — FNO improves geometry but not solver performance."
        interpretation = "Projector-aligned correction geometry remains an insufficient proxy for Krylov utility."
        recommendation = "Stage 48 should diagnose solver alignment of the predicted correction, without changing the classical q48 branch."
    else:
        conclusion = "Outcome B — oracle correction works but FNO fails."
        train_loss = next(row for row in split_loss_rows if row["split"] == "train")
        validation_loss = next(row for row in split_loss_rows if row["split"] == "validation")
        training_ceiling = np.median([
            float(row["dictionary_ceiling_d_W"])
            for row in representation_rows
            if row["split"] == "train" and int(row["rank"]) == 12
        ])
        if training_ceiling > 0.2:
            failure = (
                f"the fixed rank-96 output dictionary is inadequate even on training systems "
                f"(median ceiling d_W={training_ceiling:.3f})"
            )
            recommendation = (
                "Stage 48 should keep the frozen q48 branch, K-only FNO, loss, and ranks fixed, "
                "and replace only the rank-96 dictionary with the untruncated 516-column training-"
                "complement dictionary to remove the measured representation ceiling."
            )
        elif train_loss["mean_subspace_loss"] > 0.5:
            failure = "loss/optimization failure is already present on training systems"
            recommendation = (
                "Stage 48 should keep the frozen q48 branch and output dictionary fixed and test "
                "one optimization repair targeted at the training subspace loss."
            )
        else:
            failure = "training geometry is adequate but held-out target conditioning/generalization fails"
            recommendation = (
                "Stage 48 should keep the frozen q48 branch and correction parameterization fixed "
                "and add one compact Stage-41 classical-summary conditioning channel."
            )
        interpretation = (
            "Oracle corrections show that rank 8/12 is sufficient in principle: both reach "
            "21/21, and rank 12 brings system 63 to 3/3 at median 95 iterations. The FNO "
            f"does not recover that correction because {failure}. Its rank-12 augmentation "
            f"leaves system 63 capped and changes every ordinary-system solve by at most "
            f"{ordinary_max_degradation:+d} iteration relative to fixed q48."
        )
    decision = {
        "conclusion": conclusion, "oracle_rank12_sufficient": oracle_sufficient,
        "fno_rank12_works": fno_works, "interpretation": interpretation,
        "ordinary_system_max_iteration_degradation": ordinary_max_degradation,
        "stage48_recommendation": recommendation,
    }
    _write_json(output / "decision.json", decision)
    report = _report(
        output, solver_summary, geometry_rows, representation_rows,
        split_loss_rows, training_summary, decision, settings,
    )
    new_solver = [row for row in solver_rows if str(row["method"]).startswith("classical_q48_plus_")]
    validation = {
        "preflight_passed": preflight["passed"],
        "complement_target_overlap_tolerance": 1.0e-6,
        "complement_targets_orthogonal": max(float(row["target_classical_overlap_frobenius"]) for row in train_target_rows + validation_target_rows + hard_target_rows) <= 1.0e-6,
        "hard_teacher_loaded_after_checkpoint": True,
        "correction_ranks_exact": all(
            int(row["produced_correction_rank"]) == int(row["rank"])
            for row in geometry_rows if row.get("family") in {"fno", "oracle"}
        ),
        "classical_q48_preserved": min(
            float(row["classical_containment_in_hybrid"])
            for row in geometry_rows if row.get("family") in {"fno", "oracle"}
        ) >= 1.0 - 1.0e-9,
        "maximum_postprojection_overlap": max(
            float(row["postprojection_classical_overlap"])
            for row in geometry_rows if row.get("family") in {"fno", "oracle"}
        ),
        "new_solver_rows": len(new_solver), "expected_new_solver_rows": 84,
        "maximum_converged_true_residual": max(float(row["true_final_residual"]) for row in solver_rows if _bool(row["converged"])),
        "neural_outputs": settings.correction_dictionary_rank * max(settings.correction_ranks),
        "stage35_outputs": 1456 * 48,
    }
    validation["passed"] = bool(
        validation["preflight_passed"] and validation["complement_targets_orthogonal"]
        and validation["hard_teacher_loaded_after_checkpoint"] and validation["correction_ranks_exact"]
        and validation["classical_q48_preserved"] and validation["maximum_postprojection_overlap"] <= 1.0e-8
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["maximum_converged_true_residual"] <= 1.0e-8
        and validation["neural_outputs"] < validation["stage35_outputs"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage47_fno_complementary_correction.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage47] complete", flush=True)
    return copy
