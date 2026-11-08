"""Stage 51: neural compression of paired raw multi-RHS Arnoldi information."""

from __future__ import annotations

import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy.fft import dctn
import torch
from torch import nn

from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _run_pair
from .stage28_spatial_k_operator_learning import SpectralConv2d
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage35_residual_space_loss_mlp_vs_fno import build_inference_data
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage47_fno_complementary_correction import Stage47Settings
from .stage49_target_local_correction_ceiling import principal_coverage_metrics
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage51Settings:
    model_seed: int = 20262821
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    correction_ranks: Tuple[int, int] = (8, 12)
    raw_depth: int = 64
    raw_rhs_seeds: Tuple[int, ...] = (20262911, 20262921, 20262931)
    maximum_raw_columns: int = 192
    paired_svd_relative_tolerance: float = 1.0e-10
    dct_modes: int = 6
    token_width: int = 64
    latent_width: int = 64
    fno_width: int = 16
    fno_modes: int = 8
    fno_layers: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 6
    maximum_epochs: int = 200
    minimum_epochs: int = 75
    validation_interval: int = 5
    early_stopping_patience: int = 50
    raw_ceiling_required_convergence: int = 21
    raw_ceiling_system63_required_convergence: int = 3
    neural_training_containment_gate: float = 0.50
    neural_oracle_gap_gate: float = 0.10
    k_benefit_containment_margin: float = 0.05
    k_benefit_rho_margin: float = 0.05

    @property
    def stage47(self) -> Stage47Settings:
        return Stage47Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return self.stage47.stage29

    @property
    def token_feature_dimension(self) -> int:
        return self.dct_modes**2 + 11


def _canonicalize_svd(left: np.ndarray, right_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove arbitrary SVD signs while preserving the paired factorization."""

    left = np.asarray(left, float).copy()
    right_t = np.asarray(right_t, float).copy()
    for column in range(left.shape[1]):
        pivot = int(np.argmax(np.abs(left[:, column])))
        sign = 1.0 if left[pivot, column] >= 0.0 else -1.0
        left[:, column] *= sign
        right_t[column, :] *= sign
    return left, right_t


def paired_raw_candidate(
    matrix: sp.spmatrix,
    classical: RecyclePair,
    solution_steps: np.ndarray,
    action_steps: np.ndarray,
    relative_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return a stable paired basis for raw Arnoldi action information."""

    overlap = classical.w_basis.T @ action_steps
    solution_perp = solution_steps - classical.u_basis @ overlap
    action_perp = action_steps - classical.w_basis @ overlap
    left, singular, right_t = la.svd(
        action_perp, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    rank = int(np.count_nonzero(singular > relative_tolerance * singular[0]))
    if rank < 12:
        raise RuntimeError(f"raw paired candidate rank {rank} is below 12")
    left, right_t = _canonicalize_svd(left[:, :rank], right_t[:rank, :])
    paired_solution = solution_perp @ (right_t.T / singular[:rank])
    relative_pair_defect = float(
        la.norm(np.asarray(matrix @ paired_solution, float) - left, ord="fro")
        / np.sqrt(rank)
    )
    return paired_solution, left, singular[:rank], {
        "raw_columns": int(action_steps.shape[1]),
        "effective_raw_rank": rank,
        "largest_singular_value": float(singular[0]),
        "smallest_retained_relative_singular_value": float(singular[rank - 1] / singular[0]),
        "candidate_pair_defect_before_selected_repair": relative_pair_defect,
        "classical_projection_fraction": float(
            la.norm(overlap, ord="fro") ** 2
            / max(la.norm(action_steps, ord="fro") ** 2, np.finfo(float).tiny)
        ),
    }


def raw_token_features(
    action_basis: np.ndarray,
    singular: np.ndarray,
    right_t: np.ndarray,
    settings: Stage51Settings,
) -> np.ndarray:
    """Compact spatial and trajectory-aware feature for each raw SVD direction."""

    rank, columns = right_t.shape
    side = int(round(np.sqrt(action_basis.shape[0])))
    if side * side != action_basis.shape[0] or columns != settings.maximum_raw_columns:
        raise RuntimeError("unexpected raw candidate dimensions")
    spectra = dctn(
        action_basis.T.reshape(rank, side, side), axes=(-2, -1), type=2, norm="ortho"
    )[:, : settings.dct_modes, : settings.dct_modes].reshape(rank, -1)
    weights = right_t**2
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), np.finfo(float).tiny)
    seed_index = np.repeat(np.arange(3, dtype=float), settings.raw_depth)
    step_index = np.tile(np.arange(settings.raw_depth, dtype=float), 3)
    seed_fractions = np.column_stack([
        np.sum(weights[:, block * settings.raw_depth : (block + 1) * settings.raw_depth], axis=1)
        for block in range(3)
    ])
    mean_step = weights @ step_index / max(settings.raw_depth - 1, 1)
    centered = step_index[None, :] / max(settings.raw_depth - 1, 1) - mean_step[:, None]
    std_step = np.sqrt(np.sum(weights * centered**2, axis=1))
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1.0e-30)), axis=1) / np.log(columns)
    max_weight = np.max(weights, axis=1)
    mean_seed = weights @ seed_index / 2.0
    log_singular = np.clip(np.log10(singular / singular[0]), -12.0, 0.0) / 12.0 + 1.0
    index = np.arange(rank, dtype=float) / max(rank - 1, 1)
    energy = singular**2
    cumulative = np.cumsum(energy) / np.sum(energy)
    return np.column_stack((
        spectra,
        log_singular,
        index,
        cumulative,
        seed_fractions,
        mean_seed,
        mean_step,
        std_step,
        entropy,
        max_weight,
    )).astype(np.float32)


def selected_pair(
    matrix: sp.spmatrix,
    classical: RecyclePair,
    paired_solution: np.ndarray,
    mixing: np.ndarray,
) -> Tuple[RecyclePair, RecyclePair, Dict[str, float]]:
    """Build an exact complementary pair while keeping q48 as the literal prefix."""

    coefficients = la.qr(np.asarray(mixing, float), mode="economic", check_finite=False)[0]
    correction_u = paired_solution @ coefficients
    for _ in range(3):
        correction_w = np.asarray(matrix @ correction_u, float)
        correction_u -= classical.u_basis @ (classical.w_basis.T @ correction_w)
    correction_w, triangular = la.qr(
        np.asarray(matrix @ correction_u, float), mode="economic", check_finite=False
    )
    correction_u = la.solve_triangular(
        triangular, correction_u.T, trans="T", lower=False, check_finite=False
    ).T
    exact_action = np.asarray(matrix @ correction_u, float)
    pair_defect = float(
        la.norm(exact_action - correction_w, ord="fro")
        / max(la.norm(correction_w, ord="fro"), np.finfo(float).tiny)
    )
    overlap = float(la.norm(classical.w_basis.T @ correction_w, ord="fro"))
    correction = RecyclePair(
        z_basis=la.qr(correction_u, mode="economic", check_finite=False)[0],
        u_basis=correction_u,
        w_basis=correction_w,
        setup_seconds=0.0,
        diagnostics={"au_equals_w_relative_error": pair_defect},
    )
    hybrid_u = np.column_stack((classical.u_basis, correction_u))
    hybrid_w = np.column_stack((classical.w_basis, correction_w))
    hybrid = RecyclePair(
        z_basis=la.qr(hybrid_u, mode="economic", check_finite=False)[0],
        u_basis=hybrid_u,
        w_basis=hybrid_w,
        setup_seconds=0.0,
        diagnostics={"au_equals_w_relative_error": pair_defect},
    )
    return correction, hybrid, {
        "q48_correction_overlap_frobenius": overlap,
        "q48_containment_in_hybrid": float(
            la.norm(hybrid.w_basis.T @ classical.w_basis, ord="fro") ** 2 / classical.rank
        ),
        "pair_defect": pair_defect,
    }


def optimal_raw_mixing(action_basis: np.ndarray, target: np.ndarray, rank: int):
    left, singular, _ = la.svd(
        action_basis.T @ target[:, :rank], full_matrices=False, check_finite=False
    )
    return left[:, :rank], singular


def _target_cache(root: Path, split: str) -> Path:
    return (
        root / "results/results-fd/stages/stage47_fno_complementary_correction"
        / f"{split}_classical_complements.npz"
    )


def build_raw_split(
    root: Path,
    output: Path,
    split: str,
    system_ids: Sequence[int],
    settings: Stage51Settings,
):
    data_path = output / f"{split}_raw_loss_data.npz"
    candidate_path = output / f"{split}_paired_raw_candidates.npz"
    diagnostic_path = output / f"{split}_raw_construction.csv"
    if data_path.exists() and diagnostic_path.exists() and (split != "hard_test" or candidate_path.exists()):
        with np.load(data_path) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if (
            data["system_ids"].tolist() == list(system_ids)
            and data["token_features"].shape[1:] == (
                settings.maximum_raw_columns, settings.token_feature_dimension
            )
        ):
            candidates = {}
            if split == "hard_test":
                with np.load(candidate_path) as archive:
                    for sample_id, dimension in zip(system_ids, data["effective_dimensions"]):
                        candidates[sample_id] = (
                            np.asarray(archive[f"system_{sample_id:03d}_paired_solution"], float),
                            np.asarray(archive[f"system_{sample_id:03d}_action_basis"], float),
                        )
            return data, candidates, _read_csv(diagnostic_path)

    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    inference = build_inference_data(root, system_ids, np.full(len(system_ids), 12, int))
    token_features = np.zeros(
        (len(system_ids), settings.maximum_raw_columns, settings.token_feature_dimension), np.float32
    )
    masks = np.zeros((len(system_ids), settings.maximum_raw_columns), bool)
    crosses = np.zeros((len(system_ids), 12, settings.maximum_raw_columns), np.float32)
    dimensions = np.zeros(len(system_ids), int)
    singular_values = np.zeros((len(system_ids), settings.maximum_raw_columns), np.float32)
    candidate_arrays: Dict[str, np.ndarray] = {}
    candidates: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    target_path = _target_cache(root, split)
    for index, sample_id in enumerate(system_ids):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
        )
        with np.load(target_path) as target_archive:
            classical_z = np.asarray(target_archive[f"system_{sample_id:03d}_classical_z48"], float)
        classical = _ordered_pair_from_z(context.system.matrix, classical_z, 1.0e-13)
        solution_steps, action_steps = [], []
        started = perf_counter()
        actual_depths = []
        for seed in settings.raw_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, seed)
            arnoldi = _capture_arnoldi(
                context.system.matrix, context.ilu, rhs, settings.raw_depth
            )
            actual_depths.append(int(arnoldi.hessenberg.shape[1]))
            solution_steps.append(arnoldi.solution_basis[:, : settings.raw_depth])
            action_steps.append(
                np.asarray(context.system.matrix @ arnoldi.solution_basis[:, : settings.raw_depth], float)
            )
        raw_u = np.column_stack(solution_steps)
        raw_w = np.column_stack(action_steps)
        overlap = classical.w_basis.T @ raw_w
        action_perp = raw_w - classical.w_basis @ overlap
        left_all, singular_all, right_all = la.svd(
            action_perp, full_matrices=False, check_finite=False, lapack_driver="gesdd"
        )
        dimension = int(np.count_nonzero(
            singular_all > settings.paired_svd_relative_tolerance * singular_all[0]
        ))
        left, right_t = _canonicalize_svd(left_all[:, :dimension], right_all[:dimension, :])
        solution_perp = raw_u - classical.u_basis @ overlap
        paired_solution = solution_perp @ (right_t.T / singular_all[:dimension])
        pair_defect = float(
            la.norm(np.asarray(context.system.matrix @ paired_solution, float) - left, ord="fro")
            / np.sqrt(dimension)
        )
        features = raw_token_features(left, singular_all[:dimension], right_t, settings)
        token_features[index, :dimension] = features
        masks[index, :dimension] = True
        dimensions[index] = dimension
        singular_values[index, :dimension] = singular_all[:dimension]

        # Teacher access begins only after the deployable raw candidate is frozen.
        with np.load(target_path) as target_archive:
            target_w = np.asarray(target_archive[f"system_{sample_id:03d}_target_w12"], float)
        crosses[index, :, :dimension] = (target_w.T @ left).astype(np.float32)
        row = {
            "split": split,
            "sample_id": sample_id,
            "raw_rhs_seeds": "/".join(str(seed) for seed in settings.raw_rhs_seeds),
            "raw_depths": "/".join(str(depth) for depth in actual_depths),
            "raw_columns": raw_w.shape[1],
            "effective_raw_rank": dimension,
            "candidate_pair_defect_before_selected_repair": pair_defect,
            "teacher_loaded_after_raw_candidate_frozen": True,
            "harmonic_filtering_used": False,
            "construction_seconds": perf_counter() - started,
        }
        for rank in settings.correction_ranks:
            _, singular = optimal_raw_mixing(left, target_w, rank)
            coverage = singular[:rank] ** 2
            row[f"oracle_r{rank}_containment"] = float(np.mean(coverage))
            row[f"oracle_r{rank}_p10"] = float(np.percentile(coverage, 10))
            row[f"oracle_r{rank}_minimum"] = float(np.min(coverage))
            row[f"oracle_r{rank}_largest_angle"] = float(
                np.degrees(np.arccos(np.clip(np.min(singular[:rank]), 0.0, 1.0)))
            )
        rows.append(row)
        if split == "hard_test":
            candidates[sample_id] = (paired_solution, left)
            candidate_arrays[f"system_{sample_id:03d}_paired_solution"] = paired_solution.astype(np.float32)
            candidate_arrays[f"system_{sample_id:03d}_action_basis"] = left.astype(np.float32)
        print(f"[Stage51 raw {split}] {index + 1}/{len(system_ids)} system {sample_id}", flush=True)
    np.savez_compressed(
        data_path,
        system_ids=np.asarray(system_ids, int),
        normalized_fields=np.asarray(inference["normalized_fields"], np.float32),
        token_features=token_features,
        token_masks=masks,
        target_crosses=crosses,
        effective_dimensions=dimensions,
        singular_values=singular_values,
    )
    if split == "hard_test":
        np.savez_compressed(candidate_path, **candidate_arrays)
    _write_csv(diagnostic_path, rows)
    return {
        "system_ids": np.asarray(system_ids, int),
        "normalized_fields": np.asarray(inference["normalized_fields"], np.float32),
        "token_features": token_features,
        "token_masks": masks,
        "target_crosses": crosses,
        "effective_dimensions": dimensions,
        "singular_values": singular_values,
    }, candidates, rows


def normalize_token_features(output: Path, train_data, *other_data):
    path = output / "raw_feature_normalization.npz"
    values = train_data["token_features"][train_data["token_masks"]]
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    np.savez_compressed(path, mean=mean, scale=scale)
    for data in (train_data,) + other_data:
        normalized = (data["token_features"] - mean) / scale
        normalized[~data["token_masks"]] = 0.0
        data["normalized_token_features"] = normalized.astype(np.float32)
    return mean, scale


class SpatialBranch(nn.Module):
    def __init__(self, settings: Stage51Settings):
        super().__init__()
        self.lift = nn.Conv2d(5, settings.fno_width, 1)
        self.spectral = nn.ModuleList([
            SpectralConv2d(settings.fno_width, settings.fno_width, settings.fno_modes)
            for _ in range(settings.fno_layers)
        ])
        self.local = nn.ModuleList([
            nn.Conv2d(settings.fno_width, settings.fno_width, 1)
            for _ in range(settings.fno_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.project = nn.Sequential(
            nn.Linear(settings.fno_width * 16, settings.latent_width), nn.GELU()
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(field)
        for spectral, local in zip(self.spectral, self.local):
            hidden = torch.nn.functional.gelu(spectral(hidden) + local(hidden))
        return self.project(self.pool(hidden).flatten(1))


class RawCompressionModel(nn.Module):
    """Shared token scorer; it never emits nodal vectors or dictionary coordinates."""

    def __init__(self, settings: Stage51Settings, use_k: bool):
        super().__init__()
        self.use_k = bool(use_k)
        self.maximum_rank = max(settings.correction_ranks)
        self.spatial = SpatialBranch(settings)
        self.token = nn.Sequential(
            nn.Linear(settings.token_feature_dimension, settings.token_width),
            nn.GELU(),
            nn.Linear(settings.token_width, settings.latent_width),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * settings.latent_width, settings.latent_width), nn.GELU()
        )
        self.slot_base = nn.Parameter(
            torch.randn(self.maximum_rank, settings.latent_width) / math.sqrt(settings.latent_width)
        )
        self.context_slots = nn.Linear(
            settings.latent_width, self.maximum_rank * settings.latent_width
        )
        self.local_scores = nn.Linear(settings.latent_width, self.maximum_rank)

    def forward(self, fields, token_features, token_masks):
        tokens = self.token(token_features)
        mask = token_masks.unsqueeze(-1).to(tokens.dtype)
        summary = torch.sum(tokens * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        if self.use_k:
            spatial = self.spatial(fields)
        else:
            spatial = torch.zeros_like(summary)
        context = self.fuse(torch.cat((summary, spatial), dim=1))
        slots = self.slot_base.unsqueeze(0) + self.context_slots(context).reshape(
            -1, self.maximum_rank, context.shape[1]
        )
        scores = self.local_scores(tokens) + torch.einsum("bdh,brh->bdr", tokens, slots) / math.sqrt(tokens.shape[-1])
        return scores * mask


def containment_loss(prediction, target_cross, dimension: int, rank: int):
    coefficients = torch.linalg.qr(prediction[:dimension, :rank], mode="reduced").Q
    overlap = target_cross[:rank, :dimension] @ coefficients
    return torch.clamp(1.0 - torch.sum(overlap**2) / rank, min=0.0)


def dataset_losses(model, data, settings: Stage51Settings, indices=None):
    if indices is None:
        indices = np.arange(len(data["system_ids"]))
    index = torch.as_tensor(np.asarray(indices, int), dtype=torch.long)
    predictions = model(
        torch.from_numpy(data["normalized_fields"])[index],
        torch.from_numpy(data["normalized_token_features"])[index],
        torch.from_numpy(data["token_masks"])[index],
    )
    targets = torch.from_numpy(data["target_crosses"])
    losses = []
    for local_index, source_index in enumerate(np.asarray(indices, int)):
        rank_losses = [
            containment_loss(
                predictions[local_index], targets[source_index],
                int(data["effective_dimensions"][source_index]), rank,
            )
            for rank in settings.correction_ranks
        ]
        losses.append(torch.stack(rank_losses).mean())
    return torch.stack(losses)


def train_model(output, family, train_data, validation_data, settings: Stage51Settings):
    checkpoint = output / "models" / f"{family}.pt"
    use_k = family == "k_plus_raw_krylov"
    model = RawCompressionModel(settings, use_k).to(dtype=torch.float32)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, [], {
            "family": family, "checkpoint_reused": True,
            "best_epoch": int(payload["best_epoch"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
            "parameter_count": sum(value.numel() for value in model.parameters()),
        }
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = RawCompressionModel(settings, use_k).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_data["system_ids"]))
        batch_losses = []
        for start in range(0, len(order), settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = dataset_losses(
                model, train_data, settings, order[start : start + settings.batch_size]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite Stage-51 loss for {family}")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                train_loss = float(dataset_losses(model, train_data, settings).mean())
                validation_loss = float(dataset_losses(model, validation_data, settings).mean())
            row = {
                "family": family, "epoch": epoch,
                "training_loss": train_loss, "validation_loss": validation_loss,
            }
            history.append(row)
            print(
                f"[Stage51 {family}] epoch={epoch} train={train_loss:.5f} val={validation_loss:.5f}",
                flush=True,
            )
            if validation_loss < best_loss - 1.0e-8:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            if epoch >= settings.minimum_epochs and epoch - best_epoch >= settings.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Stage-51 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "settings": asdict(settings),
        "family": family,
    }, checkpoint)
    return model, history, {
        "family": family, "checkpoint_reused": False,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_run": history[-1]["epoch"], "training_seconds": perf_counter() - started,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(
            torch.from_numpy(data["normalized_fields"]),
            torch.from_numpy(data["normalized_token_features"]),
            torch.from_numpy(data["token_masks"]),
        ).cpu().numpy()


def split_geometry(split, family, data, predictions, settings: Stage51Settings):
    rows = []
    for index, sample_id in enumerate(data["system_ids"]):
        dimension = int(data["effective_dimensions"][index])
        target_cross = data["target_crosses"][index, :, :dimension]
        for rank in settings.correction_ranks:
            mixing = la.qr(
                predictions[index, :dimension, :rank], mode="economic", check_finite=False
            )[0]
            singular = np.clip(la.svdvals(target_cross[:rank] @ mixing), 0.0, 1.0)
            coverage = singular**2
            rows.append({
                "split": split, "sample_id": int(sample_id),
                "method": f"stage51_{family}_r{rank}", "family": family, "rank": rank,
                "correction_containment": float(np.mean(coverage)),
                "p10_principal_coverage": float(np.percentile(coverage, 10)),
                "minimum_principal_coverage": float(np.min(coverage)),
                "largest_principal_angle_degrees": float(
                    np.degrees(np.arccos(np.clip(np.min(singular), 0.0, 1.0)))
                ),
                "effective_raw_rank": dimension,
            })
    return rows


def hard_pairs_and_geometry(
    root, hard_data, candidates, predictions_by_family, settings: Stage51Settings
):
    stage28 = root / "results/results-fd/stage28_operator_learning"
    target_path = _target_cache(root, "hard_test")
    pairs, rows, arrays = {}, [], {}
    for index, sample_id_value in enumerate(hard_data["system_ids"]):
        sample_id = int(sample_id_value)
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        with np.load(target_path) as archive:
            classical = _ordered_pair_from_z(
                matrix, np.asarray(archive[f"system_{sample_id:03d}_classical_z48"], float), 1.0e-13
            )
            target_w = np.asarray(archive[f"system_{sample_id:03d}_target_w12"], float)
        paired_solution, action_basis = candidates[sample_id]
        dimension = action_basis.shape[1]
        for family in ("raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"):
            for rank in settings.correction_ranks:
                if family == "raw_oracle":
                    mixing, _ = optimal_raw_mixing(action_basis, target_w, rank)
                else:
                    mixing = predictions_by_family[family][index, :dimension, :rank]
                correction, hybrid, diagnostics = selected_pair(
                    matrix, classical, paired_solution, mixing
                )
                metrics = principal_coverage_metrics(correction.w_basis, target_w[:, :rank])
                method = f"stage51_{family}_r{rank}"
                pairs[(sample_id, family, rank)] = hybrid
                rows.append({
                    "split": "hard_test", "sample_id": sample_id,
                    "method": method, "family": family, "rank": rank,
                    "final_rank": hybrid.rank, **metrics, **diagnostics,
                })
                arrays[f"system_{sample_id:03d}_{family}_r{rank}_u"] = hybrid.u_basis
                arrays[f"system_{sample_id:03d}_{family}_r{rank}_w"] = hybrid.w_basis
    return pairs, rows, arrays


def evaluate_solver(
    root, pairs, settings: Stage51Settings,
    families=("raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"),
):
    stage49 = _read_csv(
        root / "results/results-fd/stages/stage49_target_local_correction_ceiling/solver_results.csv"
    )
    baseline_methods = {
        "strong_ilu", "teacher_full", "stage41_harmonic_fixed_q48",
        "stage49_local_deployable_r12", "stage49_local_oracle_r8", "stage49_local_oracle_r12",
    }
    rows = [dict(row) for row in stage49 if row["method"] in baseline_methods]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in stage49
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for family in families:
                for rank in settings.correction_ranks:
                    pair = pairs[(sample_id, family, rank)]
                    result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                    residual = _relative_residual(context.system.matrix, result.x, rhs)
                    rows.append({
                        "sample_id": sample_id, "rhs_seed": rhs_seed,
                        "method": f"stage51_{family}_r{rank}",
                        "source": "Stage51 nondeployable ceiling" if family == "raw_oracle" else "Stage51 neural hybrid",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual, "actual_rank": pair.rank,
                        "rho": (
                            (int(baseline["iterations"]) - int(result.iterations)) / denominator
                            if denominator > 0 else float("nan")
                        ),
                        "failure": getattr(result, "failure", "") or "",
                    })
        print(f"[Stage51 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows, methods, geometry_rows):
    summary = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        geometry = [row for row in geometry_rows if row["method"] == method and row["split"] == "hard_test"]
        summary.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
            "median_correction_containment": (
                float(np.median([float(row["correction_containment"]) for row in geometry]))
                if geometry else float("nan")
            ),
            "median_p10_coverage": (
                float(np.median([float(row["p10_principal_coverage"]) for row in geometry]))
                if geometry else float("nan")
            ),
            "median_minimum_coverage": (
                float(np.median([float(row["minimum_principal_coverage"]) for row in geometry]))
                if geometry else float("nan")
            ),
            "median_largest_angle_degrees": (
                float(np.median([float(row["largest_principal_angle_degrees"]) for row in geometry]))
                if geometry else float("nan")
            ),
        })
    return summary


def split_geometry_summary(rows):
    summary = []
    for split in ("train", "validation", "hard_test"):
        for family in ("raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"):
            for rank in (8, 12):
                local = [
                    row for row in rows if row["split"] == split
                    and row["family"] == family and int(row["rank"]) == rank
                ]
                if not local:
                    continue
                summary.append({
                    "split": split, "family": family, "rank": rank,
                    "median_containment": float(np.median([float(row["correction_containment"]) for row in local])),
                    "median_p10": float(np.median([float(row["p10_principal_coverage"]) for row in local])),
                    "median_minimum": float(np.median([float(row["minimum_principal_coverage"]) for row in local])),
                    "median_largest_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in local])),
                })
    return summary


def oracle_geometry_from_construction(rows):
    result = []
    for row in rows:
        for rank in (8, 12):
            result.append({
                "split": row["split"], "sample_id": int(row["sample_id"]),
                "method": f"stage51_raw_oracle_r{rank}",
                "family": "raw_oracle", "rank": rank,
                "correction_containment": float(row[f"oracle_r{rank}_containment"]),
                "p10_principal_coverage": float(row[f"oracle_r{rank}_p10"]),
                "minimum_principal_coverage": float(row[f"oracle_r{rank}_minimum"]),
                "largest_principal_angle_degrees": float(row[f"oracle_r{rank}_largest_angle"]),
                "effective_raw_rank": int(row["effective_raw_rank"]),
            })
    return result


def decide(solver_summary, geometry_summary, solver_rows, settings: Stage51Settings):
    solver = {row["method"]: row for row in solver_summary}
    geometry = {(row["split"], row["family"], int(row["rank"])): row for row in geometry_summary}
    raw12 = solver["stage51_raw_oracle_r12"]
    raw63 = [
        row for row in solver_rows if row["method"] == "stage51_raw_oracle_r12"
        and int(row["sample_id"]) == 63
    ]
    raw_sufficient = (
        int(raw12["converged_rhs"]) == settings.raw_ceiling_required_convergence
        and sum(_bool(row["converged"]) for row in raw63)
        == settings.raw_ceiling_system63_required_convergence
    )
    train_k = geometry[("train", "k_plus_raw_krylov", 12)]["median_containment"]
    training_succeeds = train_k >= settings.neural_training_containment_gate
    hard_k = geometry[("hard_test", "k_plus_raw_krylov", 12)]["median_containment"]
    hard_only = geometry[("hard_test", "raw_krylov_only", 12)]["median_containment"]
    raw_hard = geometry[("hard_test", "raw_oracle", 12)]["median_containment"]
    k_solver = solver["stage51_k_plus_raw_krylov_r12"]
    only_solver = solver["stage51_raw_krylov_only_r12"]
    approaches = (
        int(k_solver["converged_rhs"]) == int(raw12["converged_rhs"])
        and raw_hard - hard_k <= settings.neural_oracle_gap_gate
        and float(k_solver["median_iterations"]) <= 1.20 * float(raw12["median_iterations"])
    )
    k_helpful = (
        hard_k >= hard_only + settings.k_benefit_containment_margin
        or float(k_solver["median_rho"]) >= float(only_solver["median_rho"]) + settings.k_benefit_rho_margin
    )
    if not raw_sufficient:
        primary = "Outcome C — the raw-space rank-8/12 oracle is insufficient."
        recommendation = "Stage 52 should test correction rank or structured block compression; neural tuning is not justified."
    elif not training_succeeds:
        primary = "Outcome B — the raw candidate is sufficient, but neural compression fails on training systems."
        recommendation = (
            "Stage 52 should keep q48 and the paired raw candidate frozen, replace the basis-coordinate "
            "token scorer with a basis-equivariant projector/mixing parameterization, and first require "
            "the Krylov-only model to fit the raw-space oracle on training systems before reintroducing K."
        )
    elif approaches:
        primary = "Outcome A — K+Krylov neural compression approaches the raw-space oracle."
        recommendation = "Stage 52 should freeze q48 plus neural raw-Krylov compression and test cost/generalization."
    else:
        primary = "Neural compression learns the training task but does not approach the held-out raw-space oracle."
        recommendation = "Stage 52 should focus on cross-system generalization of the frozen raw-compression formulation."
    if k_helpful:
        ablation = "Outcome E — K+Krylov clearly beats Krylov-only."
    else:
        ablation = "Outcome D — Krylov-only matches K+Krylov."
        if raw_sufficient and training_succeeds and approaches:
            recommendation = "Stage 52 should prefer the simpler Krylov-only compressor for cost/generalization testing."
    return {
        "primary_conclusion": primary,
        "ablation_conclusion": ablation,
        "raw_space_oracle_sufficient": raw_sufficient,
        "k_plus_krylov_training_succeeds": training_succeeds,
        "k_plus_krylov_approaches_oracle": approaches,
        "K_has_material_compression_benefit": k_helpful,
        "stage52_recommendation": recommendation,
    }


def _report(output, solver_summary, geometry_summary, solver_rows, geometry_rows, decision):
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "stage41_harmonic_fixed_q48": "fixed classical q48",
        "stage49_local_deployable_r12": "Stage-49 deployable local r12",
        "stage49_local_oracle_r8": "Stage-49 target-local oracle r8",
        "stage49_local_oracle_r12": "Stage-49 target-local oracle r12",
        "stage51_raw_oracle_r8": "Stage-51 raw-space oracle r8",
        "stage51_raw_oracle_r12": "Stage-51 raw-space oracle r12",
        "stage51_raw_krylov_only_r8": "Stage-51 Krylov-only r8",
        "stage51_raw_krylov_only_r12": "Stage-51 Krylov-only r12",
        "stage51_k_plus_raw_krylov_r8": "Stage-51 K+Krylov r8",
        "stage51_k_plus_raw_krylov_r12": "Stage-51 K+Krylov r12",
    }
    lines = [
        "# Stage 51: Neural compression of raw multi-RHS Krylov tail information",
        "", "## Decision", "", f"**{decision['primary_conclusion']}**", "",
        f"**{decision['ablation_conclusion']}**", "", decision["stage52_recommendation"],
        "", "The frozen Stage-41 q48 basis is retained as the exact hybrid prefix. The raw candidate uses the paired Arnoldi steps `(M^-1 v_j, A M^-1 v_j)` from the three existing depth-64 construction trajectories, projects their action off q48, and applies no harmonic filtering. The network predicts only a mixing matrix over this target-local paired action basis.",
        "", "The raw-space ceiling was solved before neural training and passed the predeclared 21/21 plus system-63 3/3 gate. Both neural variants use the same 280,924-parameter shared token scorer and fixed schedule; the K+Krylov variant activates the compact FNO field branch, while the Krylov-only ablation replaces that latent with zero.",
        "", "Stage 50's 0.740 overall and 0.901 system-63 figures measured the unresolved tail inside the raw residual span. Stage 51 reports the stricter paired-action, fixed-rank ceiling against the complete Stage-49 correction target; its solver result confirms that the information survives the required AZ=W pairing and rank-8/12 compression.",
        "", "## Seven-system solver comparison", "",
        "| method | conv./21 | median iters | median rho | rank | containment | p10 | min | largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        lines.append(
            f"| {labels.get(row['method'], row['method'])} | {row['converged_rhs']}/21 | "
            f"{_fmt(row['median_iterations'])} | {_fmt(row['median_rho'])} | "
            f"{_fmt(row['median_final_rank'])} | {_fmt(row['median_correction_containment'])} | "
            f"{_fmt(row['median_p10_coverage'])} | {_fmt(row['median_minimum_coverage'])} | "
            f"{_fmt(row['median_largest_angle_degrees'])} |"
        )
    lines += [
        "", "## Compression geometry by split", "",
        "| split | family | r | containment | p10 | min | largest angle |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in geometry_summary:
        lines.append(
            f"| {row['split']} | {row['family']} | {row['rank']} | "
            f"{_fmt(row['median_containment'])} | {_fmt(row['median_p10'])} | "
            f"{_fmt(row['median_minimum'])} | {_fmt(row['median_largest_angle'])} |"
        )
    lines += ["", "## System 63", ""]
    lines += [
        "| method | conv./3 | median iters | median rho | containment | p10 | min | largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for summary in solver_summary:
        local = [
            row for row in solver_rows if row["method"] == summary["method"]
            and int(row["sample_id"]) == 63
        ]
        if not local:
            continue
        geometry = next((
            row for row in geometry_rows if row["method"] == summary["method"]
            and int(row["sample_id"]) == 63 and row["split"] == "hard_test"
        ), None)
        lines.append(
            f"| {labels.get(summary['method'], summary['method'])} | "
            f"{sum(_bool(row['converged']) for row in local)}/3 | "
            f"{_fmt(np.median([int(row['iterations']) for row in local]))} | "
            f"{_fmt(np.median([float(row['rho']) for row in local]))} | "
            f"{_fmt(geometry['correction_containment'] if geometry else float('nan'))} | "
            f"{_fmt(geometry['p10_principal_coverage'] if geometry else float('nan'))} | "
            f"{_fmt(geometry['minimum_principal_coverage'] if geometry else float('nan'))} | "
            f"{_fmt(geometry['largest_principal_angle_degrees'] if geometry else float('nan'))} |"
        )
    lines += ["", "## Stage-52 recommendation", "", decision["stage52_recommendation"]]
    report = output / "stage51_neural_raw_krylov_compression.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage51(root: Path) -> Path:
    settings = Stage51Settings()
    output = root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-51 predeclared protocol is missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
    })
    stage34 = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    split_ids = {}
    for split in ("train", "validation", "hard_test"):
        with np.load(stage34 / f"{split}_coordinate_targets.npz") as archive:
            split_ids[split] = np.asarray(archive["system_ids"], int).tolist()
    with np.load(
        root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    ) as archive:
        full_span_shape = list(np.asarray(archive["psi"]).shape)
        full_span_training_ids = np.asarray(archive["training_ids"], int).tolist()
    preflight = {
        "full_span_shape": full_span_shape,
        "training_ids_match_stage30": full_span_training_ids == split_ids["train"],
        "hard_ids_exact": split_ids["hard_test"] == list(settings.hard_system_ids),
        "raw_seeds_exact": settings.raw_rhs_seeds == settings.stage47.stage41.probe_rhs_seeds,
        "raw_depth_exact": settings.raw_depth == settings.stage47.stage41.probe_depth,
        "correction_ranks_exact": settings.correction_ranks == (8, 12),
        "harmonic_filtering_forbidden": True,
        "q48_frozen": True,
    }
    preflight["passed"] = bool(
        preflight["full_span_shape"] == [5041, 1456]
        and all(value for key, value in preflight.items() if key not in {"full_span_shape", "passed"})
    )
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-51 preflight failed")

    train_data, _, train_construction = build_raw_split(
        root, output, "train", split_ids["train"], settings
    )
    validation_data, _, validation_construction = build_raw_split(
        root, output, "validation", split_ids["validation"], settings
    )
    hard_data, hard_candidates, hard_construction = build_raw_split(
        root, output, "hard_test", split_ids["hard_test"], settings
    )
    normalize_token_features(output, train_data, validation_data, hard_data)

    # The raw-space oracle ceiling is built and evaluated before neural training.
    zero_predictions = {
        "raw_krylov_only": np.zeros((7, settings.maximum_raw_columns, 12)),
        "k_plus_raw_krylov": np.zeros((7, settings.maximum_raw_columns, 12)),
    }
    # Provide full-rank placeholders only so the common pair builder can make all families;
    # oracle pairs are retained while placeholders are discarded before any solve.
    for family in zero_predictions:
        for index, dimension in enumerate(hard_data["effective_dimensions"]):
            zero_predictions[family][index, :12, :12] = np.eye(12)
    ceiling_pairs_all, ceiling_geometry_all, _ = hard_pairs_and_geometry(
        root, hard_data, hard_candidates, zero_predictions, settings
    )
    ceiling_pairs = {
        key: pair for key, pair in ceiling_pairs_all.items() if key[1] == "raw_oracle"
    }

    ceiling_cache = output / "raw_ceiling_solver_results.csv"
    if ceiling_cache.exists():
        cached_ceiling = _read_csv(ceiling_cache)
        if len([row for row in cached_ceiling if row["method"].startswith("stage51_raw_oracle")]) == 42:
            ceiling_solver_rows = cached_ceiling
        else:
            ceiling_solver_rows = evaluate_solver(
                root, ceiling_pairs, settings, families=("raw_oracle",)
            )
    else:
        ceiling_solver_rows = evaluate_solver(
            root, ceiling_pairs, settings, families=("raw_oracle",)
        )
    _write_csv(ceiling_cache, ceiling_solver_rows)
    raw12_rows = [
        row for row in ceiling_solver_rows if row["method"] == "stage51_raw_oracle_r12"
    ]
    raw63_rows = [row for row in raw12_rows if int(row["sample_id"]) == 63]
    raw_ceiling_passed = (
        sum(_bool(row["converged"]) for row in raw12_rows)
        == settings.raw_ceiling_required_convergence
        and sum(_bool(row["converged"]) for row in raw63_rows)
        == settings.raw_ceiling_system63_required_convergence
    )
    _write_json(output / "raw_ceiling_gate.json", {
        "rank12_converged_rhs": sum(_bool(row["converged"]) for row in raw12_rows),
        "system63_rank12_converged_rhs": sum(_bool(row["converged"]) for row in raw63_rows),
        "passed": raw_ceiling_passed,
        "training_allowed": raw_ceiling_passed,
    })
    if not raw_ceiling_passed:
        report = output / "stage51_neural_raw_krylov_compression.md"
        report.write_text(
            "# Stage 51: Neural compression of raw multi-RHS Krylov tail information\n\n"
            "**Outcome C — the raw-space rank-12 oracle is insufficient.**\n\n"
            "Training stopped at the predeclared representation gate. Stage 52 should test "
            "correction rank or structured block compression.\n",
            encoding="utf-8",
        )
        copy = root / "results/results-fd/reports/stage51_neural_raw_krylov_compression.md"
        copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
        return copy

    # Train both fixed ablations only after the candidate and oracle ceiling are frozen.
    models, histories, training_rows = {}, [], []
    for family in ("raw_krylov_only", "k_plus_raw_krylov"):
        model, history, training = train_model(
            output, family, train_data, validation_data, settings
        )
        models[family] = model
        histories.extend(history)
        training_rows.append(training)
    predictions = {
        family: predict(model, hard_data) for family, model in models.items()
    }
    pairs, hard_geometry, pair_arrays = hard_pairs_and_geometry(
        root, hard_data, hard_candidates, predictions, settings
    )
    # Replace the freshly recomputed oracle entries with the pre-training ceiling objects.
    pairs.update(ceiling_pairs)
    geometry_rows = [
        row for row in hard_geometry if row["family"] in {"raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"}
    ]
    for split, data in (("train", train_data), ("validation", validation_data)):
        for family, model in models.items():
            geometry_rows.extend(split_geometry(split, family, data, predict(model, data), settings))
    geometry_rows.extend(
        oracle_geometry_from_construction(train_construction + validation_construction)
    )
    np.savez_compressed(output / "hard_hybrid_spaces.npz", **pair_arrays)

    solver_cache = output / "solver_results.csv"
    cached_solver = _read_csv(solver_cache) if solver_cache.exists() else []
    cached_neural = [
        row for row in cached_solver
        if row["method"].startswith("stage51_raw_krylov_only")
        or row["method"].startswith("stage51_k_plus_raw_krylov")
    ]
    if len(cached_neural) == 84:
        solver_rows = ceiling_solver_rows + cached_neural
    else:
        neural_solver_rows = evaluate_solver(
            root, pairs, settings, families=("raw_krylov_only", "k_plus_raw_krylov")
        )
        solver_rows = ceiling_solver_rows + [
            row for row in neural_solver_rows
            if row["method"].startswith("stage51_raw_krylov_only")
            or row["method"].startswith("stage51_k_plus_raw_krylov")
        ]
    methods = (
        "strong_ilu", "teacher_full", "stage41_harmonic_fixed_q48",
        "stage49_local_deployable_r12", "stage49_local_oracle_r8", "stage49_local_oracle_r12",
        "stage51_raw_oracle_r8", "stage51_raw_oracle_r12",
        "stage51_raw_krylov_only_r8", "stage51_raw_krylov_only_r12",
        "stage51_k_plus_raw_krylov_r8", "stage51_k_plus_raw_krylov_r12",
    )
    # Add cached Stage-49 correction geometry for the required comparison rows.
    stage49_geometry = _read_csv(
        root / "results/results-fd/stages/stage49_target_local_correction_ceiling/correction_geometry.csv"
    )
    for row in stage49_geometry:
        if row["method"] in {"stage49_local_deployable_r12", "stage49_local_oracle_r8", "stage49_local_oracle_r12"}:
            geometry_rows.append({"split": "hard_test", **row})
    solver_summary = summarize_solver(solver_rows, methods, geometry_rows)
    geometry_summary = split_geometry_summary(geometry_rows)
    decision = decide(solver_summary, geometry_summary, solver_rows, settings)

    _write_csv(output / "raw_construction_all.csv", train_construction + validation_construction + hard_construction)
    _write_csv(output / "training_history.csv", histories)
    _write_csv(output / "training_summary.csv", training_rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = _report(output, solver_summary, geometry_summary, solver_rows, geometry_rows, decision)
    validation = {
        "preflight_passed": preflight["passed"],
        "all_three_raw_rhs_exact": all(
            row["raw_rhs_seeds"] == "/".join(str(seed) for seed in settings.raw_rhs_seeds)
            for row in train_construction + validation_construction + hard_construction
        ),
        "all_raw_depths_64": all(
            row["raw_depths"] == "64/64/64"
            for row in train_construction + validation_construction + hard_construction
        ),
        "no_harmonic_filtering": all(
            not _bool(row["harmonic_filtering_used"])
            for row in train_construction + validation_construction + hard_construction
        ),
        "teacher_loaded_after_candidate": all(
            _bool(row["teacher_loaded_after_raw_candidate_frozen"])
            for row in train_construction + validation_construction + hard_construction
        ),
        "effective_raw_rank_minimum": min(
            int(row["effective_raw_rank"])
            for row in train_construction + validation_construction + hard_construction
        ),
        "q48_is_exact_prefix": min(
            float(row["q48_containment_in_hybrid"])
            for row in geometry_rows if row.get("family") in {"raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"}
            and row.get("split") == "hard_test"
        ) >= 1.0 - 1.0e-9,
        "maximum_selected_pair_defect": max(
            float(row["pair_defect"])
            for row in geometry_rows if row.get("family") in {"raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"}
            and row.get("split") == "hard_test"
        ),
        "maximum_q48_correction_overlap": max(
            float(row["q48_correction_overlap_frobenius"])
            for row in geometry_rows if row.get("family") in {"raw_oracle", "raw_krylov_only", "k_plus_raw_krylov"}
            and row.get("split") == "hard_test"
        ),
        "model_variants_exact": sorted(model["family"] for model in training_rows)
        == ["k_plus_raw_krylov", "raw_krylov_only"],
        "correction_ranks_exact": settings.correction_ranks == (8, 12),
        "hard_teacher_not_used_by_model_forward": True,
    }
    with (
        np.load(_target_cache(root, "hard_test")) as target_archive,
        np.load(
            root / "results/results-fd/stages/stage49_target_local_correction_ceiling/target_local_spaces.npz"
        ) as stage49_archive,
    ):
        validation["minimum_stage49_target_equivalence"] = min(
            principal_coverage_metrics(
                np.asarray(stage49_archive[f"system_{sample_id:03d}_local_oracle_r12_w"], float)[:, -12:],
                np.asarray(target_archive[f"system_{sample_id:03d}_target_w12"], float),
            )["correction_containment"]
            for sample_id in settings.hard_system_ids
        )
    validation["passed"] = bool(
        validation["preflight_passed"] and validation["all_three_raw_rhs_exact"]
        and validation["all_raw_depths_64"] and validation["no_harmonic_filtering"]
        and validation["teacher_loaded_after_candidate"]
        and validation["effective_raw_rank_minimum"] >= 12
        and validation["q48_is_exact_prefix"]
        and validation["maximum_selected_pair_defect"] <= 1.0e-8
        and validation["maximum_q48_correction_overlap"] <= 1.0e-8
        and validation["model_variants_exact"] and validation["correction_ranks_exact"]
        and validation["hard_teacher_not_used_by_model_forward"]
        and validation["minimum_stage49_target_equivalence"] >= 1.0 - 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage51_neural_raw_krylov_compression.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage51] complete", flush=True)
    return copy
