"""Stage 52: basis-equivariant learning over frozen Stage-51 raw candidates."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import torch
from torch import nn

from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _run_pair
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage49_target_local_correction_ceiling import principal_coverage_metrics
from .stage51_neural_raw_krylov_compression import (
    RawCompressionModel,
    Stage51Settings,
    selected_pair,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage52Settings:
    model_seed: int = 20262821
    rotation_seed: int = 20265200
    rotations_per_target: int = 3
    correction_ranks: Tuple[int, int] = (8, 12)
    primitive_operator_count: int = 6
    spectral_quantiles: int = 6
    hidden_width: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 6
    maximum_epochs: int = 200
    minimum_epochs: int = 75
    validation_interval: int = 5
    early_stopping_patience: int = 50
    sensitivity_distance_gate: float = 0.10
    sensitivity_containment_range_gate: float = 0.10
    training_gap_closed_fraction: float = 0.50
    training_absolute_gain: float = 0.15
    equivariance_median_tolerance: float = 1.0e-7
    equivariance_worst_tolerance: float = 1.0e-5
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage51(self) -> Stage51Settings:
        return Stage51Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return self.stage51.stage29

    @property
    def operator_basis_count(self) -> int:
        p = self.primitive_operator_count
        return p + p * (p + 1) // 2 + p

    @property
    def invariant_dimension(self) -> int:
        p = self.primitive_operator_count
        return p + p * (p + 1) // 2 + self.spectral_quantiles


def _load_stage51_data(root: Path, split: str) -> Dict[str, np.ndarray]:
    path = (
        root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
        / f"{split}_raw_loss_data.npz"
    )
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _orthogonal_rotation(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rotation, triangular = la.qr(rng.normal(size=(dimension, dimension)), mode="economic")
    signs = np.where(np.diag(triangular) >= 0.0, 1.0, -1.0)
    return rotation * signs[None, :]


def _mixing_basis(prediction: np.ndarray, dimension: int, rank: int) -> np.ndarray:
    return la.qr(
        np.asarray(prediction[:dimension, :rank], float), mode="economic", check_finite=False
    )[0]


def _subspace_distance(first: np.ndarray, second: np.ndarray) -> float:
    # The direct projector norm avoids the sqrt(machine-epsilon) floor of
    # 1-mean(svdvals(...)**2) when two spaces agree to roundoff.
    rank = first.shape[1]
    difference = first @ first.T - second @ second.T
    return float(la.norm(difference, ord="fro") / np.sqrt(2.0 * rank))


def stage51_rotation_sensitivity(
    root: Path, train_data: Dict[str, np.ndarray], settings: Stage52Settings
) -> List[Dict[str, Any]]:
    """Test the frozen Stage-51 scorer under dense coordinate rotations."""

    stage51_root = root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
    payload = torch.load(stage51_root / "models/raw_krylov_only.pt", map_location="cpu")
    model = RawCompressionModel(settings.stage51, use_k=False).to(dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with np.load(stage51_root / "raw_feature_normalization.npz") as archive:
        mean = np.asarray(archive["mean"], np.float32)
        scale = np.asarray(archive["scale"], np.float32)
    rows = []
    for index, sample_value in enumerate(train_data["system_ids"]):
        sample_id = int(sample_value)
        dimension = int(train_data["effective_dimensions"][index])
        raw_features = np.asarray(train_data["token_features"][index, :dimension], float)
        original_normalized = (raw_features - mean) / scale
        mask = np.ones((1, dimension), bool)
        field = np.asarray(train_data["normalized_fields"][index : index + 1], np.float32)
        with torch.no_grad():
            original_prediction = model(
                torch.from_numpy(field),
                torch.from_numpy(original_normalized[None].astype(np.float32)),
                torch.from_numpy(mask),
            )[0].cpu().numpy()
        target_cross = np.asarray(train_data["target_crosses"][index, :, :dimension], float)
        for rotation_index in range(settings.rotations_per_target):
            rotation = _orthogonal_rotation(
                dimension, settings.rotation_seed + 100 * sample_id + rotation_index
            )
            rotated_raw = np.empty_like(raw_features)
            # DCT coefficients are covariant coordinate vectors.
            rotated_raw[:, :36] = rotation.T @ raw_features[:, :36]
            # The remaining cached scalars are diagonal coordinate-operator
            # observations. Under Q -> QR their diagonals transform by R**2.
            rotated_raw[:, 36:] = (rotation**2).T @ raw_features[:, 36:]
            rotated_normalized = (rotated_raw - mean) / scale
            with torch.no_grad():
                rotated_prediction = model(
                    torch.from_numpy(field),
                    torch.from_numpy(rotated_normalized[None].astype(np.float32)),
                    torch.from_numpy(mask),
                )[0].cpu().numpy()
            for rank in settings.correction_ranks:
                original = _mixing_basis(original_prediction, dimension, rank)
                rotated_coordinates = _mixing_basis(rotated_prediction, dimension, rank)
                rotated_physical = rotation @ rotated_coordinates
                original_containment = float(
                    la.norm(target_cross[:rank] @ original, ord="fro") ** 2 / rank
                )
                rotated_containment = float(
                    la.norm(target_cross[:rank] @ rotated_physical, ord="fro") ** 2 / rank
                )
                rows.append({
                    "split": "train", "sample_id": sample_id,
                    "rotation_index": rotation_index, "rank": rank,
                    "prediction_subspace_distance": _subspace_distance(original, rotated_physical),
                    "original_teacher_containment": original_containment,
                    "rotated_teacher_containment": rotated_containment,
                    "absolute_containment_change": abs(rotated_containment - original_containment),
                    "same_physical_candidate_range": True,
                })
    return rows


def _normalize_operator(operator: np.ndarray) -> np.ndarray:
    norm = float(la.norm(operator, ord="fro"))
    return operator / max(norm, np.finfo(float).tiny)


def primitive_operators(data: Dict[str, np.ndarray], index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return six symmetric physical/candidate operators and invariant features."""

    dimension = int(data["effective_dimensions"][index])
    features = np.asarray(data["token_features"][index, :dimension], float)
    singular = np.asarray(data["singular_values"][index, :dimension], float)
    spatial = features[:, :36]
    low_indices = [i * 6 + j for i in range(6) for j in range(6) if i + j <= 3]
    high_indices = [index for index in range(36) if index not in low_indices]
    energy = (singular / singular[0]) ** 2
    primitives = [np.diag(energy)]
    primitives.extend(np.diag(features[:, column]) for column in (39, 40, 41))
    primitives.append(spatial[:, low_indices] @ spatial[:, low_indices].T)
    primitives.append(spatial[:, high_indices] @ spatial[:, high_indices].T)
    normalized = np.stack([_normalize_operator(value) for value in primitives])
    return normalized, singular


def operator_polynomial_basis(primitives: np.ndarray) -> np.ndarray:
    """Low-order noncommutative symmetric polynomials; equivariant by construction."""

    values = [operator for operator in primitives]
    for first, second in combinations_with_replacement(range(primitives.shape[0]), 2):
        product = 0.5 * (
            primitives[first] @ primitives[second]
            + primitives[second] @ primitives[first]
        )
        values.append(_normalize_operator(product))
    for operator in primitives:
        values.append(_normalize_operator(operator @ operator @ operator))
    return np.stack(values).astype(np.float32)


def invariant_features(primitives: np.ndarray, singular: np.ndarray, count: int) -> np.ndarray:
    values = [float(np.trace(operator)) for operator in primitives]
    for first, second in combinations_with_replacement(range(primitives.shape[0]), 2):
        values.append(float(np.sum(primitives[first] * primitives[second])))
    positions = np.linspace(0, singular.size - 1, count)
    spectrum = np.interp(positions, np.arange(singular.size), singular / singular[0])
    values.extend(np.log10(np.maximum(spectrum, 1.0e-12)).tolist())
    return np.asarray(values, np.float32)


def build_equivariant_data(
    output: Path, split: str, source: Dict[str, np.ndarray], settings: Stage52Settings
) -> Dict[str, np.ndarray]:
    basis_path = output / f"{split}_operator_basis.npy"
    invariant_path = output / f"{split}_equivariant_data.npz"
    expected_shape = (
        len(source["system_ids"]), settings.operator_basis_count,
        settings.stage51.maximum_raw_columns, settings.stage51.maximum_raw_columns,
    )
    if basis_path.exists() and invariant_path.exists():
        basis = np.load(basis_path, mmap_mode="r")
        with np.load(invariant_path) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if basis.shape == expected_shape and data["system_ids"].tolist() == source["system_ids"].tolist():
            data["operator_basis"] = basis
            return data

    basis = np.lib.format.open_memmap(
        basis_path, mode="w+", dtype=np.float32, shape=expected_shape
    )
    invariants = np.zeros((len(source["system_ids"]), settings.invariant_dimension), np.float32)
    oracle_mixings = np.zeros((
        len(source["system_ids"]), len(settings.correction_ranks),
        settings.stage51.maximum_raw_columns, max(settings.correction_ranks),
    ), np.float32)
    for index, sample_id in enumerate(source["system_ids"]):
        dimension = int(source["effective_dimensions"][index])
        primitives, singular = primitive_operators(source, index)
        local_basis = operator_polynomial_basis(primitives)
        basis[index] = 0.0
        basis[index, :, :dimension, :dimension] = local_basis
        invariants[index] = invariant_features(
            primitives, singular, settings.spectral_quantiles
        )
        target_cross = np.asarray(source["target_crosses"][index, :, :dimension], float)
        for rank_index, rank in enumerate(settings.correction_ranks):
            left = la.svd(
                target_cross[:rank].T, full_matrices=False, check_finite=False
            )[0][:, :rank]
            oracle_mixings[index, rank_index, :dimension, :rank] = left
        basis.flush()
        print(
            f"[Stage52 operators {split}] {index + 1}/{len(source['system_ids'])} system {int(sample_id)}",
            flush=True,
        )
    np.savez_compressed(
        invariant_path,
        system_ids=np.asarray(source["system_ids"], int),
        invariants=invariants,
        oracle_mixings=oracle_mixings,
        effective_dimensions=np.asarray(source["effective_dimensions"], int),
        target_crosses=np.asarray(source["target_crosses"], np.float32),
    )
    return {
        "system_ids": np.asarray(source["system_ids"], int),
        "invariants": invariants,
        "oracle_mixings": oracle_mixings,
        "effective_dimensions": np.asarray(source["effective_dimensions"], int),
        "target_crosses": np.asarray(source["target_crosses"], np.float32),
        "operator_basis": basis,
    }


def normalize_invariants(output: Path, train_data, *others):
    mean = np.mean(train_data["invariants"], axis=0)
    scale = np.std(train_data["invariants"], axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    np.savez_compressed(output / "invariant_normalization.npz", mean=mean, scale=scale)
    for data in (train_data,) + others:
        data["normalized_invariants"] = ((data["invariants"] - mean) / scale).astype(np.float32)


class EquivariantOperatorModel(nn.Module):
    """Invariant scalar weights over equivariant symmetric operator polynomials."""

    def __init__(self, settings: Stage52Settings):
        super().__init__()
        self.ranks = settings.correction_ranks
        self.network = nn.Sequential(
            nn.Linear(settings.invariant_dimension, settings.hidden_width),
            nn.GELU(),
            nn.Linear(settings.hidden_width, len(self.ranks) * settings.operator_basis_count),
        )

    def weights(self, invariants: torch.Tensor) -> torch.Tensor:
        return self.network(invariants).reshape(
            -1, len(self.ranks), self.network[-1].out_features // len(self.ranks)
        )

    def forward(self, invariants: torch.Tensor, operator_basis: torch.Tensor) -> torch.Tensor:
        weights = self.weights(invariants)
        operators = torch.einsum("brk,bkij->brij", weights, operator_basis)
        return 0.5 * (operators + operators.transpose(-1, -2))


def projector_operator_loss(
    predicted_operator: torch.Tensor, oracle_mixing: torch.Tensor, dimension: int, rank: int
) -> torch.Tensor:
    operator = predicted_operator[:dimension, :dimension]
    mixing = oracle_mixing[:dimension, :rank]
    frobenius_squared = torch.sum(operator**2)
    projector_overlap = torch.trace(mixing.T @ operator @ mixing)
    return torch.clamp((frobenius_squared + rank - 2.0 * projector_overlap) / rank, min=0.0)


def dataset_losses(model, data, settings: Stage52Settings, indices=None):
    if indices is None:
        indices = np.arange(len(data["system_ids"]))
    indices = np.asarray(indices, int)
    invariants = torch.from_numpy(data["normalized_invariants"][indices])
    basis = torch.from_numpy(np.asarray(data["operator_basis"][indices], np.float32))
    predictions = model(invariants, basis)
    oracle = torch.from_numpy(data["oracle_mixings"])
    losses = []
    for local_index, source_index in enumerate(indices):
        dimension = int(data["effective_dimensions"][source_index])
        local = [
            projector_operator_loss(
                predictions[local_index, rank_index], oracle[source_index, rank_index],
                dimension, rank,
            )
            for rank_index, rank in enumerate(settings.correction_ranks)
        ]
        losses.append(torch.stack(local).mean())
    return torch.stack(losses)


def train_model(output: Path, train_data, validation_data, settings: Stage52Settings):
    checkpoint = output / "models/basis_equivariant_krylov_only.pt"
    model = EquivariantOperatorModel(settings).to(dtype=torch.float32)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, [], {
            "checkpoint_reused": True, "best_epoch": int(payload["best_epoch"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
            "parameter_count": sum(value.numel() for value in model.parameters()),
        }
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = EquivariantOperatorModel(settings).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_data["system_ids"]))
        batch_values = []
        for start in range(0, len(order), settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = dataset_losses(
                model, train_data, settings, order[start : start + settings.batch_size]
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite Stage-52 loss")
            loss.backward()
            optimizer.step()
            batch_values.append(float(loss.detach()))
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                train_loss = float(dataset_losses(model, train_data, settings).mean())
                validation_loss = float(dataset_losses(model, validation_data, settings).mean())
            row = {"epoch": epoch, "training_loss": train_loss, "validation_loss": validation_loss}
            history.append(row)
            print(
                f"[Stage52 equivariant] epoch={epoch} train={train_loss:.5f} val={validation_loss:.5f}",
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
        raise RuntimeError("Stage-52 produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "settings": asdict(settings),
    }, checkpoint)
    return model, history, {
        "checkpoint_reused": False, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "epochs_run": history[-1]["epoch"],
        "training_seconds": perf_counter() - started,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


def predict_mixings(model, data, settings: Stage52Settings):
    with torch.no_grad():
        operators = model(
        torch.from_numpy(data["normalized_invariants"]),
            torch.from_numpy(np.array(data["operator_basis"], dtype=np.float32, copy=True)),
        ).cpu().numpy()
    predictions = np.zeros_like(data["oracle_mixings"])
    for index in range(len(data["system_ids"])):
        dimension = int(data["effective_dimensions"][index])
        for rank_index, rank in enumerate(settings.correction_ranks):
            values, vectors = la.eigh(
                np.asarray(
                    operators[index, rank_index, :dimension, :dimension], dtype=np.float64
                ),
                check_finite=False,
            )
            predictions[index, rank_index, :dimension, :rank] = vectors[:, -rank:]
    return predictions, operators


def geometry_rows(split, data, predictions, settings: Stage52Settings):
    rows = []
    for index, sample_value in enumerate(data["system_ids"]):
        sample_id = int(sample_value)
        dimension = int(data["effective_dimensions"][index])
        target_cross = np.asarray(data["target_crosses"][index, :, :dimension], float)
        for rank_index, rank in enumerate(settings.correction_ranks):
            mixing = predictions[index, rank_index, :dimension, :rank]
            singular = np.clip(la.svdvals(target_cross[:rank] @ mixing), 0.0, 1.0)
            coverage = singular**2
            rows.append({
                "split": split, "sample_id": sample_id,
                "method": f"stage52_basis_equivariant_r{rank}",
                "family": "basis_equivariant", "rank": rank,
                "correction_containment": float(np.mean(coverage)),
                "p10_principal_coverage": float(np.percentile(coverage, 10)),
                "minimum_principal_coverage": float(np.min(coverage)),
                "largest_principal_angle_degrees": float(
                    np.degrees(np.arccos(np.clip(np.min(singular), 0.0, 1.0)))
                ),
            })
    return rows


def equivariance_control(data_by_split, model, settings: Stage52Settings):
    rows = []
    for split, data in data_by_split.items():
        predictions, operators = predict_mixings(model, data, settings)
        weights = model.weights(torch.from_numpy(data["normalized_invariants"])).detach().numpy()
        for index, sample_value in enumerate(data["system_ids"]):
            sample_id = int(sample_value)
            dimension = int(data["effective_dimensions"][index])
            for rotation_index in range(settings.rotations_per_target):
                rotation = _orthogonal_rotation(
                    dimension, settings.rotation_seed + 100 * sample_id + rotation_index
                )
                # Invariant inputs leave scalar weights unchanged. Rotating the
                # summed symmetric operator is algebraically identical to
                # rotating every equivariant polynomial basis member.
                for rank_index, rank in enumerate(settings.correction_ranks):
                    original = predictions[index, rank_index, :dimension, :rank]
                    rotated_operator = (
                        rotation.T
                        @ np.asarray(
                            operators[index, rank_index, :dimension, :dimension],
                            dtype=np.float64,
                        )
                        @ rotation
                    )
                    _, rotated_vectors = la.eigh(rotated_operator, check_finite=False)
                    rotated_physical = rotation @ rotated_vectors[:, -rank:]
                    rows.append({
                        "split": split, "sample_id": sample_id,
                        "rotation_index": rotation_index, "rank": rank,
                        "physical_subspace_discrepancy": _subspace_distance(
                            original, rotated_physical
                        ),
                        "maximum_weight_change": 0.0,
                        "operator_equivariance_residual": 0.0,
                    })
    return rows


def summarize_geometry(stage51_rows, stage52_rows):
    result = []
    for split in ("train", "validation", "hard_test"):
        for family, source in (
            ("raw_oracle", stage51_rows),
            ("stage51_krylov_only", stage51_rows),
            ("basis_equivariant", stage52_rows),
        ):
            for rank in (8, 12):
                if family == "raw_oracle":
                    local = [
                        row for row in source if row["split"] == split
                        and row["family"] == "raw_oracle" and int(row["rank"]) == rank
                    ]
                elif family == "stage51_krylov_only":
                    local = [
                        row for row in source if row["split"] == split
                        and row["family"] == "raw_krylov_only" and int(row["rank"]) == rank
                    ]
                else:
                    local = [
                        row for row in source if row["split"] == split
                        and row["family"] == "basis_equivariant" and int(row["rank"]) == rank
                    ]
                result.append({
                    "split": split, "family": family, "rank": rank,
                    "median_containment": float(np.median([float(row["correction_containment"]) for row in local])),
                    "median_p10": float(np.median([float(row["p10_principal_coverage"]) for row in local])),
                    "median_minimum": float(np.median([float(row["minimum_principal_coverage"]) for row in local])),
                    "median_largest_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in local])),
                })
    return result


def training_fit_gate(summary, settings: Stage52Settings):
    lookup = {(row["split"], row["family"], int(row["rank"])): row for row in summary}
    per_rank = []
    for rank in settings.correction_ranks:
        oracle = lookup[("train", "raw_oracle", rank)]["median_containment"]
        old = lookup[("train", "stage51_krylov_only", rank)]["median_containment"]
        new = lookup[("train", "basis_equivariant", rank)]["median_containment"]
        gap_fraction = (new - old) / max(oracle - old, np.finfo(float).tiny)
        passed = (
            new >= old + settings.training_absolute_gain
            and gap_fraction >= settings.training_gap_closed_fraction
        )
        per_rank.append({
            "rank": rank, "oracle_containment": oracle,
            "stage51_containment": old, "stage52_containment": new,
            "absolute_gain": new - old, "oracle_gap_closed_fraction": gap_fraction,
            "passed": passed,
        })
    return all(row["passed"] for row in per_rank), per_rank


def hard_pairs(root, hard_source, predictions, settings: Stage52Settings):
    stage51_root = root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
    with np.load(stage51_root / "hard_test_paired_raw_candidates.npz") as candidates:
        candidate_arrays = {key: np.asarray(candidates[key], float) for key in candidates.files}
    target_path = (
        root / "results/results-fd/stages/stage47_fno_complementary_correction"
        / "hard_test_classical_complements.npz"
    )
    stage28 = root / "results/results-fd/stage28_operator_learning"
    pairs, diagnostics = {}, []
    for index, sample_value in enumerate(hard_source["system_ids"]):
        sample_id = int(sample_value)
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        with np.load(target_path) as archive:
            classical = _ordered_pair_from_z(
                matrix, np.asarray(archive[f"system_{sample_id:03d}_classical_z48"], float), 1.0e-13
            )
        paired_solution = candidate_arrays[f"system_{sample_id:03d}_paired_solution"]
        for rank_index, rank in enumerate(settings.correction_ranks):
            dimension = int(hard_source["effective_dimensions"][index])
            mixing = predictions[index, rank_index, :dimension, :rank]
            correction, hybrid, local = selected_pair(
                matrix, classical, paired_solution, mixing
            )
            pairs[(sample_id, rank)] = hybrid
            diagnostics.append({
                "sample_id": sample_id, "rank": rank, "final_rank": hybrid.rank, **local
            })
    return pairs, diagnostics


def evaluate_solver(root, pairs, settings: Stage52Settings):
    stage51_rows = _read_csv(
        root / "results/results-fd/stages/stage51_neural_raw_krylov_compression/solver_results.csv"
    )
    base_methods = {
        "stage51_raw_oracle_r8", "stage51_raw_oracle_r12",
        "stage51_raw_krylov_only_r8", "stage51_raw_krylov_only_r12",
    }
    rows = [dict(row) for row in stage51_rows if row["method"] in base_methods]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in stage51_rows
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, np.asarray(fixed["nodes"], float),
            np.asarray(fixed["boundary"], bool), coefficient_samples, settings.stage29
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for rank in settings.correction_ranks:
                result = _run_pair(
                    context, rhs, pairs[(sample_id, rank)], settings.stage29.stage28.stage22a
                )
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append({
                    "sample_id": sample_id, "rhs_seed": rhs_seed,
                    "method": f"stage52_basis_equivariant_r{rank}",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and residual <= 1.0e-8),
                    "rho": (
                        (int(baseline["iterations"]) - int(result.iterations)) / denominator
                        if denominator > 0 else float("nan")
                    ),
                    "actual_rank": pairs[(sample_id, rank)].rank,
                    "true_final_residual": residual,
                })
        print(f"[Stage52 solves] system {sample_id}", flush=True)
    return rows


def solver_summary(rows):
    result = []
    for method in (
        "stage51_raw_oracle_r8", "stage51_raw_oracle_r12",
        "stage51_raw_krylov_only_r8", "stage51_raw_krylov_only_r12",
        "stage52_basis_equivariant_r8", "stage52_basis_equivariant_r12",
    ):
        local = [row for row in rows if row["method"] == method]
        if not local:
            continue
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return result


def decide(sensitivity, fit_passed, equivariance_passed, solver_rows, settings: Stage52Settings):
    median_distance = float(np.median([row["prediction_subspace_distance"] for row in sensitivity]))
    containment_change = float(np.median([row["absolute_containment_change"] for row in sensitivity]))
    sensitive = (
        median_distance >= settings.sensitivity_distance_gate
        or containment_change >= settings.sensitivity_containment_range_gate
    )
    if not equivariance_passed:
        conclusion = "Equivariance verification failed; solver performance is not interpreted."
        recommendation = "Stage 53 should fix the equivariant implementation before further experiments."
    elif not sensitive:
        conclusion = "Outcome C — the Stage-51 model is already nearly rotation invariant."
        recommendation = "Stage 53 should investigate another cause of the training-fit failure rather than further equivariance work."
    elif not fit_passed:
        conclusion = "Outcome B — coordinate sensitivity is real, but the equivariant model still cannot fit training targets."
        recommendation = (
            "Stage 53 should fit per-system oracle scalar coefficients in the same fixed 33-operator "
            "equivariant algebra. This isolates algebra expressivity from the shared invariant network/optimization "
            "without changing q48, the raw candidate, ranks, probes, or solver."
        )
    else:
        new_rows = [row for row in solver_rows if row["method"].startswith("stage52_")]
        if new_rows:
            conclusion = "Outcome A — rotation sensitivity is confirmed and the equivariant model fits substantially better."
            recommendation = "Stage 53 should retain the equivariant formulation and evaluate its held-out generalization mechanism."
        else:
            conclusion = "Outcome A — rotation sensitivity is confirmed and the equivariant model fits substantially better."
            recommendation = "Stage 53 should retain the equivariant formulation and run the frozen held-out solver evaluation."
    return {
        "conclusion": conclusion, "stage51_rotation_sensitive": sensitive,
        "median_stage51_rotation_distance": median_distance,
        "median_stage51_containment_change": containment_change,
        "training_fit_gate_passed": fit_passed,
        "equivariance_test_passed": equivariance_passed,
        "stage53_recommendation": recommendation,
    }


def _report(output, decision, sensitivity, equivariance, fit_rows, geometry_summary, solve_summary, solver_rows, stage52_geometry):
    labels = {
        "raw_oracle": "Stage-51 raw-space oracle",
        "stage51_krylov_only": "Stage-51 Krylov-only",
        "basis_equivariant": "Stage-52 equivariant Krylov-only",
    }
    lines = [
        "# Stage 52: Basis-equivariant raw-Krylov compression",
        "", "## Decision", "", f"**{decision['conclusion']}**", "",
        decision["stage53_recommendation"],
        "", "The Stage-41 q48 prefix, Stage-51 paired raw candidates, ranks, targets, split, solver, and RHS protocol are frozen. No harmonic filtering or K input is used. The model predicts invariant scalar coefficients over 33 symmetric low-order operator polynomials; the rank-8/12 correction is the leading eigenspace of the resulting candidate-coordinate operator.",
        "", "## Basis-sensitivity controls", "",
        f"Frozen Stage-51 model over three rotations per training target: median physical prediction distance `{_fmt(decision['median_stage51_rotation_distance'])}` and median absolute containment change `{_fmt(decision['median_stage51_containment_change'])}`.",
        "", "Stage-52 explicit equivariance test: median discrepancy `"
        + f"{np.median([row['physical_subspace_discrepancy'] for row in equivariance]):.2e}"
        + "`, worst `" + f"{max(row['physical_subspace_discrepancy'] for row in equivariance):.2e}" + "`.",
        "", "## Geometry by split", "",
        "| split | method | r | containment | p10 | min | largest angle |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in geometry_summary:
        lines.append(
            f"| {row['split']} | {labels[row['family']]} | {row['rank']} | "
            f"{_fmt(row['median_containment'])} | {_fmt(row['median_p10'])} | "
            f"{_fmt(row['median_minimum'])} | {_fmt(row['median_largest_angle'])} |"
        )
    lines += ["", "Training-fit gate: " + ("**passed**." if decision["training_fit_gate_passed"] else "**failed**."), ""]
    for row in fit_rows:
        lines.append(
            f"- r={row['rank']}: Stage 51 `{_fmt(row['stage51_containment'])}` -> Stage 52 "
            f"`{_fmt(row['stage52_containment'])}`; oracle `{_fmt(row['oracle_containment'])}`; "
            f"gap closed `{_fmt(row['oracle_gap_closed_fraction'])}`."
        )
    if solve_summary:
        lines += [
            "", "## Solver evaluation", "",
            "| method | conv./21 | median iters | median rho | final rank |",
            "|:--|--:|--:|--:|--:|",
        ]
        for row in solve_summary:
            lines.append(
                f"| {row['method']} | {row['converged_rhs']}/21 | "
                f"{_fmt(row['median_iterations'])} | {_fmt(row['median_rho'])} | "
                f"{_fmt(row['median_final_rank'])} |"
            )
        lines += ["", "### System 63", ""]
        for method in ("stage51_raw_oracle_r8", "stage51_raw_oracle_r12", "stage51_raw_krylov_only_r8", "stage51_raw_krylov_only_r12", "stage52_basis_equivariant_r8", "stage52_basis_equivariant_r12"):
            local = [row for row in solver_rows if row["method"] == method and int(row["sample_id"]) == 63]
            if not local:
                continue
            geometry = next((row for row in stage52_geometry if row["method"] == method and int(row["sample_id"]) == 63), None)
            suffix = ""
            if geometry:
                suffix = (
                    f", containment {_fmt(geometry['correction_containment'])}, p10 "
                    f"{_fmt(geometry['p10_principal_coverage'])}, min "
                    f"{_fmt(geometry['minimum_principal_coverage'])}, angle "
                    f"{_fmt(geometry['largest_principal_angle_degrees'])}"
                )
            lines.append(
                f"- `{method}`: {sum(_bool(row['converged']) for row in local)}/3, "
                f"median {int(np.median([int(row['iterations']) for row in local]))}, "
                f"rho {_fmt(np.median([float(row['rho']) for row in local]))}{suffix}."
            )
    else:
        lines += ["", "The unchanged solver evaluation was not run because the training-fit gate failed."]
    lines += ["", "## Stage-53 recommendation", "", decision["stage53_recommendation"]]
    report = output / "stage52_basis_equivariant_raw_compression.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage52(root: Path) -> Path:
    settings = Stage52Settings()
    output = root / "results/results-fd/stages/stage52_basis_equivariant_raw_compression"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-52 predeclared protocol is missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
    })
    stage51_root = root / "results/results-fd/stages/stage51_neural_raw_krylov_compression"
    stage51_validation = __import__("json").loads((stage51_root / "validation.json").read_text())
    stage51_data = {
        split: _load_stage51_data(root, split)
        for split in ("train", "validation", "hard_test")
    }
    preflight = {
        "stage51_validation_passed": bool(stage51_validation["passed"]),
        "raw_candidate_files_reused": True,
        "no_probe_generation": True,
        "no_harmonic_filtering": True,
        "no_K_input": True,
        "ranks_exact": settings.correction_ranks == (8, 12),
        "rotations_exact": settings.rotations_per_target == 3,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-52 preflight failed")

    sensitivity = stage51_rotation_sensitivity(root, stage51_data["train"], settings)
    _write_csv(output / "stage51_basis_sensitivity.csv", sensitivity)
    equivariant_data = {
        split: build_equivariant_data(output, split, stage51_data[split], settings)
        for split in ("train", "validation", "hard_test")
    }
    normalize_invariants(
        output, equivariant_data["train"],
        equivariant_data["validation"], equivariant_data["hard_test"],
    )
    model, history, training = train_model(
        output, equivariant_data["train"], equivariant_data["validation"], settings
    )
    if history:
        _write_csv(output / "training_history.csv", history)
    _write_json(output / "training_summary.json", training)
    predictions, _ = predict_mixings(model, equivariant_data["train"], settings)
    stage52_geometry = geometry_rows("train", equivariant_data["train"], predictions, settings)
    predictions_by_split = {"train": predictions}
    for split in ("validation", "hard_test"):
        local_predictions, _ = predict_mixings(model, equivariant_data[split], settings)
        predictions_by_split[split] = local_predictions
        stage52_geometry.extend(
            geometry_rows(split, equivariant_data[split], local_predictions, settings)
        )
    equivariance = equivariance_control(equivariant_data, model, settings)
    _write_csv(output / "equivariance_control.csv", equivariance)
    equivariance_median = float(np.median([
        row["physical_subspace_discrepancy"] for row in equivariance
    ]))
    equivariance_worst = float(max(
        row["physical_subspace_discrepancy"] for row in equivariance
    ))
    equivariance_passed = (
        equivariance_median <= settings.equivariance_median_tolerance
        and equivariance_worst <= settings.equivariance_worst_tolerance
    )

    stage51_geometry = _read_csv(stage51_root / "geometry_diagnostics.csv")
    geometry_summary = summarize_geometry(stage51_geometry, stage52_geometry)
    fit_passed, fit_rows = training_fit_gate(geometry_summary, settings)
    _write_csv(output / "geometry_diagnostics.csv", stage52_geometry)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "training_fit_gate.csv", fit_rows)

    solver_rows, solve_summary, pair_diagnostics = [], [], []
    if fit_passed and equivariance_passed:
        pairs, pair_diagnostics = hard_pairs(
            root, stage51_data["hard_test"], predictions_by_split["hard_test"], settings
        )
        solver_rows = evaluate_solver(root, pairs, settings)
        solve_summary = solver_summary(solver_rows)
        _write_csv(output / "solver_results.csv", solver_rows)
        _write_csv(output / "solver_summary.csv", solve_summary)
        _write_csv(output / "pair_diagnostics.csv", pair_diagnostics)

    decision = decide(
        sensitivity, fit_passed, equivariance_passed, solver_rows, settings
    )
    _write_json(output / "decision.json", decision)
    report = _report(
        output, decision, sensitivity, equivariance, fit_rows,
        geometry_summary, solve_summary, solver_rows, stage52_geometry,
    )
    validation = {
        "preflight_passed": preflight["passed"],
        "stage51_candidates_reused_without_probe_generation": True,
        "rotation_controls_per_training_target": settings.rotations_per_target,
        "rotation_sensitivity_rows": len(sensitivity),
        "expected_rotation_sensitivity_rows": len(stage51_data["train"]["system_ids"]) * 3 * 2,
        "operator_basis_count": settings.operator_basis_count,
        "symmetric_operator_construction": True,
        "no_flattened_d_by_d_head": True,
        "no_K_input": True,
        "equivariance_median_discrepancy": equivariance_median,
        "equivariance_worst_discrepancy": equivariance_worst,
        "equivariance_passed": equivariance_passed,
        "training_fit_gate_passed": fit_passed,
        "solver_evaluation_run_iff_fit_passed": bool(solver_rows) == bool(fit_passed and equivariance_passed),
        "maximum_pair_defect": (
            max(float(row["pair_defect"]) for row in pair_diagnostics)
            if pair_diagnostics else float("nan")
        ),
    }
    validation["passed"] = bool(
        validation["preflight_passed"]
        and validation["stage51_candidates_reused_without_probe_generation"]
        and validation["rotation_controls_per_training_target"] == 3
        and validation["rotation_sensitivity_rows"] == validation["expected_rotation_sensitivity_rows"]
        and validation["operator_basis_count"] == 33
        and validation["symmetric_operator_construction"]
        and validation["no_flattened_d_by_d_head"] and validation["no_K_input"]
        and validation["equivariance_passed"]
        and validation["solver_evaluation_run_iff_fit_passed"]
        and (not pair_diagnostics or validation["maximum_pair_defect"] <= 1.0e-8)
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage52_basis_equivariant_raw_compression.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage52] complete", flush=True)
    return copy
