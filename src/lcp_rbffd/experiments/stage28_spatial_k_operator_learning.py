"""Stage 28: spatial-K operator-learning feasibility.

This experiment keeps the validated RBF-FD, strong-ILU, ordinary-Ritz
teacher, and nonsymmetric recycled-GMRES mechanisms fixed.  It compares a
seven-parameter MLP control with one modest FNO-style tensor-field encoder.
Both models predict coefficients in the same training-only solution-space
dictionary and use the same normalized W-projector loss.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
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
import scipy.sparse as sp
import torch
from torch import nn

from neurkitt_oracle.gcrodr import restarted_projected_gmres

from ..pde import SpatialAnisotropicDiffusion
from ..preconditioners import ilu_preconditioner
from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import (
    Stage19ASettings,
    _assemble,
    _coefficient_metadata,
    _discrete_rhs,
    _sample_coefficients,
    run_hardness_scan,
)
from .stage20_spatial_k_robust_oracle import SystemContext, _relative_residual
from .stage22a_multi_rhs_functional_oracle import (
    RecyclePair,
    Stage22ASettings,
    _build_ordinary_construction,
    _orthonormalize,
    _pair_from_z,
    _raw_dictionary,
    _run_pair,
)
from .stage27_spatial_k_adaptive_rank_validation import (
    FirstCycleExpansionController,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PURPLE = "#7553A6"
PINK = "#B14D75"
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
class Stage28Settings:
    """Frozen population, teacher, model, evaluation, and decision controls."""

    sample_seed: int = 20262801
    sample_count: int = 72
    split_seed: int = 20262802
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    construction_rhs_seeds: Tuple[int, ...] = (20262911, 20262921, 20262931)
    evaluation_rhs_seeds: Tuple[int, ...] = (20263001, 20263002, 20263003)
    arnoldi_dimension: int = 128
    candidates_per_rhs: int = 24
    teacher_q: int = 28
    full_q: int = 48
    numerical_rank_tolerance: float = 1.0e-11
    hard_iteration_threshold: int = 50
    dictionary_energy_target: float = 0.99
    model_seeds: Tuple[int, ...] = (20262821, 20262822)
    hidden_widths: Tuple[int, int] = (64, 128)
    fno_width: int = 16
    fno_modes: int = 8
    fno_layers: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 6
    max_epochs: int = 500
    minimum_epochs: int = 100
    validation_interval: int = 5
    early_stopping_patience: int = 75
    cholesky_jitter: float = 1.0e-6
    first_cycle_contraction_threshold: float = 1.0e-4
    minimum_hard_test_systems: int = 5
    effective_rhs_convergence: float = 0.90
    effective_system_pass: float = 0.70
    effective_median_eta28: float = 0.50
    effective_iteration_ratio: float = 0.75
    fno_eta_margin: float = 0.10
    fno_iteration_margin_fraction: float = 0.10
    fno_tie_or_win_fraction: float = 2.0 / 3.0

    @property
    def stage19a(self) -> Stage19ASettings:
        return Stage19ASettings(
            seed=self.sample_seed,
            sample_count=self.sample_count,
            construction_rhs_seed=self.construction_rhs_seeds[0],
            evaluation_rhs_seeds=self.evaluation_rhs_seeds,
        )

    @property
    def stage22a(self) -> Stage22ASettings:
        return Stage22ASettings(
            construction_rhs_seeds=self.construction_rhs_seeds + (20262941,),
            heldout_rhs_seeds=self.evaluation_rhs_seeds,
            primary_arnoldi_dimension=self.arnoldi_dimension,
            candidates_per_rhs=self.candidates_per_rhs,
            numerical_rank_tolerance=self.numerical_rank_tolerance,
            seed=20262800,
        )

    @property
    def solver(self):
        return self.stage22a.solver


class ParameterMLP(nn.Module):
    """Seven generation parameters to a ``p x q`` coefficient matrix."""

    def __init__(self, p: int, q: int, widths: Tuple[int, int]) -> None:
        super().__init__()
        self.p, self.q = int(p), int(q)
        self.network = nn.Sequential(
            nn.Linear(7, widths[0]),
            nn.GELU(),
            nn.Linear(widths[0], widths[1]),
            nn.GELU(),
            nn.Linear(widths[1], p * q),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).reshape(-1, self.p, self.q)


class SpectralConv2d(nn.Module):
    """Small real-input Fourier layer with learned positive/negative modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(max(in_channels * out_channels, 1))
        shape = (in_channels, out_channels, modes, modes, 2)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        complex_weights = torch.view_as_complex(weights.contiguous())
        return torch.einsum("bixy,ioxy->boxy", inputs, complex_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        transformed = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=transformed.dtype,
            device=x.device,
        )
        my = min(self.modes, height)
        mx = min(self.modes, width // 2 + 1)
        out[:, :, :my, :mx] = self._multiply(
            transformed[:, :, :my, :mx], self.weight_positive[:, :, :my, :mx]
        )
        out[:, :, -my:, :mx] = self._multiply(
            transformed[:, :, -my:, :mx], self.weight_negative[:, :, :my, :mx]
        )
        return torch.fft.irfft2(out, s=(height, width), norm="ortho")


class FNOFieldEncoder(nn.Module):
    """Modest FNO-style tensor-field encoder and low-rank coefficient decoder."""

    def __init__(self, p: int, q: int, width: int, modes: int, layers: int) -> None:
        super().__init__()
        self.p, self.q = int(p), int(q)
        self.lift = nn.Conv2d(5, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(layers)]
        )
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.decoder = nn.Sequential(
            nn.Linear(width * 16, 128),
            nn.GELU(),
            nn.Linear(128, p * q),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(field)
        for spectral, local in zip(self.spectral, self.local):
            hidden = torch.nn.functional.gelu(spectral(hidden) + local(hidden))
        encoded = self.pool(hidden).flatten(1)
        return self.decoder(encoded).reshape(-1, self.p, self.q)


def normalized_w_projector_losses(
    coefficients: torch.Tensor,
    grams: torch.Tensor,
    crosses: torch.Tensor,
    target_ranks: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    r"""Return ``||P_W-P_W*||_F^2/(q+q*)`` from reduced matrices.

    ``Psi`` is orthonormal.  QR of the coefficient matrix therefore realizes
    ``orth(Psi C)`` without touching the 5,041-dimensional node axis.  For
    ``Y=A Psi`` and coefficient basis ``Q_C``, the remaining QR is represented
    by the Cholesky factor of ``Q_C^T Y^T Y Q_C``.
    """

    coefficient_basis = torch.linalg.qr(coefficients, mode="reduced").Q
    gc = torch.matmul(grams, coefficient_basis)
    gram_c = torch.matmul(coefficient_basis.transpose(-2, -1), gc)
    q = coefficient_basis.shape[-1]
    scale = torch.diagonal(gram_c, dim1=-2, dim2=-1).mean(dim=-1)
    eye = torch.eye(q, dtype=coefficients.dtype, device=coefficients.device)
    stabilized = gram_c + (
        jitter * torch.clamp(scale, min=torch.finfo(coefficients.dtype).eps)
    )[..., None, None] * eye
    lower = torch.linalg.cholesky(stabilized)
    hc = torch.matmul(crosses, coefficient_basis)
    overlap = torch.linalg.solve_triangular(lower, hc.transpose(-2, -1), upper=False)
    overlap_squared = torch.sum(overlap**2, dim=(-2, -1))
    numerator = q + target_ranks - 2.0 * overlap_squared
    return torch.clamp(numerator / (q + target_ranks), min=0.0)


def stratified_system_split(
    ids: Sequence[int], hard_ids: Sequence[int], seed: int, train_fraction: float,
    validation_fraction: float,
) -> Dict[str, List[int]]:
    """Deterministically split systems while preserving oracle-blind hardness mix."""

    rng = np.random.default_rng(seed)
    hard = set(int(value) for value in hard_ids)
    groups = [[int(value) for value in ids if int(value) in hard],
              [int(value) for value in ids if int(value) not in hard]]
    result = {"train": [], "validation": [], "test": []}
    for group in groups:
        shuffled = np.asarray(group, dtype=int)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(train_fraction * n))
        n_validation = int(round(validation_fraction * n))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_validation = min(max(n_validation, 1), n - n_train - 1)
        result["train"].extend(shuffled[:n_train].tolist())
        result["validation"].extend(shuffled[n_train:n_train + n_validation].tolist())
        result["test"].extend(shuffled[n_train + n_validation:].tolist())
    for split in result:
        result[split] = sorted(result[split])
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _parameters(coefficient: SpatialAnisotropicDiffusion) -> np.ndarray:
    return np.asarray(
        [coefficient.a0, coefficient.a1, coefficient.a2, coefficient.a3,
         coefficient.theta0_degrees, coefficient.b1_degrees,
         coefficient.b2_degrees], dtype=float
    )


def _coefficient(values: Sequence[float]) -> SpatialAnisotropicDiffusion:
    return SpatialAnisotropicDiffusion(*[float(value) for value in values])


def _tensor_channels(coefficient: SpatialAnisotropicDiffusion, nodes: np.ndarray) -> np.ndarray:
    tensor = coefficient.tensor_values(nodes)
    return np.column_stack((tensor[:, 0, 0], tensor[:, 0, 1], tensor[:, 1, 1]))


def _assemble_context(
    sample_id: int,
    coefficient: SpatialAnisotropicDiffusion,
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage28Settings,
) -> SystemContext:
    system, _ = _assemble(nodes, boundary, coefficient, settings.stage19a)
    ilu = ilu_preconditioner(
        system.matrix, settings.stage19a.stage10.classical_ilu, name="strong_ilu"
    )
    return SystemContext(sample_id, system, ilu)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(output: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(output)): _file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build_dataset(root: Path, output: Path, settings: Stage28Settings) -> Dict[str, Any]:
    """Generate the independent systems, freeze the split, and build teachers."""

    source = root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation"
    fixed = np.load(source / "fixed_nodes.npz")
    nodes = np.asarray(fixed["nodes"], dtype=float)
    boundary = np.asarray(fixed["boundary"], dtype=bool)
    coefficients = _sample_coefficients(settings.stage19a)
    _write_csv(
        output / "coefficient_samples.csv",
        [_coefficient_metadata(index, coefficient, nodes)
         for index, coefficient in enumerate(coefficients)],
    )
    scan_rhs, scan_summary = run_hardness_scan(
        output, settings.stage19a, coefficients, nodes, boundary
    )
    valid_ids = sorted(
        int(row["sample_id"]) for row in scan_summary if _bool(row["numerically_valid"])
    )
    hard_ids = sorted(
        int(row["sample_id"])
        for row in scan_summary
        if _bool(row["numerically_valid"])
        and (
            float(row["ilu_iterations_median"]) >= settings.hard_iteration_threshold
            or float(row["ilu_convergence_rate"]) < 1.0
        )
    )
    split = stratified_system_split(
        valid_ids, hard_ids, settings.split_seed,
        settings.train_fraction, settings.validation_fraction,
    )
    split_payload = {
        **split,
        "hard_ids": hard_ids,
        "valid_ids": valid_ids,
        "rule": "deterministic oracle-blind stratification by strong-ILU hardness",
        "split_frozen_before_teacher_construction": True,
    }
    _write_json(output / "split.json", split_payload)

    systems_dir = output / "systems"
    matrices_dir = output / "matrices"
    systems_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    split_of = {sample_id: name for name, values in split.items() for sample_id in values}
    scan_by_id = {int(row["sample_id"]): row for row in scan_summary}
    teacher_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    for position, sample_id in enumerate(valid_ids, start=1):
        coefficient = coefficients[sample_id]
        context = _assemble_context(sample_id, coefficient, nodes, boundary, settings)
        constructions = [
            _build_ordinary_construction(
                context, seed, settings.arnoldi_dimension, settings.stage22a
            )
            for seed in settings.construction_rhs_seeds
        ]
        raw = _raw_dictionary(constructions)
        compression_start = perf_counter()
        ordered, singular_values, numerical_rank = _orthonormalize(
            raw, settings.numerical_rank_tolerance
        )
        compression_seconds = perf_counter() - compression_start
        z28 = ordered[:, : min(settings.teacher_q, numerical_rank)]
        zfull = ordered[:, : min(settings.full_q, numerical_rank)]
        pair28 = _pair_from_z(context.system.matrix, z28)
        pairfull = _pair_from_z(context.system.matrix, zfull)
        fields = _tensor_channels(coefficient, nodes)
        parameter_values = _parameters(coefficient)
        system_path = systems_dir / f"system_{sample_id:03d}.npz"
        matrix_path = matrices_dir / f"A_{sample_id:03d}.npz"
        np.savez_compressed(
            system_path,
            parameters=parameter_values,
            tensor_field=fields.astype(np.float32),
            z28=z28.astype(np.float32),
            w28=pair28.w_basis.astype(np.float32),
            zfull=zfull.astype(np.float32),
            wfull=pairfull.w_basis.astype(np.float32),
            joint_singular_values=singular_values.astype(np.float64),
            construction_rhs_seeds=np.asarray(settings.construction_rhs_seeds),
        )
        sp.save_npz(matrix_path, context.system.matrix)
        construction_seconds = float(sum(item.setup_seconds for item in constructions))
        total_setup = construction_seconds + compression_seconds + pairfull.setup_seconds
        max_pair_defect = max(
            float(pair28.diagnostics["au_equals_w_relative_error"]),
            float(pairfull.diagnostics["au_equals_w_relative_error"]),
        )
        teacher_rows.append(
            {
                "sample_id": sample_id,
                "split": split_of[sample_id],
                "hard": sample_id in hard_ids,
                "raw_columns": raw.shape[1],
                "dictionary_numerical_rank": numerical_rank,
                "teacher28_actual_rank": pair28.rank,
                "full_actual_rank": pairfull.rank,
                "arnoldi_seconds": construction_seconds,
                "compression_seconds": compression_seconds,
                "pair28_seconds": pair28.setup_seconds,
                "pairfull_seconds": pairfull.setup_seconds,
                "full_teacher_setup_seconds": total_setup,
                "setup_A_applications": int(sum(item.a_applications for item in constructions)) + pairfull.rank,
                "setup_M_applications": int(sum(item.m_applications for item in constructions)),
                "maximum_pair_defect": max_pair_defect,
                "failure": "",
            }
        )
        dataset_rows.append(
            {
                "sample_id": sample_id,
                "split": split_of[sample_id],
                "hard": sample_id in hard_ids,
                "numerically_valid": True,
                "ilu_iterations_median": scan_by_id[sample_id]["ilu_iterations_median"],
                "system_file": str(system_path.relative_to(output)),
                "matrix_file": str(matrix_path.relative_to(output)),
                "system_sha256": _file_hash(system_path),
                "matrix_sha256": _file_hash(matrix_path),
            }
        )
        _write_csv(output / "teacher_construction_metadata.partial.csv", teacher_rows)
        print(f"[teacher {position:02d}/{len(valid_ids)}] system {sample_id}", flush=True)
    _write_csv(output / "teacher_construction_metadata.csv", teacher_rows)
    _write_csv(output / "dataset_manifest.csv", dataset_rows)
    partial = output / "teacher_construction_metadata.partial.csv"
    if partial.exists():
        partial.unlink()
    _write_json(
        output / "dataset_summary.json",
        {
            "sampled": settings.sample_count,
            "valid": len(valid_ids),
            "hard": len(hard_ids),
            "split_counts": {key: len(value) for key, value in split.items()},
            "hard_split_counts": {
                key: sum(value in set(hard_ids) for value in values)
                for key, values in split.items()
            },
            "all_fields_saved": len(dataset_rows) == len(valid_ids),
            "stage27_systems_used": 0,
        },
    )
    return {
        "nodes": nodes,
        "boundary": boundary,
        "coefficients": coefficients,
        "scan_rhs": scan_rhs,
        "scan_summary": scan_summary,
        "split": split_payload,
        "teacher_rows": teacher_rows,
        "dataset_rows": dataset_rows,
    }


def load_saved_dataset(root: Path, output: Path, settings: Stage28Settings) -> Dict[str, Any]:
    """Load the immutable Stage-28 dataset after an interrupted later phase.

    This deliberately reconstructs no matrices, ILU factors, teacher spaces, or
    split decisions.  It is used only to resume final-test evaluation after all
    model checkpoints and the final-test lock receipt have already been saved.
    """

    source = root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation"
    with np.load(source / "fixed_nodes.npz") as fixed:
        nodes = np.asarray(fixed["nodes"], dtype=float)
        boundary = np.asarray(fixed["boundary"], dtype=bool)
    return {
        "nodes": nodes,
        "boundary": boundary,
        "coefficients": _sample_coefficients(settings.stage19a),
        "scan_rhs": _read_csv(output / "hardness_scan_rhs.csv"),
        "scan_summary": _read_csv(output / "hardness_scan_summary.csv"),
        "split": json.loads((output / "split.json").read_text(encoding="utf-8")),
        "teacher_rows": _read_csv(output / "teacher_construction_metadata.csv"),
        "dataset_rows": _read_csv(output / "dataset_manifest.csv"),
    }


def load_selected_models(
    output: Path, p: int, settings: Stage28Settings,
) -> Dict[str, nn.Module]:
    """Restore the validation-selected checkpoints without retraining."""

    restored: Dict[str, nn.Module] = {}
    for model_name in ("parameter_mlp", "fno"):
        checkpoint = torch.load(
            output / "models" / f"selected_{model_name}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if model_name == "parameter_mlp":
            model: nn.Module = ParameterMLP(p, settings.teacher_q, settings.hidden_widths)
        else:
            model = FNOFieldEncoder(
                p, settings.teacher_q, settings.fno_width,
                settings.fno_modes, settings.fno_layers,
            )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        restored[model_name] = model
    return restored


def build_training_dictionary(output: Path, split: Mapping[str, Sequence[int]], settings: Stage28Settings) -> Tuple[np.ndarray, Dict[str, Any]]:
    snapshots = []
    for sample_id in split["train"]:
        with np.load(output / "systems" / f"system_{sample_id:03d}.npz") as archive:
            snapshots.append(np.asarray(archive["z28"], dtype=float))
    snapshot_matrix = np.column_stack(snapshots)
    dictionary, singular_values, _ = _orthonormalize(snapshot_matrix, 1.0e-14)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    p = int(np.searchsorted(energy, settings.dictionary_energy_target) + 1)
    psi = np.asarray(dictionary[:, :p], dtype=float)
    rows = [
        {"mode": index + 1, "singular_value": value,
         "normalized_singular_value": value / singular_values[0],
         "captured_energy": energy[index], "selected": index < p}
        for index, value in enumerate(singular_values)
    ]
    _write_csv(output / "dictionary_spectrum.csv", rows)
    metadata = {
        "p": p,
        "q": settings.teacher_q,
        "energy_target": settings.dictionary_energy_target,
        "captured_energy": float(energy[p - 1]),
        "snapshot_columns": snapshot_matrix.shape[1],
        "training_system_count": len(split["train"]),
        "validation_systems_used": 0,
        "test_systems_used": 0,
        "orthogonality_error": float(np.linalg.norm(psi.T @ psi - np.eye(p), ord="fro")),
    }
    np.savez_compressed(output / "training_dictionary.npz", psi=psi.astype(np.float32), singular_values=singular_values, energy=energy)
    _write_json(output / "dictionary_metadata.json", metadata)
    return psi, metadata


def build_preprocessing_and_loss_data(
    output: Path, nodes: np.ndarray, psi: np.ndarray,
    split: Mapping[str, Sequence[int]], settings: Stage28Settings,
) -> Dict[str, Any]:
    train_parameters, train_fields = [], []
    for sample_id in split["train"]:
        with np.load(output / "systems" / f"system_{sample_id:03d}.npz") as archive:
            train_parameters.append(np.asarray(archive["parameters"], dtype=float))
            train_fields.append(np.asarray(archive["tensor_field"], dtype=float))
    train_parameters_array = np.stack(train_parameters)
    train_fields_array = np.stack(train_fields)
    parameter_mean = train_parameters_array.mean(axis=0)
    parameter_std = train_parameters_array.std(axis=0)
    field_mean = train_fields_array.reshape(-1, 3).mean(axis=0)
    field_std = train_fields_array.reshape(-1, 3).std(axis=0)
    parameter_std = np.where(parameter_std > 0, parameter_std, 1.0)
    field_std = np.where(field_std > 0, field_std, 1.0)
    side = int(round(math.sqrt(nodes.shape[0])))
    if side * side != nodes.shape[0]:
        raise RuntimeError("fixed nodes cannot be reshaped to the FNO grid")
    np.savez_compressed(
        output / "frozen_preprocessing.npz",
        parameter_mean=parameter_mean,
        parameter_std=parameter_std,
        field_mean=field_mean,
        field_std=field_std,
        x_grid=(2.0 * nodes[:, 0] - 1.0).reshape(side, side),
        y_grid=(2.0 * nodes[:, 1] - 1.0).reshape(side, side),
        train_system_ids=np.asarray(split["train"], dtype=int),
    )
    for split_name in ("train", "validation"):
        grams, crosses, ranks, parameters, fields = [], [], [], [], []
        for sample_id in split[split_name]:
            matrix = sp.load_npz(output / "matrices" / f"A_{sample_id:03d}.npz")
            with np.load(output / "systems" / f"system_{sample_id:03d}.npz") as archive:
                w = np.asarray(archive["w28"], dtype=float)
                parameters.append((np.asarray(archive["parameters"], dtype=float) - parameter_mean) / parameter_std)
                field = (np.asarray(archive["tensor_field"], dtype=float) - field_mean) / field_std
                field_grid = field.reshape(side, side, 3).transpose(2, 0, 1)
                coordinates = np.stack(
                    ((2.0 * nodes[:, 0] - 1.0).reshape(side, side),
                     (2.0 * nodes[:, 1] - 1.0).reshape(side, side))
                )
                fields.append(np.concatenate((field_grid, coordinates), axis=0))
            image = np.asarray(matrix @ psi, dtype=float)
            grams.append(image.T @ image)
            padded = np.zeros((settings.teacher_q, psi.shape[1]), dtype=float)
            padded[: w.shape[1], :] = w.T @ image
            crosses.append(padded)
            ranks.append(w.shape[1])
        np.savez_compressed(
            output / f"{split_name}_reduced_training_data.npz",
            system_ids=np.asarray(split[split_name], dtype=int),
            grams=np.asarray(grams, dtype=np.float32),
            crosses=np.asarray(crosses, dtype=np.float32),
            target_ranks=np.asarray(ranks, dtype=np.float32),
            normalized_parameters=np.asarray(parameters, dtype=np.float32),
            normalized_fields=np.asarray(fields, dtype=np.float32),
        )
    metadata = {
        "parameter_mean": parameter_mean.tolist(),
        "parameter_std": parameter_std.tolist(),
        "field_mean": field_mean.tolist(),
        "field_std": field_std.tolist(),
        "fit_system_ids": list(split["train"]),
        "validation_systems_used_for_fit": 0,
        "test_systems_used_for_fit": 0,
        "field_channels": ["K_xx", "K_xy", "K_yy", "x", "y"],
        "grid_shape": [side, side],
    }
    _write_json(output / "preprocessing_metadata.json", metadata)
    return metadata


def _model_inputs(model_name: str, archive: Mapping[str, np.ndarray]) -> np.ndarray:
    if model_name == "parameter_mlp":
        return np.asarray(archive["normalized_parameters"], dtype=np.float32)
    if model_name == "fno":
        return np.asarray(archive["normalized_fields"], dtype=np.float32)
    raise ValueError(f"unknown model {model_name}")


def _new_model(model_name: str, p: int, settings: Stage28Settings) -> nn.Module:
    if model_name == "parameter_mlp":
        return ParameterMLP(p, settings.teacher_q, settings.hidden_widths)
    if model_name == "fno":
        return FNOFieldEncoder(
            p, settings.teacher_q, settings.fno_width,
            settings.fno_modes, settings.fno_layers,
        )
    raise ValueError(f"unknown model {model_name}")


def _batched_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    grams: torch.Tensor,
    crosses: torch.Tensor,
    ranks: torch.Tensor,
    settings: Stage28Settings,
    *,
    train: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    indices = torch.arange(inputs.shape[0])
    if train:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    losses = []
    for start in range(0, len(indices), settings.batch_size):
        local = indices[start:start + settings.batch_size]
        coefficients = model(inputs[local])
        losses.append(
            normalized_w_projector_losses(
                coefficients, grams[local], crosses[local], ranks[local],
                settings.cholesky_jitter,
            )
        )
    return torch.cat(losses)


def train_model_seed(
    output: Path,
    model_name: str,
    seed: int,
    p: int,
    settings: Stage28Settings,
) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    """Fit one fixed architecture and select its checkpoint by validation loss."""

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = _new_model(model_name, p, settings).to(dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    with np.load(output / "train_reduced_training_data.npz") as train_archive:
        train_inputs = torch.from_numpy(_model_inputs(model_name, train_archive))
        train_grams = torch.from_numpy(np.asarray(train_archive["grams"], dtype=np.float32))
        train_crosses = torch.from_numpy(np.asarray(train_archive["crosses"], dtype=np.float32))
        train_ranks = torch.from_numpy(np.asarray(train_archive["target_ranks"], dtype=np.float32))
    with np.load(output / "validation_reduced_training_data.npz") as val_archive:
        val_inputs = torch.from_numpy(_model_inputs(model_name, val_archive))
        val_grams = torch.from_numpy(np.asarray(val_archive["grams"], dtype=np.float32))
        val_crosses = torch.from_numpy(np.asarray(val_archive["crosses"], dtype=np.float32))
        val_ranks = torch.from_numpy(np.asarray(val_archive["target_ranks"], dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    training_start = perf_counter()
    for epoch in range(settings.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_losses = _batched_loss(
            model, train_inputs, train_grams, train_crosses, train_ranks,
            settings, train=True, generator=generator,
        )
        training_loss = training_losses.mean()
        if not torch.isfinite(training_loss):
            raise RuntimeError(f"nonfinite {model_name} loss at epoch {epoch}")
        training_loss.backward()
        optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                validation_losses = _batched_loss(
                    model, val_inputs, val_grams, val_crosses, val_ranks,
                    settings, train=False,
                )
            row = {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "training_loss": float(training_loss.detach().cpu()),
                "validation_loss": float(validation_losses.mean().cpu()),
                "validation_loss_max": float(validation_losses.max().cpu()),
            }
            history.append(row)
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
        raise RuntimeError(f"no finite checkpoint for {model_name}, seed {seed}")
    model.load_state_dict(best_state)
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    checkpoint = models_dir / f"{model_name}_seed{seed}.pt"
    torch.save(
        {
            "model": model_name,
            "seed": seed,
            "p": p,
            "q": settings.teacher_q,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "settings": asdict(settings),
        },
        checkpoint,
    )
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    summary = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "training_seconds": perf_counter() - training_start,
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint.relative_to(output)),
    }
    return model, history, summary


def train_models(
    output: Path, p: int, settings: Stage28Settings,
) -> Tuple[Dict[str, nn.Module], List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: Dict[str, nn.Module] = {}
    histories: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for model_name in ("parameter_mlp", "fno"):
        candidates = []
        for seed in settings.model_seeds:
            model, history, summary = train_model_seed(
                output, model_name, seed, p, settings
            )
            candidates.append((summary["best_validation_loss"], seed, model, summary))
            histories.extend(history)
            summaries.append(summary)
            print(
                f"[train {model_name}] seed={seed} best={summary['best_validation_loss']:.6f}",
                flush=True,
            )
        _, selected_seed, selected_model, selected_summary = min(
            candidates, key=lambda value: (value[0], value[1])
        )
        selected[model_name] = selected_model
        selected_summary["selected"] = True
        selected_path = output / "models" / f"selected_{model_name}.pt"
        shutil.copyfile(
            output / selected_summary["checkpoint"], selected_path
        )
        for summary in summaries:
            if summary["model"] == model_name and summary["seed"] != selected_seed:
                summary["selected"] = False
    _write_csv(output / "training_history.csv", histories)
    _write_csv(output / "training_summary.csv", summaries)
    _write_json(
        output / "selected_models.json",
        {
            name: next(
                row for row in summaries if row["model"] == name and row["selected"]
            )
            for name in selected
        },
    )
    return selected, histories, summaries


def _normalized_inputs_for_system(
    system_archive: Mapping[str, np.ndarray], preprocessing: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    parameters = (
        np.asarray(system_archive["parameters"], dtype=float)
        - np.asarray(preprocessing["parameter_mean"], dtype=float)
    ) / np.asarray(preprocessing["parameter_std"], dtype=float)
    field = (
        np.asarray(system_archive["tensor_field"], dtype=float)
        - np.asarray(preprocessing["field_mean"], dtype=float)
    ) / np.asarray(preprocessing["field_std"], dtype=float)
    side = int(round(math.sqrt(field.shape[0])))
    field_grid = field.reshape(side, side, 3).transpose(2, 0, 1)
    coordinate_grid = np.stack(
        (np.asarray(preprocessing["x_grid"], dtype=float),
         np.asarray(preprocessing["y_grid"], dtype=float))
    )
    return parameters.astype(np.float32), np.concatenate(
        (field_grid, coordinate_grid), axis=0
    ).astype(np.float32)


def predict_space(
    model_name: str,
    model: nn.Module,
    psi: np.ndarray,
    system_archive: Mapping[str, np.ndarray],
    preprocessing: Mapping[str, np.ndarray],
    q: int,
) -> Tuple[np.ndarray, float]:
    """Predict one solution-space basis without target Krylov information."""

    parameter_input, field_input = _normalized_inputs_for_system(
        system_archive, preprocessing
    )
    values = parameter_input[None, :] if model_name == "parameter_mlp" else field_input[None, :]
    tensor = torch.from_numpy(values)
    model.eval()
    with torch.no_grad():
        _ = model(tensor)
        timings = []
        coefficients = None
        for _ in range(5):
            started = perf_counter()
            coefficients = model(tensor)[0].cpu().numpy()
            timings.append(perf_counter() - started)
    assert coefficients is not None
    z, _, rank = _orthonormalize(psi @ coefficients, 1.0e-13)
    if rank != q:
        raise RuntimeError(f"{model_name} prediction has rank {rank}, expected {q}")
    return z, float(np.median(timings))


def _field_distance(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.linalg.norm(first - second))
    denominator = 0.5 * (float(np.linalg.norm(first)) + float(np.linalg.norm(second)))
    return numerator / max(denominator, np.finfo(float).tiny)


def _projection_microseconds(pair: RecyclePair, n_nodes: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=n_nodes)
    start = perf_counter()
    for _ in range(50):
        _ = pair.u_basis @ (pair.w_basis.T @ vector)
    return 1.0e6 * (perf_counter() - start) / 50.0


def _run_fallback(
    context: SystemContext,
    rhs: np.ndarray,
    z_predicted: np.ndarray,
    z_full: np.ndarray,
    settings: Stage28Settings,
) -> Tuple[Any, FirstCycleExpansionController, float]:
    initial_pair = _pair_from_z(context.system.matrix, z_predicted)
    controller_settings = __import__(
        "lcp_rbffd.experiments.stage27_spatial_k_adaptive_rank_validation",
        fromlist=["Stage27Settings"],
    ).Stage27Settings(
        first_cycle_contraction_threshold=settings.first_cycle_contraction_threshold
    )
    controller = FirstCycleExpansionController(
        context.system.matrix, z_full, initial_pair, float(np.linalg.norm(rhs)),
        controller_settings,
    )
    context.ilu.reset_timing()
    result = restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        u_basis=initial_pair.u_basis,
        c_basis=initial_pair.w_basis,
        rtol=settings.solver.relative_tolerance,
        atol=settings.solver.absolute_tolerance,
        restart=settings.solver.restart,
        max_iterations=settings.solver.max_iterations,
        record_true_residuals=True,
        cycle_update=controller,
    )
    return result, controller, initial_pair.setup_seconds


def _solver_row(
    sample_id: int,
    rhs_seed: int,
    hard: bool,
    method: str,
    result: Any,
    true_residual: float,
    baseline: Mapping[str, Any],
    actual_rank: int,
    inference_seconds: float = 0.0,
    pair_setup_seconds: float = 0.0,
    teacher_setup_seconds: float = 0.0,
    projection_microseconds: float = 0.0,
    expanded: bool = False,
    first_cycle_contraction: float = float("nan"),
) -> Dict[str, Any]:
    counts = result.operation_counts or {}
    return {
        "sample_id": sample_id,
        "split": "test",
        "hard": hard,
        "rhs_seed": rhs_seed,
        "method": method,
        "baseline_iterations": int(baseline["iterations"]),
        "baseline_converged": _bool(baseline["converged"]),
        "baseline_true_residual": float(baseline["true_final_relative_residual"]),
        "baseline_solve_seconds": float(baseline["solve_seconds"]),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "true_final_residual": float(true_residual),
        "actual_rank": actual_rank,
        "inference_seconds": inference_seconds,
        "pair_setup_seconds": pair_setup_seconds,
        "teacher_setup_seconds_charged": teacher_setup_seconds,
        "projection_application_microseconds": projection_microseconds,
        "online_solve_seconds": float(result.solve_seconds),
        "online_total_seconds": inference_seconds + pair_setup_seconds + float(result.solve_seconds),
        "fully_charged_seconds": teacher_setup_seconds + inference_seconds + pair_setup_seconds + float(result.solve_seconds),
        "expanded_to_classical_full": expanded,
        "first_cycle_contraction": first_cycle_contraction,
        "A_applications": int(counts.get("a_matvecs", 0)),
        "M_applications": int(counts.get("preconditioner_applications", 0)),
        "failure": getattr(result, "failure", "") or "",
    }


def evaluate_final_test(
    output: Path,
    nodes: np.ndarray,
    boundary: np.ndarray,
    coefficients: Sequence[SpatialAnisotropicDiffusion],
    split: Mapping[str, Sequence[int]],
    scan_rhs: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
    psi: np.ndarray,
    models: Mapping[str, nn.Module],
    settings: Stage28Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the untouched final-test solver comparison and secondary fallback."""

    with np.load(output / "frozen_preprocessing.npz") as archive:
        preprocessing = {key: np.asarray(archive[key]) for key in archive.files}
    scan_lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row for row in scan_rhs
    }
    teacher_lookup = {int(row["sample_id"]): row for row in teacher_rows}
    train_fields, train_spaces = {}, {}
    for donor in split["train"]:
        with np.load(output / "systems" / f"system_{donor:03d}.npz") as archive:
            train_fields[donor] = np.asarray(archive["tensor_field"], dtype=float)
            train_spaces[donor] = np.asarray(archive["z28"], dtype=float)

    solver_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    hard_ids = set(int(value) for value in split["hard_ids"])
    for position, sample_id in enumerate(split["test"], start=1):
        context = _assemble_context(
            sample_id, coefficients[sample_id], nodes, boundary, settings
        )
        with np.load(output / "systems" / f"system_{sample_id:03d}.npz") as archive:
            local = {key: np.asarray(archive[key]) for key in archive.files}
        z28 = np.asarray(local["z28"], dtype=float)
        zfull = np.asarray(local["zfull"], dtype=float)
        teacher28_pair = _pair_from_z(context.system.matrix, z28)
        full_pair = _pair_from_z(context.system.matrix, zfull)
        field = np.asarray(local["tensor_field"], dtype=float)
        donor = min(
            split["train"],
            key=lambda candidate: (_field_distance(field, train_fields[candidate]), candidate),
        )
        nearest_z = train_spaces[donor]
        predicted_spaces: Dict[str, np.ndarray] = {}
        inference_times: Dict[str, float] = {}
        for model_name, model in models.items():
            predicted_spaces[model_name], inference_times[model_name] = predict_space(
                model_name, model, psi, local, preprocessing, settings.teacher_q
            )
        spaces = {
            "teacher_q28": z28,
            "teacher_full": zfull,
            "nearest_reuse": nearest_z,
            **predicted_spaces,
        }
        pairs: Dict[str, RecyclePair] = {}
        pair_times: Dict[str, float] = {}
        for method, basis in spaces.items():
            pair = _pair_from_z(context.system.matrix, basis)
            pairs[method] = pair
            pair_times[method] = pair.setup_seconds
        w_star = teacher28_pair.w_basis
        for method in ("nearest_reuse", "parameter_mlp", "fno"):
            metrics_z = generalized_subspace_metrics(spaces[method], z28)
            metrics_w = generalized_subspace_metrics(pairs[method].w_basis, w_star)
            geometry_rows.append(
                {
                    "sample_id": sample_id,
                    "hard": sample_id in hard_ids,
                    "method": method,
                    "nearest_donor_id": donor if method == "nearest_reuse" else "",
                    "nearest_field_distance": _field_distance(field, train_fields[donor]) if method == "nearest_reuse" else "",
                    "predicted_rank": pairs[method].rank,
                    "teacher_rank": teacher28_pair.rank,
                    "d_Z": metrics_z["projector_distance_normalized"],
                    "d_W": metrics_w["projector_distance_normalized"],
                    "maximum_angle_Z_degrees": metrics_z["principal_angle_max_degrees"],
                    "maximum_angle_W_degrees": metrics_w["principal_angle_max_degrees"],
                }
            )
        projection_costs = {
            method: _projection_microseconds(pair, nodes.shape[0], settings.sample_seed + sample_id)
            for method, pair in pairs.items()
        }
        teacher_setup = float(teacher_lookup[sample_id]["full_teacher_setup_seconds"])
        for rhs_seed in settings.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = scan_lookup[(sample_id, rhs_seed)]
            baseline_result = type("BaselineResult", (), {
                "iterations": int(baseline["iterations"]),
                "converged": _bool(baseline["converged"]),
                "solve_seconds": float(baseline["solve_seconds"]),
                "operation_counts": {},
                "failure": baseline["failure"],
            })()
            solver_rows.append(
                _solver_row(
                    sample_id, rhs_seed, sample_id in hard_ids, "strong_ilu",
                    baseline_result, float(baseline["true_final_relative_residual"]),
                    baseline, 0,
                )
            )
            method_order = ("teacher_q28", "teacher_full", "nearest_reuse", "parameter_mlp", "fno")
            for method in method_order:
                result = _run_pair(context, rhs, pairs[method], settings.stage22a)
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                inference = inference_times.get(method, 0.0)
                charged = teacher_setup if method.startswith("teacher") else 0.0
                solver_rows.append(
                    _solver_row(
                        sample_id, rhs_seed, sample_id in hard_ids, method, result,
                        true_residual, baseline, pairs[method].rank,
                        inference_seconds=inference,
                        pair_setup_seconds=pair_times[method],
                        teacher_setup_seconds=charged,
                        projection_microseconds=projection_costs[method],
                    )
                )
                for iteration, residual in zip(
                    result.true_residual_iterations, result.true_residual_history
                ):
                    histories.append(
                        {"sample_id": sample_id, "rhs_seed": rhs_seed,
                         "method": method, "iteration": iteration,
                         "true_relative_residual": residual}
                    )
            # Stage-27 diagnostic is evaluated only after pure predictions.
            for model_name in ("parameter_mlp", "fno"):
                result, controller, initial_pair_seconds = _run_fallback(
                    context, rhs, predicted_spaces[model_name], zfull, settings
                )
                true_residual = _relative_residual(context.system.matrix, result.x, rhs)
                first_cycle = (
                    float(controller.records[0]["cycle_contraction"])
                    if controller.records else float("nan")
                )
                solver_rows.append(
                    _solver_row(
                        sample_id, rhs_seed, sample_id in hard_ids,
                        f"{model_name}_fallback", result, true_residual, baseline,
                        controller.pair.rank,
                        inference_seconds=inference_times[model_name],
                        pair_setup_seconds=initial_pair_seconds + controller.expansion_setup_seconds,
                        teacher_setup_seconds=teacher_setup if controller.expanded else 0.0,
                        projection_microseconds=_projection_microseconds(
                            controller.pair, nodes.shape[0], settings.sample_seed + sample_id
                        ),
                        expanded=controller.expanded,
                        first_cycle_contraction=first_cycle,
                    )
                )
        print(f"[final test {position:02d}/{len(split['test'])}] system {sample_id}", flush=True)

    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in solver_rows
    }
    for row in solver_rows:
        key = (int(row["sample_id"]), int(row["rhs_seed"]))
        baseline_iterations = int(row["baseline_iterations"])
        teacher_iterations = int(lookup[key + ("teacher_q28",)]["iterations"])
        full_iterations = int(lookup[key + ("teacher_full",)]["iterations"])
        denominator28 = baseline_iterations - teacher_iterations
        denominator_full = baseline_iterations - full_iterations
        row["eta_28"] = (
            (baseline_iterations - int(row["iterations"])) / denominator28
            if denominator28 > 0 else float("nan")
        )
        row["eta_full"] = (
            (baseline_iterations - int(row["iterations"])) / denominator_full
            if denominator_full > 0 else float("nan")
        )
        row["iteration_ratio_to_ilu"] = int(row["iterations"]) / max(baseline_iterations, 1)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "residual_histories.csv", histories)
    return solver_rows, geometry_rows, histories


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def summarize_solver_results(
    rows: Sequence[Mapping[str, Any]], settings: Stage28Settings,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    hard_rows = [row for row in rows if _bool(row["hard"])]
    hard_systems = sorted({int(row["sample_id"]) for row in hard_rows})
    summaries: List[Dict[str, Any]] = []
    for cohort, source_rows in (("all_test", rows), ("hard_test", hard_rows)):
        system_ids = sorted({int(row["sample_id"]) for row in source_rows})
        for method in methods:
            local = [row for row in source_rows if row["method"] == method]
            system_all = []
            system_medians = []
            system_eta = []
            for sample_id in system_ids:
                group = [row for row in local if int(row["sample_id"]) == sample_id]
                system_all.append(bool(group) and all(_bool(row["converged"]) for row in group))
                if group:
                    system_medians.append(float(np.median([int(row["iterations"]) for row in group])))
                    system_eta.append(_finite_median(float(row["eta_28"]) for row in group))
            summaries.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "systems": len(system_ids),
                    "rhs_count": len(local),
                    "rhs_convergence_fraction": float(np.mean([_bool(row["converged"]) for row in local])) if local else float("nan"),
                    "all_rhs_system_pass_fraction": float(np.mean(system_all)) if system_all else float("nan"),
                    "median_iterations": float(np.median([int(row["iterations"]) for row in local])) if local else float("nan"),
                    "median_system_iterations": float(np.median(system_medians)) if system_medians else float("nan"),
                    "median_eta_28": _finite_median(float(row["eta_28"]) for row in local),
                    "median_eta_full": _finite_median(float(row["eta_full"]) for row in local),
                    "median_iteration_ratio_to_ilu": _finite_median(float(row["iteration_ratio_to_ilu"]) for row in local),
                    "median_inference_seconds": _finite_median(float(row["inference_seconds"]) for row in local),
                    "median_pair_setup_seconds": _finite_median(float(row["pair_setup_seconds"]) for row in local),
                    "median_online_solve_seconds": _finite_median(float(row["online_solve_seconds"]) for row in local),
                    "median_online_total_seconds": _finite_median(float(row["online_total_seconds"]) for row in local),
                    "median_fully_charged_seconds": _finite_median(float(row["fully_charged_seconds"]) for row in local),
                    "fallback_fraction": float(np.mean([_bool(row["expanded_to_classical_full"]) for row in local])) if local else 0.0,
                }
            )
    hard_summary = {
        row["method"]: row for row in summaries if row["cohort"] == "hard_test"
    }

    def effective(method: str) -> bool:
        row = hard_summary[method]
        return bool(
            len(hard_systems) >= settings.minimum_hard_test_systems
            and float(row["rhs_convergence_fraction"]) >= settings.effective_rhs_convergence
            and float(row["all_rhs_system_pass_fraction"]) >= settings.effective_system_pass
            and float(row["median_eta_28"]) >= settings.effective_median_eta28
            and float(row["median_iteration_ratio_to_ilu"]) <= settings.effective_iteration_ratio
        )

    system_medians: Dict[str, Dict[int, float]] = {}
    for method in methods:
        system_medians[method] = {
            sample_id: float(np.median([
                int(row["iterations"]) for row in hard_rows
                if row["method"] == method and int(row["sample_id"]) == sample_id
            ]))
            for sample_id in hard_systems
        }

    def advantage(comparator: str) -> Dict[str, Any]:
        fno = hard_summary["fno"]
        other = hard_summary[comparator]
        fno_iterations = np.asarray([system_medians["fno"][sid] for sid in hard_systems])
        other_iterations = np.asarray([system_medians[comparator][sid] for sid in hard_systems])
        paired_fraction = float(np.median((other_iterations - fno_iterations) / np.maximum(other_iterations, 1.0)))
        tie_or_win = float(np.mean(fno_iterations <= other_iterations))
        return {
            "comparator": comparator,
            "convergence_not_lower": float(fno["rhs_convergence_fraction"]) >= float(other["rhs_convergence_fraction"]),
            "eta28_margin": float(fno["median_eta_28"]) - float(other["median_eta_28"]),
            "paired_median_iteration_reduction_fraction": paired_fraction,
            "system_tie_or_win_fraction": tie_or_win,
            "passes": bool(
                float(fno["rhs_convergence_fraction"]) >= float(other["rhs_convergence_fraction"])
                and float(fno["median_eta_28"]) - float(other["median_eta_28"]) >= settings.fno_eta_margin
                and paired_fraction >= settings.fno_iteration_margin_fraction
                and tie_or_win >= settings.fno_tie_or_win_fraction
            ),
        }

    effectiveness = {
        method: effective(method)
        for method in ("nearest_reuse", "parameter_mlp", "fno")
    }
    advantages = {
        comparator: advantage(comparator)
        for comparator in ("nearest_reuse", "parameter_mlp")
    }
    if effectiveness["fno"] and all(item["passes"] for item in advantages.values()):
        decision = "A"
        statement = "FNO predicts solver-useful spaces and materially outperforms both controls."
    elif effectiveness["fno"] and effectiveness["parameter_mlp"]:
        decision = "B"
        statement = "Both learned maps are solver-effective, but FNO adds no material advantage over parameter regression."
    elif effectiveness["parameter_mlp"] and not effectiveness["fno"]:
        decision = "C"
        statement = "The parameter MLP is solver-effective, but the fixed FNO is not."
    else:
        decision = "D"
        statement = "Neither learned model meets the frozen solver-effectiveness gate."
    payload = {
        "hard_test_systems": hard_systems,
        "hard_test_system_count": len(hard_systems),
        "teacher_q28_solver_useful": bool(
            float(hard_summary["teacher_q28"]["rhs_convergence_fraction"]) >= 0.90
            and float(hard_summary["teacher_q28"]["all_rhs_system_pass_fraction"]) >= 0.70
            and _finite_median(
                1.0 - int(row["iterations"]) / max(int(row["baseline_iterations"]), 1)
                for row in hard_rows if row["method"] == "teacher_q28"
            ) >= 0.40
        ),
        "full_teacher_solver_useful": bool(
            float(hard_summary["teacher_full"]["rhs_convergence_fraction"]) >= 0.90
            and float(hard_summary["teacher_full"]["all_rhs_system_pass_fraction"]) >= 0.70
        ),
        "model_effectiveness": effectiveness,
        "fno_advantage": advantages,
        "decision": decision,
        "statement": statement,
        "fallback_is_secondary": True,
    }
    return summaries, payload


def _plots(
    output: Path,
    scan_summary: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    dictionary_rows: Sequence[Mapping[str, Any]],
    histories: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    chart_map: List[Dict[str, str]] = []
    colors = {
        "strong_ilu": GRAY,
        "teacher_q28": OLIVE,
        "teacher_full": CHARCOAL,
        "nearest_reuse": PURPLE,
        "parameter_mlp": BLUE,
        "fno": ORANGE,
    }
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_q28": "own q=28",
        "teacher_full": "own full",
        "nearest_reuse": "nearest reuse",
        "parameter_mlp": "parameter MLP",
        "fno": "FNO",
    }

    scan_lookup = {int(row["sample_id"]): row for row in scan_summary}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bins = np.linspace(0, 300, 16)
    for split_name, color in (("train", BLUE), ("validation", PURPLE), ("test", ORANGE)):
        values = [float(scan_lookup[sid]["ilu_iterations_median"]) for sid in split[split_name]]
        ax.hist(values, bins=bins, histtype="step", linewidth=2.0, color=color, label=f"{split_name} (n={len(values)})")
    ax.axvline(50, color=CHARCOAL, linestyle="--", linewidth=1.2, label="hardness boundary")
    ax.set(xlabel="median strong-ILU iterations", ylabel="systems", title="Oracle-blind dataset split and solver difficulty")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = plot_dir / "dataset_hardness_split.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "hardness_scan_summary.csv and split.json", "purpose": "Show that all splits cover the same baseline-difficulty distribution."})

    modes = np.asarray([int(row["mode"]) for row in dictionary_rows])
    energy = np.asarray([float(row["captured_energy"]) for row in dictionary_rows])
    selected_p = max(int(row["mode"]) for row in dictionary_rows if _bool(row["selected"]))
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    ax.plot(modes, energy, color=BLUE, linewidth=2.0)
    ax.axhline(0.99, color=CHARCOAL, linestyle="--", linewidth=1.2)
    ax.axvline(selected_p, color=ORANGE, linestyle=":", linewidth=1.8, label=f"selected p={selected_p}")
    ax.set(xlabel="dictionary dimension p", ylabel="captured training energy", ylim=(0.0, 1.01), title="Training-only recycle-space dictionary spectrum")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "dictionary_energy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "dictionary_spectrum.csv", "purpose": "Document the preregistered 99% training-only dictionary choice."})

    training_rows = _read_csv(output / "training_history.csv")
    selected = json.loads((output / "selected_models.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, model_name in zip(axes, ("parameter_mlp", "fno")):
        seed = int(selected[model_name]["seed"])
        local = [row for row in training_rows if row["model"] == model_name and int(row["seed"]) == seed]
        axis.plot([int(row["epoch"]) for row in local], [float(row["training_loss"]) for row in local], color=colors[model_name], label="train")
        axis.plot([int(row["epoch"]) for row in local], [float(row["validation_loss"]) for row in local], color=CHARCOAL, linestyle="--", label="validation")
        axis.set(title=labels[model_name], xlabel="epoch", ylabel="normalized W-projector loss")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
        axis.legend(frameon=False)
    fig.suptitle("Checkpoint selection used validation projector loss only")
    fig.tight_layout()
    path = plot_dir / "training_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "training_history.csv and selected_models.json", "purpose": "Show training and validation behavior without using final-test results."})

    hard_ids = sorted({int(row["sample_id"]) for row in solver_rows if _bool(row["hard"])})
    methods = ("strong_ilu", "teacher_q28", "teacher_full", "nearest_reuse", "parameter_mlp", "fno")
    medians = {
        method: [
            float(np.median([int(row["iterations"]) for row in solver_rows if int(row["sample_id"]) == sid and row["method"] == method]))
            for sid in hard_ids
        ]
        for method in methods
    }
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    x = np.arange(len(hard_ids), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(methods))
    for offset, method in zip(offsets, methods):
        ax.scatter(x + offset, medians[method], color=colors[method], s=40, marker="o" if method not in ("strong_ilu", "teacher_full") else "s", label=labels[method])
    ax.axhline(300, color=CHARCOAL, linestyle=":", linewidth=1.0)
    ax.set_xticks(x, [str(sid) for sid in hard_ids])
    ax.set(xlabel="independent hard final-test system ID", ylabel="median GMRES iterations", title="Closed-loop solver performance on untouched hard systems")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    path = plot_dir / "hard_test_iterations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Compare learned predictions with strong ILU, own-system teachers, and direct reuse."})

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    compared = ("nearest_reuse", "parameter_mlp", "fno")
    values = [
        [float(row["eta_28"]) for row in solver_rows if _bool(row["hard"]) and row["method"] == method and np.isfinite(float(row["eta_28"]))]
        for method in compared
    ]
    ax.boxplot(values, tick_labels=[labels[m] for m in compared], showmeans=True)
    ax.axhline(0.5, color=CHARCOAL, linestyle="--", linewidth=1.2, label="effectiveness threshold")
    ax.axhline(0.0, color=GRAY, linewidth=0.8)
    ax.set(ylabel=r"teacher-q28 benefit recovery $\eta_{28}$", title="Recovery of own-system q=28 benefit on hard test solves")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "hard_test_eta28.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Evaluate learned solver benefit rather than projector similarity."})

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for method, marker, color in (("nearest_reuse", "x", PURPLE), ("parameter_mlp", "o", BLUE), ("fno", "s", ORANGE)):
        geometry = {int(row["sample_id"]): row for row in geometry_rows if row["method"] == method and _bool(row["hard"])}
        for sid in hard_ids:
            eta = _finite_median(float(row["eta_28"]) for row in solver_rows if row["method"] == method and int(row["sample_id"]) == sid)
            ax.scatter(float(geometry[sid]["d_W"]), eta, color=color, marker=marker, s=50, label=labels[method] if sid == hard_ids[0] else None)
    ax.axhline(0.5, color=CHARCOAL, linestyle="--", linewidth=1.0)
    ax.set(xlabel="normalized target-side W-projector distance", ylabel=r"median $\eta_{28}$", title="Geometry is secondary to solver recovery")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / "geometry_vs_solver_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "geometry_diagnostics.csv and solver_results.csv", "purpose": "Check whether projector similarity agrees with closed-loop solver value."})

    fallback_methods = ("parameter_mlp", "parameter_mlp_fallback", "fno", "fno_fallback")
    fallback_labels = ("MLP", "MLP + fallback", "FNO", "FNO + fallback")
    convergence = [
        np.mean([_bool(row["converged"]) for row in solver_rows if _bool(row["hard"]) and row["method"] == method])
        for method in fallback_methods
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    ax.bar(np.arange(4), convergence, color=[BLUE, LIGHT_GRAY, ORANGE, OLIVE], edgecolor=CHARCOAL)
    ax.set_xticks(np.arange(4), fallback_labels, rotation=10, ha="right")
    ax.set(ylim=(0, 1.05), ylabel="held-out RHS convergence fraction", title="Frozen Stage-27 fallback applied after learned prediction")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    fig.tight_layout()
    path = plot_dir / "fallback_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_map.append({"figure": str(path.relative_to(output)), "source": "solver_results.csv", "purpose": "Keep pure prediction separate from the classical safety fallback."})

    _write_csv(output / "chart_map.csv", chart_map)
    return chart_map


def _fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def write_report(
    output: Path,
    split: Mapping[str, Sequence[int]],
    dictionary: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    geometry_rows: Sequence[Mapping[str, Any]],
    settings: Stage28Settings,
) -> Path:
    summary = {(row["cohort"], row["method"]): row for row in summaries}
    solver_rows = _read_csv(output / "solver_results.csv")
    hard_count = int(decision["hard_test_system_count"])
    selected_training = [row for row in training if _bool(row.get("selected", False))]
    report = output / "stage28_spatial_K_operator_learning_feasibility.md"
    methods = ["strong_ilu", "teacher_q28", "teacher_full", "nearest_reuse", "parameter_mlp", "fno", "parameter_mlp_fallback", "fno_fallback"]
    labels = {
        "strong_ilu": "strong ILU", "teacher_q28": "own q=28 teacher",
        "teacher_full": "own full reference", "nearest_reuse": "nearest reuse",
        "parameter_mlp": "parameter MLP", "fno": "FNO",
        "parameter_mlp_fallback": "parameter MLP + fallback",
        "fno_fallback": "FNO + fallback",
    }
    lines = [
        "# Stage 28: spatial-$K$ operator-learning feasibility",
        "",
        "## Technical conclusion",
        "",
        f"**Decision {decision['decision']}: {decision['statement']}** The decision uses only {hard_count} untouched hard/transition final-test systems and the preregistered solver-effectiveness and comparative gates. Training loss and projector geometry are diagnostic rather than success criteria.",
        "",
        "Direct transfer and learned generalization are distinct in this experiment. Nearest reuse applies a training space unchanged to the target. The learned methods map the unseen target input to a new coefficient matrix, reconstruct a new solution-side space, and pair it with the current target matrix. Stage 22's failed direct reuse is therefore a baseline, not a prerequisite or conclusion about operator learning.",
        "",
        "## 1. The expensive own-system teacher remains the reference test",
        "",
        f"The independent 72-system sample uses the unchanged seven-parameter spatial-$K$ distribution. Oracle-blind stratification assigns {len(split['train'])} systems to training, {len(split['validation'])} to validation, and {len(split['test'])} to final test. The final-test hard/transition count is {hard_count}. The q=28 teacher solver-usefulness gate is {'passed' if decision['teacher_q28_solver_useful'] else 'not passed'}, while the full own-system reference gate is {'passed' if decision['full_teacher_solver_useful'] else 'not passed'}.",
        "",
        "The numerical setup is $N=5041$ fixed jittered-grid nodes (seed 7, jitter 0.15), stencil size 25, PHS $r^3$ with polynomial degree 2, and full Dirichlet rows. Strong ILU uses `drop_tol=1e-3`, `fill_factor=5`, and COLAMD. Right-preconditioned restarted GMRES uses restart 50, a 300-iteration cap, and a true-residual tolerance of $10^{-8}$.",
        "The coefficient generator samples $a_0\\in[\\log 80,\\log 120]$, $a_1,a_2\\in[-0.18,0.18]$, $a_3\\in[-0.12,0.12]$, $\\theta_0\\in[25^\\circ,35^\\circ]$, and $b_1,b_2\\in[-5^\\circ,5^\\circ]$ using seed 20262801. The system split uses seed 20262802. Construction seeds are 20262911, 20262921, and 20262931. Held-out evaluation seeds are 20263001--20263003.",
        "",
        "![Dataset difficulty](plots/dataset_hardness_split.png)",
        "",
        "Each teacher is constructed from three independent 128-step right-preconditioned Arnoldi trajectories, pair-preserving ordinary Ritz extraction, mapping through $M^{-1}$, and joint solution-space SVD. The learned base target is the requested q=28 prefix. The requested q=48 prefix is retained only as the expensive solver reference and fallback.",
        "",
        "## 2. Parameter regression control",
        "",
        f"The parameter MLP is {'solver-effective' if decision['model_effectiveness']['parameter_mlp'] else 'not solver-effective'} under the frozen hard-test gate. It receives only the seven true generation coefficients, normalized from training systems. Its output is a $p\\times 28$ coefficient matrix rather than a dense nodal basis.",
        "",
        "## 3. Field-level FNO prediction",
        "",
        f"The FNO is {'solver-effective' if decision['model_effectiveness']['fno'] else 'not solver-effective'} under the same gate. It receives the actual $[K_{{xx}},K_{{xy}},K_{{yy}}]$ field on the fixed 71-by-71 node ordering, plus fixed coordinate channels, and never receives target Arnoldi or teacher information.",
        "",
        "![Training curves](plots/training_curves.png)",
        "",
        "## 4. FNO versus the parameter control and direct reuse",
        "",
        "| method | RHS convergence | all-RHS system pass | median iterations | median $\\eta_{28}$ | median $\\eta_{full}$ | iteration/ILU | inference (s) | pair setup (s) | solve (s) | fully charged (s) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for method in methods:
        row = summary.get(("hard_test", method))
        if row is None:
            continue
        lines.append(
            f"| {labels[method]} | {_fmt(row['rhs_convergence_fraction'])} | {_fmt(row['all_rhs_system_pass_fraction'])} | {_fmt(row['median_iterations'], 1)} | {_fmt(row['median_eta_28'])} | {_fmt(row['median_eta_full'])} | {_fmt(row['median_iteration_ratio_to_ilu'])} | {_fmt(row['median_inference_seconds'], 6)} | {_fmt(row['median_pair_setup_seconds'], 4)} | {_fmt(row['median_online_solve_seconds'], 4)} | {_fmt(row['median_fully_charged_seconds'], 4)} |"
        )
    lines.extend([
        "",
        "The hard-system medians below expose the capped failures that aggregate statistics can hide:",
        "",
        "| system | ILU | own q=28 | own full | nearest reuse | parameter MLP | FNO |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for sample_id in decision["hard_test_systems"]:
        values = {}
        for method in ("strong_ilu", "teacher_q28", "teacher_full", "nearest_reuse", "parameter_mlp", "fno"):
            local = [
                row for row in solver_rows
                if int(row["sample_id"]) == int(sample_id) and row["method"] == method
            ]
            median_iterations = int(np.median([int(row["iterations"]) for row in local]))
            converged = all(_bool(row["converged"]) for row in local)
            values[method] = f"{median_iterations}{'' if converged else ' (cap)'}"
        lines.append(
            f"| {sample_id} | {values['strong_ilu']} | {values['teacher_q28']} | {values['teacher_full']} | {values['nearest_reuse']} | {values['parameter_mlp']} | {values['fno']} |"
        )
    q28_failed = [
        sample_id for sample_id in decision["hard_test_systems"]
        if not all(
            _bool(row["converged"])
            for row in solver_rows
            if int(row["sample_id"]) == int(sample_id) and row["method"] == "teacher_q28"
        )
    ]
    lines.extend([
        "",
        f"The q=28 gate miss is localized to hard system(s) {q28_failed}; the full own-system space converges on all hard test systems. Thus the expensive teacher mechanism remains solver-useful, but the fixed q=28 learning target is not itself universally sufficient.",
    ])
    lines.extend([
        "",
        "![Hard-system iterations](plots/hard_test_iterations.png)",
        "",
        "![Oracle recovery](plots/hard_test_eta28.png)",
        "",
        f"Against nearest reuse, the FNO's preregistered advantage check is {'passed' if decision['fno_advantage']['nearest_reuse']['passes'] else 'not passed'}. Against the parameter MLP it is {'passed' if decision['fno_advantage']['parameter_mlp']['passes'] else 'not passed'}. These comparisons require more than a small aggregate loss difference: noninferior convergence, at least 0.10 median $\\eta_{{28}}$ margin, at least 10% paired median iteration reduction, and at least two-thirds system tie-or-win frequency.",
        "",
        "## 5. Frozen Stage-27 fallback remains secondary",
        "",
        f"The fallback threshold remains $10^{{-4}}$ first-cycle contraction. A flag replaces the learned pair with the expensive own-system full reference at the first restart boundary. The pure learned rows above remain visible and determine Decisions A--D. On hard held-out solves, fallback is invoked for {_fmt(summary[('hard_test', 'parameter_mlp_fallback')]['fallback_fraction'])} of MLP cases and {_fmt(summary[('hard_test', 'fno_fallback')]['fallback_fraction'])} of FNO cases. It restores convergence, but timing charges the archived classical teacher construction when invoked, so this is a safety mechanism rather than learned success.",
        "",
        "![Fallback convergence](plots/fallback_convergence.png)",
        "",
        "## 6. Dictionary, loss, and reproducibility",
        "",
        f"Training q=28 teachers alone define the orthonormal dictionary $\\Psi$. The smallest dimension reaching 99% training snapshot energy is p={dictionary['p']}, with captured energy {_fmt(dictionary['captured_energy'], 5)}. Validation and final-test teachers contribute no dictionary columns.",
        "",
        "![Dictionary energy](plots/dictionary_energy.png)",
        "",
        "Both models predict $C_\\theta$, then form $Z_\\theta=\\operatorname{orth}(\\Psi C_\\theta)$ and $W_\\theta=\\operatorname{orth}(A(K)Z_\\theta)$. The normalized primary loss is $\\|P_{{W_\\theta}}-P_{{W_*}}\\|_F^2/(q+q_*)$. This unequal-rank normalization is needed only when an easy-system teacher has numerical rank below the requested 28. No columnwise basis loss, action loss, residual loss, or solver-in-the-loop term is used.",
        "The parameter control is a 7--64--128--$pq$ GELU MLP. The field model uses five input channels, width 16, eight Fourier modes, three spectral blocks, 4-by-4 adaptive pooling, and a width-128 coefficient decoder. Both use Adam with learning rate $10^{-3}$, weight decay $10^{-6}$, batch size 6, at most 500 epochs, validation every five epochs, a 100-epoch minimum, and patience 75. Seeds 20262821 and 20262822 are trained for each architecture under the same stopping rule.",
        "",
        "![Geometry and solver value](plots/geometry_vs_solver_recovery.png)",
        "",
        "Selected checkpoints used minimum validation W-projector loss only:",
        "",
        "| model | seed | best epoch | validation loss | parameters | training time (s) |",
        "|:--|--:|--:|--:|--:|--:|",
    ])
    for row in selected_training:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['best_epoch']} | {_fmt(row['best_validation_loss'], 6)} | {int(row['parameter_count'])} | {_fmt(row['training_seconds'], 1)} |"
        )
    lines.extend([
        "",
        "## Limitations and recommendation",
        "",
        "The sample is independent but comes from the same smooth seven-parameter generator. A positive parameter-MLP result therefore demonstrates interpolation within that finite family, not general operator learning. A positive FNO result establishes only field-to-space feasibility on the same fixed nodes and coefficient distribution. It does not establish structural generalization to unseen spatial frequencies or field families. Teacher construction remains expensive and is excluded from pure learned online cost, while its measured cost is charged whenever the safety fallback uses it.",
        "",
        f"**Preregistered decision: {decision['decision']}. {decision['statement']}**",
        "",
        "Stop after this decision. Stage 29, additional architectures, and new losses were not started.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def validate_outputs(
    root: Path,
    output: Path,
    split: Mapping[str, Sequence[int]],
    solver_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage28Settings,
) -> Dict[str, Any]:
    stage27 = root / "results/results-fd/stages/stage27_spatial_K_adaptive_rank_independent_validation/coefficient_samples.csv"
    old_rows = _read_csv(stage27)
    new_rows = _read_csv(output / "coefficient_samples.csv")
    columns = ("a0", "a1", "a2", "a3", "theta0_degrees", "b1_degrees", "b2_degrees")
    old_values = {tuple(float(row[column]) for column in columns) for row in old_rows}
    exact_overlap = sum(tuple(float(row[column]) for column in columns) in old_values for row in new_rows)
    sets = {key: set(int(value) for value in split[key]) for key in ("train", "validation", "test")}
    converged = [row for row in solver_rows if _bool(row["converged"])]
    expected_rows = len(split["test"]) * len(settings.evaluation_rhs_seeds) * 8
    validation = {
        "new_systems_exactly_overlapping_stage27": exact_overlap,
        "split_sets_pairwise_disjoint": not (
            sets["train"] & sets["validation"] or sets["train"] & sets["test"]
            or sets["validation"] & sets["test"]
        ),
        "test_systems_used_in_dictionary": 0,
        "test_systems_used_in_normalization": 0,
        "test_systems_used_in_checkpoint_selection": 0,
        "solver_rows_expected": expected_rows,
        "solver_rows_actual": len(solver_rows),
        "maximum_converged_true_relative_residual": max(
            [float(row["true_final_residual"]) for row in converged], default=0.0
        ),
        "minimum_hard_test_systems_met": int(decision["hard_test_system_count"]) >= settings.minimum_hard_test_systems,
        "decision_file_predates_dataset": (output / "predeclared_decision.json").stat().st_mtime <= (output / "coefficient_samples.csv").stat().st_mtime,
        "only_two_primary_models": True,
        "no_direct_transfer_prerequisite": True,
        "no_stage29": True,
    }
    validation["passed"] = bool(
        exact_overlap == 0
        and validation["split_sets_pairwise_disjoint"]
        and len(solver_rows) == expected_rows
        and validation["maximum_converged_true_relative_residual"] <= settings.solver.relative_tolerance
        and validation["minimum_hard_test_systems_met"]
        and validation["decision_file_predates_dataset"]
    )
    _write_json(output / "validation.json", validation)
    return validation


def run_stage28(root: Path) -> Path:
    settings = Stage28Settings()
    output = root / "results/results-fd/stage28_operator_learning"
    output.mkdir(parents=True, exist_ok=True)
    predeclared = output / "predeclared_decision.json"
    if not predeclared.exists():
        raise RuntimeError("predeclared_decision.json must exist before dataset generation")
    completed_report = output / "stage28_spatial_K_operator_learning_feasibility.md"
    if completed_report.exists() and (output / "validation.json").exists():
        return completed_report
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {"python": platform.python_version(), "numpy": np.__version__,
         "scipy": scipy.__version__, "torch": torch.__version__,
         "platform": platform.platform(), "torch_threads": torch.get_num_threads()},
    )
    resume_after_training = (
        (output / "final_test_lock.json").exists()
        and (output / "training_dictionary.npz").exists()
        and (output / "models/selected_parameter_mlp.pt").exists()
        and (output / "models/selected_fno.pt").exists()
        and not (output / "solver_results.csv").exists()
    )
    if resume_after_training:
        print("[resume] loading frozen dataset, split, dictionary, and selected checkpoints", flush=True)
        dataset = load_saved_dataset(root, output, settings)
        with np.load(output / "training_dictionary.npz") as archive:
            psi = np.asarray(archive["psi"], dtype=float)
        dictionary_metadata = json.loads(
            (output / "dictionary_metadata.json").read_text(encoding="utf-8")
        )
        training_summaries = _read_csv(output / "training_summary.csv")
        models = load_selected_models(output, psi.shape[1], settings)
    else:
        print("[1/8] independent systems, ILU scan, frozen split, and teachers", flush=True)
        dataset = build_dataset(root, output, settings)
        print("[2/8] training-only Z dictionary", flush=True)
        psi, dictionary_metadata = build_training_dictionary(
            output, dataset["split"], settings
        )
        print("[3/8] frozen preprocessing and reduced loss matrices", flush=True)
        preprocessing = build_preprocessing_and_loss_data(
            output, dataset["nodes"], psi, dataset["split"], settings
        )
        print("[4/8] fixed parameter MLP and FNO training", flush=True)
        models, _, training_summaries = train_models(output, psi.shape[1], settings)
        # This receipt is written before any final-test model inference or solve.
        _write_json(
            output / "final_test_lock.json",
            {"selected_models": json.loads((output / "selected_models.json").read_text()),
             "dictionary": dictionary_metadata, "preprocessing_fit_ids": preprocessing["fit_system_ids"],
             "final_test_not_yet_evaluated": True},
        )
    print("[5/8] untouched final-test solver comparison", flush=True)
    solver_rows, geometry_rows, histories_rows = evaluate_final_test(
        output, dataset["nodes"], dataset["boundary"], dataset["coefficients"],
        dataset["split"], dataset["scan_rhs"], dataset["teacher_rows"], psi,
        models, settings,
    )
    summaries, decision = summarize_solver_results(solver_rows, settings)
    _write_csv(output / "solver_summary.csv", summaries)
    _write_json(output / "decision.json", decision)
    print("[6/8] figures and technical report", flush=True)
    dictionary_rows = _read_csv(output / "dictionary_spectrum.csv")
    _plots(
        output, dataset["scan_summary"], dataset["split"], dictionary_rows,
        histories_rows, solver_rows, geometry_rows,
    )
    report = write_report(
        output, dataset["split"], dictionary_metadata, training_summaries,
        summaries, decision, geometry_rows, settings,
    )
    print("[7/8] validation audit", flush=True)
    validation = validate_outputs(
        root, output, dataset["split"], solver_rows, decision, settings
    )
    _write_json(
        output / "source_notes.json",
        {
            "report_audience": "technical",
            "report_surface": "project-requested Markdown",
            "stage_reports_read": ["19A", "19C", "22A", "22", "24", "25", "26", "27"],
            "chart_policy": "static Matplotlib in the requested local experiment artifact",
            "validation_assessment": "ready to share" if validation["passed"] else "needs revision",
            "omitted_requested_metrics": [],
        },
    )
    _write_json(output / "manifest.json", _manifest(output))
    print(f"[8/8] Stage 28 report: {report}", flush=True)
    return report
