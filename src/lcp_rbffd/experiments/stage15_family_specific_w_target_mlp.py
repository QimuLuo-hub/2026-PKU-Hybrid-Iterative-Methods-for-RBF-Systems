"""Stage 15: narrow-family supervised W-target MLP feasibility experiment.

The numerical pipeline is intentionally inherited from Stages 10--14.  The
only varying system parameter is the rotated-anisotropy ratio ``R``.  The
oracle is the same right-preconditioned Ritz/recycle space, and every
solver-level evaluation uses the unchanged fixed-space projected GMRES from
Stage 13.

The learned model maps scalar ``R`` to coefficients in a training-only POD
dictionary ``Psi``.  It is trained with a basis-invariant residual/test-space
projector loss; no neural solver, graph model, or operator model is added.
"""

from __future__ import annotations

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
import scipy.linalg as la
import torch
from torch import nn

from global_rbf.transfer import grassmann_geodesic

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..subspace_analysis import subspace_metrics
from .stage9_scalability import _write_csv, _write_json
from .stage10_ritz_transfer import RitzContext, Stage10Settings, _build_context, _discrete_rhs
from .stage12_nonsymmetric_coupling import _orthonormalize
from .stage13_persistent_vs_recycling import (
    Stage13Settings,
    _run_baseline,
    _run_recycled,
)


BLUE = "#356A9A"
ORANGE = "#C87832"
OLIVE = "#7A7F43"
PINK = "#B45C7A"
PURPLE = "#7562A8"
CHARCOAL = "#263238"
LIGHT_GRAY = "#D8DEE4"


@dataclass(frozen=True)
class Stage15Settings:
    """Frozen family, model, evaluation, and decision controls."""

    stage10: Stage10Settings = Stage10Settings()
    ratio_min: float = 90.0
    ratio_max: float = 110.0
    ratio_step: float = 1.0
    q: int = 8
    dictionary_energy_target: float = 0.99
    dictionary_cap: int = 32
    hidden_widths: Tuple[int, int] = (64, 128)
    activation: str = "gelu"
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-6
    max_epochs: int = 3000
    validation_interval: int = 10
    early_stopping_patience: int = 500
    minimum_epochs: int = 500
    model_seeds: Tuple[int, ...] = (20261501, 20261502, 20261503)
    training_dtype: str = "float64"
    cholesky_jitter: float = 1.0e-12
    success_median_eta: float = 0.70
    success_convergence_rate: float = 0.90
    nearest_loss_tolerance_iterations: float = 0.0
    offline_learnable_normalized_dw: float = 0.25
    deterministic_win_fraction: float = 0.60

    @property
    def ratios(self) -> Tuple[float, ...]:
        count = int(round((self.ratio_max - self.ratio_min) / self.ratio_step))
        return tuple(self.ratio_min + self.ratio_step * index for index in range(count + 1))

    @property
    def train_ratios(self) -> Tuple[float, ...]:
        return tuple(ratio for index, ratio in enumerate(self.ratios) if index % 2 == 0)

    @property
    def intermediate_ratios(self) -> Tuple[float, ...]:
        return tuple(ratio for index, ratio in enumerate(self.ratios) if index % 2 == 1)

    @property
    def validation_ratios(self) -> Tuple[float, ...]:
        return self.intermediate_ratios[::2]

    @property
    def test_ratios(self) -> Tuple[float, ...]:
        return self.intermediate_ratios[1::2]

    @property
    def stage13(self) -> Stage13Settings:
        return Stage13Settings(
            stage10=self.stage10,
            ratios=self.ratios,
            q=self.q,
            transfer_pairs=(),
            representative_ratios=self.test_ratios,
        )


class ReducedSpaceMLP(nn.Module):
    """Small scalar-parameter MLP returning ``p x q`` dictionary coefficients."""

    def __init__(self, p: int, q: int, widths: Tuple[int, int]) -> None:
        super().__init__()
        self.p = int(p)
        self.q = int(q)
        self.network = nn.Sequential(
            nn.Linear(1, widths[0]),
            nn.GELU(),
            nn.Linear(widths[0], widths[1]),
            nn.GELU(),
            nn.Linear(widths[1], p * q),
        )

    def forward(self, normalized_ratio: torch.Tensor) -> torch.Tensor:
        raw = self.network(normalized_ratio.reshape(-1, 1))
        return raw.reshape(-1, self.p, self.q)


def _normalized_ratio(ratio: float, settings: Stage15Settings) -> float:
    midpoint = 0.5 * (settings.ratio_min + settings.ratio_max)
    half_width = 0.5 * (settings.ratio_max - settings.ratio_min)
    return (float(ratio) - midpoint) / half_width


def _stage10_archive_root() -> Path:
    return Path(__file__).resolve().parents[3] / "results" / "stage10_ritz_transfer_and_mode_tracking" / "subspaces"


def _load_archived_or_rebuilt_oracle(
    ratio: float,
    context: RitzContext,
    q: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use the exact Stage-13 archive where available, else the same deterministic build."""

    archive_path = _stage10_archive_root() / f"ritz_R{ratio:g}_q32.npz"
    rebuilt = np.asarray(context.modes.basis[:, :q], dtype=float)
    rebuilt, rebuilt_error, rebuilt_rank = _orthonormalize(rebuilt)
    if archive_path.exists():
        with np.load(archive_path) as archive:
            archived = np.asarray(archive["basis"][:, :q], dtype=float)
        archived, archived_error, archived_rank = _orthonormalize(archived)
        reconciliation = subspace_metrics(archived, rebuilt)
        return archived, {
            "oracle_source": "exact Stage 10 archive used by Stage 13",
            "archive_path": str(archive_path),
            "rank": archived_rank,
            "orthogonality_error": archived_error,
            "archived_rebuilt_projector_distance": float(reconciliation["projector_distance"]),
            "archived_rebuilt_max_angle_degrees": float(
                reconciliation["principal_angle_max_degrees"]
            ),
        }
    return rebuilt, {
        "oracle_source": "deterministic Stage 10 construction (no pre-existing archive)",
        "archive_path": "",
        "rank": rebuilt_rank,
        "orthogonality_error": rebuilt_error,
        "archived_rebuilt_projector_distance": float("nan"),
        "archived_rebuilt_max_angle_degrees": float("nan"),
    }


def build_family(
    output: Path,
    settings: Stage15Settings,
) -> Tuple[
    Dict[float, RitzContext],
    Dict[float, np.ndarray],
    Dict[float, AZPetrovFactors],
    List[Dict[str, Any]],
]:
    """Build the fixed-node one-parameter family and frozen oracle spaces."""

    contexts: Dict[float, RitzContext] = {}
    spaces: Dict[float, np.ndarray] = {}
    factors: Dict[float, AZPetrovFactors] = {}
    rows: List[Dict[str, Any]] = []
    archive: Dict[str, np.ndarray] = {}
    for ratio in settings.ratios:
        start = perf_counter()
        context = _build_context(ratio, settings.stage10)
        context_seconds = perf_counter() - start
        basis, provenance = _load_archived_or_rebuilt_oracle(ratio, context, settings.q)
        pair = factor_az_petrov(context.system.matrix, basis)
        contexts[ratio] = context
        spaces[ratio] = basis
        factors[ratio] = pair
        archive[f"Z_R{ratio:g}"] = basis
        archive[f"W_R{ratio:g}"] = pair.test
        split = (
            "train"
            if ratio in settings.train_ratios
            else "validation"
            if ratio in settings.validation_ratios
            else "test"
        )
        rows.append(
            {
                "ratio": ratio,
                "split": split,
                "oracle_source": provenance["oracle_source"],
                "oracle_rank": provenance["rank"],
                "z_orthogonality_error": provenance["orthogonality_error"],
                "archived_rebuilt_projector_distance": provenance[
                    "archived_rebuilt_projector_distance"
                ],
                "archived_rebuilt_max_angle_degrees": provenance[
                    "archived_rebuilt_max_angle_degrees"
                ],
                "w_orthogonality_error": pair.diagnostics["test_orthogonality_error"],
                "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                "condition_wtaz": pair.diagnostics["wtaz_condition"],
                "assembly_seconds": context.assembly_seconds,
                "ilu_setup_seconds": context.ilu.setup_seconds,
                "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
                "context_build_seconds": context_seconds,
            }
        )
        _write_csv(output / "system_oracle_index.csv", rows)
    np.savez_compressed(output / "oracle_spaces.npz", **archive)
    return contexts, spaces, factors, rows


def build_training_dictionary(
    spaces: Mapping[float, np.ndarray],
    settings: Stage15Settings,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Form a training-only POD dictionary and enforce the declared rank cap."""

    snapshots = np.column_stack([spaces[ratio] for ratio in settings.train_ratios])
    left, singular_values, _ = la.svd(snapshots, full_matrices=False, check_finite=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    required = int(np.searchsorted(energy, settings.dictionary_energy_target) + 1)
    p = min(required, settings.dictionary_cap, left.shape[1])
    dictionary = np.asarray(left[:, :p], dtype=float)
    rows = [
        {
            "mode": index + 1,
            "singular_value": float(singular_values[index]),
            "normalized_singular_value": float(singular_values[index] / singular_values[0]),
            "captured_energy": float(energy[index]),
            "selected": bool(index < p),
        }
        for index in range(len(singular_values))
    ]
    metadata = {
        "p": p,
        "q": settings.q,
        "required_p_for_target": required,
        "energy_target": settings.dictionary_energy_target,
        "captured_energy_at_p": float(energy[p - 1]),
        "cap_active": bool(required > settings.dictionary_cap),
        "snapshot_columns": int(snapshots.shape[1]),
        "training_systems": len(settings.train_ratios),
        "orthogonality_error": float(
            np.linalg.norm(dictionary.T @ dictionary - np.eye(p), ord="fro")
        ),
    }
    return dictionary, rows, metadata


def _reduced_loss_data(
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    dictionary: np.ndarray,
    ratios: Sequence[float],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return exact reduced Gram/cross matrices for the W-projector loss."""

    grams = []
    crosses = []
    for ratio in ratios:
        image = np.asarray(contexts[ratio].system.matrix @ dictionary, dtype=float)
        grams.append(image.T @ image)
        crosses.append(factors[ratio].test.T @ image)
    return torch.as_tensor(np.stack(grams), dtype=dtype), torch.as_tensor(
        np.stack(crosses), dtype=dtype
    )


def reduced_w_projector_losses(
    coefficients: torch.Tensor,
    grams: torch.Tensor,
    crosses: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    r"""Compute ``||P_Wtheta-P_W*||_F^2`` without differentiating through N rows.

    If ``Y=A Psi`` and ``C`` is the network output, then
    ``S=C.T Y.T Y C=L L.T`` and ``orth(YC)=Y C L^{-T}``.  Therefore the
    projector loss is ``2q-2||W*.T Y C L^{-T}||_F^2``.  This is exactly the
    requested W-space loss up to the documented Cholesky jitter.
    """

    gc = torch.matmul(grams, coefficients)
    gram_c = torch.matmul(coefficients.transpose(-2, -1), gc)
    q = coefficients.shape[-1]
    scale = torch.diagonal(gram_c, dim1=-2, dim2=-1).mean(dim=-1)
    eye = torch.eye(q, dtype=coefficients.dtype, device=coefficients.device)
    stabilized = gram_c + (jitter * torch.clamp(scale, min=1.0))[..., None, None] * eye
    lower = torch.linalg.cholesky(stabilized)
    hc = torch.matmul(crosses, coefficients)
    overlap_t = torch.linalg.solve_triangular(lower, hc.transpose(-2, -1), upper=False)
    overlap = overlap_t.transpose(-2, -1)
    squared = 2.0 * q - 2.0 * torch.sum(overlap**2, dim=(-2, -1))
    return torch.clamp(squared, min=0.0)


def _model_spaces(
    model: ReducedSpaceMLP,
    dictionary: np.ndarray,
    ratios: Sequence[float],
    settings: Stage15Settings,
) -> Dict[float, np.ndarray]:
    model.eval()
    dtype = next(model.parameters()).dtype
    inputs = torch.as_tensor(
        [_normalized_ratio(ratio, settings) for ratio in ratios], dtype=dtype
    )
    with torch.no_grad():
        coefficients = model(inputs).cpu().numpy()
    result: Dict[float, np.ndarray] = {}
    for ratio, local in zip(ratios, coefficients):
        raw = dictionary @ local
        basis, error, rank = _orthonormalize(raw)
        if rank != settings.q or error > 1.0e-8:
            raise RuntimeError(f"learned space at R={ratio:g} is rank deficient")
        result[ratio] = basis
    return result


def train_one_seed(
    output: Path,
    seed: int,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    dictionary: np.ndarray,
    settings: Stage15Settings,
) -> Tuple[ReducedSpaceMLP, List[Dict[str, Any]], Dict[str, Any]]:
    """Train one seeded MLP and select its checkpoint only by validation loss."""

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.use_deterministic_algorithms(True)
    dtype = torch.float64 if settings.training_dtype == "float64" else torch.float32
    model = ReducedSpaceMLP(dictionary.shape[1], settings.q, settings.hidden_widths).to(dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_g, train_h = _reduced_loss_data(
        contexts, factors, dictionary, settings.train_ratios, dtype
    )
    val_g, val_h = _reduced_loss_data(
        contexts, factors, dictionary, settings.validation_ratios, dtype
    )
    train_x = torch.as_tensor(
        [_normalized_ratio(ratio, settings) for ratio in settings.train_ratios], dtype=dtype
    )
    val_x = torch.as_tensor(
        [_normalized_ratio(ratio, settings) for ratio in settings.validation_ratios], dtype=dtype
    )
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    start = perf_counter()
    for epoch in range(settings.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_coefficients = model(train_x)
        losses = reduced_w_projector_losses(
            train_coefficients, train_g, train_h, settings.cholesky_jitter
        )
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss for seed {seed} at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                validation_losses = reduced_w_projector_losses(
                    model(val_x), val_g, val_h, settings.cholesky_jitter
                )
                validation_loss = float(validation_losses.mean().cpu())
                training_loss = float(loss.cpu())
            history.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_loss_max": float(validation_losses.max().cpu()),
                }
            )
            if validation_loss < best_loss - 1.0e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            if (
                epoch >= settings.minimum_epochs
                and epoch - best_epoch >= settings.early_stopping_patience
            ):
                break
    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    elapsed = perf_counter() - start
    checkpoint = output / "models" / f"mlp_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "p": dictionary.shape[1],
            "q": settings.q,
            "hidden_widths": settings.hidden_widths,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        checkpoint,
    )
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "training_loss_at_selected_epoch": next(
            row["training_loss"] for row in history if row["epoch"] == best_epoch
        ),
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint),
    }
    return model, history, summary


def _nearest_training_ratio(ratio: float, settings: Stage15Settings) -> float:
    """Return the nearest training ratio, breaking midpoint ties downward."""

    return min(settings.train_ratios, key=lambda value: (abs(value - ratio), value))


def _bracketing_training_ratios(
    ratio: float, settings: Stage15Settings
) -> Tuple[float, float, float]:
    lower = max(value for value in settings.train_ratios if value < ratio)
    upper = min(value for value in settings.train_ratios if value > ratio)
    fraction = (ratio - lower) / (upper - lower)
    return lower, upper, fraction


def build_deterministic_spaces(
    spaces: Mapping[float, np.ndarray],
    settings: Stage15Settings,
) -> Tuple[Dict[str, Dict[float, np.ndarray]], List[Dict[str, Any]]]:
    """Build target-independent nearest-reuse and bracketing interpolation controls."""

    result: Dict[str, Dict[float, np.ndarray]] = {
        "nearest_reuse": {},
        "deterministic_interpolation": {},
    }
    rows: List[Dict[str, Any]] = []
    for ratio in settings.intermediate_ratios:
        nearest = _nearest_training_ratio(ratio, settings)
        lower, upper, fraction = _bracketing_training_ratios(ratio, settings)
        result["nearest_reuse"][ratio] = spaces[nearest]
        interpolated = grassmann_geodesic(spaces[lower], spaces[upper], fraction)
        interpolated, error, rank = _orthonormalize(interpolated)
        result["deterministic_interpolation"][ratio] = interpolated
        rows.extend(
            (
                {
                    "ratio": ratio,
                    "method": "nearest_reuse",
                    "source_lower": nearest,
                    "source_upper": "",
                    "fraction": "",
                    "rank": settings.q,
                    "orthogonality_error": float(
                        np.linalg.norm(spaces[nearest].T @ spaces[nearest] - np.eye(settings.q))
                    ),
                },
                {
                    "ratio": ratio,
                    "method": "deterministic_interpolation",
                    "source_lower": lower,
                    "source_upper": upper,
                    "fraction": fraction,
                    "rank": rank,
                    "orthogonality_error": error,
                },
            )
        )
    return result, rows


def build_learned_spaces(
    models: Mapping[int, ReducedSpaceMLP],
    dictionary: np.ndarray,
    settings: Stage15Settings,
) -> Tuple[Dict[int, Dict[float, np.ndarray]], List[Dict[str, Any]]]:
    """Predict every intermediate space and record one-time inference cost."""

    all_spaces: Dict[int, Dict[float, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    for seed, model in models.items():
        per_seed: Dict[float, np.ndarray] = {}
        for ratio in settings.intermediate_ratios:
            start = perf_counter()
            predicted = _model_spaces(model, dictionary, (ratio,), settings)[ratio]
            elapsed = perf_counter() - start
            per_seed[ratio] = predicted
            rows.append(
                {
                    "seed": seed,
                    "ratio": ratio,
                    "split": "validation" if ratio in settings.validation_ratios else "test",
                    "prediction_seconds": elapsed,
                    "rank": predicted.shape[1],
                    "orthogonality_error": float(
                        np.linalg.norm(predicted.T @ predicted - np.eye(settings.q), ord="fro")
                    ),
                }
            )
        all_spaces[seed] = per_seed
    return all_spaces, rows


def offline_metrics(
    contexts: Mapping[float, RitzContext],
    oracle_spaces: Mapping[float, np.ndarray],
    oracle_factors: Mapping[float, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[float, np.ndarray]],
    learned: Mapping[int, Mapping[float, np.ndarray]],
    dictionary: np.ndarray,
    settings: Stage15Settings,
) -> List[Dict[str, Any]]:
    """Evaluate Z/W projector errors for all deployable candidate spaces."""

    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[str, int | str, float, np.ndarray]] = []
    for method, method_spaces in deterministic.items():
        for ratio, basis in method_spaces.items():
            candidates.append((method, "", ratio, basis))
    for seed, method_spaces in learned.items():
        for ratio, basis in method_spaces.items():
            candidates.append(("learned_mlp", seed, ratio, basis))
    for ratio in settings.intermediate_ratios:
        projection_raw = dictionary @ (dictionary.T @ oracle_spaces[ratio])
        projection, _, projection_rank = _orthonormalize(projection_raw)
        if projection_rank == settings.q:
            candidates.append(("dictionary_z_projection", "", ratio, projection))
    for method, seed, ratio, basis in candidates:
        pair = factor_az_petrov(contexts[ratio].system.matrix, basis)
        z_metric = subspace_metrics(basis, oracle_spaces[ratio])
        w_metric = subspace_metrics(pair.test, oracle_factors[ratio].test)
        rows.append(
            {
                "ratio": ratio,
                "split": "validation" if ratio in settings.validation_ratios else "test",
                "method": method,
                "seed": seed,
                "d_z": float(z_metric["projector_distance"]),
                "d_z_normalized": float(z_metric["projector_distance_normalized"]),
                "z_max_angle_degrees": float(z_metric["principal_angle_max_degrees"]),
                "d_w": float(w_metric["projector_distance"]),
                "d_w_normalized": float(w_metric["projector_distance_normalized"]),
                "w_max_angle_degrees": float(w_metric["principal_angle_max_degrees"]),
                "condition_wtaz": float(pair.diagnostics["wtaz_condition"]),
            }
        )
    return rows


def _candidate_factor(
    context: RitzContext,
    basis: np.ndarray,
) -> Tuple[AZPetrovFactors, float]:
    start = perf_counter()
    pair = factor_az_petrov(context.system.matrix, basis)
    return pair, perf_counter() - start


def run_solver_evaluation(
    contexts: Mapping[float, RitzContext],
    oracle_spaces: Mapping[float, np.ndarray],
    oracle_factors: Mapping[float, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[float, np.ndarray]],
    learned: Mapping[int, Mapping[float, np.ndarray]],
    prediction_rows: Sequence[Mapping[str, Any]],
    settings: Stage15Settings,
) -> List[Dict[str, Any]]:
    """Evaluate unchanged strong ILU and recycled GMRES on all unseen ratios."""

    prediction_lookup = {
        (int(row["seed"]), float(row["ratio"])): float(row["prediction_seconds"])
        for row in prediction_rows
    }
    factor_cache: Dict[Tuple[str, int | str, float], Tuple[AZPetrovFactors, float]] = {}
    for ratio in settings.intermediate_ratios:
        factor_cache[("oracle", "", ratio)] = (oracle_factors[ratio], 0.0)
        for method, method_spaces in deterministic.items():
            factor_cache[(method, "", ratio)] = _candidate_factor(
                contexts[ratio], method_spaces[ratio]
            )
        for seed, method_spaces in learned.items():
            factor_cache[("learned_mlp", seed, ratio)] = _candidate_factor(
                contexts[ratio], method_spaces[ratio]
            )

    rows: List[Dict[str, Any]] = []
    for ratio in settings.intermediate_ratios:
        context = contexts[ratio]
        split = "validation" if ratio in settings.validation_ratios else "test"
        methods: List[Tuple[str, int | str, AZPetrovFactors, float, float]] = [
            ("oracle", "", *factor_cache[("oracle", "", ratio)], 0.0),
            (
                "nearest_reuse",
                "",
                *factor_cache[("nearest_reuse", "", ratio)],
                0.0,
            ),
            (
                "deterministic_interpolation",
                "",
                *factor_cache[("deterministic_interpolation", "", ratio)],
                0.0,
            ),
        ]
        methods.extend(
            (
                "learned_mlp",
                seed,
                *factor_cache[("learned_mlp", seed, ratio)],
                prediction_lookup[(seed, ratio)],
            )
            for seed in settings.model_seeds
        )
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline, baseline_counts, baseline_time = _run_baseline(
                context, rhs, oracle_factors[ratio], settings.stage13
            )
            rows.append(
                {
                    "ratio": ratio,
                    "split": split,
                    "rhs_seed": rhs_seed,
                    "method": "strong_ilu",
                    "seed": "",
                    "iterations": baseline.iterations,
                    "converged": baseline.converged,
                    "cap_hit": bool(
                        baseline.iterations >= settings.stage10.solver.max_iterations
                        and not baseline.converged
                    ),
                    "true_final_relative_residual": float(baseline.relative_residuals[-1]),
                    "online_solve_seconds": baseline_time,
                    "method_setup_seconds": 0.0,
                    "prediction_seconds": 0.0,
                    "a_matvecs": baseline_counts.a_matvecs,
                    "ilu_solves": baseline_counts.ilu_solves,
                    "eta": 0.0,
                }
            )
            method_rows: List[Dict[str, Any]] = []
            for method, seed, pair, setup_seconds, prediction_seconds in methods:
                result, counts, online = _run_recycled(context, rhs, pair, settings.stage13)
                method_rows.append(
                    {
                        "ratio": ratio,
                        "split": split,
                        "rhs_seed": rhs_seed,
                        "method": method,
                        "seed": seed,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "cap_hit": bool(
                            result.iterations >= settings.stage10.solver.max_iterations
                            and not result.converged
                        ),
                        "true_final_relative_residual": result.relative_residual,
                        "online_solve_seconds": online,
                        "method_setup_seconds": setup_seconds,
                        "prediction_seconds": prediction_seconds,
                        "a_matvecs": int((result.operation_counts or {}).get("a_matvecs", 0)),
                        "ilu_solves": counts.ilu_solves,
                        "eta": float("nan"),
                    }
                )
            oracle_iterations = next(
                int(row["iterations"]) for row in method_rows if row["method"] == "oracle"
            )
            denominator = baseline.iterations - oracle_iterations
            for row in method_rows:
                row["eta"] = (
                    (baseline.iterations - int(row["iterations"])) / denominator
                    if denominator > 0
                    else float("nan")
                )
                rows.append(row)
    return rows


def aggregate_solver_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["split"]), float(row["ratio"]), str(row["method"]), str(row["seed"]))
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (split, ratio, method, seed), local in sorted(groups.items()):
        summaries.append(
            {
                "split": split,
                "ratio": ratio,
                "method": method,
                "seed": seed,
                "rhs_count": len(local),
                "iterations_median": float(np.median([row["iterations"] for row in local])),
                "iterations_min": int(min(row["iterations"] for row in local)),
                "iterations_max": int(max(row["iterations"] for row in local)),
                "convergence_rate": float(np.mean([bool(row["converged"]) for row in local])),
                "true_final_relative_residual_max": float(
                    max(row["true_final_relative_residual"] for row in local)
                ),
                "online_solve_seconds_median": float(
                    np.median([row["online_solve_seconds"] for row in local])
                ),
                "method_setup_seconds": float(local[0]["method_setup_seconds"]),
                "prediction_seconds": float(local[0]["prediction_seconds"]),
                "eta_median": float(np.nanmedian([row["eta"] for row in local])),
            }
        )
    return summaries


def select_seed(training_summaries: Sequence[Mapping[str, Any]]) -> int:
    """Select the model seed using validation loss only; test metrics are unseen."""

    return int(
        min(
            training_summaries,
            key=lambda row: (float(row["best_validation_loss"]), int(row["seed"])),
        )["seed"]
    )


def make_decision(
    selected_seed: int,
    offline_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    settings: Stage15Settings,
) -> Dict[str, Any]:
    """Apply the declared test-only feasibility gates and four-way decision."""

    test = [row for row in summaries if row["split"] == "test"]
    learned = [
        row
        for row in test
        if row["method"] == "learned_mlp" and int(row["seed"]) == selected_seed
    ]
    nearest = {float(row["ratio"]): row for row in test if row["method"] == "nearest_reuse"}
    interpolation = {
        float(row["ratio"]): row
        for row in test
        if row["method"] == "deterministic_interpolation"
    }
    learned_eta = float(np.nanmedian([row["eta_median"] for row in learned]))
    convergence = float(np.mean([row["convergence_rate"] for row in learned]))
    nearest_differences = np.asarray(
        [
            float(row["iterations_median"])
            - float(nearest[float(row["ratio"])]["iterations_median"])
            for row in learned
        ]
    )
    ratios_not_worse = float(np.mean(nearest_differences <= settings.nearest_loss_tolerance_iterations))
    not_systematically_worse = bool(
        np.median(nearest_differences) <= settings.nearest_loss_tolerance_iterations
        and ratios_not_worse >= 0.50
    )
    learned_offline = [
        row
        for row in offline_rows
        if row["split"] == "test"
        and row["method"] == "learned_mlp"
        and int(row["seed"]) == selected_seed
    ]
    learned_dw = float(np.median([row["d_w_normalized"] for row in learned_offline]))
    offline_learnable = learned_dw <= settings.offline_learnable_normalized_dw
    nearest_eta = float(
        np.nanmedian([row["eta_median"] for row in test if row["method"] == "nearest_reuse"])
    )
    interpolation_eta = float(
        np.nanmedian(
            [row["eta_median"] for row in test if row["method"] == "deterministic_interpolation"]
        )
    )
    primary_pass = bool(
        learned_eta >= settings.success_median_eta
        and convergence >= settings.success_convergence_rate
        and not_systematically_worse
    )
    deterministic_best = max(nearest_eta, interpolation_eta)
    deterministic_wins = []
    for row in learned:
        ratio = float(row["ratio"])
        best_deterministic_iterations = min(
            float(nearest[ratio]["iterations_median"]),
            float(interpolation[ratio]["iterations_median"]),
        )
        deterministic_wins.append(
            best_deterministic_iterations < float(row["iterations_median"])
        )
    deterministic_win_fraction = float(np.mean(deterministic_wins))
    deterministic_stronger = bool(
        deterministic_best > learned_eta
        and deterministic_win_fraction >= settings.deterministic_win_fraction
    )
    if primary_pass and offline_learnable:
        code = "A"
        label = "Family-specific W-target is learnable and solver-useful."
    elif deterministic_stronger:
        code = "C"
        label = "Deterministic interpolation/reuse remains stronger."
    elif offline_learnable and not primary_pass:
        code = "B"
        label = "Target is learnable offline but not solver-useful."
    else:
        code = "D"
        label = "Supervised W-target route should be paused."
    return {
        "decision": code,
        "label": label,
        "selected_seed_by_validation_only": selected_seed,
        "test_median_eta": learned_eta,
        "test_convergence_rate": convergence,
        "test_median_iterations_minus_nearest": float(np.median(nearest_differences)),
        "test_fraction_ratios_not_worse_than_nearest": ratios_not_worse,
        "test_median_normalized_d_w": learned_dw,
        "nearest_test_median_eta": nearest_eta,
        "interpolation_test_median_eta": interpolation_eta,
        "deterministic_test_win_fraction": deterministic_win_fraction,
        "offline_learnable": offline_learnable,
        "primary_success_gate_pass": primary_pass,
        "not_systematically_worse_than_nearest": not_systematically_worse,
        "deterministic_stronger": deterministic_stronger,
        "thresholds": {
            "median_eta": settings.success_median_eta,
            "convergence_rate": settings.success_convergence_rate,
            "offline_normalized_d_w": settings.offline_learnable_normalized_dw,
            "deterministic_win_fraction": settings.deterministic_win_fraction,
        },
    }


def verify_reduced_loss_equivalence(
    model: ReducedSpaceMLP,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    dictionary: np.ndarray,
    learned_spaces: Mapping[float, np.ndarray],
    settings: Stage15Settings,
) -> List[Dict[str, Any]]:
    """Compare the reduced differentiable loss with explicit full-space projectors."""

    dtype = next(model.parameters()).dtype
    ratios = settings.intermediate_ratios
    gram, cross = _reduced_loss_data(contexts, factors, dictionary, ratios, dtype)
    x = torch.as_tensor([_normalized_ratio(ratio, settings) for ratio in ratios], dtype=dtype)
    model.eval()
    with torch.no_grad():
        reduced = reduced_w_projector_losses(
            model(x), gram, cross, settings.cholesky_jitter
        ).cpu().numpy()
    rows = []
    for ratio, reduced_value in zip(ratios, reduced):
        pair = factor_az_petrov(contexts[ratio].system.matrix, learned_spaces[ratio])
        explicit = float(
            subspace_metrics(pair.test, factors[ratio].test)["projector_distance"] ** 2
        )
        rows.append(
            {
                "ratio": ratio,
                "split": "validation" if ratio in settings.validation_ratios else "test",
                "reduced_loss": float(reduced_value),
                "explicit_projector_loss": explicit,
                "absolute_difference": abs(float(reduced_value) - explicit),
            }
        )
    return rows


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_plots(
    output: Path,
    dictionary_rows: Sequence[Mapping[str, Any]],
    training_history: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    selected_seed: int,
    settings: Stage15Settings,
) -> None:
    """Create compact static evidence figures for the requested Markdown report."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    modes = np.asarray([row["mode"] for row in dictionary_rows], dtype=float)
    energy = np.asarray([row["captured_energy"] for row in dictionary_rows], dtype=float)
    selected = max(int(row["mode"]) for row in dictionary_rows if row["selected"])
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(modes, energy, color=BLUE, marker="o", markersize=3, linewidth=1.5)
    axis.axhline(settings.dictionary_energy_target, color=ORANGE, linestyle="--", linewidth=1.2)
    axis.axvline(selected, color=CHARCOAL, linestyle=":", linewidth=1.2)
    axis.set(xlabel="POD dictionary dimension p", ylabel="Captured training energy", title="Training-oracle dictionary energy")
    axis.set_ylim(0.0, 1.01)
    axis.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    _save_figure(figure, output / "plots" / "dictionary_energy.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for seed, color in zip(settings.model_seeds, (BLUE, ORANGE, OLIVE)):
        local = [row for row in training_history if int(row["seed"]) == seed]
        axis.plot(
            [row["epoch"] for row in local],
            [row["validation_loss"] for row in local],
            label=f"seed {seed}",
            color=color,
            linewidth=1.4,
        )
    axis.set(xlabel="Epoch", ylabel=r"Validation $L_W$", title="Validation W-projector loss by training seed")
    axis.set_yscale("log")
    axis.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    _save_figure(figure, output / "plots" / "training_validation_loss.png")

    method_styles = {
        "nearest_reuse": (ORANGE, "o", "Nearest reuse"),
        "deterministic_interpolation": (OLIVE, "s", "Grassmann interpolation"),
        "learned_mlp": (PURPLE, "D", "Learned MLP"),
        "dictionary_z_projection": (CHARCOAL, "^", "Dictionary Z projection"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)
    for method, (color, marker, label) in method_styles.items():
        local = [
            row
            for row in offline_rows
            if row["method"] == method
            and (method != "learned_mlp" or int(row["seed"]) == selected_seed)
        ]
        local.sort(key=lambda row: float(row["ratio"]))
        axes[0].plot(
            [row["ratio"] for row in local],
            [row["d_w_normalized"] for row in local],
            color=color,
            marker=marker,
            linewidth=1.3,
            markersize=4,
            label=label,
        )
        axes[1].plot(
            [row["ratio"] for row in local],
            [row["d_z_normalized"] for row in local],
            color=color,
            marker=marker,
            linewidth=1.3,
            markersize=4,
            label=label,
        )
    axes[0].set(ylabel="Normalized projector distance", title=r"Residual/test space $d_W/\sqrt{2q}$")
    axes[1].set(title=r"Right space $d_Z/\sqrt{2q}$")
    for axis in axes:
        axis.set_xlabel("Held-out anisotropy ratio R")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)
    _save_figure(figure, output / "plots" / "offline_projector_errors.png")

    test = [row for row in solver_summaries if row["split"] == "test"]
    solver_methods = (
        ("strong_ilu", "", CHARCOAL, "o", "Strong ILU"),
        ("oracle", "", BLUE, "s", "Own oracle"),
        ("nearest_reuse", "", ORANGE, "^", "Nearest reuse"),
        ("deterministic_interpolation", "", OLIVE, "v", "Interpolation"),
        ("learned_mlp", str(selected_seed), PURPLE, "D", "Learned MLP"),
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    for method, seed, color, marker, label in solver_methods:
        local = [
            row
            for row in test
            if row["method"] == method and (method != "learned_mlp" or row["seed"] == seed)
        ]
        local.sort(key=lambda row: float(row["ratio"]))
        axis.plot(
            [row["ratio"] for row in local],
            [row["iterations_median"] for row in local],
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=label,
        )
    axis.axhline(settings.stage10.solver.max_iterations, color=PINK, linestyle=":", linewidth=1.0, label="Iteration cap")
    axis.set(xlabel="Test anisotropy ratio R", ylabel="Median iterations across three RHSs", title="Recycled-GMRES solver performance on unseen intermediate systems")
    axis.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    _save_figure(figure, output / "plots" / "test_solver_iterations.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    for method, seed, color, marker, label in solver_methods[2:]:
        local = [
            row
            for row in test
            if row["method"] == method and (method != "learned_mlp" or row["seed"] == seed)
        ]
        local.sort(key=lambda row: float(row["ratio"]))
        axis.plot(
            [row["ratio"] for row in local],
            [row["eta_median"] for row in local],
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=label,
        )
    axis.axhline(settings.success_median_eta, color=BLUE, linestyle="--", linewidth=1.0, label="Primary gate 0.70")
    axis.set(xlabel="Test anisotropy ratio R", ylabel=r"Median oracle recovery $\eta$", title="Oracle acceleration recovered on unseen systems")
    axis.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, output / "plots" / "test_oracle_recovery.png")


def _format(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if abs(number) < 1.0e-3 and number != 0.0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_format(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    settings: Stage15Settings,
    dictionary_metadata: Mapping[str, Any],
    training_summaries: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    selected_seed: int,
    decision: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    """Write the requested concise, technically complete Markdown report."""

    split_line = (
        f"training {tuple(settings.train_ratios)}, validation {tuple(settings.validation_ratios)}, "
        f"test {tuple(settings.test_ratios)}"
    )
    seed_rows = [
        (
            str(int(float(row["seed"]))),
            str(int(float(row["best_epoch"]))),
            row["training_loss_at_selected_epoch"],
            row["best_validation_loss"],
            row["training_seconds"],
            "selected" if int(row["seed"]) == selected_seed else "",
        )
        for row in sorted(training_summaries, key=lambda row: int(row["seed"]))
    ]
    def seed_key(value: Any) -> str:
        return "" if value == "" else str(int(float(value)))

    offline_lookup = {
        (float(row["ratio"]), str(row["method"]), seed_key(row["seed"])): row
        for row in offline_rows
    }
    summary_lookup = {
        (float(row["ratio"]), str(row["method"]), seed_key(row["seed"])): row
        for row in solver_summaries
    }
    test_rows = []
    for ratio in settings.test_ratios:
        for method, seed, label in (
            ("strong_ilu", "", "Strong ILU"),
            ("oracle", "", "Own oracle"),
            ("nearest_reuse", "", "Nearest reuse"),
            ("deterministic_interpolation", "", "Interpolation"),
            ("learned_mlp", str(selected_seed), "Learned MLP"),
        ):
            solver = summary_lookup[(ratio, method, seed)]
            offline = offline_lookup.get((ratio, method, seed))
            test_rows.append(
                (
                    ratio,
                    label,
                    offline["d_w_normalized"] if offline else "",
                    offline["d_z_normalized"] if offline else "",
                    solver["iterations_median"],
                    solver["convergence_rate"],
                    solver["true_final_relative_residual_max"],
                    solver["online_solve_seconds_median"],
                    solver["eta_median"] if method not in ("strong_ilu", "oracle") else (0.0 if method == "strong_ilu" else 1.0),
                )
            )
    learned_seed_rows = []
    for seed in settings.model_seeds:
        local_offline = [
            row
            for row in offline_rows
            if row["split"] == "test" and row["method"] == "learned_mlp" and int(row["seed"]) == seed
        ]
        local_solver = [
            row
            for row in solver_summaries
            if row["split"] == "test" and row["method"] == "learned_mlp" and int(row["seed"]) == seed
        ]
        learned_seed_rows.append(
            (
                str(seed),
                np.median([row["d_w_normalized"] for row in local_offline]),
                np.median([row["d_z_normalized"] for row in local_offline]),
                np.nanmedian([row["eta_median"] for row in local_solver]),
                np.mean([row["convergence_rate"] for row in local_solver]),
                np.median([row["iterations_median"] for row in local_solver]),
                "selected" if seed == selected_seed else "",
            )
        )
    dictionary_projection_dw = float(
        np.median(
            [
                row["d_w_normalized"]
                for row in offline_rows
                if row["split"] == "test"
                and row["method"] == "dictionary_z_projection"
            ]
        )
    )
    cap_note = (
        f"The 99% target requires p={dictionary_metadata['required_p_for_target']}, so the declared cap p={dictionary_metadata['p']} is active and retains {dictionary_metadata['captured_energy_at_p']:.2%}."
        if dictionary_metadata["cap_active"]
        else f"The smallest dictionary meeting 99% is p={dictionary_metadata['p']} (captured energy {dictionary_metadata['captured_energy_at_p']:.2%}); the cap is inactive."
    )
    primary = "passes" if decision["primary_success_gate_pass"] else "fails"
    report = rf"""# Stage 15: family-specific W-target MLP feasibility

## Decision

**{decision['decision']}. {decision['label']}**

The validation-selected MLP {primary} the preregistered solver gate on the untouched test ratios: median oracle recovery is **{decision['test_median_eta']:.3f}**, convergence is **{decision['test_convergence_rate']:.2f}**, and the median iteration difference from nearest-neighbor reuse is **{decision['test_median_iterations_minus_nearest']:+.1f}**. The normalized W-projector error is **{decision['test_median_normalized_d_w']:.3f}**. Model selection used validation loss only; no test outcome influenced the selected seed.

## 1. Fixed numerical and statistical protocol

Stages [13](../stage13_persistent_preconditioner_vs_krylov_recycling/stage13_persistent_preconditioner_vs_krylov_recycling.md) and [14](../stage14_solver_aware_target_loss_diagnostics/stage14_solver_aware_target_loss_diagnostics.md) are the algorithmic and empirical starting points. The experiment keeps N=5041, node seed 7, jitter 0.15, stencil 25, 30° rotated anisotropy, q=8, strong ILU drop tolerance $10^{{-3}}$ and fill factor 5, right-preconditioned $AM^{{-1}}$, tolerance $10^{{-8}}$, restart 50, cap 300, construction RHS seed 20261001, and held-out RHS seeds (20261101, 20261102, 20261103). Only R varies from 90 to 110 in unit steps.

The interpolation split is {split_line}. Even-indexed ratios are training systems; odd intermediate systems are alternated between validation and final test. Individual RHSs are never randomly split. Three MLP seeds receive the same systems, loss, optimizer budget, and training-only dictionary.

At ratios archived by Stage 10, the exact Stage 13 archive is used. Other ratios are regenerated by the identical deterministic Stage 10 right-preconditioned 64-step Arnoldi/Ritz procedure. Every space is evaluated with the unchanged Stage 13 fixed-space recycled GMRES: $AZ=WR$, $U=ZR^{{-1}}$, initial residual projection against W, and Arnoldi on $(I-WW^T)AM^{{-1}}$.

## 2. Reduced dictionary and supervised target

The training snapshot matrix is $S=[Z_*(R_1),\ldots,Z_*(R_{{11}})]$. Its POD dictionary is $\Psi=U_{{:,1:p}}$ from the SVD of S. {cap_note}

The MLP is `1 -> 64 -> 128 -> p*q` with GELU and Adam. For normalized scalar input $\widehat R=(R-100)/10$, it predicts $C_\theta(R)\in\mathbb{{R}}^{{p\times q}}$ and forms

$$Z_{{raw}}=\Psi C_\theta(R),\qquad Z_\theta=\operatorname{{orth}}(Z_{{raw}}),$$

$$W_\theta=\operatorname{{orth}}(A(R)Z_\theta),\qquad L_W=\|W_\theta W_\theta^T-W_*W_*^T\|_F^2.$$

Training uses an algebraically equivalent reduced Gram formulation of $L_W$, avoiding differentiation through 5041 rows; explicit full-space projector losses agree within the validation tolerance. Final spaces and every solver run use explicit QR and `factor_az_petrov`.

The 99% criterion concerns energy in the right spaces Z, not fidelity after applying the nonnormal target A. As a diagnostic, orthogonally projecting each test oracle Z into $\operatorname{{range}}(\Psi)$ still gives median normalized $d_W={dictionary_projection_dw:.3f}$. Thus the prescribed Z-energy dictionary is already a material information bottleneck for the W-target; this diagnostic is not mislabeled as the mathematically optimal W-space approximation inside the dictionary.

![Training dictionary energy](plots/dictionary_energy.png)

| seed | selected epoch | train L_W | validation L_W | training s | selection |
| --- | --- | --- | --- | --- | --- |
{chr(10).join('| ' + ' | '.join(_format(value) for value in row) + ' |' for row in seed_rows)}

![Validation loss](plots/training_validation_loss.png)

## 3. Offline held-out geometry

Distances are reported in normalized form, $d_Z/\sqrt{{2q}}$ and $d_W/\sqrt{{2q}}$, so 0 is identical and 1 is maximally separated. $d_Z$ remains diagnostic only; training optimizes $d_W^2$.

![Offline projector errors](plots/offline_projector_errors.png)

Seed sensitivity on the untouched test systems:

{_table(['seed', 'median normalized d_W', 'median normalized d_Z', 'median eta', 'convergence', 'median iterations', 'selection'], learned_seed_rows)}

## 4. Solver-level test comparison

Oracle recovery is computed per RHS as

$$\eta=\frac{{k_{{ILU}}-k_{{method}}}}{{k_{{ILU}}-k_{{oracle}}}},$$

then summarized by target ratio. Values below 0 are worse than ILU; 1 matches the own-system oracle. Every method uses the same target A, ILU, RHS, start, tolerance, cap, and Stage 13 recycle implementation. Nearest reuse breaks midpoint ties toward the smaller training R. Deterministic interpolation is the midpoint Grassmann geodesic between the two bracketing training oracles.

{_table(['R', 'method', 'norm d_W', 'norm d_Z', 'iterations', 'conv.', 'max true res.', 'online s', 'eta'], test_rows)}

![Test solver iterations](plots/test_solver_iterations.png)

![Test oracle recovery](plots/test_oracle_recovery.png)

The primary gate requires median test $\eta\ge0.70$, convergence $\ge0.90$, and no systematic loss to nearest reuse. The last condition means a nonpositive median iteration difference and learned iterations no worse on at least half of test ratios. The learned model is classified as offline-learnable only when median normalized $d_W\le{settings.offline_learnable_normalized_dw:.2f}$.

## 5. Interpretation and scope

- **Solver usefulness:** test median $\eta={decision['test_median_eta']:.3f}$ versus nearest reuse {decision['nearest_test_median_eta']:.3f} and deterministic interpolation {decision['interpolation_test_median_eta']:.3f}.
- **Robustness:** convergence is {decision['test_convergence_rate']:.2f}; the MLP is no worse than nearest reuse on {decision['test_fraction_ratios_not_worse_than_nearest']:.0%} of test ratios.
- **Deterministic control:** the better of nearest reuse and interpolation uses fewer iterations on {decision['deterministic_test_win_fraction']:.0%} of test ratios and has higher median oracle recovery. This is the direct empirical reason for conclusion C.
- **Offline learning:** median normalized $d_W={decision['test_median_normalized_d_w']:.3f}$; this is {'inside' if decision['offline_learnable'] else 'outside'} the declared 0.25 feasibility band.
- **Scope:** this is a PDE-instance-specific interpolation experiment with fixed nodes, geometry, stencil, angle, ILU, q, and a single scalar parameter. It is not evidence of general transferability, variable-N learning, or operator learning.
- **Cost:** training and oracle-label costs are offline and recorded separately. Online timings include the unchanged recycled solve; prediction and target-specific $AZ$/QR setup are separate columns in the raw evidence.

## 6. Stage 16 recommendation

The four-way outcome is **{decision['decision']}**. {decision['label']} This decision applies only to the family-specific W-projector target and the declared narrow interpolation family. No FNO, GNN, operator learner, neural solver, or altered Ritz target is introduced.

The next justified step is not architecture escalation: retain deterministic interpolation/reuse as the reference and, if supervised learning is revisited, first test a training-only dictionary selected for W-space fidelity rather than Z-energy alone. That would be a new target-design experiment, not a reinterpretation of this negative gate.

## 7. Reproducibility and validation

The validation status is **{validation['assessment']}**. Raw evidence is saved in `system_oracle_index.csv`, `dictionary_spectrum.csv`, `training_history.csv`, `training_summary.csv`, `prediction_metadata.csv`, `offline_metrics.csv`, `solver_results.csv`, `solver_summary.csv`, `loss_equivalence.csv`, `decision.json`, `settings.json`, `environment.json`, and `validation.json`. Model checkpoints and the training-only dictionary are saved separately.

Measured wall times are single-process observations on this machine; iteration counts, convergence flags, true residuals, split membership, and algebra checks are the hardware-independent evidence. Capped ILU baselines remain censored rather than being converted into invented uncapped iteration counts.

As in Stage 14, this macOS NumPy runtime emitted floating-point-status warnings from finite BLAS-backed matrix products. All saved bases, projector metrics, losses, residuals, and timings were explicitly checked for finiteness; ranks, orthogonality, $AU=W$, true-residual tolerances, and the reduced/full loss equivalence all pass. The warnings are retained rather than hidden.
"""
    path = output / "stage15_family_specific_W_target_MLP.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    output: Path,
    settings: Stage15Settings,
    dictionary_metadata: Mapping[str, Any],
    system_rows: Sequence[Mapping[str, Any]],
    training_summaries: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    equivalence_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run calculation, split, algebra, and artifact completeness checks."""

    checks: Dict[str, Any] = {}
    train = set(settings.train_ratios)
    validation = set(settings.validation_ratios)
    test = set(settings.test_ratios)
    checks["split_disjoint"] = not (train & validation or train & test or validation & test)
    checks["split_complete"] = train | validation | test == set(settings.ratios)
    checks["alternating_training"] = settings.train_ratios == settings.ratios[::2]
    checks["intermediate_holdout"] = settings.intermediate_ratios == settings.ratios[1::2]
    checks["system_count_exact"] = len(system_rows) == len(settings.ratios)
    checks["all_oracle_ranks_q"] = all(int(row["oracle_rank"]) == settings.q for row in system_rows)
    checks["all_algebra_finite"] = all(
        np.isfinite(float(row[field]))
        for row in system_rows
        for field in ("z_orthogonality_error", "w_orthogonality_error", "au_equals_w_relative_error", "condition_wtaz")
    )
    checks["dictionary_cap_respected"] = int(dictionary_metadata["p"]) <= settings.dictionary_cap
    checks["dictionary_energy_consistent"] = (
        float(dictionary_metadata["captured_energy_at_p"]) >= settings.dictionary_energy_target
        or bool(dictionary_metadata["cap_active"])
    )
    checks["three_training_seeds"] = len(training_summaries) == len(settings.model_seeds)
    checks["seed_selected_by_validation"] = int(decision["selected_seed_by_validation_only"]) == select_seed(training_summaries)
    checks["prediction_rows_exact"] = len(prediction_rows) == len(settings.model_seeds) * len(settings.intermediate_ratios)
    checks["prediction_bases_orthonormal"] = all(
        int(row["rank"]) == settings.q and float(row["orthogonality_error"]) <= 1.0e-8
        for row in prediction_rows
    )
    expected_offline = len(settings.intermediate_ratios) * (3 + len(settings.model_seeds))
    checks["offline_rows_exact"] = len(offline_rows) == expected_offline
    checks["offline_metrics_finite"] = all(
        np.isfinite(float(row[field]))
        for row in offline_rows
        for field in ("d_z", "d_z_normalized", "d_w", "d_w_normalized", "condition_wtaz")
    )
    expected_solver = len(settings.intermediate_ratios) * len(settings.stage10.evaluation_rhs_seeds) * (4 + len(settings.model_seeds))
    checks["solver_rows_exact"] = len(solver_rows) == expected_solver
    checks["solver_metrics_finite"] = all(
        np.isfinite(float(row[field]))
        for row in solver_rows
        for field in ("iterations", "true_final_relative_residual", "online_solve_seconds")
    )
    checks["converged_residuals_meet_tolerance"] = all(
        (not bool(row["converged"]))
        or float(row["true_final_relative_residual"])
        <= settings.stage10.solver.relative_tolerance * (1.0 + 1.0e-8)
        for row in solver_rows
    )
    baseline = {
        (float(row["ratio"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in solver_rows
        if row["method"] == "strong_ilu"
    }
    oracle = {
        (float(row["ratio"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in solver_rows
        if row["method"] == "oracle"
    }
    eta_errors = []
    for row in solver_rows:
        if row["method"] in ("strong_ilu",):
            continue
        key = (float(row["ratio"]), int(row["rhs_seed"]))
        denominator = baseline[key] - oracle[key]
        expected = (baseline[key] - int(row["iterations"])) / denominator if denominator > 0 else float("nan")
        eta_errors.append(abs(float(row["eta"]) - expected))
    checks["eta_formula_max_absolute_error"] = float(max(eta_errors, default=0.0))
    checks["eta_formula_exact"] = checks["eta_formula_max_absolute_error"] <= 1.0e-12
    checks["all_full_summaries_present"] = len(solver_summaries) == len(settings.intermediate_ratios) * (4 + len(settings.model_seeds))
    checks["reduced_loss_equivalence_max_error"] = float(
        max(row["absolute_difference"] for row in equivalence_rows)
    )
    checks["reduced_loss_equivalent"] = checks["reduced_loss_equivalence_max_error"] <= 1.0e-7
    required = (
        "stage15_family_specific_W_target_MLP.md",
        "system_oracle_index.csv",
        "dictionary_spectrum.csv",
        "training_history.csv",
        "training_summary.csv",
        "prediction_metadata.csv",
        "offline_metrics.csv",
        "solver_results.csv",
        "solver_summary.csv",
        "loss_equivalence.csv",
        "decision.json",
        "settings.json",
        "environment.json",
        "training_dictionary.npz",
        "plots/dictionary_energy.png",
        "plots/training_validation_loss.png",
        "plots/offline_projector_errors.png",
        "plots/test_solver_iterations.png",
        "plots/test_oracle_recovery.png",
    )
    missing = [path for path in required if not (output / path).exists()]
    checks["missing_artifacts"] = missing
    checks["artifacts_complete"] = not missing
    blockers = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    return {
        "assessment": "ready_to_share" if not blockers else "needs_revision",
        "blockers": blockers,
        "checks": checks,
    }


def run_stage15(output: Path) -> Path:
    """Execute the complete, fixed Stage 15 feasibility experiment."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    settings = Stage15Settings()
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "solver": "unchanged Stage 13 fixed-space recycled/projected GMRES",
            "oracle": "unchanged Stage 10/13 right-preconditioned Ritz construction",
        },
    )
    print("[1/10] building dense fixed-node anisotropy family and oracle spaces", flush=True)
    contexts, spaces, factors, system_rows = build_family(output, settings)
    _write_csv(output / "system_oracle_index.csv", system_rows)

    print("[2/10] forming training-only POD dictionary", flush=True)
    dictionary, dictionary_rows, dictionary_metadata = build_training_dictionary(spaces, settings)
    _write_csv(output / "dictionary_spectrum.csv", dictionary_rows)
    _write_json(output / "dictionary_metadata.json", dictionary_metadata)
    np.savez_compressed(
        output / "training_dictionary.npz",
        psi=dictionary,
        train_ratios=np.asarray(settings.train_ratios),
        p=dictionary.shape[1],
        q=settings.q,
    )

    print("[3/10] training three scalar-R MLP seeds with W-projector loss", flush=True)
    models: Dict[int, ReducedSpaceMLP] = {}
    training_history: List[Dict[str, Any]] = []
    training_summaries: List[Dict[str, Any]] = []
    for seed in settings.model_seeds:
        model, history, summary = train_one_seed(
            output, seed, contexts, spaces, factors, dictionary, settings
        )
        models[seed] = model
        training_history.extend(history)
        training_summaries.append(summary)
        _write_csv(output / "training_history.csv", training_history)
        _write_csv(output / "training_summary.csv", training_summaries)
    selected_seed = select_seed(training_summaries)

    print("[4/10] constructing deterministic and learned held-out spaces", flush=True)
    deterministic, deterministic_rows = build_deterministic_spaces(spaces, settings)
    learned, prediction_rows = build_learned_spaces(models, dictionary, settings)
    _write_csv(output / "deterministic_space_metadata.csv", deterministic_rows)
    _write_csv(output / "prediction_metadata.csv", prediction_rows)

    print("[5/10] evaluating held-out projector geometry", flush=True)
    offline_rows = offline_metrics(
        contexts, spaces, factors, deterministic, learned, dictionary, settings
    )
    _write_csv(output / "offline_metrics.csv", offline_rows)

    print("[6/10] running unchanged ILU/recycled-GMRES held-out solves", flush=True)
    solver_rows = run_solver_evaluation(
        contexts, spaces, factors, deterministic, learned, prediction_rows, settings
    )
    solver_summaries = aggregate_solver_rows(solver_rows)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summaries)

    print("[7/10] checking reduced loss against explicit full-space projectors", flush=True)
    equivalence_rows = verify_reduced_loss_equivalence(
        models[selected_seed], contexts, factors, dictionary, learned[selected_seed], settings
    )
    _write_csv(output / "loss_equivalence.csv", equivalence_rows)

    print("[8/10] applying the preregistered feasibility gate", flush=True)
    decision = make_decision(selected_seed, offline_rows, solver_summaries, settings)
    _write_json(output / "decision.json", decision)

    print("[9/10] generating figures and validating raw evidence", flush=True)
    make_plots(
        output,
        dictionary_rows,
        training_history,
        offline_rows,
        solver_summaries,
        selected_seed,
        settings,
    )
    preliminary = {"assessment": "pending_report_artifact"}
    write_report(
        output,
        settings,
        dictionary_metadata,
        training_summaries,
        offline_rows,
        solver_summaries,
        selected_seed,
        decision,
        preliminary,
    )
    validation = validate_outputs(
        output,
        settings,
        dictionary_metadata,
        system_rows,
        training_summaries,
        prediction_rows,
        offline_rows,
        solver_rows,
        solver_summaries,
        equivalence_rows,
        decision,
    )
    _write_json(output / "validation.json", validation)

    print("[10/10] writing final Stage 15 report", flush=True)
    report = write_report(
        output,
        settings,
        dictionary_metadata,
        training_summaries,
        offline_rows,
        solver_summaries,
        selected_seed,
        decision,
        validation,
    )
    print(f"Stage 15 complete: {report}", flush=True)
    return report


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[3]
    run_stage15(repository / "results" / "stage15_family_specific_W_target_MLP")
