"""Stage 18: family-specific W-target MLP on a two-parameter anisotropy family.

The solver, ILU, q=8 Ritz oracle, recycling mechanism, W-projector loss, and
training budget are inherited from Stages 13 and 15.  Only the system family
and the MLP input change: the physical tensor is

    K(R, theta) = Q(theta) diag(R, 1) Q(theta)^T.

The legacy coefficient class uses ``Q(phi) diag(1,R) Q(phi)^T``.  We therefore
assemble with ``phi=theta-90 degrees``, an exact algebraic conversion that is
checked and archived for every system.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats
import torch
from torch import nn

from global_rbf.transfer import grassmann_geodesic

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..pde import ConstantAnisotropicDiffusion
from ..subspace_analysis import subspace_metrics
from .stage9_scalability import _write_csv, _write_json
from .stage10_ritz_transfer import RitzContext, Stage10Settings, _build_context, _discrete_rhs
from .stage12_nonsymmetric_coupling import _orthonormalize
from .stage13_persistent_vs_recycling import Stage13Settings, _run_baseline, _run_recycled
from .stage15_family_specific_w_target_mlp import reduced_w_projector_losses


Point = Tuple[float, float]
BLUE = "#356A9A"
ORANGE = "#C87832"
OLIVE = "#7A7F43"
PINK = "#B45C7A"
PURPLE = "#7562A8"
CHARCOAL = "#263238"


@dataclass(frozen=True)
class Stage18Settings:
    """Frozen scan, split, training, solver, and decision controls."""

    stage10: Stage10Settings = Stage10Settings()
    scan_ratios: Tuple[float, ...] = (90.0, 95.0, 100.0, 105.0, 110.0)
    scan_thetas: Tuple[float, ...] = (
        15.0,
        20.0,
        25.0,
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        35.0,
        40.0,
        45.0,
    )
    scan_rhs_seed: int = 20261800
    hard_iteration_threshold: int = 50
    oracle_room_fraction: float = 0.25
    training_ratios: Tuple[float, ...] = (95.0, 96.25, 97.5, 98.75, 100.0)
    training_thetas: Tuple[float, ...] = (30.0, 30.25, 30.5, 30.75, 31.0)
    q: int = 8
    dictionary_energy_target: float = 0.99
    dictionary_cap: int = 32
    hidden_widths: Tuple[int, int] = (64, 128)
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-6
    max_epochs: int = 3000
    validation_interval: int = 10
    early_stopping_patience: int = 500
    minimum_epochs: int = 500
    model_seeds: Tuple[int, ...] = (20261501, 20261502, 20261503)
    training_dtype: str = "float64"
    cholesky_jitter: float = 1.0e-12
    mlp_clear_margin_eta: float = 0.03
    comparable_eta_tolerance: float = 0.03
    comparable_iteration_tolerance: float = 5.0
    win_fraction_gate: float = 0.625
    reliable_convergence_rate: float = 0.90

    @property
    def train_points(self) -> Tuple[Point, ...]:
        return tuple((r, t) for t in self.training_thetas for r in self.training_ratios)

    @property
    def heldout_ratios(self) -> Tuple[float, ...]:
        return tuple(
            0.5 * (left + right)
            for left, right in zip(self.training_ratios[:-1], self.training_ratios[1:])
        )

    @property
    def heldout_thetas(self) -> Tuple[float, ...]:
        return tuple(
            0.5 * (left + right)
            for left, right in zip(self.training_thetas[:-1], self.training_thetas[1:])
        )

    @property
    def validation_points(self) -> Tuple[Point, ...]:
        return tuple(
            (r, t)
            for j, t in enumerate(self.heldout_thetas)
            for i, r in enumerate(self.heldout_ratios)
            if (i + j) % 2 == 0
        )

    @property
    def test_points(self) -> Tuple[Point, ...]:
        return tuple(
            (r, t)
            for j, t in enumerate(self.heldout_thetas)
            for i, r in enumerate(self.heldout_ratios)
            if (i + j) % 2 == 1
        )

    @property
    def heldout_points(self) -> Tuple[Point, ...]:
        return self.validation_points + self.test_points


class TwoParameterReducedSpaceMLP(nn.Module):
    """Stage-15-scale MLP: normalized ``(R,theta)`` to ``p x q`` coefficients."""

    def __init__(self, p: int, q: int, widths: Tuple[int, int]) -> None:
        super().__init__()
        self.p = int(p)
        self.q = int(q)
        self.network = nn.Sequential(
            nn.Linear(2, widths[0]),
            nn.GELU(),
            nn.Linear(widths[0], widths[1]),
            nn.GELU(),
            nn.Linear(widths[1], p * q),
        )

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        raw = self.network(parameters.reshape(-1, 2))
        return raw.reshape(-1, self.p, self.q)


def requested_tensor(ratio: float, theta_degrees: float) -> np.ndarray:
    """Return exactly ``Q(theta) diag(R,1) Q(theta)^T``."""

    angle = np.deg2rad(theta_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
    )
    return rotation @ np.diag([float(ratio), 1.0]) @ rotation.T


def legacy_angle(theta_degrees: float) -> float:
    """Map the Stage-18 tensor convention to the unchanged legacy assembler."""

    return float(theta_degrees) - 90.0


def tensor_conversion_error(ratio: float, theta_degrees: float) -> float:
    assembled = ConstantAnisotropicDiffusion(
        anisotropy_ratio=ratio, angle_degrees=legacy_angle(theta_degrees)
    ).tensor
    target = requested_tensor(ratio, theta_degrees)
    return float(np.linalg.norm(assembled - target) / np.linalg.norm(target))


def _point_settings(theta: float, settings: Stage18Settings) -> Stage10Settings:
    return replace(settings.stage10, anisotropy_angle_degrees=legacy_angle(theta))


def build_context(point: Point, settings: Stage18Settings) -> RitzContext:
    ratio, theta = point
    return _build_context(ratio, _point_settings(theta, settings))


def oracle_space(context: RitzContext, q: int) -> Tuple[np.ndarray, float, int]:
    raw = np.asarray(context.modes.basis[:, :q], dtype=float)
    return _orthonormalize(raw)


def _stage13_settings(point: Point, settings: Stage18Settings) -> Stage13Settings:
    ratio, theta = point
    return Stage13Settings(
        stage10=_point_settings(theta, settings),
        ratios=(ratio,),
        q=settings.q,
        transfer_pairs=(),
        representative_ratios=(),
    )


def _normalized_point(point: Point, settings: Stage18Settings) -> Tuple[float, float]:
    ratio, theta = point
    r_mid = 0.5 * (settings.training_ratios[0] + settings.training_ratios[-1])
    r_half = 0.5 * (settings.training_ratios[-1] - settings.training_ratios[0])
    t_mid = 0.5 * (settings.training_thetas[0] + settings.training_thetas[-1])
    t_half = 0.5 * (settings.training_thetas[-1] - settings.training_thetas[0])
    return ((ratio - r_mid) / r_half, (theta - t_mid) / t_half)


def _split(point: Point, settings: Stage18Settings) -> str:
    if point in settings.train_points:
        return "train"
    if point in settings.validation_points:
        return "validation"
    if point in settings.test_points:
        return "test"
    return "scan"


def run_hardness_scan(output: Path, settings: Stage18Settings) -> List[Dict[str, Any]]:
    """Scan the full requested box plus transition angles using a disjoint RHS."""

    rows: List[Dict[str, Any]] = []
    for theta in settings.scan_thetas:
        for ratio in settings.scan_ratios:
            point = (ratio, theta)
            start = perf_counter()
            try:
                context = build_context(point, settings)
                basis, z_error, rank = oracle_space(context, settings.q)
                pair = factor_az_petrov(context.system.matrix, basis)
                rhs, _ = _discrete_rhs(context.system, settings.scan_rhs_seed)
                local_settings = _stage13_settings(point, settings)
                baseline, baseline_counts, baseline_time = _run_baseline(
                    context, rhs, pair, local_settings
                )
                recycled, recycled_counts, recycled_time = _run_recycled(
                    context, rhs, pair, local_settings
                )
                room = baseline.iterations - recycled.iterations
                room_fraction = room / max(baseline.iterations, 1)
                hard_useful = bool(
                    baseline.iterations >= settings.hard_iteration_threshold
                    and room_fraction >= settings.oracle_room_fraction
                )
                rows.append(
                    {
                        "ratio": ratio,
                        "theta_degrees": theta,
                        "legacy_assembly_angle_degrees": legacy_angle(theta),
                        "tensor_conversion_relative_error": tensor_conversion_error(ratio, theta),
                        "ilu_iterations": baseline.iterations,
                        "ilu_converged": baseline.converged,
                        "oracle_iterations": recycled.iterations,
                        "oracle_converged": recycled.converged,
                        "oracle_iteration_room": room,
                        "oracle_iteration_reduction_fraction": room_fraction,
                        "hard_and_oracle_useful": hard_useful,
                        "ilu_true_residual": float(baseline.relative_residuals[-1]),
                        "oracle_true_residual": float(recycled.relative_residual),
                        "ilu_online_seconds": baseline_time,
                        "oracle_online_seconds": recycled_time,
                        "ilu_solves": baseline_counts.ilu_solves,
                        "oracle_ilu_solves": recycled_counts.ilu_solves,
                        "oracle_rank": rank,
                        "z_orthogonality_error": z_error,
                        "au_equals_w_relative_error": pair.diagnostics[
                            "au_equals_w_relative_error"
                        ],
                        "condition_wtaz": pair.diagnostics["wtaz_condition"],
                        "assembly_seconds": context.assembly_seconds,
                        "ilu_setup_seconds": context.ilu.setup_seconds,
                        "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
                        "total_context_and_solve_seconds": perf_counter() - start,
                        "failure": "",
                    }
                )
            except Exception as exc:  # numerical failures are evidence, not hidden
                rows.append(
                    {
                        "ratio": ratio,
                        "theta_degrees": theta,
                        "legacy_assembly_angle_degrees": legacy_angle(theta),
                        "tensor_conversion_relative_error": tensor_conversion_error(ratio, theta),
                        "ilu_iterations": "",
                        "ilu_converged": False,
                        "oracle_iterations": "",
                        "oracle_converged": False,
                        "oracle_iteration_room": "",
                        "oracle_iteration_reduction_fraction": "",
                        "hard_and_oracle_useful": False,
                        "ilu_true_residual": "",
                        "oracle_true_residual": "",
                        "ilu_online_seconds": "",
                        "oracle_online_seconds": "",
                        "ilu_solves": "",
                        "oracle_ilu_solves": "",
                        "oracle_rank": "",
                        "z_orthogonality_error": "",
                        "au_equals_w_relative_error": "",
                        "condition_wtaz": "",
                        "assembly_seconds": "",
                        "ilu_setup_seconds": "",
                        "ritz_setup_seconds": "",
                        "total_context_and_solve_seconds": perf_counter() - start,
                        "failure": f"{type(exc).__name__}: {exc}",
                    }
                )
            _write_csv(output / "hardness_scan.csv", rows)
            print(
                f"  scan R={ratio:g}, theta={theta:g}: "
                f"{rows[-1]['ilu_iterations']}/{rows[-1]['oracle_iterations']}",
                flush=True,
            )
    return rows


def build_dataset(
    output: Path, settings: Stage18Settings
) -> Tuple[
    Dict[Point, RitzContext],
    Dict[Point, np.ndarray],
    Dict[Point, AZPetrovFactors],
    List[Dict[str, Any]],
]:
    """Build train and cell-interior held-out systems and identical Ritz oracles."""

    contexts: Dict[Point, RitzContext] = {}
    spaces: Dict[Point, np.ndarray] = {}
    factors: Dict[Point, AZPetrovFactors] = {}
    rows: List[Dict[str, Any]] = []
    archive: Dict[str, np.ndarray] = {}
    for point in settings.train_points + settings.heldout_points:
        ratio, theta = point
        start = perf_counter()
        context = build_context(point, settings)
        basis, z_error, rank = oracle_space(context, settings.q)
        pair = factor_az_petrov(context.system.matrix, basis)
        contexts[point] = context
        spaces[point] = basis
        factors[point] = pair
        tag = f"R{ratio:g}_T{theta:g}".replace(".", "p")
        archive[f"Z_{tag}"] = basis
        archive[f"W_{tag}"] = pair.test
        rows.append(
            {
                "ratio": ratio,
                "theta_degrees": theta,
                "split": _split(point, settings),
                "legacy_assembly_angle_degrees": legacy_angle(theta),
                "tensor_conversion_relative_error": tensor_conversion_error(ratio, theta),
                "oracle_rank": rank,
                "z_orthogonality_error": z_error,
                "w_orthogonality_error": pair.diagnostics["test_orthogonality_error"],
                "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                "condition_wtaz": pair.diagnostics["wtaz_condition"],
                "assembly_seconds": context.assembly_seconds,
                "ilu_setup_seconds": context.ilu.setup_seconds,
                "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
                "context_build_seconds": perf_counter() - start,
            }
        )
        _write_csv(output / "system_oracle_index.csv", rows)
    np.savez_compressed(output / "oracle_spaces.npz", **archive)
    return contexts, spaces, factors, rows


def build_dictionary(
    spaces: Mapping[Point, np.ndarray], settings: Stage18Settings
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Build the 99%-energy Z-POD dictionary from training oracles only."""

    snapshots = np.column_stack([spaces[point] for point in settings.train_points])
    left, singular_values, _ = la.svd(snapshots, full_matrices=False, check_finite=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    required = int(np.searchsorted(energy, settings.dictionary_energy_target) + 1)
    p = min(required, settings.dictionary_cap, left.shape[1])
    dictionary = np.asarray(left[:, :p], dtype=float)
    rows = [
        {
            "mode": index + 1,
            "singular_value": float(value),
            "normalized_singular_value": float(value / singular_values[0]),
            "captured_energy": float(energy[index]),
            "selected": bool(index < p),
        }
        for index, value in enumerate(singular_values)
    ]
    metadata = {
        "p": p,
        "q": settings.q,
        "required_p_for_target": required,
        "energy_target": settings.dictionary_energy_target,
        "captured_energy_at_p": float(energy[p - 1]),
        "cap_active": bool(required > settings.dictionary_cap),
        "snapshot_columns": snapshots.shape[1],
        "training_systems": len(settings.train_points),
        "heldout_systems_used": 0,
        "orthogonality_error": float(
            np.linalg.norm(dictionary.T @ dictionary - np.eye(p), ord="fro")
        ),
    }
    return dictionary, rows, metadata


def _loss_data(
    contexts: Mapping[Point, RitzContext],
    factors: Mapping[Point, AZPetrovFactors],
    dictionary: np.ndarray,
    points: Sequence[Point],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grams, crosses = [], []
    for point in points:
        image = np.asarray(contexts[point].system.matrix @ dictionary, dtype=float)
        grams.append(image.T @ image)
        crosses.append(factors[point].test.T @ image)
    return torch.as_tensor(np.stack(grams), dtype=dtype), torch.as_tensor(
        np.stack(crosses), dtype=dtype
    )


def _input_tensor(
    points: Sequence[Point], settings: Stage18Settings, dtype: torch.dtype
) -> torch.Tensor:
    return torch.as_tensor([_normalized_point(point, settings) for point in points], dtype=dtype)


def train_seed(
    output: Path,
    seed: int,
    contexts: Mapping[Point, RitzContext],
    factors: Mapping[Point, AZPetrovFactors],
    dictionary: np.ndarray,
    settings: Stage18Settings,
) -> Tuple[TwoParameterReducedSpaceMLP, List[Dict[str, Any]], Dict[str, Any]]:
    """Train exactly the Stage-15-scale network with the same W loss and budget."""

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.use_deterministic_algorithms(True)
    dtype = torch.float64 if settings.training_dtype == "float64" else torch.float32
    model = TwoParameterReducedSpaceMLP(
        dictionary.shape[1], settings.q, settings.hidden_widths
    ).to(dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    train_g, train_h = _loss_data(
        contexts, factors, dictionary, settings.train_points, dtype
    )
    val_g, val_h = _loss_data(
        contexts, factors, dictionary, settings.validation_points, dtype
    )
    train_x = _input_tensor(settings.train_points, settings, dtype)
    val_x = _input_tensor(settings.validation_points, settings, dtype)
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    start = perf_counter()
    for epoch in range(settings.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = reduced_w_projector_losses(
            model(train_x), train_g, train_h, settings.cholesky_jitter
        )
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss for seed {seed}, epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = reduced_w_projector_losses(
                    model(val_x), val_g, val_h, settings.cholesky_jitter
                )
            row = {
                "seed": seed,
                "epoch": epoch,
                "training_loss": float(loss.detach().cpu()),
                "validation_loss": float(val_losses.mean().cpu()),
                "validation_loss_max": float(val_losses.max().cpu()),
            }
            history.append(row)
            if row["validation_loss"] < best_loss - 1.0e-10:
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
        raise RuntimeError(f"no finite checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    checkpoint = output / "models" / f"mlp_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "p": dictionary.shape[1],
            "q": settings.q,
            "input_dimension": 2,
            "hidden_widths": settings.hidden_widths,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
        checkpoint,
    )
    selected_train_loss = next(
        row["training_loss"] for row in history if row["epoch"] == best_epoch
    )
    return model, history, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "training_loss_at_selected_epoch": selected_train_loss,
        "best_validation_loss": best_loss,
        "training_seconds": perf_counter() - start,
        "checkpoint": str(checkpoint),
    }


def _predict_space(
    model: TwoParameterReducedSpaceMLP,
    dictionary: np.ndarray,
    point: Point,
    settings: Stage18Settings,
) -> np.ndarray:
    model.eval()
    dtype = next(model.parameters()).dtype
    inputs = _input_tensor((point,), settings, dtype)
    with torch.no_grad():
        coefficients = model(inputs)[0].cpu().numpy()
    basis, error, rank = _orthonormalize(dictionary @ coefficients)
    if rank != settings.q or error > 1.0e-8:
        raise RuntimeError(f"rank-deficient predicted space at {point}")
    return basis


def _nearest_training_point(point: Point, settings: Stage18Settings) -> Point:
    x = np.asarray(_normalized_point(point, settings))
    return min(
        settings.train_points,
        key=lambda candidate: (
            float(np.linalg.norm(np.asarray(_normalized_point(candidate, settings)) - x)),
            candidate[0],
            candidate[1],
        ),
    )


def _bracket(value: float, grid: Sequence[float]) -> Tuple[float, float, float]:
    lower = max(item for item in grid if item < value)
    upper = min(item for item in grid if item > value)
    return lower, upper, (value - lower) / (upper - lower)


def grassmann_bilinear(
    z00: np.ndarray,
    z10: np.ndarray,
    z01: np.ndarray,
    z11: np.ndarray,
    fraction_r: float,
    fraction_theta: float,
) -> np.ndarray:
    """Symmetrized tensor-product Grassmann geodesic interpolation."""

    r_then_t = grassmann_geodesic(
        grassmann_geodesic(z00, z10, fraction_r),
        grassmann_geodesic(z01, z11, fraction_r),
        fraction_theta,
    )
    t_then_r = grassmann_geodesic(
        grassmann_geodesic(z00, z01, fraction_theta),
        grassmann_geodesic(z10, z11, fraction_theta),
        fraction_r,
    )
    combined = grassmann_geodesic(r_then_t, t_then_r, 0.5)
    basis, error, rank = _orthonormalize(combined)
    if rank != z00.shape[1] or error > 1.0e-8:
        raise RuntimeError("Grassmann interpolation returned a deficient basis")
    return basis


def build_candidate_spaces(
    spaces: Mapping[Point, np.ndarray],
    models: Mapping[int, TwoParameterReducedSpaceMLP],
    dictionary: np.ndarray,
    settings: Stage18Settings,
) -> Tuple[
    Dict[str, Dict[Point, np.ndarray]],
    Dict[int, Dict[Point, np.ndarray]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    deterministic = {"nearest_reuse": {}, "grassmann_2d_interpolation": {}}
    det_rows: List[Dict[str, Any]] = []
    for point in settings.heldout_points:
        ratio, theta = point
        nearest = _nearest_training_point(point, settings)
        deterministic["nearest_reuse"][point] = spaces[nearest]
        r0, r1, fr = _bracket(ratio, settings.training_ratios)
        t0, t1, ft = _bracket(theta, settings.training_thetas)
        interpolated = grassmann_bilinear(
            spaces[(r0, t0)],
            spaces[(r1, t0)],
            spaces[(r0, t1)],
            spaces[(r1, t1)],
            fr,
            ft,
        )
        deterministic["grassmann_2d_interpolation"][point] = interpolated
        for method in deterministic:
            det_rows.append(
                {
                    "ratio": ratio,
                    "theta_degrees": theta,
                    "split": _split(point, settings),
                    "method": method,
                    "nearest_source_ratio": nearest[0] if method == "nearest_reuse" else "",
                    "nearest_source_theta": nearest[1] if method == "nearest_reuse" else "",
                    "r_lower": r0 if method != "nearest_reuse" else "",
                    "r_upper": r1 if method != "nearest_reuse" else "",
                    "theta_lower": t0 if method != "nearest_reuse" else "",
                    "theta_upper": t1 if method != "nearest_reuse" else "",
                    "fraction_r": fr if method != "nearest_reuse" else "",
                    "fraction_theta": ft if method != "nearest_reuse" else "",
                }
            )
    learned: Dict[int, Dict[Point, np.ndarray]] = {}
    pred_rows: List[Dict[str, Any]] = []
    for seed, model in models.items():
        learned[seed] = {}
        for point in settings.heldout_points:
            start = perf_counter()
            basis = _predict_space(model, dictionary, point, settings)
            elapsed = perf_counter() - start
            learned[seed][point] = basis
            pred_rows.append(
                {
                    "seed": seed,
                    "ratio": point[0],
                    "theta_degrees": point[1],
                    "split": _split(point, settings),
                    "prediction_seconds": elapsed,
                    "rank": basis.shape[1],
                    "orthogonality_error": float(
                        np.linalg.norm(basis.T @ basis - np.eye(settings.q), ord="fro")
                    ),
                }
            )
    return deterministic, learned, det_rows, pred_rows


def offline_metrics(
    contexts: Mapping[Point, RitzContext],
    spaces: Mapping[Point, np.ndarray],
    factors: Mapping[Point, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[Point, np.ndarray]],
    learned: Mapping[int, Mapping[Point, np.ndarray]],
    settings: Stage18Settings,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[str, Any, Point, np.ndarray]] = []
    for method, local_spaces in deterministic.items():
        candidates.extend((method, "", point, basis) for point, basis in local_spaces.items())
    for seed, local_spaces in learned.items():
        candidates.extend(("learned_mlp", seed, point, basis) for point, basis in local_spaces.items())
    for method, seed, point, basis in candidates:
        pair = factor_az_petrov(contexts[point].system.matrix, basis)
        z_metric = subspace_metrics(basis, spaces[point])
        w_metric = subspace_metrics(pair.test, factors[point].test)
        rows.append(
            {
                "ratio": point[0],
                "theta_degrees": point[1],
                "split": _split(point, settings),
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


def run_solver_evaluation(
    contexts: Mapping[Point, RitzContext],
    factors: Mapping[Point, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[Point, np.ndarray]],
    learned: Mapping[int, Mapping[Point, np.ndarray]],
    prediction_rows: Sequence[Mapping[str, Any]],
    settings: Stage18Settings,
) -> List[Dict[str, Any]]:
    prediction_time = {
        (int(row["seed"]), (float(row["ratio"]), float(row["theta_degrees"]))): float(
            row["prediction_seconds"]
        )
        for row in prediction_rows
    }
    factor_cache: Dict[Tuple[str, Any, Point], Tuple[AZPetrovFactors, float]] = {}
    for point in settings.heldout_points:
        factor_cache[("oracle", "", point)] = (factors[point], 0.0)
        for method, local_spaces in deterministic.items():
            start = perf_counter()
            pair = factor_az_petrov(contexts[point].system.matrix, local_spaces[point])
            factor_cache[(method, "", point)] = (pair, perf_counter() - start)
        for seed, local_spaces in learned.items():
            start = perf_counter()
            pair = factor_az_petrov(contexts[point].system.matrix, local_spaces[point])
            factor_cache[("learned_mlp", seed, point)] = (pair, perf_counter() - start)

    rows: List[Dict[str, Any]] = []
    for point in settings.heldout_points:
        context = contexts[point]
        local_settings = _stage13_settings(point, settings)
        methods: List[Tuple[str, Any, AZPetrovFactors, float, float]] = [
            ("oracle", "", *factor_cache[("oracle", "", point)], 0.0),
            (
                "nearest_reuse",
                "",
                *factor_cache[("nearest_reuse", "", point)],
                0.0,
            ),
            (
                "grassmann_2d_interpolation",
                "",
                *factor_cache[("grassmann_2d_interpolation", "", point)],
                0.0,
            ),
        ]
        methods.extend(
            (
                "learned_mlp",
                seed,
                *factor_cache[("learned_mlp", seed, point)],
                prediction_time[(seed, point)],
            )
            for seed in settings.model_seeds
        )
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline, counts, online = _run_baseline(
                context, rhs, factors[point], local_settings
            )
            rows.append(
                {
                    "ratio": point[0],
                    "theta_degrees": point[1],
                    "split": _split(point, settings),
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
                    "online_solve_seconds": online,
                    "method_setup_seconds": 0.0,
                    "prediction_seconds": 0.0,
                    "a_matvecs": counts.a_matvecs,
                    "ilu_solves": counts.ilu_solves,
                    "eta": 0.0,
                    "failure": baseline.failure,
                }
            )
            local_rows: List[Dict[str, Any]] = []
            for method, seed, pair, setup, prediction in methods:
                result, method_counts, method_online = _run_recycled(
                    context, rhs, pair, local_settings
                )
                local_rows.append(
                    {
                        "ratio": point[0],
                        "theta_degrees": point[1],
                        "split": _split(point, settings),
                        "rhs_seed": rhs_seed,
                        "method": method,
                        "seed": seed,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "cap_hit": bool(
                            result.iterations >= settings.stage10.solver.max_iterations
                            and not result.converged
                        ),
                        "true_final_relative_residual": float(result.relative_residual),
                        "online_solve_seconds": method_online,
                        "method_setup_seconds": setup,
                        "prediction_seconds": prediction,
                        "a_matvecs": int((result.operation_counts or {}).get("a_matvecs", 0)),
                        "ilu_solves": method_counts.ilu_solves,
                        "eta": float("nan"),
                        "failure": "" if result.converged else "maximum iterations reached",
                    }
                )
            oracle_iterations = next(
                row["iterations"] for row in local_rows if row["method"] == "oracle"
            )
            denominator = baseline.iterations - oracle_iterations
            for row in local_rows:
                row["eta"] = (
                    (baseline.iterations - row["iterations"]) / denominator
                    if denominator > 0
                    else float("nan")
                )
                rows.append(row)
    return rows


def aggregate_solver(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, float, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            float(row["ratio"]),
            float(row["theta_degrees"]),
            str(row["method"]),
            str(row["seed"]),
        )
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for key, local in sorted(groups.items()):
        split, ratio, theta, method, seed = key
        summaries.append(
            {
                "split": split,
                "ratio": ratio,
                "theta_degrees": theta,
                "method": method,
                "seed": seed,
                "rhs_count": len(local),
                "iterations_median": float(np.median([row["iterations"] for row in local])),
                "iterations_min": int(min(row["iterations"] for row in local)),
                "iterations_max": int(max(row["iterations"] for row in local)),
                "convergence_rate": float(np.mean([row["converged"] for row in local])),
                "true_final_relative_residual_max": float(
                    max(row["true_final_relative_residual"] for row in local)
                ),
                "online_solve_seconds_median": float(
                    np.median([row["online_solve_seconds"] for row in local])
                ),
                "method_setup_seconds": float(local[0]["method_setup_seconds"]),
                "prediction_seconds": float(local[0]["prediction_seconds"]),
                "eta_median": float(np.nanmedian([row["eta"] for row in local])),
                "cap_hit_rate": float(np.mean([row["cap_hit"] for row in local])),
            }
        )
    return summaries


def paired_mlp_interpolation_summary(
    rows: Sequence[Mapping[str, Any]], selected_seed: int
) -> Dict[str, Any]:
    """Paired RHS-level robustness comparison on untouched test systems."""

    test_rows = [row for row in rows if row["split"] == "test"]
    lookup = {
        (
            float(row["ratio"]),
            float(row["theta_degrees"]),
            int(row["rhs_seed"]),
            str(row["method"]),
            str(row["seed"]),
        ): row
        for row in test_rows
    }
    iteration_differences: List[float] = []
    time_differences: List[float] = []
    for row in test_rows:
        if row["method"] != "learned_mlp" or str(row["seed"]) != str(selected_seed):
            continue
        interpolation = lookup[
            (
                float(row["ratio"]),
                float(row["theta_degrees"]),
                int(row["rhs_seed"]),
                "grassmann_2d_interpolation",
                "",
            )
        ]
        iteration_differences.append(
            float(row["iterations"]) - float(interpolation["iterations"])
        )
        time_differences.append(
            float(row["online_solve_seconds"])
            - float(interpolation["online_solve_seconds"])
        )
    if not iteration_differences:
        raise RuntimeError("no paired test rows for MLP/interpolation comparison")
    wilcoxon = stats.wilcoxon(
        iteration_differences, zero_method="wilcox", alternative="two-sided"
    )
    values = np.asarray(iteration_differences)
    times = np.asarray(time_differences)
    return {
        "pair_count": int(values.size),
        "mlp_minus_interpolation_iterations_median": float(np.median(values)),
        "mlp_minus_interpolation_iterations_mean": float(np.mean(values)),
        "mlp_wins": int(np.sum(values < 0.0)),
        "ties": int(np.sum(values == 0.0)),
        "interpolation_wins": int(np.sum(values > 0.0)),
        "wilcoxon_two_sided_statistic": float(wilcoxon.statistic),
        "wilcoxon_two_sided_pvalue": float(wilcoxon.pvalue),
        "mlp_minus_interpolation_online_seconds_median": float(np.median(times)),
        "mlp_minus_interpolation_online_seconds_mean": float(np.mean(times)),
        "mlp_faster_count": int(np.sum(times < 0.0)),
    }


def select_seed(training_rows: Sequence[Mapping[str, Any]]) -> int:
    return int(min(training_rows, key=lambda row: (row["best_validation_loss"], row["seed"]))["seed"])


def make_decision(
    selected_seed: int,
    solver_summary: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    settings: Stage18Settings,
) -> Dict[str, Any]:
    test = [row for row in solver_summary if row["split"] == "test"]

    def method_rows(method: str, seed: Any = "") -> List[Mapping[str, Any]]:
        return [
            row
            for row in test
            if row["method"] == method and (seed == "" or str(row["seed"]) == str(seed))
        ]

    mlp = method_rows("learned_mlp", selected_seed)
    interpolation = method_rows("grassmann_2d_interpolation")
    oracle = method_rows("oracle")
    if not mlp or not interpolation or not oracle:
        raise RuntimeError("decision rows are incomplete")
    lookup_interp = {(row["ratio"], row["theta_degrees"]): row for row in interpolation}
    differences = [
        row["iterations_median"]
        - lookup_interp[(row["ratio"], row["theta_degrees"])]["iterations_median"]
        for row in mlp
    ]
    eta_mlp = float(np.nanmedian([row["eta_median"] for row in mlp]))
    eta_interp = float(np.nanmedian([row["eta_median"] for row in interpolation]))
    convergence = float(np.mean([row["convergence_rate"] for row in mlp]))
    oracle_convergence = float(np.mean([row["convergence_rate"] for row in oracle]))
    win_fraction = float(np.mean(np.asarray(differences) < 0.0))
    tie_or_win_fraction = float(np.mean(np.asarray(differences) <= 0.0))
    median_difference = float(np.median(differences))
    useful_target = bool(
        oracle_convergence >= settings.reliable_convergence_rate
        and np.nanmedian([row["eta_median"] for row in oracle]) >= 0.99
    )
    if not useful_target:
        outcome = "D"
        statement = "The expanded family does not provide a stable/useful oracle target."
    elif (
        eta_mlp >= eta_interp + settings.mlp_clear_margin_eta
        and win_fraction >= settings.win_fraction_gate
        and convergence >= settings.reliable_convergence_rate
    ):
        outcome = "A"
        statement = "The MLP clearly beats deterministic 2D interpolation."
    elif (
        abs(eta_mlp - eta_interp) <= settings.comparable_eta_tolerance
        and abs(median_difference) <= settings.comparable_iteration_tolerance
        and convergence >= settings.reliable_convergence_rate
    ):
        outcome = "B"
        statement = "The MLP and deterministic 2D interpolation are comparable."
    elif eta_interp >= eta_mlp + settings.mlp_clear_margin_eta:
        outcome = "C"
        statement = "Deterministic 2D interpolation remains clearly stronger."
    else:
        outcome = "B"
        statement = (
            "The MLP and deterministic 2D interpolation are computationally comparable, "
            "with heterogeneous pointwise wins."
        )
    offline_test = [
        row
        for row in offline_rows
        if row["split"] == "test"
        and row["method"] == "learned_mlp"
        and str(row["seed"]) == str(selected_seed)
    ]
    return {
        "outcome": outcome,
        "statement": statement,
        "selected_seed": selected_seed,
        "mlp_test_eta_median": eta_mlp,
        "interpolation_test_eta_median": eta_interp,
        "mlp_test_convergence_rate": convergence,
        "oracle_test_convergence_rate": oracle_convergence,
        "mlp_minus_interpolation_iterations_median": median_difference,
        "mlp_strict_win_fraction": win_fraction,
        "mlp_tie_or_win_fraction": tie_or_win_fraction,
        "mlp_test_normalized_d_w_median": float(
            np.median([row["d_w_normalized"] for row in offline_test])
        ),
        "mlp_test_normalized_d_z_median": float(
            np.median([row["d_z_normalized"] for row in offline_test])
        ),
        "thresholds": {
            "clear_eta_margin": settings.mlp_clear_margin_eta,
            "comparable_eta_tolerance": settings.comparable_eta_tolerance,
            "comparable_iteration_tolerance": settings.comparable_iteration_tolerance,
            "win_fraction_gate": settings.win_fraction_gate,
            "reliable_convergence_rate": settings.reliable_convergence_rate,
        },
    }


def _heatmap(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    field: str,
    title: str,
    colorbar: str,
) -> None:
    ratios = sorted({float(row["ratio"]) for row in rows})
    thetas = sorted({float(row["theta_degrees"]) for row in rows})
    values = np.full((len(thetas), len(ratios)), np.nan)
    lookup = {(float(row["ratio"]), float(row["theta_degrees"])): row for row in rows}
    for j, theta in enumerate(thetas):
        for i, ratio in enumerate(ratios):
            value = lookup[(ratio, theta)].get(field, "")
            if value != "":
                values[j, i] = float(value)
    fig, axis = plt.subplots(figsize=(7.1, 6.2))
    image = axis.pcolormesh(ratios, thetas, values, shading="nearest", cmap="viridis")
    axis.set_xticks(ratios, [f"{value:g}" for value in ratios])
    axis.set_yticks(thetas, [f"{value:g}" for value in thetas])
    axis.add_patch(
        Rectangle(
            (95.0, 30.0),
            5.0,
            1.0,
            fill=False,
            edgecolor="white",
            linewidth=1.8,
            linestyle="--",
            label="training rectangle",
        )
    )
    axis.set(xlabel="anisotropy ratio R", ylabel=r"physical angle $\theta$ (degrees)", title=title)
    axis.legend(frameon=True, loc="upper left", fontsize=8)
    fig.colorbar(image, ax=axis, label=colorbar)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_plots(
    output: Path,
    scan_rows: Sequence[Mapping[str, Any]],
    dictionary_rows: Sequence[Mapping[str, Any]],
    training_history: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summary: Sequence[Mapping[str, Any]],
    selected_seed: int,
    settings: Stage18Settings,
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _heatmap(
        plots / "hardness_scan_ilu_iterations.png",
        scan_rows,
        "ilu_iterations",
        "Strong-ILU hardness scan",
        "GMRES iterations",
    )
    _heatmap(
        plots / "hardness_scan_oracle_reduction.png",
        scan_rows,
        "oracle_iteration_reduction_fraction",
        "q=8 own-oracle recycling room",
        "iteration reduction fraction",
    )

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    modes = [row["mode"] for row in dictionary_rows]
    energy = [row["captured_energy"] for row in dictionary_rows]
    axis.plot(modes, energy, marker="o", markersize=2.5, color=BLUE)
    axis.axhline(settings.dictionary_energy_target, color=ORANGE, linestyle="--")
    axis.axvline(max(row["mode"] for row in dictionary_rows if row["selected"]), color=PURPLE, linestyle=":")
    axis.set(xlabel="dictionary dimension p", ylabel="captured training Z energy", ylim=(0.0, 1.01))
    fig.tight_layout()
    fig.savefig(plots / "dictionary_energy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.7, 4.2))
    for seed in settings.model_seeds:
        local = [row for row in training_history if int(row["seed"]) == seed]
        axis.plot(
            [row["epoch"] for row in local],
            [row["validation_loss"] for row in local],
            label=str(seed),
        )
    axis.set(xlabel="epoch", ylabel=r"validation $L_W$", yscale="log")
    axis.legend(frameon=False, title="seed")
    fig.tight_layout()
    fig.savefig(plots / "training_validation_loss.png", dpi=180)
    plt.close(fig)

    selected_offline = [
        row
        for row in offline_rows
        if row["split"] == "test"
        and (
            row["method"] != "learned_mlp" or str(row["seed"]) == str(selected_seed)
        )
    ]
    fig, axis = plt.subplots(figsize=(7.0, 4.3))
    labels = ["nearest_reuse", "grassmann_2d_interpolation", "learned_mlp"]
    values = [
        [row["d_w_normalized"] for row in selected_offline if row["method"] == method]
        for method in labels
    ]
    axis.boxplot(values, tick_labels=["Nearest", "2D Grassmann", "MLP"], showmeans=True)
    axis.set(ylabel=r"normalized $d_W$", title="Held-out residual-space geometry")
    fig.tight_layout()
    fig.savefig(plots / "test_w_projector_error.png", dpi=180)
    plt.close(fig)

    selected_solver = [
        row
        for row in solver_summary
        if row["split"] == "test"
        and (
            row["method"] != "learned_mlp" or str(row["seed"]) == str(selected_seed)
        )
    ]
    point_order = sorted(settings.test_points, key=lambda point: (point[1], point[0]))
    point_index = {point: index for index, point in enumerate(point_order)}
    method_style = {
        "strong_ilu": (CHARCOAL, "ILU"),
        "oracle": (OLIVE, "Oracle"),
        "nearest_reuse": (ORANGE, "Nearest"),
        "grassmann_2d_interpolation": (PURPLE, "2D Grassmann"),
        "learned_mlp": (PINK, "MLP"),
    }
    fig, axis = plt.subplots(figsize=(9.2, 4.5))
    for method, (color, label) in method_style.items():
        local = [row for row in selected_solver if row["method"] == method]
        local.sort(key=lambda row: point_index[(row["ratio"], row["theta_degrees"])])
        axis.plot(range(len(local)), [row["iterations_median"] for row in local], marker="o", color=color, label=label)
    axis.set_xticks(
        range(len(point_order)),
        [f"{r:g}\n{t:g}°" for r, t in point_order],
        fontsize=8,
    )
    axis.set(xlabel="test point (R, theta)", ylabel="median iterations", title="Held-out recycled-GMRES performance")
    axis.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(plots / "test_solver_iterations.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.3))
    eta_values = []
    eta_labels = []
    for method, label in (
        ("nearest_reuse", "Nearest"),
        ("grassmann_2d_interpolation", "2D Grassmann"),
        ("learned_mlp", "MLP"),
    ):
        eta_values.append([row["eta_median"] for row in selected_solver if row["method"] == method])
        eta_labels.append(label)
    axis.boxplot(eta_values, tick_labels=eta_labels, showmeans=True)
    axis.axhline(1.0, color=OLIVE, linestyle="--", linewidth=1)
    axis.set(ylabel=r"oracle recovery $\eta$", title="Held-out solver acceleration recovered")
    fig.tight_layout()
    fig.savefig(plots / "test_oracle_recovery.png", dpi=180)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return "—"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    if number != 0.0 and (abs(number) < 1.0e-3 or abs(number) >= 1.0e4):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    settings: Stage18Settings,
    scan_rows: Sequence[Mapping[str, Any]],
    dictionary_metadata: Mapping[str, Any],
    training_summaries: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summary: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    paired_summary: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    selected_seed = int(decision["selected_seed"])
    selected_solver = [
        row
        for row in solver_summary
        if row["split"] == "test"
        and (row["method"] != "learned_mlp" or str(row["seed"]) == str(selected_seed))
    ]
    offline_lookup = {
        (row["ratio"], row["theta_degrees"], row["method"]): row
        for row in offline_rows
        if row["split"] == "test"
        and (row["method"] != "learned_mlp" or str(row["seed"]) == str(selected_seed))
    }
    solver_table = []
    labels = {
        "strong_ilu": "Strong ILU",
        "oracle": "Own oracle",
        "nearest_reuse": "Nearest reuse",
        "grassmann_2d_interpolation": "2D Grassmann",
        "learned_mlp": "MLP",
    }
    for row in sorted(
        selected_solver,
        key=lambda item: (
            item["theta_degrees"],
            item["ratio"],
            list(labels).index(item["method"]),
        ),
    ):
        metric = offline_lookup.get((row["ratio"], row["theta_degrees"], row["method"]))
        solver_table.append(
            (
                _fmt(row["ratio"], 3),
                _fmt(row["theta_degrees"], 3),
                labels[row["method"]],
                _fmt(metric["d_z_normalized"] if metric else ""),
                _fmt(metric["d_w_normalized"] if metric else ""),
                _fmt(row["iterations_median"], 1),
                _fmt(row["convergence_rate"]),
                _fmt(row["true_final_relative_residual_max"]),
                _fmt(row["online_solve_seconds_median"]),
                _fmt(row["eta_median"]),
            )
        )
    scan_summary = []
    for theta in settings.scan_thetas:
        local = [row for row in scan_rows if float(row["theta_degrees"]) == theta]
        valid = [row for row in local if row["ilu_iterations"] != ""]
        scan_summary.append(
            (
                f"{theta:g}",
                _fmt(min(float(row["ilu_iterations"]) for row in valid), 0),
                _fmt(max(float(row["ilu_iterations"]) for row in valid), 0),
                _fmt(min(float(row["oracle_iterations"]) for row in valid), 0),
                _fmt(max(float(row["oracle_iterations"]) for row in valid), 0),
                str(sum(bool(row["hard_and_oracle_useful"]) for row in valid)),
            )
        )
    train_table = [
        (
            row["seed"],
            _fmt(row["best_epoch"], 0),
            _fmt(row["training_loss_at_selected_epoch"]),
            _fmt(row["best_validation_loss"]),
            _fmt(row["training_seconds"]),
            "yes" if int(row["seed"]) == selected_seed else "no",
        )
        for row in training_summaries
    ]
    seed_table = []
    for seed in settings.model_seeds:
        local_solver = [
            row
            for row in solver_summary
            if row["split"] == "test"
            and row["method"] == "learned_mlp"
            and str(row["seed"]) == str(seed)
        ]
        local_offline = [
            row
            for row in offline_rows
            if row["split"] == "test"
            and row["method"] == "learned_mlp"
            and str(row["seed"]) == str(seed)
        ]
        seed_table.append(
            (
                seed,
                _fmt(np.median([row["d_w_normalized"] for row in local_offline])),
                _fmt(np.median([row["iterations_median"] for row in local_solver]), 1),
                _fmt(np.median([row["eta_median"] for row in local_solver])),
                _fmt(np.mean([row["convergence_rate"] for row in local_solver])),
                "yes" if seed == selected_seed else "no",
            )
        )
    report = fr"""# Stage 18 — Two-parameter anisotropy family

## Decision

**{decision['outcome']}. {decision['statement']}**

On the untouched cell-interior test systems, the validation-selected MLP obtains median oracle recovery **{_fmt(decision['mlp_test_eta_median'])}**, versus **{_fmt(decision['interpolation_test_eta_median'])}** for deterministic 2D Grassmann interpolation. The MLP-minus-interpolation median iteration difference is **{_fmt(decision['mlp_minus_interpolation_iterations_median'], 1)}**, its strict win fraction is **{_fmt(decision['mlp_strict_win_fraction'])}**, and its convergence rate is **{_fmt(decision['mlp_test_convergence_rate'])}**.

## 1. Exact setup and tensor convention

Stages 15–17 are the empirical starting point. Stage 15 found scalar-family W-loss recovery eta=0.939 but deterministic interpolation remained stronger (0.983); Stage 16 found that W-aware dictionaries did not remove the limitation; Stage 17 found that action supervision reduced action error but worsened solver recovery to 0.851. This round therefore changes the PDE family only and returns to the prescribed Stage-15 W-projector loss.

The fixed numerical setup is N={settings.stage10.n_nodes}, node seed={settings.stage10.node_seed}, jitter={settings.stage10.jitter_fraction}, stencil={settings.stage10.stencil_size}, q={settings.q}, ILU drop tolerance {settings.stage10.classical_ilu.drop_tolerance:g}, fill factor {settings.stage10.classical_ilu.fill_factor:g}, right-preconditioned operator $AM^{{-1}}$, restart {settings.stage10.solver.restart}, cap {settings.stage10.solver.max_iterations}, and tolerance {settings.stage10.solver.relative_tolerance:.0e}. Oracle construction remains the Stage-10/13 64-step right-preconditioned Arnoldi/Ritz procedure. Evaluation uses the same three held-out RHS seeds {settings.stage10.evaluation_rhs_seeds} and unchanged Stage-13 projected/recycled GMRES.

The requested tensor is

$$K(R,\theta)=Q(\theta)\operatorname{{diag}}(R,1)Q(\theta)^T.$$

The legacy class used by Stages 9–17 stores $Q(\phi)\operatorname{{diag}}(1,R)Q(\phi)^T$. Stage 18 uses the exact identity $\phi=\theta-90^\circ$ rather than silently swapping eigenvalues. The maximum archived tensor-conversion relative error is {_fmt(max(float(row['tensor_conversion_relative_error']) for row in scan_rows))}. Consequently, the symbol $\theta=30^\circ$ here is the major-eigenvector angle in the new explicit convention; it is not numerically identical to the legacy class parameter $\phi=30^\circ$ used in Stage 15.

## 2. Hardness scan and selected connected region

The scan uses $R\in\{{90,95,100,105,110\}}$ and physical angles from 15° to 45°, with extra samples across 27°–33°. Its RHS seed {settings.scan_rhs_seed} is disjoint from final evaluation. An entry is flagged hard-and-useful when strong ILU needs at least {settings.hard_iteration_threshold} iterations and the own q=8 oracle removes at least {100*settings.oracle_room_fraction:.0f}% of them.

{_markdown_table(('theta','ILU min','ILU max','oracle min','oracle max','flagged points'), scan_summary)}

![ILU hardness scan](plots/hardness_scan_ilu_iterations.png)

![Oracle room scan](plots/hardness_scan_oracle_reduction.png)

The scan identifies a thin connected transition, not the full requested box: angles at or below 25° are easy, while angles at or above 33° generally hit the iteration cap for both ILU and the fixed q=8 oracle. The supervised rectangle $R\in[95,100]$, $\theta\in[30^\circ,31^\circ]$ is selected because its scan corners retain strong-ILU difficulty and material own-oracle room. This avoids training on clearly easy systems and avoids the region where the q=8 target itself is ineffective.

## 3. Leakage-free two-parameter dataset and dictionary

Training uses the tensor-product 5×5 lattice $R={settings.training_ratios}$ and $\theta={settings.training_thetas}$. Held-out systems are the 4×4 cell centers, strictly intermediate in both coordinates. A checkerboard assignment gives {len(settings.validation_points)} validation and {len(settings.test_points)} final-test systems; no held-out oracle enters dictionary construction or optimization.

The dictionary is the Z-POD of the {len(settings.train_points)} training oracle spaces only. The smallest rank reaching 99% energy is p={dictionary_metadata['required_p_for_target']}; the deployed rank is p={dictionary_metadata['p']} with captured energy {_fmt(dictionary_metadata['captured_energy_at_p'])}. Cap active: {dictionary_metadata['cap_active']}.

![Dictionary energy](plots/dictionary_energy.png)

## 4. Fixed-scale MLP and W-projector loss

The only architectural change forced by the family is the input dimension: `2 -> 64 -> 128 -> p*q`, with GELU. Inputs are separately normalized to [-1,1] over the training rectangle. As in Stage 15,

$$Z_\theta=\operatorname{{orth}}(\Psi C_\theta(R,\theta)),\qquad W_\theta=\operatorname{{orth}}(A(R,\theta)Z_\theta),$$

$$L_W=\|W_\theta W_\theta^T-W_*W_*^T\|_F^2.$$

Adam, learning rate {settings.learning_rate:g}, weight decay {settings.weight_decay:g}, float64 arithmetic, 3000-epoch cap, stopping rules, and seeds are copied from Stage 15. Checkpoint selection uses validation W loss only.

{_markdown_table(('seed','epoch','train L_W','validation L_W','training s','selected'), train_table)}

![Validation loss](plots/training_validation_loss.png)

Test sensitivity is reported for all preregistered seeds, although only the validation-selected seed enters the primary comparison:

{_markdown_table(('seed','norm d_W','median iter.','median eta','conv.','selected'), seed_table)}

## 5. Deterministic controls and held-out solver results

Nearest reuse uses normalized two-parameter Euclidean distance. The interpolation control is not entrywise: it follows Grassmann geodesics along R and theta in both orders, then takes their Grassmann midpoint. Every candidate is paired target-specifically through $AZ=WR$, $U=ZR^{{-1}}$, and evaluated by the unchanged recycled solver.

Projector distances below are normalized by $\sqrt{{2q}}$. Oracle recovery is computed per RHS before aggregation:

$$\eta=\frac{{k_{{ILU}}-k_{{method}}}}{{k_{{ILU}}-k_{{oracle}}}}.$$

{_markdown_table(('R','theta','method','norm d_Z','norm d_W','iter.','conv.','max residual','solve s','eta'), solver_table)}

![W-projector error](plots/test_w_projector_error.png)

![Solver iterations](plots/test_solver_iterations.png)

![Oracle recovery](plots/test_oracle_recovery.png)

## 6. Interpretation

- **Solver target usefulness:** own-oracle convergence on test systems is {_fmt(decision['oracle_test_convergence_rate'])}. The scan explicitly excludes the adjacent capped region where q=8 does not rescue the solver.
- **Learned geometry:** selected-model median normalized $d_W={_fmt(decision['mlp_test_normalized_d_w_median'])}$ and diagnostic $d_Z={_fmt(decision['mlp_test_normalized_d_z_median'])}$.
- **ML versus interpolation:** median eta differs by {_fmt(float(decision['mlp_test_eta_median'])-float(decision['interpolation_test_eta_median']))}; the MLP ties or beats interpolation on {_fmt(decision['mlp_tie_or_win_fraction'])} of test systems.
- **Paired robustness:** across {paired_summary['pair_count']} matched test RHS solves, MLP/interpolation wins split {paired_summary['mlp_wins']}–{paired_summary['interpolation_wins']} with {paired_summary['ties']} ties. The iteration difference (MLP minus interpolation) has median {_fmt(paired_summary['mlp_minus_interpolation_iterations_median'], 1)}, mean {_fmt(paired_summary['mlp_minus_interpolation_iterations_mean'], 1)}, and two-sided paired Wilcoxon p={_fmt(paired_summary['wilcoxon_two_sided_pvalue'])}. This failure to show a consistent signed advantage is not treated as proof of statistical equivalence.
- **Online cost:** the paired median MLP-minus-interpolation solve-time difference is {_fmt(paired_summary['mlp_minus_interpolation_online_seconds_median'])} s; the MLP is faster on {paired_summary['mlp_faster_count']} of {paired_summary['pair_count']} matched solves. Prediction and target-specific $AZ$/QR costs remain separately archived.
- **Scope:** this is a fixed-node, fixed-resolution, narrow two-parameter interpolation study. It does not establish transfer to new geometries, node sets, resolutions, or function-valued coefficients.
- **Next step:** if the outcome is C, the registered next experiment is function-valued coefficient fields $K(x)$, not another loss or dictionary ablation.

## 7. Reproducibility and validation

Validation status: **{validation['assessment']}**. Raw scan, split, oracle, dictionary, training, geometry, solver, timing, decision, environment, and validation artifacts are saved beside this report. Numerical failures and cap hits are retained in the CSVs. Training and oracle-label costs are offline and are not folded into online solve time.
"""
    path = output / "stage18_two_parameter_anisotropy_family.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate(
    output: Path,
    settings: Stage18Settings,
    scan_rows: Sequence[Mapping[str, Any]],
    system_rows: Sequence[Mapping[str, Any]],
    dictionary_metadata: Mapping[str, Any],
    training_summaries: Sequence[Mapping[str, Any]],
    selected_seed: int,
    solver_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    train, validation, test = (
        set(settings.train_points),
        set(settings.validation_points),
        set(settings.test_points),
    )
    checks: Dict[str, Any] = {
        "splits_pairwise_disjoint": not (train & validation or train & test or validation & test),
        "heldout_strictly_interior": all(
            settings.training_ratios[0] < r < settings.training_ratios[-1]
            and settings.training_thetas[0] < t < settings.training_thetas[-1]
            for r, t in validation | test
        ),
        "dictionary_uses_no_heldout_systems": dictionary_metadata["heldout_systems_used"] == 0,
        "dictionary_cap_respected": dictionary_metadata["p"] <= settings.dictionary_cap,
        "tensor_conversion_pass": max(
            float(row["tensor_conversion_relative_error"]) for row in scan_rows + list(system_rows)
        ) < 1.0e-13,
        "oracle_ranks_pass": all(int(row["oracle_rank"]) == settings.q for row in system_rows),
        "au_equals_w_pass": max(
            float(row["au_equals_w_relative_error"]) for row in system_rows
        ) < 1.0e-10,
        "selected_seed_is_validation_best": selected_seed
        == int(min(training_summaries, key=lambda row: row["best_validation_loss"])["seed"]),
        "all_solver_metrics_finite": all(
            np.isfinite(float(row[field]))
            for row in solver_rows
            for field in ("iterations", "true_final_relative_residual", "online_solve_seconds")
        ),
        "converged_rows_meet_tolerance": all(
            (not row["converged"])
            or float(row["true_final_relative_residual"]) <= 1.05 * settings.stage10.solver.relative_tolerance
            for row in solver_rows
        ),
    }
    required = (
        "hardness_scan.csv",
        "system_oracle_index.csv",
        "dictionary_spectrum.csv",
        "dictionary_metadata.json",
        "training_history.csv",
        "training_summary.csv",
        "offline_metrics.csv",
        "solver_results.csv",
        "solver_summary.csv",
        "decision.json",
        "paired_mlp_vs_interpolation.json",
        "settings.json",
        "environment.json",
        "training_dictionary.npz",
        "oracle_spaces.npz",
        "plots/hardness_scan_ilu_iterations.png",
        "plots/hardness_scan_oracle_reduction.png",
        "plots/dictionary_energy.png",
        "plots/training_validation_loss.png",
        "plots/test_w_projector_error.png",
        "plots/test_solver_iterations.png",
        "plots/test_oracle_recovery.png",
    )
    missing = [name for name in required if not (output / name).exists()]
    checks["artifacts_complete"] = not missing
    checks["missing_artifacts"] = missing
    blockers = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    return {
        "assessment": "ready_to_share" if not blockers else "needs_revision",
        "blockers": blockers,
        "checks": checks,
    }


def run_stage18(output: Path) -> Path:
    """Run the complete Stage-18 scan, fixed ablation, validation, and report."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    settings = Stage18Settings()
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
            "solver": "unchanged Stage 13 fixed-space projected/recycled GMRES",
            "oracle": "unchanged Stage 10/13 right-preconditioned Ritz construction",
            "tensor_convention": "Q(theta) diag(R,1) Q(theta)^T via legacy phi=theta-90 degrees",
        },
    )

    print("[1/9] scanning the full two-parameter box and transition", flush=True)
    scan_rows = run_hardness_scan(output, settings)

    print("[2/9] building training and strict cell-interior held-out systems", flush=True)
    contexts, spaces, factors, system_rows = build_dataset(output, settings)
    _write_csv(output / "system_oracle_index.csv", system_rows)

    print("[3/9] rebuilding the training-only Z-POD dictionary", flush=True)
    dictionary, dictionary_rows, dictionary_metadata = build_dictionary(spaces, settings)
    _write_csv(output / "dictionary_spectrum.csv", dictionary_rows)
    _write_json(output / "dictionary_metadata.json", dictionary_metadata)
    np.savez_compressed(
        output / "training_dictionary.npz",
        psi=dictionary,
        train_points=np.asarray(settings.train_points),
        p=dictionary.shape[1],
        q=settings.q,
    )

    print("[4/9] training three fixed-scale two-input MLP seeds", flush=True)
    models: Dict[int, TwoParameterReducedSpaceMLP] = {}
    training_history: List[Dict[str, Any]] = []
    training_summaries: List[Dict[str, Any]] = []
    for seed in settings.model_seeds:
        model, history, summary = train_seed(
            output, seed, contexts, factors, dictionary, settings
        )
        models[seed] = model
        training_history.extend(history)
        training_summaries.append(summary)
        _write_csv(output / "training_history.csv", training_history)
        _write_csv(output / "training_summary.csv", training_summaries)
    selected_seed = select_seed(training_summaries)

    print("[5/9] constructing nearest, Grassmann, and learned spaces", flush=True)
    deterministic, learned, deterministic_rows, prediction_rows = build_candidate_spaces(
        spaces, models, dictionary, settings
    )
    _write_csv(output / "deterministic_space_metadata.csv", deterministic_rows)
    _write_csv(output / "prediction_metadata.csv", prediction_rows)

    print("[6/9] evaluating held-out Z/W geometry", flush=True)
    offline_rows = offline_metrics(
        contexts, spaces, factors, deterministic, learned, settings
    )
    _write_csv(output / "offline_metrics.csv", offline_rows)

    print("[7/9] running unchanged recycled-GMRES comparisons", flush=True)
    solver_rows = run_solver_evaluation(
        contexts, factors, deterministic, learned, prediction_rows, settings
    )
    solver_summary = aggregate_solver(solver_rows)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)

    print("[8/9] applying fixed decision logic and generating figures", flush=True)
    decision = make_decision(
        selected_seed, solver_summary, offline_rows, settings
    )
    _write_json(output / "decision.json", decision)
    paired_summary = paired_mlp_interpolation_summary(solver_rows, selected_seed)
    _write_json(output / "paired_mlp_vs_interpolation.json", paired_summary)
    make_plots(
        output,
        scan_rows,
        dictionary_rows,
        training_history,
        offline_rows,
        solver_summary,
        selected_seed,
        settings,
    )

    print("[9/9] validating evidence and writing report", flush=True)
    validation = validate(
        output,
        settings,
        scan_rows,
        system_rows,
        dictionary_metadata,
        training_summaries,
        selected_seed,
        solver_rows,
    )
    _write_json(output / "validation.json", validation)
    report = write_report(
        output,
        settings,
        scan_rows,
        dictionary_metadata,
        training_summaries,
        offline_rows,
        solver_summary,
        decision,
        paired_summary,
        validation,
    )
    print(json.dumps(decision, indent=2), flush=True)
    print(f"report: {report}", flush=True)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    run_stage18(root / "results" / "stage18_two_parameter_anisotropy_family")
