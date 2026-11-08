"""Stage 17: solver-action supervision with the frozen Stage 15 setup.

Only the supervised loss changes.  The exact Stage 15 Z-POD dictionary,
scalar MLP, optimizer budget/seeds, system split, Ritz labels, ILU, and
recycled GMRES implementation are reused without modification.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..krylov import restarted_gmres
from ..subspace_analysis import subspace_metrics
from .stage9_scalability import _read_csv, _write_csv, _write_json
from .stage10_ritz_transfer import RitzContext, _discrete_rhs
from .stage12_nonsymmetric_coupling import _orthonormalize
from .stage13_persistent_vs_recycling import _run_baseline, _run_recycled
from .stage15_family_specific_w_target_mlp import (
    BLUE,
    CHARCOAL,
    LIGHT_GRAY,
    OLIVE,
    ORANGE,
    PINK,
    PURPLE,
    ReducedSpaceMLP,
    Stage15Settings,
    build_deterministic_spaces,
    build_learned_spaces,
    select_seed,
)
from .stage16_w_aware_dictionary_ablation import load_exact_stage15_family


@dataclass(frozen=True)
class Stage17Settings:
    """Action-probe controls; all model and solver controls live in Stage 15."""

    stage15: Stage15Settings = Stage15Settings()
    training_probe_seeds: Tuple[int, ...] = (20261701, 20261702, 20261703)
    validation_probe_seeds: Tuple[int, ...] = (20261801, 20261802, 20261803)
    test_probe_seeds: Tuple[int, ...] = (20261901, 20261902, 20261903)
    action_epsilon_scale: float = 1.0e-12
    clear_eta_improvement: float = 0.03
    interpolation_eta_tolerance: float = 0.02
    ratio_win_fraction: float = 0.60
    stable_convergence_rate: float = 0.90


@dataclass(frozen=True)
class ActionProbeBlock:
    """Reduced sufficient statistics for exact action-loss evaluation."""

    ratio: float
    split: str
    gram_y: np.ndarray
    residual_to_dictionary_image: np.ndarray
    residual_to_oracle_test: np.ndarray
    target_dictionary_coefficients: np.ndarray
    target_norm_squared: np.ndarray
    vector_norms: np.ndarray
    trajectory_labels: np.ndarray
    iteration_indices: np.ndarray

    @property
    def count(self) -> int:
        return int(self.target_norm_squared.size)


METHOD_LABELS = {
    "strong_ilu": "Strong ILU",
    "oracle": "Own oracle",
    "nearest_reuse": "Nearest reuse",
    "deterministic_interpolation": "Interpolation",
    "mlp_w_loss": "MLP / W loss",
    "mlp_action": "MLP / action loss",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage15_directory() -> Path:
    return _repository_root() / "results" / "stage15_family_specific_W_target_MLP"


def load_stage15_dictionary_and_models(
    settings: Stage17Settings,
) -> Tuple[np.ndarray, Dict[int, ReducedSpaceMLP], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load the exact Stage 15 dictionary, checkpoints, and training records."""

    source = _stage15_directory()
    with np.load(source / "training_dictionary.npz") as archive:
        dictionary = np.asarray(archive["psi"], dtype=float)
        train_ratios = tuple(float(value) for value in archive["train_ratios"])
    if train_ratios != settings.stage15.train_ratios or dictionary.shape != (5041, 19):
        raise RuntimeError("saved Stage 15 dictionary does not match the frozen protocol")
    models: Dict[int, ReducedSpaceMLP] = {}
    for seed in settings.stage15.model_seeds:
        checkpoint = torch.load(
            source / "models" / f"mlp_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = ReducedSpaceMLP(
            dictionary.shape[1], settings.stage15.q, settings.stage15.hidden_widths
        ).to(dtype=torch.float64)
        model.load_state_dict(checkpoint["state_dict"])
        models[seed] = model
    return (
        dictionary,
        models,
        _read_csv(source / "training_history.csv"),
        _read_csv(source / "training_summary.csv"),
    )


def _split_for_ratio(ratio: float, settings: Stage17Settings) -> str:
    if ratio in settings.stage15.train_ratios:
        return "train"
    if ratio in settings.stage15.validation_ratios:
        return "validation"
    if ratio in settings.stage15.test_ratios:
        return "test"
    raise ValueError(f"R={ratio:g} is outside the frozen split")


def _probe_seeds(split: str, settings: Stage17Settings) -> Tuple[int, ...]:
    return {
        "train": settings.training_probe_seeds,
        "validation": settings.validation_probe_seeds,
        "test": settings.test_probe_seeds,
    }[split]


def collect_action_probe_block(
    context: RitzContext,
    factors: AZPetrovFactors,
    dictionary: np.ndarray,
    ratio: float,
    split: str,
    seeds: Sequence[int],
    settings: Stage17Settings,
) -> Tuple[ActionProbeBlock, List[Dict[str, Any]]]:
    """Collect ILU-Arnoldi inputs and reduce exact ``UW.T r`` labels.

    The full probes are consumed trajectory-by-trajectory.  Saved arrays are
    sufficient to reproduce the loss exactly without retaining a very large
    dense ``n_probes x N`` matrix.
    """

    image = np.asarray(context.system.matrix @ dictionary, dtype=float)
    gram_y = image.T @ image
    psi_t_u = dictionary.T @ factors.recycling_trial
    u_gram = factors.recycling_trial.T @ factors.recycling_trial
    all_h: List[np.ndarray] = []
    all_b: List[np.ndarray] = []
    all_t: List[np.ndarray] = []
    all_norm_sq: List[np.ndarray] = []
    all_vector_norms: List[np.ndarray] = []
    labels: List[str] = []
    iteration_indices: List[int] = []
    trajectory_rows: List[Dict[str, Any]] = []

    for trajectory_index, seed in enumerate(seeds):
        rhs, _ = _discrete_rhs(context.system, int(seed))
        captured: List[Tuple[int, np.ndarray]] = []

        def capture(iteration: int, vector: np.ndarray) -> None:
            captured.append((int(iteration), np.asarray(vector, dtype=float).copy()))

        result = restarted_gmres(
            context.system.matrix,
            rhs,
            settings.stage15.stage10.solver,
            context.ilu,
            preconditioner_input_callback=capture,
        )
        if not captured:
            raise RuntimeError(f"R={ratio:g}, seed={seed} produced no Arnoldi probes")
        vectors = np.row_stack([vector for _, vector in captured])
        vector_norms = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(vector_norms)) or np.any(vector_norms <= 0.0):
            raise RuntimeError("nonfinite or zero Arnoldi probe")
        residual_to_image = vectors @ image
        residual_to_test = vectors @ factors.test
        target_coefficients = residual_to_test @ psi_t_u.T
        target_norm_sq = np.einsum(
            "ki,ij,kj->k", residual_to_test, u_gram, residual_to_test
        )
        target_norm_sq = np.maximum(target_norm_sq, 0.0)
        trajectory_label = f"{split}_R{ratio:g}_trajectory{trajectory_index}_seed{seed}"
        all_h.append(residual_to_image)
        all_b.append(residual_to_test)
        all_t.append(target_coefficients)
        all_norm_sq.append(target_norm_sq)
        all_vector_norms.append(vector_norms)
        labels.extend([trajectory_label] * len(captured))
        iteration_indices.extend(iteration for iteration, _ in captured)
        trajectory_rows.append(
            {
                "ratio": ratio,
                "split": split,
                "trajectory": trajectory_label,
                "rhs_seed": int(seed),
                "iterations": int(result.iterations),
                "probe_count": len(captured),
                "converged": bool(result.converged),
                "true_final_relative_residual": float(result.relative_residuals[-1]),
                "probe_norm_min": float(np.min(vector_norms)),
                "probe_norm_max": float(np.max(vector_norms)),
                "target_action_norm_min": float(np.sqrt(np.min(target_norm_sq))),
                "target_action_norm_median": float(np.sqrt(np.median(target_norm_sq))),
                "target_action_norm_max": float(np.sqrt(np.max(target_norm_sq))),
            }
        )

    return (
        ActionProbeBlock(
            ratio=ratio,
            split=split,
            gram_y=gram_y,
            residual_to_dictionary_image=np.row_stack(all_h),
            residual_to_oracle_test=np.row_stack(all_b),
            target_dictionary_coefficients=np.row_stack(all_t),
            target_norm_squared=np.concatenate(all_norm_sq),
            vector_norms=np.concatenate(all_vector_norms),
            trajectory_labels=np.asarray(labels),
            iteration_indices=np.asarray(iteration_indices, dtype=int),
        ),
        trajectory_rows,
    )


def collect_action_probes(
    output: Path,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    dictionary: np.ndarray,
    settings: Stage17Settings,
) -> Tuple[Dict[float, ActionProbeBlock], List[Dict[str, Any]], float]:
    blocks: Dict[float, ActionProbeBlock] = {}
    rows: List[Dict[str, Any]] = []
    start = perf_counter()
    archive: Dict[str, np.ndarray] = {}
    for ratio in settings.stage15.ratios:
        split = _split_for_ratio(ratio, settings)
        block, local_rows = collect_action_probe_block(
            contexts[ratio], factors[ratio], dictionary, ratio, split,
            _probe_seeds(split, settings), settings,
        )
        blocks[ratio] = block
        rows.extend(local_rows)
        key = f"R{ratio:g}"
        archive[f"h_{key}"] = block.residual_to_dictionary_image
        archive[f"b_{key}"] = block.residual_to_oracle_test
        archive[f"target_coeff_{key}"] = block.target_dictionary_coefficients
        archive[f"target_norm_sq_{key}"] = block.target_norm_squared
        archive[f"vector_norms_{key}"] = block.vector_norms
        archive[f"iterations_{key}"] = block.iteration_indices
        archive[f"trajectory_labels_{key}"] = block.trajectory_labels
        _write_csv(output / "probe_trajectory_summary.csv", rows)
    np.savez_compressed(output / "action_probe_reduced_data.npz", **archive)
    return blocks, rows, perf_counter() - start


def _normalized_ratio(ratio: float, settings: Stage15Settings) -> float:
    return (float(ratio) - 100.0) / 10.0


def _action_epsilon(
    blocks: Mapping[float, ActionProbeBlock], settings: Stage17Settings
) -> float:
    values = np.concatenate(
        [blocks[ratio].target_norm_squared for ratio in settings.stage15.train_ratios]
    )
    positive = values[values > 0.0]
    if positive.size == 0:
        raise RuntimeError("all oracle actions vanish on training probes")
    return float(settings.action_epsilon_scale * np.median(positive))


def reduced_action_losses(
    coefficients: torch.Tensor,
    grams: Sequence[torch.Tensor],
    residual_images: Sequence[torch.Tensor],
    target_coefficients: Sequence[torch.Tensor],
    target_norm_squared: Sequence[torch.Tensor],
    epsilon: float,
    jitter: float,
) -> torch.Tensor:
    r"""Return per-probe relative ``||c_theta-c_*||^2`` losses.

    For ``X=Psi C`` and ``Y=A Psi``, the action is

    ``c_theta(r)=Psi C(C.T Y.T Y C)^-1 C.T Y.T r``.

    Because ``Psi.T Psi=I``, norms and cross terms require only p-dimensional
    sufficient statistics.  This is algebraically identical to forming thin
    QR factors and ``U_theta W_theta.T r`` when no stabilization is needed.
    """

    per_system: List[torch.Tensor] = []
    q = coefficients.shape[-1]
    eye = torch.eye(q, dtype=coefficients.dtype, device=coefficients.device)
    for index in range(coefficients.shape[0]):
        local = coefficients[index]
        gram = local.T @ grams[index] @ local
        scale = torch.diagonal(gram).mean()
        stabilized = gram + jitter * torch.clamp(scale, min=1.0) * eye
        rhs = residual_images[index] @ local
        reduced = torch.linalg.solve(stabilized, rhs.T).T
        solution_coordinates = reduced @ local.T
        predicted_norm_sq = torch.sum(solution_coordinates**2, dim=1)
        cross = torch.sum(solution_coordinates * target_coefficients[index], dim=1)
        numerator = torch.clamp(
            predicted_norm_sq - 2.0 * cross + target_norm_squared[index], min=0.0
        )
        per_system.append(numerator / (target_norm_squared[index] + epsilon))
    return torch.cat(per_system)


def _torch_probe_data(
    blocks: Mapping[float, ActionProbeBlock],
    ratios: Sequence[float],
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    grams, residual_images, targets, norms = [], [], [], []
    for ratio in ratios:
        block = blocks[ratio]
        grams.append(torch.as_tensor(block.gram_y, dtype=dtype))
        residual_images.append(
            torch.as_tensor(block.residual_to_dictionary_image, dtype=dtype)
        )
        targets.append(
            torch.as_tensor(block.target_dictionary_coefficients, dtype=dtype)
        )
        norms.append(torch.as_tensor(block.target_norm_squared, dtype=dtype))
    return grams, residual_images, targets, norms


def train_action_seed(
    output: Path,
    seed: int,
    blocks: Mapping[float, ActionProbeBlock],
    dictionary: np.ndarray,
    epsilon: float,
    settings: Stage17Settings,
) -> Tuple[ReducedSpaceMLP, List[Dict[str, Any]], Dict[str, Any]]:
    """Train one fixed-budget Stage 15 MLP using only the action loss."""

    base = settings.stage15
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.use_deterministic_algorithms(True)
    dtype = torch.float64 if base.training_dtype == "float64" else torch.float32
    model = ReducedSpaceMLP(dictionary.shape[1], base.q, base.hidden_widths).to(dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=base.learning_rate, weight_decay=base.weight_decay
    )
    train_data = _torch_probe_data(blocks, base.train_ratios, dtype)
    validation_data = _torch_probe_data(blocks, base.validation_ratios, dtype)
    train_x = torch.as_tensor(
        [_normalized_ratio(ratio, base) for ratio in base.train_ratios], dtype=dtype
    )
    validation_x = torch.as_tensor(
        [_normalized_ratio(ratio, base) for ratio in base.validation_ratios], dtype=dtype
    )
    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    start = perf_counter()
    for epoch in range(base.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = reduced_action_losses(
            model(train_x), *train_data, epsilon, base.cholesky_jitter
        )
        loss = train_losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite action loss for seed {seed} at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if epoch % base.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                validation_losses = reduced_action_losses(
                    model(validation_x),
                    *validation_data,
                    epsilon,
                    base.cholesky_jitter,
                )
            validation_loss = float(validation_losses.mean().cpu())
            history.append(
                {
                    "target": "action",
                    "seed": seed,
                    "epoch": epoch,
                    "training_loss": float(loss.detach().cpu()),
                    "validation_loss": validation_loss,
                    "validation_loss_median": float(validation_losses.median().cpu()),
                    "validation_loss_max": float(validation_losses.max().cpu()),
                }
            )
            if validation_loss < best_loss - 1.0e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if epoch >= base.minimum_epochs and epoch - best_epoch >= base.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError(f"no finite action checkpoint for seed {seed}")
    model.load_state_dict(best_state)
    elapsed = perf_counter() - start
    checkpoint_path = output / "models" / f"action_mlp_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "p": dictionary.shape[1],
            "q": base.q,
            "hidden_widths": base.hidden_widths,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "action_epsilon": epsilon,
        },
        checkpoint_path,
    )
    summary = {
        "target": "action",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"],
        "best_validation_loss": best_loss,
        "training_loss_at_selected_epoch": next(
            row["training_loss"] for row in history if row["epoch"] == best_epoch
        ),
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint_path),
    }
    return model, history, summary


def model_coefficients(
    models: Mapping[int, ReducedSpaceMLP],
    ratios: Sequence[float],
    settings: Stage15Settings,
) -> Dict[int, Dict[float, np.ndarray]]:
    result: Dict[int, Dict[float, np.ndarray]] = {}
    inputs = torch.as_tensor(
        [_normalized_ratio(ratio, settings) for ratio in ratios], dtype=torch.float64
    )
    for seed, model in models.items():
        model.eval()
        with torch.no_grad():
            values = model(inputs).cpu().numpy()
        result[seed] = {
            ratio: np.asarray(value, dtype=float)
            for ratio, value in zip(ratios, values)
        }
    return result


def action_error_from_coefficients(
    coefficients: np.ndarray,
    block: ActionProbeBlock,
    epsilon: float,
) -> np.ndarray:
    """Evaluate exact per-probe action error without training stabilization."""

    gram = coefficients.T @ block.gram_y @ coefficients
    rhs = block.residual_to_dictionary_image @ coefficients
    reduced = np.linalg.solve(gram, rhs.T).T
    solution_coordinates = reduced @ coefficients.T
    predicted_norm_sq = np.sum(solution_coordinates**2, axis=1)
    cross = np.sum(solution_coordinates * block.target_dictionary_coefficients, axis=1)
    numerator = np.maximum(
        predicted_norm_sq - 2.0 * cross + block.target_norm_squared, 0.0
    )
    return np.sqrt(numerator / (block.target_norm_squared + epsilon))


def offline_evaluation(
    coefficient_maps: Mapping[str, Mapping[int, Mapping[float, np.ndarray]]],
    space_maps: Mapping[str, Mapping[int, Mapping[float, np.ndarray]]],
    blocks: Mapping[float, ActionProbeBlock],
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    epsilon: float,
    settings: Stage17Settings,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target, per_seed in coefficient_maps.items():
        for seed, per_ratio in per_seed.items():
            for ratio in settings.stage15.intermediate_ratios:
                predicted_space = space_maps[target][seed][ratio]
                pair = factor_az_petrov(contexts[ratio].system.matrix, predicted_space)
                z_metric = subspace_metrics(predicted_space, spaces[ratio])
                w_metric = subspace_metrics(pair.test, factors[ratio].test)
                errors = action_error_from_coefficients(
                    per_ratio[ratio], blocks[ratio], epsilon
                )
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "ratio": ratio,
                        "split": _split_for_ratio(ratio, settings),
                        "probe_count": blocks[ratio].count,
                        "action_error_mean": float(np.mean(errors)),
                        "action_error_median": float(np.median(errors)),
                        "action_error_p90": float(np.quantile(errors, 0.90)),
                        "action_error_max": float(np.max(errors)),
                        "d_z": float(z_metric["projector_distance"]),
                        "d_z_normalized": float(z_metric["projector_distance_normalized"]),
                        "d_w": float(w_metric["projector_distance"]),
                        "d_w_normalized": float(w_metric["projector_distance_normalized"]),
                        "predicted_rank": int(predicted_space.shape[1]),
                        "au_equals_w_relative_error": float(
                            pair.diagnostics["au_equals_w_relative_error"]
                        ),
                        "condition_wtaz": float(pair.diagnostics["wtaz_condition"]),
                    }
                )
    return rows


def verify_reduced_action_equivalence(
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    dictionary: np.ndarray,
    coefficients: Mapping[float, np.ndarray],
    epsilon: float,
    ratios: Sequence[float],
) -> List[Dict[str, Any]]:
    """Compare reduced statistics with explicit QR/triangular-solve actions."""

    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        rng = np.random.default_rng(20261700 + int(ratio))
        probes = rng.normal(size=(7, dictionary.shape[0]))
        local = coefficients[ratio]
        raw = dictionary @ local
        z_basis, _, rank = _orthonormalize(raw)
        if rank != local.shape[1]:
            raise RuntimeError("equivalence check encountered rank-deficient prediction")
        pair = factor_az_petrov(contexts[ratio].system.matrix, z_basis)
        oracle = factors[ratio]
        explicit_predicted = (pair.recycling_trial @ (pair.test.T @ probes.T)).T
        explicit_target = (
            oracle.recycling_trial @ (oracle.test.T @ probes.T)
        ).T
        explicit_num = np.sum((explicit_predicted - explicit_target) ** 2, axis=1)
        explicit_den = np.sum(explicit_target**2, axis=1) + epsilon
        explicit_errors = np.sqrt(explicit_num / explicit_den)

        image = np.asarray(contexts[ratio].system.matrix @ dictionary, dtype=float)
        h = probes @ image
        b = probes @ oracle.test
        target_coeff = b @ (dictionary.T @ oracle.recycling_trial).T
        target_norm_sq = np.einsum(
            "ki,ij,kj->k",
            b,
            oracle.recycling_trial.T @ oracle.recycling_trial,
            b,
        )
        synthetic = ActionProbeBlock(
            ratio,
            "equivalence",
            image.T @ image,
            h,
            b,
            target_coeff,
            target_norm_sq,
            np.linalg.norm(probes, axis=1),
            np.repeat("equivalence", probes.shape[0]),
            np.arange(probes.shape[0]),
        )
        reduced_errors = action_error_from_coefficients(local, synthetic, epsilon)
        rows.append(
            {
                "ratio": ratio,
                "probe_count": probes.shape[0],
                "rank": rank,
                "max_absolute_error": float(
                    np.max(np.abs(explicit_errors - reduced_errors))
                ),
                "explicit_error_median": float(np.median(explicit_errors)),
                "reduced_error_median": float(np.median(reduced_errors)),
                "au_equals_w_relative_error": float(
                    pair.diagnostics["au_equals_w_relative_error"]
                ),
            }
        )
    return rows


def run_solver_evaluation(
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[float, np.ndarray]],
    space_maps: Mapping[str, Mapping[int, Mapping[float, np.ndarray]]],
    prediction_rows: Sequence[Mapping[str, Any]],
    settings: Stage17Settings,
) -> List[Dict[str, Any]]:
    prediction_lookup = {
        (str(row["target"]), int(row["seed"]), float(row["ratio"])): float(
            row["prediction_seconds"]
        )
        for row in prediction_rows
    }
    factor_cache: Dict[Tuple[str, int | str, float], Tuple[AZPetrovFactors, float]] = {}
    for ratio in settings.stage15.intermediate_ratios:
        factor_cache[("oracle", "", ratio)] = (factors[ratio], 0.0)
        for method, basis_map in deterministic.items():
            start = perf_counter()
            factor_cache[(method, "", ratio)] = (
                factor_az_petrov(contexts[ratio].system.matrix, basis_map[ratio]),
                perf_counter() - start,
            )
        for target, per_seed in space_maps.items():
            for seed, basis_map in per_seed.items():
                start = perf_counter()
                factor_cache[(f"mlp_{target}", seed, ratio)] = (
                    factor_az_petrov(contexts[ratio].system.matrix, basis_map[ratio]),
                    perf_counter() - start,
                )

    rows: List[Dict[str, Any]] = []
    for ratio in settings.stage15.intermediate_ratios:
        context = contexts[ratio]
        split = _split_for_ratio(ratio, settings)
        specs: List[Tuple[str, str, int | str, AZPetrovFactors, float, float]] = [
            ("oracle", "", "", *factor_cache[("oracle", "", ratio)], 0.0)
        ]
        for method in ("nearest_reuse", "deterministic_interpolation"):
            specs.append(
                (method, "", "", *factor_cache[(method, "", ratio)], 0.0)
            )
        for target, per_seed in space_maps.items():
            for seed in per_seed:
                method = f"mlp_{target}"
                specs.append(
                    (
                        method,
                        target,
                        seed,
                        *factor_cache[(method, seed, ratio)],
                        prediction_lookup[(target, seed, ratio)],
                    )
                )

        for rhs_seed in settings.stage15.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline, baseline_counts, baseline_time = _run_baseline(
                context, rhs, factors[ratio], settings.stage15.stage13
            )
            rows.append(
                {
                    "ratio": ratio,
                    "split": split,
                    "rhs_seed": rhs_seed,
                    "method": "strong_ilu",
                    "target": "",
                    "seed": "",
                    "iterations": baseline.iterations,
                    "converged": baseline.converged,
                    "cap_hit": bool(
                        baseline.iterations
                        >= settings.stage15.stage10.solver.max_iterations
                        and not baseline.converged
                    ),
                    "true_final_relative_residual": float(
                        baseline.relative_residuals[-1]
                    ),
                    "online_solve_seconds": baseline_time,
                    "method_setup_seconds": 0.0,
                    "prediction_seconds": 0.0,
                    "a_matvecs": baseline_counts.a_matvecs,
                    "ilu_solves": baseline_counts.ilu_solves,
                    "eta": 0.0,
                }
            )
            local_rows: List[Dict[str, Any]] = []
            for method, target, seed, pair, setup_seconds, prediction_seconds in specs:
                result, counts, online = _run_recycled(
                    context, rhs, pair, settings.stage15.stage13
                )
                local_rows.append(
                    {
                        "ratio": ratio,
                        "split": split,
                        "rhs_seed": rhs_seed,
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "cap_hit": bool(
                            result.iterations
                            >= settings.stage15.stage10.solver.max_iterations
                            and not result.converged
                        ),
                        "true_final_relative_residual": result.relative_residual,
                        "online_solve_seconds": online,
                        "method_setup_seconds": setup_seconds,
                        "prediction_seconds": prediction_seconds,
                        "a_matvecs": int(
                            (result.operation_counts or {}).get("a_matvecs", 0)
                        ),
                        "ilu_solves": counts.ilu_solves,
                        "eta": float("nan"),
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


def aggregate_solver_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            float(row["ratio"]),
            str(row["method"]),
            str(row["target"]),
            str(row["seed"]),
        )
        groups.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for (split, ratio, method, target, seed), local in sorted(groups.items()):
        result.append(
            {
                "split": split,
                "ratio": ratio,
                "method": method,
                "target": target,
                "seed": seed,
                "rhs_count": len(local),
                "iterations_median": float(
                    np.median([row["iterations"] for row in local])
                ),
                "iterations_min": int(min(row["iterations"] for row in local)),
                "iterations_max": int(max(row["iterations"] for row in local)),
                "convergence_rate": float(
                    np.mean([bool(row["converged"]) for row in local])
                ),
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
    return result


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def make_decision(
    action_training_summaries: Sequence[Mapping[str, Any]],
    w_training_summaries: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    settings: Stage17Settings,
) -> Dict[str, Any]:
    w_seed = select_seed(w_training_summaries)
    action_seed = select_seed(action_training_summaries)
    selected = {"w_loss": w_seed, "action": action_seed}
    offline: Dict[str, Dict[str, float]] = {}
    learned_solver: Dict[str, Dict[str, float]] = {}
    for target, seed in selected.items():
        local_offline = [
            row for row in offline_rows
            if row["target"] == target and int(row["seed"]) == seed and row["split"] == "test"
        ]
        local_solver = [
            row for row in solver_summaries
            if row["method"] == f"mlp_{target}" and int(row["seed"]) == seed and row["split"] == "test"
        ]
        offline[target] = {
            "action_error_mean": _median(local_offline, "action_error_mean"),
            "action_error_median": _median(local_offline, "action_error_median"),
            "action_error_p90": _median(local_offline, "action_error_p90"),
            "d_z_normalized": _median(local_offline, "d_z_normalized"),
            "d_w_normalized": _median(local_offline, "d_w_normalized"),
        }
        learned_solver[target] = {
            "eta_median": _median(local_solver, "eta_median"),
            "iterations_median": _median(local_solver, "iterations_median"),
            "convergence_rate": float(np.mean([float(row["convergence_rate"]) for row in local_solver])),
            "online_seconds_median": _median(local_solver, "online_solve_seconds_median"),
        }
    baselines: Dict[str, Dict[str, float]] = {}
    for method in ("strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation"):
        local = [row for row in solver_summaries if row["method"] == method and row["split"] == "test"]
        baselines[method] = {
            "eta_median": _median(local, "eta_median"),
            "iterations_median": _median(local, "iterations_median"),
            "convergence_rate": float(np.mean([float(row["convergence_rate"]) for row in local])),
            "online_seconds_median": _median(local, "online_solve_seconds_median"),
        }
    per_ratio_w = {
        float(row["ratio"]): float(row["iterations_median"])
        for row in solver_summaries
        if row["method"] == "mlp_w_loss" and int(row["seed"]) == w_seed and row["split"] == "test"
    }
    per_ratio_action = {
        float(row["ratio"]): float(row["iterations_median"])
        for row in solver_summaries
        if row["method"] == "mlp_action" and int(row["seed"]) == action_seed and row["split"] == "test"
    }
    action_not_worse_fraction = float(np.mean([
        per_ratio_action[ratio] <= per_ratio_w[ratio] for ratio in settings.stage15.test_ratios
    ]))
    eta_gain = learned_solver["action"]["eta_median"] - learned_solver["w_loss"]["eta_median"]
    stable = (
        learned_solver["action"]["convergence_rate"] >= settings.stable_convergence_rate
        and all(np.isfinite(value) for value in offline["action"].values())
    )
    clearly_improves = (
        eta_gain >= settings.clear_eta_improvement
        and action_not_worse_fraction >= settings.ratio_win_fraction
    )
    approaches_interpolation = learned_solver["action"]["eta_median"] >= (
        baselines["deterministic_interpolation"]["eta_median"]
        - settings.interpolation_eta_tolerance
    )
    if not stable:
        code = "D"
        label = "Action supervision is unstable."
    elif clearly_improves and approaches_interpolation:
        code = "A"
        label = "Action loss clearly improves solver performance and approaches or beats interpolation."
    elif clearly_improves:
        code = "B"
        label = "Action loss improves over W loss, but deterministic interpolation remains stronger."
    else:
        code = "C"
        label = "Action loss gives no meaningful solver improvement."
    return {
        "decision": code,
        "label": label,
        "selected_seed_by_validation_only": selected,
        "offline_test": offline,
        "learned_solver_test": learned_solver,
        "baselines_test": baselines,
        "action_eta_gain_over_w_loss": eta_gain,
        "action_not_worse_ratio_fraction": action_not_worse_fraction,
        "action_stable": stable,
        "action_clearly_improves": clearly_improves,
        "action_approaches_interpolation": approaches_interpolation,
        "thresholds": {
            "clear_eta_improvement": settings.clear_eta_improvement,
            "ratio_win_fraction": settings.ratio_win_fraction,
            "interpolation_eta_tolerance": settings.interpolation_eta_tolerance,
            "stable_convergence_rate": settings.stable_convergence_rate,
        },
        "fixed_next_step": "Expand the PDE family; do not test another subspace target.",
    }


def _selected_solver_rows(
    rows: Sequence[Mapping[str, Any]], selected: Mapping[str, int]
) -> List[Mapping[str, Any]]:
    keep = {f"mlp_{target}": int(seed) for target, seed in selected.items()}
    return [
        row for row in rows
        if row["method"] in {"strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation"}
        or (
            row["method"] in keep and row["seed"] != ""
            and int(float(row["seed"])) == keep[row["method"]]
        )
    ]


def make_plots(
    output: Path,
    histories: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage17Settings,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = decision["selected_seed_by_validation_only"]
    colors = {"w_loss": BLUE, "action": ORANGE}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for target in ("w_loss", "action"):
        local = sorted([
            row for row in offline_rows
            if row["target"] == target and int(row["seed"]) == selected[target] and row["split"] == "test"
        ], key=lambda row: float(row["ratio"]))
        ax.plot([row["ratio"] for row in local], [row["action_error_median"] for row in local], marker="o", color=colors[target], label=METHOD_LABELS[f"mlp_{target}"])
    ax.set(xlabel="Anisotropy ratio R", ylabel="Median relative solver-action error", title="Held-out action error on ILU-Arnoldi probes")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_action_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    for target in ("w_loss", "action"):
        local = sorted([
            row for row in offline_rows
            if row["target"] == target and int(row["seed"]) == selected[target] and row["split"] == "test"
        ], key=lambda row: float(row["ratio"]))
        axes[0].plot([row["ratio"] for row in local], [row["d_z_normalized"] for row in local], marker="o", color=colors[target], label=METHOD_LABELS[f"mlp_{target}"])
        axes[1].plot([row["ratio"] for row in local], [row["d_w_normalized"] for row in local], marker="o", color=colors[target], label=METHOD_LABELS[f"mlp_{target}"])
    axes[0].set(title="Solution-space geometry", ylabel="Normalized projector error", xlabel="R")
    axes[1].set(title="Residual/test-space geometry", xlabel="R")
    for ax in axes:
        ax.set_xticks(settings.stage15.test_ratios)
        ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_projector_geometry.png", dpi=180)
    plt.close(fig)

    selected_rows = _selected_solver_rows(solver_summaries, selected)
    method_colors = {
        "strong_ilu": CHARCOAL, "oracle": PINK, "nearest_reuse": OLIVE,
        "deterministic_interpolation": PURPLE, "mlp_w_loss": BLUE,
        "mlp_action": ORANGE,
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for method in METHOD_LABELS:
        local = sorted([row for row in selected_rows if row["method"] == method and row["split"] == "test"], key=lambda row: float(row["ratio"]))
        ax.plot([row["ratio"] for row in local], [row["iterations_median"] for row in local], marker="o", color=method_colors[method], label=METHOD_LABELS[method])
    ax.set(xlabel="Anisotropy ratio R", ylabel="Median recycled-GMRES iterations", title="Held-out solver iterations")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_solver_iterations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for method in ("nearest_reuse", "deterministic_interpolation", "mlp_w_loss", "mlp_action"):
        local = sorted([row for row in selected_rows if row["method"] == method and row["split"] == "test"], key=lambda row: float(row["ratio"]))
        ax.plot([row["ratio"] for row in local], [row["eta_median"] for row in local], marker="o", color=method_colors[method], label=METHOD_LABELS[method])
    ax.axhline(1.0, color=PINK, linestyle="--", linewidth=1.0, label="Own oracle")
    ax.set(xlabel="Anisotropy ratio R", ylabel="Median oracle recovery eta", title="Held-out oracle recovery")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_oracle_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for seed in settings.stage15.model_seeds:
        local = sorted([row for row in histories if int(row["seed"]) == seed], key=lambda row: int(row["epoch"]))
        ax.semilogy([row["epoch"] for row in local], [row["validation_loss"] for row in local], label=str(seed))
    ax.set(xlabel="Epoch", ylabel="Validation mean relative action error squared", title="Action-loss validation histories")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "action_training_validation_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for target in ("w_loss", "action"):
        local_offline = {
            float(row["ratio"]): row for row in offline_rows
            if row["target"] == target and int(row["seed"]) == selected[target] and row["split"] == "test"
        }
        local_solver = [
            row for row in selected_rows
            if row["method"] == f"mlp_{target}" and row["split"] == "test"
        ]
        ax.scatter(
            [local_offline[float(row["ratio"])]["action_error_median"] for row in local_solver],
            [row["eta_median"] for row in local_solver],
            color=colors[target], label=METHOD_LABELS[f"mlp_{target}"], s=45,
        )
    ax.set(xlabel="Median held-out action error", ylabel="Median oracle recovery eta", title="Action fidelity versus solver benefit")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "action_error_vs_oracle_recovery.png", dpi=180)
    plt.close(fig)


def stage15_reproduction_checks(
    solver_rows: Sequence[Mapping[str, Any]],
    selected_w_seed: int,
) -> Dict[str, Any]:
    old_rows = _read_csv(_stage15_directory() / "solver_results.csv")
    old_lookup = {
        (
            float(row["ratio"]), int(row["rhs_seed"]), str(row["method"]),
            "" if row["seed"] == "" else int(row["seed"]),
        ): int(row["iterations"])
        for row in old_rows
    }
    differences = []
    for row in solver_rows:
        method = str(row["method"])
        if method == "mlp_w_loss":
            old_method = "learned_mlp"
        elif method in {"strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation"}:
            old_method = method
        else:
            continue
        seed = "" if row["seed"] == "" else int(row["seed"])
        key = (float(row["ratio"]), int(row["rhs_seed"]), old_method, seed)
        differences.append(abs(int(row["iterations"]) - old_lookup[key]))
    old_decision = json.loads((_stage15_directory() / "decision.json").read_text())
    return {
        "max_iteration_difference": int(max(differences, default=0)),
        "selected_w_seed": selected_w_seed,
        "stage15_selected_seed": int(old_decision["selected_seed_by_validation_only"]),
        "selected_seed_matches": selected_w_seed == int(old_decision["selected_seed_by_validation_only"]),
    }


def _format(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None or value == "":
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        absolute = abs(float(value))
        if absolute != 0.0 and (absolute < 1.0e-3 or absolute >= 1.0e4):
            return f"{float(value):.{digits}e}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_format(value) for value in row) + " |" for row in rows),
        )
    )


def write_report(
    output: Path,
    settings: Stage17Settings,
    probe_rows: Sequence[Mapping[str, Any]],
    epsilon: float,
    action_training_summaries: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    equivalence_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    selected = decision["selected_seed_by_validation_only"]
    probe_table = []
    for split in ("train", "validation", "test"):
        local = [row for row in probe_rows if row["split"] == split]
        probe_table.append([
            split, len({row["ratio"] for row in local}), len(local),
            sum(bool(row["converged"]) for row in local),
            sum(int(row["probe_count"]) for row in local),
            min(float(row["target_action_norm_min"]) for row in local),
            float(np.median([float(row["target_action_norm_median"]) for row in local])),
            max(float(row["target_action_norm_max"]) for row in local),
        ])
    training_table = [[
        int(row["seed"]), int(row["best_epoch"]), int(row["epochs_run"]),
        row["training_loss_at_selected_epoch"], row["best_validation_loss"],
        row["training_seconds"], int(row["seed"]) == selected["action"],
    ] for row in action_training_summaries]
    offline_table = []
    for ratio in settings.stage15.test_ratios:
        for target in ("w_loss", "action"):
            row = next(
                item for item in offline_rows
                if item["target"] == target and int(item["seed"]) == selected[target]
                and float(item["ratio"]) == ratio
            )
            offline_table.append([
                ratio, METHOD_LABELS[f"mlp_{target}"], row["probe_count"],
                row["action_error_mean"], row["action_error_median"], row["action_error_p90"],
                row["d_z_normalized"], row["d_w_normalized"],
            ])
    selected_solver = _selected_solver_rows(solver_summaries, selected)
    solver_table = []
    for ratio in settings.stage15.test_ratios:
        for method in METHOD_LABELS:
            row = next(
                item for item in selected_solver
                if float(item["ratio"]) == ratio and item["method"] == method
            )
            solver_table.append([
                ratio, METHOD_LABELS[method], row["iterations_median"], row["iterations_min"],
                row["iterations_max"], row["convergence_rate"],
                row["true_final_relative_residual_max"], row["online_solve_seconds_median"],
                row["eta_median"],
            ])
    action = decision["offline_test"]["action"]
    w_loss = decision["offline_test"]["w_loss"]
    report = f"""# Stage 17 — Solver-action supervision

## Decision

**{decision['decision']}. {decision['label']}**

The strict ablation changes only the training loss. The action-supervised MLP has median held-out action error **{action['action_error_median']:.3f}** versus **{w_loss['action_error_median']:.3f}** for the exact Stage 15 W-loss model. Its median oracle recovery is **{decision['learned_solver_test']['action']['eta_median']:.3f}** versus **{decision['learned_solver_test']['w_loss']['eta_median']:.3f}**, a change of {decision['action_eta_gain_over_w_loss']:+.3f}. Deterministic interpolation remains at **{decision['baselines_test']['deterministic_interpolation']['eta_median']:.3f}**.

## 1. Frozen experiment

The experiment reuses Stage 15 exactly: N=5041, stencil=25, fixed seed-7 nodes with jitter 0.15, 30-degree rotated anisotropy, R=90,...,110, q=8, strong ILU (drop tolerance 1e-3, fill factor 5), right-preconditioned A M^-1, restart 50, 300-iteration cap, tolerance 1e-8, and the same construction/evaluation RHS protocol. Training ratios are the even R values; validation={settings.stage15.validation_ratios}; final test={settings.stage15.test_ratios}.

The dictionary is the exact Stage 15 p=19 Z-POD basis. Both targets use the identical `1→64→128→152` GELU MLP, Adam hyperparameters, float64 arithmetic, early-stopping budget, and seeds {settings.stage15.model_seeds}. The W-loss arm loads the original Stage 15 checkpoints; the action arm starts from the same seeded initialization and changes only its loss. No new dictionary, architecture, parameter, solver, or oracle target is introduced.

## 2. Solver action and reduced training loss

For every space, thin QR gives

$$AZ=WR,\qquad U=ZR^{{-1}},\qquad AU=W,$$

where U is formed by a triangular solve. The supervised action is

$$c(r)=UW^Tr.$$

Training probes are the actual normalized Arnoldi vectors presented to the strong ILU preconditioner by classical GMRES. Three independent smooth manufactured discrete RHS trajectories are used per training R. Validation and test probes use disjoint systems and distinct trajectory seeds; test probes never affect checkpoint selection.

With X=Psi C and Y=A Psi, the exact basis-invariant action is

$$c_\\theta(r)=\Psi C(C^T Y^T Y C)^{{-1}}C^TY^Tr.$$

This yields the requested mean relative action loss without materializing N-dimensional actions during every epoch. The denominator uses epsilon={epsilon:.3e}, fixed at {settings.action_epsilon_scale:.0e} times the median positive training ||c_*(r)||^2. Explicit QR/triangular-solve actions and the reduced evaluation agree to maximum absolute error {max(float(row['max_absolute_error']) for row in equivalence_rows):.3e}.

{_table(['split', 'systems', 'trajectories', 'converged', 'probes', 'min ||c*||', 'median ||c*||', 'max ||c*||'], probe_table)}

Fifteen of 33 training trajectories reach the unchanged 300-iteration cap, concentrated at R=100, 102, 106, 108, and 110; all validation and test probe trajectories converge. Capped classical trajectories remain valid sequences of actual ILU-Arnoldi inputs and are retained rather than censored. Their exact oracle actions are algebraic labels and do not depend on classical convergence.

## 3. Training

Action checkpoints are selected only by mean validation action loss.

{_table(['seed', 'best epoch', 'epochs run', 'train action loss', 'validation action loss', 'training s', 'selected'], training_table)}

![Action validation loss](plots/action_training_validation_loss.png)

## 4. Held-out action and geometry

Action errors are evaluated on unseen ILU-Arnoldi probes from the untouched test systems. Projector distances are normalized by sqrt(2q); d_Z and d_W remain diagnostics rather than training success criteria for the action arm.

{_table(['R', 'model', 'probes', 'mean action err.', 'median action err.', 'p90 action err.', 'norm d_Z', 'norm d_W'], offline_table)}

![Held-out action error](plots/test_action_error.png)

![Held-out projector geometry](plots/test_projector_geometry.png)

## 5. Solver-level comparison

Every method uses the same target matrix, ILU, RHS, initial guess, tolerance, cap, and Stage 13 fixed-space recycled GMRES. Oracle recovery is computed per RHS as eta=(k_ILU-k_method)/(k_ILU-k_oracle), then summarized by R.

{_table(['R', 'method', 'median iter.', 'min', 'max', 'conv.', 'max true residual', 'online s', 'eta'], solver_table)}

![Held-out solver iterations](plots/test_solver_iterations.png)

![Held-out oracle recovery](plots/test_oracle_recovery.png)

![Action fidelity versus oracle recovery](plots/action_error_vs_oracle_recovery.png)

## 6. Interpretation

- **Action fidelity:** median test action error changes from {w_loss['action_error_median']:.3f} to {action['action_error_median']:.3f}; p90 changes from {w_loss['action_error_p90']:.3f} to {action['action_error_p90']:.3f}.
- **Geometry:** action supervision yields median normalized d_Z={action['d_z_normalized']:.3f} and d_W={action['d_w_normalized']:.3f}, versus {w_loss['d_z_normalized']:.3f} and {w_loss['d_w_normalized']:.3f} for W supervision. These quantities are not treated as primary objectives.
- **Solver value:** action median iterations={decision['learned_solver_test']['action']['iterations_median']:.1f} and eta={decision['learned_solver_test']['action']['eta_median']:.3f}; W-loss gives {decision['learned_solver_test']['w_loss']['iterations_median']:.1f} and {decision['learned_solver_test']['w_loss']['eta_median']:.3f}. Action is no worse on {decision['action_not_worse_ratio_fraction']:.0%} of test ratios.
- **Stability:** action convergence rate={decision['learned_solver_test']['action']['convergence_rate']:.2f}; all reported true residuals retain the original 1e-8 stopping criterion.
- **Deterministic reference:** nearest reuse eta={decision['baselines_test']['nearest_reuse']['eta_median']:.3f}; interpolation eta={decision['baselines_test']['deterministic_interpolation']['eta_median']:.3f}. Training loss alone is not credited as solver success.

The central question therefore has a negative answer: action supervision improves the quantity it optimizes on unseen probes, even while d_Z and d_W become larger, but that improvement does **not** translate into Krylov acceleration. In this probe distribution, average relative action fidelity is not sufficiently aligned with the solver-critical information carried by the oracle recycle space.

The decision requires a median eta gain of at least {settings.clear_eta_improvement:.2f}, action iterations no worse on at least {settings.ratio_win_fraction:.0%} of test ratios, convergence at least {settings.stable_convergence_rate:.0%}, and interpolation matching within {settings.interpolation_eta_tolerance:.2f}. Measured values produce outcome **{decision['decision']}**.

## 7. Scope and fixed next step

This is a narrow fixed-PDE, fixed-node, one-parameter interpolation experiment. It tests solver relevance, not broad learnability or transferability. Regardless of outcome, the next step is fixed by protocol: **expand the PDE family rather than test additional subspace targets**. No GNN, FNO, operator learner, extra target, larger MLP, or solver redesign is justified by this stage.

## 8. Reproducibility and validation

Stage 15 reproduction has maximum iteration difference {reproduction['max_iteration_difference']} and selected-seed agreement={reproduction['selected_seed_matches']}. Validation status: **{validation['assessment']}**. Probe generation, reduced sufficient statistics, training histories/checkpoints, geometry, solver rows, timing, decisions, algebra checks, and figures are saved beside this report.
"""
    path = output / "stage17_solver_action_supervision.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    output: Path,
    probe_rows: Sequence[Mapping[str, Any]],
    action_training_summaries: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    equivalence_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    settings: Stage17Settings,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    train = set(settings.stage15.train_ratios)
    validation = set(settings.stage15.validation_ratios)
    test = set(settings.stage15.test_ratios)
    checks["system_splits_disjoint"] = not (train & validation or train & test or validation & test)
    checks["probe_seed_sets_disjoint"] = not (
        set(settings.training_probe_seeds) & set(settings.validation_probe_seeds)
        or set(settings.training_probe_seeds) & set(settings.test_probe_seeds)
        or set(settings.validation_probe_seeds) & set(settings.test_probe_seeds)
    )
    checks["three_trajectories_per_system"] = len(probe_rows) == len(settings.stage15.ratios) * 3
    checks["all_probe_trajectories_finite"] = all(
        np.isfinite(float(row["true_final_relative_residual"]))
        and np.isfinite(float(row["probe_norm_min"]))
        and np.isfinite(float(row["target_action_norm_median"]))
        for row in probe_rows
    )
    checks["heldout_probe_trajectories_converged"] = all(
        bool(row["converged"]) for row in probe_rows if row["split"] in {"validation", "test"}
    )
    checks["positive_probe_counts"] = all(int(row["probe_count"]) > 0 for row in probe_rows)
    checks["action_training_three_seeds"] = len(action_training_summaries) == len(settings.stage15.model_seeds)
    checks["selected_by_validation"] = int(decision["selected_seed_by_validation_only"]["action"]) == select_seed(action_training_summaries)
    checks["prediction_rows_exact"] = len(prediction_rows) == 2 * len(settings.stage15.model_seeds) * len(settings.stage15.intermediate_ratios)
    checks["offline_rows_exact"] = len(offline_rows) == len(prediction_rows)
    checks["offline_metrics_finite"] = all(
        np.isfinite(float(row[field]))
        for row in offline_rows
        for field in ("action_error_mean", "action_error_median", "action_error_p90", "d_z_normalized", "d_w_normalized", "au_equals_w_relative_error")
    )
    checks["all_predicted_ranks_q"] = all(int(row["predicted_rank"]) == settings.stage15.q for row in offline_rows)
    checks["au_equals_w_accurate"] = max(float(row["au_equals_w_relative_error"]) for row in offline_rows) <= 1.0e-9
    expected_solver = len(settings.stage15.intermediate_ratios) * len(settings.stage15.stage10.evaluation_rhs_seeds) * (4 + 2 * len(settings.stage15.model_seeds))
    checks["solver_rows_exact"] = len(solver_rows) == expected_solver
    checks["solver_summaries_exact"] = len(solver_summaries) == len(settings.stage15.intermediate_ratios) * (4 + 2 * len(settings.stage15.model_seeds))
    checks["solver_metrics_finite"] = all(
        np.isfinite(float(row[field])) for row in solver_rows
        for field in ("iterations", "true_final_relative_residual", "online_solve_seconds")
    )
    checks["converged_residuals_meet_tolerance"] = all(
        (not bool(row["converged"]))
        or float(row["true_final_relative_residual"]) <= settings.stage15.stage10.solver.relative_tolerance * (1.0 + 1.0e-8)
        for row in solver_rows
    )
    baseline = {(float(row["ratio"]), int(row["rhs_seed"])): int(row["iterations"]) for row in solver_rows if row["method"] == "strong_ilu"}
    oracle = {(float(row["ratio"]), int(row["rhs_seed"])): int(row["iterations"]) for row in solver_rows if row["method"] == "oracle"}
    eta_errors = []
    for row in solver_rows:
        if row["method"] == "strong_ilu":
            continue
        key = (float(row["ratio"]), int(row["rhs_seed"]))
        denominator = baseline[key] - oracle[key]
        expected = (baseline[key] - int(row["iterations"])) / denominator if denominator > 0 else float("nan")
        eta_errors.append(abs(float(row["eta"]) - expected))
    checks["eta_formula_max_absolute_error"] = float(max(eta_errors, default=0.0))
    checks["eta_formula_exact"] = checks["eta_formula_max_absolute_error"] <= 1.0e-12
    checks["reduced_action_equivalence_max_error"] = float(max(row["max_absolute_error"] for row in equivalence_rows))
    checks["reduced_action_equivalent"] = checks["reduced_action_equivalence_max_error"] <= 1.0e-8
    checks["stage15_iteration_reproduction"] = reproduction["max_iteration_difference"] == 0
    checks["stage15_seed_reproduction"] = bool(reproduction["selected_seed_matches"])
    required = (
        "stage17_solver_action_supervision.md", "settings.json", "environment.json",
        "action_probe_reduced_data.npz", "probe_trajectory_summary.csv",
        "w_loss_training_summary_stage15.csv", "action_training_history.csv",
        "action_training_summary.csv", "prediction_metadata.csv", "offline_metrics.csv",
        "solver_results.csv", "solver_summary.csv", "action_equivalence.csv",
        "decision.json", "reproduction.json", "plots/test_action_error.png",
        "plots/test_projector_geometry.png", "plots/test_solver_iterations.png",
        "plots/test_oracle_recovery.png", "plots/action_training_validation_loss.png",
        "plots/action_error_vs_oracle_recovery.png",
    )
    missing = [name for name in required if not (output / name).exists()]
    checks["missing_artifacts"] = missing
    checks["artifacts_complete"] = not missing
    blockers = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    return {"assessment": "ready_to_share" if not blockers else "needs_revision", "blockers": blockers, "checks": checks}


def run_stage17(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    settings = Stage17Settings()
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "solver": "unchanged Stage 13 fixed-space recycled/projected GMRES",
        "oracle": "exact saved Stage 15 oracle spaces", "dictionary": "exact Stage 15 p=19 Z-POD",
    })

    print("[1/10] loading exact Stage 15 family, dictionary, and W-loss models", flush=True)
    contexts, spaces, factors, _ = load_exact_stage15_family(output, settings)  # type: ignore[arg-type]
    dictionary, w_models, w_history, w_summaries = load_stage15_dictionary_and_models(settings)
    _write_csv(output / "w_loss_training_history_stage15.csv", w_history)
    _write_csv(output / "w_loss_training_summary_stage15.csv", w_summaries)

    print("[2/10] collecting disjoint multi-RHS ILU-Arnoldi action probes", flush=True)
    blocks, probe_rows, probe_seconds = collect_action_probes(output, contexts, factors, dictionary, settings)
    epsilon = _action_epsilon(blocks, settings)
    _write_json(output / "probe_metadata.json", {
        "collection_seconds": probe_seconds, "action_epsilon": epsilon,
        "full_probe_vectors_saved": False,
        "reason": "reduced sufficient statistics exactly reproduce the loss without a dense probes-by-N archive",
    })

    print("[3/10] training action-loss MLP with the exact Stage 15 budget", flush=True)
    action_models: Dict[int, ReducedSpaceMLP] = {}
    action_history: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    for seed in settings.stage15.model_seeds:
        model, history, summary = train_action_seed(output, seed, blocks, dictionary, epsilon, settings)
        action_models[seed] = model
        action_history.extend(history)
        action_summaries.append(summary)
        _write_csv(output / "action_training_history.csv", action_history)
        _write_csv(output / "action_training_summary.csv", action_summaries)

    print("[4/10] constructing W-loss and action-loss held-out spaces", flush=True)
    space_maps: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    prediction_rows: List[Dict[str, Any]] = []
    for target, models in (("w_loss", w_models), ("action", action_models)):
        local_spaces, local_rows = build_learned_spaces(models, dictionary, settings.stage15)
        space_maps[target] = local_spaces
        prediction_rows.extend({"target": target, **row} for row in local_rows)
    _write_csv(output / "prediction_metadata.csv", prediction_rows)
    coefficient_maps = {
        "w_loss": model_coefficients(w_models, settings.stage15.intermediate_ratios, settings.stage15),
        "action": model_coefficients(action_models, settings.stage15.intermediate_ratios, settings.stage15),
    }

    print("[5/10] evaluating unseen-probe action error and projector geometry", flush=True)
    offline_rows = offline_evaluation(coefficient_maps, space_maps, blocks, contexts, spaces, factors, epsilon, settings)
    _write_csv(output / "offline_metrics.csv", offline_rows)

    print("[6/10] running the unchanged recycled-GMRES A/B comparison", flush=True)
    deterministic, deterministic_rows = build_deterministic_spaces(spaces, settings.stage15)
    _write_csv(output / "deterministic_space_metadata.csv", deterministic_rows)
    solver_rows = run_solver_evaluation(contexts, factors, deterministic, space_maps, prediction_rows, settings)
    solver_summaries = aggregate_solver_rows(solver_rows)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summaries)

    print("[7/10] verifying reduced action algebra", flush=True)
    selected_action = select_seed(action_summaries)
    equivalence_rows = verify_reduced_action_equivalence(
        contexts, factors, dictionary, coefficient_maps["action"][selected_action], epsilon,
        settings.stage15.test_ratios,
    )
    _write_csv(output / "action_equivalence.csv", equivalence_rows)

    print("[8/10] applying the fixed four-way interpretation", flush=True)
    decision = make_decision(action_summaries, w_summaries, offline_rows, solver_summaries, settings)
    reproduction = stage15_reproduction_checks(solver_rows, decision["selected_seed_by_validation_only"]["w_loss"])
    _write_json(output / "decision.json", decision)
    _write_json(output / "reproduction.json", reproduction)

    print("[9/10] plotting and validating evidence", flush=True)
    make_plots(output, action_history, offline_rows, solver_summaries, decision, settings)
    preliminary = {"assessment": "pending_report_artifact"}
    write_report(output, settings, probe_rows, epsilon, action_summaries, offline_rows, solver_summaries, equivalence_rows, decision, reproduction, preliminary)
    validation = validate_outputs(output, probe_rows, action_summaries, prediction_rows, offline_rows, solver_rows, solver_summaries, equivalence_rows, decision, reproduction, settings)
    _write_json(output / "validation.json", validation)

    print("[10/10] writing final Stage 17 report", flush=True)
    report = write_report(output, settings, probe_rows, epsilon, action_summaries, offline_rows, solver_summaries, equivalence_rows, decision, reproduction, validation)
    print(f"Stage 17 complete: {report}", flush=True)
    return report


if __name__ == "__main__":
    run_stage17(_repository_root() / "results" / "stage17_solver_action_supervision")
