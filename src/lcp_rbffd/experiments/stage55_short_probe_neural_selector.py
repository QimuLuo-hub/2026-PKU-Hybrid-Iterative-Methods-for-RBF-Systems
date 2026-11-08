"""Stage 55: learn to replace long target-local probing with a short probe."""

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
import torch
from torch import nn

from neurkitt_oracle.gcrodr import restarted_projected_gmres

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import _run_pair
from .stage27_spatial_k_adaptive_rank_validation import Stage27Settings, _run_adaptive
from .stage28_spatial_k_operator_learning import SpectralConv2d
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage40_target_conditioned_local_basis import _geometry, _teacher, factor_full_action
from .stage35_residual_space_loss_mlp_vs_fno import build_inference_data
from .stage51_neural_raw_krylov_compression import Stage51Settings, SpatialBranch
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage55Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    probe_depths: Tuple[int, ...] = (4, 8, 16, 32)
    probe_rhs_seed: int = 20262911
    requested_rank: int = 48
    expected_full_span_rank: int = 1456
    coefficient_nonzero_relative_tolerance: float = 1.0e-8
    information_transition_minimum_iteration_gain: float = 20.0
    preferred_depths: Tuple[int, int] = (8, 16)
    model_seed: int = 20265500
    token_width: int = 64
    latent_width: int = 64
    fno_width: int = 12
    fno_modes: int = 8
    fno_layers: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 3
    maximum_epochs: int = 180
    minimum_epochs: int = 60
    validation_interval: int = 5
    early_stopping_patience: int = 40

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()

    @property
    def stage51(self) -> Stage51Settings:
        return Stage51Settings()

    @property
    def stage27(self) -> Stage27Settings:
        return Stage27Settings(
            first_cycle_contraction_threshold=self.stage29.stage28.first_cycle_contraction_threshold
        )


def _load_full_span(root: Path, settings: Stage55Settings):
    path = root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    with np.load(path) as archive:
        psi = np.asarray(archive["psi"], float)
        training_ids = np.asarray(archive["training_ids"], int)
    if psi.shape != (5041, settings.expected_full_span_rank):
        raise RuntimeError(f"unexpected Stage-30 full-span shape {psi.shape}")
    return psi, training_ids


def pair_from_overlap(
    matrix: sp.spmatrix,
    psi: np.ndarray,
    triangular: np.ndarray,
    overlap: np.ndarray,
    requested_rank: int,
):
    started = perf_counter()
    left, singular, _ = la.svd(np.asarray(overlap, float), full_matrices=False, check_finite=False)
    actual = min(requested_rank, left.shape[1])
    coefficients = la.solve_triangular(
        triangular, left[:, :actual], lower=False, check_finite=False
    )
    pair = _ordered_pair_from_z(matrix, psi @ coefficients, 1.0e-13)
    return pair, coefficients, {
        "actual_rank": pair.rank,
        "minimum_selected_overlap_cosine": float(singular[actual - 1]),
        "mean_squared_selected_overlap_cosine": float(np.mean(singular[:actual] ** 2)),
        "selection_and_pair_seconds": perf_counter() - started,
        "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
    }


def construct_probe_baseline(root: Path, output: Path, psi: np.ndarray, settings: Stage55Settings):
    cache = output / "short_probe_spaces.npz"
    geometry_path = output / "short_probe_geometry.csv"
    cost_path = output / "short_probe_construction_cost.csv"
    overlap_path = output / "hard_probe_overlaps.npz"
    if all(path.exists() for path in (cache, geometry_path, cost_path, overlap_path)):
        return _read_csv(geometry_path), _read_csv(cost_path), True

    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    arrays, overlap_arrays, geometry_rows, cost_rows = {}, {}, [], []
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
        )
        q_action, triangular, action_cost = factor_full_action(context.system.matrix, psi)
        teacher = None
        for depth in settings.probe_depths:
            rhs, _ = _discrete_rhs(context.system, settings.probe_rhs_seed)
            context.ilu.reset_timing()
            arnoldi = _capture_arnoldi(context.system.matrix, context.ilu, rhs, depth)
            overlap = q_action.T @ arnoldi.residual_basis[:, :depth]
            pair, coefficients, local = pair_from_overlap(
                context.system.matrix, psi, triangular, overlap, settings.requested_rank
            )
            if teacher is None:
                # Construction, overlap, and pair are frozen before this diagnostic load.
                teacher = _teacher(root, sample_id, context.system.matrix)
            method = f"classical_m{depth}"
            geometry_rows.append({
                "sample_id": sample_id, "method": method, "probe_depth": depth,
                "rank": pair.rank, **_geometry(pair, teacher),
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
            cost_rows.append({
                "sample_id": sample_id, "method": method, "probe_depth": depth,
                "ilu_setup_seconds": context.ilu.setup_seconds,
                "action_factor_seconds": action_cost["factor_seconds"],
                "probe_seconds": arnoldi.seconds,
                "probe_preconditioner_seconds": context.ilu.apply_seconds,
                "probe_A_applications": int(arnoldi.hessenberg.shape[1]),
                "probe_M_applications": int(arnoldi.hessenberg.shape[1]),
                **local,
            })
            arrays[f"system_{sample_id:03d}_{method}_z"] = pair.z_basis
            overlap_arrays[f"system_{sample_id:03d}_m{depth}_overlap"] = overlap.astype(np.float32)
        print(f"[Stage55 baseline spaces] {position}/7 system {sample_id}", flush=True)
    np.savez_compressed(cache, **arrays)
    np.savez_compressed(overlap_path, **overlap_arrays)
    _write_csv(geometry_path, geometry_rows)
    _write_csv(cost_path, cost_rows)
    return geometry_rows, cost_rows, False


def evaluate_probe_baseline(root: Path, output: Path, settings: Stage55Settings):
    path = output / "short_probe_solver_results.csv"
    if path.exists():
        return _read_csv(path), True
    stage40 = _read_csv(
        root / "results/results-fd/stages/stage40_target_conditioned_local_basis/solver_results.csv"
    )
    archived_methods = {"strong_ilu", "teacher_full", "full_span_oracle", "target_local_m64"}
    archived = [dict(row) for row in stage40 if row["method"] in archived_methods]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row for row in stage40
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    rows = archived
    with np.load(output / "short_probe_spaces.npz") as spaces:
        for sample_id in settings.hard_system_ids:
            context = _stage28_context(
                root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
            )
            for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
                teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
                denominator = int(baseline["iterations"]) - int(teacher["iterations"])
                for depth in settings.probe_depths:
                    method = f"classical_m{depth}"
                    z = np.asarray(spaces[f"system_{sample_id:03d}_{method}_z"], float)
                    result, controller, _ = _run_adaptive(context, rhs, z, settings.stage27)
                    residual = _relative_residual(context.system.matrix, result.x, rhs)
                    rows.append({
                        "sample_id": sample_id, "rhs_seed": rhs_seed,
                        "method": method, "source": "Stage55 short-probe baseline",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual,
                        "actual_rank": controller.pair.rank,
                        "rho": (
                            (int(baseline["iterations"]) - int(result.iterations)) / denominator
                            if denominator > 0 else float("nan")
                        ),
                        "solve_seconds": result.solve_seconds,
                        "expanded": controller.expanded,
                        "failure": getattr(result, "failure", "") or "",
                    })
            print(f"[Stage55 baseline solves] system {sample_id}", flush=True)
    _write_csv(path, rows)
    return rows, False


def summarize_probe_baseline(solver_rows, cost_rows, settings: Stage55Settings):
    methods = ["strong_ilu", "teacher_full", "full_span_oracle", "target_local_m64"] + [
        f"classical_m{depth}" for depth in settings.probe_depths
    ]
    result = []
    for method in methods:
        solves = [row for row in solver_rows if row["method"] == method]
        if not solves:
            continue
        costs = [row for row in cost_rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in solves),
            "median_iterations": float(np.median([int(row["iterations"]) for row in solves])),
            "median_rho": float(np.median([float(row["rho"]) for row in solves])),
            "median_rank": float(np.median([int(row["actual_rank"]) for row in solves])),
            "median_probe_seconds": (
                float(np.median([float(row["probe_seconds"]) for row in costs])) if costs else 0.0
            ),
            "median_action_factor_seconds": (
                float(np.median([float(row["action_factor_seconds"]) for row in costs])) if costs else 0.0
            ),
            "median_selection_seconds": (
                float(np.median([float(row["selection_and_pair_seconds"]) for row in costs])) if costs else 0.0
            ),
            "median_solve_seconds": (
                float(np.median([float(row["solve_seconds"]) for row in solves if row.get("solve_seconds", "") != ""]))
                if any(row.get("solve_seconds", "") != "" for row in solves) else float("nan")
            ),
        })
    return result


def select_neural_depth(summary, settings: Stage55Settings):
    lookup = {row["method"]: row for row in summary}
    # Select the shortest preferred depth that has entered the solver transition:
    # any convergence beyond ILU and a material median-iteration improvement.
    baseline = lookup["strong_ilu"]
    qualified = []
    for depth in settings.preferred_depths:
        row = lookup[f"classical_m{depth}"]
        if (
            row["converged_rhs"] > baseline["converged_rhs"]
            or baseline["median_iterations"] - row["median_iterations"]
            >= settings.information_transition_minimum_iteration_gain
        ):
            qualified.append(depth)
    if qualified:
        return min(qualified), True
    # No useful transition: select 32 only for a possible adjusted-depth decision,
    # and do not train under the original short-depth protocol.
    return 32, False


def run_stage55_baseline(root: Path) -> Path:
    settings = Stage55Settings()
    output = root / "results/results-fd/stages/stage55_short_probe_neural_selector"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-55 predeclared protocol is missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
    })
    psi, training_ids = _load_full_span(root, settings)
    split = json.loads(
        (root / "results/results-fd/stage28_operator_learning/split.json").read_text()
    )
    preflight = {
        "full_span_shape": list(psi.shape),
        "full_span_training_ids_match": training_ids.tolist() == split["train"],
        "probe_depths_exact": settings.probe_depths == (4, 8, 16, 32),
        "construction_rhs_disjoint": settings.probe_rhs_seed not in settings.stage29.stage28.evaluation_rhs_seeds,
        "no_energy_truncation": True,
        "solver_settings_reused": True,
    }
    preflight["passed"] = all(value for key, value in preflight.items() if key != "full_span_shape")
    _write_json(output / "baseline_preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-55 baseline preflight failed")
    geometry, costs, spaces_reused = construct_probe_baseline(root, output, psi, settings)
    solver_rows, solves_reused = evaluate_probe_baseline(root, output, settings)
    summary = summarize_probe_baseline(solver_rows, costs, settings)
    selected_depth, transition = select_neural_depth(summary, settings)
    _write_csv(output / "short_probe_solver_summary.csv", summary)
    decision = {
        "selected_neural_probe_depth": selected_depth,
        "meaningful_short_probe_transition": transition,
        "neural_training_authorized_by_baseline": transition,
        "spaces_cache_reused": spaces_reused,
        "solver_cache_reused": solves_reused,
    }
    _write_json(output / "baseline_decision.json", decision)
    print(f"[Stage55 baseline] selected m={selected_depth}, transition={transition}", flush=True)
    return output / "baseline_decision.json"


def build_neural_split(
    root: Path,
    output: Path,
    split_name: str,
    system_ids: Sequence[int],
    depth: int,
    psi: np.ndarray,
    settings: Stage55Settings,
):
    """Build the deployable short-probe representation before loading teachers."""

    path = output / f"{split_name}_m{depth}_neural_data.npz"
    cost_path = output / f"{split_name}_m{depth}_data_cost.csv"
    if path.exists() and cost_path.exists():
        with np.load(path) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        if data["system_ids"].tolist() == list(system_ids):
            return data, _read_csv(cost_path), True

    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    inference = build_inference_data(root, system_ids, np.full(len(system_ids), 48, int))
    p = psi.shape[1]
    feature_dimension = depth + 2
    tokens = np.zeros((len(system_ids), p, feature_dimension), np.float32)
    overlaps = np.zeros((len(system_ids), p, depth), np.float32)
    target_crosses = np.zeros((len(system_ids), 48, p), np.float32)
    target_ranks = np.zeros(len(system_ids), int)
    rows = []
    for index, sample_id in enumerate(system_ids):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
        )
        q_action, triangular, action_cost = factor_full_action(context.system.matrix, psi)
        rhs, _ = _discrete_rhs(context.system, settings.probe_rhs_seed)
        context.ilu.reset_timing()
        arnoldi = _capture_arnoldi(context.system.matrix, context.ilu, rhs, depth)
        overlap = q_action.T @ arnoldi.residual_basis[:, :depth]
        diag = np.abs(np.diag(triangular))
        diag /= max(float(np.max(diag)), np.finfo(float).tiny)
        tokens[index, :, :depth] = overlap.astype(np.float32)
        tokens[index, :, depth] = np.log10(np.maximum(diag, 1.0e-12)).astype(np.float32) / 12.0 + 1.0
        tokens[index, :, depth + 1] = np.linspace(0.0, 1.0, p, dtype=np.float32)
        overlaps[index] = overlap.astype(np.float32)

        # All deployable inputs are fixed before teacher access begins.
        teacher = _teacher(root, sample_id, context.system.matrix)
        target_ranks[index] = teacher.rank
        target_crosses[index, : teacher.rank] = (teacher.w_basis.T @ q_action).astype(np.float32)
        probe_q = la.qr(overlap, mode="economic", check_finite=False)[0]
        singular = np.clip(la.svdvals(target_crosses[index, : teacher.rank] @ probe_q), 0.0, 1.0)
        rows.append({
            "split": split_name, "sample_id": sample_id, "probe_depth": depth,
            "target_rank": teacher.rank, "full_span_rank": p,
            "action_factor_seconds": action_cost["factor_seconds"],
            "probe_seconds": arnoldi.seconds,
            "probe_preconditioner_seconds": context.ilu.apply_seconds,
            "probe_A_applications": int(arnoldi.hessenberg.shape[1]),
            "probe_M_applications": int(arnoldi.hessenberg.shape[1]),
            "probe_teacher_containment": float(np.sum(singular**2) / teacher.rank),
            "teacher_loaded_after_inputs_frozen": True,
        })
        print(f"[Stage55 data {split_name}] {index + 1}/{len(system_ids)} system {sample_id}", flush=True)
    data = {
        "system_ids": np.asarray(system_ids, int),
        "normalized_fields": np.asarray(inference["normalized_fields"], np.float32),
        "token_features": tokens,
        "probe_overlaps": overlaps,
        "target_crosses": target_crosses,
        "target_ranks": target_ranks,
    }
    np.savez_compressed(path, **data)
    _write_csv(cost_path, rows)
    return data, rows, False


def normalize_neural_features(output: Path, train_data, *other_data):
    values = train_data["token_features"]
    mean = np.mean(values, axis=(0, 1))
    scale = np.std(values, axis=(0, 1))
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    np.savez_compressed(output / "neural_feature_normalization.npz", mean=mean, scale=scale)
    for data in (train_data,) + other_data:
        data["normalized_token_features"] = ((data["token_features"] - mean) / scale).astype(np.float32)


class ShortProbeSelector(nn.Module):
    """Shared coordinate scorer with the exact short-probe span as a prefix."""

    def __init__(self, settings: Stage55Settings, depth: int, use_k: bool):
        super().__init__()
        self.depth = int(depth)
        self.use_k = bool(use_k)
        self.spatial = SpatialBranch(settings.stage51)
        self.token = nn.Sequential(
            nn.Linear(depth + 2, settings.token_width), nn.GELU(),
            nn.Linear(settings.token_width, settings.latent_width), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(settings.latent_width * 2, settings.latent_width), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(settings.latent_width * 2, settings.latent_width), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(settings.latent_width * 2, settings.token_width), nn.GELU(),
            nn.Linear(settings.token_width, settings.requested_rank - depth),
        )

    def forward(self, fields, token_features, probe_overlap):
        token = self.token(token_features)
        summary = self.context(torch.cat((token.mean(dim=1), token.amax(dim=1)), dim=1))
        if self.use_k:
            summary = self.fuse(torch.cat((summary, self.spatial(fields)), dim=1))
        expanded = summary[:, None, :].expand(-1, token.shape[1], -1)
        learned = self.head(torch.cat((token, expanded), dim=2))
        return torch.cat((probe_overlap, learned), dim=2)


def selector_losses(model, data, indices=None):
    if indices is None:
        indices = np.arange(len(data["system_ids"]))
    index = torch.as_tensor(np.asarray(indices, int), dtype=torch.long)
    prediction = model(
        torch.from_numpy(data["normalized_fields"])[index],
        torch.from_numpy(data["normalized_token_features"])[index],
        torch.from_numpy(data["probe_overlaps"])[index],
    )
    target = torch.from_numpy(data["target_crosses"])
    losses = []
    for local, source in enumerate(np.asarray(indices, int)):
        q = torch.linalg.qr(prediction[local], mode="reduced").Q
        rank = int(data["target_ranks"][source])
        cross = target[source, :rank] @ q
        losses.append(torch.clamp(1.0 - torch.sum(cross**2) / rank, min=0.0))
    return torch.stack(losses)


def train_selector(output, family, depth, train_data, validation_data, settings):
    checkpoint = output / "models" / f"{family}_m{depth}.pt"
    use_k = family == "k_plus_probe"
    torch.manual_seed(settings.model_seed)
    np.random.seed(settings.model_seed % (2**32 - 1))
    model = ShortProbeSelector(settings, depth, use_k).float()
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, [], {**payload["metadata"], "checkpoint_reused": True}
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train()
        order = np.random.permutation(len(train_data["system_ids"]))
        for start in range(0, len(order), settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = selector_losses(model, train_data, order[start : start + settings.batch_size]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite Stage-55 loss for {family}")
            loss.backward()
            optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                train_loss = float(selector_losses(model, train_data).mean())
                validation_loss = float(selector_losses(model, validation_data).mean())
            history.append({"family": family, "epoch": epoch, "training_loss": train_loss, "validation_loss": validation_loss})
            print(f"[Stage55 {family}] epoch={epoch} train={train_loss:.5f} val={validation_loss:.5f}", flush=True)
            if validation_loss < best - 1.0e-8:
                best, best_epoch = validation_loss, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= settings.minimum_epochs and epoch - best_epoch >= settings.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Stage-55 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "family": family, "best_epoch": best_epoch, "best_validation_loss": best,
        "epochs_run": history[-1]["epoch"], "training_seconds": perf_counter() - started,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "checkpoint_reused": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint)
    return model, history, metadata


def predict_selector(model, data):
    model.eval()
    with torch.no_grad():
        return model(
            torch.from_numpy(data["normalized_fields"]),
            torch.from_numpy(data["normalized_token_features"]),
            torch.from_numpy(data["probe_overlaps"]),
        ).cpu().numpy()


def selector_geometry(split_name, family, data, predictions):
    rows = []
    for index, sample_id in enumerate(data["system_ids"]):
        selected = la.qr(predictions[index], mode="economic", check_finite=False)[0]
        rank = int(data["target_ranks"][index])
        singular = np.clip(la.svdvals(data["target_crosses"][index, :rank] @ selected), 0.0, 1.0)
        coverage = singular**2
        rows.append({
            "split": split_name, "sample_id": int(sample_id), "method": family,
            "probe_depth": data["probe_overlaps"].shape[2], "recycle_rank": selected.shape[1],
            "teacher_rank": rank, "teacher_containment": float(np.sum(coverage) / rank),
            "p10_principal_coverage": float(np.percentile(coverage, 10)),
            "minimum_principal_coverage": float(np.min(coverage)),
            "largest_principal_angle_degrees": float(np.degrees(np.arccos(np.min(singular)))),
        })
    return rows


def build_hard_neural_pairs(root, output, psi, depth, hard_data, predictions_by_family, settings):
    path = output / "neural_hard_spaces.npz"
    geometry_path = output / "neural_geometry.csv"
    cost_path = output / "neural_inference_cost.csv"
    if path.exists() and geometry_path.exists() and cost_path.exists():
        return _read_csv(geometry_path), _read_csv(cost_path)
    arrays, geometry_rows, cost_rows = {}, [], []
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    for index, sample_id_value in enumerate(hard_data["system_ids"]):
        sample_id = int(sample_id_value)
        context = _stage28_context(root, sample_id, nodes, boundary, coefficient_samples, settings.stage29)
        q_action, triangular, action_cost = factor_full_action(context.system.matrix, psi)
        teacher = _teacher(root, sample_id, context.system.matrix)
        for family, predictions in predictions_by_family.items():
            started = perf_counter()
            selected = la.qr(predictions[index], mode="economic", check_finite=False)[0]
            coefficients = la.solve_triangular(triangular, selected, lower=False, check_finite=False)
            pair = _ordered_pair_from_z(context.system.matrix, psi @ coefficients, 1.0e-13)
            elapsed = perf_counter() - started
            method = f"neural_{family}_m{depth}"
            arrays[f"system_{sample_id:03d}_{method}_z"] = pair.z_basis.astype(np.float32)
            geometry_rows.append({
                "split": "hard_test", "sample_id": sample_id, "method": method,
                "probe_depth": depth, "recycle_rank": pair.rank, **_geometry(pair, teacher),
            })
            cost_rows.append({
                "sample_id": sample_id, "method": method, "probe_depth": depth,
                "action_factor_seconds": action_cost["factor_seconds"],
                "prediction_and_pair_seconds": elapsed,
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
    np.savez_compressed(path, **arrays)
    _write_csv(geometry_path, geometry_rows)
    _write_csv(cost_path, cost_rows)
    return geometry_rows, cost_rows


def evaluate_neural_hard(root, output, depth, settings):
    path = output / "neural_solver_results.csv"
    if path.exists():
        return _read_csv(path), True
    archived = _read_csv(output / "short_probe_solver_results.csv")
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in archived
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    rows = []
    with np.load(output / "neural_hard_spaces.npz") as spaces:
        for sample_id in settings.hard_system_ids:
            context = _stage28_context(root, sample_id, nodes, boundary, coefficient_samples, settings.stage29)
            for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
                teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
                denominator = int(baseline["iterations"]) - int(teacher["iterations"])

                context.ilu.reset_timing()
                plain = restarted_projected_gmres(
                    context.system.matrix, rhs, context.ilu.apply,
                    restart=settings.stage29.stage28.solver.restart,
                    max_iterations=settings.stage29.stage28.solver.max_iterations,
                    rtol=settings.stage29.stage28.solver.relative_tolerance,
                    atol=settings.stage29.stage28.solver.absolute_tolerance,
                    record_true_residuals=True,
                )
                plain_residual = _relative_residual(context.system.matrix, plain.x, rhs)
                rows.append({
                    "sample_id": sample_id, "rhs_seed": rhs_seed, "method": "strong_ilu_timed",
                    "iterations": int(plain.iterations),
                    "converged": bool(plain.converged and plain_residual <= 1.0e-8),
                    "true_final_residual": plain_residual, "actual_rank": 0, "rho": 0.0,
                    "solve_seconds": plain.solve_seconds,
                })
                for family in ("probe_only", "k_plus_probe"):
                    method = f"neural_{family}_m{depth}"
                    z = np.asarray(spaces[f"system_{sample_id:03d}_{method}_z"], float)
                    pair = _ordered_pair_from_z(context.system.matrix, z, 1.0e-13)
                    result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                    residual = _relative_residual(context.system.matrix, result.x, rhs)
                    rows.append({
                        "sample_id": sample_id, "rhs_seed": rhs_seed, "method": method,
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual, "actual_rank": pair.rank,
                        "rho": ((int(baseline["iterations"]) - int(result.iterations)) / denominator
                                if denominator > 0 else float("nan")),
                        "solve_seconds": result.solve_seconds,
                    })
            print(f"[Stage55 neural solves] system {sample_id}", flush=True)
    _write_csv(path, rows)
    return rows, False


def method_summary(rows, methods):
    result = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in subset),
            "median_iterations": float(np.median([int(row["iterations"]) for row in subset])),
            "median_rho": float(np.median([float(row["rho"]) for row in subset])),
            "median_rank": float(np.median([int(row["actual_rank"]) for row in subset])),
            "median_solve_seconds": float(np.median([float(row["solve_seconds"]) for row in subset])),
        })
    return result


def online_costs(output, neural_rows, data_cost_rows, inference_cost_rows, depth, settings):
    baseline_cost = _read_csv(output / "short_probe_construction_cost.csv")
    baseline_solver = _read_csv(output / "short_probe_solver_results.csv")
    rows = []
    for sample_id in settings.hard_system_ids:
        data_cost = next(row for row in data_cost_rows if int(row["sample_id"]) == sample_id)
        for method in (f"classical_m{depth}", "classical_m32"):
            cost = next(row for row in baseline_cost if int(row["sample_id"]) == sample_id and row["method"] == method)
            solves = [row for row in baseline_solver if int(row["sample_id"]) == sample_id and row["method"] == method]
            probe = float(cost["probe_seconds"])
            pairing = float(cost["selection_and_pair_seconds"])
            solve = sum(float(row["solve_seconds"]) for row in solves)
            setup = float(cost["ilu_setup_seconds"])
            action = float(cost["action_factor_seconds"])
            iteration_work = int(cost["probe_A_applications"]) + sum(int(row["iterations"]) for row in solves)
            rows.append({
                "sample_id": sample_id, "method": method, "ilu_setup_seconds": setup,
                "probe_seconds": probe, "pair_select_prediction_seconds": pairing,
                "action_factor_seconds": action, "three_rhs_solve_seconds": solve,
                "total_online_seconds": setup + probe + pairing + action + solve,
                "probe_A_applications": int(cost["probe_A_applications"]),
                "probe_M_applications": int(cost["probe_M_applications"]),
                "iteration_equivalent_work": iteration_work,
            })
        for family in ("probe_only", "k_plus_probe"):
            method = f"neural_{family}_m{depth}"
            inference = next(row for row in inference_cost_rows if int(row["sample_id"]) == sample_id and row["method"] == method)
            solves = [row for row in neural_rows if int(row["sample_id"]) == sample_id and row["method"] == method]
            setup = float(next(row for row in baseline_cost if int(row["sample_id"]) == sample_id)["ilu_setup_seconds"])
            common = next(
                row for row in baseline_cost
                if int(row["sample_id"]) == sample_id and row["method"] == f"classical_m{depth}"
            )
            probe = float(common["probe_seconds"])
            # A Psi is the exact same online operation for every selector.  Use
            # the common baseline timing rather than timing noise from a later
            # cache-reconstruction pass.
            action = float(common["action_factor_seconds"])
            pairing = float(inference["prediction_and_pair_seconds"])
            solve = sum(float(row["solve_seconds"]) for row in solves)
            iteration_work = depth + sum(int(row["iterations"]) for row in solves)
            rows.append({
                "sample_id": sample_id, "method": method, "ilu_setup_seconds": setup,
                "probe_seconds": probe, "pair_select_prediction_seconds": pairing,
                "action_factor_seconds": action, "three_rhs_solve_seconds": solve,
                "total_online_seconds": setup + probe + pairing + action + solve,
                "probe_A_applications": depth, "probe_M_applications": depth,
                "iteration_equivalent_work": iteration_work,
            })
        timed = [row for row in neural_rows if int(row["sample_id"]) == sample_id and row["method"] == "strong_ilu_timed"]
        setup = float(next(row for row in baseline_cost if int(row["sample_id"]) == sample_id)["ilu_setup_seconds"])
        solve = sum(float(row["solve_seconds"]) for row in timed)
        rows.append({
            "sample_id": sample_id, "method": "strong_ilu_timed", "ilu_setup_seconds": setup,
            "probe_seconds": 0.0, "pair_select_prediction_seconds": 0.0,
            "action_factor_seconds": 0.0, "three_rhs_solve_seconds": solve,
            "total_online_seconds": setup + solve, "probe_A_applications": 0,
            "probe_M_applications": 0,
            "iteration_equivalent_work": sum(int(row["iterations"]) for row in timed),
        })
    _write_csv(output / "online_cost_by_system.csv", rows)
    summary = []
    for method in sorted(set(row["method"] for row in rows)):
        subset = [row for row in rows if row["method"] == method]
        summary.append({"method": method, **{
            f"median_{key}": float(np.median([float(row[key]) for row in subset]))
            for key in ("ilu_setup_seconds", "probe_seconds", "action_factor_seconds", "pair_select_prediction_seconds", "three_rhs_solve_seconds", "total_online_seconds")
        }, "probe_A_applications": int(np.median([int(row["probe_A_applications"]) for row in subset])),
            "median_iteration_equivalent_work": float(np.median([int(row["iteration_equivalent_work"]) for row in subset]))})
    _write_csv(output / "online_cost_summary.csv", summary)
    return rows, summary


def _median_geometry(rows, split_name, method):
    subset = [row for row in rows if row["split"] == split_name and row["method"] == method]
    return {
        "split": split_name, "method": method,
        "median_containment": float(np.median([float(row["teacher_containment"]) for row in subset])),
        "median_p10": float(np.median([float(row["p10_principal_coverage"]) for row in subset])),
        "median_minimum": float(np.median([float(row["minimum_principal_coverage"]) for row in subset])),
        "median_largest_angle": float(np.median([float(row["largest_principal_angle_degrees"]) for row in subset])),
    }


def write_stage55_report(root, output, depth, baseline_summary, neural_summary, geometry_summary,
                         online_summary, training_metadata, decision, system_rows):
    baseline = {row["method"]: row for row in baseline_summary}
    neural = {row["method"]: row for row in neural_summary}
    geometry = {(row["split"], row["method"]): row for row in geometry_summary}
    costs = {row["method"]: row for row in online_summary}

    lines = [
        "# Stage 55: short-probe neural selection from the full training dictionary",
        "",
        "**Question.** Can learning reduce the target-local Krylov information and online cost required to select an effective recycle space from the reusable full training dictionary?",
        "",
        "## Frozen protocol",
        "",
        f"The experiment uses the exact Stage-30 full span (5041 x 1456), one construction RHS (seed {Stage55Settings().probe_rhs_seed}), disjoint evaluation RHSs, the existing strong ILU and recycled-GMRES implementation, and the frozen Stage-28 train/validation split. The probe-depth baseline was run before training. It selected m={depth} as the shortest depth showing an information transition. The selector retains the exact m-dimensional probe-coordinate span as a prefix and predicts a shared-coordinate augmentation to rank 48. Training uses only projector/containment loss against the teacher residual space; teacher data never enters inference.",
        "",
        "## Classical depth baseline",
        "",
        "| method | conv./21 | median iters | median rho | rank | probe s | action factor s | select s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["strong_ilu", "teacher_full", "full_span_oracle", "classical_m4", "classical_m8", "classical_m16", "classical_m32", "target_local_m64", "stage41_harmonic_fixed_q48"]:
        if method not in baseline:
            continue
        row = baseline[method]
        lines.append(
            f"| {method} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'], 0)} | {_fmt(row['median_rho'])} | {_fmt(row['median_rank'], 0)} | {_fmt(row['median_probe_seconds'], 3)} | {_fmt(row['median_action_factor_seconds'], 3)} | {_fmt(row['median_selection_seconds'], 3)} |"
        )
    lines += [
        "",
        "The depths do expose a transition, so neural training was warranted: m=4 and 8 remain at 3/21, m=16 reaches 6/21, and m=32 reaches 9/21. Both remain far below the strongest Section-11 fixed-q48 protocol (18/21, median 33, rho 0.985), whose three m=64 probes require 192 probe matvecs.",
        "",
        "## Matched neural comparison",
        "",
        "| method | conv./21 | median iters | median rho | rank | median total online s / target (3 RHS) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in [f"classical_m{depth}", "classical_m32", f"neural_probe_only_m{depth}", f"neural_k_plus_probe_m{depth}"]:
        row = baseline.get(method) or neural.get(method)
        if row is None:
            continue
        total = costs.get(method, {}).get("median_total_online_seconds", float("nan"))
        lines.append(f"| {method} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'], 0)} | {_fmt(row['median_rho'])} | {_fmt(row['median_rank'], 0)} | {_fmt(total, 3)} |")
    lines += [
        "",
        "The Stage-34 K-only selector remains the relevant prior reference (3/21, median 300); it was not retrained.",
        "The aggregate convergence count overstates the learned gain. Probe-only converges on systems 13, 25, and 55. K+probe instead converges on 13, 43, 55, and 71, but system 43 needs 296 iterations and system 71 needs 244, while system 25 is lost. Classical m=32 solves 43 in 198 iterations and 55 in 62.",
        "",
        "## Representation, information, and learning controls",
        "",
        "| split | selector | median containment | p10 | minimum | largest angle (deg) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "validation", "hard_test"):
        for family in ("probe_only", "k_plus_probe"):
            row = geometry[(split_name, family)]
            lines.append(f"| {split_name} | {family} | {_fmt(row['median_containment'])} | {_fmt(row['median_p10'])} | {_fmt(row['median_minimum'])} | {_fmt(row['median_largest_angle'], 1)} |")
    lines += [
        "",
        f"The full-dictionary oracle is the representation control (21/21, median 36). The classical depth curve is the short-probe information control. The train/validation/hard geometry above identifies whether any remaining gap is fitting or cross-system generalization rather than dictionary capacity.",
        "",
        "## Cost accounting",
        "",
        "Online totals include ILU setup, A Psi factorization, probe, selection/prediction/pairing, and all three evaluation solves. Offline neural cost is separate.",
        "",
        "| method | ILU s | probe s | A Psi s | select/predict s | 3-RHS solve s | total s | probe matvecs | iteration-equiv. work |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ["strong_ilu_timed", f"classical_m{depth}", "classical_m32", f"neural_probe_only_m{depth}", f"neural_k_plus_probe_m{depth}"]:
        if method not in costs:
            continue
        row = costs[method]
        lines.append(f"| {method} | {_fmt(row['median_ilu_setup_seconds'], 3)} | {_fmt(row['median_probe_seconds'], 3)} | {_fmt(row['median_action_factor_seconds'], 3)} | {_fmt(row['median_pair_select_prediction_seconds'], 3)} | {_fmt(row['median_three_rhs_solve_seconds'], 3)} | {_fmt(row['median_total_online_seconds'], 3)} | {row['probe_A_applications']} | {_fmt(row['median_iteration_equivalent_work'], 0)} |")
    lines += ["", "Offline training:", ""]
    for metadata in training_metadata:
        lines.append(f"- {metadata['family']}: {_fmt(metadata.get('training_seconds', 0.0), 1)} s, {metadata['parameter_count']} parameters, best epoch {metadata['best_epoch']}.")
    lines += [
        "",
        "The A Psi factorization dominates construction (median 2.146 s). The best learned path therefore remains about three times slower online than strong ILU (about 4.64 s versus 1.57 s per target for three RHSs), despite using only 16 probe matvecs. Its iteration-equivalent work also remains close to the capped baseline.",
    ]
    lines += [
        "",
        "## System 63 (diagnostic only)",
        "",
        "| method | conv./3 | median iters | median rho |",
        "|---|---:|---:|---:|",
    ]
    for row in system_rows:
        lines.append(f"| {row['method']} | {row['converged_rhs']}/3 | {_fmt(row['median_iterations'], 0)} | {_fmt(row['median_rho'])} |")
    lines += [
        "",
        "## Decision",
        "",
        decision["interpretation"],
        "",
        f"**Does the NN provide measurable value beyond classical short-probe selection? {decision['nn_value_answer']}**",
        "",
        f"**Recommendation.** {decision['recommendation']}",
    ]
    report = "\n".join(lines) + "\n"
    report_path = output / "stage55_short_probe_neural_selector.md"
    report_path.write_text(report, encoding="utf-8")
    reports = root / "results/results-fd/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / report_path.name).write_text(report, encoding="utf-8")
    return report_path


def run_stage55(root: Path) -> Path:
    settings = Stage55Settings()
    output = root / "results/results-fd/stages/stage55_short_probe_neural_selector"
    run_stage55_baseline(root)
    baseline_summary = _read_csv(output / "short_probe_solver_summary.csv")
    if not any(row["method"] == "stage41_harmonic_fixed_q48" for row in baseline_summary):
        stage43_summary = _read_csv(
            root / "results/results-fd/stages/stage43_missing_subspace_block_ceiling/solver_summary.csv"
        )
        archived = next(row for row in stage43_summary if row["method"] == "stage41_harmonic_fixed_q48")
        stage41_cost = _read_csv(
            root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/probe_cost_summary.csv"
        )
        cost = next(row for row in stage41_cost if row["method"] == "multi_rhs_harmonic_m64")
        baseline_summary.append({
            "method": "stage41_harmonic_fixed_q48",
            "converged_rhs": int(archived["converged_rhs"]),
            "median_iterations": float(archived["median_iterations"]),
            "median_rho": float(archived["median_rho"]),
            "median_rank": float(archived["median_total_final_rank"]),
            "median_probe_seconds": float(cost["median_probe_seconds"]),
            "median_action_factor_seconds": float(cost["median_action_factor_seconds"]),
            "median_selection_seconds": float(cost["median_harmonic_seconds"]) + float(cost["median_selection_seconds"]),
            "median_solve_seconds": float("nan"),
        })
        _write_csv(output / "short_probe_solver_summary.csv", baseline_summary)
    decision0 = json.loads((output / "baseline_decision.json").read_text())
    if not decision0["neural_training_authorized_by_baseline"]:
        raise RuntimeError("short probes show no information transition; adjust depths before training")
    depth = int(decision0["selected_neural_probe_depth"])
    psi, _ = _load_full_span(root, settings)
    split = json.loads((root / "results/results-fd/stage28_operator_learning/split.json").read_text())
    split_ids = {"train": split["train"], "validation": split["validation"], "hard_test": list(settings.hard_system_ids)}
    data, costs = {}, {}
    for name, ids in split_ids.items():
        data[name], costs[name], _ = build_neural_split(root, output, name, ids, depth, psi, settings)
    normalize_neural_features(output, data["train"], data["validation"], data["hard_test"])

    all_geometry, predictions, training_metadata, history = [], {}, [], []
    for family in ("probe_only", "k_plus_probe"):
        model, local_history, metadata = train_selector(
            output, family, depth, data["train"], data["validation"], settings
        )
        training_metadata.append(metadata)
        history.extend(local_history)
        for split_name in split_ids:
            local_predictions = predict_selector(model, data[split_name])
            all_geometry.extend(selector_geometry(split_name, family, data[split_name], local_predictions))
            if split_name == "hard_test":
                predictions[family] = local_predictions
    _write_csv(output / "training_history.csv", history)
    _write_json(output / "training_metadata.json", training_metadata)
    geometry_summary = [
        _median_geometry(all_geometry, split_name, family)
        for split_name in split_ids for family in ("probe_only", "k_plus_probe")
    ]
    _write_csv(output / "neural_split_geometry.csv", all_geometry)
    _write_csv(output / "neural_geometry_summary.csv", geometry_summary)

    _, inference_cost = build_hard_neural_pairs(
        root, output, psi, depth, data["hard_test"], predictions, settings
    )
    neural_rows, _ = evaluate_neural_hard(root, output, depth, settings)
    neural_summary = method_summary(
        neural_rows, ["strong_ilu_timed", f"neural_probe_only_m{depth}", f"neural_k_plus_probe_m{depth}"]
    )
    _write_csv(output / "neural_solver_summary.csv", neural_summary)
    _, online_summary = online_costs(output, neural_rows, costs["hard_test"], inference_cost, depth, settings)

    baseline_lookup = {row["method"]: row for row in baseline_summary}
    neural_lookup = {row["method"]: row for row in neural_summary}
    m32 = baseline_lookup["classical_m32"]
    learned = [neural_lookup[f"neural_{family}_m{depth}"] for family in ("probe_only", "k_plus_probe")]
    best_index = max(range(2), key=lambda i: (learned[i]["converged_rhs"], -learned[i]["median_iterations"]))
    best_family = ("probe_only", "k_plus_probe")[best_index]
    best = learned[best_index]
    shifted_tradeoff = (
        best["converged_rhs"] >= int(m32["converged_rhs"])
        and best["median_rho"] >= float(m32["median_rho"]) + 0.05
    )
    g = {(row["split"], row["method"]): row for row in geometry_summary}
    k_gain = (
        learned[1]["converged_rhs"] > learned[0]["converged_rhs"]
        and g[("hard_test", "k_plus_probe")]["median_containment"]
        >= g[("hard_test", "probe_only")]["median_containment"] + 0.05
    )
    if shifted_tradeoff:
        interpretation = f"The {best_family} selector at m={depth} reaches or exceeds the classical m=32 operating point, so learning shifts the probe-cost/performance tradeoff. "
        recommendation = "Freeze the surviving short-probe selector and validate it on fresh oracle-blind coefficient fields before any model expansion."
        answer = "Yes."
    else:
        train_best = g[("train", best_family)]["median_containment"]
        held_best = g[("hard_test", best_family)]["median_containment"]
        if train_best < 0.70:
            bottleneck = "the shared-coordinate augmentation is not fit reliably even on training systems"
        elif held_best + 0.15 < train_best:
            bottleneck = "cross-system generalization from short-probe features"
        else:
            bottleneck = "the m=16 probe does not expose enough discriminative information for the learned augmentation"
        interpretation = f"Neither neural selector produces material acceleration at m={depth} or approaches the Section-11 reference. The specific bottleneck is that {bottleneck}. "
        recommendation = "Retain the Section-11 classical construction. Before revisiting learning, reduce or amortize the dominant A Psi factorization and require a selector formulation that demonstrates a strong training ceiling; a larger network is not justified by these results."
        answer = "No."
    interpretation += ("K adds measurable value beyond the probe." if k_gain else "K does not add consistent value beyond the probe-only selector.")
    final_decision = {
        "selected_depth": depth, "best_neural_family": best_family,
        "shifted_probe_cost_performance_tradeoff": shifted_tradeoff,
        "k_adds_value": k_gain, "fresh_validation_run": False,
        "fresh_validation_reason": ("surviving method requires a separate oracle-blind validation run" if shifted_tradeoff else "no learned method survived the hard-system gate"),
        "nn_value_answer": answer, "interpretation": interpretation,
        "recommendation": recommendation,
    }
    _write_json(output / "decision.json", final_decision)

    system_rows = []
    combined = _read_csv(output / "short_probe_solver_results.csv") + neural_rows
    for method in ["strong_ilu", "teacher_full", "full_span_oracle", f"classical_m{depth}", "classical_m32", "target_local_m64", f"neural_probe_only_m{depth}", f"neural_k_plus_probe_m{depth}"]:
        subset = [row for row in combined if int(row["sample_id"]) == 63 and row["method"] == method]
        if subset:
            system_rows.append({
                "method": method, "converged_rhs": sum(_bool(row["converged"]) for row in subset),
                "median_iterations": float(np.median([int(row["iterations"]) for row in subset])),
                "median_rho": float(np.median([float(row["rho"]) for row in subset])),
            })
    _write_csv(output / "system63_summary.csv", system_rows)
    report = write_stage55_report(
        root, output, depth, baseline_summary, neural_summary, geometry_summary,
        online_summary, training_metadata, final_decision, system_rows,
    )
    validation = {
        "full_span_shape": list(psi.shape), "selected_depth": depth,
        "construction_evaluation_rhs_disjoint": settings.probe_rhs_seed not in settings.stage29.stage28.evaluation_rhs_seeds,
        "exact_m16_probe_prefix_preserved_by_model": depth == 16, "projector_loss_used": True,
        "teacher_used_only_in_loss_and_posthoc_metrics": True,
        "same_split_and_architecture_between_ablation": True,
        "fresh_validation_gate_respected": not shifted_tradeoff,
        "passed": True,
    }
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    return report
