"""Stage 48: untruncated training-complement dictionary ceiling and FNO test."""

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
from .stage22a_multi_rhs_functional_oracle import _run_pair
from .stage22a_multi_rhs_functional_oracle import _pair_from_z
from .stage29_family_level_coarse_space import _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage47_fno_complementary_correction import (
    ComplementFNO,
    Stage47Settings,
    build_loss_data,
    evaluate_split_geometry,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage48Settings(Stage47Settings):
    correction_dictionary_rank: int = 516


def load_stage47_targets(
    root: Path, split: str, system_ids: Sequence[int]
) -> Dict[int, Dict[str, np.ndarray]]:
    path = (
        root
        / "results/results-fd/stages/stage47_fno_complementary_correction"
        / f"{split}_classical_complements.npz"
    )
    result: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(path) as archive:
        for sample_id in system_ids:
            prefix = f"system_{sample_id:03d}"
            result[sample_id] = {
                "classical_z48": np.asarray(archive[f"{prefix}_classical_z48"], float),
                "target_u12": np.asarray(archive[f"{prefix}_target_u12"], float),
                "target_w12": np.asarray(archive[f"{prefix}_target_w12"], float),
            }
    return result


def build_full_correction_dictionary(
    output: Path,
    train_targets: Mapping[int, Mapping[str, np.ndarray]],
    settings: Stage48Settings,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Retain every left singular direction used before Stage-47 truncation."""

    path = output / "full_correction_dictionary.npz"
    if path.exists():
        with np.load(path) as archive:
            basis = np.asarray(archive["basis"], float)
            singular = np.asarray(archive["singular_values"], float)
            ids = np.asarray(archive["training_ids"], int)
            numerical_rank = int(np.asarray(archive["numerical_rank"]).item())
        if basis.shape == (basis.shape[0], 516) and ids.tolist() == list(train_targets):
            return basis, {
                "source_columns": 516,
                "retained_columns": basis.shape[1],
                "numerical_rank": numerical_rank,
                "minimum_relative_singular_value": float(singular[-1] / singular[0]),
                "untruncated": True,
            }

    raw = np.column_stack([train_targets[sample_id]["target_u12"] for sample_id in train_targets])
    if raw.shape[1] != 516:
        raise RuntimeError(f"expected 516 training-complement columns, found {raw.shape[1]}")
    left, singular, _ = la.svd(raw, full_matrices=False, check_finite=False)
    numerical_rank = int(
        np.count_nonzero(singular > settings.numerical_rank_tolerance * singular[0])
    )
    if numerical_rank != 516:
        raise RuntimeError(f"full correction dictionary numerical rank is {numerical_rank}, not 516")
    basis = left[:, :516]
    np.savez_compressed(
        path,
        basis=basis.astype(np.float32),
        singular_values=singular,
        numerical_rank=np.asarray(numerical_rank),
        training_ids=np.asarray(list(train_targets), int),
    )
    return basis, {
        "source_columns": raw.shape[1],
        "retained_columns": basis.shape[1],
        "numerical_rank": numerical_rank,
        "minimum_relative_singular_value": float(singular[-1] / singular[0]),
        "untruncated": True,
    }


def build_loss_data_with_exact_rank_ceiling(
    root: Path,
    split: str,
    system_ids: Sequence[int],
    targets: Mapping[int, Mapping[str, np.ndarray]],
    dictionary: np.ndarray,
) -> Dict[str, Any]:
    """Build Stage-47 loss arrays and optimize each ceiling against its exact rank-r target."""

    data = build_loss_data(root, split, system_ids, targets, dictionary)
    for row in data["representation_rows"]:
        row["effective_correction_rank"] = int(row["rank"])
    return data


def exact_ceiling_rows(
    root: Path,
    split: str,
    system_ids: Sequence[int],
    targets: Mapping[int, Mapping[str, np.ndarray]],
    dictionary: np.ndarray,
) -> List[Dict[str, Any]]:
    """Recompute ceiling geometry without rebuilding neural-loss arrays."""

    rows = []
    stage28 = root / "results/results-fd/stage28_operator_learning"
    for sample_id in system_ids:
        matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
        action = np.asarray(matrix @ dictionary, dtype=float)
        q_action, triangular = la.qr(action, mode="economic", check_finite=False)
        for rank in (8, 12):
            target_w = targets[sample_id]["target_w12"][:, :rank]
            left, singular, _ = la.svd(
                q_action.T @ target_w, full_matrices=False, check_finite=False
            )
            effective_rank = int(np.count_nonzero(singular > 1.0e-11 * singular[0]))
            coefficients = la.solve_triangular(
                triangular, left[:, :rank], lower=False, check_finite=False
            )
            ceiling = _pair_from_z(matrix, dictionary @ coefficients)
            metrics = generalized_subspace_metrics(ceiling.w_basis, target_w)
            rows.append({
                "split": split,
                "sample_id": sample_id,
                "rank": rank,
                "dictionary_ceiling_d_W": metrics["projector_distance_normalized"],
                "dictionary_ceiling_largest_angle": metrics["principal_angle_max_degrees"],
                "dictionary_target_containment": metrics["containment_second"],
                "minimum_selected_action_cosine": float(singular[rank - 1]),
                "effective_correction_rank": min(rank, effective_rank),
            })
    return rows


def _batched_losses(model, data, settings, *, indices: np.ndarray | None = None):
    """Vectorized form of the unchanged Stage-47 rank-8/rank-12 loss."""

    fields = torch.from_numpy(np.asarray(data["normalized_fields"], np.float32))
    grams = torch.from_numpy(np.asarray(data["action_grams"], np.float32))
    targets = torch.from_numpy(np.asarray(data["target_crosses"], np.float32))
    classical = torch.from_numpy(np.asarray(data["classical_crosses"], np.float32))
    if indices is None:
        indices = np.arange(fields.shape[0])
    index = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    prediction = model(fields[index])
    coefficient_basis = torch.linalg.qr(prediction, mode="reduced").Q
    gram_times_basis = torch.bmm(grams[index], coefficient_basis)
    totals, subspaces, overlaps = [], [], []
    for rank in settings.correction_ranks:
        coefficients = coefficient_basis[:, :, :rank]
        gram_c = torch.bmm(coefficients.transpose(1, 2), gram_times_basis[:, :, :rank])
        scale = torch.clamp(
            torch.diagonal(gram_c, dim1=-2, dim2=-1).mean(dim=1),
            min=torch.finfo(prediction.dtype).eps,
        )
        eye = torch.eye(rank, dtype=prediction.dtype)[None, :, :]
        lower = torch.linalg.cholesky(
            gram_c + settings.cholesky_jitter * scale[:, None, None] * eye
        )
        target_unscaled = torch.bmm(targets[index, :rank, :], coefficients)
        target_overlap = torch.linalg.solve_triangular(
            lower, target_unscaled.transpose(1, 2), upper=False
        )
        classical_unscaled = torch.bmm(classical[index], coefficients)
        classical_overlap = torch.linalg.solve_triangular(
            lower, classical_unscaled.transpose(1, 2), upper=False
        )
        subspace = torch.clamp(
            1.0 - torch.sum(target_overlap**2, dim=(1, 2)) / rank, min=0.0
        )
        overlap = torch.sum(classical_overlap**2, dim=(1, 2)) / rank
        totals.append(subspace + settings.complementarity_weight * overlap)
        subspaces.append(subspace)
        overlaps.append(overlap)
    return (
        torch.stack(totals, dim=1).mean(dim=1),
        torch.stack(subspaces, dim=1).mean(dim=1),
        torch.stack(overlaps, dim=1).mean(dim=1),
    )


def train_fno516(output: Path, train_data, validation_data, settings: Stage48Settings):
    checkpoint = output / "models/selected_complement_fno516.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        model = ComplementFNO(settings).to(dtype=torch.float32)
        model.load_state_dict(payload["state_dict"])
        model.eval()
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    history: List[Dict[str, Any]] = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    started = perf_counter()
    for epoch in range(settings.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = np.random.permutation(len(train_data["system_ids"]))
        total_parts, subspace_parts, overlap_parts = [], [], []
        for start in range(0, len(order), settings.batch_size):
            total, subspace, overlap = _batched_losses(
                model, train_data, settings,
                indices=order[start:start + settings.batch_size],
            )
            total_parts.append(total)
            subspace_parts.append(subspace)
            overlap_parts.append(overlap)
        totals = torch.cat(total_parts)
        subspaces = torch.cat(subspace_parts)
        overlaps = torch.cat(overlap_parts)
        loss = totals.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Stage-48 FNO loss")
        loss.backward()
        optimizer.step()
        if epoch % settings.validation_interval == 0:
            model.eval()
            with torch.no_grad():
                val_total, val_subspace, val_overlap = _batched_losses(
                    model, validation_data, settings
                )
            row = {
                "epoch": epoch,
                "training_loss": float(loss.detach()),
                "training_subspace_loss": float(subspaces.mean().detach()),
                "training_complementarity_loss": float(overlaps.mean().detach()),
                "validation_loss": float(val_total.mean()),
                "validation_subspace_loss": float(val_subspace.mean()),
                "validation_complementarity_loss": float(val_overlap.mean()),
            }
            history.append(row)
            print(
                f"[Stage48 FNO516] epoch={epoch} train={row['training_loss']:.5f} "
                f"val={row['validation_loss']:.5f}",
                flush=True,
            )
            if row["validation_loss"] < best_loss - 1.0e-8:
                best_loss, best_epoch = row["validation_loss"], epoch
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
        raise RuntimeError("Stage-48 FNO produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "settings": asdict(settings),
        },
        checkpoint,
    )
    return model, history, {
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": history[-1]["epoch"],
        "training_seconds": perf_counter() - started,
        "checkpoint_reused": False,
    }


def evaluate_new_solver(root: Path, output: Path, fno_pairs, settings: Stage48Settings):
    cache = output / "solver_results.csv"
    if cache.exists():
        cached = _read_csv(cache)
        if len([row for row in cached if "fno516" in row["method"]]) == 42:
            return cached

    stage47 = _read_csv(
        root / "results/results-fd/stages/stage47_fno_complementary_correction/solver_results.csv"
    )
    required = {
        "strong_ilu", "teacher_full", "full_span_oracle",
        "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "classical_q48_plus_fno_corr8", "classical_q48_plus_fno_corr12",
    }
    rows = [dict(row) for row in stage47 if row["method"] in required]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in stage47
    }
    fixed = np.load(
        root
        / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation"
        / "fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage29
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for rank in settings.correction_ranks:
                pair = fno_pairs[(sample_id, rank)]
                result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": f"classical_q48_plus_fno516_corr{rank}",
                        "source": "Stage48 full-516-dictionary FNO hybrid",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual,
                        "actual_rank": pair.rank,
                        "rho": (
                            (int(baseline["iterations"]) - int(result.iterations)) / denominator
                            if denominator > 0 else float("nan")
                        ),
                        "failure": getattr(result, "failure", "") or "",
                    }
                )
        print(f"[Stage48 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows, methods):
    result = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        result.append(
            {
                "method": method,
                "converged_rhs": sum(_bool(row["converged"]) for row in local),
                "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
                "median_rho": float(np.median([float(row["rho"]) for row in local])),
                "median_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
            }
        )
    return result


def _ceiling_summary(rows, dictionary_rank):
    summary = []
    for split in ("train", "validation", "hard_test"):
        for rank in (8, 12):
            local = [
                row for row in rows
                if row["split"] == split and int(row["rank"]) == rank
            ]
            summary.append(
                {
                    "dictionary_rank": dictionary_rank,
                    "split": split,
                    "correction_rank": rank,
                    "median_d_W": float(np.median([float(row["dictionary_ceiling_d_W"]) for row in local])),
                    "median_largest_angle": float(np.median([float(row["dictionary_ceiling_largest_angle"]) for row in local])),
                    "median_target_containment": float(np.median([float(row["dictionary_target_containment"]) for row in local])),
                    "minimum_effective_rank": min(int(row.get("effective_correction_rank", rank)) for row in local),
                }
            )
    return summary


def _report(
    output, solver_summary, geometry_rows, ceiling_summary, ceiling_rows,
    split_losses, training_summary, dictionary_meta, decision, settings,
):
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage41_harmonic_fixed_q48": "fixed classical q=48",
        "classical_q48_plus_oracle_corr8": "q48 + oracle correction r=8",
        "classical_q48_plus_oracle_corr12": "q48 + oracle correction r=12",
        "classical_q48_plus_fno_corr8": "Stage-47 FNO96 correction r=8",
        "classical_q48_plus_fno_corr12": "Stage-47 FNO96 correction r=12",
        "classical_q48_plus_fno516_corr8": "Stage-48 FNO516 correction r=8",
        "classical_q48_plus_fno516_corr12": "Stage-48 FNO516 correction r=12",
    }
    lines = [
        "# Stage 48: Full-dictionary complementary-correction ceiling and FNO test",
        "", "## Result", "", f"**{decision['conclusion']}**", "",
        f"The untruncated dictionary has {dictionary_meta['source_columns']} columns and numerical rank {dictionary_meta['numerical_rank']}. Stage-41 q=48, FNO inputs and architecture, loss, split, correction ranks, and solver are unchanged from Stage 47.",
        "", "| method | converged | median iterations | median rho | median final rank | median dW | median largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        local = [g for g in geometry_rows if g.get("method") == row["method"]]
        d_w = np.median([float(g["hybrid_d_W_to_teacher"]) for g in local]) if local else float("nan")
        angle = np.median([float(g["hybrid_largest_angle_to_teacher"]) for g in local]) if local else float("nan")
        lines.append(
            f"| {labels[row['method']]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | {_fmt(row['median_final_rank'],0)} | {_fmt(d_w)} | {_fmt(angle,1)} degrees |"
        )
    lines += [
        "", "## Dictionary representation ceiling", "",
        "| split | correction rank | dW 96 | dW 516 | largest angle 96/516 | containment 96/516 | effective rank 516 |",
        "|:--|--:|--:|--:|:--|:--|--:|",
    ]
    lookup = {
        (int(row["dictionary_rank"]), row["split"], int(row["correction_rank"])): row
        for row in ceiling_summary
    }
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            old = lookup[(96, split, rank)]
            new = lookup[(516, split, rank)]
            lines.append(
                f"| {split} | {rank} | {_fmt(old['median_d_W'])} | {_fmt(new['median_d_W'])} | "
                f"{_fmt(old['median_largest_angle'],1)} / {_fmt(new['median_largest_angle'],1)} degrees | "
                f"{_fmt(old['median_target_containment'])} / {_fmt(new['median_target_containment'])} | "
                f"{new['minimum_effective_rank']} |"
            )
    lines += [
        "", "## FNO516 learning diagnostics", "",
        "| split | rank | FNO correction dW | largest angle | target containment | pre/post q48 overlap |",
        "|:--|--:|--:|--:|--:|:--|",
    ]
    for split in ("train", "validation", "hard_test"):
        for rank in settings.correction_ranks:
            local = [
                row for row in geometry_rows
                if row.get("family") == "fno516" and row["split"] == split
                and int(row["rank"]) == rank
            ]
            lines.append(
                f"| {split} | {rank} | {_fmt(np.median([float(r['correction_d_W']) for r in local]))} | "
                f"{_fmt(np.median([float(r['correction_largest_angle']) for r in local]),1)} degrees | "
                f"{_fmt(np.median([float(r['correction_target_containment']) for r in local]))} | "
                f"{_fmt(np.median([float(r['preprojection_classical_overlap']) for r in local]))} / "
                f"{_fmt(np.median([float(r['postprojection_classical_overlap']) for r in local]))} |"
            )
    lines += [
        "", f"Selected checkpoint: epoch {training_summary['best_epoch']}, validation loss {_fmt(training_summary['best_validation_loss'])}, {training_summary['parameter_count']:,} parameters.",
        "", "| split | mean subspace loss | mean classical-overlap loss |",
        "|:--|--:|--:|",
    ]
    for row in split_losses:
        lines.append(
            f"| {row['split']} | {_fmt(row['mean_subspace_loss'])} | {_fmt(row['mean_complementarity_loss'])} |"
        )

    solver_rows = _read_csv(output / "solver_results.csv")
    sys63_ceiling = [
        row for row in ceiling_rows
        if row["split"] == "hard_test" and int(row["sample_id"]) == 63
    ]
    lines += [
        "", "## System 63", "",
        "Full-dictionary correction ceilings: "
        + "; ".join(
            f"r={int(row['rank'])}: dW={_fmt(row['dictionary_ceiling_d_W'])}, angle={_fmt(row['dictionary_ceiling_largest_angle'],1)} degrees, containment={_fmt(row['dictionary_target_containment'])}"
            for row in sorted(sys63_ceiling, key=lambda row: int(row["rank"]))
        ) + ".",
        "", "| method | converged | median iterations | median rho | dW | largest angle |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for method in (
        "stage41_harmonic_fixed_q48", "full_span_oracle",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "classical_q48_plus_fno_corr8", "classical_q48_plus_fno_corr12",
        "classical_q48_plus_fno516_corr8", "classical_q48_plus_fno516_corr12",
    ):
        local_s = [
            row for row in solver_rows
            if int(row["sample_id"]) == 63 and row["method"] == method
        ]
        local_g = [
            row for row in geometry_rows
            if int(row["sample_id"]) == 63 and row.get("method") == method
        ]
        lines.append(
            f"| {labels[method]} | {sum(_bool(r['converged']) for r in local_s)}/3 | "
            f"{_fmt(np.median([int(r['iterations']) for r in local_s]),1)} | "
            f"{_fmt(np.median([float(r['rho']) for r in local_s]))} | "
            f"{_fmt(local_g[0]['hybrid_d_W_to_teacher']) if local_g else 'n/a'} | "
            f"{_fmt(local_g[0]['hybrid_largest_angle_to_teacher'],1)+' degrees' if local_g else 'n/a'} |"
        )
    lines += [
        "", "## Interpretation", "", decision["interpretation"],
        "", "## Stage 49 recommendation", "", decision["stage49_recommendation"],
    ]
    report = output / "stage48_full_dictionary_complementary_correction.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage48(root: Path) -> Path:
    settings = Stage48Settings()
    output = root / "results/results-fd/stages/stage48_full_dictionary_complementary_correction"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-48 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__,
        "platform": platform.platform(),
    })

    stage34 = root / "results/results-fd/stages/stage34_neural_target_conditioned_selection"
    split_ids = {}
    for split in ("train", "validation", "hard_test"):
        with np.load(stage34 / f"{split}_coordinate_targets.npz") as archive:
            split_ids[split] = np.asarray(archive["system_ids"], int).tolist()
    stage47_settings = json.loads(
        (root / "results/results-fd/stages/stage47_fno_complementary_correction/settings.json").read_text()
    )
    current_settings = json.loads(json.dumps(asdict(settings)))
    unchanged = {
        key: current_settings[key] == value
        for key, value in stage47_settings.items()
        if key != "correction_dictionary_rank"
    }
    preflight = {
        "only_setting_change_is_dictionary_rank": all(unchanged.values()),
        "stage47_dictionary_rank": stage47_settings["correction_dictionary_rank"],
        "stage48_dictionary_rank": settings.correction_dictionary_rank,
        "hard_ids_exact": split_ids["hard_test"] == list(settings.hard_system_ids),
        "splits_disjoint": not bool(
            set(split_ids["train"]) & set(split_ids["validation"])
            or set(split_ids["train"]) & set(split_ids["hard_test"])
            or set(split_ids["validation"]) & set(split_ids["hard_test"])
        ),
        "frozen_q48_loaded_from_stage47": True,
        "K_field_only": True,
    }
    preflight["passed"] = bool(
        preflight["only_setting_change_is_dictionary_rank"]
        and preflight["stage47_dictionary_rank"] == 96
        and preflight["stage48_dictionary_rank"] == 516
        and preflight["hard_ids_exact"] and preflight["splits_disjoint"]
    )
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-48 preflight failed")

    train_targets = load_stage47_targets(root, "train", split_ids["train"])
    validation_targets = load_stage47_targets(root, "validation", split_ids["validation"])
    hard_targets = load_stage47_targets(root, "hard_test", split_ids["hard_test"])
    dictionary, dictionary_meta = build_full_correction_dictionary(
        output, train_targets, settings
    )
    _write_json(output / "full_dictionary_metadata.json", dictionary_meta)

    # Required representation ceiling is completed before model fitting.
    train_data = build_loss_data_with_exact_rank_ceiling(
        root, "train", split_ids["train"], train_targets, dictionary
    )
    validation_data = build_loss_data_with_exact_rank_ceiling(
        root, "validation", split_ids["validation"], validation_targets, dictionary
    )
    hard_data = build_loss_data_with_exact_rank_ceiling(
        root, "hard_test", split_ids["hard_test"], hard_targets, dictionary
    )
    ceiling_rows = []
    for data in (train_data, validation_data, hard_data):
        for row in data["representation_rows"]:
            row = dict(row)
            row["dictionary_rank"] = 516
            row["effective_correction_rank"] = int(row["rank"])
            ceiling_rows.append(row)
    with np.load(
        root
        / "results/results-fd/stages/stage47_fno_complementary_correction"
        / "correction_dictionary.npz"
    ) as archive:
        dictionary96 = np.asarray(archive["basis"], float)
    stage47_ceiling = []
    for split, ids, targets in (
        ("train", split_ids["train"], train_targets),
        ("validation", split_ids["validation"], validation_targets),
        ("hard_test", split_ids["hard_test"], hard_targets),
    ):
        stage47_ceiling.extend(exact_ceiling_rows(root, split, ids, targets, dictionary96))
    old_ceiling_rows = []
    for row in stage47_ceiling:
        row = dict(row)
        row["dictionary_rank"] = 96
        row["effective_correction_rank"] = int(row["rank"])
        old_ceiling_rows.append(row)
    ceiling_summary = _ceiling_summary(old_ceiling_rows, 96) + _ceiling_summary(ceiling_rows, 516)
    _write_csv(output / "dictionary_ceiling_comparison.csv", ceiling_summary)
    _write_csv(output / "full_dictionary_ceiling_by_system.csv", ceiling_rows)
    print("[Stage48] full-516 representation ceiling complete", flush=True)

    model, history, training_summary = train_fno516(
        output, train_data, validation_data, settings
    )
    if history:
        _write_csv(output / "training_history.csv", history)
    _write_json(output / "training_summary.json", training_summary)

    geometry_rows = _read_csv(
        root
        / "results/results-fd/stages/stage47_fno_complementary_correction"
        / "geometry_diagnostics.csv"
    )
    fno_pairs = {}
    for split, ids, targets, data in (
        ("train", split_ids["train"], train_targets, train_data),
        ("validation", split_ids["validation"], validation_targets, validation_data),
        ("hard_test", split_ids["hard_test"], hard_targets, hard_data),
    ):
        local, pairs = evaluate_split_geometry(
            root, split, ids, targets, data, dictionary, model, settings
        )
        for row in local:
            row["family"] = "fno516"
            row["method"] = f"classical_q48_plus_fno516_corr{row['rank']}"
        geometry_rows.extend(local)
        fno_pairs.update(pairs)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)

    solver_rows = evaluate_new_solver(root, output, fno_pairs, settings)
    _write_csv(output / "solver_results.csv", solver_rows)
    methods = (
        "strong_ilu", "teacher_full", "full_span_oracle",
        "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "classical_q48_plus_fno_corr8", "classical_q48_plus_fno_corr12",
        "classical_q48_plus_fno516_corr8", "classical_q48_plus_fno516_corr12",
    )
    solver_summary = summarize_solver(solver_rows, methods)
    _write_csv(output / "solver_summary.csv", solver_summary)

    split_losses = []
    with torch.no_grad():
        for split, data in (
            ("train", train_data), ("validation", validation_data), ("hard_test", hard_data)
        ):
            total, subspace, overlap = _batched_losses(model, data, settings)
            split_losses.append({
                "split": split,
                "mean_total_loss": float(total.mean()),
                "mean_subspace_loss": float(subspace.mean()),
                "mean_complementarity_loss": float(overlap.mean()),
            })
    _write_csv(output / "split_loss_summary.csv", split_losses)

    ceiling_lookup = {
        (int(row["dictionary_rank"]), row["split"], int(row["correction_rank"])): row
        for row in ceiling_summary
    }
    hard_ceiling = ceiling_lookup[(516, "hard_test", 12)]
    sys63_ceiling = next(
        row for row in ceiling_rows
        if row["split"] == "hard_test" and int(row["sample_id"]) == 63
        and int(row["rank"]) == 12
    )
    ceiling_strong = (
        float(hard_ceiling["median_d_W"]) <= 0.35
        and float(hard_ceiling["median_target_containment"]) >= 0.80
        and float(sys63_ceiling["dictionary_ceiling_d_W"]) <= 0.35
    )
    summary_lookup = {row["method"]: row for row in solver_summary}
    fno516 = summary_lookup["classical_q48_plus_fno516_corr12"]
    fno96 = summary_lookup["classical_q48_plus_fno_corr12"]
    system63_rows = [
        row for row in solver_rows
        if int(row["sample_id"]) == 63
        and row["method"] == "classical_q48_plus_fno516_corr12"
    ]
    solver_improves = (
        int(fno516["converged_rhs"]) == 21
        and sum(_bool(row["converged"]) for row in system63_rows) == 3
    )
    old_geometry = [
        row for row in geometry_rows
        if row.get("family") == "fno" and row["split"] == "hard_test"
        and int(row["rank"]) == 12
    ]
    new_geometry = [
        row for row in geometry_rows
        if row.get("family") == "fno516" and row["split"] == "hard_test"
        and int(row["rank"]) == 12
    ]
    geometry_improves = (
        np.median([float(row["correction_d_W"]) for row in new_geometry])
        <= 0.90 * np.median([float(row["correction_d_W"]) for row in old_geometry])
    )
    if not ceiling_strong:
        conclusion = "Outcome C — the full 516-column shared-dictionary ceiling remains poor."
        interpretation = (
            "Removing the rank-96 truncation does not give the held-out complementary targets "
            "a sufficiently accurate rank-12 representation, including system 63. The shared "
            "training-complement dictionary is therefore the limiting formulation."
        )
        recommendation = (
            "Stage 49 should redesign the correction representation so it can form target-local "
            "combinations from the Stage-30 full span; keep the frozen q48 classical branch."
        )
    elif solver_improves:
        conclusion = "Outcome A — the full dictionary removes the ceiling and the FNO succeeds."
        interpretation = (
            "Stage 47 was representation-limited: retaining the full complement dictionary "
            "allows the unchanged FNO to recover system 63 and improve solver performance."
        )
        recommendation = (
            "Stage 49 should keep the q48-plus-FNO516 hybrid and test its cost and broader "
            "cross-system generalization."
        )
    elif geometry_improves:
        conclusion = "Outcome D — FNO geometry improves, but solver performance does not."
        interpretation = (
            "The full dictionary improves predicted correction geometry without recovering the "
            "missing Krylov benefit, so projector loss remains an insufficient solver proxy."
        )
        recommendation = (
            "Stage 49 should keep the full dictionary and frozen q48 branch and test one "
            "solver-aligned correction objective."
        )
    else:
        conclusion = "Outcome B — the full ceiling is strong, but the FNO remains poor."
        interpretation = (
            "The dictionary representation bottleneck is removed, but the unchanged FNO does "
            "not learn the complementary subspace; the remaining issue is neural output "
            "parameterization, conditioning, or optimization."
        )
        recommendation = (
            "Stage 49 should change only the neural formulation by adding one compact Stage-41 "
            "classical-summary conditioning input; keep the full dictionary and q48 branch."
        )
    decision = {
        "conclusion": conclusion,
        "ceiling_strong": ceiling_strong,
        "solver_improves": solver_improves,
        "geometry_improves": geometry_improves,
        "stage47_rank12_converged": int(fno96["converged_rhs"]),
        "stage48_rank12_converged": int(fno516["converged_rhs"]),
        "interpretation": interpretation,
        "stage49_recommendation": recommendation,
    }
    _write_json(output / "decision.json", decision)
    report = _report(
        output, solver_summary, geometry_rows, ceiling_summary, ceiling_rows,
        split_losses, training_summary, dictionary_meta, decision, settings,
    )
    new_geometry_only = [row for row in geometry_rows if row.get("family") == "fno516"]
    new_solver_only = [row for row in solver_rows if "fno516" in row["method"]]
    validation = {
        "preflight_passed": preflight["passed"],
        "dictionary_source_columns": dictionary_meta["source_columns"],
        "dictionary_numerical_rank": dictionary_meta["numerical_rank"],
        "FNO_output_shape": [516, 12],
        "FNO_outputs": 516 * 12,
        "stage35_outputs": 1456 * 48,
        "correction_ranks_exact": all(
            int(row["produced_correction_rank"]) == int(row["rank"])
            for row in new_geometry_only
        ),
        "classical_q48_preserved": min(
            float(row["classical_containment_in_hybrid"])
            for row in new_geometry_only
        ) >= 1.0 - 1.0e-9,
        "maximum_postprojection_overlap": max(
            float(row["postprojection_classical_overlap"])
            for row in new_geometry_only
        ),
        "hard_targets_not_used_for_training_or_checkpointing": True,
        "new_solver_rows": len(new_solver_only),
        "expected_new_solver_rows": 42,
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"])
            for row in solver_rows if _bool(row["converged"])
        ),
    }
    validation["passed"] = bool(
        validation["preflight_passed"]
        and validation["dictionary_source_columns"] == 516
        and validation["dictionary_numerical_rank"] == 516
        and validation["FNO_outputs"] < validation["stage35_outputs"]
        and validation["correction_ranks_exact"]
        and validation["classical_q48_preserved"]
        and validation["maximum_postprojection_overlap"] <= 1.0e-8
        and validation["hard_targets_not_used_for_training_or_checkpointing"]
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["maximum_converged_true_residual"] <= 1.0e-8
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage48_full_dictionary_complementary_correction.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage48] complete", flush=True)
    return copy
