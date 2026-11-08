"""Stage 53: per-system expressivity ceiling of the fixed Stage-52 algebra."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import torch

from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage52_basis_equivariant_raw_compression import (
    EquivariantOperatorModel,
    Stage52Settings,
    evaluate_solver as evaluate_stage52_solver,
    hard_pairs,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage53Settings:
    optimization_seed: int = 20265300
    correction_ranks: Tuple[int, int] = (8, 12)
    random_restarts: int = 5
    maximum_steps: int = 200
    learning_rate: float = 3.0e-2
    least_squares_perturbation: float = 2.0e-2
    algebra_fraction_gate: float = 0.80
    learned_fraction_gate: float = 0.80

    @property
    def stage52(self) -> Stage52Settings:
        return Stage52Settings()


def _load_split(stage52_root: Path, split: str) -> Dict[str, np.ndarray]:
    with np.load(stage52_root / f"{split}_equivariant_data.npz") as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    data["operator_basis"] = np.load(
        stage52_root / f"{split}_operator_basis.npy", mmap_mode="r"
    )
    return data


def _load_stage52_weights(stage52_root: Path, data, settings: Stage53Settings) -> np.ndarray:
    with np.load(stage52_root / "invariant_normalization.npz") as archive:
        mean = np.asarray(archive["mean"], np.float32)
        scale = np.asarray(archive["scale"], np.float32)
    invariants = ((data["invariants"] - mean) / scale).astype(np.float32)
    payload = torch.load(
        stage52_root / "models/basis_equivariant_krylov_only.pt", map_location="cpu"
    )
    model = EquivariantOperatorModel(settings.stage52).to(dtype=torch.float32)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        return model.weights(torch.from_numpy(invariants)).cpu().numpy().astype(float)


def _top_eigenspace(operator_basis: np.ndarray, coefficients: np.ndarray, rank: int):
    operator = np.einsum("k,kij->ij", coefficients, operator_basis, optimize=True)
    operator = 0.5 * (operator + operator.T)
    values, vectors = la.eigh(operator, check_finite=False, driver="evr")
    return vectors[:, -rank:], values


def _coordinate_containment(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(la.norm(target.T @ prediction, ord="fro") ** 2 / target.shape[1])


def _least_squares_coefficients(operator_basis: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Projection of the target projector onto the fixed operator algebra."""

    projector = target @ target.T
    gram = np.einsum("kij,lij->kl", operator_basis, operator_basis, optimize=True)
    right = np.einsum("kij,ij->k", operator_basis, projector, optimize=True)
    ridge = 1.0e-10 * max(float(np.trace(gram)) / gram.shape[0], 1.0)
    return la.solve(gram + ridge * np.eye(gram.shape[0]), right, assume_a="sym")


def optimize_coefficients(
    operator_basis: np.ndarray,
    target: np.ndarray,
    learned_initialization: np.ndarray,
    rank: int,
    seed: int,
    settings: Stage53Settings,
):
    """Optimize six independent starts through the top eigenspace.

    The first start is the frozen Stage-52 output. Five random starts follow;
    one is a small random perturbation of the operator least-squares solution.
    This is still a per-system teacher-informed ceiling and is never deployed.
    """

    basis = np.asarray(operator_basis, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rng = np.random.default_rng(seed)
    starts = np.empty((1 + settings.random_restarts, basis.shape[0]), float)
    starts[0] = np.asarray(learned_initialization, float)
    least_squares = _least_squares_coefficients(basis, target)
    starts[1] = least_squares + settings.least_squares_perturbation * rng.normal(
        size=basis.shape[0]
    )
    starts[2:] = rng.normal(size=(settings.random_restarts - 1, basis.shape[0]))
    norms = la.norm(starts, axis=1)
    invalid = norms <= np.finfo(float).tiny
    starts[invalid] = rng.normal(size=(int(np.count_nonzero(invalid)), basis.shape[0]))

    basis_tensor = torch.from_numpy(basis)
    target_tensor = torch.from_numpy(target)
    alpha = torch.nn.Parameter(torch.from_numpy(starts).to(dtype=torch.float64))
    optimizer = torch.optim.Adam([alpha], lr=settings.learning_rate)
    best_objectives = np.full(alpha.shape[0], -np.inf)
    best_steps = np.zeros(alpha.shape[0], int)
    best_coefficients = starts.copy()
    initial_objectives = None
    started = perf_counter()
    for step in range(settings.maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        normalized = alpha / torch.linalg.vector_norm(
            alpha, dim=1, keepdim=True
        ).clamp_min(1.0e-14)
        operators = torch.einsum("bk,kij->bij", normalized, basis_tensor)
        operators = 0.5 * (operators + operators.transpose(-1, -2))
        _, vectors = torch.linalg.eigh(operators)
        leading = vectors[:, :, -rank:]
        overlap = torch.einsum("di,bdj->bij", target_tensor, leading)
        objectives = torch.sum(overlap**2, dim=(1, 2)) / rank
        values = objectives.detach().cpu().numpy()
        current = normalized.detach().cpu().numpy()
        if initial_objectives is None:
            initial_objectives = values.copy()
        improved = values > best_objectives
        best_objectives[improved] = values[improved]
        best_steps[improved] = step
        best_coefficients[improved] = current[improved]
        if step == settings.maximum_steps:
            break
        loss = -torch.sum(objectives)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite Stage-53 eigenspace objective")
        loss.backward()
        if not torch.all(torch.isfinite(alpha.grad)):
            raise RuntimeError("nonfinite Stage-53 eigenspace gradient")
        optimizer.step()

    winner = int(np.argmax(best_objectives))
    mixing, eigenvalues = _top_eigenspace(basis, best_coefficients[winner], rank)
    eigengap = float(eigenvalues[-rank] - eigenvalues[-rank - 1])
    rows = []
    for restart in range(1 + settings.random_restarts):
        rows.append({
            "restart": restart,
            "initialization": "stage52" if restart == 0 else "random",
            "initial_objective": float(initial_objectives[restart]),
            "best_objective": float(best_objectives[restart]),
            "best_step": int(best_steps[restart]),
            "selected": restart == winner,
        })
    diagnostics = {
        "selected_restart": winner,
        "best_raw_oracle_alignment": float(best_objectives[winner]),
        "final_top_eigengap": eigengap,
        "optimization_seconds": perf_counter() - started,
    }
    return best_coefficients[winner], mixing, rows, diagnostics


def coverage_metrics(target_cross: np.ndarray, mixing: np.ndarray, rank: int) -> Dict[str, float]:
    singular = np.clip(
        la.svdvals(np.asarray(target_cross[:rank], float) @ np.asarray(mixing, float)),
        0.0,
        1.0,
    )
    coverage = singular**2
    return {
        "correction_containment": float(np.mean(coverage)),
        "p10_principal_coverage": float(np.percentile(coverage, 10)),
        "minimum_principal_coverage": float(np.min(coverage)),
        "largest_principal_angle_degrees": float(
            np.degrees(np.arccos(np.clip(np.min(singular), 0.0, 1.0)))
        ),
    }


def _split_cache(output: Path, split: str) -> Path:
    return output / f"{split}_algebra_oracle.npz"


def optimize_split(output: Path, split: str, data, learned_weights, settings: Stage53Settings):
    cache = _split_cache(output, split)
    expected = (len(data["system_ids"]), len(settings.correction_ranks), 192, 12)
    if cache.exists():
        with np.load(cache) as archive:
            predictions = np.asarray(archive["predictions"])
            coefficients = np.asarray(archive["coefficients"])
        if predictions.shape == expected and coefficients.shape[:2] == expected[:2]:
            return predictions, coefficients, [], True

    predictions = np.zeros(expected, np.float32)
    coefficients = np.zeros((expected[0], expected[1], settings.stage52.operator_basis_count), np.float64)
    restart_rows: List[Dict[str, Any]] = []
    for index, sample_value in enumerate(data["system_ids"]):
        sample_id = int(sample_value)
        dimension = int(data["effective_dimensions"][index])
        basis = np.asarray(
            data["operator_basis"][index, :, :dimension, :dimension], dtype=np.float64
        )
        for rank_index, rank in enumerate(settings.correction_ranks):
            target = np.asarray(
                data["oracle_mixings"][index, rank_index, :dimension, :rank], float
            )
            coefficient, mixing, local_rows, diagnostics = optimize_coefficients(
                basis,
                target,
                learned_weights[index, rank_index],
                rank,
                settings.optimization_seed + 10000 * index + 100 * rank + len(split),
                settings,
            )
            coefficients[index, rank_index] = coefficient
            predictions[index, rank_index, :dimension, :rank] = mixing
            for row in local_rows:
                restart_rows.append({
                    "split": split,
                    "sample_id": sample_id,
                    "rank": rank,
                    **row,
                    **diagnostics,
                })
        print(
            f"[Stage53 {split}] {index + 1}/{len(data['system_ids'])} system {sample_id}",
            flush=True,
        )
    np.savez_compressed(cache, predictions=predictions, coefficients=coefficients)
    return predictions, coefficients, restart_rows, False


def learned_mixings(data, learned_weights, settings: Stage53Settings) -> np.ndarray:
    predictions = np.zeros_like(data["oracle_mixings"])
    for index in range(len(data["system_ids"])):
        dimension = int(data["effective_dimensions"][index])
        basis = np.asarray(data["operator_basis"][index, :, :dimension, :dimension], float)
        for rank_index, rank in enumerate(settings.correction_ranks):
            mixing, _ = _top_eigenspace(basis, learned_weights[index, rank_index], rank)
            predictions[index, rank_index, :dimension, :rank] = mixing
    return predictions


def geometry_rows(split, data, algebra_predictions, stage52_predictions, settings: Stage53Settings):
    rows = []
    for index, sample_value in enumerate(data["system_ids"]):
        sample_id = int(sample_value)
        dimension = int(data["effective_dimensions"][index])
        target_cross = np.asarray(data["target_crosses"][index, :, :dimension], float)
        for rank_index, rank in enumerate(settings.correction_ranks):
            for family, prediction in (
                ("raw_oracle", data["oracle_mixings"]),
                ("algebra_oracle", algebra_predictions),
                ("stage52_shared", stage52_predictions),
            ):
                mixing = np.asarray(prediction[index, rank_index, :dimension, :rank], float)
                metrics = coverage_metrics(target_cross, mixing, rank)
                raw_target = np.asarray(
                    data["oracle_mixings"][index, rank_index, :dimension, :rank], float
                )
                rows.append({
                    "split": split,
                    "sample_id": sample_id,
                    "family": family,
                    "rank": rank,
                    "effective_raw_rank": dimension,
                    "raw_oracle_coordinate_alignment": _coordinate_containment(
                        raw_target, mixing
                    ),
                    **metrics,
                })
    return rows


def summarize_geometry(rows):
    summary = []
    for split in ("train", "validation", "hard_test"):
        for rank in (8, 12):
            local_by_family = {
                family: [
                    row for row in rows
                    if row["split"] == split and row["family"] == family
                    and int(row["rank"]) == rank
                ]
                for family in ("raw_oracle", "algebra_oracle", "stage52_shared")
            }
            values = {}
            for family, local in local_by_family.items():
                values[family] = {
                    "containment": float(np.median([row["correction_containment"] for row in local])),
                    "p10": float(np.median([row["p10_principal_coverage"] for row in local])),
                    "minimum": float(np.median([row["minimum_principal_coverage"] for row in local])),
                    "angle": float(np.median([row["largest_principal_angle_degrees"] for row in local])),
                }
            raw = values["raw_oracle"]["containment"]
            algebra = values["algebra_oracle"]["containment"]
            learned = values["stage52_shared"]["containment"]
            for family in values:
                summary.append({
                    "split": split,
                    "family": family,
                    "rank": rank,
                    "median_containment": values[family]["containment"],
                    "median_p10": values[family]["p10"],
                    "median_minimum": values[family]["minimum"],
                    "median_largest_angle": values[family]["angle"],
                    "delta_algebra": raw - algebra,
                    "delta_learning": algebra - learned,
                    "algebra_fraction": algebra / max(raw, np.finfo(float).tiny),
                    "learned_fraction": learned / max(algebra, np.finfo(float).tiny),
                })
    return summary


def _hard_fraction_rows(rows, settings: Stage53Settings):
    result = []
    for sample_id in settings.stage52.hard_system_ids:
        for rank in settings.correction_ranks:
            lookup = {
                row["family"]: row for row in rows
                if row["split"] == "hard_test" and int(row["sample_id"]) == sample_id
                and int(row["rank"]) == rank
            }
            raw = float(lookup["raw_oracle"]["correction_containment"])
            algebra = float(lookup["algebra_oracle"]["correction_containment"])
            learned = float(lookup["stage52_shared"]["correction_containment"])
            result.append({
                "sample_id": sample_id,
                "rank": rank,
                "raw_containment": raw,
                "algebra_containment": algebra,
                "stage52_containment": learned,
                "algebra_fraction": algebra / max(raw, np.finfo(float).tiny),
                "learned_fraction": learned / max(algebra, np.finfo(float).tiny),
                "raw_p10": lookup["raw_oracle"]["p10_principal_coverage"],
                "algebra_p10": lookup["algebra_oracle"]["p10_principal_coverage"],
                "stage52_p10": lookup["stage52_shared"]["p10_principal_coverage"],
                "raw_minimum": lookup["raw_oracle"]["minimum_principal_coverage"],
                "algebra_minimum": lookup["algebra_oracle"]["minimum_principal_coverage"],
                "stage52_minimum": lookup["stage52_shared"]["minimum_principal_coverage"],
                "algebra_largest_angle": lookup["algebra_oracle"]["largest_principal_angle_degrees"],
            })
    return result


def decide(summary, hard_rows, settings: Stage53Settings):
    summary_lookup = {
        (row["split"], int(row["rank"])): row
        for row in summary if row["family"] == "algebra_oracle"
    }
    train_strong = all(
        summary_lookup[("train", rank)]["algebra_fraction"] >= settings.algebra_fraction_gate
        for rank in settings.correction_ranks
    )
    hard_strong = all(
        summary_lookup[("hard_test", rank)]["algebra_fraction"] >= settings.algebra_fraction_gate
        for rank in settings.correction_ranks
    )
    system63 = [row for row in hard_rows if int(row["sample_id"]) == 63]
    system63_strong = all(
        row["algebra_fraction"] >= settings.algebra_fraction_gate for row in system63
    )
    learned_far = any(
        summary_lookup[(split, rank)]["learned_fraction"] < settings.learned_fraction_gate
        for split in ("train", "hard_test") for rank in settings.correction_ranks
    )
    if train_strong and hard_strong and system63_strong:
        if learned_far:
            outcome = "Outcome D"
            conclusion = "The fixed algebra is strong, but Stage 52 remains far below its per-system ceiling."
        else:
            outcome = "Outcome A"
            conclusion = "The fixed 33-operator algebra approaches the raw-space oracle."
        bottleneck = "coefficient inference"
        recommendation = (
            "Stage 54 should characterize the per-system oracle coefficient map—its identifiability, "
            "smoothness, and predictability from the existing invariant features—before selecting another predictor."
        )
    elif train_strong and (not hard_strong or not system63_strong):
        outcome = "Outcome C"
        conclusion = "The algebra works on training systems but is not robust on the hard systems."
        bottleneck = "algebra expressivity"
        recommendation = (
            "Stage 54 should identify which equivariant operator interactions are missing on hard systems, "
            "without training another coefficient predictor."
        )
    else:
        outcome = "Outcome B"
        conclusion = "The per-system oracle remains substantially below the raw-space oracle."
        bottleneck = "algebra expressivity"
        recommendation = (
            "Stage 54 should redesign the equivariant operator family using a focused analysis of the "
            "raw-oracle projector components absent from the current 33-operator span."
        )
    return {
        "outcome": outcome,
        "conclusion": conclusion,
        "dominant_bottleneck": bottleneck,
        "training_algebra_strong": train_strong,
        "hard_algebra_strong": hard_strong,
        "system63_algebra_strong": system63_strong,
        "stage52_far_below_algebra": learned_far,
        "solver_gate_passed": hard_strong,
        "stage54_recommendation": recommendation,
    }


def _solver_summary(rows):
    result = []
    methods = [
        "stage51_raw_oracle_r8", "stage51_raw_oracle_r12",
        "stage53_algebra_oracle_r8", "stage53_algebra_oracle_r12",
    ]
    for method in methods:
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


def _report(output, decision, summary, hard_rows, restart_rows, solver_rows, solve_summary):
    labels = {
        "raw_oracle": "Stage-51 raw-space oracle",
        "algebra_oracle": "Stage-53 per-system algebra oracle",
        "stage52_shared": "Stage-52 shared model",
    }
    lines = [
        "# Stage 53: Per-system oracle ceiling of the fixed equivariant operator algebra",
        "", "## Decision", "",
        f"**{decision['outcome']} — {decision['conclusion']}**", "",
        f"**Dominant bottleneck: {decision['dominant_bottleneck']}.**", "",
        decision["stage54_recommendation"], "",
        "All Stage-41/51 candidates, targets, probes, q48 spaces, ranks, splits, and all 33 cached Stage-52 symmetric operators were frozen. K(x) was not used and no network was trained. Each system and rank used one frozen Stage-52 initialization plus exactly five deterministic random initializations; the best raw-oracle coordinate alignment after 200 Adam steps was retained.",
        "", "## Ceiling hierarchy", "",
        "Containment is the primary metric. The final four columns quantify the same lower-tail geometry.",
        "", "| split | method | r | containment | p10 | min | largest angle |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['split']} | {labels[row['family']]} | {row['rank']} | "
            f"{_fmt(row['median_containment'])} | {_fmt(row['median_p10'])} | "
            f"{_fmt(row['median_minimum'])} | {_fmt(row['median_largest_angle'])} |"
        )
    lines += [
        "", "## Failure decomposition", "",
        "| split | r | algebra gap | learning gap | algebra/raw | Stage52/algebra |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in summary:
        if row["family"] != "algebra_oracle":
            continue
        lines.append(
            f"| {row['split']} | {row['rank']} | {_fmt(row['delta_algebra'])} | "
            f"{_fmt(row['delta_learning'])} | {_fmt(row['algebra_fraction'])} | "
            f"{_fmt(row['learned_fraction'])} |"
        )
    lines += [
        "", "The algebra gap is raw-oracle minus algebra-oracle containment. The learning gap is algebra-oracle minus Stage-52 containment. Fractions use the corresponding upstream ceiling as denominator.",
        "", "## Seven hard systems", "",
        "| system | r | raw C (p10/min) | algebra C (p10/min) | Stage52 C (p10/min) | algebra/raw | angle |",
        "|--:|--:|:--|:--|:--|--:|--:|",
    ]
    for row in hard_rows:
        lines.append(
            f"| {row['sample_id']} | {row['rank']} | {_fmt(row['raw_containment'])} "
            f"({_fmt(row['raw_p10'])}/{_fmt(row['raw_minimum'])}) | "
            f"{_fmt(row['algebra_containment'])} ({_fmt(row['algebra_p10'])}/"
            f"{_fmt(row['algebra_minimum'])}) | {_fmt(row['stage52_containment'])} "
            f"({_fmt(row['stage52_p10'])}/{_fmt(row['stage52_minimum'])}) | "
            f"{_fmt(row['algebra_fraction'])} | {_fmt(row['algebra_largest_angle'])} |"
        )
    lines += ["", "### System 63", ""]
    for row in hard_rows:
        if int(row["sample_id"]) != 63:
            continue
        lines.append(
            f"- r={row['rank']}: raw containment `{_fmt(row['raw_containment'])}`; algebra "
            f"`{_fmt(row['algebra_containment'])}` ({_fmt(row['algebra_fraction'])} of raw), "
            f"p10 `{_fmt(row['algebra_p10'])}`, minimum `{_fmt(row['algebra_minimum'])}`, "
            f"largest angle `{_fmt(row['algebra_largest_angle'])}`; Stage-52 containment "
            f"`{_fmt(row['stage52_containment'])}`."
        )
    selected = [row for row in restart_rows if _bool(row["selected"])]
    lines += [
        "", "## Optimization robustness", "",
        f"The run recorded `{len(restart_rows)}` restart outcomes. Across the `{len(selected)}` selected system/rank fits, "
        f"the median best raw-oracle coordinate alignment was `{_fmt(np.median([float(row['best_raw_oracle_alignment']) for row in selected]))}`. "
        f"A Stage-52 initialization won `{sum(int(row['selected_restart']) == 0 for row in selected)}` fits; a random initialization won the remaining `{sum(int(row['selected_restart']) != 0 for row in selected)}`.",
    ]
    if solve_summary:
        lines += [
            "", "## Optional solver ceiling", "",
            "The predeclared hard-test containment gate passed, so the unchanged recycled-GMRES ceiling was evaluated.",
            "", "| method | converged / 21 | median iterations | median rho | final rank |",
            "|:--|--:|--:|--:|--:|",
        ]
        for row in solve_summary:
            lines.append(
                f"| {row['method']} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'])} | "
                f"{_fmt(row['median_rho'])} | {_fmt(row['median_final_rank'])} |"
            )
        lines += ["", "System 63:", ""]
        for rank in (8, 12):
            local = [
                row for row in solver_rows if row["method"] == f"stage53_algebra_oracle_r{rank}"
                and int(row["sample_id"]) == 63
            ]
            lines.append(
                f"- r={rank}: {sum(_bool(row['converged']) for row in local)}/3, median "
                f"{int(np.median([int(row['iterations']) for row in local]))}, rho "
                f"{_fmt(np.median([float(row['rho']) for row in local]))}."
            )
    else:
        lines += [
            "", "## Optional solver ceiling", "",
            "The solver was not run because the Stage-53 algebra oracle did not reach 80% of raw-oracle median containment on the hard split for both ranks.",
        ]
    lines += [
        "", "## Stage-54 recommendation", "", decision["stage54_recommendation"], "",
        f"The dominant bottleneck is **{decision['dominant_bottleneck']}**.",
    ]
    report = output / "stage53_equivariant_algebra_oracle_ceiling.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage53(root: Path) -> Path:
    settings = Stage53Settings()
    output = root / "results/results-fd/stages/stage53_equivariant_algebra_oracle_ceiling"
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "predeclared_protocol.json"
    if not protocol_path.exists():
        raise RuntimeError("Stage-53 predeclared protocol is missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform(),
    })
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    stage52_root = root / "results/results-fd/stages/stage52_basis_equivariant_raw_compression"
    validation52 = json.loads((stage52_root / "validation.json").read_text())
    data_by_split = {
        split: _load_split(stage52_root, split)
        for split in ("train", "validation", "hard_test")
    }
    preflight = {
        "stage52_validation_passed": bool(validation52["passed"]),
        "cached_operator_count_exact": all(
            data["operator_basis"].shape[1] == 33 for data in data_by_split.values()
        ),
        "cached_splits_exact": [
            len(data_by_split[split]["system_ids"])
            for split in ("train", "validation", "hard_test")
        ] == [43, 15, 7],
        "ranks_exact": settings.correction_ranks == (8, 12),
        "random_restarts_exact": settings.random_restarts == 5,
        "no_neural_training": True,
        "no_K_input": True,
        "no_probe_generation": True,
        "fixed_operator_algebra": True,
    }
    preflight["passed"] = all(preflight.values())
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-53 preflight failed")

    geometry, restart_rows, predictions_by_split = [], [], {}
    coefficients_by_split = {}
    cache_reused = {}
    for split, data in data_by_split.items():
        weights = _load_stage52_weights(stage52_root, data, settings)
        algebra, coefficients, local_restarts, reused = optimize_split(
            output, split, data, weights, settings
        )
        stage52_predictions = learned_mixings(data, weights, settings)
        predictions_by_split[split] = algebra
        coefficients_by_split[split] = coefficients
        restart_rows.extend(local_restarts)
        cache_reused[split] = reused
        geometry.extend(geometry_rows(split, data, algebra, stage52_predictions, settings))
    if restart_rows:
        _write_csv(output / "optimization_restarts.csv", restart_rows)
    elif (output / "optimization_restarts.csv").exists():
        restart_rows = _read_csv(output / "optimization_restarts.csv")
    else:
        raise RuntimeError("Stage-53 restart diagnostics are missing")
    _write_csv(output / "geometry_diagnostics.csv", geometry)
    summary = summarize_geometry(geometry)
    _write_csv(output / "geometry_summary.csv", summary)
    hard_rows = _hard_fraction_rows(geometry, settings)
    _write_csv(output / "hard_system_ceiling.csv", hard_rows)
    decision = decide(summary, hard_rows, settings)

    solver_rows: List[Dict[str, Any]] = []
    solve_summary: List[Dict[str, Any]] = []
    pair_diagnostics: List[Dict[str, Any]] = []
    if decision["solver_gate_passed"]:
        pairs, pair_diagnostics = hard_pairs(
            root, data_by_split["hard_test"], predictions_by_split["hard_test"], settings.stage52
        )
        solver_rows = evaluate_stage52_solver(root, pairs, settings.stage52)
        for row in solver_rows:
            if row["method"].startswith("stage52_basis_equivariant_r"):
                row["method"] = row["method"].replace(
                    "stage52_basis_equivariant", "stage53_algebra_oracle"
                )
                row["source"] = "Stage53 nondeployable per-system algebra ceiling"
        solve_summary = _solver_summary(solver_rows)
        _write_csv(output / "solver_results.csv", solver_rows)
        _write_csv(output / "solver_summary.csv", solve_summary)
        _write_csv(output / "pair_diagnostics.csv", pair_diagnostics)

    _write_json(output / "decision.json", decision)
    report = _report(
        output, decision, summary, hard_rows, restart_rows, solver_rows, solve_summary
    )
    expected_fits = (43 + 15 + 7) * 2
    cached_stage52_geometry = _read_csv(stage52_root / "geometry_diagnostics.csv")
    cached_lookup = {
        (row["split"], int(row["sample_id"]), int(row["rank"])): row
        for row in cached_stage52_geometry
    }
    reproduction_errors = [
        abs(
            float(row["correction_containment"])
            - float(cached_lookup[(row["split"], int(row["sample_id"]), int(row["rank"]))]["correction_containment"])
        )
        for row in geometry if row["family"] == "stage52_shared"
    ]
    validation = {
        "preflight_passed": preflight["passed"],
        "cached_stage52_operators_reused": True,
        "operator_basis_count": 33,
        "fit_count": expected_fits,
        "expected_fit_count": expected_fits,
        "restart_rows": len(restart_rows),
        "expected_restart_rows": expected_fits * (1 + settings.random_restarts),
        "stage52_initialization_rows": sum(
            str(row["initialization"]) == "stage52" for row in restart_rows
        ),
        "no_neural_training": True,
        "no_K_input": True,
        "no_new_probes": True,
        "maximum_stage52_containment_reproduction_error": max(reproduction_errors),
        "solver_gate_passed": decision["solver_gate_passed"],
        "solver_evaluation_run_iff_gate_passed": bool(solver_rows) == bool(decision["solver_gate_passed"]),
        "maximum_pair_defect": (
            max(float(row["pair_defect"]) for row in pair_diagnostics)
            if pair_diagnostics else float("nan")
        ),
        "split_cache_reused": cache_reused,
    }
    validation["passed"] = bool(
        validation["preflight_passed"]
        and validation["cached_stage52_operators_reused"]
        and validation["operator_basis_count"] == 33
        and validation["fit_count"] == validation["expected_fit_count"]
        and validation["restart_rows"] == validation["expected_restart_rows"]
        and validation["stage52_initialization_rows"] == expected_fits
        and validation["no_neural_training"] and validation["no_K_input"]
        and validation["no_new_probes"]
        and validation["maximum_stage52_containment_reproduction_error"] <= 1.0e-6
        and validation["solver_evaluation_run_iff_gate_passed"]
        and (not pair_diagnostics or validation["maximum_pair_defect"] <= 1.0e-8)
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    report_copy = root / "results/results-fd/reports/stage53_equivariant_algebra_oracle_ceiling.md"
    report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage53] complete", flush=True)
    return report_copy
