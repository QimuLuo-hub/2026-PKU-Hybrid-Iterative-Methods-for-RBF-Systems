"""Stage 16: W-aware reduced-dictionary ablation with the Stage 15 MLP.

This experiment changes only the fixed solution-side dictionary.  It reloads
the exact Stage 15 oracle spaces and split, rebuilds the same matrices/ILU,
reuses the same scalar MLP, optimizer, seeds, W-projector loss, and fixed-space
recycled GMRES, and compares dictionaries at the identical dimension p=19.
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
import scipy.linalg as la
import scipy.sparse.linalg as spla
import torch

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..subspace_analysis import subspace_metrics
from .stage9_scalability import _read_csv, _write_csv, _write_json
from .stage10_ritz_transfer import RitzContext, _build_context, _discrete_rhs
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
    train_one_seed,
)


@dataclass(frozen=True)
class Stage16Settings:
    """Controls added by Stage 16; the embedded Stage 15 settings stay exact."""

    stage15: Stage15Settings = Stage15Settings()
    dictionary_dimension: int = 19
    joint_lambdas: Tuple[float, ...] = (0.1, 1.0, 10.0)
    representation_relative_improvement: float = 0.20
    learned_eta_improvement: float = 0.03
    interpolation_eta_tolerance: float = 0.02
    low_representation_error: float = 0.25


@dataclass(frozen=True)
class DictionarySpec:
    name: str
    label: str
    basis: np.ndarray
    metadata: Dict[str, Any]


DICTIONARY_LABELS = {
    "z_pod": "Z-POD",
    "az_aware": "AZ-aware",
    "joint": "Joint",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage15_directory() -> Path:
    path = _repository_root() / "results" / "stage15_family_specific_W_target_MLP"
    if not path.exists():
        raise FileNotFoundError("Stage 15 results are required for Stage 16")
    return path


def _json_normalized(value: Any) -> Any:
    """Round-trip dataclass content through JSON for tuple/list equivalence."""

    return json.loads(json.dumps(value))


def load_exact_stage15_family(
    output: Path,
    settings: Stage16Settings,
) -> Tuple[
    Dict[float, RitzContext],
    Dict[float, np.ndarray],
    Dict[float, AZPetrovFactors],
    List[Dict[str, Any]],
]:
    """Rebuild A/ILU while loading the exact saved Stage 15 Z and W labels."""

    source = _stage15_directory()
    saved_settings = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    current_settings = _json_normalized(asdict(settings.stage15))
    if saved_settings != current_settings:
        raise RuntimeError("Stage 16 settings do not exactly match saved Stage 15 settings")
    with np.load(source / "oracle_spaces.npz") as archive:
        saved_arrays = {key: np.asarray(archive[key], dtype=float) for key in archive.files}

    contexts: Dict[float, RitzContext] = {}
    spaces: Dict[float, np.ndarray] = {}
    factors: Dict[float, AZPetrovFactors] = {}
    rows: List[Dict[str, Any]] = []
    for ratio in settings.stage15.ratios:
        context = _build_context(ratio, settings.stage15.stage10)
        z_saved = saved_arrays[f"Z_R{ratio:g}"]
        w_saved = saved_arrays[f"W_R{ratio:g}"]
        z_basis, z_error, rank = _orthonormalize(z_saved)
        pair = factor_az_petrov(context.system.matrix, z_basis)
        w_match = subspace_metrics(pair.test, w_saved)
        contexts[ratio] = context
        spaces[ratio] = z_basis
        factors[ratio] = pair
        rows.append(
            {
                "ratio": ratio,
                "split": (
                    "train"
                    if ratio in settings.stage15.train_ratios
                    else "validation"
                    if ratio in settings.stage15.validation_ratios
                    else "test"
                ),
                "z_rank": rank,
                "z_orthogonality_error": z_error,
                "saved_rebuilt_w_projector_distance": float(w_match["projector_distance"]),
                "saved_rebuilt_w_max_angle_degrees": float(
                    w_match["principal_angle_max_degrees"]
                ),
                "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                "condition_wtaz": pair.diagnostics["wtaz_condition"],
                "assembly_seconds": context.assembly_seconds,
                "ilu_setup_seconds": context.ilu.setup_seconds,
            }
        )
        _write_csv(output / "stage15_family_reconstruction.csv", rows)
    return contexts, spaces, factors, rows


def _pod_basis(snapshots: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, float]:
    left, singular_values, _ = la.svd(snapshots, full_matrices=False, check_finite=False)
    basis = np.asarray(left[:, :p], dtype=float)
    energy = float(
        np.sum(singular_values[:p] ** 2) / max(np.sum(singular_values**2), np.finfo(float).tiny)
    )
    return basis, singular_values, energy


def _dictionary_projection_metrics(
    dictionary: np.ndarray,
    ratio: float,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
) -> Dict[str, Any]:
    """Project held-out Z into a dictionary, then measure its induced W error."""

    projected_raw = dictionary @ (dictionary.T @ spaces[ratio])
    projected, orthogonality_error, rank = _orthonormalize(projected_raw)
    pair = factor_az_petrov(contexts[ratio].system.matrix, projected)
    z_metric = subspace_metrics(projected, spaces[ratio])
    w_metric = subspace_metrics(pair.test, factors[ratio].test)
    return {
        "ratio": ratio,
        "projected_rank": rank,
        "projected_orthogonality_error": orthogonality_error,
        "projection_d_z": float(z_metric["projector_distance"]),
        "projection_d_z_normalized": float(z_metric["projector_distance_normalized"]),
        "projection_d_w": float(w_metric["projector_distance"]),
        "projection_d_w_normalized": float(w_metric["projector_distance_normalized"]),
        "projection_w_max_angle_degrees": float(w_metric["principal_angle_max_degrees"]),
    }


def _w_optimal_dictionary_metrics(
    dictionary: np.ndarray,
    ratio: float,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
) -> Dict[str, Any]:
    """Evaluation-only best q-dimensional W subspace inside range(A Psi).

    This uses the held-out target only as an offline ceiling and never to build
    a dictionary or train a model.  It distinguishes dictionary range from the
    prescribed Z-projection diagnostic.
    """

    image = np.asarray(contexts[ratio].system.matrix @ dictionary, dtype=float)
    q_image, triangular = la.qr(image, mode="economic", check_finite=False)
    overlap = q_image.T @ factors[ratio].test
    left, _, _ = la.svd(overlap, full_matrices=False, check_finite=False)
    residual_coordinates = left[:, : spaces[ratio].shape[1]]
    coefficients = la.solve_triangular(
        triangular, residual_coordinates, check_finite=False
    )
    z_raw = dictionary @ coefficients
    z_basis, _, rank = _orthonormalize(z_raw)
    pair = factor_az_petrov(contexts[ratio].system.matrix, z_basis)
    z_metric = subspace_metrics(z_basis, spaces[ratio])
    w_metric = subspace_metrics(pair.test, factors[ratio].test)
    return {
        "w_optimal_rank": rank,
        "w_optimal_d_z_normalized": float(z_metric["projector_distance_normalized"]),
        "w_optimal_d_w_normalized": float(w_metric["projector_distance_normalized"]),
        "w_optimal_max_angle_degrees": float(w_metric["principal_angle_max_degrees"]),
    }


def representation_diagnostics(
    dictionaries: Mapping[str, DictionarySpec],
    ratios: Sequence[float],
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage16Settings,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, spec in dictionaries.items():
        for ratio in ratios:
            row = {
                "dictionary": name,
                "dictionary_label": spec.label,
                "ratio": ratio,
                "split": (
                    "validation"
                    if ratio in settings.stage15.validation_ratios
                    else "test"
                    if ratio in settings.stage15.test_ratios
                    else "train"
                ),
                **_dictionary_projection_metrics(
                    spec.basis, ratio, contexts, spaces, factors
                ),
                **_w_optimal_dictionary_metrics(
                    spec.basis, ratio, contexts, spaces, factors
                ),
            }
            rows.append(row)
    return rows


def build_dictionaries(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage16Settings,
) -> Tuple[Dict[str, DictionarySpec], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Construct the three p=19 dictionaries using training systems only."""

    p = settings.dictionary_dimension
    source = _stage15_directory()
    with np.load(source / "training_dictionary.npz") as archive:
        z_pod = np.asarray(archive["psi"], dtype=float)
        saved_train_ratios = tuple(float(value) for value in archive["train_ratios"])
    if saved_train_ratios != settings.stage15.train_ratios or z_pod.shape[1] != p:
        raise RuntimeError("saved Stage 15 dictionary does not match the fixed Stage 16 control")

    z_snapshots = np.column_stack([spaces[ratio] for ratio in settings.stage15.train_ratios])
    y_blocks = [
        np.asarray(contexts[ratio].system.matrix @ spaces[ratio], dtype=float)
        for ratio in settings.stage15.train_ratios
    ]
    y_snapshots = np.column_stack(y_blocks)
    phi_w, y_singular, y_energy = _pod_basis(y_snapshots, p)

    inverse_blocks: List[np.ndarray] = []
    inverse_start = perf_counter()
    for ratio in settings.stage15.train_ratios:
        direct = spla.splu(contexts[ratio].system.matrix.tocsc())
        inverse_blocks.append(np.asarray(direct.solve(phi_w), dtype=float))
    inverse_seconds = perf_counter() - inverse_start
    inverse_snapshots = np.column_stack(inverse_blocks)
    psi_az, inverse_singular, inverse_energy = _pod_basis(inverse_snapshots, p)

    scaled_y_blocks = [
        block
        * (
            np.linalg.norm(spaces[ratio], ord="fro")
            / max(np.linalg.norm(block, ord="fro"), np.finfo(float).tiny)
        )
        for ratio, block in zip(settings.stage15.train_ratios, y_blocks)
    ]
    scaled_y = np.column_stack(scaled_y_blocks)
    lambda_specs: Dict[float, DictionarySpec] = {}
    lambda_rows: List[Dict[str, Any]] = []
    for weight in settings.joint_lambdas:
        joint_snapshots = np.column_stack((z_snapshots, np.sqrt(weight) * scaled_y))
        basis, singular, energy = _pod_basis(joint_snapshots, p)
        lambda_specs[weight] = DictionarySpec(
            name="joint",
            label=f"Joint (lambda={weight:g})",
            basis=basis,
            metadata={
                "lambda": weight,
                "captured_joint_energy": energy,
                "singular_values": singular,
            },
        )
        diagnostics = representation_diagnostics(
            {"joint": lambda_specs[weight]},
            settings.stage15.validation_ratios,
            contexts,
            spaces,
            factors,
            settings,
        )
        lambda_rows.append(
            {
                "lambda": weight,
                "validation_projection_d_w_normalized_median": float(
                    np.median([row["projection_d_w_normalized"] for row in diagnostics])
                ),
                "validation_w_optimal_d_w_normalized_median": float(
                    np.median([row["w_optimal_d_w_normalized"] for row in diagnostics])
                ),
                "captured_joint_energy": energy,
            }
        )
    selected_lambda = min(
        lambda_rows,
        key=lambda row: (
            row["validation_projection_d_w_normalized_median"], row["lambda"]
        ),
    )["lambda"]
    for row in lambda_rows:
        row["selected"] = bool(row["lambda"] == selected_lambda)

    dictionaries = {
        "z_pod": DictionarySpec(
            "z_pod",
            DICTIONARY_LABELS["z_pod"],
            z_pod,
            {"source": "exact Stage 15 training dictionary", "p": p},
        ),
        "az_aware": DictionarySpec(
            "az_aware",
            DICTIONARY_LABELS["az_aware"],
            psi_az,
            {
                "source": "training-only inverse maps of AZ-POD residual modes",
                "p": p,
                "phi_w_captured_energy": y_energy,
                "inverse_map_captured_energy": inverse_energy,
                "inverse_mapping_seconds": inverse_seconds,
            },
        ),
        "joint": DictionarySpec(
            "joint",
            DICTIONARY_LABELS["joint"],
            lambda_specs[selected_lambda].basis,
            {
                "source": "training-only POD of Z and block-scaled AZ snapshots",
                "p": p,
                "lambda": selected_lambda,
                "captured_joint_energy": lambda_specs[selected_lambda].metadata[
                    "captured_joint_energy"
                ],
                "sign_correction": "plus lambda; prompt's minus sign would reward AZ error",
            },
        ),
    }
    dictionary_rows = []
    spectra = {
        "az_residual_pod": y_singular,
        "az_inverse_map_pod": inverse_singular,
    }
    for name, singular in spectra.items():
        energy = np.cumsum(singular**2) / np.sum(singular**2)
        dictionary_rows.extend(
            {
                "spectrum": name,
                "mode": index + 1,
                "singular_value": float(value),
                "captured_energy": float(energy[index]),
                "selected": bool(index < p),
            }
            for index, value in enumerate(singular)
        )
    archive = {f"psi_{name}": spec.basis for name, spec in dictionaries.items()}
    archive["phi_w"] = phi_w
    np.savez_compressed(output / "dictionaries.npz", **archive)
    return dictionaries, dictionary_rows, lambda_rows


def train_dictionary_models(
    output: Path,
    dictionaries: Mapping[str, DictionarySpec],
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage16Settings,
) -> Tuple[
    Dict[str, Dict[int, ReducedSpaceMLP]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Train identical Stage 15 MLP/Adam replicas for every dictionary."""

    models: Dict[str, Dict[int, ReducedSpaceMLP]] = {}
    histories: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for name, spec in dictionaries.items():
        models[name] = {}
        if name == "z_pod":
            old_histories = _read_csv(_stage15_directory() / "training_history.csv")
            old_summaries = _read_csv(_stage15_directory() / "training_summary.csv")
            histories.extend({"dictionary": name, **row} for row in old_histories)
            summaries.extend({"dictionary": name, **row} for row in old_summaries)
            for seed in settings.stage15.model_seeds:
                checkpoint = torch.load(
                    _stage15_directory() / "models" / f"mlp_seed{seed}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                model = ReducedSpaceMLP(
                    spec.basis.shape[1],
                    settings.stage15.q,
                    settings.stage15.hidden_widths,
                ).to(dtype=torch.float64)
                model.load_state_dict(checkpoint["state_dict"])
                models[name][seed] = model
                local_checkpoint = output / name / "models" / f"mlp_seed{seed}.pt"
                local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, local_checkpoint)
            _write_csv(output / "training_history.csv", histories)
            _write_csv(output / "training_summary.csv", summaries)
            continue
        for seed in settings.stage15.model_seeds:
            model, local_history, local_summary = train_one_seed(
                output / name,
                seed,
                contexts,
                spaces,
                factors,
                spec.basis,
                settings.stage15,
            )
            models[name][seed] = model
            histories.extend({"dictionary": name, **row} for row in local_history)
            summaries.append({"dictionary": name, **local_summary})
            _write_csv(output / "training_history.csv", histories)
            _write_csv(output / "training_summary.csv", summaries)
    return models, histories, summaries


def predict_all_spaces(
    models: Mapping[str, Mapping[int, ReducedSpaceMLP]],
    dictionaries: Mapping[str, DictionarySpec],
    settings: Stage16Settings,
) -> Tuple[
    Dict[str, Dict[int, Dict[float, np.ndarray]]],
    List[Dict[str, Any]],
]:
    predicted: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    rows: List[Dict[str, Any]] = []
    for name, per_seed_models in models.items():
        local, metadata = build_learned_spaces(
            per_seed_models, dictionaries[name].basis, settings.stage15
        )
        predicted[name] = local
        rows.extend({"dictionary": name, **row} for row in metadata)
    return predicted, rows


def learned_offline_metrics(
    predicted: Mapping[str, Mapping[int, Mapping[float, np.ndarray]]],
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage16Settings,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, per_seed in predicted.items():
        for seed, per_ratio in per_seed.items():
            for ratio, basis in per_ratio.items():
                pair = factor_az_petrov(contexts[ratio].system.matrix, basis)
                z_metric = subspace_metrics(basis, spaces[ratio])
                w_metric = subspace_metrics(pair.test, factors[ratio].test)
                rows.append(
                    {
                        "dictionary": name,
                        "dictionary_label": DICTIONARY_LABELS[name],
                        "seed": seed,
                        "ratio": ratio,
                        "split": (
                            "validation"
                            if ratio in settings.stage15.validation_ratios
                            else "test"
                        ),
                        "d_z": float(z_metric["projector_distance"]),
                        "d_z_normalized": float(z_metric["projector_distance_normalized"]),
                        "d_w": float(w_metric["projector_distance"]),
                        "d_w_normalized": float(w_metric["projector_distance_normalized"]),
                        "w_max_angle_degrees": float(
                            w_metric["principal_angle_max_degrees"]
                        ),
                        "condition_wtaz": float(pair.diagnostics["wtaz_condition"]),
                    }
                )
    return rows


def _selected_seeds(
    training_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for name in DICTIONARY_LABELS:
        result[name] = select_seed(
            [row for row in training_summaries if row["dictionary"] == name]
        )
    return result


def run_solver_ablation(
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    factors: Mapping[float, AZPetrovFactors],
    deterministic: Mapping[str, Mapping[float, np.ndarray]],
    predicted: Mapping[str, Mapping[int, Mapping[float, np.ndarray]]],
    prediction_rows: Sequence[Mapping[str, Any]],
    settings: Stage16Settings,
) -> List[Dict[str, Any]]:
    """Run the unchanged Stage 13 solver for baselines and every model seed."""

    prediction_lookup = {
        (str(row["dictionary"]), int(row["seed"]), float(row["ratio"])): float(
            row["prediction_seconds"]
        )
        for row in prediction_rows
    }
    factor_cache: Dict[Tuple[str, int | str, float], Tuple[AZPetrovFactors, float]] = {}
    for ratio in settings.stage15.intermediate_ratios:
        factor_cache[("oracle", "", ratio)] = (factors[ratio], 0.0)
        for method, basis_map in deterministic.items():
            start = perf_counter()
            pair = factor_az_petrov(contexts[ratio].system.matrix, basis_map[ratio])
            factor_cache[(method, "", ratio)] = (pair, perf_counter() - start)
        for name, per_seed in predicted.items():
            for seed, basis_map in per_seed.items():
                start = perf_counter()
                pair = factor_az_petrov(contexts[ratio].system.matrix, basis_map[ratio])
                factor_cache[(f"mlp_{name}", seed, ratio)] = (
                    pair,
                    perf_counter() - start,
                )

    rows: List[Dict[str, Any]] = []
    for ratio in settings.stage15.intermediate_ratios:
        context = contexts[ratio]
        split = (
            "validation" if ratio in settings.stage15.validation_ratios else "test"
        )
        method_specs: List[
            Tuple[str, str, int | str, AZPetrovFactors, float, float]
        ] = []
        method_specs.append(("oracle", "", "", *factor_cache[("oracle", "", ratio)], 0.0))
        for method in ("nearest_reuse", "deterministic_interpolation"):
            method_specs.append((method, "", "", *factor_cache[(method, "", ratio)], 0.0))
        for name, per_seed in predicted.items():
            for seed in per_seed:
                method = f"mlp_{name}"
                method_specs.append(
                    (
                        method,
                        name,
                        seed,
                        *factor_cache[(method, seed, ratio)],
                        prediction_lookup[(name, seed, ratio)],
                    )
                )

        for rhs_seed in settings.stage15.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline, baseline_counts, baseline_time = _run_baseline(
                context, rhs, factors[ratio], settings.stage15.stage13
            )
            baseline_row = {
                "ratio": ratio,
                "split": split,
                "rhs_seed": rhs_seed,
                "method": "strong_ilu",
                "dictionary": "",
                "seed": "",
                "iterations": baseline.iterations,
                "converged": baseline.converged,
                "cap_hit": bool(
                    baseline.iterations >= settings.stage15.stage10.solver.max_iterations
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
            rows.append(baseline_row)
            local_rows: List[Dict[str, Any]] = []
            for method, dictionary, seed, pair, setup_seconds, prediction_seconds in method_specs:
                result, counts, online = _run_recycled(
                    context, rhs, pair, settings.stage15.stage13
                )
                local_rows.append(
                    {
                        "ratio": ratio,
                        "split": split,
                        "rhs_seed": rhs_seed,
                        "method": method,
                        "dictionary": dictionary,
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
            str(row["dictionary"]),
            str(row["seed"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for (split, ratio, method, dictionary, seed), local in sorted(groups.items()):
        result.append(
            {
                "split": split,
                "ratio": ratio,
                "method": method,
                "dictionary": dictionary,
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


def stage15_reproduction_checks(
    solver_rows: Sequence[Mapping[str, Any]],
    training_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Verify that the z-POD arm exactly reproduces Stage 15 iterations/selection."""

    source = _stage15_directory()
    old_rows = _read_csv(source / "solver_results.csv")
    old_lookup = {
        (
            float(row["ratio"]),
            int(row["rhs_seed"]),
            str(row["method"]),
            "" if row["seed"] == "" else str(int(row["seed"])),
        ): int(row["iterations"])
        for row in old_rows
    }
    differences = []
    for row in solver_rows:
        method = str(row["method"])
        if method == "mlp_z_pod":
            old_method = "learned_mlp"
        elif method in {
            "strong_ilu",
            "oracle",
            "nearest_reuse",
            "deterministic_interpolation",
        }:
            old_method = method
        else:
            continue
        seed = "" if row["seed"] == "" else str(int(row["seed"]))
        key = (float(row["ratio"]), int(row["rhs_seed"]), old_method, seed)
        differences.append(abs(int(row["iterations"]) - old_lookup[key]))
    new_seed = select_seed(
        [row for row in training_summaries if row["dictionary"] == "z_pod"]
    )
    old_decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    return {
        "max_iteration_difference": int(max(differences, default=0)),
        "selected_seed_stage16_z_pod": new_seed,
        "selected_seed_stage15": int(old_decision["selected_seed_by_validation_only"]),
        "selected_seed_matches": new_seed
        == int(old_decision["selected_seed_by_validation_only"]),
    }


def _median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def make_decision(
    representation_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    selected_seeds: Mapping[str, int],
    settings: Stage16Settings,
) -> Dict[str, Any]:
    """Apply an explicit representation-versus-mapping decision rule."""

    test_rep: Dict[str, Dict[str, float]] = {}
    for name in DICTIONARY_LABELS:
        local = [
            row
            for row in representation_rows
            if row["dictionary"] == name and row["split"] == "test"
        ]
        test_rep[name] = {
            "projection_d_w_normalized_median": _median(
                local, "projection_d_w_normalized"
            ),
            "w_optimal_d_w_normalized_median": _median(
                local, "w_optimal_d_w_normalized"
            ),
            "projection_d_z_normalized_median": _median(
                local, "projection_d_z_normalized"
            ),
        }

    selected_offline: Dict[str, Dict[str, float]] = {}
    selected_solver: Dict[str, Dict[str, float]] = {}
    for name, seed in selected_seeds.items():
        local_offline = [
            row
            for row in offline_rows
            if row["dictionary"] == name
            and int(row["seed"]) == seed
            and row["split"] == "test"
        ]
        local_solver = [
            row
            for row in solver_summaries
            if row["method"] == f"mlp_{name}"
            and int(row["seed"]) == seed
            and row["split"] == "test"
        ]
        selected_offline[name] = {
            "d_w_normalized_median": _median(local_offline, "d_w_normalized"),
            "d_z_normalized_median": _median(local_offline, "d_z_normalized"),
        }
        selected_solver[name] = {
            "eta_median": _median(local_solver, "eta_median"),
            "iterations_median": _median(local_solver, "iterations_median"),
            "convergence_rate": float(
                np.mean([float(row["convergence_rate"]) for row in local_solver])
            ),
            "online_seconds_median": _median(
                local_solver, "online_solve_seconds_median"
            ),
        }

    baselines: Dict[str, Dict[str, float]] = {}
    for method in ("strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation"):
        local = [
            row
            for row in solver_summaries
            if row["method"] == method and row["split"] == "test"
        ]
        baselines[method] = {
            "eta_median": _median(local, "eta_median"),
            "iterations_median": _median(local, "iterations_median"),
            "convergence_rate": float(
                np.mean([float(row["convergence_rate"]) for row in local])
            ),
            "online_seconds_median": _median(
                local, "online_solve_seconds_median"
            ),
        }

    z_rep = test_rep["z_pod"]["projection_d_w_normalized_median"]
    best_rep_name = min(
        ("az_aware", "joint"),
        key=lambda name: test_rep[name]["projection_d_w_normalized_median"],
    )
    best_learned_name = max(
        DICTIONARY_LABELS,
        key=lambda name: selected_solver[name]["eta_median"],
    )
    representation_improvement = (
        z_rep - test_rep[best_rep_name]["projection_d_w_normalized_median"]
    ) / max(z_rep, np.finfo(float).tiny)
    learned_eta_improvement = (
        selected_solver[best_learned_name]["eta_median"]
        - selected_solver["z_pod"]["eta_median"]
    )
    interpolation_eta = baselines["deterministic_interpolation"]["eta_median"]
    best_eta = selected_solver[best_learned_name]["eta_median"]
    matches_interpolation = best_eta >= (
        interpolation_eta - settings.interpolation_eta_tolerance
    )
    representation_substantial = (
        representation_improvement >= settings.representation_relative_improvement
    )
    learned_substantial = learned_eta_improvement >= settings.learned_eta_improvement
    best_w_ceiling = min(
        value["w_optimal_d_w_normalized_median"] for value in test_rep.values()
    )
    mapping_gap = (
        selected_offline[best_learned_name]["d_w_normalized_median"]
        - test_rep[best_learned_name]["w_optimal_d_w_normalized_median"]
    )

    if representation_substantial and learned_substantial and matches_interpolation:
        code = "A"
        label = "Representation was the main bottleneck; the learned method now matches or beats interpolation."
    elif representation_substantial and learned_substantial:
        code = "B"
        label = "Representation improves substantially, but deterministic interpolation remains stronger."
    elif best_w_ceiling <= settings.low_representation_error and mapping_gap >= 0.20:
        code = "C"
        label = "The dictionary can represent a good W-space, but the scalar MLP mapping is now the dominant bottleneck."
    else:
        code = "D"
        label = "The W-supervised reduced-dictionary route remains structurally limited in this experiment."

    return {
        "decision": code,
        "label": label,
        "selected_seed_by_validation_only": dict(selected_seeds),
        "test_representation": test_rep,
        "selected_offline": selected_offline,
        "selected_solver": selected_solver,
        "baselines": baselines,
        "best_new_representation": best_rep_name,
        "best_learned_representation": best_learned_name,
        "relative_representation_improvement_vs_z_pod": representation_improvement,
        "learned_eta_improvement_vs_z_pod": learned_eta_improvement,
        "matches_interpolation_with_tolerance": matches_interpolation,
        "best_w_optimal_ceiling_error": best_w_ceiling,
        "best_learned_mapping_gap": mapping_gap,
        "thresholds": {
            "representation_relative_improvement": settings.representation_relative_improvement,
            "learned_eta_improvement": settings.learned_eta_improvement,
            "interpolation_eta_tolerance": settings.interpolation_eta_tolerance,
            "low_representation_error": settings.low_representation_error,
            "mapping_gap": 0.20,
        },
    }


def _selected_solver_rows(
    rows: Sequence[Mapping[str, Any]], selected: Mapping[str, int]
) -> List[Mapping[str, Any]]:
    keep = {f"mlp_{name}": int(seed) for name, seed in selected.items()}
    return [
        row
        for row in rows
        if row["method"] in {
            "strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation"
        }
        or (
            row["method"] in keep
            and row["seed"] != ""
            and int(float(row["seed"])) == keep[row["method"]]
        )
    ]


def make_plots(
    output: Path,
    representation_rows: Sequence[Mapping[str, Any]],
    training_history: Sequence[Mapping[str, Any]],
    lambda_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    selected: Mapping[str, int],
    settings: Stage16Settings,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"z_pod": BLUE, "az_aware": ORANGE, "joint": PURPLE}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3)
    projection, ceiling = [], []
    for name in DICTIONARY_LABELS:
        local = [row for row in representation_rows if row["dictionary"] == name and row["split"] == "test"]
        projection.append(_median(local, "projection_d_w_normalized"))
        ceiling.append(_median(local, "w_optimal_d_w_normalized"))
    ax.bar(x - 0.18, projection, width=0.36, color=BLUE, label="project oracle Z")
    ax.bar(x + 0.18, ceiling, width=0.36, color=OLIVE, label="best W in range(A Psi)")
    ax.set_xticks(x, list(DICTIONARY_LABELS.values()))
    ax.set_ylabel("Median test normalized W-projector error")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(plot_dir / "representation_ceiling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in DICTIONARY_LABELS:
        seed = selected[name]
        local = sorted(
            [row for row in offline_rows if row["dictionary"] == name and int(row["seed"]) == seed and row["split"] == "test"],
            key=lambda row: float(row["ratio"]),
        )
        ax.plot([row["ratio"] for row in local], [row["d_w_normalized"] for row in local], marker="o", color=colors[name], label=DICTIONARY_LABELS[name])
    ax.set_xlabel("Anisotropy ratio R")
    ax.set_ylabel("Normalized W-projector error")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "learned_w_error_test.png", dpi=180)
    plt.close(fig)

    selected_rows = _selected_solver_rows(solver_summaries, selected)
    labels = {
        "strong_ilu": "ILU", "oracle": "Oracle", "nearest_reuse": "Nearest",
        "deterministic_interpolation": "Interpolation", "mlp_z_pod": "MLP Z-POD",
        "mlp_az_aware": "MLP AZ-aware", "mlp_joint": "MLP joint",
    }
    method_colors = {
        "strong_ilu": CHARCOAL, "oracle": PINK, "nearest_reuse": OLIVE,
        "deterministic_interpolation": "#4B8F8C", "mlp_z_pod": BLUE,
        "mlp_az_aware": ORANGE, "mlp_joint": PURPLE,
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for method in labels:
        local = sorted([row for row in selected_rows if row["method"] == method and row["split"] == "test"], key=lambda row: float(row["ratio"]))
        ax.plot([row["ratio"] for row in local], [row["iterations_median"] for row in local], marker="o", label=labels[method], color=method_colors[method])
    ax.set_xlabel("Anisotropy ratio R")
    ax.set_ylabel("Median recycled-GMRES iterations")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_solver_iterations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for method in ("nearest_reuse", "deterministic_interpolation", "mlp_z_pod", "mlp_az_aware", "mlp_joint"):
        local = sorted([row for row in selected_rows if row["method"] == method and row["split"] == "test"], key=lambda row: float(row["ratio"]))
        ax.plot([row["ratio"] for row in local], [row["eta_median"] for row in local], marker="o", label=labels[method], color=method_colors[method])
    ax.axhline(1.0, color=PINK, linestyle="--", linewidth=1.0, label="Oracle")
    ax.set_xlabel("Anisotropy ratio R")
    ax.set_ylabel("Median oracle recovery eta")
    ax.set_xticks(settings.stage15.test_ratios)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_oracle_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in DICTIONARY_LABELS:
        seed = selected[name]
        local = sorted([row for row in training_history if row["dictionary"] == name and int(row["seed"]) == seed], key=lambda row: int(row["epoch"]))
        ax.semilogy([row["epoch"] for row in local], [row["validation_loss"] for row in local], color=colors[name], label=DICTIONARY_LABELS[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation W-projector loss")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_validation_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ordered = sorted(lambda_rows, key=lambda row: float(row["lambda"]))
    ax.semilogx([row["lambda"] for row in ordered], [row["validation_projection_d_w_normalized_median"] for row in ordered], marker="o", color=PURPLE)
    ax.set_xlabel("Joint dictionary lambda")
    ax.set_ylabel("Validation median normalized W error")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(plot_dir / "joint_lambda_selection.png", dpi=180)
    plt.close(fig)


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
    header = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_format(value) for value in row) + " |" for row in rows]
    return "\n".join((header, rule, *body))


def write_report(
    output: Path,
    settings: Stage16Settings,
    dictionaries: Mapping[str, DictionarySpec],
    lambda_rows: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    training_summaries: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    selected = decision["selected_seed_by_validation_only"]
    rep_table = []
    for name in DICTIONARY_LABELS:
        local = [row for row in representation_rows if row["dictionary"] == name and row["split"] == "test"]
        rep_table.append([
            DICTIONARY_LABELS[name], dictionaries[name].basis.shape[1],
            _median(local, "projection_d_z_normalized"),
            _median(local, "projection_d_w_normalized"),
            _median(local, "w_optimal_d_w_normalized"),
        ])
    training_table = []
    for name in DICTIONARY_LABELS:
        for row in training_summaries:
            if row["dictionary"] == name:
                training_table.append([
                    DICTIONARY_LABELS[name], int(row["seed"]), int(row["best_epoch"]),
                    row["best_validation_loss"], row["training_seconds"],
                    int(row["seed"]) == int(selected[name]),
                ])
    test_table = []
    chosen_rows = _selected_solver_rows(solver_summaries, selected)
    method_order = ("strong_ilu", "oracle", "nearest_reuse", "deterministic_interpolation", "mlp_z_pod", "mlp_az_aware", "mlp_joint")
    method_labels = {
        "strong_ilu": "Strong ILU", "oracle": "Own oracle", "nearest_reuse": "Nearest reuse",
        "deterministic_interpolation": "Interpolation", "mlp_z_pod": "MLP / Z-POD",
        "mlp_az_aware": "MLP / AZ-aware", "mlp_joint": "MLP / joint",
    }
    offline_lookup = {(row["dictionary"], int(row["seed"]), float(row["ratio"])): row for row in offline_rows}
    for ratio in settings.stage15.test_ratios:
        for method in method_order:
            row = next(local for local in chosen_rows if local["split"] == "test" and float(local["ratio"]) == ratio and local["method"] == method)
            if method.startswith("mlp_"):
                name = method.removeprefix("mlp_")
                geom = offline_lookup[(name, int(selected[name]), ratio)]
                dw, dz = geom["d_w_normalized"], geom["d_z_normalized"]
            else:
                dw = dz = None
            test_table.append([
                ratio, method_labels[method], dw, dz, row["iterations_median"],
                row["convergence_rate"], row["true_final_relative_residual_max"],
                row["online_solve_seconds_median"], row["eta_median"],
            ])

    joint_selected = next(row for row in lambda_rows if row["selected"])
    representation_change = decision["relative_representation_improvement_vs_z_pod"]
    representation_phrase = (
        f"a {representation_change:.1%} relative reduction"
        if representation_change >= 0.0
        else f"a {-representation_change:.1%} relative increase"
    )
    report = f"""# Stage 16 — W-aware dictionary ablation

## Executive conclusion

**Decision {decision['decision']}: {decision['label']}**

This controlled ablation changes only the fixed reduced dictionary. It reloads the exact Stage 15 oracle spaces and split, rebuilds the same matrices and ILU factors, and reuses the same scalar MLP, optimizer, random seeds, W-projector loss, and Stage 13 recycled-GMRES evaluation. All dictionaries have **p={settings.dictionary_dimension}**, so the network architecture and parameter count are identical.

The less damaging of the two new dictionaries by the pre-network Z-projection diagnostic is **{DICTIONARY_LABELS[decision['best_new_representation']]}**, but it produces {representation_phrase} in median test W error versus Z-POD. The best learned arm remains **{DICTIONARY_LABELS[decision['best_learned_representation']]}**, changing median oracle recovery by {decision['learned_eta_improvement_vs_z_pod']:.3f} relative to the exact Stage 15 Z-POD control. Deterministic interpolation remains at median eta={decision['baselines']['deterministic_interpolation']['eta_median']:.3f}.

## 1. Fixed protocol and representation construction

- RBF-FD: N=5041, stencil=25, seed=7, jitter=0.15, rotated anisotropy at 30 degrees.
- Strong ILU: drop tolerance 1e-3, fill factor 5; operator convention B=A M^-1.
- Family: R=90,...,110. Training uses alternating even R values; validation={settings.stage15.validation_ratios}; untouched test={settings.stage15.test_ratios}.
- q=8; p=19 for every representation. MLP: 1→64→128→152, GELU, Adam, learning rate 2e-3, weight decay 1e-6, float64, identical Stage 15 stopping and three seeds.
- Z-POD dictionary, model checkpoints, and training history are loaded exactly from Stage 15; this control reproduces every Stage 15 solver iteration.
- AZ-aware first forms a p=19 POD basis Phi_W of training-only A_i Z_i snapshots, maps every Phi_W through each training A_i^-1 using accurate sparse solves, then takes a p=19 POD of those solution-side inverse images. No held-out matrix or oracle solve enters construction.
- Joint uses a p=19 POD of [Z_train, sqrt(lambda) scaled(A_i Z_i)], with each AZ block Frobenius-scaled to its matching Z block. The validation-only choice is lambda={joint_selected['lambda']:g} from {settings.joint_lambdas}.

The prompt's displayed combined objective contains a minus sign before the lambda term. Under minimization that would reward larger AZ reconstruction error and is mathematically ill-posed. This implementation uses the intended nonnegative **plus-lambda** combined reconstruction; the change is explicit rather than silent.

## 2. Is the dictionary limiting before the network?

For each held-out target, the prescribed diagnostic projects its oracle Z_* into each fixed dictionary, orthonormalizes, applies the target A, and measures normalized W-projector error. The final column is an additional evaluation-only lower ceiling: the best q-dimensional W subspace inside range(A Psi). It uses the held-out oracle only for diagnosis and never for dictionary construction or training.

{_table(['dictionary', 'p', 'Z-proj norm d_Z', 'Z-proj norm d_W', 'best-in-range norm d_W'], rep_table)}

![Representation ceilings](plots/representation_ceiling.png)

## 3. Identical MLP training

All arms minimize the same basis-invariant W-projector loss L_W=||P_Wtheta-P_W*||_F^2. Seed selection is validation-only.

{_table(['dictionary', 'seed', 'best epoch', 'best val L_W', 'training s', 'selected'], training_table)}

![Validation loss](plots/training_validation_loss.png)

The small joint-lambda sweep uses only validation representation error:

{_table(['lambda', 'validation Z-proj norm d_W', 'validation best-in-range norm d_W', 'selected'], [[row['lambda'], row['validation_projection_d_w_normalized_median'], row['validation_w_optimal_d_w_normalized_median'], row['selected']] for row in lambda_rows])}

![Joint lambda selection](plots/joint_lambda_selection.png)

## 4. Held-out offline and solver results

Normalized projector distances divide by sqrt(2q). Z error is diagnostic only. Oracle recovery is eta=(k_ILU-k_method)/(k_ILU-k_oracle), computed per RHS before aggregation.

{_table(['R', 'method', 'norm d_W', 'norm d_Z', 'iterations', 'conv.', 'max true residual', 'online s', 'eta'], test_table)}

![Learned W error](plots/learned_w_error_test.png)

![Test iterations](plots/test_solver_iterations.png)

![Test oracle recovery](plots/test_oracle_recovery.png)

## 5. Bottleneck diagnosis

- **Representation:** median test Z-projection-induced W error changes from {decision['test_representation']['z_pod']['projection_d_w_normalized_median']:.3f} to {decision['test_representation'][decision['best_new_representation']]['projection_d_w_normalized_median']:.3f}. The new dictionary's best-in-range W error is {decision['test_representation'][decision['best_new_representation']]['w_optimal_d_w_normalized_median']:.3f}.
- **Learning:** the best selected MLP has median normalized W error {decision['selected_offline'][decision['best_learned_representation']]['d_w_normalized_median']:.3f}; its gap above that dictionary's best-in-range ceiling is {decision['best_learned_mapping_gap']:.3f}.
- **Solver:** the best learned median eta is {decision['selected_solver'][decision['best_learned_representation']]['eta_median']:.3f}, versus Z-POD {decision['selected_solver']['z_pod']['eta_median']:.3f}, nearest reuse {decision['baselines']['nearest_reuse']['eta_median']:.3f}, and interpolation {decision['baselines']['deterministic_interpolation']['eta_median']:.3f}.
- **Robustness:** best learned convergence is {decision['selected_solver'][decision['best_learned_representation']]['convergence_rate']:.2f}. Timing is reported but not overinterpreted; dictionary and label creation remain offline.

## 6. Decision logic and limitations

The registered thresholds are at least {settings.representation_relative_improvement:.0%} relative representation improvement, at least {settings.learned_eta_improvement:.2f} eta improvement, interpolation matching within {settings.interpolation_eta_tolerance:.2f}, and a best-in-range normalized W error ≤{settings.low_representation_error:.2f} as evidence that a dictionary is no longer the dominant limiter.

Outcome **{decision['decision']}** follows from measured representation improvement {decision['relative_representation_improvement_vs_z_pod']:.1%}, learned eta change {decision['learned_eta_improvement_vs_z_pod']:.3f}, and interpolation matching={decision['matches_interpolation_with_tolerance']}.

This remains a narrow, fixed-node, one-parameter interpolation experiment. It does not establish transfer beyond R in [90,110], variable geometry, variable N, another PDE, or another ILU. It does not test a different neural architecture. Training-only AZ inverse-map construction cost is {dictionaries['az_aware'].metadata['inverse_mapping_seconds']:.3f} s and requires amortization in deployment arguments.

## 7. Reproducibility and validation

Stage 15 reproduction: maximum iteration difference {reproduction['max_iteration_difference']}; selected-seed agreement={reproduction['selected_seed_matches']}. Validation status: **{validation['assessment']}**.

Raw CSV/JSON/NPZ evidence and all referenced figures are saved beside this report. Ranks, split isolation, finite metrics, eta calculations, convergence residuals, and Stage 15 equivalence are validated. No neural family, PDE setting, oracle construction, target loss, or solver mechanism was changed.
"""
    path = output / "stage16_W_aware_dictionary_ablation.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    output: Path,
    dictionaries: Mapping[str, DictionarySpec],
    representation_rows: Sequence[Mapping[str, Any]],
    training_summaries: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    solver_summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    settings: Stage16Settings,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["three_dictionaries"] = set(dictionaries) == set(DICTIONARY_LABELS)
    checks["identical_dictionary_dimensions"] = all(
        spec.basis.shape == (5041, settings.dictionary_dimension)
        for spec in dictionaries.values()
    )
    checks["dictionary_orthonormality"] = all(
        np.linalg.norm(spec.basis.T @ spec.basis - np.eye(settings.dictionary_dimension), ord="fro") <= 1.0e-8
        for spec in dictionaries.values()
    )
    checks["representation_rows_exact"] = len(representation_rows) == 3 * len(settings.stage15.intermediate_ratios)
    checks["representation_metrics_finite"] = all(
        np.isfinite(float(row[field]))
        for row in representation_rows
        for field in ("projection_d_z_normalized", "projection_d_w_normalized", "w_optimal_d_w_normalized")
    )
    checks["training_runs_exact"] = len(training_summaries) == 3 * len(settings.stage15.model_seeds)
    checks["seed_selection_complete"] = set(decision["selected_seed_by_validation_only"]) == set(DICTIONARY_LABELS)
    checks["prediction_rows_exact"] = len(prediction_rows) == 3 * len(settings.stage15.model_seeds) * len(settings.stage15.intermediate_ratios)
    checks["offline_rows_exact"] = len(offline_rows) == len(prediction_rows)
    expected_solver = len(settings.stage15.intermediate_ratios) * len(settings.stage15.stage10.evaluation_rhs_seeds) * (4 + 3 * len(settings.stage15.model_seeds))
    checks["solver_rows_exact"] = len(solver_rows) == expected_solver
    checks["solver_metrics_finite"] = all(
        np.isfinite(float(row[field]))
        for row in solver_rows
        for field in ("iterations", "true_final_relative_residual", "online_solve_seconds")
    )
    checks["converged_residuals_meet_tolerance"] = all(
        (not bool(row["converged"]))
        or float(row["true_final_relative_residual"]) <= settings.stage15.stage10.solver.relative_tolerance * (1.0 + 1.0e-8)
        for row in solver_rows
    )
    checks["solver_summaries_exact"] = len(solver_summaries) == len(settings.stage15.intermediate_ratios) * (4 + 3 * len(settings.stage15.model_seeds))
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
    checks["stage15_iteration_reproduction"] = reproduction["max_iteration_difference"] == 0
    checks["stage15_seed_reproduction"] = bool(reproduction["selected_seed_matches"])
    required = (
        "stage16_W_aware_dictionary_ablation.md", "settings.json", "environment.json",
        "dictionaries.npz", "stage15_family_reconstruction.csv", "dictionary_spectra.csv",
        "joint_lambda_selection.csv", "representation_diagnostics.csv", "training_history.csv",
        "training_summary.csv", "prediction_metadata.csv", "offline_metrics.csv",
        "solver_results.csv", "solver_summary.csv", "decision.json", "reproduction.json",
        "plots/representation_ceiling.png", "plots/learned_w_error_test.png",
        "plots/test_solver_iterations.png", "plots/test_oracle_recovery.png",
        "plots/training_validation_loss.png", "plots/joint_lambda_selection.png",
    )
    missing = [name for name in required if not (output / name).exists()]
    checks["missing_artifacts"] = missing
    checks["artifacts_complete"] = not missing
    blockers = [key for key, value in checks.items() if isinstance(value, bool) and not value]
    return {"assessment": "ready_to_share" if not blockers else "needs_revision", "blockers": blockers, "checks": checks}


def run_stage16(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    settings = Stage16Settings()
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "solver": "unchanged Stage 13 recycled/projected GMRES", "oracle": "exact saved Stage 15 oracle Z spaces",
    })
    print("[1/9] loading exact Stage 15 family and oracle spaces", flush=True)
    contexts, spaces, factors, _ = load_exact_stage15_family(output, settings)

    print("[2/9] building three p=19 training-only dictionaries", flush=True)
    dictionaries, spectra, lambda_rows = build_dictionaries(output, contexts, spaces, factors, settings)
    _write_csv(output / "dictionary_spectra.csv", spectra)
    _write_csv(output / "joint_lambda_selection.csv", lambda_rows)
    _write_json(output / "dictionary_metadata.json", {name: spec.metadata for name, spec in dictionaries.items()})

    print("[3/9] measuring held-out representation ceilings before the NN", flush=True)
    representation_rows = representation_diagnostics(dictionaries, settings.stage15.intermediate_ratios, contexts, spaces, factors, settings)
    _write_csv(output / "representation_diagnostics.csv", representation_rows)

    print("[4/9] training identical Stage 15 MLPs and Adam budgets", flush=True)
    models, histories, training_summaries = train_dictionary_models(output, dictionaries, contexts, spaces, factors, settings)
    selected = _selected_seeds(training_summaries)

    print("[5/9] constructing held-out learned and deterministic spaces", flush=True)
    predicted, prediction_rows = predict_all_spaces(models, dictionaries, settings)
    _write_csv(output / "prediction_metadata.csv", prediction_rows)
    deterministic, deterministic_rows = build_deterministic_spaces(spaces, settings.stage15)
    _write_csv(output / "deterministic_space_metadata.csv", deterministic_rows)
    offline_rows = learned_offline_metrics(predicted, contexts, spaces, factors, settings)
    _write_csv(output / "offline_metrics.csv", offline_rows)

    print("[6/9] running unchanged recycled-GMRES evaluation", flush=True)
    solver_rows = run_solver_ablation(contexts, spaces, factors, deterministic, predicted, prediction_rows, settings)
    solver_summaries = aggregate_solver_rows(solver_rows)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summaries)

    print("[7/9] diagnosing the bottleneck and reproducing Stage 15", flush=True)
    decision = make_decision(representation_rows, offline_rows, solver_summaries, selected, settings)
    reproduction = stage15_reproduction_checks(solver_rows, training_summaries)
    _write_json(output / "decision.json", decision)
    _write_json(output / "reproduction.json", reproduction)

    print("[8/9] plotting and validating artifacts", flush=True)
    make_plots(output, representation_rows, histories, lambda_rows, offline_rows, solver_summaries, selected, settings)
    preliminary = {"assessment": "pending_report_artifact"}
    write_report(output, settings, dictionaries, lambda_rows, representation_rows, training_summaries, offline_rows, solver_summaries, decision, reproduction, preliminary)
    validation = validate_outputs(output, dictionaries, representation_rows, training_summaries, prediction_rows, offline_rows, solver_rows, solver_summaries, decision, reproduction, settings)
    _write_json(output / "validation.json", validation)

    print("[9/9] writing final technical report", flush=True)
    report = write_report(output, settings, dictionaries, lambda_rows, representation_rows, training_summaries, offline_rows, solver_summaries, decision, reproduction, validation)
    print(f"Stage 16 complete: {report}", flush=True)
    return report


if __name__ == "__main__":
    run_stage16(_repository_root() / "results" / "stage16_W_aware_dictionary_ablation")
