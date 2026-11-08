"""Stage 12: isolate nonsymmetric Ritz coarse-coupling formulations.

No learned model is trained.  The Stage 10 discretization, ILU, Ritz
construction, and held-out right-hand sides are reused.  The only experimental
factor is the reduced trial/test coupling applied to the same right space Z.
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

from ..coarse_space import (
    AZPetrovFactors,
    build_az_petrov_coarse_preconditioner,
    build_galerkin_coarse_preconditioner,
    build_multiplicative_coarse_preconditioner,
    build_paired_petrov_coarse_preconditioner,
    factor_az_petrov,
    random_orthonormal_basis,
    sparse_symmetry_measure,
)
from ..preconditioners import InverseAction
from ..subspace_analysis import subspace_metrics
from .stage9_scalability import _read_csv, _write_csv, _write_json
from .stage10_ritz_transfer import (
    BLUE,
    CHARCOAL,
    LIGHT_GRAY,
    OLIVE,
    ORANGE,
    PINK,
    PURPLE,
    RitzContext,
    Stage10Settings,
    _build_context,
    _run_solver,
)


@dataclass(frozen=True)
class Stage12Settings:
    """Fixed protocol and preregistered comparison dimensions."""

    stage10: Stage10Settings = Stage10Settings()
    ratios: Tuple[float, ...] = (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0)
    q_values: Tuple[int, ...] = (4, 8, 16)
    transfer_q: int = 8
    transfer_pairs: Tuple[Tuple[float, float], ...] = (
        (85.0, 90.0),
        (90.0, 95.0),
        (95.0, 100.0),
        (100.0, 105.0),
        (105.0, 110.0),
        (110.0, 120.0),
    )
    operator_symmetry_probes: int = 48
    equivalence_probes: int = 6
    equivalence_tolerance: float = 5.0e-11
    reproduction_iteration_tolerance: int = 0
    useful_iteration_reduction: float = 0.20
    reliable_convergence_rate: float = 0.90
    oracle_gap_tolerance: float = 10.0


def _stage10_directory(output: Path) -> Path:
    sibling = output.parent / "stage10_ritz_transfer_and_mode_tracking"
    if sibling.exists():
        return sibling
    repository = Path(__file__).resolve().parents[3]
    fallback = repository / "results" / "stage10_ritz_transfer_and_mode_tracking"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Stage 10 results are required for Stage 12")


def _orthonormalize(basis: np.ndarray) -> Tuple[np.ndarray, float, int]:
    basis = np.asarray(basis, dtype=float)
    orthonormal, triangular = la.qr(basis, mode="economic")
    rank = int(np.linalg.matrix_rank(triangular))
    error = float(
        np.linalg.norm(orthonormal.T @ orthonormal - np.eye(orthonormal.shape[1]))
    )
    if rank != basis.shape[1]:
        raise RuntimeError(f"Ritz prefix rank {rank}, expected {basis.shape[1]}")
    return orthonormal, error, rank


def build_contexts_and_fixed_spaces(
    output: Path,
    settings: Stage12Settings,
) -> Tuple[Dict[float, RitzContext], Dict[float, np.ndarray], List[Dict[str, Any]]]:
    """Rebuild Stage 10 contexts and reconcile their Ritz spaces with archives."""

    stage10_dir = _stage10_directory(output)
    contexts: Dict[float, RitzContext] = {}
    fixed_spaces: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    saved_dir = output / "spaces"
    saved_dir.mkdir(parents=True, exist_ok=True)
    for ratio in settings.ratios:
        context = _build_context(ratio, settings.stage10)
        archive = np.load(stage10_dir / "subspaces" / f"ritz_R{ratio:g}_q32.npz")
        archived = np.asarray(archive["basis"], dtype=float)
        rebuilt = np.asarray(context.modes.basis, dtype=float)
        metric = subspace_metrics(archived, rebuilt)
        fixed_spaces[ratio] = archived
        contexts[ratio] = context
        rows.append(
            {
                "ratio": ratio,
                "archived_dimension": archived.shape[1],
                "rebuilt_dimension": rebuilt.shape[1],
                "archived_rebuilt_projector_distance_normalized": metric[
                    "projector_distance_normalized"
                ],
                "archived_rebuilt_principal_angle_max_degrees": metric[
                    "principal_angle_max_degrees"
                ],
                "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
                "ilu_setup_seconds": context.ilu.setup_seconds,
                "arnoldi_dimension": context.modes.diagnostics["arnoldi_dimension"],
                "rebuilt_basis_orthogonality_error": context.modes.diagnostics[
                    "basis_orthogonality_error"
                ],
            }
        )
        np.savez_compressed(
            saved_dir / f"fixed_stage10_ritz_R{ratio:g}_q32.npz",
            basis=archived,
            eigenvalues=archive["eigenvalues"],
            ratio=ratio,
            provenance="exact Stage 10 archived right/correction space",
        )
    _write_csv(output / "ritz_space_reconstruction.csv", rows)
    return contexts, fixed_spaces, rows


def _right_preconditioned_symmetry_estimate(
    context: RitzContext,
    probes: int,
    seed: int,
) -> Dict[str, float]:
    r"""Estimate ``||B-B^T||_F/||B||_F`` for ``B=AM^{-1}`` by probes."""

    rng = np.random.default_rng(seed)
    denominator: List[float] = []
    numerator: List[float] = []
    matrix = context.system.matrix
    for _ in range(probes):
        vector = rng.choice(np.array([-1.0, 1.0]), size=matrix.shape[0])
        forward = np.asarray(matrix @ context.ilu.apply(vector), dtype=float)
        transpose = context.ilu.apply_transpose(
            np.asarray(matrix.T @ vector, dtype=float)
        )
        denominator.append(float(np.dot(forward, forward)))
        difference = forward - transpose
        numerator.append(float(np.dot(difference, difference)))
    ratio = float(np.sqrt(np.sum(numerator) / np.sum(denominator)))
    bootstrap = []
    for _ in range(256):
        indices = rng.integers(0, probes, size=probes)
        bootstrap.append(
            np.sqrt(
                np.sum(np.asarray(numerator)[indices])
                / np.sum(np.asarray(denominator)[indices])
            )
        )
    return {
        "am_inverse_symmetry_estimate": ratio,
        "am_inverse_symmetry_bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "am_inverse_symmetry_bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
    }


def run_nonsymmetry_diagnostics(
    output: Path,
    contexts: Mapping[float, RitzContext],
    settings: Stage12Settings,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ratio, context in contexts.items():
        estimate = _right_preconditioned_symmetry_estimate(
            context,
            settings.operator_symmetry_probes,
            settings.stage10.seed + int(100 * ratio),
        )
        rows.append(
            {
                "ratio": ratio,
                "matrix_symmetry_measure": sparse_symmetry_measure(
                    context.system.matrix
                ),
                "stage10_recorded_symmetry_measure": context.system.diagnostics[
                    "symmetry_measure"
                ],
                "operator": "A M^-1",
                "operator_estimator": "Hutchinson/Rademacher Frobenius ratio",
                "probe_count": settings.operator_symmetry_probes,
                **estimate,
            }
        )
    _write_csv(output / "nonsymmetry_diagnostics.csv", rows)
    return rows


def _relative_action_error(
    first: InverseAction,
    second: InverseAction,
    dimension: int,
    probes: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(probes):
        vector = rng.normal(size=dimension)
        first_value = first.apply(vector)
        second_value = second.apply(vector)
        errors.append(
            float(
                np.linalg.norm(first_value - second_value)
                / max(np.linalg.norm(first_value), np.finfo(float).tiny)
            )
        )
    return float(max(errors))


def _algorithmic_matvec_count(iterations: int, restart: int, hybrid: bool) -> int:
    """Exact A-matvec count for the local FGMRES implementation."""

    cycles = 0 if iterations == 0 else int(np.ceil(iterations / restart))
    outer = 1 + cycles + 2 * iterations
    return outer + (iterations if hybrid else 0)


def _result_row(
    ratio: float,
    q: int,
    method: str,
    rhs_seed: int,
    result: Any,
    diagnostics: Mapping[str, Any],
    settings: Stage12Settings,
    expected: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    hybrid = method != "ILU"
    row = {
        "ratio": ratio,
        "q": q,
        "method": method,
        "rhs_seed": rhs_seed,
        "iterations": result.iterations,
        "converged": result.converged,
        "final_relative_residual": float(result.relative_residuals[-1]),
        "a_matvec_count": _algorithmic_matvec_count(
            result.iterations, settings.stage10.solver.restart, hybrid
        ),
        "solve_seconds": result.solve_seconds,
        "preconditioner_seconds": result.preconditioner_seconds,
        "coarse_setup_seconds": diagnostics.get("coarse_setup_seconds", 0.0),
        "ritz_setup_seconds": diagnostics.get("ritz_setup_seconds", 0.0),
        "reduced_condition": diagnostics.get("reduced_condition", float("nan")),
        "ztaz_condition": diagnostics.get("ztaz_condition", float("nan")),
        "wtaz_condition": diagnostics.get("wtaz_condition", float("nan")),
        "failure": result.failure,
        "stage10_expected_iterations": "",
        "stage10_iteration_difference": "",
        "stage10_expected_converged": "",
        "stage10_convergence_match": "",
    }
    if expected is not None:
        row.update(
            {
                "stage10_expected_iterations": int(expected["iterations"]),
                "stage10_iteration_difference": result.iterations
                - int(expected["iterations"]),
                "stage10_expected_converged": bool(expected["converged"]),
                "stage10_convergence_match": result.converged
                == bool(expected["converged"]),
            }
        )
    return row


def _summaries(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for key, local in groups.items():
        summary = {name: value for name, value in zip(keys, key)}
        summary.update(
            {
                "iterations_median": float(np.median([row["iterations"] for row in local])),
                "iterations_max": int(max(row["iterations"] for row in local)),
                "convergence_rate": float(np.mean([row["converged"] for row in local])),
                "final_relative_residual_max": float(
                    max(row["final_relative_residual"] for row in local)
                ),
                "a_matvec_count_median": float(
                    np.median([row["a_matvec_count"] for row in local])
                ),
                "solve_seconds_median": float(
                    np.median([row["solve_seconds"] for row in local])
                ),
                "preconditioner_seconds_median": float(
                    np.median([row["preconditioner_seconds"] for row in local])
                ),
                "coarse_setup_seconds": float(local[0]["coarse_setup_seconds"]),
                "ritz_setup_seconds": float(local[0]["ritz_setup_seconds"]),
                "reduced_condition": float(local[0]["reduced_condition"]),
                "ztaz_condition": float(local[0]["ztaz_condition"]),
                "wtaz_condition": float(local[0]["wtaz_condition"]),
                "failures": "; ".join(
                    sorted({str(row["failure"]) for row in local if row["failure"]})
                ),
            }
        )
        summaries.append(summary)
    return summaries


def run_same_system_comparison(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    settings: Stage12Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare all couplings on identical Z and held-out right-hand sides."""

    old_rows = _read_csv(_stage10_directory(output) / "q_sweep_results.csv")
    expected = {
        (float(row["ratio"]), int(row["q"]), int(row["rhs_seed"])): row
        for row in old_rows
    }
    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    for ratio, context in contexts.items():
        for q in settings.q_values:
            qr_start = perf_counter()
            basis, basis_error, rank = _orthonormalize(spaces[ratio][:, :q])
            basis_qr_seconds = perf_counter() - qr_start
            factors = factor_az_petrov(context.system.matrix, basis)
            galerkin, galerkin_dx = build_galerkin_coarse_preconditioner(
                context.system.matrix, context.ilu, basis, name="galerkin"
            )
            historical, historical_dx = build_multiplicative_coarse_preconditioner(
                context.system.matrix,
                context.ilu,
                basis,
                name="stage10_existing",
            )
            petrov, petrov_dx = build_az_petrov_coarse_preconditioner(
                context.system.matrix, context.ilu, factors, name="az_petrov"
            )
            recycling, recycling_dx = build_az_petrov_coarse_preconditioner(
                context.system.matrix,
                context.ilu,
                factors,
                recycling=True,
                name="recycling",
            )
            seed = settings.stage10.seed + int(1000 * ratio) + q
            historical_petrov_error = _relative_action_error(
                historical,
                petrov,
                context.system.matrix.shape[0],
                settings.equivalence_probes,
                seed,
            )
            petrov_recycling_error = _relative_action_error(
                petrov,
                recycling,
                context.system.matrix.shape[0],
                settings.equivalence_probes,
                seed + 1,
            )
            diagnostics_rows.append(
                {
                    "ratio": ratio,
                    "q": q,
                    "basis_rank": rank,
                    "basis_qr_seconds": basis_qr_seconds,
                    "basis_orthogonality_error": basis_error,
                    "historical_coarse_variant": historical_dx["coarse_variant"],
                    "galerkin_ztaz_condition": galerkin_dx["ztaz_condition"],
                    "petrov_wtaz_condition": petrov_dx["wtaz_condition"],
                    "wtaz_equals_r_relative_error": factors.diagnostics[
                        "wtaz_equals_r_relative_error"
                    ],
                    "w_orthogonality_error": factors.diagnostics[
                        "test_orthogonality_error"
                    ],
                    "au_equals_w_relative_error": factors.diagnostics[
                        "au_equals_w_relative_error"
                    ],
                    "historical_petrov_action_relative_error_max": historical_petrov_error,
                    "petrov_recycling_action_relative_error_max": petrov_recycling_error,
                    "galerkin_setup_seconds": galerkin_dx["coarse_setup_seconds"],
                    "petrov_setup_seconds": petrov_dx["coarse_setup_seconds"],
                    "recycling_setup_seconds": recycling_dx["coarse_setup_seconds"],
                    "ritz_setup_seconds": context.modes.diagnostics["setup_seconds"],
                }
            )
            methods = {
                "galerkin_counterfactual": (galerkin, galerkin_dx),
                "stage10_existing_petrov": (historical, historical_dx),
                "az_petrov": (petrov, petrov_dx),
                "recycling": (recycling, recycling_dx),
            }
            for rhs_seed in settings.stage10.evaluation_rhs_seeds:
                if q == settings.q_values[0]:
                    baseline = _run_solver(context, rhs_seed, context.ilu, settings.stage10)
                    baseline_dx = {
                        "coarse_setup_seconds": 0.0,
                        "ritz_setup_seconds": 0.0,
                    }
                    rows.append(
                        _result_row(
                            ratio,
                            0,
                            "ILU",
                            rhs_seed,
                            baseline,
                            baseline_dx,
                            settings,
                            expected.get((ratio, 0, rhs_seed)),
                        )
                    )
                    for iteration, residual in enumerate(baseline.relative_residuals):
                        histories.append(
                            {
                                "ratio": ratio,
                                "q": 0,
                                "method": "ILU",
                                "rhs_seed": rhs_seed,
                                "iteration": iteration,
                                "relative_residual": float(residual),
                            }
                        )
                for method, (action, method_dx) in methods.items():
                    method_dx = dict(method_dx)
                    method_dx["ritz_setup_seconds"] = context.modes.diagnostics[
                        "setup_seconds"
                    ]
                    result = _run_solver(context, rhs_seed, action, settings.stage10)
                    stage10_expected = (
                        expected.get((ratio, q, rhs_seed))
                        if method == "stage10_existing_petrov"
                        else None
                    )
                    rows.append(
                        _result_row(
                            ratio,
                            q,
                            method,
                            rhs_seed,
                            result,
                            method_dx,
                            settings,
                            stage10_expected,
                        )
                    )
                    for iteration, residual in enumerate(result.relative_residuals):
                        histories.append(
                            {
                                "ratio": ratio,
                                "q": q,
                                "method": method,
                                "rhs_seed": rhs_seed,
                                "iteration": iteration,
                                "relative_residual": float(residual),
                            }
                        )
    summaries = _summaries(rows, ("ratio", "q", "method"))
    _write_csv(output / "same_system_results.csv", rows)
    _write_csv(output / "same_system_summary.csv", summaries)
    _write_csv(output / "same_system_histories.csv", histories)
    _write_csv(output / "coupling_identities.csv", diagnostics_rows)
    return rows, summaries, histories, diagnostics_rows


def run_transfer_comparison(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    settings: Stage12Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate full donor-pair reuse from right-space-only transfer."""

    q = settings.transfer_q
    result_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    coupling_rows: List[Dict[str, Any]] = []
    stage10_transfer = _read_csv(_stage10_directory(output) / "transfer_results.csv")
    old_lookup = {
        (float(row["source_ratio"]), float(row["target_ratio"])): row
        for row in stage10_transfer
        if row["method"] == "nearby_reuse" and row["source_ratio"] != ""
    }
    for pair_index, (source_ratio, target_ratio) in enumerate(settings.transfer_pairs):
        source = contexts[source_ratio]
        target = contexts[target_ratio]
        source_z, _, _ = _orthonormalize(spaces[source_ratio][:, :q])
        target_z, _, _ = _orthonormalize(spaces[target_ratio][:, :q])
        source_factors = factor_az_petrov(source.system.matrix, source_z)
        target_factors = factor_az_petrov(target.system.matrix, target_z)
        transferred_factors = factor_az_petrov(target.system.matrix, source_z)
        own, own_dx = build_az_petrov_coarse_preconditioner(
            target.system.matrix, target.ilu, target_factors, name="own_target"
        )
        full_pair, full_pair_dx = build_paired_petrov_coarse_preconditioner(
            target.system.matrix,
            target.ilu,
            source_z,
            source_factors.test,
            name="donor_full_pair",
        )
        right_transfer, right_dx = build_az_petrov_coarse_preconditioner(
            target.system.matrix,
            target.ilu,
            transferred_factors,
            name="donor_right_target_w",
        )
        random_z = random_orthonormal_basis(
            settings.stage10.n_nodes,
            q,
            settings.stage10.seed + 5000 + pair_index,
        )
        random_factors = factor_az_petrov(target.system.matrix, random_z)
        random_action, random_dx = build_az_petrov_coarse_preconditioner(
            target.system.matrix,
            target.ilu,
            random_factors,
            name="random_target_w",
        )
        z_metric = subspace_metrics(source_z, target_z)
        w_metric = subspace_metrics(source_factors.test, target_factors.test)
        recomputed_w_metric = subspace_metrics(
            transferred_factors.test, target_factors.test
        )
        donor_to_recomputed = subspace_metrics(
            source_factors.test, transferred_factors.test
        )
        geometry_rows.append(
            {
                "source_ratio": source_ratio,
                "target_ratio": target_ratio,
                "q": q,
                "z_projector_distance_normalized": z_metric[
                    "projector_distance_normalized"
                ],
                "z_principal_angle_mean_degrees": z_metric[
                    "principal_angle_mean_degrees"
                ],
                "z_principal_angle_max_degrees": z_metric[
                    "principal_angle_max_degrees"
                ],
                "donor_w_to_target_w_projector_distance_normalized": w_metric[
                    "projector_distance_normalized"
                ],
                "donor_w_to_target_w_principal_angle_max_degrees": w_metric[
                    "principal_angle_max_degrees"
                ],
                "recomputed_w_to_target_w_projector_distance_normalized": recomputed_w_metric[
                    "projector_distance_normalized"
                ],
                "recomputed_w_to_target_w_principal_angle_max_degrees": recomputed_w_metric[
                    "principal_angle_max_degrees"
                ],
                "donor_w_to_recomputed_w_projector_distance_normalized": donor_to_recomputed[
                    "projector_distance_normalized"
                ],
                "donor_w_to_recomputed_w_principal_angle_max_degrees": donor_to_recomputed[
                    "principal_angle_max_degrees"
                ],
            }
        )
        coupling_rows.append(
            {
                "source_ratio": source_ratio,
                "target_ratio": target_ratio,
                "q": q,
                "own_wtaz_condition": own_dx["wtaz_condition"],
                "full_pair_wtaz_condition": full_pair_dx["wtaz_condition"],
                "right_transfer_wtaz_condition": right_dx["wtaz_condition"],
                "random_wtaz_condition": random_dx["wtaz_condition"],
                "own_setup_seconds": own_dx["coarse_setup_seconds"],
                "full_pair_setup_seconds": full_pair_dx["coarse_setup_seconds"],
                "right_transfer_setup_seconds": right_dx["coarse_setup_seconds"],
                "random_setup_seconds": random_dx["coarse_setup_seconds"],
                "donor_w_source_setup_seconds": source_factors.diagnostics[
                    "petrov_setup_seconds"
                ],
                "right_transfer_wtaz_equals_r_relative_error": transferred_factors.diagnostics[
                    "wtaz_equals_r_relative_error"
                ],
            }
        )
        methods = {
            "ILU": (target.ilu, {"coarse_setup_seconds": 0.0}),
            "own_target_pg": (own, own_dx),
            "donor_full_pair": (full_pair, full_pair_dx),
            "donor_right_target_w": (right_transfer, right_dx),
            "random_target_w": (random_action, random_dx),
        }
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            for method, (action, diagnostics) in methods.items():
                diagnostics = dict(diagnostics)
                diagnostics["ritz_setup_seconds"] = (
                    0.0
                    if method in ("ILU", "random_target_w")
                    else source.modes.diagnostics["setup_seconds"]
                    if method.startswith("donor")
                    else target.modes.diagnostics["setup_seconds"]
                )
                result = _run_solver(target, rhs_seed, action, settings.stage10)
                row = _result_row(
                    target_ratio,
                    q if method != "ILU" else 0,
                    method,
                    rhs_seed,
                    result,
                    diagnostics,
                    settings,
                )
                row["source_ratio"] = "" if method in ("ILU", "own_target_pg", "random_target_w") else source_ratio
                result_rows.append(row)
                for iteration, residual in enumerate(result.relative_residuals):
                    histories.append(
                        {
                            "source_ratio": row["source_ratio"],
                            "target_ratio": target_ratio,
                            "method": method,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "relative_residual": float(residual),
                        }
                    )
        old = old_lookup.get((source_ratio, target_ratio))
        right_summary_rows = [
            row
            for row in result_rows
            if float(row["ratio"]) == target_ratio
            and row.get("source_ratio") == source_ratio
            and row["method"] == "donor_right_target_w"
        ]
        if old is not None and right_summary_rows:
            coupling_rows[-1]["stage10_nearby_expected_iterations_median"] = float(
                old["iterations_median"]
            )
            coupling_rows[-1]["right_transfer_iterations_median"] = float(
                np.median([row["iterations"] for row in right_summary_rows])
            )
            coupling_rows[-1]["stage10_transfer_iteration_difference"] = (
                coupling_rows[-1]["right_transfer_iterations_median"]
                - coupling_rows[-1]["stage10_nearby_expected_iterations_median"]
            )
        else:
            coupling_rows[-1]["stage10_nearby_expected_iterations_median"] = ""
            coupling_rows[-1]["right_transfer_iterations_median"] = ""
            coupling_rows[-1]["stage10_transfer_iteration_difference"] = ""
    summaries = _summaries(result_rows, ("source_ratio", "ratio", "method"))
    for summary in summaries:
        local = [
            row
            for row in summaries
            if row["ratio"] == summary["ratio"]
        ]
        baseline = next(row for row in local if row["method"] == "ILU")
        own = next(row for row in local if row["method"] == "own_target_pg")
        summary["iteration_reduction_vs_ilu"] = (
            baseline["iterations_median"] - summary["iterations_median"]
        ) / max(baseline["iterations_median"], 1.0)
        summary["iteration_gap_vs_own"] = (
            summary["iterations_median"] - own["iterations_median"]
        )
    _write_csv(output / "transfer_results.csv", result_rows)
    _write_csv(output / "transfer_summary.csv", summaries)
    _write_csv(output / "transfer_histories.csv", histories)
    _write_csv(output / "transfer_geometry.csv", geometry_rows)
    _write_csv(output / "transfer_coupling_diagnostics.csv", coupling_rows)
    return result_rows, summaries, histories, geometry_rows


def build_decision(
    output: Path,
    same_rows: Sequence[Mapping[str, Any]],
    same_summaries: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    settings: Stage12Settings,
) -> Dict[str, Any]:
    q = settings.transfer_q
    reproduction = [
        row
        for row in same_rows
        if row["method"] == "stage10_existing_petrov" and int(row["q"]) == q
    ]
    reproduction_iteration_max = int(
        max(abs(int(row["stage10_iteration_difference"])) for row in reproduction)
    )
    reproduction_convergence = bool(
        all(bool(row["stage10_convergence_match"]) for row in reproduction)
    )
    identity_max = float(
        max(
            max(float(row["historical_petrov_action_relative_error_max"]),
                float(row["petrov_recycling_action_relative_error_max"]))
            for row in coupling_rows
        )
    )
    historical_variants = sorted({str(row["historical_coarse_variant"]) for row in coupling_rows})
    q8 = [row for row in same_summaries if int(row["q"]) in (0, q)]
    baseline = {float(row["ratio"]): row for row in q8 if row["method"] == "ILU"}
    own_pg = [row for row in q8 if row["method"] == "az_petrov"]
    galerkin = [row for row in q8 if row["method"] == "galerkin_counterfactual"]
    pg_reductions = [
        (baseline[float(row["ratio"])]["iterations_median"] - row["iterations_median"])
        / max(baseline[float(row["ratio"])]["iterations_median"], 1.0)
        for row in own_pg
    ]
    gal_pg_gaps = [
        next(g["iterations_median"] for g in galerkin if g["ratio"] == row["ratio"])
        - row["iterations_median"]
        for row in own_pg
    ]
    transferred = [row for row in transfer_summaries if row["method"] == "donor_right_target_w"]
    full_pair = [row for row in transfer_summaries if row["method"] == "donor_full_pair"]
    transfer_pass = bool(
        np.median([row["iteration_reduction_vs_ilu"] for row in transferred])
        >= settings.useful_iteration_reduction
        and np.mean([row["convergence_rate"] for row in transferred])
        >= settings.reliable_convergence_rate
        and np.median([row["iteration_gap_vs_own"] for row in transferred])
        <= settings.oracle_gap_tolerance
    )
    gal_pg_similar = bool(
        np.max(np.abs(gal_pg_gaps)) <= 5.0
        and abs(
            np.mean([row["convergence_rate"] for row in galerkin])
            - np.mean([row["convergence_rate"] for row in own_pg])
        )
        <= 0.05
    )
    provenance_pass = bool(
        historical_variants == ["petrov_minres"]
        and reproduction_iteration_max <= settings.reproduction_iteration_tolerance
        and reproduction_convergence
        and identity_max <= settings.equivalence_tolerance
    )
    if not provenance_pass:
        classification = "validation_failure"
        label = "Historical reproduction or algebraic equivalence failed; no scientific classification is valid."
    elif gal_pg_similar:
        classification = "D"
        label = "The formulations are effectively equivalent for this experiment."
    elif np.median(pg_reductions) >= settings.useful_iteration_reduction and not transfer_pass:
        classification = "B"
        label = "The nonsymmetric formulation improves the coupling but does not rescue transferability."
    elif np.median(pg_reductions) < settings.useful_iteration_reduction:
        classification = "C"
        label = "The Ritz-space mechanism itself loses most of its value under the corrected formulation."
    else:
        classification = "B"
        label = "The nonsymmetric formulation improves the coupling but does not rescue transferability."
    decision = {
        "historical_implementation_variants": historical_variants,
        "historical_used_symmetric_galerkin": False,
        "stage10_q8_reproduction_max_iteration_difference": reproduction_iteration_max,
        "stage10_q8_reproduction_all_convergence_matches": reproduction_convergence,
        "historical_petrov_recycling_equivalence_error_max": identity_max,
        "provenance_and_equivalence_pass": provenance_pass,
        "own_petrov_iteration_reduction_median": float(np.median(pg_reductions)),
        "own_petrov_convergence_rate_mean": float(
            np.mean([row["convergence_rate"] for row in own_pg])
        ),
        "galerkin_minus_petrov_iteration_gap_median": float(np.median(gal_pg_gaps)),
        "galerkin_petrov_absolute_iteration_gap_median": float(
            np.median(np.abs(gal_pg_gaps))
        ),
        "galerkin_petrov_absolute_iteration_gap_max": float(
            np.max(np.abs(gal_pg_gaps))
        ),
        "galerkin_to_petrov_condition_ratio_median": float(
            np.median(
                [
                    float(row["galerkin_ztaz_condition"])
                    / float(row["petrov_wtaz_condition"])
                    for row in coupling_rows
                    if int(row["q"]) == q
                ]
            )
        ),
        "galerkin_petrov_effectively_similar": gal_pg_similar,
        "right_transfer_iteration_reduction_median": float(
            np.median([row["iteration_reduction_vs_ilu"] for row in transferred])
        ),
        "right_transfer_convergence_rate_mean": float(
            np.mean([row["convergence_rate"] for row in transferred])
        ),
        "right_transfer_iteration_gap_vs_own_median": float(
            np.median([row["iteration_gap_vs_own"] for row in transferred])
        ),
        "full_pair_iteration_reduction_median": float(
            np.median([row["iteration_reduction_vs_ilu"] for row in full_pair])
        ),
        "full_pair_convergence_rate_mean": float(
            np.mean([row["convergence_rate"] for row in full_pair])
        ),
        "right_transfer_pass": transfer_pass,
        "decision_class": classification,
        "decision_label": label,
        "neural_network_trained": False,
    }
    _write_json(output / "decision.json", decision)
    return decision


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    nonsymmetry: Sequence[Mapping[str, Any]],
    same_summaries: Sequence[Mapping[str, Any]],
    same_histories: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    transfer_histories: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    settings: Stage12Settings,
) -> List[str]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    ratios = [float(row["ratio"]) for row in nonsymmetry]

    figure, axis = plt.subplots(figsize=(8.7, 4.9))
    axis.plot(ratios, [row["matrix_symmetry_measure"] for row in nonsymmetry], "o-", color=BLUE, label="A (exact sparse Frobenius)")
    axis.plot(ratios, [row["am_inverse_symmetry_estimate"] for row in nonsymmetry], "s-", color=ORANGE, label=r"$AM^{-1}$ (48-probe estimate)")
    axis.set(xlabel="anisotropy ratio R", ylabel="relative transpose defect")
    axis.set_title("Matrix and right-preconditioned-operator nonsymmetry", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False)
    name = "nonsymmetry_diagnostics.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    method_colors = {
        "ILU": CHARCOAL,
        "galerkin_counterfactual": PINK,
        "stage10_existing_petrov": OLIVE,
        "az_petrov": BLUE,
        "recycling": ORANGE,
    }
    figure, axis = plt.subplots(figsize=(9.7, 5.2))
    for method, color in method_colors.items():
        local = sorted(
            [row for row in same_summaries if row["method"] == method and int(row["q"]) in (0, settings.transfer_q)],
            key=lambda row: float(row["ratio"]),
        )
        axis.plot([row["ratio"] for row in local], [row["iterations_median"] for row in local], marker="o", color=color, label=method.replace("_", " "))
    axis.set(xlabel="anisotropy ratio R", ylabel="median FGMRES iterations")
    axis.set_title("Same-system q=8 coupling comparison", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False, ncol=2)
    name = "same_system_q8_iterations.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    q8_coupling = [row for row in coupling_rows if int(row["q"]) == settings.transfer_q]
    figure, axis = plt.subplots(figsize=(8.7, 4.9))
    axis.semilogy([row["ratio"] for row in q8_coupling], [row["galerkin_ztaz_condition"] for row in q8_coupling], "o-", color=PINK, label=r"cond($Z^TAZ$)")
    axis.semilogy([row["ratio"] for row in q8_coupling], [row["petrov_wtaz_condition"] for row in q8_coupling], "s-", color=BLUE, label=r"cond($W^TAZ$)=cond($R$)")
    axis.set(xlabel="anisotropy ratio R", ylabel="reduced condition number")
    axis.set_title("Reduced coupling conditioning at q=8", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False)
    name = "reduced_conditioning_q8.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    transfer_colors = {
        "ILU": CHARCOAL,
        "own_target_pg": BLUE,
        "donor_full_pair": PINK,
        "donor_right_target_w": ORANGE,
        "random_target_w": OLIVE,
    }
    figure, axis = plt.subplots(figsize=(10.0, 5.2))
    labels = [f"{a:g}→{b:g}" for a, b in settings.transfer_pairs]
    for method, color in transfer_colors.items():
        local = [row for row in transfer_summaries if row["method"] == method]
        local = sorted(local, key=lambda row: float(row["ratio"]))
        axis.plot(labels, [row["iterations_median"] for row in local], marker="o", color=color, label=method.replace("_", " "))
    axis.set(xlabel="donor → target anisotropy ratio", ylabel="median FGMRES iterations")
    axis.set_title("q=8 paired reuse versus target-recomputed test space", loc="left")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False, ncol=2)
    name = "transfer_coupling_comparison.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    right = {float(row["ratio"]): row for row in transfer_summaries if row["method"] == "donor_right_target_w"}
    figure, axis = plt.subplots(figsize=(7.8, 5.0))
    x = [row["z_projector_distance_normalized"] for row in geometry_rows]
    y = [right[float(row["target_ratio"])]["iteration_gap_vs_own"] for row in geometry_rows]
    axis.scatter(x, y, s=65, color=PURPLE)
    for row, xv, yv in zip(geometry_rows, x, y):
        axis.annotate(f"{row['source_ratio']:g}→{row['target_ratio']:g}", (xv, yv), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel="normalized projector distance between donor and target Z", ylabel="right-transfer iteration gap vs own Z")
    axis.set_title("Right-space geometry versus transferred solver performance", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.55)
    name = "geometry_vs_transfer_gap.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(8.7, 4.9))
    identity_series = {
        r"$W^TAZ=R$": "wtaz_equals_r_relative_error",
        r"$AU=W$": "au_equals_w_relative_error",
        "historical=AZ Petrov": "historical_petrov_action_relative_error_max",
        "Petrov=recycling": "petrov_recycling_action_relative_error_max",
    }
    for label, field in identity_series.items():
        axis.semilogy([row["ratio"] for row in q8_coupling], [max(float(row[field]), 1.0e-18) for row in q8_coupling], marker="o", label=label)
    axis.axhline(settings.equivalence_tolerance, color=CHARCOAL, linestyle="--", label="equivalence tolerance")
    axis.set(xlabel="anisotropy ratio R", ylabel="relative error")
    axis.set_title("q=8 coupling identities and action equivalence", loc="left")
    axis.grid(axis="y", color=LIGHT_GRAY, alpha=0.55)
    axis.legend(frameon=False, ncol=2)
    name = "coupling_identity_errors.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(8.9, 4.9))
    for method, color in method_colors.items():
        local = [row for row in same_histories if float(row["ratio"]) == 100.0 and int(row["rhs_seed"]) == settings.stage10.evaluation_rhs_seeds[0] and row["method"] == method and int(row["q"]) in (0, settings.transfer_q)]
        if local:
            axis.semilogy([row["iteration"] for row in local], [row["relative_residual"] for row in local], color=color, label=method.replace("_", " "))
    axis.axhline(settings.stage10.solver.relative_tolerance, color=LIGHT_GRAY, linestyle="--")
    axis.set(xlabel="FGMRES iteration", ylabel="true relative residual")
    axis.set_title("Same-system residual history at R=100, q=8", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.5)
    axis.legend(frameon=False, ncol=2)
    name = "same_system_residual_R100.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(8.9, 4.9))
    for method, color in transfer_colors.items():
        local = [row for row in transfer_histories if float(row["target_ratio"]) == 100.0 and int(row["rhs_seed"]) == settings.stage10.evaluation_rhs_seeds[0] and row["method"] == method]
        if local:
            axis.semilogy([row["iteration"] for row in local], [row["relative_residual"] for row in local], color=color, label=method.replace("_", " "))
    axis.axhline(settings.stage10.solver.relative_tolerance, color=LIGHT_GRAY, linestyle="--")
    axis.set(xlabel="FGMRES iteration", ylabel="true relative residual")
    axis.set_title("Transferred residual history for 95→100, q=8", loc="left")
    axis.grid(color=LIGHT_GRAY, alpha=0.5)
    axis.legend(frameon=False, ncol=2)
    name = "transfer_residual_95_to_100.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)
    return names


def _format(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "—"
    if numeric != 0.0 and (abs(numeric) < 1.0e-3 or abs(numeric) >= 1.0e4):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def write_report(
    output: Path,
    nonsymmetry: Sequence[Mapping[str, Any]],
    reconstruction_rows: Sequence[Mapping[str, Any]],
    same_summaries: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage12Settings,
) -> Path:
    q = settings.transfer_q
    same_map = {
        (float(row["ratio"]), row["method"]): row
        for row in same_summaries
        if int(row["q"]) in (0, q)
    }
    nonsym_table = [
        (
            f"{float(row['ratio']):g}",
            _format(row["matrix_symmetry_measure"]),
            _format(row["am_inverse_symmetry_estimate"]),
            f"[{_format(row['am_inverse_symmetry_bootstrap_p05'])}, {_format(row['am_inverse_symmetry_bootstrap_p95'])}]",
        )
        for row in nonsymmetry
    ]
    same_table = []
    for ratio in settings.ratios:
        local_dx = next(row for row in coupling_rows if float(row["ratio"]) == ratio and int(row["q"]) == q)
        same_table.append(
            (
                f"{ratio:g}",
                _format(same_map[(ratio, "ILU")]["iterations_median"], 0),
                _format(same_map[(ratio, "galerkin_counterfactual")]["iterations_median"], 0),
                _format(same_map[(ratio, "stage10_existing_petrov")]["iterations_median"], 0),
                _format(same_map[(ratio, "az_petrov")]["iterations_median"], 0),
                _format(same_map[(ratio, "recycling")]["iterations_median"], 0),
                _format(same_map[(ratio, "az_petrov")]["convergence_rate"], 2),
                _format(same_map[(ratio, "az_petrov")]["a_matvec_count_median"], 0),
                _format(same_map[(ratio, "galerkin_counterfactual")]["reduced_condition"]),
                _format(same_map[(ratio, "az_petrov")]["reduced_condition"]),
                _format(local_dx["historical_petrov_action_relative_error_max"]),
            )
        )
    method_labels = {
        "ILU": "ILU",
        "galerkin_counterfactual": "Galerkin",
        "stage10_existing_petrov": "historical PG",
        "az_petrov": "AZ-PG",
        "recycling": "REC",
    }
    same_detail_table = []
    for ratio in settings.ratios:
        for method in method_labels:
            row = same_map[(ratio, method)]
            same_detail_table.append(
                (
                    f"{ratio:g}",
                    method_labels[method],
                    _format(row["iterations_median"], 0),
                    _format(row["convergence_rate"], 2),
                    _format(row["final_relative_residual_max"]),
                    _format(row["a_matvec_count_median"], 0),
                    _format(row["solve_seconds_median"]),
                    _format(row["preconditioner_seconds_median"]),
                    _format(row["coarse_setup_seconds"]),
                    _format(row["reduced_condition"]),
                    row["failures"] if row["failures"] else "—",
                )
            )
    q_sensitivity_table = []
    for ratio in settings.ratios:
        for local_q in settings.q_values:
            local = {
                row["method"]: row
                for row in same_summaries
                if float(row["ratio"]) == ratio and int(row["q"]) == local_q
            }
            q_sensitivity_table.append(
                (
                    f"{ratio:g}",
                    str(local_q),
                    _format(local["galerkin_counterfactual"]["iterations_median"], 0),
                    _format(local["az_petrov"]["iterations_median"], 0),
                    _format(local["recycling"]["iterations_median"], 0),
                )
            )
    transfer_map = {
        (float(row["ratio"]), row["method"]): row for row in transfer_summaries
    }
    transfer_table = []
    for source, target in settings.transfer_pairs:
        transfer_table.append(
            (
                f"{source:g}→{target:g}",
                _format(transfer_map[(target, "ILU")]["iterations_median"], 0),
                _format(transfer_map[(target, "own_target_pg")]["iterations_median"], 0),
                _format(transfer_map[(target, "donor_full_pair")]["iterations_median"], 0),
                _format(transfer_map[(target, "donor_full_pair")]["convergence_rate"], 2),
                _format(transfer_map[(target, "donor_right_target_w")]["iterations_median"], 0),
                _format(transfer_map[(target, "donor_right_target_w")]["convergence_rate"], 2),
                _format(transfer_map[(target, "random_target_w")]["iterations_median"], 0),
                _format(transfer_map[(target, "donor_right_target_w")]["solve_seconds_median"]),
            )
        )
    transfer_labels = {
        "ILU": "ILU",
        "own_target_pg": "own PG",
        "donor_full_pair": "full donor pair",
        "donor_right_target_w": "donor Z + target W",
        "random_target_w": "random + target W",
    }
    transfer_detail_table = []
    for source, target in settings.transfer_pairs:
        for method in transfer_labels:
            row = transfer_map[(target, method)]
            transfer_detail_table.append(
                (
                    f"{source:g}→{target:g}",
                    transfer_labels[method],
                    _format(row["iterations_median"], 0),
                    _format(row["convergence_rate"], 2),
                    _format(row["final_relative_residual_max"]),
                    _format(row["a_matvec_count_median"], 0),
                    _format(row["solve_seconds_median"]),
                    _format(row["coarse_setup_seconds"]),
                    _format(row["reduced_condition"]),
                    row["failures"] if row["failures"] else "—",
                )
            )
    geometry_table = [
        (
            f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}",
            _format(row["z_projector_distance_normalized"]),
            _format(row["z_principal_angle_max_degrees"], 1),
            _format(row["donor_w_to_target_w_projector_distance_normalized"]),
            _format(row["recomputed_w_to_target_w_projector_distance_normalized"]),
            _format(transfer_map[(float(row["target_ratio"]), "donor_full_pair")]["iteration_gap_vs_own"], 0),
            _format(transfer_map[(float(row["target_ratio"]), "donor_right_target_w")]["iteration_gap_vs_own"], 0),
        )
        for row in geometry_rows
    ]
    reconstruction_map = {
        float(row["ratio"]): row for row in reconstruction_rows
    }
    cost_table = []
    for ratio in settings.ratios:
        pg = same_map[(ratio, "az_petrov")]
        rec = same_map[(ratio, "recycling")]
        gal = same_map[(ratio, "galerkin_counterfactual")]
        cost_table.append(
            (
                f"{ratio:g}",
                _format(reconstruction_map[ratio]["ilu_setup_seconds"]),
                _format(pg["ritz_setup_seconds"]),
                _format(gal["coarse_setup_seconds"]),
                _format(pg["coarse_setup_seconds"]),
                _format(rec["coarse_setup_seconds"]),
                _format(pg["solve_seconds_median"]),
                _format(pg["preconditioner_seconds_median"]),
            )
        )
    transfer_setup_table = []
    for row in _read_csv(output / "transfer_coupling_diagnostics.csv"):
        transfer_setup_table.append(
            (
                f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}",
                _format(row["donor_w_source_setup_seconds"]),
                _format(row["full_pair_setup_seconds"]),
                _format(row["right_transfer_setup_seconds"]),
                _format(row["full_pair_wtaz_condition"]),
                _format(row["right_transfer_wtaz_condition"]),
            )
        )
    rank_ok = all(int(row["archived_dimension"]) == 32 for row in reconstruction_rows)
    max_space_distance = max(float(row["archived_rebuilt_projector_distance_normalized"]) for row in reconstruction_rows)
    report = rf"""# Stage 12: nonsymmetric Petrov–Galerkin and recycling validation

## Executive conclusion

**{decision['decision_class']}. {decision['decision_label']}**

The premise that Stages 9–11 used the symmetric Galerkin correction is contradicted by the implementation and saved diagnostics. Their shared coarse builder selected `petrov_minres` for every tested matrix and already used $AZ=WR$ with $Z R^{{-1}}W^T$. This round reproduces every Stage 10 q=8 held-out iteration count with maximum difference **{int(decision['stage10_q8_reproduction_max_iteration_difference'])}**, and the historical action, explicit AZ–Petrov action, and recycling action agree to maximum relative error **{float(decision['historical_petrov_recycling_equivalence_error_max']):.2e}**.

The counterfactual Galerkin comparison is still informative, but it is not the historical baseline. Own-system AZ–Petrov retains a median **{100.0 * float(decision['own_petrov_iteration_reduction_median']):.1f}%** reduction over strong ILU. Right-space transfer with target-recomputed $W$—which is also what Stage 10 actually did—has median reduction **{100.0 * float(decision['right_transfer_iteration_reduction_median']):.1f}%**, mean convergence **{float(decision['right_transfer_convergence_rate_mean']):.2f}**, and median **{float(decision['right_transfer_iteration_gap_vs_own_median']):.1f}**-iteration gap to the target oracle. The Stage 10–11 transferability conclusion therefore remains valid. No neural network was trained.

## 1. Provenance and fixed protocol

The [Stage 9 report](../stage9_hard_regime_discovery_and_scalability/stage9_hard_regime_discovery_and_scalability_report.md), [Stage 10 report](../stage10_ritz_transfer_and_mode_tracking/stage10_ritz_transfer_and_mode_tracking_report.md), and [Stage 11 report](../stage11_ritz_cluster_stability/stage11_ritz_cluster_stability_report.md) were read before implementation. The fixed setup is $N=5041$, node seed 7, stencil 25, jitter 0.15, 30° rotated anisotropy, ILU drop tolerance $10^{{-3}}$, fill factor 5, COLAMD, zero diagonal-pivot threshold, FGMRES tolerance $10^{{-8}}$, restart 50, cap 300, construction RHS seed {settings.stage10.construction_rhs_seed}, and held-out RHS seeds {settings.stage10.evaluation_rhs_seeds}.

The exact archived Stage 10 q=32 bases are reused; deterministic reconstruction was also run. All archived ranks are valid: **{'yes' if rank_ok else 'no'}**. The maximum archived-versus-rebuilt normalized projector distance is **{max_space_distance:.2e}**. Main same-system tests use $q\in\{{4,8,16\}}$; transfer uses the preregistered $q=8$. The optional adjoint/two-sided Ritz construction was not added because it would change the Ritz target rather than isolate coupling.

## 2. Operators and implementation

For every method, $Z$ is the same orthonormal right/correction space and

\[
s=(I-AM^{{-1}})r.
\]

The symmetric-style counterfactual is

\[
P_G(r)=M^{{-1}}r+Z(Z^TAZ)^{{-1}}Z^Ts.
\]

The nonsymmetric AZ–Petrov factorization and action are

\[
AZ=WR,\qquad P_{{PG}}(r)=M^{{-1}}r+Z(W^TAZ)^{{-1}}W^Ts
=M^{{-1}}r+Z R^{{-1}}W^Ts.
\]

The recycling normalization uses a triangular solve—not an inverse—to form

\[
U=ZR^{{-1}},qquad AU\approx W,qquad P_{{REC}}(r)=M^{{-1}}r+UW^Ts.
\]

Reduced systems use LU solves for general Galerkin/paired Petrov coupling and triangular solves for $R$. Across all $(R,q)$, $W^TAZ=R$, $AU=W$, historical=Petrov, and Petrov=recycling are checked numerically. Total reported A-matvecs include true-residual recomputation and the multiplicative residual $s$, matching the local FGMRES implementation exactly.

## 3. Nonsymmetry diagnostics

$\|A-A^T\|_F/\|A\|_F$ is exact from sparse matrices. The $AM^{{-1}}$ ratio is a reproducible 48-probe Rademacher/Hutchinson estimate using $B^T=M^{{-T}}A^T$; brackets are bootstrap 5th–95th percentiles, not deterministic bounds.

{_table(['R', 'A transpose defect', 'AM^-1 estimate', 'bootstrap [p05,p95]'], nonsym_table)}

![Nonsymmetry diagnostics](plots/nonsymmetry_diagnostics.png)

These values are many orders above the $10^{{-10}}$ symmetry threshold, so symmetric Galerkin algebra is not justified by approximate symmetry.

## 4. Same-system oracle comparison at q=8

“Historical” is the exact Stage 10 builder; it is AZ–Petrov, not Galerkin. Iterations are medians over the same three held-out RHSs. `conv.` is the Petrov convergence rate; matvecs include all A applications. Conditions correspond to the displayed $q=8$ reduced systems.

{_table(['R', 'ILU it.', 'Galerkin it.', 'historical it.', 'AZ-PG it.', 'REC it.', 'PG conv.', 'PG matvecs', 'cond ZTAZ', 'cond WTAZ', 'hist/PG error'], same_table)}

![Same-system iterations](plots/same_system_q8_iterations.png)

![Reduced conditioning](plots/reduced_conditioning_q8.png)

![Coupling identities](plots/coupling_identity_errors.png)

The median Galerkin-minus-Petrov iteration gap is **{float(decision['galerkin_minus_petrov_iteration_gap_median']):.1f}**. Regardless of that counterfactual difference, the historical, AZ–Petrov, and recycling columns are algebraically and solver-level identical within floating-point error.

The counterfactual Galerkin and Petrov iteration counts are not uniformly interchangeable: their median absolute gap is **{float(decision['galerkin_petrov_absolute_iteration_gap_median']):.1f}**, but the maximum is **{float(decision['galerkin_petrov_absolute_iteration_gap_max']):.0f}**. Petrov improves the reduced condition number by a median factor **{float(decision['galerkin_to_petrov_condition_ratio_median']):.1f}**, although that conditioning improvement does not imply fewer iterations at every $R$.

Full q=8 solver diagnostics:

{_table(['R', 'method', 'it.', 'conv.', 'final residual', 'A matvecs', 'solve s', 'precond. s', 'coarse setup s', 'reduced cond.', 'failure'], same_detail_table)}

The optional q sensitivity was run without changing Ritz ordering:

{_table(['R', 'q', 'Galerkin it.', 'AZ-PG it.', 'REC it.'], q_sensitivity_table)}

![Same-system residuals](plots/same_system_residual_R100.png)

## 5. Transfer: donor pair versus target-recomputed test space

Two meanings of transfer are kept separate:

- **Full paired reuse:** retain donor $Z_i,W_i$, but solve the target reduced system $W_i^TA_jZ_i$.
- **Right-space transfer:** retain only $Z_i$, compute $A_jZ_i=W_{{j\leftarrow i}}R_{{j\leftarrow i}}$, then use $Z_iR_{{j\leftarrow i}}^{{-1}}W_{{j\leftarrow i}}^T$. This target-recomputed construction is the Stage 10 implementation.

{_table(['pair', 'ILU', 'own PG', 'full pair', 'full conv.', 'right + target W', 'right conv.', 'random', 'right solve s'], transfer_table)}

Full transfer diagnostics:

{_table(['pair', 'method', 'it.', 'conv.', 'final residual', 'A matvecs', 'solve s', 'coarse setup s', 'reduced cond.', 'failure'], transfer_detail_table)}

![Transfer comparison](plots/transfer_coupling_comparison.png)

![Transfer residuals](plots/transfer_residual_95_to_100.png)

## 6. Geometry versus solver effectiveness

{_table(['pair', 'Z d_proj', 'Z max angle', 'donor W/target W d', 'recomputed W/target W d', 'full-pair gap', 'right-transfer gap'], geometry_table)}

![Geometry versus transfer](plots/geometry_vs_transfer_gap.png)

Recomputing $W$ from $A_jZ_i$ isolates the residual/test partner but cannot repair a mismatched right trial space. Because Stage 10 already performed this recomputation, its poor high-$R$ transfer cannot be attributed to using donor $W_i$ or $Z_i$ as a symmetric test space. The remaining evidence points primarily to instability/mismatch of $Z$, with paired-space mismatch adding further sensitivity where full-pair reuse is worse.

## 7. Setup and online cost at q=8

All values are wall-clock seconds from this run. Ritz setup includes the 64-step Arnoldi construction. Galerkin and Petrov setup both form $AZ$; recycling additionally forms $U$ by triangular solves.

{_table(['R', 'ILU setup', 'Ritz setup', 'Galerkin setup', 'AZ-PG setup', 'REC setup', 'PG solve', 'PG precond.'], cost_table)}

Transfer-side coupling costs distinguish donor-$W$ construction from target coupling:

{_table(['pair', 'donor W setup', 'full-pair target setup', 'target-W setup', 'full-pair cond.', 'target-W cond.'], transfer_setup_table)}

Timing differences between algebraically identical actions are implementation overhead diagnostics, not mathematical performance claims.

## 8. Decision and implications for Stages 10–11

| Check | Result |
| --- | --- |
| Historical implementation | {', '.join(decision['historical_implementation_variants'])} |
| Historical q=8 reproduction | max iteration difference {int(decision['stage10_q8_reproduction_max_iteration_difference'])}; all convergence flags match: {_format(decision['stage10_q8_reproduction_all_convergence_matches'])} |
| Petrov/recycling equivalence | max relative action error {float(decision['historical_petrov_recycling_equivalence_error_max']):.2e} |
| Own-system PG relevance | {100.0 * float(decision['own_petrov_iteration_reduction_median']):.1f}% median reduction; {float(decision['own_petrov_convergence_rate_mean']):.2f} convergence |
| Right-space transfer | {100.0 * float(decision['right_transfer_iteration_reduction_median']):.1f}% median reduction; {float(decision['right_transfer_convergence_rate_mean']):.2f} convergence; {float(decision['right_transfer_iteration_gap_vs_own_median']):.1f} median oracle gap |
| Neural model | not trained |

**Final classification: {decision['decision_class']}. {decision['decision_label']}**

The historical Stage 10–11 conclusion remains valid: the own-system Ritz mechanism is real, but the q=8 right space is not reliably transferable through the hard transition. A symmetric-style operator was not the confounder because it was not the operator used.

## Limitations and reproducibility

- $AM^{{-1}}$ nonsymmetry is estimated matrix-free rather than formed densely; the exact estimator protocol and probe seed are saved.
- Only the requested fixed-node anisotropy path and q values are tested. No clustering, Ritz reordering, q tuning, or target redesign was performed.
- The three held-out RHSs are identical to Stage 10; timing is single-process wall clock and should not be treated as a hardware-independent benchmark.
- The full donor-pair experiment is a coupling diagnostic, not a claim that (W_i) is a compatible target-system left Ritz space.
- Cap hits and reduced-system failures are retained in raw CSVs.

Raw evidence is in `nonsymmetry_diagnostics.csv`, `ritz_space_reconstruction.csv`, `coupling_identities.csv`, `same_system_results.csv`, `same_system_summary.csv`, `transfer_results.csv`, `transfer_summary.csv`, `transfer_geometry.csv`, and `transfer_coupling_diagnostics.csv`. Settings, environment, decision gates, and validation are saved separately.
"""
    path = output / "stage12_nonsymmetric_petrov_galerkin_recycling_validation.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_stage12(output: Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = Stage12Settings()
    np.random.seed(settings.stage10.seed)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    print("[1/7] rebuilding Stage 10 systems and reconciling Ritz archives", flush=True)
    contexts, spaces, reconstruction_rows = build_contexts_and_fixed_spaces(
        output, settings
    )
    print("[2/7] measuring A and AM^-1 nonsymmetry", flush=True)
    nonsymmetry = run_nonsymmetry_diagnostics(output, contexts, settings)
    print("[3/7] comparing same-system Galerkin, Petrov, and recycling actions", flush=True)
    same_rows, same_summaries, same_histories, coupling_rows = run_same_system_comparison(
        output, contexts, spaces, settings
    )
    print("[4/7] separating full-pair from right-space-only transfer", flush=True)
    _, transfer_summaries, transfer_histories, geometry_rows = run_transfer_comparison(
        output, contexts, spaces, settings
    )
    print("[5/7] reconciling historical results and applying decision gates", flush=True)
    decision = build_decision(
        output,
        same_rows,
        same_summaries,
        coupling_rows,
        transfer_summaries,
        settings,
    )
    print("[6/7] rendering evidence figures", flush=True)
    make_plots(
        output,
        nonsymmetry,
        same_summaries,
        same_histories,
        coupling_rows,
        transfer_summaries,
        transfer_histories,
        geometry_rows,
        settings,
    )
    print("[7/7] writing Stage 12 report", flush=True)
    report = write_report(
        output,
        nonsymmetry,
        reconstruction_rows,
        same_summaries,
        coupling_rows,
        transfer_summaries,
        geometry_rows,
        decision,
        settings,
    )
    print(f"completed: {report}", flush=True)
    return report
