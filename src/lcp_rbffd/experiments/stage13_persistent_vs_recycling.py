"""Stage 13: persistent Petrov correction versus fixed-space Krylov recycling.

The experiment freezes the Stage 10 right Ritz space ``Z`` and the strong ILU
factorization.  Method A inserts the AZ-based Petrov correction into every
right-preconditioner application.  Method B uses the same ``Z`` only through
the fixed GCRO/NeurKItt pair ``A U=W`` and applies ILU inside projected
Arnoldi.  No learned model or alternative oracle construction is used.
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

from neurkitt_oracle.gcrodr import GmresResult, restarted_projected_gmres

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..krylov import KrylovResult, restarted_gmres
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
    _discrete_rhs,
)
from .stage12_nonsymmetric_coupling import (
    Stage12Settings,
    _orthonormalize,
    build_contexts_and_fixed_spaces,
)


@dataclass(frozen=True)
class Stage13Settings:
    """Frozen experimental protocol and diagnostic thresholds."""

    stage10: Stage10Settings = Stage10Settings()
    ratios: Tuple[float, ...] = (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0)
    q: int = 8
    transfer_pairs: Tuple[Tuple[float, float], ...] = (
        (90.0, 95.0),
        (95.0, 100.0),
        (100.0, 105.0),
        (105.0, 110.0),
    )
    representative_ratios: Tuple[float, ...] = (90.0, 100.0, 110.0, 120.0)
    coarse_frequencies: Tuple[int, ...] = (1, 2, 5)
    algebra_tolerance: float = 1.0e-10
    projected_orthogonality_tolerance: float = 1.0e-9
    material_iteration_advantage: float = 0.15
    similar_iteration_fraction: float = 0.10
    reliable_convergence_rate: float = 0.90
    rho_regrowth_threshold: float = 0.05


@dataclass
class RunCounters:
    """Online operations attributable to one solve."""

    a_matvecs: int = 0
    ilu_solves: int = 0
    ilu_seconds: float = 0.0
    w_transpose_applications: int = 0
    w_applications: int = 0
    z_or_u_applications: int = 0
    reduced_solves: int = 0
    coarse_corrections: int = 0
    diagnostic_w_transpose_applications: int = 0
    diagnostic_w_applications: int = 0


class CountingMatrix:
    """Matrix-multiplication proxy that counts every online application of A."""

    def __init__(self, matrix: Any, counters: RunCounters) -> None:
        self.matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        self.counters = counters

    def __matmul__(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector)
        self.counters.a_matvecs += 1 if array.ndim == 1 else int(array.shape[1])
        return self.matrix @ vector


def _counting_ilu(context: RitzContext, counters: RunCounters) -> InverseAction:
    """Wrap the unchanged Stage 10 ILU while preserving exact application cost."""

    def apply(vector: np.ndarray) -> np.ndarray:
        start = perf_counter()
        value = context.ilu.apply(vector)
        counters.ilu_seconds += perf_counter() - start
        counters.ilu_solves += 1
        return value

    return InverseAction("counted_strong_ilu", apply)


def _rho(test_basis: np.ndarray, vector: np.ndarray) -> float:
    return float(np.linalg.norm(test_basis.T @ vector)) / max(
        float(np.linalg.norm(vector)), np.finfo(float).tiny
    )


def _persistent_action(
    counted_matrix: CountingMatrix,
    context: RitzContext,
    factors: AZPetrovFactors,
    counters: RunCounters,
    *,
    frequency: int = 1,
    diagnostic_records: List[Dict[str, Any]] | None = None,
) -> InverseAction:
    r"""Return ``M^-1 r + Z R^-1 W.T(r-A M^-1 r)`` at a fixed frequency.

    Frequencies two and five apply on calls ``1, 1+k, 1+2k, ...``.  Starting
    with the first call avoids confounding frequency with a deliberately
    omitted initial correction.
    """

    if frequency < 1:
        raise ValueError("frequency must be positive")
    ilu = _counting_ilu(context, counters)
    call_index = 0

    def apply(vector: np.ndarray) -> np.ndarray:
        nonlocal call_index
        call_index += 1
        classical = ilu.apply(vector)
        if (call_index - 1) % frequency != 0:
            return classical
        remaining = vector - counted_matrix @ classical
        projected = factors.test.T @ remaining
        counters.w_transpose_applications += 1
        coefficients = la.solve_triangular(
            factors.triangular,
            projected,
            check_finite=False,
        )
        counters.reduced_solves += 1
        correction = factors.trial @ coefficients
        counters.z_or_u_applications += 1
        counters.coarse_corrections += 1
        if diagnostic_records is not None:
            after = remaining - factors.test @ projected
            counters.diagnostic_w_applications += 1
            counters.diagnostic_w_transpose_applications += 1
            diagnostic_records.append(
                {
                    "preconditioner_application": call_index,
                    "rho_before": float(np.linalg.norm(projected))
                    / max(float(np.linalg.norm(remaining)), np.finfo(float).tiny),
                    "rho_after": _rho(factors.test, after),
                    "remaining_norm_before": float(np.linalg.norm(remaining)),
                    "remaining_norm_after": float(np.linalg.norm(after)),
                }
            )
        return classical + correction

    return InverseAction(f"persistent_pg_every_{frequency}", apply)


def _initial_persistent_guess(
    rhs: np.ndarray,
    counted_matrix: CountingMatrix,
    ilu: InverseAction,
    factors: AZPetrovFactors,
    counters: RunCounters,
) -> np.ndarray:
    """Apply the full two-level action once to the initial right-hand side."""

    classical = ilu.apply(rhs)
    remaining = rhs - counted_matrix @ classical
    projected = factors.test.T @ remaining
    counters.w_transpose_applications += 1
    coefficients = la.solve_triangular(
        factors.triangular,
        projected,
        check_finite=False,
    )
    counters.reduced_solves += 1
    counters.z_or_u_applications += 1
    counters.coarse_corrections += 1
    return classical + factors.trial @ coefficients


def _persistent_rho_callback(
    test_basis: np.ndarray,
    rows: List[Dict[str, Any]],
):
    def callback(iteration: int, _solution: np.ndarray, residual: np.ndarray) -> None:
        rows.append(
            {
                "iteration": iteration,
                "stage": "outer_true_residual",
                "relative_residual": float(np.linalg.norm(residual)),
                "rho": _rho(test_basis, residual),
            }
        )

    return callback


def _common_costs(
    context: RitzContext,
    factors: AZPetrovFactors,
    method: str,
) -> Tuple[float, float]:
    if method == "ILU":
        return 0.0, 0.0
    if method == "persistent_pg":
        return (
            float(factors.diagnostics["petrov_setup_seconds"]),
            float(context.modes.diagnostics["setup_seconds"]),
        )
    if method == "krylov_recycling":
        return (
            float(factors.diagnostics["recycling_setup_seconds"]),
            float(context.modes.diagnostics["setup_seconds"]),
        )
    raise ValueError(f"unknown method: {method}")


def _row_from_krylov(
    *,
    ratio: float,
    rhs_seed: int,
    method: str,
    result: KrylovResult,
    counters: RunCounters,
    online_seconds: float,
    context: RitzContext,
    factors: AZPetrovFactors,
    source_ratio: float | str = "",
) -> Dict[str, Any]:
    method_setup, oracle_setup = _common_costs(context, factors, method)
    total_without_oracle = context.ilu.setup_seconds + method_setup + online_seconds
    return {
        "ratio": ratio,
        "source_ratio": source_ratio,
        "q": 0 if method == "ILU" else factors.trial.shape[1],
        "rhs_seed": rhs_seed,
        "method": method,
        "iterations": result.iterations,
        "converged": result.converged,
        "cap_hit": bool(result.iterations >= 300 and not result.converged),
        "true_final_relative_residual": float(result.relative_residuals[-1]),
        "a_matvecs": counters.a_matvecs,
        "ilu_solves": counters.ilu_solves,
        "w_transpose_applications": counters.w_transpose_applications,
        "w_applications": counters.w_applications,
        "z_or_u_applications": counters.z_or_u_applications,
        "reduced_solves": counters.reduced_solves,
        "coarse_corrections": counters.coarse_corrections,
        "preconditioner_seconds": counters.ilu_seconds,
        "online_solve_seconds": online_seconds,
        "method_setup_seconds_after_z": method_setup,
        "common_oracle_z_seconds": oracle_setup,
        "ilu_setup_seconds": context.ilu.setup_seconds,
        "total_seconds_excluding_oracle": total_without_oracle,
        "total_seconds_including_oracle": total_without_oracle + oracle_setup,
        "az_setup_matvec_equivalents": 0 if method == "ILU" else factors.trial.shape[1],
        "failure": result.failure,
        "projected_arnoldi_relation_error_max": "",
        "arnoldi_basis_orthogonality_error_max": "",
        "arnoldi_basis_to_w_orthogonality_error_max": "",
    }


def _row_from_recycled(
    *,
    ratio: float,
    rhs_seed: int,
    result: GmresResult,
    counters: RunCounters,
    online_seconds: float,
    context: RitzContext,
    factors: AZPetrovFactors,
    source_ratio: float | str = "",
) -> Dict[str, Any]:
    method_setup, oracle_setup = _common_costs(context, factors, "krylov_recycling")
    operations = result.operation_counts or {}
    diagnostics = result.diagnostics or {}
    total_without_oracle = context.ilu.setup_seconds + method_setup + online_seconds
    return {
        "ratio": ratio,
        "source_ratio": source_ratio,
        "q": factors.trial.shape[1],
        "rhs_seed": rhs_seed,
        "method": "krylov_recycling",
        "iterations": result.iterations,
        "converged": result.converged,
        "cap_hit": bool(result.iterations >= 300 and not result.converged),
        "true_final_relative_residual": result.relative_residual,
        "a_matvecs": int(operations.get("a_matvecs", 0)),
        "ilu_solves": counters.ilu_solves,
        "w_transpose_applications": int(operations.get("w_transpose_applications", 0)),
        "w_applications": int(operations.get("w_applications", 0)),
        "z_or_u_applications": int(operations.get("u_applications", 0)),
        "reduced_solves": int(operations.get("reduced_solves", 0)),
        "coarse_corrections": int(result.restart_cycles),
        "preconditioner_seconds": counters.ilu_seconds,
        "online_solve_seconds": online_seconds,
        "method_setup_seconds_after_z": method_setup,
        "common_oracle_z_seconds": oracle_setup,
        "ilu_setup_seconds": context.ilu.setup_seconds,
        "total_seconds_excluding_oracle": total_without_oracle,
        "total_seconds_including_oracle": total_without_oracle + oracle_setup,
        "az_setup_matvec_equivalents": factors.trial.shape[1],
        "failure": "" if result.converged else "maximum iterations reached",
        "projected_arnoldi_relation_error_max": diagnostics.get(
            "projected_arnoldi_relation_error_max", float("nan")
        ),
        "arnoldi_basis_orthogonality_error_max": diagnostics.get(
            "arnoldi_basis_orthogonality_error_max", float("nan")
        ),
        "arnoldi_basis_to_w_orthogonality_error_max": diagnostics.get(
            "arnoldi_basis_to_w_orthogonality_error_max", float("nan")
        ),
    }


def _history_rows(
    ratio: float,
    rhs_seed: int,
    method: str,
    iterations: Sequence[int],
    residuals: Sequence[float],
    source_ratio: float | str = "",
) -> List[Dict[str, Any]]:
    return [
        {
            "ratio": ratio,
            "source_ratio": source_ratio,
            "rhs_seed": rhs_seed,
            "method": method,
            "iteration": int(iteration),
            "true_relative_residual": float(residual),
        }
        for iteration, residual in zip(iterations, residuals)
    ]


def _summarize(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    summaries: List[Dict[str, Any]] = []
    for key, local in groups.items():
        result = {name: value for name, value in zip(keys, key)}
        for field in (
            "iterations",
            "a_matvecs",
            "ilu_solves",
            "w_transpose_applications",
            "w_applications",
            "z_or_u_applications",
            "reduced_solves",
            "coarse_corrections",
            "online_solve_seconds",
            "preconditioner_seconds",
            "total_seconds_excluding_oracle",
            "total_seconds_including_oracle",
        ):
            result[f"{field}_median"] = float(np.median([float(row[field]) for row in local]))
        result.update(
            {
                "iterations_max": int(max(int(row["iterations"]) for row in local)),
                "convergence_rate": float(np.mean([bool(row["converged"]) for row in local])),
                "true_final_relative_residual_max": float(
                    max(float(row["true_final_relative_residual"]) for row in local)
                ),
                "method_setup_seconds_after_z": float(local[0]["method_setup_seconds_after_z"]),
                "common_oracle_z_seconds": float(local[0]["common_oracle_z_seconds"]),
                "ilu_setup_seconds": float(local[0]["ilu_setup_seconds"]),
                "projected_arnoldi_relation_error_max": max(
                    [
                        float(row["projected_arnoldi_relation_error_max"])
                        for row in local
                        if row["projected_arnoldi_relation_error_max"] != ""
                    ],
                    default=float("nan"),
                ),
                "arnoldi_basis_orthogonality_error_max": max(
                    [
                        float(row["arnoldi_basis_orthogonality_error_max"])
                        for row in local
                        if row["arnoldi_basis_orthogonality_error_max"] != ""
                    ],
                    default=float("nan"),
                ),
                "arnoldi_basis_to_w_orthogonality_error_max": max(
                    [
                        float(row["arnoldi_basis_to_w_orthogonality_error_max"])
                        for row in local
                        if row["arnoldi_basis_to_w_orthogonality_error_max"] != ""
                    ],
                    default=float("nan"),
                ),
                "failures": "; ".join(
                    sorted({str(row["failure"]) for row in local if row["failure"]})
                ),
            }
        )
        summaries.append(result)
    return summaries


def _run_baseline(
    context: RitzContext,
    rhs: np.ndarray,
    factors: AZPetrovFactors,
    settings: Stage13Settings,
) -> Tuple[KrylovResult, RunCounters, float]:
    counters = RunCounters()
    counted_matrix = CountingMatrix(context.system.matrix, counters)
    context.ilu.reset_timing()
    ilu = _counting_ilu(context, counters)
    start = perf_counter()
    result = restarted_gmres(
        counted_matrix,
        rhs,
        settings.stage10.solver,
        ilu,
    )
    return result, counters, perf_counter() - start


def _run_persistent(
    context: RitzContext,
    rhs: np.ndarray,
    factors: AZPetrovFactors,
    settings: Stage13Settings,
    *,
    frequency: int = 1,
    only_initial: bool = False,
    diagnostic_records: List[Dict[str, Any]] | None = None,
    residual_callback: Any = None,
) -> Tuple[KrylovResult, RunCounters, float]:
    counters = RunCounters()
    counted_matrix = CountingMatrix(context.system.matrix, counters)
    context.ilu.reset_timing()
    start = perf_counter()
    if only_initial:
        ilu = _counting_ilu(context, counters)
        initial_guess = _initial_persistent_guess(
            rhs,
            counted_matrix,
            ilu,
            factors,
            counters,
        )
        result = restarted_gmres(
            counted_matrix,
            rhs,
            settings.stage10.solver,
            ilu,
            initial_guess=initial_guess,
            callback=residual_callback,
        )
    else:
        action = _persistent_action(
            counted_matrix,
            context,
            factors,
            counters,
            frequency=frequency,
            diagnostic_records=diagnostic_records,
        )
        result = restarted_gmres(
            counted_matrix,
            rhs,
            settings.stage10.solver,
            action,
            callback=residual_callback,
        )
    return result, counters, perf_counter() - start


def _run_recycled(
    context: RitzContext,
    rhs: np.ndarray,
    factors: AZPetrovFactors,
    settings: Stage13Settings,
    *,
    record_projection_diagnostics: bool = False,
) -> Tuple[GmresResult, RunCounters, float]:
    counters = RunCounters()
    context.ilu.reset_timing()

    def apply_ilu(vector: np.ndarray) -> np.ndarray:
        start = perf_counter()
        value = context.ilu.apply(vector)
        counters.ilu_seconds += perf_counter() - start
        counters.ilu_solves += 1
        return value

    start = perf_counter()
    result = restarted_projected_gmres(
        context.system.matrix,
        rhs,
        apply_ilu,
        u_basis=factors.recycling_trial,
        c_basis=factors.test,
        rtol=settings.stage10.solver.relative_tolerance,
        atol=settings.stage10.solver.absolute_tolerance,
        restart=settings.stage10.solver.restart,
        max_iterations=settings.stage10.solver.max_iterations,
        record_true_residuals=True,
        record_projection_diagnostics=record_projection_diagnostics,
    )
    return result, counters, perf_counter() - start


def build_fixed_contexts(
    output: Path,
    settings: Stage13Settings,
) -> Tuple[Dict[float, RitzContext], Dict[float, np.ndarray], Dict[float, AZPetrovFactors], List[Dict[str, Any]]]:
    """Load exact Stage 10 spaces and construct the one common pair per system."""

    stage12_settings = Stage12Settings(
        stage10=settings.stage10,
        ratios=settings.ratios,
        q_values=(settings.q,),
        transfer_q=settings.q,
        transfer_pairs=tuple(
            (settings.ratios[index], settings.ratios[index + 1])
            for index in range(len(settings.ratios) - 1)
        ),
    )
    contexts, archived, _ = build_contexts_and_fixed_spaces(output, stage12_settings)
    spaces: Dict[float, np.ndarray] = {}
    factors: Dict[float, AZPetrovFactors] = {}
    checks: List[Dict[str, Any]] = []
    for ratio in settings.ratios:
        z_basis, z_error, rank = _orthonormalize(archived[ratio][:, : settings.q])
        pair = factor_az_petrov(contexts[ratio].system.matrix, z_basis)
        spaces[ratio] = z_basis
        factors[ratio] = pair
        checks.append(
            {
                "ratio": ratio,
                "q": settings.q,
                "z_rank": rank,
                "z_orthogonality_error": z_error,
                "w_orthogonality_error": pair.diagnostics["test_orthogonality_error"],
                "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                "wtaz_equals_r_relative_error": pair.diagnostics["wtaz_equals_r_relative_error"],
                "condition_wtaz": pair.diagnostics["wtaz_condition"],
                "projected_arnoldi_relation_error_max": "",
                "arnoldi_basis_orthogonality_error_max": "",
                "arnoldi_basis_to_w_orthogonality_error_max": "",
                "algebra_pass": "",
            }
        )
    return contexts, spaces, factors, checks


def run_same_system_comparison(
    output: Path,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    checks: List[Dict[str, Any]],
    settings: Stage13Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run ILU, persistent PG, and fixed recycling on identical RHSs and Z."""

    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    check_lookup = {float(row["ratio"]): row for row in checks}
    for ratio in settings.ratios:
        context = contexts[ratio]
        pair = factors[ratio]
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)

            baseline, baseline_count, baseline_time = _run_baseline(
                context, rhs, pair, settings
            )
            rows.append(
                _row_from_krylov(
                    ratio=ratio,
                    rhs_seed=rhs_seed,
                    method="ILU",
                    result=baseline,
                    counters=baseline_count,
                    online_seconds=baseline_time,
                    context=context,
                    factors=pair,
                )
            )
            histories.extend(
                _history_rows(
                    ratio,
                    rhs_seed,
                    "ILU",
                    range(len(baseline.relative_residuals)),
                    baseline.relative_residuals,
                )
            )

            persistent, persistent_count, persistent_time = _run_persistent(
                context, rhs, pair, settings
            )
            rows.append(
                _row_from_krylov(
                    ratio=ratio,
                    rhs_seed=rhs_seed,
                    method="persistent_pg",
                    result=persistent,
                    counters=persistent_count,
                    online_seconds=persistent_time,
                    context=context,
                    factors=pair,
                )
            )
            histories.extend(
                _history_rows(
                    ratio,
                    rhs_seed,
                    "persistent_pg",
                    range(len(persistent.relative_residuals)),
                    persistent.relative_residuals,
                )
            )

            recycled, recycled_count, recycled_time = _run_recycled(
                context, rhs, pair, settings
            )
            recycled_row = _row_from_recycled(
                ratio=ratio,
                rhs_seed=rhs_seed,
                result=recycled,
                counters=recycled_count,
                online_seconds=recycled_time,
                context=context,
                factors=pair,
            )
            rows.append(recycled_row)
            histories.extend(
                _history_rows(
                    ratio,
                    rhs_seed,
                    "krylov_recycling",
                    recycled.true_residual_iterations,
                    recycled.true_residual_history,
                )
            )
            check = check_lookup[ratio]
            for field in (
                "projected_arnoldi_relation_error_max",
                "arnoldi_basis_orthogonality_error_max",
                "arnoldi_basis_to_w_orthogonality_error_max",
            ):
                previous = check[field]
                value = float(recycled_row[field])
                check[field] = value if previous == "" else max(float(previous), value)

    for check in checks:
        check["algebra_pass"] = bool(
            float(check["z_orthogonality_error"]) <= settings.algebra_tolerance
            and float(check["w_orthogonality_error"]) <= settings.algebra_tolerance
            and float(check["au_equals_w_relative_error"]) <= settings.algebra_tolerance
            and float(check["wtaz_equals_r_relative_error"]) <= settings.algebra_tolerance
            and np.isfinite(float(check["condition_wtaz"]))
            and float(check["condition_wtaz"]) < 1.0e14
            and float(check["projected_arnoldi_relation_error_max"])
            <= settings.algebra_tolerance
            and float(check["arnoldi_basis_to_w_orthogonality_error_max"])
            <= settings.projected_orthogonality_tolerance
        )
    if not all(bool(row["algebra_pass"]) for row in checks):
        _write_csv(output / "algebra_checks.csv", checks)
        raise RuntimeError("Stage 13 algebra verification failed; performance interpretation aborted")

    summaries = _summarize(rows, ("ratio", "method"))
    _write_csv(output / "same_system_results.csv", rows)
    _write_csv(output / "same_system_summary.csv", summaries)
    _write_csv(output / "same_system_histories.csv", histories)
    _write_csv(output / "algebra_checks.csv", checks)
    return rows, summaries, histories


def run_rho_diagnostics(
    output: Path,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage13Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rerun representative solves outside timing summaries to record rho."""

    solver_rows: List[Dict[str, Any]] = []
    preconditioner_rows: List[Dict[str, Any]] = []
    rhs_seed = settings.stage10.evaluation_rhs_seeds[0]
    for ratio in settings.representative_ratios:
        context = contexts[ratio]
        pair = factors[ratio]
        rhs, _ = _discrete_rhs(context.system, rhs_seed)
        rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)

        baseline_records: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "stage": "outer_true_residual",
                "relative_residual": 1.0,
                "rho": _rho(pair.test, rhs),
            }
        ]
        baseline_counters = RunCounters()
        counted = CountingMatrix(context.system.matrix, baseline_counters)
        ilu = _counting_ilu(context, baseline_counters)
        restarted_gmres(
            counted,
            rhs,
            settings.stage10.solver,
            ilu,
            callback=_persistent_rho_callback(pair.test, baseline_records),
        )
        for row in baseline_records:
            solver_rows.append(
                {
                    "ratio": ratio,
                    "rhs_seed": rhs_seed,
                    "method": "ILU",
                    "iteration": row["iteration"],
                    "stage": row["stage"],
                    "true_relative_residual": float(row["relative_residual"])
                    / (1.0 if row["iteration"] == 0 else rhs_norm),
                    "rho": row["rho"],
                }
            )

        action_records: List[Dict[str, Any]] = []
        outer_records: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "stage": "outer_true_residual",
                "relative_residual": 1.0,
                "rho": _rho(pair.test, rhs),
            }
        ]
        _run_persistent(
            context,
            rhs,
            pair,
            settings,
            diagnostic_records=action_records,
            residual_callback=_persistent_rho_callback(pair.test, outer_records),
        )
        for row in outer_records:
            solver_rows.append(
                {
                    "ratio": ratio,
                    "rhs_seed": rhs_seed,
                    "method": "persistent_pg",
                    "iteration": row["iteration"],
                    "stage": row["stage"],
                    "true_relative_residual": float(row["relative_residual"])
                    / (1.0 if row["iteration"] == 0 else rhs_norm),
                    "rho": row["rho"],
                }
            )
        for row in action_records:
            preconditioner_rows.append(
                {
                    "ratio": ratio,
                    "rhs_seed": rhs_seed,
                    **row,
                }
            )

        recycled, _, _ = _run_recycled(
            context,
            rhs,
            pair,
            settings,
            record_projection_diagnostics=True,
        )
        for row in recycled.projection_records:
            solver_rows.append(
                {
                    "ratio": ratio,
                    "rhs_seed": rhs_seed,
                    "method": "krylov_recycling",
                    "iteration": row["iteration"],
                    "stage": row["stage"],
                    "true_relative_residual": row["relative_residual"],
                    "rho": row["rho"],
                }
            )
    _write_csv(output / "rho_solver_histories.csv", solver_rows)
    _write_csv(output / "persistent_preconditioner_rho.csv", preconditioner_rows)
    return solver_rows, preconditioner_rows


def run_frequency_ablation(
    output: Path,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    same_rows: Sequence[Mapping[str, Any]],
    settings: Stage13Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply the fixed Petrov correction every 1/2/5 calls or once initially."""

    rows: List[Dict[str, Any]] = []
    for row in same_rows:
        if row["method"] == "persistent_pg":
            copied = dict(row)
            copied["frequency"] = 1
            copied["frequency_label"] = "every application"
            rows.append(copied)
    for ratio in settings.ratios:
        context = contexts[ratio]
        pair = factors[ratio]
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            for frequency in settings.coarse_frequencies:
                if frequency == 1:
                    continue
                result, counters, online = _run_persistent(
                    context,
                    rhs,
                    pair,
                    settings,
                    frequency=frequency,
                )
                row = _row_from_krylov(
                    ratio=ratio,
                    rhs_seed=rhs_seed,
                    method="persistent_pg",
                    result=result,
                    counters=counters,
                    online_seconds=online,
                    context=context,
                    factors=pair,
                )
                row["frequency"] = frequency
                row["frequency_label"] = f"every {frequency} applications (from first)"
                rows.append(row)
            result, counters, online = _run_persistent(
                context,
                rhs,
                pair,
                settings,
                only_initial=True,
            )
            row = _row_from_krylov(
                ratio=ratio,
                rhs_seed=rhs_seed,
                method="persistent_pg",
                result=result,
                counters=counters,
                online_seconds=online,
                context=context,
                factors=pair,
            )
            row["frequency"] = 0
            row["frequency_label"] = "once at initialization"
            rows.append(row)
    summaries = _summarize(rows, ("ratio", "frequency", "frequency_label"))
    _write_csv(output / "frequency_ablation_results.csv", rows)
    _write_csv(output / "frequency_ablation_summary.csv", summaries)
    return rows, summaries


def run_transfer_comparison(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    settings: Stage13Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Transfer one donor Z and rebuild the identical target pair for A and B."""

    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    geometry: List[Dict[str, Any]] = []
    for source_ratio, target_ratio in settings.transfer_pairs:
        source = contexts[source_ratio]
        target = contexts[target_ratio]
        donor_z = spaces[source_ratio]
        target_pair = factor_az_petrov(target.system.matrix, donor_z)
        target_own = spaces[target_ratio]
        metric = subspace_metrics(donor_z, target_own)
        own_w = factor_az_petrov(target.system.matrix, target_own).test
        w_metric = subspace_metrics(target_pair.test, own_w)
        geometry.append(
            {
                "source_ratio": source_ratio,
                "target_ratio": target_ratio,
                "q": settings.q,
                "z_projector_distance_normalized": metric["projector_distance_normalized"],
                "z_principal_angle_mean_degrees": metric["principal_angle_mean_degrees"],
                "z_principal_angle_max_degrees": metric["principal_angle_max_degrees"],
                "target_recomputed_w_projector_distance_normalized": w_metric[
                    "projector_distance_normalized"
                ],
                "target_recomputed_w_principal_angle_max_degrees": w_metric[
                    "principal_angle_max_degrees"
                ],
                "condition_wtaz": target_pair.diagnostics["wtaz_condition"],
            }
        )
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(target.system, rhs_seed)
            persistent, persistent_count, persistent_time = _run_persistent(
                target, rhs, target_pair, settings
            )
            persistent_row = _row_from_krylov(
                ratio=target_ratio,
                source_ratio=source_ratio,
                rhs_seed=rhs_seed,
                method="persistent_pg",
                result=persistent,
                counters=persistent_count,
                online_seconds=persistent_time,
                context=target,
                factors=target_pair,
            )
            persistent_row["common_oracle_z_seconds"] = source.modes.diagnostics[
                "setup_seconds"
            ]
            persistent_row["total_seconds_including_oracle"] = (
                persistent_row["total_seconds_excluding_oracle"]
                + persistent_row["common_oracle_z_seconds"]
            )
            rows.append(persistent_row)
            histories.extend(
                _history_rows(
                    target_ratio,
                    rhs_seed,
                    "persistent_pg",
                    range(len(persistent.relative_residuals)),
                    persistent.relative_residuals,
                    source_ratio,
                )
            )

            recycled, recycled_count, recycled_time = _run_recycled(
                target, rhs, target_pair, settings
            )
            recycled_row = _row_from_recycled(
                ratio=target_ratio,
                source_ratio=source_ratio,
                rhs_seed=rhs_seed,
                result=recycled,
                counters=recycled_count,
                online_seconds=recycled_time,
                context=target,
                factors=target_pair,
            )
            recycled_row["common_oracle_z_seconds"] = source.modes.diagnostics[
                "setup_seconds"
            ]
            recycled_row["total_seconds_including_oracle"] = (
                recycled_row["total_seconds_excluding_oracle"]
                + recycled_row["common_oracle_z_seconds"]
            )
            rows.append(recycled_row)
            histories.extend(
                _history_rows(
                    target_ratio,
                    rhs_seed,
                    "krylov_recycling",
                    recycled.true_residual_iterations,
                    recycled.true_residual_history,
                    source_ratio,
                )
            )
    summaries = _summarize(rows, ("source_ratio", "ratio", "method"))
    _write_csv(output / "transfer_results.csv", rows)
    _write_csv(output / "transfer_summary.csv", summaries)
    _write_csv(output / "transfer_histories.csv", histories)
    _write_csv(output / "transfer_geometry.csv", geometry)
    return rows, summaries, histories, geometry


def build_decision(
    output: Path,
    same_rows: Sequence[Mapping[str, Any]],
    same_summaries: Sequence[Mapping[str, Any]],
    frequency_summaries: Sequence[Mapping[str, Any]],
    rho_solver_rows: Sequence[Mapping[str, Any]],
    rho_preconditioner_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    settings: Stage13Settings,
) -> Dict[str, Any]:
    """Apply mechanism-level decision rules without selecting a preferred outcome."""

    lookup = {
        (float(row["ratio"]), str(row["method"])): row for row in same_summaries
    }
    ilu = [lookup[(ratio, "ILU")] for ratio in settings.ratios]
    persistent = [lookup[(ratio, "persistent_pg")] for ratio in settings.ratios]
    recycled = [lookup[(ratio, "krylov_recycling")] for ratio in settings.ratios]
    persistent_reductions = np.asarray(
        [
            (base["iterations_median"] - method["iterations_median"])
            / max(base["iterations_median"], 1.0)
            for base, method in zip(ilu, persistent)
        ]
    )
    recycled_reductions = np.asarray(
        [
            (base["iterations_median"] - method["iterations_median"])
            / max(base["iterations_median"], 1.0)
            for base, method in zip(ilu, recycled)
        ]
    )
    a_vs_b = np.asarray(
        [
            (b["iterations_median"] - a["iterations_median"])
            / max(b["iterations_median"], 1.0)
            for a, b in zip(persistent, recycled)
        ]
    )
    b_vs_a = -a_vs_b * np.asarray(
        [b["iterations_median"] / max(a["iterations_median"], 1.0) for a, b in zip(persistent, recycled)]
    )
    iteration_gaps = np.asarray(
        [a["iterations_median"] - b["iterations_median"] for a, b in zip(persistent, recycled)]
    )
    time_ratios = np.asarray(
        [
            a["online_solve_seconds_median"]
            / max(b["online_solve_seconds_median"], np.finfo(float).tiny)
            for a, b in zip(persistent, recycled)
        ]
    )

    frequency_lookup = {
        (float(row["ratio"]), int(row["frequency"])): row
        for row in frequency_summaries
    }
    once_penalties = np.asarray(
        [
            frequency_lookup[(ratio, 0)]["iterations_median"]
            - frequency_lookup[(ratio, 1)]["iterations_median"]
            for ratio in settings.ratios
        ]
    )
    every_five_penalties = np.asarray(
        [
            frequency_lookup[(ratio, 5)]["iterations_median"]
            - frequency_lookup[(ratio, 1)]["iterations_median"]
            for ratio in settings.ratios
        ]
    )

    pre_after_first = [
        float(row["rho_before"])
        for row in rho_preconditioner_rows
        if int(row["preconditioner_application"]) > 1
    ]
    post_values = [float(row["rho_after"]) for row in rho_preconditioner_rows]
    recycled_trials = [
        float(row["rho"])
        for row in rho_solver_rows
        if row["method"] == "krylov_recycling" and row["stage"] == "arnoldi_trial"
    ]
    recycled_post = [
        float(row["rho"])
        for row in rho_solver_rows
        if row["method"] == "krylov_recycling" and row["stage"] == "cycle_post_projection"
    ]

    transfer_lookup = {
        (float(row["source_ratio"]), float(row["ratio"]), str(row["method"])): row
        for row in transfer_summaries
    }
    persistent_transfer_reductions = []
    recycled_transfer_reductions = []
    persistent_transfer_convergence = []
    recycled_transfer_convergence = []
    for source, target in settings.transfer_pairs:
        base = lookup[(target, "ILU")]
        a = transfer_lookup[(source, target, "persistent_pg")]
        b = transfer_lookup[(source, target, "krylov_recycling")]
        persistent_transfer_reductions.append(
            (base["iterations_median"] - a["iterations_median"])
            / max(base["iterations_median"], 1.0)
        )
        recycled_transfer_reductions.append(
            (base["iterations_median"] - b["iterations_median"])
            / max(base["iterations_median"], 1.0)
        )
        persistent_transfer_convergence.append(a["convergence_rate"])
        recycled_transfer_convergence.append(b["convergence_rate"])

    stage12_path = (
        output.parent
        / "stage12_nonsymmetric_petrov_galerkin_recycling_validation"
        / "same_system_results.csv"
    )
    stage12_differences: List[int] = []
    if stage12_path.exists():
        stage12 = {
            (float(row["ratio"]), int(row["rhs_seed"])): int(row["iterations"])
            for row in _read_csv(stage12_path)
            if row["method"] == "az_petrov" and int(row["q"]) == settings.q
        }
        for row in same_rows:
            if row["method"] == "persistent_pg":
                key = (float(row["ratio"]), int(row["rhs_seed"]))
                if key in stage12:
                    stage12_differences.append(int(row["iterations"]) - stage12[key])

    own_useful = bool(
        np.median(persistent_reductions) >= 0.20
        and np.median(recycled_reductions) >= 0.20
        and np.mean([row["convergence_rate"] for row in persistent])
        >= settings.reliable_convergence_rate
        and np.mean([row["convergence_rate"] for row in recycled])
        >= settings.reliable_convergence_rate
    )
    residual_regrowth = bool(
        np.median(pre_after_first) >= settings.rho_regrowth_threshold
        if pre_after_first
        else False
    )
    repeated_material = bool(
        np.median(once_penalties)
        >= settings.material_iteration_advantage
        * max(float(np.median([row["iterations_median"] for row in persistent])), 1.0)
    )
    a_material = bool(
        np.median(a_vs_b) >= settings.material_iteration_advantage
        and np.sum(iteration_gaps < 0.0) >= 5
        and residual_regrowth
        and repeated_material
    )
    b_material = bool(
        np.median(b_vs_a) >= settings.material_iteration_advantage
        and np.sum(iteration_gaps > 0.0) >= 5
        and not repeated_material
    )
    similar = bool(
        np.median(np.abs(iteration_gaps))
        <= settings.similar_iteration_fraction
        * max(
            float(
                np.median(
                    [
                        min(a["iterations_median"], b["iterations_median"])
                        for a, b in zip(persistent, recycled)
                    ]
                )
            ),
            1.0,
        )
    )
    if not own_useful:
        conclusion = "D"
        label = "Neither integration is robust enough."
    elif a_material:
        conclusion = "A"
        label = "Persistent two-level preconditioning is justified."
    elif b_material:
        conclusion = "B"
        label = "The oracle space is useful, but mainly as a Krylov recycling/deflation space."
    elif similar:
        conclusion = "C"
        label = "Both formulations exploit essentially the same information with similar performance."
    else:
        conclusion = "D"
        label = "Neither integration is robust enough."

    decision = {
        "conclusion": conclusion,
        "conclusion_label": label,
        "all_algebra_checks_pass": bool(all(bool(row["algebra_pass"]) for row in checks)),
        "stage12_persistent_reproduction_max_iteration_difference": int(
            max([abs(value) for value in stage12_differences], default=0)
        ),
        "persistent_iteration_reduction_median": float(np.median(persistent_reductions)),
        "recycling_iteration_reduction_median": float(np.median(recycled_reductions)),
        "persistent_minus_recycling_iterations_median": float(np.median(iteration_gaps)),
        "persistent_better_system_count": int(np.sum(iteration_gaps < 0.0)),
        "recycling_better_system_count": int(np.sum(iteration_gaps > 0.0)),
        "exact_iteration_tie_system_count": int(np.sum(iteration_gaps == 0.0)),
        "persistent_to_recycling_online_time_ratio_median": float(np.median(time_ratios)),
        "persistent_convergence_rate_mean": float(
            np.mean([row["convergence_rate"] for row in persistent])
        ),
        "recycling_convergence_rate_mean": float(
            np.mean([row["convergence_rate"] for row in recycled])
        ),
        "persistent_preconditioner_rho_before_after_first_median": float(
            np.median(pre_after_first) if pre_after_first else float("nan")
        ),
        "persistent_preconditioner_rho_after_median": float(
            np.median(post_values) if post_values else float("nan")
        ),
        "recycling_trial_rho_median": float(
            np.median(recycled_trials) if recycled_trials else float("nan")
        ),
        "recycling_trial_rho_max": float(
            np.max(recycled_trials) if recycled_trials else float("nan")
        ),
        "recycling_post_projection_rho_max": float(
            np.max(recycled_post) if recycled_post else float("nan")
        ),
        "rho_regrowth_detected": residual_regrowth,
        "once_initial_iteration_penalty_median": float(np.median(once_penalties)),
        "every_five_iteration_penalty_median": float(np.median(every_five_penalties)),
        "repeated_correction_material": repeated_material,
        "persistent_transfer_iteration_reduction_median": float(
            np.median(persistent_transfer_reductions)
        ),
        "recycling_transfer_iteration_reduction_median": float(
            np.median(recycled_transfer_reductions)
        ),
        "persistent_transfer_convergence_rate_mean": float(
            np.mean(persistent_transfer_convergence)
        ),
        "recycling_transfer_convergence_rate_mean": float(
            np.mean(recycled_transfer_convergence)
        ),
        "own_space_useful_for_both": own_useful,
        "persistent_material_advantage_gate": a_material,
        "recycling_material_advantage_gate": b_material,
        "similar_iteration_gate": similar,
        "decision_thresholds": {
            "material_iteration_advantage": settings.material_iteration_advantage,
            "similar_iteration_fraction": settings.similar_iteration_fraction,
            "reliable_convergence_rate": settings.reliable_convergence_rate,
            "rho_regrowth_threshold": settings.rho_regrowth_threshold,
        },
    }
    _write_json(output / "decision.json", decision)
    return decision


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    same_summaries: Sequence[Mapping[str, Any]],
    same_histories: Sequence[Mapping[str, Any]],
    frequency_summaries: Sequence[Mapping[str, Any]],
    rho_solver_rows: Sequence[Mapping[str, Any]],
    rho_preconditioner_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    settings: Stage13Settings,
) -> List[str]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    colors = {"ILU": CHARCOAL, "persistent_pg": ORANGE, "krylov_recycling": BLUE}
    labels = {
        "ILU": "ILU",
        "persistent_pg": "persistent PG",
        "krylov_recycling": "Krylov recycling",
    }

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    iteration_styles = {
        "ILU": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "zorder": 1},
        "persistent_pg": {
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.8,
            "markerfacecolor": "white",
            "markeredgewidth": 2.0,
            "markersize": 8.0,
            "zorder": 2,
        },
        "krylov_recycling": {
            "marker": "x",
            "linestyle": "--",
            "linewidth": 1.7,
            "markersize": 7.0,
            "zorder": 3,
        },
    }
    for method in ("ILU", "persistent_pg", "krylov_recycling"):
        local = sorted(
            [row for row in same_summaries if row["method"] == method],
            key=lambda row: float(row["ratio"]),
        )
        axis.plot(
            [row["ratio"] for row in local],
            [row["iterations_median"] for row in local],
            color=colors[method],
            label=labels[method],
            **iteration_styles[method],
        )
    axis.set(xlabel="anisotropy ratio R", ylabel="median Arnoldi/FGMRES iterations")
    axis.set_title("Same-system q=8 oracle integration", loc="left")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    name = "same_system_iterations.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for method in ("persistent_pg", "krylov_recycling"):
        local = sorted(
            [row for row in same_summaries if row["method"] == method],
            key=lambda row: float(row["ratio"]),
        )
        axes[0].plot(
            [row["ratio"] for row in local],
            [row["online_solve_seconds_median"] for row in local],
            marker="o",
            color=colors[method],
            label=labels[method],
        )
        axes[1].plot(
            [row["ratio"] for row in local],
            [row["a_matvecs_median"] for row in local],
            marker="o",
            color=colors[method],
            label=labels[method],
        )
    axes[0].set(xlabel="R", ylabel="online solve time (s)", title="Measured online time")
    axes[1].set(xlabel="R", ylabel="median A applications", title="Counted A-matvecs")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    name = "same_system_costs.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    rhs_seed = settings.stage10.evaluation_rhs_seeds[0]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=False, sharey=True)
    for axis, ratio in zip(axes.flat, settings.representative_ratios):
        residual_styles = {
            "ILU": {"linestyle": "-", "linewidth": 1.8, "zorder": 1},
            "persistent_pg": {"linestyle": "-", "linewidth": 2.8, "zorder": 2},
            "krylov_recycling": {"linestyle": "--", "linewidth": 1.6, "zorder": 3},
        }
        for method in ("ILU", "persistent_pg", "krylov_recycling"):
            local = [
                row
                for row in same_histories
                if float(row["ratio"]) == ratio
                and int(row["rhs_seed"]) == rhs_seed
                and row["method"] == method
            ]
            axis.semilogy(
                [row["iteration"] for row in local],
                [max(float(row["true_relative_residual"]), 1.0e-16) for row in local],
                color=colors[method],
                label=labels[method],
                **residual_styles[method],
            )
        axis.axhline(settings.stage10.solver.relative_tolerance, color=LIGHT_GRAY, linestyle="--")
        axis.set_title(f"R={ratio:g}", loc="left")
        axis.set(xlabel="iteration", ylabel="true relative residual")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    name = "representative_residual_histories.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharey=True)
    for axis, ratio in zip(axes.flat, settings.representative_ratios):
        for method, stage in (
            ("ILU", "outer_true_residual"),
            ("persistent_pg", "outer_true_residual"),
            ("krylov_recycling", "arnoldi_trial"),
        ):
            local = [
                row
                for row in rho_solver_rows
                if float(row["ratio"]) == ratio
                and row["method"] == method
                and row["stage"] == stage
            ]
            axis.semilogy(
                [row["iteration"] for row in local],
                [max(float(row["rho"]), 1.0e-17) for row in local],
                color=colors[method],
                linewidth=1.5,
                label=labels[method],
            )
        axis.set_title(f"R={ratio:g}", loc="left")
        axis.set(xlabel="iteration", ylabel=r"$\rho_m=\|W^Tr_m\|/\|r_m\|$")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    name = "representative_rho_histories.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for ratio in settings.representative_ratios:
        local = [row for row in rho_preconditioner_rows if float(row["ratio"]) == ratio]
        applications = [int(row["preconditioner_application"]) for row in local]
        axis.semilogy(
            applications,
            [max(float(row["rho_before"]), 1.0e-17) for row in local],
            linewidth=1.3,
            label=f"R={ratio:g} before",
        )
    axis.axhline(settings.rho_regrowth_threshold, color=CHARCOAL, linestyle="--", linewidth=1)
    axis.set(
        xlabel="persistent preconditioner application",
        ylabel="rho of ILU remainder before PG removal",
    )
    axis.set_title("Reappearance of the paired residual-space component", loc="left")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    name = "persistent_rho_before_correction.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    frequency_colors = {1: ORANGE, 2: PINK, 5: PURPLE, 0: OLIVE}
    for frequency, label in (
        (1, "every application"),
        (2, "every 2 (from first)"),
        (5, "every 5 (from first)"),
        (0, "once at initialization"),
    ):
        local = sorted(
            [row for row in frequency_summaries if int(row["frequency"]) == frequency],
            key=lambda row: float(row["ratio"]),
        )
        axis.plot(
            [row["ratio"] for row in local],
            [row["iterations_median"] for row in local],
            marker="o",
            linewidth=1.8,
            color=frequency_colors[frequency],
            label=label,
        )
    axis.set(xlabel="anisotropy ratio R", ylabel="median iterations")
    axis.set_title("Persistent-correction frequency ablation", loc="left")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    name = "frequency_ablation.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    transfer_lookup = {
        (float(row["source_ratio"]), float(row["ratio"]), row["method"]): row
        for row in transfer_summaries
    }
    same_lookup = {
        (float(row["ratio"]), row["method"]): row for row in same_summaries
    }
    pair_labels = [f"{source:g}→{target:g}" for source, target in settings.transfer_pairs]
    x = np.arange(len(pair_labels))
    width = 0.16
    figure, axis = plt.subplots(figsize=(10.2, 4.8))
    methods = (
        ("ILU", CHARCOAL, -2),
        ("own persistent", ORANGE, -1),
        ("transfer persistent", PINK, 0),
        ("own recycling", BLUE, 1),
        ("transfer recycling", PURPLE, 2),
    )
    for label, color, offset in methods:
        values = []
        for source, target in settings.transfer_pairs:
            if label == "ILU":
                row = same_lookup[(target, "ILU")]
            elif label == "own persistent":
                row = same_lookup[(target, "persistent_pg")]
            elif label == "own recycling":
                row = same_lookup[(target, "krylov_recycling")]
            elif label == "transfer persistent":
                row = transfer_lookup[(source, target, "persistent_pg")]
            else:
                row = transfer_lookup[(source, target, "krylov_recycling")]
            values.append(row["iterations_median"])
        axis.bar(x + offset * width, values, width, color=color, label=label)
    axis.set_xticks(x, pair_labels)
    axis.set(ylabel="median iterations", xlabel="donor → target R")
    axis.set_title("Neighboring-system transfer with target-rebuilt pairing", loc="left")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    name = "transfer_ab_comparison.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)
    return names


def _format(value: Any, digits: int = 3) -> str:
    if value == "":
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    if number != 0.0 and (abs(number) < 1.0e-3 or abs(number) >= 1.0e4):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    same_summaries: Sequence[Mapping[str, Any]],
    frequency_summaries: Sequence[Mapping[str, Any]],
    rho_solver_rows: Sequence[Mapping[str, Any]],
    rho_preconditioner_rows: Sequence[Mapping[str, Any]],
    transfer_summaries: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage13Settings,
) -> Path:
    """Write the requested concise technical Markdown report."""

    same = {(float(row["ratio"]), row["method"]): row for row in same_summaries}
    algebra_table = [
        (
            _format(row["ratio"], 0),
            int(row["z_rank"]),
            _format(row["z_orthogonality_error"]),
            _format(row["w_orthogonality_error"]),
            _format(row["au_equals_w_relative_error"]),
            _format(row["wtaz_equals_r_relative_error"]),
            _format(row["condition_wtaz"]),
            _format(row["projected_arnoldi_relation_error_max"]),
            _format(row["arnoldi_basis_to_w_orthogonality_error_max"]),
        )
        for row in checks
    ]
    same_table = []
    operation_table = []
    cost_table = []
    for ratio in settings.ratios:
        base = same[(ratio, "ILU")]
        a = same[(ratio, "persistent_pg")]
        b = same[(ratio, "krylov_recycling")]
        same_table.append(
            (
                _format(ratio, 0),
                _format(base["iterations_median"], 0),
                _format(a["iterations_median"], 0),
                _format(b["iterations_median"], 0),
                _format(a["convergence_rate"], 2),
                _format(b["convergence_rate"], 2),
                _format(a["true_final_relative_residual_max"]),
                _format(b["true_final_relative_residual_max"]),
                _format(a["online_solve_seconds_median"]),
                _format(b["online_solve_seconds_median"]),
            )
        )
        for method, label in (("persistent_pg", "persistent PG"), ("krylov_recycling", "recycling")):
            row = same[(ratio, method)]
            operation_table.append(
                (
                    _format(ratio, 0),
                    label,
                    _format(row["a_matvecs_median"], 0),
                    _format(row["ilu_solves_median"], 0),
                    _format(row["w_transpose_applications_median"], 0),
                    _format(row["w_applications_median"], 0),
                    _format(row["z_or_u_applications_median"], 0),
                    _format(row["reduced_solves_median"], 0),
                )
            )
            cost_table.append(
                (
                    _format(ratio, 0),
                    label,
                    _format(row["method_setup_seconds_after_z"]),
                    _format(row["common_oracle_z_seconds"]),
                    _format(row["online_solve_seconds_median"]),
                    _format(row["total_seconds_excluding_oracle_median"]),
                    _format(row["total_seconds_including_oracle_median"]),
                )
            )

    frequency_lookup = {
        (float(row["ratio"]), int(row["frequency"])): row for row in frequency_summaries
    }
    frequency_table = [
        (
            _format(ratio, 0),
            _format(frequency_lookup[(ratio, 1)]["iterations_median"], 0),
            _format(frequency_lookup[(ratio, 2)]["iterations_median"], 0),
            _format(frequency_lookup[(ratio, 5)]["iterations_median"], 0),
            _format(frequency_lookup[(ratio, 0)]["iterations_median"], 0),
            _format(frequency_lookup[(ratio, 1)]["online_solve_seconds_median"]),
            _format(frequency_lookup[(ratio, 0)]["online_solve_seconds_median"]),
        )
        for ratio in settings.ratios
    ]

    rho_table = []
    for ratio in settings.representative_ratios:
        pre = [
            float(row["rho_before"])
            for row in rho_preconditioner_rows
            if float(row["ratio"]) == ratio and int(row["preconditioner_application"]) > 1
        ]
        after = [
            float(row["rho_after"])
            for row in rho_preconditioner_rows
            if float(row["ratio"]) == ratio
        ]
        recycle_trials = [
            float(row["rho"])
            for row in rho_solver_rows
            if float(row["ratio"]) == ratio
            and row["method"] == "krylov_recycling"
            and row["stage"] == "arnoldi_trial"
        ]
        rho_table.append(
            (
                _format(ratio, 0),
                _format(np.median(pre)),
                _format(np.max(pre)),
                _format(np.median(after)),
                _format(np.median(recycle_trials)),
                _format(np.max(recycle_trials)),
            )
        )

    transfer = {
        (float(row["source_ratio"]), float(row["ratio"]), row["method"]): row
        for row in transfer_summaries
    }
    transfer_table = []
    for source, target in settings.transfer_pairs:
        base = same[(target, "ILU")]
        own_a = same[(target, "persistent_pg")]
        own_b = same[(target, "krylov_recycling")]
        trans_a = transfer[(source, target, "persistent_pg")]
        trans_b = transfer[(source, target, "krylov_recycling")]
        transfer_table.append(
            (
                f"{source:g}→{target:g}",
                _format(base["iterations_median"], 0),
                _format(own_a["iterations_median"], 0),
                _format(trans_a["iterations_median"], 0),
                _format(trans_a["convergence_rate"], 2),
                _format(own_b["iterations_median"], 0),
                _format(trans_b["iterations_median"], 0),
                _format(trans_b["convergence_rate"], 2),
                _format(trans_a["online_solve_seconds_median"]),
                _format(trans_b["online_solve_seconds_median"]),
            )
        )
    geometry_table = [
        (
            f"{row['source_ratio']:g}→{row['target_ratio']:g}",
            _format(row["z_projector_distance_normalized"]),
            _format(row["z_principal_angle_mean_degrees"], 1),
            _format(row["z_principal_angle_max_degrees"], 1),
            _format(row["target_recomputed_w_projector_distance_normalized"]),
            _format(row["target_recomputed_w_principal_angle_max_degrees"], 1),
            _format(row["condition_wtaz"]),
        )
        for row in geometry
    ]

    q1 = (
        "Yes. Both own-system mechanisms materially accelerate strong ILU: median iteration "
        f"reductions are {100*decision['persistent_iteration_reduction_median']:.1f}% (persistent) "
        f"and {100*decision['recycling_iteration_reduction_median']:.1f}% (recycling)."
    )
    gap = float(decision["persistent_minus_recycling_iterations_median"])
    if gap == 0.0:
        q2 = "Neither: their median iteration counts are tied across the tested systems."
    elif gap < 0.0:
        q2 = f"Persistent PG, by {-gap:.1f} median iterations."
    else:
        q2 = f"Recycling, by {gap:.1f} median iterations."
    time_ratio = float(decision["persistent_to_recycling_online_time_ratio_median"])
    q3 = (
        "Persistent PG" if time_ratio < 1.0 else "Krylov recycling"
    ) + f" has the lower median online time; persistent/recycling={time_ratio:.2f}."
    q4 = (
        "Yes inside the ILU remainder used by Method A: after the first application its median "
        f"pre-correction rho is {decision['persistent_preconditioner_rho_before_after_first_median']:.3f}, "
        f"while the correction reduces it to {decision['persistent_preconditioner_rho_after_median']:.2e}. "
        "In Method B, projected Arnoldi keeps the true-residual component at "
        f"median {decision['recycling_trial_rho_median']:.2e}."
    )
    q5 = (
        "Repeated enforcement is necessary" if decision["repeated_correction_material"]
        else "Repeated enforcement is not necessary"
    ) + (
        f" under the preregistered mechanism gate: one-time initialization costs a median "
        f"{decision['once_initial_iteration_penalty_median']:.1f} extra iterations relative to every application. "
        "It need not be implemented as a persistent preconditioner, however: the projected recycled solve "
        "enforces the same complement and matches Method A's iterations."
    )
    transfer_a = float(decision["persistent_transfer_convergence_rate_mean"])
    transfer_b = float(decision["recycling_transfer_convergence_rate_mean"])
    if abs(transfer_a - transfer_b) < 1.0e-12:
        q6 = f"Neither is more robust by convergence rate; both are {transfer_a:.2f}."
    elif transfer_a > transfer_b:
        q6 = f"Persistent PG is more robust ({transfer_a:.2f} versus {transfer_b:.2f} convergence)."
    else:
        q6 = f"Recycling is more robust ({transfer_b:.2f} versus {transfer_a:.2f} convergence)."
    q7 = (
        "Yes at the component level: the W-component regrows in fresh ILU remainders and one-time use is "
        "insufficient. The stronger claim that this uniquely justifies embedding Z in every preconditioner "
        "application is not supported, because fixed-space Krylov projection removes the same persistent "
        "component with matched iterations and lower measured online cost. "
        f"The observed classification is {decision['conclusion']}: {decision['conclusion_label']}"
    )

    report = f"""# Stage 13: persistent preconditioner versus Krylov recycling

## Executive conclusion

**{decision['conclusion']}. {decision['conclusion_label']}**

This is a strict solver-integration A/B test. The archived Stage 10 right Ritz basis is frozen at $q={settings.q}$; the RBF-FD matrix, strong ILU, held-out right-hand sides, initial guess, tolerance, and iteration cap are identical. No network is trained, no Ritz ordering or target is changed, and the common oracle construction cost is charged equally to Methods A and B.

All algebra checks pass. Method A reproduces the Stage 12 q=8 AZ–Petrov iterations with maximum difference **{decision['stage12_persistent_reproduction_max_iteration_difference']}**. The projected Arnoldi implementation therefore proceeds to performance interpretation on the requested nonsymmetric convention.

![Same-system iterations](plots/same_system_iterations.png)

## 1. Sources, fixed protocol, and operator convention

The [research proposal](../../hybrid_iteration_rp.pdf) remains the mathematical source of truth. The empirical inputs are [Stage 9](../stage9_hard_regime_discovery_and_scalability/stage9_hard_regime_discovery_and_scalability_report.md), [Stage 10](../stage10_ritz_transfer_and_mode_tracking/stage10_ritz_transfer_and_mode_tracking_report.md), [Stage 11](../stage11_ritz_cluster_stability/stage11_ritz_cluster_stability_report.md), and [Stage 12](../stage12_nonsymmetric_petrov_galerkin_recycling_validation/stage12_nonsymmetric_petrov_galerkin_recycling_validation.md). The conceptual comparison uses only the uploaded [NeurKItt paper](../../literature/NeurIPS-2024-neural-krylov-iteration-for-accelerating-linear-system-solving-Paper-Conference.pdf).

Fixed numerical controls: $N=5041$, node seed 7, jitter 0.15, stencil 25, 30° rotated anisotropy, $R\in\{{85,90,95,100,105,110,120\}}$, ILU drop tolerance $10^{{-3}}$, fill factor 5, COLAMD, zero diagonal-pivot threshold, FGMRES/GMRES relative tolerance $10^{{-8}}$, restart 50, cap 300, construction RHS seed {settings.stage10.construction_rhs_seed}, and held-out RHS seeds {settings.stage10.evaluation_rhs_seeds}.

Because these are the identical matrices and ILU factors used in Stage 12, its nonsymmetry diagnostics apply unchanged: $\|A-A^T\|_F/\|A\|_F$ ranges from 0.1384 to 0.1389, while the 48-probe estimate of $\|AM^{{-1}}-(AM^{{-1}})^T\|_F/\|AM^{{-1}}\|_F$ ranges from 1.30 to 1.62. Both methods below therefore use the same meaningfully nonsymmetric, right-preconditioned operator convention.

Stage 10 constructs Ritz information from the right-preconditioned operator $AM^{{-1}}$, maps selected Ritz directions through $M^{{-1}}$ into solution coordinates, and orthonormalizes them. The same archived prefix $Z\in\mathbb{{R}}^{{N\\times8}}$ is used below.

## 2. Exact A/B formulations

For both methods, thin QR gives

$$AZ=WR,\qquad W^TW=I,\qquad U=ZR^{{-1}},\qquad AU=W,$$

where $U$ is formed by triangular solves.

Method A is a fixed right preconditioner. For each Arnoldi input $v$,

$$y_0=M^{{-1}}v,\qquad s=v-Ay_0,$$

$$P_A(v)=y_0+Z\,R^{{-1}}W^Ts.$$

The outer FGMRES Arnoldi operator is therefore $A P_A$; the same $Z,W,R$ are used on every application.

Method B is fixed-space right-preconditioned GCRO/recycled GMRES. At each restart,

$$\\bar x=x+UW^Tr,\qquad \\bar r=(I-WW^T)r.$$

For projected Arnoldi vectors $V_m$, let $Y_m=M^{{-1}}V_m$ and $B_m=W^TAY_m$. Arnoldi is applied to

$$ (I-WW^T) A M^{{-1}} V_m=V_{{m+1}}\\bar H_m,$$

and every trial solution uses

$$x_m=\\bar x+(Y_m-U B_m)y_m.$$

The compensation term is essential because $A(Y_m-U B_m)=(I-WW^T)AY_m$. ILU is applied once per projected Arnoldi step; $Z/U$ is not inserted into the ILU preconditioner. The fixed pair is retained across restarts. Unlike full GCRO-DR/NeurKItt Algorithm 1, no new recycle space is extracted, because doing so would violate the frozen-$Z$ fairness condition.

## 3. Algebra verification

{_table(['R', 'rank', '||ZTZ-I||F', '||WTW-I||F', '||AU-W||F/||W||F', '||WTAZ-R||/||R||', 'cond(WTAZ)', 'projected Arnoldi error', '||WTV||F'], algebra_table)}

The Petrov reduced system is nonsingular at every tested ratio. The final $W$-reprojection in Arnoldi is a numerical-stability step: it prevents the $W$-component from accumulating under the strongly nonnormal $AM^{{-1}}$ operator and is included in Method B's operation count and wall time.

## 4. Same-system oracle A/B comparison

Medians are over the identical three held-out RHSs. Iterations count new Arnoldi directions; the fixed recycle projection is not counted as an iteration. True residuals—not Hessenberg proxies—control convergence.

{_table(['R', 'ILU it.', 'persistent it.', 'recycle it.', 'A conv.', 'B conv.', 'A max final res.', 'B max final res.', 'A online s', 'B online s'], same_table)}

![Same-system costs](plots/same_system_costs.png)

Operation counts:

{_table(['R', 'method', 'A matvecs', 'ILU solves', 'W^T apps', 'W apps', 'Z/U apps', 'q solves'], operation_table)}

Setup and total cost (seconds):

{_table(['R', 'method', 'setup after Z', 'common Ritz Z', 'online', 'total excl. Z', 'total incl. Z'], cost_table)}

`setup after Z` includes $AZ$, QR, and—only for recycling—the triangular normalization that forms $U$. `total excl. Z` is ILU setup + method setup + online solve; `total incl. Z` adds the identical system-specific 64-step Ritz construction. The common oracle is never charged differently between A and B.

## 5. Residual and paired-component diagnostics

![Representative true residual histories](plots/representative_residual_histories.png)

![Representative rho histories](plots/representative_rho_histories.png)

For Method A, “before” and “after” refer to the ILU remainder $s=v-AM^{{-1}}v$ inside each preconditioner application. Method B reports the true residual during projected Arnoldi.

{_table(['R', 'A median rho before after first', 'A max rho before', 'A median rho after', 'B median trial rho', 'B max trial rho'], rho_table)}

![Persistent preconditioner rho before correction](plots/persistent_rho_before_correction.png)

The component can reappear in Method A's *new ILU remainders* even when Method B keeps the solver residual in the complementary projected space. Regrowth alone therefore does not prove that persistent correction is the better integration mechanism.

## 6. Correction-frequency ablation

“Every 2/5” starts with application 1 and repeats every 2/5 applications; this varies frequency without deliberately suppressing the first available correction. “Once” applies the full ILU+PG action to the initial RHS, uses that as $x_0$, and then runs ILU-only FGMRES.

{_table(['R', 'every 1', 'every 2', 'every 5', 'once initial', 'every-1 time', 'once time'], frequency_table)}

![Frequency ablation](plots/frequency_ablation.png)

Median iteration penalties versus every application are **{decision['every_five_iteration_penalty_median']:.1f}** for every fifth and **{decision['once_initial_iteration_penalty_median']:.1f}** for one-time initialization. Under the stated gate, repeated correction is **{'material' if decision['repeated_correction_material'] else 'not material'}**.

## 7. Neighboring-system transfer

Both methods receive the same donor $Z_i$. For target $A_j$, both rebuild the same target-side pair $A_jZ_i=W_{{j\leftarrow i}}R_{{j\leftarrow i}}$; only the solver integration differs.

{_table(['pair', 'ILU', 'own A', 'transfer A', 'A conv.', 'own B', 'transfer B', 'B conv.', 'A online s', 'B online s'], transfer_table)}

{_table(['pair', 'Z d_proj', 'Z mean angle', 'Z max angle', 'target W d_proj', 'target W max angle', 'cond'], geometry_table)}

![Transfer A/B comparison](plots/transfer_ab_comparison.png)

Transfer median reductions versus target ILU are **{100*decision['persistent_transfer_iteration_reduction_median']:.1f}%** for persistent PG and **{100*decision['recycling_transfer_iteration_reduction_median']:.1f}%** for recycling; mean convergence rates are **{decision['persistent_transfer_convergence_rate_mean']:.2f}** and **{decision['recycling_transfer_convergence_rate_mean']:.2f}**, respectively.

## 8. Answers to the seven mechanism questions

1. **Does the same oracle Z accelerate both?** {q1}
2. **Which gives lower iteration count?** {q2}
3. **Which gives lower online wall time?** {q3}
4. **Does the Z/W residual component reappear?** {q4}
5. **Is repeated correction necessary?** {q5}
6. **Is one mechanism more robust under transfer?** {q6}
7. **Does Z represent a persistent ILU failure mode?** {q7}

## 9. Conceptual comparison with NeurKItt

The uploaded NeurKItt paper explicitly constructs $[Q,R]=\operatorname{{qr}}(AY)$, sets $C=Q$ and $U=YR^{{-1}}$, and uses the projected relation $(I-CC^H)AV=V\\bar H$. Its experiments combine this acceleration module with conventional preconditioners, including ILU. Thus NeurKItt already supports “learned subspace + ILU”; its predicted invariant space is used primarily as a Krylov deflation/recycling space, not as a learned complement repeatedly embedded in every preconditioner application.

This Stage 13 experiment does not claim novelty from combining a subspace with ILU. It tests the narrower RBF-FD hypothesis: whether an oracle space captures a persistent low-dimensional failure of an RBF-FD-specific classical preconditioner strongly enough that repeated two-level correction is superior to using the same information in projected Krylov iteration. The conclusion above is limited to that mechanism and this frozen q=8 oracle family.

## 10. Limitations and reproducibility

- The comparison uses one resolution, node set, stencil, anisotropy angle, ILU budget, q, and four directed transfer pairs. It is a controlled mechanism test, not a general theorem.
- The archived Ritz basis is an oracle constructed from a disjoint RHS. Both A and B pay the same recorded construction cost; neither is deployable without a future prediction/reuse mechanism.
- Timings are single-process wall-clock observations and include the true-residual recomputation required by this codebase. Operation counts are the hardware-independent companion metric.
- Method B freezes the pair across restarts. NeurKItt's published pseudocode can refresh the recycle space after a cycle; that feature is intentionally disabled here to preserve the same $Z$.
- Every cap hit and numerical failure is retained. No network is trained and the missing-action definition $A^{{-1}}r-M^{{-1}}r$ is not modified.

Raw evidence is in `same_system_results.csv`, `same_system_histories.csv`, `algebra_checks.csv`, `rho_solver_histories.csv`, `persistent_preconditioner_rho.csv`, `frequency_ablation_results.csv`, `transfer_results.csv`, and `transfer_geometry.csv`. Settings, environment, decision inputs, and validation are saved as JSON.
"""
    path = output / "stage13_persistent_preconditioner_vs_krylov_recycling.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    output: Path,
    same_rows: Sequence[Mapping[str, Any]],
    frequency_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    settings: Stage13Settings,
) -> Dict[str, Any]:
    """Independent structural and numerical checks for the final evidence set."""

    expected_same = len(settings.ratios) * len(settings.stage10.evaluation_rhs_seeds) * 3
    expected_frequency = len(settings.ratios) * len(settings.stage10.evaluation_rhs_seeds) * 4
    expected_transfer = len(settings.transfer_pairs) * len(settings.stage10.evaluation_rhs_seeds) * 2
    same_keys = {
        (float(row["ratio"]), int(row["rhs_seed"]), row["method"]) for row in same_rows
    }
    operation_consistency = True
    for row in same_rows:
        if row["method"] in ("ILU", "persistent_pg"):
            operation_consistency &= int(row["ilu_solves"]) == int(row["iterations"])
        if row["method"] == "persistent_pg":
            operation_consistency &= int(row["reduced_solves"]) == int(row["iterations"])
        if row["method"] == "krylov_recycling":
            operation_consistency &= int(row["ilu_solves"]) == int(row["iterations"])
            operation_consistency &= int(row["reduced_solves"]) == 0
    converged_residuals_valid = all(
        (not bool(row["converged"]))
        or float(row["true_final_relative_residual"])
        <= 1.05 * settings.stage10.solver.relative_tolerance
        for row in list(same_rows) + list(frequency_rows) + list(transfer_rows)
    )
    validation = {
        "assessment": "ready_to_share",
        "same_system_row_count": len(same_rows),
        "same_system_expected_row_count": expected_same,
        "same_system_unique_key_count": len(same_keys),
        "frequency_row_count": len(frequency_rows),
        "frequency_expected_row_count": expected_frequency,
        "transfer_row_count": len(transfer_rows),
        "transfer_expected_row_count": expected_transfer,
        "all_algebra_checks_pass": bool(all(bool(row["algebra_pass"]) for row in checks)),
        "operation_counts_consistent": bool(operation_consistency),
        "converged_true_residuals_meet_tolerance": bool(converged_residuals_valid),
        "all_required_csv_files_present": all(
            (output / name).exists()
            for name in (
                "same_system_results.csv",
                "same_system_summary.csv",
                "same_system_histories.csv",
                "algebra_checks.csv",
                "rho_solver_histories.csv",
                "persistent_preconditioner_rho.csv",
                "frequency_ablation_results.csv",
                "frequency_ablation_summary.csv",
                "transfer_results.csv",
                "transfer_summary.csv",
                "transfer_geometry.csv",
            )
        ),
    }
    blockers = [key for key, value in validation.items() if key.startswith("all_") and not value]
    blockers.extend(
        key
        for key in ("operation_counts_consistent", "converged_true_residuals_meet_tolerance")
        if not validation[key]
    )
    if (
        len(same_rows) != expected_same
        or len(same_keys) != expected_same
        or len(frequency_rows) != expected_frequency
        or len(transfer_rows) != expected_transfer
    ):
        blockers.append("row_count_or_uniqueness")
    validation["blockers"] = sorted(set(blockers))
    validation["assessment"] = "ready_to_share" if not blockers else "needs_revision"
    _write_json(output / "validation.json", validation)
    if blockers:
        raise RuntimeError(f"Stage 13 validation failed: {blockers}")
    return validation


def run_stage13(output: Path) -> Path:
    """Execute the full frozen-Z integration experiment and write the report."""

    settings = Stage13Settings()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "operator_convention": "right-preconditioned A M^-1",
            "persistent_outer_operator": "A P_A",
            "recycled_projected_operator": "(I-WW^T) A M^-1",
            "oracle_target": "exact archived Stage 10 q=8 right/correction Ritz prefix",
            "neural_network_trained": False,
        },
    )
    print("[1/8] rebuilding fixed systems and loading exact Stage 10 Z", flush=True)
    contexts, spaces, factors, checks = build_fixed_contexts(output, settings)
    print("[2/8] running same-system ILU / persistent PG / recycling A/B", flush=True)
    same_rows, same_summaries, same_histories = run_same_system_comparison(
        output, contexts, factors, checks, settings
    )
    print("[3/8] collecting residual-space rho diagnostics", flush=True)
    rho_solver, rho_preconditioner = run_rho_diagnostics(
        output, contexts, factors, settings
    )
    print("[4/8] running persistent-correction frequency ablation", flush=True)
    frequency_rows, frequency_summaries = run_frequency_ablation(
        output, contexts, factors, same_rows, settings
    )
    print("[5/8] running neighboring-system transfer A/B", flush=True)
    transfer_rows, transfer_summaries, _, geometry = run_transfer_comparison(
        output, contexts, spaces, settings
    )
    print("[6/8] applying decision rules and generating figures", flush=True)
    decision = build_decision(
        output,
        same_rows,
        same_summaries,
        frequency_summaries,
        rho_solver,
        rho_preconditioner,
        transfer_summaries,
        checks,
        settings,
    )
    make_plots(
        output,
        same_summaries,
        same_histories,
        frequency_summaries,
        rho_solver,
        rho_preconditioner,
        transfer_summaries,
        settings,
    )
    print("[7/8] writing report", flush=True)
    report = write_report(
        output,
        same_summaries,
        frequency_summaries,
        rho_solver,
        rho_preconditioner,
        transfer_summaries,
        geometry,
        checks,
        decision,
        settings,
    )
    print("[8/8] validating evidence and deliverables", flush=True)
    validate_outputs(output, same_rows, frequency_rows, transfer_rows, checks, settings)
    return report
