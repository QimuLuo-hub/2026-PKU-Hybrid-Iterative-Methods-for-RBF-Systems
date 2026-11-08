"""Harder RBF-FD regimes, subspace transfer, and direct-ML baseline."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr
import torch

from ..coarse_space import (
    build_multiplicative_coarse_preconditioner,
    random_orthonormal_basis,
)
from ..config import ILUConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..models.direct_mlp import (
    DirectMLPConfig,
    correction_metrics,
    make_direct_preconditioner,
    train_direct_mlp,
)
from ..pde import ManufacturedSolution, random_smooth_solution
from ..pod import compute_pod, projection_error_curve
from ..preconditioners import (
    InverseAction,
    identity_preconditioner,
    ilu_preconditioner,
    smaller_stencil_ilu_preconditioner,
)
from ..subspace_analysis import (
    energy_dimension,
    interpolate_basis_to_probe,
    krylov_ritz_basis,
    standardized_parameter_distances,
    subspace_metrics,
)
from .harder_regimes import (
    SystemSpec,
    budget_specs,
    build_system,
    controlled_system_families,
    make_snapshot_pair,
    manufactured_problem,
    selected_analysis_specs,
    stack_ml_samples,
    system_metadata,
)
from .stage6_reporting import (
    analyze_evidence,
    create_plots,
    write_optional_assessments,
    write_report,
    write_validation_report,
)


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
CHARCOAL = "#30343B"
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
class Stage6Settings:
    """All experiment values for the reproducible A-F run."""

    seed: int = 20260829
    solver: SolverConfig = SolverConfig(
        relative_tolerance=1.0e-8,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=800,
        breakdown_tolerance=1.0e-14,
    )
    classical_ilu: ILUConfig = ILUConfig(
        drop_tolerance=1.0e-3,
        fill_factor=5.0,
        diagonal_pivot_threshold=0.0,
        permutation="COLAMD",
    )
    budget_drop_tolerances: Tuple[float, ...] = (
        3.0e-2,
        1.0e-2,
        3.0e-3,
        1.0e-3,
        1.0e-4,
    )
    budget_fill_factors: Tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
    snapshot_train_each: int = 20
    snapshot_test_each: int = 10
    pod_q_values: Tuple[int, ...] = (5, 10, 15, 20, 30, 40)
    ritz_max_q: int = 30
    ritz_arnoldi_dimension: int = 60
    common_subspace_q: int = 10
    probe_side: int = 25
    ml_n_nodes: int = 900
    ml_warp_strength: float = 0.95
    ml_train_betas: Tuple[float, ...] = (0.0, 0.6, 1.2, 1.8)
    ml_validation_betas: Tuple[float, ...] = (0.3, 1.5)
    ml_test_betas: Tuple[float, ...] = (0.9, 1.65)
    ml_train_each: int = 48
    ml_validation_each: int = 24
    ml_test_each: int = 24
    mlp: DirectMLPConfig = DirectMLPConfig(
        hidden_width=64,
        hidden_layers=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=32,
        max_epochs=250,
        patience=30,
        seed=20260829,
    )


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _native(row.get(key, "")) for key in fields}
            )


def _solution_metrics(
    result: KrylovResult,
    exact: np.ndarray,
    matrix,
    rhs: np.ndarray,
) -> Dict[str, float]:
    return {
        "relative_solution_error": float(
            np.linalg.norm(result.solution - exact)
            / max(np.linalg.norm(exact), 1.0e-300)
        ),
        "final_relative_residual": float(
            np.linalg.norm(rhs - matrix @ result.solution)
            / max(np.linalg.norm(rhs), 1.0e-300)
        ),
    }


def _record_history(
    rows: List[Dict[str, Any]],
    stage: str,
    system: str,
    rhs_name: str,
    method: str,
    q: int,
    result: KrylovResult,
) -> None:
    for iteration, residual in enumerate(result.relative_residuals):
        rows.append(
            {
                "stage": stage,
                "system": system,
                "rhs": rhs_name,
                "method": method,
                "q": q,
                "iteration": iteration,
                "relative_residual": float(residual),
            }
        )


def _build_default_ilu(system, settings: Stage6Settings) -> InverseAction:
    return ilu_preconditioner(
        system.matrix,
        settings.classical_ilu,
        name="ilu_drop1e-3_fill5",
    )


def _stage_a(
    settings: Stage6Settings,
    histories: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, SystemSpec],
]:
    rows: List[Dict[str, Any]] = []
    systems: Dict[str, Any] = {}
    specs: Dict[str, SystemSpec] = {}
    for spec in controlled_system_families():
        specs[spec.name] = spec
        try:
            system, assembly_seconds = build_system(spec)
            systems[spec.name] = system
            rhs, exact = manufactured_problem(system)
            direct_start = perf_counter()
            direct = np.asarray(spla.spsolve(system.matrix, rhs), dtype=float)
            direct_seconds = perf_counter() - direct_start
            discretization_error = float(
                np.linalg.norm(direct - exact)
                / max(np.linalg.norm(exact), 1.0e-300)
            )
            direct_residual = float(
                np.linalg.norm(rhs - system.matrix @ direct)
                / max(np.linalg.norm(rhs), 1.0e-300)
            )

            unpreconditioned = restarted_gmres(
                system.matrix,
                rhs,
                settings.solver,
                identity_preconditioner(),
            )
            classical = _build_default_ilu(system, settings)
            preconditioned = restarted_gmres(
                system.matrix,
                rhs,
                settings.solver,
                classical,
            )
            metadata = system_metadata(
                spec, system, assembly_seconds
            )
            rows.append(
                {
                    **metadata,
                    "relative_discretization_error": discretization_error,
                    "direct_relative_residual": direct_residual,
                    "direct_solve_seconds": direct_seconds,
                    "unpreconditioned_iterations": (
                        unpreconditioned.iterations
                    ),
                    "unpreconditioned_converged": (
                        unpreconditioned.converged
                    ),
                    "unpreconditioned_solve_seconds": (
                        unpreconditioned.solve_seconds
                    ),
                    "unpreconditioned_total_seconds": (
                        unpreconditioned.solve_seconds
                    ),
                    "classical_iterations": preconditioned.iterations,
                    "classical_converged": preconditioned.converged,
                    "classical_setup_seconds": classical.setup_seconds,
                    "classical_solve_seconds": (
                        preconditioned.solve_seconds
                    ),
                    "classical_application_seconds": (
                        preconditioned.preconditioner_seconds
                    ),
                    "classical_application_seconds_per_call": (
                        preconditioned.preconditioner_seconds
                        / max(classical.apply_count, 1)
                    ),
                    "classical_total_seconds": (
                        classical.setup_seconds
                        + preconditioned.solve_seconds
                    ),
                    "classical_factor_fill_ratio": classical.diagnostics[
                        "factor_fill_ratio"
                    ],
                    "classical_final_relative_residual": (
                        preconditioned.relative_residuals[-1]
                    ),
                    "classical_failure": preconditioned.failure,
                }
            )
            _record_history(
                histories,
                "A",
                spec.name,
                "manufactured_1_1",
                "none",
                0,
                unpreconditioned,
            )
            _record_history(
                histories,
                "A",
                spec.name,
                "manufactured_1_1",
                "classical",
                0,
                preconditioned,
            )
            if not unpreconditioned.converged:
                failures.append(
                    {
                        "stage": "A",
                        "system": spec.name,
                        "operation": "unpreconditioned GMRES",
                        "error": unpreconditioned.failure,
                    }
                )
            if not preconditioned.converged:
                failures.append(
                    {
                        "stage": "A",
                        "system": spec.name,
                        "operation": "classical GMRES",
                        "error": preconditioned.failure,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "stage": "A",
                    "system": spec.name,
                    "operation": "system benchmark",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows, systems, specs


def _mark_pareto(rows: List[Dict[str, Any]]) -> None:
    """Mark nondominated setup-plus-application cost versus iteration count."""

    by_system = sorted({str(row["system"]) for row in rows})
    for system in by_system:
        subset = [row for row in rows if row["system"] == system]
        for row in subset:
            dominated = any(
                (
                    other["preconditioner_cost_seconds"]
                    <= row["preconditioner_cost_seconds"]
                    and other["iterations"] <= row["iterations"]
                    and (
                        other["preconditioner_cost_seconds"]
                        < row["preconditioner_cost_seconds"]
                        or other["iterations"] < row["iterations"]
                    )
                )
                for other in subset
                if other.get("converged")
            )
            row["pareto_optimal"] = bool(
                row.get("converged") and not dominated
            )


def _stage_b(
    systems: Dict[str, Any],
    specs: Dict[str, SystemSpec],
    stage_a_rows: Sequence[Dict[str, Any]],
    settings: Stage6Settings,
    failures: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    unpreconditioned_iterations = {
        str(row["system"]): int(row["unpreconditioned_iterations"])
        for row in stage_a_rows
    }
    for spec in budget_specs():
        system = systems[spec.name]
        rhs, _ = manufactured_problem(system)
        for drop_tolerance in settings.budget_drop_tolerances:
            for fill_factor in settings.budget_fill_factors:
                label = (
                    f"ilu_drop{drop_tolerance:g}_fill{fill_factor:g}"
                )
                try:
                    action = ilu_preconditioner(
                        system.matrix,
                        ILUConfig(
                            drop_tolerance=drop_tolerance,
                            fill_factor=fill_factor,
                            diagonal_pivot_threshold=0.0,
                            permutation="COLAMD",
                        ),
                        name=label,
                    )
                    result = restarted_gmres(
                        system.matrix,
                        rhs,
                        settings.solver,
                        action,
                    )
                    application = result.preconditioner_seconds
                    rows.append(
                        {
                            "system": spec.name,
                            "family": spec.family,
                            "preconditioner": label,
                            "preconditioner_kind": "ilu_A",
                            "drop_tolerance": drop_tolerance,
                            "fill_factor": fill_factor,
                            "factor_fill_ratio": action.diagnostics[
                                "factor_fill_ratio"
                            ],
                            "setup_seconds": action.setup_seconds,
                            "application_seconds": application,
                            "application_seconds_per_call": (
                                application / max(action.apply_count, 1)
                            ),
                            "preconditioner_cost_seconds": (
                                action.setup_seconds + application
                            ),
                            "iterations": result.iterations,
                            "iteration_improvement": (
                                unpreconditioned_iterations[spec.name]
                                - result.iterations
                            ),
                            "convergence_factor": (
                                unpreconditioned_iterations[spec.name]
                                / max(result.iterations, 1)
                            ),
                            "solve_seconds": result.solve_seconds,
                            "total_seconds": (
                                action.setup_seconds
                                + result.solve_seconds
                            ),
                            "converged": result.converged,
                            "final_relative_residual": (
                                result.relative_residuals[-1]
                            ),
                            "failure": result.failure,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "stage": "B",
                            "system": spec.name,
                            "operation": label,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        smaller = max(
            9,
            min(13, spec.stencil_size - 4),
        )
        if smaller < spec.stencil_size:
            try:
                action, cheaper = smaller_stencil_ilu_preconditioner(
                    system,
                    smaller,
                    settings.classical_ilu,
                )
                result = restarted_gmres(
                    system.matrix,
                    rhs,
                    settings.solver,
                    action,
                )
                rows.append(
                    {
                        "system": spec.name,
                        "family": spec.family,
                        "preconditioner": action.name,
                        "preconditioner_kind": "smaller_stencil_ilu",
                        "drop_tolerance": (
                            settings.classical_ilu.drop_tolerance
                        ),
                        "fill_factor": settings.classical_ilu.fill_factor,
                        "factor_fill_ratio": action.diagnostics[
                            "factor_fill_ratio"
                        ],
                        "setup_seconds": action.setup_seconds,
                        "application_seconds": (
                            result.preconditioner_seconds
                        ),
                        "application_seconds_per_call": (
                            result.preconditioner_seconds
                            / max(action.apply_count, 1)
                        ),
                        "preconditioner_cost_seconds": (
                            action.setup_seconds
                            + result.preconditioner_seconds
                        ),
                        "iterations": result.iterations,
                        "iteration_improvement": (
                            unpreconditioned_iterations[spec.name]
                            - result.iterations
                        ),
                        "convergence_factor": (
                            unpreconditioned_iterations[spec.name]
                            / max(result.iterations, 1)
                        ),
                        "solve_seconds": result.solve_seconds,
                        "total_seconds": (
                            action.setup_seconds
                            + result.solve_seconds
                        ),
                        "converged": result.converged,
                        "final_relative_residual": (
                            result.relative_residuals[-1]
                        ),
                        "cheaper_matrix_nnz": cheaper.matrix.nnz,
                        "failure": result.failure,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "stage": "B",
                        "system": spec.name,
                        "operation": "smaller-stencil ILU",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    _mark_pareto(rows)
    return rows


def _stage_c(
    systems: Dict[str, Any],
    settings: Stage6Settings,
    output: Path,
    failures: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Dict[str, float]],
]:
    summary_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    pods: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    costs: Dict[str, Dict[str, float]] = {}
    snapshot_directory = output / "system_snapshots"
    snapshot_directory.mkdir(parents=True, exist_ok=True)

    for index, spec in enumerate(selected_analysis_specs()):
        try:
            system = systems[spec.name]
            action = _build_default_ilu(system, settings)
            train, test, diagnostics = make_snapshot_pair(
                system,
                action,
                settings.solver,
                settings.snapshot_train_each,
                settings.snapshot_test_each,
                settings.seed + 1000 * index,
            )
            svd_start = perf_counter()
            pod = compute_pod(train.corrections)
            svd_seconds = perf_counter() - svd_start
            error_curve = projection_error_curve(
                pod.basis,
                test.corrections,
                pod.basis.shape[1],
            )
            dimensions = {
                threshold: energy_dimension(
                    pod.captured_energy, threshold
                )
                for threshold in (0.90, 0.95, 0.99)
            }
            row: Dict[str, Any] = {
                "system": spec.name,
                "family": spec.family,
                "n_nodes": spec.n_nodes,
                "stencil_size": spec.stencil_size,
                "node_irregularity": system.diagnostics[
                    "nearest_neighbor_cv"
                ],
                "coefficient_beta": spec.beta,
                "coefficient_contrast": spec.coefficient_contrast,
                "train_snapshots": train.corrections.shape[1],
                "test_snapshots": test.corrections.shape[1],
                "snapshot_total_seconds": diagnostics[
                    "snapshot_total_seconds"
                ],
                "svd_seconds": svd_seconds,
                "reference_space_total_seconds": (
                    diagnostics["snapshot_total_seconds"]
                    + svd_seconds
                ),
                "max_reference_relative_residual": diagnostics[
                    "max_reference_relative_residual"
                ],
            }
            for threshold, q in dimensions.items():
                key = int(round(100 * threshold))
                errors = error_curve[q - 1]
                row[f"q{key}"] = q
                row[f"q{key}_over_n"] = q / spec.n_nodes
                row[f"eta_q{key}_mean"] = float(np.mean(errors))
                row[f"eta_q{key}_p90"] = float(
                    np.quantile(errors, 0.9)
                )
            joint_candidates = [
                q
                for q in range(1, pod.basis.shape[1] + 1)
                if pod.captured_energy[q - 1] >= 0.95
                and float(np.mean(error_curve[q - 1])) <= 0.25
            ]
            joint_q = joint_candidates[0] if joint_candidates else 0
            row["q_joint_energy95_eta25"] = joint_q
            row["q_joint_over_n"] = (
                joint_q / spec.n_nodes if joint_q else float("nan")
            )
            row["eta_joint_mean"] = (
                float(np.mean(error_curve[joint_q - 1]))
                if joint_q
                else float("nan")
            )
            row["eta_at_qmax_mean"] = float(np.mean(error_curve[-1]))
            summary_rows.append(row)
            for q in range(1, pod.basis.shape[1] + 1):
                errors = error_curve[q - 1]
                curve_rows.append(
                    {
                        "system": spec.name,
                        "family": spec.family,
                        "n_nodes": spec.n_nodes,
                        "q": q,
                        "q_over_n": q / spec.n_nodes,
                        "normalized_singular_value": float(
                            pod.singular_values[q - 1]
                            / pod.singular_values[0]
                        ),
                        "captured_energy": float(
                            pod.captured_energy[q - 1]
                        ),
                        "heldout_eta_mean": float(np.mean(errors)),
                        "heldout_eta_p90": float(
                            np.quantile(errors, 0.9)
                        ),
                    }
                )
            pods[spec.name] = pod
            snapshots[spec.name] = (train, test)
            costs[spec.name] = {
                **diagnostics,
                "svd_seconds": svd_seconds,
            }
            np.savez_compressed(
                snapshot_directory / f"{spec.name}.npz",
                train_residuals=train.residuals,
                train_corrections=train.corrections,
                train_labels=train.labels,
                test_residuals=test.residuals,
                test_corrections=test.corrections,
                test_labels=test.labels,
                basis=pod.basis,
                singular_values=pod.singular_values,
                captured_energy=pod.captured_energy,
                definition="A^{-1} r - M^{-1} r",
            )
        except Exception as exc:
            failures.append(
                {
                    "stage": "C",
                    "system": spec.name,
                    "operation": "unresolved-correction analysis",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return summary_rows, curve_rows, pods, snapshots, costs


def _stage_d(
    systems: Dict[str, Any],
    pods: Dict[str, Any],
    pod_summary: Sequence[Dict[str, Any]],
    pod_costs: Dict[str, Dict[str, float]],
    settings: Stage6Settings,
    histories: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare POD, random, and Arnoldi-Ritz reference spaces in FGMRES."""

    rows: List[Dict[str, Any]] = []
    pod_summary_by_system = {
        str(row["system"]): row for row in pod_summary
    }
    selected_names = [spec.name for spec in budget_specs()]
    for system_index, system_name in enumerate(selected_names):
        system = systems[system_name]
        pod = pods[system_name]
        base = _build_default_ilu(system, settings)
        max_q = min(
            settings.ritz_max_q,
            pod.basis.shape[1],
            system.matrix.shape[0] - 1,
        )
        try:
            ritz_basis, ritz_diagnostics = krylov_ritz_basis(
                system.matrix,
                base,
                max_q,
                settings.ritz_arnoldi_dimension,
                settings.seed + 3000 + system_index,
            )
        except Exception as exc:
            ritz_basis = None
            ritz_diagnostics = {"setup_seconds": float("nan")}
            failures.append(
                {
                    "stage": "D",
                    "system": system_name,
                    "operation": "Ritz basis",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        summary = pod_summary_by_system[system_name]
        q_values = sorted(
            {
                q
                for q in (
                    *settings.pod_q_values,
                    int(summary["q90"]),
                    int(summary["q95"]),
                    int(summary["q99"]),
                )
                if 0 < q <= max_q
            }
        )
        manufactured_cases = [
            ("manufactured_1_1", ManufacturedSolution()),
            (
                "heldout_smooth_1",
                random_smooth_solution(
                    settings.seed + 4000 + system_index, max_mode=4
                ),
            ),
            (
                "heldout_smooth_2",
                random_smooth_solution(
                    settings.seed + 5000 + system_index, max_mode=4
                ),
            ),
        ]
        cases = [
            (
                name,
                *manufactured_problem(system, manufactured),
            )
            for name, manufactured in manufactured_cases
        ]

        for rhs_name, rhs, exact in cases:
            result = restarted_gmres(
                system.matrix, rhs, settings.solver, base
            )
            rows.append(
                {
                    "system": system_name,
                    "rhs": rhs_name,
                    "method_family": "classical",
                    "method": base.name,
                    "q": 0,
                    "captured_energy": 0.0,
                    "basis_setup_seconds": 0.0,
                    "coarse_setup_seconds": 0.0,
                    "setup_seconds": base.setup_seconds,
                    "solve_seconds": result.solve_seconds,
                    "total_seconds": (
                        base.setup_seconds + result.solve_seconds
                    ),
                    "iterations": result.iterations,
                    "converged": result.converged,
                    **_solution_metrics(
                        result, exact, system.matrix, rhs
                    ),
                    "failure": result.failure,
                }
            )
            _record_history(
                histories,
                "D",
                system_name,
                rhs_name,
                "classical",
                0,
                result,
            )

        for q in q_values:
            random_start = perf_counter()
            random_basis = random_orthonormal_basis(
                system.matrix.shape[0],
                q,
                settings.seed + 6000 + q + system_index,
            )
            random_setup = perf_counter() - random_start
            basis_choices = [
                (
                    "pod",
                    pod.basis[:, :q],
                    pod_costs[system_name]["snapshot_total_seconds"]
                    + pod_costs[system_name]["svd_seconds"],
                ),
                ("random", random_basis, random_setup),
            ]
            if ritz_basis is not None:
                basis_choices.append(
                    (
                        "ritz",
                        ritz_basis[:, :q],
                        float(ritz_diagnostics["setup_seconds"]),
                    )
                )
            for family, basis, basis_setup in basis_choices:
                try:
                    action, diagnostics = (
                        build_multiplicative_coarse_preconditioner(
                            system.matrix,
                            base,
                            basis,
                            symmetry_threshold=1.0e-10,
                            name=family,
                        )
                    )
                    total_setup = (
                        action.setup_seconds + basis_setup
                    )
                    for rhs_name, rhs, exact in cases:
                        result = restarted_gmres(
                            system.matrix,
                            rhs,
                            settings.solver,
                            action,
                        )
                        rows.append(
                            {
                                "system": system_name,
                                "rhs": rhs_name,
                                "method_family": family,
                                "method": action.name,
                                "q": q,
                                "captured_energy": float(
                                    pod.captured_energy[q - 1]
                                )
                                if family == "pod"
                                else float("nan"),
                                "basis_setup_seconds": basis_setup,
                                "coarse_setup_seconds": diagnostics[
                                    "coarse_setup_seconds"
                                ],
                                "setup_seconds": total_setup,
                                "solve_seconds": result.solve_seconds,
                                "total_seconds": (
                                    total_setup + result.solve_seconds
                                ),
                                "iterations": result.iterations,
                                "converged": result.converged,
                                "coarse_variant": diagnostics[
                                    "coarse_variant"
                                ],
                                "reduced_condition": diagnostics[
                                    "reduced_condition"
                                ],
                                **_solution_metrics(
                                    result,
                                    exact,
                                    system.matrix,
                                    rhs,
                                ),
                                "failure": result.failure,
                            }
                        )
                        _record_history(
                            histories,
                            "D",
                            system_name,
                            rhs_name,
                            family,
                            q,
                            result,
                        )
                except Exception as exc:
                    failures.append(
                        {
                            "stage": "D",
                            "system": system_name,
                            "operation": f"{family}_q{q}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return rows


def _write_matrix_csv(
    path: Path,
    names: Sequence[str],
    matrix: np.ndarray,
) -> None:
    rows = []
    for index, name in enumerate(names):
        row: Dict[str, Any] = {"system": name}
        row.update(
            {
                other: float(matrix[index, column])
                for column, other in enumerate(names)
            }
        )
        rows.append(row)
    _write_csv(path, rows)


def _stage_e(
    systems: Dict[str, Any],
    pods: Dict[str, Any],
    settings: Stage6Settings,
    output: Path,
    failures: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, np.ndarray],
]:
    """Compare independently constructed spaces on a common physical probe grid."""

    specs = [
        spec
        for spec in selected_analysis_specs()
        if spec.name in pods
    ]
    q = min(
        settings.common_subspace_q,
        min(pods[spec.name].basis.shape[1] for spec in specs),
    )
    axis = np.linspace(0.0, 1.0, settings.probe_side)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    probe_nodes = np.column_stack((xx.ravel(), yy.ravel()))
    common_bases: Dict[str, np.ndarray] = {}
    subspace_directory = output / "reference_subspaces"
    subspace_directory.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        try:
            common = interpolate_basis_to_probe(
                systems[spec.name].nodes,
                pods[spec.name].basis[:, :q],
                probe_nodes,
            )
            common_bases[spec.name] = common
            projector = common @ common.T
            np.savez_compressed(
                subspace_directory / f"{spec.name}.npz",
                local_basis=pods[spec.name].basis[:, :q],
                common_probe_basis=common,
                common_probe_projector=projector,
                probe_nodes=probe_nodes,
                q=q,
                n_nodes=spec.n_nodes,
                stencil_size=spec.stencil_size,
                node_distribution=spec.node_distribution,
                jitter_fraction=spec.jitter_fraction,
                warp_strength=spec.warp_strength,
                coefficient_beta=spec.beta,
                coefficient_contrast=spec.coefficient_contrast,
            )
        except Exception as exc:
            failures.append(
                {
                    "stage": "E",
                    "system": spec.name,
                    "operation": "common-grid interpolation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    specs = [spec for spec in specs if spec.name in common_bases]
    names = [spec.name for spec in specs]
    parameter_rows = [
        [
            math.log(spec.n_nodes),
            systems[spec.name].diagnostics["nearest_neighbor_cv"],
            spec.stencil_size,
            spec.beta,
            spec.coefficient_frequency,
        ]
        for spec in specs
    ]
    parameter_distance = standardized_parameter_distances(
        parameter_rows
    )
    projector_matrix = np.zeros((len(specs), len(specs)))
    angle_matrix = np.zeros((len(specs), len(specs)))
    rows: List[Dict[str, Any]] = []
    for first_index, first in enumerate(specs):
        for second_index, second in enumerate(specs):
            metrics = subspace_metrics(
                common_bases[first.name],
                common_bases[second.name],
            )
            projector_matrix[first_index, second_index] = metrics[
                "projector_distance"
            ]
            angle_matrix[first_index, second_index] = metrics[
                "principal_angle_mean_degrees"
            ]
            rows.append(
                {
                    "system_i": first.name,
                    "system_j": second.name,
                    "family_i": first.family,
                    "family_j": second.family,
                    "q": q,
                    "projector_distance": metrics[
                        "projector_distance"
                    ],
                    "projector_distance_normalized": metrics[
                        "projector_distance_normalized"
                    ],
                    "principal_angle_mean_degrees": metrics[
                        "principal_angle_mean_degrees"
                    ],
                    "principal_angle_max_degrees": metrics[
                        "principal_angle_max_degrees"
                    ],
                    "principal_angles_degrees": json.dumps(
                        np.asarray(
                            metrics["principal_angles_degrees"]
                        ).tolist()
                    ),
                    "parameter_distance": parameter_distance[
                        first_index, second_index
                    ],
                    "delta_log_n": abs(
                        math.log(first.n_nodes)
                        - math.log(second.n_nodes)
                    ),
                    "delta_irregularity": abs(
                        systems[first.name].diagnostics[
                            "nearest_neighbor_cv"
                        ]
                        - systems[second.name].diagnostics[
                            "nearest_neighbor_cv"
                        ]
                    ),
                    "delta_stencil_size": abs(
                        first.stencil_size - second.stencil_size
                    ),
                    "delta_beta": abs(first.beta - second.beta),
                }
            )

    upper = np.triu_indices(len(specs), k=1)
    overall_correlation = spearmanr(
        parameter_distance[upper],
        projector_matrix[upper],
    )
    within_family: Dict[str, Dict[str, float]] = {}
    for family in sorted({spec.family for spec in specs}):
        family_indices = [
            index
            for index, spec in enumerate(specs)
            if spec.family == family
        ]
        if len(family_indices) < 3:
            continue
        distances = []
        projectors = []
        for position, first_index in enumerate(family_indices):
            for second_index in family_indices[position + 1 :]:
                distances.append(
                    parameter_distance[first_index, second_index]
                )
                projectors.append(
                    projector_matrix[first_index, second_index]
                )
        correlation = spearmanr(distances, projectors)
        within_family[family] = {
            "spearman_rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "pairs": len(distances),
        }
    summary = {
        "q": q,
        "probe_nodes": int(probe_nodes.shape[0]),
        "systems": names,
        "overall_spearman_rho": float(
            overall_correlation.statistic
        ),
        "overall_spearman_p_value": float(
            overall_correlation.pvalue
        ),
        "within_family": within_family,
        "interpolation_note": (
            "Each independently fitted nodal POD basis was linearly "
            "interpolated to the same physical probe grid and then "
            "re-orthonormalized before basis-invariant comparison."
        ),
    }
    _write_matrix_csv(
        output / "projector_distance_matrix.csv",
        names,
        projector_matrix,
    )
    _write_matrix_csv(
        output / "principal_angle_mean_matrix.csv",
        names,
        angle_matrix,
    )
    _write_matrix_csv(
        output / "parameter_distance_matrix.csv",
        names,
        parameter_distance,
    )
    (output / "subspace_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    matrices = {
        "projector": projector_matrix,
        "angle": angle_matrix,
        "parameter": parameter_distance,
        "names": np.asarray(names),
    }
    return rows, summary, matrices


def _ml_spec(beta: float, split: str, settings: Stage6Settings) -> SystemSpec:
    return SystemSpec(
        name=f"ML_{split}_beta{int(round(100 * beta)):03d}",
        family=f"F_ML_{split}",
        n_nodes=settings.ml_n_nodes,
        stencil_size=25,
        node_distribution="warped_grid",
        jitter_fraction=0.0,
        warp_strength=settings.ml_warp_strength,
        beta=beta,
    )


def _collect_ml_systems(
    betas: Sequence[float],
    split: str,
    each: int,
    settings: Stage6Settings,
    seed_offset: int,
    failures: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    systems: Dict[str, Dict[str, Any]] = {}
    for index, beta in enumerate(betas):
        spec = _ml_spec(beta, split, settings)
        try:
            system, assembly_seconds = build_system(spec)
            action = _build_default_ilu(system, settings)
            train, test, diagnostics = make_snapshot_pair(
                system,
                action,
                settings.solver,
                each,
                max(4, settings.ml_test_each // 2),
                settings.seed + seed_offset + 100 * index,
            )
            systems[spec.name] = {
                "spec": spec,
                "system": system,
                "assembly_seconds": assembly_seconds,
                "train": train,
                "test": test,
                "snapshot_diagnostics": diagnostics,
            }
        except Exception as exc:
            failures.append(
                {
                    "stage": "F",
                    "system": spec.name,
                    "operation": "ML data generation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return systems


def _stack_entries(
    entries: Sequence[Dict[str, Any]],
    snapshot_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual_blocks = []
    correction_blocks = []
    features = []
    labels = []
    residual_sources = []
    for entry in entries:
        snapshot = entry[snapshot_key]
        residual_blocks.append(snapshot.residuals)
        correction_blocks.append(snapshot.corrections)
        features.append([entry["spec"].beta])
        labels.extend([entry["spec"].name] * snapshot.residuals.shape[1])
        residual_sources.extend(snapshot.labels.tolist())
    residuals, corrections, system_features = stack_ml_samples(
        residual_blocks,
        correction_blocks,
        features,
    )
    return (
        residuals,
        corrections,
        system_features,
        np.asarray(labels),
        np.asarray(residual_sources),
    )


def _append_ml_solver_row(
    rows: List[Dict[str, Any]],
    histories: List[Dict[str, Any]],
    split: str,
    system_name: str,
    beta: float,
    rhs_name: str,
    family: str,
    q: int,
    action: InverseAction,
    result: KrylovResult,
    exact: np.ndarray,
    matrix,
    rhs: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    additional_setup_seconds: float = 0.0,
) -> None:
    setup = action.setup_seconds + additional_setup_seconds
    rows.append(
        {
            "split": split,
            "system": system_name,
            "coefficient_beta": beta,
            "coefficient_contrast": float(np.exp(2.0 * abs(beta))),
            "rhs": rhs_name,
            "method_family": family,
            "method": action.name,
            "q": q,
            "nonlinear_action": not action.is_linear,
            "training_seconds": training_seconds,
            "setup_seconds": setup,
            "inference_seconds": inference_seconds,
            "solve_seconds": result.solve_seconds,
            "online_total_seconds": setup + result.solve_seconds,
            "single_solve_total_with_training_seconds": (
                training_seconds + setup + result.solve_seconds
            ),
            "iterations": result.iterations,
            "converged": result.converged,
            **_solution_metrics(result, exact, matrix, rhs),
            "failure": result.failure,
        }
    )
    _record_history(
        histories,
        "F",
        system_name,
        rhs_name,
        family,
        q,
        result,
    )


def _stage_f(
    settings: Stage6Settings,
    output: Path,
    histories: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, float]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """Train and evaluate the fixed-size direct missing-action MLP."""

    train_systems = _collect_ml_systems(
        settings.ml_train_betas,
        "train",
        settings.ml_train_each,
        settings,
        10000,
        failures,
    )
    validation_systems = _collect_ml_systems(
        settings.ml_validation_betas,
        "validation",
        settings.ml_validation_each,
        settings,
        20000,
        failures,
    )
    test_systems = _collect_ml_systems(
        settings.ml_test_betas,
        "test",
        settings.ml_test_each,
        settings,
        30000,
        failures,
    )
    if (
        len(train_systems) != len(settings.ml_train_betas)
        or len(validation_systems)
        != len(settings.ml_validation_betas)
        or len(test_systems) != len(settings.ml_test_betas)
    ):
        raise RuntimeError("one or more required ML systems failed")

    train_entries = list(train_systems.values())
    validation_entries = list(validation_systems.values())
    test_entries = list(test_systems.values())
    train_r, train_d, train_xi, train_labels, train_sources = _stack_entries(
        train_entries, "train"
    )
    validation_r, validation_d, validation_xi, validation_labels, validation_sources = (
        _stack_entries(validation_entries, "train")
    )
    known_r, known_d, known_xi, known_labels, known_sources = _stack_entries(
        train_entries, "test"
    )
    unseen_r, unseen_d, unseen_xi, unseen_labels, unseen_sources = _stack_entries(
        test_entries, "train"
    )

    training = train_direct_mlp(
        train_r,
        train_d,
        train_xi,
        validation_r,
        validation_d,
        validation_xi,
        ("coefficient_beta",),
        settings.mlp,
    )
    training.predictor.save(
        output / "direct_mlp_checkpoint.pt",
        settings.mlp,
        training.history,
    )
    _write_csv(output / "direct_mlp_training_history.csv", training.history)
    np.savez_compressed(
        output / "direct_mlp_datasets.npz",
        train_residuals=train_r,
        train_corrections=train_d,
        train_system_features=train_xi,
        train_system_labels=train_labels,
        train_residual_sources=train_sources,
        validation_residuals=validation_r,
        validation_corrections=validation_d,
        validation_system_features=validation_xi,
        validation_system_labels=validation_labels,
        validation_residual_sources=validation_sources,
        known_test_residuals=known_r,
        known_test_corrections=known_d,
        known_test_system_features=known_xi,
        known_test_system_labels=known_labels,
        known_test_residual_sources=known_sources,
        unseen_test_residuals=unseen_r,
        unseen_test_corrections=unseen_d,
        unseen_test_system_features=unseen_xi,
        unseen_test_system_labels=unseen_labels,
        unseen_test_residual_sources=unseen_sources,
        definition="A(xi)^{-1} r - M(xi)^{-1} r",
    )

    correction_rows: List[Dict[str, Any]] = []
    correction_sets = [
        (
            "training",
            train_r,
            train_d,
            train_xi,
            train_labels,
            train_sources,
        ),
        (
            "validation_unseen_systems",
            validation_r,
            validation_d,
            validation_xi,
            validation_labels,
            validation_sources,
        ),
        (
            "known_systems_unseen_residuals",
            known_r,
            known_d,
            known_xi,
            known_labels,
            known_sources,
        ),
        (
            "unseen_parameter_systems",
            unseen_r,
            unseen_d,
            unseen_xi,
            unseen_labels,
            unseen_sources,
        ),
    ]
    for split, residuals, corrections, features, labels, sources in correction_sets:
        predictions = training.predictor.predict_batch(
            residuals, features
        )
        metrics = correction_metrics(predictions, corrections)
        for index in range(residuals.shape[0]):
            correction_rows.append(
                {
                    "split": split,
                    "system": labels[index],
                    "residual_source": sources[index],
                    "coefficient_beta": features[index, 0],
                    "relative_correction_error": metrics[
                        "relative_error"
                    ][index],
                    "cosine_similarity": metrics[
                        "cosine_similarity"
                    ][index],
                    "target_norm": float(
                        np.linalg.norm(corrections[index])
                    ),
                    "prediction_norm": float(
                        np.linalg.norm(predictions[index])
                    ),
                }
            )

    solver_rows: List[Dict[str, Any]] = []
    solver_direction_rows: List[Dict[str, Any]] = []
    evaluation_entries = []
    for entry in train_entries:
        if entry["spec"].beta in (0.6, 1.2):
            evaluation_entries.append(("known_system", entry))
    evaluation_entries.extend(
        ("unseen_parameter", entry) for entry in test_entries
    )
    for system_index, (split, entry) in enumerate(evaluation_entries):
        system = entry["system"]
        spec = entry["spec"]
        reference_start = perf_counter()
        reference_pod = compute_pod(entry["train"].corrections)
        reference_seconds = perf_counter() - reference_start
        q95 = energy_dimension(reference_pod.captured_energy, 0.95)
        pod_action, _ = build_multiplicative_coarse_preconditioner(
            system.matrix,
            _build_default_ilu(system, settings),
            reference_pod.basis[:, :q95],
            symmetry_threshold=1.0e-10,
            name="pod",
        )
        exact_factor = spla.splu(system.matrix.tocsc())
        cases = [
            ("manufactured_1_1", ManufacturedSolution()),
            (
                "heldout_smooth_1",
                random_smooth_solution(
                    settings.seed + 40000 + system_index, max_mode=4
                ),
            ),
            (
                "heldout_smooth_2",
                random_smooth_solution(
                    settings.seed + 50000 + system_index, max_mode=4
                ),
            ),
        ]
        for rhs_name, manufactured in cases:
            rhs, exact = manufactured_problem(system, manufactured)

            classical = _build_default_ilu(system, settings)
            classical_result = restarted_gmres(
                system.matrix,
                rhs,
                settings.solver,
                classical,
            )
            _append_ml_solver_row(
                solver_rows,
                histories,
                split,
                spec.name,
                spec.beta,
                rhs_name,
                "classical",
                0,
                classical,
                classical_result,
                exact,
                system.matrix,
                rhs,
                training.training_seconds,
                0.0,
            )

            pod_result = restarted_gmres(
                system.matrix,
                rhs,
                settings.solver,
                pod_action,
            )
            _append_ml_solver_row(
                solver_rows,
                histories,
                split,
                spec.name,
                spec.beta,
                rhs_name,
                "pod",
                q95,
                pod_action,
                pod_result,
                exact,
                system.matrix,
                rhs,
                training.training_seconds,
                0.0,
                additional_setup_seconds=(
                    entry["snapshot_diagnostics"][
                        "snapshot_total_seconds"
                    ]
                    + reference_seconds
                ),
            )

            learned_base = _build_default_ilu(system, settings)
            traced_inputs: List[np.ndarray] = []
            learned = make_direct_preconditioner(
                learned_base,
                training.predictor,
                np.array([spec.beta]),
                trace_inputs=traced_inputs,
                trace_limit=50,
            )
            # Warm the Torch path before timing this right-preconditioner.
            training.predictor.predict(
                np.ones(system.matrix.shape[0]), np.array([spec.beta])
            )
            training.predictor.reset_timing()
            learned_result = restarted_gmres(
                system.matrix,
                rhs,
                settings.solver,
                learned,
            )
            inference_seconds = training.predictor.inference_seconds
            _append_ml_solver_row(
                solver_rows,
                histories,
                split,
                spec.name,
                spec.beta,
                rhs_name,
                "direct_mlp",
                0,
                learned,
                learned_result,
                exact,
                system.matrix,
                rhs,
                training.training_seconds,
                inference_seconds,
            )
            if traced_inputs:
                traced = np.row_stack(traced_inputs)
                traced_features = np.full(
                    (traced.shape[0], 1), spec.beta, dtype=float
                )
                predicted = training.predictor.predict_batch(
                    traced, traced_features
                )
                exact_actions = np.row_stack(
                    [exact_factor.solve(vector) for vector in traced]
                )
                classical_actions = np.row_stack(
                    [learned_base.apply(vector) for vector in traced]
                )
                targets = exact_actions - classical_actions
                metrics = correction_metrics(predicted, targets)
                for trace_index in range(traced.shape[0]):
                    target_norm = float(np.linalg.norm(targets[trace_index]))
                    base_norm = float(
                        np.linalg.norm(classical_actions[trace_index])
                    )
                    solver_direction_rows.append(
                        {
                            "split": split,
                            "system": spec.name,
                            "rhs": rhs_name,
                            "arnoldi_application_index": trace_index,
                            "relative_correction_error": metrics[
                                "relative_error"
                            ][trace_index],
                            "cosine_similarity": metrics[
                                "cosine_similarity"
                            ][trace_index],
                            "prediction_over_target_norm": float(
                                np.linalg.norm(predicted[trace_index])
                                / max(target_norm, 1.0e-300)
                            ),
                            "target_over_classical_norm": (
                                target_norm / max(base_norm, 1.0e-300)
                            ),
                        }
                    )
            if not learned_result.converged:
                failures.append(
                    {
                        "stage": "F",
                        "system": spec.name,
                        "operation": f"direct MLP FGMRES/{rhs_name}",
                        "error": learned_result.failure,
                    }
                )

    summary = {
        "training_seconds": training.training_seconds,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "train_systems": [entry["spec"].name for entry in train_entries],
        "train_betas": [entry["spec"].beta for entry in train_entries],
        "validation_systems": [
            entry["spec"].name for entry in validation_entries
        ],
        "validation_betas": [
            entry["spec"].beta for entry in validation_entries
        ],
        "test_systems": [entry["spec"].name for entry in test_entries],
        "test_betas": [entry["spec"].beta for entry in test_entries],
        "model_parameters": int(
            sum(
                parameter.numel()
                for parameter in training.predictor.model.parameters()
            )
        ),
        "hidden_width": settings.mlp.hidden_width,
        "hidden_layers": settings.mlp.hidden_layers,
        "train_samples": int(train_r.shape[0]),
        "validation_samples": int(validation_r.shape[0]),
    }
    (output / "direct_mlp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return (
        solver_rows,
        correction_rows,
        training.history,
        summary,
        solver_direction_rows,
    )


def _software_environment() -> Dict[str, Any]:
    """Return the numerical runtime recorded with every reproducible run."""

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


def run_stage6_experiment(
    output_directory: Path,
    settings: Optional[Stage6Settings] = None,
) -> Dict[str, Any]:
    """Run required Stages A--F, apply evidence gates, and write the report."""

    chosen = Stage6Settings() if settings is None else settings
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    histories: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    (output / "experiment_config.json").write_text(
        json.dumps(asdict(chosen), indent=2), encoding="utf-8"
    )
    (output / "software_environment.json").write_text(
        json.dumps(_software_environment(), indent=2), encoding="utf-8"
    )

    print("[A] assembling and benchmarking controlled harder regimes", flush=True)
    stage_a, systems, specs = _stage_a(chosen, histories, failures)
    _write_csv(output / "harder_regime_systems.csv", stage_a)

    print("[B] sweeping classical preconditioner budgets", flush=True)
    stage_b = _stage_b(
        systems, specs, stage_a, chosen, failures
    )
    _write_csv(output / "preconditioner_budget.csv", stage_b)

    print("[C] generating exact missing-action snapshots and POD diagnostics", flush=True)
    (
        stage_c_summary,
        stage_c_curves,
        pods,
        snapshots,
        pod_costs,
    ) = _stage_c(systems, chosen, output, failures)
    _write_csv(output / "pod_system_summary.csv", stage_c_summary)
    _write_csv(output / "pod_curves.csv", stage_c_curves)

    print("[D] testing POD, random, and Ritz spaces in the Krylov solve", flush=True)
    stage_d = _stage_d(
        systems,
        pods,
        stage_c_summary,
        pod_costs,
        chosen,
        histories,
        failures,
    )
    _write_csv(output / "krylov_subspace_results.csv", stage_d)

    print("[E] comparing independently fitted difficult subspaces", flush=True)
    stage_e_rows, stage_e_summary, stage_e_matrices = _stage_e(
        systems, pods, chosen, output, failures
    )
    _write_csv(output / "subspace_pairwise.csv", stage_e_rows)

    print("[F] training and evaluating the fixed-size direct-complement MLP", flush=True)
    (
        ml_solver_rows,
        ml_correction_rows,
        ml_history,
        ml_summary,
        ml_solver_direction_rows,
    ) = _stage_f(chosen, output, histories, failures)
    _write_csv(
        output / "direct_mlp_solver_results.csv", ml_solver_rows
    )
    _write_csv(
        output / "direct_mlp_correction_metrics.csv",
        ml_correction_rows,
    )
    _write_csv(
        output / "direct_mlp_solver_direction_metrics.csv",
        ml_solver_direction_rows,
    )
    _write_csv(output / "residual_histories.csv", histories)
    _write_csv(output / "failures.csv", failures)

    print("[report] synthesizing evidence, gates, figures, and Markdown", flush=True)
    evidence = analyze_evidence(
        stage_a,
        stage_b,
        stage_c_summary,
        stage_d,
        stage_e_rows,
        stage_e_summary,
        ml_correction_rows,
        ml_solver_rows,
        ml_solver_direction_rows,
    )
    (output / "evidence_summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    charts = create_plots(
        output,
        stage_a,
        stage_b,
        stage_c_summary,
        stage_c_curves,
        stage_d,
        stage_e_rows,
        stage_e_matrices,
        ml_correction_rows,
        ml_solver_rows,
        ml_history,
        histories,
    )
    optional_paths = write_optional_assessments(output, evidence)
    report_path = write_report(
        output,
        evidence,
        stage_a,
        stage_b,
        stage_c_summary,
        stage_d,
        stage_e_summary,
        ml_correction_rows,
        ml_solver_rows,
        ml_solver_direction_rows,
        ml_summary,
        failures,
    )
    validation_path = write_validation_report(
        output, evidence, failures, len(charts)
    )
    manifest = {
        "output_directory": str(output),
        "report": str(report_path),
        "validation": str(validation_path),
        "optional_evidence_gated_outputs": [
            str(path) for path in optional_paths
        ],
        "gates": evidence["gates"],
        "raw_tables": [
            "harder_regime_systems.csv",
            "preconditioner_budget.csv",
            "pod_system_summary.csv",
            "pod_curves.csv",
            "krylov_subspace_results.csv",
            "subspace_pairwise.csv",
            "direct_mlp_correction_metrics.csv",
            "direct_mlp_solver_results.csv",
            "direct_mlp_solver_direction_metrics.csv",
            "residual_histories.csv",
            "failures.csv",
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] report: {report_path}", flush=True)
    return manifest
