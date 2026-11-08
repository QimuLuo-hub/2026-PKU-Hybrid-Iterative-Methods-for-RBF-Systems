"""Run global-RBF Stage 1: discretization, classical baselines, and structure.

The script intentionally stops before neural modeling.  It writes raw CSV
evidence and static research figures into one output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.collocation import GlobalRBFSystem, assemble_poisson_system, matrix_diagnostics
from global_rbf.krylov import FGMRESResult, fgmres
from global_rbf.nodes import square_nodes
from global_rbf.preconditioners import (
    Preconditioner,
    dense_block_jacobi_preconditioner,
    exact_lu_preconditioner,
    global_residual_sparse_approximate_inverse,
    identity_preconditioner,
    jacobi_preconditioner,
    local_truncation_preconditioner,
    low_precision_lu_preconditioner,
)
from global_rbf.structure import (
    CorrectionData,
    build_correction_data,
    harmonic_ritz_space,
    normalized_columns,
    oracle_coarse_preconditioner,
    pod_metrics,
    residual_history_space,
    smallest_right_singular_space,
)


SEED = 20260830
RTOL = 1.0e-10
PALETTE = {
    "blue": "#1F5A94",
    "gold": "#C58C24",
    "orange": "#D06B32",
    "olive": "#708238",
    "pink": "#B4557D",
    "ink": "#252A31",
    "grey": "#8A9099",
    "light": "#DCE2E8",
}


@dataclass(frozen=True)
class SystemConfig:
    study: str
    side: int
    distribution: str
    kernel: str
    epsilon: float | None

    @property
    def node_count(self) -> int:
        return self.side**2

    @property
    def config_id(self) -> str:
        eps = "none" if self.epsilon is None else f"{self.epsilon:g}"
        return f"{self.study}_{self.kernel}_{self.distribution}_N{self.node_count}_eps{eps}"


@dataclass
class StructureArtifact:
    config: SystemConfig
    system: GlobalRBFSystem
    base: Preconditioner
    trajectory: FGMRESResult
    data: CorrectionData
    snapshot_setup_seconds: float


def core_epsilon(kernel: str, side: int) -> float | None:
    if kernel == "gaussian":
        return 0.4 * side
    if kernel == "imq":
        return 0.2 * side
    if kernel == "phs3":
        return None
    raise ValueError(kernel)


def core_configs(quick: bool) -> list[SystemConfig]:
    sides = [10, 15] if quick else [10, 15, 20, 25, 30]
    return [
        SystemConfig("core", side, distribution, kernel, core_epsilon(kernel, side))
        for distribution in ("tensor", "jittered")
        for kernel in ("gaussian", "imq", "phs3")
        for side in sides
    ]


def shape_configs(quick: bool) -> list[SystemConfig]:
    sides = [15] if quick else [15, 20]
    epsilons = [2.0, 4.0, 6.0, 8.0] if quick else [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    return [
        SystemConfig("shape", side, distribution, kernel, epsilon)
        for distribution in ("tensor", "jittered")
        for kernel in ("gaussian", "imq")
        for side in sides
        for epsilon in epsilons
    ]


def structure_configs(quick: bool) -> list[SystemConfig]:
    if quick:
        specs = [("gaussian", "tensor", 10), ("imq", "tensor", 15)]
    else:
        specs = [
            ("gaussian", "tensor", 10),
            ("gaussian", "tensor", 15),
            ("gaussian", "tensor", 20),
            ("gaussian", "jittered", 15),
            ("gaussian", "jittered", 20),
            ("imq", "tensor", 10),
            ("imq", "tensor", 15),
            ("imq", "tensor", 20),
            ("imq", "jittered", 15),
            ("imq", "jittered", 20),
        ]
    return [SystemConfig("structure", side, distribution, kernel, core_epsilon(kernel, side)) for kernel, distribution, side in specs]


def oracle_ids(quick: bool) -> set[tuple[str, str, int]]:
    if quick:
        return {("imq", "tensor", 15)}
    return {
        ("gaussian", "tensor", 15),
        ("gaussian", "jittered", 20),
        ("imq", "tensor", 15),
        ("imq", "tensor", 20),
        ("imq", "jittered", 15),
    }


def assemble(config: SystemConfig) -> GlobalRBFSystem:
    nodes, boundary = square_nodes(
        config.side,
        config.distribution,
        seed=SEED + config.side,
    )
    return assemble_poisson_system(
        nodes,
        boundary,
        kernel=config.kernel,
        epsilon=config.epsilon,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config_fields(config: SystemConfig) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "study": config.study,
        "N": config.node_count,
        "system_dimension": config.node_count + (6 if config.kernel == "phs3" else 0),
        "side": config.side,
        "distribution": config.distribution,
        "kernel": config.kernel,
        "epsilon": config.epsilon,
    }


def characterize(configs: list[SystemConfig], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        print(f"[characterize {index}/{len(configs)}] {config.config_id}", flush=True)
        try:
            system = assemble(config)
            factor_start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", la.LinAlgWarning)
                lu, piv = la.lu_factor(system.A, check_finite=False)
            factor_seconds = perf_counter() - factor_start
            solve_start = perf_counter()
            x = la.lu_solve((lu, piv), system.b, check_finite=False)
            triangular_solve_seconds = perf_counter() - solve_start
            metrics = system.error_metrics(x, evaluation_side=41)
            diag_start = perf_counter()
            diag = matrix_diagnostics(system.A, exact_svd=True)
            diagnostics_seconds = perf_counter() - diag_start
            exposure = diag["condition_2"] * np.finfo(float).eps
            if caught or exposure >= 1.0:
                interpretation = "roundoff-dominated native basis"
            elif metrics["relative_residual"] > 1.0e-8:
                interpretation = "reference residual is inadequate"
            elif metrics["grid_rel_l2"] > 0.1:
                interpretation = "linear solve resolved; discretization inaccurate"
            else:
                interpretation = "resolved native-basis solve"
            row = config_fields(config)
            row.update(
                {
                    **diag,
                    **metrics,
                    "roundoff_exposure_kappa_u": exposure,
                    "reference_linalg_warning": bool(caught),
                    "numerical_interpretation": interpretation,
                    "assembly_seconds": system.assembly_seconds,
                    "factorization_seconds": factor_seconds,
                    "triangular_solve_seconds": triangular_solve_seconds,
                    "direct_total_seconds": system.assembly_seconds + factor_seconds + triangular_solve_seconds,
                    "diagnostics_seconds": diagnostics_seconds,
                    "factor_memory_bytes": int(lu.nbytes + piv.nbytes),
                }
            )
            rows.append(row)
        except Exception as exc:
            failures.append({**config_fields(config), "stage": "characterization", "error": repr(exc)})
    return rows


def solver_run(
    system: GlobalRBFSystem,
    preconditioner: Preconditioner,
    b: np.ndarray,
    *,
    repeats: int,
    capture: bool = False,
) -> tuple[FGMRESResult, float]:
    results = [
        fgmres(
            system.A,
            b,
            preconditioner.apply,
            rtol=RTOL,
            restart=min(100, system.dimension),
            max_iterations=min(400, 2 * system.dimension),
            capture_residual_vectors=capture and repeat == 0,
            capture_arnoldi=capture and repeat == 0,
        )
        for repeat in range(repeats)
    ]
    primary = results[0]
    median_seconds = float(np.median([result.solve_seconds for result in results]))
    return primary, median_seconds


def classical_baselines(
    configs: list[SystemConfig],
    failures: list[dict[str, Any]],
    *,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    repeats = 1 if quick else 3
    for index, config in enumerate(configs, start=1):
        print(f"[classical {index}/{len(configs)}] {config.config_id}", flush=True)
        try:
            system = assemble(config)
            reference_lu = la.lu_factor(system.A, check_finite=False)
            reference = la.lu_solve(reference_lu, system.b, check_finite=False)
        except Exception as exc:
            failures.append({**config_fields(config), "stage": "classical_system", "error": repr(exc)})
            continue

        builders: list[tuple[str, Callable[[], Preconditioner]]] = [
            ("identity", lambda: identity_preconditioner(system.dimension)),
        ]
        if config.kernel != "phs3":
            builders.extend(
                [
                    ("jacobi", lambda: jacobi_preconditioner(system.A)),
                    (
                        "block_jacobi_3x3",
                        lambda: dense_block_jacobi_preconditioner(system.A, system.nodes, blocks_per_axis=3),
                    ),
                    (
                        "local_truncation_k49",
                        lambda: local_truncation_preconditioner(system.A, system.nodes, neighbors=49),
                    ),
                ]
            )
            if config.node_count <= (225 if quick else 400):
                builders.append(
                    (
                        "global_residual_lsai_k25",
                        lambda: global_residual_sparse_approximate_inverse(system.A, system.nodes, support_size=25),
                    )
                )
        else:
            builders.append(("jacobi", lambda: jacobi_preconditioner(system.A)))
        builders.append(("pivoted_lu_fp32", lambda: low_precision_lu_preconditioner(system.A)))

        for method_name, builder in builders:
            try:
                preconditioner = builder()
                result, median_seconds = solver_run(
                    system,
                    preconditioner,
                    system.b,
                    repeats=repeats,
                    capture=True,
                )
                row = config_fields(config)
                row.update(
                    {
                        "method": method_name,
                        "solver_type": "right-preconditioned FGMRES",
                        "setup_seconds": preconditioner.setup_seconds,
                        "application_seconds": preconditioner.benchmark_application(system.dimension, repeats=40),
                        "preconditioner_memory_bytes": preconditioner.memory_bytes,
                        "matrix_plus_preconditioner_bytes": system.A.nbytes + preconditioner.memory_bytes,
                        "gmres_iterations": result.iterations,
                        "converged": result.converged,
                        "relative_residual": result.relative_residual,
                        "solve_seconds_median": median_seconds,
                        "total_seconds_first_rhs": preconditioner.setup_seconds + median_seconds,
                        "coefficient_rel_error_vs_fp64_lu": float(
                            np.linalg.norm(result.x - reference) / np.linalg.norm(reference)
                        ),
                        "notes": preconditioner.notes,
                    }
                )
                rows.append(row)
                for point, residual in enumerate(result.residual_history):
                    histories.append(
                        {
                            "config_id": config.config_id,
                            "method": method_name,
                            "history_point": point,
                            "relative_residual": residual,
                        }
                    )
            except Exception as exc:
                failures.append(
                    {**config_fields(config), "stage": "classical_preconditioner", "method": method_name, "error": repr(exc)}
                )

        try:
            direct = exact_lu_preconditioner(system.A)
            start = perf_counter()
            x = direct.apply(system.b)
            solve_seconds = perf_counter() - start
            residual = np.linalg.norm(system.b - system.A @ x) / np.linalg.norm(system.b)
            row = config_fields(config)
            row.update(
                {
                    "method": "pivoted_lu_fp64_direct",
                    "solver_type": "direct dense solve",
                    "setup_seconds": direct.setup_seconds,
                    "application_seconds": direct.benchmark_application(system.dimension, repeats=40),
                    "preconditioner_memory_bytes": direct.memory_bytes,
                    "matrix_plus_preconditioner_bytes": system.A.nbytes + direct.memory_bytes,
                    "gmres_iterations": 0,
                    "converged": residual <= RTOL,
                    "relative_residual": residual,
                    "solve_seconds_median": solve_seconds,
                    "total_seconds_first_rhs": direct.setup_seconds + solve_seconds,
                    "coefficient_rel_error_vs_fp64_lu": float(np.linalg.norm(x - reference) / np.linalg.norm(reference)),
                    "notes": direct.notes,
                }
            )
            rows.append(row)
        except Exception as exc:
            failures.append({**config_fields(config), "stage": "direct_ceiling", "error": repr(exc)})
    return rows, histories


def related_trajectory_residuals(config: SystemConfig, scales: tuple[float, ...]) -> list[np.ndarray]:
    residuals: list[np.ndarray] = []
    if config.epsilon is None:
        return residuals
    for scale in scales:
        related = SystemConfig("related", config.side, config.distribution, config.kernel, config.epsilon * scale)
        try:
            system = assemble(related)
            preconditioner = low_precision_lu_preconditioner(system.A)
            result = fgmres(
                system.A,
                system.b,
                preconditioner.apply,
                rtol=RTOL,
                restart=min(100, system.dimension),
                max_iterations=min(240, 2 * system.dimension),
                capture_residual_vectors=True,
            )
            residuals.extend(result.residual_vectors[:80])
        except Exception:
            continue
    return residuals


def structure_analysis(
    configs: list[SystemConfig],
    failures: list[dict[str, Any]],
    *,
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, StructureArtifact]]:
    metric_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    artifacts: dict[str, StructureArtifact] = {}
    keep = oracle_ids(quick)
    rng = np.random.default_rng(SEED + 91)
    for index, config in enumerate(configs, start=1):
        print(f"[structure {index}/{len(configs)}] {config.config_id}", flush=True)
        try:
            system = assemble(config)
            base = low_precision_lu_preconditioner(system.A)
            trajectory = fgmres(
                system.A,
                system.b,
                base.apply,
                rtol=RTOL,
                restart=min(100, system.dimension),
                max_iterations=min(300, 2 * system.dimension),
                capture_residual_vectors=True,
                capture_arnoldi=True,
            )
            synthetic_count = system.dimension if not quick else min(system.dimension, 80)
            test_count = max(48, system.dimension // 2) if not quick else 40
            synthetic = rng.standard_normal((system.dimension, synthetic_count))
            own_trajectory = trajectory.residual_vectors[:100]
            related = related_trajectory_residuals(config, (0.95, 1.05))
            train_parts: list[np.ndarray] = [synthetic]
            if own_trajectory:
                train_parts.append(np.column_stack(own_trajectory))
            if related:
                train_parts.append(np.column_stack(related))
            train = np.column_stack(train_parts)
            test = rng.standard_normal((system.dimension, test_count))
            start = perf_counter()
            data = build_correction_data(
                system.A,
                base,
                train_residuals=train,
                test_residuals=test,
            )
            snapshot_seconds = perf_counter() - start
            q_values = [q for q in (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128) if q <= system.dimension]
            for metric in pod_metrics(data, q_values):
                metric_rows.append(
                    {
                        **config_fields(config),
                        **metric,
                        "base_method": base.name,
                        "train_samples": data.train_corrections.shape[1],
                        "test_samples": data.test_corrections.shape[1],
                    }
                )
            leading = data.singular_values[0]
            for rank, value in enumerate(data.singular_values, start=1):
                singular_rows.append(
                    {
                        **config_fields(config),
                        "rank": rank,
                        "singular_value": value,
                        "singular_value_normalized": value / leading,
                    }
                )
            exact_test = la.solve(system.A, data.test_residuals, assume_a="gen", check_finite=False)
            correction_ratio = np.linalg.norm(data.test_corrections, axis=0) / np.linalg.norm(exact_test, axis=0)
            summary_rows.append(
                {
                    **config_fields(config),
                    "base_method": base.name,
                    "base_iterations": trajectory.iterations,
                    "base_converged": trajectory.converged,
                    "base_relative_residual": trajectory.relative_residual,
                    "synthetic_train_samples": synthetic_count,
                    "own_trajectory_samples": len(own_trajectory),
                    "related_trajectory_samples": len(related),
                    "test_samples": test_count,
                    "snapshot_and_svd_seconds": snapshot_seconds,
                    "median_correction_over_exact_action": float(np.median(correction_ratio)),
                    "q90_correction_over_exact_action": float(np.quantile(correction_ratio, 0.9)),
                }
            )
            key = (config.kernel, config.distribution, config.side)
            if key in keep:
                artifacts[config.config_id] = StructureArtifact(
                    config,
                    system,
                    base,
                    trajectory,
                    data,
                    snapshot_seconds,
                )
        except Exception as exc:
            failures.append({**config_fields(config), "stage": "structure", "error": repr(exc)})
    return metric_rows, singular_rows, summary_rows, artifacts


def mode_rhs(system: GlobalRBFSystem, kx: int, ky: int) -> np.ndarray:
    x = system.nodes
    u = np.sin(kx * np.pi * x[:, 0]) * np.sin(ky * np.pi * x[:, 1])
    f = (kx**2 + ky**2) * np.pi**2 * u
    rhs = np.where(system.boundary, u, f)
    if system.dimension > system.node_count:
        rhs = np.concatenate([rhs, np.zeros(system.dimension - system.node_count)])
    return rhs


def oracle_analysis(
    artifacts: dict[str, StructureArtifact],
    failures: list[dict[str, Any]],
    *,
    quick: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    q_values = [4, 8] if quick else [4, 8, 12, 16, 24, 32]
    rhs_specs = [("mode_1_1", 1, 1)] if quick else [
        ("mode_1_1", 1, 1),
        ("mode_1_2", 1, 2),
        ("mode_2_1", 2, 1),
        ("mode_2_2", 2, 2),
    ]
    for index, artifact in enumerate(artifacts.values(), start=1):
        config, system, base, trajectory, data = (
            artifact.config,
            artifact.system,
            artifact.base,
            artifact.trajectory,
            artifact.data,
        )
        print(f"[oracle {index}/{len(artifacts)}] {config.config_id}", flush=True)
        max_q = min(max(q_values), system.dimension)
        candidates: dict[str, tuple[np.ndarray, float, int]] = {
            "correction_pod": (data.pod_basis[:, :max_q], artifact.snapshot_setup_seconds, data.train_corrections.nbytes),
        }
        start = perf_counter()
        residual_space = residual_history_space(trajectory.residual_vectors, base, max_q)
        candidates["preconditioned_residual_pod"] = (
            residual_space,
            perf_counter() - start,
            sum(r.nbytes for r in trajectory.residual_vectors),
        )
        try:
            start = perf_counter()
            harmonic = harmonic_ritz_space(trajectory, max_q)
            candidates["harmonic_ritz"] = (
                harmonic,
                perf_counter() - start,
                (trajectory.preconditioned_vectors.nbytes if trajectory.preconditioned_vectors is not None else 0),
            )
        except Exception as exc:
            failures.append({**config_fields(config), "stage": "harmonic_ritz_basis", "error": repr(exc)})
        start = perf_counter()
        singular = smallest_right_singular_space(system.A, max_q)
        candidates["smallest_right_singular"] = (singular, perf_counter() - start, system.A.nbytes)

        rhs_map = {name: mode_rhs(system, kx, ky) for name, kx, ky in rhs_specs}
        for rhs_name, b in rhs_map.items():
            base_result, base_seconds = solver_run(system, base, b, repeats=1)
            rows.append(
                {
                    **config_fields(config),
                    "rhs": rhs_name,
                    "space": "classical_only",
                    "q_requested": 0,
                    "q_actual": 0,
                    "basis_setup_seconds": 0.0,
                    "basis_source_memory_bytes": 0,
                    "coarse_setup_seconds": 0.0,
                    "coarse_memory_bytes": 0,
                    "gmres_iterations": base_result.iterations,
                    "converged": base_result.converged,
                    "relative_residual": base_result.relative_residual,
                    "solve_seconds": base_seconds,
                    "dense_matvec_equivalent": base_result.iterations,
                    "total_seconds_first_rhs": base_seconds,
                    "composition": "classical only",
                }
            )
        for space_name, (basis, basis_setup, source_memory) in candidates.items():
            for q in q_values:
                q_actual = min(q, basis.shape[1])
                if q_actual == 0:
                    continue
                try:
                    hybrid = oracle_coarse_preconditioner(
                        system.A,
                        base,
                        basis[:, :q_actual],
                        name=f"{space_name}_q{q_actual}",
                        composition="residual_complement",
                    )
                    coarse_setup = hybrid.setup_seconds - base.setup_seconds
                    coarse_memory = hybrid.memory_bytes - base.memory_bytes
                    for rhs_name, b in rhs_map.items():
                        result, seconds = solver_run(system, hybrid, b, repeats=1)
                        rows.append(
                            {
                                **config_fields(config),
                                "rhs": rhs_name,
                                "space": space_name,
                                "q_requested": q,
                                "q_actual": q_actual,
                                "basis_setup_seconds": basis_setup,
                                "basis_source_memory_bytes": source_memory,
                                "coarse_setup_seconds": coarse_setup,
                                "coarse_memory_bytes": coarse_memory,
                                "gmres_iterations": result.iterations,
                                "converged": result.converged,
                                "relative_residual": result.relative_residual,
                                "solve_seconds": seconds,
                                "dense_matvec_equivalent": 2 * result.iterations,
                                "total_seconds_first_rhs": basis_setup + coarse_setup + seconds,
                                "composition": "classical plus residual-complement Petrov-Galerkin",
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            **config_fields(config),
                            "stage": "oracle",
                            "space": space_name,
                            "q": q,
                            "error": repr(exc),
                        }
                    )
    return rows


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["light"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_blossom(fig: plt.Figure) -> None:
    center_x, center_y = 0.975, 0.965
    color = PALETTE["blue"]
    for dx, dy in ((-0.008, 0), (0.008, 0), (0, -0.012), (0, 0.012)):
        fig.add_artist(Circle((center_x + dx, center_y + dy), 0.005, transform=fig.transFigure, color=color, alpha=0.85))


def finish_figure(fig: plt.Figure, path: Path) -> None:
    add_blossom(fig)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(output: Path) -> list[dict[str, Any]]:
    setup_plot_style()
    chart_map: list[dict[str, Any]] = []
    systems = read_csv(output / "system_characterization.csv")
    classical = read_csv(output / "classical_baselines.csv")
    pod = read_csv(output / "correction_pod_metrics.csv")
    singular = read_csv(output / "correction_singular_values.csv")
    oracle = read_csv(output / "oracle_results.csv")

    core = [r for r in systems if r["study"] == "core"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharex=True, sharey=True)
    for ax, kernel in zip(axes, ("gaussian", "imq", "phs3")):
        for distribution, style, color in (("tensor", "-", PALETTE["blue"]), ("jittered", "--", PALETTE["orange"])):
            selected = sorted(
                (r for r in core if r["kernel"] == kernel and r["distribution"] == distribution),
                key=lambda r: int(r["N"]),
            )
            ax.plot(
                [int(r["N"]) for r in selected],
                [float(r["grid_rel_l2"]) for r in selected],
                style,
                color=color,
                marker="o" if distribution == "tensor" else "s",
                markerfacecolor="white" if distribution == "jittered" else color,
                label=distribution,
            )
        ax.set_title(kernel.upper() if kernel != "phs3" else "PHS $r^3$ + degree-2 polynomials")
        ax.set_xlabel("RBF centers / collocation nodes, N")
        ax.set_yscale("log")
    axes[0].set_ylabel("Relative $L^2$ error on 41×41 grid")
    axes[0].legend(loc="best")
    fig.suptitle("Global RBF Poisson discretization convergence", x=0.05, ha="left", fontweight="bold")
    fig.text(0.05, 0.905, "Unit square; manufactured $u=\\sin(\\pi x)\\sin(\\pi y)$; scaled-ε core path", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    path = output / "discretization_convergence.png"
    finish_figure(fig, path)
    chart_map.append({"section": "correctness", "question": "Does the method converge with N?", "family": "trend", "fields": "N, grid_rel_l2, kernel, distribution", "claim": "All three supported kernel paths converge on the resolved core sequence.", "palette": "hard two-root cap", "path": path.name})

    shape = [r for r in systems if r["study"] == "shape" and r["distribution"] == "tensor"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex="col")
    for col, kernel in enumerate(("gaussian", "imq")):
        for side, color, marker in ((15, PALETTE["blue"], "o"), (20, PALETTE["orange"], "s")):
            selected = sorted(
                (r for r in shape if r["kernel"] == kernel and int(r["side"]) == side),
                key=lambda r: float(r["epsilon"]),
            )
            if not selected:
                continue
            label = f"N={side**2}"
            axes[0, col].plot([float(r["epsilon"]) for r in selected], [float(r["grid_rel_l2"]) for r in selected], color=color, marker=marker, label=label)
            axes[1, col].plot([float(r["epsilon"]) for r in selected], [float(r["condition_2"]) for r in selected], color=color, marker=marker, label=label)
        axes[0, col].set_title(kernel.upper())
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Shape parameter ε")
        axes[0, col].legend(loc="best")
    axes[0, 0].set_ylabel("Relative grid $L^2$ error")
    axes[1, 0].set_ylabel("2-norm condition number")
    fig.suptitle("Shape-parameter sensitivity of the native global basis", x=0.07, ha="left", fontweight="bold")
    fig.text(0.07, 0.935, "Tensor nodes; flat kernels are accurate before native-basis roundoff dominates", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    path = output / "shape_parameter_sensitivity.png"
    finish_figure(fig, path)
    chart_map.append({"section": "conditioning", "question": "How does ε trade accuracy against conditioning?", "family": "trend + uncertainty", "fields": "epsilon, grid_rel_l2, condition_2, N, kernel", "claim": "The most accurate flat cases can be unusable as linear-algebra references.", "palette": "hard two-root cap", "path": path.name})

    iterative = [r for r in classical if r["method"] != "pivoted_lu_fp64_direct" and r["kernel"] in {"gaussian", "imq"} and r["distribution"] == "tensor"]
    methods = ["identity", "jacobi", "block_jacobi_3x3", "local_truncation_k49", "global_residual_lsai_k25", "pivoted_lu_fp32"]
    method_style = {
        "identity": (PALETTE["grey"], "o", ":"),
        "jacobi": (PALETTE["gold"], "s", ":"),
        "block_jacobi_3x3": (PALETTE["olive"], "^", "--"),
        "local_truncation_k49": (PALETTE["pink"], "v", "--"),
        "global_residual_lsai_k25": (PALETTE["orange"], "D", "-."),
        "pivoted_lu_fp32": (PALETTE["blue"], "o", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    for ax, kernel in zip(axes, ("gaussian", "imq")):
        for method in methods:
            selected = sorted((r for r in iterative if r["kernel"] == kernel and r["method"] == method), key=lambda r: int(r["N"]))
            if not selected:
                continue
            color, marker, linestyle = method_style[method]
            ax.plot([int(r["N"]) for r in selected], [int(r["gmres_iterations"]) for r in selected], color=color, marker=marker, linestyle=linestyle, markerfacecolor="white" if method != "pivoted_lu_fp32" else color, label=method.replace("_", " "))
        ax.axhline(400, color=PALETTE["ink"], linewidth=0.8, linestyle=":")
        ax.set_title(kernel.upper())
        ax.set_xlabel("N")
    axes[0].set_ylabel("FGMRES iterations (400 is the run cap)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Classical preconditioner iteration counts", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.91, "Tensor-node core path; tolerance $10^{-10}$; unconverged runs remain at their stopping iteration", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 0.84, 0.88])
    path = output / "classical_preconditioner_iterations.png"
    finish_figure(fig, path)
    chart_map.append({"section": "classical baselines", "question": "Which classical preconditioner is strongest?", "family": "comparison", "fields": "N, method, gmres_iterations, kernel", "claim": "FP32 pivoted LU is the only consistently strong iterative baseline; exact FP64 LU remains the time ceiling.", "palette": "relaxed multi-category", "path": path.name})

    char_lookup = {r["config_id"]: r for r in core}
    points = []
    for row in classical:
        if row["method"] == "pivoted_lu_fp32" and row["config_id"] in char_lookup:
            points.append((row, char_lookup[row["config_id"]]))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for kernel, color in (("gaussian", PALETTE["blue"]), ("imq", PALETTE["orange"]), ("phs3", PALETTE["olive"])):
        for distribution, marker in (("tensor", "o"), ("jittered", "s")):
            selected = [(r, c) for r, c in points if r["kernel"] == kernel and r["distribution"] == distribution]
            if not selected:
                continue
            ax.scatter([float(c["condition_2"]) for r, c in selected], [int(r["gmres_iterations"]) for r, c in selected], color=color, marker=marker, facecolors="none" if distribution == "jittered" else color, label=f"{kernel}, {distribution}")
    ax.set_xscale("log")
    ax.set_xlabel("2-norm condition number")
    ax.set_ylabel("FP32-LU-preconditioned FGMRES iterations")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Conditioning and FP32-LU solver difficulty", x=0.09, ha="left", fontweight="bold")
    fig.text(0.09, 0.91, "Core resolved sequences; marker shape distinguishes node distribution", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 0.79, 0.88])
    path = output / "conditioning_vs_solver_difficulty.png"
    finish_figure(fig, path)
    chart_map.append({"section": "conditioning", "question": "When does the global system become hard?", "family": "relationship", "fields": "condition_2, gmres_iterations, kernel, distribution", "claim": "Solver work rises sharply once FP32 factorization error is amplified by conditioning.", "palette": "relaxed multi-category", "path": path.name})

    available_ids = {r["config_id"] for r in pod}
    desired_specs = [
        ("gaussian", "tensor", 225),
        ("gaussian", "tensor", 400),
        ("gaussian", "jittered", 400),
        ("imq", "tensor", 225),
        ("imq", "tensor", 400),
        ("imq", "jittered", 400),
    ]
    chosen_ids = []
    for kernel, distribution, N in desired_specs:
        matches = sorted(
            config_id
            for config_id in available_ids
            if f"_{kernel}_{distribution}_N{N}_" in config_id
        )
        if matches:
            chosen_ids.append(matches[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["olive"], PALETTE["pink"], PALETTE["gold"], PALETTE["ink"]]
    for config_id, color in zip(chosen_ids, colors):
        selected = sorted((r for r in singular if r["config_id"] == config_id and int(r["rank"]) <= 80), key=lambda r: int(r["rank"]))
        label = config_id.replace("structure_", "").replace("_eps", ", ε=").replace("_N", ", N=")
        ax.plot([int(r["rank"]) for r in selected], [float(r["singular_value_normalized"]) for r in selected], color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Correction snapshot singular-value rank")
    ax.set_ylabel("$\sigma_i / \sigma_1$")
    ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Singular-value decay of the missing FP32-LU inverse action", x=0.08, ha="left", fontweight="bold")
    fig.text(0.08, 0.91, "Normalized residual inputs; synthetic, own-GMRES, and neighboring-ε samples", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = output / "correction_singular_value_decay.png"
    finish_figure(fig, path)
    chart_map.append({"section": "missing inverse action", "question": "Is the unresolved correction low dimensional?", "family": "distribution/rank decay", "fields": "rank, normalized singular value, configuration", "claim": "Decay exists, but strength varies materially with N, kernel, and node distribution.", "palette": "relaxed multi-category", "path": path.name})

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for config_id, color in zip(chosen_ids, colors):
        selected = sorted((r for r in pod if r["config_id"] == config_id), key=lambda r: float(r["q_over_n"]))
        label = config_id.replace("structure_", "").replace("_eps", ", ε=").replace("_N", ", N=")
        ax.plot([float(r["q_over_n"]) for r in selected], [float(r["heldout_projection_error_median"]) for r in selected], color=color, marker="o", markersize=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Retained dimension q / system dimension")
    ax.set_ylabel("Median held-out relative projection error")
    ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Held-out correction projection error", x=0.08, ha="left", fontweight="bold")
    fig.text(0.08, 0.91, "Test residuals are independent Gaussian directions; no training-column reuse", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = output / "heldout_projection_error.png"
    finish_figure(fig, path)
    chart_map.append({"section": "missing inverse action", "question": "Does POD generalize to held-out residual directions?", "family": "trend", "fields": "q_over_n, heldout_projection_error_median, configuration", "claim": "Small q can work in favorable cases, but error is not uniformly negligible across systems.", "palette": "relaxed multi-category", "path": path.name})

    oracle_nonbase = [r for r in oracle if r["space"] != "classical_only" and r["converged"] == "True"]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in oracle_nonbase:
        grouped[(row["space"], int(row["q_requested"]))].append(float(row["gmres_iterations"]))
    base_values = [float(r["gmres_iterations"]) for r in oracle if r["space"] == "classical_only"]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    styles = {
        "correction_pod": (PALETTE["blue"], "o"),
        "preconditioned_residual_pod": (PALETTE["orange"], "s"),
        "harmonic_ritz": (PALETTE["olive"], "^"),
        "smallest_right_singular": (PALETTE["pink"], "D"),
    }
    for space, (color, marker) in styles.items():
        qs = sorted({q for candidate, q in grouped if candidate == space})
        if not qs:
            continue
        ax.plot(qs, [np.median(grouped[(space, q)]) for q in qs], color=color, marker=marker, label=space.replace("_", " "))
    if base_values:
        ax.axhline(np.median(base_values), color=PALETTE["ink"], linestyle="--", label="classical FP32 LU median")
    ax.set_xlabel("Requested oracle dimension q")
    ax.set_ylabel("Median FGMRES iterations across systems and RHSs")
    ax.legend(loc="best", fontsize=8)
    fig.suptitle("Oracle difficult-space iteration sensitivity", x=0.08, ha="left", fontweight="bold")
    fig.text(0.08, 0.91, "System-own spaces; residual-complement Petrov–Galerkin correction; four smooth RHSs", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = output / "oracle_iteration_sensitivity.png"
    finish_figure(fig, path)
    chart_map.append({"section": "oracle correction", "question": "Which difficult space is solver relevant?", "family": "comparison + benchmark", "fields": "q, median iterations, space", "claim": "The q=32 smallest-right-singular oracle is strongest; correction POD helps at larger q, while residual POD is inconsistent.", "palette": "relaxed multi-category", "path": path.name})
    return chart_map


def write_environment(output: Path, quick: bool) -> None:
    payload = {
        "generated_at_local": "2026-08-30",
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "quick": quick,
        "seed": SEED,
        "rtol": RTOL,
        "blas_thread_env": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")
        },
    }
    (output / "software_environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/global_rbf_stage1"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_environment(args.output, args.quick)
    config_payload = {
        "core": [asdict(c) for c in core_configs(args.quick)],
        "shape": [asdict(c) for c in shape_configs(args.quick)],
        "structure": [asdict(c) for c in structure_configs(args.quick)],
        "oracle_systems": sorted(list(oracle_ids(args.quick))),
        "rtol": RTOL,
        "max_iterations": "min(400, 2*dimension)",
        "restart": "min(100, dimension)",
        "seed": SEED,
        "manufactured_solution": "sin(pi*x)*sin(pi*y)",
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    failures: list[dict[str, Any]] = []
    systems = characterize(core_configs(args.quick) + shape_configs(args.quick), failures)
    write_csv(args.output / "system_characterization.csv", systems)
    classical, histories = classical_baselines(core_configs(args.quick), failures, quick=args.quick)
    write_csv(args.output / "classical_baselines.csv", classical)
    write_csv(args.output / "classical_residual_histories.csv", histories)
    metrics, singular, summaries, artifacts = structure_analysis(structure_configs(args.quick), failures, quick=args.quick)
    write_csv(args.output / "correction_pod_metrics.csv", metrics)
    write_csv(args.output / "correction_singular_values.csv", singular)
    write_csv(args.output / "correction_structure_summary.csv", summaries)
    oracle = oracle_analysis(artifacts, failures, quick=args.quick)
    write_csv(args.output / "oracle_results.csv", oracle)
    write_csv(args.output / "failures.csv", failures)
    chart_map = plot_results(args.output)
    write_csv(args.output / "chart_map.csv", chart_map)
    print(f"Wrote Stage-1 evidence to {args.output}", flush=True)
    print(f"Failures recorded: {len(failures)} ({Counter(row.get('stage') for row in failures)})", flush=True)


if __name__ == "__main__":
    main()
