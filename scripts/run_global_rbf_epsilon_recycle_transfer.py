#!/usr/bin/env python3
"""Transfer q=16 harmonic-Ritz recycle spaces across nearby RBF epsilons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import (
    GmresResult,
    build_augmentation,
    restarted_projected_gmres,
)


EPSILONS = (7.6, 7.8, 8.0, 8.2, 8.4)
SIDE = 20
DIMENSION = SIDE**2
Q = 16
LS_SUPPORT_SIZE = 81
RTOL = 1.0e-10
ATOL = 0.0
RESTART = 100
# Epsilon 7.6 needs 2,493 iterations with the fixed restart/tolerance/initial
# guess.  This is only a full-convergence safety cap; all mathematical solver
# settings and the selected N=400 discretization remain unchanged.
MAX_ITERATIONS = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_epsilon_recycle_transfer"),
    )
    return parser.parse_args()


def _epsilon_key(epsilon: float) -> str:
    return f"eps_{epsilon:.1f}".replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _solve_repeated(
    matrix: np.ndarray,
    rhs: np.ndarray,
    apply_preconditioner,
    *,
    u_basis: np.ndarray | None,
    c_basis: np.ndarray | None,
    repeats: int,
    warmup: bool,
) -> tuple[GmresResult, float, list[float]]:
    kwargs = {
        "u_basis": u_basis,
        "c_basis": c_basis,
        "x0": np.zeros(matrix.shape[0]),
        "rtol": RTOL,
        "atol": ATOL,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
    }
    if warmup:
        restarted_projected_gmres(matrix, rhs, apply_preconditioner, **kwargs)
    runs = [
        restarted_projected_gmres(matrix, rhs, apply_preconditioner, **kwargs)
        for _ in range(repeats)
    ]
    if len({run.iterations for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagree on iteration count")
    if len({run.converged for run in runs}) != 1:
        raise RuntimeError("deterministic repeats disagree on convergence")
    if np.ptp([run.relative_residual for run in runs]) > 1.0e-13:
        raise RuntimeError("deterministic repeats disagree on final residual")
    times = [float(run.solve_seconds) for run in runs]
    return runs[0], float(np.median(times)), times


def _augmentation_diagnostics(
    matrix: np.ndarray,
    u_basis: np.ndarray,
    c_basis: np.ndarray,
) -> tuple[float, float]:
    q = c_basis.shape[1]
    return (
        float(np.linalg.norm(matrix @ u_basis - c_basis, ord="fro")),
        float(np.linalg.norm(c_basis.T @ c_basis - np.eye(q), ord="fro")),
    )


def _subspace_metrics(
    source_basis: np.ndarray,
    target_basis: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    angles = np.sort(la.subspace_angles(source_basis, target_basis))[::-1]
    degrees = np.rad2deg(angles)
    projector_frobenius = float(np.sqrt(2.0 * np.sum(np.sin(angles) ** 2)))
    normalized = projector_frobenius / np.sqrt(2.0 * source_basis.shape[1])
    return (
        degrees,
        float(np.max(degrees)),
        float(np.median(degrees)),
        projector_frobenius,
        float(normalized),
    )


def _record_histories(
    estimated_rows: list[dict[str, Any]],
    true_rows: list[dict[str, Any]],
    *,
    solve_id: str,
    role: str,
    source_epsilon: float | None,
    target_epsilon: float,
    result: GmresResult,
) -> None:
    estimated_rows.extend(
        {
            "solve_id": solve_id,
            "role": role,
            "source_epsilon": source_epsilon,
            "target_epsilon": target_epsilon,
            "history_index": index,
            "relative_residual_estimate_or_restart_true_residual": value,
        }
        for index, value in enumerate(result.residual_history)
    )
    true_rows.extend(
        {
            "solve_id": solve_id,
            "role": role,
            "source_epsilon": source_epsilon,
            "target_epsilon": target_epsilon,
            "iteration": iteration,
            "true_relative_residual": value,
        }
        for iteration, value in zip(
            result.true_residual_iterations,
            result.true_residual_history,
        )
    )


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _matrix_rows(
    pair_rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in EPSILONS:
        output: dict[str, Any] = {"epsilon_source": source}
        for target in EPSILONS:
            pair = next(
                row
                for row in pair_rows
                if row["epsilon_source"] == source
                and row["epsilon_target"] == target
            )
            output[f"target_{target:.1f}"] = pair[field]
        rows.append(output)
    return rows


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    cmap,
    value_format: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(EPSILONS)), [f"{value:.1f}" for value in EPSILONS])
    ax.set_yticks(range(len(EPSILONS)), [f"{value:.1f}" for value in EPSILONS])
    ax.set_xlabel("Target epsilon")
    ax.set_ylabel("Source epsilon")
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold")
    threshold = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value < threshold else "#20242a",
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def _make_plots(output_dir: Path, pair_rows: list[dict[str, Any]]) -> None:
    blue_low = LinearSegmentedColormap.from_list(
        "blue_low",
        ["#214f78", "#78a8cf", "#f6f8fa"],
    )
    orange_low = LinearSegmentedColormap.from_list(
        "orange_low",
        ["#9a541f", "#e2a66e", "#f9f7f4"],
    )
    ratio_matrix = np.asarray(
        [
            [
                next(
                    row
                    for row in pair_rows
                    if row["epsilon_source"] == source
                    and row["epsilon_target"] == target
                )["transferred_baseline_iteration_ratio"]
                for target in EPSILONS
            ]
            for source in EPSILONS
        ],
        dtype=float,
    )
    largest_angle_matrix = np.asarray(
        [
            [
                next(
                    row
                    for row in pair_rows
                    if row["epsilon_source"] == source
                    and row["epsilon_target"] == target
                )["largest_principal_angle_degrees"]
                for target in EPSILONS
            ]
            for source in EPSILONS
        ],
        dtype=float,
    )
    projector_matrix = np.asarray(
        [
            [
                next(
                    row
                    for row in pair_rows
                    if row["epsilon_source"] == source
                    and row["epsilon_target"] == target
                )["projector_distance_frobenius"]
                for target in EPSILONS
            ]
            for source in EPSILONS
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
    _annotated_heatmap(
        ax,
        ratio_matrix,
        title="Transferred recycle iteration ratio",
        colorbar_label="Transferred iterations / baseline iterations",
        cmap=blue_low,
        value_format=".3f",
        vmin=float(np.nanmin(ratio_matrix)),
        vmax=max(1.0, float(np.nanmax(ratio_matrix))),
    )
    fig.suptitle(
        "Global Gaussian-RBF epsilon transfer, N=400, q=16",
        x=0.08,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#20242a",
    )
    fig.savefig(output_dir / "transferred_iteration_ratio_heatmap.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    _annotated_heatmap(
        axes[0],
        largest_angle_matrix,
        title="Largest principal angle",
        colorbar_label="Degrees",
        cmap=orange_low,
        value_format=".1f",
        vmin=0.0,
        vmax=max(float(np.nanmax(largest_angle_matrix)), 1.0),
    )
    _annotated_heatmap(
        axes[1],
        projector_matrix,
        title="Projector distance",
        colorbar_label="Frobenius norm",
        cmap=orange_low,
        value_format=".2f",
        vmin=0.0,
        vmax=max(float(np.nanmax(projector_matrix)), 0.1),
    )
    fig.suptitle(
        "Source-to-target harmonic-Ritz subspace similarity",
        x=0.04,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#20242a",
    )
    fig.savefig(output_dir / "subspace_similarity_heatmaps.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    source_shades = ["#1f4e79", "#356c9b", "#5589b2", "#78a6ca", "#a1c1da"]
    markers = ["o", "s", "^", "D", "P"]
    for color, marker, source in zip(source_shades, markers, EPSILONS):
        selected = [row for row in pair_rows if row["epsilon_source"] == source]
        ax.scatter(
            [row["epsilon_distance"] for row in selected],
            [row["transferred_baseline_iteration_ratio"] for row in selected],
            label=f"source {source:.1f}",
            color=color,
            edgecolor="#20242a",
            linewidth=0.5,
            marker=marker,
            s=52,
            zorder=3,
        )
    distance_values = sorted({row["epsilon_distance"] for row in pair_rows})
    medians = [
        float(
            np.median(
                [
                    row["transferred_baseline_iteration_ratio"]
                    for row in pair_rows
                    if np.isclose(row["epsilon_distance"], distance)
                ]
            )
        )
        for distance in distance_values
    ]
    ax.plot(
        distance_values,
        medians,
        color="#d27a2c",
        marker="o",
        linewidth=2.0,
        label="distance-group median",
        zorder=4,
    )
    for level, style, label in (
        (0.85, "--", "strong threshold 0.85"),
        (0.95, ":", "useful threshold 0.95"),
        (1.00, "-.", "baseline 1.00"),
    ):
        ax.axhline(level, color="#555b63", linestyle=style, linewidth=1.0, label=label)
    ax.set_xlabel("Absolute epsilon distance")
    ax.set_ylabel("Transferred iterations / baseline iterations")
    ax.set_title(
        "Iteration ratio versus epsilon distance\n"
        "25 ordered source-target pairs; lower is better",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        linespacing=1.35,
    )
    ax.set_xticks(distance_values)
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="best")
    fig.savefig(output_dir / "iteration_ratio_vs_epsilon_distance.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random_basis, _ = la.qr(
        np.random.default_rng(args.random_seed).standard_normal((DIMENSION, Q)),
        mode="economic",
        check_finite=False,
    )
    systems: dict[float, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    true_histories: list[dict[str, Any]] = []
    solver_diagnostics: list[dict[str, Any]] = []
    source_arrays: dict[str, np.ndarray] = {"matched_random_Y": random_basis}
    pair_arrays: dict[str, np.ndarray] = {}
    cycle_arrays: dict[str, np.ndarray] = {}
    cycle_index_rows: list[dict[str, Any]] = []

    for epsilon in EPSILONS:
        key = _epsilon_key(epsilon)
        print(f"[source epsilon={epsilon:.1f}] assembling and extracting", flush=True)
        system = assemble_tensor_gaussian_poisson_blocked(SIDE, epsilon=epsilon)
        preconditioner = rbf_local_cardinal_preconditioner(
            system.A,
            system.nodes,
            support_size=LS_SUPPORT_SIZE,
        )
        extraction_start = perf_counter()
        trajectory = fgmres(
            system.A,
            system.b,
            preconditioner.apply,
            x0=np.zeros(DIMENSION),
            rtol=RTOL,
            atol=ATOL,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
            capture_arnoldi=True,
        )
        extraction_seconds = perf_counter() - extraction_start
        if not trajectory.converged:
            raise RuntimeError(
                f"source epsilon={epsilon:.1f} did not converge within validated settings"
            )
        if not trajectory.arnoldi_cycles:
            raise RuntimeError(f"source epsilon={epsilon:.1f} has no saved Arnoldi cycles")
        construction_start = perf_counter()
        oracle_basis = harmonic_ritz_space(trajectory, Q)
        construction_seconds = perf_counter() - construction_start
        if oracle_basis.shape != (DIMENSION, Q):
            raise RuntimeError(
                f"epsilon={epsilon:.1f} harmonic space has shape {oracle_basis.shape}"
            )
        own_u, own_c = build_augmentation(system.A, oracle_basis)
        own_au_defect, own_ctc_defect = _augmentation_diagnostics(
            system.A,
            own_u,
            own_c,
        )
        source_arrays[f"{key}_Y"] = oracle_basis
        source_arrays[f"{key}_U"] = own_u
        source_arrays[f"{key}_C"] = own_c
        source_arrays[f"{key}_extraction_solution"] = trajectory.x
        for cycle_index, cycle in enumerate(trajectory.arnoldi_cycles):
            prefix = f"{key}_cycle_{cycle_index:02d}"
            cycle_arrays[f"{prefix}_V"] = cycle.arnoldi_vectors
            cycle_arrays[f"{prefix}_Z"] = cycle.preconditioned_vectors
            cycle_arrays[f"{prefix}_H"] = cycle.hessenberg
            cycle_index_rows.append(
                {
                    "epsilon": epsilon,
                    "cycle_index": cycle_index,
                    "first_iteration": cycle.first_iteration,
                    "krylov_dimension": cycle.hessenberg.shape[1],
                    "V_key": f"{prefix}_V",
                    "Z_key": f"{prefix}_Z",
                    "H_key": f"{prefix}_H",
                }
            )
        source_rows.append(
            {
                "epsilon": epsilon,
                "converged": trajectory.converged,
                "iterations": trajectory.iterations,
                "true_relative_residual": trajectory.relative_residual,
                "ls_acbf_setup_seconds": preconditioner.setup_seconds,
                "baseline_extraction_seconds_offline": extraction_seconds,
                "harmonic_construction_seconds_offline": construction_seconds,
                "restart_cycles": len(trajectory.arnoldi_cycles),
                "last_cycle_dimension": trajectory.arnoldi_cycles[-1].hessenberg.shape[1],
                "own_AU_minus_C_frobenius": own_au_defect,
                "own_CHC_minus_I_frobenius": own_ctc_defect,
            }
        )
        histories.extend(
            {
                "solve_id": f"{key}_source_extraction",
                "role": "source_extraction",
                "source_epsilon": epsilon,
                "target_epsilon": epsilon,
                "history_index": index,
                "relative_residual_estimate_or_restart_true_residual": value,
            }
            for index, value in enumerate(trajectory.residual_history)
        )
        systems[epsilon] = {
            "system": system,
            "preconditioner": preconditioner,
            "trajectory": trajectory,
            "oracle_basis": oracle_basis,
        }

    np.savez_compressed(args.output_dir / "source_recycle_spaces.npz", **source_arrays)
    np.savez_compressed(args.output_dir / "source_krylov_cycles.npz", **cycle_arrays)
    _write_csv(args.output_dir / "source_krylov_cycle_index.csv", cycle_index_rows)
    _write_csv(args.output_dir / "source_extractions.csv", source_rows)

    target_controls: dict[float, dict[str, Any]] = {}
    for epsilon in EPSILONS:
        print(f"[target epsilon={epsilon:.1f}] baseline/oracle/random", flush=True)
        data = systems[epsilon]
        system = data["system"]
        preconditioner = data["preconditioner"]
        oracle_basis = data["oracle_basis"]
        rhs_norm = float(np.linalg.norm(system.b))

        baseline_result, baseline_time, baseline_repeats = _solve_repeated(
            system.A,
            system.b,
            preconditioner.apply,
            u_basis=None,
            c_basis=None,
            repeats=args.repeats,
            warmup=not args.no_warmup,
        )
        if (
            baseline_result.iterations != data["trajectory"].iterations
            or baseline_result.converged != data["trajectory"].converged
        ):
            raise RuntimeError(
                f"target epsilon={epsilon:.1f} baseline disagrees with extraction"
            )
        oracle_u, oracle_c = build_augmentation(system.A, oracle_basis)
        oracle_result, oracle_time, oracle_repeats = _solve_repeated(
            system.A,
            system.b,
            preconditioner.apply,
            u_basis=oracle_u,
            c_basis=oracle_c,
            repeats=args.repeats,
            warmup=not args.no_warmup,
        )
        random_u, random_c = build_augmentation(system.A, random_basis)
        random_result, random_time, random_repeats = _solve_repeated(
            system.A,
            system.b,
            preconditioner.apply,
            u_basis=random_u,
            c_basis=random_c,
            repeats=args.repeats,
            warmup=not args.no_warmup,
        )
        oracle_au, oracle_ctc = _augmentation_diagnostics(
            system.A,
            oracle_u,
            oracle_c,
        )
        random_au, random_ctc = _augmentation_diagnostics(
            system.A,
            random_u,
            random_c,
        )
        true_residuals = {
            "baseline": float(
                np.linalg.norm(system.b - system.A @ baseline_result.x) / rhs_norm
            ),
            "oracle": float(
                np.linalg.norm(system.b - system.A @ oracle_result.x) / rhs_norm
            ),
            "random": float(
                np.linalg.norm(system.b - system.A @ random_result.x) / rhs_norm
            ),
        }
        target_rows.append(
            {
                "epsilon_target": epsilon,
                "baseline_converged": baseline_result.converged,
                "baseline_iterations": baseline_result.iterations,
                "baseline_true_relative_residual": true_residuals["baseline"],
                "baseline_solve_seconds": baseline_time,
                "oracle_converged": oracle_result.converged,
                "oracle_iterations": oracle_result.iterations,
                "oracle_true_relative_residual": true_residuals["oracle"],
                "oracle_solve_seconds": oracle_time,
                "oracle_baseline_iteration_ratio": oracle_result.iterations
                / baseline_result.iterations,
                "random_converged": random_result.converged,
                "random_iterations": random_result.iterations,
                "random_true_relative_residual": true_residuals["random"],
                "random_solve_seconds": random_time,
                "random_baseline_iteration_ratio": random_result.iterations
                / baseline_result.iterations,
                "oracle_AU_minus_C_frobenius": oracle_au,
                "oracle_CHC_minus_I_frobenius": oracle_ctc,
                "random_AU_minus_C_frobenius": random_au,
                "random_CHC_minus_I_frobenius": random_ctc,
            }
        )
        key = _epsilon_key(epsilon)
        pair_arrays[f"{key}_baseline_solution"] = baseline_result.x
        pair_arrays[f"{key}_oracle_U"] = oracle_u
        pair_arrays[f"{key}_oracle_C"] = oracle_c
        pair_arrays[f"{key}_oracle_solution"] = oracle_result.x
        pair_arrays[f"{key}_random_U"] = random_u
        pair_arrays[f"{key}_random_C"] = random_c
        pair_arrays[f"{key}_random_solution"] = random_result.x
        for role, result in (
            ("baseline", baseline_result),
            ("target_oracle", oracle_result),
            ("matched_random", random_result),
        ):
            _record_histories(
                histories,
                true_histories,
                solve_id=f"{key}_{role}",
                role=role,
                source_epsilon=epsilon if role == "target_oracle" else None,
                target_epsilon=epsilon,
                result=result,
            )
        for role, result, times in (
            ("baseline", baseline_result, baseline_repeats),
            ("target_oracle", oracle_result, oracle_repeats),
            ("matched_random", random_result, random_repeats),
        ):
            solver_diagnostics.append(
                {
                    "solve_id": f"{key}_{role}",
                    "role": role,
                    "source_epsilon": epsilon if role == "target_oracle" else None,
                    "target_epsilon": epsilon,
                    "solve_seconds_min": min(times),
                    "solve_seconds_median": float(np.median(times)),
                    "solve_seconds_max": max(times),
                    **(result.diagnostics or {}),
                }
            )
        target_controls[epsilon] = {
            "baseline_result": baseline_result,
            "baseline_time": baseline_time,
            "oracle_result": oracle_result,
            "oracle_time": oracle_time,
            "random_result": random_result,
            "random_time": random_time,
            "true_residuals": true_residuals,
        }

    _write_csv(args.output_dir / "target_controls.csv", target_rows)

    for source in EPSILONS:
        source_basis = systems[source]["oracle_basis"]
        for target in EPSILONS:
            target_data = systems[target]
            system = target_data["system"]
            preconditioner = target_data["preconditioner"]
            target_basis = target_data["oracle_basis"]
            control = target_controls[target]
            print(
                f"[transfer source={source:.1f} -> target={target:.1f}]",
                flush=True,
            )
            augmentation_start = perf_counter()
            transferred_u, transferred_c = build_augmentation(
                system.A,
                source_basis,
            )
            augmentation_seconds = perf_counter() - augmentation_start
            au_defect, ctc_defect = _augmentation_diagnostics(
                system.A,
                transferred_u,
                transferred_c,
            )
            transferred_result, transferred_time, transferred_repeats = _solve_repeated(
                system.A,
                system.b,
                preconditioner.apply,
                u_basis=transferred_u,
                c_basis=transferred_c,
                repeats=args.repeats,
                warmup=not args.no_warmup,
            )
            rhs_norm = float(np.linalg.norm(system.b))
            transferred_true_residual = float(
                np.linalg.norm(system.b - system.A @ transferred_result.x)
                / rhs_norm
            )
            angles, largest_angle, median_angle, projector_distance, normalized_distance = (
                _subspace_metrics(source_basis, target_basis)
            )
            epsilon_distance = round(float(abs(source - target)), 10)
            baseline_result = control["baseline_result"]
            oracle_result = control["oracle_result"]
            random_result = control["random_result"]
            transfer_ratio = transferred_result.iterations / baseline_result.iterations
            if transfer_ratio <= 0.85:
                classification = "strong_positive"
            elif transfer_ratio <= 0.95 and transferred_result.iterations < random_result.iterations:
                classification = "weak_but_useful"
            else:
                classification = "no_useful_transfer"
            pair_row = {
                "epsilon_source": source,
                "epsilon_target": target,
                "epsilon_distance": epsilon_distance,
                "baseline_converged": baseline_result.converged,
                "transferred_converged": transferred_result.converged,
                "oracle_converged": oracle_result.converged,
                "random_converged": random_result.converged,
                "baseline_iterations": baseline_result.iterations,
                "transferred_iterations": transferred_result.iterations,
                "oracle_iterations": oracle_result.iterations,
                "random_iterations": random_result.iterations,
                "transferred_baseline_iteration_ratio": transfer_ratio,
                "oracle_baseline_iteration_ratio": oracle_result.iterations
                / baseline_result.iterations,
                "random_baseline_iteration_ratio": random_result.iterations
                / baseline_result.iterations,
                "transferred_oracle_iteration_ratio": transferred_result.iterations
                / oracle_result.iterations,
                "transferred_random_iteration_ratio": transferred_result.iterations
                / random_result.iterations,
                "baseline_true_relative_residual": control["true_residuals"]["baseline"],
                "transferred_true_relative_residual": transferred_true_residual,
                "oracle_true_relative_residual": control["true_residuals"]["oracle"],
                "random_true_relative_residual": control["true_residuals"]["random"],
                "baseline_solve_seconds": control["baseline_time"],
                "transferred_solve_seconds": transferred_time,
                "oracle_solve_seconds": control["oracle_time"],
                "random_solve_seconds": control["random_time"],
                "transferred_baseline_solve_time_ratio": transferred_time
                / control["baseline_time"],
                "target_augmentation_setup_seconds_offline": augmentation_seconds,
                "target_AU_minus_C_frobenius": au_defect,
                "target_CHC_minus_I_frobenius": ctc_defect,
                "largest_principal_angle_degrees": largest_angle,
                "median_principal_angle_degrees": median_angle,
                "projector_distance_frobenius": projector_distance,
                "projector_distance_normalized": normalized_distance,
                "transfer_classification": classification,
            }
            pair_rows.append(pair_row)
            for angle_index, angle in enumerate(angles):
                angle_rows.append(
                    {
                        "epsilon_source": source,
                        "epsilon_target": target,
                        "epsilon_distance": epsilon_distance,
                        "angle_index_descending": angle_index,
                        "principal_angle_degrees": float(angle),
                    }
                )
            source_key = _epsilon_key(source)
            target_key = _epsilon_key(target)
            prefix = f"source_{source_key}_target_{target_key}"
            pair_arrays[f"{prefix}_U"] = transferred_u
            pair_arrays[f"{prefix}_C"] = transferred_c
            pair_arrays[f"{prefix}_solution"] = transferred_result.x
            _record_histories(
                histories,
                true_histories,
                solve_id=prefix,
                role="transferred",
                source_epsilon=source,
                target_epsilon=target,
                result=transferred_result,
            )
            solver_diagnostics.append(
                {
                    "solve_id": prefix,
                    "role": "transferred",
                    "source_epsilon": source,
                    "target_epsilon": target,
                    "solve_seconds_min": min(transferred_repeats),
                    "solve_seconds_median": transferred_time,
                    "solve_seconds_max": max(transferred_repeats),
                    **(transferred_result.diagnostics or {}),
                }
            )
            print(
                f"  baseline={baseline_result.iterations} "
                f"transfer={transferred_result.iterations} "
                f"oracle={oracle_result.iterations} random={random_result.iterations} "
                f"ratio={transfer_ratio:.3f} angle_max={largest_angle:.1f}deg",
                flush=True,
            )

    _write_csv(args.output_dir / "pair_results.csv", pair_rows)
    _write_csv(args.output_dir / "principal_angles.csv", angle_rows)
    _write_csv(args.output_dir / "residual_histories.csv", histories)
    _write_csv(args.output_dir / "true_residual_histories.csv", true_histories)
    _write_csv(args.output_dir / "solver_diagnostics.csv", solver_diagnostics)
    np.savez_compressed(args.output_dir / "target_and_pair_recycle_arrays.npz", **pair_arrays)

    ratio_matrix_rows = _matrix_rows(pair_rows, "transferred_baseline_iteration_ratio")
    angle_matrix_rows = _matrix_rows(pair_rows, "largest_principal_angle_degrees")
    projector_matrix_rows = _matrix_rows(pair_rows, "projector_distance_frobenius")
    _write_csv(args.output_dir / "transferred_iteration_ratio_matrix.csv", ratio_matrix_rows)
    _write_csv(args.output_dir / "largest_principal_angle_matrix.csv", angle_matrix_rows)
    _write_csv(args.output_dir / "projector_distance_matrix.csv", projector_matrix_rows)

    distance_rows: list[dict[str, Any]] = []
    for distance in sorted({row["epsilon_distance"] for row in pair_rows}):
        group = [
            row
            for row in pair_rows
            if np.isclose(row["epsilon_distance"], distance)
        ]
        ratios = np.asarray(
            [row["transferred_baseline_iteration_ratio"] for row in group]
        )
        distance_rows.append(
            {
                "epsilon_distance": distance,
                "pair_count": len(group),
                "median_transferred_baseline_iteration_ratio": float(np.median(ratios)),
                "minimum_transferred_baseline_iteration_ratio": float(np.min(ratios)),
                "maximum_transferred_baseline_iteration_ratio": float(np.max(ratios)),
                "median_transferred_oracle_iteration_ratio": float(
                    np.median(
                        [row["transferred_oracle_iteration_ratio"] for row in group]
                    )
                ),
                "median_random_baseline_iteration_ratio": float(
                    np.median(
                        [row["random_baseline_iteration_ratio"] for row in group]
                    )
                ),
                "median_largest_principal_angle_degrees": float(
                    np.median(
                        [row["largest_principal_angle_degrees"] for row in group]
                    )
                ),
                "median_projector_distance_frobenius": float(
                    np.median(
                        [row["projector_distance_frobenius"] for row in group]
                    )
                ),
                "strong_positive_count": sum(
                    row["transfer_classification"] == "strong_positive"
                    for row in group
                ),
                "weak_but_useful_count": sum(
                    row["transfer_classification"] == "weak_but_useful"
                    for row in group
                ),
                "no_useful_transfer_count": sum(
                    row["transfer_classification"] == "no_useful_transfer"
                    for row in group
                ),
            }
        )
    _write_csv(args.output_dir / "distance_summary.csv", distance_rows)

    all_distance = np.asarray([row["epsilon_distance"] for row in pair_rows])
    all_ratios = np.asarray(
        [row["transferred_baseline_iteration_ratio"] for row in pair_rows]
    )
    all_angles = np.asarray(
        [row["largest_principal_angle_degrees"] for row in pair_rows]
    )
    all_projectors = np.asarray(
        [row["projector_distance_frobenius"] for row in pair_rows]
    )
    off_diagonal = all_distance > 0.0
    correlations = {
        "epsilon_distance_vs_iteration_ratio_all_pairs": _safe_correlation(
            all_distance,
            all_ratios,
        ),
        "epsilon_distance_vs_iteration_ratio_off_diagonal": _safe_correlation(
            all_distance[off_diagonal],
            all_ratios[off_diagonal],
        ),
        "largest_angle_vs_iteration_ratio_all_pairs": _safe_correlation(
            all_angles,
            all_ratios,
        ),
        "largest_angle_vs_iteration_ratio_off_diagonal": _safe_correlation(
            all_angles[off_diagonal],
            all_ratios[off_diagonal],
        ),
        "projector_distance_vs_iteration_ratio_all_pairs": _safe_correlation(
            all_projectors,
            all_ratios,
        ),
        "projector_distance_vs_iteration_ratio_off_diagonal": _safe_correlation(
            all_projectors[off_diagonal],
            all_ratios[off_diagonal],
        ),
    }

    nearby = [
        row for row in pair_rows if np.isclose(row["epsilon_distance"], 0.2)
    ]
    nearby_strong_fraction = float(
        np.mean(
            [row["transferred_baseline_iteration_ratio"] <= 0.85 for row in nearby]
        )
    )
    nearby_useful_fraction = float(
        np.mean(
            [
                row["transferred_baseline_iteration_ratio"] <= 0.95
                and row["transferred_iterations"] < row["random_iterations"]
                for row in nearby
            ]
        )
    )
    nearby_positive = nearby_useful_fraction >= 0.75
    next_experiment = (
        "Use Grassmann-geodesic interpolation from epsilon 7.8 and 8.2 to "
        "predict the q=16 space at epsilon 8.0, then compare it only with "
        "nearest-neighbor reuse, matched random, and the own-system oracle."
        if nearby_positive
        else "Do not interpolate yet; diagnose harmonic-mode correspondence and crossings across adjacent epsilons."
    )

    _make_plots(args.output_dir, pair_rows)

    configuration = {
        "epsilon_values": list(EPSILONS),
        "N": DIMENSION,
        "side": SIDE,
        "kernel": "global Gaussian RBF",
        "pde": "2D Poisson, exact u=sin(pi x) sin(pi y), Dirichlet boundary rows",
        "nodes": "tensor square grid",
        "right_preconditioner": "LS-ACBF k=81 constructed separately for each epsilon",
        "q": Q,
        "rtol": RTOL,
        "atol": ATOL,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
        "iteration_cap_note": (
            "raised from the epsilon=8 benchmark's 1000-step safety cap to 3000 "
            "because epsilon=7.6 converges in 2493 iterations; restart, tolerance, "
            "initial guess, algorithm, system size, and preconditioner are unchanged"
        ),
        "initial_guess": "zero vector for every solve",
        "timing_repeats": args.repeats,
        "timing_warmup": not args.no_warmup,
        "random_seed": args.random_seed,
        "random_control": (
            "one fixed deterministic coefficient-space random orthonormal q=16 "
            "basis, target-normalized separately and reused across source rows"
        ),
        "transfer_rule": (
            "transfer Y_source as a coefficient-space subspace only; for every "
            "target recompute A_target Y_source=C R and U=Y_source R^-1"
        ),
        "recycle_integration": (
            "existing validated fixed-subspace projected Arnoldi with right "
            "preconditioning and AU=C, C^H C=I"
        ),
        "solve_time_definition": (
            "median warmed online solver time; excludes source extraction, LS-ACBF "
            "setup, random/oracle construction, and target augmentation construction"
        ),
        "A_inverse_used": False,
        "bare_A_invariant_subspaces_used": False,
        "ML_used": False,
    }
    summary = {
        "configuration": configuration,
        "source_extractions": source_rows,
        "target_controls": target_rows,
        "pair_results": pair_rows,
        "distance_summary": distance_rows,
        "correlations": correlations,
        "decision": {
            "nearby_definition": "adjacent grid values with absolute epsilon distance 0.2",
            "nearby_pair_count": len(nearby),
            "nearby_median_iteration_ratio": float(
                np.median(
                    [row["transferred_baseline_iteration_ratio"] for row in nearby]
                )
            ),
            "nearby_strong_positive_fraction": nearby_strong_fraction,
            "nearby_useful_fraction": nearby_useful_fraction,
            "nearby_transfer_positive": nearby_positive,
            "next_experiment": next_experiment,
            "ML_trained": False,
        },
        "saved_artifacts": {
            "source_spaces": "source_recycle_spaces.npz",
            "source_cycles": "source_krylov_cycles.npz",
            "source_cycle_index": "source_krylov_cycle_index.csv",
            "pair_representations_and_solutions": "target_and_pair_recycle_arrays.npz",
            "all_solver_histories": "residual_histories.csv",
            "restart_true_residual_histories": "true_residual_histories.csv",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "chart_contract.md").write_text(
        """# Chart map

- Transfer matrix: annotated 5x5 heatmap of transferred/baseline iteration ratios; source rows and target columns; single-root blue scale; exact values shown.
- Geometry matrix: paired annotated 5x5 heatmaps for largest principal angle and Frobenius projector distance; source rows and target columns; single-root orange scale.
- Distance relationship: 25-pair scatter of absolute epsilon distance versus transferred/baseline iteration ratio, with source epsilon distinguished by blue shade and marker, distance-group median in orange, and 0.85/0.95/1.0 reference lines.
- Final QA surface: exported PNGs at 200 dpi, inspected at native resolution.
""",
        encoding="utf-8",
    )
    print(
        f"nearby median ratio={summary['decision']['nearby_median_iteration_ratio']:.3f} "
        f"strong_fraction={nearby_strong_fraction:.3f} "
        f"useful_fraction={nearby_useful_fraction:.3f}; wrote {args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
