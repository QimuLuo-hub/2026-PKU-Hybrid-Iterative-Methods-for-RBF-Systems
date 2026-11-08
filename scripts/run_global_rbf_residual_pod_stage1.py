#!/usr/bin/env python3
"""Stage-1 diagnostics for normalized true-residual history POD spaces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space


EPSILONS = (7.8, 8.0, 8.2, 8.4)
UNORDERED_ADJACENT = ((7.8, 8.0), (8.0, 8.2), (8.2, 8.4))
Q_VALUES = (4, 8, 12, 16)
SIDE = 20
N = SIDE**2
RTOL = 1.0e-10
ATOL = 0.0
RESTART = 100
MAX_ITERATIONS = 3000
SNAPSHOT_STRIDE = 10
LS_SUPPORT_SIZE = 81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_residual_pod_stage1"),
    )
    parser.add_argument(
        "--previous-harmonic-results",
        type=Path,
        default=Path("results/global_rbf_harmonic_cluster_diagnostic/adjacent_transfer_results.csv"),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(epsilon: float) -> str:
    return f"eps_{epsilon:.1f}".replace(".", "p")


def _sample_true_residuals(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = {
        cycle.first_iteration + cycle.hessenberg.shape[1]
        for cycle in (result.arnoldi_cycles or [])
    }
    selected: dict[int, np.ndarray] = {}
    for iteration, residual in zip(
        result.residual_vector_iterations,
        result.residual_vectors,
    ):
        if iteration % SNAPSHOT_STRIDE == 0 or iteration in boundaries:
            selected[int(iteration)] = np.asarray(residual, dtype=float).copy()
    if result.iterations not in selected:
        raise RuntimeError("final restart boundary is missing from true residual snapshots")
    iterations = np.asarray(sorted(selected), dtype=int)
    residuals = np.column_stack([selected[int(iteration)] for iteration in iterations])
    norms = np.linalg.norm(residuals, axis=0)
    if np.any(norms == 0.0):
        raise RuntimeError("zero residual cannot be normalized")
    normalized = residuals / norms
    return iterations, residuals, normalized


def _subspace_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    angles = np.sort(np.rad2deg(la.subspace_angles(left, right)))[::-1]
    distance = float(np.sqrt(2.0 * np.sum(np.sin(np.deg2rad(angles)) ** 2)))
    return (
        float(np.max(angles)),
        float(np.median(angles)),
        distance / np.sqrt(2.0 * left.shape[1]),
    )


def _persistence_summary(iterations: np.ndarray, values: np.ndarray) -> dict[str, float]:
    count = len(values)
    quartile = max(1, count // 4)
    early = values[:quartile]
    penultimate = values[-2 * quartile : -quartile] if count >= 2 * quartile else values[:quartile]
    late = values[-quartile:]
    half_start = count // 2
    late_iterations = iterations[half_start:].astype(float)
    late_values = values[half_start:]
    normalized_iterations = (late_iterations - late_iterations[0]) / max(
        late_iterations[-1] - late_iterations[0],
        1.0,
    )
    slope = float(np.polyfit(normalized_iterations, late_values, 1)[0]) if len(late_values) > 1 else 0.0
    correlation = stats.spearmanr(late_iterations, late_values)
    return {
        "early_quartile_median": float(np.median(early)),
        "penultimate_quartile_median": float(np.median(penultimate)),
        "late_quartile_median": float(np.median(late)),
        "late_minus_penultimate_median": float(np.median(late) - np.median(penultimate)),
        "late_half_slope_per_normalized_iteration": slope,
        "late_half_spearman_rho": float(correlation.statistic),
        "late_half_spearman_p": float(correlation.pvalue),
    }


def _make_plots(
    output_dir: Path,
    spectra_rows: list[dict[str, Any]],
    persistence_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
) -> None:
    epsilon_colors = {7.8: "#1f4e79", 8.0: "#477da8", 8.2: "#d27a2c", 8.4: "#8b5a2b"}
    q_colors = {4: "#1f4e79", 8: "#477da8", 12: "#d27a2c", 16: "#8b5a2b"}
    markers = {4: "o", 8: "s", 12: "^", 16: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    for epsilon in EPSILONS:
        selected = [row for row in spectra_rows if row["epsilon"] == epsilon]
        axes[0].semilogy(
            [row["mode_index"] for row in selected],
            [row["singular_value"] for row in selected],
            color=epsilon_colors[epsilon],
            linewidth=1.6,
            label=f"epsilon {epsilon:.1f}",
        )
        axes[1].plot(
            [row["mode_index"] for row in selected],
            [row["cumulative_energy"] for row in selected],
            color=epsilon_colors[epsilon],
            linewidth=1.6,
        )
    axes[0].set_title("Normalized residual-history singular values", loc="left", fontweight="semibold")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Singular value")
    axes[0].legend(frameon=False)
    axes[1].set_title("Cumulative residual-history energy", loc="left", fontweight="semibold")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Cumulative energy")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].set_xticks(list(Q_VALUES))
    for axis in axes:
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("True-residual history POD decay, N=400", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "residual_pod_singular_values_and_energy.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharey=True, constrained_layout=True)
    for axis, epsilon in zip(axes.ravel(), EPSILONS):
        for q in Q_VALUES:
            selected = [
                row for row in persistence_rows if row["epsilon"] == epsilon and row["q"] == q
            ]
            axis.plot(
                [row["iteration"] for row in selected],
                [row["rho"] for row in selected],
                color=q_colors[q],
                marker=markers[q],
                markevery=max(1, len(selected) // 8),
                markersize=3.5,
                linewidth=1.2,
                label=f"q={q}",
            )
        axis.set_title(f"epsilon {epsilon:.1f}", loc="left", fontweight="semibold")
        axis.set_xlabel("FGMRES iteration")
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("POD projection energy rho")
    axes[1, 0].set_ylabel("POD projection energy rho")
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle("Persistence of normalized true residuals in own-history POD spaces", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "residual_pod_persistence.png", dpi=200)
    plt.close(fig)

    pair_labels = [f"{left:.1f}-{right:.1f}" for left, right in UNORDERED_ADJACENT]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    width = 0.18
    x = np.arange(len(pair_labels), dtype=float)
    for offset, q in enumerate(Q_VALUES):
        residual_rows = [
            next(
                row for row in geometry_rows
                if row["pair_label"] == pair and row["q"] == q and row["space"] == "residual_pod"
            )
            for pair in pair_labels
        ]
        harmonic_rows = [
            next(
                row for row in geometry_rows
                if row["pair_label"] == pair and row["q"] == q and row["space"] == "harmonic_ritz"
            )
            for pair in pair_labels
        ]
        position = x + (offset - 1.5) * width
        axes[0].bar(
            position,
            [row["normalized_projector_distance"] for row in residual_rows],
            width,
            color=q_colors[q],
            edgecolor="#20242a",
            linewidth=0.5,
            label=f"POD q={q}",
        )
        axes[0].scatter(
            position,
            [row["normalized_projector_distance"] for row in harmonic_rows],
            marker="x",
            color="#20242a",
            s=35,
            zorder=4,
        )
        axes[1].bar(
            position,
            [row["largest_principal_angle_degrees"] for row in residual_rows],
            width,
            color=q_colors[q],
            edgecolor="#20242a",
            linewidth=0.5,
        )
        axes[1].scatter(
            position,
            [row["largest_principal_angle_degrees"] for row in harmonic_rows],
            marker="x",
            color="#20242a",
            s=35,
            zorder=4,
        )
    axes[0].set_title("Normalized projector distance", loc="left", fontweight="semibold")
    axes[1].set_title("Largest principal angle", loc="left", fontweight="semibold")
    axes[0].set_ylabel("Distance; bars=POD, x=harmonic Ritz")
    axes[1].set_ylabel("Degrees; bars=POD, x=harmonic Ritz")
    for axis in axes:
        axis.set_xticks(x, pair_labels)
        axis.set_xlabel("Adjacent epsilon pair")
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Cross-epsilon geometry: residual POD versus harmonic Ritz", fontsize=15, fontweight="bold")
    fig.savefig(output_dir / "residual_pod_vs_harmonic_geometry.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_q16: dict[tuple[float, float], dict[str, str]] = {}
    if args.previous_harmonic_results.exists():
        with args.previous_harmonic_results.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["candidate"] == "q16":
                    pair = tuple(sorted((float(row["epsilon_source"]), float(row["epsilon_target"]))))
                    previous_q16[pair] = row

    systems: dict[float, Any] = {}
    pod_bases: dict[float, dict[int, np.ndarray]] = {}
    harmonic_bases: dict[float, dict[int, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    baseline_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
    persistence_summary_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []

    for epsilon in EPSILONS:
        print(f"[stage1 epsilon={epsilon:.1f}] baseline and true snapshots", flush=True)
        system = assemble_tensor_gaussian_poisson_blocked(SIDE, epsilon=epsilon)
        preconditioner = rbf_local_cardinal_preconditioner(
            system.A,
            system.nodes,
            support_size=LS_SUPPORT_SIZE,
        )
        trajectory = fgmres(
            system.A,
            system.b,
            preconditioner.apply,
            x0=np.zeros(N),
            rtol=RTOL,
            atol=ATOL,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
            capture_residual_vectors=True,
            capture_arnoldi=True,
        )
        if not trajectory.converged:
            raise RuntimeError(f"baseline failed at epsilon={epsilon}")
        iterations, residuals, normalized = _sample_true_residuals(trajectory)
        left, singular_values, _ = la.svd(normalized, full_matrices=False, check_finite=False)
        energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        key = _key(epsilon)
        arrays[f"{key}_snapshot_iterations"] = iterations
        arrays[f"{key}_true_residual_snapshots"] = residuals
        arrays[f"{key}_normalized_residual_snapshots"] = normalized
        arrays[f"{key}_singular_values"] = singular_values
        arrays[f"{key}_left_singular_vectors"] = left
        arrays[f"{key}_baseline_solution"] = trajectory.x
        baseline_rows.append(
            {
                "epsilon": epsilon,
                "converged": trajectory.converged,
                "iterations": trajectory.iterations,
                "true_relative_residual": trajectory.relative_residual,
                "solve_seconds_with_snapshot_capture": trajectory.solve_seconds,
                "snapshot_count": len(iterations),
                "restart_boundary_count": len(trajectory.arnoldi_cycles or []),
                "ls_acbf_setup_seconds": preconditioner.setup_seconds,
                "ls_acbf_memory_bytes": preconditioner.memory_bytes,
            }
        )
        snapshot_rows.extend(
            {
                "epsilon": epsilon,
                "snapshot_index": index,
                "iteration": int(iteration),
                "true_residual_norm": float(norm),
                "true_relative_residual": float(norm / np.linalg.norm(system.b)),
                "is_restart_boundary": bool(
                    iteration in {
                        cycle.first_iteration + cycle.hessenberg.shape[1]
                        for cycle in (trajectory.arnoldi_cycles or [])
                    }
                ),
                "is_stride_10": bool(iteration % SNAPSHOT_STRIDE == 0),
            }
            for index, (iteration, norm) in enumerate(zip(iterations, np.linalg.norm(residuals, axis=0)))
        )
        spectra_rows.extend(
            {
                "epsilon": epsilon,
                "mode_index": index,
                "singular_value": float(value),
                "cumulative_energy": float(energy[index - 1]),
            }
            for index, value in enumerate(singular_values, start=1)
        )
        pod_bases[epsilon] = {}
        harmonic_bases[epsilon] = {}
        for q in Q_VALUES:
            basis = left[:, :q]
            harmonic = harmonic_ritz_space(trajectory, q)
            pod_bases[epsilon][q] = basis
            harmonic_bases[epsilon][q] = harmonic
            arrays[f"{key}_pod_q{q}"] = basis
            arrays[f"{key}_harmonic_q{q}"] = harmonic
            rho = np.sum((basis.T @ normalized) ** 2, axis=0)
            summary = _persistence_summary(iterations, rho)
            energy_rows.append(
                {
                    "epsilon": epsilon,
                    "q": q,
                    "E_q": float(energy[q - 1]),
                    "q_over_snapshot_count": q / len(iterations),
                    **summary,
                }
            )
            persistence_summary_rows.append({"epsilon": epsilon, "q": q, **summary})
            persistence_rows.extend(
                {
                    "epsilon": epsilon,
                    "q": q,
                    "snapshot_index": index,
                    "iteration": int(iteration),
                    "rho": float(value),
                }
                for index, (iteration, value) in enumerate(zip(iterations, rho))
            )
        systems[epsilon] = (system, preconditioner, trajectory)

    maximum_previous_q16_geometry_error = 0.0
    for left_epsilon, right_epsilon in UNORDERED_ADJACENT:
        pair_label = f"{left_epsilon:.1f}-{right_epsilon:.1f}"
        for q in Q_VALUES:
            for space, bases in (("residual_pod", pod_bases), ("harmonic_ritz", harmonic_bases)):
                largest, median, normalized_distance = _subspace_metrics(
                    bases[left_epsilon][q],
                    bases[right_epsilon][q],
                )
                geometry_rows.append(
                    {
                        "pair_label": pair_label,
                        "epsilon_left": left_epsilon,
                        "epsilon_right": right_epsilon,
                        "q": q,
                        "space": space,
                        "largest_principal_angle_degrees": largest,
                        "median_principal_angle_degrees": median,
                        "normalized_projector_distance": normalized_distance,
                    }
                )
                if space == "harmonic_ritz" and q == 16 and (left_epsilon, right_epsilon) in previous_q16:
                    previous = previous_q16[(left_epsilon, right_epsilon)]
                    maximum_previous_q16_geometry_error = max(
                        maximum_previous_q16_geometry_error,
                        abs(largest - float(previous["largest_principal_angle_degrees"])),
                        abs(median - float(previous["median_principal_angle_degrees"])),
                        abs(normalized_distance - float(previous["projector_distance_normalized"])),
                    )

    comparison_rows: list[dict[str, Any]] = []
    for q in Q_VALUES:
        residual = [row for row in geometry_rows if row["q"] == q and row["space"] == "residual_pod"]
        harmonic = [row for row in geometry_rows if row["q"] == q and row["space"] == "harmonic_ritz"]
        comparison_rows.append(
            {
                "q": q,
                "residual_pod_median_largest_angle_degrees": float(np.median([row["largest_principal_angle_degrees"] for row in residual])),
                "harmonic_ritz_median_largest_angle_degrees": float(np.median([row["largest_principal_angle_degrees"] for row in harmonic])),
                "residual_pod_median_normalized_projector_distance": float(np.median([row["normalized_projector_distance"] for row in residual])),
                "harmonic_ritz_median_normalized_projector_distance": float(np.median([row["normalized_projector_distance"] for row in harmonic])),
                "pod_has_smaller_projector_distance_for_all_pairs": bool(
                    all(
                        pod_row["normalized_projector_distance"] < harmonic_row["normalized_projector_distance"]
                        for pod_row, harmonic_row in zip(residual, harmonic)
                    )
                ),
            }
        )

    persistence_gate_rows = []
    for q in Q_VALUES:
        selected = [row for row in persistence_summary_rows if row["q"] == q]
        persistence_gate_rows.append(
            {
                "q": q,
                "late_quartile_exceeds_penultimate_count": sum(
                    row["late_minus_penultimate_median"] > 0.0 for row in selected
                ),
                "positive_late_half_slope_count": sum(
                    row["late_half_slope_per_normalized_iteration"] > 0.0
                    for row in selected
                ),
                "epsilon_count": len(selected),
            }
        )

    summary = {
        "configuration": {
            "N": N,
            "epsilon_values": list(EPSILONS),
            "right_preconditioner": "LS-ACBF k=81",
            "restart": RESTART,
            "rtol": RTOL,
            "atol": ATOL,
            "initial_guess": "zero vector",
            "snapshot_rule": "exact true residual every 10 iterations and every restart boundary, duplicates removed",
            "q_values": list(Q_VALUES),
            "ML_used": False,
            "stage2_executed": False,
        },
        "comparison": comparison_rows,
        "stage1_decision": {
            "passed": False,
            "stage2_executed": False,
            "reason": (
                "Stop rule applied: late residual concentration is not systematic across "
                "epsilon, and residual-POD adjacent-epsilon geometry is less stable than "
                "harmonic Ritz at every tested rank."
            ),
            "persistence_counts": persistence_gate_rows,
            "geometric_stability": (
                "Residual POD has a larger median normalized projector distance than "
                "harmonic Ritz for q=4,8,12,16, and is not closer for every adjacent pair "
                "at any rank."
            ),
        },
        "previous_q16_geometry_maximum_absolute_error": maximum_previous_q16_geometry_error,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    _write_csv(args.output_dir / "baseline_results.csv", baseline_rows)
    _write_csv(args.output_dir / "snapshot_metadata.csv", snapshot_rows)
    _write_csv(args.output_dir / "singular_value_spectra.csv", spectra_rows)
    _write_csv(args.output_dir / "energy_and_persistence_summary.csv", energy_rows)
    _write_csv(args.output_dir / "persistence.csv", persistence_rows)
    _write_csv(args.output_dir / "persistence_summary.csv", persistence_summary_rows)
    _write_csv(args.output_dir / "adjacent_geometry.csv", geometry_rows)
    _write_csv(args.output_dir / "geometry_comparison_summary.csv", comparison_rows)
    np.savez_compressed(args.output_dir / "stage1_arrays.npz", **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _make_plots(args.output_dir, spectra_rows, persistence_rows, geometry_rows)
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
