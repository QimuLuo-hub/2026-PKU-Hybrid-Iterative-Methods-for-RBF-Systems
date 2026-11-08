#!/usr/bin/env python3
"""Diagnose whether a q=16 harmonic-Ritz cluster cut caused poor transfer."""

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
import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.krylov import FGMRESResult, fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked
from global_rbf.structure import harmonic_ritz_space
from neurkitt_oracle.gcrodr import build_augmentation
try:
    from scripts.run_global_rbf_epsilon_recycle_transfer import (
        ATOL,
        MAX_ITERATIONS,
        RESTART,
        RTOL,
        _augmentation_diagnostics,
        _epsilon_key,
        _solve_repeated,
        _subspace_metrics,
        _write_csv,
    )
except ModuleNotFoundError:
    from run_global_rbf_epsilon_recycle_transfer import (
        ATOL,
        MAX_ITERATIONS,
        RESTART,
        RTOL,
        _augmentation_diagnostics,
        _epsilon_key,
        _solve_repeated,
        _subspace_metrics,
        _write_csv,
    )


EPSILONS = (7.8, 8.0, 8.2, 8.4)
ADJACENT_PAIRS = (
    (7.8, 8.0),
    (8.0, 7.8),
    (8.0, 8.2),
    (8.2, 8.0),
    (8.2, 8.4),
    (8.4, 8.2),
)
FIXED_RANKS = (12, 16, 20, 24)
TRANSFER_FIXED_RANKS = (16, 20, 24)
GAP_THRESHOLD = 0.05
SPECTRUM_COUNT = 30
SIDE = 20
N = SIDE**2
LS_SUPPORT_SIZE = 81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260902)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_harmonic_cluster_diagnostic"),
    )
    return parser.parse_args()


def harmonic_ritz_values(result: FGMRESResult, minimum_dimension: int = 30) -> tuple[np.ndarray, Any]:
    """Return finite harmonic Ritz values sorted by the existing |theta| rule."""

    cycles = result.arnoldi_cycles or []
    cycle = next(
        (item for item in reversed(cycles) if item.hessenberg.shape[1] >= minimum_dimension),
        None,
    )
    if cycle is None:
        raise ValueError(f"no Arnoldi cycle has dimension at least {minimum_dimension}")
    hbar = cycle.hessenberg
    m = hbar.shape[1]
    hm = hbar[:m, :]
    tail = hbar[m, m - 1] if hbar.shape[0] > m else 0.0
    left = hm.T @ hm
    left[-1, -1] += tail**2
    values = la.eigvals(left, hm.T, check_finite=False)
    values = values[np.isfinite(values)]
    values = values[np.argsort(np.abs(values))]
    if len(values) < minimum_dimension:
        raise ValueError("insufficient finite harmonic Ritz values")
    return values, cycle


def adjacent_relative_gaps(magnitudes: np.ndarray) -> np.ndarray:
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.ndim != 1 or len(magnitudes) < 2:
        raise ValueError("at least two magnitudes are required")
    return np.diff(magnitudes) / np.maximum(magnitudes[:-1], np.finfo(float).tiny)


def cluster_end(magnitudes: np.ndarray, boundary_rank: int) -> int:
    """Extend a prefix boundary through every following sub-5% gap."""

    if boundary_rank < 1 or boundary_rank > len(magnitudes):
        raise ValueError("boundary rank is outside the spectrum")
    gaps = adjacent_relative_gaps(magnitudes)
    end = int(boundary_rank)
    while end < len(magnitudes) and gaps[end - 1] < GAP_THRESHOLD:
        end += 1
    return end


def common_cluster_boundary(
    source_magnitudes: np.ndarray,
    target_magnitudes: np.ndarray,
    start: int = 16,
) -> int:
    """Find the first shared prefix boundary that cuts neither endpoint cluster."""

    boundary = int(start)
    while True:
        extended = max(
            cluster_end(source_magnitudes, boundary),
            cluster_end(target_magnitudes, boundary),
        )
        if extended == boundary:
            return boundary
        boundary = extended


def _spectrum_rows(epsilon: float, values: np.ndarray) -> list[dict[str, Any]]:
    magnitudes = np.abs(values)
    gaps = adjacent_relative_gaps(magnitudes)
    cluster_id = 1
    rows: list[dict[str, Any]] = []
    for index in range(SPECTRUM_COUNT):
        if index > 0 and gaps[index - 1] >= GAP_THRESHOLD:
            cluster_id += 1
        rows.append(
            {
                "epsilon": epsilon,
                "mode_index": index + 1,
                "real_part": float(values[index].real),
                "imaginary_part": float(values[index].imag),
                "magnitude": float(magnitudes[index]),
                "relative_gap_from_previous": "" if index == 0 else float(gaps[index - 1]),
                "relative_gap_to_next": float(gaps[index]),
                "same_cluster_as_next": bool(gaps[index] < GAP_THRESHOLD),
                "cluster_id_within_first_30": cluster_id,
            }
        )
    return rows


def _save_cycles(
    epsilon: float,
    trajectory: FGMRESResult,
    arrays: dict[str, np.ndarray],
    index_rows: list[dict[str, Any]],
) -> None:
    key = _epsilon_key(epsilon)
    for cycle_index, cycle in enumerate(trajectory.arnoldi_cycles or []):
        prefix = f"{key}_cycle_{cycle_index:03d}"
        arrays[f"{prefix}_V"] = cycle.arnoldi_vectors
        arrays[f"{prefix}_Z"] = cycle.preconditioned_vectors
        arrays[f"{prefix}_H"] = cycle.hessenberg
        index_rows.append(
            {
                "epsilon": epsilon,
                "cycle_index": cycle_index,
                "start_iteration": cycle.first_iteration,
                "dimension": cycle.hessenberg.shape[1],
                "V_key": f"{prefix}_V",
                "Z_key": f"{prefix}_Z",
                "H_key": f"{prefix}_H",
            }
        )


def _record_histories(
    estimated_rows: list[dict[str, Any]],
    true_rows: list[dict[str, Any]],
    *,
    solve_id: str,
    role: str,
    source_epsilon: float | None,
    target_epsilon: float,
    result: Any,
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
    iterations = getattr(result, "true_residual_iterations", [result.iterations])
    values = getattr(result, "true_residual_history", [result.relative_residual])
    true_rows.extend(
        {
            "solve_id": solve_id,
            "role": role,
            "source_epsilon": source_epsilon,
            "target_epsilon": target_epsilon,
            "iteration": iteration,
            "true_relative_residual": value,
        }
        for iteration, value in zip(iterations, values)
    )


def _make_plots(
    output_dir: Path,
    spectrum_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
) -> None:
    colors = {7.8: "#1f4e79", 8.0: "#477da8", 8.2: "#d27a2c", 8.4: "#8b5a2b"}
    markers = {7.8: "o", 8.0: "s", 8.2: "^", 8.4: "D"}

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), constrained_layout=True)
    for epsilon in EPSILONS:
        selected = [row for row in spectrum_rows if row["epsilon"] == epsilon]
        indices = [row["mode_index"] for row in selected]
        magnitudes = [row["magnitude"] for row in selected]
        gaps = [row["relative_gap_to_next"] for row in selected]
        axes[0].plot(
            indices,
            magnitudes,
            color=colors[epsilon],
            marker=markers[epsilon],
            markersize=3.5,
            linewidth=1.5,
            label=f"epsilon {epsilon:.1f}",
        )
        axes[1].plot(
            indices[11:24],
            gaps[11:24],
            color=colors[epsilon],
            marker=markers[epsilon],
            markersize=4.0,
            linewidth=1.4,
        )
    axes[0].axvline(16.5, color="#555b63", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Harmonic-Ritz magnitude |theta|")
    axes[0].set_title("First 30 harmonic-Ritz magnitudes", loc="left", fontweight="semibold")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].grid(axis="y", color="#d9dde2", linewidth=0.7)
    axes[1].axhline(
        GAP_THRESHOLD,
        color="#555b63",
        linestyle="--",
        linewidth=1.2,
        label="5% cluster boundary criterion",
    )
    axes[1].set_xlabel("Mode index i (gap shown from i to i+1)")
    axes[1].set_ylabel("Adjacent relative gap")
    axes[1].set_title("Adjacent gaps around the q=16 boundary", loc="left", fontweight="semibold")
    axes[1].set_xticks(range(12, 25))
    axes[1].grid(axis="y", color="#d9dde2", linewidth=0.7)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Global Gaussian-RBF harmonic-Ritz cluster diagnostic, N=400",
        fontsize=15,
        fontweight="bold",
        color="#20242a",
    )
    fig.savefig(output_dir / "harmonic_ritz_spectra_and_gaps.png", dpi=200)
    plt.close(fig)

    pair_labels = [f"{source:.1f}->{target:.1f}" for source, target in ADJACENT_PAIRS]
    candidate_labels = ("q16", "q20", "q24", "cluster_aware")
    candidate_colors = ("#356c9b", "#78a6ca", "#d27a2c", "#8b5a2b")
    x = np.arange(len(pair_labels), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    for offset, (candidate, color) in enumerate(zip(candidate_labels, candidate_colors)):
        selected = [
            next(
                row
                for row in transfer_rows
                if row["pair_label"] == pair_label and row["candidate"] == candidate
            )
            for pair_label in pair_labels
        ]
        ax.bar(
            x + (offset - 1.5) * width,
            [row["transferred_baseline_iteration_ratio"] for row in selected],
            width,
            label=candidate.replace("_", " "),
            color=color,
            edgecolor="#20242a",
            linewidth=0.5,
        )
    ax.axhline(1.0, color="#555b63", linestyle="-.", linewidth=1.2, label="baseline")
    ax.axhline(0.90, color="#555b63", linestyle="--", linewidth=1.2, label="positive threshold")
    ax.set_xticks(x, pair_labels)
    ax.set_ylabel("Transferred iterations / baseline iterations")
    ax.set_xlabel("Directed adjacent epsilon transfer")
    ax.set_title("Nearby transfer by fixed and cluster-aware rank", loc="left", fontweight="semibold")
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, fontsize=9)
    fig.savefig(output_dir / "adjacent_transfer_iteration_ratios.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warmup = not args.no_warmup

    random_max, _ = la.qr(
        np.random.default_rng(args.random_seed).standard_normal((N, max(FIXED_RANKS))),
        mode="economic",
        check_finite=False,
    )
    systems: dict[float, dict[str, Any]] = {}
    spectrum_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    self_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    true_residual_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"matched_random_q24": random_max}
    cycle_arrays: dict[str, np.ndarray] = {}
    cycle_index_rows: list[dict[str, Any]] = []

    for epsilon in EPSILONS:
        print(f"[source epsilon={epsilon:.1f}] baseline extraction", flush=True)
        system = assemble_tensor_gaussian_poisson_blocked(SIDE, epsilon=epsilon)
        preconditioner = rbf_local_cardinal_preconditioner(
            system.A,
            system.nodes,
            support_size=LS_SUPPORT_SIZE,
        )
        started = perf_counter()
        trajectory = fgmres(
            system.A,
            system.b,
            preconditioner.apply,
            x0=np.zeros(N),
            rtol=RTOL,
            atol=ATOL,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
            capture_arnoldi=True,
        )
        extraction_seconds = perf_counter() - started
        if not trajectory.converged:
            raise RuntimeError(f"baseline extraction did not converge for epsilon={epsilon}")
        values, selected_cycle = harmonic_ritz_values(trajectory, SPECTRUM_COUNT)
        magnitudes = np.abs(values)
        local_cluster_rank = cluster_end(magnitudes, 16)
        spectrum_rows.extend(_spectrum_rows(epsilon, values))
        _save_cycles(epsilon, trajectory, cycle_arrays, cycle_index_rows)
        key = _epsilon_key(epsilon)
        arrays[f"{key}_extraction_solution"] = trajectory.x
        _record_histories(
            residual_rows,
            true_residual_rows,
            solve_id=f"{key}_source_extraction",
            role="source_extraction",
            source_epsilon=epsilon,
            target_epsilon=epsilon,
            result=trajectory,
        )
        source_rows.append(
            {
                "epsilon": epsilon,
                "converged": trajectory.converged,
                "iterations": trajectory.iterations,
                "true_relative_residual": trajectory.relative_residual,
                "extraction_seconds_offline": extraction_seconds,
                "ls_acbf_setup_seconds": preconditioner.setup_seconds,
                "ls_acbf_memory_bytes": preconditioner.memory_bytes,
                "selected_cycle_start_iteration": selected_cycle.first_iteration,
                "selected_cycle_dimension": selected_cycle.hessenberg.shape[1],
                "q_cluster_local": local_cluster_rank,
                "gap_after_mode_16": adjacent_relative_gaps(magnitudes)[15],
            }
        )
        systems[epsilon] = {
            "system": system,
            "preconditioner": preconditioner,
            "trajectory": trajectory,
            "values": values,
            "magnitudes": magnitudes,
            "q_cluster": local_cluster_rank,
            "bases": {},
        }

    pair_cluster_ranks: dict[tuple[float, float], int] = {}
    for source, target in ADJACENT_PAIRS:
        pair_cluster_ranks[(source, target)] = common_cluster_boundary(
            systems[source]["magnitudes"],
            systems[target]["magnitudes"],
            16,
        )

    needed_ranks = set(FIXED_RANKS)
    needed_ranks.update(item["q_cluster"] for item in systems.values())
    needed_ranks.update(pair_cluster_ranks.values())
    if max(needed_ranks) > random_max.shape[1]:
        raise RuntimeError("cluster-aware rank exceeds the random control capacity")

    solve_cache: dict[tuple[float, int, str], tuple[Any, float, list[float], np.ndarray, np.ndarray]] = {}
    baseline_cache: dict[float, tuple[Any, float, list[float]]] = {}
    for epsilon in EPSILONS:
        data = systems[epsilon]
        system = data["system"]
        preconditioner = data["preconditioner"]
        print(f"[target epsilon={epsilon:.1f}] self-oracle controls", flush=True)
        baseline_result, baseline_time, baseline_repeats = _solve_repeated(
            system.A,
            system.b,
            preconditioner.apply,
            u_basis=None,
            c_basis=None,
            repeats=args.repeats,
            warmup=warmup,
        )
        baseline_cache[epsilon] = (baseline_result, baseline_time, baseline_repeats)
        key = _epsilon_key(epsilon)
        arrays[f"{key}_baseline_solution"] = baseline_result.x
        _record_histories(
            residual_rows,
            true_residual_rows,
            solve_id=f"{key}_baseline",
            role="baseline",
            source_epsilon=None,
            target_epsilon=epsilon,
            result=baseline_result,
        )

        for q in sorted(needed_ranks):
            basis = harmonic_ritz_space(data["trajectory"], q)
            if basis.shape != (N, q):
                raise RuntimeError(f"epsilon={epsilon} q={q} produced shape {basis.shape}")
            data["bases"][q] = basis
            arrays[f"{key}_Y_q{q}"] = basis
            oracle_u, oracle_c = build_augmentation(system.A, basis)
            oracle_result, oracle_time, oracle_repeats = _solve_repeated(
                system.A,
                system.b,
                preconditioner.apply,
                u_basis=oracle_u,
                c_basis=oracle_c,
                repeats=args.repeats,
                warmup=warmup,
            )
            random_basis = random_max[:, :q]
            random_u, random_c = build_augmentation(system.A, random_basis)
            random_result, random_time, random_repeats = _solve_repeated(
                system.A,
                system.b,
                preconditioner.apply,
                u_basis=random_u,
                c_basis=random_c,
                repeats=args.repeats,
                warmup=warmup,
            )
            solve_cache[(epsilon, q, "oracle")] = (
                oracle_result,
                oracle_time,
                oracle_repeats,
                oracle_u,
                oracle_c,
            )
            solve_cache[(epsilon, q, "random")] = (
                random_result,
                random_time,
                random_repeats,
                random_u,
                random_c,
            )
            for role, result, elapsed, timings, u_basis, c_basis in (
                ("oracle", oracle_result, oracle_time, oracle_repeats, oracle_u, oracle_c),
                ("random", random_result, random_time, random_repeats, random_u, random_c),
            ):
                arrays[f"{key}_{role}_q{q}_U"] = u_basis
                arrays[f"{key}_{role}_q{q}_C"] = c_basis
                arrays[f"{key}_{role}_q{q}_solution"] = result.x
                _record_histories(
                    residual_rows,
                    true_residual_rows,
                    solve_id=f"{key}_{role}_q{q}",
                    role=role,
                    source_epsilon=None,
                    target_epsilon=epsilon,
                    result=result,
                )
            base_iterations = baseline_result.iterations
            self_rows.append(
                {
                    "epsilon": epsilon,
                    "candidate": "fixed" if q in FIXED_RANKS else "cluster_only",
                    "is_local_cluster_rank": q == data["q_cluster"],
                    "q": q,
                    "q_over_N": q / N,
                    "baseline_converged": baseline_result.converged,
                    "baseline_iterations": base_iterations,
                    "baseline_true_relative_residual": baseline_result.relative_residual,
                    "baseline_solve_seconds": baseline_time,
                    "oracle_converged": oracle_result.converged,
                    "oracle_iterations": oracle_result.iterations,
                    "oracle_baseline_iteration_ratio": oracle_result.iterations / base_iterations,
                    "oracle_true_relative_residual": oracle_result.relative_residual,
                    "oracle_solve_seconds": oracle_time,
                    "random_converged": random_result.converged,
                    "random_iterations": random_result.iterations,
                    "random_baseline_iteration_ratio": random_result.iterations / base_iterations,
                    "random_true_relative_residual": random_result.relative_residual,
                    "random_solve_seconds": random_time,
                    "oracle_timing_repeats_seconds": json.dumps(oracle_repeats),
                    "random_timing_repeats_seconds": json.dumps(random_repeats),
                    "baseline_timing_repeats_seconds": json.dumps(baseline_repeats),
                }
            )

    transfer_cache: dict[tuple[float, float, int], tuple[Any, float, list[float], np.ndarray, np.ndarray]] = {}
    for source, target in ADJACENT_PAIRS:
        pair_label = f"{source:.1f}->{target:.1f}"
        cluster_rank = pair_cluster_ranks[(source, target)]
        candidates = (
            ("q16", 16),
            ("q20", 20),
            ("q24", 24),
            ("cluster_aware", cluster_rank),
        )
        for candidate, q in candidates:
            print(f"[transfer {pair_label}] {candidate} q={q}", flush=True)
            cache_key = (source, target, q)
            target_system = systems[target]["system"]
            target_preconditioner = systems[target]["preconditioner"]
            source_basis = systems[source]["bases"][q]
            target_basis = systems[target]["bases"][q]
            if cache_key not in transfer_cache:
                transferred_u, transferred_c = build_augmentation(target_system.A, source_basis)
                transferred_result, transferred_time, transferred_repeats = _solve_repeated(
                    target_system.A,
                    target_system.b,
                    target_preconditioner.apply,
                    u_basis=transferred_u,
                    c_basis=transferred_c,
                    repeats=args.repeats,
                    warmup=warmup,
                )
                transfer_cache[cache_key] = (
                    transferred_result,
                    transferred_time,
                    transferred_repeats,
                    transferred_u,
                    transferred_c,
                )
                key = f"source_{_epsilon_key(source)}_target_{_epsilon_key(target)}_q{q}"
                arrays[f"{key}_U"] = transferred_u
                arrays[f"{key}_C"] = transferred_c
                arrays[f"{key}_solution"] = transferred_result.x
                _record_histories(
                    residual_rows,
                    true_residual_rows,
                    solve_id=key,
                    role="transferred",
                    source_epsilon=source,
                    target_epsilon=target,
                    result=transferred_result,
                )
            transferred_result, transferred_time, transferred_repeats, transferred_u, transferred_c = transfer_cache[cache_key]
            oracle_result, oracle_time, _, _, _ = solve_cache[(target, q, "oracle")]
            random_result, random_time, _, _, _ = solve_cache[(target, q, "random")]
            baseline_result, baseline_time, _ = baseline_cache[target]
            angles, largest, median, projector, normalized = _subspace_metrics(
                source_basis,
                target_basis,
            )
            au_defect, ctc_defect = _augmentation_diagnostics(
                target_system.A,
                transferred_u,
                transferred_c,
            )
            base_iterations = baseline_result.iterations
            transfer_rows.append(
                {
                    "pair_label": pair_label,
                    "epsilon_source": source,
                    "epsilon_target": target,
                    "candidate": candidate,
                    "q": q,
                    "q_over_N": q / N,
                    "source_local_q_cluster": systems[source]["q_cluster"],
                    "target_local_q_cluster": systems[target]["q_cluster"],
                    "pair_common_q_cluster": cluster_rank,
                    "baseline_converged": baseline_result.converged,
                    "baseline_iterations": base_iterations,
                    "baseline_true_relative_residual": baseline_result.relative_residual,
                    "baseline_solve_seconds": baseline_time,
                    "transferred_converged": transferred_result.converged,
                    "transferred_iterations": transferred_result.iterations,
                    "transferred_baseline_iteration_ratio": transferred_result.iterations / base_iterations,
                    "transferred_true_relative_residual": transferred_result.relative_residual,
                    "transferred_solve_seconds": transferred_time,
                    "transferred_baseline_solve_time_ratio": transferred_time / baseline_time,
                    "oracle_converged": oracle_result.converged,
                    "oracle_iterations": oracle_result.iterations,
                    "oracle_baseline_iteration_ratio": oracle_result.iterations / base_iterations,
                    "oracle_true_relative_residual": oracle_result.relative_residual,
                    "oracle_solve_seconds": oracle_time,
                    "random_converged": random_result.converged,
                    "random_iterations": random_result.iterations,
                    "random_baseline_iteration_ratio": random_result.iterations / base_iterations,
                    "random_true_relative_residual": random_result.relative_residual,
                    "random_solve_seconds": random_time,
                    "target_AU_minus_C_frobenius": au_defect,
                    "target_CHC_minus_I_frobenius": ctc_defect,
                    "largest_principal_angle_degrees": largest,
                    "median_principal_angle_degrees": median,
                    "projector_distance_frobenius": projector,
                    "projector_distance_normalized": normalized,
                    "transferred_timing_repeats_seconds": json.dumps(transferred_repeats),
                }
            )
            angle_rows.extend(
                {
                    "pair_label": pair_label,
                    "candidate": candidate,
                    "q": q,
                    "angle_index": index,
                    "principal_angle_degrees": float(angle),
                }
                for index, angle in enumerate(angles, start=1)
            )

    cluster_rows = [row for row in transfer_rows if row["candidate"] == "cluster_aware"]
    q16_rows = [row for row in transfer_rows if row["candidate"] == "q16"]
    cluster_ratios = np.asarray(
        [row["transferred_baseline_iteration_ratio"] for row in cluster_rows],
        dtype=float,
    )
    q16_ratios = np.asarray(
        [row["transferred_baseline_iteration_ratio"] for row in q16_rows],
        dtype=float,
    )
    cluster_projectors = np.asarray(
        [row["projector_distance_normalized"] for row in cluster_rows],
        dtype=float,
    )
    q16_projectors = np.asarray(
        [row["projector_distance_normalized"] for row in q16_rows],
        dtype=float,
    )
    cluster_angles = np.asarray(
        [row["largest_principal_angle_degrees"] for row in cluster_rows],
        dtype=float,
    )
    q16_angles = np.asarray(
        [row["largest_principal_angle_degrees"] for row in q16_rows],
        dtype=float,
    )
    positive_count = int(np.count_nonzero(cluster_ratios <= 0.90))
    clearly_better_random_count = sum(
        row["transferred_iterations"] < row["random_iterations"] for row in cluster_rows
    )
    geometry_improves = bool(
        np.median(cluster_projectors) < np.median(q16_projectors)
        and np.median(cluster_angles) < np.median(q16_angles)
    )
    if (
        positive_count >= 4
        and np.median(cluster_ratios) < np.median(q16_ratios)
        and clearly_better_random_count >= 4
    ):
        decision = "POSITIVE"
    elif 0.90 <= np.median(cluster_ratios) <= 0.95 and geometry_improves:
        decision = "WEAK"
    else:
        decision = "NEGATIVE"

    summary = {
        "configuration": {
            "epsilon_values": list(EPSILONS),
            "N": N,
            "side": SIDE,
            "kernel": "global Gaussian RBF",
            "pde": "same 2D Poisson manufactured problem as validated epsilon transfer",
            "right_preconditioner": "LS-ACBF k=81 rebuilt for each epsilon",
            "fixed_self_ranks": list(FIXED_RANKS),
            "fixed_transfer_ranks": list(TRANSFER_FIXED_RANKS),
            "cluster_gap_threshold": GAP_THRESHOLD,
            "spectrum_count": SPECTRUM_COUNT,
            "rtol": RTOL,
            "atol": ATOL,
            "restart": RESTART,
            "max_iterations": MAX_ITERATIONS,
            "initial_guess": "zero vector",
            "timing_repeats": args.repeats,
            "timing_warmup": warmup,
            "random_seed": args.random_seed,
            "recycling": "existing validated right-preconditioned projected-Arnoldi implementation",
            "pair_cluster_rule": "smallest common boundary >=16 that cuts no sub-5% cluster at either endpoint",
            "ML_used": False,
        },
        "local_cluster_ranks": {
            f"{epsilon:.1f}": systems[epsilon]["q_cluster"] for epsilon in EPSILONS
        },
        "pair_common_cluster_ranks": {
            f"{source:.1f}->{target:.1f}": rank
            for (source, target), rank in pair_cluster_ranks.items()
        },
        "decision": {
            "classification": decision,
            "cluster_aware_median_iteration_ratio": float(np.median(cluster_ratios)),
            "q16_median_iteration_ratio": float(np.median(q16_ratios)),
            "cluster_aware_pairs_ratio_le_0p90": positive_count,
            "cluster_aware_pairs_better_than_random": clearly_better_random_count,
            "cluster_aware_pairs_better_than_baseline": int(np.count_nonzero(cluster_ratios < 1.0)),
            "cluster_aware_median_largest_angle_degrees": float(np.median(cluster_angles)),
            "q16_median_largest_angle_degrees": float(np.median(q16_angles)),
            "cluster_aware_median_normalized_projector_distance": float(np.median(cluster_projectors)),
            "q16_median_normalized_projector_distance": float(np.median(q16_projectors)),
            "geometry_improves_in_both_medians": geometry_improves,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }

    _write_csv(args.output_dir / "harmonic_ritz_spectra.csv", spectrum_rows)
    _write_csv(args.output_dir / "source_extractions.csv", source_rows)
    _write_csv(args.output_dir / "self_oracle_results.csv", self_rows)
    _write_csv(args.output_dir / "adjacent_transfer_results.csv", transfer_rows)
    _write_csv(args.output_dir / "principal_angles.csv", angle_rows)
    _write_csv(args.output_dir / "residual_histories.csv", residual_rows)
    _write_csv(args.output_dir / "true_residual_histories.csv", true_residual_rows)
    _write_csv(args.output_dir / "source_krylov_cycle_index.csv", cycle_index_rows)
    np.savez_compressed(args.output_dir / "recycle_spaces_and_solutions.npz", **arrays)
    np.savez_compressed(args.output_dir / "source_krylov_cycles.npz", **cycle_arrays)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _make_plots(args.output_dir, spectrum_rows, transfer_rows)
    print(json.dumps(summary["decision"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
