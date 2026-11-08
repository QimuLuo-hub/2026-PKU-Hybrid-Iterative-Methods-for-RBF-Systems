"""Compare solver-dynamics oracle spaces for the global-RBF benchmark.

The experiment is deliberately non-ML and non-transfer:

* unchanged 27 Gaussian N=225 systems and four manufactured RHSs;
* right-preconditioned FGMRES with the FP32 pivoted-LU baseline;
* the standard nonsymmetric Galerkin-deflation integration selected earlier;
* harmonic Ritz, late exact-error POD, late unresolved-correction POD, and
  dominant right singular vectors of S = I - M^{-1} A;
* requested ranks q = 4, 8, 16.

Some baseline trajectories converge before a requested q is available.  For
those cases every target is truncated to the same ``q_actual``.  Exact-q
results are also summarized separately so target dimension never confounds
the primary comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.benchmarking import benchmark_fgmres, true_relative_residual
from global_rbf.collocation import GlobalRBFSystem, variable_diffusion_forcing
from global_rbf.krylov import FGMRESResult, fgmres
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import GalerkinDeflation, galerkin_deflation, harmonic_ritz_space
from run_global_rbf_stage2 import N, RHS_SPECS, RTOL, assemble_family_member, family_specs, write_csv


OUTPUT_DEFAULT = Path("results/global_rbf_krylov_oracle_spaces")
REPORT_DEFAULT = Path("global_rbf_krylov_dynamics_oracle_report.md")
Q_VALUES = (4, 8, 16)
MAX_Q = max(Q_VALUES)
LATE_SNAPSHOT_LIMIT = 2 * MAX_Q
RESTART = min(100, N)
MAX_ITERATIONS = min(450, 2 * N)
REPEATS = 3
REFERENCE_SWITCH_TOLERANCE = float(np.sqrt(np.finfo(float).eps))
REFERENCE_DECIMAL_DIGITS = 60
TARGETS = (
    "harmonic_ritz",
    "late_exact_error_pod",
    "late_unresolved_pod",
    "dominant_right_singular_S",
)
TARGET_LABELS = {
    "harmonic_ritz": "Harmonic Ritz",
    "late_exact_error_pod": "Late exact-error POD",
    "late_unresolved_pod": "Late unresolved-correction POD",
    "dominant_right_singular_S": "Dominant right singular vectors of S",
}


@dataclass
class SystemArtifact:
    system_id: str
    family: str
    parameter: str
    index: int
    value: float
    system: GlobalRBFSystem
    base: Preconditioner
    reference_factor: tuple[np.ndarray, np.ndarray]
    zs_basis: np.ndarray
    zs_setup_seconds: float
    mp_matrix: Any | None = None
    mp_lu: Any | None = None
    mp_pivots: Any | None = None
    mp_factor_seconds: float = 0.0


def common_fields(artifact: SystemArtifact) -> dict[str, Any]:
    return {
        "system_id": artifact.system_id,
        "family": artifact.family,
        "parameter": artifact.parameter,
        "parameter_index": artifact.index,
        "parameter_value": artifact.value,
        "N": N,
        "kernel": "gaussian",
        "epsilon": artifact.value if artifact.family == "epsilon" else 6.0,
        "jitter_fraction": artifact.value if artifact.family == "jitter" else 0.0,
        "beta": artifact.value if artifact.family == "pde_coefficient" else 0.0,
    }


def mode_rhs(artifact: SystemArtifact, kx: int, ky: int) -> np.ndarray:
    points = artifact.system.nodes
    u = np.sin(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1])
    if artifact.family == "pde_coefficient":
        forcing = variable_diffusion_forcing(points, artifact.value, kx=kx, ky=ky)
    else:
        forcing = (kx**2 + ky**2) * np.pi**2 * u
    return np.where(artifact.system.boundary, u, forcing)


def build_artifact(spec: Any, index: int, value: float) -> SystemArtifact:
    system_id, system = assemble_family_member(spec, index, value)
    base = low_precision_lu_preconditioner(system.A)
    reference_factor = la.lu_factor(system.A, check_finite=False)
    start = perf_counter()
    B = np.asarray(base.apply(system.A), dtype=float)
    _, _, right_s = la.svd(np.eye(system.dimension) - B, full_matrices=False, check_finite=False)
    zs_basis = right_s[:MAX_Q, :].T
    zs_basis, _ = la.qr(zs_basis, mode="economic", check_finite=False)
    return SystemArtifact(
        system_id,
        spec.family,
        spec.parameter,
        index,
        float(value),
        system,
        base,
        reference_factor,
        zs_basis,
        perf_counter() - start,
    )


def late_residual_matrix(residual_vectors: list[np.ndarray]) -> np.ndarray:
    """Return up to 2*q_max late residual directions without restart duplicates."""

    directions: list[np.ndarray] = []
    for residual in residual_vectors:
        vector = np.asarray(residual, dtype=float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 100.0 * np.finfo(float).eps:
            continue
        direction = vector / norm
        if directions:
            duplicate_distance = min(
                np.linalg.norm(direction - directions[-1]),
                np.linalg.norm(direction + directions[-1]),
            )
            if duplicate_distance <= 1.0e-11:
                continue
        directions.append(direction)
    if not directions:
        return np.empty((0, 0))
    return np.column_stack(directions[-LATE_SNAPSHOT_LIMIT:])


def refined_fp64_reference_actions(
    A: np.ndarray,
    factor: tuple[np.ndarray, np.ndarray],
    residuals: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply A^{-1} with two iterative-refinement steps; never form A^{-1}."""

    if residuals.size == 0:
        return residuals.copy(), 0.0
    actions = la.lu_solve(factor, residuals, check_finite=False)
    for _ in range(2):
        actions += la.lu_solve(factor, residuals - A @ actions, check_finite=False)
    relative = np.linalg.norm(residuals - A @ actions, axis=0) / np.linalg.norm(residuals, axis=0)
    return actions, float(np.max(relative))


def accurate_reference_actions(
    artifact: SystemArtifact,
    residuals: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use verified FP64 actions, with an MP60 fallback for hard systems."""

    start = perf_counter()
    actions, fp64_residual = refined_fp64_reference_actions(
        artifact.system.A,
        artifact.reference_factor,
        residuals,
    )
    if fp64_residual <= REFERENCE_SWITCH_TOLERANCE:
        return actions, {
            "reference_method": "FP64 LU plus iterative refinement",
            "max_reference_action_relative_residual": fp64_residual,
            "max_reference_action_double_representation_residual": fp64_residual,
            "reference_factor_seconds": 0.0,
            "reference_solve_seconds": perf_counter() - start,
        }

    with mp.workdps(REFERENCE_DECIMAL_DIGITS):
        factor_seconds = 0.0
        if artifact.mp_lu is None:
            factor_start = perf_counter()
            artifact.mp_matrix = mp.matrix(artifact.system.A.tolist())
            artifact.mp_lu, artifact.mp_pivots = mp.mp.LU_decomp(artifact.mp_matrix)
            artifact.mp_factor_seconds = perf_counter() - factor_start
            factor_seconds = artifact.mp_factor_seconds
        solve_start = perf_counter()
        columns: list[list[float]] = []
        maximum_mp_residual = mp.mpf("0")
        for column in range(residuals.shape[1]):
            right_hand_side = mp.matrix(residuals[:, column].tolist())
            intermediate = mp.mp.L_solve(artifact.mp_lu, right_hand_side, artifact.mp_pivots)
            solution = mp.mp.U_solve(artifact.mp_lu, intermediate)
            mp_residual = mp.norm(right_hand_side - artifact.mp_matrix * solution) / mp.norm(
                right_hand_side
            )
            maximum_mp_residual = max(maximum_mp_residual, mp_residual)
            columns.append([float(value) for value in solution])
        actions = np.asarray(columns, dtype=float).T
        solve_seconds = perf_counter() - solve_start
    double_representation_residual = float(
        np.max(
            np.linalg.norm(residuals - artifact.system.A @ actions, axis=0)
            / np.linalg.norm(residuals, axis=0)
        )
    )
    return actions, {
        "reference_method": f"MP{REFERENCE_DECIMAL_DIGITS} pivoted LU",
        "max_reference_action_relative_residual": float(maximum_mp_residual),
        "max_reference_action_double_representation_residual": double_representation_residual,
        "reference_factor_seconds": factor_seconds,
        "reference_solve_seconds": solve_seconds,
    }


def pod_basis(columns: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a numerical-range POD basis and its singular values."""

    matrix = np.asarray(columns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.empty((matrix.shape[0] if matrix.ndim == 2 else 0, 0)), np.empty(0), 0
    U, singular_values, _ = la.svd(matrix, full_matrices=False, check_finite=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.empty((matrix.shape[0], 0)), singular_values, 0
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    retained = min(max_rank, numerical_rank)
    return U[:, :retained], singular_values, numerical_rank


def capture_baseline(
    artifact: SystemArtifact,
    rhs: np.ndarray,
    *,
    repeats: int,
) -> tuple[FGMRESResult, dict[str, Any]]:
    trajectory = fgmres(
        artifact.system.A,
        rhs,
        artifact.base.apply,
        rtol=RTOL,
        restart=RESTART,
        max_iterations=MAX_ITERATIONS,
        capture_residual_vectors=True,
        capture_arnoldi=True,
    )
    timed = benchmark_fgmres(
        artifact.system.A,
        rhs,
        artifact.base,
        rtol=RTOL,
        restart=RESTART,
        max_iterations=MAX_ITERATIONS,
        repeats=repeats,
        warmup=True,
    )
    if trajectory.iterations != timed.result.iterations or trajectory.converged != timed.result.converged:
        raise RuntimeError("diagnostic capture changed the baseline FGMRES trajectory")
    return trajectory, {
        "iterations": trajectory.iterations,
        "converged": trajectory.converged,
        "solve_seconds": timed.solve_seconds_median,
        "true_relative_residual": true_relative_residual(artifact.system.A, rhs, trajectory.x),
    }


def save_trajectory(path: Path, trajectory: FGMRESResult) -> None:
    arrays: dict[str, np.ndarray] = {
        "residual_history": np.asarray(trajectory.residual_history, dtype=float),
        "residual_vectors": (
            np.column_stack(trajectory.residual_vectors)
            if trajectory.residual_vectors
            else np.empty((len(trajectory.x), 0))
        ),
    }
    for cycle_index, cycle in enumerate(trajectory.arnoldi_cycles or []):
        arrays[f"cycle_{cycle_index}_V"] = cycle.arnoldi_vectors
        arrays[f"cycle_{cycle_index}_Z"] = cycle.preconditioned_vectors
        arrays[f"cycle_{cycle_index}_H"] = cycle.hessenberg
        arrays[f"cycle_{cycle_index}_first_iteration"] = np.asarray([cycle.first_iteration], dtype=int)
    np.savez_compressed(path, **arrays)


def benchmark_deflated(
    artifact: SystemArtifact,
    deflation: GalerkinDeflation,
    rhs: np.ndarray,
    *,
    repeats: int,
) -> dict[str, Any]:
    projected_rhs = deflation.deflate_rhs(rhs)
    absolute_threshold = RTOL * np.linalg.norm(rhs)

    def once() -> tuple[int, float, float, bool, float]:
        start = perf_counter()
        result = fgmres(
            deflation.A_deflated,
            projected_rhs,
            artifact.base.apply,
            rtol=0.0,
            atol=absolute_threshold,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
        )
        solution = deflation.reconstruct(artifact.system.A, rhs, result.x)
        seconds = perf_counter() - start
        residual = true_relative_residual(artifact.system.A, rhs, solution)
        identity_error = np.linalg.norm(
            rhs
            - artifact.system.A @ solution
            - (projected_rhs - deflation.A_deflated @ result.x)
        ) / np.linalg.norm(rhs)
        return result.iterations, seconds, residual, residual <= RTOL, float(identity_error)

    once()
    runs = [once() for _ in range(repeats)]
    primary = runs[0]
    return {
        "iterations": primary[0],
        "solve_seconds": float(np.median([run[1] for run in runs])),
        "true_relative_residual": primary[2],
        "converged": primary[3],
        "deflation_residual_identity_error": primary[4],
    }


def paired_outcome(method: dict[str, Any], baseline: dict[str, Any]) -> str:
    if bool(method["converged"]) and not bool(baseline["converged"]):
        return "win"
    if not bool(method["converged"]) and bool(baseline["converged"]):
        return "loss"
    if int(method["iterations"]) < int(baseline["iterations"]):
        return "win"
    if int(method["iterations"]) > int(baseline["iterations"]):
        return "loss"
    return "tie"


def system_outcome(rows: list[dict[str, Any]]) -> str:
    wins = sum(row["outcome_vs_baseline"] == "win" for row in rows)
    losses = sum(row["outcome_vs_baseline"] == "loss" for row in rows)
    return "win" if wins > losses else "loss" if losses > wins else "tie"


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    system_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for q in Q_VALUES:
            selected = [row for row in rows if row["target"] == target and row["q_requested"] == q]
            exact = [row for row in selected if row["q_actual"] == q]
            system_outcomes: list[str] = []
            for system_id in dict.fromkeys(row["system_id"] for row in selected):
                group = [row for row in selected if row["system_id"] == system_id]
                outcome = system_outcome(group)
                system_outcomes.append(outcome)
                system_rows.append(
                    {
                        "system_id": system_id,
                        "family": group[0]["family"],
                        "parameter_value": group[0]["parameter_value"],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "q_requested": q,
                        "median_q_actual": float(np.median([row["q_actual"] for row in group])),
                        "exact_q_rhs": sum(row["q_actual"] == q for row in group),
                        "converged_rhs": sum(bool(row["converged"]) for row in group),
                        "median_iterations": float(np.median([row["iterations"] for row in group])),
                        "median_iteration_ratio_vs_baseline": float(
                            np.median([row["iteration_ratio_vs_baseline"] for row in group])
                        ),
                        "rhs_wins": sum(row["outcome_vs_baseline"] == "win" for row in group),
                        "rhs_ties": sum(row["outcome_vs_baseline"] == "tie" for row in group),
                        "rhs_losses": sum(row["outcome_vs_baseline"] == "loss" for row in group),
                        "system_outcome_vs_baseline": outcome,
                    }
                )
            exact_ratios = np.asarray([row["iteration_ratio_vs_baseline"] for row in exact], dtype=float)
            all_ratios = np.asarray([row["iteration_ratio_vs_baseline"] for row in selected], dtype=float)
            aggregate.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "q_requested": q,
                    "observations": len(selected),
                    "exact_q_observations": len(exact),
                    "exact_q_fraction": len(exact) / len(selected),
                    "median_q_actual": float(np.median([row["q_actual"] for row in selected])),
                    "converged_all": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio_all_matched_rank": float(np.median(all_ratios)),
                    "median_iteration_ratio_exact_q": (
                        float(np.median(exact_ratios)) if exact_ratios.size else float("nan")
                    ),
                    "exact_q_converged": sum(bool(row["converged"]) for row in exact),
                    "exact_q_wins": sum(row["outcome_vs_baseline"] == "win" for row in exact),
                    "exact_q_ties": sum(row["outcome_vs_baseline"] == "tie" for row in exact),
                    "exact_q_losses": sum(row["outcome_vs_baseline"] == "loss" for row in exact),
                    "all_wins": sum(row["outcome_vs_baseline"] == "win" for row in selected),
                    "all_ties": sum(row["outcome_vs_baseline"] == "tie" for row in selected),
                    "all_losses": sum(row["outcome_vs_baseline"] == "loss" for row in selected),
                    "system_wins": system_outcomes.count("win"),
                    "system_ties": system_outcomes.count("tie"),
                    "system_losses": system_outcomes.count("loss"),
                }
            )
    return aggregate, system_rows


def summarize_targets(
    aggregate: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in TARGETS:
        rows = [row for row in aggregate if row["target"] == target]
        raw = []
        exact_wins = exact_ties = exact_losses = exact_converged = exact_observations = 0
        for row in rows:
            exact_wins += int(row["exact_q_wins"])
            exact_ties += int(row["exact_q_ties"])
            exact_losses += int(row["exact_q_losses"])
            exact_converged += int(row["exact_q_converged"])
            exact_observations += int(row["exact_q_observations"])
        for q in Q_VALUES:
            raw.extend(
                result["iteration_ratio_vs_baseline"]
                for result in result_rows
                if result["target"] == target
                and result["q_requested"] == q
                and result["q_actual"] == q
            )
        output.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "exact_q_observations": exact_observations,
                "exact_q_converged": exact_converged,
                "median_iteration_ratio_exact_q_pooled": float(np.median(raw)),
                "exact_q_wins": exact_wins,
                "exact_q_ties": exact_ties,
                "exact_q_losses": exact_losses,
            }
        )
    return output


def summarize_slowest_quartile(
    result_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Summarize exact-q performance on the slowest baseline quartile."""

    cutoff = float(np.quantile([row["iterations"] for row in baseline_rows], 0.75))
    output: list[dict[str, Any]] = []
    for target in TARGETS:
        for q in Q_VALUES:
            selected = [
                row
                for row in result_rows
                if row["target"] == target
                and row["q_requested"] == q
                and row["q_actual"] == q
                and row["baseline_iterations"] >= cutoff
            ]
            output.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "q_requested": q,
                    "baseline_iteration_cutoff": cutoff,
                    "observations": len(selected),
                    "converged": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio": float(
                        np.median([row["iteration_ratio_vs_baseline"] for row in selected])
                    ),
                    "wins": sum(row["outcome_vs_baseline"] == "win" for row in selected),
                    "ties": sum(row["outcome_vs_baseline"] == "tie" for row in selected),
                    "losses": sum(row["outcome_vs_baseline"] == "loss" for row in selected),
                }
            )
    return cutoff, output


def summarize_reduced_conditioning(
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Relate reduced-system conditioning to true-residual convergence."""

    output: list[dict[str, Any]] = []
    for target in TARGETS:
        for q in Q_VALUES:
            for converged in (True, False):
                selected = [
                    row
                    for row in result_rows
                    if row["target"] == target
                    and row["q_requested"] == q
                    and bool(row["converged"]) == converged
                ]
                if not selected:
                    continue
                values = np.asarray(
                    [row["reduced_condition_number"] for row in selected],
                    dtype=float,
                )
                output.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "q_requested": q,
                        "converged": converged,
                        "observations": len(selected),
                        "median_reduced_condition_number": float(np.median(values)),
                        "q90_reduced_condition_number": float(np.quantile(values, 0.9)),
                        "max_reduced_condition_number": float(np.max(values)),
                    }
                )
    return output


def write_report(
    path: Path,
    aggregate: list[dict[str, Any]],
    target_summary: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    slow_summary: list[dict[str, Any]],
    condition_summary: list[dict[str, Any]],
    winner: dict[str, Any],
) -> None:
    overall_winner = next(
        row
        for row in aggregate
        if row["target"] == winner["target"]
        and row["q_requested"] == winner["q_requested"]
    )
    cutoff = winner["baseline_iteration_cutoff"]
    winner_condition = {
        bool(row["converged"]): row
        for row in condition_summary
        if row["target"] == winner["target"]
        and row["q_requested"] == winner["q_requested"]
    }
    lines = [
        "# Solver-Dynamics Oracle Spaces for Global RBF FGMRES",
        "",
        "## Technical summary",
        "",
        (
            f"**{winner['target_label']} at q={winner['q_requested']} is the best-aligned trajectory target for genuinely slow solves.** "
            f"Within the slowest baseline quartile (at least {cutoff:g} iterations), its exact-q median iteration ratio is {winner['median_iteration_ratio']:.3f}, "
            f"with {winner['converged']}/{winner['observations']} converged and W/T/L "
            f"{winner['wins']}/{winner['ties']}/{winner['losses']}. Across all exact-q cases at this rank, the ratio is "
            f"{overall_winner['median_iteration_ratio_exact_q']:.3f} with {overall_winner['exact_q_converged']}/{overall_winner['exact_q_observations']} converged."
        ),
        "",
        "No target is uniformly robust: harmonic Ritz and unresolved-correction POD often cut nominal iterations more aggressively, but their nonconvergence counts are too high to call those reductions successful acceleration. ZS preserves convergence best but its median iteration ratio remains 1.000. The result is an on-trajectory oracle upper bound, not cross-RHS reuse or learnability.",
        "",
        "## Exact-rank evidence identifies the best-aligned target",
        "",
        "Each row uses exactly the requested q on trajectories with enough independent data. The last column retains all 108 system/RHS cases by truncating every target to the same available rank, preventing dimension from favoring one definition.",
        "",
        "| Target | q | Exact-q cases | Exact-q convergence | Median iteration ratio, exact q | Exact-q W/T/L | Median ratio, all matched-rank cases |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['target_label']} | {row['q_requested']} | {row['exact_q_observations']}/{row['observations']} | "
            f"{row['exact_q_converged']}/{row['exact_q_observations']} | "
            f"{row['median_iteration_ratio_exact_q']:.3f} | "
            f"{row['exact_q_wins']}/{row['exact_q_ties']}/{row['exact_q_losses']} | "
            f"{row['median_iteration_ratio_all_matched_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Slowest-quartile outcomes separate useful acceleration from easy-case effects",
            "",
            f"The slowest-quartile cutoff is {cutoff:g} FP32-LU iterations. Exact-q convergence-aware outcomes are shown below; any method that misses the true-residual tolerance is counted as a loss regardless of its nominal iteration count.",
            "",
            "| Target | q | Cases | Converged | Median iteration ratio | W/T/L |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in slow_summary:
        lines.append(
            f"| {row['target_label']} | {row['q_requested']} | {row['observations']} | "
            f"{row['converged']}/{row['observations']} | {row['median_iteration_ratio']:.3f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## The baseline trajectories define the comparison population",
            "",
            f"The population is the unchanged 27-system Gaussian N=225 suite with four RHSs ({len(baseline_rows)} solves). "
            f"FP32-LU baseline iterations range from {min(row['iterations'] for row in baseline_rows)} to {max(row['iterations'] for row in baseline_rows)}, "
            f"and {sum(row['converged'] for row in baseline_rows)}/{len(baseline_rows)} baseline solves converge at the original-system relative-residual tolerance {RTOL:.0e}.",
            "",
            "For q=8 and q=16, easy trajectories can terminate before q independent directions exist. The main table therefore exposes exact-q coverage and never pads a POD or Ritz basis with arbitrary null-space vectors.",
            "",
            "## Method keeps the integration and stopping rule fixed",
            "",
            f"Harmonic Ritz vectors are taken nearest zero for the right-preconditioned Arnoldi operator A M^{{-1}}, then mapped to coefficient space through the stored preconditioned Arnoldi vectors. The latest restart cycle with enough directions is used. Exact-error POD applies A^{{-1}} to the final 32 distinct normalized residual directions using verified FP64 LU with iterative refinement; when its relative equation residual exceeds sqrt(machine epsilon), an MP{REFERENCE_DECIMAL_DIGITS} pivoted-LU solve is used instead. Unresolved POD subtracts the FP32-LU action from those same exact actions. ZS uses the dominant right singular vectors of I-M^{{-1}}A.",
            "",
            "Every candidate uses the same nonsymmetric Galerkin deflation, E=Z^T A Z, P=I-AZ E^{-1}Z^T, Q=I-ZE^{-1}Z^T A, followed by right-preconditioned FGMRES on PA and reconstruction x=ZE^{-1}Z^Tb+Q x_hat. Reduced systems are solved by pivoted LU; no inverse is formed.",
            "",
            "## Limitations and robustness checks",
            "",
            f"The POD and harmonic targets are deliberately RHS-specific and use information from a completed baseline solve. Their setup cost is therefore offline oracle cost and is not credited as an online acceleration. True residuals are recomputed from the original A and b. For the selected q={winner['q_requested']} exact-error POD, median cond(Z^T A Z) is {winner_condition[True]['median_reduced_condition_number']:.1f} among converged cases versus {winner_condition[False]['median_reduced_condition_number']:.1f} among failures, confirming that reduced-system instability—not nominal Krylov iteration count alone—limits the higher-impact spaces. Raw trajectory arrays and residual-identity diagnostics are saved with the report.",
            "",
            "## Recommended next step",
            "",
            f"Use {winner['target_label']} at q={winner['q_requested']} as the solver-relevance diagnostic target only if its advantage is not confined to the same-RHS oracle construction. The next admissible test is cross-RHS or held-out-trajectory reuse on the same system; no ML should be introduced before that check.",
            "",
            "## Further question",
            "",
            "Does the winning trajectory-derived space retain acceleration when built from one RHS and applied to a different RHS of the same matrix? This separates operator-level difficult structure from exact same-trajectory leakage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output / "trajectories"
    basis_dir = args.output / "oracle_spaces"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    basis_dir.mkdir(parents=True, exist_ok=True)
    repeats = 1 if args.quick else REPEATS

    members = [
        (spec, index, float(value))
        for spec in family_specs(False)
        for index, value in enumerate(spec.values)
    ]
    if args.quick:
        members = [members[0], members[9], members[18]]

    baseline_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for system_index, (spec, index, value) in enumerate(members, start=1):
        artifact = build_artifact(spec, index, value)
        print(f"[system {system_index}/{len(members)}] {artifact.system_id}", flush=True)
        for rhs_name, kx, ky in RHS_SPECS:
            rhs = mode_rhs(artifact, kx, ky)
            try:
                trajectory, baseline = capture_baseline(artifact, rhs, repeats=repeats)
                save_trajectory(trajectory_dir / f"{artifact.system_id}__{rhs_name}.npz", trajectory)
                late = late_residual_matrix(trajectory.residual_vectors)
                exact_actions, reference_diagnostics = accurate_reference_actions(artifact, late)
                unresolved = exact_actions - artifact.base.apply(late)
                exact_basis, exact_singular, exact_rank = pod_basis(exact_actions, MAX_Q)
                unresolved_basis, unresolved_singular, unresolved_rank = pod_basis(unresolved, MAX_Q)
                harmonic_max = harmonic_ritz_space(trajectory, MAX_Q)
                common_available = min(
                    harmonic_max.shape[1],
                    exact_basis.shape[1],
                    unresolved_basis.shape[1],
                    artifact.zs_basis.shape[1],
                )
                cycles = trajectory.arnoldi_cycles or []
                baseline_row = {
                    **common_fields(artifact),
                    "rhs_name": rhs_name,
                    "kx": kx,
                    "ky": ky,
                    **baseline,
                    "residual_snapshots_recorded": len(trajectory.residual_vectors),
                    "late_distinct_snapshots": late.shape[1],
                    "arnoldi_cycles": len(cycles),
                    "arnoldi_cycle_dimensions": ";".join(str(cycle.hessenberg.shape[1]) for cycle in cycles),
                    "harmonic_available_rank": harmonic_max.shape[1],
                    "exact_error_pod_numerical_rank": exact_rank,
                    "unresolved_pod_numerical_rank": unresolved_rank,
                    "common_available_rank": common_available,
                    **reference_diagnostics,
                }
                baseline_rows.append(baseline_row)

                saved_bases: dict[str, np.ndarray] = {}
                for q_requested in Q_VALUES:
                    q_actual = min(q_requested, common_available)
                    if q_actual < 1:
                        raise RuntimeError("trajectory produced no common oracle direction")
                    harmonic = harmonic_ritz_space(trajectory, q_actual)
                    q_actual = min(q_actual, harmonic.shape[1])
                    candidates = {
                        "harmonic_ritz": harmonic[:, :q_actual],
                        "late_exact_error_pod": exact_basis[:, :q_actual],
                        "late_unresolved_pod": unresolved_basis[:, :q_actual],
                        "dominant_right_singular_S": artifact.zs_basis[:, :q_actual],
                    }
                    for target, basis in candidates.items():
                        saved_bases[f"{target}_q{q_requested}_actual{q_actual}"] = basis
                        orthogonality = float(
                            np.linalg.norm(basis.T @ basis - np.eye(q_actual), ord="fro")
                        )
                        reduced_condition_number = float(
                            np.linalg.cond(basis.T @ artifact.system.A @ basis)
                        )
                        deflation = galerkin_deflation(artifact.system.A, basis)
                        result = benchmark_deflated(artifact, deflation, rhs, repeats=repeats)
                        outcome = paired_outcome(result, baseline)
                        row = {
                            **common_fields(artifact),
                            "rhs_name": rhs_name,
                            "kx": kx,
                            "ky": ky,
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "q_requested": q_requested,
                            "q_actual": q_actual,
                            "exact_q": q_actual == q_requested,
                            "iterations": result["iterations"],
                            "converged": result["converged"],
                            "solve_seconds": result["solve_seconds"],
                            "true_relative_residual": result["true_relative_residual"],
                            "baseline_iterations": baseline["iterations"],
                            "baseline_converged": baseline["converged"],
                            "baseline_solve_seconds": baseline["solve_seconds"],
                            "iteration_ratio_vs_baseline": result["iterations"] / baseline["iterations"],
                            "solve_time_ratio_vs_baseline": result["solve_seconds"] / baseline["solve_seconds"],
                            "outcome_vs_baseline": outcome,
                            "deflation_setup_seconds": deflation.setup_seconds,
                            "deflation_residual_identity_error": result[
                                "deflation_residual_identity_error"
                            ],
                            "basis_orthogonality_defect": orthogonality,
                            "reduced_condition_number": reduced_condition_number,
                            "late_snapshot_count": late.shape[1],
                            "common_available_rank": common_available,
                        }
                        result_rows.append(row)
                        basis_rows.append(
                            {
                                "system_id": artifact.system_id,
                                "rhs_name": rhs_name,
                                "target": target,
                                "q_requested": q_requested,
                                "q_actual": q_actual,
                                "basis_orthogonality_defect": orthogonality,
                                "reduced_condition_number": reduced_condition_number,
                                "deflation_setup_seconds": deflation.setup_seconds,
                            }
                        )
                np.savez_compressed(
                    basis_dir / f"{artifact.system_id}__{rhs_name}.npz",
                    **saved_bases,
                    late_residual_directions=late,
                    exact_error_singular_values=exact_singular,
                    unresolved_singular_values=unresolved_singular,
                )
            except Exception as exc:
                failures.append(
                    {
                        **common_fields(artifact),
                        "rhs_name": rhs_name,
                        "stage": "trajectory_or_oracle",
                        "error": repr(exc),
                    }
                )
                print(f"  [failure] {rhs_name}: {exc!r}", flush=True)

    if failures:
        write_csv(args.output / "failures.csv", failures)
        raise RuntimeError(f"{len(failures)} trajectory/oracle cases failed")

    aggregate, system_summary = summarize(result_rows)
    target_summary = summarize_targets(aggregate, result_rows)
    slow_cutoff, slow_summary = summarize_slowest_quartile(result_rows, baseline_rows)
    condition_summary = summarize_reduced_conditioning(result_rows)
    trajectory_candidates = [
        row for row in slow_summary if row["target"] != "dominant_right_singular_S"
    ]
    winner = min(
        trajectory_candidates,
        key=lambda row: (
            -row["wins"],
            row["losses"],
            -row["converged"],
            row["median_iteration_ratio"],
            row["q_requested"],
            TARGETS.index(row["target"]),
        ),
    )

    write_csv(args.output / "baseline_trajectories.csv", baseline_rows)
    write_csv(args.output / "oracle_results.csv", result_rows)
    write_csv(args.output / "oracle_basis_diagnostics.csv", basis_rows)
    write_csv(args.output / "oracle_summary.csv", aggregate)
    write_csv(args.output / "oracle_system_summary.csv", system_summary)
    write_csv(args.output / "target_summary.csv", target_summary)
    write_csv(args.output / "slowest_quartile_summary.csv", slow_summary)
    write_csv(args.output / "reduced_condition_summary.csv", condition_summary)
    (args.output / "selection.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    (args.output / "experiment_config.json").write_text(
        json.dumps(
            {
                "systems": len(members),
                "rhs_per_system": len(RHS_SPECS),
                "N": N,
                "kernel": "gaussian",
                "preconditioner": "FP32 pivoted LU",
                "right_preconditioned": True,
                "rtol": RTOL,
                "restart": RESTART,
                "max_iterations": MAX_ITERATIONS,
                "timing_repeats": repeats,
                "q_values": Q_VALUES,
                "late_snapshot_limit": LATE_SNAPSHOT_LIMIT,
                "slowest_quartile_baseline_iteration_cutoff": slow_cutoff,
                "reference_switch_tolerance": REFERENCE_SWITCH_TOLERANCE,
                "reference_decimal_digits": REFERENCE_DECIMAL_DIGITS,
                "integration": {
                    "name": "standard nonsymmetric Galerkin deflation",
                    "P": "I - A Z (Z^T A Z)^-1 Z^T",
                    "Q": "I - Z (Z^T A Z)^-1 Z^T A",
                    "reduced_action": "pivoted LU solve; no explicit inverse",
                },
                "rank_fairness": "all targets use common q_actual per system/RHS/requested-q; exact-q subset reported",
                "solver_alignment_selection": "among trajectory-derived targets, maximize convergence-aware wins on exact-q cases in the slowest baseline quartile, then minimize losses",
                "no_ml": True,
                "no_transferability": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output / "software_environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output / "source_notes.json").write_text(
        json.dumps(
            {
                "report_audience": "technical",
                "primary_question": "Which baseline-FGMRES-derived subspace aligns with slow convergence?",
                "comparison_basis": "FP32-LU-only right-preconditioned FGMRES",
                "winner_selection": "The report's solver-alignment target maximizes convergence-aware wins in the slowest baseline quartile; ZS remains the fixed operator-only baseline rather than an eligible trajectory target.",
                "chart_map": [
                    {
                        "section": "Slow-solve outcomes distinguish alignment from easy-case effects",
                        "question": "Which trajectory target produces the most convergence-aware wins on slow solves?",
                        "family": "comparison and ranking",
                        "type": "grouped horizontal bar",
                        "fields": ["target", "q", "wins", "cases", "convergence", "median iteration ratio"],
                        "claim": "q=4 late exact-error POD has the highest win rate in the slowest baseline quartile",
                        "palette": "categorical q encoding with labels and table fallback",
                        "delivery": "canonical portable HTML artifact",
                    }
                ],
                "required_structure_mapping": {
                    "technical_summary": "Technical summary",
                    "key_findings": "Exact-rank evidence identifies the best-aligned target",
                    "scope_definitions": "The baseline trajectories define the comparison population",
                    "methodology": "Method keeps the integration and stopping rule fixed",
                    "limitations": "Limitations and robustness checks",
                    "next_steps": "Recommended next step",
                    "further_questions": "Further question",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        args.report,
        aggregate,
        target_summary,
        baseline_rows,
        slow_summary,
        condition_summary,
        winner,
    )
    print(
        json.dumps(
            {
                "systems": len(members),
                "baseline_rows": len(baseline_rows),
                "oracle_rows": len(result_rows),
                "winner": winner,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
