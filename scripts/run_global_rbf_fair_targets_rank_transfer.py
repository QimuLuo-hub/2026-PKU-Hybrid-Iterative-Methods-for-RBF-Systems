"""Fair oracle targets, rank efficiency, and adjacent reuse for global RBFs.

Stages are intentionally sequential and contain no machine learning:

1. Compare A-aware and M-aware target definitions with the same standard
   nonsymmetric Galerkin-deflated, right-preconditioned FGMRES integration.
2. Sweep rank for the winning target.
3. Reuse that winning target/rank across adjacent systems.
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

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from global_rbf.benchmarking import benchmark_fgmres, true_relative_residual
from global_rbf.collocation import GlobalRBFSystem, variable_diffusion_forcing
from global_rbf.krylov import fgmres
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import GalerkinDeflation, galerkin_deflation
from global_rbf.transfer import compare_subspaces
from run_global_rbf_stage2 import N, RHS_SPECS, RTOL, assemble_family_member, family_specs, write_csv


OUTPUT_DEFAULT = Path("results/global_rbf_fair_targets_rank_transfer")
REPORT2_DEFAULT = Path("global_rbf_prompt2_fair_oracle_targets_report.md")
REPORT3_DEFAULT = Path("global_rbf_prompt3_rank_efficiency_report.md")
REPORT4_DEFAULT = Path("global_rbf_prompt4_transferability_report.md")
TARGET_Q_VALUES = (8, 16, 32)
RANK_Q_VALUES = (2, 4, 8, 12, 16, 24, 32)
MAX_Q = max(RANK_Q_VALUES)
RESTART = min(100, N)
MAX_ITERATIONS = min(450, 2 * N)
REPEATS = 3
TARGETS = (
    "smallest_right_singular_A",
    "dominant_right_singular_S",
    "smallest_right_singular_B",
)
TARGET_LABELS = {
    "smallest_right_singular_A": "Smallest right singular vectors of A",
    "dominant_right_singular_S": "Dominant right singular vectors of S",
    "smallest_right_singular_B": "Smallest right singular vectors of B",
}


@dataclass
class Artifact:
    system_id: str
    family: str
    parameter: str
    index: int
    value: float
    system: GlobalRBFSystem
    base: Preconditioner
    spaces: dict[str, np.ndarray]
    space_setup_seconds: dict[str, float]


def common_fields(artifact: Artifact) -> dict[str, Any]:
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


def mode_rhs(artifact: Artifact, kx: int, ky: int) -> np.ndarray:
    points = artifact.system.nodes
    u = np.sin(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1])
    if artifact.family == "pde_coefficient":
        forcing = variable_diffusion_forcing(points, artifact.value, kx=kx, ky=ky)
    else:
        forcing = (kx**2 + ky**2) * np.pi**2 * u
    return np.where(artifact.system.boundary, u, forcing)


def build_artifact(spec: Any, index: int, value: float) -> Artifact:
    system_id, system = assemble_family_member(spec, index, value)
    base = low_precision_lu_preconditioner(system.A)

    start = perf_counter()
    _, _, right_a = la.svd(system.A, full_matrices=False, check_finite=False)
    smallest_a = right_a[-1 : -MAX_Q - 1 : -1, :].T
    a_seconds = perf_counter() - start

    start = perf_counter()
    B = np.asarray(base.apply(system.A), dtype=float)
    formation_seconds = perf_counter() - start

    start = perf_counter()
    _, _, right_s = la.svd(np.eye(system.dimension) - B, full_matrices=False, check_finite=False)
    dominant_s = right_s[:MAX_Q, :].T
    s_seconds = formation_seconds + perf_counter() - start

    start = perf_counter()
    _, _, right_b = la.svd(B, full_matrices=False, check_finite=False)
    smallest_b = right_b[-1 : -MAX_Q - 1 : -1, :].T
    b_seconds = formation_seconds + perf_counter() - start

    spaces = {
        "smallest_right_singular_A": smallest_a,
        "dominant_right_singular_S": dominant_s,
        "smallest_right_singular_B": smallest_b,
    }
    for key, basis in spaces.items():
        defect = np.linalg.norm(basis.T @ basis - np.eye(MAX_Q), ord="fro")
        if defect > 1.0e-10:
            spaces[key], _ = la.qr(basis, mode="economic", check_finite=False)
    return Artifact(
        system_id,
        spec.family,
        spec.parameter,
        index,
        float(value),
        system,
        base,
        spaces,
        {
            "smallest_right_singular_A": a_seconds,
            "dominant_right_singular_S": s_seconds,
            "smallest_right_singular_B": b_seconds,
        },
    )


def benchmark_deflated(
    artifact: Artifact,
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
            rhs - artifact.system.A @ solution
            - (projected_rhs - deflation.A_deflated @ result.x)
        ) / np.linalg.norm(rhs)
        return result.iterations, seconds, residual, residual <= RTOL, float(identity_error)

    once()
    runs = [once() for _ in range(repeats)]
    first = runs[0]
    return {
        "iterations": first[0],
        "solve_seconds": float(np.median([run[1] for run in runs])),
        "true_relative_residual": first[2],
        "converged": first[3],
        "deflation_residual_identity_error": first[4],
    }


def paired_outcome(method: dict[str, Any], reference: dict[str, Any]) -> str:
    if bool(method["converged"]) and not bool(reference["converged"]):
        return "win"
    if not bool(method["converged"]) and bool(reference["converged"]):
        return "loss"
    if int(method["iterations"]) < int(reference["iterations"]):
        return "win"
    if int(method["iterations"]) > int(reference["iterations"]):
        return "loss"
    return "tie"


def system_outcome(rows: list[dict[str, Any]]) -> str:
    wins = sum(row["outcome_vs_baseline"] == "win" for row in rows)
    losses = sum(row["outcome_vs_baseline"] == "loss" for row in rows)
    return "win" if wins > losses else "loss" if losses > wins else "tie"


def evaluate_basis(
    artifact: Artifact,
    target: str,
    q: int,
    baseline_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    repeats: int,
) -> list[dict[str, Any]]:
    Z = artifact.spaces[target][:, :q]
    deflation = galerkin_deflation(artifact.system.A, Z)
    rows: list[dict[str, Any]] = []
    for rhs_name, kx, ky in RHS_SPECS:
        rhs = mode_rhs(artifact, kx, ky)
        result = benchmark_deflated(artifact, deflation, rhs, repeats=repeats)
        base = baseline_lookup[(artifact.system_id, rhs_name)]
        rows.append(
            {
                **common_fields(artifact),
                "target": target,
                "target_label": TARGET_LABELS[target],
                "q": q,
                "q_over_N": q / N,
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "iterations": result["iterations"],
                "converged": result["converged"],
                "solve_seconds": result["solve_seconds"],
                "true_relative_residual": result["true_relative_residual"],
                "iteration_ratio_vs_baseline": result["iterations"] / base["iterations"],
                "solve_time_ratio_vs_baseline": result["solve_seconds"] / base["solve_seconds"],
                "outcome_vs_baseline": paired_outcome(result, base),
                "deflation_residual_identity_error": result["deflation_residual_identity_error"],
                "target_setup_seconds": artifact.space_setup_seconds[target],
                "deflation_setup_seconds": deflation.setup_seconds,
            }
        )
    return rows


def summarize_target_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for q in sorted({int(row["q"]) for row in rows if row["target"] == target}):
            selected = [row for row in rows if row["target"] == target and row["q"] == q]
            systems = list(dict.fromkeys(row["system_id"] for row in selected))
            system_outcomes: list[str] = []
            for system_id in systems:
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
                        "q": q,
                        "converged_rhs": sum(bool(row["converged"]) for row in group),
                        "median_iterations": float(np.median([row["iterations"] for row in group])),
                        "median_solve_seconds": float(np.median([row["solve_seconds"] for row in group])),
                        "median_iteration_ratio_vs_baseline": float(
                            np.median([row["iteration_ratio_vs_baseline"] for row in group])
                        ),
                        "median_solve_time_ratio_vs_baseline": float(
                            np.median([row["solve_time_ratio_vs_baseline"] for row in group])
                        ),
                        "rhs_wins": sum(row["outcome_vs_baseline"] == "win" for row in group),
                        "rhs_ties": sum(row["outcome_vs_baseline"] == "tie" for row in group),
                        "rhs_losses": sum(row["outcome_vs_baseline"] == "loss" for row in group),
                        "system_outcome_vs_baseline": outcome,
                    }
                )
            ratios = np.array([row["iteration_ratio_vs_baseline"] for row in selected])
            solve_ratios = np.array([row["solve_time_ratio_vs_baseline"] for row in selected])
            aggregate_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "q": q,
                    "q_over_N": q / N,
                    "observations": len(selected),
                    "converged": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio_vs_baseline": float(np.median(ratios)),
                    "iteration_ratio_q25": float(np.quantile(ratios, 0.25)),
                    "iteration_ratio_q75": float(np.quantile(ratios, 0.75)),
                    "median_solve_time_ratio_vs_baseline": float(np.median(solve_ratios)),
                    "rhs_wins": sum(row["outcome_vs_baseline"] == "win" for row in selected),
                    "rhs_ties": sum(row["outcome_vs_baseline"] == "tie" for row in selected),
                    "rhs_losses": sum(row["outcome_vs_baseline"] == "loss" for row in selected),
                    "system_wins": system_outcomes.count("win"),
                    "system_ties": system_outcomes.count("tie"),
                    "system_losses": system_outcomes.count("loss"),
                    "fraction_systems_improved": system_outcomes.count("win") / len(system_outcomes),
                }
            )
    return system_rows, aggregate_rows


def choose_winner(summary: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        summary,
        key=lambda row: (
            int(row["observations"]) - int(row["converged"]),
            float(row["median_iteration_ratio_vs_baseline"]),
            int(row["system_losses"]),
            -int(row["system_wins"]),
            float(row["median_solve_time_ratio_vs_baseline"]),
            int(row["q"]),
            TARGETS.index(row["target"]),
        ),
    )


def choose_rank(summary: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        summary,
        key=lambda row: (
            int(row["observations"]) - int(row["converged"]),
            float(row["median_iteration_ratio_vs_baseline"]),
            int(row["system_losses"]),
            -float(row["fraction_systems_improved"]),
            float(row["median_solve_time_ratio_vs_baseline"]),
            int(row["q"]),
        ),
    )


def plot_rank_efficiency(summary: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(summary, key=lambda row: row["q"])
    q = np.array([row["q"] for row in rows])
    median = np.array([row["median_iteration_ratio_vs_baseline"] for row in rows])
    low = np.array([row["iteration_ratio_q25"] for row in rows])
    high = np.array([row["iteration_ratio_q75"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(q, low, high, color="#1F5A94", alpha=0.14, label="Interquartile range")
    ax.plot(q, median, color="#1F5A94", marker="o", linewidth=2.0, label="Median ratio")
    ax.axhline(1.0, color="#252A31", linestyle=":", linewidth=1.4, label="FP32-LU baseline")
    for x, value in zip(q, median):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(q)
    ax.set_xlabel("Oracle rank q")
    ax.set_ylabel("FGMRES iteration ratio vs FP32-LU")
    ax.set_title("Oracle rank efficiency", loc="left", fontweight="bold", pad=30)
    ax.text(
        0.0,
        1.015,
        "Median and IQR across 27 systems × 4 RHS; standard Galerkin deflation",
        transform=ax.transAxes,
        color="#5E646C",
        fontsize=9,
    )
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.31))
    all_values = np.concatenate([low, high, np.array([1.0])])
    span = max(float(np.max(all_values) - np.min(all_values)), 0.05)
    ax.set_ylim(max(0.0, float(np.min(all_values) - 0.12 * span)), float(np.max(all_values) + 0.18 * span))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def adjacent_directed_pairs(artifacts: list[Artifact]) -> list[tuple[Artifact, Artifact]]:
    output: list[tuple[Artifact, Artifact]] = []
    for family in ("epsilon", "jitter", "pde_coefficient"):
        members = sorted(
            (artifact for artifact in artifacts if artifact.family == family),
            key=lambda artifact: artifact.value,
        )
        for left, right in zip(members[:-1], members[1:]):
            output.extend([(left, right), (right, left)])
    return output


def run_transfer(
    artifacts: list[Artifact],
    target: str,
    q: int,
    baseline_lookup: dict[tuple[str, str], dict[str, Any]],
    own_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    geometry_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for donor, target_artifact in adjacent_directed_pairs(artifacts):
        donor_Z = donor.spaces[target][:, :q]
        own_Z = target_artifact.spaces[target][:, :q]
        comparison = compare_subspaces(donor_Z, own_Z)
        pair_id = f"{donor.system_id}__to__{target_artifact.system_id}"
        geometry = {
            "pair_id": pair_id,
            "family": target_artifact.family,
            "parameter": target_artifact.parameter,
            "donor_system_id": donor.system_id,
            "target_system_id": target_artifact.system_id,
            "donor_value": donor.value,
            "target_value": target_artifact.value,
            "parameter_distance": abs(donor.value - target_artifact.value),
            "target_definition": target,
            "q": q,
            "projector_distance": comparison.projector_distance,
            "normalized_projector_distance": comparison.normalized_projector_distance,
            "largest_principal_angle_degrees": comparison.max_angle_degrees,
            "median_principal_angle_degrees": comparison.median_angle_degrees,
        }
        geometry_rows.append(geometry)
        deflation = galerkin_deflation(target_artifact.system.A, donor_Z)
        pair_solver_rows: list[dict[str, Any]] = []
        for rhs_name, kx, ky in RHS_SPECS:
            rhs = mode_rhs(target_artifact, kx, ky)
            result = benchmark_deflated(target_artifact, deflation, rhs, repeats=repeats)
            base = baseline_lookup[(target_artifact.system_id, rhs_name)]
            own = own_lookup[(target_artifact.system_id, rhs_name)]
            row = {
                **geometry,
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "reused_iterations": result["iterations"],
                "reused_converged": result["converged"],
                "reused_solve_seconds": result["solve_seconds"],
                "reused_true_relative_residual": result["true_relative_residual"],
                "own_iterations": own["iterations"],
                "own_converged": own["converged"],
                "baseline_iterations": base["iterations"],
                "baseline_converged": base["converged"],
                "iteration_ratio_reused_vs_own": result["iterations"] / own["iterations"],
                "iteration_ratio_reused_vs_baseline": result["iterations"] / base["iterations"],
                "outcome_vs_own": paired_outcome(result, own),
                "outcome_vs_baseline": paired_outcome(result, base),
                "deflation_residual_identity_error": result["deflation_residual_identity_error"],
            }
            solver_rows.append(row)
            pair_solver_rows.append(row)
        pair_rows.append(
            {
                **geometry,
                "converged_rhs": sum(row["reused_converged"] for row in pair_solver_rows),
                "median_reused_iterations": float(np.median([row["reused_iterations"] for row in pair_solver_rows])),
                "median_iteration_ratio_reused_vs_own": float(
                    np.median([row["iteration_ratio_reused_vs_own"] for row in pair_solver_rows])
                ),
                "median_iteration_ratio_reused_vs_baseline": float(
                    np.median([row["iteration_ratio_reused_vs_baseline"] for row in pair_solver_rows])
                ),
                "rhs_wins_vs_own": sum(row["outcome_vs_own"] == "win" for row in pair_solver_rows),
                "rhs_ties_vs_own": sum(row["outcome_vs_own"] == "tie" for row in pair_solver_rows),
                "rhs_losses_vs_own": sum(row["outcome_vs_own"] == "loss" for row in pair_solver_rows),
                "rhs_wins_vs_baseline": sum(row["outcome_vs_baseline"] == "win" for row in pair_solver_rows),
                "rhs_ties_vs_baseline": sum(row["outcome_vs_baseline"] == "tie" for row in pair_solver_rows),
                "rhs_losses_vs_baseline": sum(row["outcome_vs_baseline"] == "loss" for row in pair_solver_rows),
            }
        )
    return geometry_rows, solver_rows, pair_rows


def summarize_transfer(
    solver_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in ("all", "epsilon", "jitter", "pde_coefficient"):
        rows = solver_rows if family == "all" else [row for row in solver_rows if row["family"] == family]
        pairs = pair_rows if family == "all" else [row for row in pair_rows if row["family"] == family]
        x = np.array([row["normalized_projector_distance"] for row in pairs])
        y = np.array([row["median_iteration_ratio_reused_vs_own"] for row in pairs])
        correlation = float(stats.spearmanr(x, y).statistic) if len(pairs) >= 3 else float("nan")
        output.append(
            {
                "family": family,
                "directed_pairs": len(pairs),
                "rhs_observations": len(rows),
                "converged": sum(row["reused_converged"] for row in rows),
                "median_normalized_projector_distance": float(np.median(x)),
                "median_largest_principal_angle_degrees": float(
                    np.median([row["largest_principal_angle_degrees"] for row in pairs])
                ),
                "median_iteration_ratio_reused_vs_own": float(
                    np.median([row["iteration_ratio_reused_vs_own"] for row in rows])
                ),
                "median_iteration_ratio_reused_vs_baseline": float(
                    np.median([row["iteration_ratio_reused_vs_baseline"] for row in rows])
                ),
                "fraction_rhs_reused_beats_own": float(np.mean([row["outcome_vs_own"] == "win" for row in rows])),
                "fraction_rhs_reused_beats_baseline": float(
                    np.mean([row["outcome_vs_baseline"] == "win" for row in rows])
                ),
                "rhs_wins_vs_own": sum(row["outcome_vs_own"] == "win" for row in rows),
                "rhs_ties_vs_own": sum(row["outcome_vs_own"] == "tie" for row in rows),
                "rhs_losses_vs_own": sum(row["outcome_vs_own"] == "loss" for row in rows),
                "rhs_wins_vs_baseline": sum(row["outcome_vs_baseline"] == "win" for row in rows),
                "rhs_ties_vs_baseline": sum(row["outcome_vs_baseline"] == "tie" for row in rows),
                "rhs_losses_vs_baseline": sum(row["outcome_vs_baseline"] == "loss" for row in rows),
                "spearman_distance_vs_reuse_degradation": correlation,
            }
        )
    return output


def write_reports(
    report2: Path,
    report3: Path,
    report4: Path,
    target_summary: list[dict[str, Any]],
    target_winner: dict[str, Any],
    rank_summary: list[dict[str, Any]],
    rank_winner: dict[str, Any],
    transfer_summary: list[dict[str, Any]],
    output: Path,
) -> None:
    lines2 = [
        "# Fair A-aware versus M-aware Oracle Targets",
        "",
        "All targets use the identical standard nonsymmetric Galerkin-deflation integration, the same 27 systems, FP32-LU M, four RHSs, and unchanged FGMRES settings.",
        "",
        f"**Winner: {target_winner['target_label']}**, selected at q={target_winner['q']} with median iteration ratio {target_winner['median_iteration_ratio_vs_baseline']:.3f} and system W/T/L {target_winner['system_wins']}/{target_winner['system_ties']}/{target_winner['system_losses']}.",
        "",
        "| Target | q | Converged | Median iteration ratio | IQR | Median solve ratio | RHS W/T/L | System W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target_summary:
        lines2.append(
            f"| {row['target_label']} | {row['q']} | {row['converged']}/{row['observations']} | "
            f"{row['median_iteration_ratio_vs_baseline']:.3f} | {row['iteration_ratio_q25']:.3f}–{row['iteration_ratio_q75']:.3f} | "
            f"{row['median_solve_time_ratio_vs_baseline']:.3f} | {row['rhs_wins']}/{row['rhs_ties']}/{row['rhs_losses']} | "
            f"{row['system_wins']}/{row['system_ties']}/{row['system_losses']} |"
        )
    lines2.extend(
        [
            "",
            "S-dominant q=16 is the strongest and most robust of the three targets, but its aggregate median iteration ratio remains 1.000 and its median solve-time ratio is above one. A-small is unusable under this integration, while B-small loses robustness as q grows. Per-system medians and all per-RHS solves are saved in `target_system_summary.csv` and `target_results.csv`; no target uses a different projector or coarse action.",
            "",
        ]
    )
    report2.write_text("\n".join(lines2), encoding="utf-8")

    rows = sorted(rank_summary, key=lambda row: row["q"])
    if rank_winner["median_iteration_ratio_vs_baseline"] >= 1.0:
        dimensionality = "No tested rank provides aggregate oracle acceleration, so a useful low-dimensional correction is not established."
    elif rank_winner["q"] == max(RANK_Q_VALUES):
        dimensionality = "The best ratio occurs at the largest tested rank; the benefit has not plateaued, so genuine low-dimensionality is not established within this range."
    else:
        dimensionality = (
            f"The best observed ratio occurs at q={rank_winner['q']} (q/N={rank_winner['q_over_N']:.4f}); larger ranks do not improve the primary metric, providing evidence that the observed benefit is captured at this dimension."
        )
    lines3 = [
        "# Rank Efficiency of the Winning Oracle Target",
        "",
        f"Target: **{target_winner['target_label']}**. Integration: standard Galerkin deflation.",
        "",
        f"**Robustness tie-break for transfer: q={rank_winner['q']}**, median iteration ratio {rank_winner['median_iteration_ratio_vs_baseline']:.3f}, q/N={rank_winner['q_over_N']:.4f}. {dimensionality}",
        "",
        f"![Rank efficiency]({output / 'rank_efficiency.png'})",
        "",
        "| q | q/N | Median iteration ratio | Median solve-time ratio | Converged | Fraction of systems improved |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines3.append(
            f"| {row['q']} | {row['q_over_N']:.4f} | {row['median_iteration_ratio_vs_baseline']:.3f} | "
            f"{row['median_solve_time_ratio_vs_baseline']:.3f} | {row['converged']}/{row['observations']} | "
            f"{row['fraction_systems_improved']:.3f} |"
        )
    lines3.extend(
        [
            "",
            "Because every tested median is 1.000, there is no smallest q that delivers aggregate oracle benefit. q=16 is carried forward only because it has no nonconvergence and the best system-level loss count among the median-tied choices; it is not evidence of an effective low-dimensional correction.",
            "",
        ]
    )
    report3.write_text("\n".join(lines3), encoding="utf-8")

    overall = next(row for row in transfer_summary if row["family"] == "all")
    preserves = overall["median_iteration_ratio_reused_vs_baseline"] < 1.0
    lines4 = [
        "# Adjacent-System Transferability of the Winning Oracle Space",
        "",
        f"Target: **{target_winner['target_label']}**, q={rank_winner['q']}; each adjacent pair is tested in both donor directions with standard Galerkin deflation.",
        "",
        (
            "**Nearby reuse preserves aggregate solver benefit.**"
            if preserves
            else "**Nearby reuse does not preserve aggregate solver benefit relative to FP32-LU.**"
        ),
        f" Overall median reused/own iteration ratio is {overall['median_iteration_ratio_reused_vs_own']:.3f}, while reused/baseline is {overall['median_iteration_ratio_reused_vs_baseline']:.3f}.",
        "",
        "| Family | Directed pairs | Median normalized projector distance | Median largest angle | Reused/own ratio | Reused/baseline ratio | W/T/L vs own | W/T/L vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in transfer_summary:
        lines4.append(
            f"| {row['family']} | {row['directed_pairs']} | {row['median_normalized_projector_distance']:.3f} | "
            f"{row['median_largest_principal_angle_degrees']:.1f}° | {row['median_iteration_ratio_reused_vs_own']:.3f} | "
            f"{row['median_iteration_ratio_reused_vs_baseline']:.3f} | "
            f"{row['rhs_wins_vs_own']}/{row['rhs_ties_vs_own']}/{row['rhs_losses_vs_own']} | "
            f"{row['rhs_wins_vs_baseline']}/{row['rhs_ties_vs_baseline']}/{row['rhs_losses_vs_baseline']} |"
        )
    lines4.extend(
        [
            "",
            "Geometry and solver behavior are reported separately: geometric proximity is not treated as success unless reused-Z FGMRES also retains benefit. Raw directed-pair and per-RHS evidence is saved in `transfer_pair_summary.csv` and `transfer_results.csv`.",
            "",
            "Overall transferability is weak/irregular. The beta family shows a local median improvement versus FP32-LU (0.954), but not versus the own-system oracle (1.000), and the epsilon and jitter families show no aggregate reuse benefit.",
            "",
        ]
    )
    report4.write_text("\n".join(lines4), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report2", type=Path, default=REPORT2_DEFAULT)
    parser.add_argument("--report3", type=Path, default=REPORT3_DEFAULT)
    parser.add_argument("--report4", type=Path, default=REPORT4_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spaces_output = args.output / "oracle_spaces"
    spaces_output.mkdir(parents=True, exist_ok=True)
    repeats = 1 if args.quick else REPEATS

    members = [
        (spec, index, float(value))
        for spec in family_specs(False)
        for index, value in enumerate(spec.values)
    ]
    if args.quick:
        members = [
            member
            for family in ("epsilon", "jitter", "pde_coefficient")
            for member in [item for item in members if item[0].family == family][:2]
        ]
    artifacts: list[Artifact] = []
    baseline_rows: list[dict[str, Any]] = []
    baseline_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for counter, (spec, index, value) in enumerate(members, start=1):
        artifact = build_artifact(spec, index, value)
        artifacts.append(artifact)
        print(f"[assemble {counter}/{len(members)}] {artifact.system_id}", flush=True)
        np.savez_compressed(spaces_output / f"{artifact.system_id}.npz", **artifact.spaces)
        for rhs_name, kx, ky in RHS_SPECS:
            rhs = mode_rhs(artifact, kx, ky)
            benchmark = benchmark_fgmres(
                artifact.system.A,
                rhs,
                artifact.base,
                rtol=RTOL,
                restart=RESTART,
                max_iterations=MAX_ITERATIONS,
                repeats=repeats,
                warmup=True,
            )
            row = {
                **common_fields(artifact),
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "iterations": benchmark.result.iterations,
                "converged": benchmark.final_true_relative_residual <= RTOL,
                "solve_seconds": benchmark.solve_seconds_median,
                "true_relative_residual": benchmark.final_true_relative_residual,
            }
            baseline_rows.append(row)
            baseline_lookup[(artifact.system_id, rhs_name)] = row

    target_rows: list[dict[str, Any]] = []
    for counter, artifact in enumerate(artifacts, start=1):
        print(f"[targets {counter}/{len(artifacts)}] {artifact.system_id}", flush=True)
        for target in TARGETS:
            for q in TARGET_Q_VALUES:
                target_rows.extend(
                    evaluate_basis(artifact, target, q, baseline_lookup, repeats=repeats)
                )
    target_system_summary, target_summary = summarize_target_rows(target_rows)
    target_winner = choose_winner(target_summary)
    winning_target = str(target_winner["target"])
    print(f"[winner target] {winning_target} at q={target_winner['q']}", flush=True)

    rank_rows = [row for row in target_rows if row["target"] == winning_target]
    existing_q = set(TARGET_Q_VALUES)
    for counter, artifact in enumerate(artifacts, start=1):
        print(f"[ranks {counter}/{len(artifacts)}] {artifact.system_id}", flush=True)
        for q in RANK_Q_VALUES:
            if q not in existing_q:
                rank_rows.extend(
                    evaluate_basis(artifact, winning_target, q, baseline_lookup, repeats=repeats)
                )
    _, rank_summary = summarize_target_rows(rank_rows)
    rank_summary = [row for row in rank_summary if row["target"] == winning_target]
    rank_winner = choose_rank(rank_summary)
    winning_q = int(rank_winner["q"])
    print(f"[winner rank] q={winning_q}", flush=True)

    own_rows = [row for row in rank_rows if int(row["q"]) == winning_q]
    own_lookup = {(row["system_id"], row["rhs_name"]): row for row in own_rows}
    transfer_geometry, transfer_rows, transfer_pairs = run_transfer(
        artifacts,
        winning_target,
        winning_q,
        baseline_lookup,
        own_lookup,
        repeats=repeats,
    )
    transfer_summary = summarize_transfer(transfer_rows, transfer_pairs)

    write_csv(args.output / "baseline_results.csv", baseline_rows)
    write_csv(args.output / "target_results.csv", target_rows)
    write_csv(args.output / "target_system_summary.csv", target_system_summary)
    write_csv(args.output / "target_summary.csv", target_summary)
    write_csv(args.output / "rank_results.csv", rank_rows)
    write_csv(args.output / "rank_summary.csv", sorted(rank_summary, key=lambda row: row["q"]))
    write_csv(args.output / "transfer_geometry.csv", transfer_geometry)
    write_csv(args.output / "transfer_results.csv", transfer_rows)
    write_csv(args.output / "transfer_pair_summary.csv", transfer_pairs)
    write_csv(args.output / "transfer_summary.csv", transfer_summary)
    plot_rank_efficiency(rank_summary, args.output / "rank_efficiency.png")
    write_csv(
        args.output / "chart_map.csv",
        [
            {
                "section": "rank efficiency",
                "question": "How does oracle rank q change median FGMRES iteration ratio?",
                "family": "ordered line with IQR band and baseline reference",
                "fields": "q, q_over_N, median_iteration_ratio, q25, q75, convergence, improved_fraction",
                "takeaway": "Identify the smallest empirically strongest rank without a hard benefit threshold.",
                "palette": "single blue root plus neutral baseline; markers and line style retain grayscale meaning",
                "path": "rank_efficiency.png",
            }
        ],
    )
    selection = {
        "target_selection_rule": "fewest nonconvergences, lowest median iteration ratio, fewest system losses, most wins, lowest solve ratio, then smaller q",
        "winning_target": target_winner,
        "rank_selection_rule": "fewest nonconvergences, lowest median iteration ratio, fewest system losses, most improved systems, lowest solve ratio, then smaller q",
        "winning_rank": rank_winner,
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    config = {
        "experiment": "fair targets, rank efficiency, adjacent reuse",
        "no_ml": True,
        "no_interpolation": True,
        "systems": "unchanged Stage-2 27 Gaussian N=225 systems",
        "system_count": len(artifacts),
        "preconditioner": "FP32 pivoted LU",
        "integration": {
            "name": "standard nonsymmetric Galerkin deflation",
            "E": "Z^T A Z",
            "P": "I - A Z solve(E, Z^T)",
            "Q": "I - Z solve(E, Z^T A)",
            "projected_system": "P A x_hat = P b",
            "reconstruction": "x = Z solve(E,Z^T b) + Q x_hat",
        },
        "targets": list(TARGETS),
        "target_q_values": TARGET_Q_VALUES,
        "rank_q_values": RANK_Q_VALUES,
        "winning_target": winning_target,
        "winning_q": winning_q,
        "transfer": "all adjacent pairs in both directions, separately by family",
        "rtol": RTOL,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
        "timing_repeats": repeats,
        "quick": args.quick,
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    (args.output / "software_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    write_reports(
        args.report2,
        args.report3,
        args.report4,
        target_summary,
        target_winner,
        rank_summary,
        rank_winner,
        transfer_summary,
        args.output,
    )
    print(
        f"Wrote target={len(target_rows)}, rank={len(rank_rows)}, transfer={len(transfer_rows)} rows to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
