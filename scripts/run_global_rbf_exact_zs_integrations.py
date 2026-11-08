"""Compare four integrations of the exact q=8 S-oracle on 27 systems.

No learned model is used.  The systems, right-hand sides, FP32-LU
preconditioner, and FGMRES tolerances are inherited from Stage 2.
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

import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.benchmarking import benchmark_fgmres, true_relative_residual
from global_rbf.collocation import GlobalRBFSystem, variable_diffusion_forcing
from global_rbf.krylov import fgmres
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import (
    GalerkinDeflation,
    additive_galerkin_coarse_preconditioner,
    galerkin_deflation,
    residual_minimizing_coarse_preconditioner,
)
from run_global_rbf_stage2 import (
    N,
    RHS_SPECS,
    RTOL,
    assemble_family_member,
    family_specs,
    write_csv,
)


OUTPUT_DEFAULT = Path("results/global_rbf_exact_ZS_coarse_integrations")
REPORT_DEFAULT = Path("global_rbf_exact_ZS_coarse_integration_report.md")
Q = 8
RESTART = min(100, N)
MAX_ITERATIONS = min(450, 2 * N)
REPEATS = 3
METHODS = (
    "baseline_fp32_lu",
    "additive_galerkin",
    "additive_residual_minimizing",
    "standard_galerkin_deflation",
)
LABELS = {
    "baseline_fp32_lu": "M only",
    "additive_galerkin": "Additive Galerkin",
    "additive_residual_minimizing": "Additive residual-minimizing",
    "standard_galerkin_deflation": "Standard Galerkin deflation",
}


@dataclass
class SystemArtifact:
    system_id: str
    family: str
    parameter: str
    value: float
    system: GlobalRBFSystem
    base: Preconditioner
    Z: np.ndarray
    oracle_setup_seconds: float
    oracle_sigma_q: float
    oracle_sigma_q_plus_1: float


def mode_rhs(artifact: SystemArtifact, kx: int, ky: int) -> np.ndarray:
    points = artifact.system.nodes
    u = np.sin(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1])
    if artifact.family == "pde_coefficient":
        f = variable_diffusion_forcing(points, artifact.value, kx=kx, ky=ky)
    else:
        f = (kx**2 + ky**2) * np.pi**2 * u
    return np.where(artifact.system.boundary, u, f)


def build_artifact(family: str, parameter: str, index: int, value: float) -> SystemArtifact:
    system_id, system = assemble_family_member(
        next(spec for spec in family_specs(False) if spec.family == family),
        index,
        value,
    )
    base = low_precision_lu_preconditioner(system.A)
    start = perf_counter()
    B = np.asarray(base.apply(system.A), dtype=float)
    _, singular_values, right_t = la.svd(
        np.eye(system.dimension) - B,
        full_matrices=False,
        check_finite=False,
    )
    Z = right_t[:Q, :].T
    Z, _ = la.qr(Z, mode="economic", check_finite=False)
    return SystemArtifact(
        system_id,
        family,
        parameter,
        float(value),
        system,
        base,
        Z,
        perf_counter() - start,
        float(singular_values[Q - 1]),
        float(singular_values[Q]),
    )


def common_fields(artifact: SystemArtifact) -> dict[str, Any]:
    return {
        "system_id": artifact.system_id,
        "family": artifact.family,
        "parameter": artifact.parameter,
        "parameter_value": artifact.value,
        "N": N,
        "kernel": "gaussian",
        "epsilon": artifact.value if artifact.family == "epsilon" else 6.0,
        "jitter_fraction": artifact.value if artifact.family == "jitter" else 0.0,
        "beta": artifact.value if artifact.family == "pde_coefficient" else 0.0,
        "q": Q,
    }


def benchmark_deflated(
    artifact: SystemArtifact,
    deflation: GalerkinDeflation,
    rhs: np.ndarray,
    *,
    repeats: int,
) -> dict[str, Any]:
    projected_rhs = deflation.deflate_rhs(rhs)
    original_threshold = RTOL * np.linalg.norm(rhs)

    def solve_once() -> tuple[int, float, float, bool, float]:
        start = perf_counter()
        result = fgmres(
            deflation.A_deflated,
            projected_rhs,
            artifact.base.apply,
            rtol=0.0,
            atol=original_threshold,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
        )
        solution = deflation.reconstruct(artifact.system.A, rhs, result.x)
        elapsed = perf_counter() - start
        residual = true_relative_residual(artifact.system.A, rhs, solution)
        identity_error = np.linalg.norm(
            rhs - artifact.system.A @ solution
            - (projected_rhs - deflation.A_deflated @ result.x)
        ) / np.linalg.norm(rhs)
        return result.iterations, elapsed, residual, residual <= RTOL, float(identity_error)

    solve_once()  # timing warm-up, matching the other three methods
    runs = [solve_once() for _ in range(repeats)]
    first = runs[0]
    return {
        "iterations": first[0],
        "solve_seconds": float(np.median([run[1] for run in runs])),
        "true_relative_residual": first[2],
        "converged": first[3],
        "deflation_residual_identity_error": first[4],
    }


def outcome(method: dict[str, Any], baseline: dict[str, Any]) -> str:
    if bool(method["converged"]) and not bool(baseline["converged"]):
        return "win"
    if not bool(method["converged"]) and bool(baseline["converged"]):
        return "loss"
    if int(method["iterations"]) < int(baseline["iterations"]):
        return "win"
    if int(method["iterations"]) > int(baseline["iterations"]):
        return "loss"
    return "tie"


def summarize_systems(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    system_ids = list(dict.fromkeys(row["system_id"] for row in rows))
    for system_id in system_ids:
        system_rows = [row for row in rows if row["system_id"] == system_id]
        common = system_rows[0]
        for method in METHODS:
            selected = [row for row in system_rows if row["method"] == method]
            counts = {key: sum(row["outcome_vs_baseline"] == key for row in selected) for key in ("win", "tie", "loss")}
            output.append(
                {
                    "system_id": system_id,
                    "family": common["family"],
                    "parameter": common["parameter"],
                    "parameter_value": common["parameter_value"],
                    "method": method,
                    "method_label": LABELS[method],
                    "rhs_count": len(selected),
                    "converged_rhs": sum(bool(row["converged"]) for row in selected),
                    "median_iterations": float(np.median([row["iterations"] for row in selected])),
                    "median_solve_seconds": float(np.median([row["solve_seconds"] for row in selected])),
                    "median_iteration_ratio_vs_baseline": float(
                        np.median([row["iteration_ratio_vs_baseline"] for row in selected])
                    ),
                    "median_solve_time_ratio_vs_baseline": float(
                        np.median([row["solve_time_ratio_vs_baseline"] for row in selected])
                    ),
                    "rhs_wins": counts["win"],
                    "rhs_ties": counts["tie"],
                    "rhs_losses": counts["loss"],
                }
            )
    return output


def system_outcome(method_row: dict[str, Any], base_row: dict[str, Any]) -> str:
    method_all = int(method_row["converged_rhs"]) == int(method_row["rhs_count"])
    base_all = int(base_row["converged_rhs"]) == int(base_row["rhs_count"])
    if method_all and not base_all:
        return "win"
    if not method_all and base_all:
        return "loss"
    left = float(method_row["median_iterations"])
    right = float(base_row["median_iterations"])
    return "win" if left < right else "loss" if left > right else "tie"


def summarize_aggregate(
    rows: list[dict[str, Any]],
    system_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        selected_systems = [row for row in system_rows if row["method"] == method]
        base_by_system = {
            row["system_id"]: row
            for row in system_rows
            if row["method"] == "baseline_fp32_lu"
        }
        system_outcomes = [system_outcome(row, base_by_system[row["system_id"]]) for row in selected_systems]
        ratios = np.array([row["iteration_ratio_vs_baseline"] for row in selected], dtype=float)
        solve_ratios = np.array([row["solve_time_ratio_vs_baseline"] for row in selected], dtype=float)
        output.append(
            {
                "method": method,
                "method_label": LABELS[method],
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
            }
        )
    return output


def summarize_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family in ("epsilon", "jitter", "pde_coefficient"):
        for method in METHODS:
            selected = [row for row in rows if row["family"] == family and row["method"] == method]
            output.append(
                {
                    "family": family,
                    "method": method,
                    "method_label": LABELS[method],
                    "observations": len(selected),
                    "converged": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio_vs_baseline": float(
                        np.median([row["iteration_ratio_vs_baseline"] for row in selected])
                    ),
                    "rhs_wins": sum(row["outcome_vs_baseline"] == "win" for row in selected),
                    "rhs_ties": sum(row["outcome_vs_baseline"] == "tie" for row in selected),
                    "rhs_losses": sum(row["outcome_vs_baseline"] == "loss" for row in selected),
                }
            )
    return output


def report_cell(row: dict[str, Any]) -> str:
    return (
        f"{row['median_iterations']:.1f}; {row['converged_rhs']}/{row['rhs_count']}; "
        f"{row['median_iteration_ratio_vs_baseline']:.3f}; {row['median_solve_seconds']:.4f}s"
    )


def write_report(
    report: Path,
    aggregate: list[dict[str, Any]],
    family_summary: list[dict[str, Any]],
    system_summary: list[dict[str, Any]],
    output: Path,
) -> None:
    by_method = {row["method"]: row for row in aggregate}
    best = min(
        (row for row in aggregate if row["method"] != "baseline_fp32_lu"),
        key=lambda row: row["median_iteration_ratio_vs_baseline"],
    )
    stable = best["median_iteration_ratio_vs_baseline"] < 1.0 and best["system_losses"] == 0
    lines = [
        "# Exact Z_S Coarse-Integration Experiment",
        "",
        "## Answer",
        "",
        (
            "**Yes: exact Z_S stably reduces iterations under the best tested integration.**"
            if stable
            else "**No: exact Z_S does not stably reduce iterations across the unchanged 27-system suite.**"
        ),
        "",
        f"The strongest integration was **{best['method_label']}**, with aggregate median iteration ratio **{best['median_iteration_ratio_vs_baseline']:.3f}** and system win/tie/loss **{best['system_wins']}/{best['system_ties']}/{best['system_losses']}**. The result is reported without ML and without changing the systems, M, or FGMRES tolerances.",
        "",
        "## Aggregate results",
        "",
        "Win/tie/loss is convergence-aware: convergence beats nonconvergence; otherwise iteration counts are compared. There are 108 system-RHS observations and 27 system-level comparisons per method.",
        "",
        "| Method | Converged RHS | Median iteration ratio | IQR | Median solve-time ratio | RHS W/T/L | System W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"| {row['method_label']} | {row['converged']}/{row['observations']} | "
            f"{row['median_iteration_ratio_vs_baseline']:.3f} | "
            f"{row['iteration_ratio_q25']:.3f}–{row['iteration_ratio_q75']:.3f} | "
            f"{row['median_solve_time_ratio_vs_baseline']:.3f} | "
            f"{row['rhs_wins']}/{row['rhs_ties']}/{row['rhs_losses']} | "
            f"{row['system_wins']}/{row['system_ties']}/{row['system_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Results by system family",
            "",
            "| Family | Method | Converged RHS | Median iteration ratio | RHS W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in family_summary:
        if row["method"] == "baseline_fp32_lu":
            continue
        lines.append(
            f"| {row['family']} | {row['method_label']} | {row['converged']}/{row['observations']} | "
            f"{row['median_iteration_ratio_vs_baseline']:.3f} | "
            f"{row['rhs_wins']}/{row['rhs_ties']}/{row['rhs_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Per-system results",
            "",
            "Each cell is `median iterations; converged RHS/4; median iteration ratio; median solve time`.",
            "",
            "| System | M only | Additive Galerkin | Residual-minimizing | Standard deflation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    systems = list(dict.fromkeys(row["system_id"] for row in system_summary))
    lookup = {(row["system_id"], row["method"]): row for row in system_summary}
    for system_id in systems:
        lines.append(
            f"| {system_id} | "
            + " | ".join(report_cell(lookup[(system_id, method)]) for method in METHODS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exact integrations",
            "",
            "All methods use the same exact q=8 basis formed from the dominant right singular vectors of S = I - M^(-1)A.",
            "",
            "1. **Additive Galerkin:** M^(-1)r + Z solve(Z^T A Z, Z^T r).",
            "2. **Residual-minimizing:** M^(-1)r + Z y, where y minimizes ||r-AZy||_2. The reduced least-squares action uses the SVD of AZ; no normal equations are formed.",
            "3. **Standard nonsymmetric Galerkin deflation:** E=Z^T A Z, P=I-AZ E^(-1)Z^T, Q=I-Z E^(-1)Z^T A. Right-preconditioned FGMRES solves P A x_hat=P b with M^(-1), then reconstructs x=Z E^(-1)Z^T b+Q x_hat. Reduced systems use pivoted LU solves, never explicit inverses. The projected solve uses the unchanged absolute threshold tol*||b|| so convergence is measured against the original system.",
            "",
            "Solve times exclude oracle-space and coarse setup but include FGMRES and deflated-solution reconstruction. All final residuals are recomputed from the original A and b.",
            "",
            "## Raw evidence",
            "",
            f"- `{output / 'solver_results.csv'}`",
            f"- `{output / 'system_summary.csv'}`",
            f"- `{output / 'aggregate_summary.csv'}`",
            f"- `{output / 'family_summary.csv'}`",
            f"- `{output / 'validation_report.md'}`",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spaces = args.output / "oracle_spaces"
    spaces.mkdir(parents=True, exist_ok=True)

    specs = family_specs(False)
    members = [
        (spec, index, float(value))
        for spec in specs
        for index, value in enumerate(spec.values)
    ]
    if args.quick:
        members = [next(member for member in members if member[0].family == family) for family in ("epsilon", "jitter", "pde_coefficient")]
    rhs_specs = RHS_SPECS[:1] if args.quick else RHS_SPECS
    repeats = 1 if args.quick else REPEATS

    raw_rows: list[dict[str, Any]] = []
    system_metadata: list[dict[str, Any]] = []
    for counter, (spec, index, value) in enumerate(members, start=1):
        artifact = build_artifact(spec.family, spec.parameter, index, value)
        print(f"[system {counter}/{len(members)}] {artifact.system_id}", flush=True)
        np.savez_compressed(spaces / f"{artifact.system_id}.npz", Z=artifact.Z)
        additive = additive_galerkin_coarse_preconditioner(
            artifact.system.A,
            artifact.base,
            artifact.Z,
            name="additive_galerkin_q8",
        )
        residual_minimizing = residual_minimizing_coarse_preconditioner(
            artifact.system.A,
            artifact.base,
            artifact.Z,
            name="additive_residual_minimizing_q8",
        )
        deflation = galerkin_deflation(artifact.system.A, artifact.Z)
        system_metadata.append(
            {
                **common_fields(artifact),
                "oracle_setup_seconds": artifact.oracle_setup_seconds,
                "oracle_sigma_q": artifact.oracle_sigma_q,
                "oracle_sigma_q_plus_1": artifact.oracle_sigma_q_plus_1,
                "oracle_boundary_relative_gap": (
                    (artifact.oracle_sigma_q - artifact.oracle_sigma_q_plus_1)
                    / artifact.oracle_sigma_q
                ),
                "base_setup_seconds": artifact.base.setup_seconds,
                "additive_setup_seconds": additive.setup_seconds - artifact.base.setup_seconds,
                "residual_minimizing_setup_seconds": residual_minimizing.setup_seconds - artifact.base.setup_seconds,
                "deflation_setup_seconds": deflation.setup_seconds,
            }
        )

        for rhs_name, kx, ky in rhs_specs:
            rhs = mode_rhs(artifact, kx, ky)
            results: dict[str, dict[str, Any]] = {}
            for method, preconditioner in (
                ("baseline_fp32_lu", artifact.base),
                ("additive_galerkin", additive),
                ("additive_residual_minimizing", residual_minimizing),
            ):
                benchmark = benchmark_fgmres(
                    artifact.system.A,
                    rhs,
                    preconditioner,
                    rtol=RTOL,
                    restart=RESTART,
                    max_iterations=MAX_ITERATIONS,
                    repeats=repeats,
                    warmup=True,
                )
                results[method] = {
                    "iterations": benchmark.result.iterations,
                    "solve_seconds": benchmark.solve_seconds_median,
                    "true_relative_residual": benchmark.final_true_relative_residual,
                    "converged": benchmark.final_true_relative_residual <= RTOL,
                    "deflation_residual_identity_error": 0.0,
                }
            results["standard_galerkin_deflation"] = benchmark_deflated(
                artifact,
                deflation,
                rhs,
                repeats=repeats,
            )
            baseline = results["baseline_fp32_lu"]
            for method in METHODS:
                result = results[method]
                raw_rows.append(
                    {
                        **common_fields(artifact),
                        "rhs_name": rhs_name,
                        "kx": kx,
                        "ky": ky,
                        "method": method,
                        "method_label": LABELS[method],
                        "iterations": result["iterations"],
                        "converged": result["converged"],
                        "solve_seconds": result["solve_seconds"],
                        "true_relative_residual": result["true_relative_residual"],
                        "iteration_ratio_vs_baseline": result["iterations"] / baseline["iterations"],
                        "solve_time_ratio_vs_baseline": result["solve_seconds"] / baseline["solve_seconds"],
                        "outcome_vs_baseline": outcome(result, baseline),
                        "deflation_residual_identity_error": result["deflation_residual_identity_error"],
                        "rtol": RTOL,
                        "restart": RESTART,
                        "max_iterations": MAX_ITERATIONS,
                        "timing_repeats": repeats,
                    }
                )

    system_summary = summarize_systems(raw_rows)
    aggregate = summarize_aggregate(raw_rows, system_summary)
    family_summary = summarize_families(raw_rows)
    write_csv(args.output / "solver_results.csv", raw_rows)
    write_csv(args.output / "system_summary.csv", system_summary)
    write_csv(args.output / "aggregate_summary.csv", aggregate)
    write_csv(args.output / "family_summary.csv", family_summary)
    write_csv(args.output / "system_metadata.csv", system_metadata)
    config = {
        "experiment": "exact q=8 Z_S coarse-integration comparison",
        "no_ml": True,
        "system_count": len(members),
        "systems": "unchanged Stage-2 9 epsilon + 9 jitter + 9 beta systems",
        "rhs_specs": rhs_specs,
        "preconditioner": "FP32 pivoted LU",
        "oracle": "top q=8 right singular vectors of S=I-M^{-1}A",
        "methods": list(METHODS),
        "rtol": RTOL,
        "restart": RESTART,
        "max_iterations": MAX_ITERATIONS,
        "timing_repeats": repeats,
        "warmup": 1,
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
    write_report(args.report, aggregate, family_summary, system_summary, args.output)
    print(f"Wrote {len(raw_rows)} solve rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
