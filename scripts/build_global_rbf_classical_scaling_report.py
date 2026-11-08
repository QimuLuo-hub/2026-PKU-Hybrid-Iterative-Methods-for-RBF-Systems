#!/usr/bin/env python3
"""Build the canonical portable-report input for classical global-RBF scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_classical_scaling"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _scientific(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}e}"


def main() -> int:
    args = parse_args()
    raw_systems = _read_csv(args.results_dir / "systems.csv")
    raw_results = _read_csv(args.results_dir / "benchmark.csv")
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((args.results_dir / "validation.json").read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    systems: list[dict[str, object]] = []
    for order, row in enumerate(raw_systems, start=1):
        condition = float(row["condition_2_estimate"])
        measured = row["condition_estimate_measured"] == "True"
        systems.append(
            {
                "sort_order": order,
                "N": int(row["N"]),
                "grid": f"{row['side']}×{row['side']}",
                "epsilon": float(row["epsilon"]),
                "epsilon_h": float(row["epsilon_times_h"]),
                "matrix_memory_mb": float(row["matrix_memory_megabytes_decimal"]),
                "assembly_seconds": float(row["assembly_seconds"]),
                "condition_2": condition,
                "condition_display": f"{condition:.2e}" + ("" if measured else " est."),
                "log10_condition": math.log10(condition),
                "condition_method": "Exact SVD" if measured else "Extrapolated",
                "sigma_max": float(row["sigma_max"]),
                "sigma_min_display": _scientific(float(row["sigma_min"])),
            }
        )

    method_order = {"none": 0, "jacobi": 1, "ls_acbf": 2, "fp32_lu": 3}
    method_labels = {
        "none": "None",
        "jacobi": "Jacobi",
        "ls_acbf": "LS-ACBF (k=49)",
        "fp32_lu": "FP32 pivoted LU",
    }
    results: list[dict[str, object]] = []
    measured_results: list[dict[str, object]] = []
    for row in sorted(
        raw_results,
        key=lambda item: (int(item["N"]), method_order[item["method"]]),
    ):
        measured = row["run_status"] == "measured"
        setup = _optional_float(row["setup_seconds"])
        setup_projected = _optional_float(row["setup_seconds_projected"])
        residual = _optional_float(row["final_true_relative_residual"])
        rhs_defect = _optional_float(row["am_inv_rhs_inverse_defect"])
        n = int(row["N"])
        method = row["method"]
        if measured:
            setup_display = f"{setup:.3g} measured"
            iterations_display = f"{int(row['gmres_iterations']):,} cap"
            solve_display = f"{float(row['solve_seconds']):.3g}"
            converged = "No"
            residual_display = _scientific(residual)
            rhs_defect_display = _scientific(rhs_defect)
        else:
            setup_display = f"{setup_projected:.3g} projected"
            iterations_display = "Not run"
            solve_display = "—"
            converged = "Not run"
            residual_display = "—"
            rhs_defect_display = "—"
        result = {
            "sort_order": len(results) + 1,
            "N": n,
            "method": method_labels[method],
            "run_status": "Measured" if measured else "Feasibility cutoff",
            "setup_seconds": setup,
            "setup_seconds_projected": setup_projected,
            "setup_display": setup_display,
            "iterations": int(row["gmres_iterations"]) if measured else None,
            "iterations_display": iterations_display,
            "solve_seconds": float(row["solve_seconds"]) if measured else None,
            "solve_display": solve_display,
            "first_rhs_seconds": float(row["first_rhs_seconds"]) if measured else None,
            "converged": converged,
            "final_relative_residual": residual,
            "residual_display": residual_display,
            "residual_digits": -math.log10(residual) if residual is not None else None,
            "am_rhs_inverse_defect": rhs_defect,
            "am_rhs_defect_display": rhs_defect_display,
            "am_restricted_singular_spread": _optional_float(
                row["am_inv_random_subspace_condition"]
            ),
            "am_compression_median_distance_to_one": _optional_float(
                row["am_inv_compression_median_distance_to_one"]
            ),
            "am_compression_nonnormality": _optional_float(
                row["am_inv_compression_nonnormality"]
            ),
            "preconditioner_additivity_defect": _optional_float(
                row["preconditioner_additivity_defect"]
            ),
            "preconditioner_memory_mb": int(row["preconditioner_memory_bytes"]) / 1e6,
            "matrix_plus_preconditioner_mb": int(row["matrix_plus_preconditioner_bytes"]) / 1e6,
        }
        results.append(result)
        if measured:
            measured_results.append(result)

    by_key = {(row["N"], row["method"]): row for row in results}
    cond_first = systems[0]["condition_2"]
    cond_last = systems[-1]["condition_2"]
    memory_first = systems[0]["matrix_memory_mb"]
    memory_last = systems[-1]["matrix_memory_mb"]
    none_first = by_key[(900, "None")]
    none_last = by_key[(10000, "None")]
    fp32_first = by_key[(900, "FP32 pivoted LU")]
    fp32_last = by_key[(10000, "FP32 pivoted LU")]
    ls_last_measured = by_key[(2500, "LS-ACBF (k=49)")]
    ls_projected = by_key[(10000, "LS-ACBF (k=49)")]
    fit = summary["condition_extrapolation"]

    title = "Global Gaussian-RBF classical scaling"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Validated scaling study of classical right-preconditioned GMRES for a dense 2D global Gaussian-RBF Poisson collocation benchmark.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "residual_digits",
                    "title": "No classical variant reaches the ten-digit target",
                    "subtitle": "Residual reduction after the common 1,000-iteration cap; 10 digits are required",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "How does preconditioner effectiveness change as the dense RBF problem grows?",
                    "rationale": "Residual digits keep the six-order range readable while preserving the common true-residual criterion.",
                    "dataset": "measured_results",
                    "sourceId": "benchmark_source",
                    "encodings": {
                        "x": {"field": "N", "type": "ordinal", "label": "Unknowns N"},
                        "y": {
                            "field": "residual_digits",
                            "type": "quantitative",
                            "label": "Residual reduction, −log₁₀(relative residual)",
                        },
                        "color": {"field": "method", "type": "nominal", "label": "Method"},
                        "tooltip": [
                            {
                                "field": "final_relative_residual",
                                "type": "quantitative",
                                "label": "Final true relative residual",
                            },
                            {"field": "iterations", "type": "quantitative", "label": "Iterations"},
                        ],
                    },
                    "yAxisTitle": "Residual reduction digits",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "classical-preconditioners"},
                    "legend": {"position": "bottom", "interactive": True, "title": "Method"},
                    "labels": {"values": "all"},
                    "settings": {
                        "groupMode": "grouped",
                        "orientation": "vertical",
                        "sort": "custom",
                        "showValues": True,
                    },
                },
                {
                    "id": "solve_scaling",
                    "title": "A fixed 1,000-iteration workload becomes expensive",
                    "subtitle": "Median warmed solve time; setup excluded and LS-ACBF stops at its declared feasibility cutoff",
                    "type": "line",
                    "intent": "trend",
                    "question": "How does the cost of the same capped Krylov work scale with N?",
                    "rationale": "The ordered size axis exposes the dense matvec and preconditioner-application scaling directly.",
                    "dataset": "measured_results",
                    "sourceId": "benchmark_source",
                    "encodings": {
                        "x": {"field": "N", "type": "ordinal", "label": "Unknowns N"},
                        "y": {
                            "field": "solve_seconds",
                            "type": "quantitative",
                            "label": "Solve time",
                            "unit": "s",
                        },
                        "color": {"field": "method", "type": "nominal", "label": "Method"},
                        "tooltip": [
                            {"field": "solve_seconds", "type": "quantitative", "label": "Solve time", "unit": "s"},
                            {"field": "first_rhs_seconds", "type": "quantitative", "label": "Setup + first solve", "unit": "s"},
                        ],
                    },
                    "yAxisTitle": "Solve time (s), setup excluded",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "classical-preconditioners"},
                    "legend": {"position": "bottom", "interactive": True, "title": "Method"},
                    "settings": {"showPoints": True, "sort": "ascending"},
                },
                {
                    "id": "resource_scaling",
                    "title": "Dense storage reaches 800 MB at N=10,000",
                    "subtitle": "FP64 collocation matrix only; storage follows 8N² bytes exactly",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "When does matrix storage itself become a practical constraint?",
                    "rationale": "Five requested problem sizes make the quadratic storage trajectory explicit.",
                    "dataset": "systems",
                    "sourceId": "systems_source",
                    "encodings": {
                        "x": {"field": "N", "type": "ordinal", "label": "Unknowns N"},
                        "y": {
                            "field": "matrix_memory_mb",
                            "type": "quantitative",
                            "label": "Matrix memory",
                            "unit": "MB",
                        },
                        "tooltip": [
                            {"field": "condition_display", "type": "nominal", "label": "κ₂(A)"},
                            {"field": "condition_method", "type": "nominal", "label": "Condition method"},
                        ],
                    },
                    "yAxisTitle": "Dense matrix memory (MB)",
                    "layout": "full",
                    "palette": {"kind": "sequential", "name": "memory"},
                    "labels": {"values": "all"},
                    "settings": {"orientation": "vertical", "sort": "ascending", "showValues": True},
                },
            ],
            "tables": [
                {
                    "id": "system_detail",
                    "title": "System scaling and condition estimates",
                    "subtitle": "The N=10,000 condition is the sole extrapolated value",
                    "dataset": "systems",
                    "sourceId": "systems_source",
                    "defaultSort": {"field": "sort_order", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "sort_order", "label": "#", "format": "number"},
                        {"field": "grid", "label": "Grid", "type": "text"},
                        {"field": "N", "label": "N", "format": "number"},
                        {"field": "epsilon_h", "label": "εh", "format": "number"},
                        {"field": "matrix_memory_mb", "label": "Matrix MB", "format": "number"},
                        {"field": "condition_display", "label": "κ₂(A)", "type": "text"},
                        {"field": "condition_method", "label": "Method", "type": "text"},
                    ],
                },
                {
                    "id": "benchmark_detail",
                    "title": "Exact benchmark results",
                    "subtitle": "Setup is measured unless marked projected; solve time excludes setup",
                    "dataset": "results",
                    "sourceId": "benchmark_source",
                    "defaultSort": {"field": "sort_order", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "sort_order", "label": "#", "format": "number"},
                        {"field": "N", "label": "N", "format": "number"},
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "setup_display", "label": "Setup (s)", "type": "text"},
                        {"field": "iterations_display", "label": "Iterations", "type": "text"},
                        {"field": "solve_display", "label": "Solve (s)", "type": "text"},
                        {"field": "converged", "label": "Converged", "type": "text"},
                        {"field": "residual_display", "label": "Final rel. residual", "type": "text"},
                        {"field": "am_rhs_defect_display", "label": "||b-AM⁻¹b||/||b||", "type": "text"},
                    ],
                },
                {
                    "id": "spectral_detail",
                    "title": "A M⁻¹ response and compression diagnostics",
                    "subtitle": "Fixed 16-dimensional random probe; these are not full condition estimates",
                    "dataset": "measured_results",
                    "sourceId": "benchmark_source",
                    "defaultSort": {"field": "sort_order", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "sort_order", "label": "#", "format": "number"},
                        {"field": "N", "label": "N", "format": "number"},
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "am_restricted_singular_spread", "label": "Restricted σ spread", "format": "number"},
                        {"field": "am_compression_median_distance_to_one", "label": "Median |λ−1|", "format": "number"},
                        {"field": "am_compression_nonnormality", "label": "Compression nonnormality", "format": "number"},
                        {"field": "am_rhs_inverse_defect", "label": "RHS inverse defect", "format": "number"},
                    ],
                },
            ],
            "sources": [
                {"id": "systems_source", "label": "System and conditioning results", "path": "systems.csv"},
                {"id": "benchmark_source", "label": "Classical right-preconditioned GMRES sweep", "path": "benchmark.csv"},
                {"id": "validation_source", "label": "Independent numerical validation", "path": "validation.json"},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Answer\n\n"
                        "**Krylov difficulty rises before memory becomes the only problem, and none of the four classical choices is a practical scaling solution in this native global basis.** "
                        "Every measured run reaches the common **1,000-iteration cap without satisfying the 1e-10 true-residual tolerance**. "
                        f"Across the requested range, κ₂(A) rises from **{cond_first:.2e}** to **{cond_last:.2e}** while matrix storage rises from **{memory_first:.2f} MB** to **{memory_last:.0f} MB**. "
                        f"The unpreconditioned capped solve grows from **{none_first['solve_seconds']:.3f} s** to **{none_last['solve_seconds']:.2f} s**.\n\n"
                        "Jacobi is essentially neutral, LS-ACBF improves the terminal residual but its setup is already dominant, and FP32 LU loses inverse quality rapidly as N grows. "
                        "The practical next step is therefore a more stable/localized formulation or a stronger scalable preconditioner—not a larger dense run of this same setup."
                    ),
                },
                {
                    "id": "difficulty_heading",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Iteration counts are right-censored\n\n"
                        "Because every measured row ends at 1,000 iterations, iteration count alone cannot rank the methods. The chart uses **−log₁₀ of the final true relative residual**: ten digits mark convergence. LS-ACBF preserves roughly 5.5–5.8 digits through N=2,500, but still misses tolerance; FP32 LU falls from 5.7 digits at N=900 to essentially zero at N=10,000."
                    ),
                },
                {"id": "residual_chart", "type": "chart", "chartId": "residual_digits", "layout": "full"},
                {
                    "id": "cost_heading",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Fixed-work solve cost grows sharply\n\n"
                        f"For no preconditioner, the same 1,000 Arnoldi steps cost **{none_last['solve_seconds'] / none_first['solve_seconds']:.1f}×** more at N=10,000 than at N=900. "
                        "Jacobi gives no persistent residual benefit and is slower at the largest size. FP32 LU adds dense triangular-solve cost; at N=10,000 its **32.65 s setup plus 30.38 s capped solve** totals 63.03 s for the first right-hand side."
                    ),
                },
                {"id": "solve_chart", "type": "chart", "chartId": "solve_scaling", "layout": "full"},
                {
                    "id": "resource_heading",
                    "type": "markdown",
                    "sourceId": "systems_source",
                    "body": (
                        "## Dense storage is quadratic\n\n"
                        "The matrix follows 8N² bytes exactly. FP32 LU stores another 400.04 MB of factors at N=10,000, bringing matrix-plus-preconditioner storage to **1.20 GB** before Krylov work arrays and process overhead."
                    ),
                },
                {"id": "resource_chart", "type": "chart", "chartId": "resource_scaling", "layout": "full"},
                {"id": "system_table", "type": "table", "tableId": "system_detail", "layout": "full"},
                {
                    "id": "preconditioner_heading",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Preconditioner practicality\n\n"
                        f"**LS-ACBF:** the fixed 49-point, 7×7 footprint reduces the terminal residual by about one to two orders of magnitude through N=2,500, but setup grows from 4.01 s to **{ls_last_measured['setup_seconds']:.2f} s**. A fit to those measured setups has exponent **{validation['calculation_spot_checks']['LS_ACBF_setup_power_law_exponent']:.3f}**; under the declared cutoff this projects to **{ls_projected['setup_seconds_projected']:.0f} s (8.2 min)** at N=10,000.\n\n"
                        f"**FP32 pivoted LU:** the right-hand-side inverse-response defect grows from **{fp32_first['am_rhs_inverse_defect']:.3g}** at N=900 to **{fp32_last['am_rhs_inverse_defect']:.1f}** at N=10,000, while terminal residual worsens from **{fp32_first['final_relative_residual']:.2e}** to **{fp32_last['final_relative_residual']:.3f}**. Factorization remains computable, but it no longer behaves as a useful inverse in this conditioning regime.\n\n"
                        "**Jacobi:** diagonal scaling changes response magnitudes but not the capped convergence outcome; its final residual remains in the same 10⁻⁵–10⁻⁴ band as no preconditioner."
                    ),
                },
                {"id": "benchmark_table", "type": "table", "tableId": "benchmark_detail", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "systems_source",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "- **Problem:** 2D Poisson collocation with global Gaussian RBFs, exact solution sin(πx)sin(πy), tensor square grids, and Dirichlet boundary rows.\n"
                        "- **Comparable shape regime:** ε=0.4√N, so εh varies only from 0.414 to 0.404.\n"
                        "- **Solver:** right-preconditioned restarted FGMRES, restart 100, zero initial guess, rtol 1e-10, at most 1,000 Arnoldi steps; three warmed timings on one CPU thread. For the linear none/Jacobi/LS maps, this is mathematically right-preconditioned GMRES.\n"
                        "- **Solve time:** median warmed Krylov solve, excluding preconditioner construction. **First-RHS time** is setup plus solve.\n"
                        "- **Convergence:** original-system true residual ||b−Ax||₂/||b||₂≤1e-10.\n"
                        "- **LS-ACBF:** row-sparse least-squares approximate-cardinal inverse with 49 nearest collocation equations per coefficient."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Conditioning and preconditioned-operator diagnostics\n\n"
                        "κ₂(A) is computed by dense SVD through N=4,900. At N=10,000, σmax is measured with the tensor-structured operator and κ₂(A) is extrapolated from the four smaller exact values using a log-log power law. The fit exponent is "
                        f"**{fit['exponent']:.3f}** with R² **{fit['r_squared']:.4f}**; a stricter fit trained only through N=2,500 overpredicts the held-out N=4,900 value by **{100.0 * (fit['holdout_validation']['prediction_over_actual'] - 1.0):.1f}%**.\n\n"
                        "For AM⁻¹, each measured method is probed on a fixed 16-dimensional random orthonormal subspace. The output includes restricted singular spread, compression eigenvalues/nonnormality, probe gains, and the right-hand-side inverse-response defect ||b−AM⁻¹b||₂/||b||₂. These are **spectral-response diagnostics, not a full condition estimate**; a random compression can miss the tiny singular directions that dominate κ₂.\n\n"
                        "The FP32 triangular solve is roundoff-dependent and measurably non-additive on this ill-conditioned family, so its response diagnostics do not describe one exact linear operator. Flexible GMRES is retained for that case; the linear variants were independently cross-checked against SciPy GMRES."
                    ),
                },
                {"id": "spectral_table", "type": "table", "tableId": "spectral_detail", "layout": "full"},
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "validation_source",
                    "body": (
                        f"## Validation\n\n**Assessment: {validation['overall_assessment']}.** All requested sizes and methods are represented; dense memory recomputes as 8N²; εh remains comparable; blocked dense and tensor-structured actions agree to better than 7e-16 relative error. The N=900 condition number recomputes exactly. SciPy GMRES independently reproduces the 1,000-step cap and non-convergence for none, Jacobi, and LS-ACBF; a repeated FP32-FGMRES run reproduces its recorded residual. No ML model or oracle subspace is created or used."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and uncertainty\n\n"
                        "- All iteration counts are right-censored at 1,000; solve-time comparisons are costs for equal capped work, not converged time-to-solution.\n"
                        "- N=10,000 κ₂(A) and LS-ACBF setup at N=4,900/10,000 are estimates, explicitly labeled; all other reported setup/solve values are measured.\n"
                        "- Random-subspace AM⁻¹ diagnostics are deliberately lightweight and cannot certify the full spectrum or condition number.\n"
                        "- FP32 LU is a finite-precision response map here, not a fixed exact-linear operator.\n"
                        "- Wall-clock values are workstation-specific single-thread medians. Matrix/factor memory, residuals, and cap outcomes are more portable."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Keep the same PDE and εh regime, but test a stable-basis or localized RBF formulation before increasing N further.\n"
                        "2. Evaluate scalable sparse, hierarchical, or domain-decomposition preconditioners with both setup amortization and true residual tracked.\n"
                        "3. If this native global basis must be retained, use the present run as a stress-test regression and raise the iteration cap only after a preconditioner demonstrates materially better residual decay.\n"
                        "4. Defer ML and oracle recycling until a classical baseline reaches the target tolerance reliably."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Is the observed difficulty primarily native-basis conditioning, nonnormality, or boundary-row scaling?\n"
                        "- Would residual equilibration plus a localized basis make Jacobi or ILU-type methods meaningful?\n"
                        "- At what N does a matrix-free structured matvec plus scalable preconditioner overtake dense storage and BLAS throughput?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "systems": systems,
                "results": results,
                "measured_results": measured_results,
            },
        },
        "sources": [
            {
                "id": "systems_source",
                "label": "System and conditioning results",
                "path": "systems.csv",
                "query": {
                    "engine": "duckdb",
                    "description": "Load the five requested square-grid systems and their matrix/condition diagnostics.",
                    "language": "sql",
                    "sql": "SELECT row_number() OVER () AS sort_order, side, N, epsilon, epsilon_times_h, matrix_memory_megabytes_decimal AS matrix_memory_mb, assembly_seconds, sigma_max, sigma_min, condition_2_estimate AS condition_2, condition_estimation_method, condition_estimate_measured FROM read_csv_auto('systems.csv') ORDER BY N",
                    "tables_used": ["systems.csv"],
                    "filters": ["N in (900,1600,2500,4900,10000)", "Gaussian kernel", "epsilon=0.4*side"],
                    "metric_definitions": {
                        "matrix_memory_mb": "Dense FP64 matrix bytes divided by 1e6.",
                        "condition_2": "Exact dense-SVD 2-norm condition through N=4900; labeled power-law estimate at N=10000.",
                    },
                },
            },
            {
                "id": "benchmark_source",
                "label": "Classical right-preconditioned GMRES sweep",
                "path": "benchmark.csv",
                "query": {
                    "engine": "duckdb",
                    "description": "Load measured classical solver rows and the two declared LS-ACBF feasibility projections.",
                    "language": "sql",
                    "sql": "SELECT * FROM read_csv_auto('benchmark.csv') ORDER BY N, CASE method WHEN 'none' THEN 0 WHEN 'jacobi' THEN 1 WHEN 'ls_acbf' THEN 2 ELSE 3 END",
                    "tables_used": ["benchmark.csv"],
                    "filters": ["rtol=1e-10", "restart=100", "max_iterations=1000", "timing_repeats=3"],
                    "metric_definitions": {
                        "gmres_iterations": "Arnoldi expansions, capped at 1000.",
                        "solve_seconds": "Median warmed Krylov solve excluding preconditioner setup.",
                        "am_inv_rhs_inverse_defect": "||b-A M^-1 b||2 / ||b||2; a response diagnostic, not a condition estimate.",
                    },
                },
            },
            {
                "id": "validation_source",
                "label": "Independent numerical validation",
                "path": "validation.json",
            },
        ],
    }
    (args.results_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.results_dir / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
