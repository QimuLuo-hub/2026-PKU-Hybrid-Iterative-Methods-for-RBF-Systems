#!/usr/bin/env python3
"""Build the canonical portable-report input for the recycle-space benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/neurkitt_recycle_space"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (args.results_dir / "validation.json").read_text(encoding="utf-8")
    )
    raw_rows = _read_csv(args.results_dir / "benchmark.csv")
    raw_angles = _read_csv(args.results_dir / "principal_angles.csv")
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )

    space_labels = {
        "baseline": "Baseline",
        "QA": "QA",
        "QB": "QB",
        "random": "Random",
    }
    comparison = []
    for sort_order, row in enumerate(raw_rows, start=1):
        ratio = float(row["iteration_ratio_over_baseline"])
        comparison.append(
            {
                "sort_order": sort_order,
                "method": row["method"],
                "preconditioner": row["preconditioner"],
                "recycle_space": row["recycle_space"],
                "variant": space_labels[row["recycle_space"]],
                "k": int(row["k"]),
                "iterations": int(row["iterations"]),
                "converged": "Yes" if row["converged"] == "True" else "No",
                "solve_time_seconds": float(
                    row["solve_time_seconds_excluding_space_construction"]
                ),
                "krylov_seconds_median": float(row["krylov_seconds_median"]),
                "final_true_relative_residual": float(
                    row["final_true_relative_residual"]
                ),
                "iteration_ratio": ratio,
                "iteration_ratio_display": f"{ratio:.3f}",
                "restart_cycles": int(row["restart_cycles"]),
            }
        )
    angles = [
        {
            "preconditioner": row["preconditioner"],
            "angle_rank": int(row["canonical_angle_index_largest_first"]),
            "principal_angle_degrees": float(row["principal_angle_degrees"]),
        }
        for row in raw_angles
    ]
    by_key = {(row["preconditioner"], row["recycle_space"]): row for row in comparison}
    jacobi = summary["qb_diagnostics"]["Jacobi"]
    ilu = summary["qb_diagnostics"]["ILU(0)"]
    jacobi_qa = by_key[("Jacobi", "QA")]
    jacobi_qb = by_key[("Jacobi", "QB")]
    ilu_qa = by_key[("ILU(0)", "QA")]
    ilu_qb = by_key[("ILU(0)", "QB")]
    jacobi_qb_gain = 100.0 * (
        jacobi_qa["iterations"] - jacobi_qb["iterations"]
    ) / jacobi_qa["iterations"]
    ilu_qb_gain = 100.0 * (ilu_qa["iterations"] - ilu_qb["iterations"]) / ilu_qa[
        "iterations"
    ]

    title = "QA versus QB on right-preconditioned Darcy"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Oracle-only QA versus QB recycle-space comparison on the validated NeurKItt Darcy N=9604 benchmark.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "iteration_comparison",
                    "title": "QB is only marginally better than QA",
                    "subtitle": "Restarted right-preconditioned GMRES, k=10, true relative residual tolerance 1e-8",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "Does targeting the preconditioned operator reduce iterations beyond the validated A-invariant space?",
                    "rationale": "Grouped bars show QA, QB, and the random control against the paired preconditioned baseline.",
                    "dataset": "comparison",
                    "sourceId": "benchmark_source",
                    "encodings": {
                        "x": {
                            "field": "preconditioner",
                            "type": "ordinal",
                            "label": "Right preconditioner",
                        },
                        "y": {
                            "field": "iterations",
                            "type": "quantitative",
                            "label": "Arnoldi iterations",
                            "format": "number",
                        },
                        "color": {
                            "field": "variant",
                            "type": "nominal",
                            "label": "Recycle space",
                        },
                        "tooltip": [
                            {
                                "field": "iteration_ratio",
                                "type": "quantitative",
                                "label": "Iterations / baseline",
                            },
                            {
                                "field": "solve_time_seconds",
                                "type": "quantitative",
                                "label": "Solve time",
                                "unit": "s",
                            },
                            {
                                "field": "final_true_relative_residual",
                                "type": "quantitative",
                                "label": "Final true relative residual",
                            },
                        ],
                    },
                    "yAxisTitle": "Arnoldi iterations",
                    "valueFormat": "number",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "recycle-space"},
                    "legend": {
                        "position": "bottom",
                        "interactive": True,
                        "title": "Recycle space",
                    },
                    "labels": {"values": "all"},
                    "settings": {
                        "groupMode": "grouped",
                        "orientation": "vertical",
                        "sort": "custom",
                        "showValues": True,
                    },
                },
                {
                    "id": "angle_spectrum",
                    "title": "ILU changes most of the ten-dimensional space",
                    "subtitle": "Canonical angles QA versus QB, ordered from largest to smallest",
                    "type": "line",
                    "intent": "comparison",
                    "question": "How broadly does each right preconditioner rotate the targeted invariant space?",
                    "rationale": "The full canonical-angle spectrum distinguishes a single rotated direction from broad subspace change.",
                    "dataset": "angles",
                    "sourceId": "angle_source",
                    "encodings": {
                        "x": {
                            "field": "angle_rank",
                            "type": "ordinal",
                            "label": "Canonical angle rank (largest first)",
                        },
                        "y": {
                            "field": "principal_angle_degrees",
                            "type": "quantitative",
                            "label": "Principal angle",
                            "unit": "degrees",
                        },
                        "color": {
                            "field": "preconditioner",
                            "type": "nominal",
                            "label": "Right preconditioner",
                        },
                        "tooltip": [
                            {
                                "field": "principal_angle_degrees",
                                "type": "quantitative",
                                "label": "Principal angle",
                                "unit": "degrees",
                            }
                        ],
                    },
                    "yAxisTitle": "Principal angle (degrees)",
                    "layout": "full",
                    "palette": {"kind": "categorical", "name": "preconditioners"},
                    "legend": {
                        "position": "bottom",
                        "interactive": True,
                        "title": "Right preconditioner",
                    },
                    "settings": {"showPoints": True, "sort": "ascending"},
                },
            ],
            "tables": [
                {
                    "id": "benchmark_detail",
                    "title": "Exact benchmark results",
                    "subtitle": "Solve time excludes recycle-space construction but includes preconditioner and AU=C setup",
                    "dataset": "comparison",
                    "sourceId": "benchmark_source",
                    "defaultSort": {"field": "sort_order", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "sort_order", "label": "Order", "format": "number"},
                        {"field": "preconditioner", "label": "Right preconditioner", "type": "text"},
                        {"field": "variant", "label": "Recycle space", "type": "text"},
                        {"field": "iterations", "label": "Iterations", "format": "number"},
                        {"field": "iteration_ratio_display", "label": "Iterations / baseline", "type": "text"},
                        {"field": "solve_time_seconds", "label": "Solve time (s)", "format": "number"},
                        {"field": "converged", "label": "Converged", "type": "text"},
                    ],
                }
            ],
            "sources": [
                {
                    "id": "benchmark_source",
                    "label": "Deterministic QA/QB benchmark",
                    "path": "benchmark.csv",
                },
                {
                    "id": "angle_source",
                    "label": "QA versus QB canonical angles",
                    "path": "principal_angles.csv",
                },
                {
                    "id": "validation_source",
                    "label": "Independent SciPy and Arnoldi validation",
                    "path": "validation.json",
                },
                {
                    "id": "paper_source",
                    "label": "NeurKItt paper",
                    "url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/e88870ec82f2469b0ddf32c817920c68-Paper-Conference.pdf",
                },
                {
                    "id": "code_source",
                    "label": "Official NeurKItt implementation",
                    "url": "https://github.com/smart-JLuo/NeurKItt",
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Answer\n\n**Geometrically, yes—especially for ILU(0). Operationally, not much on this instance.** "
                        f"Jacobi gives a largest QA–QB principal angle of **{jacobi['largest_principal_angle_degrees']:.1f}°** but a median of only **{jacobi['median_principal_angle_degrees']:.1f}°**. "
                        f"ILU(0) changes the space more broadly: **{ilu['largest_principal_angle_degrees']:.1f}°** largest and **{ilu['median_principal_angle_degrees']:.1f}°** median. "
                        f"Yet QB improves on QA by only **{jacobi_qa['iterations']} → {jacobi_qb['iterations']} iterations ({jacobi_qb_gain:.1f}%)** with Jacobi and **{ilu_qa['iterations']} → {ilu_qb['iterations']} ({ilu_qb_gain:.1f}%)** with ILU.\n\n"
                        "Both exact spaces retain the strong oracle benefit, while the random control remains at baseline. The practical conclusion for this N=9,604 sample is that right preconditioning rotates the exact low invariant space, but QA already captures almost all solver-relevant acceleration available at k=10."
                    ),
                },
                {
                    "id": "iterations_heading",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Space quality, not augmentation alone, drives the gain\n\n"
                        "QA and QB cut 73%–75% of the paired baseline iterations. The fixed random orthonormal control gives ratios of **1.001** with Jacobi and **0.991** with ILU, so the improvement is specific to the low-mode content rather than the mere presence of ten dense directions."
                    ),
                },
                {"id": "iteration_chart", "type": "chart", "chartId": "iteration_comparison", "layout": "full"},
                {
                    "id": "angles_heading",
                    "type": "markdown",
                    "sourceId": "angle_source",
                    "body": (
                        "## The angle spectrum separates Jacobi from ILU\n\n"
                        "Jacobi's 40.3° maximum is concentrated in a few directions: six of ten angles are below 10°. ILU changes most of the subspace, with six angles above 30°. This is a material geometric change, even though the two exact spaces remain nearly equivalent as restart accelerators on this right-hand side."
                    ),
                },
                {"id": "angle_chart", "type": "chart", "chartId": "angle_spectrum", "layout": "full"},
                {
                    "id": "results_heading",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Exact results at a common true-residual tolerance\n\n"
                        "All eight runs satisfy **||b-Ax||₂ / ||b||₂ ≤ 1e-8**. Wall-clock values are local warmed medians and should be treated as supporting evidence; iteration ratios are the portable comparison."
                    ),
                },
                {"id": "benchmark_table", "type": "table", "tableId": "benchmark_detail", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "- **Problem:** the validated seed-0 NeurKItt Darcy construction on a 100×100 coefficient grid, producing a 98×98 interior system with **9,604 unknowns** and 47,628 nonzeros.\n"
                        "- **Solver:** right-preconditioned GMRES, restart 40, zero initial guess, k=10, maximum 20,000 Arnoldi steps.\n"
                        "- **QA:** the exact invariant subspace of A for its ten smallest-magnitude eigenvalues.\n"
                        "- **QB:** the real invariant subspace of B=A M⁻¹ for its ten smallest-magnitude modes. The ILU target contains one complex-conjugate pair, retained as its real and imaginary invariant plane.\n"
                        "- **Random:** one fixed Gaussian orthonormal physical candidate basis, shared across both preconditioners.\n"
                        "- **Solve time:** preconditioner setup + AU=C thin QR + median of three warmed Krylov solves. QA/QB eigensolvers and random-basis generation are excluded.\n"
                        "- **Principal angle:** canonical QA-versus-QB angle; the headline angle is the largest, and the full spectrum is shown."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Projected-Arnoldi mapping for QB\n\n"
                        "The validated implementation is unchanged. For every candidate Y it computes **AY=CR** and **U=Y R⁻¹**, so **AU=C** and **CᴴC=I**, then runs Arnoldi on **(I-CCᴴ)A** over right-preconditioned directions.\n\n"
                        "For QB, an eigenbasis Q of B=A M⁻¹ is mapped to the physical candidate **Y=M⁻¹Q** before normalization. Therefore AY=BQ and the resulting projector C spans QB, while U supplies the corresponding physical correction. QA enters as the same physical A-invariant basis used in the validated benchmark. No neural network is instantiated, trained, or evaluated."
                    ),
                },
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "validation_source",
                    "body": (
                        "## Validation and robustness checks\n\n"
                        f"**Assessment: {validation['overall_assessment']}.** Independent SciPy GMRES reproduces all eight iteration counts exactly and passes the original-system tolerance. The prior validated baseline and QA counts also reproduce exactly. Across QA, QB, and random spaces, the maximum ||AU-C||F is **2.75e-13**; the largest audited corrected-operator Arnoldi defect is **3.70e-14**. Targeted invariant-space defects are at most **1.93e-13**, and the largest selected eigenpair residual is **1.03e-12**."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations\n\n"
                        "- This is one deterministic Darcy realization and one right-hand side; it establishes the behavior of the validated regression case, not a distributional claim.\n"
                        "- N=9,604 is the manageable surrogate, not the paper's larger main benchmark.\n"
                        "- SciPy SuperLU ILU(0)-style factorization with natural ordering is not bitwise equivalent to PETSc ILU.\n"
                        "- Exact eigenspace cost is intentionally excluded. These are oracle comparisons, not end-to-end production timings.\n"
                        "- QA and QB may separate differently for other permeability seeds, restart lengths, k values, or right-hand sides."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next checks\n\n"
                        "1. Repeat this exact QA/QB/random design over a declared Darcy seed set and report distributions of largest/median angles and QA-versus-QB iteration deltas.\n"
                        "2. Sweep k and restart jointly; the value of the rotated QB directions may become clearer when the restart bottleneck is tighter.\n"
                        "3. If QB is approximated in future work, keep this oracle result as the upper-bound regression target and retain the true-residual and SciPy cross-checks."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does the broad ILU rotation remain utility-neutral across right-hand sides?\n"
                        "- Are harmonic Ritz or residual-weighted troublesome modes more predictive than smallest-magnitude eigenvalues for this nonnormal B?\n"
                        "- At what k does QB's modest edge become large enough to justify a preconditioner-specific recycle-space construction?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"comparison": comparison, "angles": angles},
        },
        "sources": [
            {
                "id": "benchmark_source",
                "label": "Deterministic QA/QB benchmark",
                "path": "benchmark.csv",
                "query": {
                    "engine": "duckdb",
                    "description": "Reconstruct the eight reviewed method rows from the benchmark CSV.",
                    "language": "sql",
                    "sql": "SELECT row_number() OVER () AS sort_order, method, preconditioner, recycle_space, CASE recycle_space WHEN 'baseline' THEN 'Baseline' WHEN 'QA' THEN 'QA' WHEN 'QB' THEN 'QB' ELSE 'Random' END AS variant, k, iterations, CASE WHEN converged THEN 'Yes' ELSE 'No' END AS converged, solve_time_seconds_excluding_space_construction AS solve_time_seconds, krylov_seconds_median, final_true_relative_residual, iteration_ratio_over_baseline AS iteration_ratio, printf('%.3f', iteration_ratio_over_baseline) AS iteration_ratio_display, restart_cycles FROM read_csv_auto('benchmark.csv')",
                    "tables_used": ["benchmark.csv"],
                    "filters": ["grid_size=100", "seed=0", "rtol=1e-8", "restart=40", "k=10"],
                    "metric_definitions": {
                        "iterations": "Arnoldi expansions until the original-system true relative residual is at most 1e-8.",
                        "solve_time_seconds": "Preconditioner setup plus AU=C normalization plus median warmed Krylov solve; candidate-space construction excluded.",
                        "iteration_ratio": "Method iterations divided by its paired preconditioned baseline iterations.",
                    },
                },
            },
            {
                "id": "angle_source",
                "label": "QA versus QB canonical angles",
                "path": "principal_angles.csv",
                "query": {
                    "engine": "duckdb",
                    "description": "Load all ten QA-versus-QB canonical angles for each right preconditioner.",
                    "language": "sql",
                    "sql": "SELECT preconditioner, canonical_angle_index_largest_first AS angle_rank, principal_angle_degrees FROM read_csv_auto('principal_angles.csv') ORDER BY preconditioner, angle_rank",
                    "tables_used": ["principal_angles.csv"],
                    "filters": ["k=10", "angles ordered largest first"],
                    "metric_definitions": {
                        "principal_angle_degrees": "Canonical angle between orthonormal QA and QB, reported in degrees."
                    },
                },
            },
            {
                "id": "validation_source",
                "label": "Independent SciPy and projected-Arnoldi validation",
                "path": "validation.json",
            },
            {
                "id": "paper_source",
                "label": "Neural Krylov Iteration for Accelerating Linear System Solving",
                "url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/e88870ec82f2469b0ddf32c817920c68-Paper-Conference.pdf",
            },
            {
                "id": "code_source",
                "label": "Official NeurKItt implementation",
                "url": "https://github.com/smart-JLuo/NeurKItt",
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
