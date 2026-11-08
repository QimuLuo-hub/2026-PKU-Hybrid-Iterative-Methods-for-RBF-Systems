#!/usr/bin/env python3
"""Build the compact portable report for the large-N Krylov recycle test."""

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
        default=Path("results/global_rbf_krylov_recycle_largeN"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with (args.results_dir / "benchmark.csv").open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    validation = json.loads(
        (args.results_dir / "validation.json").read_text(encoding="utf-8")
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    rows = []
    for order, row in enumerate(raw_rows, start=1):
        q = int(row["q"])
        variant = (
            "Baseline"
            if row["recycle_space"] == "baseline"
            else ("HR" if row["recycle_space"] == "harmonic_ritz" else "Random")
            + f" q={q}"
        )
        rows.append(
            {
                "sort_order": order,
                "method": row["method"],
                "method_display": {
                    "baseline": "Baseline",
                    "harmonic_ritz": "Harmonic Ritz",
                    "random": "Random",
                }[row["recycle_space"]],
                "variant": variant,
                "q_display": "—" if q == 0 else str(q),
                "iterations": int(row["iterations"]),
                "true_residual": float(row["true_relative_residual"]),
                "true_residual_e5": 1.0e5 * float(row["true_relative_residual"]),
                "true_residual_display": f"{float(row['true_relative_residual']):.6e}",
                "solve_seconds": float(row["solve_seconds"]),
                "iteration_ratio": float(row["iteration_ratio"]),
                "iteration_ratio_display": f"{float(row['iteration_ratio']):.3f}",
            }
        )

    title = "Large-N RBF Krylov recycling"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Compact comparison of baseline, exact harmonic-Ritz, and random recycle spaces on the N=4900 Jacobi-preconditioned global Gaussian-RBF case.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "residual_comparison",
                    "title": "True residual after the common 1,000-step cap",
                    "subtitle": "Lower is better; the shared tolerance is 1e-10",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "Did a Krylov-derived recycle space improve the capped true residual?",
                    "rationale": "All iteration counts tie at the cap, so true residual is the only effectiveness discriminator.",
                    "dataset": "results",
                    "sourceId": "benchmark_source",
                    "encodings": {
                        "x": {"field": "variant", "type": "ordinal", "label": "Method"},
                        "y": {
                            "field": "true_residual_e5",
                            "type": "quantitative",
                            "label": "True relative residual (×10⁻⁵)",
                        },
                        "tooltip": [
                            {"field": "iterations", "type": "quantitative", "label": "Iterations"},
                            {"field": "solve_seconds", "type": "quantitative", "label": "Solve time", "unit": "s"},
                            {"field": "iteration_ratio", "type": "quantitative", "label": "Iteration ratio"},
                        ],
                    },
                    "yAxisTitle": "True relative residual (×10⁻⁵)",
                    "layout": "full",
                    "palette": {"kind": "sequential", "name": "residual"},
                    "labels": {"values": "all"},
                    "settings": {"orientation": "vertical", "sort": "custom", "showValues": True},
                }
            ],
            "tables": [
                {
                    "id": "results_table",
                    "title": "N=4,900, right Jacobi, restart 100, rtol 1e-10",
                    "subtitle": "All iteration counts are capped at 1,000",
                    "dataset": "results",
                    "sourceId": "benchmark_source",
                    "defaultSort": {"field": "sort_order", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "sort_order", "label": "#", "format": "number"},
                        {"field": "method_display", "label": "Method", "type": "text"},
                        {"field": "q_display", "label": "q", "type": "text"},
                        {"field": "iterations", "label": "Iterations", "format": "number"},
                        {"field": "true_residual_display", "label": "True relative residual", "type": "text"},
                        {"field": "solve_seconds", "label": "Solve time (s)", "format": "number"},
                        {"field": "iteration_ratio_display", "label": "Iteration ratio", "type": "text"},
                    ],
                }
            ],
            "sources": [
                {"id": "benchmark_source", "label": "Validated large-N recycle benchmark", "path": "benchmark.csv"},
                {"id": "validation_source", "label": "Numerical validation", "path": "validation.json"},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "result_note",
                    "type": "markdown",
                    "sourceId": "benchmark_source",
                    "body": (
                        "## Result\n\nNo tested recycle space reduced the capped iteration count. "
                        "The exact harmonic-Ritz spaces also finished with larger true residuals than baseline, while the matched random controls stayed near baseline."
                    ),
                },
                {"id": "table", "type": "table", "tableId": "results_table", "layout": "full"},
                {"id": "residual_chart", "type": "chart", "chartId": "residual_comparison", "layout": "full"},
                {
                    "id": "timing_note",
                    "type": "markdown",
                    "sourceId": "validation_source",
                    "body": (
                        "## Timing definition\n\nSolve time is the median of three warmed projected-Arnoldi solves. "
                        "It excludes shared Jacobi setup, the baseline extraction run, harmonic-Ritz/random-space construction, and one-time AU=C normalization. "
                        f"Validation: **{validation['overall_assessment']}**."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"results": rows},
        },
        "sources": [
            {
                "id": "benchmark_source",
                "label": "Validated large-N recycle benchmark",
                "path": "benchmark.csv",
                "query": {
                    "engine": "duckdb",
                    "description": "Load the seven requested baseline, harmonic-Ritz, and random-control rows.",
                    "language": "sql",
                    "sql": "SELECT row_number() OVER () AS sort_order, method, q, iterations, true_relative_residual, solve_seconds, iteration_ratio FROM read_csv_auto('benchmark.csv')",
                    "tables_used": ["benchmark.csv"],
                    "filters": ["N=4900", "right Jacobi", "restart=100", "rtol=1e-10", "max_iterations=1000"],
                    "metric_definitions": {
                        "iterations": "Arnoldi expansions, right-censored at 1000.",
                        "true_relative_residual": "||b-Ax||2 / ||b||2 in the original system.",
                        "solve_seconds": "Median of three warmed projected-Arnoldi solves; all construction excluded.",
                        "iteration_ratio": "Method iterations divided by baseline iterations.",
                    },
                },
            },
            {
                "id": "validation_source",
                "label": "Numerical validation",
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
