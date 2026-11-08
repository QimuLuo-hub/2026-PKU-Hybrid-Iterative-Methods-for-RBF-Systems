"""Validate the saved three-way global-RBF classical benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = {"none", "fp32_lu", "rbf_local_cardinal_ls"}
FAMILIES = {"epsilon", "jitter", "pde_coefficient"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=2.0e-11, atol=1.0e-14, equal_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/global_rbf_classical_preconditioners"),
    )
    args = parser.parse_args()
    output = args.results
    rows = read_csv(output / "benchmark_results.csv")
    summary_rows = read_csv(output / "benchmark_summary.csv")
    probe_rows = read_csv(output / "support_probe.csv")
    failures_text = (output / "failures.csv").read_text(encoding="utf-8")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("row count", len(rows) == 81, f"observed {len(rows)} rows; expected 81")
    system_ids = sorted({row["system_id"] for row in rows})
    check("system count", len(system_ids) == 27, f"observed {len(system_ids)} systems; expected 27")
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row["system_id"]].add(row["method"])
    check(
        "method completeness",
        all(methods == METHODS for methods in grouped.values()),
        "each system contains none, FP32-LU, and local LS-ACBF",
    )
    check(
        "family coverage",
        {row["family"] for row in rows} == FAMILIES,
        f"families={sorted({row['family'] for row in rows})}",
    )
    check("failure ledger", failures_text == "", f"failure-ledger bytes={len(failures_text)}")

    required = {
        "setup_seconds",
        "solve_seconds",
        "fgmres_iterations",
        "final_true_residual_norm",
        "final_true_relative_residual",
        "preconditioner_memory_bytes",
        "kernel",
        "epsilon",
        "N",
        "node_family",
        "pde_parameter",
    }
    missing = sorted(required - set(rows[0]))
    check("required fields", not missing, f"missing fields={missing}")

    numeric_fields = (
        "setup_seconds",
        "solve_seconds",
        "final_true_residual_norm",
        "final_true_relative_residual",
        "preconditioner_memory_bytes",
        "matrix_memory_bytes",
    )
    finite_nonnegative = all(
        np.isfinite(float(row[field])) and float(row[field]) >= 0.0
        for row in rows
        for field in numeric_fields
    )
    check("finite nonnegative measurements", finite_nonnegative, "all timing, residual, and memory values pass")

    convergence_consistent = all(
        as_bool(row["converged"]) == (float(row["final_true_relative_residual"]) <= float(row["fgmres_rtol"]))
        for row in rows
    )
    check(
        "true-residual convergence rule",
        convergence_consistent,
        "saved convergence flag agrees with original-system true relative residual and rtol",
    )

    summary_lookup = {(row["family"], row["method"]): row for row in summary_rows}
    aggregate_ok = True
    for family in ("all", "epsilon", "jitter", "pde_coefficient"):
        family_rows = rows if family == "all" else [row for row in rows if row["family"] == family]
        for method in METHODS:
            selected = [row for row in family_rows if row["method"] == method]
            saved = summary_lookup[(family, method)]
            expected = {
                "systems": len(selected),
                "converged_systems": sum(as_bool(row["converged"]) for row in selected),
                "median_iterations_all": float(np.median([int(row["fgmres_iterations"]) for row in selected])),
                "median_setup_seconds": float(np.median([float(row["setup_seconds"]) for row in selected])),
                "median_solve_seconds": float(np.median([float(row["solve_seconds"]) for row in selected])),
                "median_first_rhs_seconds": float(
                    np.median([float(row["total_seconds_first_rhs"]) for row in selected])
                ),
                "median_preconditioner_memory_bytes": float(
                    np.median([float(row["preconditioner_memory_bytes"]) for row in selected])
                ),
            }
            for field, value in expected.items():
                aggregate_ok &= close(float(saved[field]), float(value))
    check("aggregate recomputation", aggregate_ok, "saved 12-row summary matches independent recomputation")

    probe_by_size: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in probe_rows:
        probe_by_size[int(row["support_size"])].append(row)
    probe_counts = {
        size: sum(as_bool(row["converged"]) for row in selected)
        for size, selected in sorted(probe_by_size.items())
    }
    check(
        "support probe completeness",
        len(probe_rows) == 45 and all(len(selected) == 9 for selected in probe_by_size.values()),
        f"rows={len(probe_rows)}, convergence counts={probe_counts}",
    )
    check(
        "fixed support robustness",
        probe_counts.get(161) == 9 and any(count < 9 for size, count in probe_counts.items() if size < 161),
        f"representative convergence counts by support={probe_counts}",
    )
    figure = output / "classical_preconditioner_comparison.png"
    check("figure artifact", figure.exists() and figure.stat().st_size > 20_000, f"bytes={figure.stat().st_size}")

    by_system: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_system[row["system_id"]][row["method"]] = row
    paired = []
    for system_id, methods in by_system.items():
        fp32 = methods["fp32_lu"]
        acbf = methods["rbf_local_cardinal_ls"]
        setup_difference = float(acbf["setup_seconds"]) - float(fp32["setup_seconds"])
        solve_saving = float(fp32["solve_seconds"]) - float(acbf["solve_seconds"])
        paired.append(
            {
                "system_id": system_id,
                "family": fp32["family"],
                "parameter_value": float(fp32["family_parameter_value"]),
                "fp32_iterations": int(fp32["fgmres_iterations"]),
                "acbf_iterations": int(acbf["fgmres_iterations"]),
                "iteration_ratio": int(acbf["fgmres_iterations"]) / int(fp32["fgmres_iterations"]),
                "solve_time_ratio": float(acbf["solve_seconds"]) / float(fp32["solve_seconds"]),
                "first_rhs_time_ratio": float(acbf["total_seconds_first_rhs"])
                / float(fp32["total_seconds_first_rhs"]),
                "acbf_break_even_rhs": setup_difference / solve_saving if solve_saving > 0.0 else None,
            }
        )

    overall = {
        method: next(row for row in summary_rows if row["family"] == "all" and row["method"] == method)
        for method in METHODS
    }
    acbf_winners = [item for item in paired if item["acbf_iterations"] < item["fp32_iterations"]]
    evidence = {
        "validation_status": "Ready to share" if all(item["passed"] for item in checks) else "Needs revision",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "systems": len(system_ids),
        "benchmark_rows": len(rows),
        "converged_systems": {
            method: int(float(overall[method]["converged_systems"])) for method in METHODS
        },
        "median_iterations": {
            method: float(overall[method]["median_iterations_all"]) for method in METHODS
        },
        "median_setup_seconds": {
            method: float(overall[method]["median_setup_seconds"]) for method in METHODS
        },
        "median_solve_seconds": {
            method: float(overall[method]["median_solve_seconds"]) for method in METHODS
        },
        "median_first_rhs_seconds": {
            method: float(overall[method]["median_first_rhs_seconds"]) for method in METHODS
        },
        "median_preconditioner_memory_bytes": {
            method: float(overall[method]["median_preconditioner_memory_bytes"]) for method in METHODS
        },
        "acbf_vs_fp32": {
            "systems_with_fewer_iterations": len(acbf_winners),
            "winner_systems": acbf_winners,
            "median_paired_iteration_ratio": float(np.median([item["iteration_ratio"] for item in paired])),
            "median_paired_solve_time_ratio": float(np.median([item["solve_time_ratio"] for item in paired])),
            "median_paired_first_rhs_time_ratio": float(
                np.median([item["first_rhs_time_ratio"] for item in paired])
            ),
            "preconditioner_memory_ratio": float(
                overall["rbf_local_cardinal_ls"]["median_preconditioner_memory_bytes"]
            )
            / float(overall["fp32_lu"]["median_preconditioner_memory_bytes"]),
        },
        "support_probe_converged_of_9": probe_counts,
        "fixed_support_size": 161,
        "fixed_support_fraction": 161 / 225,
        "recommendation": "Use FP32-LU as primary M; retain local LS-ACBF as an RBF-specific secondary baseline.",
    }
    (output / "evidence_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    validation = {"status": evidence["validation_status"], "checks": checks}
    (output / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    lines = [
        "# Validation report",
        "",
        f"**Overall assessment: {evidence['validation_status']}.**",
        "",
        "The validation independently checked result completeness, requested fields, convergence against the saved true residual, aggregate calculations, support-probe coverage, and the rendered figure artifact.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['name']} | {'Pass' if item['passed'] else 'Fail'} | {item['detail']} |")
    lines.extend(
        [
            "",
            "The implementation-level unit test separately recomputes `b - A @ x` from the original unpreconditioned matrix and right-hand side and compares it with both saved true-residual helpers.",
            "",
            "Timing values are host-specific. The fixed-N Gaussian family does not establish scaling behavior for the local-cardinal construction.",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["validation_status"], "passed": evidence["checks_passed"], "total": evidence["checks_total"]}))


if __name__ == "__main__":
    main()
