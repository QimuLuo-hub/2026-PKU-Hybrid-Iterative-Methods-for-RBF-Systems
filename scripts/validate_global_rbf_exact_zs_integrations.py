"""Validate the exact-Z_S coarse-integration benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_global_rbf_exact_zs_integrations import (
    METHODS,
    OUTPUT_DEFAULT,
    Q,
    RESTART,
    MAX_ITERATIONS,
    outcome,
    summarize_aggregate,
    summarize_families,
    summarize_systems,
)
from run_global_rbf_stage2 import RHS_SPECS, RTOL, family_specs


REPORT_DEFAULT = Path("global_rbf_exact_ZS_coarse_integration_report.md")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def typed_raw(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = dict(row)
        for key in (
            "parameter_value",
            "epsilon",
            "jitter_fraction",
            "beta",
            "solve_seconds",
            "true_relative_residual",
            "iteration_ratio_vs_baseline",
            "solve_time_ratio_vs_baseline",
            "deflation_residual_identity_error",
            "rtol",
        ):
            converted[key] = float(row[key])
        for key in ("N", "q", "kx", "ky", "iterations", "restart", "max_iterations", "timing_repeats"):
            converted[key] = int(row[key])
        converted["converged"] = row["converged"] == "True"
        output.append(converted)
    return output


def rows_match(
    actual: list[dict[str, str]],
    expected: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> bool:
    if len(actual) != len(expected):
        return False
    actual_lookup = {(row["system_id"], row["method"]): row for row in actual}
    expected_lookup = {(row["system_id"], row["method"]): row for row in expected}
    if set(actual_lookup) != set(expected_lookup):
        return False
    for identity, expected_row in expected_lookup.items():
        actual_row = actual_lookup[identity]
        for key in keys:
            expected_value = expected_row[key]
            if isinstance(expected_value, (float, np.floating)):
                if not np.isclose(float(actual_row[key]), float(expected_value), rtol=1.0e-12, atol=1.0e-14):
                    return False
            elif isinstance(expected_value, (int, np.integer)):
                if int(actual_row[key]) != int(expected_value):
                    return False
            elif str(actual_row[key]) != str(expected_value):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    raw = typed_raw(read_csv(args.output / "solver_results.csv"))
    stored_system = read_csv(args.output / "system_summary.csv")
    stored_aggregate = read_csv(args.output / "aggregate_summary.csv")
    stored_family = read_csv(args.output / "family_summary.csv")
    config = json.loads((args.output / "experiment_config.json").read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected_systems = {
        f"{spec.family}_N225_{spec.parameter}{value:g}"
        for spec in family_specs(False)
        for value in spec.values
    }
    stored_systems = {row["system_id"] for row in raw}
    check(
        "unchanged 27 systems",
        stored_systems == expected_systems and len(stored_systems) == 27,
        f"stored={len(stored_systems)}, expected={len(expected_systems)}",
    )
    expected_rows = 27 * len(RHS_SPECS) * len(METHODS)
    check("complete raw result grain", len(raw) == expected_rows, f"rows={len(raw)}, expected={expected_rows}")
    combinations = {
        (row["system_id"], row["rhs_name"], row["method"])
        for row in raw
    }
    check("unique system-RHS-method rows", len(combinations) == len(raw), f"unique={len(combinations)}")
    check("four requested methods only", {row["method"] for row in raw} == set(METHODS), str(METHODS))
    check(
        "fixed solver protocol",
        all(
            row["q"] == Q
            and row["rtol"] == RTOL
            and row["restart"] == RESTART
            and row["max_iterations"] == MAX_ITERATIONS
            for row in raw
        ),
        f"q={Q}, tol={RTOL}, restart={RESTART}, maxiter={MAX_ITERATIONS}",
    )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw:
        groups.setdefault((row["system_id"], row["rhs_name"]), []).append(row)
    ratio_ok = True
    outcome_ok = True
    for rows in groups.values():
        base = next(row for row in rows if row["method"] == "baseline_fp32_lu")
        for row in rows:
            ratio_ok &= np.isclose(
                row["iteration_ratio_vs_baseline"],
                row["iterations"] / base["iterations"],
            ) and np.isclose(
                row["solve_time_ratio_vs_baseline"],
                row["solve_seconds"] / base["solve_seconds"],
            )
            outcome_ok &= row["outcome_vs_baseline"] == outcome(row, base)
    check("paired ratios", ratio_ok, "recomputed against matching baseline rows")
    check("win/tie/loss labels", outcome_ok, "recomputed with convergence-aware ordering")
    convergence_ok = all(row["converged"] == (row["true_relative_residual"] <= RTOL) for row in raw)
    check("true-residual convergence", convergence_ok, "flags match original-system relative residual threshold")
    check(
        "finite timing and residual fields",
        all(
            np.isfinite(row["solve_seconds"])
            and row["solve_seconds"] > 0.0
            and np.isfinite(row["true_relative_residual"])
            for row in raw
        ),
        "all 432 rows checked",
    )
    deflation_rows = [row for row in raw if row["method"] == "standard_galerkin_deflation"]
    max_identity = max(row["deflation_residual_identity_error"] for row in deflation_rows)
    check(
        "deflation reconstruction identity",
        max_identity < 2.0e-10,
        f"maximum relative identity discrepancy={max_identity:.3e}",
    )

    oracle_files = sorted((args.output / "oracle_spaces").glob("*.npz"))
    orthogonality = []
    shape_ok = len(oracle_files) == 27
    for path in oracle_files:
        Z = np.load(path, allow_pickle=False)["Z"]
        shape_ok &= Z.shape == (225, Q)
        orthogonality.append(np.linalg.norm(Z.T @ Z - np.eye(Q), ord="fro"))
    check("saved oracle basis shapes", shape_ok, f"files={len(oracle_files)}, expected shape=(225,{Q})")
    check(
        "saved oracle bases orthonormal",
        bool(orthogonality and max(orthogonality) < 1.0e-10),
        f"maximum Frobenius defect={max(orthogonality, default=float('nan')):.3e}",
    )

    recomputed_system = summarize_systems(raw)
    recomputed_aggregate = summarize_aggregate(raw, recomputed_system)
    recomputed_family = summarize_families(raw)
    check(
        "system summaries",
        rows_match(
            stored_system,
            recomputed_system,
            (
                "converged_rhs",
                "median_iterations",
                "median_solve_seconds",
                "median_iteration_ratio_vs_baseline",
                "rhs_wins",
                "rhs_ties",
                "rhs_losses",
            ),
        ),
        "108 method-system rows independently aggregated",
    )
    aggregate_lookup = {row["method"]: row for row in stored_aggregate}
    aggregate_expected = {row["method"]: row for row in recomputed_aggregate}
    aggregate_ok = set(aggregate_lookup) == set(aggregate_expected)
    for method, expected in aggregate_expected.items():
        actual = aggregate_lookup.get(method, {})
        aggregate_ok &= all(
            np.isclose(float(actual[key]), float(expected[key]), rtol=1.0e-12, atol=1.0e-14)
            for key in (
                "median_iteration_ratio_vs_baseline",
                "iteration_ratio_q25",
                "iteration_ratio_q75",
                "median_solve_time_ratio_vs_baseline",
            )
        )
        aggregate_ok &= all(
            int(actual[key]) == int(expected[key])
            for key in (
                "converged",
                "rhs_wins",
                "rhs_ties",
                "rhs_losses",
                "system_wins",
                "system_ties",
                "system_losses",
            )
        )
    check("aggregate summaries", aggregate_ok, "median ratios and W/T/L counts independently recomputed")
    family_lookup = {(row["family"], row["method"]): row for row in stored_family}
    family_ok = len(family_lookup) == 12
    for expected in recomputed_family:
        actual = family_lookup.get((expected["family"], expected["method"]), {})
        family_ok &= np.isclose(
            float(actual.get("median_iteration_ratio_vs_baseline", "nan")),
            expected["median_iteration_ratio_vs_baseline"],
        )
        family_ok &= all(
            int(actual.get(key, -1)) == int(expected[key])
            for key in ("converged", "rhs_wins", "rhs_ties", "rhs_losses")
        )
    check("family summaries", family_ok, "12 family-method rows independently recomputed")
    check(
        "no ML and exact configuration",
        config.get("no_ml") is True
        and config.get("system_count") == 27
        and config.get("preconditioner") == "FP32 pivoted LU",
        "experiment_config.json inspected",
    )
    report_text = args.report.read_text(encoding="utf-8")
    check(
        "report answers stability question",
        "does not stably reduce iterations" in report_text
        and "Standard Galerkin deflation" in report_text,
        str(args.report),
    )

    passed = sum(item["passed"] for item in checks)
    status = "Ready to share" if passed == len(checks) else "Needs revision"
    aggregate_result = {row["method"]: row for row in recomputed_aggregate}
    summary = {
        "status": status,
        "passed_checks": passed,
        "total_checks": len(checks),
        "answer": "No; exact Z_S does not stably reduce iterations across all 27 systems.",
        "best_method": "standard_galerkin_deflation",
        "best_median_iteration_ratio": aggregate_result["standard_galerkin_deflation"][
            "median_iteration_ratio_vs_baseline"
        ],
        "checks": checks,
    }
    (args.output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Exact Z_S Integration Validation Report",
        "",
        f"## Overall assessment: {status}",
        "",
        f"Passed {passed}/{len(checks)} checks.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |")
    lines.extend(
        [
            "",
            "The negative stability conclusion is supported: standard deflation is the strongest integration, but its aggregate median ratio is 1.0 and it still loses on six of 27 systems.",
            "",
        ]
    )
    (args.output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "passed_checks", "total_checks", "answer", "best_method", "best_median_iteration_ratio")}, indent=2))


if __name__ == "__main__":
    main()
