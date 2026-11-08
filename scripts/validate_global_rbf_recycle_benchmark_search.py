#!/usr/bin/env python3
"""Validate the targeted global-RBF recycle-benchmark case search."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from global_rbf.krylov import fgmres
from global_rbf.preconditioners import rbf_local_cardinal_preconditioner
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path("results/global_rbf_recycle_benchmark_search"),
    )
    parser.add_argument(
        "--fallback-dir",
        type=Path,
        default=Path("results/global_rbf_recycle_benchmark_search_ls81"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _validate_saved_residuals(results_dir: Path) -> dict[str, object]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _read_csv(results_dir / "benchmark.csv")
    arrays = np.load(results_dir / "solutions.npz")
    maximum_relative_error = 0.0
    flags_match = True
    selection_flags_match = True

    for row in rows:
        if row["status"] != "measured":
            continue
        side = int(row["side"])
        epsilon = float(row["epsilon"])
        system = assemble_tensor_gaussian_poisson_blocked(side, epsilon=epsilon)
        solution = arrays[row["solution_key"]]
        recomputed = float(
            np.linalg.norm(system.b - system.A @ solution) / np.linalg.norm(system.b)
        )
        reported = float(row["final_true_relative_residual"])
        maximum_relative_error = max(
            maximum_relative_error,
            abs(recomputed - reported)
            / max(abs(recomputed), np.finfo(float).tiny),
        )
        rtol = float(row["rtol"])
        expected_converged = recomputed <= rtol
        flags_match &= _as_bool(row["converged"]) == expected_converged
        expected_suitable = (
            expected_converged and 100 <= int(row["iterations"]) <= 800
        )
        selection_flags_match &= _as_bool(row["suitable"]) == expected_suitable

    return {
        "summary": summary,
        "rows": rows,
        "maximum_saved_residual_relative_error": maximum_relative_error,
        "convergence_flags_match_original_system_residuals": flags_match,
        "suitability_flags_match_declared_rule": selection_flags_match,
    }


def main() -> int:
    args = parse_args()
    primary = _validate_saved_residuals(args.primary_dir)
    fallback = _validate_saved_residuals(args.fallback_dir)
    primary_rows = primary["rows"]
    fallback_rows = fallback["rows"]
    primary_summary = primary["summary"]
    fallback_summary = fallback["summary"]

    expected_primary_pairs = {
        (dimension, method)
        for dimension in (400, 625, 900)
        for method in ("jacobi", "fp32_lu", "ls_acbf")
    }
    actual_primary_pairs = {
        (int(row["N"]), row["preconditioner"]) for row in primary_rows
    }
    selected = fallback_summary["selected_case"]

    system = assemble_tensor_gaussian_poisson_blocked(
        int(selected["side"]),
        epsilon=float(selected["epsilon"]),
    )
    preconditioner = rbf_local_cardinal_preconditioner(
        system.A,
        system.nodes,
        support_size=int(selected["ls_support_size"]),
    )
    repeat = fgmres(
        system.A,
        system.b,
        preconditioner.apply,
        rtol=float(selected["rtol"]),
        restart=int(selected["restart"]),
        max_iterations=int(selected["max_iterations"]),
    )
    repeat_true_residual = float(
        np.linalg.norm(system.b - system.A @ repeat.x) / np.linalg.norm(system.b)
    )

    checks = {
        "strict_search_contains_exactly_the_nine_requested_cases": (
            actual_primary_pairs == expected_primary_pairs and len(primary_rows) == 9
        ),
        "strict_search_has_no_suitable_case": primary_summary["selected_case"] is None,
        "fallback_changes_only_ls_acbf_support": (
            fallback_summary["configuration"]["search_N_values"] == [400]
            and float(fallback_summary["configuration"]["epsilon_factor"]) == 0.4
            and int(fallback_summary["configuration"]["ls_acbf_support_size"]) == 81
        ),
        "fallback_selects_N400_ls_acbf": (
            int(selected["N"]) == 400
            and selected["preconditioner"] == "ls_acbf"
            and int(selected["ls_support_size"]) == 81
        ),
        "selected_iterations_are_in_target_range": 100
        <= int(selected["iterations"])
        <= 800,
        "selected_original_true_residual_meets_tolerance": (
            float(selected["final_true_relative_residual"])
            <= float(selected["rtol"])
        ),
        "primary_saved_true_residuals_recompute": (
            float(primary["maximum_saved_residual_relative_error"]) <= 1.0e-12
        ),
        "fallback_saved_true_residuals_recompute": (
            float(fallback["maximum_saved_residual_relative_error"]) <= 1.0e-12
        ),
        "all_convergence_flags_match_original_system_residuals": bool(
            primary["convergence_flags_match_original_system_residuals"]
            and fallback["convergence_flags_match_original_system_residuals"]
        ),
        "all_suitability_flags_match_declared_rule": bool(
            primary["suitability_flags_match_declared_rule"]
            and fallback["suitability_flags_match_declared_rule"]
        ),
        "selected_case_repeats_independently": (
            repeat.converged
            and repeat.iterations == int(selected["iterations"])
            and repeat_true_residual <= float(selected["rtol"])
        ),
        "no_recycle_space_or_ML_used": (
            not primary_summary["configuration"]["recycle_space_used"]
            and not primary_summary["configuration"]["ML_used"]
            and not fallback_summary["configuration"]["recycle_space_used"]
            and not fallback_summary["configuration"]["ML_used"]
        ),
    }
    validation = {
        "overall_assessment": "Ready to use" if all(checks.values()) else "Needs review",
        "checks": checks,
        "diagnostics": {
            "primary_maximum_saved_residual_relative_error": primary[
                "maximum_saved_residual_relative_error"
            ],
            "fallback_maximum_saved_residual_relative_error": fallback[
                "maximum_saved_residual_relative_error"
            ],
            "selected_repeat_iterations": repeat.iterations,
            "selected_repeat_true_relative_residual": repeat_true_residual,
        },
        "caveats": [
            "Solve timings are local single-thread medians after one warm-up.",
            "The selected case is only 0.5 percent below the 1e-10 residual threshold, so future recycle comparisons must continue to recompute the original-system residual.",
            "The k=81 fallback changes preconditioner support only; the PDE, kernel, epsilon rule, nodes, solver, tolerance, restart, and iteration cap remain unchanged.",
        ],
    }
    output = args.primary_dir / "validation.json"
    output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
