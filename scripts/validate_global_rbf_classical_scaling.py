#!/usr/bin/env python3
"""Validate the global Gaussian-RBF classical scaling experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.preconditioners import (
    identity_preconditioner,
    jacobi_preconditioner,
    low_precision_lu_preconditioner,
    rbf_local_cardinal_preconditioner,
)
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked


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


def _fit_power_law(sizes: list[float], values: list[float]) -> tuple[float, float]:
    exponent, intercept = np.polyfit(np.log(sizes), np.log(values), 1)
    return float(exponent), float(intercept)


def _scipy_right_gmres(
    matrix: np.ndarray,
    rhs: np.ndarray,
    apply,
    *,
    rtol: float,
    restart: int,
    max_iterations: int,
) -> dict[str, object]:
    operator = spla.LinearOperator(
        matrix.shape,
        matvec=lambda vector: matrix @ apply(vector),
        dtype=float,
    )
    callbacks: list[float] = []
    transformed, info = spla.gmres(
        operator,
        rhs,
        rtol=rtol,
        atol=0.0,
        restart=restart,
        maxiter=int(np.ceil(max_iterations / restart)),
        callback=callbacks.append,
        callback_type="pr_norm",
    )
    solution = apply(transformed)
    relative_residual = np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs)
    return {
        "scipy_info": int(info),
        "scipy_iterations": len(callbacks),
        "scipy_true_relative_residual": float(relative_residual),
        "scipy_converged": bool(relative_residual <= rtol and info == 0),
    }


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = summary["configuration"]
    systems = _read_csv(args.results_dir / "systems.csv")
    rows = _read_csv(args.results_dir / "benchmark.csv")
    measured = [row for row in rows if row["run_status"] == "measured"]
    skipped = [row for row in rows if row["run_status"] == "not_run_feasibility_cutoff"]

    expected_sizes = [900, 1600, 2500, 4900, 10000]
    size_coverage_pass = sorted(int(row["N"]) for row in systems) == expected_sizes
    method_coverage_pass = all(
        {row["method"] for row in rows if int(row["N"]) == dimension}
        == {"none", "jacobi", "ls_acbf", "fp32_lu"}
        for dimension in expected_sizes
    )
    memory_formula_pass = all(
        int(row["matrix_memory_bytes"]) == 8 * int(row["N"]) ** 2
        for row in systems
    )
    shape_regime_pass = max(float(row["epsilon_times_h"]) for row in systems) - min(
        float(row["epsilon_times_h"]) for row in systems
    ) <= 0.011
    structured_action_pass = max(
        float(row["structured_dense_matvec_relative_defect"]) for row in systems
    ) <= 2.0e-14

    condition_values = [float(row["condition_2_estimate"]) for row in systems]
    condition_monotone_pass = all(
        right > left for left, right in zip(condition_values, condition_values[1:])
    )
    exact_rows = [row for row in systems if row["condition_estimate_measured"] == "True"]
    extrapolated_rows = [
        row for row in systems if row["condition_estimate_measured"] == "False"
    ]
    condition_method_pass = (
        len(exact_rows) == 4
        and len(extrapolated_rows) == 1
        and int(extrapolated_rows[0]["N"]) == 10000
    )
    holdout = summary["condition_extrapolation"]["holdout_validation"]
    condition_holdout_pass = (
        0.5 <= float(holdout["prediction_over_actual"]) <= 1.5
        and float(summary["condition_extrapolation"]["r_squared"]) >= 0.98
    )

    n900_system = assemble_tensor_gaussian_poisson_blocked(30, epsilon=12.0)
    exact_singular_values = la.svdvals(n900_system.A, check_finite=False)
    recomputed_condition = float(exact_singular_values[0] / exact_singular_values[-1])
    recorded_condition = float(next(row for row in systems if row["N"] == "900")["condition_2_estimate"])
    n900_condition_relative_error = abs(recomputed_condition - recorded_condition) / recorded_condition
    n900_condition_pass = n900_condition_relative_error <= 2.0e-12

    convergence_label_pass = all(
        (row["converged"] == "True")
        == (float(row["final_true_relative_residual"]) <= float(configuration["rtol"]))
        for row in measured
    )
    finite_metrics_pass = all(
        np.isfinite(
            [
                float(row["setup_seconds"]),
                float(row["solve_seconds"]),
                float(row["final_true_relative_residual"]),
                float(row["am_inv_rhs_inverse_defect"]),
                float(row["am_inv_random_subspace_condition"]),
            ]
        ).all()
        for row in measured
    )
    cap_consistency_pass = all(
        int(row["gmres_iterations"]) == int(configuration["max_iterations"])
        and row["converged"] == "False"
        for row in measured
    )

    measured_ls = [row for row in measured if row["method"] == "ls_acbf"]
    ls_exponent, ls_intercept = _fit_power_law(
        [float(row["N"]) for row in measured_ls],
        [float(row["setup_seconds"]) for row in measured_ls],
    )
    projected_ls_pass = len(skipped) == 2
    for row in skipped:
        predicted = float(np.exp(ls_intercept + ls_exponent * np.log(float(row["N"]))))
        projected_ls_pass &= abs(
            predicted - float(row["setup_seconds_projected"])
        ) / predicted <= 1.0e-12
        projected_ls_pass &= row["method"] == "ls_acbf"

    fp32_rows = [row for row in measured if row["method"] == "fp32_lu"]
    fp32_deterioration_pass = all(
        right > left
        for left, right in zip(
            [float(row["am_inv_rhs_inverse_defect"]) for row in fp32_rows],
            [float(row["am_inv_rhs_inverse_defect"]) for row in fp32_rows][1:],
        )
    )

    benchmark_lookup = {(row["N"], row["method"]): row for row in rows}
    preconditioners = {
        "none": identity_preconditioner(n900_system.dimension),
        "jacobi": jacobi_preconditioner(n900_system.A),
        "ls_acbf": rbf_local_cardinal_preconditioner(
            n900_system.A,
            n900_system.nodes,
            support_size=int(configuration["ls_acbf_support_size"]),
        ),
    }
    scipy_checks: list[dict[str, object]] = []
    scipy_linear_pass = True
    for method, preconditioner in preconditioners.items():
        check = _scipy_right_gmres(
            n900_system.A,
            n900_system.b,
            preconditioner.apply,
            rtol=float(configuration["rtol"]),
            restart=int(configuration["restart"]),
            max_iterations=int(configuration["max_iterations"]),
        )
        expected = benchmark_lookup[("900", method)]
        check.update(
            {
                "method": method,
                "benchmark_iterations": int(expected["gmres_iterations"]),
                "benchmark_converged": expected["converged"] == "True",
            }
        )
        check["iteration_and_convergence_match"] = (
            check["scipy_iterations"] == check["benchmark_iterations"]
            and check["scipy_converged"] == check["benchmark_converged"]
        )
        scipy_linear_pass &= bool(check["iteration_and_convergence_match"])
        scipy_checks.append(check)

    fp32 = low_precision_lu_preconditioner(n900_system.A)
    fp32_repeat = benchmark_fgmres(
        n900_system.A,
        n900_system.b,
        fp32,
        rtol=float(configuration["rtol"]),
        restart=int(configuration["restart"]),
        max_iterations=int(configuration["max_iterations"]),
        repeats=1,
        warmup=False,
    )
    fp32_expected = benchmark_lookup[("900", "fp32_lu")]
    fp32_repeat_pass = (
        fp32_repeat.result.iterations == int(fp32_expected["gmres_iterations"])
        and fp32_repeat.result.converged == (fp32_expected["converged"] == "True")
        and abs(
            fp32_repeat.final_true_relative_residual
            - float(fp32_expected["final_true_relative_residual"])
        )
        / float(fp32_expected["final_true_relative_residual"])
        <= 0.35
    )
    fp32_nonlinearity_documented = all(
        float(row["preconditioner_additivity_defect"]) >= 1.0e-3
        for row in fp32_rows
    )

    no_ml_pass = bool(configuration["no_ml"]) and bool(
        configuration["no_oracle_subspaces"]
    )
    ready = all(
        (
            size_coverage_pass,
            method_coverage_pass,
            memory_formula_pass,
            shape_regime_pass,
            structured_action_pass,
            condition_monotone_pass,
            condition_method_pass,
            condition_holdout_pass,
            n900_condition_pass,
            convergence_label_pass,
            finite_metrics_pass,
            cap_consistency_pass,
            projected_ls_pass,
            fp32_deterioration_pass,
            scipy_linear_pass,
            fp32_repeat_pass,
            fp32_nonlinearity_documented,
            no_ml_pass,
        )
    )
    validation = {
        "overall_assessment": "Ready to share" if ready else "Needs revision",
        "checks": {
            "all_requested_sizes_present": size_coverage_pass,
            "all_four_methods_represented_at_each_size": method_coverage_pass,
            "dense_matrix_memory_matches_8N_squared": memory_formula_pass,
            "epsilon_h_regime_is_comparable": shape_regime_pass,
            "structured_and_dense_actions_match": structured_action_pass,
            "condition_estimates_increase_monotonically": condition_monotone_pass,
            "condition_method_is_exact_through_4900_and_extrapolated_only_at_10000": condition_method_pass,
            "condition_extrapolation_holdout_within_50_percent": condition_holdout_pass,
            "N900_condition_recomputes": n900_condition_pass,
            "convergence_flags_match_true_residual_rule": convergence_label_pass,
            "all_measured_metrics_are_finite": finite_metrics_pass,
            "all_measured_runs_consistently_reach_the_cap_without_convergence": cap_consistency_pass,
            "LS_ACBF_setup_projections_recompute": projected_ls_pass,
            "FP32_inverse_response_deteriorates_monotonically": fp32_deterioration_pass,
            "SciPy_GMRES_matches_linear_preconditioner_cap_and_convergence": scipy_linear_pass,
            "FP32_FGMRES_repeat_reproduces": fp32_repeat_pass,
            "FP32_nonlinearity_is_measured_and_documented": fp32_nonlinearity_documented,
            "no_ML_or_oracle_spaces": no_ml_pass,
        },
        "calculation_spot_checks": {
            "N900_condition_2_recomputed": recomputed_condition,
            "N900_condition_relative_error": n900_condition_relative_error,
            "LS_ACBF_setup_power_law_exponent": ls_exponent,
            "N4900_condition_holdout_prediction_over_actual": float(
                holdout["prediction_over_actual"]
            ),
        },
        "independent_scipy_linear_GMRES_checks": scipy_checks,
        "FP32_validation": {
            "reason_for_separate_validation": (
                "FP32 triangular solves introduce roundoff-dependent non-additivity; "
                "the existing flexible-GMRES implementation is appropriate, while a "
                "standard fixed-operator SciPy GMRES cross-check is not equivalent."
            ),
            "repeat_iterations": fp32_repeat.result.iterations,
            "repeat_true_relative_residual": fp32_repeat.final_true_relative_residual,
            "repeat_pass": fp32_repeat_pass,
        },
        "caveats": [
            "All iteration counts are right-censored at 1,000 because no measured method reaches 1e-10.",
            "The N=10,000 condition number is a log-log extrapolation; its largest singular value is measured and the fit is holdout-checked at N=4,900.",
            "LS-ACBF setup is measured through N=2,500 and projected at N=4,900 and 10,000 under a predeclared feasibility cutoff.",
            "Random-compression AM^-1 diagnostics do not estimate the full condition number and can miss tiny singular directions.",
            "FP32 response diagnostics describe an inexact finite-precision map, not the spectrum of one exact linear operator.",
            "Wall-clock timings are local single-thread CPU measurements; memory and iteration-cap outcomes are more portable.",
        ],
    }
    (args.results_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
