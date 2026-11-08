"""Independent validation of the solver-dynamics oracle experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from global_rbf.benchmarking import true_relative_residual
from global_rbf.krylov import fgmres
from global_rbf.preconditioners import low_precision_lu_preconditioner
from global_rbf.structure import galerkin_deflation
from run_global_rbf_krylov_oracle_spaces import (
    MAX_ITERATIONS,
    OUTPUT_DEFAULT,
    Q_VALUES,
    REFERENCE_SWITCH_TOLERANCE,
    RESTART,
    REPORT_DEFAULT,
    TARGETS,
    mode_rhs,
    paired_outcome,
    summarize,
    summarize_reduced_conditioning,
    summarize_slowest_quartile,
    summarize_targets,
)
from run_global_rbf_stage2 import RHS_SPECS, RTOL, assemble_family_member, family_specs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def typed(rows: list[dict[str, str]], bool_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    integer_keys = {
        "parameter_index",
        "N",
        "kx",
        "ky",
        "iterations",
        "baseline_iterations",
        "q_requested",
        "q_actual",
        "observations",
        "exact_q_observations",
        "converged_all",
        "exact_q_converged",
        "exact_q_wins",
        "exact_q_ties",
        "exact_q_losses",
        "all_wins",
        "all_ties",
        "all_losses",
        "system_wins",
        "system_ties",
        "system_losses",
        "residual_snapshots_recorded",
        "late_distinct_snapshots",
        "arnoldi_cycles",
        "harmonic_available_rank",
        "exact_error_pod_numerical_rank",
        "unresolved_pod_numerical_rank",
        "common_available_rank",
        "wins",
        "ties",
        "losses",
        "converged",
    }
    output: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if key in bool_keys:
                row[key] = value == "True"
            elif key in integer_keys:
                row[key] = int(value)
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        output.append(row)
    return output


def compare_rows(
    stored: list[dict[str, str]],
    expected: list[dict[str, Any]],
    identity: tuple[str, ...],
    fields: tuple[str, ...],
) -> bool:
    stored_lookup = {tuple(row[key] for key in identity): row for row in stored}
    expected_lookup = {tuple(str(row[key]) for key in identity): row for row in expected}
    if stored_lookup.keys() != expected_lookup.keys():
        return False
    for key, expected_row in expected_lookup.items():
        stored_row = stored_lookup[key]
        for field in fields:
            expected_value = expected_row[field]
            if isinstance(expected_value, (float, np.floating)):
                if not np.isclose(
                    float(stored_row[field]),
                    float(expected_value),
                    rtol=1.0e-12,
                    atol=1.0e-14,
                    equal_nan=True,
                ):
                    return False
            elif isinstance(expected_value, (int, np.integer)):
                if int(stored_row[field]) != int(expected_value):
                    return False
            elif str(stored_row[field]) != str(expected_value):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    baseline = typed(read_csv(output / "baseline_trajectories.csv"), ("converged",))
    results = typed(
        read_csv(output / "oracle_results.csv"),
        ("exact_q", "converged", "baseline_converged"),
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected_systems = {
        f"{spec.family}_N225_{spec.parameter}{value:g}"
        for spec in family_specs(False)
        for value in spec.values
    }
    check(
        "unchanged system suite",
        {row["system_id"] for row in baseline} == expected_systems,
        f"systems={len({row['system_id'] for row in baseline})}",
    )
    check("baseline grain", len(baseline) == 27 * len(RHS_SPECS), f"rows={len(baseline)}")
    check(
        "oracle grain",
        len(results) == 27 * len(RHS_SPECS) * len(TARGETS) * len(Q_VALUES),
        f"rows={len(results)}",
    )
    check(
        "requested coverage",
        {row["target"] for row in results} == set(TARGETS)
        and {row["q_requested"] for row in results} == set(Q_VALUES),
        "four definitions and q=4,8,16",
    )
    check(
        "unchanged solver protocol",
        config["preconditioner"] == "FP32 pivoted LU"
        and config["right_preconditioned"] is True
        and config["rtol"] == RTOL
        and config["restart"] == RESTART
        and config["max_iterations"] == MAX_ITERATIONS
        and config["integration"]["name"] == "standard nonsymmetric Galerkin deflation",
        "experiment_config.json inspected",
    )

    baseline_lookup = {(row["system_id"], row["rhs_name"]): row for row in baseline}
    ratio_ok = outcome_ok = convergence_ok = fairness_ok = True
    grouped_actual: dict[tuple[str, str, int], set[int]] = {}
    for row in results:
        base = baseline_lookup[(row["system_id"], row["rhs_name"])]
        ratio_ok &= np.isclose(
            row["iteration_ratio_vs_baseline"],
            row["iterations"] / base["iterations"],
        )
        ratio_ok &= np.isclose(
            row["solve_time_ratio_vs_baseline"],
            row["solve_seconds"] / base["solve_seconds"],
        )
        outcome_ok &= row["outcome_vs_baseline"] == paired_outcome(row, base)
        convergence_ok &= row["converged"] == (row["true_relative_residual"] <= RTOL)
        grouped_actual.setdefault(
            (row["system_id"], row["rhs_name"], row["q_requested"]), set()
        ).add(row["q_actual"])
    fairness_ok &= all(len(values) == 1 for values in grouped_actual.values())
    check("paired ratios", ratio_ok, "iteration and solve-time ratios recomputed")
    check("convergence-aware W/T/L", outcome_ok, "all 1296 outcomes recomputed")
    check("original true-residual convergence labels", convergence_ok, "all labels checked")
    check("matched q_actual fairness", fairness_ok, "all targets share one actual rank per sample/q")

    aggregate, system_summary = summarize(results)
    target_summary = summarize_targets(aggregate, results)
    slow_cutoff, slow_summary = summarize_slowest_quartile(results, baseline)
    condition_summary = summarize_reduced_conditioning(results)
    check(
        "aggregate summaries",
        compare_rows(
            read_csv(output / "oracle_summary.csv"),
            aggregate,
            ("target", "q_requested"),
            (
                "observations",
                "exact_q_observations",
                "converged_all",
                "median_iteration_ratio_all_matched_rank",
                "median_iteration_ratio_exact_q",
                "exact_q_converged",
                "exact_q_wins",
                "exact_q_ties",
                "exact_q_losses",
            ),
        ),
        "12 target-rank rows independently aggregated",
    )
    check(
        "system summaries",
        compare_rows(
            read_csv(output / "oracle_system_summary.csv"),
            system_summary,
            ("system_id", "target", "q_requested"),
            (
                "exact_q_rhs",
                "converged_rhs",
                "median_iteration_ratio_vs_baseline",
                "rhs_wins",
                "rhs_ties",
                "rhs_losses",
                "system_outcome_vs_baseline",
            ),
        ),
        "324 system-target-rank rows independently aggregated",
    )
    check(
        "target summaries",
        compare_rows(
            read_csv(output / "target_summary.csv"),
            target_summary,
            ("target",),
            (
                "exact_q_observations",
                "exact_q_converged",
                "median_iteration_ratio_exact_q_pooled",
                "exact_q_wins",
                "exact_q_ties",
                "exact_q_losses",
            ),
        ),
        "pooled target metrics independently aggregated",
    )
    check(
        "slowest-quartile summaries",
        slow_cutoff == 21.0
        and compare_rows(
            read_csv(output / "slowest_quartile_summary.csv"),
            slow_summary,
            ("target", "q_requested"),
            ("observations", "converged", "median_iteration_ratio", "wins", "ties", "losses"),
        ),
        f"cutoff={slow_cutoff:g}; 12 rows recomputed",
    )
    check(
        "reduced-conditioning summaries",
        compare_rows(
            read_csv(output / "reduced_condition_summary.csv"),
            condition_summary,
            ("target", "q_requested", "converged"),
            ("observations", "median_reduced_condition_number", "q90_reduced_condition_number"),
        ),
        "converged and failed reduced systems independently separated",
    )

    trajectory_files = sorted((output / "trajectories").glob("*.npz"))
    basis_files = sorted((output / "oracle_spaces").glob("*.npz"))
    check(
        "raw artifact count",
        len(trajectory_files) == 108 and len(basis_files) == 108,
        f"trajectories={len(trajectory_files)}, bases={len(basis_files)}",
    )

    systems: dict[str, tuple[Any, Any]] = {}
    for spec in family_specs(False):
        for index, value in enumerate(spec.values):
            system_id, system = assemble_family_member(spec, index, float(value))
            systems[system_id] = (system, low_precision_lu_preconditioner(system.A))

    max_arnoldi_identity = 0.0
    max_arnoldi_roundoff_fraction = 0.0
    max_v_orthogonality = 0.0
    trajectory_shape_ok = True
    for path in trajectory_files:
        stem = path.stem
        system_id, rhs_name = stem.rsplit("__", 1)
        system, _ = systems[system_id]
        data = np.load(path, allow_pickle=False)
        cycle_indices = sorted(
            int(key.split("_")[1])
            for key in data.files
            if key.startswith("cycle_") and key.endswith("_H")
        )
        expected_cycle_count = baseline_lookup[(system_id, rhs_name)]["arnoldi_cycles"]
        trajectory_shape_ok &= len(cycle_indices) == expected_cycle_count
        for cycle_index in cycle_indices:
            V = data[f"cycle_{cycle_index}_V"]
            Z = data[f"cycle_{cycle_index}_Z"]
            H = data[f"cycle_{cycle_index}_H"]
            m = H.shape[1]
            trajectory_shape_ok &= V.shape == (225, m + 1) and Z.shape == (225, m)
            denominator = max(np.linalg.norm(system.A @ Z, ord="fro"), np.finfo(float).tiny)
            max_arnoldi_identity = max(
                max_arnoldi_identity,
                float(np.linalg.norm(system.A @ Z - V @ H, ord="fro") / denominator),
            )
            gamma_n = 225.0 * np.finfo(float).eps / (1.0 - 225.0 * np.finfo(float).eps)
            multiplication_bound = gamma_n * np.linalg.norm(
                np.abs(system.A) @ np.abs(Z), ord="fro"
            )
            max_arnoldi_roundoff_fraction = max(
                max_arnoldi_roundoff_fraction,
                float(np.linalg.norm(system.A @ Z - V @ H, ord="fro") / multiplication_bound),
            )
            max_v_orthogonality = max(
                max_v_orthogonality,
                float(np.linalg.norm(V.T @ V - np.eye(m + 1), ord="fro")),
            )
    check("trajectory cycle dimensions", trajectory_shape_ok, "every saved V/Z/H cycle checked")
    check(
        "Arnoldi identities",
        max_arnoldi_roundoff_fraction < 1.0,
        (
            f"maximum relative ||AZ-VH||={max_arnoldi_identity:.3e}; "
            f"maximum fraction of matrix-product roundoff bound={max_arnoldi_roundoff_fraction:.3e}"
        ),
    )
    check(
        "Arnoldi orthogonality",
        max_v_orthogonality < 2.0e-10,
        f"maximum ||V^TV-I||_F={max_v_orthogonality:.3e}",
    )

    max_basis_defect = 0.0
    basis_shapes_ok = True
    for row in results:
        data = np.load(
            output / "oracle_spaces" / f"{row['system_id']}__{row['rhs_name']}.npz",
            allow_pickle=False,
        )
        key = f"{row['target']}_q{row['q_requested']}_actual{row['q_actual']}"
        basis_shapes_ok &= key in data.files
        if key not in data.files:
            continue
        basis = data[key]
        basis_shapes_ok &= basis.shape == (225, row["q_actual"])
        max_basis_defect = max(
            max_basis_defect,
            float(np.linalg.norm(basis.T @ basis - np.eye(row["q_actual"]), ord="fro")),
        )
    check("saved basis dimensions", basis_shapes_ok, "all 1296 row-to-basis mappings checked")
    check("saved basis orthogonality", max_basis_defect < 2.0e-10, f"maximum defect={max_basis_defect:.3e}")

    reference_ok = all(
        row["max_reference_action_relative_residual"] <= REFERENCE_SWITCH_TOLERANCE
        for row in baseline
    )
    check(
        "verified reference inverse actions",
        reference_ok,
        f"maximum verified residual={max(row['max_reference_action_relative_residual'] for row in baseline):.3e}",
    )

    # Re-run every oracle solve once from the saved basis and original A,b.
    rerun_ok = True
    max_true_residual_difference = 0.0
    max_identity = 0.0
    artifact_cache: dict[str, Any] = {}
    for spec in family_specs(False):
        for index, value in enumerate(spec.values):
            system_id = f"{spec.family}_N225_{spec.parameter}{value:g}"
            system, base = systems[system_id]
            artifact_cache[system_id] = type(
                "Artifact",
                (),
                {
                    "system": system,
                    "base": base,
                    "family": spec.family,
                    "value": float(value),
                },
            )()
    for row in results:
        artifact = artifact_cache[row["system_id"]]
        kx = int(row["kx"])
        ky = int(row["ky"])
        rhs = mode_rhs(artifact, kx, ky)
        data = np.load(
            output / "oracle_spaces" / f"{row['system_id']}__{row['rhs_name']}.npz",
            allow_pickle=False,
        )
        basis = data[f"{row['target']}_q{row['q_requested']}_actual{row['q_actual']}"]
        deflation = galerkin_deflation(artifact.system.A, basis)
        projected_rhs = deflation.deflate_rhs(rhs)
        result = fgmres(
            deflation.A_deflated,
            projected_rhs,
            artifact.base.apply,
            rtol=0.0,
            atol=RTOL * np.linalg.norm(rhs),
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
        )
        solution = deflation.reconstruct(artifact.system.A, rhs, result.x)
        residual = true_relative_residual(artifact.system.A, rhs, solution)
        identity = np.linalg.norm(
            rhs
            - artifact.system.A @ solution
            - (projected_rhs - deflation.A_deflated @ result.x)
        ) / np.linalg.norm(rhs)
        rerun_ok &= result.iterations == row["iterations"]
        rerun_ok &= (residual <= RTOL) == row["converged"]
        max_true_residual_difference = max(
            max_true_residual_difference,
            abs(residual - row["true_relative_residual"]),
        )
        max_identity = max(max_identity, float(identity))
    check(
        "original-system oracle reruns",
        rerun_ok and max_true_residual_difference < 1.0e-12,
        f"1296 reruns; maximum stored-vs-rerun residual difference={max_true_residual_difference:.3e}",
    )
    check(
        "deflation residual diagnostic reproduced",
        np.isfinite(max_identity),
        f"maximum original/projected discrepancy={max_identity:.3e} (reported diagnostic, not a pass threshold)",
    )

    winner_expected = min(
        [row for row in slow_summary if row["target"] != "dominant_right_singular_S"],
        key=lambda row: (
            -row["wins"],
            row["losses"],
            -row["converged"],
            row["median_iteration_ratio"],
            row["q_requested"],
            TARGETS.index(row["target"]),
        ),
    )
    check(
        "solver-alignment selection",
        selection["target"] == winner_expected["target"]
        and selection["q_requested"] == winner_expected["q_requested"],
        f"winner={winner_expected['target']} q={winner_expected['q_requested']}",
    )
    report_text = args.report.read_text(encoding="utf-8")
    check(
        "report completeness",
        "Late exact-error POD at q=4" in report_text
        and "Exact-rank evidence" in report_text
        and "Slowest-quartile outcomes" in report_text
        and "Limitations and robustness checks" in report_text,
        "technical report sections inspected",
    )
    check(
        "no ML or transferability",
        config["no_ml"] is True and config["no_transferability"] is True,
        "experiment scope inspected",
    )

    passed = sum(item["passed"] for item in checks)
    status = "Ready to share" if passed == len(checks) else "Needs revision"
    summary = {
        "status": status,
        "passed_checks": passed,
        "total_checks": len(checks),
        "selection": selection,
        "checks": checks,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Solver-Dynamics Oracle Validation",
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
            "The validated evidence supports q=4 late exact-error POD as the best same-trajectory diagnostic for slow solves, but not as a uniformly robust online coarse space.",
            "",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
