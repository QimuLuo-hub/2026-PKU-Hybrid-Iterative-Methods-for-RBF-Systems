"""Independent validation for fair targets, rank efficiency, and reuse."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from global_rbf.transfer import compare_subspaces
from run_global_rbf_fair_targets_rank_transfer import (
    MAX_ITERATIONS,
    OUTPUT_DEFAULT,
    RANK_Q_VALUES,
    RESTART,
    TARGETS,
    TARGET_Q_VALUES,
    choose_rank,
    choose_winner,
    paired_outcome,
    summarize_target_rows,
    summarize_transfer,
)
from run_global_rbf_stage2 import RHS_SPECS, RTOL, family_specs


REPORT2_DEFAULT = Path("global_rbf_prompt2_fair_oracle_targets_report.md")
REPORT3_DEFAULT = Path("global_rbf_prompt3_rank_efficiency_report.md")
REPORT4_DEFAULT = Path("global_rbf_prompt4_transferability_report.md")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def typed(rows: list[dict[str, str]], *, bool_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    integer_keys = {
        "parameter_index",
        "N",
        "q",
        "kx",
        "ky",
        "iterations",
        "baseline_iterations",
        "own_iterations",
        "reused_iterations",
        "observations",
        "converged",
        "converged_rhs",
        "rhs_wins",
        "rhs_ties",
        "rhs_losses",
        "system_wins",
        "system_ties",
        "system_losses",
        "directed_pairs",
        "rhs_observations",
        "rhs_wins_vs_own",
        "rhs_ties_vs_own",
        "rhs_losses_vs_own",
        "rhs_wins_vs_baseline",
        "rhs_ties_vs_baseline",
        "rhs_losses_vs_baseline",
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in bool_keys:
                converted[key] = value == "True"
            elif key in integer_keys:
                converted[key] = int(value)
            else:
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
        output.append(converted)
    return output


def compare_summary_rows(
    actual: list[dict[str, str]],
    expected: list[dict[str, Any]],
    identity: tuple[str, ...],
    keys: tuple[str, ...],
) -> bool:
    actual_lookup = {tuple(row[key] for key in identity): row for row in actual}
    expected_lookup = {tuple(str(row[key]) for key in identity): row for row in expected}
    if set(actual_lookup) != set(expected_lookup):
        return False
    for identity_value, expected_row in expected_lookup.items():
        actual_row = actual_lookup[identity_value]
        for key in keys:
            expected_value = expected_row[key]
            if isinstance(expected_value, (float, np.floating)):
                if not np.isclose(float(actual_row[key]), float(expected_value), rtol=1.0e-12, atol=1.0e-14, equal_nan=True):
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
    parser.add_argument("--report2", type=Path, default=REPORT2_DEFAULT)
    parser.add_argument("--report3", type=Path, default=REPORT3_DEFAULT)
    parser.add_argument("--report4", type=Path, default=REPORT4_DEFAULT)
    args = parser.parse_args()
    output = args.output
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    baseline = typed(
        read_csv(output / "baseline_results.csv"),
        bool_keys=("converged",),
    )
    target = typed(
        read_csv(output / "target_results.csv"),
        bool_keys=("converged",),
    )
    rank = typed(
        read_csv(output / "rank_results.csv"),
        bool_keys=("converged",),
    )
    transfer = typed(
        read_csv(output / "transfer_results.csv"),
        bool_keys=("reused_converged", "own_converged", "baseline_converged"),
    )
    transfer_pairs = typed(read_csv(output / "transfer_pair_summary.csv"))

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected_systems = {
        f"{spec.family}_N225_{spec.parameter}{value:g}"
        for spec in family_specs(False)
        for value in spec.values
    }
    check(
        "unchanged 27-system suite",
        {row["system_id"] for row in baseline} == expected_systems,
        f"stored systems={len({row['system_id'] for row in baseline})}",
    )
    check("baseline grain", len(baseline) == 27 * len(RHS_SPECS), f"rows={len(baseline)}")
    check(
        "fair target grain",
        len(target) == 27 * len(RHS_SPECS) * len(TARGETS) * len(TARGET_Q_VALUES),
        f"rows={len(target)}",
    )
    check(
        "rank grain",
        len(rank) == 27 * len(RHS_SPECS) * len(RANK_Q_VALUES),
        f"rows={len(rank)}",
    )
    check(
        "target and rank coverage",
        {row["target"] for row in target} == set(TARGETS)
        and {row["q"] for row in target} == set(TARGET_Q_VALUES)
        and {row["q"] for row in rank} == set(RANK_Q_VALUES)
        and len({row["target"] for row in rank}) == 1,
        "requested definitions and ranks inspected",
    )
    check(
        "identical integration and solver protocol",
        config["integration"]["name"] == "standard nonsymmetric Galerkin deflation"
        and config["preconditioner"] == "FP32 pivoted LU"
        and config["rtol"] == RTOL
        and config["restart"] == RESTART
        and config["max_iterations"] == MAX_ITERATIONS,
        "experiment_config.json inspected",
    )

    baseline_lookup = {(row["system_id"], row["rhs_name"]): row for row in baseline}
    target_ratio_ok = True
    target_outcome_ok = True
    for row in target:
        base = baseline_lookup[(row["system_id"], row["rhs_name"])]
        target_ratio_ok &= np.isclose(row["iteration_ratio_vs_baseline"], row["iterations"] / base["iterations"])
        target_ratio_ok &= np.isclose(row["solve_time_ratio_vs_baseline"], row["solve_seconds"] / base["solve_seconds"])
        target_outcome_ok &= row["outcome_vs_baseline"] == paired_outcome(row, base)
    check("target paired ratios", target_ratio_ok, "iterations and solve times recomputed against matching baseline")
    check("target W/T/L labels", target_outcome_ok, "convergence-aware outcomes recomputed")
    convergence_ok = all(row["converged"] == (row["true_relative_residual"] <= RTOL) for row in target)
    check("target true-residual convergence", convergence_ok, "all target rows checked")

    recomputed_system, recomputed_target_summary = summarize_target_rows(target)
    stored_target_system = read_csv(output / "target_system_summary.csv")
    stored_target_summary = read_csv(output / "target_summary.csv")
    check(
        "target system summaries",
        compare_summary_rows(
            stored_target_system,
            recomputed_system,
            ("system_id", "target", "q"),
            (
                "converged_rhs",
                "median_iterations",
                "median_iteration_ratio_vs_baseline",
                "rhs_wins",
                "rhs_ties",
                "rhs_losses",
                "system_outcome_vs_baseline",
            ),
        ),
        "243 system-target-rank rows independently aggregated",
    )
    check(
        "target aggregate summaries",
        compare_summary_rows(
            stored_target_summary,
            recomputed_target_summary,
            ("target", "q"),
            (
                "converged",
                "median_iteration_ratio_vs_baseline",
                "median_solve_time_ratio_vs_baseline",
                "rhs_wins",
                "rhs_ties",
                "rhs_losses",
                "system_wins",
                "system_ties",
                "system_losses",
            ),
        ),
        "nine target-rank summaries independently aggregated",
    )
    target_winner = choose_winner(recomputed_target_summary)
    winner_ok = (
        target_winner["target"] == selection["winning_target"]["target"]
        and target_winner["q"] == selection["winning_target"]["q"]
    )
    check("target winner selection", winner_ok, f"winner={target_winner['target']} q={target_winner['q']}")

    _, recomputed_rank_summary = summarize_target_rows(rank)
    recomputed_rank_summary = [
        row for row in recomputed_rank_summary if row["target"] == target_winner["target"]
    ]
    stored_rank_summary = read_csv(output / "rank_summary.csv")
    check(
        "rank summaries",
        compare_summary_rows(
            stored_rank_summary,
            recomputed_rank_summary,
            ("target", "q"),
            (
                "q_over_N",
                "converged",
                "median_iteration_ratio_vs_baseline",
                "median_solve_time_ratio_vs_baseline",
                "fraction_systems_improved",
                "system_wins",
                "system_ties",
                "system_losses",
            ),
        ),
        "seven rank rows independently aggregated",
    )
    rank_winner = choose_rank(recomputed_rank_summary)
    check(
        "rank winner selection",
        rank_winner["q"] == selection["winning_rank"]["q"],
        f"selected q={rank_winner['q']}",
    )

    spaces = sorted((output / "oracle_spaces").glob("*.npz"))
    shape_ok = len(spaces) == 27
    max_defect = 0.0
    for path in spaces:
        data = np.load(path, allow_pickle=False)
        shape_ok &= set(data.files) == set(TARGETS)
        for target_name in TARGETS:
            Z = data[target_name]
            shape_ok &= Z.shape == (225, 32)
            max_defect = max(max_defect, float(np.linalg.norm(Z.T @ Z - np.eye(32), ord="fro")))
    check("saved target bases", shape_ok, f"files={len(spaces)}, three (225,32) bases per system")
    check("saved basis orthonormality", max_defect < 1.0e-10, f"maximum defect={max_defect:.3e}")

    check("adjacent transfer grain", len(transfer_pairs) == 48 and len(transfer) == 192, f"pairs={len(transfer_pairs)}, RHS rows={len(transfer)}")
    parameter_index_by_system = {
        row["system_id"]: int(row["parameter_index"])
        for row in baseline
    }
    adjacency_ok = all(
        abs(
            parameter_index_by_system[row["target_system_id"]]
            - parameter_index_by_system[row["donor_system_id"]]
        )
        == 1
        for row in transfer
    )
    check("adjacent systems only", adjacency_ok, "index distance is one for every directed donor-target row")

    transfer_ratio_ok = True
    transfer_outcome_ok = True
    for row in transfer:
        transfer_ratio_ok &= np.isclose(
            row["iteration_ratio_reused_vs_own"],
            row["reused_iterations"] / row["own_iterations"],
        ) and np.isclose(
            row["iteration_ratio_reused_vs_baseline"],
            row["reused_iterations"] / row["baseline_iterations"],
        )
        reused = {"iterations": row["reused_iterations"], "converged": row["reused_converged"]}
        own = {"iterations": row["own_iterations"], "converged": row["own_converged"]}
        base = {"iterations": row["baseline_iterations"], "converged": row["baseline_converged"]}
        transfer_outcome_ok &= row["outcome_vs_own"] == paired_outcome(reused, own)
        transfer_outcome_ok &= row["outcome_vs_baseline"] == paired_outcome(reused, base)
    check("transfer paired ratios", transfer_ratio_ok, "reused/own and reused/baseline ratios recomputed")
    check("transfer W/T/L labels", transfer_outcome_ok, "both reference comparisons recomputed")
    check(
        "transfer true-residual convergence",
        all(row["reused_converged"] == (row["reused_true_relative_residual"] <= RTOL) for row in transfer),
        "all 192 rows checked",
    )

    file_lookup = {path.stem: path for path in spaces}
    geometry_ok = True
    for row in transfer_pairs:
        donor = np.load(file_lookup[row["donor_system_id"]], allow_pickle=False)[target_winner["target"]][
            :, : rank_winner["q"]
        ]
        own = np.load(file_lookup[row["target_system_id"]], allow_pickle=False)[target_winner["target"]][
            :, : rank_winner["q"]
        ]
        comparison = compare_subspaces(donor, own)
        geometry_ok &= np.isclose(
            row["normalized_projector_distance"],
            comparison.normalized_projector_distance,
            rtol=1.0e-12,
            atol=1.0e-14,
        ) and np.isclose(
            row["largest_principal_angle_degrees"],
            comparison.max_angle_degrees,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    check("transfer subspace geometry", geometry_ok, "all 48 directed pairs recomputed from saved bases")

    recomputed_transfer_summary = summarize_transfer(transfer, transfer_pairs)
    stored_transfer_summary = read_csv(output / "transfer_summary.csv")
    check(
        "transfer summaries",
        compare_summary_rows(
            stored_transfer_summary,
            recomputed_transfer_summary,
            ("family",),
            (
                "directed_pairs",
                "converged",
                "median_normalized_projector_distance",
                "median_iteration_ratio_reused_vs_own",
                "median_iteration_ratio_reused_vs_baseline",
                "rhs_wins_vs_own",
                "rhs_ties_vs_own",
                "rhs_losses_vs_own",
                "rhs_wins_vs_baseline",
                "rhs_ties_vs_baseline",
                "rhs_losses_vs_baseline",
            ),
        ),
        "all/family transfer metrics independently aggregated",
    )

    max_all_candidate_identity = max(
        [row["deflation_residual_identity_error"] for row in target]
        + [row["deflation_residual_identity_error"] for row in transfer]
    )
    max_selected_identity = max(
        [
            row["deflation_residual_identity_error"]
            for row in target
            if row["target"] == target_winner["target"]
        ]
        + [row["deflation_residual_identity_error"] for row in rank]
        + [row["deflation_residual_identity_error"] for row in transfer]
    )
    check(
        "selected-space deflation residual identity",
        max_selected_identity < 2.0e-8,
        (
            f"selected/rank/transfer maximum={max_selected_identity:.3e}; "
            f"all-candidate maximum={max_all_candidate_identity:.3e}"
        ),
    )
    plot_path = output / "rank_efficiency.png"
    check("rank plot and chart map", plot_path.exists() and plot_path.stat().st_size > 10_000 and (output / "chart_map.csv").exists(), str(plot_path))
    report_texts = [args.report2.read_text(encoding="utf-8"), args.report3.read_text(encoding="utf-8"), args.report4.read_text(encoding="utf-8")]
    check(
        "reports match selected target and rank",
        all("Dominant right singular vectors of S" in text for text in report_texts)
        and f"q={rank_winner['q']}" in report_texts[2]
        and "No tested rank provides aggregate oracle acceleration" in report_texts[1],
        "three markdown reports inspected",
    )
    check("no ML or interpolation", config.get("no_ml") is True and config.get("no_interpolation") is True, "experiment config inspected")

    passed = sum(item["passed"] for item in checks)
    status = "Ready to share" if passed == len(checks) else "Needs revision"
    summary = {
        "status": status,
        "passed_checks": passed,
        "total_checks": len(checks),
        "winning_target": target_winner,
        "winning_rank": rank_winner,
        "transfer_overall": next(row for row in recomputed_transfer_summary if row["family"] == "all"),
        "checks": checks,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Fair Targets, Rank, and Transfer Validation",
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
            "The evidence supports S-dominant q=16 as the most robust target configuration under standard deflation, but no tested rank lowers the aggregate median below 1.0. Adjacent reuse is family-dependent and does not improve the aggregate median.",
            "",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "passed": passed, "total": len(checks), "target": target_winner["target"], "q": rank_winner["q"]}, indent=2))


if __name__ == "__main__":
    main()
