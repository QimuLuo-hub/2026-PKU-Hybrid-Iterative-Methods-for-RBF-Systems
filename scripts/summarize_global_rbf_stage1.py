"""Recompute Stage-1 headline statistics from the saved CSV evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: dict[str, str], key: str) -> int:
    return int(row[key])


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/global_rbf_stage1"))
    args = parser.parse_args()
    systems = rows(args.input / "system_characterization.csv")
    classical = rows(args.input / "classical_baselines.csv")
    pod = rows(args.input / "correction_pod_metrics.csv")
    structure = rows(args.input / "correction_structure_summary.csv")
    oracle = rows(args.input / "oracle_results.csv")
    failures = rows(args.input / "failures.csv")

    core = [row for row in systems if row["study"] == "core"]
    convergence: list[dict[str, object]] = []
    for kernel in ("gaussian", "imq", "phs3"):
        for distribution in ("tensor", "jittered"):
            selected = sorted(
                (row for row in core if row["kernel"] == kernel and row["distribution"] == distribution),
                key=lambda row: i(row, "N"),
            )
            slope = np.polyfit(
                np.log([i(row, "N") for row in selected]),
                np.log([f(row, "grid_rel_l2") for row in selected]),
                1,
            )[0]
            convergence.append(
                {
                    "kernel": kernel,
                    "distribution": distribution,
                    "N_first": i(selected[0], "N"),
                    "N_last": i(selected[-1], "N"),
                    "grid_rel_l2_first": f(selected[0], "grid_rel_l2"),
                    "grid_rel_l2_last": f(selected[-1], "grid_rel_l2"),
                    "empirical_loglog_slope": float(slope),
                    "condition_2_first": f(selected[0], "condition_2"),
                    "condition_2_last": f(selected[-1], "condition_2"),
                    "roundoff_exposure_last": f(selected[-1], "roundoff_exposure_kappa_u"),
                }
            )

    shape = [row for row in systems if row["study"] == "shape"]
    shape_optima: list[dict[str, object]] = []
    for kernel in ("gaussian", "imq"):
        for distribution in ("tensor", "jittered"):
            for N in (225, 400):
                selected = [row for row in shape if row["kernel"] == kernel and row["distribution"] == distribution and i(row, "N") == N]
                best_error = min(selected, key=lambda row: f(row, "grid_rel_l2"))
                resolved = [row for row in selected if row["numerical_interpretation"] == "resolved native-basis solve"]
                best_resolved = min(resolved, key=lambda row: f(row, "grid_rel_l2")) if resolved else None
                shape_optima.append(
                    {
                        "kernel": kernel,
                        "distribution": distribution,
                        "N": N,
                        "raw_min_error_epsilon": f(best_error, "epsilon"),
                        "raw_min_grid_rel_l2": f(best_error, "grid_rel_l2"),
                        "raw_min_condition_2": f(best_error, "condition_2"),
                        "raw_min_interpretation": best_error["numerical_interpretation"],
                        "best_resolved_epsilon": f(best_resolved, "epsilon") if best_resolved else None,
                        "best_resolved_grid_rel_l2": f(best_resolved, "grid_rel_l2") if best_resolved else None,
                        "best_resolved_condition_2": f(best_resolved, "condition_2") if best_resolved else None,
                    }
                )

    classical_summary: list[dict[str, object]] = []
    for method in sorted({row["method"] for row in classical}):
        selected = [row for row in classical if row["method"] == method]
        converged = [row for row in selected if row["converged"] == "True"]
        classical_summary.append(
            {
                "method": method,
                "systems": len(selected),
                "converged": len(converged),
                "convergence_fraction": len(converged) / len(selected),
                "median_iterations_converged": median([f(row, "gmres_iterations") for row in converged]),
                "median_total_seconds_first_rhs": median([f(row, "total_seconds_first_rhs") for row in selected]),
                "median_setup_seconds": median([f(row, "setup_seconds") for row in selected]),
                "median_memory_bytes": median([f(row, "preconditioner_memory_bytes") for row in selected]),
            }
        )

    classical_lookup = {(row["config_id"], row["method"]): row for row in classical}
    direct_vs_fp32: list[dict[str, object]] = []
    for config_id in sorted({row["config_id"] for row in classical}):
        fp32 = classical_lookup.get((config_id, "pivoted_lu_fp32"))
        direct = classical_lookup.get((config_id, "pivoted_lu_fp64_direct"))
        if not fp32 or not direct:
            continue
        fp_setup, fp_solve = f(fp32, "setup_seconds"), f(fp32, "solve_seconds_median")
        d_setup, d_solve = f(direct, "setup_seconds"), f(direct, "solve_seconds_median")
        denominator = fp_solve - d_solve
        break_even = (d_setup - fp_setup) / denominator if denominator > 0 else float("inf")
        direct_vs_fp32.append(
            {
                "config_id": config_id,
                "N": i(fp32, "N"),
                "kernel": fp32["kernel"],
                "distribution": fp32["distribution"],
                "fp32_iterations": i(fp32, "gmres_iterations"),
                "fp32_converged": fp32["converged"] == "True",
                "first_rhs_time_ratio_fp32_over_direct": f(fp32, "total_seconds_first_rhs") / f(direct, "total_seconds_first_rhs"),
                "per_rhs_time_ratio_fp32_over_direct": fp_solve / d_solve,
                "break_even_rhs_count_if_positive": float(break_even),
                "memory_ratio_fp32_factor_over_fp64_factor": f(fp32, "preconditioner_memory_bytes") / f(direct, "preconditioner_memory_bytes"),
            }
        )

    pod_by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pod:
        pod_by_config[row["config_id"]].append(row)
    structure_summary: list[dict[str, object]] = []
    structure_lookup = {row["config_id"]: row for row in structure}
    for config_id, selected in pod_by_config.items():
        selected = sorted(selected, key=lambda row: i(row, "q"))
        q99 = next((row for row in selected if f(row, "captured_energy") >= 0.99), selected[-1])
        q10 = next((row for row in selected if f(row, "heldout_projection_error_median") <= 0.10), selected[-1])
        q01 = next((row for row in selected if f(row, "heldout_projection_error_median") <= 0.01), None)
        source = structure_lookup[config_id]
        structure_summary.append(
            {
                "config_id": config_id,
                "N": i(source, "N"),
                "kernel": source["kernel"],
                "distribution": source["distribution"],
                "base_iterations": i(source, "base_iterations"),
                "median_correction_over_exact_action": f(source, "median_correction_over_exact_action"),
                "q_for_99pct_snapshot_energy": i(q99, "q"),
                "q_over_n_for_99pct_energy": f(q99, "q_over_n"),
                "heldout_error_at_q99": f(q99, "heldout_projection_error_median"),
                "q_for_median_heldout_error_le_0p10": i(q10, "q"),
                "q_over_n_for_error_le_0p10": f(q10, "q_over_n"),
                "q_for_median_heldout_error_le_0p01": i(q01, "q") if q01 else None,
                "q_over_n_for_error_le_0p01": f(q01, "q_over_n") if q01 else None,
            }
        )

    base_oracle = {(row["config_id"], row["rhs"]): row for row in oracle if row["space"] == "classical_only"}
    oracle_grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for row in oracle:
        if row["space"] == "classical_only":
            continue
        base = base_oracle[(row["config_id"], row["rhs"])]
        oracle_grouped[(row["space"], i(row, "q_requested"))].append(
            {
                "iteration_ratio": f(row, "gmres_iterations") / f(base, "gmres_iterations"),
                "solve_time_ratio": f(row, "solve_seconds") / f(base, "solve_seconds"),
                "matvec_ratio": f(row, "dense_matvec_equivalent") / f(base, "dense_matvec_equivalent"),
                "first_rhs_total_ratio": f(row, "total_seconds_first_rhs") / f(base, "solve_seconds"),
                "improved": float(i(row, "gmres_iterations") < i(base, "gmres_iterations")),
            }
        )
    oracle_summary: list[dict[str, object]] = []
    for (space, q), values in sorted(oracle_grouped.items()):
        oracle_summary.append(
            {
                "space": space,
                "q": q,
                "comparisons": len(values),
                "median_iteration_ratio_vs_classical": median([v["iteration_ratio"] for v in values]),
                "median_solve_time_ratio_vs_classical": median([v["solve_time_ratio"] for v in values]),
                "median_dense_matvec_ratio_vs_classical": median([v["matvec_ratio"] for v in values]),
                "median_first_rhs_total_ratio_vs_classical_solve": median([v["first_rhs_total_ratio"] for v in values]),
                "fraction_with_fewer_iterations": float(np.mean([v["improved"] for v in values])),
            }
        )

    fp32_points = []
    core_lookup = {row["config_id"]: row for row in core}
    for row in classical:
        if row["method"] == "pivoted_lu_fp32" and row["config_id"] in core_lookup:
            fp32_points.append((np.log10(f(core_lookup[row["config_id"]], "condition_2")), f(row, "gmres_iterations")))
    correlation = float(np.corrcoef(np.array(fp32_points).T)[0, 1])

    payload = {
        "counts": {
            "characterized_systems": len(systems),
            "core_systems": len(core),
            "classical_rows": len(classical),
            "structure_systems": len(structure),
            "oracle_rows": len(oracle),
            "failures": len(failures),
        },
        "core_convergence": convergence,
        "shape_optima": shape_optima,
        "classical_method_summary": classical_summary,
        "direct_vs_fp32": direct_vs_fp32,
        "structure_summary": structure_summary,
        "oracle_summary": oracle_summary,
        "correlation_log10_condition_vs_fp32_iterations": correlation,
    }
    (args.input / "evidence_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
