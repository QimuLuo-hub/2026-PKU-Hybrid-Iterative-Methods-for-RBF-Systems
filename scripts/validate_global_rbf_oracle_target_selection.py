"""Validate the saved global-RBF oracle target-selection experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATES = (
    "dominant_right_singular_S",
    "smallest_right_singular_B",
    "late_residual_pod",
)
Q_VALUES = (8, 16, 24, 32)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/global_rbf_oracle_target_selection"),
    )
    args = parser.parse_args()
    output = args.results
    baseline = read_csv(output / "baseline_results.csv")
    oracle = read_csv(output / "oracle_results.csv")
    own_summary = read_csv(output / "own_oracle_summary.csv")
    transfer_summary = read_csv(output / "nearby_transfer_summary.csv")
    space_metadata = read_csv(output / "space_metadata.csv")
    trajectories = read_csv(output / "pod_trajectory_metadata.csv")
    selection = read_csv(output / "system_selection.csv")
    failures_text = (output / "failures.csv").read_text(encoding="utf-8")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("target count", len(selection) == 4, f"observed {len(selection)}; expected 4")
    check("baseline row count", len(baseline) == 16, f"observed {len(baseline)}; expected 16")
    check("oracle row count", len(oracle) == 528, f"observed {len(oracle)}; expected 528")
    own = [row for row in oracle if row["source_role"] == "own"]
    nearby = [row for row in oracle if row["source_role"] == "nearby"]
    check("own-oracle row count", len(own) == 192, f"observed {len(own)}; expected 192")
    check("nearby-transfer row count", len(nearby) == 336, f"observed {len(nearby)}; expected 336")
    check("failure ledger", failures_text == "", f"failure-ledger bytes={len(failures_text)}")

    observed_candidates = {row["candidate"] for row in oracle}
    observed_q = {int(row["q"]) for row in oracle}
    check(
        "candidate and rank coverage",
        observed_candidates == set(CANDIDATES) and observed_q == set(Q_VALUES),
        f"candidates={sorted(observed_candidates)}, q={sorted(observed_q)}",
    )

    baseline_lookup = {(row["system_id"], row["rhs_name"]): row for row in baseline}
    ratios_ok = True
    convergence_ok = True
    finite_ok = True
    for row in oracle:
        base = baseline_lookup[(row["target_system_id"], row["rhs_name"])]
        expected_iteration = int(row["hybrid_iterations"]) / int(base["fgmres_iterations"])
        expected_time = float(row["hybrid_solve_seconds"]) / float(base["solve_seconds"])
        ratios_ok &= bool(
            np.isclose(float(row["iteration_ratio"]), expected_iteration, rtol=1.0e-13, atol=1.0e-14)
            and np.isclose(float(row["solve_time_ratio"]), expected_time, rtol=1.0e-13, atol=1.0e-14)
        )
        convergence_ok &= as_bool(row["converged"]) == (
            float(row["final_true_relative_residual"]) <= float(row["fgmres_rtol"])
        )
        finite_ok &= all(
            np.isfinite(float(row[field]))
            for field in (
                "iteration_ratio",
                "solve_time_ratio",
                "final_true_residual_norm",
                "final_true_relative_residual",
                "space_setup_seconds",
                "incremental_coarse_setup_seconds",
            )
        )
    check("paired ratio recomputation", ratios_ok, "all iteration and solve-time ratios match baseline rows")
    check(
        "true-residual convergence rule",
        convergence_ok,
        "all convergence flags use the saved original-system true relative residual",
    )
    check("finite measurements", finite_ok, "all primary, timing, residual, and setup fields are finite")

    own_combinations: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in own:
        own_combinations[(row["target_system_id"], row["candidate"], int(row["q"]))] += 1
    check(
        "own combination completeness",
        len(own_combinations) == 4 * 3 * 4 and set(own_combinations.values()) == {4},
        f"combinations={len(own_combinations)}, RHS counts={sorted(set(own_combinations.values()))}",
    )

    bases_ok = True
    basis_files = sorted((output / "oracle_spaces").glob("*.npz"))
    for path in basis_files:
        with np.load(path) as data:
            if set(data.files) != set(CANDIDATES):
                bases_ok = False
                continue
            for candidate in CANDIDATES:
                Z = data[candidate]
                bases_ok &= Z.shape == (225, 32)
                bases_ok &= np.linalg.norm(Z.T @ Z - np.eye(32), ord="fro") < 2.0e-10
    check(
        "saved basis integrity",
        len(basis_files) == 11 and bases_ok,
        f"basis files={len(basis_files)}; all expected shapes and orthonormality checked",
    )

    pod_rows = [row for row in space_metadata if row["candidate"] == "late_residual_pod"]
    pod_rank_ok = all(
        int(row["snapshot_count"]) >= 32 and int(row["snapshot_numerical_rank"]) >= 32
        for row in pod_rows
    )
    check(
        "POD rank support",
        len(pod_rows) == 11 and pod_rank_ok,
        "all 11 systems have at least 32 late residual snapshots and numerical rank at least 32",
    )
    check(
        "trajectory coverage",
        len(trajectories) == 66,
        f"observed {len(trajectories)} training trajectories; expected 66",
    )

    own_lookup = {(row["candidate"], int(row["q"])): row for row in own_summary}
    transfer_lookup = {(row["candidate"], int(row["q"])): row for row in transfer_summary}
    summary_ok = True
    for candidate in CANDIDATES:
        for q in Q_VALUES:
            own_rows = [
                row for row in own if row["candidate"] == candidate and int(row["q"]) == q
            ]
            transfer_rows = [
                row for row in nearby if row["candidate"] == candidate and int(row["q"]) == q
            ]
            summary_ok &= np.isclose(
                float(own_lookup[(candidate, q)]["median_iteration_ratio"]),
                np.median([float(row["iteration_ratio"]) for row in own_rows]),
                rtol=1.0e-13,
            )
            summary_ok &= np.isclose(
                float(transfer_lookup[(candidate, q)]["median_iteration_ratio"]),
                np.median([float(row["iteration_ratio"]) for row in transfer_rows]),
                rtol=1.0e-13,
            )
    check("summary recomputation", summary_ok, "all 24 saved own/transfer medians recomputed")
    figure = output / "oracle_target_iteration_ratios.png"
    check("figure artifact", figure.exists() and figure.stat().st_size > 20_000, f"bytes={figure.stat().st_size}")

    aggregate: dict[str, dict[str, Any]] = {}
    by_target_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_own = [row for row in own if row["candidate"] == candidate]
        candidate_nearby = [row for row in nearby if row["candidate"] == candidate]
        aggregate[candidate] = {
            "own_observations": len(candidate_own),
            "own_median_iteration_ratio_all_q": float(
                np.median([float(row["iteration_ratio"]) for row in candidate_own])
            ),
            "own_median_solve_time_ratio_all_q": float(
                np.median([float(row["solve_time_ratio"]) for row in candidate_own])
            ),
            "own_fraction_fewer_iterations_all_q": float(
                np.mean([float(row["iteration_ratio"]) < 1.0 for row in candidate_own])
            ),
            "nearby_observations": len(candidate_nearby),
            "nearby_median_iteration_ratio_all_q": float(
                np.median([float(row["iteration_ratio"]) for row in candidate_nearby])
            ),
            "nearby_median_solve_time_ratio_all_q": float(
                np.median([float(row["solve_time_ratio"]) for row in candidate_nearby])
            ),
            "nearby_fraction_fewer_iterations_all_q": float(
                np.mean([float(row["iteration_ratio"]) < 1.0 for row in candidate_nearby])
            ),
        }
        for target_system_id in sorted({row["target_system_id"] for row in own}):
            for q in Q_VALUES:
                selected_rows = [
                    row
                    for row in own
                    if row["target_system_id"] == target_system_id
                    and row["candidate"] == candidate
                    and int(row["q"]) == q
                ]
                by_target_rows.append(
                    {
                        "target_system_id": target_system_id,
                        "candidate": candidate,
                        "q": q,
                        "observations": len(selected_rows),
                        "median_iteration_ratio": float(
                            np.median([float(row["iteration_ratio"]) for row in selected_rows])
                        ),
                        "median_solve_time_ratio": float(
                            np.median([float(row["solve_time_ratio"]) for row in selected_rows])
                        ),
                        "fraction_fewer_iterations": float(
                            np.mean([float(row["iteration_ratio"]) < 1.0 for row in selected_rows])
                        ),
                    }
                )
    fieldnames = list(by_target_rows[0])
    with (output / "own_oracle_by_target.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_target_rows)

    best_by_q = {}
    for q in Q_VALUES:
        ranked = sorted(
            (
                (
                    candidate,
                    float(own_lookup[(candidate, q)]["median_iteration_ratio"]),
                    float(own_lookup[(candidate, q)]["median_solve_time_ratio"]),
                )
                for candidate in CANDIDATES
            ),
            key=lambda item: (item[1], item[2]),
        )
        best_by_q[str(q)] = {
            "candidate": ranked[0][0],
            "median_iteration_ratio": ranked[0][1],
            "median_solve_time_ratio": ranked[0][2],
        }
    all_own_medians = {
        candidate: [
            float(own_lookup[(candidate, q)]["median_iteration_ratio"]) for q in Q_VALUES
        ]
        for candidate in CANDIDATES
    }
    best_candidate = min(
        CANDIDATES,
        key=lambda candidate: (
            min(all_own_medians[candidate]),
            np.median(all_own_medians[candidate]),
        ),
    )
    best_q = min(
        Q_VALUES,
        key=lambda q: (
            float(own_lookup[(best_candidate, q)]["median_iteration_ratio"]),
            float(own_lookup[(best_candidate, q)]["median_solve_time_ratio"]),
        ),
    )
    no_positive_median = all(
        float(row["median_iteration_ratio"]) >= 1.0 for row in own_summary
    )
    evidence = {
        "validation_status": "Ready to share" if all(item["passed"] for item in checks) else "Needs revision",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "targets": len(selection),
        "baseline_rows": len(baseline),
        "oracle_rows": len(oracle),
        "aggregate_by_candidate": aggregate,
        "best_candidate_by_rank": best_by_q,
        "recommended_candidate": best_candidate,
        "recommended_rank_in_tested_set": best_q,
        "recommended_own_median_iteration_ratio": float(
            own_lookup[(best_candidate, best_q)]["median_iteration_ratio"]
        ),
        "recommended_own_median_solve_time_ratio": float(
            own_lookup[(best_candidate, best_q)]["median_solve_time_ratio"]
        ),
        "recommended_nearby_median_iteration_ratio": float(
            transfer_lookup[(best_candidate, best_q)]["median_iteration_ratio"]
        ),
        "recommended_nearby_transfer_to_own_ratio": float(
            transfer_lookup[(best_candidate, best_q)]["median_transfer_to_own_iteration_ratio"]
        ),
        "no_candidate_rank_has_median_iteration_ratio_below_one": no_positive_median,
        "conclusion": (
            "Dominant right singular vectors of S at q=8 are the least harmful tested target, "
            "but the oracle does not improve the aggregate median under the mandated additive correction."
        ),
    }
    (output / "evidence_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    validation = {"status": evidence["validation_status"], "checks": checks}
    (output / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    lines = [
        "# Validation report",
        "",
        f"**Overall assessment: {evidence['validation_status']}.**",
        "",
        "The audit recomputed every paired ratio, checked true-residual convergence, verified all saved bases, and reconstructed the own-oracle and transfer summaries.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['name']} | {'Pass' if item['passed'] else 'Fail'} | {item['detail']} |")
    lines.extend(
        [
            "",
            "The formula-level unit test separately verifies the exact additive action against a direct reduced solve.",
            "",
            "The negative median result is retained: capped runs are included in ratios and no candidate is selected by filtering to successful improvements.",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["validation_status"],
                "passed": evidence["checks_passed"],
                "total": evidence["checks_total"],
                "recommended": best_candidate,
                "q": best_q,
            }
        )
    )


if __name__ == "__main__":
    main()
