#!/usr/bin/env python3
"""Validate Stage 10 result completeness and cross-artifact consistency."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(output: Path) -> dict:
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    ratios = sorted(set(settings["base_ratios"] + settings["transition_ratios"]))
    q_values = [0] + settings["q_values"]
    rhs_seeds = settings["evaluation_rhs_seeds"]
    metadata = read_rows(output / "system_metadata.csv")
    assert len(metadata) == len(ratios)
    for field in (
        "n_nodes",
        "node_seed",
        "stencil_size",
        "jitter_fraction",
        "anisotropy_angle_degrees",
        "nnz",
    ):
        assert len({row[field] for row in metadata}) == 1, field
    assert [float(row["ratio"]) for row in metadata] == ratios

    q_rows = read_rows(output / "q_sweep_results.csv")
    expected_solve_keys = {
        (ratio, q, seed)
        for ratio in ratios
        for q in q_values
        for seed in rhs_seeds
    }
    observed_solve_keys = {
        (float(row["ratio"]), int(float(row["q"])), int(float(row["rhs_seed"])))
        for row in q_rows
    }
    assert observed_solve_keys == expected_solve_keys
    for row in q_rows:
        assert 0 <= int(float(row["iterations"])) <= settings["solver"]["max_iterations"]
        residual = float(row["final_relative_residual"])
        assert math.isfinite(residual) and residual >= 0.0
        if row["converged"] == "True":
            assert residual <= 1.05 * settings["solver"]["relative_tolerance"]

    transfer = read_rows(output / "transfer_results.csv")
    methods = {"ILU", "own_oracle", "nearby_reuse", "far_reuse", "random_control"}
    assert {(float(row["target_ratio"]), row["method"]) for row in transfer} == {
        (ratio, method) for ratio in ratios for method in methods
    }
    q_track = int(
        json.loads((output / "tracking_dimension.json").read_text(encoding="utf-8"))[
            "q_track"
        ]
    )
    q_summary = read_rows(output / "q_sweep_summary.csv")
    for ratio in ratios:
        for method, q in (("ILU", 0), ("own_oracle", q_track)):
            transfer_row = next(
                row
                for row in transfer
                if float(row["target_ratio"]) == ratio and row["method"] == method
            )
            sweep_row = next(
                row
                for row in q_summary
                if float(row["ratio"]) == ratio and int(float(row["q"])) == q
            )
            assert float(transfer_row["iterations_median"]) == float(
                sweep_row["iterations_median"]
            )
            assert float(transfer_row["convergence_rate"]) == float(
                sweep_row["convergence_rate"]
            )

    pairs = read_rows(output / "subspace_pairs.csv")
    expected_pairs = len(ratios) * (len(ratios) - 1) // 2
    for q in settings["q_values"]:
        local = [row for row in pairs if int(float(row["q"])) == q]
        assert len(local) == expected_pairs
        assert all(
            0.0 <= float(row["normalized_projector_distance"]) <= 1.0 + 1.0e-12
            for row in local
        )
        data = np.load(output / "distance_matrices" / f"distances_q{q}.npz")
        distance = data["normalized_projector_distance"]
        angles = data["principal_angle_mean_degrees"]
        assert np.allclose(distance, distance.T)
        assert np.allclose(np.diag(distance), 0.0, atol=1.0e-7)
        assert np.allclose(angles, angles.T)
        assert np.allclose(np.diag(angles), 0.0, atol=1.0e-5)

    tracked_pairs = [row for row in pairs if int(float(row["q"])) == q_track]
    correlation = float(
        spearmanr(
            [float(row["absolute_parameter_distance"]) for row in tracked_pairs],
            [float(row["normalized_projector_distance"]) for row in tracked_pairs],
        ).statistic
    )
    stored_correlation = next(
        row
        for row in read_rows(output / "subspace_correlations.csv")
        if int(float(row["q"])) == q_track
    )
    assert np.isclose(correlation, float(stored_correlation["spearman_correlation"]))

    assignments = read_rows(output / "mode_track_assignments.csv")
    assert len(assignments) == q_track * len(ratios)
    for ratio in ratios:
        identifiers = [
            int(float(row["track_id"]))
            for row in assignments
            if float(row["ratio"]) == ratio
        ]
        assert sorted(identifiers) == list(range(q_track))

    maximum_error = max(float(row["relative_discretization_error"]) for row in metadata)
    assert maximum_error < 0.02
    return {
        "systems": len(ratios),
        "q_sweep_solve_rows": len(q_rows),
        "transfer_summaries": len(transfer),
        "q_track": q_track,
        "q_track_spearman": correlation,
        "maximum_relative_discretization_error": maximum_error,
        "nonconverged_q_sweeps": sum(row["converged"] == "False" for row in q_rows),
        "unit_consistency": "passed",
        "completeness": "passed",
        "cross_artifact_reconciliation": "passed",
    }


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    result_directory = repository / "results" / "stage10_ritz_transfer_and_mode_tracking"
    summary = validate(result_directory)
    path = result_directory / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
