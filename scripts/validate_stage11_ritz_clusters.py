#!/usr/bin/env python3
"""Validate Stage 11 completeness, invariants, and reported decision metrics."""

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_bool(value: str) -> bool:
    return value == "True"


def validate(output: Path) -> dict:
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
    ratios = [float(value) for value in settings["ratios"]]
    maximum_modes = max(settings["stage10"]["q_values"])

    atom_rows = read_rows(output / "ritz_atoms.csv")
    cluster_rows = read_rows(output / "ritz_clusters.csv")
    assert {float(row["ratio"]) for row in atom_rows} == set(ratios)
    assert {float(row["ratio"]) for row in cluster_rows} == set(ratios)
    complex_cut_count = 0
    for ratio in ratios:
        atoms = [row for row in atom_rows if float(row["ratio"]) == ratio]
        clusters = [row for row in cluster_rows if float(row["ratio"]) == ratio]
        assert [int(row["start"]) for row in atoms] == [
            0, *[int(row["stop"]) for row in atoms[:-1]]
        ]
        assert int(atoms[-1]["stop"]) == maximum_modes
        assert all(int(row["dimension"]) in (1, 2) for row in atoms)
        assert [int(row["start"]) for row in clusters] == [
            0, *[int(row["stop"]) for row in clusters[:-1]]
        ]
        assert int(clusters[-1]["stop"]) == maximum_modes
        crossing = [row for row in clusters if as_bool(row["contains_q8_cut"])]
        assert len(crossing) == 1
        complex_cut_count += sum(as_bool(row["crosses_q8_cut"]) for row in atoms)

        archive = np.load(output / "clusters" / f"ritz_clusters_R{ratio:g}.npz")
        basis = archive["basis"]
        assert basis.shape == (settings["stage10"]["n_nodes"], maximum_modes)
        assert np.allclose(basis.T @ basis, np.eye(maximum_modes), atol=2.0e-10)
        assert np.array_equal(
            archive["atom_spans"],
            np.asarray([[int(row["start"]), int(row["stop"])] for row in atoms]),
        )
        assert np.array_equal(
            archive["cluster_spans"],
            np.asarray([[int(row["start"]), int(row["stop"])] for row in clusters]),
        )
    assert complex_cut_count == decision["q8_complex_plane_cut_count"]

    assignments = read_rows(output / "cluster_track_assignments.csv")
    assert len(assignments) == len(cluster_rows)
    assert {
        (float(row["ratio"]), int(row["cluster_id"])) for row in assignments
    } == {
        (float(row["ratio"]), int(row["cluster_id"])) for row in cluster_rows
    }

    matches = read_rows(output / "cluster_neighbor_matches.csv")
    adjacent = set(zip(ratios[:-1], ratios[1:]))
    assert {
        (float(row["source_ratio"]), float(row["target_ratio"])) for row in matches
    } == adjacent
    assert all(
        0.0 <= float(row["projector_distance_normalized"]) <= 1.0 + 1.0e-12
        for row in matches
    )
    for row in matches:
        expected_stable = (
            as_bool(row["dimension_equal"])
            and float(row["projector_distance_normalized"])
            <= settings["stable_projector_distance"]
            and float(row["principal_angle_max_degrees"])
            <= settings["stable_max_principal_angle_degrees"]
            and float(row["spectral_factor"]) <= settings["stable_spectral_factor"]
        )
        assert as_bool(row["stable_pair"]) == expected_stable

    importance = read_rows(output / "cluster_solver_importance.csv")
    assert len(importance) == len(cluster_rows)
    assert {
        (float(row["ratio"]), int(row["cluster_id"])) for row in importance
    } == {
        (float(row["ratio"]), int(row["cluster_id"])) for row in cluster_rows
    }
    assert all(
        0.0 <= float(row[field]) <= 1.0
        for row in importance
        for field in ("baseline_convergence_rate", "add_convergence_rate", "remove_convergence_rate")
    )

    transfer = read_rows(output / "cluster_transfer_results.csv")
    methods = {
        "ILU",
        "own_full_q8",
        "nearby_full_q8_reuse",
        "random_q8",
        "own_cluster",
        "nearby_cluster_reuse",
        "random_cluster",
    }
    assert {(float(row["target_ratio"]), row["method"]) for row in transfer} == {
        (ratio, method) for ratio in ratios for method in methods
    }
    for row in transfer:
        iterations = float(row["iterations_median"])
        assert 0.0 <= iterations <= settings["stage10"]["solver"]["max_iterations"]
        assert 0.0 <= float(row["convergence_rate"]) <= 1.0
        assert math.isfinite(float(row["final_relative_residual_max"]))

    robustness = read_rows(output / "cluster_robustness_pairs.csv")
    assert len(robustness) == 2 * len(settings["robustness_ratios"]) * 3
    assert {row["variation"] for row in robustness} == {
        "construction_rhs",
        "node_perturbation",
    }
    assert all(
        0.0 <= float(row["projector_distance_normalized"]) <= 1.0 + 1.0e-12
        for row in robustness
    )

    expanded = read_rows(output / "expanded_crossing_blocks.csv")
    expanded_solves = read_rows(output / "expanded_crossing_solver_results.csv")
    assert len(expanded) == len(ratios) - 1
    assert len(expanded_solves) == 2 * len(expanded)
    assert all(
        int(row["expanded_dimension"]) <= settings["maximum_expanded_dimension"]
        for row in expanded
        if as_bool(row["expanded_available"])
    )

    report = output / "stage11_ritz_cluster_stability_report.md"
    report_text = report.read_text(encoding="utf-8")
    assert not any(ord(character) < 32 and character not in "\n\t" for character in report_text)
    assert "No neural network was trained" in report_text
    assert "should be paused" in report_text
    image_links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report_text)
    assert len(image_links) >= 8
    assert all((output / link).exists() for link in image_links)

    assert decision["new_learned_subspace_target_exists"] is False
    assert decision["pause_current_ritz_learned_subspace_route"] is True
    assert decision["stable_critical_track_count"] == 0
    return {
        "systems": len(ratios),
        "ritz_atoms": len(atom_rows),
        "gap_clusters": len(cluster_rows),
        "neighbor_cluster_matches": len(matches),
        "held_out_cluster_importance_summaries": len(importance),
        "transfer_summaries": len(transfer),
        "robustness_pair_summaries": len(robustness),
        "expanded_neighbor_blocks": len(expanded),
        "complex_plane_cuts_detected": complex_cut_count,
        "stable_critical_tracks": decision["stable_critical_track_count"],
        "completeness": "passed",
        "linear_algebra_invariants": "passed",
        "cross_artifact_reconciliation": "passed",
        "report_integrity": "passed",
    }


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    result_directory = repository / "results" / "stage11_ritz_cluster_stability"
    summary = validate(result_directory)
    path = result_directory / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
