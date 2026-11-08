#!/usr/bin/env python3
"""Independently validate saved Stage-1 residual-history POD diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la

from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked


EPSILONS = (7.8, 8.0, 8.2, 8.4)
Q_VALUES = (4, 8, 12, 16)
SIDE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/global_rbf_residual_pod_stage1"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(epsilon: float) -> str:
    return f"eps_{epsilon:.1f}".replace(".", "p")


def subspace_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    angles = np.sort(np.rad2deg(la.subspace_angles(left, right)))[::-1]
    normalized = np.sqrt(2.0 * np.sum(np.sin(np.deg2rad(angles)) ** 2))
    normalized /= np.sqrt(2.0 * left.shape[1])
    return float(np.max(angles)), float(np.median(angles)), float(normalized)


def finite_max(current: float, *values: float) -> float:
    checked = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(checked)):
        raise AssertionError(f"non-finite validation value: {checked}")
    return max(current, *(float(value) for value in checked))


def main() -> int:
    args = parse_args()
    arrays = np.load(args.results_dir / "stage1_arrays.npz")
    baselines = read_csv(args.results_dir / "baseline_results.csv")
    energies = read_csv(args.results_dir / "energy_and_persistence_summary.csv")
    persistence = read_csv(args.results_dir / "persistence.csv")
    geometry = read_csv(args.results_dir / "adjacent_geometry.csv")
    snapshot_meta = read_csv(args.results_dir / "snapshot_metadata.csv")
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))

    max_solution_residual_error = 0.0
    max_normalization_error = 0.0
    max_basis_orthogonality_error = 0.0
    max_singular_value_error = 0.0
    max_energy_error = 0.0
    max_persistence_error = 0.0
    max_geometry_error = 0.0

    for epsilon in EPSILONS:
        prefix = key(epsilon)
        raw = arrays[f"{prefix}_true_residual_snapshots"]
        normalized = arrays[f"{prefix}_normalized_residual_snapshots"]
        iterations = arrays[f"{prefix}_snapshot_iterations"]
        solution = arrays[f"{prefix}_baseline_solution"]
        singular_values = arrays[f"{prefix}_singular_values"]
        assert np.all(np.isfinite(raw)) and np.all(np.isfinite(normalized))
        assert np.all(np.diff(iterations) > 0)
        assert iterations[0] == 0
        baseline = next(row for row in baselines if float(row["epsilon"]) == epsilon)
        assert iterations[-1] == int(baseline["iterations"])
        assert baseline["converged"] == "True"
        system = assemble_tensor_gaussian_poisson_blocked(SIDE, epsilon=epsilon)
        direct_relative = np.linalg.norm(system.b - system.A @ solution) / np.linalg.norm(system.b)
        max_solution_residual_error = finite_max(
            max_solution_residual_error,
            abs(direct_relative - float(baseline["true_relative_residual"])),
        )
        recomputed_normalized = raw / np.linalg.norm(raw, axis=0)
        max_normalization_error = finite_max(
            max_normalization_error,
            float(np.max(np.abs(normalized - recomputed_normalized))),
            float(np.max(np.abs(np.linalg.norm(normalized, axis=0) - 1.0))),
        )
        left, recomputed_singular_values, _ = la.svd(
            normalized, full_matrices=False, check_finite=True
        )
        max_singular_value_error = finite_max(
            max_singular_value_error,
            float(np.max(np.abs(singular_values - recomputed_singular_values))),
        )
        cumulative = np.cumsum(recomputed_singular_values**2) / np.sum(
            recomputed_singular_values**2
        )
        for q in Q_VALUES:
            basis = arrays[f"{prefix}_pod_q{q}"]
            max_basis_orthogonality_error = finite_max(
                max_basis_orthogonality_error,
                float(np.linalg.norm(basis.T @ basis - np.eye(q), ord="fro")),
                float(np.linalg.norm(basis @ basis.T - left[:, :q] @ left[:, :q].T, ord="fro")),
            )
            energy_row = next(
                row
                for row in energies
                if float(row["epsilon"]) == epsilon and int(row["q"]) == q
            )
            max_energy_error = finite_max(
                max_energy_error, abs(cumulative[q - 1] - float(energy_row["E_q"]))
            )
            recomputed_rho = np.sum((basis.T @ normalized) ** 2, axis=0)
            saved_rho = np.array(
                [
                    float(row["rho"])
                    for row in persistence
                    if float(row["epsilon"]) == epsilon and int(row["q"]) == q
                ]
            )
            max_persistence_error = finite_max(
                max_persistence_error,
                float(np.max(np.abs(recomputed_rho - saved_rho))),
            )

        meta = [row for row in snapshot_meta if float(row["epsilon"]) == epsilon]
        assert len(meta) == raw.shape[1]
        assert all(row["is_stride_10"] == "True" or row["is_restart_boundary"] == "True" for row in meta)

    for row in geometry:
        epsilon_left = float(row["epsilon_left"])
        epsilon_right = float(row["epsilon_right"])
        q = int(row["q"])
        suffix = "pod" if row["space"] == "residual_pod" else "harmonic"
        left = arrays[f"{key(epsilon_left)}_{suffix}_q{q}"]
        right = arrays[f"{key(epsilon_right)}_{suffix}_q{q}"]
        actual = subspace_metrics(left, right)
        expected = (
            float(row["largest_principal_angle_degrees"]),
            float(row["median_principal_angle_degrees"]),
            float(row["normalized_projector_distance"]),
        )
        max_geometry_error = finite_max(
            max_geometry_error, *(abs(a - b) for a, b in zip(actual, expected))
        )

    assert max_solution_residual_error < 1.0e-13
    assert max_normalization_error < 1.0e-12
    assert max_basis_orthogonality_error < 1.0e-10
    assert max_singular_value_error < 1.0e-12
    assert max_energy_error < 1.0e-12
    assert max_persistence_error < 1.0e-12
    assert max_geometry_error < 1.0e-10
    assert summary["stage1_decision"]["passed"] is False
    assert summary["stage1_decision"]["stage2_executed"] is False

    validation = {
        "passed": True,
        "checks": {
            "maximum_true_relative_residual_identity_error": max_solution_residual_error,
            "maximum_snapshot_normalization_error": max_normalization_error,
            "maximum_basis_or_projector_error": max_basis_orthogonality_error,
            "maximum_singular_value_error": max_singular_value_error,
            "maximum_energy_error": max_energy_error,
            "maximum_persistence_error": max_persistence_error,
            "maximum_geometry_error": max_geometry_error,
            "previous_harmonic_q16_geometry_error": summary[
                "previous_q16_geometry_maximum_absolute_error"
            ],
        },
        "stage2_absent_by_gate": True,
    }
    (args.results_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
