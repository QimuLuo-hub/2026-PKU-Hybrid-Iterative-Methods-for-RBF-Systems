"""Recompute integrity and solver-utility checks for the offline dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la

from build_global_rbf_subspace_dataset import (
    ENERGY_TARGET,
    KERNEL,
    N,
    Q,
    RHS_SPECS,
    SIDE,
    parameter_pairs,
    parameter_values,
    selected_oracle_basis,
)
from global_rbf.collocation import assemble_variable_diffusion_system
from global_rbf.nodes import square_nodes
from global_rbf.offline_dataset import (
    coefficient_representation,
    fit_shared_basis,
    projector_reconstruction_error,
    reconstructed_basis,
    validate_parameter_splits,
)
from global_rbf.preconditioners import low_precision_lu_preconditioner


OUTPUT_DEFAULT = Path("results/global_rbf_subspace_dataset")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(output / "dataset.npz", allow_pickle=False)
    sample_rows = read_csv(output / "samples.csv")
    solver_rows = read_csv(output / "solver_diagnostics.csv")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    expected_pairs = parameter_pairs(parameter_values(bool(manifest["quick"])))
    stored_pairs = {
        split: data["parameter"][data["split"] == split]
        for split in ("train", "validation", "test")
    }
    try:
        validate_parameter_splits(stored_pairs, require_marginal_disjoint=True)
        splits_valid = True
        split_detail = "pair-level and epsilon/beta marginal values are disjoint"
    except ValueError as exc:
        splits_valid = False
        split_detail = str(exc)
    check("parameter-disjoint splits", splits_valid, split_detail)
    expected_order = np.concatenate(
        [expected_pairs[split] for split in ("train", "validation", "test")]
    )
    check(
        "complete parameter grid",
        np.array_equal(data["parameter"], expected_order),
        f"stored={len(data['parameter'])}, expected={len(expected_order)}",
    )

    sample_count = len(data["sample_id"])
    p = int(data["p"])
    expected_shapes = {
        "parameter": (sample_count, 2),
        "Z": (sample_count, N, Q),
        "C": (sample_count, p, Q),
        "W": (N, p),
        "classical_iterations": (sample_count, len(RHS_SPECS)),
        "oracle_iterations": (sample_count, len(RHS_SPECS)),
    }
    shape_ok = all(data[name].shape == shape for name, shape in expected_shapes.items())
    check("array shapes", shape_ok, json.dumps({name: list(data[name].shape) for name in expected_shapes}))
    check(
        "sample identifiers unique",
        len(set(data["sample_id"].tolist())) == sample_count,
        f"unique={len(set(data['sample_id'].tolist()))}, rows={sample_count}",
    )
    check(
        "metadata and solver grain",
        len(sample_rows) == sample_count and len(solver_rows) == sample_count * len(RHS_SPECS),
        f"samples.csv={len(sample_rows)}, solver rows={len(solver_rows)}",
    )

    W = data["W"]
    w_orthogonality = float(np.linalg.norm(W.T @ W - np.eye(p), ord="fro"))
    z_orthogonality = np.array(
        [np.linalg.norm(Z.T @ Z - np.eye(Q), ord="fro") for Z in data["Z"]]
    )
    check("W orthonormal", w_orthogonality < 1.0e-10, f"Frobenius defect={w_orthogonality:.3e}")
    check(
        "all Z orthonormal",
        float(np.max(z_orthogonality)) < 1.0e-10,
        f"maximum Frobenius defect={np.max(z_orthogonality):.3e}",
    )

    cumulative = data["shared_basis_cumulative_energy"]
    target = float(data["energy_target"])
    minimal_prefix = cumulative[p - 1] >= target and (p == 1 or cumulative[p - 2] < target)
    check(
        "smallest 99.9% energy prefix",
        bool(minimal_prefix and abs(target - ENERGY_TARGET) < 1.0e-15),
        f"p={p}, E_p={cumulative[p - 1]:.10f}, E_(p-1)={cumulative[p - 2] if p > 1 else 0.0:.10f}",
    )
    train_mask = data["split"] == "train"
    refit = fit_shared_basis([Z for Z in data["Z"][train_mask]], energy_target=target)
    shared_projector_error = float(
        np.linalg.norm(W @ W.T - refit.basis @ refit.basis.T, ord="fro")
    )
    check(
        "W fitted from training only",
        refit.retained_dimension == p and shared_projector_error < 1.0e-8,
        f"refit p={refit.retained_dimension}, projector error={shared_projector_error:.3e}",
    )

    coefficient_errors = []
    reconstruction_errors = []
    for index, Z in enumerate(data["Z"]):
        expected_C = W.T @ Z
        coefficient_errors.append(np.linalg.norm(data["C"][index] - expected_C, ord="fro"))
        Z_hat = reconstructed_basis(W, data["C"][index])
        error, _ = projector_reconstruction_error(Z, Z_hat)
        reconstruction_errors.append(error)
    coefficient_errors = np.asarray(coefficient_errors)
    reconstruction_errors = np.asarray(reconstruction_errors)
    stored_errors = data["projector_reconstruction_error_fro"]
    check(
        "C equals W^T Z",
        float(np.max(coefficient_errors)) < 1.0e-10,
        f"maximum coefficient discrepancy={np.max(coefficient_errors):.3e}",
    )
    check(
        "projector errors independently reproduced",
        np.allclose(reconstruction_errors, stored_errors, rtol=1.0e-10, atol=1.0e-12),
        f"maximum stored/recomputed difference={np.max(np.abs(reconstruction_errors-stored_errors)):.3e}",
    )

    arrays_finite = all(
        np.all(np.isfinite(data[name]))
        for name in (
            "parameter",
            "Z",
            "C",
            "W",
            "classical_true_relative_residual",
            "oracle_true_relative_residual",
            "projector_reconstruction_error_fro",
        )
    )
    check("finite numeric arrays", arrays_finite, "all required numeric arrays inspected")

    classical_expected = data["classical_true_relative_residual"] <= manifest["solver"]["rtol"]
    oracle_expected = data["oracle_true_relative_residual"] <= manifest["solver"]["rtol"]
    residual_flags_match = np.array_equal(classical_expected, data["classical_converged"]) and np.array_equal(
        oracle_expected, data["oracle_converged"]
    )
    check("true-residual convergence flags", residual_flags_match, "recomputed from stored original-system residuals")
    counts_valid = bool(
        np.all(data["classical_iterations"] > 0)
        and np.all(data["oracle_iterations"] > 0)
        and np.all(data["classical_iterations"] <= manifest["solver"]["max_iterations"])
        and np.all(data["oracle_iterations"] <= manifest["solver"]["max_iterations"])
    )
    check("FGMRES iteration ranges", counts_valid, f"maximum allowed={manifest['solver']['max_iterations']}")
    derived_acceleration = np.all(data["classical_converged"], axis=1) & np.all(
        data["oracle_converged"], axis=1
    ) & (
        np.median(data["oracle_iterations"], axis=1)
        < np.median(data["classical_iterations"], axis=1)
    )
    check(
        "oracle acceleration flags",
        np.array_equal(derived_acceleration, data["oracle_acceleration_verified"]),
        "strict median-iteration improvement with all RHS solves converged",
    )

    sample_file_ok = True
    for index, identifier in enumerate(data["sample_id"]):
        path = output / "samples" / f"{identifier}.npz"
        if not path.exists():
            sample_file_ok = False
            break
        item = np.load(path, allow_pickle=False)
        if not (
            np.array_equal(item["parameter"], data["parameter"][index])
            and np.allclose(item["Z"], data["Z"][index])
            and np.allclose(item["C"], data["C"][index])
            and np.array_equal(item["classical_iterations"], data["classical_iterations"][index])
            and np.array_equal(item["oracle_iterations"], data["oracle_iterations"][index])
        ):
            sample_file_ok = False
            break
    check("per-sample NPZ files", sample_file_ok, f"checked {sample_count} files")

    # Rebuild one system per split to trace stored Z back to the selected oracle.
    nodes, boundary = square_nodes(SIDE, "tensor")
    spot_errors: list[float] = []
    for split in ("train", "validation", "test"):
        index = int(np.flatnonzero(data["split"] == split)[0])
        epsilon, beta = data["parameter"][index]
        system = assemble_variable_diffusion_system(
            nodes,
            boundary,
            beta=float(beta),
            kernel=KERNEL,
            epsilon=float(epsilon),
        )
        base = low_precision_lu_preconditioner(system.A)
        rebuilt, _ = selected_oracle_basis(system.A, base)
        error, _ = projector_reconstruction_error(data["Z"][index], rebuilt)
        spot_errors.append(error)
    check(
        "selected-oracle basis spot checks",
        max(spot_errors) < 1.0e-8,
        f"train/validation/test projector errors={spot_errors}",
    )

    reference_grid_errors = np.array([float(row["reference_grid_rel_l2"]) for row in sample_rows])
    reference_residuals = np.array([float(row["reference_relative_residual"]) for row in sample_rows])
    check(
        "reference discretization diagnostics finite",
        bool(np.all(np.isfinite(reference_grid_errors)) and np.all(np.isfinite(reference_residuals))),
        f"grid relative error range={reference_grid_errors.min():.3e}-{reference_grid_errors.max():.3e}; residual max={reference_residuals.max():.3e}",
    )

    passed = sum(bool(item["passed"]) for item in checks)
    integrity_ready = passed == len(checks)
    paired_ratios = data["oracle_iterations"] / data["classical_iterations"]
    aggregate_ratio = float(np.median(paired_ratios))
    verified_count = int(np.count_nonzero(data["oracle_acceleration_verified"]))
    status = "Ready to share" if integrity_ready and aggregate_ratio < 1.0 else (
        "Share with caveats" if integrity_ready else "Needs revision"
    )

    split_summary: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        mask = data["split"] == split
        split_summary.append(
            {
                "split": split,
                "systems": int(np.count_nonzero(mask)),
                "median_projector_error": float(np.median(stored_errors[mask])),
                "max_projector_error": float(np.max(stored_errors[mask])),
                "median_normalized_projector_error": float(
                    np.median(data["normalized_projector_reconstruction_error"][mask])
                ),
                "median_paired_oracle_iteration_ratio": float(np.median(paired_ratios[mask])),
                "verified_accelerated_systems": int(
                    np.count_nonzero(data["oracle_acceleration_verified"][mask])
                ),
                "both_all_rhs_converged_systems": int(
                    np.count_nonzero(
                        np.all(data["classical_converged"][mask], axis=1)
                        & np.all(data["oracle_converged"][mask], axis=1)
                    )
                ),
            }
        )

    summary = {
        "status": status,
        "passed_checks": passed,
        "total_checks": len(checks),
        "sample_count": sample_count,
        "q": Q,
        "p": p,
        "p_over_N": p / N,
        "retained_training_energy": float(cumulative[p - 1]),
        "aggregate_median_paired_oracle_iteration_ratio": aggregate_ratio,
        "verified_accelerated_systems": verified_count,
        "split_summary": split_summary,
        "checks": checks,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Offline Dataset Validation Report",
        "",
        f"## Overall assessment: {status}",
        "",
        f"All {passed}/{len(checks)} integrity checks passed." if integrity_ready else f"Only {passed}/{len(checks)} integrity checks passed.",
        "",
        "The dataset is structurally reproducible and leakage-free at the parameter-value level. "
        + (
            "The oracle also improves the aggregate median iteration count."
            if aggregate_ratio < 1.0
            else "The oracle does not improve the aggregate median iteration count, so target utility remains an explicit caveat."
        ),
        "",
        "## Split diagnostics",
        "",
        "| Split | Systems | Both solvers converged on all RHSs | Median projector error | Maximum projector error | Median paired iteration ratio | Verified accelerated systems |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_summary:
        lines.append(
            f"| {row['split']} | {row['systems']} | {row['both_all_rhs_converged_systems']}/{row['systems']} | {row['median_projector_error']:.6g} | "
            f"{row['max_projector_error']:.6g} | {row['median_paired_oracle_iteration_ratio']:.4f} | "
            f"{row['verified_accelerated_systems']}/{row['systems']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            "| Check | Result | Detail |",
            "|---|---|---|",
        ]
    )
    for item in checks:
        lines.append(
            f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Required caveat before ML",
            "",
            f"The shared basis uses p={p} of N={N} ambient directions. Reconstruction quality and oracle acceleration must both be considered: a small projector error does not make the target solver-useful, and an oracle that does not beat M cannot justify a learned approximation.",
            "",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "passed_checks", "total_checks", "p", "p_over_N", "aggregate_median_paired_oracle_iteration_ratio", "verified_accelerated_systems")}, indent=2))


if __name__ == "__main__":
    main()
