"""Build an offline difficult-subspace dataset for global RBF collocation.

The experiment fixes N=225 tensor nodes, a Gaussian kernel, and the variable-
diffusion PDE ``-div(exp(beta*x) grad u)=f``.  Only epsilon and beta vary.
The selected oracle target is the q=8 dominant right-singular subspace of
``S = I - M^{-1} A`` for the established FP32 pivoted-LU preconditioner.

No neural network is constructed or trained here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.collocation import (
    GlobalRBFSystem,
    assemble_variable_diffusion_system,
    variable_diffusion_forcing,
)
from global_rbf.nodes import square_nodes
from global_rbf.offline_dataset import (
    align_basis_to_reference,
    coefficient_representation,
    fit_shared_basis,
    orthonormalize_basis,
    projector_reconstruction_error,
    reconstructed_basis,
    validate_parameter_splits,
)
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import additive_galerkin_coarse_preconditioner


OUTPUT_DEFAULT = Path("results/global_rbf_subspace_dataset")
SIDE = 15
N = SIDE**2
KERNEL = "gaussian"
Q = 8
ENERGY_TARGET = 0.999
RTOL = 1.0e-10
RESTART = 100
MAX_ITERATIONS = 450
RHS_SPECS = (
    ("heldout_mode_2_3", 2, 3),
    ("heldout_mode_3_2", 3, 2),
    ("heldout_mode_1_4", 1, 4),
    ("heldout_mode_4_1", 4, 1),
)


@dataclass
class Sample:
    sample_id: str
    split: str
    epsilon: float
    beta: float
    Z: np.ndarray
    classical_iterations: np.ndarray
    oracle_iterations: np.ndarray
    classical_converged: np.ndarray
    oracle_converged: np.ndarray
    classical_true_relative_residual: np.ndarray
    oracle_true_relative_residual: np.ndarray
    metadata: dict[str, Any]


def parameter_values(quick: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if quick:
        return {
            "train": (np.array([5.0, 5.5, 6.0]), np.array([0.0, 0.5, 1.0])),
            "validation": (np.array([5.25]), np.array([0.25])),
            "test": (np.array([5.75]), np.array([0.75])),
        }
    return {
        "train": (np.linspace(5.0, 6.0, 9), np.linspace(0.0, 1.0, 9)),
        "validation": (
            np.array([5.0625, 5.3125, 5.5625, 5.8125]),
            np.array([0.0625, 0.3125, 0.5625, 0.8125]),
        ),
        "test": (
            np.array([5.1875, 5.4375, 5.6875, 5.9375]),
            np.array([0.1875, 0.4375, 0.6875, 0.9375]),
        ),
    }


def parameter_pairs(values: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    pairs: dict[str, np.ndarray] = {}
    for split, (epsilons, betas) in values.items():
        pairs[split] = np.array(
            [(float(epsilon), float(beta)) for epsilon in epsilons for beta in betas],
            dtype=float,
        )
    validate_parameter_splits(pairs, require_marginal_disjoint=True)
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_id(epsilon: float, beta: float) -> str:
    return f"epsilon{epsilon:.4f}_beta{beta:.4f}"


def mode_rhs(system: GlobalRBFSystem, beta: float, kx: int, ky: int) -> np.ndarray:
    points = system.nodes
    u = np.sin(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1])
    f = variable_diffusion_forcing(points, beta, kx=kx, ky=ky)
    return np.where(system.boundary, u, f)


def selected_oracle_basis(
    A: np.ndarray,
    preconditioner: Preconditioner,
) -> tuple[np.ndarray, dict[str, float]]:
    start = perf_counter()
    B = np.asarray(preconditioner.apply(A), dtype=float)
    formation_seconds = perf_counter() - start
    start = perf_counter()
    _, singular_values, right_t = la.svd(
        np.eye(A.shape[0]) - B,
        full_matrices=False,
        check_finite=False,
    )
    decomposition_seconds = perf_counter() - start
    Z = orthonormalize_basis(right_t[:Q, :].T)
    relative_boundary_gap = (
        float((singular_values[Q - 1] - singular_values[Q]) / singular_values[Q - 1])
        if Q < len(singular_values) and singular_values[Q - 1] != 0.0
        else float("nan")
    )
    return Z, {
        "preconditioned_operator_formation_seconds": formation_seconds,
        "oracle_svd_seconds": decomposition_seconds,
        "oracle_sigma_q": float(singular_values[Q - 1]),
        "oracle_sigma_q_plus_1": float(singular_values[Q]),
        "oracle_boundary_relative_gap": relative_boundary_gap,
    }


def build_sample(
    split: str,
    epsilon: float,
    beta: float,
    nodes: np.ndarray,
    boundary: np.ndarray,
    solver_rows: list[dict[str, Any]],
) -> Sample:
    identifier = sample_id(epsilon, beta)
    system = assemble_variable_diffusion_system(
        nodes,
        boundary,
        beta=beta,
        kernel=KERNEL,
        epsilon=epsilon,
    )
    base = low_precision_lu_preconditioner(system.A)
    Z, oracle_setup = selected_oracle_basis(system.A, base)
    hybrid = additive_galerkin_coarse_preconditioner(
        system.A,
        base,
        Z,
        name=f"dominant_right_singular_S_q{Q}",
    )

    reference_solution, reference_solve_seconds = system.solve_reference()
    reference_metrics = system.error_metrics(reference_solution, evaluation_side=31)

    classical_iterations: list[int] = []
    oracle_iterations: list[int] = []
    classical_converged: list[bool] = []
    oracle_converged: list[bool] = []
    classical_residuals: list[float] = []
    oracle_residuals: list[float] = []
    for rhs_name, kx, ky in RHS_SPECS:
        rhs = mode_rhs(system, beta, kx, ky)
        classical = benchmark_fgmres(
            system.A,
            rhs,
            base,
            rtol=RTOL,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
            repeats=1,
            warmup=False,
        )
        oracle = benchmark_fgmres(
            system.A,
            rhs,
            hybrid,
            rtol=RTOL,
            restart=RESTART,
            max_iterations=MAX_ITERATIONS,
            repeats=1,
            warmup=False,
        )
        base_ok = classical.final_true_relative_residual <= RTOL
        oracle_ok = oracle.final_true_relative_residual <= RTOL
        classical_iterations.append(classical.result.iterations)
        oracle_iterations.append(oracle.result.iterations)
        classical_converged.append(base_ok)
        oracle_converged.append(oracle_ok)
        classical_residuals.append(classical.final_true_relative_residual)
        oracle_residuals.append(oracle.final_true_relative_residual)
        solver_rows.append(
            {
                "sample_id": identifier,
                "split": split,
                "epsilon": epsilon,
                "beta": beta,
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "classical_iterations": classical.result.iterations,
                "oracle_iterations": oracle.result.iterations,
                "iteration_ratio": oracle.result.iterations / classical.result.iterations,
                "oracle_used_fewer_iterations": oracle.result.iterations < classical.result.iterations,
                "classical_converged": base_ok,
                "oracle_converged": oracle_ok,
                "classical_true_relative_residual": classical.final_true_relative_residual,
                "oracle_true_relative_residual": oracle.final_true_relative_residual,
                "classical_solve_seconds": classical.solve_seconds_median,
                "oracle_solve_seconds": oracle.solve_seconds_median,
            }
        )

    classical_array = np.asarray(classical_iterations, dtype=np.int32)
    oracle_array = np.asarray(oracle_iterations, dtype=np.int32)
    classical_ok_array = np.asarray(classical_converged, dtype=bool)
    oracle_ok_array = np.asarray(oracle_converged, dtype=bool)
    ratios = oracle_array / classical_array
    median_base = float(np.median(classical_array))
    median_oracle = float(np.median(oracle_array))
    metadata = {
        "sample_id": identifier,
        "split": split,
        "epsilon": epsilon,
        "beta": beta,
        "N": N,
        "side": SIDE,
        "node_family": "tensor",
        "kernel": KERNEL,
        "pde_form": "-div(exp(beta*x) grad u)=f",
        "preconditioner": "FP32 pivoted LU",
        "oracle_definition": "q=8 dominant right singular vectors of I-M^{-1}A",
        "q": Q,
        "assembly_seconds": system.assembly_seconds,
        "classical_setup_seconds": base.setup_seconds,
        "hybrid_setup_seconds": hybrid.setup_seconds,
        "reference_solve_seconds": reference_solve_seconds,
        "matrix_memory_bytes": system.A.nbytes,
        "classical_preconditioner_memory_bytes": base.memory_bytes,
        "hybrid_preconditioner_memory_bytes": hybrid.memory_bytes,
        "reference_nodal_rel_l2": reference_metrics["nodal_rel_l2"],
        "reference_grid_rel_l2": reference_metrics["grid_rel_l2"],
        "reference_relative_residual": reference_metrics["relative_residual"],
        "classical_iterations_median": median_base,
        "oracle_iterations_median": median_oracle,
        "oracle_iteration_ratio_median_of_paired_ratios": float(np.median(ratios)),
        "oracle_to_classical_median_iteration_ratio": median_oracle / median_base,
        "oracle_acceleration_fraction": float(np.mean(oracle_array < classical_array)),
        "oracle_acceleration_verified": bool(
            np.all(classical_ok_array) and np.all(oracle_ok_array) and median_oracle < median_base
        ),
        "classical_all_converged": bool(np.all(classical_ok_array)),
        "oracle_all_converged": bool(np.all(oracle_ok_array)),
        **oracle_setup,
    }
    return Sample(
        identifier,
        split,
        epsilon,
        beta,
        Z,
        classical_array,
        oracle_array,
        classical_ok_array,
        oracle_ok_array,
        np.asarray(classical_residuals, dtype=float),
        np.asarray(oracle_residuals, dtype=float),
        metadata,
    )


def write_dataset_report(
    output: Path,
    samples: list[Sample],
    metadata_rows: list[dict[str, Any]],
    p: int,
    retained_energy: float,
) -> None:
    lines = [
        "# Global RBF Difficult-Subspace Offline Dataset",
        "",
        "No neural network is trained by this pipeline.",
        "",
        "## Fixed problem and target",
        "",
        f"- N: {N} ({SIDE} x {SIDE} tensor nodes)",
        f"- Kernel: {KERNEL}",
        "- PDE: -div(exp(beta*x) grad u) = f with Dirichlet data",
        "- Classical M: FP32 pivoted LU",
        f"- Oracle Z: dominant right singular vectors of S = I - M^(-1) A, q = {Q}",
        "- Hybrid verification: the same additive Galerkin correction used in target selection",
        "",
        "## Split and representation summary",
        "",
        "| Split | Systems | Both solvers converged on all RHSs | Median projector error | Maximum projector error | Median oracle iteration ratio | Systems with verified median acceleration |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        rows = [row for row in metadata_rows if row["split"] == split]
        errors = np.array([row["projector_reconstruction_error_fro"] for row in rows])
        ratios = np.array([row["oracle_iteration_ratio_median_of_paired_ratios"] for row in rows])
        accelerated = sum(bool(row["oracle_acceleration_verified"]) for row in rows)
        fully_converged = sum(
            bool(row["classical_all_converged"]) and bool(row["oracle_all_converged"])
            for row in rows
        )
        lines.append(
            f"| {split} | {len(rows)} | {fully_converged}/{len(rows)} | {np.median(errors):.6g} | {np.max(errors):.6g} | "
            f"{np.median(ratios):.4f} | {accelerated}/{len(rows)} |"
        )
    lines.extend(
        [
            "",
            f"The shared training-only basis has p = {p} and captures {retained_energy:.8%} of concatenated training-basis energy; the requested minimum was {ENERGY_TARGET:.4%}.",
            "Parameter values are marginally disjoint across train, validation, and test. Multiple right-hand sides are stored as diagnostics within each system sample and never create duplicate ML samples.",
            "",
            "`oracle_acceleration_verified` means that both solvers converged for all four diagnostic right-hand sides and the oracle median iteration count was strictly lower than the classical median. No samples were filtered based on this outcome.",
            "",
            "## Files",
            "",
            "- `dataset.npz`: combined ML-ready arrays",
            "- `shared_basis_W.npy`: shared training-only basis",
            "- `samples/`: one compressed NPZ per parameter pair",
            "- `samples.csv`: one row per system sample",
            "- `solver_diagnostics.csv`: paired within-system FGMRES evidence",
            "- `reconstruction_diagnostics.csv`: projector errors before ML",
            "- `shared_basis_spectrum.csv`: training energy curve and retained p",
            "- `split_values.json`: explicit disjoint parameter values",
            "- `manifest.json`: shapes, definitions, and provenance",
            "- `validation_report.md`: independently recomputed integrity checks",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples_output = args.output / "samples"
    samples_output.mkdir(parents=True, exist_ok=True)

    values = parameter_values(args.quick)
    pairs = parameter_pairs(values)
    nodes, boundary = square_nodes(SIDE, "tensor")
    solver_rows: list[dict[str, Any]] = []
    samples: list[Sample] = []
    total = sum(len(split_pairs) for split_pairs in pairs.values())
    counter = 0
    for split in ("train", "validation", "test"):
        for epsilon, beta in pairs[split]:
            counter += 1
            print(
                f"[system {counter}/{total}] {split} epsilon={epsilon:g} beta={beta:g}",
                flush=True,
            )
            samples.append(
                build_sample(
                    split,
                    float(epsilon),
                    float(beta),
                    nodes,
                    boundary,
                    solver_rows,
                )
            )

    anchor = min(
        (sample for sample in samples if sample.split == "train"),
        key=lambda sample: (sample.epsilon - 5.5) ** 2 + (sample.beta - 0.5) ** 2,
    )
    for sample in samples:
        sample.Z = align_basis_to_reference(sample.Z, anchor.Z)
    training_bases = [sample.Z for sample in samples if sample.split == "train"]
    shared = fit_shared_basis(training_bases, energy_target=ENERGY_TARGET)
    W = shared.basis
    p = shared.retained_dimension
    retained_energy = float(shared.cumulative_energy[p - 1])

    coefficients: list[np.ndarray] = []
    reconstruction_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for sample in samples:
        C = coefficient_representation(W, sample.Z)
        Z_hat = reconstructed_basis(W, C)
        error, normalized_error = projector_reconstruction_error(sample.Z, Z_hat)
        coefficients.append(C)
        sample.metadata.update(
            {
                "shared_basis_dimension_p": p,
                "shared_basis_training_energy": retained_energy,
                "coefficient_rows": p,
                "coefficient_columns": Q,
                "projector_reconstruction_error_fro": error,
                "normalized_projector_reconstruction_error": normalized_error,
            }
        )
        metadata_rows.append(sample.metadata)
        reconstruction_rows.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "epsilon": sample.epsilon,
                "beta": sample.beta,
                "q": Q,
                "p": p,
                "projector_reconstruction_error_fro": error,
                "normalized_projector_reconstruction_error": normalized_error,
            }
        )
        np.savez_compressed(
            samples_output / f"{sample.sample_id}.npz",
            parameter=np.array([sample.epsilon, sample.beta], dtype=float),
            parameter_names=np.array(["epsilon", "beta"]),
            split=np.array(sample.split),
            Z=sample.Z,
            C=C,
            classical_iterations=sample.classical_iterations,
            oracle_iterations=sample.oracle_iterations,
            classical_converged=sample.classical_converged,
            oracle_converged=sample.oracle_converged,
            classical_true_relative_residual=sample.classical_true_relative_residual,
            oracle_true_relative_residual=sample.oracle_true_relative_residual,
            projector_reconstruction_error_fro=np.array(error),
            normalized_projector_reconstruction_error=np.array(normalized_error),
        )

    parameter_array = np.array([[sample.epsilon, sample.beta] for sample in samples])
    normalized_parameters = np.column_stack(
        [
            (parameter_array[:, 0] - 5.0) / (6.0 - 5.0),
            parameter_array[:, 1],
        ]
    )
    np.save(args.output / "shared_basis_W.npy", W)
    np.savez_compressed(
        args.output / "dataset.npz",
        sample_id=np.array([sample.sample_id for sample in samples]),
        split=np.array([sample.split for sample in samples]),
        parameter_names=np.array(["epsilon", "beta"]),
        parameter=parameter_array,
        parameter_normalized=normalized_parameters,
        Z=np.stack([sample.Z for sample in samples]),
        C=np.stack(coefficients),
        W=W,
        q=np.array(Q, dtype=np.int32),
        p=np.array(p, dtype=np.int32),
        energy_target=np.array(ENERGY_TARGET),
        retained_training_energy=np.array(retained_energy),
        shared_basis_singular_values=shared.singular_values,
        shared_basis_cumulative_energy=shared.cumulative_energy,
        rhs_names=np.array([spec[0] for spec in RHS_SPECS]),
        classical_iterations=np.stack([sample.classical_iterations for sample in samples]),
        oracle_iterations=np.stack([sample.oracle_iterations for sample in samples]),
        classical_converged=np.stack([sample.classical_converged for sample in samples]),
        oracle_converged=np.stack([sample.oracle_converged for sample in samples]),
        classical_true_relative_residual=np.stack(
            [sample.classical_true_relative_residual for sample in samples]
        ),
        oracle_true_relative_residual=np.stack(
            [sample.oracle_true_relative_residual for sample in samples]
        ),
        oracle_acceleration_verified=np.array(
            [sample.metadata["oracle_acceleration_verified"] for sample in samples],
            dtype=bool,
        ),
        projector_reconstruction_error_fro=np.array(
            [row["projector_reconstruction_error_fro"] for row in reconstruction_rows]
        ),
        normalized_projector_reconstruction_error=np.array(
            [row["normalized_projector_reconstruction_error"] for row in reconstruction_rows]
        ),
    )

    spectrum_rows = [
        {
            "component": index + 1,
            "singular_value": value,
            "cumulative_training_subspace_energy": shared.cumulative_energy[index],
            "retained_in_W": index < p,
        }
        for index, value in enumerate(shared.singular_values)
    ]
    write_csv(args.output / "samples.csv", metadata_rows)
    write_csv(args.output / "solver_diagnostics.csv", solver_rows)
    write_csv(args.output / "reconstruction_diagnostics.csv", reconstruction_rows)
    write_csv(args.output / "shared_basis_spectrum.csv", spectrum_rows)
    (args.output / "split_values.json").write_text(
        json.dumps(
            {
                split: {
                    "epsilon": epsilons.tolist(),
                    "beta": betas.tolist(),
                    "system_count": len(pairs[split]),
                }
                for split, (epsilons, betas) in values.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset": "global RBF selected difficult-subspace offline dataset",
        "quick": args.quick,
        "sample_grain": "one dense collocation system per (epsilon, beta) pair",
        "parameter_names": ["epsilon", "beta"],
        "parameter_normalization": {"epsilon": "(epsilon-5)/1", "beta": "beta"},
        "fixed": {
            "N": N,
            "side": SIDE,
            "node_topology": "15x15 tensor grid on [0,1]^2",
            "kernel": KERNEL,
            "pde": "-div(exp(beta*x) grad u)=f",
            "classical_preconditioner": "FP32 pivoted LU",
        },
        "oracle": {
            "definition": "dominant right singular vectors of S=I-M^{-1}A",
            "q": Q,
            "hybrid_verification": "M^{-1}r + Z solve(Z^T A Z, Z^T r)",
            "acceleration_verified_definition": (
                "both classical and oracle converge for all four diagnostic RHSs and "
                "median oracle iterations are strictly lower"
            ),
        },
        "shared_basis": {
            "training_only": True,
            "energy_target": ENERGY_TARGET,
            "p": p,
            "retained_training_energy": retained_energy,
            "coefficient_definition": "C = W^T Z after reference-gauge Procrustes alignment",
            "reconstruction": "Z_hat = orth(W C)",
            "diagnostic": "||Z Z^T - Z_hat Z_hat^T||_F",
        },
        "splits": {split: len(split_pairs) for split, split_pairs in pairs.items()},
        "marginal_parameter_values_disjoint": True,
        "rhs_diagnostics": [
            {"name": name, "kx": kx, "ky": ky} for name, kx, ky in RHS_SPECS
        ],
        "solver": {
            "rtol": RTOL,
            "restart": RESTART,
            "max_iterations": MAX_ITERATIONS,
            "true_residual_from_original_A_b": True,
        },
        "array_shapes": {
            "parameter": list(parameter_array.shape),
            "Z": [len(samples), N, Q],
            "C": [len(samples), p, Q],
            "W": [N, p],
            "iteration_counts": [len(samples), len(RHS_SPECS)],
        },
        "basis_alignment_reference_sample": anchor.sample_id,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_dataset_report(args.output, samples, metadata_rows, p, retained_energy)
    print(
        f"Wrote {len(samples)} systems, W shape {W.shape}, and C shape {(len(samples), p, Q)} to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
