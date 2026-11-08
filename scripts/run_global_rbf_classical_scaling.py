#!/usr/bin/env python3
"""Scale the existing global Gaussian-RBF benchmark with classical GMRES only."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.preconditioners import (
    Preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
    low_precision_lu_preconditioner,
    rbf_local_cardinal_preconditioner,
)
from global_rbf.scaling import (
    TensorGaussianPoissonOperator,
    assemble_tensor_gaussian_poisson_blocked,
)


DEFAULT_SIDES = (30, 40, 50, 70, 100)
METHOD_ORDER = ("none", "jacobi", "ls_acbf", "fp32_lu")
METHOD_LABELS = {
    "none": "None",
    "jacobi": "Jacobi",
    "ls_acbf": "LS-ACBF k=49",
    "fp32_lu": "FP32 pivoted LU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sides",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIDES),
        help="Square-grid side lengths; defaults give N=900,1600,2500,4900,10000.",
    )
    parser.add_argument("--epsilon-factor", type=float, default=0.4)
    parser.add_argument("--ls-support-size", type=int, default=49)
    parser.add_argument("--ls-max-n", type=int, default=2500)
    parser.add_argument("--exact-svd-max-n", type=int, default=4900)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-probes", type=int, default=16)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_classical_scaling"),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fit_power_law(
    sizes: list[float],
    values: list[float],
) -> dict[str, float]:
    if len(sizes) < 2:
        raise ValueError("at least two observations are required for extrapolation")
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    exponent, intercept = np.polyfit(x, y, 1)
    fitted = intercept + exponent * x
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = (
        1.0 - np.sum((y - fitted) ** 2) / denominator
        if denominator > np.finfo(float).tiny
        else 1.0
    )
    return {
        "exponent": float(exponent),
        "log_intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def _predict_power_law(size: float, fit: dict[str, float]) -> float:
    return float(
        np.exp(fit["log_intercept"] + fit["exponent"] * np.log(float(size)))
    )


def _condition_diagnostics(
    matrix: np.ndarray,
    structured: TensorGaussianPoissonOperator,
    *,
    exact: bool,
    prior_exact_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    start = perf_counter()
    dimension = matrix.shape[0]
    if exact:
        singular_values = la.svdvals(matrix, check_finite=False)
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        condition = sigma_max / sigma_min
        return {
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "condition_2_estimate": condition,
            "condition_estimation_method": "exact dense SVD",
            "condition_estimate_measured": True,
            "condition_fit_exponent": None,
            "condition_fit_r_squared": None,
            "condition_diagnostics_seconds": perf_counter() - start,
        }

    exact_sizes = [float(row["N"]) for row in prior_exact_rows]
    exact_conditions = [float(row["condition_2_estimate"]) for row in prior_exact_rows]
    fit = _fit_power_law(exact_sizes, exact_conditions)
    condition = _predict_power_law(dimension, fit)
    rng = np.random.default_rng(7100 + dimension)
    v0 = rng.standard_normal(dimension)
    v0 /= np.linalg.norm(v0)
    sigma_max = float(
        spla.svds(
            structured.as_linear_operator(),
            k=1,
            which="LM",
            solver="propack",
            tol=1.0e-9,
            maxiter=20000,
            v0=v0,
            return_singular_vectors=False,
        )[0]
    )
    return {
        "sigma_max": sigma_max,
        "sigma_min": sigma_max / condition,
        "condition_2_estimate": condition,
        "condition_estimation_method": (
            "log-log power-law extrapolation from exact SVDs at smaller requested N; "
            "sigma_max measured by structured PROPACK"
        ),
        "condition_estimate_measured": False,
        "condition_fit_exponent": fit["exponent"],
        "condition_fit_r_squared": fit["r_squared"],
        "condition_diagnostics_seconds": perf_counter() - start,
    }


def _apply_columns(preconditioner: Preconditioner, basis: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [preconditioner.apply(basis[:, column]) for column in range(basis.shape[1])]
    )


def _spectral_diagnostics(
    structured: TensorGaussianPoissonOperator,
    rhs: np.ndarray,
    preconditioner: Preconditioner,
    probe_basis: np.ndarray,
) -> dict[str, float]:
    mapped = _apply_columns(preconditioner, probe_basis)
    images = np.column_stack(
        [structured.matvec(mapped[:, column]) for column in range(mapped.shape[1])]
    )
    singular_values = la.svdvals(images, check_finite=False)
    compression = probe_basis.T @ images
    eigenvalues = la.eigvals(compression, check_finite=False)
    compression_norm = max(float(np.linalg.norm(compression, ord="fro")), np.finfo(float).tiny)
    commutator = compression.T @ compression - compression @ compression.T
    gains = np.linalg.norm(images, axis=0)
    identity_defect = np.linalg.norm(images - probe_basis, ord="fro") / np.sqrt(
        probe_basis.shape[1]
    )
    rhs_correction = preconditioner.apply(rhs)
    rhs_inverse_defect = np.linalg.norm(rhs - structured.matvec(rhs_correction)) / np.linalg.norm(rhs)

    rng = np.random.default_rng(991 + structured.dimension)
    left = rng.standard_normal(structured.dimension)
    right = rng.standard_normal(structured.dimension)
    applied_left = preconditioner.apply(left)
    applied_right = preconditioner.apply(right)
    additivity_defect = np.linalg.norm(
        preconditioner.apply(left + right) - applied_left - applied_right
    ) / max(
        np.linalg.norm(applied_left) + np.linalg.norm(applied_right),
        np.finfo(float).tiny,
    )
    return {
        "am_inv_random_subspace_condition": float(
            singular_values[0] / max(singular_values[-1], np.finfo(float).tiny)
        ),
        "am_inv_probe_gain_min": float(gains.min()),
        "am_inv_probe_gain_median": float(np.median(gains)),
        "am_inv_probe_gain_max": float(gains.max()),
        "am_inv_identity_defect_frobenius_per_direction": float(identity_defect),
        "am_inv_rhs_inverse_defect": float(rhs_inverse_defect),
        "am_inv_compression_eigenvalue_min_magnitude": float(np.min(np.abs(eigenvalues))),
        "am_inv_compression_eigenvalue_max_magnitude": float(np.max(np.abs(eigenvalues))),
        "am_inv_compression_median_distance_to_one": float(
            np.median(np.abs(eigenvalues - 1.0))
        ),
        "am_inv_compression_max_distance_to_one": float(
            np.max(np.abs(eigenvalues - 1.0))
        ),
        "am_inv_compression_real_part_min": float(eigenvalues.real.min()),
        "am_inv_compression_real_part_max": float(eigenvalues.real.max()),
        "am_inv_compression_max_abs_imaginary_part": float(
            np.max(np.abs(eigenvalues.imag))
        ),
        "am_inv_compression_nonnormality": float(
            np.linalg.norm(commutator, ord="fro") / compression_norm**2
        ),
        "preconditioner_additivity_defect": float(additivity_defect),
    }


def _ls_memory_bytes(dimension: int, support_size: int) -> int:
    k = min(dimension, support_size)
    return int(8 * dimension * k + 4 * dimension * k + 4 * (dimension + 1))


def _method_builder(
    method: str,
    matrix: np.ndarray,
    nodes: np.ndarray,
    *,
    support_size: int,
) -> Callable[[], Preconditioner]:
    if method == "none":
        return lambda: identity_preconditioner(matrix.shape[0])
    if method == "jacobi":
        return lambda: jacobi_preconditioner(matrix)
    if method == "ls_acbf":
        return lambda: rbf_local_cardinal_preconditioner(
            matrix,
            nodes,
            support_size=support_size,
        )
    if method == "fp32_lu":
        return lambda: low_precision_lu_preconditioner(matrix)
    raise ValueError(method)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.random_probes < 2:
        raise ValueError("random_probes must be at least two")
    sides = sorted(set(args.sides))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    measured_ls_setups: list[tuple[int, float]] = []

    for side_index, side in enumerate(sides, start=1):
        dimension = side**2
        epsilon = args.epsilon_factor * side
        print(
            f"[system {side_index}/{len(sides)}] side={side}, N={dimension}, epsilon={epsilon:g}",
            flush=True,
        )
        system = assemble_tensor_gaussian_poisson_blocked(
            side,
            epsilon=epsilon,
        )
        structured = TensorGaussianPoissonOperator(side, epsilon)
        rng = np.random.default_rng(20260901 + dimension)
        probe_basis, _ = la.qr(
            rng.standard_normal((dimension, args.random_probes)),
            mode="economic",
            check_finite=False,
        )
        consistency_probe = rng.standard_normal(dimension)
        structured_relative_defect = float(
            np.linalg.norm(system.A @ consistency_probe - structured.matvec(consistency_probe))
            / np.linalg.norm(system.A @ consistency_probe)
        )
        condition_fields = _condition_diagnostics(
            system.A,
            structured,
            exact=dimension <= args.exact_svd_max_n,
            prior_exact_rows=[
                row for row in system_rows if row["condition_estimate_measured"]
            ],
        )
        system_row: dict[str, Any] = {
            "side": side,
            "N": dimension,
            "kernel": "gaussian",
            "epsilon": epsilon,
            "epsilon_times_h": epsilon / (side - 1),
            "pde": "2D Poisson with manufactured sin(pi x) sin(pi y) solution",
            "nodes": "tensor square grid",
            "matrix_storage": "dense FP64",
            "matrix_memory_bytes": int(system.A.nbytes),
            "matrix_memory_megabytes_decimal": system.A.nbytes / 1.0e6,
            "assembly_seconds": system.assembly_seconds,
            "structured_dense_matvec_relative_defect": structured_relative_defect,
            **condition_fields,
        }
        system_rows.append(system_row)
        _write_csv(args.output_dir / "systems.csv", system_rows)

        for method in METHOD_ORDER:
            print(f"  [{method}]", flush=True)
            base_row: dict[str, Any] = {
                "side": side,
                "N": dimension,
                "epsilon": epsilon,
                "epsilon_times_h": epsilon / (side - 1),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "solver": "right-preconditioned FGMRES (fixed classical preconditioner)",
                "rtol": args.rtol,
                "restart": args.restart,
                "max_iterations": args.max_iterations,
                "timing_repeats": args.repeats,
                "warmup": not args.no_warmup,
                "matrix_memory_bytes": int(system.A.nbytes),
                "condition_2_estimate_A": condition_fields["condition_2_estimate"],
                "condition_estimation_method_A": condition_fields[
                    "condition_estimation_method"
                ],
                "ls_support_size": args.ls_support_size if method == "ls_acbf" else 0,
            }
            if method == "ls_acbf" and dimension > args.ls_max_n:
                if len(measured_ls_setups) >= 2:
                    fit = _fit_power_law(
                        [item[0] for item in measured_ls_setups],
                        [item[1] for item in measured_ls_setups],
                    )
                    projected_setup = _predict_power_law(dimension, fit)
                    projected_method = (
                        f"power-law projection from measured LS-ACBF setups; "
                        f"exponent={fit['exponent']:.3f}"
                    )
                else:
                    projected_setup = None
                    projected_method = "insufficient measured setup points"
                benchmark_rows.append(
                    {
                        **base_row,
                        "run_status": "not_run_feasibility_cutoff",
                        "feasible": False,
                        "feasibility_reason": (
                            f"predeclared LS-ACBF setup cutoff N>{args.ls_max_n}; "
                            "one global tall least-squares fit per row"
                        ),
                        "setup_seconds": None,
                        "setup_seconds_projected": projected_setup,
                        "setup_projection_method": projected_method,
                        "preconditioner_memory_bytes": _ls_memory_bytes(
                            dimension,
                            args.ls_support_size,
                        ),
                        "matrix_plus_preconditioner_bytes": int(
                            system.A.nbytes
                            + _ls_memory_bytes(dimension, args.ls_support_size)
                        ),
                        "gmres_iterations": None,
                        "converged": None,
                        "solve_seconds": None,
                        "first_rhs_seconds": None,
                        "final_true_relative_residual": None,
                    }
                )
                _write_csv(args.output_dir / "benchmark.csv", benchmark_rows)
                continue

            builder = _method_builder(
                method,
                system.A,
                system.nodes,
                support_size=args.ls_support_size,
            )
            try:
                preconditioner = builder()
                if method == "ls_acbf":
                    measured_ls_setups.append(
                        (dimension, preconditioner.setup_seconds)
                    )
                diagnostics_start = perf_counter()
                spectral = _spectral_diagnostics(
                    structured,
                    system.b,
                    preconditioner,
                    probe_basis,
                )
                spectral_seconds = perf_counter() - diagnostics_start
                application_seconds = preconditioner.benchmark_application(
                    dimension,
                    repeats=10,
                    seed=dimension,
                )
                benchmark = benchmark_fgmres(
                    system.A,
                    system.b,
                    preconditioner,
                    rtol=args.rtol,
                    restart=args.restart,
                    max_iterations=args.max_iterations,
                    repeats=args.repeats,
                    warmup=not args.no_warmup,
                )
                result = benchmark.result
                converged = benchmark.final_true_relative_residual <= args.rtol
                row = {
                    **base_row,
                    "run_status": "measured",
                    "feasible": True,
                    "feasibility_reason": "measured on this workstation",
                    "setup_seconds": preconditioner.setup_seconds,
                    "setup_seconds_projected": None,
                    "setup_projection_method": None,
                    "preconditioner_application_seconds": application_seconds,
                    "preconditioner_memory_bytes": preconditioner.memory_bytes,
                    "matrix_plus_preconditioner_bytes": int(
                        system.A.nbytes + preconditioner.memory_bytes
                    ),
                    "spectral_diagnostics_seconds": spectral_seconds,
                    "gmres_iterations": result.iterations,
                    "converged": converged,
                    "solve_seconds": benchmark.solve_seconds_median,
                    "first_rhs_seconds": (
                        preconditioner.setup_seconds + benchmark.solve_seconds_median
                    ),
                    "final_true_residual_norm": benchmark.final_true_residual_norm,
                    "final_true_relative_residual": benchmark.final_true_relative_residual,
                    "restart_cycles": int(np.ceil(result.iterations / args.restart)),
                    "preconditioner_notes": preconditioner.notes,
                    **spectral,
                }
                benchmark_rows.append(row)
                histories.extend(
                    {
                        "N": dimension,
                        "side": side,
                        "method": method,
                        "history_index": history_index,
                        "relative_residual_estimate_or_restart_true_residual": value,
                    }
                    for history_index, value in enumerate(result.residual_history)
                )
                print(
                    f"    setup={preconditioner.setup_seconds:.3f}s "
                    f"iterations={result.iterations} converged={converged} "
                    f"relres={benchmark.final_true_relative_residual:.3e} "
                    f"solve={benchmark.solve_seconds_median:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    **base_row,
                    "run_status": "failed",
                    "feasible": False,
                    "feasibility_reason": repr(exc),
                    "setup_seconds": None,
                    "setup_seconds_projected": None,
                    "preconditioner_memory_bytes": None,
                    "matrix_plus_preconditioner_bytes": None,
                    "gmres_iterations": None,
                    "converged": False,
                    "solve_seconds": None,
                    "first_rhs_seconds": None,
                    "final_true_relative_residual": None,
                }
                benchmark_rows.append(failure)
                failures.append(
                    {"side": side, "N": dimension, "method": method, "error": repr(exc)}
                )
                print(f"    FAILED: {exc!r}", flush=True)
            _write_csv(args.output_dir / "benchmark.csv", benchmark_rows)
            _write_csv(args.output_dir / "residual_history.csv", histories)
            _write_csv(args.output_dir / "failures.csv", failures)

        preconditioner = None
        builder = None
        benchmark = None
        result = None
        probe_basis = None
        structured = None
        consistency_probe = None
        del system
        gc.collect()

    exact_rows = [row for row in system_rows if row["condition_estimate_measured"]]
    condition_fit = _fit_power_law(
        [row["N"] for row in exact_rows],
        [row["condition_2_estimate"] for row in exact_rows],
    )
    holdout: dict[str, Any] | None = None
    if len(exact_rows) >= 4:
        training = exact_rows[:-1]
        validation_row = exact_rows[-1]
        holdout_fit = _fit_power_law(
            [row["N"] for row in training],
            [row["condition_2_estimate"] for row in training],
        )
        predicted = _predict_power_law(validation_row["N"], holdout_fit)
        holdout = {
            "held_out_N": validation_row["N"],
            "actual_condition_2": validation_row["condition_2_estimate"],
            "predicted_condition_2": predicted,
            "prediction_over_actual": predicted
            / validation_row["condition_2_estimate"],
        }

    summary = {
        "configuration": {
            "sides": sides,
            "N_values": [side**2 for side in sides],
            "kernel": "gaussian",
            "epsilon_rule": "epsilon = 0.4 * grid side by default",
            "epsilon_factor": args.epsilon_factor,
            "pde": "2D Poisson, exact u=sin(pi x) sin(pi y), Dirichlet boundary rows",
            "node_distribution": "tensor square grids",
            "solver": "right-preconditioned FGMRES with fixed classical maps (GMRES-equivalent)",
            "rtol": args.rtol,
            "atol": 0.0,
            "restart": args.restart,
            "max_iterations": args.max_iterations,
            "timing_repeats": args.repeats,
            "warmup": not args.no_warmup,
            "ls_acbf_support_size": args.ls_support_size,
            "ls_acbf_support_interpretation": "fixed 7x7 nearest-node footprint; comparable because epsilon*h is nearly constant",
            "ls_acbf_measured_max_N": args.ls_max_n,
            "exact_svd_max_N": args.exact_svd_max_n,
            "random_spectral_probe_dimension": args.random_probes,
            "no_ml": True,
            "no_oracle_subspaces": True,
        },
        "condition_extrapolation": {
            **condition_fit,
            "formula": "condition_2 approximately exp(log_intercept) * N^exponent",
            "holdout_validation": holdout,
        },
        "systems": system_rows,
        "results": benchmark_rows,
        "failures": failures,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "systems.csv", system_rows)
    _write_csv(args.output_dir / "benchmark.csv", benchmark_rows)
    _write_csv(args.output_dir / "residual_history.csv", histories)
    _write_csv(args.output_dir / "failures.csv", failures)
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
