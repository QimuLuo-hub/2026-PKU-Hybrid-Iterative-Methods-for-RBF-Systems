#!/usr/bin/env python3
"""Find one converged, nontrivial global-RBF case for recycle experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import scipy

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.preconditioners import (
    Preconditioner,
    jacobi_preconditioner,
    low_precision_lu_preconditioner,
    rbf_local_cardinal_preconditioner,
)
from global_rbf.scaling import assemble_tensor_gaussian_poisson_blocked


DEFAULT_SIDES = (20, 25, 30)
METHOD_ORDER = ("jacobi", "fp32_lu", "ls_acbf")
METHOD_LABELS = {
    "jacobi": "Jacobi",
    "fp32_lu": "FP32 pivoted LU",
}


def _method_label(method: str, ls_support_size: int) -> str:
    if method == "ls_acbf":
        return f"LS-ACBF k={ls_support_size}"
    return METHOD_LABELS[method]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", type=int, nargs="+", default=list(DEFAULT_SIDES))
    parser.add_argument("--epsilon-factor", type=float, default=0.4)
    parser.add_argument("--ls-support-size", type=int, default=49)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--target-min-iterations", type=int, default=100)
    parser.add_argument("--target-max-iterations", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/global_rbf_recycle_benchmark_search"),
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _builder(
    method: str,
    matrix: np.ndarray,
    nodes: np.ndarray,
    *,
    ls_support_size: int,
) -> Callable[[], Preconditioner]:
    if method == "jacobi":
        return lambda: jacobi_preconditioner(matrix)
    if method == "fp32_lu":
        return lambda: low_precision_lu_preconditioner(matrix)
    if method == "ls_acbf":
        return lambda: rbf_local_cardinal_preconditioner(
            matrix,
            nodes,
            support_size=ls_support_size,
        )
    raise ValueError(method)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.target_min_iterations < 0:
        raise ValueError("target-min-iterations must be nonnegative")
    if args.target_max_iterations < args.target_min_iterations:
        raise ValueError("target-max-iterations must be at least target-min-iterations")

    sides = sorted(set(args.sides))
    expected_sides = set(DEFAULT_SIDES)
    if not set(sides).issubset(expected_sides):
        raise ValueError(
            "this targeted search is restricted to sides 20, 25, and 30 "
            "(N=400, 625, and 900)"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    saved_solutions: dict[str, np.ndarray] = {}
    selected: dict[str, Any] | None = None

    for side in sides:
        dimension = side**2
        epsilon = args.epsilon_factor * side
        print(f"[N={dimension}] epsilon={epsilon:g}", flush=True)
        system = assemble_tensor_gaussian_poisson_blocked(side, epsilon=epsilon)
        rhs_norm = float(np.linalg.norm(system.b))

        for method in METHOD_ORDER:
            method_label = _method_label(method, args.ls_support_size)
            print(f"  [{method_label}]", flush=True)
            base_row = {
                "N": dimension,
                "side": side,
                "kernel": "gaussian",
                "epsilon": epsilon,
                "epsilon_times_h": epsilon / (side - 1),
                "node_family": "tensor square grid",
                "pde_parameter": "constant diffusion beta=0",
                "preconditioner": method,
                "preconditioner_label": method_label,
                "solver": "right-preconditioned FGMRES",
                "rtol": args.rtol,
                "restart": args.restart,
                "max_iterations": args.max_iterations,
                "timing_repeats": args.repeats,
                "warmup": not args.no_warmup,
                "ls_support_size": args.ls_support_size if method == "ls_acbf" else 0,
                "matrix_memory_bytes": int(system.A.nbytes),
            }
            try:
                preconditioner = _builder(
                    method,
                    system.A,
                    system.nodes,
                    ls_support_size=args.ls_support_size,
                )()
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
                true_relative_residual = benchmark.final_true_relative_residual
                converged = bool(
                    result.converged and true_relative_residual <= args.rtol
                )
                suitable = bool(
                    converged
                    and args.target_min_iterations
                    <= result.iterations
                    <= args.target_max_iterations
                )
                solution_key = f"N{dimension}_{method}"
                saved_solutions[solution_key] = result.x
                row = {
                    **base_row,
                    "status": "measured",
                    "converged": converged,
                    "iterations": result.iterations,
                    "final_true_residual_norm": benchmark.final_true_residual_norm,
                    "final_true_relative_residual": true_relative_residual,
                    "rhs_norm": rhs_norm,
                    "setup_seconds": preconditioner.setup_seconds,
                    "solve_seconds": benchmark.solve_seconds_median,
                    "preconditioner_memory_bytes": preconditioner.memory_bytes,
                    "suitable": suitable,
                    "solution_key": solution_key,
                }
                rows.append(row)
                print(
                    f"    iterations={result.iterations} converged={converged} "
                    f"true_relres={true_relative_residual:.6e} "
                    f"setup={preconditioner.setup_seconds:.4f}s "
                    f"solve={benchmark.solve_seconds_median:.4f}s "
                    f"suitable={suitable}",
                    flush=True,
                )
                if suitable:
                    selected = dict(row)
                    break
            except Exception as exc:
                row = {
                    **base_row,
                    "status": "failed",
                    "converged": False,
                    "iterations": None,
                    "final_true_residual_norm": None,
                    "final_true_relative_residual": None,
                    "rhs_norm": rhs_norm,
                    "setup_seconds": None,
                    "solve_seconds": None,
                    "preconditioner_memory_bytes": None,
                    "suitable": False,
                    "solution_key": None,
                }
                rows.append(row)
                failures.append(
                    {
                        "N": dimension,
                        "preconditioner": method,
                        "error": repr(exc),
                    }
                )
                print(f"    FAILED: {exc!r}", flush=True)

            _write_csv(args.output_dir / "benchmark.csv", rows)
            _write_csv(args.output_dir / "failures.csv", failures)
            np.savez_compressed(args.output_dir / "solutions.npz", **saved_solutions)

        if selected is not None:
            break

    _write_csv(args.output_dir / "benchmark.csv", rows)
    _write_csv(args.output_dir / "failures.csv", failures)
    np.savez_compressed(args.output_dir / "solutions.npz", **saved_solutions)

    configuration = {
        "search_N_values": [side**2 for side in sides],
        "method_order": list(METHOD_ORDER),
        "kernel": "gaussian",
        "epsilon_rule": "epsilon = 0.4 * square-grid side",
        "epsilon_factor": args.epsilon_factor,
        "nodes": "tensor square grid",
        "pde": "2D Poisson, exact u=sin(pi x) sin(pi y), Dirichlet boundary rows",
        "pde_parameter": "constant diffusion beta=0",
        "solver": "right-preconditioned FGMRES",
        "rtol": args.rtol,
        "atol": 0.0,
        "restart": args.restart,
        "max_iterations": args.max_iterations,
        "timing_repeats": args.repeats,
        "warmup": not args.no_warmup,
        "ls_acbf_support_size": args.ls_support_size,
        "selection_rule": {
            "minimum_iterations": args.target_min_iterations,
            "maximum_iterations": args.target_max_iterations,
            "maximum_true_relative_residual": args.rtol,
        },
        "stop_rule": "stop immediately after the first suitable measured case",
        "ML_used": False,
        "recycle_space_used": False,
    }
    summary = {
        "configuration": configuration,
        "results": rows,
        "selected_case": selected,
        "failures": failures,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selected_case.json").write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output_dir}; selected="
        f"{None if selected is None else (selected['N'], selected['preconditioner'])}",
        flush=True,
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
