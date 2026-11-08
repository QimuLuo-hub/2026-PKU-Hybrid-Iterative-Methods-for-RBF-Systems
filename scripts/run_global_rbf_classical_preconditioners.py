"""Benchmark three classical baselines on the existing global-RBF families.

The matrix families, assembly routines, FGMRES tolerance, restart, iteration
cap, and CSV conventions are imported from the Stage 2 experiment.  No neural
model or oracle coarse space is used here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.preconditioners import (
    Preconditioner,
    identity_preconditioner,
    low_precision_lu_preconditioner,
    rbf_local_cardinal_preconditioner,
)
from run_global_rbf_stage2 import (
    BASE_EPSILON,
    N,
    RTOL,
    FamilySpec,
    assemble_family_member,
    family_specs,
    write_csv,
)


DEFAULT_OUTPUT = Path("results/global_rbf_classical_preconditioners")
DEFAULT_SUPPORT_SIZE = 161
DEFAULT_REPEATS = 5
SUPPORT_PROBE_SIZES = (25, 49, 81, 121)
METHOD_ORDER = ("none", "fp32_lu", "rbf_local_cardinal_ls")
METHOD_LABELS = {
    "none": "None",
    "fp32_lu": "FP32 LU",
    "rbf_local_cardinal_ls": "Local LS-ACBF",
}
METHOD_COLORS = {
    "none": "#8A9099",
    "fp32_lu": "#1F5A94",
    "rbf_local_cardinal_ls": "#C58C24",
}
FAMILY_MARKERS = {
    "epsilon": "o",
    "jitter": "s",
    "pde_coefficient": "^",
}


def system_metadata(spec: FamilySpec, index: int, value: float, system_id: str) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "family": spec.family,
        "family_parameter_name": spec.parameter,
        "family_parameter_index": index,
        "family_parameter_value": value,
        "kernel": "gaussian",
        "epsilon": value if spec.family == "epsilon" else BASE_EPSILON,
        "N": N,
        "node_family": "frozen_jitter" if spec.family == "jitter" else "tensor",
        "jitter_fraction": value if spec.family == "jitter" else 0.0,
        "pde_operator": "variable_diffusion" if spec.family == "pde_coefficient" else "poisson",
        "pde_parameter_name": "beta",
        "pde_parameter": value if spec.family == "pde_coefficient" else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in ("all", "epsilon", "jitter", "pde_coefficient"):
        family_rows = rows if family == "all" else [row for row in rows if row["family"] == family]
        for method in METHOD_ORDER:
            selected = [row for row in family_rows if row["method"] == method]
            converged = [row for row in selected if bool(row["converged"])]
            iterations = np.array([int(row["fgmres_iterations"]) for row in selected], dtype=float)
            converged_iterations = np.array(
                [int(row["fgmres_iterations"]) for row in converged],
                dtype=float,
            )
            summaries.append(
                {
                    "family": family,
                    "method": method,
                    "systems": len(selected),
                    "converged_systems": len(converged),
                    "convergence_fraction": len(converged) / len(selected),
                    "median_iterations_all": float(np.median(iterations)),
                    "median_iterations_converged": (
                        float(np.median(converged_iterations)) if len(converged_iterations) else float("nan")
                    ),
                    "median_setup_seconds": float(np.median([row["setup_seconds"] for row in selected])),
                    "median_solve_seconds": float(np.median([row["solve_seconds"] for row in selected])),
                    "median_first_rhs_seconds": float(
                        np.median([row["total_seconds_first_rhs"] for row in selected])
                    ),
                    "median_application_seconds": float(
                        np.median([row["application_seconds"] for row in selected])
                    ),
                    "median_preconditioner_memory_bytes": float(
                        np.median([row["preconditioner_memory_bytes"] for row in selected])
                    ),
                    "max_final_true_relative_residual": float(
                        np.max([row["final_true_relative_residual"] for row in selected])
                    ),
                }
            )
    return summaries


def benchmark_method(
    method: str,
    builder: Callable[[], Preconditioner],
    system: Any,
    metadata: dict[str, Any],
    *,
    repeats: int,
    support_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preconditioner = builder()
    benchmark = benchmark_fgmres(
        system.A,
        system.b,
        preconditioner,
        rtol=RTOL,
        restart=min(100, system.dimension),
        max_iterations=min(450, 2 * system.dimension),
        repeats=repeats,
        warmup=True,
    )
    reference_lu = la.lu_factor(system.A, check_finite=False)
    reference = la.lu_solve(reference_lu, system.b, check_finite=False)
    result = benchmark.result
    relative_error = float(np.linalg.norm(result.x - reference) / np.linalg.norm(reference))
    row = {
        **metadata,
        "method": method,
        "preconditioner_name": preconditioner.name,
        "solver_type": "right-preconditioned FGMRES",
        "fgmres_rtol": RTOL,
        "fgmres_restart": min(100, system.dimension),
        "fgmres_max_iterations": min(450, 2 * system.dimension),
        "timing_repeats": repeats,
        "setup_seconds": preconditioner.setup_seconds,
        "solve_seconds": benchmark.solve_seconds_median,
        "solve_seconds_median": benchmark.solve_seconds_median,
        "total_seconds_first_rhs": preconditioner.setup_seconds + benchmark.solve_seconds_median,
        "application_seconds": preconditioner.benchmark_application(system.dimension, repeats=100),
        "fgmres_iterations": result.iterations,
        "converged": benchmark.final_true_relative_residual <= RTOL,
        "final_true_residual_norm": benchmark.final_true_residual_norm,
        "final_true_relative_residual": benchmark.final_true_relative_residual,
        "coefficient_relative_error_vs_fp64_lu": relative_error,
        "preconditioner_memory_bytes": preconditioner.memory_bytes,
        "matrix_memory_bytes": system.A.nbytes,
        "matrix_plus_preconditioner_bytes": system.A.nbytes + preconditioner.memory_bytes,
        "assembly_seconds": system.assembly_seconds,
        "local_support_size": support_size if method == "rbf_local_cardinal_ls" else 0,
        "local_support_fraction": support_size / system.dimension if method == "rbf_local_cardinal_ls" else 0.0,
        "notes": preconditioner.notes,
    }
    history = [
        {
            "system_id": metadata["system_id"],
            "family": metadata["family"],
            "method": method,
            "history_point": point,
            "relative_residual_estimate_or_restart_true_residual": residual,
        }
        for point, residual in enumerate(result.residual_history)
    ]
    return row, history


def plot_results(rows: list[dict[str, Any]], output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    rng = np.random.default_rng(20260830)

    for ax, field, ylabel, log_scale in (
        (axes[0], "fgmres_iterations", "FGMRES iterations", False),
        (axes[1], "total_seconds_first_rhs", "First-RHS setup + solve time (s)", True),
    ):
        data = [np.array([float(row[field]) for row in rows if row["method"] == method]) for method in METHOD_ORDER]
        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#252A31", "linewidth": 1.7},
            whiskerprops={"color": "#5E646C"},
            capprops={"color": "#5E646C"},
        )
        for patch, method in zip(boxes["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.48)
            patch.set_edgecolor("#252A31")
        for x_position, method in zip(positions, METHOD_ORDER):
            selected = [row for row in rows if row["method"] == method]
            jitter = rng.uniform(-0.13, 0.13, size=len(selected))
            for dx, row in zip(jitter, selected):
                ax.scatter(
                    x_position + dx,
                    float(row[field]),
                    marker=FAMILY_MARKERS[row["family"]],
                    s=28,
                    facecolor=METHOD_COLORS[method],
                    edgecolor="#252A31",
                    linewidth=0.45,
                    alpha=0.82,
                    zorder=3,
                )
                if field == "fgmres_iterations" and not bool(row["converged"]):
                    ax.scatter(
                        x_position + dx,
                        float(row[field]),
                        marker="x",
                        s=45,
                        color="#252A31",
                        linewidth=1.0,
                        zorder=4,
                    )
        ax.set_xticks(positions, [METHOD_LABELS[method] for method in METHOD_ORDER])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DCE2E8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if log_scale:
            ax.set_yscale("log")

    axes[0].set_title("Iteration distributions", loc="left", fontweight="bold")
    axes[0].set_ylim(0, 475)
    axes[1].set_title("First-right-hand-side cost", loc="left", fontweight="bold")
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#252A31",
            label=family.replace("_", " "),
            markersize=6,
        )
        for family, marker in FAMILY_MARKERS.items()
    ]
    family_handles.append(
        Line2D([0], [0], marker="x", color="#252A31", linestyle="none", label="iteration cap")
    )
    fig.suptitle("Global RBF classical preconditioner benchmark", x=0.075, ha="left", fontweight="bold")
    fig.text(
        0.075,
        0.935,
        "27 Gaussian N=225 systems; FGMRES relative tolerance 1e-10; points show individual systems",
        ha="left",
        color="#5E646C",
        fontsize=9,
    )
    fig.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        frameon=False,
        fontsize=8,
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path = output / "classical_preconditioner_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run_support_probe(
    specs: tuple[FamilySpec, ...],
    *,
    support_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sizes = (*SUPPORT_PROBE_SIZES, support_size)
    for spec in specs:
        for index in (0, len(spec.values) // 2, len(spec.values) - 1):
            value = spec.values[index]
            system_id, system = assemble_family_member(spec, index, value)
            metadata = system_metadata(spec, index, value, system_id)
            for k in sizes:
                preconditioner = rbf_local_cardinal_preconditioner(system.A, system.nodes, support_size=k)
                benchmark = benchmark_fgmres(
                    system.A,
                    system.b,
                    preconditioner,
                    rtol=RTOL,
                    restart=min(100, system.dimension),
                    max_iterations=min(450, 2 * system.dimension),
                    repeats=1,
                    warmup=False,
                )
                rows.append(
                    {
                        **metadata,
                        "support_size": k,
                        "support_fraction": k / system.dimension,
                        "setup_seconds": preconditioner.setup_seconds,
                        "fgmres_iterations": benchmark.result.iterations,
                        "converged": benchmark.final_true_relative_residual <= RTOL,
                        "final_true_relative_residual": benchmark.final_true_relative_residual,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--support-size", type=int, default=DEFAULT_SUPPORT_SIZE)
    parser.add_argument("--skip-support-probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = family_specs(args.quick)
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = sum(len(spec.values) for spec in specs)
    counter = 0
    for spec in specs:
        for index, value in enumerate(spec.values):
            counter += 1
            system_id, system = assemble_family_member(spec, index, value)
            metadata = system_metadata(spec, index, value, system_id)
            print(f"[system {counter}/{total}] {system_id}", flush=True)
            builders: tuple[tuple[str, Callable[[], Preconditioner]], ...] = (
                ("none", lambda: identity_preconditioner(system.dimension)),
                ("fp32_lu", lambda: low_precision_lu_preconditioner(system.A)),
                (
                    "rbf_local_cardinal_ls",
                    lambda: rbf_local_cardinal_preconditioner(
                        system.A,
                        system.nodes,
                        support_size=args.support_size,
                    ),
                ),
            )
            for method, builder in builders:
                print(f"  {method}", flush=True)
                try:
                    row, method_history = benchmark_method(
                        method,
                        builder,
                        system,
                        metadata,
                        repeats=args.repeats,
                        support_size=args.support_size,
                    )
                    rows.append(row)
                    histories.extend(method_history)
                except Exception as exc:
                    failures.append({**metadata, "method": method, "error": repr(exc)})

    write_csv(args.output / "benchmark_results.csv", rows)
    write_csv(args.output / "residual_histories.csv", histories)
    write_csv(args.output / "benchmark_summary.csv", summarize(rows))
    write_csv(args.output / "failures.csv", failures)

    if not args.skip_support_probe and not args.quick:
        print("[support probe] representative endpoints and midpoints", flush=True)
        support_rows = run_support_probe(specs, support_size=args.support_size)
        write_csv(args.output / "support_probe.csv", support_rows)

    plot_path = plot_results(rows, args.output)
    write_csv(
        args.output / "chart_map.csv",
        [
            {
                "section": "three-way classical comparison",
                "question": "Which classical preconditioner should be the primary hybrid baseline?",
                "family": "distribution and cost comparison",
                "fields": "method, family, fgmres_iterations, total_seconds_first_rhs, converged",
                "claim": "Compare robustness, iteration work, and first-RHS cost without hiding capped solves.",
                "palette": "single method color plus family marker shape",
                "path": plot_path.name,
            }
        ],
    )

    config = {
        "experiment": "global RBF three-way classical preconditioner benchmark",
        "quick": args.quick,
        "families": [
            {"family": spec.family, "parameter": spec.parameter, "values": spec.values}
            for spec in specs
        ],
        "N": N,
        "kernel": "gaussian",
        "base_epsilon": BASE_EPSILON,
        "fgmres_rtol": RTOL,
        "fgmres_restart": min(100, N),
        "fgmres_max_iterations": min(450, 2 * N),
        "timing_repeats": args.repeats,
        "warmup_solves": 1,
        "local_cardinal_support_size": args.support_size,
        "local_cardinal_support_fraction": args.support_size / N,
        "support_probe_sizes": [] if args.skip_support_probe else [*SUPPORT_PROBE_SIZES, args.support_size],
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    (args.output / "software_environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} benchmark rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
