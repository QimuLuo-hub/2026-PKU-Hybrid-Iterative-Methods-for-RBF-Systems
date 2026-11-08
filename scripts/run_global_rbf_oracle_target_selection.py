"""Select a solver-relevant oracle difficult-space definition for global RBFs.

The experiment uses the established FP32 pivoted-LU preconditioner and
compares only the three requested spaces under one identical additive
Galerkin correction.  No learned model is trained.
"""

from __future__ import annotations

import argparse
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

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la

from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.collocation import GlobalRBFSystem, variable_diffusion_forcing
from global_rbf.krylov import fgmres
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import additive_galerkin_coarse_preconditioner
from run_global_rbf_stage2 import (
    BASE_EPSILON,
    N,
    RTOL,
    FamilySpec,
    assemble_family_member,
    write_csv,
)


OUTPUT_DEFAULT = Path("results/global_rbf_oracle_target_selection")
Q_VALUES = (8, 16, 24, 32)
MAX_Q = max(Q_VALUES)
REPEATS = 3
TRAIN_RHS_SPECS = (
    ("train_mode_1_1", 1, 1),
    ("train_mode_1_2", 1, 2),
    ("train_mode_2_1", 2, 1),
    ("train_mode_2_2", 2, 2),
    ("train_mode_1_3", 1, 3),
    ("train_mode_3_1", 3, 1),
)
EVALUATION_RHS_SPECS = (
    ("heldout_mode_2_3", 2, 3),
    ("heldout_mode_3_2", 3, 2),
    ("heldout_mode_1_4", 1, 4),
    ("heldout_mode_4_1", 4, 1),
)
CANDIDATES = (
    "dominant_right_singular_S",
    "smallest_right_singular_B",
    "late_residual_pod",
)
CANDIDATE_LABELS = {
    "dominant_right_singular_S": "Dominant right singular vectors of S",
    "smallest_right_singular_B": "Smallest right singular vectors of B",
    "late_residual_pod": "Late-residual POD",
}
COLORS = {
    "dominant_right_singular_S": "#1F5A94",
    "smallest_right_singular_B": "#C58C24",
    "late_residual_pod": "#D06B32",
}
MARKERS = {
    "dominant_right_singular_S": "o",
    "smallest_right_singular_B": "s",
    "late_residual_pod": "^",
}
LINESTYLES = {
    "dominant_right_singular_S": "-",
    "smallest_right_singular_B": "--",
    "late_residual_pod": "-.",
}


@dataclass(frozen=True)
class TargetConfig:
    family: str
    parameter: str
    value: float
    nearby_values: tuple[float, ...]
    selection_reason: str


TARGET_CONFIGS = (
    TargetConfig(
        "epsilon",
        "epsilon",
        5.0,
        (5.25,),
        "hardest saved FP32-LU epsilon system (271 iterations on the original RHS)",
    ),
    TargetConfig(
        "epsilon",
        "epsilon",
        5.75,
        (5.5, 6.0),
        "second-hardest saved FP32-LU epsilon system (147 iterations)",
    ),
    TargetConfig(
        "jitter",
        "jitter_fraction",
        0.0275,
        (0.0, 0.055),
        "hardest nonzero-jitter member without duplicating the tensor epsilon=6 system",
    ),
    TargetConfig(
        "pde_coefficient",
        "beta",
        0.125,
        (0.0, 0.25),
        "hardest nonzero variable-diffusion member (23 iterations)",
    ),
)


@dataclass
class SpaceArtifact:
    system_id: str
    family: str
    parameter: str
    value: float
    system: GlobalRBFSystem
    base: Preconditioner
    spaces: dict[str, np.ndarray]
    space_setup_seconds: dict[str, float]
    space_metadata: list[dict[str, Any]]
    trajectory_metadata: list[dict[str, Any]]


def make_family_spec(family: str, parameter: str, value: float) -> FamilySpec:
    return FamilySpec(family, parameter, (float(value),))


def assemble_artifact_system(family: str, parameter: str, value: float) -> tuple[str, GlobalRBFSystem]:
    spec = make_family_spec(family, parameter, value)
    return assemble_family_member(spec, 0, float(value))


def mode_rhs(system: GlobalRBFSystem, family: str, parameter_value: float, kx: int, ky: int) -> np.ndarray:
    points = system.nodes
    u = np.sin(kx * np.pi * points[:, 0]) * np.sin(ky * np.pi * points[:, 1])
    if family == "pde_coefficient":
        forcing = variable_diffusion_forcing(points, parameter_value, kx=kx, ky=ky)
    else:
        forcing = (kx**2 + ky**2) * np.pi**2 * u
    return np.where(system.boundary, u, forcing)


def system_fields(family: str, parameter: str, value: float, system_id: str) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "family": family,
        "parameter": parameter,
        "parameter_value": value,
        "kernel": "gaussian",
        "epsilon": value if family == "epsilon" else BASE_EPSILON,
        "N": N,
        "node_family": "frozen_jitter" if family == "jitter" else "tensor",
        "jitter_fraction": value if family == "jitter" else 0.0,
        "pde_parameter_beta": value if family == "pde_coefficient" else 0.0,
        "pde_operator": "variable_diffusion" if family == "pde_coefficient" else "poisson",
    }


def build_space_artifact(
    family: str,
    parameter: str,
    value: float,
    *,
    spaces_output: Path,
) -> SpaceArtifact:
    system_id, system = assemble_artifact_system(family, parameter, value)
    print(f"[space] {system_id}", flush=True)
    base = low_precision_lu_preconditioner(system.A)

    operator_start = perf_counter()
    B = np.asarray(base.apply(system.A), dtype=float)
    operator_seconds = perf_counter() - operator_start

    s_start = perf_counter()
    S = np.eye(system.dimension) - B
    _, _, Vh_s = la.svd(S, full_matrices=False, check_finite=False)
    s_decomposition_seconds = perf_counter() - s_start
    dominant_s = Vh_s[:MAX_Q, :].T
    del S, Vh_s

    b_start = perf_counter()
    _, _, Vh_b = la.svd(B, full_matrices=False, check_finite=False)
    b_decomposition_seconds = perf_counter() - b_start
    smallest_b = Vh_b[-1 : -MAX_Q - 1 : -1, :].T
    del B, Vh_b

    pod_start = perf_counter()
    late_snapshots: list[np.ndarray] = []
    trajectory_rows: list[dict[str, Any]] = []
    for rhs_name, kx, ky in TRAIN_RHS_SPECS:
        rhs = mode_rhs(system, family, value, kx, ky)
        result = fgmres(
            system.A,
            rhs,
            base.apply,
            rtol=RTOL,
            restart=min(100, system.dimension),
            max_iterations=min(450, 2 * system.dimension),
            capture_residual_vectors=True,
        )
        residuals = result.residual_vectors
        late_start = len(residuals) // 2
        scale = float(np.linalg.norm(rhs))
        selected = [residual / scale for residual in residuals[late_start:]]
        late_snapshots.extend(selected)
        trajectory_rows.append(
            {
                **system_fields(family, parameter, value, system_id),
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "fgmres_iterations": result.iterations,
                "converged": result.converged,
                "final_true_relative_residual": result.relative_residual,
                "residual_snapshots": len(residuals),
                "late_start_index": late_start,
                "late_snapshots_used": len(selected),
            }
        )
    snapshot_matrix = np.column_stack(late_snapshots)
    U_pod, pod_singular_values, _ = la.svd(snapshot_matrix, full_matrices=False, check_finite=False)
    late_pod = U_pod[:, :MAX_Q]
    pod_seconds = perf_counter() - pod_start
    pod_tolerance = (
        max(snapshot_matrix.shape) * np.finfo(float).eps * pod_singular_values[0]
        if len(pod_singular_values)
        else 0.0
    )
    pod_numerical_rank = int(np.count_nonzero(pod_singular_values > pod_tolerance))

    spaces = {
        "dominant_right_singular_S": dominant_s,
        "smallest_right_singular_B": smallest_b,
        "late_residual_pod": late_pod,
    }
    setup = {
        "dominant_right_singular_S": operator_seconds + s_decomposition_seconds,
        "smallest_right_singular_B": operator_seconds + b_decomposition_seconds,
        "late_residual_pod": pod_seconds,
    }
    common = system_fields(family, parameter, value, system_id)
    metadata_rows = [
        {
            **common,
            "candidate": "dominant_right_singular_S",
            "candidate_definition": CANDIDATE_LABELS["dominant_right_singular_S"],
            "space_rank_saved": MAX_Q,
            "space_setup_seconds": setup["dominant_right_singular_S"],
            "basis_memory_bytes": dominant_s.nbytes,
            "preconditioned_operator_formation_seconds": operator_seconds,
            "trajectory_count": 0,
            "snapshot_count": 0,
            "snapshot_numerical_rank": 0,
        },
        {
            **common,
            "candidate": "smallest_right_singular_B",
            "candidate_definition": CANDIDATE_LABELS["smallest_right_singular_B"],
            "space_rank_saved": MAX_Q,
            "space_setup_seconds": setup["smallest_right_singular_B"],
            "basis_memory_bytes": smallest_b.nbytes,
            "preconditioned_operator_formation_seconds": operator_seconds,
            "trajectory_count": 0,
            "snapshot_count": 0,
            "snapshot_numerical_rank": 0,
        },
        {
            **common,
            "candidate": "late_residual_pod",
            "candidate_definition": CANDIDATE_LABELS["late_residual_pod"],
            "space_rank_saved": MAX_Q,
            "space_setup_seconds": setup["late_residual_pod"],
            "basis_memory_bytes": late_pod.nbytes,
            "preconditioned_operator_formation_seconds": 0.0,
            "trajectory_count": len(TRAIN_RHS_SPECS),
            "snapshot_count": snapshot_matrix.shape[1],
            "snapshot_numerical_rank": pod_numerical_rank,
        },
    ]
    spaces_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        spaces_output / f"{system_id}.npz",
        dominant_right_singular_S=dominant_s,
        smallest_right_singular_B=smallest_b,
        late_residual_pod=late_pod,
    )
    return SpaceArtifact(
        system_id,
        family,
        parameter,
        value,
        system,
        base,
        spaces,
        setup,
        metadata_rows,
        trajectory_rows,
    )


def summarize_own(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for q in Q_VALUES:
            selected = [
                row
                for row in rows
                if row["source_role"] == "own" and row["candidate"] == candidate and row["q"] == q
            ]
            ratios = np.array([row["iteration_ratio"] for row in selected], dtype=float)
            solve_ratios = np.array([row["solve_time_ratio"] for row in selected], dtype=float)
            output.append(
                {
                    "candidate": candidate,
                    "candidate_definition": CANDIDATE_LABELS[candidate],
                    "q": q,
                    "observations": len(selected),
                    "converged": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio": float(np.median(ratios)),
                    "iteration_ratio_q25": float(np.quantile(ratios, 0.25)),
                    "iteration_ratio_q75": float(np.quantile(ratios, 0.75)),
                    "median_solve_time_ratio": float(np.median(solve_ratios)),
                    "solve_time_ratio_q25": float(np.quantile(solve_ratios, 0.25)),
                    "solve_time_ratio_q75": float(np.quantile(solve_ratios, 0.75)),
                    "fraction_fewer_iterations": float(np.mean(ratios < 1.0)),
                    "median_hybrid_iterations": float(
                        np.median([row["hybrid_iterations"] for row in selected])
                    ),
                }
            )
    return output


def summarize_transfer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    own_lookup = {
        (row["target_system_id"], row["candidate"], row["q"], row["rhs_name"]): row
        for row in rows
        if row["source_role"] == "own"
    }
    output: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for q in Q_VALUES:
            selected = [
                row
                for row in rows
                if row["source_role"] == "nearby" and row["candidate"] == candidate and row["q"] == q
            ]
            ratios = np.array([row["iteration_ratio"] for row in selected], dtype=float)
            solve_ratios = np.array([row["solve_time_ratio"] for row in selected], dtype=float)
            degradation = []
            for row in selected:
                own = own_lookup[(row["target_system_id"], candidate, q, row["rhs_name"])]
                degradation.append(row["hybrid_iterations"] / own["hybrid_iterations"])
            output.append(
                {
                    "candidate": candidate,
                    "candidate_definition": CANDIDATE_LABELS[candidate],
                    "q": q,
                    "observations": len(selected),
                    "converged": sum(bool(row["converged"]) for row in selected),
                    "median_iteration_ratio": float(np.median(ratios)),
                    "iteration_ratio_q25": float(np.quantile(ratios, 0.25)),
                    "iteration_ratio_q75": float(np.quantile(ratios, 0.75)),
                    "median_solve_time_ratio": float(np.median(solve_ratios)),
                    "median_transfer_to_own_iteration_ratio": float(np.median(degradation)),
                    "transfer_to_own_iteration_ratio_q75": float(np.quantile(degradation, 0.75)),
                    "fraction_fewer_iterations_than_base": float(np.mean(ratios < 1.0)),
                }
            )
    return output


def plot_summary(
    own_summary: list[dict[str, Any]],
    transfer_summary: list[dict[str, Any]],
    output: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for ax, rows, title in (
        (axes[0], own_summary, "System-own oracle spaces"),
        (axes[1], transfer_summary, "Adjacent-parameter reused spaces"),
    ):
        for candidate in CANDIDATES:
            selected = sorted(
                [row for row in rows if row["candidate"] == candidate],
                key=lambda row: row["q"],
            )
            q = np.array([row["q"] for row in selected], dtype=float)
            median = np.array([row["median_iteration_ratio"] for row in selected], dtype=float)
            low = np.array([row["iteration_ratio_q25"] for row in selected], dtype=float)
            high = np.array([row["iteration_ratio_q75"] for row in selected], dtype=float)
            ax.fill_between(q, low, high, color=COLORS[candidate], alpha=0.12)
            ax.plot(
                q,
                median,
                color=COLORS[candidate],
                marker=MARKERS[candidate],
                linestyle=LINESTYLES[candidate],
                linewidth=1.8,
                markersize=6,
                label=CANDIDATE_LABELS[candidate],
            )
        ax.axhline(1.0, color="#252A31", linestyle=":", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xticks(Q_VALUES)
        ax.set_xlabel("Coarse rank q")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#DCE2E8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median FGMRES iteration ratio vs FP32-LU")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88), frameon=False, ncol=3)
    fig.suptitle("Oracle difficult-space solver comparison", x=0.075, ha="left", fontweight="bold")
    fig.text(
        0.075,
        0.935,
        "Median and interquartile range across held-out RHSs; log ratio scale; lower is better",
        ha="left",
        color="#5E646C",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    path = output / "oracle_target_iteration_ratios.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spaces_output = args.output / "oracle_spaces"

    targets = TARGET_CONFIGS[:1] if args.quick else TARGET_CONFIGS
    evaluation_specs = EVALUATION_RHS_SPECS[:1] if args.quick else EVALUATION_RHS_SPECS
    repeats = 1 if args.quick else REPEATS

    required_keys: list[tuple[str, str, float]] = []
    for target in targets:
        required_keys.append((target.family, target.parameter, target.value))
        for value in target.nearby_values[:1] if args.quick else target.nearby_values:
            required_keys.append((target.family, target.parameter, value))
    unique_keys = list(dict.fromkeys(required_keys))

    artifacts: dict[tuple[str, float], SpaceArtifact] = {}
    space_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for family, parameter, value in unique_keys:
        try:
            artifact = build_space_artifact(
                family,
                parameter,
                value,
                spaces_output=spaces_output,
            )
            artifacts[(family, value)] = artifact
            space_rows.extend(artifact.space_metadata)
            trajectory_rows.extend(artifact.trajectory_metadata)
        except Exception as exc:
            failures.append(
                {
                    "stage": "space_construction",
                    "family": family,
                    "parameter": parameter,
                    "parameter_value": value,
                    "error": repr(exc),
                }
            )

    selection_rows = []
    baseline_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets, start=1):
        target_artifact = artifacts[(target.family, target.value)]
        print(f"[target {target_index}/{len(targets)}] {target_artifact.system_id}", flush=True)
        selection_rows.append(
            {
                **system_fields(
                    target.family,
                    target.parameter,
                    target.value,
                    target_artifact.system_id,
                ),
                "selection_reason": target.selection_reason,
                "nearby_values": ";".join(str(value) for value in target.nearby_values),
            }
        )
        baseline_lookup: dict[str, dict[str, Any]] = {}
        for rhs_name, kx, ky in evaluation_specs:
            rhs = mode_rhs(target_artifact.system, target.family, target.value, kx, ky)
            baseline = benchmark_fgmres(
                target_artifact.system.A,
                rhs,
                target_artifact.base,
                rtol=RTOL,
                restart=min(100, target_artifact.system.dimension),
                max_iterations=min(450, 2 * target_artifact.system.dimension),
                repeats=repeats,
                warmup=True,
            )
            row = {
                **system_fields(
                    target.family,
                    target.parameter,
                    target.value,
                    target_artifact.system_id,
                ),
                "rhs_name": rhs_name,
                "kx": kx,
                "ky": ky,
                "method": "fp32_lu",
                "fgmres_iterations": baseline.result.iterations,
                "converged": baseline.final_true_relative_residual <= RTOL,
                "solve_seconds": baseline.solve_seconds_median,
                "final_true_residual_norm": baseline.final_true_residual_norm,
                "final_true_relative_residual": baseline.final_true_relative_residual,
                "preconditioner_setup_seconds": target_artifact.base.setup_seconds,
                "preconditioner_memory_bytes": target_artifact.base.memory_bytes,
            }
            baseline_rows.append(row)
            baseline_lookup[rhs_name] = row

        source_specs = [("own", target.value)]
        nearby_values = target.nearby_values[:1] if args.quick else target.nearby_values
        source_specs.extend(("nearby", value) for value in nearby_values)
        for source_role, source_value in source_specs:
            source = artifacts[(target.family, source_value)]
            for candidate in CANDIDATES:
                for q in Q_VALUES:
                    Z = source.spaces[candidate][:, :q]
                    try:
                        hybrid = additive_galerkin_coarse_preconditioner(
                            target_artifact.system.A,
                            target_artifact.base,
                            Z,
                            name=f"{candidate}_q{q}_{source_role}",
                        )
                        incremental_setup = hybrid.setup_seconds - target_artifact.base.setup_seconds
                        application_seconds = hybrid.benchmark_application(
                            target_artifact.system.dimension,
                            repeats=50,
                        )
                        for rhs_name, kx, ky in evaluation_specs:
                            rhs = mode_rhs(
                                target_artifact.system,
                                target.family,
                                target.value,
                                kx,
                                ky,
                            )
                            benchmark = benchmark_fgmres(
                                target_artifact.system.A,
                                rhs,
                                hybrid,
                                rtol=RTOL,
                                restart=min(100, target_artifact.system.dimension),
                                max_iterations=min(450, 2 * target_artifact.system.dimension),
                                repeats=repeats,
                                warmup=True,
                            )
                            baseline = baseline_lookup[rhs_name]
                            oracle_rows.append(
                                {
                                    "target_system_id": target_artifact.system_id,
                                    "target_family": target.family,
                                    "target_parameter": target.parameter,
                                    "target_parameter_value": target.value,
                                    "source_system_id": source.system_id,
                                    "source_role": source_role,
                                    "source_parameter_value": source_value,
                                    "parameter_distance": abs(source_value - target.value),
                                    "candidate": candidate,
                                    "candidate_definition": CANDIDATE_LABELS[candidate],
                                    "q": q,
                                    "q_over_n": q / target_artifact.system.dimension,
                                    "rhs_name": rhs_name,
                                    "kx": kx,
                                    "ky": ky,
                                    "base_iterations": int(baseline["fgmres_iterations"]),
                                    "hybrid_iterations": benchmark.result.iterations,
                                    "iteration_ratio": benchmark.result.iterations
                                    / int(baseline["fgmres_iterations"]),
                                    "base_solve_seconds": float(baseline["solve_seconds"]),
                                    "hybrid_solve_seconds": benchmark.solve_seconds_median,
                                    "solve_time_ratio": benchmark.solve_seconds_median
                                    / float(baseline["solve_seconds"]),
                                    "converged": benchmark.final_true_relative_residual <= RTOL,
                                    "final_true_residual_norm": benchmark.final_true_residual_norm,
                                    "final_true_relative_residual": benchmark.final_true_relative_residual,
                                    "space_setup_seconds": source.space_setup_seconds[candidate],
                                    "incremental_coarse_setup_seconds": incremental_setup,
                                    "hybrid_application_seconds": application_seconds,
                                    "hybrid_preconditioner_memory_bytes": hybrid.memory_bytes,
                                    "fgmres_rtol": RTOL,
                                    "fgmres_restart": min(100, target_artifact.system.dimension),
                                    "fgmres_max_iterations": min(
                                        450,
                                        2 * target_artifact.system.dimension,
                                    ),
                                    "timing_repeats": repeats,
                                }
                            )
                    except Exception as exc:
                        failures.append(
                            {
                                "stage": "hybrid_solver",
                                "target_system_id": target_artifact.system_id,
                                "source_system_id": source.system_id,
                                "source_role": source_role,
                                "candidate": candidate,
                                "q": q,
                                "error": repr(exc),
                            }
                        )

    own_summary = summarize_own(oracle_rows)
    transfer_summary = summarize_transfer(oracle_rows)
    write_csv(args.output / "system_selection.csv", selection_rows)
    write_csv(args.output / "space_metadata.csv", space_rows)
    write_csv(args.output / "pod_trajectory_metadata.csv", trajectory_rows)
    write_csv(args.output / "baseline_results.csv", baseline_rows)
    write_csv(args.output / "oracle_results.csv", oracle_rows)
    write_csv(args.output / "own_oracle_summary.csv", own_summary)
    write_csv(args.output / "nearby_transfer_summary.csv", transfer_summary)
    write_csv(args.output / "failures.csv", failures)

    plot_path = plot_summary(own_summary, transfer_summary, args.output)
    write_csv(
        args.output / "chart_map.csv",
        [
            {
                "section": "oracle target selection",
                "question": "Which difficult-space definition most reduces FGMRES work under one correction?",
                "family": "ordered rank comparison with interquartile band",
                "fields": "candidate, q, median_iteration_ratio, q25, q75, source_role",
                "claim": "Compare system-own effectiveness and adjacent-parameter reuse on the same ratio scale.",
                "palette": "three candidate roots plus neutral baseline",
                "path": plot_path.name,
            }
        ],
    )

    config = {
        "experiment": "global RBF oracle difficult-space target selection",
        "primary_preconditioner": "FP32 pivoted LU",
        "definitions": {
            "B": "M^{-1} A",
            "S": "I - M^{-1} A",
            "hybrid": "M^{-1}r + Z solve(Z^T A Z, Z^T r)",
            "residual_pod": (
                "dominant POD modes of the last half of each M-only FGMRES residual trajectory; "
                "residuals scaled by the corresponding RHS 2-norm"
            ),
        },
        "targets": [
            {
                "family": target.family,
                "parameter": target.parameter,
                "value": target.value,
                "nearby_values": target.nearby_values[:1] if args.quick else target.nearby_values,
                "selection_reason": target.selection_reason,
            }
            for target in targets
        ],
        "q_values": Q_VALUES,
        "training_rhs": TRAIN_RHS_SPECS,
        "evaluation_rhs": evaluation_specs,
        "evaluation_rhs_are_held_out_from_pod": True,
        "fgmres_rtol": RTOL,
        "fgmres_restart": min(100, N),
        "fgmres_max_iterations": min(450, 2 * N),
        "timing_repeats": repeats,
        "warmup_solves": 1,
        "quick": args.quick,
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
    print(
        f"Wrote {len(oracle_rows)} hybrid rows and {len(baseline_rows)} baseline rows to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
