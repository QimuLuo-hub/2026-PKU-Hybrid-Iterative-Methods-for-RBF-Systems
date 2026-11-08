"""Run Stage 2: transferability of global-RBF difficult subspaces.

This experiment is deliberately non-ML.  It holds the coefficient-space
dimension fixed, constructs one-parameter Gaussian system families, and tests
the q=24/32/40 smallest-right-singular subspaces geometrically and in FGMRES.
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
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import scipy
import scipy.linalg as la
import scipy.stats as stats

from global_rbf.collocation import (
    GlobalRBFSystem,
    assemble_poisson_system,
    assemble_variable_diffusion_system,
    variable_diffusion_forcing,
)
from global_rbf.krylov import FGMRESResult, fgmres
from global_rbf.nodes import square_nodes
from global_rbf.preconditioners import Preconditioner, low_precision_lu_preconditioner
from global_rbf.structure import oracle_coarse_preconditioner
from global_rbf.transfer import (
    compare_subspaces,
    grassmann_geodesic,
    orthonormalize,
    smallest_right_singular_spaces,
)


SEED = 20260830
SIDE = 15
N = SIDE**2
BASE_EPSILON = 6.0
Q_VALUES = (24, 32, 40)
RTOL = 1.0e-10
RHS_SPECS = (("mode_1_1", 1, 1), ("mode_1_2", 1, 2), ("mode_2_1", 2, 1), ("mode_2_2", 2, 2))
PALETTE = {
    "blue": "#1F5A94",
    "gold": "#C58C24",
    "orange": "#D06B32",
    "olive": "#708238",
    "pink": "#B4557D",
    "ink": "#252A31",
    "grey": "#8A9099",
    "light": "#DCE2E8",
}


@dataclass(frozen=True)
class FamilySpec:
    family: str
    parameter: str
    values: tuple[float, ...]


@dataclass
class SystemArtifact:
    family: str
    parameter: str
    index: int
    value: float
    system_id: str
    system: GlobalRBFSystem
    base: Preconditioner
    spaces: dict[int, np.ndarray]
    singular_values: np.ndarray
    svd_seconds: float
    characterization: dict[str, Any]


def family_specs(quick: bool) -> tuple[FamilySpec, ...]:
    if quick:
        return (
            FamilySpec("epsilon", "epsilon", (5.5, 6.0, 6.5)),
            FamilySpec("jitter", "jitter_fraction", (0.0, 0.11, 0.22)),
            FamilySpec("pde_coefficient", "beta", (0.0, 0.5, 1.0)),
        )
    return (
        FamilySpec("epsilon", "epsilon", tuple(np.linspace(5.0, 7.0, 9))),
        FamilySpec("jitter", "jitter_fraction", tuple(np.linspace(0.0, 0.22, 9))),
        FamilySpec("pde_coefficient", "beta", tuple(np.linspace(0.0, 1.0, 9))),
    )


def assemble_family_member(spec: FamilySpec, index: int, value: float) -> tuple[str, GlobalRBFSystem]:
    if spec.family == "epsilon":
        nodes, boundary = square_nodes(SIDE, "tensor")
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=value)
    elif spec.family == "jitter":
        nodes, boundary = square_nodes(
            SIDE,
            "jittered",
            jitter_fraction=value,
            seed=SEED + SIDE,
        )
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=BASE_EPSILON)
    elif spec.family == "pde_coefficient":
        nodes, boundary = square_nodes(SIDE, "tensor")
        system = assemble_variable_diffusion_system(
            nodes,
            boundary,
            beta=value,
            kernel="gaussian",
            epsilon=BASE_EPSILON,
        )
    else:
        raise ValueError(spec.family)
    system_id = f"{spec.family}_N{N}_{spec.parameter}{value:g}"
    return system_id, system


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mode_rhs(artifact: SystemArtifact, kx: int, ky: int) -> np.ndarray:
    system = artifact.system
    x = system.nodes
    u = np.sin(kx * np.pi * x[:, 0]) * np.sin(ky * np.pi * x[:, 1])
    if artifact.family == "pde_coefficient":
        forcing = variable_diffusion_forcing(x, artifact.value, kx=kx, ky=ky)
    else:
        forcing = (kx**2 + ky**2) * np.pi**2 * u
    return np.where(system.boundary, u, forcing)


def run_solver(
    artifact: SystemArtifact,
    preconditioner: Preconditioner,
    rhs: np.ndarray,
    *,
    repeats: int,
) -> tuple[FGMRESResult, float]:
    # One unrecorded warm-up limits first-call effects in sub-millisecond runs.
    fgmres(
        artifact.system.A,
        rhs,
        preconditioner.apply,
        rtol=RTOL,
        restart=min(100, N),
        max_iterations=min(450, 2 * N),
    )
    results = [
        fgmres(
            artifact.system.A,
            rhs,
            preconditioner.apply,
            rtol=RTOL,
            restart=min(100, N),
            max_iterations=min(450, 2 * N),
        )
        for _ in range(repeats)
    ]
    return results[0], float(np.median([result.solve_seconds for result in results]))


def build_system_artifacts(
    specs: tuple[FamilySpec, ...],
) -> tuple[list[SystemArtifact], list[dict[str, Any]]]:
    artifacts: list[SystemArtifact] = []
    rows: list[dict[str, Any]] = []
    total = sum(len(spec.values) for spec in specs)
    counter = 0
    for spec in specs:
        for index, value in enumerate(spec.values):
            counter += 1
            system_id, system = assemble_family_member(spec, index, value)
            print(f"[systems {counter}/{total}] {system_id}", flush=True)
            spaces, singular_values, svd_seconds = smallest_right_singular_spaces(system.A, Q_VALUES)
            base = low_precision_lu_preconditioner(system.A)
            lu = la.lu_factor(system.A, check_finite=False)
            solution = la.lu_solve(lu, system.b, check_finite=False)
            errors = system.error_metrics(solution, evaluation_side=41)
            condition = float(singular_values[0] / singular_values[-1])
            characterization = {
                "system_id": system_id,
                "family": spec.family,
                "parameter": spec.parameter,
                "parameter_index": index,
                "parameter_value": value,
                "N": N,
                "epsilon": value if spec.family == "epsilon" else BASE_EPSILON,
                "jitter_fraction": value if spec.family == "jitter" else 0.0,
                "beta": value if spec.family == "pde_coefficient" else 0.0,
                "coefficient_contrast": float(np.exp(value)) if spec.family == "pde_coefficient" else 1.0,
                "condition_2": condition,
                "sigma_max": float(singular_values[0]),
                "sigma_min": float(singular_values[-1]),
                "roundoff_exposure_kappa_u": condition * np.finfo(float).eps,
                "assembly_seconds": system.assembly_seconds,
                "svd_seconds": svd_seconds,
                "fp32_lu_setup_seconds": base.setup_seconds,
                "matrix_memory_bytes": system.A.nbytes,
                **errors,
            }
            for q in Q_VALUES:
                excluded = singular_values[-q - 1]
                included = singular_values[-q]
                characterization[f"boundary_gap_ratio_q{q}"] = float(excluded / included)
                characterization[f"boundary_relative_gap_q{q}"] = float((excluded - included) / excluded)
            artifact = SystemArtifact(
                spec.family,
                spec.parameter,
                index,
                value,
                system_id,
                system,
                base,
                spaces,
                singular_values,
                svd_seconds,
                characterization,
            )
            artifacts.append(artifact)
            rows.append(characterization)
    return artifacts, rows


def family_groups(artifacts: list[SystemArtifact]) -> dict[str, list[SystemArtifact]]:
    groups: dict[str, list[SystemArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(artifact.family, []).append(artifact)
    for values in groups.values():
        values.sort(key=lambda artifact: artifact.value)
    return groups


def pairwise_geometry(
    groups: dict[str, list[SystemArtifact]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    for family, members in groups.items():
        span = members[-1].value - members[0].value
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                raw_distance = abs(right.value - left.value)
                normalized_parameter_distance = raw_distance / span if span else 0.0
                adjacent = right.index == left.index + 1
                for q in Q_VALUES:
                    comparison = compare_subspaces(left.spaces[q], right.spaces[q])
                    pair_id = f"{left.system_id}__{right.system_id}__q{q}"
                    pair_rows.append(
                        {
                            "pair_id": pair_id,
                            "family": family,
                            "parameter": left.parameter,
                            "left_system_id": left.system_id,
                            "right_system_id": right.system_id,
                            "left_value": left.value,
                            "right_value": right.value,
                            "parameter_distance": raw_distance,
                            "normalized_parameter_distance": normalized_parameter_distance,
                            "index_distance": right.index - left.index,
                            "adjacent": adjacent,
                            "q": q,
                            "q_over_n": q / N,
                            "projector_distance": comparison.projector_distance,
                            "normalized_projector_distance": comparison.normalized_projector_distance,
                            "principal_angle_max_degrees": comparison.max_angle_degrees,
                            "principal_angle_median_degrees": comparison.median_angle_degrees,
                            "principal_angle_rms_degrees": comparison.rms_angle_degrees,
                        }
                    )
                    for angle_index, angle in enumerate(np.sort(comparison.principal_angles_radians)):
                        angle_rows.append(
                            {
                                "pair_id": pair_id,
                                "family": family,
                                "q": q,
                                "angle_index": angle_index + 1,
                                "principal_angle_degrees": float(np.degrees(angle)),
                            }
                        )
    return pair_rows, angle_rows


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def correlation_summary(
    groups: dict[str, list[SystemArtifact]],
    pair_rows: list[dict[str, Any]],
    *,
    bootstrap_reps: int,
    permutation_reps: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 71)
    output: list[dict[str, Any]] = []
    for family, members in groups.items():
        for q in Q_VALUES:
            rows = [row for row in pair_rows if row["family"] == family and row["q"] == q]
            x = np.array([row["normalized_parameter_distance"] for row in rows])
            y = np.array([row["normalized_projector_distance"] for row in rows])
            spearman = safe_spearman(x, y)
            pearson = float(stats.pearsonr(x, y).statistic)
            bootstrap = []
            for _ in range(bootstrap_reps):
                indices = rng.integers(0, len(rows), size=len(rows))
                value = safe_spearman(x[indices], y[indices])
                if np.isfinite(value):
                    bootstrap.append(value)

            parameter_matrix = np.abs(
                np.subtract.outer(
                    np.array([member.value for member in members]),
                    np.array([member.value for member in members]),
                )
            )
            distance_matrix = np.zeros_like(parameter_matrix)
            lookup = {
                tuple(sorted((row["left_system_id"], row["right_system_id"]))): row["normalized_projector_distance"]
                for row in rows
            }
            for i, left in enumerate(members):
                for j, right in enumerate(members):
                    if i != j:
                        distance_matrix[i, j] = lookup[tuple(sorted((left.system_id, right.system_id)))]
            triangle = np.triu_indices(len(members), 1)
            permuted_correlations = []
            for _ in range(permutation_reps):
                permutation = rng.permutation(len(members))
                permuted = distance_matrix[np.ix_(permutation, permutation)]
                permuted_correlations.append(safe_spearman(parameter_matrix[triangle], permuted[triangle]))
            permuted_correlations = np.array(permuted_correlations)
            permutation_p = float(
                (1 + np.count_nonzero(np.abs(permuted_correlations) >= abs(spearman)))
                / (permutation_reps + 1)
            )
            adjacent = y[np.array([bool(row["adjacent"]) for row in rows])]
            far = y[np.array([int(row["index_distance"]) == len(members) - 1 for row in rows])]
            output.append(
                {
                    "family": family,
                    "q": q,
                    "pair_count": len(rows),
                    "spearman_rho": spearman,
                    "spearman_pair_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "spearman_pair_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "pearson_r": pearson,
                    "mantel_label_permutation_p_two_sided": permutation_p,
                    "adjacent_distance_median": float(np.median(adjacent)),
                    "adjacent_distance_q90": float(np.quantile(adjacent, 0.9)),
                    "endpoint_distance": float(far[0]),
                }
            )
    return output


def numerical_variability(
    groups: dict[str, list[SystemArtifact]],
    *,
    perturbation_reps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + 83)
    for family, members in groups.items():
        reference = members[len(members) // 2]
        cross_spaces, _, _ = smallest_right_singular_spaces(
            reference.system.A,
            Q_VALUES,
            lapack_driver="gesvd",
        )
        for q in Q_VALUES:
            comparison = compare_subspaces(reference.spaces[q], cross_spaces[q])
            rows.append(
                {
                    "family": family,
                    "system_id": reference.system_id,
                    "q": q,
                    "perturbation_type": "lapack_driver_crosscheck",
                    "relative_frobenius_perturbation": 0.0,
                    "replicate": 0,
                    "normalized_projector_distance": comparison.normalized_projector_distance,
                    "projector_distance": comparison.projector_distance,
                    "principal_angle_max_degrees": comparison.max_angle_degrees,
                    "principal_angle_median_degrees": comparison.median_angle_degrees,
                }
            )
        for relative_scale in (1.0e-14, 1.0e-12):
            for replicate in range(perturbation_reps):
                noise = rng.standard_normal(reference.system.A.shape)
                noise *= relative_scale * np.linalg.norm(reference.system.A, ord="fro") / np.linalg.norm(noise, ord="fro")
                perturbed_spaces, _, _ = smallest_right_singular_spaces(reference.system.A + noise, Q_VALUES)
                for q in Q_VALUES:
                    comparison = compare_subspaces(reference.spaces[q], perturbed_spaces[q])
                    rows.append(
                        {
                            "family": family,
                            "system_id": reference.system_id,
                            "q": q,
                            "perturbation_type": "dense_backward_perturbation",
                            "relative_frobenius_perturbation": relative_scale,
                            "replicate": replicate,
                            "normalized_projector_distance": comparison.normalized_projector_distance,
                            "projector_distance": comparison.projector_distance,
                            "principal_angle_max_degrees": comparison.max_angle_degrees,
                            "principal_angle_median_degrees": comparison.median_angle_degrees,
                        }
                    )
    return rows


def log_histogram_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Common histogram mass on a log-distance scale.

    A linear [0, 1] grid hides the many-order separation between SVD backward
    variability and genuine parameter changes, so overlap is measured after a
    documented log10 transform.  The common-language and KS measures below
    remain bin-free companions.
    """

    floor = 10.0 * np.finfo(float).eps
    left_log = np.log10(np.maximum(left, floor))
    right_log = np.log10(np.maximum(right, floor))
    lower = min(np.min(left_log), np.min(right_log))
    upper = max(np.max(left_log), np.max(right_log))
    if upper == lower:
        return 1.0
    bins = np.linspace(lower, upper, 61)
    left_counts, _ = np.histogram(left_log, bins=bins)
    right_counts, _ = np.histogram(right_log, bins=bins)
    left_mass = left_counts / max(1, np.sum(left_counts))
    right_mass = right_counts / max(1, np.sum(right_counts))
    return float(np.sum(np.minimum(left_mass, right_mass)))


def variability_overlap_summary(
    groups: dict[str, list[SystemArtifact]],
    pair_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in groups:
        for q in Q_VALUES:
            within = np.array(
                [
                    row["normalized_projector_distance"]
                    for row in variability_rows
                    if row["family"] == family
                    and row["q"] == q
                    and row["perturbation_type"] == "dense_backward_perturbation"
                    and row["relative_frobenius_perturbation"] == 1.0e-12
                ]
            )
            adjacent = np.array(
                [
                    row["normalized_projector_distance"]
                    for row in pair_rows
                    if row["family"] == family and row["q"] == q and row["adjacent"]
                ]
            )
            ks = stats.ks_2samp(within, adjacent, alternative="two-sided", method="auto")
            common_language = float(np.mean(adjacent[:, None] > within[None, :]))
            rows.append(
                {
                    "family": family,
                    "q": q,
                    "within_count": len(within),
                    "between_adjacent_count": len(adjacent),
                    "within_median": float(np.median(within)),
                    "within_q90": float(np.quantile(within, 0.9)),
                    "between_adjacent_median": float(np.median(adjacent)),
                    "between_adjacent_q10": float(np.quantile(adjacent, 0.1)),
                    "log_histogram_overlap_coefficient": log_histogram_overlap(within, adjacent),
                    "probability_between_exceeds_within": common_language,
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                }
            )
    return rows


def random_space(n: int, q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return orthonormalize(rng.standard_normal((n, q)))


def reuse_solver_study(
    groups: dict[str, list[SystemArtifact]],
    *,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, members in groups.items():
        for target_position, target in enumerate(members):
            print(f"[reuse] {target.system_id}", flush=True)
            if target_position == 0:
                nearby = members[1]
            elif target_position == len(members) - 1:
                nearby = members[-2]
            else:
                nearby = members[target_position - 1]
            far = max(members, key=lambda member: abs(member.value - target.value))
            rhs_map = {
                rhs_name: mode_rhs(target, kx, ky)
                for rhs_name, kx, ky in RHS_SPECS
            }
            for rhs_name, rhs in rhs_map.items():
                result, solve_seconds = run_solver(target, target.base, rhs, repeats=repeats)
                rows.append(
                    {
                        "family": family,
                        "target_system_id": target.system_id,
                        "target_parameter_value": target.value,
                        "source_type": "classical_only",
                        "source_system_id": target.system_id,
                        "source_parameter_value": target.value,
                        "source_parameter_distance": 0.0,
                        "q": 0,
                        "rhs": rhs_name,
                        "gmres_iterations": result.iterations,
                        "converged": result.converged,
                        "relative_residual": result.relative_residual,
                        "solve_seconds_median": solve_seconds,
                        "basis_setup_seconds": 0.0,
                        "target_svd_seconds_avoided": 0.0,
                        "coarse_setup_seconds": 0.0,
                        "coarse_memory_bytes": 0,
                        "total_seconds_available_basis": solve_seconds,
                        "total_seconds_system_own_first_rhs": solve_seconds,
                    }
                )
            for q in Q_VALUES:
                source_specs = (
                    ("own_oracle", target, target.spaces[q], target.svd_seconds),
                    ("nearby_reuse", nearby, nearby.spaces[q], nearby.svd_seconds),
                    ("far_reuse", far, far.spaces[q], far.svd_seconds),
                    (
                        "random",
                        target,
                        random_space(N, q, SEED + 1000 * target.index + 10 * q + len(family)),
                        0.0,
                    ),
                )
                for source_type, source, basis, basis_setup in source_specs:
                    subspace_comparison = compare_subspaces(target.spaces[q], basis)
                    hybrid = oracle_coarse_preconditioner(
                        target.system.A,
                        target.base,
                        basis,
                        name=f"{source_type}_{source.system_id}_q{q}",
                        composition="residual_complement",
                    )
                    coarse_setup = hybrid.setup_seconds - target.base.setup_seconds
                    coarse_memory = hybrid.memory_bytes - target.base.memory_bytes
                    for rhs_name, rhs in rhs_map.items():
                        result, solve_seconds = run_solver(target, hybrid, rhs, repeats=repeats)
                        rows.append(
                            {
                                "family": family,
                                "target_system_id": target.system_id,
                                "target_parameter_value": target.value,
                                "source_type": source_type,
                                "source_system_id": source.system_id if source_type != "random" else "haar_random",
                                "source_parameter_value": source.value if source_type != "random" else float("nan"),
                                "source_parameter_distance": abs(source.value - target.value) if source_type != "random" else float("nan"),
                                "q": q,
                                "source_target_normalized_projector_distance": subspace_comparison.normalized_projector_distance,
                                "source_target_projector_distance": subspace_comparison.projector_distance,
                                "source_target_max_angle_degrees": subspace_comparison.max_angle_degrees,
                                "source_target_median_angle_degrees": subspace_comparison.median_angle_degrees,
                                "rhs": rhs_name,
                                "gmres_iterations": result.iterations,
                                "converged": result.converged,
                                "relative_residual": result.relative_residual,
                                "solve_seconds_median": solve_seconds,
                                "basis_setup_seconds": basis_setup,
                                "target_svd_seconds_avoided": target.svd_seconds if source_type != "own_oracle" else 0.0,
                                "coarse_setup_seconds": coarse_setup,
                                "coarse_memory_bytes": coarse_memory,
                                "total_seconds_available_basis": coarse_setup + solve_seconds,
                                "total_seconds_system_own_first_rhs": basis_setup + coarse_setup + solve_seconds,
                            }
                        )
    return rows


def interpolation_study(
    groups: dict[str, list[SystemArtifact]],
    *,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    geometry_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for family, members in groups.items():
        for left, target, right in zip(members[:-2], members[1:-1], members[2:]):
            t = (target.value - left.value) / (right.value - left.value)
            print(
                f"[interpolation] {family}: {left.value:g} -> {target.value:g} -> {right.value:g}",
                flush=True,
            )
            rhs_map = {name: mode_rhs(target, kx, ky) for name, kx, ky in RHS_SPECS}
            for q in Q_VALUES:
                start = perf_counter()
                interpolated = grassmann_geodesic(left.spaces[q], right.spaces[q], t)
                interpolation_seconds = perf_counter() - start
                own_comparison = compare_subspaces(interpolated, target.spaces[q])
                left_comparison = compare_subspaces(left.spaces[q], target.spaces[q])
                right_comparison = compare_subspaces(right.spaces[q], target.spaces[q])
                endpoint_comparison = compare_subspaces(left.spaces[q], right.spaces[q])
                nearest_endpoint_distance = min(
                    left_comparison.normalized_projector_distance,
                    right_comparison.normalized_projector_distance,
                )
                geometry_rows.append(
                    {
                        "family": family,
                        "left_system_id": left.system_id,
                        "target_system_id": target.system_id,
                        "right_system_id": right.system_id,
                        "left_value": left.value,
                        "target_value": target.value,
                        "right_value": right.value,
                        "interpolation_fraction": t,
                        "q": q,
                        "interpolation_seconds": interpolation_seconds,
                        "interpolated_normalized_projector_distance": own_comparison.normalized_projector_distance,
                        "interpolated_projector_distance": own_comparison.projector_distance,
                        "interpolated_max_angle_degrees": own_comparison.max_angle_degrees,
                        "nearest_endpoint_normalized_projector_distance": nearest_endpoint_distance,
                        "geometry_ratio_vs_nearest_endpoint": own_comparison.normalized_projector_distance / nearest_endpoint_distance,
                        "endpoint_max_angle_degrees": endpoint_comparison.max_angle_degrees,
                    }
                )
                hybrid = oracle_coarse_preconditioner(
                    target.system.A,
                    target.base,
                    interpolated,
                    name=f"grassmann_interpolated_q{q}",
                    composition="residual_complement",
                )
                coarse_setup = hybrid.setup_seconds - target.base.setup_seconds
                for rhs_name, rhs in rhs_map.items():
                    result, solve_seconds = run_solver(target, hybrid, rhs, repeats=repeats)
                    solver_rows.append(
                        {
                            "family": family,
                            "left_system_id": left.system_id,
                            "target_system_id": target.system_id,
                            "right_system_id": right.system_id,
                            "target_parameter_value": target.value,
                            "q": q,
                            "rhs": rhs_name,
                            "gmres_iterations": result.iterations,
                            "converged": result.converged,
                            "relative_residual": result.relative_residual,
                            "solve_seconds_median": solve_seconds,
                            "interpolation_seconds": interpolation_seconds,
                            "coarse_setup_seconds": coarse_setup,
                            "total_seconds_available_endpoints": interpolation_seconds + coarse_setup + solve_seconds,
                        }
                    )
    return geometry_rows, solver_rows


def reuse_summary(reuse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["family"], row["target_system_id"], row["q"], row["rhs"], row["source_type"]): row
        for row in reuse_rows
        if row["source_type"] != "classical_only"
    }
    base_lookup = {
        (row["family"], row["target_system_id"], row["rhs"]): row
        for row in reuse_rows
        if row["source_type"] == "classical_only"
    }
    summary: list[dict[str, Any]] = []
    families = sorted({row["family"] for row in reuse_rows})
    for family in families:
        targets = sorted({row["target_system_id"] for row in reuse_rows if row["family"] == family})
        for q in Q_VALUES:
            for source_type in ("own_oracle", "nearby_reuse", "far_reuse", "random"):
                selected = [
                    lookup[(family, target, q, rhs, source_type)]
                    for target in targets
                    for rhs, _, _ in RHS_SPECS
                ]
                own = [
                    lookup[(family, target, q, rhs, "own_oracle")]
                    for target in targets
                    for rhs, _, _ in RHS_SPECS
                ]
                base = [
                    base_lookup[(family, target, rhs)]
                    for target in targets
                    for rhs, _, _ in RHS_SPECS
                ]
                iteration_ratio_own = np.array([row["gmres_iterations"] for row in selected]) / np.array(
                    [row["gmres_iterations"] for row in own]
                )
                iteration_ratio_base = np.array([row["gmres_iterations"] for row in selected]) / np.array(
                    [row["gmres_iterations"] for row in base]
                )
                solve_ratio_own = np.array([row["solve_seconds_median"] for row in selected]) / np.array(
                    [row["solve_seconds_median"] for row in own]
                )
                summary.append(
                    {
                        "family": family,
                        "q": q,
                        "source_type": source_type,
                        "comparison_count": len(selected),
                        "converged_fraction": float(np.mean([row["converged"] for row in selected])),
                        "median_iterations": float(np.median([row["gmres_iterations"] for row in selected])),
                        "median_iteration_ratio_vs_own": float(np.median(iteration_ratio_own)),
                        "q90_iteration_ratio_vs_own": float(np.quantile(iteration_ratio_own, 0.9)),
                        "median_iteration_ratio_vs_classical": float(np.median(iteration_ratio_base)),
                        "fraction_fewer_iterations_than_classical": float(np.mean(iteration_ratio_base < 1.0)),
                        "median_solve_ratio_vs_own": float(np.median(solve_ratio_own)),
                        "median_available_basis_total_seconds": float(
                            np.median([row["total_seconds_available_basis"] for row in selected])
                        ),
                    }
                )
    return summary


def geometry_solver_correlation(reuse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Relate source-target geometry to solver degradation at target grain."""

    rows: list[dict[str, Any]] = []
    families = sorted({row["family"] for row in reuse_rows})
    for family in families:
        for q in Q_VALUES:
            for source_scope, source_types in (
                ("structured_reuse", {"nearby_reuse", "far_reuse"}),
                ("all_nonoracle_spaces", {"nearby_reuse", "far_reuse", "random"}),
            ):
                candidates = [
                    row
                    for row in reuse_rows
                    if row["family"] == family and row["q"] == q and row["source_type"] in source_types
                ]
                grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
                for row in candidates:
                    grouped.setdefault((row["target_system_id"], row["source_type"]), []).append(row)
                distances = []
                max_angles = []
                iteration_ratios = []
                for (target_id, _), values in grouped.items():
                    own = [
                        row
                        for row in reuse_rows
                        if row["family"] == family
                        and row["target_system_id"] == target_id
                        and row["q"] == q
                        and row["source_type"] == "own_oracle"
                    ]
                    distances.append(values[0]["source_target_normalized_projector_distance"])
                    max_angles.append(values[0]["source_target_max_angle_degrees"])
                    iteration_ratios.append(
                        np.median([row["gmres_iterations"] for row in values])
                        / np.median([row["gmres_iterations"] for row in own])
                    )
                distances = np.asarray(distances)
                max_angles = np.asarray(max_angles)
                iteration_ratios = np.asarray(iteration_ratios)
                rows.append(
                    {
                        "family": family,
                        "q": q,
                        "source_scope": source_scope,
                        "target_source_count": len(distances),
                        "spearman_projector_distance_vs_iteration_degradation": safe_spearman(
                            distances, iteration_ratios
                        ),
                        "spearman_max_angle_vs_iteration_degradation": safe_spearman(
                            max_angles, iteration_ratios
                        ),
                        "median_projector_distance": float(np.median(distances)),
                        "median_max_angle_degrees": float(np.median(max_angles)),
                        "median_iteration_degradation": float(np.median(iteration_ratios)),
                    }
                )
    return rows


def interpolation_summary(
    interpolation_rows: list[dict[str, Any]],
    reuse_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    own_lookup = {
        (row["family"], row["target_system_id"], row["q"], row["rhs"]): row
        for row in reuse_rows
        if row["source_type"] == "own_oracle"
    }
    nearby_lookup = {
        (row["family"], row["target_system_id"], row["q"], row["rhs"]): row
        for row in reuse_rows
        if row["source_type"] == "nearby_reuse"
    }
    rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in interpolation_rows}):
        for q in Q_VALUES:
            selected = [row for row in interpolation_rows if row["family"] == family and row["q"] == q]
            own = [own_lookup[(family, row["target_system_id"], q, row["rhs"])] for row in selected]
            nearby = [nearby_lookup[(family, row["target_system_id"], q, row["rhs"])] for row in selected]
            interpolation_iterations = np.array([row["gmres_iterations"] for row in selected])
            own_iterations = np.array([row["gmres_iterations"] for row in own])
            nearby_iterations = np.array([row["gmres_iterations"] for row in nearby])
            rows.append(
                {
                    "family": family,
                    "q": q,
                    "comparison_count": len(selected),
                    "converged_fraction": float(np.mean([row["converged"] for row in selected])),
                    "median_iterations": float(np.median(interpolation_iterations)),
                    "median_iteration_ratio_vs_own": float(np.median(interpolation_iterations / own_iterations)),
                    "median_iteration_ratio_vs_nearby": float(np.median(interpolation_iterations / nearby_iterations)),
                    "fraction_better_than_nearby": float(np.mean(interpolation_iterations < nearby_iterations)),
                    "median_solve_seconds": float(np.median([row["solve_seconds_median"] for row in selected])),
                }
            )
    return rows


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["light"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_blossom(fig: plt.Figure) -> None:
    center_x, center_y = 0.978, 0.975
    for dx, dy in ((-0.007, 0), (0.007, 0), (0, -0.010), (0, 0.010)):
        fig.add_artist(
            Circle(
                (center_x + dx, center_y + dy),
                0.0045,
                transform=fig.transFigure,
                color=PALETTE["blue"],
                alpha=0.85,
            )
        )


def finish_figure(fig: plt.Figure, path: Path) -> None:
    add_blossom(fig)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_results(
    output: Path,
    groups: dict[str, list[SystemArtifact]],
    pair_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    reuse_rows: list[dict[str, Any]],
    reuse_summaries: list[dict[str, Any]],
    interpolation_geometry: list[dict[str, Any]],
    interpolation_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    setup_plot_style()
    chart_map: list[dict[str, Any]] = []
    family_order = ("epsilon", "jitter", "pde_coefficient")
    family_titles = {
        "epsilon": "Gaussian shape ε",
        "jitter": "Frozen node-jitter path",
        "pde_coefficient": "Diffusion coefficient β",
    }
    q_styles = {
        24: (PALETTE["gold"], "o"),
        32: (PALETTE["blue"], "s"),
        40: (PALETTE["pink"], "D"),
    }

    fig, axes = plt.subplots(3, 3, figsize=(11.2, 9.0), sharex=True, sharey=True)
    for row_index, q in enumerate(Q_VALUES):
        for column_index, family in enumerate(family_order):
            ax = axes[row_index, column_index]
            selected = [row for row in pair_rows if row["family"] == family and row["q"] == q]
            color, marker = q_styles[q]
            ax.scatter(
                [row["normalized_parameter_distance"] for row in selected],
                [row["normalized_projector_distance"] for row in selected],
                color=color,
                marker=marker,
                s=32,
                facecolor="white" if q != 32 else color,
                linewidth=1.1,
            )
            correlation = next(row for row in correlations if row["family"] == family and row["q"] == q)
            ax.text(
                0.04,
                0.93,
                f"Spearman ρ={correlation['spearman_rho']:.2f}",
                transform=ax.transAxes,
                va="top",
                color=PALETTE["ink"],
            )
            if row_index == 0:
                ax.set_title(family_titles[family])
            if row_index == 2:
                ax.set_xlabel("Normalized parameter distance")
            if column_index == 0:
                ax.set_ylabel(f"q={q}\nNormalized projector distance")
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.03, 1.03)
    fig.suptitle("Difficult-space distance versus parameter distance", x=0.06, ha="left", fontweight="bold")
    fig.text(
        0.06,
        0.955,
        "All unordered pairs at N=225; projector distance divided by √(2q)",
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = output / "subspace_distance_vs_parameter_distance.png"
    finish_figure(fig, path)
    chart_map.append(
        {
            "section": "geometric transfer",
            "question": "Do nearby parameters have closer difficult spaces?",
            "family": "faceted relationship",
            "fields": "normalized parameter distance, normalized projector distance, family, q",
            "claim": "Correlation strength and regularity differ by parameter family and q.",
            "palette": "single-root per q",
            "path": path.name,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    for ax, family in zip(axes, family_order):
        members = groups[family]
        for q in Q_VALUES:
            selected = [
                row
                for row in pair_rows
                if row["family"] == family and row["q"] == q and row["adjacent"]
            ]
            color, marker = q_styles[q]
            midpoints = [(row["left_value"] + row["right_value"]) / 2.0 for row in selected]
            ax.plot(
                midpoints,
                [row["principal_angle_max_degrees"] for row in selected],
                color=color,
                marker=marker,
                markerfacecolor="white" if q != 32 else color,
                label=f"q={q}",
            )
        ax.set_title(family_titles[family])
        ax.set_xlabel(f"Adjacent-pair midpoint ({members[0].parameter})")
    axes[0].set_ylabel("Largest principal angle (degrees)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Principal-angle variation between adjacent systems", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.90, "Largest angle in each q-dimensional pair; smaller is closer", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 0.92, 0.85])
    path = output / "principal_angle_trends.png"
    finish_figure(fig, path)
    chart_map.append(
        {
            "section": "geometric transfer",
            "question": "Are adjacent-space angles smooth along each path?",
            "family": "small-multiple ordered comparison",
            "fields": "adjacent midpoint, largest principal angle, family, q",
            "claim": "Angle regularity identifies whether transfer is uniformly local or path dependent.",
            "palette": "three-root categorical",
            "path": path.name,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, family in zip(axes, family_order):
        positions = []
        values = []
        labels = []
        position = 1
        for q in Q_VALUES:
            within = [
                row["normalized_projector_distance"]
                for row in variability_rows
                if row["family"] == family
                and row["q"] == q
                and row["perturbation_type"] == "dense_backward_perturbation"
                and row["relative_frobenius_perturbation"] == 1.0e-12
            ]
            adjacent = [
                row["normalized_projector_distance"]
                for row in pair_rows
                if row["family"] == family and row["q"] == q and row["adjacent"]
            ]
            values.extend([within, adjacent])
            positions.extend([position, position + 0.8])
            labels.extend([f"q{q}\nwithin", f"q{q}\nadjacent"])
            position += 2.4
        boxes = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
        for box_index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(PALETTE["light"] if box_index % 2 == 0 else PALETTE["blue"])
            box.set_edgecolor(PALETTE["ink"])
            box.set_alpha(0.9)
        ax.set_xticks(positions, labels, fontsize=7)
        ax.set_title(family_titles[family])
    axes[0].set_ylabel("Normalized projector distance")
    fig.suptitle("Reference variability versus adjacent-system variation", x=0.06, ha="left", fontweight="bold")
    fig.text(
        0.06,
        0.90,
        "Within: 12 dense backward perturbations of relative Frobenius size 10⁻¹²",
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    path = output / "within_between_variability.png"
    finish_figure(fig, path)
    chart_map.append(
        {
            "section": "reference stability",
            "question": "Is cross-system variation larger than numerical SVD variability?",
            "family": "distribution comparison",
            "fields": "normalized projector distance, source of variation, family, q",
            "claim": "Within-system backward perturbations provide the noise floor for between-system comparisons.",
            "palette": "hard two-root cap",
            "path": path.name,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    source_styles = {
        "own_oracle": (PALETTE["ink"], "o"),
        "nearby_reuse": (PALETTE["blue"], "s"),
        "far_reuse": (PALETTE["orange"], "D"),
        "random": (PALETTE["grey"], "^")
    }
    for ax, family in zip(axes, family_order):
        for source_type, (color, marker) in source_styles.items():
            selected = [
                row for row in reuse_summaries if row["family"] == family and row["source_type"] == source_type
            ]
            ax.plot(
                [row["q"] for row in selected],
                [row["median_iteration_ratio_vs_own"] for row in selected],
                color=color,
                marker=marker,
                markerfacecolor="white" if source_type != "nearby_reuse" else color,
                label=source_type.replace("_", " "),
            )
        ax.axhline(1.0, color=PALETTE["ink"], linestyle=":", linewidth=0.9)
        ax.set_title(family_titles[family])
        ax.set_xlabel("Difficult-space dimension q")
        ax.set_xticks(Q_VALUES)
    axes[0].set_ylabel("Median iteration ratio versus own oracle")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("FGMRES degradation under difficult-space reuse", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.90, "Four sine-mode right-hand sides per target; lower is better", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 0.86, 0.84])
    path = output / "reuse_solver_degradation.png"
    finish_figure(fig, path)
    chart_map.append(
        {
            "section": "solver reuse",
            "question": "Does nearby reuse preserve the system-own oracle gain?",
            "family": "comparison with benchmark",
            "fields": "q, median iteration ratio versus own, source type, family",
            "claim": "Nearby, far, and random spaces are compared against the same target-system oracle.",
            "palette": "relaxed multi-category",
            "path": path.name,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x_positions = np.arange(len(family_order))
    offsets = {24: -0.22, 32: 0.0, 40: 0.22}
    for q in Q_VALUES:
        color, marker = q_styles[q]
        geometry_medians = []
        solver_medians = []
        for family in family_order:
            geometry_medians.append(
                float(
                    np.median(
                        [
                            row["geometry_ratio_vs_nearest_endpoint"]
                            for row in interpolation_geometry
                            if row["family"] == family and row["q"] == q
                        ]
                    )
                )
            )
            solver_medians.append(
                next(
                    row["median_iteration_ratio_vs_nearby"]
                    for row in interpolation_summaries
                    if row["family"] == family and row["q"] == q
                )
            )
        axes[0].scatter(x_positions + offsets[q], geometry_medians, color=color, marker=marker, label=f"q={q}")
        axes[1].scatter(x_positions + offsets[q], solver_medians, color=color, marker=marker, label=f"q={q}")
    for ax in axes:
        ax.axhline(1.0, color=PALETTE["ink"], linestyle=":", linewidth=0.9)
        ax.set_xticks(x_positions, [family_titles[family] for family in family_order], rotation=15, ha="right")
    axes[0].set_ylabel("Median geometry ratio vs nearest endpoint")
    axes[1].set_ylabel("Median iteration ratio vs nearby reuse")
    axes[0].set_title("Geometric approximation")
    axes[1].set_title("Target-system FGMRES")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Grassmann geodesic interpolation at interior parameters", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.90, "Values below one improve on the nearer endpoint space", color=PALETTE["grey"])
    fig.tight_layout(rect=[0, 0, 0.88, 0.84])
    path = output / "grassmann_interpolation.png"
    finish_figure(fig, path)
    chart_map.append(
        {
            "section": "interpolation",
            "question": "Does Grassmann interpolation improve on endpoint reuse?",
            "family": "paired benchmark comparison",
            "fields": "family, q, geometry ratio, iteration ratio",
            "claim": "Subspace geometry and solver impact are evaluated separately.",
            "palette": "three-root categorical",
            "path": path.name,
        }
    )
    return chart_map


def evidence_summary(
    system_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    reuse_summaries: list[dict[str, Any]],
    geometry_solver_rows: list[dict[str, Any]],
    interpolation_geometry: list[dict[str, Any]],
    interpolation_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    q32_correlations = [row for row in correlations if row["q"] == 32]
    q32_overlap = [row for row in overlap_rows if row["q"] == 32]
    q32_reuse = [row for row in reuse_summaries if row["q"] == 32]
    q32_interpolation = [row for row in interpolation_summaries if row["q"] == 32]
    return {
        "scope": {
            "N": N,
            "kernel": "gaussian",
            "q_values": list(Q_VALUES),
            "families": ["epsilon", "jitter", "pde_coefficient"],
            "neural_network_trained": False,
        },
        "numerical_validity": {
            "max_grid_relative_l2_error": max(row["grid_rel_l2"] for row in system_rows),
            "max_relative_residual": max(row["relative_residual"] for row in system_rows),
            "max_normwise_backward_error": max(row["normwise_backward_error"] for row in system_rows),
            "max_condition_number": max(row["condition_2"] for row in system_rows),
            "max_roundoff_exposure_kappa_u": max(row["roundoff_exposure_kappa_u"] for row in system_rows),
        },
        "q32_parameter_distance_correlations": {
            row["family"]: {
                "spearman_rho": row["spearman_rho"],
                "pearson_r": row["pearson_r"],
                "permutation_p": row["mantel_label_permutation_p_two_sided"],
                "adjacent_distance_median": row["adjacent_distance_median"],
                "endpoint_distance": row["endpoint_distance"],
            }
            for row in q32_correlations
        },
        "q32_within_between_separation": {
            row["family"]: {
                "within_median": row["within_median"],
                "adjacent_median": row["between_adjacent_median"],
                "log_histogram_overlap_coefficient": row["log_histogram_overlap_coefficient"],
                "probability_between_exceeds_within": row["probability_between_exceeds_within"],
            }
            for row in q32_overlap
        },
        "q32_solver_reuse": {
            row["family"]: {
                source["source_type"]: {
                    "median_iteration_ratio_vs_own": source["median_iteration_ratio_vs_own"],
                    "median_iteration_ratio_vs_classical": source["median_iteration_ratio_vs_classical"],
                    "median_solve_ratio_vs_own": source["median_solve_ratio_vs_own"],
                }
                for source in q32_reuse
                if source["family"] == row["family"]
            }
            for row in q32_reuse
            if row["source_type"] == "own_oracle"
        },
        "q32_geometry_solver_association": {
            row["family"]: {
                "spearman_projector_distance_vs_iteration_degradation": row[
                    "spearman_projector_distance_vs_iteration_degradation"
                ],
                "spearman_max_angle_vs_iteration_degradation": row[
                    "spearman_max_angle_vs_iteration_degradation"
                ],
            }
            for row in geometry_solver_rows
            if row["q"] == 32 and row["source_scope"] == "structured_reuse"
        },
        "q32_grassmann_interpolation": {
            row["family"]: {
                "median_geometry_ratio_vs_nearest_endpoint": float(
                    np.median(
                        [
                            geometry["geometry_ratio_vs_nearest_endpoint"]
                            for geometry in interpolation_geometry
                            if geometry["family"] == row["family"] and geometry["q"] == 32
                        ]
                    )
                ),
                "median_iteration_ratio_vs_own": row["median_iteration_ratio_vs_own"],
                "median_iteration_ratio_vs_nearby": row["median_iteration_ratio_vs_nearby"],
                "fraction_better_than_nearby": row["fraction_better_than_nearby"],
            }
            for row in q32_interpolation
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/global_rbf_stage2"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--solver-repeats", type=int, default=5)
    parser.add_argument("--perturbation-reps", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=1999)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    specs = family_specs(args.quick)
    artifacts, system_rows = build_system_artifacts(specs)
    groups = family_groups(artifacts)
    write_csv(args.output / "systems.csv", system_rows)

    pair_rows, angle_rows = pairwise_geometry(groups)
    write_csv(args.output / "pairwise_subspace_metrics.csv", pair_rows)
    write_csv(args.output / "principal_angles.csv", angle_rows)
    correlations = correlation_summary(
        groups,
        pair_rows,
        bootstrap_reps=args.bootstrap_reps,
        permutation_reps=args.permutation_reps,
    )
    write_csv(args.output / "parameter_distance_correlations.csv", correlations)

    variability_rows = numerical_variability(groups, perturbation_reps=args.perturbation_reps)
    write_csv(args.output / "reference_subspace_variability.csv", variability_rows)
    overlap_rows = variability_overlap_summary(groups, pair_rows, variability_rows)
    write_csv(args.output / "within_between_overlap.csv", overlap_rows)

    reuse_rows = reuse_solver_study(groups, repeats=args.solver_repeats)
    write_csv(args.output / "reuse_solver_results.csv", reuse_rows)
    reuse_summaries = reuse_summary(reuse_rows)
    write_csv(args.output / "reuse_solver_summary.csv", reuse_summaries)
    geometry_solver_rows = geometry_solver_correlation(reuse_rows)
    write_csv(args.output / "geometry_solver_correlations.csv", geometry_solver_rows)

    interpolation_geometry, interpolation_solver = interpolation_study(
        groups,
        repeats=args.solver_repeats,
    )
    write_csv(args.output / "interpolation_geometry.csv", interpolation_geometry)
    write_csv(args.output / "interpolation_solver_results.csv", interpolation_solver)
    interpolation_summaries = interpolation_summary(interpolation_solver, reuse_rows)
    write_csv(args.output / "interpolation_summary.csv", interpolation_summaries)

    chart_map = plot_results(
        args.output,
        groups,
        pair_rows,
        correlations,
        variability_rows,
        reuse_rows,
        reuse_summaries,
        interpolation_geometry,
        interpolation_summaries,
    )
    write_csv(args.output / "chart_map.csv", chart_map)

    summary = evidence_summary(
        system_rows,
        correlations,
        overlap_rows,
        reuse_summaries,
        geometry_solver_rows,
        interpolation_geometry,
        interpolation_summaries,
    )
    summary["run_seconds"] = perf_counter() - start
    (args.output / "evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {
        "seed": SEED,
        "side": SIDE,
        "N": N,
        "base_epsilon": BASE_EPSILON,
        "q_values": list(Q_VALUES),
        "rtol": RTOL,
        "rhs_specs": RHS_SPECS,
        "families": [
            {"family": spec.family, "parameter": spec.parameter, "values": spec.values}
            for spec in specs
        ],
        "quick": args.quick,
        "solver_repeats": args.solver_repeats,
        "perturbation_reps": args.perturbation_reps,
        "bootstrap_reps": args.bootstrap_reps,
        "permutation_reps": args.permutation_reps,
        "interpolation": "shortest Grassmann geodesic from principal vector pairs",
        "within_system_perturbations": [1.0e-14, 1.0e-12],
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads_requested": 1,
    }
    (args.output / "software_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
