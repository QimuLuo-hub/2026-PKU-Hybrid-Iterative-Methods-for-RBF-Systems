"""Stage 19A: spatial anisotropy hardness scan and q=8 oracle validation.

No learned model is constructed.  The experiment preserves the Stage 10--13
right-preconditioned Ritz construction and fixed-space recycled GMRES, while
replacing only the constant diffusion tensor by the user-specified smooth
seven-parameter tensor field.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as spla
import scipy.stats as stats
from scipy.stats import qmc

from neurkitt_oracle.gcrodr import restarted_projected_gmres

from ..coarse_space import factor_az_petrov
from ..config import DiscretizationConfig, RBFConfig
from ..krylov import restarted_gmres
from ..nodes import generate_nodes
from ..pde import ManufacturedSolution, SpatialAnisotropicDiffusion, random_smooth_solution
from ..preconditioners import ilu_preconditioner
from ..rbffd import RBFSystem, assemble_on_nodes
from ..subspace_analysis import nested_krylov_ritz_modes, subspace_metrics
from .stage9_scalability import _write_csv, _write_json
from .stage10_ritz_transfer import Stage10Settings


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
PURPLE = "#7553A6"
CHARCOAL = "#30343B"
LIGHT_GRAY = "#D7DBE2"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": CHARCOAL})


@dataclass(frozen=True)
class Stage19ASettings:
    """Frozen sampling, solver, validation, and decision controls."""

    seed: int = 20261901
    sample_count: int = 36
    oracle_subset_count: int = 18
    n_nodes: int = 5041
    stencil_size: int = 25
    node_seed: int = 7
    jitter_fraction: float = 0.15
    q: int = 8
    arnoldi_dimension: int = 64
    construction_rhs_seed: int = 20261001
    evaluation_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    a0_range: Tuple[float, float] = (float(np.log(80.0)), float(np.log(120.0)))
    a1_range: Tuple[float, float] = (-0.18, 0.18)
    a2_range: Tuple[float, float] = (-0.18, 0.18)
    a3_range: Tuple[float, float] = (-0.12, 0.12)
    theta0_range_degrees: Tuple[float, float] = (25.0, 35.0)
    b1_range_degrees: Tuple[float, float] = (-5.0, 5.0)
    b2_range_degrees: Tuple[float, float] = (-5.0, 5.0)
    meaningful_error_limit: float = 0.25
    significant_iterations: int = 50
    strongly_hard_iterations: int = 100
    strong_oracle_reduction: float = 0.40
    substantial_family_fraction: float = 0.25
    consistent_oracle_fraction: float = 0.70
    required_oracle_convergence: float = 0.90

    @property
    def stage10(self) -> Stage10Settings:
        return Stage10Settings()

    @property
    def config(self) -> DiscretizationConfig:
        return DiscretizationConfig(
            n_nodes=self.n_nodes,
            stencil_size=self.stencil_size,
            polynomial_degree=2,
            node_distribution="jittered_grid",
            jitter_fraction=self.jitter_fraction,
            seed=self.node_seed,
            rbf=RBFConfig(kernel="phs", order=3, shape_parameter=1.0),
            compute_local_condition_numbers=False,
        )

    @property
    def parameter_ranges(self) -> Tuple[Tuple[float, float], ...]:
        return (
            self.a0_range,
            self.a1_range,
            self.a2_range,
            self.a3_range,
            self.theta0_range_degrees,
            self.b1_range_degrees,
            self.b2_range_degrees,
        )


def _sample_coefficients(settings: Stage19ASettings) -> List[SpatialAnisotropicDiffusion]:
    unit = qmc.LatinHypercube(d=7, seed=settings.seed).random(settings.sample_count)
    ranges = np.asarray(settings.parameter_ranges, dtype=float)
    values = qmc.scale(unit, ranges[:, 0], ranges[:, 1])
    return [SpatialAnisotropicDiffusion(*map(float, row)) for row in values]


def _discrete_rhs(system: RBFSystem, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    exact = random_smooth_solution(seed, 6).values(system.nodes)
    exact[system.boundary] = 0.0
    return np.asarray(system.matrix @ exact, dtype=float), exact


def _relative_error(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), 1e-300))


def _safe_spearman(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float]:
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
        return float("nan"), float("nan")
    result = stats.spearmanr(x[finite], y[finite])
    return float(result.statistic), float(result.pvalue)


def _coefficient_metadata(
    sample_id: int,
    coefficient: SpatialAnisotropicDiffusion,
    nodes: np.ndarray,
) -> Dict[str, Any]:
    log_ratio = coefficient.log_ratio(nodes)
    ratios = np.exp(log_ratio)
    angles = np.rad2deg(coefficient.angle_radians(nodes))
    return {
        "sample_id": sample_id,
        "a0": coefficient.a0,
        "a1": coefficient.a1,
        "a2": coefficient.a2,
        "a3": coefficient.a3,
        "theta0_degrees": coefficient.theta0_degrees,
        "b1_degrees": coefficient.b1_degrees,
        "b2_degrees": coefficient.b2_degrees,
        "ratio_min_nodes": float(np.min(ratios)),
        "ratio_max_nodes": float(np.max(ratios)),
        "ratio_mean_nodes": float(np.mean(ratios)),
        "log_ratio_std_nodes": float(np.std(log_ratio)),
        "theta_min_degrees_nodes": float(np.min(angles)),
        "theta_max_degrees_nodes": float(np.max(angles)),
        "theta_mean_degrees_nodes": float(np.mean(angles)),
        "theta_std_degrees_nodes": float(np.std(angles)),
        "minimum_tensor_eigenvalue": 1.0,
    }


def _assemble(
    nodes: np.ndarray,
    boundary: np.ndarray,
    coefficient: SpatialAnisotropicDiffusion,
    settings: Stage19ASettings,
) -> Tuple[RBFSystem, float]:
    start = perf_counter()
    system = assemble_on_nodes(nodes, boundary, settings.config, coefficient)
    return system, perf_counter() - start


def run_hardness_scan(
    output: Path,
    settings: Stage19ASettings,
    coefficients: Sequence[SpatialAnisotropicDiffusion],
    nodes: np.ndarray,
    boundary: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the full, oracle-blind ILU scan and manufactured-error check."""

    solve_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    manufactured = ManufacturedSolution()
    for sample_id, coefficient in enumerate(coefficients):
        metadata = _coefficient_metadata(sample_id, coefficient, nodes)
        print(f"[scan {sample_id + 1:02d}/{len(coefficients)}]", flush=True)
        try:
            system, assembly_seconds = _assemble(nodes, boundary, coefficient, settings)
            continuous_rhs = manufactured.spatial_anisotropic_right_hand_side(
                system.nodes, system.boundary, coefficient
            )
            direct_start = perf_counter()
            lu = spla.splu(system.matrix.tocsc())
            numerical = np.asarray(lu.solve(continuous_rhs), dtype=float)
            direct_seconds = perf_counter() - direct_start
            exact = manufactured.values(system.nodes)
            discretization_error = _relative_error(numerical, exact)
            direct_residual = _relative_error(system.matrix @ numerical, continuous_rhs)
            ilu = ilu_preconditioner(
                system.matrix, settings.stage10.classical_ilu, name="strong_ilu"
            )
            local_results = []
            for rhs_seed in settings.evaluation_rhs_seeds:
                rhs, discrete_exact = _discrete_rhs(system, rhs_seed)
                ilu.reset_timing()
                result = restarted_gmres(
                    system.matrix, rhs, settings.stage10.solver, ilu
                )
                true_residual = _relative_error(system.matrix @ result.solution, rhs)
                solution_error = _relative_error(result.solution, discrete_exact)
                row = {
                    **metadata,
                    "rhs_seed": rhs_seed,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "true_final_relative_residual": true_residual,
                    "discrete_solution_relative_error": solution_error,
                    "solve_seconds": result.solve_seconds,
                    "preconditioner_apply_seconds": result.preconditioner_seconds,
                    "failure": result.failure,
                }
                solve_rows.append(row)
                local_results.append(row)
            valid = bool(
                np.isfinite(discretization_error)
                and discretization_error <= settings.meaningful_error_limit
                and direct_residual <= 1.0e-9
            )
            summary_rows.append(
                {
                    **metadata,
                    "assembly_seconds": assembly_seconds,
                    "nnz": system.matrix.nnz,
                    "symmetry_measure": system.diagnostics["symmetry_measure"],
                    "manufactured_relative_error": discretization_error,
                    "manufactured_direct_residual": direct_residual,
                    "manufactured_direct_seconds": direct_seconds,
                    "numerically_valid": valid,
                    "ilu_setup_seconds": ilu.setup_seconds,
                    "ilu_fill_ratio": ilu.diagnostics["factor_fill_ratio"],
                    "ilu_factor_memory_mib": ilu.diagnostics["factor_memory_bytes"] / 2**20,
                    "ilu_iterations_min": min(row["iterations"] for row in local_results),
                    "ilu_iterations_median": float(np.median([row["iterations"] for row in local_results])),
                    "ilu_iterations_max": max(row["iterations"] for row in local_results),
                    "ilu_convergence_rate": float(np.mean([row["converged"] for row in local_results])),
                    "ilu_true_residual_max": max(row["true_final_relative_residual"] for row in local_results),
                    "ilu_solve_seconds_median": float(np.median([row["solve_seconds"] for row in local_results])),
                    "failure": "; ".join(sorted({str(row["failure"]) for row in local_results if row["failure"]})),
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    **metadata,
                    "assembly_seconds": float("nan"),
                    "nnz": 0,
                    "symmetry_measure": float("nan"),
                    "manufactured_relative_error": float("nan"),
                    "manufactured_direct_residual": float("nan"),
                    "manufactured_direct_seconds": float("nan"),
                    "numerically_valid": False,
                    "ilu_setup_seconds": float("nan"),
                    "ilu_fill_ratio": float("nan"),
                    "ilu_factor_memory_mib": float("nan"),
                    "ilu_iterations_min": settings.stage10.solver.max_iterations,
                    "ilu_iterations_median": float(settings.stage10.solver.max_iterations),
                    "ilu_iterations_max": settings.stage10.solver.max_iterations,
                    "ilu_convergence_rate": 0.0,
                    "ilu_true_residual_max": float("nan"),
                    "ilu_solve_seconds_median": float("nan"),
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_csv(output / "hardness_scan_rhs.csv", solve_rows)
    _write_csv(output / "hardness_scan_summary.csv", summary_rows)
    return solve_rows, summary_rows


def _stratified_oracle_ids(
    scan: Sequence[Mapping[str, Any]], settings: Stage19ASettings
) -> List[int]:
    """Select evenly spaced ILU difficulty ranks without inspecting an oracle."""

    valid = sorted(
        (row for row in scan if bool(row["numerically_valid"])),
        key=lambda row: (float(row["ilu_iterations_median"]), int(row["sample_id"])),
    )
    if len(valid) <= settings.oracle_subset_count:
        return [int(row["sample_id"]) for row in valid]
    positions = np.linspace(0, len(valid) - 1, settings.oracle_subset_count)
    indices = sorted(set(int(round(value)) for value in positions))
    return [int(valid[index]["sample_id"]) for index in indices]


def run_oracle_validation(
    output: Path,
    settings: Stage19ASettings,
    coefficients: Sequence[SpatialAnisotropicDiffusion],
    nodes: np.ndarray,
    boundary: np.ndarray,
    scan_summary: Sequence[Mapping[str, Any]],
    scan_rhs: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, np.ndarray]]:
    selected_ids = _stratified_oracle_ids(scan_summary, settings)
    _write_json(
        output / "oracle_selection.json",
        {
            "sample_ids": selected_ids,
            "rule": "evenly spaced order statistics of median ILU iterations among numerically valid systems; oracle outcomes unavailable at selection",
        },
    )
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    spaces: Dict[int, np.ndarray] = {}
    subspace_dir = output / "subspaces"
    subspace_dir.mkdir(parents=True, exist_ok=True)
    scan_by_id = {int(row["sample_id"]): row for row in scan_summary}
    for position, sample_id in enumerate(selected_ids):
        coefficient = coefficients[sample_id]
        print(f"[oracle {position + 1:02d}/{len(selected_ids)}] sample {sample_id:02d}", flush=True)
        try:
            system, assembly_seconds = _assemble(nodes, boundary, coefficient, settings)
            ilu = ilu_preconditioner(
                system.matrix, settings.stage10.classical_ilu, name="strong_ilu"
            )
            construction_rhs, _ = _discrete_rhs(system, settings.construction_rhs_seed)
            modes = nested_krylov_ritz_modes(
                system.matrix,
                ilu,
                max_modes=settings.q,
                arnoldi_dimension=settings.arnoldi_dimension,
                seed=settings.seed + sample_id,
                initial_vector=construction_rhs,
            )
            z_basis, _ = la.qr(modes.basis[:, : settings.q], mode="economic")
            pair = factor_az_petrov(system.matrix, z_basis)
            spaces[sample_id] = z_basis
            np.savez_compressed(
                subspace_dir / f"sample_{sample_id:02d}_q8.npz",
                Z=z_basis,
                W=pair.test,
                U=pair.recycling_trial,
                ritz_values=modes.eigenvalues,
            )
            local = []
            for rhs_seed in settings.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(system, rhs_seed)
                ilu.reset_timing()
                result = restarted_projected_gmres(
                    system.matrix,
                    rhs,
                    ilu.apply,
                    u_basis=pair.recycling_trial,
                    c_basis=pair.test,
                    rtol=settings.stage10.solver.relative_tolerance,
                    atol=settings.stage10.solver.absolute_tolerance,
                    restart=settings.stage10.solver.restart,
                    max_iterations=settings.stage10.solver.max_iterations,
                    record_true_residuals=True,
                )
                baseline_row = next(
                    row for row in scan_rhs
                    if int(row["sample_id"]) == sample_id and int(row["rhs_seed"]) == rhs_seed
                )
                k_ilu = int(baseline_row["iterations"])
                gamma = (k_ilu - result.iterations) / max(k_ilu, 1)
                true_residual = _relative_error(system.matrix @ result.x, rhs)
                record = {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "q": settings.q,
                    "k_ilu": k_ilu,
                    "k_oracle": result.iterations,
                    "gamma_oracle": gamma,
                    "ilu_converged": bool(baseline_row["converged"]),
                    "oracle_converged": result.converged,
                    "oracle_true_final_relative_residual": true_residual,
                    "oracle_solve_seconds": result.solve_seconds,
                    "oracle_failure": "" if result.converged else "maximum iterations reached",
                }
                rows.append(record)
                local.append(record)
            summaries.append(
                {
                    "sample_id": sample_id,
                    "q": settings.q,
                    "selection_rank": position,
                    "ilu_iterations_median": scan_by_id[sample_id]["ilu_iterations_median"],
                    "ilu_convergence_rate": scan_by_id[sample_id]["ilu_convergence_rate"],
                    "oracle_iterations_median": float(np.median([row["k_oracle"] for row in local])),
                    "oracle_iterations_min": min(row["k_oracle"] for row in local),
                    "oracle_iterations_max": max(row["k_oracle"] for row in local),
                    "oracle_convergence_rate": float(np.mean([row["oracle_converged"] for row in local])),
                    "gamma_oracle_median": float(np.median([row["gamma_oracle"] for row in local])),
                    "gamma_oracle_min": min(row["gamma_oracle"] for row in local),
                    "ritz_setup_seconds": modes.diagnostics["setup_seconds"],
                    "pair_setup_seconds": pair.diagnostics["recycling_setup_seconds"],
                    "assembly_seconds_repeated": assembly_seconds,
                    "z_rank": z_basis.shape[1],
                    "z_orthogonality_error": float(np.linalg.norm(z_basis.T @ z_basis - np.eye(settings.q))),
                    "au_equals_w_relative_error": pair.diagnostics["au_equals_w_relative_error"],
                    "reduced_condition": pair.diagnostics["reduced_condition"],
                }
            )
        except Exception as exc:
            summaries.append(
                {
                    "sample_id": sample_id,
                    "q": settings.q,
                    "selection_rank": position,
                    "ilu_iterations_median": scan_by_id[sample_id]["ilu_iterations_median"],
                    "ilu_convergence_rate": scan_by_id[sample_id]["ilu_convergence_rate"],
                    "oracle_iterations_median": settings.stage10.solver.max_iterations,
                    "oracle_iterations_min": settings.stage10.solver.max_iterations,
                    "oracle_iterations_max": settings.stage10.solver.max_iterations,
                    "oracle_convergence_rate": 0.0,
                    "gamma_oracle_median": float("nan"),
                    "gamma_oracle_min": float("nan"),
                    "ritz_setup_seconds": float("nan"),
                    "pair_setup_seconds": float("nan"),
                    "assembly_seconds_repeated": float("nan"),
                    "z_rank": 0,
                    "z_orthogonality_error": float("nan"),
                    "au_equals_w_relative_error": float("nan"),
                    "reduced_condition": float("nan"),
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_csv(output / "oracle_rhs_results.csv", rows)
    _write_csv(output / "oracle_summary.csv", summaries)
    return rows, summaries, spaces


def analyze_spaces(
    output: Path,
    settings: Stage19ASettings,
    coefficients: Sequence[SpatialAnisotropicDiffusion],
    nodes: np.ndarray,
    scan_summary: Sequence[Mapping[str, Any]],
    oracle_summary: Sequence[Mapping[str, Any]],
    spaces: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    """Analyze every difficult oracle-selected system, without outcome filtering."""

    scan_by_id = {int(row["sample_id"]): row for row in scan_summary}
    relevant_ids = sorted(
        sample_id for sample_id in spaces
        if float(scan_by_id[sample_id]["ilu_iterations_median"]) >= settings.significant_iterations
    )
    if len(relevant_ids) < 2:
        relevant_ids = sorted(spaces)
    field_vectors: Dict[int, np.ndarray] = {}
    for sample_id in relevant_ids:
        coefficient = coefficients[sample_id]
        field_vectors[sample_id] = np.concatenate(
            (coefficient.log_ratio(nodes), coefficient.angle_radians(nodes))
        ) / np.sqrt(nodes.shape[0])

    pair_rows: List[Dict[str, Any]] = []
    distance_matrix = np.zeros((len(relevant_ids), len(relevant_ids)))
    for i, first_id in enumerate(relevant_ids):
        for j, second_id in enumerate(relevant_ids):
            if j < i:
                distance_matrix[i, j] = distance_matrix[j, i]
                continue
            if i == j:
                metrics = {
                    "projector_distance": 0.0,
                    "projector_distance_normalized": 0.0,
                    "principal_angle_mean_degrees": 0.0,
                    "principal_angle_max_degrees": 0.0,
                    "principal_angles_degrees": np.zeros(settings.q),
                }
            else:
                metrics = subspace_metrics(spaces[first_id], spaces[second_id])
            distance_matrix[i, j] = float(metrics["projector_distance_normalized"])
            if j <= i:
                continue
            c_first, c_second = coefficients[first_id], coefficients[second_id]
            log_difference = c_first.log_ratio(nodes) - c_second.log_ratio(nodes)
            angle_difference = c_first.angle_radians(nodes) - c_second.angle_radians(nodes)
            field_distance = float(
                np.sqrt(np.mean(log_difference**2) + np.mean(angle_difference**2))
            )
            pair_rows.append(
                {
                    "sample_i": first_id,
                    "sample_j": second_id,
                    "field_distance": field_distance,
                    "log_ratio_rms_distance": float(np.sqrt(np.mean(log_difference**2))),
                    "theta_rms_distance_degrees": float(np.rad2deg(np.sqrt(np.mean(angle_difference**2)))),
                    "parameter_euclidean_distance_scaled": float(np.linalg.norm(field_vectors[first_id] - field_vectors[second_id])),
                    "z_projector_distance": metrics["projector_distance"],
                    "z_projector_distance_normalized": metrics["projector_distance_normalized"],
                    "principal_angle_mean_degrees": metrics["principal_angle_mean_degrees"],
                    "principal_angle_max_degrees": metrics["principal_angle_max_degrees"],
                    "principal_angles_degrees": ";".join(f"{value:.6g}" for value in metrics["principal_angles_degrees"]),
                    "absolute_ilu_iteration_difference": abs(
                        float(scan_by_id[first_id]["ilu_iterations_median"])
                        - float(scan_by_id[second_id]["ilu_iterations_median"])
                    ),
                }
            )
    _write_csv(output / "subspace_pairwise_metrics.csv", pair_rows)
    np.savez_compressed(
        output / "subspace_distance_matrices.npz",
        sample_ids=np.asarray(relevant_ids),
        normalized_projector_distance=distance_matrix,
    )

    nearest_rows = []
    for sample_id in relevant_ids:
        candidates = [row for row in pair_rows if sample_id in (row["sample_i"], row["sample_j"])]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda row: row["field_distance"])
        neighbor = int(nearest["sample_j"] if nearest["sample_i"] == sample_id else nearest["sample_i"])
        nearest_rows.append(
            {
                "sample_id": sample_id,
                "nearest_sample_id": neighbor,
                "field_distance": nearest["field_distance"],
                "z_projector_distance_normalized": nearest["z_projector_distance_normalized"],
                "principal_angle_max_degrees": nearest["principal_angle_max_degrees"],
            }
        )
    _write_csv(output / "nearest_field_neighbors.csv", nearest_rows)

    snapshot = np.column_stack([spaces[sample_id] for sample_id in relevant_ids])
    singular_values = la.svdvals(snapshot)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    dimensions = {}
    for threshold in (0.95, 0.99, 0.995):
        hits = np.flatnonzero(energy >= threshold)
        dimensions[str(threshold)] = int(hits[0] + 1) if hits.size else 0
    spectrum_rows = [
        {
            "mode": index + 1,
            "singular_value": value,
            "normalized_singular_value": value / singular_values[0],
            "captured_energy": energy[index],
        }
        for index, value in enumerate(singular_values)
    ]
    _write_csv(output / "oracle_dictionary_spectrum.csv", spectrum_rows)

    field_subspace_rho, field_subspace_p = _safe_spearman(
        [row["field_distance"] for row in pair_rows],
        [row["z_projector_distance_normalized"] for row in pair_rows],
    )
    field_difficulty_rho, field_difficulty_p = _safe_spearman(
        [row["field_distance"] for row in pair_rows],
        [row["absolute_ilu_iteration_difference"] for row in pair_rows],
    )
    result = {
        "solver_relevant_definition": f"median ILU iterations >= {settings.significant_iterations}; fallback to full oracle subset if fewer than two",
        "solver_relevant_sample_ids": relevant_ids,
        "solver_relevant_count": len(relevant_ids),
        "dictionary_dimensions": dimensions,
        "dictionary_dimension_fractions_of_N": {
            key: value / settings.n_nodes for key, value in dimensions.items()
        },
        "field_vs_subspace_spearman": field_subspace_rho,
        "field_vs_subspace_pvalue": field_subspace_p,
        "field_vs_abs_difficulty_spearman": field_difficulty_rho,
        "field_vs_abs_difficulty_pvalue": field_difficulty_p,
        "nearest_neighbor_subspace_distance_median": float(np.median([row["z_projector_distance_normalized"] for row in nearest_rows])) if nearest_rows else float("nan"),
        "principal_angle_max_median_degrees": float(np.median([row["principal_angle_max_degrees"] for row in pair_rows])) if pair_rows else float("nan"),
    }
    _write_json(output / "space_analysis_summary.json", result)
    return {**result, "distance_matrix": distance_matrix, "spectrum": spectrum_rows, "pairs": pair_rows, "nearest": nearest_rows}


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    settings: Stage19ASettings,
    scan: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> None:
    plot_dir = output / "plots"
    iterations = np.asarray([row["ilu_iterations_median"] for row in scan], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bins = np.arange(0, settings.stage10.solver.max_iterations + 25, 25)
    ax.hist(iterations, bins=bins, color=BLUE, edgecolor="white")
    ax.axvline(settings.significant_iterations, color=ORANGE, linestyle="--", label="significant (50)")
    ax.axvline(settings.strongly_hard_iterations, color=PINK, linestyle=":", label="strongly hard (100)")
    ax.set(xlabel="Median strong-ILU GMRES iterations", ylabel="Systems", title="Full oracle-blind hardness scan")
    ax.legend(frameon=False)
    _save_figure(fig, plot_dir / "hardness_histogram.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    points = ax.scatter(
        [row["theta_mean_degrees_nodes"] for row in scan],
        [row["ratio_mean_nodes"] for row in scan],
        c=iterations,
        cmap="viridis",
        vmin=0,
        vmax=settings.stage10.solver.max_iterations,
        s=55,
        edgecolor="white",
        linewidth=0.5,
    )
    fig.colorbar(points, ax=ax, label="Median ILU iterations")
    ax.set(xlabel="Mean physical angle (degrees)", ylabel="Mean anisotropy ratio", title="Sampled spatial fields and ILU difficulty")
    _save_figure(fig, plot_dir / "coefficient_field_hardness.png")

    successful_oracle = [row for row in oracle if np.isfinite(float(row["gamma_oracle_median"]))]
    if successful_oracle:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = np.arange(len(successful_oracle))
        width = 0.38
        ax.bar(x - width / 2, [row["ilu_iterations_median"] for row in successful_oracle], width, color=LIGHT_GRAY, label="ILU")
        ax.bar(x + width / 2, [row["oracle_iterations_median"] for row in successful_oracle], width, color=OLIVE, label="ILU + q=8 oracle recycle")
        ax.set_xticks(x, [str(int(row["sample_id"])) for row in successful_oracle], rotation=0)
        ax.set(xlabel="Sample ID (ILU-difficulty stratified)", ylabel="Median iterations", title="Own-system q=8 oracle validation")
        ax.legend(frameon=False)
        _save_figure(fig, plot_dir / "oracle_iteration_comparison.png")

    relevant_ids = analysis["solver_relevant_sample_ids"]
    if len(relevant_ids) >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        image = ax.imshow(analysis["distance_matrix"], vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(relevant_ids)), relevant_ids, rotation=90)
        ax.set_yticks(range(len(relevant_ids)), relevant_ids)
        ax.set(title="Normalized q=8 Z-projector distance", xlabel="Sample ID", ylabel="Sample ID")
        fig.colorbar(image, ax=ax)
        _save_figure(fig, plot_dir / "z_projector_distance_heatmap.png")

        fig, ax = plt.subplots(figsize=(6.6, 4.5))
        ax.scatter(
            [row["field_distance"] for row in analysis["pairs"]],
            [row["z_projector_distance_normalized"] for row in analysis["pairs"]],
            color=PURPLE,
            alpha=0.75,
        )
        ax.set(xlabel=r"Field distance $[\mathrm{RMS}(\Delta\log R)^2+\mathrm{RMS}(\Delta\theta)^2]^{1/2}$", ylabel="Normalized Z-projector distance", title=f"Field distance vs subspace distance (Spearman {analysis['field_vs_subspace_spearman']:.2f})")
        _save_figure(fig, plot_dir / "field_vs_subspace_distance.png")

    spectrum = analysis["spectrum"]
    if spectrum:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        axes[0].semilogy([row["mode"] for row in spectrum], [row["normalized_singular_value"] for row in spectrum], color=BLUE)
        axes[0].set(xlabel="Dictionary mode p", ylabel=r"$\sigma_p/\sigma_1$", title="Oracle-Z snapshot spectrum")
        axes[1].plot([row["mode"] for row in spectrum], [row["captured_energy"] for row in spectrum], color=OLIVE)
        for value in (0.95, 0.99, 0.995):
            axes[1].axhline(value, color=LIGHT_GRAY, linewidth=0.8)
        axes[1].set(xlabel="Dictionary dimension p", ylabel="Captured energy", ylim=(0, 1.01), title="Common dictionary energy")
        _save_figure(fig, plot_dir / "oracle_dictionary_spectrum.png")


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    if abs(numeric) >= 1e4 or (0 < abs(numeric) < 1e-3):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def decide(
    scan: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    settings: Stage19ASettings,
) -> Tuple[str, Dict[str, Any]]:
    valid = [row for row in scan if bool(row["numerically_valid"])]
    efficient = [row for row in valid if float(row["ilu_iterations_median"]) <= 20]
    significant = [row for row in valid if float(row["ilu_iterations_median"]) >= settings.significant_iterations]
    strongly_hard = [row for row in valid if float(row["ilu_iterations_median"]) >= settings.strongly_hard_iterations or float(row["ilu_convergence_rate"]) < 1.0]
    relevant_oracles = [
        row for row in oracle
        if int(row["sample_id"]) in {int(item["sample_id"]) for item in significant}
        and np.isfinite(float(row["gamma_oracle_median"]))
    ]
    strong = [
        row for row in relevant_oracles
        if float(row["gamma_oracle_median"]) >= settings.strong_oracle_reduction
        and float(row["oracle_convergence_rate"]) >= settings.required_oracle_convergence
    ]
    valid_fraction = len(valid) / max(len(scan), 1)
    efficient_fraction = len(efficient) / max(len(valid), 1)
    significant_fraction = len(significant) / max(len(valid), 1)
    strong_fraction = len(strong) / max(len(relevant_oracles), 1)
    oracle_convergence = float(np.mean([row["oracle_convergence_rate"] for row in relevant_oracles])) if relevant_oracles else 0.0
    diagnostics = {
        "valid_count": len(valid),
        "valid_fraction": valid_fraction,
        "efficient_count": len(efficient),
        "efficient_fraction_among_valid": efficient_fraction,
        "significant_count": len(significant),
        "significant_fraction_among_valid": significant_fraction,
        "strongly_hard_count": len(strongly_hard),
        "relevant_oracle_count": len(relevant_oracles),
        "strong_oracle_count": len(strong),
        "strong_oracle_fraction": strong_fraction,
        "oracle_convergence_rate_mean": oracle_convergence,
    }
    if efficient_fraction >= 0.75:
        decision = "C"
    elif (
        significant_fraction >= settings.substantial_family_fraction
        and len(relevant_oracles) >= 5
        and strong_fraction >= settings.consistent_oracle_fraction
        and oracle_convergence >= settings.required_oracle_convergence
    ):
        decision = "A"
    elif len(strong) >= 3 and strong_fraction >= 0.30:
        decision = "B"
    else:
        decision = "D"
    diagnostics["decision"] = decision
    return decision, diagnostics


def write_report(
    root: Path,
    output: Path,
    settings: Stage19ASettings,
    scan: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    decision: str,
    diagnostics: Mapping[str, Any],
) -> Path:
    valid = [row for row in scan if bool(row["numerically_valid"])]
    decision_text = {
        "A": "A substantial hard family exists and q=8 Ritz oracle remains strongly solver-useful. Proceed to operator learning.",
        "B": "The oracle remains useful only in a restricted part of coefficient-field space. Define that region explicitly before learning.",
        "C": "Strong ILU solves most spatially varying systems efficiently. Operator learning is not justified for this family.",
        "D": "The existing q=8 Ritz oracle loses solver value. Do not train an operator model; revisit the target/mechanism first.",
    }[decision]
    scan_sorted = sorted(scan, key=lambda row: int(row["sample_id"]))
    oracle_sorted = sorted(oracle, key=lambda row: float(row["ilu_iterations_median"]))
    realized = {
        key: (min(float(row[key]) for row in scan), max(float(row[key]) for row in scan))
        for key in ("a0", "a1", "a2", "a3", "theta0_degrees", "b1_degrees", "b2_degrees", "ratio_min_nodes", "ratio_max_nodes", "theta_min_degrees_nodes", "theta_max_degrees_nodes")
    }
    med_iterations = np.asarray([float(row["ilu_iterations_median"]) for row in valid])
    med_errors = np.asarray([float(row["manufactured_relative_error"]) for row in valid])
    report = fr"""# Stage 19A — spatial-coefficient oracle validation

## Decision

**Decision {decision}: {decision_text}**

This is an oracle-validation experiment, not an ML experiment. No neural model, operator learner, new loss, new Ritz rule, changed recycle mechanism, or post-hoc q adjustment was used. The q=8 space is the same smallest-magnitude Ritz construction from Arnoldi on the right-preconditioned operator $B=A M^{{-1}}$ used in Stages 10–13.

The preregistered gates were: a *significant* system has median ILU iterations $\ge {settings.significant_iterations}$; a substantial family has at least {100*settings.substantial_family_fraction:.0f}% significant valid systems; a strongly useful oracle has median $\gamma_{{oracle}}\ge {settings.strong_oracle_reduction:.2f}$ and convergence rate $\ge {settings.required_oracle_convergence:.2f}$; and at least {100*settings.consistent_oracle_fraction:.0f}% of difficulty-stratified significant oracle systems must pass for Decision A. The observed counts were {diagnostics['significant_count']}/{diagnostics['valid_count']} significant valid systems and {diagnostics['strong_oracle_count']}/{diagnostics['relevant_oracle_count']} strongly useful tested hard oracles.

## Exact setup

- PDE: $-\nabla\cdot(K(x,y)\nabla u)=f$, with $K=Q(\theta)\operatorname{{diag}}(R,1)Q(\theta)^T$ exactly as requested.
- Spatial fields: $\log R=a_0+a_1\sin(\pi x)+a_2\cos(\pi y)+a_3\sin(\pi x)\cos(\pi y)$ and $\theta=\theta_0+b_1\sin(\pi x)+b_2\cos(\pi y)$.
- RBF-FD: N={settings.n_nodes}, fixed jittered-grid nodes (seed {settings.node_seed}, jitter {settings.jitter_fraction}), stencil {settings.stencil_size}, PHS $r^3$, polynomial degree 2, full Dirichlet rows.
- Strong ILU: `drop_tol=1e-3`, `fill_factor=5`, `COLAMD`, zero diagonal-pivot threshold; unchanged from Stages 9–18.
- GMRES/recycled GMRES: right preconditioning, tolerance $10^{{-8}}$, restart 50, cap 300. Construction RHS seed {settings.construction_rhs_seed}; disjoint held-out RHS seeds {settings.evaluation_rhs_seeds}.
- Oracle: 64-step Arnoldi on $AM^{{-1}}$, smallest-magnitude Ritz selection, mapping through $M^{{-1}}$, real/imaginary expansion and ordered orthonormalization; q=8. Recycling uses $AZ=WR$, $U=ZR^{{-1}}$, $AU=W$, and projected Arnoldi, exactly as Stage 13.
- Sample: {settings.sample_count}-point seeded Latin hypercube (seed {settings.seed}). The {settings.oracle_subset_count}-system oracle subset was fixed by evenly spaced order statistics of median ILU difficulty before any oracle outcome was computed.

The assembled variable operator includes the product-rule term

$$-\nabla\cdot(K\nabla u)=-K:\nabla^2u-(\nabla\cdot K)\cdot\nabla u.$$

Analytic tensor derivatives were unit-tested against centered differences. A spatially constant member reproduces the established constant-anisotropy matrix and manufactured RHS to roundoff after the documented legacy angle conversion.

### Configured and realized coefficient ranges

| quantity | configured range | realized parameter/node range |
|---|---:|---:|
| $a_0$ | [{settings.a0_range[0]:.4f}, {settings.a0_range[1]:.4f}] | [{realized['a0'][0]:.4f}, {realized['a0'][1]:.4f}] |
| $a_1$ | {settings.a1_range} | [{realized['a1'][0]:.4f}, {realized['a1'][1]:.4f}] |
| $a_2$ | {settings.a2_range} | [{realized['a2'][0]:.4f}, {realized['a2'][1]:.4f}] |
| $a_3$ | {settings.a3_range} | [{realized['a3'][0]:.4f}, {realized['a3'][1]:.4f}] |
| $\theta_0$ (deg) | {settings.theta0_range_degrees} | [{realized['theta0_degrees'][0]:.3f}, {realized['theta0_degrees'][1]:.3f}] |
| $b_1$ (deg) | {settings.b1_range_degrees} | [{realized['b1_degrees'][0]:.3f}, {realized['b1_degrees'][1]:.3f}] |
| $b_2$ (deg) | {settings.b2_range_degrees} | [{realized['b2_degrees'][0]:.3f}, {realized['b2_degrees'][1]:.3f}] |
| nodewise $R$ | implied, uniformly $\ge1$ | [{realized['ratio_min_nodes'][0]:.2f}, {realized['ratio_max_nodes'][1]:.2f}] |
| nodewise $\theta$ (deg) | implied | [{realized['theta_min_degrees_nodes'][0]:.2f}, {realized['theta_max_degrees_nodes'][1]:.2f}] |

Every $K(x)$ is SPD with pointwise eigenvalues $1$ and $R(x)$. The conservative family bounds give roughly $R\in[49.5,194]$ and $\theta\in[15^\circ,45^\circ]$; only one smooth spatial frequency is present. These ranges were fixed from the Stage-18 transition envelope, before oracle evaluation, and were not tuned to make q=8 succeed.

## Full classical hardness scan

Among {len(scan)} sampled fields, {len(valid)} were numerically valid under the preregistered manufactured-error threshold {settings.meaningful_error_limit:.2f}. Median held-out ILU iterations had median {_fmt(np.median(med_iterations) if med_iterations.size else np.nan,1)}, range [{_fmt(np.min(med_iterations) if med_iterations.size else np.nan,0)}, {_fmt(np.max(med_iterations) if med_iterations.size else np.nan,0)}]. {diagnostics['efficient_count']} valid systems needed at most 20 iterations, {diagnostics['significant_count']} needed at least 50, and {diagnostics['strongly_hard_count']} needed at least 100 or had a cap/failure. Manufactured relative error had median {_fmt(np.median(med_errors) if med_errors.size else np.nan,4)} and maximum {_fmt(np.max(med_errors) if med_errors.size else np.nan,4)}.

![Hardness distribution](results/stage19A_spatial_coefficient_oracle_validation/plots/hardness_histogram.png)

![Coefficient fields and hardness](results/stage19A_spatial_coefficient_oracle_validation/plots/coefficient_field_hardness.png)

| ID | nodewise R min–max | nodewise theta min–max (deg) | discretization error | ILU setup (s) | ILU iter median [min,max] | convergence | true residual max |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in scan_sorted:
        report += f"| {int(row['sample_id'])} | {_fmt(row['ratio_min_nodes'],1)}–{_fmt(row['ratio_max_nodes'],1)} | {_fmt(row['theta_min_degrees_nodes'],1)}–{_fmt(row['theta_max_degrees_nodes'],1)} | {_fmt(row['manufactured_relative_error'],4)} | {_fmt(row['ilu_setup_seconds'],3)} | {_fmt(row['ilu_iterations_median'],0)} [{int(row['ilu_iterations_min'])},{int(row['ilu_iterations_max'])}] | {_fmt(row['ilu_convergence_rate'],2)} | {_fmt(row['ilu_true_residual_max'],2)} |\n"

    report += fr"""

The table is the complete random sample, not a successful-oracle subset. Raw per-RHS results, including failures, are in [`hardness_scan_rhs.csv`](results/stage19A_spatial_coefficient_oracle_validation/hardness_scan_rhs.csv). Matrix nnz, nonsymmetry, ILU fill/memory, timings, and all seven coefficients are retained in [`hardness_scan_summary.csv`](results/stage19A_spatial_coefficient_oracle_validation/hardness_scan_summary.csv).

## q=8 own-system oracle validation

The oracle subset spans the ordered ILU-difficulty distribution. It was selected without observing $Z_*$ or any oracle solve. For each selected system, the construction trajectory is disjoint from all three held-out evaluation trajectories.

| ID | median $k_{{ILU}}$ | median $k_{{oracle}}$ | median $\gamma_{{oracle}}$ | oracle convergence | Ritz setup (s) | recycle-pair setup (s) | rank / $\|AU-W\|/\|W\|$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in oracle_sorted:
        report += f"| {int(row['sample_id'])} | {_fmt(row['ilu_iterations_median'],0)} | {_fmt(row['oracle_iterations_median'],0)} | {_fmt(row['gamma_oracle_median'],3)} | {_fmt(row['oracle_convergence_rate'],2)} | {_fmt(row['ritz_setup_seconds'],3)} | {_fmt(row['pair_setup_seconds'],4)} | {int(row['z_rank'])} / {_fmt(row['au_equals_w_relative_error'],2)} |\n"

    report += fr"""

![Oracle iteration comparison](results/stage19A_spatial_coefficient_oracle_validation/plots/oracle_iteration_comparison.png)

$\gamma_{{oracle}}=(k_{{ILU}}-k_{{oracle}})/k_{{ILU}}$ is evaluated per held-out RHS before taking system medians. Cap-hit iteration reductions are descriptive and are not treated as exact iteration savings beyond the cap.

## Oracle-space complexity

The complexity analysis includes every oracle-selected system with median ILU iterations at least {settings.significant_iterations}; it does **not** filter spaces by oracle success. This leaves {analysis['solver_relevant_count']} systems: {analysis['solver_relevant_sample_ids']}.

- Common oracle-Z dictionary dimensions: p95={analysis['dictionary_dimensions']['0.95']} ({analysis['dictionary_dimension_fractions_of_N']['0.95']:.3%} of N), p99={analysis['dictionary_dimensions']['0.99']} ({analysis['dictionary_dimension_fractions_of_N']['0.99']:.3%}), p99.5={analysis['dictionary_dimensions']['0.995']} ({analysis['dictionary_dimension_fractions_of_N']['0.995']:.3%}).
- Field distance versus normalized Z-projector distance: Spearman $\rho={_fmt(analysis['field_vs_subspace_spearman'],3)}$, p={_fmt(analysis['field_vs_subspace_pvalue'],3)}.
- Field distance versus absolute ILU-difficulty difference: Spearman $\rho={_fmt(analysis['field_vs_abs_difficulty_spearman'],3)}$, p={_fmt(analysis['field_vs_abs_difficulty_pvalue'],3)}.
- Median nearest-field-neighbor normalized Z-projector distance: {_fmt(analysis['nearest_neighbor_subspace_distance_median'],3)}.
- Median pairwise maximum principal angle: {_fmt(analysis['principal_angle_max_median_degrees'],1)} degrees.

![Z projector distance matrix](results/stage19A_spatial_coefficient_oracle_validation/plots/z_projector_distance_heatmap.png)

![Field versus subspace distance](results/stage19A_spatial_coefficient_oracle_validation/plots/field_vs_subspace_distance.png)

![Oracle dictionary spectrum](results/stage19A_spatial_coefficient_oracle_validation/plots/oracle_dictionary_spectrum.png)

The concatenated-basis spectrum is invariant to rotations within each q-dimensional oracle block because its covariance is $\sum_i Z_iZ_i^T$. A compact global dictionary can therefore be diagnosed without pretending individual Ritz vectors are identifiable. Projector and principal-angle results remain the primary geometry diagnostics.

## Interpretation

1. **Does a useful q=8 target survive function-valued input?** The answer follows the preregistered Decision {decision} gate above. “Useful” here means solver acceleration on three disjoint held-out RHSs, not POD energy or geometric proximity.
2. **Is the family genuinely hard?** The full scan contains {diagnostics['significant_count']} significant and {diagnostics['strongly_hard_count']} strongly hard valid systems out of {diagnostics['valid_count']}; easy systems were retained and reported rather than removed.
3. **Is the discretization meaningful?** Validity requires finite direct-solve error $\le{settings.meaningful_error_limit}$ and direct relative residual $\le10^{{-9}}$. Failures remain in the full table and raw files.
4. **Does low-dimensionality of each own-system q=8 space imply an easy learning problem?** No. The dictionary dimensions and cross-system geometry quantify family complexity; they do not establish predictability from the coefficient field or the economics of operator learning.
5. **What may proceed?** Only the action in Decision {decision}. No operator model should be trained if the decision is C or D; Decision B requires an explicit restricted region first. Even Decision A is evidence for attempting—not proof of—operator learning.

## Reproducibility and audit trail

- Code: `src/lcp_rbffd/experiments/stage19a_spatial_coefficient_oracle.py`; runner: `scripts/run_stage19a_spatial_coefficient_oracle.py`.
- Raw results: [`results/stage19A_spatial_coefficient_oracle_validation`](results/stage19A_spatial_coefficient_oracle_validation/).
- Tests: analytic field SPD/divergence and constant-field equivalence are in `tests/test_pde.py` and `tests/test_rbffd.py`.
- Software: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}; platform `{platform.platform()}`.
- No explicit inverse was formed. $R^{{-1}}$ in the recycle normalization used triangular solves.
- No neural-network module was imported or executed in this experiment.
"""
    path = root / "stage19A_spatial_coefficient_oracle_validation.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    root: Path,
    output: Path,
    settings: Stage19ASettings,
    scan_rhs: Sequence[Mapping[str, Any]],
    scan: Sequence[Mapping[str, Any]],
    oracle_rhs: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    spaces: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    checks = {
        "sample_count_exact": len(scan) == settings.sample_count,
        "scan_rhs_count_exact": len(scan_rhs) == settings.sample_count * len(settings.evaluation_rhs_seeds),
        "oracle_subset_not_oversized": len(oracle) <= settings.oracle_subset_count,
        "oracle_rhs_complete_for_successful_spaces": len(oracle_rhs) == len(spaces) * len(settings.evaluation_rhs_seeds),
        "all_saved_spaces_rank_q": all(value.shape == (settings.n_nodes, settings.q) for value in spaces.values()),
        "all_saved_spaces_orthonormal": all(np.linalg.norm(value.T @ value - np.eye(settings.q)) < 1e-10 for value in spaces.values()),
        "oracle_selection_is_difficulty_only": json.loads((output / "oracle_selection.json").read_text())["rule"].startswith("evenly spaced order statistics"),
        "report_exists": (root / "stage19A_spatial_coefficient_oracle_validation.md").exists(),
        "required_plots_exist": all((output / "plots" / name).exists() for name in (
            "hardness_histogram.png", "coefficient_field_hardness.png",
            "oracle_iteration_comparison.png", "oracle_dictionary_spectrum.png",
        )),
    }
    result = {"checks": checks, "all_checks_pass": all(checks.values())}
    _write_json(output / "validation.json", result)
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 19A output validation failed: {checks}")
    return result


def run_stage19a(root: Path) -> Path:
    settings = Stage19ASettings()
    output = root / "results" / "stage19A_spatial_coefficient_oracle_validation"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))
    coefficients = _sample_coefficients(settings)
    nodes, boundary = generate_nodes(settings.config)
    sample_rows = [_coefficient_metadata(i, coefficient, nodes) for i, coefficient in enumerate(coefficients)]
    _write_csv(output / "coefficient_samples.csv", sample_rows)
    np.savez_compressed(output / "fixed_nodes.npz", nodes=nodes, boundary=boundary)

    scan_rhs, scan = run_hardness_scan(output, settings, coefficients, nodes, boundary)
    oracle_rhs, oracle, spaces = run_oracle_validation(
        output, settings, coefficients, nodes, boundary, scan, scan_rhs
    )
    analysis = analyze_spaces(output, settings, coefficients, nodes, scan, oracle, spaces)
    decision, diagnostics = decide(scan, oracle, settings)
    _write_json(output / "decision.json", diagnostics)
    make_plots(output, settings, scan, oracle, analysis)
    report = write_report(root, output, settings, scan, oracle, analysis, decision, diagnostics)
    validate_outputs(root, output, settings, scan_rhs, scan, oracle_rhs, oracle, spaces)
    _write_json(
        output / "manifest.json",
        {
            "report": str(report),
            "decision": decision,
            "raw_csv": sorted(path.name for path in output.glob("*.csv")),
            "plots": sorted(path.name for path in (output / "plots").glob("*.png")),
            "subspaces": sorted(path.name for path in (output / "subspaces").glob("*.npz")),
        },
    )
    print(f"Stage 19A complete: decision {decision}; report {report}", flush=True)
    return report
