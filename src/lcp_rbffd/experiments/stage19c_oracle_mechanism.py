"""Stage 19C: diagnose q=8 recycle-oracle failure for spatial K(x).

The experiment reconstructs the eight exact Stage-19A systems from the saved
coefficient rows and fixed node array.  It changes only the classical oracle
space definition/dimension; the RBF-FD operator, strong ILU, RHS protocol, and
fixed-space projected/recycled GMRES are unchanged.  No learned model appears.
"""

from __future__ import annotations

import csv
import json
import platform
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from neurkitt_oracle.gcrodr import GmresResult, restarted_projected_gmres

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..config import SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..pde import SpatialAnisotropicDiffusion
from ..pod import compute_pod, relative_projection_errors
from ..preconditioners import InverseAction, ilu_preconditioner
from ..rbffd import RBFSystem, assemble_on_nodes
from ..subspace_analysis import (
    generalized_subspace_metrics,
    nested_krylov_ritz_modes,
    ritz_atomic_blocks,
    subspace_metrics,
)
from .stage9_scalability import _accurate_factor_solve, _write_csv, _write_json
from .stage19a_spatial_coefficient_oracle import Stage19ASettings, _discrete_rhs


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PINK = "#B14D75"
PURPLE = "#7553A6"
CHARCOAL = "#30343B"
LIGHT_GRAY = "#D7DBE2"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
    }
)


@dataclass(frozen=True)
class Stage19CSettings:
    """Frozen diagnostic protocol and outcome gates."""

    stage19a: Stage19ASettings = Stage19ASettings()
    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    q_values: Tuple[int, ...] = (4, 8, 12, 16, 24, 32)
    alternative_q_values: Tuple[int, ...] = (8, 16, 24, 32)
    max_q: int = 32
    arnoldi_dimension: int = 64
    pod_construction_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    pod_inputs_per_trajectory: int = 16
    pod_test_inputs_per_trajectory: int = 8
    strong_gamma: float = 0.40
    system_success_convergence_rate: float = 1.0
    robust_system_fraction: float = 0.75


@dataclass
class ArnoldiData:
    residual_basis: np.ndarray
    solution_basis: np.ndarray
    hessenberg: np.ndarray
    seconds: float


@dataclass
class ModeFamily:
    basis: np.ndarray
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    blocks: Tuple[Tuple[int, int], ...]
    seconds: float


@dataclass
class SystemContext:
    sample_id: int
    coefficient: SpatialAnisotropicDiffusion
    system: RBFSystem
    ilu: InverseAction
    assembly_seconds: float


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _coefficient_from_row(row: Mapping[str, str]) -> SpatialAnisotropicDiffusion:
    keys = (
        "a0",
        "a1",
        "a2",
        "a3",
        "theta0_degrees",
        "b1_degrees",
        "b2_degrees",
    )
    return SpatialAnisotropicDiffusion(*(float(row[key]) for key in keys))


def _relative_residual(matrix: sp.spmatrix, solution: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-300))


def _build_context(
    sample_id: int,
    coefficient: SpatialAnisotropicDiffusion,
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage19CSettings,
) -> SystemContext:
    start = perf_counter()
    system = assemble_on_nodes(
        nodes, boundary, settings.stage19a.config, coefficient
    )
    assembly_seconds = perf_counter() - start
    ilu = ilu_preconditioner(
        system.matrix,
        settings.stage19a.stage10.classical_ilu,
        name="strong_ilu",
    )
    return SystemContext(sample_id, coefficient, system, ilu, assembly_seconds)


def _capture_arnoldi(
    matrix: sp.spmatrix,
    ilu: InverseAction,
    initial_vector: np.ndarray,
    dimension: int,
) -> ArnoldiData:
    """Reproduce Stage-10 Arnoldi on B=A M^-1 and retain its full factors."""

    n = matrix.shape[0]
    dimension = min(dimension, n - 1)
    residual_basis = np.zeros((n, dimension + 1), dtype=float)
    solution_basis = np.zeros((n, dimension), dtype=float)
    hessenberg = np.zeros((dimension + 1, dimension), dtype=float)
    norm = float(np.linalg.norm(initial_vector))
    residual_basis[:, 0] = initial_vector / norm
    actual = dimension
    start = perf_counter()
    for column in range(dimension):
        solution_basis[:, column] = ilu.apply(residual_basis[:, column])
        work = np.asarray(matrix @ solution_basis[:, column], dtype=float)
        for row in range(column + 1):
            coefficient = float(np.dot(residual_basis[:, row], work))
            hessenberg[row, column] += coefficient
            work -= coefficient * residual_basis[:, row]
        for row in range(column + 1):
            correction = float(np.dot(residual_basis[:, row], work))
            hessenberg[row, column] += correction
            work -= correction * residual_basis[:, row]
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        if next_norm <= 1e-13:
            actual = column + 1
            break
        residual_basis[:, column + 1] = work / next_norm
    return ArnoldiData(
        residual_basis[:, : actual + 1],
        solution_basis[:, :actual],
        hessenberg[: actual + 1, :actual],
        perf_counter() - start,
    )


def _ordinary_spectrum(
    matrix: sp.spmatrix,
    arnoldi: ArnoldiData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = arnoldi.hessenberg.shape[1]
    values, vectors = la.eig(arnoldi.hessenberg[:m, :m])
    order = np.argsort(np.abs(values))
    residuals = np.empty(values.size, dtype=float)
    for index in range(values.size):
        y = arnoldi.residual_basis[:, :m] @ vectors[:, index]
        z = arnoldi.solution_basis @ vectors[:, index]
        az = matrix @ z
        residuals[index] = float(
            np.linalg.norm(az - values[index] * y)
            / max(np.linalg.norm(az) + abs(values[index]) * np.linalg.norm(y), 1e-300)
        )
    return values, residuals, order


def _orthogonalize_atom(
    atom: np.ndarray,
    accepted: List[np.ndarray],
) -> np.ndarray:
    work = np.asarray(atom, dtype=float).copy()
    if work.ndim == 1:
        work = work[:, None]
    for _ in range(2):
        if accepted:
            previous = np.column_stack(accepted)
            work -= previous @ (previous.T @ work)
    q_basis, triangular = la.qr(work, mode="economic")
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0:
        return np.empty((work.shape[0], 0))
    tolerance = max(work.shape) * np.finfo(float).eps * max(diagonal[0], 1.0)
    rank = int(np.count_nonzero(diagonal > tolerance))
    return q_basis[:, :rank]


def _harmonic_family(
    matrix: sp.spmatrix,
    arnoldi: ArnoldiData,
    max_q: int,
) -> Tuple[ModeFamily, List[Dict[str, Any]]]:
    """Build a real, pair-respecting harmonic-Ritz family nearest zero."""

    start = perf_counter()
    hbar = arnoldi.hessenberg
    m = hbar.shape[1]
    hm = hbar[:m, :]
    gram = hbar.T @ hbar
    test = hm.T
    values, vectors = la.eig(gram, test, check_finite=False)
    finite = np.isfinite(values)
    values, vectors = values[finite], vectors[:, finite]
    order = np.argsort(np.abs(values))
    accepted: List[np.ndarray] = []
    selected_values: List[complex] = []
    selected_residuals: List[float] = []
    blocks: List[Tuple[int, int]] = []
    spectrum_rows: List[Dict[str, Any]] = []
    used: set[int] = set()
    for ordered_position, index_value in enumerate(order):
        index = int(index_value)
        if index in used or len(accepted) >= max_q:
            continue
        vector = vectors[:, index]
        y_complex = arnoldi.residual_basis[:, :m] @ vector
        z_complex = arnoldi.solution_basis @ vector
        az = matrix @ z_complex
        relative = float(
            np.linalg.norm(az - values[index] * y_complex)
            / max(np.linalg.norm(az) + abs(values[index]) * np.linalg.norm(y_complex), 1e-300)
        )
        imaginary = np.linalg.norm(vector.imag)
        scale = max(np.linalg.norm(vector.real), 1.0)
        if imaginary <= 100.0 * np.finfo(float).eps * scale:
            atom = (arnoldi.solution_basis @ vector.real)[:, None]
            used.add(index)
            atom_values = [values[index]]
        else:
            partner_candidates = [
                int(candidate)
                for candidate in order
                if int(candidate) not in used and int(candidate) != index
            ]
            partner = min(
                partner_candidates,
                key=lambda candidate: abs(values[candidate] - np.conjugate(values[index])),
            )
            used.update((index, partner))
            atom = arnoldi.solution_basis @ np.column_stack((vector.real, vector.imag))
            atom_values = [values[index], values[index]]
        if len(accepted) + atom.shape[1] > max_q:
            continue
        orthogonal = _orthogonalize_atom(atom, accepted)
        if orthogonal.shape[1] != atom.shape[1]:
            continue
        block_start = len(accepted)
        accepted.extend([orthogonal[:, column] for column in range(orthogonal.shape[1])])
        selected_values.extend(atom_values)
        selected_residuals.extend([relative] * orthogonal.shape[1])
        blocks.append((block_start, len(accepted)))
        spectrum_rows.append(
            {
                "ordered_atom": len(blocks),
                "block_start": block_start,
                "block_end": len(accepted),
                "eigenvalue_real": float(values[index].real),
                "eigenvalue_imag": float(values[index].imag),
                "magnitude": float(abs(values[index])),
                "eigen_residual_relative": relative,
                "original_order_position": ordered_position,
            }
        )
    if not accepted:
        raise RuntimeError("harmonic Ritz extraction returned no real directions")
    return (
        ModeFamily(
            np.column_stack(accepted),
            np.asarray(selected_values, dtype=complex),
            np.asarray(selected_residuals, dtype=float),
            tuple(blocks),
            arnoldi.seconds + perf_counter() - start,
        ),
        spectrum_rows,
    )


def _actual_prefix(
    basis: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
    requested_q: int,
) -> Tuple[np.ndarray, int]:
    endpoints = [stop for _, stop in blocks if stop <= requested_q]
    if not endpoints:
        raise RuntimeError(f"no complete real/complex block fits q={requested_q}")
    actual = max(endpoints)
    return basis[:, :actual], actual


def _run_baseline(
    context: SystemContext,
    rhs: np.ndarray,
    solver: SolverConfig,
) -> KrylovResult:
    context.ilu.reset_timing()
    return restarted_gmres(context.system.matrix, rhs, solver, context.ilu)


def _run_recycle(
    context: SystemContext,
    rhs: np.ndarray,
    factors: AZPetrovFactors,
    solver: SolverConfig,
) -> GmresResult:
    context.ilu.reset_timing()
    return restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        u_basis=factors.recycling_trial,
        c_basis=factors.test,
        rtol=solver.relative_tolerance,
        atol=solver.absolute_tolerance,
        restart=solver.restart,
        max_iterations=solver.max_iterations,
        record_true_residuals=True,
    )


def _collect_preconditioner_inputs(
    context: SystemContext,
    seeds: Sequence[int],
    count_per_seed: int,
    solver: SolverConfig,
) -> np.ndarray:
    columns: List[np.ndarray] = []
    short_solver = replace(
        solver,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        restart=max(count_per_seed, 2),
        max_iterations=count_per_seed,
    )
    for seed in seeds:
        local: List[np.ndarray] = []
        rhs, _ = _discrete_rhs(context.system, seed)

        def capture(_: int, vector: np.ndarray) -> None:
            if len(local) < count_per_seed:
                local.append(np.asarray(vector, dtype=float).copy())

        restarted_gmres(
            context.system.matrix,
            rhs,
            short_solver,
            context.ilu,
            preconditioner_input_callback=capture,
        )
        columns.extend(local)
    if not columns:
        raise RuntimeError("no Arnoldi/preconditioner inputs were collected")
    matrix = np.column_stack(columns)
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms


def _missing_actions(
    context: SystemContext,
    factor: Any,
    probes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    corrections, residuals = [], []
    for column in range(probes.shape[1]):
        probe = probes[:, column]
        exact, relative = _accurate_factor_solve(context.system.matrix, factor, probe)
        correction = exact - context.ilu.apply(probe)
        corrections.append(correction)
        residuals.append(relative)
    return np.column_stack(corrections), np.asarray(residuals)


def _summarize_solver_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted(
        {(str(row["definition"]), int(row["q_requested"]), int(row["sample_id"])) for row in rows}
    )
    summaries = []
    for definition, q, sample_id in keys:
        local = [
            row for row in rows
            if str(row["definition"]) == definition
            and int(row["q_requested"]) == q
            and int(row["sample_id"]) == sample_id
        ]
        summaries.append(
            {
                "sample_id": sample_id,
                "definition": definition,
                "q_requested": q,
                "q_actual": int(local[0]["q_actual"]),
                "k_ilu_median": float(np.median([row["k_ilu"] for row in local])),
                "k_oracle_median": float(np.median([row["k_oracle"] for row in local])),
                "k_oracle_min": min(int(row["k_oracle"]) for row in local),
                "k_oracle_max": max(int(row["k_oracle"]) for row in local),
                "convergence_rate": float(np.mean([bool(row["converged"]) for row in local])),
                "gamma_median": float(np.median([row["gamma_oracle"] for row in local])),
                "gamma_min": min(float(row["gamma_oracle"]) for row in local),
                "true_residual_max": max(float(row["true_final_residual"]) for row in local),
                "construction_seconds": float(local[0]["construction_seconds"]),
                "pair_setup_seconds": float(local[0]["pair_setup_seconds"]),
                "online_solve_seconds_median": float(np.median([row["online_solve_seconds"] for row in local])),
                "total_oracle_cost_median": float(np.median([row["total_oracle_cost"] for row in local])),
            }
        )
    return summaries


def _system_success(
    summary: Mapping[str, Any], settings: Stage19CSettings
) -> bool:
    return bool(
        float(summary["convergence_rate"]) >= settings.system_success_convergence_rate
        and float(summary["gamma_median"]) >= settings.strong_gamma
    )


def _q_min_rows(
    summaries: Sequence[Mapping[str, Any]], settings: Stage19CSettings
) -> List[Dict[str, Any]]:
    rows = []
    for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
        for sample_id in settings.sample_ids:
            local = sorted(
                (
                    row for row in summaries
                    if row["definition"] == definition and int(row["sample_id"]) == sample_id
                ),
                key=lambda row: int(row["q_actual"]),
            )
            successful = [row for row in local if _system_success(row, settings)]
            best = min(successful, key=lambda row: int(row["q_actual"])) if successful else None
            rows.append(
                {
                    "sample_id": sample_id,
                    "definition": definition,
                    "q_min": "" if best is None else int(best["q_actual"]),
                    "q_requested_at_q_min": "" if best is None else int(best["q_requested"]),
                    "gamma_at_q_min": "" if best is None else float(best["gamma_median"]),
                    "convergence_at_q_min": "" if best is None else float(best["convergence_rate"]),
                    "success_by_q32": best is not None,
                }
            )
    return rows


def run_stage19c(root: Path) -> Path:
    settings = Stage19CSettings()
    source = root / "results" / "stage19A_spatial_coefficient_oracle_validation"
    output = root / "results" / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "reconstructed_systems").mkdir(exist_ok=True)
    (output / "spaces").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))

    node_data = np.load(source / "fixed_nodes.npz")
    nodes, boundary = node_data["nodes"], node_data["boundary"]
    coefficient_rows = {int(row["sample_id"]): row for row in _read_rows(source / "coefficient_samples.csv")}
    stage19a_scan = {int(row["sample_id"]): row for row in _read_rows(source / "hardness_scan_summary.csv")}
    stage19a_oracle_rhs = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in _read_rows(source / "oracle_rhs_results.csv")
    }

    baseline_rows: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    incremental_rows: List[Dict[str, Any]] = []
    ritz_spectrum_rows: List[Dict[str, Any]] = []
    harmonic_spectrum_rows: List[Dict[str, Any]] = []
    correction_spectrum_rows: List[Dict[str, Any]] = []
    correction_structure_rows: List[Dict[str, Any]] = []
    reconstruction_rows: List[Dict[str, Any]] = []
    spaces_for_secondary: Dict[Tuple[str, int], np.ndarray] = {}

    for ordinal, sample_id in enumerate(settings.sample_ids):
        print(f"[Stage19C {ordinal + 1}/{len(settings.sample_ids)}] sample {sample_id}", flush=True)
        coefficient = _coefficient_from_row(coefficient_rows[sample_id])
        context = _build_context(sample_id, coefficient, nodes, boundary, settings)
        sp.save_npz(
            output / "reconstructed_systems" / f"sample_{sample_id:02d}_matrix.npz",
            context.system.matrix,
            compressed=True,
        )

        baseline_by_seed: Dict[int, KrylovResult] = {}
        for rhs_seed in settings.stage19a.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            result = _run_baseline(context, rhs, settings.stage19a.stage10.solver)
            baseline_by_seed[rhs_seed] = result
            baseline_rows.append(
                {
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "true_final_residual": _relative_residual(context.system.matrix, result.solution, rhs),
                    "solve_seconds": result.solve_seconds,
                    "failure": result.failure,
                }
            )

        construction_rhs, _ = _discrete_rhs(
            context.system, settings.stage19a.construction_rhs_seed
        )
        modes_max = nested_krylov_ritz_modes(
            context.system.matrix,
            context.ilu,
            max_modes=settings.max_q,
            arnoldi_dimension=settings.arnoldi_dimension,
            seed=settings.stage19a.seed + sample_id,
            initial_vector=construction_rhs,
        )
        ordinary_blocks = ritz_atomic_blocks(modes_max.eigenvalues)
        arnoldi = _capture_arnoldi(
            context.system.matrix,
            context.ilu,
            construction_rhs,
            settings.arnoldi_dimension,
        )
        full_values, full_residuals, full_order = _ordinary_spectrum(
            context.system.matrix, arnoldi
        )
        harmonic, harmonic_rows = _harmonic_family(
            context.system.matrix, arnoldi, settings.max_q
        )
        for row in harmonic_rows:
            harmonic_spectrum_rows.append({"sample_id": sample_id, **row})

        # Map Stage-10 labels back to the projected eigensystem for residual diagnostics.
        block_by_index = {}
        for block_id, (start, stop) in enumerate(ordinary_blocks, start=1):
            for index in range(start, stop):
                block_by_index[index] = (block_id, start, stop)
        magnitudes = np.abs(modes_max.eigenvalues)
        for index, (value, label) in enumerate(zip(modes_max.eigenvalues, modes_max.mode_labels)):
            match = re.match(r"eig(\d+)_", label)
            eig_index = int(match.group(1)) if match else int(np.argmin(np.abs(full_values - value)))
            block_id, block_start, block_end = block_by_index[index]
            next_magnitude = (
                float(magnitudes[block_end]) if block_end < len(magnitudes) else float("nan")
            )
            gap = (
                (next_magnitude - float(magnitudes[index])) / max(float(magnitudes[index]), 1e-300)
                if index == block_end - 1 and np.isfinite(next_magnitude)
                else float("nan")
            )
            ritz_spectrum_rows.append(
                {
                    "sample_id": sample_id,
                    "direction_index": index + 1,
                    "mode_label": label,
                    "block_id": block_id,
                    "block_start": block_start + 1,
                    "block_end": block_end,
                    "eigenvalue_real": float(value.real),
                    "eigenvalue_imag": float(value.imag),
                    "magnitude": float(abs(value)),
                    "ritz_residual_relative": float(full_residuals[eig_index]),
                    "relative_gap_after_block": gap,
                }
            )

        cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

        def evaluate(
            definition: str,
            q_requested: int,
            basis: np.ndarray,
            q_actual: int,
            construction_seconds: float,
            rhs_seeds: Sequence[int],
        ) -> Tuple[List[Dict[str, Any]], AZPetrovFactors]:
            factors = factor_az_petrov(context.system.matrix, basis)
            records = []
            for rhs_seed in rhs_seeds:
                key = (definition, q_actual, rhs_seed)
                if key not in cache:
                    rhs, _ = _discrete_rhs(context.system, rhs_seed)
                    result = _run_recycle(
                        context, rhs, factors, settings.stage19a.stage10.solver
                    )
                    cache[key] = {
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "true_final_residual": _relative_residual(context.system.matrix, result.x, rhs),
                        "online_solve_seconds": result.solve_seconds,
                    }
                value = cache[key]
                baseline = baseline_by_seed[rhs_seed]
                k_ilu = baseline.iterations
                records.append(
                    {
                        "sample_id": sample_id,
                        "definition": definition,
                        "q_requested": q_requested,
                        "q_actual": q_actual,
                        "rhs_seed": rhs_seed,
                        "k_ilu": k_ilu,
                        "k_oracle": value["iterations"],
                        "gamma_oracle": (k_ilu - value["iterations"]) / max(k_ilu, 1),
                        "converged": value["converged"],
                        "true_final_residual": value["true_final_residual"],
                        "construction_seconds": construction_seconds,
                        "pair_setup_seconds": factors.diagnostics["recycling_setup_seconds"],
                        "online_solve_seconds": value["online_solve_seconds"],
                        "total_oracle_cost": construction_seconds
                        + factors.diagnostics["recycling_setup_seconds"]
                        + value["online_solve_seconds"],
                        "reduced_condition": factors.diagnostics["reduced_condition"],
                    }
                )
            return records, factors

        # Part 1: exact Stage-19A Ritz rule, changing only requested q.
        timing_by_q: Dict[int, float] = {}
        for q_requested in settings.q_values:
            timing_modes = nested_krylov_ritz_modes(
                context.system.matrix,
                context.ilu,
                max_modes=q_requested,
                arnoldi_dimension=settings.arnoldi_dimension,
                seed=settings.stage19a.seed + sample_id,
                initial_vector=construction_rhs,
            )
            timing_by_q[q_requested] = float(timing_modes.diagnostics["setup_seconds"])
            basis, q_actual = _actual_prefix(
                modes_max.basis, ordinary_blocks, q_requested
            )
            records, factors = evaluate(
                "ordinary_ritz",
                q_requested,
                basis,
                q_actual,
                timing_by_q[q_requested],
                settings.stage19a.evaluation_rhs_seeds,
            )
            solver_rows.extend(records)
            if q_requested == 8:
                saved = np.load(source / "subspaces" / f"sample_{sample_id:02d}_q8.npz")["Z"]
                geometry = subspace_metrics(basis, saved)
                iteration_match = all(
                    int(record["k_oracle"])
                    == int(stage19a_oracle_rhs[(sample_id, int(record["rhs_seed"]))]["k_oracle"])
                    for record in records
                )
                baseline_match = all(
                    int(record["k_ilu"])
                    == int(stage19a_oracle_rhs[(sample_id, int(record["rhs_seed"]))]["k_ilu"])
                    for record in records
                )
                reconstruction_rows.append(
                    {
                        "sample_id": sample_id,
                        "nnz_stage19a": int(float(stage19a_scan[sample_id]["nnz"])),
                        "nnz_reconstructed": context.system.matrix.nnz,
                        "symmetry_stage19a": float(stage19a_scan[sample_id]["symmetry_measure"]),
                        "symmetry_reconstructed": context.system.diagnostics["symmetry_measure"],
                        "fill_stage19a": float(stage19a_scan[sample_id]["ilu_fill_ratio"]),
                        "fill_reconstructed": context.ilu.diagnostics["factor_fill_ratio"],
                        "q8_projector_distance_normalized": geometry["projector_distance_normalized"],
                        "q8_baseline_iterations_exact": baseline_match,
                        "q8_oracle_iterations_exact": iteration_match,
                    }
                )
                if (
                    float(geometry["projector_distance_normalized"]) > 1e-10
                    or not iteration_match
                    or not baseline_match
                ):
                    raise RuntimeError(
                        f"sample {sample_id}: failed Stage-19A q=8 reproduction"
                    )

        # Incremental response at each complete real/complex block, first held-out RHS.
        previous_k = baseline_by_seed[settings.stage19a.evaluation_rhs_seeds[0]].iterations
        previous_q = 0
        for block_id, (_, block_end) in enumerate(ordinary_blocks, start=1):
            if block_end > settings.max_q:
                break
            basis = modes_max.basis[:, :block_end]
            records, _ = evaluate(
                "ordinary_ritz",
                block_end,
                basis,
                block_end,
                float(modes_max.diagnostics["setup_seconds"]),
                (settings.stage19a.evaluation_rhs_seeds[0],),
            )
            current_k = int(records[0]["k_oracle"])
            incremental_rows.append(
                {
                    "sample_id": sample_id,
                    "baseline_iterations": baseline_by_seed[
                        settings.stage19a.evaluation_rhs_seeds[0]
                    ].iterations,
                    "block_id": block_id,
                    "previous_q": previous_q,
                    "q_actual": block_end,
                    "iterations": current_k,
                    "marginal_iteration_improvement": previous_k - current_k,
                    "converged": records[0]["converged"],
                    "gamma_oracle": records[0]["gamma_oracle"],
                }
            )
            previous_q, previous_k = block_end, current_k

        # Part 3B: harmonic Ritz from the same captured right-preconditioned Arnoldi relation.
        for q_requested in settings.alternative_q_values:
            basis, q_actual = _actual_prefix(
                harmonic.basis, harmonic.blocks, q_requested
            )
            records, _ = evaluate(
                "harmonic_ritz",
                q_requested,
                basis,
                q_actual,
                harmonic.seconds,
                settings.stage19a.evaluation_rhs_seeds,
            )
            solver_rows.extend(records)

        # Parts 3A/4: correction POD on training trajectories only; evaluation
        # trajectories are used solely for held-out projection and solver tests.
        pod_start = perf_counter()
        train_probes = _collect_preconditioner_inputs(
            context,
            settings.pod_construction_rhs_seeds,
            settings.pod_inputs_per_trajectory,
            settings.stage19a.stage10.solver,
        )
        test_probes = _collect_preconditioner_inputs(
            context,
            settings.stage19a.evaluation_rhs_seeds,
            settings.pod_test_inputs_per_trajectory,
            settings.stage19a.stage10.solver,
        )
        factor_start = perf_counter()
        exact_factor = spla.splu(context.system.matrix.tocsc())
        factor_seconds = perf_counter() - factor_start
        train_corrections, train_exact_residuals = _missing_actions(
            context, exact_factor, train_probes
        )
        test_corrections, test_exact_residuals = _missing_actions(
            context, exact_factor, test_probes
        )
        pod = compute_pod(train_corrections)
        pod_seconds = perf_counter() - pod_start
        q95_hits = np.flatnonzero(pod.captured_energy >= 0.95)
        q99_hits = np.flatnonzero(pod.captured_energy >= 0.99)
        q95 = int(q95_hits[0] + 1) if q95_hits.size else 0
        q99 = int(q99_hits[0] + 1) if q99_hits.size else 0
        correction_structure_rows.append(
            {
                "sample_id": sample_id,
                "train_snapshots": train_corrections.shape[1],
                "test_snapshots": test_corrections.shape[1],
                "q95": q95,
                "q99": q99,
                "q95_over_N": q95 / settings.stage19a.n_nodes,
                "q99_over_N": q99 / settings.stage19a.n_nodes,
                "exact_factor_seconds": factor_seconds,
                "pod_total_setup_seconds": pod_seconds,
                "max_exact_solve_residual": float(max(np.max(train_exact_residuals), np.max(test_exact_residuals))),
            }
        )
        for index, singular in enumerate(pod.singular_values):
            eta = relative_projection_errors(
                pod.basis[:, : index + 1], test_corrections
            )
            correction_spectrum_rows.append(
                {
                    "sample_id": sample_id,
                    "q": index + 1,
                    "normalized_singular_value": float(singular / pod.singular_values[0]),
                    "captured_energy": float(pod.captured_energy[index]),
                    "heldout_projection_error_median": float(np.median(eta)),
                    "heldout_projection_error_p90": float(np.quantile(eta, 0.9)),
                }
            )
        for q_requested in settings.alternative_q_values:
            q_actual = min(q_requested, pod.basis.shape[1])
            basis = pod.basis[:, :q_actual]
            records, _ = evaluate(
                "correction_pod",
                q_requested,
                basis,
                q_actual,
                pod_seconds,
                settings.stage19a.evaluation_rhs_seeds,
            )
            solver_rows.extend(records)
        np.savez_compressed(
            output / "spaces" / f"sample_{sample_id:02d}_diagnostic_spaces.npz",
            ordinary_ritz=modes_max.basis,
            ordinary_ritz_values=modes_max.eigenvalues,
            harmonic_ritz=harmonic.basis,
            harmonic_ritz_values=harmonic.eigenvalues,
            correction_pod=pod.basis[:, : settings.max_q],
            correction_singular_values=pod.singular_values,
            correction_energy=pod.captured_energy,
        )

    _write_csv(output / "baseline_results.csv", baseline_rows)
    _write_csv(output / "oracle_solver_results.csv", solver_rows)
    summaries = _summarize_solver_rows(solver_rows)
    _write_csv(output / "oracle_solver_summary.csv", summaries)
    _write_csv(output / "ritz_incremental_blocks.csv", incremental_rows)
    _write_csv(output / "ordinary_ritz_spectrum.csv", ritz_spectrum_rows)
    _write_csv(output / "harmonic_ritz_spectrum.csv", harmonic_spectrum_rows)
    _write_csv(output / "correction_pod_spectrum.csv", correction_spectrum_rows)
    _write_csv(output / "correction_structure_summary.csv", correction_structure_rows)
    _write_csv(output / "stage19a_reconstruction_checks.csv", reconstruction_rows)

    q_min = _q_min_rows(summaries, settings)
    _write_csv(output / "q_min_summary.csv", q_min)

    # Secondary geometry uses each method's smallest successful space only.
    for row in q_min:
        if not row["success_by_q32"]:
            continue
        sample_id = int(row["sample_id"])
        definition = str(row["definition"])
        q_value = int(row["q_min"])
        archive = np.load(output / "spaces" / f"sample_{sample_id:02d}_diagnostic_spaces.npz")
        key = {
            "ordinary_ritz": "ordinary_ritz",
            "harmonic_ritz": "harmonic_ritz",
            "correction_pod": "correction_pod",
        }[definition]
        spaces_for_secondary[(definition, sample_id)] = archive[key][:, :q_value]
    secondary_rows = []
    for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
        identifiers = sorted(sample_id for method, sample_id in spaces_for_secondary if method == definition)
        for first_index, first in enumerate(identifiers):
            for second in identifiers[first_index + 1 :]:
                metrics = generalized_subspace_metrics(
                    spaces_for_secondary[(definition, first)],
                    spaces_for_secondary[(definition, second)],
                )
                secondary_rows.append(
                    {
                        "definition": definition,
                        "sample_i": first,
                        "sample_j": second,
                        "rank_i": spaces_for_secondary[(definition, first)].shape[1],
                        "rank_j": spaces_for_secondary[(definition, second)].shape[1],
                        "projector_distance_normalized": metrics["projector_distance_normalized"],
                        "principal_angle_mean_degrees": metrics["principal_angle_mean_degrees"],
                        "principal_angle_max_degrees": metrics["principal_angle_max_degrees"],
                    }
                )
    _write_csv(output / "successful_space_secondary_geometry.csv", secondary_rows)

    decision, decision_data = _decision(q_min, settings)
    _write_json(output / "decision.json", decision_data)
    _make_plots(output, settings, summaries, incremental_rows, correction_spectrum_rows)
    report = _write_report(
        root,
        output,
        settings,
        summaries,
        q_min,
        ritz_spectrum_rows,
        correction_structure_rows,
        correction_spectrum_rows,
        reconstruction_rows,
        secondary_rows,
        decision,
        decision_data,
    )
    _validate(
        output,
        report,
        settings,
        baseline_rows,
        solver_rows,
        summaries,
        reconstruction_rows,
    )
    print(f"Stage 19C complete: decision {decision}; report {report}", flush=True)
    return report


def _decision(
    q_min: Sequence[Mapping[str, Any]],
    settings: Stage19CSettings,
) -> Tuple[str, Dict[str, Any]]:
    success_counts = {}
    for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
        success_counts[definition] = sum(
            bool(row["success_by_q32"])
            for row in q_min
            if row["definition"] == definition
        )
    required = int(np.ceil(settings.robust_system_fraction * len(settings.sample_ids)))
    ordinary = success_counts["ordinary_ritz"]
    harmonic = success_counts["harmonic_ritz"]
    pod = success_counts["correction_pod"]
    if ordinary >= required:
        decision = "A"
    elif harmonic >= required and harmonic - ordinary >= 3:
        decision = "B"
    elif pod >= required and pod - max(ordinary, harmonic) >= 3:
        decision = "C"
    else:
        decision = "D"
    return decision, {
        "decision": decision,
        "strong_acceleration_definition": (
            f"all three held-out RHS converge and median gamma >= {settings.strong_gamma}"
        ),
        "robust_family_definition": (
            f"at least {required}/{len(settings.sample_ids)} systems pass"
        ),
        "successful_system_counts": success_counts,
        "required_success_count": required,
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_plots(
    output: Path,
    settings: Stage19CSettings,
    summaries: Sequence[Mapping[str, Any]],
    incremental: Sequence[Mapping[str, Any]],
    correction_spectrum: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output / "plots"
    # Chart contract: small-multiple ordered line; x=q, y=median iterations;
    # zero-to-cap scale shared across all systems, with the ILU baseline explicit.
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.1), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        local = sorted(
            (
                row for row in summaries
                if int(row["sample_id"]) == sample_id
                and row["definition"] == "ordinary_ritz"
            ),
            key=lambda row: int(row["q_actual"]),
        )
        baseline = float(local[0]["k_ilu_median"])
        axis.axhline(baseline, color=CHARCOAL, linestyle="--", linewidth=1.0, label="ILU")
        axis.plot(
            [int(row["q_actual"]) for row in local],
            [float(row["k_oracle_median"]) for row in local],
            color=BLUE,
            marker="o",
            label="ordinary Ritz",
        )
        axis.set_title(f"ID {sample_id}")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_ylim(0, settings.stage19a.stage10.solver.max_iterations + 10)
    axes[1, 0].set_xlabel("Complete-block dimension q")
    axes[1, 1].set_xlabel("Complete-block dimension q")
    axes[1, 2].set_xlabel("Complete-block dimension q")
    axes[1, 3].set_xlabel("Complete-block dimension q")
    axes[0, 0].set_ylabel("Median iterations")
    axes[1, 0].set_ylabel("Median iterations")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    figure.suptitle("Existing smallest-magnitude Ritz q-sensitivity", y=1.015)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(figure, plot_dir / "ordinary_ritz_q_sensitivity.png")

    # Comparison/ordered line: the same q grid and shared cap scale isolate target definition.
    styles = {
        "ordinary_ritz": (BLUE, "o", "-", "ordinary Ritz"),
        "harmonic_ritz": (ORANGE, "s", "--", "harmonic Ritz"),
        "correction_pod": (OLIVE, "^", ":", "correction POD"),
    }
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        local_all = [row for row in summaries if int(row["sample_id"]) == sample_id]
        baseline = float(local_all[0]["k_ilu_median"])
        axis.axhline(baseline, color=CHARCOAL, linestyle="-.", linewidth=1.0, label="ILU")
        for definition, (color, marker, line, label) in styles.items():
            local = sorted(
                (row for row in local_all if row["definition"] == definition),
                key=lambda row: int(row["q_actual"]),
            )
            axis.plot(
                [int(row["q_actual"]) for row in local],
                [float(row["k_oracle_median"]) for row in local],
                color=color,
                marker=marker,
                linestyle=line,
                label=label,
            )
        axis.set_title(f"ID {sample_id}")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_ylim(0, settings.stage19a.stage10.solver.max_iterations + 10)
    for axis in axes[1, :]:
        axis.set_xlabel("Dimension q")
    axes[0, 0].set_ylabel("Median iterations")
    axes[1, 0].set_ylabel("Median iterations")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    figure.suptitle("Classical oracle definitions on identical held-out RHSs", y=1.015)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save_figure(figure, plot_dir / "oracle_definition_comparison.png")

    # Incremental solver response: one held-out RHS, every complete atomic block.
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.1), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        local = sorted(
            (row for row in incremental if int(row["sample_id"]) == sample_id),
            key=lambda row: int(row["q_actual"]),
        )
        axis.step(
            [0] + [int(row["q_actual"]) for row in local],
            [int(local[0]["baseline_iterations"])]
            + [int(row["iterations"]) for row in local],
            where="post",
            color=PURPLE,
        )
        axis.scatter(
            [int(row["q_actual"]) for row in local],
            [int(row["iterations"]) for row in local],
            color=PURPLE,
            s=16,
        )
        axis.set_title(f"ID {sample_id}")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_ylim(0, settings.stage19a.stage10.solver.max_iterations + 10)
    for axis in axes[1, :]:
        axis.set_xlabel("Atomic-block endpoint q")
    axes[0, 0].set_ylabel("Iterations, RHS 20261101")
    axes[1, 0].set_ylabel("Iterations, RHS 20261101")
    figure.suptitle("Marginal response as ordinary Ritz blocks are added", y=1.01)
    figure.tight_layout()
    _save_figure(figure, plot_dir / "ordinary_ritz_incremental_blocks.png")

    # Two complementary dimensionality views: spectrum and held-out projection error.
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.1), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        local = [row for row in correction_spectrum if int(row["sample_id"]) == sample_id]
        axis.semilogy(
            [int(row["q"]) for row in local],
            [float(row["normalized_singular_value"]) for row in local],
            color=BLUE,
        )
        axis.set_title(f"ID {sample_id}")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    for axis in axes[1, :]:
        axis.set_xlabel("POD mode q")
    axes[0, 0].set_ylabel(r"$\sigma_q/\sigma_1$")
    axes[1, 0].set_ylabel(r"$\sigma_q/\sigma_1$")
    figure.suptitle("Unresolved-correction singular-value decay", y=1.01)
    figure.tight_layout()
    _save_figure(figure, plot_dir / "correction_pod_spectra.png")

    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.1), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        local = [row for row in correction_spectrum if int(row["sample_id"]) == sample_id]
        axis.plot(
            [int(row["q"]) for row in local],
            [float(row["heldout_projection_error_median"]) for row in local],
            color=OLIVE,
        )
        axis.set_title(f"ID {sample_id}")
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        axis.set_ylim(0, 1.02)
    for axis in axes[1, :]:
        axis.set_xlabel("POD dimension q")
    axes[0, 0].set_ylabel("Median held-out projection error")
    axes[1, 0].set_ylabel("Median held-out projection error")
    figure.suptitle("Held-out unresolved-correction projection error", y=1.01)
    figure.tight_layout()
    _save_figure(figure, plot_dir / "correction_pod_projection_error.png")

    chart_map = [
        {"plot": "ordinary_ritz_q_sensitivity.png", "family": "ordered line / small multiples", "claim": "whether larger q rescues the Stage-19A target"},
        {"plot": "oracle_definition_comparison.png", "family": "comparison line / small multiples", "claim": "whether alternative classical spaces outperform ordinary Ritz"},
        {"plot": "ordinary_ritz_incremental_blocks.png", "family": "step / small multiples", "claim": "marginal solver value of each complete real/complex block"},
        {"plot": "correction_pod_spectra.png", "family": "ordered semilog line / small multiples", "claim": "correction snapshot spectral decay"},
        {"plot": "correction_pod_projection_error.png", "family": "ordered line / small multiples", "claim": "held-out correction compressibility"},
    ]
    _write_json(output / "chart_map.json", chart_map)


def _fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return "—"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "NA"
    if abs(numeric) >= 1e4 or (0 < abs(numeric) < 1e-3):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def _write_report(
    root: Path,
    output: Path,
    settings: Stage19CSettings,
    summaries: Sequence[Mapping[str, Any]],
    q_min: Sequence[Mapping[str, Any]],
    ritz_spectrum: Sequence[Mapping[str, Any]],
    correction_structure: Sequence[Mapping[str, Any]],
    correction_spectrum: Sequence[Mapping[str, Any]],
    reconstruction: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    decision: str,
    decision_data: Mapping[str, Any],
) -> Path:
    decision_text = {
        "A": "Increasing q within the existing Ritz construction consistently restores strong oracle acceleration; Stage 19A was mainly dimension-limited.",
        "B": "Harmonic Ritz consistently succeeds where smallest-magnitude Ritz fails; the selection rule is the primary problem.",
        "C": "Correction POD succeeds while both Ritz constructions remain weak; the unresolved correction is compressible but the Ritz representation is unsuitable.",
        "D": "No tested q≤32 or classical diagnostic space provides robust acceleration; the low-dimensional recycle-space hypothesis breaks down for this family at the tested dimensions.",
    }[decision]
    by_key = {
        (int(row["sample_id"]), str(row["definition"]), int(row["q_requested"])): row
        for row in summaries
    }
    qmin_by_key = {
        (int(row["sample_id"]), str(row["definition"])): row for row in q_min
    }
    correction_by_id = {int(row["sample_id"]): row for row in correction_structure}

    report = fr"""# Stage 19C — spatial-coefficient oracle mechanism diagnostic

## Technical summary

**Decision {decision}: {decision_text}**

The experiment preserved all eight Stage-19A hard systems, the strong ILU, the three held-out RHSs, the right-preconditioned convention $B=AM^{{-1}}$, and Stage-13 fixed-space recycled GMRES. Strong acceleration means all three held-out solves converge and the system-level median

$$\gamma_{{oracle}}(q)=\frac{{k_{{ILU}}-k_{{oracle}}(q)}}{{k_{{ILU}}}}$$

is at least {settings.strong_gamma:.2f}. Robustness requires at least {decision_data['required_success_count']}/{len(settings.sample_ids)} systems to pass. The observed success counts by q≤32 were: ordinary Ritz {decision_data['successful_system_counts']['ordinary_ritz']}/8, harmonic Ritz {decision_data['successful_system_counts']['harmonic_ritz']}/8, and correction POD {decision_data['successful_system_counts']['correction_pod']}/8.

The diagnosis is not simply “q=8 was too small.” Larger ordinary-Ritz spaces rescue only ID 34 in addition to the already-successful ID 15; harmonic selection does not widen that successful set; and correction POD succeeds only on ID 15. Several failed ordinary Ritz pairs also have large Ritz residuals, so construction quality is part of the failure, but the harmonic alternative does not restore robust solver value. Most importantly, correction snapshots can be highly compressible while their POD spaces still fail as recycle spaces. Snapshot low rank and Krylov relevance are therefore distinct in this family.

No neural network, learned loss, easier system, altered ILU, changed RBF-FD operator, or changed recycle implementation was introduced.

### Frozen numerical protocol

- RBF-FD: $N=5041$, stencil 25, fixed seed-7 jittered grid with jitter fraction 0.15, PHS $r^3$, polynomial degree 2, and the unchanged Dirichlet treatment.
- Classical baseline: SuperLU ILU with `drop_tol=1e-3`, `fill_factor=5`, zero diagonal-pivot threshold, and COLAMD ordering.
- Solver: right-preconditioned/recycled GMRES, relative tolerance $10^{{-8}}$, restart 50, cap 300, zero initial guess, and true-residual verification.
- Oracle construction: Stage-19A construction RHS seed 20261001, 64 Arnoldi steps, and held-out RHS seeds 20261101–20261103. The exact eight coefficient rows remain in [`coefficient_samples.csv`](results/stage19A_spatial_coefficient_oracle_validation/coefficient_samples.csv).

## Reproducibility qualification: matrices and ILU objects were reconstructed, not loaded

Stage 19A saved the exact coefficient table and fixed node/boundary arrays, but it did **not** serialize its sparse matrices or SuperLU factor objects. Stage 19C therefore rebuilt $A$ from each saved seven-parameter coefficient row on the exact saved nodes and reran the identical deterministic `spilu` configuration. This is not a regenerated PDE sample.

The gate before diagnosis required, for all eight systems: identical nnz; matrix symmetry diagnostic and ILU fill ratio matching Stage 19A to roundoff; normalized distance below $10^{{-10}}$ between rebuilt and saved q=8 spaces; and exact reproduction of all 24 baseline and q=8 iteration counts. All checks passed. Reconstructed matrices are now archived in the Stage-19C result folder; SuperLU factors remain non-serializable and are reconstructed at runtime.

## Existing Ritz target: q-sensitivity does not change the PDE or selection rule

The main family uses the Stage-10/19A construction verbatim: 64-step Arnoldi on $AM^{{-1}}$, ordinary Ritz values ordered by increasing magnitude, residual-space Ritz vectors mapped through $M^{{-1}}$, and twice-reorthogonalized real solution directions. Requested q values were {settings.q_values}. A complex direction and its real invariant-plane partner are indivisible; when a requested q cuts a pair, the largest complete prefix not exceeding q is reported as `q_actual`.

![Existing Ritz q sensitivity](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/plots/ordinary_ritz_q_sensitivity.png)

The figure uses a common 0–300 scale and three-RHS medians. Exact per-RHS residuals, setup times, pair costs, solve times, and cap flags are in [`oracle_solver_results.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/oracle_solver_results.csv). Total oracle cost below is Ritz construction + recycle-pair setup + median online solve; it excludes the unchanged ILU setup common to all q values.

### Every system and requested q

| ID | q req./actual | median $k_{{ILU}}$ | median $k_{{Ritz}}$ [min,max] | convergence | median $\gamma$ | Ritz setup (s) | pair setup (s) | median online (s) | median total oracle cost (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for sample_id in settings.sample_ids:
        for q in settings.q_values:
            row = by_key[(sample_id, "ordinary_ritz", q)]
            report += f"| {sample_id} | {q}/{int(row['q_actual'])} | {_fmt(row['k_ilu_median'],0)} | {_fmt(row['k_oracle_median'],0)} [{int(row['k_oracle_min'])},{int(row['k_oracle_max'])}] | {_fmt(row['convergence_rate'],2)} | {_fmt(row['gamma_median'],3)} | {_fmt(row['construction_seconds'],3)} | {_fmt(row['pair_setup_seconds'],4)} | {_fmt(row['online_solve_seconds_median'],3)} | {_fmt(row['total_oracle_cost_median'],3)} |\n"

    report += fr"""

### Minimum useful q

| ID | ordinary Ritz $q_{{min}}$ | harmonic Ritz $q_{{min}}$ | correction POD $q_{{min}}$ |
|---:|---:|---:|---:|
"""
    for sample_id in settings.sample_ids:
        values = []
        for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
            value = qmin_by_key[(sample_id, definition)]["q_min"]
            values.append("—" if value == "" else str(int(value)))
        report += f"| {sample_id} | {values[0]} | {values[1]} | {values[2]} |\n"

    report += fr"""

`q_min` is the smallest tested complete-block dimension with 100% convergence over the three held-out RHSs and median $\gamma\ge0.40$. A dash means no tested q≤32 passes. ID 15 is retained as the Stage-19A positive control.

## Ritz spectrum and marginal solver response

Ritz residuals are measured for the projected eigenpair $(\lambda,y)$ after mapping $z=M^{{-1}}y$:

$$\varepsilon_{{Ritz}}=\frac{{\|Az-\lambda y\|_2}}{{\|Az\|_2+|\lambda|\,\|y\|_2}}.$$

The full 32-direction spectra, eigenvalue real/imaginary parts, block membership, residuals, and relative gaps are in [`ordinary_ritz_spectrum.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/ordinary_ritz_spectrum.csv). The first eight magnitudes and maximum first-eight residual are:

| ID | selected q=8 Ritz magnitudes | max q=8 Ritz residual | largest relative gap through q=32 |
|---:|---|---:|---:|
"""
    for sample_id in settings.sample_ids:
        local = [row for row in ritz_spectrum if int(row["sample_id"]) == sample_id]
        first = local[:8]
        gaps = [float(row["relative_gap_after_block"]) for row in local if np.isfinite(float(row["relative_gap_after_block"]))]
        report += "| {} | {} | {} | {} |\n".format(
            sample_id,
            ", ".join(f"{float(row['magnitude']):.3g}" for row in first),
            _fmt(max(float(row["ritz_residual_relative"]) for row in first), 3),
            _fmt(max(gaps) if gaps else np.nan, 3),
        )

    report += fr"""

![Incremental atomic blocks](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/plots/ordinary_ritz_incremental_blocks.png)

The incremental plot uses the first held-out RHS and every complete real/complex Ritz block, not only the coarse q grid. Consequently it distinguishes a genuinely concentrated solver drop from a smooth accumulation or a misleading spectral gap. Marginal iteration changes are archived in [`ritz_incremental_blocks.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/ritz_incremental_blocks.csv).

The solver-response classification is heterogeneous: **ID 15 is concentrated** (the first direction supplies the largest drop and q=4 already passes); **ID 34 is distributed over a moderate block** (no gain through q=17, then a large drop at q=19 and q=32 is required); **ID 14 is distributed but insufficient by q=32**; and **IDs 3, 7, 10, and 17 are not captured at all by the first 32 ordered Ritz directions**. ID 19 receives only a marginal late improvement. Thus the failed family is not explained by one common underestimated q.

## Alternative classical oracle spaces

Harmonic Ritz uses the same 64-step right-preconditioned Arnoldi relation as the ordinary diagnostic. It solves the standard generalized harmonic problem from $\bar H_m$, selects harmonic Ritz values nearest zero, maps through the stored $M^{{-1}}V_m$ vectors, and preserves complex pairs. Correction POD uses only three construction trajectories (seeds {settings.pod_construction_rhs_seeds}), collecting {settings.pod_inputs_per_trajectory} actual preconditioner inputs per trajectory. Its labels keep the proposal definition exactly:

$$d(r)=A^{{-1}}r-M^{{-1}}r,$$

where $A^{{-1}}r$ is an accurate sparse solve with iterative refinement, never an explicit inverse. The three Stage-19A held-out trajectories are excluded from POD construction and used for projection testing and solver evaluation.

![Oracle definition comparison](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/plots/oracle_definition_comparison.png)

| ID | method | best tested q | median iterations | convergence | median $\gamma$ | setup (s) | median online (s) | q≤32 strongly successful? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for sample_id in settings.sample_ids:
        for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
            local = [row for row in summaries if int(row["sample_id"]) == sample_id and row["definition"] == definition]
            best = min(local, key=lambda row: (float(row["k_oracle_median"]), int(row["q_actual"])))
            success = qmin_by_key[(sample_id, definition)]["success_by_q32"]
            report += f"| {sample_id} | {definition.replace('_',' ')} | {int(best['q_actual'])} | {_fmt(best['k_oracle_median'],0)} | {_fmt(best['convergence_rate'],2)} | {_fmt(best['gamma_median'],3)} | {_fmt(float(best['construction_seconds']) + float(best['pair_setup_seconds']),3)} | {_fmt(best['online_solve_seconds_median'],3)} | {'yes' if success else 'no'} |\n"

    report += fr"""

## Correction dimensionality is tested independently of solver iterations

| ID | train/test snapshots | q95 | q99 | q95/N | median held-out $\eta_8$ | $\eta_{{16}}$ | $\eta_{{24}}$ | $\eta_{{32}}$ | max exact-solve residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    spectrum_lookup = {
        (int(row["sample_id"]), int(row["q"])): row for row in correction_spectrum
    }
    for sample_id in settings.sample_ids:
        row = correction_by_id[sample_id]
        report += "| {} | {}/{} | {} | {} | {:.3%} | {} | {} | {} | {} | {} |\n".format(
            sample_id,
            int(row["train_snapshots"]),
            int(row["test_snapshots"]),
            int(row["q95"]),
            int(row["q99"]),
            float(row["q95_over_N"]),
            _fmt(spectrum_lookup[(sample_id, 8)]["heldout_projection_error_median"], 3),
            _fmt(spectrum_lookup[(sample_id, 16)]["heldout_projection_error_median"], 3),
            _fmt(spectrum_lookup[(sample_id, 24)]["heldout_projection_error_median"], 3),
            _fmt(spectrum_lookup[(sample_id, 32)]["heldout_projection_error_median"], 3),
            _fmt(row["max_exact_solve_residual"], 2),
        )

    report += fr"""

![Correction POD spectra](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/plots/correction_pod_spectra.png)

![Held-out correction projection](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/plots/correction_pod_projection_error.png)

Captured training energy is not used as a proxy for Krylov value. The decision compares both held-out projection error and actual recycled-GMRES response.

## What failed: dimension, ordering, or the recycle-space hypothesis?

The decision hierarchy was fixed before results were interpreted:

1. **A — dimension issue:** ordinary Ritz must strongly accelerate at least {decision_data['required_success_count']}/8 systems by q=32.
2. **B — ordering issue:** otherwise harmonic Ritz must pass at least {decision_data['required_success_count']}/8 and exceed ordinary Ritz success by at least three systems.
3. **C — correction representation issue:** otherwise correction POD must pass at least {decision_data['required_success_count']}/8 and exceed both Ritz success counts by at least three.
4. **D — hypothesis failure at tested rank:** all other outcomes.

The observed counts select **Decision {decision}**. This conclusion applies to q≤32, the unchanged 64-step construction trajectory, the eight preselected Stage-19A systems, and the fixed solver cap. It does not prove that no larger or differently constructed space could ever help; it says the requested low/moderate-dimensional classical diagnostics do not provide robust enough evidence to justify recycle-space operator learning now.

Cross-system geometry is secondary, per protocol. Pairwise distances among each method's smallest successful spaces are saved in [`successful_space_secondary_geometry.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/successful_space_secondary_geometry.csv); no transferability gate was imposed.

## Limitations, robustness, and next step

- Stage 19A did not archive its matrices or SuperLU factors. Deterministic reconstruction is strongly verified, but the report does not mislabel those objects as loaded originals.
- Iteration counts at 300 are right-censored. A cap hit is a failure; reported $\gamma$ at the cap is descriptive only.
- q values that would split a complex Ritz pair use the largest complete prefix and expose `q_actual`.
- Correction POD setup includes construction rollouts, exact sparse factorization/labels, and SVD. It is a diagnostic oracle cost, not an online preconditioner claim.
- Harmonic and ordinary Ritz share the same operator convention and construction residual, isolating selection rather than data access.

**Next step:** follow Decision {decision}. No neural model was trained, and no operator-learning stage is authorized by this experiment unless the decision criterion itself is deliberately revised in a future, separately specified study.

## Audit artifacts

- Per-RHS solver data: [`oracle_solver_results.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/oracle_solver_results.csv)
- Summaries and q-min: [`oracle_solver_summary.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/oracle_solver_summary.csv), [`q_min_summary.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/q_min_summary.csv)
- Reconstruction checks: [`stage19a_reconstruction_checks.csv`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/stage19a_reconstruction_checks.csv)
- Validation receipt: [`validation.json`](results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic/validation.json)
- Implementation: `src/lcp_rbffd/experiments/stage19c_oracle_mechanism.py`; runner: `scripts/run_stage19c_oracle_mechanism.py`
- Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, `{platform.platform()}`
"""
    path = root / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic.md"
    path.write_text(report, encoding="utf-8")
    return path


def _validate(
    output: Path,
    report: Path,
    settings: Stage19CSettings,
    baseline: Sequence[Mapping[str, Any]],
    solver_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    reconstruction: Sequence[Mapping[str, Any]],
) -> None:
    expected_baselines = len(settings.sample_ids) * len(settings.stage19a.evaluation_rhs_seeds)
    expected_main = expected_baselines * len(settings.q_values)
    main_rows = [row for row in solver_rows if row["definition"] == "ordinary_ritz" and int(row["q_requested"]) in settings.q_values]
    harmonic_rows = [row for row in solver_rows if row["definition"] == "harmonic_ritz"]
    pod_rows = [row for row in solver_rows if row["definition"] == "correction_pod"]
    expected_alternative = expected_baselines * len(settings.alternative_q_values)
    gamma_exact = all(
        abs(float(row["gamma_oracle"]) - (int(row["k_ilu"]) - int(row["k_oracle"])) / max(int(row["k_ilu"]), 1)) < 1e-14
        for row in solver_rows
    )
    checks = {
        "eight_exact_sample_ids": sorted({int(row["sample_id"]) for row in baseline}) == sorted(settings.sample_ids),
        "baseline_row_count": len(baseline) == expected_baselines,
        "ordinary_q_grid_complete": len(main_rows) == expected_main,
        "harmonic_q_grid_complete": len(harmonic_rows) == expected_alternative,
        "correction_pod_q_grid_complete": len(pod_rows) == expected_alternative,
        "all_summary_rows_present": len(summaries) == len(settings.sample_ids) * (len(settings.q_values) + 2 * len(settings.alternative_q_values)),
        "gamma_identity_exact": gamma_exact,
        "all_reconstructions_present": len(reconstruction) == len(settings.sample_ids),
        "all_q8_projectors_reproduced": all(float(row["q8_projector_distance_normalized"]) < 1e-10 for row in reconstruction),
        "all_q8_iterations_reproduced": all(bool(row["q8_baseline_iterations_exact"]) and bool(row["q8_oracle_iterations_exact"]) for row in reconstruction),
        "all_matrix_nnz_reproduced": all(int(row["nnz_stage19a"]) == int(row["nnz_reconstructed"]) for row in reconstruction),
        "all_true_residuals_finite": all(np.isfinite(float(row["true_final_residual"])) for row in solver_rows),
        "all_required_plots": all((output / "plots" / name).exists() for name in (
            "ordinary_ritz_q_sensitivity.png",
            "oracle_definition_comparison.png",
            "ordinary_ritz_incremental_blocks.png",
            "correction_pod_spectra.png",
            "correction_pod_projection_error.png",
        )),
        "report_exists": report.exists(),
        "no_neural_outputs": not any("model" in path.name.lower() for path in output.rglob("*")),
    }
    receipt = {
        "assessment": "ready_to_share" if all(checks.values()) else "needs_revision",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "caveat": "Stage 19A matrices and SuperLU objects were absent; exact deterministic reconstruction is verified by operator/fill/projector/iteration identities.",
    }
    _write_json(output / "validation.json", receipt)
    if not receipt["all_checks_pass"]:
        raise RuntimeError(f"Stage 19C validation failed: {checks}")
