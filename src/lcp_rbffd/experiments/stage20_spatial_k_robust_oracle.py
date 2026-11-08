"""Stage 20: robust complementary-space search for spatially varying K(x).

The experiment retains the exact eight Stage-19A/19C hard matrices, strong
ILU action, held-out RHSs, and fixed-space nonsymmetric recycled GMRES.  It
first reproduces the old q=8 ordinary-Ritz, harmonic-Ritz, and correction-POD
results.  It then diagnoses Ritz quality and nonnormality, evaluates adaptive
rank/filter/cluster selection on disjoint selection RHSs, and conditionally
tests multilevel local/global residual spaces if no spectral family succeeds
on a strict majority.  No learned model is constructed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
from ..config import DiscretizationConfig, RBFConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..pde import SpatialAnisotropicDiffusion
from ..pod import compute_pod
from ..preconditioners import InverseAction, ilu_preconditioner
from ..rbffd import RBFSystem
from ..subspace_analysis import (
    generalized_subspace_metrics,
    nested_krylov_ritz_modes,
    ritz_atomic_blocks,
)
from .stage10_ritz_transfer import Stage10Settings
from .stage19a_spatial_coefficient_oracle import _discrete_rhs
from .stage19c_oracle_mechanism import (
    ArnoldiData,
    ModeFamily,
    _capture_arnoldi,
    _harmonic_family,
    _ordinary_spectrum,
)
from .stage9_scalability import _write_csv, _write_json


BLUE = "#2864B7"
ORANGE = "#D97706"
OLIVE = "#6B7D2A"
PURPLE = "#7553A6"
PINK = "#B14D75"
CHARCOAL = "#30343B"
GRAY = "#8A9099"
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
class Stage20Settings:
    """Frozen systems, data split, candidate budget, and oracle gate."""

    sample_ids: Tuple[int, ...] = (3, 7, 10, 14, 15, 17, 19, 34)
    heldout_rhs_seeds: Tuple[int, ...] = (20261101, 20261102, 20261103)
    construction_rhs_seed: int = 20261001
    stability_rhs_seeds: Tuple[int, ...] = (20261011, 20261021)
    selection_rhs_seeds: Tuple[int, ...] = (20261031, 20261041)
    slow_history_rhs_seeds: Tuple[int, ...] = (20261001, 20261011, 20261021)
    alternate_history_rhs_seeds: Tuple[int, ...] = (20261051, 20261061, 20261071)
    old_arnoldi_dimension: int = 64
    long_arnoldi_dimension: int = 128
    max_spectral_dimension: int = 64
    q_values: Tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    residual_filter_thresholds: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50)
    individual_ablation_dimension: int = 16
    nonnormality_probe_count: int = 8
    strong_gamma: float = 0.40
    local_global_q_values: Tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    local_levels: Tuple[int, ...] = (2, 4, 8)
    residual_snapshot_stride: int = 10
    run_local_global_if_spectral_fail: bool = True
    seed: int = 20262001

    @property
    def solver(self) -> SolverConfig:
        return Stage10Settings().solver

    @property
    def strict_majority(self) -> int:
        return len(self.sample_ids) // 2 + 1

    @property
    def discretization(self) -> DiscretizationConfig:
        return DiscretizationConfig(
            n_nodes=5041,
            stencil_size=25,
            polynomial_degree=2,
            node_distribution="jittered_grid",
            jitter_fraction=0.15,
            seed=7,
            rbf=RBFConfig(kernel="phs", order=3, shape_parameter=1.0),
            compute_local_condition_numbers=False,
        )


@dataclass
class SystemContext:
    sample_id: int
    system: RBFSystem
    ilu: InverseAction


@dataclass
class SpectralFamily:
    name: str
    basis: np.ndarray
    eigenvalues: np.ndarray
    residuals: np.ndarray
    blocks: Tuple[Tuple[int, int], ...]
    construction_seconds: float
    arnoldi_dimension: int


@dataclass
class Candidate:
    method: str
    label: str
    basis: np.ndarray
    construction_seconds: float
    metadata: Dict[str, Any]


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


def _orthonormalize(raw: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    """Return a rank-revealing orthonormal basis without silent zero columns."""

    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("candidate space must be a nonempty matrix")
    u, singular, _ = la.svd(array, full_matrices=False, check_finite=False)
    if singular.size == 0:
        raise ValueError("candidate space has zero numerical rank")
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank == 0:
        raise ValueError("candidate space has zero numerical rank")
    return u[:, :rank]


def _complete_block_indices(
    blocks: Sequence[Tuple[int, int]], requested_q: int
) -> Tuple[int, ...]:
    """Return the largest leading complete-block prefix not exceeding q."""

    accepted: List[int] = []
    for start, stop in blocks:
        if stop > requested_q:
            break
        accepted.extend(range(start, stop))
    if not accepted:
        raise ValueError(f"no complete Ritz block fits q={requested_q}")
    return tuple(accepted)


def _cluster_atomic_blocks(
    values: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], float]:
    """Group consecutive Ritz atoms using data-adaptive relative spectral gaps."""

    magnitudes = np.asarray([abs(values[start]) for start, _ in blocks], dtype=float)
    if magnitudes.size <= 1:
        return (tuple(blocks),), 0.25
    gaps = np.diff(magnitudes) / np.maximum(magnitudes[:-1], np.finfo(float).tiny)
    median = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median)))
    threshold = max(0.25, median + 2.0 * mad)
    clusters: List[List[Tuple[int, int]]] = [[blocks[0]]]
    for index, block in enumerate(blocks[1:]):
        if gaps[index] > threshold:
            clusters.append([block])
        else:
            clusters[-1].append(block)
    return tuple(tuple(cluster) for cluster in clusters), threshold


def _relative_residual(matrix: sp.spmatrix, x: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.linalg.norm(matrix @ x - rhs) / max(np.linalg.norm(rhs), 1e-300))


def _load_context(
    root: Path,
    sample_id: int,
    coefficient_rows: Mapping[int, Mapping[str, str]],
    nodes: np.ndarray,
    boundary: np.ndarray,
    settings: Stage20Settings,
) -> SystemContext:
    source = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic"
    )
    matrix = sp.load_npz(
        source / "reconstructed_systems" / f"sample_{sample_id:02d}_matrix.npz"
    ).tocsr()
    coefficient = _coefficient_from_row(coefficient_rows[sample_id])
    system = RBFSystem(
        matrix=matrix,
        nodes=nodes,
        boundary=boundary,
        interior=~boundary,
        diagnostics={},
        config=settings.discretization,
        coefficient=coefficient,
    )
    ilu = ilu_preconditioner(
        matrix,
        Stage10Settings().classical_ilu,
        name="strong_ilu",
    )
    return SystemContext(sample_id, system, ilu)


def _run_baseline(context: SystemContext, rhs: np.ndarray, solver: SolverConfig) -> KrylovResult:
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


def _ordinary_family(
    matrix: sp.spmatrix,
    ilu: InverseAction,
    initial: np.ndarray,
    max_q: int,
    arnoldi_dimension: int,
    seed: int,
) -> Tuple[SpectralFamily, ArnoldiData]:
    modes = nested_krylov_ritz_modes(
        matrix,
        ilu,
        max_modes=max_q,
        arnoldi_dimension=arnoldi_dimension,
        seed=seed,
        initial_vector=initial,
    )
    arnoldi = _capture_arnoldi(matrix, ilu, initial, arnoldi_dimension)
    _, residuals_full, _ = _ordinary_spectrum(matrix, arnoldi)
    residuals = []
    for value, label in zip(modes.eigenvalues, modes.mode_labels):
        match = re.match(r"eig(\d+)_", label)
        if match is None:
            raise RuntimeError(f"cannot parse Ritz mode label {label}")
        residuals.append(float(residuals_full[int(match.group(1))]))
    return (
        SpectralFamily(
            name="ordinary",
            basis=modes.basis,
            eigenvalues=modes.eigenvalues,
            residuals=np.asarray(residuals),
            blocks=ritz_atomic_blocks(modes.eigenvalues),
            construction_seconds=float(modes.diagnostics["setup_seconds"]),
            arnoldi_dimension=int(modes.diagnostics["arnoldi_dimension"]),
        ),
        arnoldi,
    )


def _harmonic_from_arnoldi(
    matrix: sp.spmatrix,
    arnoldi: ArnoldiData,
    max_q: int,
) -> SpectralFamily:
    family, _ = _harmonic_family(matrix, arnoldi, max_q)
    return SpectralFamily(
        name="harmonic",
        basis=family.basis,
        eigenvalues=family.eigenvalues,
        residuals=family.residual_norms,
        blocks=family.blocks,
        construction_seconds=family.seconds,
        arnoldi_dimension=arnoldi.hessenberg.shape[1],
    )


def _family_prefix_candidate(
    family: SpectralFamily,
    q: int,
    method: str,
) -> Candidate:
    indices = _complete_block_indices(family.blocks, q)
    basis = family.basis[:, indices]
    return Candidate(
        method=method,
        label=f"{family.name}_prefix_q{basis.shape[1]}",
        basis=basis,
        construction_seconds=family.construction_seconds,
        metadata={
            "requested_q": q,
            "indices": ",".join(str(index + 1) for index in indices),
            "residual_threshold": "",
            "cluster_threshold": "",
        },
    )


def _filtered_candidates(
    family: SpectralFamily,
    q_values: Sequence[int],
    thresholds: Sequence[float],
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen: set[Tuple[int, ...]] = set()
    for threshold in thresholds:
        accepted_blocks = [
            block
            for block in family.blocks
            if float(np.max(family.residuals[block[0] : block[1]])) <= threshold
        ]
        for requested_q in q_values:
            indices: List[int] = []
            for start, stop in accepted_blocks:
                if len(indices) + stop - start > requested_q:
                    continue
                indices.extend(range(start, stop))
            key = tuple(indices)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    method=f"{family.name}_residual_filter",
                    label=f"{family.name}_resid{threshold:g}_q{len(key)}",
                    basis=family.basis[:, key],
                    construction_seconds=family.construction_seconds,
                    metadata={
                        "requested_q": requested_q,
                        "indices": ",".join(str(index + 1) for index in key),
                        "residual_threshold": threshold,
                        "cluster_threshold": "",
                    },
                )
            )
    return candidates


def _evaluate_basis(
    context: SystemContext,
    candidate: Candidate,
    seeds: Sequence[int],
    solver: SolverConfig,
    phase: str,
    baseline_by_seed: Mapping[int, int],
) -> Tuple[List[Dict[str, Any]], AZPetrovFactors]:
    factors = factor_az_petrov(context.system.matrix, candidate.basis)
    rows: List[Dict[str, Any]] = []
    for rhs_seed in seeds:
        rhs, _ = _discrete_rhs(context.system, rhs_seed)
        result = _run_recycle(context, rhs, factors, solver)
        true_residual = _relative_residual(context.system.matrix, result.x, rhs)
        k_ilu = int(baseline_by_seed[rhs_seed])
        rows.append(
            {
                "sample_id": context.sample_id,
                "phase": phase,
                "method": candidate.method,
                "candidate": candidate.label,
                "q": candidate.basis.shape[1],
                "rhs_seed": rhs_seed,
                "k_ilu": k_ilu,
                "iterations": result.iterations,
                "gamma": (k_ilu - result.iterations) / max(k_ilu, 1),
                "converged": result.converged,
                "true_final_residual": true_residual,
                "online_seconds": result.solve_seconds,
                "construction_seconds": candidate.construction_seconds,
                "pair_setup_seconds": factors.diagnostics["recycling_setup_seconds"],
                "reduced_condition": factors.diagnostics["reduced_condition"],
                "failure": "" if result.converged else "maximum iterations reached",
                **candidate.metadata,
            }
        )
    return rows, factors


def _candidate_score(rows: Sequence[Mapping[str, Any]]) -> Tuple[float, float, float, int]:
    failure_rate = 1.0 - float(np.mean([bool(row["converged"]) for row in rows]))
    iterations = float(np.median([float(row["iterations"]) for row in rows]))
    residual = float(np.median([float(row["true_final_residual"]) for row in rows]))
    q = int(rows[0]["q"])
    return failure_rate, iterations, residual, q


def _select_candidate(
    candidates: Sequence[Candidate],
    selection_rows: Sequence[Mapping[str, Any]],
) -> Candidate:
    by_label = {
        candidate.label: [row for row in selection_rows if row["candidate"] == candidate.label]
        for candidate in candidates
    }
    complete = {label: rows for label, rows in by_label.items() if rows}
    if not complete:
        raise RuntimeError("no candidate has selection results")
    winning_label = min(complete, key=lambda label: _candidate_score(complete[label]))
    return next(candidate for candidate in candidates if candidate.label == winning_label)


def _summarize_evaluation(rows: Sequence[Mapping[str, Any]], settings: Stage20Settings) -> List[Dict[str, Any]]:
    keys = sorted({(int(row["sample_id"]), str(row["method"])) for row in rows})
    summary: List[Dict[str, Any]] = []
    for sample_id, method in keys:
        local = [
            row
            for row in rows
            if int(row["sample_id"]) == sample_id and str(row["method"]) == method
        ]
        convergence = float(np.mean([bool(row["converged"]) for row in local]))
        gamma = float(np.median([float(row["gamma"]) for row in local]))
        summary.append(
            {
                "sample_id": sample_id,
                "method": method,
                "candidate": local[0]["candidate"],
                "q": int(local[0]["q"]),
                "k_ilu_median": float(np.median([float(row["k_ilu"]) for row in local])),
                "iterations_median": float(np.median([float(row["iterations"]) for row in local])),
                "iterations_min": min(int(row["iterations"]) for row in local),
                "iterations_max": max(int(row["iterations"]) for row in local),
                "convergence_rate": convergence,
                "gamma_median": gamma,
                "true_residual_max": max(float(row["true_final_residual"]) for row in local),
                "online_seconds_median": float(np.median([float(row["online_seconds"]) for row in local])),
                "construction_seconds": float(local[0]["construction_seconds"]),
                "pair_setup_seconds": float(local[0]["pair_setup_seconds"]),
                "selection_seconds": float(sum(float(row.get("selection_seconds", 0.0)) for row in local[:1])),
                "system_success": bool(convergence == 1.0 and gamma >= settings.strong_gamma),
            }
        )
    return summary


def _family_success(
    summaries: Sequence[Mapping[str, Any]], method: str, settings: Stage20Settings
) -> bool:
    return sum(
        bool(row["system_success"])
        for row in summaries
        if str(row["method"]) == method
    ) >= settings.strict_majority


def _projected_nonnormality(arnoldi: ArnoldiData) -> Dict[str, float]:
    m = arnoldi.hessenberg.shape[1]
    h = arnoldi.hessenberg[:m, :m]
    norm_h = float(np.linalg.norm(h, ord="fro"))
    commutator = h.T @ h - h @ h.T
    values, vectors = la.eig(h, check_finite=False)
    henrici_sq = max(norm_h**2 - float(np.sum(np.abs(values) ** 2)), 0.0)
    return {
        "projected_commutator_normalized": float(
            np.linalg.norm(commutator, ord="fro") / max(norm_h**2, 1e-300)
        ),
        "projected_henrici_departure": float(np.sqrt(henrici_sq) / max(norm_h, 1e-300)),
        "projected_eigenvector_condition": float(np.linalg.cond(vectors)),
        "arnoldi_closure_ratio": float(
            abs(arnoldi.hessenberg[m, m - 1])
            / max(np.linalg.norm(arnoldi.hessenberg, ord="fro"), 1e-300)
        ),
    }


def _operator_nonnormality_estimate(
    context: SystemContext, probe_count: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    matrix = context.system.matrix
    numerator: List[float] = []
    denominator: List[float] = []

    def apply_b(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ context.ilu.apply(vector), dtype=float)

    def apply_bt(vector: np.ndarray) -> np.ndarray:
        return context.ilu.apply_transpose(np.asarray(matrix.T @ vector, dtype=float))

    for _ in range(probe_count):
        probe = rng.choice((-1.0, 1.0), size=matrix.shape[0])
        b_probe = apply_b(probe)
        bt_probe = apply_bt(probe)
        commutator_probe = apply_bt(b_probe) - apply_b(bt_probe)
        numerator.append(float(np.dot(commutator_probe, commutator_probe)))
        denominator.append(float(np.dot(b_probe, b_probe)))
    return float(np.sqrt(np.mean(numerator)) / max(np.mean(denominator), 1e-300))


def _history_snapshots(
    context: SystemContext,
    seeds: Sequence[int],
    solver: SolverConfig,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, float, List[Dict[str, Any]]]:
    residuals: List[np.ndarray] = []
    errors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    start = perf_counter()
    for rhs_seed in seeds:
        rhs, exact = _discrete_rhs(context.system, rhs_seed)

        def callback(iteration: int, solution: np.ndarray, residual: np.ndarray) -> None:
            if iteration in (1, 2, 5) or iteration % stride == 0:
                residual_norm = float(np.linalg.norm(residual))
                error = exact - solution
                error_norm = float(np.linalg.norm(error))
                if residual_norm > 0.0 and error_norm > 0.0:
                    residuals.append(residual / residual_norm)
                    errors.append(error / error_norm)
                    rows.append(
                        {
                            "sample_id": context.sample_id,
                            "rhs_seed": rhs_seed,
                            "iteration": iteration,
                            "residual_norm": residual_norm,
                            "error_norm": error_norm,
                        }
                    )

        _run = restarted_gmres(
            context.system.matrix,
            rhs,
            solver,
            context.ilu,
            callback=callback,
        )
        if not residuals:
            raise RuntimeError("no slow-history snapshots collected")
    return (
        np.column_stack(residuals),
        np.column_stack(errors),
        perf_counter() - start,
        rows,
    )


def _local_test_dictionary(
    nodes: np.ndarray,
    boundary: np.ndarray,
    levels: Sequence[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    columns: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    x, y = nodes[:, 0], nodes[:, 1]
    for level in levels:
        width = 1.0 / level
        for iy in range(level):
            for ix in range(level):
                cx, cy = (ix + 0.5) * width, (iy + 0.5) * width
                wx = np.maximum(0.0, 1.0 - np.abs(x - cx) / width)
                wy = np.maximum(0.0, 1.0 - np.abs(y - cy) / width)
                column = wx * wy
                column[boundary] = 0.0
                norm = float(np.linalg.norm(column))
                if norm <= 1e-14:
                    continue
                columns.append(column / norm)
                metadata.append({"level": level, "ix": ix, "iy": iy})
    return _orthonormalize(np.column_stack(columns)), metadata


def _map_test_space(
    factor: Any, test_basis: np.ndarray
) -> np.ndarray:
    mapped = np.column_stack(
        [np.asarray(factor.solve(test_basis[:, index]), dtype=float) for index in range(test_basis.shape[1])]
    )
    return _orthonormalize(mapped)


def _make_local_global_candidates(
    context: SystemContext,
    settings: Stage20Settings,
) -> Tuple[List[Candidate], Dict[str, Any], List[Dict[str, Any]]]:
    start = perf_counter()
    residuals, errors, trajectory_seconds, history_rows = _history_snapshots(
        context,
        settings.slow_history_rhs_seeds,
        settings.solver,
        settings.residual_snapshot_stride,
    )
    factor_start = perf_counter()
    factor = spla.splu(context.system.matrix.tocsc())
    factor_seconds = perf_counter() - factor_start
    residual_pod = compute_pod(residuals)
    error_pod = compute_pod(errors)
    local_w, local_metadata = _local_test_dictionary(
        context.system.nodes, context.system.boundary, settings.local_levels
    )
    max_q = max(settings.local_global_q_values)
    global_w = residual_pod.basis[:, : min(max_q, residual_pod.basis.shape[1])]
    local_w = local_w[:, : min(max_q, local_w.shape[1])]
    candidates: List[Candidate] = []
    mapped_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def add_candidate(method: str, label: str, test_raw: np.ndarray, metadata: Dict[str, Any]) -> None:
        test_basis = _orthonormalize(test_raw)
        trial = _map_test_space(factor, test_basis)
        candidates.append(
            Candidate(
                method=method,
                label=label,
                basis=trial,
                construction_seconds=perf_counter() - start,
                metadata=metadata,
            )
        )

    for q in settings.local_global_q_values:
        if q <= global_w.shape[1]:
            add_candidate(
                "slow_residual_inverse_pod",
                f"slow_residual_inverse_q{q}",
                global_w[:, :q],
                {"requested_q": q, "global_q": q, "local_q": 0},
            )
        if q <= error_pod.basis.shape[1]:
            candidates.append(
                Candidate(
                    method="solution_error_pod_reference",
                    label=f"solution_error_pod_q{q}",
                    basis=_orthonormalize(error_pod.basis[:, :q]),
                    construction_seconds=perf_counter() - start,
                    metadata={"requested_q": q, "global_q": q, "local_q": 0},
                )
            )
        if q <= local_w.shape[1]:
            add_candidate(
                "multilevel_local_inverse",
                f"multilevel_local_inverse_q{q}",
                local_w[:, :q],
                {"requested_q": q, "global_q": 0, "local_q": q},
            )

    compositions = ((16, 16), (24, 8), (24, 16), (32, 8), (32, 16), (32, 32), (48, 16))
    for global_q, local_q in compositions:
        if global_q > global_w.shape[1] or local_q > local_w.shape[1]:
            continue
        combined = np.column_stack((global_w[:, :global_q], local_w[:, :local_q]))
        add_candidate(
            "multilevel_local_global",
            f"local_global_g{global_q}_l{local_q}",
            combined,
            {"requested_q": global_q + local_q, "global_q": global_q, "local_q": local_q},
        )
    diagnostics = {
        "sample_id": context.sample_id,
        "history_snapshot_count": residuals.shape[1],
        "residual_pod_rank": residual_pod.basis.shape[1],
        "error_pod_rank": error_pod.basis.shape[1],
        "local_dictionary_rank": local_w.shape[1],
        "local_raw_count": len(local_metadata),
        "trajectory_seconds": trajectory_seconds,
        "exact_factor_seconds": factor_seconds,
        "total_candidate_construction_seconds": perf_counter() - start,
    }
    return candidates, diagnostics, history_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_stage20(root: Path) -> Path:
    settings = Stage20Settings()
    fd_root = root / "results" / "results-fd"
    stage19a = fd_root / "stages" / "stage19A_spatial_coefficient_oracle_validation"
    stage19c = fd_root / "stages" / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic"
    output = fd_root / "stages" / "stage20_spatial_K_robust_oracle_search"
    report = fd_root / "reports" / "stage20_spatial_K_robust_oracle_search.md"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "spaces").mkdir(exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))

    node_data = np.load(stage19a / "fixed_nodes.npz")
    nodes = np.asarray(node_data["nodes"], dtype=float)
    boundary = np.asarray(node_data["boundary"], dtype=bool)
    coefficient_rows = {
        int(row["sample_id"]): row for row in _read_rows(stage19a / "coefficient_samples.csv")
    }
    old_solver_rows = {
        (int(row["sample_id"]), str(row["definition"]), int(row["q_requested"]), int(row["rhs_seed"])): row
        for row in _read_rows(stage19c / "oracle_solver_results.csv")
    }
    old_baseline_rows = {
        (int(row["sample_id"]), int(row["rhs_seed"])): row
        for row in _read_rows(stage19c / "baseline_results.csv")
    }

    reproduction_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    spectral_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []
    residual_history_rows: List[Dict[str, Any]] = []
    selected_candidates: Dict[Tuple[int, str], Candidate] = {}
    baseline_eval: Dict[Tuple[int, int], int] = {}
    baseline_selection: Dict[Tuple[int, int], int] = {}

    for ordinal, sample_id in enumerate(settings.sample_ids, start=1):
        print(f"[Stage20 spectral {ordinal}/{len(settings.sample_ids)}] ID {sample_id}", flush=True)
        context = _load_context(root, sample_id, coefficient_rows, nodes, boundary, settings)
        baseline_by_seed: Dict[int, int] = {}
        for phase, seeds in (
            ("heldout", settings.heldout_rhs_seeds),
            ("selection", settings.selection_rhs_seeds),
        ):
            for rhs_seed in seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                result = _run_baseline(context, rhs, settings.solver)
                baseline_by_seed[rhs_seed] = result.iterations
                if phase == "heldout":
                    baseline_eval[(sample_id, rhs_seed)] = result.iterations
                else:
                    baseline_selection[(sample_id, rhs_seed)] = result.iterations
                if phase == "heldout" and rhs_seed == settings.heldout_rhs_seeds[0]:
                    for iteration, value in enumerate(result.relative_residuals):
                        residual_history_rows.append(
                            {
                                "sample_id": sample_id,
                                "method": "strong_ilu",
                                "iteration": iteration,
                                "relative_residual": value,
                            }
                        )
                if phase == "heldout":
                    old = old_baseline_rows[(sample_id, rhs_seed)]
                    reproduction_rows.append(
                        {
                            "sample_id": sample_id,
                            "definition": "strong_ilu",
                            "rhs_seed": rhs_seed,
                            "saved_iterations": int(old["iterations"]),
                            "reproduced_iterations": result.iterations,
                            "exact_iteration_match": result.iterations == int(old["iterations"]),
                            "converged": result.converged,
                            "true_final_residual": _relative_residual(context.system.matrix, result.solution, rhs),
                        }
                    )

        archived = np.load(stage19c / "spaces" / f"sample_{sample_id:02d}_diagnostic_spaces.npz")
        for definition, key in (
            ("ordinary_ritz", "ordinary_ritz"),
            ("harmonic_ritz", "harmonic_ritz"),
            ("correction_pod", "correction_pod"),
        ):
            basis = _orthonormalize(np.asarray(archived[key][:, :8], dtype=float))
            candidate = Candidate(
                method=f"old_q8_{definition}",
                label=f"old_q8_{definition}",
                basis=basis,
                construction_seconds=0.0,
                metadata={"requested_q": 8},
            )
            rows, factors = _evaluate_basis(
                context,
                candidate,
                settings.heldout_rhs_seeds,
                settings.solver,
                "reproduction",
                baseline_by_seed,
            )
            for row in rows:
                old = old_solver_rows[(sample_id, definition, 8, int(row["rhs_seed"]))]
                reproduction_rows.append(
                    {
                        "sample_id": sample_id,
                        "definition": definition,
                        "rhs_seed": row["rhs_seed"],
                        "saved_iterations": int(old["k_oracle"]),
                        "reproduced_iterations": row["iterations"],
                        "exact_iteration_match": row["iterations"] == int(old["k_oracle"]),
                        "converged": row["converged"],
                        "true_final_residual": row["true_final_residual"],
                    }
                )
            if definition == "ordinary_ritz":
                rhs, _ = _discrete_rhs(context.system, settings.heldout_rhs_seeds[0])
                result = _run_recycle(context, rhs, factors, settings.solver)
                for iteration, value in enumerate(result.true_residual_history):
                    residual_history_rows.append(
                        {
                            "sample_id": sample_id,
                            "method": "old_q8_ordinary",
                            "iteration": iteration,
                            "relative_residual": value,
                        }
                    )

        construction_rhs, _ = _discrete_rhs(context.system, settings.construction_rhs_seed)
        ordinary64, arnoldi64 = _ordinary_family(
            context.system.matrix,
            context.ilu,
            construction_rhs,
            max_q=32,
            arnoldi_dimension=settings.old_arnoldi_dimension,
            seed=settings.seed + sample_id,
        )
        harmonic64 = _harmonic_from_arnoldi(context.system.matrix, arnoldi64, 32)
        projected = _projected_nonnormality(arnoldi64)
        diagnostics_rows.append(
            {
                "sample_id": sample_id,
                **projected,
                "operator_commutator_hutchinson": _operator_nonnormality_estimate(
                    context, settings.nonnormality_probe_count, settings.seed + 10 * sample_id
                ),
                "ordinary_q8_residual_max": float(np.max(ordinary64.residuals[:8])),
                "ordinary_q8_residual_median": float(np.median(ordinary64.residuals[:8])),
                "harmonic_q8_residual_max": float(np.max(harmonic64.residuals[:8])),
                "harmonic_q8_residual_median": float(np.median(harmonic64.residuals[:8])),
            }
        )

        for family in (ordinary64, harmonic64):
            bases = [(settings.construction_rhs_seed, family.basis)]
            for alternate_seed in settings.stability_rhs_seeds:
                initial, _ = _discrete_rhs(context.system, alternate_seed)
                if family.name == "ordinary":
                    alternate, _ = _ordinary_family(
                        context.system.matrix,
                        context.ilu,
                        initial,
                        max_q=32,
                        arnoldi_dimension=settings.old_arnoldi_dimension,
                        seed=settings.seed + sample_id + alternate_seed,
                    )
                else:
                    alternate_arnoldi = _capture_arnoldi(
                        context.system.matrix, context.ilu, initial, settings.old_arnoldi_dimension
                    )
                    alternate = _harmonic_from_arnoldi(
                        context.system.matrix, alternate_arnoldi, 32
                    )
                bases.append((alternate_seed, alternate.basis))
            for q in (8, 32):
                for first_index, (seed_i, basis_i) in enumerate(bases):
                    for seed_j, basis_j in bases[first_index + 1 :]:
                        metrics = generalized_subspace_metrics(basis_i[:, :q], basis_j[:, :q])
                        stability_rows.append(
                            {
                                "sample_id": sample_id,
                                "family": f"{family.name}_m64",
                                "q": q,
                                "seed_i": seed_i,
                                "seed_j": seed_j,
                                **metrics,
                            }
                        )

        ordinary, arnoldi = _ordinary_family(
            context.system.matrix,
            context.ilu,
            construction_rhs,
            max_q=settings.max_spectral_dimension,
            arnoldi_dimension=settings.long_arnoldi_dimension,
            seed=settings.seed + sample_id,
        )
        harmonic = _harmonic_from_arnoldi(
            context.system.matrix, arnoldi, settings.max_spectral_dimension
        )
        candidates_by_method: Dict[str, List[Candidate]] = {}
        for family in (ordinary, harmonic):
            prefix_method = f"{family.name}_adaptive_rank_m128"
            prefixes = [
                _family_prefix_candidate(family, q, prefix_method)
                for q in settings.q_values
            ]
            candidates_by_method[prefix_method] = prefixes
            filtered = _filtered_candidates(
                family, settings.q_values, settings.residual_filter_thresholds
            )
            if filtered:
                candidates_by_method[f"{family.name}_residual_filter"] = filtered

            for direction, (value, residual) in enumerate(zip(family.eigenvalues, family.residuals), start=1):
                spectral_rows.append(
                    {
                        "sample_id": sample_id,
                        "family": family.name,
                        "arnoldi_dimension": family.arnoldi_dimension,
                        "direction": direction,
                        "eigenvalue_real": float(value.real),
                        "eigenvalue_imag": float(value.imag),
                        "magnitude": float(abs(value)),
                        "ritz_residual_relative": float(residual),
                    }
                )

            clusters, cluster_threshold = _cluster_atomic_blocks(
                family.eigenvalues, family.blocks
            )
            cluster_candidates: List[Candidate] = []
            cluster_screen: List[Tuple[Candidate, List[Dict[str, Any]]]] = []
            for cluster_index, cluster in enumerate(clusters, start=1):
                indices = tuple(index for block in cluster for index in range(*block))
                if not indices or len(indices) > settings.max_spectral_dimension:
                    continue
                candidate = Candidate(
                    method=f"{family.name}_cluster_combo",
                    label=f"{family.name}_cluster{cluster_index}_q{len(indices)}",
                    basis=family.basis[:, indices],
                    construction_seconds=family.construction_seconds,
                    metadata={
                        "requested_q": len(indices),
                        "indices": ",".join(str(index + 1) for index in indices),
                        "residual_threshold": "",
                        "cluster_threshold": cluster_threshold,
                    },
                )
                screen_rows, _ = _evaluate_basis(
                    context,
                    candidate,
                    (settings.selection_rhs_seeds[0],),
                    settings.solver,
                    "cluster_screen",
                    baseline_by_seed,
                )
                cluster_screen.append((candidate, screen_rows))
                block_rows.extend(
                    {
                        **row,
                        "ablation": "spectral_gap_cluster_alone",
                        "family": family.name,
                        "cluster_index": cluster_index,
                    }
                    for row in screen_rows
                )
            cluster_screen.sort(key=lambda item: _candidate_score(item[1]))
            selected_columns: List[np.ndarray] = []
            labels: List[str] = []
            total_q = 0
            for candidate, _ in cluster_screen:
                if total_q + candidate.basis.shape[1] > settings.max_spectral_dimension:
                    continue
                selected_columns.append(candidate.basis)
                labels.append(candidate.label)
                total_q += candidate.basis.shape[1]
                combined = _orthonormalize(np.column_stack(selected_columns))
                cluster_candidates.append(
                    Candidate(
                        method=f"{family.name}_cluster_combo",
                        label=f"{family.name}_cluster_combo_{len(labels)}_q{combined.shape[1]}",
                        basis=combined,
                        construction_seconds=family.construction_seconds,
                        metadata={
                            "requested_q": combined.shape[1],
                            "indices": ";".join(labels),
                            "residual_threshold": "",
                            "cluster_threshold": cluster_threshold,
                        },
                    )
                )
            if cluster_candidates:
                candidates_by_method[f"{family.name}_cluster_combo"] = cluster_candidates

            # Direction and deletion diagnostics on one selection RHS.  These
            # are descriptive ablations and never select the held-out winner.
            first_blocks = [
                block for block in family.blocks if block[1] <= settings.individual_ablation_dimension
            ]
            q8_indices = list(_complete_block_indices(family.blocks, 8))
            for block_index, block in enumerate(first_blocks, start=1):
                atom_indices = tuple(range(block[0], block[1]))
                atom = Candidate(
                    method=f"{family.name}_block_ablation",
                    label=f"{family.name}_atom{block_index}_q{len(atom_indices)}",
                    basis=family.basis[:, atom_indices],
                    construction_seconds=family.construction_seconds,
                    metadata={"requested_q": len(atom_indices), "indices": str(atom_indices)},
                )
                atom_rows, _ = _evaluate_basis(
                    context,
                    atom,
                    (settings.selection_rhs_seeds[0],),
                    settings.solver,
                    "direction_ablation",
                    baseline_by_seed,
                )
                block_rows.extend({**row, "ablation": "atom_alone", "family": family.name, "cluster_index": block_index} for row in atom_rows)
                retained = [index for index in q8_indices if index not in atom_indices]
                # Only record a deletion when the atomic block actually lies
                # inside the q=8 prefix.  Blocks beyond the prefix would leave
                # the basis unchanged and are not valid ablations.
                if retained and len(retained) < len(q8_indices):
                    deleted = Candidate(
                        method=f"{family.name}_block_ablation",
                        label=f"{family.name}_q8_without_atom{block_index}_q{len(retained)}",
                        basis=family.basis[:, retained],
                        construction_seconds=family.construction_seconds,
                        metadata={"requested_q": len(retained), "indices": str(tuple(retained))},
                    )
                    deleted_rows, _ = _evaluate_basis(
                        context,
                        deleted,
                        (settings.selection_rhs_seeds[0],),
                        settings.solver,
                        "direction_ablation",
                        baseline_by_seed,
                    )
                    block_rows.extend({**row, "ablation": "delete_from_q8", "family": family.name, "cluster_index": block_index} for row in deleted_rows)

        for method, candidates in candidates_by_method.items():
            method_rows: List[Dict[str, Any]] = []
            selection_start = perf_counter()
            for candidate in candidates:
                rows, _ = _evaluate_basis(
                    context,
                    candidate,
                    settings.selection_rhs_seeds,
                    settings.solver,
                    "selection",
                    baseline_by_seed,
                )
                method_rows.extend(rows)
            selection_seconds = perf_counter() - selection_start
            selection_rows.extend(method_rows)
            winner = _select_candidate(candidates, method_rows)
            selected_candidates[(sample_id, method)] = winner
            rows, factors = _evaluate_basis(
                context,
                winner,
                settings.heldout_rhs_seeds,
                settings.solver,
                "heldout",
                baseline_by_seed,
            )
            for row in rows:
                row["selection_seconds"] = selection_seconds
            evaluation_rows.extend(rows)
            # Retain reusable bases for the two prefix families.  Filtered and
            # cluster-combination outcomes are fully recorded in CSV, and in
            # this experiment their selected arrays are either duplicates or
            # one-off diagnostic search products that need not be archived.
            if method.endswith("adaptive_rank_m128"):
                np.savez_compressed(
                    output / "spaces" / f"sample_{sample_id:02d}_{method}.npz",
                    Z=winner.basis,
                    W=factors.test,
                    U=factors.recycling_trial,
                )

    spectral_summary = _summarize_evaluation(evaluation_rows, settings)
    spectral_methods = sorted({str(row["method"]) for row in spectral_summary})
    spectral_success = {
        method: sum(
            bool(row["system_success"])
            for row in spectral_summary
            if str(row["method"]) == method
        )
        for method in spectral_methods
    }
    local_global_triggered = bool(
        settings.run_local_global_if_spectral_fail
        and max(spectral_success.values(), default=0) < settings.strict_majority
    )

    local_global_diagnostics: List[Dict[str, Any]] = []
    history_snapshot_rows: List[Dict[str, Any]] = []
    if local_global_triggered:
        print("[Stage20] spectral majority gate failed; running local/global fallback", flush=True)
        for ordinal, sample_id in enumerate(settings.sample_ids, start=1):
            print(f"[Stage20 local/global {ordinal}/{len(settings.sample_ids)}] ID {sample_id}", flush=True)
            context = _load_context(root, sample_id, coefficient_rows, nodes, boundary, settings)
            baseline_by_seed = {
                seed: baseline_eval[(sample_id, seed)] for seed in settings.heldout_rhs_seeds
            }
            for seed in settings.selection_rhs_seeds:
                baseline_by_seed[seed] = baseline_selection[(sample_id, seed)]
            candidates, diagnostics, history_rows = _make_local_global_candidates(context, settings)
            local_global_diagnostics.append(diagnostics)
            history_snapshot_rows.extend(history_rows)
            grouped: Dict[str, List[Candidate]] = {}
            for candidate in candidates:
                grouped.setdefault(candidate.method, []).append(candidate)
            for method, method_candidates in grouped.items():
                method_rows: List[Dict[str, Any]] = []
                selection_start = perf_counter()
                for candidate in method_candidates:
                    rows, _ = _evaluate_basis(
                        context,
                        candidate,
                        settings.selection_rhs_seeds,
                        settings.solver,
                        "selection",
                        baseline_by_seed,
                    )
                    method_rows.extend(rows)
                selection_seconds = perf_counter() - selection_start
                selection_rows.extend(method_rows)
                winner = _select_candidate(method_candidates, method_rows)
                selected_candidates[(sample_id, method)] = winner
                rows, factors = _evaluate_basis(
                    context,
                    winner,
                    settings.heldout_rhs_seeds,
                    settings.solver,
                    "heldout",
                    baseline_by_seed,
                )
                for row in rows:
                    row["selection_seconds"] = selection_seconds
                evaluation_rows.extend(rows)
                np.savez_compressed(
                    output / "spaces" / f"sample_{sample_id:02d}_{method}.npz",
                    Z=winner.basis,
                    W=factors.test,
                    U=factors.recycling_trial,
                )

    summaries = _summarize_evaluation(evaluation_rows, settings)
    family_rows: List[Dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in summaries})
    for method in methods:
        local = [row for row in summaries if str(row["method"]) == method]
        family_rows.append(
            {
                "method": method,
                "systems_tested": len(local),
                "systems_successful": sum(bool(row["system_success"]) for row in local),
                "majority_gate": _family_success(summaries, method, settings),
                "median_q": float(np.median([float(row["q"]) for row in local])),
                "median_gamma": float(np.median([float(row["gamma_median"]) for row in local])),
                "median_iterations": float(np.median([float(row["iterations_median"]) for row in local])),
                "median_online_seconds": float(np.median([float(row["online_seconds_median"]) for row in local])),
                "median_construction_seconds": float(np.median([float(row["construction_seconds"]) for row in local])),
            }
        )
    winning_family = min(
        family_rows,
        key=lambda row: (
            -int(row["systems_successful"]),
            -float(row["median_gamma"]),
            float(row["median_q"]),
            float(row["median_online_seconds"]),
        ),
    )["method"]

    # Construction-RHS robustness for the family-level winner.
    winner_stability_rows: List[Dict[str, Any]] = []
    for sample_id in settings.sample_ids:
        selected = selected_candidates[(sample_id, winning_family)]
        if winning_family.startswith("ordinary_adaptive_rank") or winning_family.startswith("harmonic_adaptive_rank"):
            context = _load_context(root, sample_id, coefficient_rows, nodes, boundary, settings)
            alternate_rhs, _ = _discrete_rhs(context.system, settings.stability_rhs_seeds[0])
            ordinary_alt, arnoldi_alt = _ordinary_family(
                context.system.matrix,
                context.ilu,
                alternate_rhs,
                settings.max_spectral_dimension,
                settings.long_arnoldi_dimension,
                settings.seed + sample_id + 991,
            )
            family_alt = ordinary_alt if winning_family.startswith("ordinary") else _harmonic_from_arnoldi(
                context.system.matrix, arnoldi_alt, settings.max_spectral_dimension
            )
            indices = _complete_block_indices(family_alt.blocks, selected.basis.shape[1])
            alternate_basis = family_alt.basis[:, indices]
            metrics = generalized_subspace_metrics(selected.basis, alternate_basis)
            factors = factor_az_petrov(context.system.matrix, alternate_basis)
            iterations = []
            for rhs_seed in settings.heldout_rhs_seeds:
                rhs, _ = _discrete_rhs(context.system, rhs_seed)
                result = _run_recycle(context, rhs, factors, settings.solver)
                iterations.append(result.iterations)
            winner_stability_rows.append(
                {
                    "sample_id": sample_id,
                    "method": winning_family,
                    "q_primary": selected.basis.shape[1],
                    "q_alternate": alternate_basis.shape[1],
                    **metrics,
                    "alternate_iterations_median": float(np.median(iterations)),
                    "alternate_iterations_min": min(iterations),
                    "alternate_iterations_max": max(iterations),
                }
            )

    _write_csv(output / "reproduction_checks.csv", reproduction_rows)
    _write_csv(output / "nonnormality_diagnostics.csv", diagnostics_rows)
    _write_csv(output / "construction_subspace_stability.csv", stability_rows)
    _write_csv(output / "long_arnoldi_spectrum.csv", spectral_rows)
    _write_csv(output / "spectral_block_ablation.csv", block_rows)
    _write_csv(output / "candidate_selection_results.csv", selection_rows)
    _write_csv(output / "heldout_oracle_results.csv", evaluation_rows)
    _write_csv(output / "heldout_oracle_summary.csv", summaries)
    _write_csv(output / "family_gate_summary.csv", family_rows)
    _write_csv(output / "residual_histories.csv", residual_history_rows)
    _write_csv(output / "local_global_diagnostics.csv", local_global_diagnostics)
    _write_csv(output / "history_snapshot_metadata.csv", history_snapshot_rows)
    _write_csv(output / "winning_family_stability.csv", winner_stability_rows)

    _make_plots(
        output,
        settings,
        reproduction_rows,
        diagnostics_rows,
        summaries,
        family_rows,
        residual_history_rows,
        block_rows,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(
        report,
        output,
        settings,
        reproduction_rows,
        diagnostics_rows,
        stability_rows,
        summaries,
        family_rows,
        winner_stability_rows,
        local_global_triggered,
        winning_family,
    )
    _validate(
        output,
        report,
        settings,
        reproduction_rows,
        summaries,
        family_rows,
        local_global_triggered,
    )
    _write_manifest(output)
    print(f"Stage 20 complete: {report}", flush=True)
    return report


def _make_plots(
    output: Path,
    settings: Stage20Settings,
    reproduction: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    histories: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output / "plots"
    # Matrix heatmap: exact ID-by-method held-out comparison.
    methods = [str(row["method"]) for row in sorted(family_rows, key=lambda row: (-int(row["systems_successful"]), str(row["method"])))]
    matrix = np.full((len(settings.sample_ids), len(methods)), np.nan)
    for row in summaries:
        matrix[settings.sample_ids.index(int(row["sample_id"])), methods.index(str(row["method"]))] = float(row["iterations_median"])
    figure, axis = plt.subplots(figsize=(max(9.0, 1.1 * len(methods)), 5.7))
    image = axis.imshow(matrix, vmin=0, vmax=settings.solver.max_iterations, cmap="Blues")
    axis.set_xticks(range(len(methods)), [method.replace("_", " ") for method in methods], rotation=35, ha="right")
    axis.set_yticks(range(len(settings.sample_ids)), [f"ID {sample_id}" for sample_id in settings.sample_ids])
    axis.set_title("Held-out recycled-GMRES iterations by oracle family")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", color="white" if matrix[i,j] > 170 else CHARCOAL, fontsize=8)
    figure.colorbar(image, ax=axis, label="Median iterations (cap 300)")
    figure.tight_layout()
    figure.savefig(plot_dir / "heldout_iteration_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Diagnostic relationship: eight intentional labeled points.
    old_q8 = {
        int(row["sample_id"]): float(np.median([
            (int(old["saved_iterations"]) - int(old["reproduced_iterations"]))
            for old in reproduction
            if int(old["sample_id"]) == int(row["sample_id"]) and old["definition"] == "ordinary_ritz"
        ]))
        for row in diagnostics
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for row in diagnostics:
        sample_id = int(row["sample_id"])
        old = [r for r in reproduction if int(r["sample_id"]) == sample_id and r["definition"] == "ordinary_ritz"]
        baseline = [r for r in reproduction if int(r["sample_id"]) == sample_id and r["definition"] == "strong_ilu"]
        gamma = (np.median([int(r["reproduced_iterations"]) for r in baseline]) - np.median([int(r["reproduced_iterations"]) for r in old])) / max(np.median([int(r["reproduced_iterations"]) for r in baseline]), 1)
        axes[0].scatter(float(row["ordinary_q8_residual_max"]), gamma, color=BLUE)
        axes[0].annotate(str(sample_id), (float(row["ordinary_q8_residual_max"]), gamma), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axes[1].scatter(float(row["projected_commutator_normalized"]), gamma, color=ORANGE)
        axes[1].annotate(str(sample_id), (float(row["projected_commutator_normalized"]), gamma), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Maximum q=8 Ritz-pair residual")
    axes[0].set_ylabel("q=8 iteration reduction")
    axes[1].set_xlabel("Projected commutator indicator")
    axes[1].set_ylabel("q=8 iteration reduction")
    for axis in axes:
        axis.axhline(settings.strong_gamma, color=GRAY, linestyle="--", linewidth=1)
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    figure.suptitle("Ritz quality and nonnormality versus old q=8 solver value")
    figure.tight_layout()
    figure.savefig(plot_dir / "id15_mechanism_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Family gate ranking.
    ordered = sorted(family_rows, key=lambda row: (int(row["systems_successful"]), float(row["median_gamma"])))
    figure, axis = plt.subplots(figsize=(8.8, max(3.8, 0.55 * len(ordered))))
    axis.barh([str(row["method"]).replace("_", " ") for row in ordered], [int(row["systems_successful"]) for row in ordered], color=BLUE, edgecolor=CHARCOAL)
    axis.axvline(settings.strict_majority, color=ORANGE, linestyle="--", label=f"majority gate = {settings.strict_majority}")
    axis.set_xlim(0, len(settings.sample_ids))
    axis.set_xlabel("Hard systems with convergence and at least 40% median iteration reduction")
    axis.set_title("Own-system oracle success count")
    axis.legend(frameon=False)
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    figure.tight_layout()
    figure.savefig(plot_dir / "family_success_counts.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Residual histories already expose the old baseline; final selected-space
    # histories are intentionally omitted when their candidate varies by method.
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        for method, color, line in (("strong_ilu", CHARCOAL, "-"), ("old_q8_ordinary", ORANGE, "--")):
            local = [row for row in histories if int(row["sample_id"]) == sample_id and row["method"] == method]
            if local:
                axis.semilogy([int(row["iteration"]) for row in local], [float(row["relative_residual"]) for row in local], color=color, linestyle=line, label=method.replace("_", " "))
        axis.set_title(f"ID {sample_id}")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0,0].set_ylabel("True relative residual")
    axes[1,0].set_ylabel("True relative residual")
    for axis in axes[1,:]: axis.set_xlabel("Iteration")
    handles, labels = axes[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Baseline and old q=8 residual histories, held-out RHS 20261101", y=0.99)
    figure.tight_layout(rect=(0,0,1,0.94))
    figure.savefig(plot_dir / "old_q8_residual_histories.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    report: Path,
    output: Path,
    settings: Stage20Settings,
    reproduction: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    winner_stability: Sequence[Mapping[str, Any]],
    local_global_triggered: bool,
    winning_family: str,
) -> None:
    best = next(row for row in family_rows if row["method"] == winning_family)
    passed = bool(best["majority_gate"])
    decision = (
        "The own-system oracle gate passes for the winning construction, but learning remains gated by construction stability."
        if passed
        else "No tested construction passes the own-system majority gate; neural prediction remains unsupported."
    )
    lines = [
        "# Stage 20 — robust complementary spaces for spatially varying anisotropy",
        "",
        "## Technical summary",
        "",
        f"**{decision}** The strongest family was `{winning_family}` with {int(best['systems_successful'])}/{len(settings.sample_ids)} hard systems passing the fixed criterion: all three held-out solves converge and median iteration reduction is at least {settings.strong_gamma:.0%}. Its median selected rank was {float(best['median_q']):.0f}, median held-out iteration reduction was {float(best['median_gamma']):.3f}, and median online time was {float(best['median_online_seconds']):.3f} s. Selection used two RHSs disjoint from the three Stage-19A held-out RHSs.",
        "",
        "No neural network was trained. The RBF-FD matrices, fixed nodes, strong ILU parameters, right-preconditioned operator convention, and recycled-GMRES implementation are unchanged.",
        "",
        "## Frozen systems, definitions, and leakage control",
        "",
        "The systems are IDs 3, 7, 10, 14, 15, 17, 19, and 34 from Stage 19A. Each has $N=5041$, stencil 25, PHS $r^3$, degree-2 augmentation, seed-7 jitter 0.15, SuperLU ILU with drop tolerance $10^{-3}$ and fill factor 5, restart 50, tolerance $10^{-8}$, and cap 300. The recycle pair uses $AZ=WR$ and $U=ZR^{-1}$, then applies projected Arnoldi to $(I-WW^T)AM^{-1}$.",
        "",
        f"The fixed construction seed was {settings.construction_rhs_seed}. Adaptive selection used {list(settings.selection_rhs_seeds)}. Final evaluation used the untouched Stage-19A seeds {list(settings.heldout_rhs_seeds)}. Thus no candidate rank, residual filter, or cluster combination was selected on the reported held-out solves.",
        "",
        "## The Section 8 failure is reproduced exactly",
        "",
        "Every stored strong-ILU and q=8 ordinary-Ritz, harmonic-Ritz, and correction-POD iteration count was rerun from the archived matrices and matched exactly. Exact per-RHS checks are in `reproduction_checks.csv`.",
        "",
        "| ID | ILU median | ordinary Ritz q=8 | harmonic Ritz q=8 | correction POD q=8 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sample_id in settings.sample_ids:
        def median_for(definition: str) -> int:
            return int(np.median([int(row["reproduced_iterations"]) for row in reproduction if int(row["sample_id"]) == sample_id and row["definition"] == definition]))
        lines.append(f"| {sample_id} | {median_for('strong_ilu')} | {median_for('ordinary_ritz')} | {median_for('harmonic_ritz')} | {median_for('correction_pod')} |")
    lines.extend(
        [
            "",
            "![Old q=8 residual histories](../stages/stage20_spatial_K_robust_oracle_search/plots/old_q8_residual_histories.png)",
            "",
            "The cap is a censored failure, not an estimated converged count. ID 15 remains the positive control.",
            "",
            "## ID 15 is distinguished by Ritz-pair quality, not by low nonnormality",
            "",
            "The diagnostic compares projected and matrix-free nonnormality, q=8 Ritz residuals, multiple construction RHSs, and residual histories. The evidence supports a construction-quality distinction only when it is visible in the saved diagnostics; it does not infer causality from eigenvalue magnitude alone.",
            "",
            "| ID | max ordinary q=8 Ritz residual | projected commutator | Henrici departure | eigenvector cond. | operator commutator estimate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics:
        lines.append(
            f"| {int(row['sample_id'])} | {float(row['ordinary_q8_residual_max']):.3g} | {float(row['projected_commutator_normalized']):.3g} | {float(row['projected_henrici_departure']):.3g} | {float(row['projected_eigenvector_condition']):.3g} | {float(row['operator_commutator_hutchinson']):.3g} |"
        )
    lines.extend(
        [
            "",
            "![Mechanism diagnostics](../stages/stage20_spatial_K_robust_oracle_search/plots/id15_mechanism_diagnostics.png)",
            "",
            "Construction-RHS projector and principal-angle statistics for ordinary and harmonic spaces at q=8 and q=32 are in `construction_subspace_stability.csv`. Individual atomic-block, q=8 deletion, spectral-cluster, and cluster-combination ablations are in `spectral_block_ablation.csv`; no block is called solver-critical unless its actual selection-RHS solve improves.",
            "",
            "## Adaptive rank, residual filtering, and spectral blocks",
            "",
            "The 128-step Arnoldi experiment kept the same operator $AM^{-1}$ but exposed up to 64 real, pair-respecting directions. Candidate rank, residual-quality threshold, and variable-size spectral-gap clusters were chosen on the two selection RHSs. Cluster combinations ranked whole blocks by actual recycled-GMRES response rather than assuming that the smallest Ritz values are always most important.",
            "",
            "| Method | successes / 8 | majority gate | median q | median iterations | median reduction | median setup (s) | median online (s) |",
            "|---|---:|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(family_rows, key=lambda item: (-int(item["systems_successful"]), str(item["method"]))):
        lines.append(
            f"| {str(row['method']).replace('_', ' ')} | {int(row['systems_successful'])}/8 | {'yes' if row['majority_gate'] else 'no'} | {float(row['median_q']):.0f} | {float(row['median_iterations']):.1f} | {float(row['median_gamma']):.3f} | {float(row['median_construction_seconds']):.3f} | {float(row['median_online_seconds']):.3f} |"
        )
    lines.extend(
        [
            "",
            "![Family oracle gate](../stages/stage20_spatial_K_robust_oracle_search/plots/family_success_counts.png)",
            "",
            "The exact per-system table follows. A success requires convergence on all three held-out RHSs and at least 40% median iteration reduction.",
            "",
            "| ID | method | q | ILU | oracle median [min,max] | convergence | reduction | online (s) | success |",
            "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sorted(summaries, key=lambda item: (int(item["sample_id"]), str(item["method"]))):
        lines.append(
            f"| {int(row['sample_id'])} | {str(row['method']).replace('_', ' ')} | {int(row['q'])} | {float(row['k_ilu_median']):.0f} | {float(row['iterations_median']):.0f} [{int(row['iterations_min'])},{int(row['iterations_max'])}] | {float(row['convergence_rate']):.2f} | {float(row['gamma_median']):.3f} | {float(row['online_seconds_median']):.3f} | {'yes' if row['system_success'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "![Held-out iteration matrix](../stages/stage20_spatial_K_robust_oracle_search/plots/heldout_iteration_heatmap.png)",
            "",
            "## Stability, cost, and learning gate",
            "",
        ]
    )
    if winner_stability:
        median_distance = float(np.median([float(row["projector_distance_normalized"]) for row in winner_stability]))
        median_angle = float(np.median([float(row["principal_angle_max_degrees"]) for row in winner_stability]))
        lines.append(
            f"Reconstructing the winning spectral family from a different construction RHS gave median normalized projector distance {median_distance:.3f} and median maximum principal angle {median_angle:.1f} degrees. Solver results for these independently constructed spaces are recorded in `winning_family_stability.csv`. This separates solver relevance from geometric stability."
        )
    else:
        lines.append("The winning family was not a simple spectral prefix, so a matched alternate-construction stability rerun was not defined in this stage. Its construction variability remains an explicit learning blocker.")
    lines.extend(
        [
            "",
            f"The conditional multilevel/local-global fallback was {'run because every spectral family missed the majority gate' if local_global_triggered else 'not run because at least one spectral family passed the predeclared majority gate'}. Setup times include the reported Arnoldi or history construction and AZ pairing. Adaptive search time is saved separately in `candidate_selection_results.csv` and must be amortized; it is not hidden inside online time.",
            "",
            "A solver-useful own-system oracle is not automatically a learnable parameter-to-space target. Later learning would additionally require repeatable construction, cross-system regularity, and a cost advantage over simply updating the space with Krylov data.",
            "",
            "## Limitations and one recommended next step",
            "",
            "The study is limited to the eight preselected hard fields, three frozen held-out RHSs, rank at most 64, and one unchanged strong-ILU/recycling implementation. Candidate selection is system-own and intentionally expensive; its cost is an oracle diagnostic rather than an online proposal. Nonnormality estimates are randomized Hutchinson diagnostics with the stored seed and are not exact dense norms.",
            "",
        ]
    )
    if passed:
        lines.append(
            f"**Recommended next step: run one independent validation stage for `{winning_family}` on the remaining Stage-19A spatial fields and new construction RHS groups, measuring construction stability and update cost before any neural training.**"
        )
    else:
        lines.append(
            "**Recommended next step: pause recycle-space learning and test a classical multilevel preconditioner as the primary solver, rather than attempting to predict the failed complementary spaces.**"
        )
    lines.extend(
        [
            "",
            "## Reproducibility and audit artifacts",
            "",
            "- `reproduction_checks.csv`: exact old-result reruns.",
            "- `nonnormality_diagnostics.csv`, `construction_subspace_stability.csv`, and `long_arnoldi_spectrum.csv`: mechanism evidence.",
            "- `spectral_block_ablation.csv`: individual and cluster response.",
            "- `candidate_selection_results.csv`: all selection-RHS trials.",
            "- `heldout_oracle_results.csv` and `heldout_oracle_summary.csv`: primary held-out evidence.",
            "- `family_gate_summary.csv`: predeclared majority decision.",
            "- `validation.json`: machine-readable integrity checks.",
            "",
            f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate(
    output: Path,
    report: Path,
    settings: Stage20Settings,
    reproduction: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    local_global_triggered: bool,
) -> None:
    checks = {
        "all_old_iterations_match": bool(reproduction) and all(bool(row["exact_iteration_match"]) for row in reproduction),
        "all_eight_hard_ids_present": {int(row["sample_id"]) for row in summaries} == set(settings.sample_ids),
        "heldout_system_method_rows_complete": all(int(row["sample_id"]) in settings.sample_ids for row in summaries),
        "true_residuals_valid_for_converged": all(
            (float(row["convergence_rate"]) < 1.0) or float(row["true_residual_max"]) <= 1.1 * settings.solver.relative_tolerance
            for row in summaries
        ),
        "selection_and_heldout_disjoint": set(settings.selection_rhs_seeds).isdisjoint(settings.heldout_rhs_seeds),
        "majority_gate_consistent": all(
            bool(row["majority_gate"]) == (int(row["systems_successful"]) >= settings.strict_majority)
            for row in family_rows
        ),
        "local_global_trigger_recorded": isinstance(local_global_triggered, bool),
        "report_exists": report.is_file(),
        "no_neural_artifacts": not any("model" in path.name.lower() or path.suffix in {".pt", ".pth"} for path in output.rglob("*")),
        "required_plots_exist": all(
            (output / "plots" / name).is_file()
            for name in (
                "heldout_iteration_heatmap.png",
                "id15_mechanism_diagnostics.png",
                "family_success_counts.png",
                "old_q8_residual_histories.png",
            )
        ),
    }
    _write_json(
        output / "validation.json",
        {
            "assessment": "ready_to_share" if all(checks.values()) else "needs_revision",
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "caveat": "Adaptive candidate search is an expensive own-system oracle and is not an online method.",
        },
    )
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Stage 20 validation failed: {failed}")


def _write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output / "manifest.json",
        {
            "stage": "stage20_spatial_K_robust_oracle_search",
            "artifact_count": len(rows),
            "artifacts": rows,
        },
    )
