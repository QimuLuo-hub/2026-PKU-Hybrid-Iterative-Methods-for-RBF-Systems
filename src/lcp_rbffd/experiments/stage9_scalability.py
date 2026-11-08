"""Stage 9 hard-regime discovery and scalability experiments.

The experiment keeps the proposal's missing inverse action exactly as
``d(r)=A^-1 r-M^-1 r``.  It first searches for numerically meaningful regimes
where the unchanged strong Stage 8 ILU baseline is costly or ineffective.
Correction snapshots, oracle spaces, transfer checks, and the already-fixed
direct model are gated on evidence from that search.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.stats as stats

from ..coarse_space import build_multiplicative_coarse_preconditioner
from ..config import DiscretizationConfig, ILUConfig, RBFConfig, SolverConfig
from ..krylov import KrylovResult, restarted_gmres
from ..models.direct_control import make_stabilized_direct_preconditioner
from ..models.direct_mlp import (
    DirectMLPConfig,
    correction_metrics,
    train_direct_mlp,
)
from ..nodes import generate_nodes
from ..pde import (
    ConstantAnisotropicDiffusion,
    ExponentialCoefficient,
    LocalizedCoefficient,
    ManufacturedSolution,
    random_smooth_solution,
)
from ..pod import compute_pod, relative_projection_errors
from ..preconditioners import InverseAction, PreconditionerFailure, ilu_preconditioner
from ..rbffd import RBFSystem, assemble_system, local_weights
from ..snapshots import generate_gmres_residuals, generate_synthetic_residuals
from ..subspace_analysis import energy_dimension, krylov_ritz_basis, subspace_metrics


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
class Stage9Settings:
    """Preregistered numerical controls; the ILU matches Stages 7--8."""

    seed: int = 20260830
    solver: SolverConfig = SolverConfig(
        relative_tolerance=1.0e-8,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=300,
        breakdown_tolerance=1.0e-14,
    )
    classical_ilu: ILUConfig = ILUConfig(
        drop_tolerance=1.0e-3,
        fill_factor=5.0,
        diagonal_pivot_threshold=0.0,
        permutation="COLAMD",
    )
    scale_nodes: Tuple[int, ...] = (2500, 5041, 10000, 20164)
    contrasts: Tuple[float, ...] = (10.0, 100.0, 1000.0, 10000.0)
    anisotropy_ratios: Tuple[float, ...] = (10.0, 100.0, 1000.0)
    repeated_counts: Tuple[int, ...] = (1, 10, 50, 100, 500)
    snapshot_train_each: int = 12
    snapshot_test_each: int = 6
    oracle_q_values: Tuple[int, ...] = (4, 8, 12, 16, 24)
    ritz_arnoldi_dimension: int = 40
    meaningful_error_limit: float = 0.25
    structured_eta95_limit: float = 0.25
    structured_q95_fraction_limit: float = 0.02
    oracle_iteration_reduction_gate: float = 0.20


@dataclass(frozen=True)
class Stage9Spec:
    """One independently controlled RBF-FD regime."""

    name: str
    family: str
    n_nodes: int = 5041
    stencil_size: int = 25
    node_distribution: str = "jittered_grid"
    jitter_fraction: float = 0.15
    warp_strength: float = 0.0
    density_exponent: float = 1.0
    coefficient_kind: str = "constant"
    coefficient_contrast: float = 1.0
    coefficient_frequency: int = 1
    anisotropy_angle: float = 30.0
    seed: int = 7

    def config(self) -> DiscretizationConfig:
        return DiscretizationConfig(
            n_nodes=self.n_nodes,
            stencil_size=self.stencil_size,
            polynomial_degree=2,
            node_distribution=self.node_distribution,
            jitter_fraction=self.jitter_fraction,
            warp_strength=self.warp_strength,
            density_exponent=self.density_exponent,
            seed=self.seed,
            rbf=RBFConfig(kernel="phs", order=3, shape_parameter=1.0),
            compute_local_condition_numbers=False,
        )

    def coefficient(self) -> Optional[object]:
        if self.coefficient_kind == "constant":
            return None
        if self.coefficient_kind == "smooth_scalar":
            return ExponentialCoefficient(
                beta=0.5 * np.log(self.coefficient_contrast),
                frequency=self.coefficient_frequency,
            )
        if self.coefficient_kind == "localized_scalar":
            return LocalizedCoefficient(
                peak_contrast=self.coefficient_contrast,
                center=(0.58, 0.46),
                width=0.14,
            )
        if self.coefficient_kind == "anisotropic":
            return ConstantAnisotropicDiffusion(
                anisotropy_ratio=self.coefficient_contrast,
                angle_degrees=self.anisotropy_angle,
            )
        raise ValueError(f"unknown coefficient kind: {self.coefficient_kind}")


def discovery_specs(settings: Stage9Settings) -> List[Stage9Spec]:
    """One-factor-at-a-time scale, PDE, and node-set experiment hierarchy."""

    scale = [
        Stage9Spec(
            name=f"scale_N{n}",
            family="scale",
            n_nodes=n,
        )
        for n in settings.scale_nodes
    ]
    contrast = [
        Stage9Spec(
            name=f"smooth_C{contrast:g}".replace(".", "p"),
            family="smooth_contrast",
            coefficient_kind="smooth_scalar",
            coefficient_contrast=contrast,
        )
        for contrast in settings.contrasts
    ]
    pde_variants = [
        Stage9Spec(
            name="localized_C1000",
            family="localized_contrast",
            coefficient_kind="localized_scalar",
            coefficient_contrast=1000.0,
        )
    ] + [
        Stage9Spec(
            name=f"anisotropy_R{ratio:g}".replace(".", "p"),
            family="anisotropy",
            coefficient_kind="anisotropic",
            coefficient_contrast=ratio,
        )
        for ratio in settings.anisotropy_ratios
    ]
    geometry = [
        Stage9Spec(
            name="geometry_jitter35",
            family="node_geometry",
            jitter_fraction=0.35,
        ),
        Stage9Spec(
            name="geometry_warp95",
            family="node_geometry",
            node_distribution="warped_grid",
            jitter_fraction=0.0,
            warp_strength=0.95,
        ),
        Stage9Spec(
            name="geometry_center_refined15",
            family="node_geometry",
            node_distribution="center_clustered_grid",
            jitter_fraction=0.0,
            density_exponent=1.5,
        ),
        Stage9Spec(
            name="geometry_center_refined20",
            family="node_geometry",
            node_distribution="center_clustered_grid",
            jitter_fraction=0.0,
            density_exponent=2.0,
        ),
    ]
    return scale + contrast + pde_variants + geometry


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _native(row.get(key, "")) for key in fields})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_native), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read generated scalar CSV values back into native types."""

    def parse(value: str) -> Any:
        if value == "":
            return ""
        if value in {"True", "False"}:
            return value == "True"
        try:
            return float(value)
        except ValueError:
            return value

    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {key: parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def sparse_storage_bytes(matrix: sp.spmatrix) -> int:
    """Return exact CSR/CSC array storage, excluding Python object overhead."""

    compressed = matrix.tocsr()
    return int(
        compressed.data.nbytes
        + compressed.indices.nbytes
        + compressed.indptr.nbytes
    )


def _accurate_factor_solve(
    matrix: sp.spmatrix,
    factor: Any,
    rhs: np.ndarray,
    refinement_steps: int = 5,
) -> Tuple[np.ndarray, float]:
    """Apply sparse LU with residual-based iterative refinement."""

    rhs = np.asarray(rhs, dtype=float)
    solution = np.asarray(factor.solve(rhs), dtype=float)
    scale = max(float(np.linalg.norm(rhs)), 1.0e-300)
    previous = float(np.linalg.norm(rhs - matrix @ solution) / scale)
    for _ in range(refinement_steps):
        correction_rhs = rhs - matrix @ solution
        candidate = solution + np.asarray(factor.solve(correction_rhs), dtype=float)
        current = float(np.linalg.norm(rhs - matrix @ candidate) / scale)
        if not np.isfinite(current) or current >= previous * (1.0 - 1.0e-3):
            break
        solution, previous = candidate, current
        if current <= 1.0e-12:
            break
    return solution, previous


def _rhs(
    system: RBFSystem,
    coefficient: Optional[object],
    solution: ManufacturedSolution,
) -> np.ndarray:
    if coefficient is None:
        return solution.right_hand_side(system.nodes, system.boundary)
    if isinstance(coefficient, ConstantAnisotropicDiffusion):
        return solution.anisotropic_right_hand_side(
            system.nodes, system.boundary, coefficient
        )
    return solution.variable_coefficient_right_hand_side(
        system.nodes, system.boundary, coefficient
    )


def _local_condition_sample(system: RBFSystem, sample_count: int = 96) -> Dict[str, float]:
    """Estimate local augmented-system conditioning on deterministic rows."""

    from scipy.spatial import cKDTree

    interior = np.flatnonzero(system.interior)
    if interior.size == 0:
        return {"local_condition_median": float("nan"), "local_condition_max": float("nan")}
    selected = interior[np.linspace(0, interior.size - 1, min(sample_count, interior.size)).astype(int)]
    tree = cKDTree(system.nodes)
    conditions = []
    for row in selected:
        _, stencil = tree.query(system.nodes[row], k=system.config.stencil_size)
        _, _, condition = local_weights(
            system.nodes[np.asarray(stencil, dtype=int)],
            system.nodes[row],
            system.config.rbf,
            system.config.polynomial_degree,
            compute_condition=True,
        )
        conditions.append(condition)
    return {
        "local_condition_median": float(np.median(conditions)),
        "local_condition_max": float(np.max(conditions)),
    }


def _preconditioned_condition_proxy(
    matrix: sp.spmatrix,
    preconditioner: InverseAction,
    dimension: int,
    seed: int,
) -> float:
    """Small-Arnoldi singular-value proxy for ``cond(A M^-1)``."""

    n = matrix.shape[0]
    dimension = min(dimension, n - 1)
    rng = np.random.default_rng(seed)
    q = np.zeros((n, dimension + 1), dtype=float)
    h = np.zeros((dimension + 1, dimension), dtype=float)
    q[:, 0] = rng.normal(size=n)
    q[:, 0] /= np.linalg.norm(q[:, 0])
    actual = dimension
    for column in range(dimension):
        work = np.asarray(matrix @ preconditioner.apply(q[:, column]), dtype=float)
        for row in range(column + 1):
            h[row, column] = np.dot(q[:, row], work)
            work -= h[row, column] * q[:, row]
        for row in range(column + 1):
            correction = np.dot(q[:, row], work)
            h[row, column] += correction
            work -= correction * q[:, row]
        h[column + 1, column] = np.linalg.norm(work)
        if h[column + 1, column] <= 1.0e-13:
            actual = column + 1
            break
        q[:, column + 1] = work / h[column + 1, column]
    singular = np.linalg.svd(h[:actual, :actual], compute_uv=False)
    return float(singular[0] / max(singular[-1], np.finfo(float).tiny))


def _solve_rows(
    spec: Stage9Spec,
    system: RBFSystem,
    coefficient: Optional[object],
    preconditioner: InverseAction,
    settings: Stage9Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Solve two independent manufactured problems and retain true histories."""

    solve_rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    solutions = [ManufacturedSolution(), random_smooth_solution(settings.seed + 91, 4)]
    for rhs_index, solution in enumerate(solutions):
        rhs = _rhs(system, coefficient, solution)
        result = restarted_gmres(
            system.matrix,
            rhs,
            settings.solver,
            preconditioner,
        )
        exact = solution.values(system.nodes)
        solution_error = float(
            np.linalg.norm(result.solution - exact) / max(np.linalg.norm(exact), 1.0e-300)
        )
        solve_rows.append(
            {
                "system": spec.name,
                "rhs_index": rhs_index,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_relative_residual": float(result.relative_residuals[-1]),
                "relative_discretization_error": solution_error,
                "solve_seconds": result.solve_seconds,
                "preconditioner_seconds": result.preconditioner_seconds,
                "failure": result.failure,
            }
        )
        for iteration, residual in enumerate(result.relative_residuals):
            histories.append(
                {
                    "system": spec.name,
                    "method": "ILU",
                    "rhs_index": rhs_index,
                    "iteration": iteration,
                    "relative_residual": float(residual),
                }
            )
    return solve_rows, histories


def _nontrivial_reasons(row: Mapping[str, Any]) -> List[str]:
    reasons = []
    if not bool(row.get("ilu_converged", False)):
        reasons.append("ILU failure/stagnation")
    if float(row.get("ilu_iterations_median", 0.0)) > 20.0:
        reasons.append("ILU >20 iterations")
    if float(row.get("ilu_setup_seconds", 0.0)) >= float(row.get("ilu_solve_seconds_median", np.inf)):
        reasons.append("ILU setup >= one solve")
    if float(row.get("ilu_factor_memory_mib", 0.0)) >= 100.0 or float(
        row.get("ilu_factor_fill_ratio", 0.0)
    ) >= 4.0:
        reasons.append("substantial ILU memory/fill")
    return reasons


def run_discovery(
    output: Path,
    settings: Stage9Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Stage9Spec]]:
    """Run scale/PDE/geometry discovery with the unmodified strong ILU."""

    rows: List[Dict[str, Any]] = []
    all_histories: List[Dict[str, Any]] = []
    spec_map = {spec.name: spec for spec in discovery_specs(settings)}
    failures: List[Dict[str, Any]] = []
    for index, spec in enumerate(spec_map.values()):
        start = perf_counter()
        try:
            coefficient = spec.coefficient()
            system = assemble_system(spec.config(), coefficient)
            assembly_seconds = perf_counter() - start
            matrix_bytes = sparse_storage_bytes(system.matrix)
            local = _local_condition_sample(system)
            preconditioner = ilu_preconditioner(
                system.matrix,
                settings.classical_ilu,
                name="strong_ilu",
            )
            rng = np.random.default_rng(settings.seed + index)
            applications = [rng.normal(size=spec.n_nodes) for _ in range(20)]
            preconditioner.reset_timing()
            for vector in applications:
                preconditioner.apply(vector / np.linalg.norm(vector))
            application_seconds = preconditioner.apply_seconds / len(applications)
            condition_proxy = _preconditioned_condition_proxy(
                system.matrix,
                preconditioner,
                dimension=20,
                seed=settings.seed + 1000 + index,
            )
            # Discretization correctness is checked independently of ILU.  A
            # direct sparse factor is an offline diagnostic only; no inverse
            # matrix is formed and it is not counted as an online method.
            reference_solution = ManufacturedSolution()
            reference_rhs = _rhs(system, coefficient, reference_solution)
            reference_start = perf_counter()
            reference_factor = spla.splu(system.matrix.tocsc())
            reference_factor_seconds = perf_counter() - reference_start
            reference_numerical, reference_residual = _accurate_factor_solve(
                system.matrix, reference_factor, reference_rhs
            )
            reference_exact = reference_solution.values(system.nodes)
            reference_error = float(
                np.linalg.norm(reference_numerical - reference_exact)
                / max(np.linalg.norm(reference_exact), 1.0e-300)
            )
            solve_rows, histories = _solve_rows(
                spec, system, coefficient, preconditioner, settings
            )
            all_histories.extend(histories)
            iterations = np.asarray([row["iterations"] for row in solve_rows], dtype=float)
            solve_seconds = np.asarray([row["solve_seconds"] for row in solve_rows], dtype=float)
            errors = np.asarray(
                [row["relative_discretization_error"] for row in solve_rows], dtype=float
            )
            converged = all(bool(row["converged"]) for row in solve_rows)
            row: Dict[str, Any] = {
                "system": spec.name,
                "family": spec.family,
                "n_nodes": spec.n_nodes,
                "stencil_size": spec.stencil_size,
                "node_distribution": spec.node_distribution,
                "jitter_fraction": spec.jitter_fraction,
                "warp_strength": spec.warp_strength,
                "density_exponent": spec.density_exponent,
                "node_irregularity_cv": system.diagnostics["nearest_neighbor_cv"],
                "nearest_neighbor_ratio": system.diagnostics["nearest_neighbor_ratio"],
                "coefficient_kind": spec.coefficient_kind,
                "coefficient_contrast": spec.coefficient_contrast,
                "anisotropy_angle_degrees": spec.anisotropy_angle,
                "nnz": int(system.matrix.nnz),
                "matrix_memory_bytes": matrix_bytes,
                "matrix_memory_mib": matrix_bytes / 2.0**20,
                "assembly_seconds": assembly_seconds,
                "symmetry_measure": system.diagnostics["symmetry_measure"],
                **local,
                "preconditioned_condition_proxy": condition_proxy,
                "reference_factor_seconds": reference_factor_seconds,
                "reference_relative_residual": reference_residual,
                "ilu_drop_tolerance": settings.classical_ilu.drop_tolerance,
                "ilu_fill_factor_budget": settings.classical_ilu.fill_factor,
                "ilu_setup_seconds": preconditioner.setup_seconds,
                "ilu_factor_nnz": preconditioner.diagnostics["factor_nnz"],
                "ilu_factor_fill_ratio": preconditioner.diagnostics["factor_fill_ratio"],
                "ilu_factor_memory_bytes": preconditioner.diagnostics["factor_memory_bytes"],
                "ilu_factor_memory_mib": preconditioner.diagnostics["factor_memory_bytes"] / 2.0**20,
                "ilu_application_seconds": application_seconds,
                "ilu_iterations_median": float(np.median(iterations)),
                "ilu_iterations_max": int(np.max(iterations)),
                "ilu_solve_seconds_median": float(np.median(solve_seconds)),
                "ilu_converged": converged,
                "relative_discretization_error": reference_error,
                "iterative_solution_error_median": float(np.median(errors)),
                "relative_discretization_error_max": float(np.max(errors)),
                "total_seconds_one_solve": assembly_seconds
                + preconditioner.setup_seconds
                + float(np.median(solve_seconds)),
                "numerically_meaningful": bool(
                    reference_residual <= 1.0e-9
                    and reference_error <= settings.meaningful_error_limit
                ),
            }
            reasons = _nontrivial_reasons(row)
            row["nontrivial_hybrid_regime"] = bool(reasons)
            row["nontrivial_reasons"] = "; ".join(reasons)
            rows.append(row)
        except Exception as exc:
            failure = {
                "system": spec.name,
                "family": spec.family,
                "stage": "discovery",
                "failure": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            rows.append(
                {
                    "system": spec.name,
                    "family": spec.family,
                    "n_nodes": spec.n_nodes,
                    "stencil_size": spec.stencil_size,
                    "node_distribution": spec.node_distribution,
                    "coefficient_kind": spec.coefficient_kind,
                    "coefficient_contrast": spec.coefficient_contrast,
                    "ilu_converged": False,
                    "numerically_meaningful": False,
                    "nontrivial_hybrid_regime": True,
                    "nontrivial_reasons": "assembly/ILU/solve failure",
                    "failure": failure["failure"],
                }
            )
    _write_csv(output / "discovery_systems.csv", rows)
    _write_csv(output / "discovery_residual_histories.csv", all_histories)
    _write_csv(output / "failures.csv", failures)
    return rows, all_histories, spec_map


def _select_analysis_systems(
    discovery_rows: Sequence[Mapping[str, Any]],
    spec_map: Mapping[str, Stage9Spec],
) -> List[Stage9Spec]:
    """Select representative valid nontrivial regimes without cherry-picking."""

    valid = [
        row
        for row in discovery_rows
        if bool(row.get("nontrivial_hybrid_regime", False))
        and bool(row.get("numerically_meaningful", False))
    ]
    selected: List[str] = []

    def add(row: Optional[Mapping[str, Any]]) -> None:
        if row is not None and str(row["system"]) not in selected:
            selected.append(str(row["system"]))

    scale = [row for row in valid if row.get("family") == "scale"]
    add(max(scale, key=lambda row: float(row["n_nodes"]), default=None))
    anisotropy = [row for row in valid if row.get("family") == "anisotropy"]
    add(
        max(
            anisotropy,
            key=lambda row: (
                not bool(row.get("ilu_converged", True)),
                float(row.get("ilu_iterations_median", 0.0)),
                float(row.get("coefficient_contrast", 1.0)),
            ),
            default=None,
        )
    )
    scalar = [
        row
        for row in valid
        if row.get("family") in {"smooth_contrast", "localized_contrast"}
    ]
    add(
        max(
            scalar,
            key=lambda row: float(row.get("coefficient_contrast", 1.0)),
            default=None,
        )
    )
    geometry = [row for row in valid if row.get("family") == "node_geometry"]
    add(
        max(
            geometry,
            key=lambda row: float(row.get("node_irregularity_cv", 0.0)),
            default=None,
        )
    )
    # A failed ILU solve is scientifically important even if another member of
    # the same family was already selected.
    difficult = sorted(
        valid,
        key=lambda row: (
            bool(row.get("ilu_converged", True)),
            -float(row.get("ilu_iterations_median", 0.0)),
        ),
    )
    if difficult:
        add(difficult[0])
    return [spec_map[name] for name in selected[:5]]


def _snapshot_pair(
    system: RBFSystem,
    preconditioner: InverseAction,
    settings: Stage9Settings,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    r"""Build train/test snapshots of ``A^-1r-M^-1r`` without an inverse."""

    start = perf_counter()
    train_blocks = [
        generate_synthetic_residuals(
            system.matrix,
            system.nodes,
            system.boundary,
            settings.snapshot_train_each,
            seed + 11,
        ),
        generate_gmres_residuals(
            system.matrix,
            system.nodes,
            settings.snapshot_train_each,
            seed + 17,
            settings.solver,
            iterations_per_trajectory=12,
        ),
    ]
    test_blocks = [
        generate_synthetic_residuals(
            system.matrix,
            system.nodes,
            system.boundary,
            settings.snapshot_test_each,
            seed + 101,
        ),
        generate_gmres_residuals(
            system.matrix,
            system.nodes,
            settings.snapshot_test_each,
            seed + 107,
            settings.solver,
            iterations_per_trajectory=12,
        ),
    ]
    residual_seconds = perf_counter() - start
    train_residuals = np.column_stack(train_blocks)
    test_residuals = np.column_stack(test_blocks)
    factor_start = perf_counter()
    factor = spla.splu(system.matrix.tocsc())
    factor_seconds = perf_counter() - factor_start

    def labels(block: np.ndarray) -> Tuple[np.ndarray, float]:
        corrections = []
        maximum_residual = 0.0
        for column in range(block.shape[1]):
            residual = block[:, column]
            exact, exact_residual = _accurate_factor_solve(
                system.matrix, factor, residual
            )
            maximum_residual = max(
                maximum_residual,
                exact_residual,
            )
            corrections.append(exact - preconditioner.apply(residual))
        return np.column_stack(corrections), maximum_residual

    label_start = perf_counter()
    train_corrections, train_residual = labels(train_residuals)
    test_corrections, test_residual = labels(test_residuals)
    label_seconds = perf_counter() - label_start
    return (
        train_residuals,
        train_corrections,
        test_residuals,
        test_corrections,
        {
            "residual_generation_seconds": residual_seconds,
            "reference_factor_seconds": factor_seconds,
            "label_seconds": label_seconds,
            "snapshot_total_seconds": residual_seconds + factor_seconds + label_seconds,
            "max_reference_relative_residual": max(train_residual, test_residual),
        },
    )


def run_correction_analysis(
    output: Path,
    selected: Sequence[Stage9Spec],
    settings: Stage9Settings,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, Any]],
]:
    """Re-evaluate low-dimensionality only on selected nontrivial regimes."""

    summary: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    bases: Dict[str, np.ndarray] = {}
    contexts: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    (output / "subspaces").mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(selected):
        try:
            coefficient = spec.coefficient()
            assembly_start = perf_counter()
            system = assemble_system(spec.config(), coefficient)
            assembly_seconds = perf_counter() - assembly_start
            preconditioner = ilu_preconditioner(
                system.matrix, settings.classical_ilu, name="strong_ilu"
            )
            (
                train_residuals,
                train_corrections,
                test_residuals,
                test_corrections,
                timing,
            ) = _snapshot_pair(
                system,
                preconditioner,
                settings,
                settings.seed + 10000 * index,
            )
            svd_start = perf_counter()
            pod = compute_pod(train_corrections)
            svd_seconds = perf_counter() - svd_start
            q90 = energy_dimension(pod.captured_energy, 0.90)
            q95 = energy_dimension(pod.captured_energy, 0.95)
            q99 = energy_dimension(pod.captured_energy, 0.99)

            def eta(q: int) -> float:
                return float(
                    np.median(
                        relative_projection_errors(pod.basis[:, :q], test_corrections)
                    )
                )

            eta90, eta95, eta99 = eta(q90), eta(q95), eta(q99)
            summary.append(
                {
                    "system": spec.name,
                    "family": spec.family,
                    "n_nodes": spec.n_nodes,
                    "train_snapshots": train_corrections.shape[1],
                    "test_snapshots": test_corrections.shape[1],
                    "q90": q90,
                    "q90_over_n": q90 / spec.n_nodes,
                    "eta_q90_median": eta90,
                    "q95": q95,
                    "q95_over_n": q95 / spec.n_nodes,
                    "eta_q95_median": eta95,
                    "q99": q99,
                    "q99_over_n": q99 / spec.n_nodes,
                    "eta_q99_median": eta99,
                    "structured_at_95": bool(
                        q95 / spec.n_nodes <= settings.structured_q95_fraction_limit
                        and eta95 <= settings.structured_eta95_limit
                    ),
                    "assembly_seconds": assembly_seconds,
                    "pod_svd_seconds": svd_seconds,
                    **timing,
                }
            )
            q_max = pod.singular_values.size
            for q in range(1, q_max + 1):
                errors = relative_projection_errors(
                    pod.basis[:, :q], test_corrections
                )
                curves.append(
                    {
                        "system": spec.name,
                        "q": q,
                        "q_over_n": q / spec.n_nodes,
                        "normalized_singular_value": float(
                            pod.singular_values[q - 1] / pod.singular_values[0]
                        ),
                        "captured_energy": float(pod.captured_energy[q - 1]),
                        "heldout_projection_error_median": float(np.median(errors)),
                        "heldout_projection_error_p90": float(np.quantile(errors, 0.9)),
                    }
                )
            bases[spec.name] = pod.basis
            contexts[spec.name] = {
                "spec": spec,
                "system": system,
                "preconditioner": preconditioner,
                "train_corrections": train_corrections,
                "test_corrections": test_corrections,
                "snapshot_timing": timing,
                "pod": pod,
            }
            np.savez_compressed(
                output / "subspaces" / f"{spec.name}_correction_pod.npz",
                basis=pod.basis,
                singular_values=pod.singular_values,
                captured_energy=pod.captured_energy,
                test_corrections=test_corrections,
            )
        except Exception as exc:
            failures.append(
                {
                    "system": spec.name,
                    "stage": "correction_analysis",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_csv(output / "correction_structure_summary.csv", summary)
    _write_csv(output / "correction_structure_curves.csv", curves)
    _write_csv(output / "correction_analysis_failures.csv", failures)
    return summary, curves, bases, contexts


def _oracle_rhs_set(
    system: RBFSystem,
    coefficient: Optional[object],
    seed: int,
) -> List[np.ndarray]:
    solutions = [
        ManufacturedSolution(),
        random_smooth_solution(seed + 1, 4),
        random_smooth_solution(seed + 2, 4),
    ]
    return [_rhs(system, coefficient, solution) for solution in solutions]


def _run_method(
    system: RBFSystem,
    rhs_set: Sequence[np.ndarray],
    preconditioner: InverseAction,
    solver: SolverConfig,
    system_name: str,
    method: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: List[KrylovResult] = []
    histories: List[Dict[str, Any]] = []
    for rhs_index, rhs in enumerate(rhs_set):
        result = restarted_gmres(system.matrix, rhs, solver, preconditioner)
        results.append(result)
        for iteration, residual in enumerate(result.relative_residuals):
            histories.append(
                {
                    "system": system_name,
                    "method": method,
                    "rhs_index": rhs_index,
                    "iteration": iteration,
                    "relative_residual": float(residual),
                }
            )
    return (
        {
            "iterations_median": float(np.median([result.iterations for result in results])),
            "iterations_max": int(max(result.iterations for result in results)),
            "convergence_rate": float(np.mean([result.converged for result in results])),
            "solve_seconds_median": float(np.median([result.solve_seconds for result in results])),
            "final_relative_residual_max": float(
                max(result.relative_residuals[-1] for result in results)
            ),
            "failures": "; ".join(
                sorted({result.failure for result in results if result.failure})
            ),
        },
        histories,
    )


def run_oracle_analysis(
    output: Path,
    correction_contexts: Mapping[str, Mapping[str, Any]],
    settings: Stage9Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compare ILU with per-system correction-POD and Ritz oracle spaces."""

    rows: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    selections: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for system_index, (name, context) in enumerate(correction_contexts.items()):
        spec: Stage9Spec = context["spec"]
        system: RBFSystem = context["system"]
        coefficient = spec.coefficient()
        rhs_set = _oracle_rhs_set(system, coefficient, settings.seed + system_index * 100)
        baseline = ilu_preconditioner(system.matrix, settings.classical_ilu, name="strong_ilu")
        baseline_metrics, baseline_history = _run_method(
            system, rhs_set, baseline, settings.solver, name, "ILU"
        )
        histories.extend(baseline_history)
        rows.append(
            {
                "system": name,
                "basis_definition": "none",
                "q": 0,
                "reference_setup_seconds": 0.0,
                "coarse_setup_seconds": 0.0,
                "total_setup_seconds": baseline.setup_seconds,
                "coarse_memory_mib": 0.0,
                "reduced_condition": float("nan"),
                "iteration_reduction": 0.0,
                **baseline_metrics,
            }
        )
        pod = context["pod"]
        valid_q = [q for q in settings.oracle_q_values if q <= pod.basis.shape[1]]
        for definition in ("correction_pod", "ritz"):
            for q in valid_q:
                try:
                    reference_start = perf_counter()
                    if definition == "correction_pod":
                        basis = pod.basis[:, :q]
                        reference_seconds = float(
                            context["snapshot_timing"]["snapshot_total_seconds"]
                        ) + float(context.get("pod_svd_seconds", 0.0))
                    else:
                        ritz_base = ilu_preconditioner(
                            system.matrix, settings.classical_ilu, name="strong_ilu"
                        )
                        basis, ritz_diagnostics = krylov_ritz_basis(
                            system.matrix,
                            ritz_base,
                            q=q,
                            arnoldi_dimension=max(settings.ritz_arnoldi_dimension, q + 8),
                            seed=settings.seed + 1000 * system_index + q,
                        )
                        reference_seconds = perf_counter() - reference_start
                    base = ilu_preconditioner(
                        system.matrix, settings.classical_ilu, name="strong_ilu"
                    )
                    action, diagnostics = build_multiplicative_coarse_preconditioner(
                        system.matrix,
                        base,
                        basis,
                        name=f"{definition}_oracle",
                    )
                    method = f"ILU+{definition}_q{q}"
                    metrics, method_history = _run_method(
                        system, rhs_set, action, settings.solver, name, method
                    )
                    histories.extend(method_history)
                    reduction = (
                        baseline_metrics["iterations_median"] - metrics["iterations_median"]
                    ) / max(baseline_metrics["iterations_median"], 1.0)
                    memory_bytes = int(
                        basis.nbytes
                        + (system.matrix @ basis).nbytes
                        + 2 * q * q * np.dtype(float).itemsize
                    )
                    rows.append(
                        {
                            "system": name,
                            "basis_definition": definition,
                            "q": q,
                            "reference_setup_seconds": reference_seconds,
                            "coarse_setup_seconds": diagnostics["coarse_setup_seconds"],
                            "total_setup_seconds": base.setup_seconds
                            + reference_seconds
                            + diagnostics["coarse_setup_seconds"],
                            "coarse_memory_mib": memory_bytes / 2.0**20,
                            "coarse_variant": diagnostics["coarse_variant"],
                            "reduced_condition": diagnostics["reduced_condition"],
                            "iteration_reduction": reduction,
                            **metrics,
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "system": name,
                            "stage": "oracle",
                            "basis_definition": definition,
                            "q": q,
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    )
        candidates = [
            row
            for row in rows
            if row["system"] == name
            and row["basis_definition"] != "none"
            and float(row["convergence_rate"]) == 1.0
        ]
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    float(row["iterations_median"]),
                    float(row["solve_seconds_median"]),
                    float(row["total_setup_seconds"]),
                ),
            )
            selections.append(
                {
                    **best,
                    "oracle_useful": bool(
                        float(best["iteration_reduction"])
                        >= settings.oracle_iteration_reduction_gate
                    ),
                }
            )
    _write_csv(output / "oracle_coarse_results.csv", rows)
    _write_csv(output / "oracle_residual_histories.csv", histories)
    _write_csv(output / "oracle_selection.csv", selections)
    _write_csv(output / "oracle_failures.csv", failures)
    return rows, histories, selections


def run_transferability(
    output: Path,
    oracle_selections: Sequence[Mapping[str, Any]],
    settings: Stage9Settings,
) -> Dict[str, Any]:
    """Test Ritz-space stability and nearby reuse in the useful anisotropy family."""

    anisotropic = next(
        (
            row
            for row in oracle_selections
            if row.get("system") == "anisotropy_R100"
            and bool(row.get("oracle_useful", False))
            and row.get("basis_definition") == "ritz"
        ),
        None,
    )
    if anisotropic is None:
        result = {
            "performed": False,
            "reason": "No useful anisotropic Ritz oracle passed the gate.",
        }
        _write_json(output / "transferability_gate.json", result)
        return result

    q = int(float(anisotropic["q"]))
    ratios = (60.0, 80.0, 100.0, 120.0, 140.0)
    replicate_count = 3
    systems: Dict[float, RBFSystem] = {}
    bases: Dict[Tuple[float, int], np.ndarray] = {}
    construction_rows: List[Dict[str, Any]] = []
    (output / "transfer_subspaces").mkdir(parents=True, exist_ok=True)
    for ratio in ratios:
        spec = Stage9Spec(
            name=f"transfer_anisotropy_R{ratio:g}",
            family="anisotropy_transfer",
            coefficient_kind="anisotropic",
            coefficient_contrast=ratio,
        )
        system = assemble_system(spec.config(), spec.coefficient())
        systems[ratio] = system
        for replicate in range(replicate_count):
            base = ilu_preconditioner(system.matrix, settings.classical_ilu)
            basis, diagnostics = krylov_ritz_basis(
                system.matrix,
                base,
                q=q,
                arnoldi_dimension=max(settings.ritz_arnoldi_dimension, q + 8),
                seed=settings.seed + int(ratio) * 100 + replicate,
            )
            bases[(ratio, replicate)] = basis
            np.savez_compressed(
                output
                / "transfer_subspaces"
                / f"anisotropy_R{ratio:g}_ritz_q{q}_rep{replicate}.npz",
                basis=basis,
                ratio=ratio,
                q=q,
            )
            construction_rows.append(
                {
                    "anisotropy_ratio": ratio,
                    "replicate": replicate,
                    "q": q,
                    "ritz_setup_seconds": diagnostics["setup_seconds"],
                    "arnoldi_dimension": diagnostics["arnoldi_dimension"],
                }
            )

    variability_rows: List[Dict[str, Any]] = []
    within = []
    for ratio in ratios:
        for first in range(replicate_count):
            for second in range(first + 1, replicate_count):
                metric = subspace_metrics(bases[(ratio, first)], bases[(ratio, second)])
                within.append(metric["projector_distance_normalized"])
                variability_rows.append(
                    {
                        "comparison": "within",
                        "ratio_i": ratio,
                        "ratio_j": ratio,
                        "replicate_i": first,
                        "replicate_j": second,
                        **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
                    }
                )

    between = []
    parameter_distances = []
    distance_matrix = np.zeros((len(ratios), len(ratios)), dtype=float)
    angle_matrix = np.zeros_like(distance_matrix)
    for i, first_ratio in enumerate(ratios):
        for j, second_ratio in enumerate(ratios):
            metric = subspace_metrics(bases[(first_ratio, 0)], bases[(second_ratio, 0)])
            distance_matrix[i, j] = metric["projector_distance_normalized"]
            angle_matrix[i, j] = metric["principal_angle_mean_degrees"]
            if i < j:
                between.append(metric["projector_distance_normalized"])
                parameter_distance = abs(np.log10(first_ratio) - np.log10(second_ratio))
                parameter_distances.append(parameter_distance)
                variability_rows.append(
                    {
                        "comparison": "between",
                        "ratio_i": first_ratio,
                        "ratio_j": second_ratio,
                        "replicate_i": 0,
                        "replicate_j": 0,
                        "log10_parameter_distance": parameter_distance,
                        **{key: value for key, value in metric.items() if key != "principal_angles_degrees"},
                    }
                )

    reuse_rows: List[Dict[str, Any]] = []
    for target_ratio in ratios:
        system = systems[target_ratio]
        coefficient = ConstantAnisotropicDiffusion(target_ratio, 30.0)
        rhs = _rhs(system, coefficient, random_smooth_solution(settings.seed + 701, 4))
        baseline = ilu_preconditioner(system.matrix, settings.classical_ilu)
        baseline_result = restarted_gmres(
            system.matrix, rhs, settings.solver, baseline
        )
        for source_ratio in ratios:
            base = ilu_preconditioner(system.matrix, settings.classical_ilu)
            action, diagnostics = build_multiplicative_coarse_preconditioner(
                system.matrix,
                base,
                bases[(source_ratio, 0)],
                name=f"reused_ritz_R{source_ratio:g}",
            )
            result = restarted_gmres(system.matrix, rhs, settings.solver, action)
            reuse_rows.append(
                {
                    "source_ratio": source_ratio,
                    "target_ratio": target_ratio,
                    "absolute_ratio_distance": abs(source_ratio - target_ratio),
                    "log10_parameter_distance": abs(
                        np.log10(source_ratio) - np.log10(target_ratio)
                    ),
                    "baseline_iterations": baseline_result.iterations,
                    "reused_iterations": result.iterations,
                    "iteration_reduction": (
                        baseline_result.iterations - result.iterations
                    )
                    / max(baseline_result.iterations, 1),
                    "baseline_solve_seconds": baseline_result.solve_seconds,
                    "reused_solve_seconds": result.solve_seconds,
                    "converged": result.converged,
                    "coarse_setup_seconds": diagnostics["coarse_setup_seconds"],
                    "failure": result.failure,
                }
            )

    correlation = stats.spearmanr(parameter_distances, between)
    within_median = float(np.median(within))
    between_median = float(np.median(between))
    result = {
        "performed": True,
        "basis_definition": "ritz",
        "q": q,
        "ratios": ratios,
        "within_projector_distance_median": within_median,
        "within_projector_distance_p90": float(np.quantile(within, 0.9)),
        "between_projector_distance_median": between_median,
        "between_projector_distance_p10": float(np.quantile(between, 0.1)),
        "within_to_between_median_ratio": within_median
        / max(between_median, np.finfo(float).tiny),
        "spearman_parameter_vs_projector": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "nearby_reuse_iteration_reduction_median": float(
            np.median(
                [
                    row["iteration_reduction"]
                    for row in reuse_rows
                    if row["source_ratio"] != row["target_ratio"]
                    and row["log10_parameter_distance"] <= 0.13
                ]
            )
        ),
    }
    _write_csv(output / "transfer_construction.csv", construction_rows)
    _write_csv(output / "transfer_variability.csv", variability_rows)
    _write_csv(output / "transfer_reuse.csv", reuse_rows)
    np.savez_compressed(
        output / "transfer_distance_matrices.npz",
        ratios=np.asarray(ratios),
        normalized_projector_distance=distance_matrix,
        principal_angle_mean_degrees=angle_matrix,
    )
    _write_json(output / "transferability_gate.json", result)
    return result


def build_cost_model(
    output: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    oracle_selections: Sequence[Mapping[str, Any]],
    settings: Stage9Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build fixed-A, changing-A, and oracle repeated-solve cost accounting."""

    rows: List[Dict[str, Any]] = []
    break_even: List[Dict[str, Any]] = []
    numeric_discovery = [row for row in discovery_rows if "ilu_setup_seconds" in row]
    largest = max(
        (row for row in numeric_discovery if row.get("family") == "scale"),
        key=lambda row: float(row["n_nodes"]),
    )
    changing = [
        row
        for row in numeric_discovery
        if row.get("family") in {"smooth_contrast", "anisotropy"}
    ]
    changing_assembly = float(np.mean([float(row["assembly_seconds"]) for row in changing]))
    changing_setup = float(np.mean([float(row["ilu_setup_seconds"]) for row in changing]))
    changing_solve = float(
        np.mean([float(row["ilu_solve_seconds_median"]) for row in changing])
    )
    for k in settings.repeated_counts:
        rows.append(
            {
                "case": "A_fixed_matrix",
                "system": largest["system"],
                "method": "ILU",
                "K": k,
                "assembly_seconds": float(largest["assembly_seconds"]),
                "offline_seconds": 0.0,
                "online_setup_seconds": float(largest["ilu_setup_seconds"]),
                "solve_seconds": k * float(largest["ilu_solve_seconds_median"]),
                "online_only_total_seconds": float(largest["ilu_setup_seconds"])
                + k * float(largest["ilu_solve_seconds_median"]),
                "fully_amortized_total_seconds": float(largest["assembly_seconds"])
                + float(largest["ilu_setup_seconds"])
                + k * float(largest["ilu_solve_seconds_median"]),
                "modeled_not_rerun": True,
            }
        )
        rows.append(
            {
                "case": "B_changing_matrix",
                "system": "smooth_contrast_and_anisotropy_family_mean",
                "method": "ILU",
                "K": k,
                "assembly_seconds": k * changing_assembly,
                "offline_seconds": 0.0,
                "online_setup_seconds": k * changing_setup,
                "solve_seconds": k * changing_solve,
                "online_only_total_seconds": k * (changing_setup + changing_solve),
                "fully_amortized_total_seconds": k
                * (changing_assembly + changing_setup + changing_solve),
                "modeled_not_rerun": True,
            }
        )

    for selection in oracle_selections:
        system_name = str(selection["system"])
        baseline = next(
            row
            for row in oracle_rows
            if row["system"] == system_name and row["basis_definition"] == "none"
        )
        discovery = next(row for row in numeric_discovery if row["system"] == system_name)
        base_setup = float(discovery["ilu_setup_seconds"])
        base_solve = float(baseline["solve_seconds_median"])
        hybrid_solve = float(selection["solve_seconds_median"])
        reference_setup = float(selection["reference_setup_seconds"])
        coarse_setup = float(selection["coarse_setup_seconds"])
        denominator = base_solve - hybrid_solve
        analytical = (
            (reference_setup + coarse_setup) / denominator
            if denominator > 0.0
            else float("inf")
        )
        break_even.append(
            {
                "system": system_name,
                "basis_definition": selection["basis_definition"],
                "q": selection["q"],
                "break_even_K_full_vs_ILU": analytical,
                "hybrid_online_faster_per_solve": denominator > 0.0,
            }
        )
        for k in settings.repeated_counts:
            rows.extend(
                [
                    {
                        "case": "A_fixed_matrix",
                        "system": system_name,
                        "method": "ILU",
                        "K": k,
                        "assembly_seconds": float(discovery["assembly_seconds"]),
                        "offline_seconds": 0.0,
                        "online_setup_seconds": base_setup,
                        "solve_seconds": k * base_solve,
                        "online_only_total_seconds": base_setup + k * base_solve,
                        "fully_amortized_total_seconds": float(discovery["assembly_seconds"])
                        + base_setup
                        + k * base_solve,
                        "modeled_not_rerun": True,
                    },
                    {
                        "case": "A_fixed_matrix",
                        "system": system_name,
                        "method": f"ILU+oracle_{selection['basis_definition']}_q{int(float(selection['q']))}",
                        "K": k,
                        "assembly_seconds": float(discovery["assembly_seconds"]),
                        "offline_seconds": reference_setup,
                        "online_setup_seconds": base_setup + coarse_setup,
                        "solve_seconds": k * hybrid_solve,
                        "online_only_total_seconds": base_setup
                        + coarse_setup
                        + k * hybrid_solve,
                        "fully_amortized_total_seconds": float(discovery["assembly_seconds"])
                        + base_setup
                        + reference_setup
                        + coarse_setup
                        + k * hybrid_solve,
                        "modeled_not_rerun": True,
                    },
                ]
            )
    _write_csv(output / "repeated_solve_costs.csv", rows)
    _write_csv(output / "break_even_costs.csv", break_even)
    return rows, break_even


def _direct_system(ratio: float) -> Tuple[Stage9Spec, RBFSystem, Any]:
    spec = Stage9Spec(
        name=f"direct_anisotropy_R{ratio:g}",
        family="direct_anisotropy",
        coefficient_kind="anisotropic",
        coefficient_contrast=ratio,
    )
    system = assemble_system(spec.config(), spec.coefficient())
    factor = spla.splu(system.matrix.tocsc())
    return spec, system, factor


def _label_vectors(
    system: RBFSystem,
    factor: Any,
    base: InverseAction,
    vectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    corrections = []
    residuals = []
    for vector in np.asarray(vectors):
        exact, reference_residual = _accurate_factor_solve(system.matrix, factor, vector)
        corrections.append(exact - base.apply(vector))
        residuals.append(reference_residual)
    return np.row_stack(corrections), np.asarray(residuals)


def _collect_direct_trajectory(
    system: RBFSystem,
    factor: Any,
    settings: Stage9Settings,
    seed: int,
    predictor: Optional[Any] = None,
    features: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    max_iterations: int = 120,
) -> Tuple[np.ndarray, np.ndarray, KrylovResult, Dict[str, float]]:
    """Collect and label exact Arnoldi inputs under classical or learned policy."""

    solver = SolverConfig(
        relative_tolerance=settings.solver.relative_tolerance,
        absolute_tolerance=0.0,
        restart=50,
        max_iterations=max_iterations,
        breakdown_tolerance=settings.solver.breakdown_tolerance,
    )
    exact = random_smooth_solution(seed, 6).values(system.nodes)
    exact[system.boundary] = 0.0
    rhs = np.asarray(system.matrix @ exact, dtype=float)
    base = ilu_preconditioner(system.matrix, settings.classical_ilu, name="strong_ilu")
    audit: Dict[str, float] = {"prediction_seconds": 0.0, "applications": 0.0}
    if predictor is None:
        action = base
    else:
        assert features is not None
        action, audit = make_stabilized_direct_preconditioner(
            base,
            lambda vector: predictor.predict(vector, features),
            alpha=alpha,
            norm_limit_ratio=None,
            name="stage8_mlp",
        )
    traced: List[np.ndarray] = []
    result = restarted_gmres(
        system.matrix,
        rhs,
        solver,
        action,
        preconditioner_input_callback=lambda _, vector: traced.append(vector.copy()),
    )
    vector_array = np.row_stack(traced)
    label_base = ilu_preconditioner(system.matrix, settings.classical_ilu)
    corrections, reference_residuals = _label_vectors(
        system, factor, label_base, vector_array
    )
    diagnostics = {
        "max_reference_relative_residual": float(np.max(reference_residuals)),
        "prediction_seconds": float(audit.get("prediction_seconds", 0.0)),
    }
    return vector_array, corrections, result, diagnostics


def run_direct_if_gated(
    output: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
    oracle_selections: Sequence[Mapping[str, Any]],
    settings: Stage9Settings,
) -> Dict[str, Any]:
    """Reuse the small Stage 8 MLP/on-policy strategy only if all gates pass."""

    discovery = next(
        (row for row in discovery_rows if row.get("system") == "anisotropy_R100"),
        None,
    )
    correction = next(
        (row for row in correction_rows if row.get("system") == "anisotropy_R100"),
        None,
    )
    oracle = next(
        (row for row in oracle_selections if row.get("system") == "anisotropy_R100"),
        None,
    )
    gates = {
        "meaningful_ilu_difficulty": bool(
            discovery
            and bool(discovery.get("numerically_meaningful", False))
            and (
                not bool(discovery.get("ilu_converged", True))
                or float(discovery.get("ilu_iterations_median", 0.0)) > 20.0
            )
        ),
        "structured_unresolved_correction": bool(
            correction and bool(correction.get("structured_at_95", False))
        ),
        "repeated_parameter_family_available": True,
        "oracle_useful": bool(oracle and bool(oracle.get("oracle_useful", False))),
    }
    if not all(gates.values()):
        result = {
            "performed": False,
            "gates": gates,
            "reason": "The preregistered direct-learning gates did not all pass.",
        }
        _write_json(output / "direct_gate.json", result)
        return result

    ratios = {"train": (60.0, 80.0), "validation": (90.0,), "test": (100.0,)}
    contexts = {
        ratio: _direct_system(ratio)
        for ratio in sorted(set(sum((values for values in ratios.values()), ())))
    }
    feature = lambda ratio: np.asarray([np.log10(ratio), 30.0 / 90.0], dtype=float)
    train_blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    validation_blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []

    for split, target in (("train", train_blocks), ("validation", validation_blocks)):
        for index, ratio in enumerate(ratios[split]):
            _, system, factor = contexts[ratio]
            vectors, corrections, result, diagnostic = _collect_direct_trajectory(
                system,
                factor,
                settings,
                settings.seed + 200000 + 1000 * index + int(ratio),
                max_iterations=120,
            )
            features = np.repeat(feature(ratio)[None, :], vectors.shape[0], axis=0)
            target.append((vectors, corrections, features))
            trajectory_rows.append(
                {
                    "round": -1,
                    "split": split,
                    "ratio": ratio,
                    "method": "classical_ILU_collection",
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "solve_seconds": result.solve_seconds,
                    **diagnostic,
                }
            )

    config = DirectMLPConfig(
        hidden_width=64,
        hidden_layers=1,
        learning_rate=1.0e-3,
        weight_decay=1.0e-6,
        batch_size=8,
        max_epochs=60,
        patience=12,
        seed=settings.seed,
        input_direction_scale=1.0,
    )
    final_training = None
    for round_index in range(4):
        train_vectors = np.vstack([block[0] for block in train_blocks])
        train_corrections = np.vstack([block[1] for block in train_blocks])
        train_features = np.vstack([block[2] for block in train_blocks])
        validation_vectors = np.vstack([block[0] for block in validation_blocks])
        validation_corrections = np.vstack([block[1] for block in validation_blocks])
        validation_features = np.vstack([block[2] for block in validation_blocks])
        final_training = train_direct_mlp(
            train_vectors,
            train_corrections,
            train_features,
            validation_vectors,
            validation_corrections,
            validation_features,
            ("log10_anisotropy_ratio", "angle_over_90"),
            config,
        )
        validation_predictions = final_training.predictor.predict_batch(
            validation_vectors, validation_features
        )
        validation_metric = correction_metrics(
            validation_predictions, validation_corrections
        )
        magnitude_median = float(np.median(validation_metric["magnitude_ratio"]))
        magnitude_p95 = float(np.quantile(validation_metric["magnitude_ratio"], 0.95))
        safe = bool(0.5 <= magnitude_median <= 2.0 and magnitude_p95 <= 5.0)
        training_rows.append(
            {
                "round": round_index,
                "training_samples": train_vectors.shape[0],
                "training_seconds": final_training.training_seconds,
                "best_epoch": final_training.best_epoch,
                "best_validation_loss": final_training.best_validation_loss,
                "validation_relative_error_median": float(
                    np.median(validation_metric["relative_error"])
                ),
                "validation_magnitude_ratio_median": magnitude_median,
                "validation_magnitude_ratio_p95": magnitude_p95,
                "validation_cosine_median": float(
                    np.median(validation_metric["cosine_similarity"])
                ),
                "safe_for_integration": safe,
            }
        )
        if round_index == 3 or not safe:
            break
        new_blocks = []
        for index, ratio in enumerate(ratios["train"]):
            _, system, factor = contexts[ratio]
            vectors, corrections, result, diagnostic = _collect_direct_trajectory(
                system,
                factor,
                settings,
                settings.seed + 300000 + round_index * 10000 + index,
                predictor=final_training.predictor,
                features=feature(ratio),
                alpha=1.0,
                max_iterations=120,
            )
            # Retain every traced vector on disk, but bound the retraining set
            # by a deterministic coverage sample at this larger fixed N.
            selection = np.linspace(
                0, vectors.shape[0] - 1, min(60, vectors.shape[0]), dtype=int
            )
            new_blocks.append(
                (
                    vectors[selection],
                    corrections[selection],
                    np.repeat(feature(ratio)[None, :], selection.size, axis=0),
                )
            )
            np.savez_compressed(
                output / f"direct_onpolicy_round{round_index + 1}_R{ratio:g}.npz",
                vectors=vectors,
                corrections=corrections,
                target_definition="A^-1 v - M^-1 v",
            )
            trajectory_rows.append(
                {
                    "round": round_index,
                    "split": "aggregation",
                    "ratio": ratio,
                    "method": "MLP_alpha1",
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "solve_seconds": result.solve_seconds,
                    "samples_collected": vectors.shape[0],
                    "samples_added": selection.size,
                    **diagnostic,
                }
            )
        train_blocks.extend(new_blocks)

    assert final_training is not None
    final_training.predictor.save(output / "direct_stage9_mlp.pt", config, final_training.history)
    final_safe = bool(training_rows[-1]["safe_for_integration"])
    test_ratio = ratios["test"][0]
    _, test_system, test_factor = contexts[test_ratio]
    for trajectory_index in range(2):
        seed = settings.seed + 900000 + trajectory_index
        base_vectors, base_corrections, base_result, base_diag = _collect_direct_trajectory(
            test_system, test_factor, settings, seed, max_iterations=300
        )
        prediction = final_training.predictor.predict_batch(
            base_vectors,
            np.repeat(feature(test_ratio)[None, :], base_vectors.shape[0], axis=0),
        )
        offline_metric = correction_metrics(prediction, base_corrections)
        for index in range(base_vectors.shape[0]):
            metric_rows.append(
                {
                    "trajectory": trajectory_index,
                    "application": index,
                    "relative_error": offline_metric["relative_error"][index],
                    "magnitude_ratio": offline_metric["magnitude_ratio"][index],
                    "cosine_similarity": offline_metric["cosine_similarity"][index],
                }
            )
        trajectory_rows.append(
            {
                "round": 3,
                "split": "heldout",
                "ratio": test_ratio,
                "trajectory": trajectory_index,
                "method": "classical_ILU",
                "iterations": base_result.iterations,
                "converged": base_result.converged,
                "solve_seconds": base_result.solve_seconds,
                **base_diag,
            }
        )
        for alpha in ((0.1, 1.0) if final_safe else ()):
            vectors, corrections, result, diagnostic = _collect_direct_trajectory(
                test_system,
                test_factor,
                settings,
                seed,
                predictor=final_training.predictor,
                features=feature(test_ratio),
                alpha=alpha,
                max_iterations=300,
            )
            prediction = final_training.predictor.predict_batch(
                vectors,
                np.repeat(feature(test_ratio)[None, :], vectors.shape[0], axis=0),
            )
            metric = correction_metrics(prediction, corrections)
            trajectory_rows.append(
                {
                    "round": 3,
                    "split": "heldout",
                    "ratio": test_ratio,
                    "trajectory": trajectory_index,
                    "method": f"MLP_alpha{alpha:g}",
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "solve_seconds": result.solve_seconds,
                    "relative_error_median": float(np.median(metric["relative_error"])),
                    "magnitude_ratio_median": float(np.median(metric["magnitude_ratio"])),
                    "cosine_similarity_median": float(np.median(metric["cosine_similarity"])),
                    **diagnostic,
                }
            )
            for iteration, residual in enumerate(result.relative_residuals):
                history_rows.append(
                    {
                        "trajectory": trajectory_index,
                        "method": f"MLP_alpha{alpha:g}",
                        "iteration": iteration,
                        "relative_residual": float(residual),
                    }
                )
        for iteration, residual in enumerate(base_result.relative_residuals):
            history_rows.append(
                {
                    "trajectory": trajectory_index,
                    "method": "classical_ILU",
                    "iteration": iteration,
                    "relative_residual": float(residual),
                }
            )

    _write_csv(output / "direct_training_rounds.csv", training_rows)
    _write_csv(output / "direct_solver_results.csv", trajectory_rows)
    _write_csv(output / "direct_offline_metrics.csv", metric_rows)
    _write_csv(output / "direct_residual_histories.csv", history_rows)
    heldout = [row for row in trajectory_rows if row.get("split") == "heldout"]
    summary = {}
    for method in ("classical_ILU", "MLP_alpha0.1", "MLP_alpha1"):
        method_rows = [row for row in heldout if row.get("method") == method]
        summary[method] = (
            {
                "convergence_rate": float(np.mean([row["converged"] for row in method_rows])),
                "iterations_median": float(np.median([row["iterations"] for row in method_rows])),
                "solve_seconds_median": float(np.median([row["solve_seconds"] for row in method_rows])),
            }
            if method_rows
            else {
                "convergence_rate": 0.0,
                "iterations_median": float("inf"),
                "solve_seconds_median": float("inf"),
            }
        )
    result = {
        "performed": True,
        "gates": gates,
        "architecture": asdict(config),
        "safe_for_solver_integration": final_safe,
        "training_seconds_total": float(sum(row["training_seconds"] for row in training_rows)),
        "final_validation": training_rows[-1],
        "heldout_summary": summary,
        "direct_robust_improvement": bool(
            summary["MLP_alpha0.1"]["convergence_rate"] == 1.0
            and summary["MLP_alpha0.1"]["iterations_median"]
            < summary["classical_ILU"]["iterations_median"]
            and summary["MLP_alpha0.1"]["solve_seconds_median"]
            < summary["classical_ILU"]["solve_seconds_median"]
        ),
    }
    _write_json(output / "direct_gate.json", result)
    return result


def classify_regimes(
    output: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
    oracle_selections: Sequence[Mapping[str, Any]],
    direct: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply the requested A/B/C/D decision logic to every tested case."""

    correction_map = {row["system"]: row for row in correction_rows}
    oracle_map = {row["system"]: row for row in oracle_selections}
    rows = []
    direct_positive = bool(direct.get("direct_robust_improvement", False))
    for discovery in discovery_rows:
        name = str(discovery["system"])
        if "ilu_setup_seconds" not in discovery:
            category = "B"
            rationale = (
                "Invalid RBF-FD regime: local assembly failed, so no exploitable "
                "hybrid correction can be defined. This is outside the literal B "
                "description but is conservatively ML-negative."
            )
        else:
            hard = bool(
                not bool(discovery.get("ilu_converged", True))
                or float(discovery.get("ilu_iterations_median", 0.0)) > 20.0
                or float(discovery.get("ilu_factor_memory_mib", 0.0)) >= 100.0
                or float(discovery.get("ilu_factor_fill_ratio", 0.0)) >= 4.0
            )
            oracle_positive = bool(
                name in oracle_map and bool(oracle_map[name].get("oracle_useful", False))
            )
            if hard and name == "anisotropy_R100" and direct_positive:
                category = "C+D" if oracle_positive else "D"
                rationale = "Strong ILU is difficult and the held-out direct model is robustly faster."
            elif hard and oracle_positive:
                category = "C"
                rationale = "Strong ILU is difficult and the per-system oracle gives material acceleration."
            elif hard:
                category = "B"
                structured = bool(
                    name in correction_map
                    and correction_map[name].get("structured_at_95", False)
                )
                rationale = (
                    "Strong ILU is difficult, but no tested exploitable hybrid route passed."
                    if not structured
                    else "Correction structure exists, but solver/economic evidence is insufficient."
                )
            else:
                category = "A"
                rationale = "ILU remains fast, compact, and reliable; no ML justification."
        rows.append(
            {
                "system": name,
                "family": discovery.get("family", ""),
                "category": category,
                "rationale": rationale,
            }
        )
    _write_csv(output / "regime_classification.csv", rows)
    return rows


def _append_direct_costs(
    output: Path,
    cost_rows: List[Dict[str, Any]],
    break_even: List[Dict[str, Any]],
    discovery_rows: Sequence[Mapping[str, Any]],
    direct: Mapping[str, Any],
    settings: Stage9Settings,
) -> None:
    if not direct.get("performed", False) or not direct.get("safe_for_solver_integration", False):
        _write_csv(output / "repeated_solve_costs.csv", cost_rows)
        _write_csv(output / "break_even_costs.csv", break_even)
        return
    discovery = next(row for row in discovery_rows if row["system"] == "anisotropy_R100")
    baseline = direct["heldout_summary"]["classical_ILU"]
    learned = direct["heldout_summary"]["MLP_alpha0.1"]
    training = float(direct["training_seconds_total"])
    setup = float(discovery["ilu_setup_seconds"])
    base_solve = float(baseline["solve_seconds_median"])
    learned_solve = float(learned["solve_seconds_median"])
    denominator = base_solve - learned_solve
    break_even_k = training / denominator if denominator > 0.0 else float("inf")
    break_even.append(
        {
            "system": "anisotropy_R100",
            "basis_definition": "direct_MLP_alpha0.1",
            "q": "",
            "break_even_K_full_vs_ILU": break_even_k,
            "hybrid_online_faster_per_solve": denominator > 0.0,
        }
    )
    for k in settings.repeated_counts:
        cost_rows.append(
            {
                "case": "A_fixed_matrix",
                "system": "anisotropy_R100",
                "method": "ILU+direct_MLP_alpha0.1",
                "K": k,
                "assembly_seconds": float(discovery["assembly_seconds"]),
                "offline_seconds": training,
                "online_setup_seconds": setup,
                "solve_seconds": k * learned_solve,
                "online_only_total_seconds": setup + k * learned_solve,
                "fully_amortized_total_seconds": float(discovery["assembly_seconds"])
                + training
                + setup
                + k * learned_solve,
                "modeled_not_rerun": True,
            }
        )
    _write_csv(output / "repeated_solve_costs.csv", cost_rows)
    _write_csv(output / "break_even_costs.csv", break_even)


def make_plots(
    output: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
    correction_curves: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    transfer: Mapping[str, Any],
    direct: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Generate decision-linked figures and a chart-to-data map."""

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    chart_map: List[Dict[str, str]] = []

    scale = sorted(
        [row for row in discovery_rows if row.get("family") == "scale"],
        key=lambda row: float(row["n_nodes"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    n = np.asarray([float(row["n_nodes"]) for row in scale])
    axes[0].plot(n, [row["assembly_seconds"] for row in scale], "o-", color=BLUE, label="assembly")
    axes[0].plot(n, [row["ilu_setup_seconds"] for row in scale], "s-", color=ORANGE, label="ILU setup")
    axes[0].plot(n, [row["ilu_solve_seconds_median"] for row in scale], "^-", color=OLIVE, label="ILU solve")
    axes[0].set(xlabel="N", ylabel="seconds", title="Measured cost scaling")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[1].plot(n, [row["matrix_memory_mib"] for row in scale], "o-", color=BLUE, label="A")
    axes[1].plot(n, [row["ilu_factor_memory_mib"] for row in scale], "s-", color=ORANGE, label="ILU factors")
    axes[1].set(xlabel="N", ylabel="MiB", title="Explicit sparse storage")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "scale_cost_and_memory.png", dpi=180)
    plt.close(fig)
    chart_map.append({"plot": "plots/scale_cost_and_memory.png", "source": "discovery_systems.csv", "purpose": "N scaling of assembly, ILU, solve, and storage"})

    pde = [
        row
        for row in discovery_rows
        if row.get("family") in {"smooth_contrast", "localized_contrast", "anisotropy"}
        and "ilu_iterations_median" in row
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = [PURPLE if row["family"] == "anisotropy" else BLUE for row in pde]
    ax.bar(np.arange(len(pde)), [row["ilu_iterations_median"] for row in pde], color=colors)
    ax.axhline(20, color=ORANGE, linestyle="--", linewidth=1.3, label="20-iteration flag")
    ax.set_xticks(np.arange(len(pde)), [row["system"] for row in pde], rotation=35, ha="right")
    ax.set(ylabel="median ILU-GMRES iterations", title="PDE difficulty under unchanged strong ILU")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "pde_difficulty.png", dpi=180)
    plt.close(fig)
    chart_map.append({"plot": "plots/pde_difficulty.png", "source": "discovery_systems.csv", "purpose": "contrast and anisotropy iteration comparison"})

    if correction_curves:
        systems = list(dict.fromkeys(str(row["system"]) for row in correction_curves))
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        for system_name in systems:
            values = [row for row in correction_curves if row["system"] == system_name]
            q = [row["q"] for row in values]
            axes[0].semilogy(q, [row["normalized_singular_value"] for row in values], label=system_name)
            axes[1].plot(q, [row["captured_energy"] for row in values], label=system_name)
            axes[2].plot(q, [row["heldout_projection_error_median"] for row in values], label=system_name)
        axes[0].set(xlabel="q", ylabel=r"$\sigma_q/\sigma_1$", title="Singular decay")
        axes[1].set(xlabel="q", ylabel="captured energy", title="Training energy")
        axes[2].set(xlabel="q", ylabel="median held-out error", title="Held-out projection")
        axes[1].axhline(0.95, color=LIGHT_GRAY, linestyle="--")
        axes[2].axhline(0.25, color=LIGHT_GRAY, linestyle="--")
        axes[2].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / "correction_structure.png", dpi=180)
        plt.close(fig)
        chart_map.append({"plot": "plots/correction_structure.png", "source": "correction_structure_curves.csv", "purpose": "POD evidence and held-out generalization"})

    oracle_best = []
    for system_name in dict.fromkeys(str(row["system"]) for row in oracle_rows):
        candidates = [row for row in oracle_rows if row["system"] == system_name]
        base = next(row for row in candidates if row["basis_definition"] == "none")
        hybrids = [row for row in candidates if row["basis_definition"] != "none" and float(row["convergence_rate"]) == 1.0]
        if hybrids:
            best = min(hybrids, key=lambda row: (float(row["iterations_median"]), float(row["solve_seconds_median"])))
            oracle_best.append((system_name, base, best))
    if oracle_best:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        x = np.arange(len(oracle_best))
        width = 0.36
        axes[0].bar(x - width / 2, [item[1]["iterations_median"] for item in oracle_best], width, color=ORANGE, label="ILU")
        axes[0].bar(x + width / 2, [item[2]["iterations_median"] for item in oracle_best], width, color=BLUE, label="ILU+oracle")
        axes[1].bar(x - width / 2, [item[1]["solve_seconds_median"] for item in oracle_best], width, color=ORANGE, label="ILU")
        axes[1].bar(x + width / 2, [item[2]["solve_seconds_median"] for item in oracle_best], width, color=BLUE, label="ILU+oracle")
        labels = [item[0] for item in oracle_best]
        for ax, ylabel, title in zip(axes, ("median iterations", "median solve seconds"), ("Oracle iteration effect", "Oracle online time")):
            ax.set_xticks(x, labels, rotation=30, ha="right")
            ax.set(ylabel=ylabel, title=title)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / "oracle_benefit.png", dpi=180)
        plt.close(fig)
        chart_map.append({"plot": "plots/oracle_benefit.png", "source": "oracle_coarse_results.csv", "purpose": "strongest per-system oracle vs ILU"})

    if transfer.get("performed", False):
        with np.load(output / "transfer_distance_matrices.npz") as data:
            ratios = data["ratios"]
            distance = data["normalized_projector_distance"]
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        image = ax.imshow(distance, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(ratios)), [f"{value:g}" for value in ratios])
        ax.set_yticks(range(len(ratios)), [f"{value:g}" for value in ratios])
        ax.set(xlabel="anisotropy ratio", ylabel="anisotropy ratio", title="Ritz-space projector distance")
        fig.colorbar(image, ax=ax, label="normalized distance")
        fig.tight_layout()
        fig.savefig(plot_dir / "transfer_heatmap.png", dpi=180)
        plt.close(fig)
        chart_map.append({"plot": "plots/transfer_heatmap.png", "source": "transfer_distance_matrices.npz", "purpose": "between-system difficult-space variation"})

    fixed_cost = [
        row
        for row in cost_rows
        if row["case"] == "A_fixed_matrix" and row["system"] == "anisotropy_R100"
    ]
    if fixed_cost:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for method in dict.fromkeys(str(row["method"]) for row in fixed_cost):
            values = sorted([row for row in fixed_cost if row["method"] == method], key=lambda row: float(row["K"]))
            ax.plot([row["K"] for row in values], [row["fully_amortized_total_seconds"] for row in values], "o-", label=method)
        ax.set(xlabel="repeated solves K", ylabel="fully amortized seconds", title="Anisotropy R100 repeated-solve economics")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "repeated_solve_costs.png", dpi=180)
        plt.close(fig)
        chart_map.append({"plot": "plots/repeated_solve_costs.png", "source": "repeated_solve_costs.csv", "purpose": "full cost and break-even behavior"})

    if direct.get("performed", False) and (output / "direct_offline_metrics.csv").exists():
        with (output / "direct_offline_metrics.csv").open(newline="", encoding="utf-8") as stream:
            metrics = list(csv.DictReader(stream))
        if metrics:
            fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
            for ax, field, title in zip(
                axes,
                ("relative_error", "magnitude_ratio", "cosine_similarity"),
                ("Correction relative error", "Correction magnitude ratio", "Correction cosine"),
            ):
                values = np.asarray([float(row[field]) for row in metrics])
                ax.hist(values[np.isfinite(values)], bins=24, color=BLUE, alpha=0.85)
                ax.axvline(np.median(values), color=ORANGE, linestyle="--", label="median")
                ax.set(title=title, xlabel=field.replace("_", " "), ylabel="count")
            axes[0].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(plot_dir / "direct_validation_diagnostics.png", dpi=180)
            plt.close(fig)
            chart_map.append({"plot": "plots/direct_validation_diagnostics.png", "source": "direct_offline_metrics.csv", "purpose": "unsafe MLP held-out correction diagnostics before solver integration"})

    if direct.get("safe_for_solver_integration", False) and (output / "direct_residual_histories.csv").exists():
        with (output / "direct_residual_histories.csv").open(newline="", encoding="utf-8") as stream:
            histories = list(csv.DictReader(stream))
        if histories:
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for method in dict.fromkeys(row["method"] for row in histories):
                values = [row for row in histories if row["method"] == method and row["trajectory"] == "0"]
                ax.semilogy([float(row["iteration"]) for row in values], [float(row["relative_residual"]) for row in values], label=method)
            ax.set(xlabel="FGMRES iteration", ylabel="true relative residual", title="Held-out direct-complement trajectory")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(plot_dir / "direct_heldout_residuals.png", dpi=180)
            plt.close(fig)
            chart_map.append({"plot": "plots/direct_heldout_residuals.png", "source": "direct_residual_histories.csv", "purpose": "direct MLP solver-level comparison"})

    _write_csv(output / "chart_map.csv", chart_map)
    return chart_map


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "∞" if number > 0 else "—"
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output: Path,
    discovery_rows: Sequence[Mapping[str, Any]],
    correction_rows: Sequence[Mapping[str, Any]],
    oracle_selections: Sequence[Mapping[str, Any]],
    transfer: Mapping[str, Any],
    direct: Mapping[str, Any],
    break_even: Sequence[Mapping[str, Any]],
    classifications: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the answer-first Stage 9 technical report from saved evidence."""

    scale = sorted(
        [row for row in discovery_rows if row.get("family") == "scale"],
        key=lambda row: float(row["n_nodes"]),
    )
    pde = [
        row
        for row in discovery_rows
        if row.get("family") in {"smooth_contrast", "localized_contrast", "anisotropy"}
    ]
    geometry = [row for row in discovery_rows if row.get("family") == "node_geometry"]
    largest = scale[-1]
    hard = [
        row
        for row in discovery_rows
        if "ilu_iterations_median" in row
        and (
            not bool(row.get("ilu_converged", True))
            or float(row.get("ilu_iterations_median", 0.0)) > 20.0
        )
    ]
    best_oracle = max(
        oracle_selections,
        key=lambda row: float(row.get("iteration_reduction", 0.0)),
        default=None,
    )
    scale_table = _table(
        ("N", "nnz(A)", "A MiB", "assembly s", "ILU setup s", "factor MiB", "fill", "iterations", "solve s", "disc. error"),
        [
            (
                int(float(row["n_nodes"])),
                int(float(row["nnz"])),
                _fmt(row["matrix_memory_mib"]),
                _fmt(row["assembly_seconds"]),
                _fmt(row["ilu_setup_seconds"]),
                _fmt(row["ilu_factor_memory_mib"]),
                _fmt(row["ilu_factor_fill_ratio"]),
                _fmt(row["ilu_iterations_median"], 1),
                _fmt(row["ilu_solve_seconds_median"]),
                _fmt(row["relative_discretization_error"]),
            )
            for row in scale
        ],
    )
    pde_table = _table(
        ("system", "type", "contrast", "symmetry", "cond. proxy", "ILU it.", "setup s", "solve s", "converged", "disc. error"),
        [
            (
                row["system"],
                row["coefficient_kind"],
                _fmt(row["coefficient_contrast"], 0),
                _fmt(row["symmetry_measure"]),
                _fmt(row["preconditioned_condition_proxy"]),
                _fmt(row["ilu_iterations_median"], 1),
                _fmt(row["ilu_setup_seconds"]),
                _fmt(row["ilu_solve_seconds_median"]),
                _fmt(row["ilu_converged"]),
                _fmt(row["relative_discretization_error"]),
            )
            for row in pde
            if "ilu_iterations_median" in row
        ],
    )
    geometry_table = _table(
        ("system", "NN CV", "NN ratio", "local cond. median", "ILU it.", "status"),
        [
            (
                row["system"],
                _fmt(row.get("node_irregularity_cv")),
                _fmt(row.get("nearest_neighbor_ratio")),
                _fmt(row.get("local_condition_median")),
                _fmt(row.get("ilu_iterations_median"), 1),
                (
                    "valid"
                    if "ilu_iterations_median" in row
                    else f"invalid: {row.get('failure', 'unknown failure')}"
                ),
            )
            for row in geometry
        ],
    )
    correction_table = _table(
        ("system", "N", "q90", "eta90", "q95", "q95/N", "eta95", "q99", "eta99", "structured@95"),
        [
            (
                row["system"],
                int(float(row["n_nodes"])),
                int(float(row["q90"])),
                _fmt(row["eta_q90_median"]),
                int(float(row["q95"])),
                _fmt(row["q95_over_n"]),
                _fmt(row["eta_q95_median"]),
                int(float(row["q99"])),
                _fmt(row["eta_q99_median"]),
                _fmt(row["structured_at_95"]),
            )
            for row in correction_rows
        ],
    )
    oracle_table = _table(
        ("system", "best space", "q", "ILU it.", "oracle it.", "reduction", "oracle solve s", "reference+coarse setup s", "coarse MiB"),
        [
            (
                row["system"],
                row["basis_definition"],
                int(float(row["q"])),
                _fmt(float(row["iterations_median"]) / (1.0 - float(row["iteration_reduction"])) if float(row["iteration_reduction"]) < 1.0 else float("nan"), 1),
                _fmt(row["iterations_median"], 1),
                f"{100 * float(row['iteration_reduction']):.1f}%",
                _fmt(row["solve_seconds_median"]),
                _fmt(float(row["reference_setup_seconds"]) + float(row["coarse_setup_seconds"])),
                _fmt(row["coarse_memory_mib"]),
            )
            for row in oracle_selections
        ],
    )
    break_even_table = _table(
        ("system", "method", "break-even K", "online faster/solve"),
        [
            (
                row["system"],
                row["basis_definition"],
                _fmt(row["break_even_K_full_vs_ILU"], 1),
                _fmt(row["hybrid_online_faster_per_solve"]),
            )
            for row in break_even
        ],
    )
    classification_table = _table(
        ("system", "class", "decision rationale"),
        [(row["system"], row["category"], row["rationale"]) for row in classifications],
    )

    if transfer.get("performed", False):
        transfer_text = (
            f"For Ritz spaces of dimension q={int(transfer['q'])} across anisotropy "
            f"ratios 60–140, median normalized within-system projector distance was "
            f"{_fmt(transfer['within_projector_distance_median'])}, versus "
            f"{_fmt(transfer['between_projector_distance_median'])} between systems "
            f"(ratio {_fmt(transfer['within_to_between_median_ratio'])}). The Spearman "
            f"correlation between log-parameter distance and projector distance was "
            f"{_fmt(transfer['spearman_parameter_vs_projector'])} "
            f"(p={_fmt(transfer['spearman_pvalue'])}). Nearby reuse produced median "
            f"iteration reduction {_fmt(100 * transfer['nearby_reuse_iteration_reduction_median'], 1)}%."
        )
    else:
        transfer_text = f"Transferability was not run: {transfer.get('reason', 'gate failed')}"

    if direct.get("performed", False) and direct.get("safe_for_solver_integration", False):
        heldout = direct.get("heldout_summary", {})
        direct_text = (
            f"The unchanged Stage 8 one-hidden-layer MLP was retrained with normalized, "
            f"on-policy inputs on the fixed-N anisotropy family. Fully accumulated training "
            f"cost was {_fmt(direct['training_seconds_total'])} s. Held-out ratio-100 medians "
            f"were ILU { _fmt(heldout['classical_ILU']['iterations_median'], 1) } iterations / "
            f"{_fmt(heldout['classical_ILU']['solve_seconds_median'])} s, damped MLP "
            f"{_fmt(heldout['MLP_alpha0.1']['iterations_median'], 1)} iterations / "
            f"{_fmt(heldout['MLP_alpha0.1']['solve_seconds_median'])} s, and undamped MLP "
            f"{_fmt(heldout['MLP_alpha1']['iterations_median'], 1)} iterations / "
            f"{_fmt(heldout['MLP_alpha1']['solve_seconds_median'])} s. The robust direct gate "
            f"was {'passed' if direct.get('direct_robust_improvement', False) else 'not passed'}."
        )
    else:
        validation = direct.get("final_validation", {})
        if direct.get("performed", False):
            direct_text = (
                "The unchanged Stage 8 MLP was trained once, but failed the preregistered "
                f"validation magnitude gate before on-policy aggregation: median relative "
                f"correction error {_fmt(validation.get('validation_relative_error_median'))}, "
                f"magnitude ratio {_fmt(validation.get('validation_magnitude_ratio_median'))} "
                f"(p95 {_fmt(validation.get('validation_magnitude_ratio_p95'))}), and cosine "
                f"similarity {_fmt(validation.get('validation_cosine_median'))}. It was therefore "
                "not inserted into held-out FGMRES, and no learned iteration/time result is claimed."
            )
        else:
            direct_text = f"Direct learning was not run: {direct.get('reason', 'gate failed')}"

    subspace_prediction_ready = bool(
        best_oracle is not None
        and transfer.get("performed", False)
        and float(transfer.get("within_to_between_median_ratio", np.inf)) < 0.5
        and float(transfer.get("nearby_reuse_iteration_reduction_median", 0.0)) > 0.10
    )
    direct_ready = bool(direct.get("direct_robust_improvement", False))
    conclusion = (
        "both direct learning and difficult-space prediction"
        if direct_ready and subspace_prediction_ready
        else "direct learning"
        if direct_ready
        else "difficult-space prediction"
        if subspace_prediction_ready
        else "neither learned branch yet"
    )
    report = rf"""# Stage 9: hard-regime discovery and scalability

## Technical summary

The first genuinely nontrivial strong-ILU regime is **rotated tensor anisotropy**, not scalar coefficient contrast or scale alone. Up to N={int(float(largest['n_nodes'])):,}, the unchanged ILU remains compact ({_fmt(largest['ilu_factor_memory_mib'])} MiB), convergent ({_fmt(largest['ilu_iterations_median'], 1)} median iterations), and inexpensive ({_fmt(largest['ilu_setup_seconds'])} s setup). Smooth scalar contrast through 10^4 and a localized 10^3 inclusion also remain easy. At anisotropy ratio 100, the same ILU reaches the 300-iteration cap, while the independently solved manufactured problem has relative discretization error {_fmt(next(row['relative_discretization_error'] for row in hard if row['system']=='anisotropy_R100'))}.

The correction-POD conclusion does **not** transfer automatically: only anisotropy R100 passes the q95/N plus held-out-error criterion. System-specific Ritz oracle spaces accelerate four of five analyzed representatives; anisotropy R1000 remains stalled for every tested q. The largest benefit is {100 * float(best_oracle['iteration_reduction']):.1f}% for {best_oracle['system']} if a reference space is already available. This separates solver-relevant Ritz structure from correction reconstruction energy.

On present evidence, the project should continue with **{conclusion}**. The Ritz oracle motivates further classical target-stabilization work, but within-system variability and zero median nearby-reuse gain do not yet justify learned subspace prediction. No learned subspace network was trained.

## Visual findings

![Scale cost and memory](plots/scale_cost_and_memory.png)

![PDE difficulty](plots/pde_difficulty.png)

![Correction structure](plots/correction_structure.png)

![Oracle benefit](plots/oracle_benefit.png)

![Repeated-solve costs](plots/repeated_solve_costs.png)

## Scope, sources, and definitions

The [latest proposal](../../hybrid_iteration_rp.pdf) remains the mathematical source of truth, and the [Stage 8 report](../stage8_direct_onpolicy_and_subspace_stability/stage8_direct_onpolicy_and_subspace_stability_report.md) is the empirical starting point. No conflict was found. The unresolved action is unchanged:

\[
d(r)=A_{{\mathrm{{RBF}}}}^{{-1}}r-M_{{\mathrm{{RBF}}}}^{{-1}}r.
\]

Every reference action uses a sparse solve with iterative refinement; no inverse matrix is formed. The strong ILU is unchanged from Stages 7–8: drop tolerance 10^-3, fill-factor budget 5, COLAMD ordering, and zero diagonal-pivot threshold. Matrices are polynomially augmented PHS3 RBF-FD with stencil size 25. Timings are wall-clock observations on the environment saved in `software_environment.json`; storage is the exact byte count of sparse data/index arrays, excluding Python object overhead.

## 1. Scale: when do ILU setup and memory matter?

{scale_table}

At the largest tested N, assembly—not ILU—dominates one-solve total time. ILU factor storage grows from {_fmt(scale[0]['ilu_factor_memory_mib'])} to {_fmt(largest['ilu_factor_memory_mib'])} MiB and remains about {_fmt(largest['ilu_factor_fill_ratio'])} times nnz(A). No scale-only memory wall appears by N≈20k. ILU setup is comparable to an individual solve in some cases, so reuse matters, but the absolute cost is still subsecond.

## 2. PDE difficulty: contrast and anisotropy

{pde_table}

Smooth scalar contrast does not materially degrade ILU here. Rotated anisotropy does: ratio 10 is easy, while ratios 100 and 1000 stagnate at the cap and increase factor fill above 4. These are standard SPD tensors, not pathological sign-changing coefficients. The RBF-FD matrices are nonsymmetric because of local stencils and strong Dirichlet rows, so all oracle corrections use the implemented Petrov minimum-residual variant rather than silently applying symmetric Galerkin algebra.

## 3. Node-set difficulty

{geometry_table}

Moderate jitter and centre refinement are numerically valid and ILU-easy. The most extreme warped and centre-clustered maps fail local augmented RBF-FD solves; these failures are recorded rather than regularized or hidden. They are not valid hybrid regimes.

## 4. Unresolved-correction structure

{correction_table}

Training energy alone is misleading. For example, several systems reach 95% energy with q≤4, yet their held-out errors remain 0.48–0.84. Only anisotropy R100 provides convincing low-dimensional correction evidence under the stated held-out gate. The maximum refined reference residual is recorded per system in `correction_structure_summary.csv`; anisotropy remains the least accurate reference solve because of conditioning, which limits the precision of its snapshot claim.

## 5. Oracle difficult-space test

{oracle_table}

Ritz spaces win the per-system oracle selection in all analyzed cases. This does not imply they are cheap to reconstruct: their reference setup is included in the full cost model. The anisotropy-R100 oracle reduces the capped ILU solve to about {int(float(next(row['iterations_median'] for row in oracle_selections if row['system']=='anisotropy_R100')))} iterations and is the only case where the classical solve is genuinely hard rather than merely setup-sensitive.

## 6. Difficult-space reuse across nearby systems

{transfer_text}

![Transfer heatmap](plots/transfer_heatmap.png)

The comparison uses independently constructed per-system Ritz spaces and separately reports within-system sampling variability. Reuse is tested by applying a source system's basis to a different target system with the same node ordering; it is not a fixed-base-only transfer claim.

## 7. Direct learned complement on the gated hard regime

{direct_text}

The network architecture and optimization budget were not enlarged or tuned. The supervised target and homogeneous normalization remain exactly those of Stage 8. Solver-level convergence and wall time—not training loss—determine the direct-branch decision.

![Direct correction diagnostics](plots/direct_validation_diagnostics.png)

## 8. Repeated-solve economics

{break_even_table}

Case A reuses one factorization for many right-hand sides. Case B changes the matrix and charges assembly plus a new ILU setup every solve. K∈{{1,10,50,100,500}} totals are modeled from measured component medians and explicitly marked `modeled_not_rerun`; 500 physical solves were not repeated merely to reproduce linear accounting. Oracle full cost includes reference-space construction; online-only cost treats the reference basis as already available. Direct full cost includes all on-policy training rounds.

## 9. Regime classification

{classification_table}

The requested A–D taxonomy is not literally exhaustive for invalid discretizations. The two assembly failures are placed conservatively in B and explicitly labelled invalid; no correction or ML claim is made for them.

## 10. Answers to the research questions

1. **At what N does ILU setup or memory become significant?** Not by N≈20k in an absolute sense: setup remains {_fmt(largest['ilu_setup_seconds'])} s and factor storage {_fmt(largest['ilu_factor_memory_mib'])} MiB. Setup can exceed one solve, so it matters only in changing-matrix or low-K accounting.
2. **How does ILU behave under contrast, anisotropy, and irregularity?** Scalar contrast to 10^4 and moderate valid irregularity remain easy; rotated anisotropy ≥100 causes stagnation and higher fill. Extreme node maps invalidate local RBF-FD systems before ILU is meaningful.
3. **Which systems are genuinely nontrivial?** Anisotropy ratios 100 and 1000. The scale and scalar families are primarily amortization studies, not solver-hard regimes.
4. **Does d(r) remain low-dimensional?** Not generally. The held-out criterion passes for anisotropy R100 but fails for scale N{int(float(largest['n_nodes'])):,}, anisotropy R1000, scalar contrast 10^4, and centre refinement despite high training energy.
5. **Does the oracle become substantially more useful?** Yes for anisotropy R100: the best Ritz oracle reduces median iterations by {100 * float(next(row['iteration_reduction'] for row in oracle_selections if row['system']=='anisotropy_R100')):.1f}% and also lowers online solve time. Other representatives show iteration gains but start from easy ILU solves.
6. **Are spaces reusable?** {transfer_text}
7. **Does direct learning become more competitive?** {direct_text}
8. **What K is required?** The table above gives measured-component break-even counts. A finite K is reported only when the hybrid is faster per solve; otherwise no amortization can rescue it.
9. **Strongest future regime?** Fixed-node, smoothly parameterized rotated anisotropy around ratio 100: ILU is genuinely hard, the discretization is valid, a Ritz oracle is strong, and the family supports reuse tests.
10. **Continue with what?** **{conclusion}.** Continue classical Ritz-space stabilization and reuse experiments, but pause both learned branches. A learned subspace model remains gated on target stability, reuse, and full amortized cost; this round trains none.

## Method, robustness, and limitations

- Two manufactured right-hand sides are solved per discovery system; three are used in each oracle comparison. This is a research prototype, not a statistical performance benchmark.
- A 20-step Arnoldi singular-value ratio is reported only as a conditioning proxy, not as cond(A).
- Snapshot rank is capped by 24 training samples; small q/N therefore needs held-out projection error for interpretation.
- Anisotropic sparse-LU labels retain higher residuals than the other regimes even after refinement. Conclusions depending on fine correction differences should be revisited with a higher-precision reference solver.
- Timing variance, allocator overhead, and Python object storage are not included in sparse byte counts. Repeated K totals are algebraic extrapolations of measured medians.
- Oracle spaces use each target system's own data. They are upper bounds, not deployable learned methods.

## Next steps

1. Repeat the anisotropy-ratio family with more right-hand sides and a higher-accuracy reference solve.
2. Validate Ritz-space reuse over finer nearby parameter increments and multiple node seeds.
3. Only if the direct gate is positive, replicate it without architecture changes across several held-out anisotropy ratios.
4. Only if Ritz stability/reuse and amortized economics become positive, define—but do not yet enlarge—a graph-based subspace predictor.
"""
    report_path = output / "stage9_hard_regime_discovery_and_scalability_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_stage9(output: Path) -> Dict[str, Any]:
    """Execute Stage 9 in gated order and write all raw/report artifacts."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = Stage9Settings()
    _write_json(output / "experiment_config.json", asdict(settings))
    _write_json(
        output / "software_environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    )
    discovery, _, spec_map = run_discovery(output, settings)
    selected = _select_analysis_systems(discovery, spec_map)
    _write_json(
        output / "analysis_selection.json",
        {
            "selected": [spec.name for spec in selected],
            "rule": "largest flagged scale; hardest anisotropy plus second failed anisotropy; highest scalar contrast; most irregular valid geometry; maximum five",
        },
    )
    correction, correction_curves, _, contexts = run_correction_analysis(
        output, selected, settings
    )
    oracle_rows, _, oracle_selections = run_oracle_analysis(
        output, contexts, settings
    )
    transfer = run_transferability(output, oracle_selections, settings)
    direct = run_direct_if_gated(
        output, discovery, correction, oracle_selections, settings
    )
    cost_rows, break_even = build_cost_model(
        output, discovery, oracle_rows, oracle_selections, settings
    )
    _append_direct_costs(
        output, cost_rows, break_even, discovery, direct, settings
    )
    classifications = classify_regimes(
        output, discovery, correction, oracle_selections, direct
    )
    charts = make_plots(
        output,
        discovery,
        correction_curves,
        oracle_rows,
        cost_rows,
        transfer,
        direct,
    )
    report = write_report(
        output,
        discovery,
        correction,
        oracle_selections,
        transfer,
        direct,
        break_even,
        classifications,
    )
    evidence = {
        "report": str(report),
        "selected_analysis_systems": [spec.name for spec in selected],
        "nontrivial_solver_hard_systems": [
            row["system"]
            for row in discovery
            if "ilu_iterations_median" in row
            and (
                not bool(row.get("ilu_converged", True))
                or float(row.get("ilu_iterations_median", 0.0)) > 20.0
            )
        ],
        "structured_correction_systems": [
            row["system"] for row in correction if row["structured_at_95"]
        ],
        "useful_oracle_systems": [
            row["system"] for row in oracle_selections if row["oracle_useful"]
        ],
        "direct": direct,
        "transfer": transfer,
        "charts": charts,
    }
    _write_json(output / "evidence_summary.json", evidence)
    return evidence


def rewrite_stage9_report(output: Path) -> Path:
    """Regenerate only the report after narrative/formatting changes."""

    output = Path(output)
    discovery = _read_csv(output / "discovery_systems.csv")
    correction = _read_csv(output / "correction_structure_summary.csv")
    oracle = _read_csv(output / "oracle_selection.csv")
    break_even = _read_csv(output / "break_even_costs.csv")
    classifications = _read_csv(output / "regime_classification.csv")
    transfer = json.loads((output / "transferability_gate.json").read_text(encoding="utf-8"))
    direct = json.loads((output / "direct_gate.json").read_text(encoding="utf-8"))
    training = _read_csv(output / "direct_training_rounds.csv")
    if training:
        direct["final_validation"] = training[-1]
        _write_json(output / "direct_gate.json", direct)
    report = write_report(
        output,
        discovery,
        correction,
        oracle,
        transfer,
        direct,
        break_even,
        classifications,
    )
    evidence_path = output / "evidence_summary.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["direct"] = direct
        _write_json(evidence_path, evidence)
    return report
