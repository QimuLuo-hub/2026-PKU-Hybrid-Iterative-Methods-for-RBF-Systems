"""System specifications and reusable data generation for the Stage 6 study."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from ..config import (
    DiscretizationConfig,
    ILUConfig,
    RBFConfig,
    SolverConfig,
)
from ..pde import (
    ExponentialCoefficient,
    ManufacturedSolution,
)
from ..preconditioners import InverseAction, ilu_preconditioner
from ..rbffd import RBFSystem, assemble_system
from ..snapshots import (
    SnapshotSet,
    generate_gmres_residuals,
    generate_synthetic_residuals,
)


@dataclass(frozen=True)
class SystemSpec:
    """One controlled RBF-FD system in a one-factor-at-a-time family."""

    name: str
    family: str
    n_nodes: int = 900
    stencil_size: int = 25
    node_distribution: str = "jittered_grid"
    jitter_fraction: float = 0.15
    warp_strength: float = 0.0
    beta: float = 0.0
    coefficient_frequency: int = 1
    seed: int = 7

    def discretization_config(self) -> DiscretizationConfig:
        return DiscretizationConfig(
            n_nodes=self.n_nodes,
            stencil_size=self.stencil_size,
            polynomial_degree=2,
            node_distribution=self.node_distribution,
            jitter_fraction=self.jitter_fraction,
            warp_strength=self.warp_strength,
            seed=self.seed,
            rbf=RBFConfig(kernel="phs", order=3, shape_parameter=1.0),
            compute_local_condition_numbers=False,
        )

    def coefficient(self):
        if self.beta == 0.0:
            return None
        return ExponentialCoefficient(
            beta=self.beta,
            frequency=self.coefficient_frequency,
        )

    @property
    def coefficient_contrast(self) -> float:
        return float(np.exp(2.0 * abs(self.beta)))


def controlled_system_families() -> List[SystemSpec]:
    """Return the required resolution, geometry, stencil, and coefficient families."""

    resolution = [
        SystemSpec(
            name=f"A_N{n_nodes}",
            family="A_resolution",
            n_nodes=n_nodes,
        )
        for n_nodes in (400, 900, 1600, 2500)
    ]
    geometry = [
        SystemSpec(
            name=f"B_warp{int(round(100 * warp)):02d}",
            family="B_irregularity",
            node_distribution="warped_grid",
            jitter_fraction=0.0,
            warp_strength=warp,
        )
        for warp in (0.0, 0.60, 0.90, 0.95)
    ]
    stencil = [
        SystemSpec(
            name=f"C_s{stencil_size}",
            family="C_stencil",
            stencil_size=stencil_size,
        )
        for stencil_size in (13, 25, 37)
    ]
    coefficients = [
        SystemSpec(
            name=f"D_beta{int(round(100 * beta)):03d}",
            family="D_variable_coefficient",
            beta=beta,
        )
        for beta in (0.0, 0.55, 1.10, 1.70)
    ]
    return resolution + geometry + stencil + coefficients


def selected_analysis_specs() -> List[SystemSpec]:
    """Return every one-factor system, including each family's base control."""

    return controlled_system_families()


def budget_specs() -> List[SystemSpec]:
    """Return one challenging representative from each controlled family."""

    wanted = {"A_N2500", "B_warp95", "C_s13", "D_beta170"}
    return [
        spec for spec in controlled_system_families() if spec.name in wanted
    ]


def build_system(spec: SystemSpec) -> Tuple[RBFSystem, float]:
    """Assemble one system and return its measured setup time."""

    start = perf_counter()
    system = assemble_system(
        spec.discretization_config(),
        coefficient=spec.coefficient(),
    )
    return system, perf_counter() - start


def manufactured_problem(
    system: RBFSystem,
    solution: Optional[ManufacturedSolution] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return continuous manufactured right-hand side and exact nodal values."""

    manufactured = ManufacturedSolution() if solution is None else solution
    if system.coefficient is None:
        rhs = manufactured.right_hand_side(
            system.nodes, system.boundary
        )
    else:
        rhs = manufactured.variable_coefficient_right_hand_side(
            system.nodes,
            system.boundary,
            system.coefficient,
        )
    return rhs, manufactured.values(system.nodes)


def system_metadata(
    spec: SystemSpec,
    system: RBFSystem,
    assembly_seconds: float,
) -> Dict[str, object]:
    """Flatten system controls and measured matrix/geometry diagnostics."""

    return {
        "system": spec.name,
        "family": spec.family,
        "n_nodes": spec.n_nodes,
        "stencil_size": spec.stencil_size,
        "node_distribution": spec.node_distribution,
        "jitter_fraction": spec.jitter_fraction,
        "warp_strength": spec.warp_strength,
        "node_irregularity": system.diagnostics[
            "nearest_neighbor_cv"
        ],
        "nearest_neighbor_ratio": system.diagnostics[
            "nearest_neighbor_ratio"
        ],
        "coefficient_beta": spec.beta,
        "coefficient_frequency": spec.coefficient_frequency,
        "coefficient_contrast": spec.coefficient_contrast,
        "nnz": int(system.matrix.nnz),
        "symmetry_measure": system.diagnostics["symmetry_measure"],
        "assembly_seconds": assembly_seconds,
    }


def make_snapshot_pair(
    system: RBFSystem,
    preconditioner: InverseAction,
    solver_config: SolverConfig,
    n_train_each: int,
    n_test_each: int,
    seed: int,
) -> Tuple[SnapshotSet, SnapshotSet, Dict[str, float]]:
    r"""Generate train/test d(r)=A^-1 r-M^-1 r with one accurate LU factor."""

    start = perf_counter()
    synthetic_train = generate_synthetic_residuals(
        system.matrix,
        system.nodes,
        system.boundary,
        n_train_each,
        seed + 11,
    )
    gmres_train = generate_gmres_residuals(
        system.matrix,
        system.nodes,
        n_train_each,
        seed + 17,
        solver_config,
        iterations_per_trajectory=10,
    )
    synthetic_test = generate_synthetic_residuals(
        system.matrix,
        system.nodes,
        system.boundary,
        n_test_each,
        seed + 101,
    )
    gmres_test = generate_gmres_residuals(
        system.matrix,
        system.nodes,
        n_test_each,
        seed + 107,
        solver_config,
        iterations_per_trajectory=10,
    )
    residual_generation_seconds = perf_counter() - start

    factor_start = perf_counter()
    factor = spla.splu(system.matrix.tocsc())
    factor_seconds = perf_counter() - factor_start

    def evaluate(
        blocks: Sequence[Tuple[str, np.ndarray]],
    ) -> SnapshotSet:
        residuals = []
        corrections = []
        labels = []
        exact_residuals = []
        for label, block in blocks:
            for column in range(block.shape[1]):
                residual = block[:, column]
                exact = np.asarray(factor.solve(residual), dtype=float)
                correction = exact - preconditioner.apply(residual)
                residuals.append(residual.copy())
                corrections.append(correction)
                labels.append(label)
                exact_residuals.append(
                    float(
                        np.linalg.norm(system.matrix @ exact - residual)
                        / max(np.linalg.norm(residual), 1.0e-300)
                    )
                )
        return SnapshotSet(
            np.column_stack(residuals),
            np.column_stack(corrections),
            np.asarray(labels),
            np.asarray(exact_residuals),
        )

    correction_start = perf_counter()
    train = evaluate(
        (("synthetic", synthetic_train), ("gmres", gmres_train))
    )
    test = evaluate(
        (("synthetic", synthetic_test), ("gmres", gmres_test))
    )
    correction_seconds = perf_counter() - correction_start
    diagnostics = {
        "residual_generation_seconds": residual_generation_seconds,
        "reference_factor_seconds": factor_seconds,
        "correction_evaluation_seconds": correction_seconds,
        "snapshot_total_seconds": (
            residual_generation_seconds
            + factor_seconds
            + correction_seconds
        ),
        "max_reference_relative_residual": float(
            max(
                np.max(train.exact_solve_relative_residuals),
                np.max(test.exact_solve_relative_residuals),
            )
        ),
    }
    return train, test, diagnostics


def stack_ml_samples(
    residual_blocks: Iterable[np.ndarray],
    correction_blocks: Iterable[np.ndarray],
    feature_rows: Iterable[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert column-oriented snapshot blocks into row-oriented ML arrays."""

    residual_list = []
    correction_list = []
    feature_list = []
    for residuals, corrections, features in zip(
        residual_blocks, correction_blocks, feature_rows
    ):
        if residuals.shape != corrections.shape:
            raise ValueError("residual and correction blocks must align")
        residual_list.append(residuals.T)
        correction_list.append(corrections.T)
        feature_list.append(
            np.repeat(
                np.asarray(features, dtype=float)[None, :],
                residuals.shape[1],
                axis=0,
            )
        )
    return (
        np.vstack(residual_list),
        np.vstack(correction_list),
        np.vstack(feature_list),
    )
