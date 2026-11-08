"""Solver-aware samples of the exact missing inverse action.

The vectors collected here are the Arnoldi basis vectors passed to the right
preconditioner by :func:`lcp_rbffd.krylov.restarted_gmres`.  They are not true
residual snapshots.  For every such vector ``v`` the supervised target is
computed literally as ``A^{-1} v - M^{-1} v`` using an accurate sparse solve;
no inverse matrix is ever formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from .config import SolverConfig
from .krylov import KrylovResult, restarted_gmres
from .pde import random_smooth_solution
from .preconditioners import InverseAction
from .rbffd import RBFSystem


@dataclass
class SolverAwareSamples:
    """Row-oriented Arnoldi inputs and normalized missing-action targets."""

    vectors: np.ndarray
    vector_norms: np.ndarray
    exact_actions: np.ndarray
    classical_actions: np.ndarray
    corrections: np.ndarray
    normalized_corrections: np.ndarray
    system_features: np.ndarray
    system_labels: np.ndarray
    trajectory_labels: np.ndarray
    iteration_indices: np.ndarray
    exact_solve_relative_residuals: np.ndarray

    @property
    def normalized_vectors(self) -> np.ndarray:
        """Return exactly ``v / ||v||_2`` for every collected input."""

        return self.vectors / self.vector_norms[:, None]


def collect_preconditioner_inputs(
    system: RBFSystem,
    preconditioner: InverseAction,
    solver_config: SolverConfig,
    trajectory_seeds: Sequence[int],
    system_label: str,
    system_features: Sequence[float],
    trajectory_prefix: str,
    max_mode: int = 6,
) -> Tuple[SolverAwareSamples, List[Dict[str, object]], float]:
    r"""Collect actual preconditioner inputs and evaluate ``A^-1 v-M^-1 v``.

    Each seed defines one complete FGMRES trajectory.  The discrete right-hand
    side is ``A u`` for an independently generated smooth nodal solution, so
    the trajectory has a known exact discrete solution without conflating the
    data-collection diagnostic with RBF-FD truncation error.
    """

    if not trajectory_seeds:
        raise ValueError("at least one trajectory seed is required")
    factor_start = perf_counter()
    try:
        exact_factor = spla.splu(system.matrix.tocsc())
    except Exception as exc:
        raise RuntimeError(f"reference sparse factorization failed: {exc}") from exc
    factor_seconds = perf_counter() - factor_start

    vectors: List[np.ndarray] = []
    norms: List[float] = []
    corrections: List[np.ndarray] = []
    exact_actions: List[np.ndarray] = []
    classical_actions: List[np.ndarray] = []
    features: List[np.ndarray] = []
    system_labels: List[str] = []
    trajectory_labels: List[str] = []
    iteration_indices: List[int] = []
    exact_residuals: List[float] = []
    trajectory_rows: List[Dict[str, object]] = []
    feature_array = np.asarray(system_features, dtype=float)

    for trajectory_index, seed in enumerate(trajectory_seeds):
        trajectory_label = f"{trajectory_prefix}_{trajectory_index:03d}_seed{seed}"
        exact = random_smooth_solution(seed, max_mode=max_mode).values(system.nodes)
        exact[system.boundary] = 0.0
        rhs = np.asarray(system.matrix @ exact, dtype=float)
        collected: List[Tuple[int, np.ndarray]] = []

        def capture(iteration: int, vector: np.ndarray) -> None:
            collected.append((iteration, np.asarray(vector, dtype=float).copy()))

        result = restarted_gmres(
            system.matrix,
            rhs,
            solver_config,
            preconditioner,
            preconditioner_input_callback=capture,
        )
        trajectory_rows.append(
            {
                "system": system_label,
                "trajectory": trajectory_label,
                "rhs_seed": int(seed),
                "iterations": int(result.iterations),
                "inputs_collected": int(len(collected)),
                "converged": bool(result.converged),
                "final_relative_residual": float(result.relative_residuals[-1]),
                "solve_seconds": float(result.solve_seconds),
                "failure": result.failure,
            }
        )
        if not collected:
            raise RuntimeError(f"trajectory {trajectory_label} produced no Arnoldi inputs")

        for iteration, vector in collected:
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                raise RuntimeError("FGMRES supplied a zero or non-finite preconditioner input")
            exact_action = np.asarray(exact_factor.solve(vector), dtype=float)
            approximate_action = preconditioner.apply(vector)
            correction = exact_action - approximate_action
            reference_residual = float(
                np.linalg.norm(system.matrix @ exact_action - vector) / norm
            )
            vectors.append(vector)
            norms.append(norm)
            exact_actions.append(exact_action)
            classical_actions.append(approximate_action)
            corrections.append(correction)
            features.append(feature_array.copy())
            system_labels.append(system_label)
            trajectory_labels.append(trajectory_label)
            iteration_indices.append(iteration)
            exact_residuals.append(reference_residual)

    vector_array = np.row_stack(vectors)
    norm_array = np.asarray(norms)
    correction_array = np.row_stack(corrections)
    samples = SolverAwareSamples(
        vectors=vector_array,
        vector_norms=norm_array,
        exact_actions=np.row_stack(exact_actions),
        classical_actions=np.row_stack(classical_actions),
        corrections=correction_array,
        normalized_corrections=correction_array / norm_array[:, None],
        system_features=np.row_stack(features),
        system_labels=np.asarray(system_labels),
        trajectory_labels=np.asarray(trajectory_labels),
        iteration_indices=np.asarray(iteration_indices, dtype=int),
        exact_solve_relative_residuals=np.asarray(exact_residuals),
    )
    return samples, trajectory_rows, factor_seconds


def concatenate_solver_aware_samples(
    blocks: Sequence[SolverAwareSamples],
) -> SolverAwareSamples:
    """Concatenate systems while preserving whole-trajectory provenance."""

    if not blocks:
        raise ValueError("at least one sample block is required")
    return SolverAwareSamples(
        vectors=np.vstack([block.vectors for block in blocks]),
        vector_norms=np.concatenate([block.vector_norms for block in blocks]),
        exact_actions=np.vstack([block.exact_actions for block in blocks]),
        classical_actions=np.vstack([block.classical_actions for block in blocks]),
        corrections=np.vstack([block.corrections for block in blocks]),
        normalized_corrections=np.vstack(
            [block.normalized_corrections for block in blocks]
        ),
        system_features=np.vstack([block.system_features for block in blocks]),
        system_labels=np.concatenate([block.system_labels for block in blocks]),
        trajectory_labels=np.concatenate(
            [block.trajectory_labels for block in blocks]
        ),
        iteration_indices=np.concatenate(
            [block.iteration_indices for block in blocks]
        ),
        exact_solve_relative_residuals=np.concatenate(
            [block.exact_solve_relative_residuals for block in blocks]
        ),
    )


def label_preconditioner_inputs(
    system: RBFSystem,
    preconditioner: InverseAction,
    vectors: np.ndarray,
    system_label: str,
    system_features: Sequence[float],
    trajectory_label: str,
    iteration_indices: Optional[Sequence[int]] = None,
    exact_factor: Optional[object] = None,
) -> Tuple[SolverAwareSamples, float]:
    r"""Label arbitrary solver inputs with ``A^-1 v-M^-1 v``.

    This utility is used for on-policy aggregation after a learned action has
    changed the FGMRES trajectory.  ``exact_factor`` may be a reusable sparse
    LU factor; otherwise one is constructed.  The factor is only applied to
    vectors and the inverse matrix is never formed.
    """

    vector_array = np.asarray(vectors, dtype=float)
    if vector_array.ndim == 1:
        vector_array = vector_array[None, :]
    if vector_array.ndim != 2 or vector_array.shape[1] != system.matrix.shape[0]:
        raise ValueError("solver input block has incompatible dimensions")
    if vector_array.shape[0] == 0:
        raise ValueError("at least one solver input is required")
    norms = np.linalg.norm(vector_array, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).eps):
        raise ValueError("solver inputs must be finite and nonzero")

    factor_seconds = 0.0
    factor = exact_factor
    if factor is None:
        start = perf_counter()
        try:
            factor = spla.splu(system.matrix.tocsc())
        except Exception as exc:
            raise RuntimeError(f"reference sparse factorization failed: {exc}") from exc
        factor_seconds = perf_counter() - start

    exact_actions = np.row_stack(
        [np.asarray(factor.solve(vector), dtype=float) for vector in vector_array]
    )
    classical_actions = np.row_stack(
        [preconditioner.apply(vector) for vector in vector_array]
    )
    corrections = exact_actions - classical_actions
    relative_residuals = np.asarray(
        [
            np.linalg.norm(system.matrix @ exact - vector) / norm
            for exact, vector, norm in zip(exact_actions, vector_array, norms)
        ],
        dtype=float,
    )
    feature = np.asarray(system_features, dtype=float)
    features = np.repeat(feature[None, :], vector_array.shape[0], axis=0)
    if iteration_indices is None:
        indices = np.arange(1, vector_array.shape[0] + 1, dtype=int)
    else:
        indices = np.asarray(iteration_indices, dtype=int)
        if indices.shape != (vector_array.shape[0],):
            raise ValueError("iteration_indices must match the solver input count")

    samples = SolverAwareSamples(
        vectors=vector_array.copy(),
        vector_norms=norms,
        exact_actions=exact_actions,
        classical_actions=classical_actions,
        corrections=corrections,
        normalized_corrections=corrections / norms[:, None],
        system_features=features,
        system_labels=np.repeat(system_label, vector_array.shape[0]),
        trajectory_labels=np.repeat(trajectory_label, vector_array.shape[0]),
        iteration_indices=indices,
        exact_solve_relative_residuals=relative_residuals,
    )
    return samples, factor_seconds
