"""Generation of unresolved-correction snapshots for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import SolverConfig
from .krylov import restarted_gmres
from .pde import random_smooth_solution
from .preconditioners import InverseAction


@dataclass
class SnapshotSet:
    r"""Residuals and exact missing actions ``A^{-1}r-M^{-1}r`` as columns."""

    residuals: np.ndarray
    corrections: np.ndarray
    labels: np.ndarray
    exact_solve_relative_residuals: np.ndarray


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("cannot normalize a zero residual snapshot")
    return matrix / norms


def generate_synthetic_residuals(
    matrix: sp.spmatrix,
    nodes: np.ndarray,
    boundary: np.ndarray,
    n_samples: int,
    seed: int,
    max_mode: int = 6,
) -> np.ndarray:
    """Generate both random residual and smooth synthetic-error mechanisms.

    Even-indexed samples are isotropic random residuals. Odd-indexed samples
    start with a smooth sine-series error ``e`` and use ``r=Ae``.  This avoids
    baking a low-rank conclusion into the data generator.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    residuals: List[np.ndarray] = []
    for sample in range(n_samples):
        if sample % 2 == 0:
            residual = rng.normal(size=matrix.shape[0])
        else:
            exact_error = random_smooth_solution(seed + 1009 * sample, max_mode).values(nodes)
            # Include boundary-compatible error, but do not assume a small mode count in POD.
            exact_error[boundary] = 0.0
            residual = np.asarray(matrix @ exact_error, dtype=float)
        residuals.append(residual)
    return _normalize_columns(np.column_stack(residuals))


def generate_gmres_residuals(
    matrix: sp.spmatrix,
    nodes: np.ndarray,
    n_samples: int,
    seed: int,
    solver_config: SolverConfig,
    max_mode: int = 6,
    iterations_per_trajectory: int = 12,
) -> np.ndarray:
    """Collect true residual vectors produced inside actual unpreconditioned GMRES runs."""

    if n_samples < 1 or iterations_per_trajectory < 1:
        raise ValueError("sample and trajectory counts must be positive")
    residuals: List[np.ndarray] = []
    trajectory = 0
    while len(residuals) < n_samples:
        exact = random_smooth_solution(seed + 7919 * trajectory, max_mode).values(nodes)
        rhs = np.asarray(matrix @ exact, dtype=float)
        local_config = SolverConfig(
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
            restart=max(iterations_per_trajectory, 2),
            max_iterations=iterations_per_trajectory,
            breakdown_tolerance=solver_config.breakdown_tolerance,
        )

        def collect(_: int, __: np.ndarray, residual: np.ndarray) -> None:
            if len(residuals) < n_samples and np.linalg.norm(residual) > np.finfo(float).eps:
                residuals.append(residual)

        restarted_gmres(matrix, rhs, local_config, callback=collect)
        trajectory += 1
        if trajectory > n_samples + 10 and not residuals:
            raise RuntimeError("GMRES did not produce residual snapshots")
    return _normalize_columns(np.column_stack(residuals[:n_samples]))


def compute_missing_actions(
    matrix: sp.spmatrix,
    preconditioner: InverseAction,
    residual_blocks: Sequence[Tuple[str, np.ndarray]],
) -> SnapshotSet:
    r"""Compute ``d(r)=A^{-1}r-M^{-1}r`` without forming ``A^{-1}``.

    One accurate sparse LU factorization is reused only for offline reference
    solves.  Every returned exact-solve residual is recorded.
    """

    try:
        exact_factor = spla.splu(matrix.tocsc())
    except Exception as exc:
        raise RuntimeError(f"reference sparse factorization failed: {exc}") from exc
    all_residuals: List[np.ndarray] = []
    corrections: List[np.ndarray] = []
    labels: List[str] = []
    solve_residuals: List[float] = []
    for label, block in residual_blocks:
        if block.shape[0] != matrix.shape[0]:
            raise ValueError("residual block has incompatible dimension")
        for column in range(block.shape[1]):
            residual = block[:, column]
            exact = np.asarray(exact_factor.solve(residual), dtype=float)
            approximate = preconditioner.apply(residual)
            correction = exact - approximate
            relative = float(
                np.linalg.norm(matrix @ exact - residual) / max(np.linalg.norm(residual), 1.0e-300)
            )
            all_residuals.append(residual.copy())
            corrections.append(correction)
            labels.append(label)
            solve_residuals.append(relative)
    return SnapshotSet(
        np.column_stack(all_residuals),
        np.column_stack(corrections),
        np.asarray(labels),
        np.asarray(solve_residuals),
    )

