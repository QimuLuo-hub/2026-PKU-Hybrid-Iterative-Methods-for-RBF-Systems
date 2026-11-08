"""POD/SVD diagnostics for unresolved-correction data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PODResult:
    """Thin SVD and captured-energy curve for a snapshot matrix."""

    basis: np.ndarray
    singular_values: np.ndarray
    right_vectors_transpose: np.ndarray
    captured_energy: np.ndarray


def compute_pod(snapshot_matrix: np.ndarray) -> PODResult:
    r"""Compute ``D=U Sigma V^T`` and ``E_q=sum_{i<=q} sigma_i^2/sum_i sigma_i^2``."""

    if snapshot_matrix.ndim != 2 or snapshot_matrix.shape[1] == 0:
        raise ValueError("snapshot matrix must be a nonempty two-dimensional array")
    basis, singular_values, right_vectors_transpose = np.linalg.svd(
        snapshot_matrix, full_matrices=False
    )
    squared = singular_values**2
    total = float(np.sum(squared))
    captured = np.cumsum(squared) / total if total > 0.0 else np.zeros_like(squared)
    return PODResult(basis, singular_values, right_vectors_transpose, captured)


def relative_projection_errors(basis: np.ndarray, test_snapshots: np.ndarray) -> np.ndarray:
    r"""Return each ``||(I-ZZ^T)d||_2/||d||_2`` for orthonormal ``Z``."""

    if basis.ndim != 2 or test_snapshots.ndim != 2:
        raise ValueError("basis and snapshots must be matrices")
    if basis.shape[0] != test_snapshots.shape[0]:
        raise ValueError("basis and snapshots have incompatible row dimensions")
    norms = np.linalg.norm(test_snapshots, axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("projection error is undefined for a zero snapshot")
    residual = test_snapshots - basis @ (basis.T @ test_snapshots)
    return np.linalg.norm(residual, axis=0) / norms


def projection_error_curve(
    full_basis: np.ndarray,
    test_snapshots: np.ndarray,
    q_max: int,
) -> np.ndarray:
    """Return one row of held-out projection errors for every ``q=1,...,q_max``."""

    if q_max < 1 or q_max > full_basis.shape[1]:
        raise ValueError("q_max is outside the available POD basis")
    return np.vstack(
        [relative_projection_errors(full_basis[:, :q], test_snapshots) for q in range(1, q_max + 1)]
    )

