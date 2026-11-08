"""Subspace geometry for global-RBF difficult-space transfer studies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import scipy.linalg as la


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    q, _ = la.qr(basis, mode="economic", check_finite=False)
    return q


@dataclass(frozen=True)
class SubspaceComparison:
    principal_angles_radians: np.ndarray
    projector_distance: float
    normalized_projector_distance: float

    @property
    def max_angle_degrees(self) -> float:
        return float(np.degrees(np.max(self.principal_angles_radians, initial=0.0)))

    @property
    def median_angle_degrees(self) -> float:
        return float(np.degrees(np.median(self.principal_angles_radians)))

    @property
    def rms_angle_degrees(self) -> float:
        return float(np.degrees(np.sqrt(np.mean(self.principal_angles_radians**2))))


def compare_subspaces(left: np.ndarray, right: np.ndarray) -> SubspaceComparison:
    """Return basis-invariant angles and chordal projector distances."""

    left = orthonormalize(left)
    right = orthonormalize(right)
    if left.shape != right.shape:
        raise ValueError("subspaces must have equal ambient and intrinsic dimensions")
    singular_values = la.svdvals(left.T @ right, check_finite=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    # ||P-Q||_F^2 = 2q - 2||Z_left^T Z_right||_F^2.
    distance_squared = max(0.0, 2.0 * left.shape[1] - 2.0 * float(np.sum(singular_values**2)))
    distance = float(np.sqrt(distance_squared))
    return SubspaceComparison(
        principal_angles_radians=angles,
        projector_distance=distance,
        normalized_projector_distance=distance / np.sqrt(2.0 * left.shape[1]),
    )


def smallest_right_singular_spaces(
    A: np.ndarray,
    q_values: Iterable[int],
    *,
    lapack_driver: str = "gesdd",
) -> tuple[dict[int, np.ndarray], np.ndarray, float]:
    """Compute nested smallest-right-singular subspaces with one full SVD."""

    q_values = sorted({int(q) for q in q_values})
    if not q_values or q_values[0] <= 0 or q_values[-1] > min(A.shape):
        raise ValueError("q values must lie between one and min(A.shape)")
    start = perf_counter()
    _, singular_values, vh = la.svd(
        A,
        full_matrices=False,
        check_finite=False,
        lapack_driver=lapack_driver,
    )
    seconds = perf_counter() - start
    spaces = {}
    for q in q_values:
        # SVD right vectors are already orthonormal; QR removes only tiny drift.
        spaces[q] = orthonormalize(vh[-q:, :].T)
    return spaces, singular_values, seconds


def grassmann_geodesic(left: np.ndarray, right: np.ndarray, t: float) -> np.ndarray:
    """Interpolate equal-dimensional subspaces along their shortest geodesic.

    The construction uses principal vector pairs.  It is invariant to rotations
    or sign changes of either input basis and does not interpolate basis entries.
    ``t=0`` returns ``range(left)`` and ``t=1`` returns ``range(right)``.
    """

    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    left = orthonormalize(left)
    right = orthonormalize(right)
    if left.shape != right.shape:
        raise ValueError("subspaces must have equal ambient and intrinsic dimensions")
    u, cosines, vh = la.svd(left.T @ right, full_matrices=False, check_finite=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    angles = np.arccos(cosines)
    x_principal = left @ u
    y_principal = right @ vh.T
    normal = np.zeros_like(x_principal)
    sine = np.sin(angles)
    resolved = sine > 100.0 * np.finfo(float).eps
    if np.any(resolved):
        normal[:, resolved] = (
            y_principal[:, resolved]
            - x_principal[:, resolved] * cosines[resolved]
        ) / sine[resolved]
    raw = x_principal * np.cos(t * angles) + normal * np.sin(t * angles)
    return orthonormalize(raw)
