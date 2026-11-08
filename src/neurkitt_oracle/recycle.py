"""Utilities for comparing recycle spaces of real nonnormal operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg as la


VectorMap = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RealInvariantSubspace:
    """A real orthonormal basis assembled from real or conjugate eigenmodes."""

    basis: np.ndarray
    selected_indices: tuple[int, ...]


def apply_to_columns(apply: VectorMap, basis: np.ndarray) -> np.ndarray:
    """Apply a vector map to every column, accepting vector-only callables."""

    array = np.asarray(basis)
    if array.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    try:
        mapped = np.asarray(apply(array))
        if mapped.shape == array.shape:
            return mapped
    except (TypeError, ValueError):
        pass
    return np.column_stack([np.asarray(apply(array[:, j])) for j in range(array.shape[1])])


def apply_real_operator_to_complex(
    apply: VectorMap,
    vector: np.ndarray,
) -> np.ndarray:
    """Extend a real-linear callable to complex vectors componentwise."""

    array = np.asarray(vector)
    if np.iscomplexobj(array):
        return np.asarray(apply(array.real)) + 1j * np.asarray(apply(array.imag))
    return np.asarray(apply(array))


def real_invariant_subspace(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    dimension: int,
    *,
    imaginary_tolerance: float = 1.0e-10,
    conjugate_tolerance: float = 1.0e-7,
) -> RealInvariantSubspace:
    """Convert ordered complex eigenpairs into a real invariant subspace.

    A real eigenvector contributes one direction.  A nonreal conjugate pair
    contributes its real and imaginary parts together, preserving real
    invariance.  The function refuses to split a pair at the requested
    dimension boundary.
    """

    values = np.asarray(eigenvalues)
    vectors = np.asarray(eigenvectors)
    if values.ndim != 1 or vectors.ndim != 2:
        raise ValueError("eigenvalues and eigenvectors must be one- and two-dimensional")
    if vectors.shape[1] != values.size:
        raise ValueError("one eigenvector column is required per eigenvalue")
    if not 1 <= dimension <= vectors.shape[0]:
        raise ValueError("dimension must be positive and no larger than the ambient space")

    columns: list[np.ndarray] = []
    selected: list[int] = []
    consumed: set[int] = set()
    for index, value in enumerate(values):
        if index in consumed:
            continue
        scale = max(1.0, abs(value))
        if abs(value.imag) <= imaginary_tolerance * scale:
            if len(columns) == dimension:
                break
            columns.append(vectors[:, index].real)
            selected.append(index)
            consumed.add(index)
        else:
            candidates = [
                other
                for other in range(values.size)
                if other != index and other not in consumed
            ]
            if not candidates:
                raise ValueError("a nonreal eigenvalue is missing its conjugate partner")
            partner = min(candidates, key=lambda other: abs(values[other] - value.conjugate()))
            pair_error = abs(values[partner] - value.conjugate()) / scale
            if pair_error > conjugate_tolerance:
                raise ValueError("a nonreal eigenvalue is missing an accurate conjugate partner")
            if len(columns) + 2 > dimension:
                raise ValueError("requested real dimension would split a conjugate pair")
            columns.extend((vectors[:, index].real, vectors[:, index].imag))
            selected.extend((index, partner))
            consumed.update((index, partner))
        if len(columns) == dimension:
            break

    if len(columns) != dimension:
        raise ValueError("not enough eigenpairs to construct the requested real subspace")
    basis, triangular = la.qr(
        np.column_stack(columns),
        mode="economic",
        check_finite=False,
    )
    rank_threshold = (
        100.0
        * np.finfo(float).eps
        * max(float(np.linalg.norm(triangular, ord=2)), 1.0)
    )
    if np.any(np.abs(np.diag(triangular)) <= rank_threshold):
        raise ValueError("selected real invariant directions are numerically rank deficient")
    return RealInvariantSubspace(basis=basis, selected_indices=tuple(selected))


def principal_angles_degrees(
    first_basis: np.ndarray,
    second_basis: np.ndarray,
) -> np.ndarray:
    """Return canonical angles in degrees, ordered largest to smallest."""

    first = np.asarray(first_basis, dtype=float)
    second = np.asarray(second_basis, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("bases must be two-dimensional with equal ambient dimension")
    return np.rad2deg(la.subspace_angles(first, second))


def invariant_subspace_defect(
    apply_operator: VectorMap,
    basis: np.ndarray,
) -> float:
    """Measure ``||(I-QQ^T)BQ||_F / ||BQ||_F`` for an orthonormal ``Q``."""

    orthonormal_basis = np.asarray(basis, dtype=float)
    image = apply_to_columns(apply_operator, orthonormal_basis)
    outside = image - orthonormal_basis @ (orthonormal_basis.T @ image)
    return float(
        np.linalg.norm(outside, ord="fro")
        / max(np.linalg.norm(image, ord="fro"), np.finfo(float).tiny)
    )
