"""Utilities for an offline global-RBF difficult-subspace dataset.

The learned object is a subspace, so basis gauges are aligned only to make
the stored coefficient matrices reproducible.  All quality diagnostics are
computed from orthogonal projectors and are therefore basis invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import scipy.linalg as la


@dataclass(frozen=True)
class SharedBasisFit:
    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    retained_dimension: int
    energy_target: float


def orthonormalize_basis(basis: np.ndarray) -> np.ndarray:
    """Return an economic QR basis after validating its numerical rank."""

    matrix = np.asarray(basis, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("basis must be a nonempty matrix")
    singular_values = la.svdvals(matrix, check_finite=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    if np.count_nonzero(singular_values > tolerance) != matrix.shape[1]:
        raise ValueError("basis is numerically rank deficient")
    orthonormal, _ = la.qr(matrix, mode="economic", check_finite=False)
    return orthonormal


def align_basis_to_reference(basis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Choose the orthogonal Procrustes gauge nearest a reference basis.

    The returned matrix spans exactly the same subspace as ``basis``.  This
    removes arbitrary signs and within-subspace rotations from the stored
    coefficient target without changing projector-based diagnostics.
    """

    aligned = orthonormalize_basis(basis)
    anchor = orthonormalize_basis(reference)
    if aligned.shape != anchor.shape:
        raise ValueError("basis and reference must have the same dimensions")
    left, _, right_t = la.svd(aligned.T @ anchor, full_matrices=False, check_finite=False)
    return aligned @ (left @ right_t)


def fit_shared_basis(
    training_bases: list[np.ndarray],
    *,
    energy_target: float = 0.999,
) -> SharedBasisFit:
    """Fit ``W`` by SVD of concatenated training bases.

    Energy is the squared Frobenius energy of the concatenated orthonormal
    bases.  The retained dimension is the smallest prefix meeting the target.
    """

    if not 0.0 < energy_target <= 1.0:
        raise ValueError("energy_target must lie in (0, 1]")
    if not training_bases:
        raise ValueError("at least one training basis is required")
    normalized = [orthonormalize_basis(basis) for basis in training_bases]
    shape = normalized[0].shape
    if any(basis.shape != shape for basis in normalized):
        raise ValueError("all training bases must have the same dimensions")
    concatenated = np.concatenate(normalized, axis=1)
    left, singular_values, _ = la.svd(
        concatenated,
        full_matrices=False,
        check_finite=False,
    )
    squared = singular_values**2
    cumulative = np.cumsum(squared) / np.sum(squared)
    retained = int(np.searchsorted(cumulative, energy_target, side="left") + 1)
    return SharedBasisFit(
        basis=left[:, :retained],
        singular_values=singular_values,
        cumulative_energy=cumulative,
        retained_dimension=retained,
        energy_target=float(energy_target),
    )


def coefficient_representation(shared_basis: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return the least-squares coefficients ``C = W.T @ Z``."""

    W = np.asarray(shared_basis, dtype=float)
    Z = np.asarray(basis, dtype=float)
    if W.ndim != 2 or Z.ndim != 2 or W.shape[1] == 0 or Z.shape[1] == 0:
        raise ValueError("shared and sample bases must be nonempty matrices")
    if W.shape[0] != Z.shape[0]:
        raise ValueError("shared and sample bases must use the same ambient dimension")
    w_defect = np.linalg.norm(W.T @ W - np.eye(W.shape[1]), ord="fro")
    z_defect = np.linalg.norm(Z.T @ Z - np.eye(Z.shape[1]), ord="fro")
    if w_defect > 1.0e-8 or z_defect > 1.0e-8:
        raise ValueError("C = W.T Z requires stored orthonormal W and Z")
    return W.T @ Z


def reconstructed_basis(shared_basis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Orthonormalize ``W C`` for basis-invariant projector diagnostics."""

    W = np.asarray(shared_basis, dtype=float)
    C = np.asarray(coefficients, dtype=float)
    if W.ndim != 2 or W.shape[1] == 0 or C.ndim != 2 or C.shape[0] != W.shape[1]:
        raise ValueError("coefficient matrix has incompatible dimensions")
    defect = np.linalg.norm(W.T @ W - np.eye(W.shape[1]), ord="fro")
    if defect > 1.0e-8:
        raise ValueError("shared basis must be orthonormal")
    return orthonormalize_basis(W @ C)


def projector_reconstruction_error(
    basis: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[float, float]:
    """Return absolute and normalized Frobenius projector error."""

    Z = orthonormalize_basis(basis)
    Z_hat = orthonormalize_basis(reconstructed)
    if Z.shape != Z_hat.shape:
        raise ValueError("basis and reconstruction must have the same dimensions")
    error = float(np.linalg.norm(Z @ Z.T - Z_hat @ Z_hat.T, ord="fro"))
    normalized = error / np.sqrt(2.0 * Z.shape[1])
    return error, float(normalized)


def validate_parameter_splits(
    split_parameters: Mapping[str, np.ndarray],
    *,
    require_marginal_disjoint: bool = True,
) -> None:
    """Validate pair-level and optional marginal parameter disjointness."""

    required = {"train", "validation", "test"}
    if set(split_parameters) != required:
        raise ValueError(f"splits must be exactly {sorted(required)}")
    pairs: dict[str, set[tuple[float, float]]] = {}
    marginals: dict[str, tuple[set[float], set[float]]] = {}
    for split, values in split_parameters.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            raise ValueError(f"{split} parameters must be a nonempty (n, 2) array")
        pair_set = {tuple(row) for row in array.tolist()}
        if len(pair_set) != len(array):
            raise ValueError(f"{split} contains duplicate parameter pairs")
        pairs[split] = pair_set
        marginals[split] = (set(array[:, 0]), set(array[:, 1]))
    ordered = ("train", "validation", "test")
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if pairs[left] & pairs[right]:
                raise ValueError(f"{left} and {right} overlap in parameter pairs")
            if require_marginal_disjoint:
                for axis, name in enumerate(("epsilon", "beta")):
                    if marginals[left][axis] & marginals[right][axis]:
                        raise ValueError(f"{left} and {right} overlap in {name} values")
