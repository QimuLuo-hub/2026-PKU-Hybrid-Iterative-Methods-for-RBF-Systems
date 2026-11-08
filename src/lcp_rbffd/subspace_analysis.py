"""Krylov reference spaces and basis-invariant cross-system diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .preconditioners import InverseAction


@dataclass
class RitzModeResult:
    """Nested real solution-space Ritz directions from ``A M^{-1}``."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    mode_labels: Tuple[str, ...]
    diagnostics: Dict[str, object]


def ritz_atomic_blocks(
    eigenvalues: np.ndarray,
    relative_tolerance: float = 1.0e-8,
) -> Tuple[Tuple[int, int], ...]:
    """Return consecutive indivisible real Ritz blocks.

    ``nested_krylov_ritz_modes`` repeats a non-real Ritz value for the real and
    imaginary directions spanning its real invariant plane.  Those directions
    must never be separated by clustering or prefix completion.  Consecutive
    repeated values are therefore collapsed into one half-open index span;
    ordinary real Ritz values remain one-dimensional atoms.
    """

    values = np.asarray(eigenvalues, dtype=complex)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    blocks: List[Tuple[int, int]] = []
    start = 0
    while start < values.size:
        stop = start + 1
        scale = 1.0 + abs(values[start])
        while (
            stop < values.size
            and abs(values[stop] - values[start]) <= relative_tolerance * scale
        ):
            stop += 1
        blocks.append((start, stop))
        start = stop
    return tuple(blocks)


def nested_krylov_ritz_modes(
    matrix,
    preconditioner: InverseAction,
    max_modes: int,
    arnoldi_dimension: int,
    seed: int,
    initial_vector: Optional[np.ndarray] = None,
) -> RitzModeResult:
    r"""Return ordered real Ritz directions for the right-preconditioned operator.

    Arnoldi is applied to ``A M^{-1}``.  Projected eigenvectors are ordered by
    increasing Ritz-value magnitude, split into real/imaginary directions when
    needed, mapped to solution space with ``M^{-1}``, and orthonormalized in
    that order.  Consequently every prefix is a nested candidate coarse space.

    ``initial_vector`` permits construction from a physical residual/RHS; a
    seeded Gaussian direction is used otherwise.  The returned eigenvalue is
    repeated for the real and imaginary directions of a complex Ritz vector.
    """

    n = matrix.shape[0]
    if not 0 < max_modes < n:
        raise ValueError("max_modes must satisfy 0 < max_modes < n")
    dimension = min(max(arnoldi_dimension, max_modes + 4), n - 1)
    if initial_vector is None:
        rng = np.random.default_rng(seed)
        start_vector = rng.normal(size=n)
    else:
        start_vector = np.asarray(initial_vector, dtype=float).copy()
        if start_vector.shape != (n,):
            raise ValueError("initial_vector has incompatible dimension")
    start_norm = float(np.linalg.norm(start_vector))
    if not np.isfinite(start_norm) or start_norm <= np.finfo(float).eps:
        raise ValueError("initial_vector must be finite and nonzero")

    start = perf_counter()
    arnoldi_basis = np.zeros((n, dimension + 1), dtype=float)
    hessenberg = np.zeros((dimension + 1, dimension), dtype=float)
    arnoldi_basis[:, 0] = start_vector / start_norm
    actual_dimension = dimension
    preconditioner.reset_timing()
    for column in range(dimension):
        work = np.asarray(
            matrix @ preconditioner.apply(arnoldi_basis[:, column]), dtype=float
        )
        for row in range(column + 1):
            coefficient = float(np.dot(arnoldi_basis[:, row], work))
            hessenberg[row, column] += coefficient
            work -= coefficient * arnoldi_basis[:, row]
        for row in range(column + 1):
            correction = float(np.dot(arnoldi_basis[:, row], work))
            hessenberg[row, column] += correction
            work -= correction * arnoldi_basis[:, row]
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        if next_norm <= 1.0e-13:
            actual_dimension = column + 1
            break
        arnoldi_basis[:, column + 1] = work / next_norm

    projected = hessenberg[:actual_dimension, :actual_dimension]
    eigenvalues, eigenvectors = la.eig(projected)
    order = np.argsort(np.abs(eigenvalues))
    accepted = []
    selected_values = []
    labels = []
    for eigen_index in order:
        projected_vector = (
            arnoldi_basis[:, :actual_dimension] @ eigenvectors[:, eigen_index]
        )
        parts = [("real", np.real(projected_vector))]
        if np.linalg.norm(np.imag(projected_vector)) > 1.0e-10:
            parts.append(("imag", np.imag(projected_vector)))
        for component_name, component in parts:
            if np.linalg.norm(component) <= 1.0e-10:
                continue
            mapped = np.asarray(preconditioner.apply(component), dtype=float)
            for _ in range(2):
                for previous in accepted:
                    mapped -= previous * np.dot(previous, mapped)
            norm = float(np.linalg.norm(mapped))
            if norm <= 1.0e-10:
                continue
            accepted.append(mapped / norm)
            selected_values.append(eigenvalues[eigen_index])
            labels.append(f"eig{int(eigen_index)}_{component_name}")
            if len(accepted) == max_modes:
                break
        if len(accepted) == max_modes:
            break

    if len(accepted) < max_modes:
        raise RuntimeError(
            f"Arnoldi produced only {len(accepted)} independent Ritz directions; "
            f"{max_modes} requested"
        )
    basis = np.column_stack(accepted)
    orthogonality_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(max_modes))
    )
    result = RitzModeResult(
        basis=basis,
        eigenvalues=np.asarray(selected_values, dtype=complex),
        mode_labels=tuple(labels),
        diagnostics={
            "setup_seconds": perf_counter() - start,
            "arnoldi_dimension": actual_dimension,
            "basis_orthogonality_error": orthogonality_error,
            "preconditioner_apply_seconds": preconditioner.apply_seconds,
            "preconditioner_applications": preconditioner.apply_count,
            "selected_ritz_magnitudes": [
                float(abs(value)) for value in selected_values
            ],
        },
    )
    if orthogonality_error > 1.0e-8:
        raise RuntimeError(
            f"Ritz basis lost orthogonality (error={orthogonality_error:.3e})"
        )
    return result


def energy_dimension(captured_energy: np.ndarray, threshold: float) -> int:
    """Return the smallest one-based q whose cumulative energy reaches threshold."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    indices = np.flatnonzero(np.asarray(captured_energy) >= threshold)
    if indices.size == 0:
        return 0
    return int(indices[0] + 1)


def krylov_ritz_basis(
    matrix,
    preconditioner: InverseAction,
    q: int,
    arnoldi_dimension: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    r"""Build an approximate difficult space from Arnoldi Ritz information.

    Arnoldi is run on the right-preconditioned operator A M^{-1}. Eigenvectors
    of its projected Hessenberg matrix with smallest eigenvalue magnitude are
    mapped to solution space through M^{-1}. Complex Ritz vectors are expanded
    into real and imaginary parts before a real orthonormal basis is formed.
    """

    n = matrix.shape[0]
    if not 0 < q < n:
        raise ValueError("q must satisfy 0 < q < n")
    dimension = min(max(arnoldi_dimension, q + 2), n - 1)
    start = perf_counter()
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=n)
    initial /= np.linalg.norm(initial)
    basis = np.zeros((n, dimension + 1), dtype=float)
    hessenberg = np.zeros((dimension + 1, dimension), dtype=float)
    basis[:, 0] = initial
    actual_dimension = dimension
    preconditioner.reset_timing()

    for column in range(dimension):
        work = np.asarray(
            matrix @ preconditioner.apply(basis[:, column]), dtype=float
        )
        for row in range(column + 1):
            coefficient = float(np.dot(basis[:, row], work))
            hessenberg[row, column] += coefficient
            work -= coefficient * basis[:, row]
        for row in range(column + 1):
            correction = float(np.dot(basis[:, row], work))
            hessenberg[row, column] += correction
            work -= correction * basis[:, row]
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        if next_norm <= 1.0e-13:
            actual_dimension = column + 1
            break
        basis[:, column + 1] = work / next_norm

    projected = hessenberg[:actual_dimension, :actual_dimension]
    eigenvalues, eigenvectors = la.eig(projected)
    order = np.argsort(np.abs(eigenvalues))
    candidates = []
    for index in order:
        vector = basis[:, :actual_dimension] @ eigenvectors[:, index]
        for component in (np.real(vector), np.imag(vector)):
            if np.linalg.norm(component) <= 1.0e-10:
                continue
            mapped = preconditioner.apply(component)
            for accepted in candidates:
                mapped -= accepted * np.dot(accepted, mapped)
            norm = float(np.linalg.norm(mapped))
            if norm > 1.0e-10:
                candidates.append(mapped / norm)
            if len(candidates) == q:
                break
        if len(candidates) == q:
            break

    if len(candidates) < q:
        for column in range(actual_dimension):
            mapped = preconditioner.apply(basis[:, column])
            for accepted in candidates:
                mapped -= accepted * np.dot(accepted, mapped)
            norm = float(np.linalg.norm(mapped))
            if norm > 1.0e-10:
                candidates.append(mapped / norm)
            if len(candidates) == q:
                break
    if len(candidates) < q:
        raise RuntimeError("Arnoldi process did not yield enough independent vectors")

    result = np.column_stack(candidates[:q])
    result, _ = np.linalg.qr(result, mode="reduced")
    diagnostics: Dict[str, object] = {
        "setup_seconds": perf_counter() - start,
        "arnoldi_dimension": actual_dimension,
        "selected_ritz_magnitudes": np.abs(eigenvalues[order[:q]]).tolist(),
        "basis_orthogonality_error": float(
            np.linalg.norm(result.T @ result - np.eye(q))
        ),
    }
    return result, diagnostics


def interpolate_basis_to_probe(
    nodes: np.ndarray,
    basis: np.ndarray,
    probe_nodes: np.ndarray,
) -> np.ndarray:
    """Interpolate nodal modes to common probes and re-orthonormalize them."""

    nodes = np.asarray(nodes, dtype=float)
    basis = np.asarray(basis, dtype=float)
    probe_nodes = np.asarray(probe_nodes, dtype=float)
    if basis.shape[0] != nodes.shape[0]:
        raise ValueError("basis row count must match source nodes")
    columns = []
    for index in range(basis.shape[1]):
        linear = LinearNDInterpolator(nodes, basis[:, index])
        values = np.asarray(linear(probe_nodes), dtype=float)
        missing = ~np.isfinite(values)
        if np.any(missing):
            nearest = NearestNDInterpolator(nodes, basis[:, index])
            values[missing] = nearest(probe_nodes[missing])
        columns.append(values)
    interpolated = np.column_stack(columns)
    orthonormal, triangular = np.linalg.qr(interpolated, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= 1.0e-12:
        raise RuntimeError("interpolation made the comparison basis rank deficient")
    return orthonormal


def subspace_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> Dict[str, object]:
    """Compute projector distance and principal-angle summaries without dense projectors.

    For equal-rank orthonormal bases ``Q1`` and ``Q2``, the Frobenius projector
    distance obeys ``||P1-P2||_F = sqrt(2)||(I-P1)Q2||_F``.  Computing the
    orthogonal residual avoids both an ``n x n`` projector and catastrophic
    cancellation for nearly identical spaces.  Inputs are reduced to
    orthonormal bases when roundoff or interpolation has perturbed them.
    """

    if first.shape != second.shape:
        raise ValueError("subspace bases must have identical shapes")
    q = first.shape[1]
    identity = np.eye(q)
    if np.linalg.norm(first.T @ first - identity) > 1.0e-10:
        first, _ = la.qr(first, mode="economic")
    if np.linalg.norm(second.T @ second - identity) > 1.0e-10:
        second, _ = la.qr(second, mode="economic")
    overlap = first.T @ second
    # The algebraically equivalent trace formula
    # ``2q - 2||Q1.T Q2||_F**2`` loses all relative accuracy when the two
    # spaces nearly coincide.  The orthogonal residual has the same norm
    # identity for equal-rank orthonormal bases and remains accurate at zero.
    orthogonal_residual = second - first @ overlap
    distance = float(np.sqrt(2.0) * np.linalg.norm(orthogonal_residual, ord="fro"))
    angles = la.subspace_angles(first, second)
    degrees = np.degrees(angles)
    return {
        "projector_distance": distance,
        "projector_distance_normalized": float(
            distance / np.sqrt(2.0 * first.shape[1])
        ),
        "principal_angles_degrees": degrees,
        "principal_angle_mean_degrees": float(np.mean(degrees)),
        "principal_angle_max_degrees": float(np.max(degrees)),
    }


def generalized_subspace_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> Dict[str, object]:
    r"""Compare possibly unequal-rank subspaces without dense projectors.

    For orthonormal bases ``Q1`` and ``Q2`` with ranks ``p`` and ``q``,
    ``||P1-P2||_F^2 = p+q-2||Q1^T Q2||_F^2``.  Division by ``sqrt(p+q)``
    gives a distance in ``[0, 1]`` and retains a penalty when one subspace is
    strictly contained in a larger one.  Principal angles describe only the
    shared ``min(p, q)`` directions, so both containment fractions are also
    returned explicitly.
    """

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("subspace bases must be matrices with equal row counts")
    if first.shape[1] == 0 or second.shape[1] == 0:
        raise ValueError("subspace ranks must be positive")
    first, _ = la.qr(first, mode="economic")
    second, _ = la.qr(second, mode="economic")
    first_rank = first.shape[1]
    second_rank = second.shape[1]
    overlap = first.T @ second
    shared_energy = float(np.linalg.norm(overlap, ord="fro") ** 2)
    distance_squared = max(
        0.0,
        float(first_rank + second_rank - 2.0 * shared_energy),
    )
    angles = np.degrees(la.subspace_angles(first, second))
    return {
        "projector_distance": float(np.sqrt(distance_squared)),
        "projector_distance_normalized": float(
            np.sqrt(distance_squared) / np.sqrt(first_rank + second_rank)
        ),
        "principal_angles_degrees": angles,
        "principal_angle_mean_degrees": float(np.mean(angles)),
        "principal_angle_max_degrees": float(np.max(angles)),
        "dimension_first": int(first_rank),
        "dimension_second": int(second_rank),
        "dimension_equal": bool(first_rank == second_rank),
        "shared_energy": shared_energy,
        "containment_first": float(shared_energy / first_rank),
        "containment_second": float(shared_energy / second_rank),
    }


def standardized_parameter_distances(
    parameter_rows: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return pairwise Euclidean distances after per-parameter standardization."""

    parameters = np.asarray(parameter_rows, dtype=float)
    scale = np.std(parameters, axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    standardized = (parameters - np.mean(parameters, axis=0)) / scale
    differences = standardized[:, None, :] - standardized[None, :, :]
    return np.linalg.norm(differences, axis=2)
