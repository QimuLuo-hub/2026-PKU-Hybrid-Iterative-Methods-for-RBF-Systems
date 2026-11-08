"""Missing-inverse-action diagnostics and oracle coarse-space construction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import scipy.linalg as la

from .krylov import ArnoldiCycle, FGMRESResult
from .preconditioners import Preconditioner


@dataclass
class CorrectionData:
    train_residuals: np.ndarray
    test_residuals: np.ndarray
    train_corrections: np.ndarray
    test_corrections: np.ndarray
    singular_values: np.ndarray
    pod_basis: np.ndarray


@dataclass(frozen=True)
class GalerkinDeflation:
    """Standard nonsymmetric Galerkin deflation operators for a trial space Z.

    With ``E = Z.T A Z``, the projectors are

    ``P = I - A Z E^{-1} Z.T`` and
    ``Q = I - Z E^{-1} Z.T A``,

    and satisfy ``A Q = P A``.  Reduced actions use a pivoted LU solve; the
    inverse of E is never formed.
    """

    A_deflated: np.ndarray
    Z: np.ndarray
    AZ: np.ndarray
    reduced_factor: tuple[np.ndarray, np.ndarray]
    setup_seconds: float
    memory_bytes: int

    def deflate_rhs(self, rhs: np.ndarray) -> np.ndarray:
        coefficients = la.lu_solve(
            self.reduced_factor,
            self.Z.T @ rhs,
            check_finite=False,
        )
        return rhs - self.AZ @ coefficients

    def reconstruct(self, A: np.ndarray, rhs: np.ndarray, deflated_solution: np.ndarray) -> np.ndarray:
        coarse = self.Z @ la.lu_solve(
            self.reduced_factor,
            self.Z.T @ rhs,
            check_finite=False,
        )
        projected = deflated_solution - self.Z @ la.lu_solve(
            self.reduced_factor,
            self.Z.T @ (A @ deflated_solution),
            check_finite=False,
        )
        return coarse + projected


def normalized_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    keep = norms > 100.0 * np.finfo(float).eps
    if not np.any(keep):
        return np.empty((matrix.shape[0], 0))
    return matrix[:, keep] / norms[keep]


def correction_columns(
    lu_and_piv: tuple[np.ndarray, np.ndarray],
    preconditioner: Preconditioner,
    residuals: np.ndarray,
) -> np.ndarray:
    residuals = normalized_columns(residuals)
    exact = la.lu_solve(lu_and_piv, residuals, check_finite=False)
    classical = np.column_stack([preconditioner.apply(residuals[:, j]) for j in range(residuals.shape[1])])
    return exact - classical


def build_correction_data(
    A: np.ndarray,
    preconditioner: Preconditioner,
    *,
    train_residuals: np.ndarray,
    test_residuals: np.ndarray,
) -> CorrectionData:
    lu_and_piv = la.lu_factor(A, check_finite=False)
    train = normalized_columns(train_residuals)
    test = normalized_columns(test_residuals)
    d_train = correction_columns(lu_and_piv, preconditioner, train)
    d_test = correction_columns(lu_and_piv, preconditioner, test)
    U, singular_values, _ = la.svd(d_train, full_matrices=False, check_finite=False)
    return CorrectionData(train, test, d_train, d_test, singular_values, U)


def pod_metrics(data: CorrectionData, q_values: Iterable[int]) -> list[dict[str, float]]:
    total_energy = float(np.sum(data.singular_values**2))
    rows: list[dict[str, float]] = []
    for q_requested in q_values:
        q = min(int(q_requested), data.pod_basis.shape[1])
        Z = data.pod_basis[:, :q]
        if data.test_corrections.shape[1]:
            errors = []
            for j in range(data.test_corrections.shape[1]):
                d = data.test_corrections[:, j]
                errors.append(np.linalg.norm(d - Z @ (Z.T @ d)) / np.linalg.norm(d))
            median_error = float(np.median(errors))
            q90_error = float(np.quantile(errors, 0.9))
        else:
            median_error = q90_error = float("nan")
        captured = float(np.sum(data.singular_values[:q] ** 2) / total_energy) if total_energy else 1.0
        rows.append(
            {
                "q": q,
                "q_over_n": q / data.train_corrections.shape[0],
                "captured_energy": captured,
                "heldout_projection_error_median": median_error,
                "heldout_projection_error_q90": q90_error,
            }
        )
    return rows


def residual_history_space(
    residual_vectors: list[np.ndarray],
    preconditioner: Preconditioner,
    q: int,
) -> np.ndarray:
    if not residual_vectors:
        return np.empty((0, 0))
    updates = np.column_stack([preconditioner.apply(r) for r in residual_vectors])
    updates = normalized_columns(updates)
    U, _, _ = la.svd(updates, full_matrices=False, check_finite=False)
    return U[:, : min(q, U.shape[1])]


def _harmonic_ritz_cycle(result: FGMRESResult, q: int) -> ArnoldiCycle:
    """Choose the latest restart cycle that can support the requested rank."""

    cycles = result.arnoldi_cycles or []
    if cycles:
        sufficient = [cycle for cycle in cycles if cycle.hessenberg.shape[1] >= q]
        if sufficient:
            return sufficient[-1]
        return max(cycles, key=lambda cycle: cycle.hessenberg.shape[1])
    if result.hessenberg is None or result.preconditioned_vectors is None or result.arnoldi_vectors is None:
        raise ValueError("FGMRES result does not contain Arnoldi data")
    return ArnoldiCycle(
        result.arnoldi_vectors,
        result.preconditioned_vectors,
        result.hessenberg,
        max(0, result.iterations - result.hessenberg.shape[1]),
    )


def harmonic_ritz_space(result: FGMRESResult, q: int) -> np.ndarray:
    """Return a real coefficient-space harmonic Ritz space nearest zero.

    For right-preconditioned FGMRES, the Arnoldi operator is ``A M^{-1}``.
    Harmonic Ritz vectors live in its residual-space Krylov basis; multiplying
    their coordinates by the stored preconditioned Arnoldi vectors maps them
    to coefficient-space correction directions.  Complex-conjugate pairs are
    represented by their real and imaginary parts rather than discarding the
    imaginary component.
    """

    if q < 1:
        return np.empty((len(result.x), 0))
    cycle = _harmonic_ritz_cycle(result, q)
    Hbar = cycle.hessenberg
    m = Hbar.shape[1]
    if m == 0:
        return np.empty((len(result.x), 0))
    Hm = Hbar[:m, :]
    tail = Hbar[m, m - 1] if Hbar.shape[0] > m else 0.0
    K = Hm.T @ Hm
    K[-1, -1] += tail**2
    G = Hm.T
    eigenvalues, eigenvectors = la.eig(K, G, check_finite=False)
    finite = np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[finite]
    eigenvectors = eigenvectors[:, finite]
    order = np.argsort(np.abs(eigenvalues))
    columns: list[np.ndarray] = []
    used: set[int] = set()
    complex_tolerance = 100.0 * np.finfo(float).eps
    for index in order:
        index = int(index)
        if index in used or len(columns) >= q:
            continue
        vector = eigenvectors[:, index]
        imaginary = np.linalg.norm(vector.imag)
        scale = max(np.linalg.norm(vector.real), 1.0)
        if imaginary <= complex_tolerance * scale:
            columns.append(vector.real)
            used.add(index)
            continue
        if q - len(columns) < 2:
            continue
        partner_candidates = [
            int(candidate)
            for candidate in order
            if int(candidate) not in used and int(candidate) != index
        ]
        if partner_candidates:
            partner = min(
                partner_candidates,
                key=lambda candidate: abs(eigenvalues[candidate] - np.conjugate(eigenvalues[index])),
            )
            used.add(partner)
        used.add(index)
        columns.extend([vector.real, vector.imag])
    if not columns:
        return np.empty((len(result.x), 0))
    raw = cycle.preconditioned_vectors @ np.column_stack(columns[:q])
    U, singular_values, _ = la.svd(raw, full_matrices=False, check_finite=False)
    if singular_values.size == 0:
        return np.empty((len(result.x), 0))
    tolerance = max(raw.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    return U[:, : min(q, rank)]


def smallest_right_singular_space(A: np.ndarray, q: int) -> np.ndarray:
    _, _, Vh = la.svd(A, full_matrices=False, check_finite=False)
    # Smallest-first ordering makes prefixes mathematically nested: if a q_max
    # basis is computed once, ``basis[:, :q]`` is the span of the q smallest
    # right singular vectors rather than a middle portion of the q_max space.
    take = min(q, len(Vh))
    raw = Vh[-1 : -take - 1 : -1, :].T
    Z, _ = la.qr(raw, mode="economic", check_finite=False)
    return Z


def oracle_coarse_preconditioner(
    A: np.ndarray,
    base: Preconditioner,
    Z: np.ndarray,
    *,
    name: str,
    composition: str = "additive",
) -> Preconditioner:
    """Combine a classical inverse action and a Petrov-Galerkin correction.

    ``W=A Z`` makes the reduced solve a minimum-residual Petrov-Galerkin
    correction for the nonsymmetric Kansa matrix.  ``additive`` applies the
    requested ``Z(W^T A Z)^{-1}W^T r`` directly.  ``residual_complement``
    applies it to ``r-A M^{-1}r`` and costs an extra dense matvec.  The reduced
    matrix is factorized and solved; no explicit inverse is formed.
    """

    start = perf_counter()
    Z, _ = la.qr(np.asarray(Z), mode="economic", check_finite=False)
    AZ = A @ Z
    W = AZ
    reduced = W.T @ AZ
    factor = la.cho_factor(reduced, lower=False, check_finite=False)

    if composition not in {"additive", "residual_complement"}:
        raise ValueError(f"unknown coarse composition: {composition}")

    def apply(r: np.ndarray) -> np.ndarray:
        base_update = base.apply(r)
        coarse_input = r if composition == "additive" else r - A @ base_update
        coarse_rhs = W.T @ coarse_input
        coarse_coeff = la.cho_solve(factor, coarse_rhs, check_finite=False)
        return base_update + Z @ coarse_coeff

    memory = base.memory_bytes + Z.nbytes + W.nbytes + factor[0].nbytes
    return Preconditioner(
        name,
        apply,
        base.setup_seconds + perf_counter() - start,
        int(memory),
        f"{base.name} plus {composition} Petrov-Galerkin correction, q={Z.shape[1]}",
    )


def additive_galerkin_coarse_preconditioner(
    A: np.ndarray,
    base: Preconditioner,
    Z: np.ndarray,
    *,
    name: str,
) -> Preconditioner:
    """Apply the exact additive correction requested for target selection.

    For an orthonormalized trial basis Z, the returned action is

    M^{-1} r + Z (Z.T A Z)^{-1} Z.T r.

    The generally nonsymmetric reduced matrix is factorized with pivoted LU
    and solved on every application; no reduced inverse is formed.  This
    function is intentionally separate from oracle_coarse_preconditioner,
    whose default Petrov test basis is A Z.
    """

    start = perf_counter()
    matrix = np.asarray(A, dtype=float)
    basis = np.asarray(Z, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("additive Galerkin correction requires a square matrix")
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[0]:
        raise ValueError("coarse basis has incompatible dimensions")
    if basis.shape[1] == 0:
        raise ValueError("coarse basis must contain at least one vector")
    basis, _ = la.qr(basis, mode="economic", check_finite=False)
    reduced = basis.T @ matrix @ basis
    factor = la.lu_factor(reduced, check_finite=False)

    def apply(r: np.ndarray) -> np.ndarray:
        coarse_rhs = basis.T @ r
        coarse_coefficients = la.lu_solve(factor, coarse_rhs, check_finite=False)
        return base.apply(r) + basis @ coarse_coefficients

    memory = base.memory_bytes + basis.nbytes + factor[0].nbytes + factor[1].nbytes
    return Preconditioner(
        name,
        apply,
        base.setup_seconds + perf_counter() - start,
        int(memory),
        f"{base.name} plus additive Galerkin correction, q={basis.shape[1]}",
    )


def residual_minimizing_coarse_preconditioner(
    A: np.ndarray,
    base: Preconditioner,
    Z: np.ndarray,
    *,
    name: str,
) -> Preconditioner:
    """Apply ``M^{-1}r + Z argmin_y ||r-A Z y||_2``.

    The small least-squares problem is solved from an SVD of ``A Z``.  This is
    deliberately not a normal-equations implementation.
    """

    start = perf_counter()
    matrix = np.asarray(A, dtype=float)
    basis = np.asarray(Z, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("residual-minimizing correction requires a square matrix")
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[0] or basis.shape[1] == 0:
        raise ValueError("coarse basis has incompatible dimensions")
    basis, _ = la.qr(basis, mode="economic", check_finite=False)
    AZ = matrix @ basis
    left, singular_values, right_t = la.svd(
        AZ,
        full_matrices=False,
        check_finite=False,
    )
    tolerance = max(AZ.shape) * np.finfo(float).eps * singular_values[0]
    if np.count_nonzero(singular_values > tolerance) != basis.shape[1]:
        raise la.LinAlgError("A Z is numerically rank deficient")

    def apply(r: np.ndarray) -> np.ndarray:
        coefficients = right_t.T @ ((left.T @ r) / singular_values)
        return base.apply(r) + basis @ coefficients

    memory = (
        base.memory_bytes
        + basis.nbytes
        + AZ.nbytes
        + left.nbytes
        + singular_values.nbytes
        + right_t.nbytes
    )
    return Preconditioner(
        name,
        apply,
        base.setup_seconds + perf_counter() - start,
        int(memory),
        f"{base.name} plus additive minimum-residual coarse correction, q={basis.shape[1]}",
    )


def galerkin_deflation(A: np.ndarray, Z: np.ndarray) -> GalerkinDeflation:
    """Build ``P A`` for standard nonsymmetric Galerkin deflation.

    ``P = I-A Z E^{-1}Z.T`` and ``Q = I-Z E^{-1}Z.T A`` with
    ``E=Z.T A Z``.  The returned dense matrix is ``P A``.  All reduced
    operations use pivoted LU solves, never an explicit inverse.
    """

    start = perf_counter()
    matrix = np.asarray(A, dtype=float)
    basis = np.asarray(Z, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Galerkin deflation requires a square matrix")
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[0] or basis.shape[1] == 0:
        raise ValueError("coarse basis has incompatible dimensions")
    basis, _ = la.qr(basis, mode="economic", check_finite=False)
    AZ = matrix @ basis
    reduced = basis.T @ AZ
    factor = la.lu_factor(reduced, check_finite=False)
    projected_rows = la.lu_solve(factor, basis.T @ matrix, check_finite=False)
    deflated = matrix - AZ @ projected_rows
    memory = (
        deflated.nbytes
        + basis.nbytes
        + AZ.nbytes
        + factor[0].nbytes
        + factor[1].nbytes
    )
    return GalerkinDeflation(
        deflated,
        basis,
        AZ,
        factor,
        perf_counter() - start,
        int(memory),
    )
