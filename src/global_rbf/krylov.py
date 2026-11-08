"""Right-preconditioned flexible GMRES for dense global collocation systems."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.linalg as la


VectorMap = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ArnoldiCycle:
    """Unrotated Arnoldi data from one restarted FGMRES cycle."""

    arnoldi_vectors: np.ndarray
    preconditioned_vectors: np.ndarray
    hessenberg: np.ndarray
    first_iteration: int


@dataclass
class FGMRESResult:
    x: np.ndarray
    converged: bool
    iterations: int
    solve_seconds: float
    relative_residual: float
    residual_history: list[float]
    residual_vectors: list[np.ndarray]
    residual_vector_iterations: list[int]
    arnoldi_vectors: np.ndarray | None = None
    preconditioned_vectors: np.ndarray | None = None
    hessenberg: np.ndarray | None = None
    arnoldi_cycles: list[ArnoldiCycle] | None = None


def _givens(a: float, b: float) -> tuple[float, float]:
    if b == 0.0:
        return 1.0, 0.0
    if a == 0.0:
        return 0.0, 1.0
    scale = abs(a) + abs(b)
    norm = scale * np.sqrt((a / scale) ** 2 + (b / scale) ** 2)
    return a / norm, b / norm


def fgmres(
    A: np.ndarray,
    b: np.ndarray,
    apply_preconditioner: VectorMap | None = None,
    *,
    x0: np.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    max_iterations: int = 300,
    capture_residual_vectors: bool = False,
    capture_arnoldi: bool = False,
) -> FGMRESResult:
    """Solve ``A x=b`` with right-preconditioned restarted FGMRES.

    The stopping rule uses the true double-precision residual after every
    restart or candidate convergence, while the inner loop uses the standard
    Givens-updated GMRES residual estimate.
    """

    n = len(b)
    apply = apply_preconditioner or (lambda v: v.copy())
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float, copy=True)
    bnorm = np.linalg.norm(b)
    Ainf = np.linalg.norm(A, ord=np.inf)
    threshold = max(atol, rtol * bnorm)
    history: list[float] = []
    residual_vectors: list[np.ndarray] = []
    residual_vector_iterations: list[int] = []
    arnoldi_cycles: list[ArnoldiCycle] | None = [] if capture_arnoldi else None
    total_iterations = 0
    last_V = last_Z = last_H = None
    start = perf_counter()

    while total_iterations < max_iterations:
        r = b - A @ x
        beta = np.linalg.norm(r)
        history.append(float(beta / bnorm))
        if capture_residual_vectors:
            residual_vectors.append(r.copy())
            residual_vector_iterations.append(total_iterations)
        if beta <= threshold:
            break

        cycle_first_iteration = total_iterations
        m = min(restart, max_iterations - total_iterations, n)
        V = np.zeros((n, m + 1), dtype=float)
        Z = np.zeros((n, m), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        H_arnoldi = np.zeros((m + 1, m), dtype=float)
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        V[:, 0] = r / beta
        g[0] = beta
        used = m

        for j in range(m):
            Z[:, j] = apply(V[:, j])
            w = A @ Z[:, j]
            # Twice-modified Gram-Schmidt is important for ill-conditioned,
            # nonnormal collocation matrices.
            for _ in range(2):
                h = V[:, : j + 1].T @ w
                H[: j + 1, j] += h
                w -= V[:, : j + 1] @ h
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 10.0 * np.finfo(float).eps * Ainf:
                V[:, j + 1] = w / H[j + 1, j]
            H_arnoldi[: j + 2, j] = H[: j + 2, j]

            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            cs[j], sn[j] = _givens(H[j, j], H[j + 1, j])
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            total_iterations += 1
            history.append(float(abs(g[j + 1]) / bnorm))

            if capture_residual_vectors:
                y = la.solve_triangular(H[: j + 1, : j + 1], g[: j + 1], check_finite=False)
                residual_vectors.append(b - A @ (x + Z[:, : j + 1] @ y))
                residual_vector_iterations.append(total_iterations)

            if abs(g[j + 1]) <= threshold or H[j, j] == 0.0:
                used = j + 1
                break

        y = la.solve_triangular(H[:used, :used], g[:used], check_finite=False)
        x += Z[:, :used] @ y
        last_V, last_Z, last_H = (
            V[:, : used + 1],
            Z[:, :used],
            H_arnoldi[: used + 1, :used],
        )
        if arnoldi_cycles is not None:
            arnoldi_cycles.append(
                ArnoldiCycle(
                    last_V.copy(),
                    last_Z.copy(),
                    last_H.copy(),
                    cycle_first_iteration,
                )
            )
        true_residual = np.linalg.norm(b - A @ x)
        if true_residual <= threshold:
            break

    relative_residual = float(np.linalg.norm(b - A @ x) / bnorm)
    return FGMRESResult(
        x=x,
        converged=relative_residual <= max(atol / bnorm if bnorm else atol, rtol),
        iterations=total_iterations,
        solve_seconds=perf_counter() - start,
        relative_residual=relative_residual,
        residual_history=history,
        residual_vectors=residual_vectors,
        residual_vector_iterations=residual_vector_iterations,
        arnoldi_vectors=last_V if capture_arnoldi else None,
        preconditioned_vectors=last_Z if capture_arnoldi else None,
        hessenberg=last_H if capture_arnoldi else None,
        arnoldi_cycles=arnoldi_cycles,
    )
