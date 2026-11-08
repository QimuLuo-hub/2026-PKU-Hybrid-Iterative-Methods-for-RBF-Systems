"""Classical preconditioners for dense global RBF collocation matrices."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable
import warnings

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


@dataclass
class Preconditioner:
    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    setup_seconds: float
    memory_bytes: int
    notes: str
    apply_transpose: Callable[[np.ndarray], np.ndarray] | None = None

    def benchmark_application(self, n: int, repeats: int = 25, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((repeats, n))
        self.apply(vectors[0])
        start = perf_counter()
        for vector in vectors:
            self.apply(vector)
        return (perf_counter() - start) / repeats


def identity_preconditioner(n: int) -> Preconditioner:
    apply = lambda r: np.array(r, copy=True)
    return Preconditioner("identity", apply, 0.0, 0, "No preconditioning", apply)


def jacobi_preconditioner(A: np.ndarray) -> Preconditioner:
    start = perf_counter()
    diagonal = np.diag(A).copy()
    scale = np.linalg.norm(A, ord=np.inf)
    floor = np.finfo(float).eps * max(scale, 1.0)
    bad = np.abs(diagonal) <= floor
    if np.any(bad):
        raise la.LinAlgError(f"Jacobi has {np.count_nonzero(bad)} numerically zero diagonal entries")
    inverse_diagonal = 1.0 / diagonal
    return Preconditioner(
        "jacobi",
        lambda r: inverse_diagonal * r,
        perf_counter() - start,
        inverse_diagonal.nbytes,
        "Coefficient-space diagonal inverse",
        lambda r: inverse_diagonal * r,
    )


def dense_block_jacobi_preconditioner(
    A: np.ndarray,
    nodes: np.ndarray,
    *,
    blocks_per_axis: int = 4,
) -> Preconditioner:
    """Nonoverlapping geometric block Jacobi with pivoted dense block LU."""

    if A.shape[0] != len(nodes):
        raise ValueError("block Jacobi currently supports unaugmented systems only")
    start = perf_counter()
    bins = np.minimum((nodes * blocks_per_axis).astype(int), blocks_per_axis - 1)
    labels = bins[:, 0] + blocks_per_axis * bins[:, 1]
    factors: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        with warnings.catch_warnings():
            warnings.simplefilter("error", la.LinAlgWarning)
            lu, piv = la.lu_factor(A[np.ix_(idx, idx)], check_finite=False)
        factors.append((idx, lu, piv))

    def apply(r: np.ndarray) -> np.ndarray:
        out = np.empty_like(r, dtype=float)
        for idx, lu, piv in factors:
            out[idx] = la.lu_solve((lu, piv), r[idx], check_finite=False)
        return out

    memory = sum(idx.nbytes + lu.nbytes + piv.nbytes for idx, lu, piv in factors)
    block_sizes = [len(idx) for idx, _, _ in factors]
    return Preconditioner(
        f"block_jacobi_{blocks_per_axis}x{blocks_per_axis}",
        apply,
        perf_counter() - start,
        int(memory),
        f"{len(factors)} geometric blocks; sizes {min(block_sizes)}-{max(block_sizes)}",
    )


def local_truncation_preconditioner(
    A: np.ndarray,
    nodes: np.ndarray,
    *,
    neighbors: int = 49,
) -> Preconditioner:
    """Exact sparse LU of a distance-truncated global collocation matrix.

    This surrogate is meaningful only for kernels whose matrix entries decay
    with center distance.  It is not an RBF-FD discretization: retained entries
    remain evaluations of the original global basis functions.
    """

    if A.shape[0] != len(nodes):
        raise ValueError("local truncation currently supports unaugmented systems only")
    start = perf_counter()
    n = len(nodes)
    k = min(neighbors, n)
    _, nearest = cKDTree(nodes).query(nodes, k=k)
    if k == 1:
        nearest = nearest[:, None]
    rows = np.repeat(np.arange(n), k)
    cols = nearest.ravel()
    data = A[rows, cols]
    surrogate = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
    surrogate.sum_duplicates()
    factor = spla.splu(surrogate.tocsc(), permc_spec="COLAMD")

    def apply(r: np.ndarray) -> np.ndarray:
        return factor.solve(r)

    memory = (
        surrogate.data.nbytes
        + surrogate.indices.nbytes
        + surrogate.indptr.nbytes
        + factor.L.data.nbytes
        + factor.L.indices.nbytes
        + factor.L.indptr.nbytes
        + factor.U.data.nbytes
        + factor.U.indices.nbytes
        + factor.U.indptr.nbytes
    )
    return Preconditioner(
        f"local_truncation_k{k}",
        apply,
        perf_counter() - start,
        int(memory),
        f"Sparse LU of nearest-center truncation; {surrogate.nnz} retained entries",
    )


def local_sparse_approximate_inverse(
    A: np.ndarray,
    nodes: np.ndarray,
    *,
    support_size: int = 25,
    row_oversampling: int = 3,
) -> Preconditioner:
    """Build an ACBF-inspired localized least-squares approximate inverse.

    Column ``j`` is supported on the nearest coefficient centers and minimizes
    ``||A[R,S] z-e_j[R]||`` on an oversampled local equation set.  This is a
    practical approximate-cardinal baseline, not an implementation of any one
    published ACBF algorithm.
    """

    if A.shape[0] != len(nodes):
        raise ValueError("localized approximate inverse supports unaugmented systems only")
    start = perf_counter()
    n = len(nodes)
    k = min(support_size, n)
    kr = min(row_oversampling * k, n)
    tree = cKDTree(nodes)
    _, supports = tree.query(nodes, k=k)
    _, row_sets = tree.query(nodes, k=kr)
    if k == 1:
        supports = supports[:, None]
    if kr == 1:
        row_sets = row_sets[:, None]
    rows_out: list[int] = []
    cols_out: list[int] = []
    data_out: list[float] = []
    for j in range(n):
        support = np.asarray(supports[j], dtype=int)
        row_idx = np.unique(np.concatenate([np.asarray(row_sets[j], dtype=int), [j]]))
        target = np.zeros(len(row_idx))
        target[np.flatnonzero(row_idx == j)[0]] = 1.0
        coeff, *_ = la.lstsq(A[np.ix_(row_idx, support)], target, lapack_driver="gelsy", check_finite=False)
        rows_out.extend(support.tolist())
        cols_out.extend([j] * len(support))
        data_out.extend(coeff.tolist())
    inverse = sp.csc_matrix((data_out, (rows_out, cols_out)), shape=A.shape)

    def apply(r: np.ndarray) -> np.ndarray:
        return np.asarray(inverse @ r).ravel()

    memory = inverse.data.nbytes + inverse.indices.nbytes + inverse.indptr.nbytes
    return Preconditioner(
        f"local_lsai_k{k}",
        apply,
        perf_counter() - start,
        int(memory),
        f"Localized least-squares approximate inverse with {inverse.nnz} entries",
    )


def global_residual_sparse_approximate_inverse(
    A: np.ndarray,
    nodes: np.ndarray,
    *,
    support_size: int = 25,
) -> Preconditioner:
    """Sparse approximate inverse with local columns and global residual fits.

    For each right-hand-side coordinate ``e_j``, the nonzeros of the
    approximate inverse column are restricted geometrically, but the
    least-squares residual is minimized over *all* collocation equations.  The
    global fit is costlier than ``local_sparse_approximate_inverse`` and is a
    closer proxy for approximate-cardinal/global approximate-inverse ideas.
    """

    if A.shape[0] != len(nodes):
        raise ValueError("global-residual approximate inverse supports unaugmented systems only")
    start = perf_counter()
    n = len(nodes)
    k = min(support_size, n)
    _, supports = cKDTree(nodes).query(nodes, k=k)
    if k == 1:
        supports = supports[:, None]
    rows_out: list[int] = []
    cols_out: list[int] = []
    data_out: list[float] = []
    target = np.zeros(n)
    for j in range(n):
        support = np.asarray(supports[j], dtype=int)
        target[j] = 1.0
        coeff, *_ = la.lstsq(A[:, support], target, lapack_driver="gelsy", check_finite=False)
        target[j] = 0.0
        rows_out.extend(support.tolist())
        cols_out.extend([j] * len(support))
        data_out.extend(coeff.tolist())
    inverse = sp.csc_matrix((data_out, (rows_out, cols_out)), shape=A.shape)

    def apply(r: np.ndarray) -> np.ndarray:
        return np.asarray(inverse @ r).ravel()

    memory = inverse.data.nbytes + inverse.indices.nbytes + inverse.indptr.nbytes
    return Preconditioner(
        f"global_residual_lsai_k{k}",
        apply,
        perf_counter() - start,
        int(memory),
        f"Global-residual least-squares approximate inverse with {inverse.nnz} entries",
    )


def rbf_local_cardinal_preconditioner(
    A: np.ndarray,
    nodes: np.ndarray,
    *,
    support_size: int = 49,
) -> Preconditioner:
    """Construct a local least-squares approximate-cardinal preconditioner.

    The ``i``-th row of ``W`` is restricted to the collocation equations at
    the ``support_size`` nearest nodes to coefficient center ``i``.  Its
    weights solve

    ``min_w ||A[S_i, :].T w - e_i||_2``,

    so the assembled row-sparse matrix targets ``W A = I``.  Applying ``W``
    to a residual therefore maps collocation-equation residuals back to an
    approximate coefficient correction.  This is the least-squares
    approximate-cardinal construction for the dense RBF-PDE matrix; it is not
    an RBF-FD stencil discretization.

    The implementation uses a rank-revealing QR least-squares solve for each
    row.  It currently supports unaugmented systems, for which collocation
    equations and coefficient centers have the same node indexing.
    """

    matrix = np.asarray(A, dtype=float)
    points = np.asarray(nodes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("local-cardinal preconditioning requires a square matrix")
    if matrix.shape[0] != len(points):
        raise ValueError("local-cardinal preconditioning currently supports unaugmented systems only")
    if support_size < 1:
        raise ValueError("support_size must be positive")

    start = perf_counter()
    n = len(points)
    k = min(int(support_size), n)
    _, supports = cKDTree(points).query(points, k=k)
    supports = np.asarray(supports, dtype=np.int32)
    if k == 1:
        supports = supports[:, None]

    coefficients = np.empty((n, k), dtype=float)
    target = np.zeros(n, dtype=float)
    for i in range(n):
        support = supports[i]
        target[i] = 1.0
        coefficients[i], *_ = la.lstsq(
            matrix[support, :].T,
            target,
            lapack_driver="gelsy",
            check_finite=False,
        )
        target[i] = 0.0

    indptr = np.arange(0, (n + 1) * k, k, dtype=np.int32)
    inverse = sp.csr_matrix(
        (coefficients.ravel(), supports.ravel(), indptr),
        shape=matrix.shape,
    )

    def apply(r: np.ndarray) -> np.ndarray:
        return np.asarray(inverse @ r).ravel()

    memory = inverse.data.nbytes + inverse.indices.nbytes + inverse.indptr.nbytes
    return Preconditioner(
        f"rbf_local_cardinal_ls_k{k}",
        apply,
        perf_counter() - start,
        int(memory),
        (
            "Row-sparse RBF-PDE approximate-cardinal inverse; "
            f"{k} nearest collocation equations per coefficient and {inverse.nnz} stored weights"
        ),
        lambda r: np.asarray(inverse.T @ r).ravel(),
    )


def low_precision_lu_preconditioner(A: np.ndarray) -> Preconditioner:
    """Pivoted FP32 LU used inside a double-precision Krylov solve."""

    start = perf_counter()
    A32 = np.asarray(A, dtype=np.float32, order="F")
    with warnings.catch_warnings():
        warnings.simplefilter("error", la.LinAlgWarning)
        lu, piv = la.lu_factor(A32, check_finite=False, overwrite_a=True)

    def apply(r: np.ndarray) -> np.ndarray:
        rhs = np.asarray(r, dtype=np.float32)
        return la.lu_solve((lu, piv), rhs, check_finite=False).astype(float)

    return Preconditioner(
        "pivoted_lu_fp32",
        apply,
        perf_counter() - start,
        int(lu.nbytes + piv.nbytes),
        "FP32 pivoted LU factors; triangular solves cast back to FP64",
        lambda r: la.lu_solve(
            (lu, piv),
            np.asarray(r, dtype=np.float32),
            trans=1,
            check_finite=False,
        ).astype(float),
    )


def exact_lu_preconditioner(A: np.ndarray) -> Preconditioner:
    """Double-precision direct-solve ceiling (not a cost-reduced baseline)."""

    start = perf_counter()
    lu, piv = la.lu_factor(A, check_finite=False)

    def apply(r: np.ndarray) -> np.ndarray:
        return la.lu_solve((lu, piv), r, check_finite=False)

    return Preconditioner(
        "pivoted_lu_fp64_ceiling",
        apply,
        perf_counter() - start,
        int(lu.nbytes + piv.nbytes),
        "Exact dense LU ceiling/reference; same O(N^3) setup class as a direct solve",
        lambda r: la.lu_solve((lu, piv), r, trans=1, check_finite=False),
    )
