"""Fixed-subspace GCRO-DR/NeurKItt projected-Arnoldi iteration.

The augmentation follows the paper exactly.  Given basis ``Y``, a reduced QR
factorization ``A Y = C R`` defines ``U = Y R^{-1}``, so ``A U = C`` and
``C.T C = I``.  At every restart the initial residual is projected away from
``range(C)``.  Arnoldi then uses ``(I-C C.T) A`` on right-preconditioned search
directions ``Z``.  The correction ``Z y - U B y``, where ``B=C.T A Z``,
restores the unprojected residual identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


VectorMap = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GmresResult:
    """Result of restarted baseline or projected GMRES."""

    x: np.ndarray
    converged: bool
    iterations: int
    solve_seconds: float
    relative_residual: float
    residual_history: tuple[float, ...]
    restart_cycles: int
    true_residual_history: tuple[float, ...] = ()
    true_residual_iterations: tuple[int, ...] = ()
    projection_records: tuple[dict[str, float | int | str], ...] = ()
    operation_counts: dict[str, int] | None = None
    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class CycleArnoldiData:
    """Completed restart-cycle data exposed to an optional update callback.

    ``residual_basis`` and ``hessenberg`` describe Arnoldi for the operator
    active in that cycle.  With a recycle pair this is
    ``(I-C C.T) A M^{-1}``; otherwise it is ``A M^{-1}``.
    ``solution_basis`` contains the corresponding ``M^{-1}V_m`` columns.
    The callback may return a new paired ``(U,C)`` for the *next* cycle.
    """

    cycle: int
    start_iteration: int
    end_iteration: int
    residual_basis: np.ndarray
    solution_basis: np.ndarray
    hessenberg: np.ndarray
    coupling: np.ndarray
    recycle_u: np.ndarray
    recycle_c: np.ndarray
    start_residual: np.ndarray
    end_residual: np.ndarray
    breakdown: bool


def build_augmentation(
    matrix: sp.spmatrix,
    invariant_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``U,C`` by the paper's thin-QR prescription."""

    basis = np.asarray(invariant_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != matrix.shape[0]:
        raise ValueError("invariant_basis must have shape (matrix dimension, k)")
    ay = np.asarray(matrix @ basis)
    c_basis, triangular = la.qr(ay, mode="economic", check_finite=False)
    # U = Y R^{-1}; solve R.T U.T = Y.T without forming an inverse.
    u_basis = la.solve_triangular(
        triangular.T,
        basis.T,
        lower=True,
        check_finite=False,
    ).T
    return u_basis, c_basis


def _givens(a: float, b: float) -> tuple[float, float]:
    if b == 0.0:
        return 1.0, 0.0
    if a == 0.0:
        return 0.0, 1.0
    scale = abs(a) + abs(b)
    norm = scale * np.sqrt((a / scale) ** 2 + (b / scale) ** 2)
    return a / norm, b / norm


def _least_squares_update(
    hessenberg_rotated: np.ndarray,
    hessenberg_arnoldi: np.ndarray,
    rhs_rotated: np.ndarray,
    beta: float,
    used: int,
) -> np.ndarray:
    diagonal = np.abs(np.diag(hessenberg_rotated[:used, :used]))
    scale = max(float(np.linalg.norm(hessenberg_rotated[:used, :used])), 1.0)
    if np.all(diagonal > 100.0 * np.finfo(float).eps * scale):
        return la.solve_triangular(
            hessenberg_rotated[:used, :used],
            rhs_rotated[:used],
            check_finite=False,
        )
    least_squares_rhs = np.zeros(used + 1)
    least_squares_rhs[0] = beta
    return la.lstsq(
        hessenberg_arnoldi[: used + 1, :used],
        least_squares_rhs,
        check_finite=False,
        lapack_driver="gelsy",
    )[0]


def restarted_projected_gmres(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    apply_preconditioner: VectorMap | None = None,
    *,
    u_basis: np.ndarray | None = None,
    c_basis: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    restart: int = 40,
    max_iterations: int | None = None,
    record_true_residuals: bool = False,
    record_projection_diagnostics: bool = False,
    cycle_update: Callable[
        [CycleArnoldiData], tuple[np.ndarray, np.ndarray] | None
    ]
    | None = None,
) -> GmresResult:
    """Solve with right-preconditioned restarted GMRES or fixed augmentation.

    Passing both ``u_basis`` and ``c_basis`` activates the fixed-space
    NeurKItt/GCRO projection.  With right preconditioning, Arnoldi is applied
    to ``(I-C C.T) A M^{-1}``, while trial updates use
    ``M^{-1}V y-U(C.T A M^{-1}V)y``.  Thus the recycle information is not
    inserted into every classical-preconditioner application.

    Stopping is always based on the true residual of the original,
    unpreconditioned system.  If ``record_true_residuals`` is true, that
    residual is explicitly recomputed at every Arnoldi step, matching the
    RBF-FD FGMRES instrumentation.  ``record_projection_diagnostics`` also
    records ``||C.T r||/||r||`` before/after restart projection and at every
    trial iterate, plus projected-Arnoldi orthogonality defects.  The optional
    ``cycle_update`` hook receives completed Arnoldi data and can replace the
    paired recycle bases for the following restart cycle.  It does not alter
    the cycle from which the data were obtained.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    b = np.asarray(rhs, dtype=float)
    if b.shape != (n,):
        raise ValueError("rhs must have shape (matrix dimension,)")
    if restart < 1:
        raise ValueError("restart must be positive")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("tolerances must be nonnegative")

    augmented = (u_basis is not None) or (c_basis is not None)
    if augmented and (u_basis is None or c_basis is None):
        raise ValueError("u_basis and c_basis must be supplied together")
    if augmented:
        u = np.asarray(u_basis, dtype=float)
        c = np.asarray(c_basis, dtype=float)
        if u.shape != c.shape or u.shape[0] != n:
            raise ValueError("u_basis and c_basis must have matching shape (n, k)")
    else:
        u = c = np.empty((n, 0), dtype=float)

    apply = apply_preconditioner or (lambda vector: vector.copy())
    x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float, copy=True)
    bnorm = float(np.linalg.norm(b))
    threshold = max(atol, rtol * bnorm)
    iteration_limit = n if max_iterations is None else max_iterations
    if iteration_limit < 1:
        raise ValueError("max_iterations must be positive")

    history: list[float] = []
    true_history: list[float] = []
    true_iterations: list[int] = []
    projection_records: list[dict[str, float | int | str]] = []
    operation_counts = {
        "a_matvecs": 0,
        "preconditioner_applications": 0,
        "w_transpose_applications": 0,
        "w_applications": 0,
        "u_applications": 0,
        "projected_arnoldi_steps": 0,
        "reduced_solves": 0,
    }
    arnoldi_relation_errors: list[float] = []
    arnoldi_basis_errors: list[float] = []
    arnoldi_w_errors: list[float] = []
    total_iterations = 0
    restart_cycles = 0
    start = perf_counter()

    def apply_matrix(vector: np.ndarray) -> np.ndarray:
        operation_counts["a_matvecs"] += 1
        return np.asarray(matrix @ vector, dtype=float)

    def rho(residual: np.ndarray) -> float:
        if not augmented:
            return 0.0
        operation_counts["w_transpose_applications"] += 1
        numerator = float(np.linalg.norm(c.T @ residual))
        return numerator / max(float(np.linalg.norm(residual)), np.finfo(float).tiny)

    residual = b - apply_matrix(x)
    initial_norm = float(np.linalg.norm(residual))
    true_history.append(initial_norm / bnorm if bnorm else initial_norm)
    true_iterations.append(0)
    if record_projection_diagnostics and augmented:
        projection_records.append(
            {
                "iteration": 0,
                "restart_cycle": 0,
                "stage": "initial_unprojected",
                "relative_residual": true_history[-1],
                "rho": rho(residual),
            }
        )

    while total_iterations < iteration_limit:
        if augmented:
            pre_projection_norm = float(np.linalg.norm(residual))
            rho_before = rho(residual) if record_projection_diagnostics else float("nan")
            operation_counts["w_transpose_applications"] += 1
            coarse_coordinates = c.T @ residual
            operation_counts["u_applications"] += 1
            x += u @ coarse_coordinates
            operation_counts["w_applications"] += 1
            residual -= c @ coarse_coordinates
            if record_projection_diagnostics:
                projection_records.extend(
                    (
                        {
                            "iteration": total_iterations,
                            "restart_cycle": restart_cycles + 1,
                            "stage": "cycle_pre_projection",
                            "relative_residual": pre_projection_norm
                            / (bnorm if bnorm else 1.0),
                            "rho": rho_before,
                        },
                        {
                            "iteration": total_iterations,
                            "restart_cycle": restart_cycles + 1,
                            "stage": "cycle_post_projection",
                            "relative_residual": float(np.linalg.norm(residual))
                            / (bnorm if bnorm else 1.0),
                            "rho": rho(residual),
                        },
                    )
                )
        beta = float(np.linalg.norm(residual))
        history.append(beta / bnorm if bnorm else beta)
        if beta <= threshold:
            break

        restart_cycles += 1
        width = min(restart, iteration_limit - total_iterations, n)
        vectors = np.zeros((n, width + 1), dtype=float)
        search_directions = np.zeros((n, width), dtype=float)
        hessenberg = np.zeros((width + 1, width), dtype=float)
        hessenberg_arnoldi = np.zeros((width + 1, width), dtype=float)
        coupling = np.zeros((c.shape[1], width), dtype=float)
        projected_images = np.zeros((n, width), dtype=float)
        cosines = np.zeros(width)
        sines = np.zeros(width)
        rotated_rhs = np.zeros(width + 1)
        vectors[:, 0] = residual / beta
        cycle_start_residual = residual.copy()
        cycle_start_iteration = total_iterations
        rotated_rhs[0] = beta
        used = width
        cycle_base = x.copy()
        last_trial = x.copy()
        last_trial_residual = residual.copy()
        inner_converged = False
        breakdown = False

        for j in range(width):
            operation_counts["preconditioner_applications"] += 1
            search_directions[:, j] = np.asarray(apply(vectors[:, j]), dtype=float)
            a_z = apply_matrix(search_directions[:, j])
            if augmented:
                operation_counts["w_transpose_applications"] += 1
                coupling[:, j] = c.T @ a_z
                operation_counts["w_applications"] += 1
                work = a_z - c @ coupling[:, j]
            else:
                work = a_z.copy()
            projected_images[:, j] = work
            operation_counts["projected_arnoldi_steps"] += 1

            arnoldi_scale = max(float(np.linalg.norm(work)), 1.0)
            # Twice-modified Gram-Schmidt keeps the projected basis stable.
            for _ in range(2):
                coefficients = vectors[:, : j + 1].T @ work
                hessenberg[: j + 1, j] += coefficients
                work -= vectors[:, : j + 1] @ coefficients
            # One final C-reprojection prevents loss of the defining
            # complementary-space orthogonality under strong nonnormality.
            # Since V is already C-orthogonal to roundoff, this does not alter
            # the Arnoldi coefficients beyond floating-point noise.
            if augmented:
                operation_counts["w_transpose_applications"] += 1
                c_coefficients = c.T @ work
                operation_counts["w_applications"] += 1
                work -= c @ c_coefficients
            hessenberg[j + 1, j] = np.linalg.norm(work)
            breakdown = hessenberg[j + 1, j] <= (
                100.0 * np.finfo(float).eps * arnoldi_scale
            )
            if not breakdown:
                vectors[:, j + 1] = work / hessenberg[j + 1, j]
            hessenberg_arnoldi[: j + 2, j] = hessenberg[: j + 2, j]

            for i in range(j):
                top = (
                    cosines[i] * hessenberg[i, j]
                    + sines[i] * hessenberg[i + 1, j]
                )
                hessenberg[i + 1, j] = (
                    -sines[i] * hessenberg[i, j]
                    + cosines[i] * hessenberg[i + 1, j]
                )
                hessenberg[i, j] = top
            cosines[j], sines[j] = _givens(
                hessenberg[j, j], hessenberg[j + 1, j]
            )
            hessenberg[j, j] = (
                cosines[j] * hessenberg[j, j]
                + sines[j] * hessenberg[j + 1, j]
            )
            hessenberg[j + 1, j] = 0.0
            rotated_rhs[j + 1] = -sines[j] * rotated_rhs[j]
            rotated_rhs[j] = cosines[j] * rotated_rhs[j]

            total_iterations += 1
            estimate = abs(rotated_rhs[j + 1])
            history.append(estimate / bnorm if bnorm else estimate)
            if record_true_residuals:
                trial_coefficients = _least_squares_update(
                    hessenberg,
                    hessenberg_arnoldi,
                    rotated_rhs,
                    beta,
                    j + 1,
                )
                trial_correction = (
                    search_directions[:, : j + 1] @ trial_coefficients
                )
                if augmented:
                    operation_counts["u_applications"] += 1
                    trial_correction -= u @ (
                        coupling[:, : j + 1] @ trial_coefficients
                    )
                last_trial = cycle_base + trial_correction
                last_trial_residual = b - apply_matrix(last_trial)
                true_norm = float(np.linalg.norm(last_trial_residual))
                true_history.append(true_norm / bnorm if bnorm else true_norm)
                true_iterations.append(total_iterations)
                if record_projection_diagnostics and augmented:
                    projection_records.append(
                        {
                            "iteration": total_iterations,
                            "restart_cycle": restart_cycles,
                            "stage": "arnoldi_trial",
                            "relative_residual": true_history[-1],
                            "rho": rho(last_trial_residual),
                        }
                    )
                if true_norm <= threshold:
                    used = j + 1
                    inner_converged = True
                    break
            elif estimate <= threshold:
                used = j + 1
                break
            if breakdown:
                used = j + 1
                break

        relation_denominator = max(
            float(np.linalg.norm(projected_images[:, :used], ord="fro")),
            np.finfo(float).tiny,
        )
        arnoldi_relation_errors.append(
            float(
                np.linalg.norm(
                    projected_images[:, :used]
                    - vectors[:, : used + 1]
                    @ hessenberg_arnoldi[: used + 1, :used],
                    ord="fro",
                )
                / relation_denominator
            )
        )
        active_basis = vectors[:, : used + (0 if breakdown else 1)]
        arnoldi_basis_errors.append(
            float(
                np.linalg.norm(
                    active_basis.T @ active_basis
                    - np.eye(active_basis.shape[1]),
                    ord="fro",
                )
            )
        )
        if augmented:
            arnoldi_w_errors.append(
                float(np.linalg.norm(c.T @ active_basis, ord="fro"))
            )

        cycle_converged = False
        if record_true_residuals:
            x = last_trial
            residual = last_trial_residual
            if inner_converged:
                history.append(true_history[-1])
                cycle_converged = True
        else:
            coefficients = _least_squares_update(
                hessenberg,
                hessenberg_arnoldi,
                rotated_rhs,
                beta,
                used,
            )
            correction = search_directions[:, :used] @ coefficients
            if augmented:
                operation_counts["u_applications"] += 1
                correction -= u @ (coupling[:, :used] @ coefficients)
            x += correction
            residual = b - apply_matrix(x)
            true_residual = float(np.linalg.norm(residual))
            true_history.append(true_residual / bnorm if bnorm else true_residual)
            true_iterations.append(total_iterations)
            if true_residual <= threshold:
                history.append(true_residual / bnorm if bnorm else true_residual)
                cycle_converged = True

        if cycle_update is not None:
            replacement = cycle_update(
                CycleArnoldiData(
                    cycle=restart_cycles,
                    start_iteration=cycle_start_iteration,
                    end_iteration=total_iterations,
                    residual_basis=vectors[:, : used + 1],
                    solution_basis=search_directions[:, :used],
                    hessenberg=hessenberg_arnoldi[: used + 1, :used],
                    coupling=coupling[:, :used],
                    recycle_u=u,
                    recycle_c=c,
                    start_residual=cycle_start_residual,
                    end_residual=residual.copy(),
                    breakdown=breakdown,
                )
            )
            if replacement is not None:
                next_u = np.asarray(replacement[0], dtype=float)
                next_c = np.asarray(replacement[1], dtype=float)
                if next_u.shape != next_c.shape or next_u.ndim != 2:
                    raise ValueError(
                        "cycle_update must return matching two-dimensional U,C bases"
                    )
                if next_u.shape[0] != n:
                    raise ValueError("cycle_update returned bases with wrong row count")
                u, c = next_u, next_c
                augmented = c.shape[1] > 0

        if cycle_converged:
            break

    solve_seconds = perf_counter() - start
    final_residual = float(np.linalg.norm(residual))
    relative_residual = final_residual / bnorm if bnorm else final_residual
    return GmresResult(
        x=x,
        converged=final_residual <= threshold,
        iterations=total_iterations,
        solve_seconds=solve_seconds,
        relative_residual=relative_residual,
        residual_history=tuple(history),
        restart_cycles=restart_cycles,
        true_residual_history=tuple(true_history),
        true_residual_iterations=tuple(true_iterations),
        projection_records=tuple(projection_records),
        operation_counts=operation_counts,
        diagnostics={
            "projected_arnoldi_relation_error_max": float(
                max(arnoldi_relation_errors, default=0.0)
            ),
            "arnoldi_basis_orthogonality_error_max": float(
                max(arnoldi_basis_errors, default=0.0)
            ),
            "arnoldi_basis_to_w_orthogonality_error_max": float(
                max(arnoldi_w_errors, default=0.0)
            ),
        },
    )
