"""Restarted right-preconditioned GMRES with true-residual instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp

from .config import SolverConfig
from .preconditioners import InverseAction


ResidualCallback = Callable[[int, np.ndarray, np.ndarray], None]
PreconditionerInputCallback = Callable[[int, np.ndarray], None]


@dataclass
class KrylovResult:
    """GMRES result including every explicitly recomputed true residual."""

    solution: np.ndarray
    converged: bool
    iterations: int
    residual_norms: np.ndarray
    relative_residuals: np.ndarray
    solve_seconds: float
    preconditioner_seconds: float
    failure: str = ""


def restarted_gmres(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: SolverConfig,
    preconditioner: Optional[InverseAction] = None,
    initial_guess: Optional[np.ndarray] = None,
    callback: Optional[ResidualCallback] = None,
    preconditioner_input_callback: Optional[PreconditionerInputCallback] = None,
) -> KrylovResult:
    r"""Solve ``A x = b`` by restarted right-preconditioned GMRES.

    Arnoldi is applied to ``A P`` with ``z_j=P(v_j)`` and solution updates in
    the stored ``z_j`` vectors.  For fixed linear ``P`` (all Stage 1-5 cases),
    this is right-preconditioned GMRES.  Retaining each ``z_j`` also makes the
    implementation algebraically ready for flexible actions in later work.

    The residual is explicitly recomputed for each trial iterate, so the saved
    history is ``||b-Ax_k||/||b||`` rather than a preconditioned proxy.  When
    supplied, ``preconditioner_input_callback`` receives a copy of the actual
    Arnoldi vector immediately before every application of ``P``.  This is
    intentionally distinct from the true-residual callback.
    """

    rhs = np.asarray(rhs, dtype=float)
    n = rhs.size
    if matrix.shape != (n, n):
        raise ValueError("matrix and rhs dimensions do not agree")
    x = np.zeros(n, dtype=float) if initial_guess is None else np.array(initial_guess, float)
    if x.shape != (n,):
        raise ValueError("initial_guess has the wrong shape")
    if preconditioner is not None:
        preconditioner.reset_timing()

    rhs_norm = float(np.linalg.norm(rhs))
    scale = rhs_norm if rhs_norm > 0.0 else 1.0
    tolerance = max(config.absolute_tolerance, config.relative_tolerance * rhs_norm)
    residual = rhs - matrix @ x
    residual_norm = float(np.linalg.norm(residual))
    residual_norms: List[float] = [residual_norm]
    relative_residuals: List[float] = [residual_norm / scale]
    if residual_norm <= tolerance:
        return KrylovResult(
            x, True, 0, np.array(residual_norms), np.array(relative_residuals), 0.0, 0.0
        )

    start = perf_counter()
    iterations = 0
    failure = ""
    converged = False

    try:
        while iterations < config.max_iterations and not converged:
            residual = rhs - matrix @ x
            beta = float(np.linalg.norm(residual))
            inner_limit = min(config.restart, config.max_iterations - iterations)
            basis = np.zeros((n, inner_limit + 1), dtype=float)
            preconditioned_basis = np.zeros((n, inner_limit), dtype=float)
            hessenberg = np.zeros((inner_limit + 1, inner_limit), dtype=float)
            basis[:, 0] = residual / beta
            cycle_base = x.copy()
            last_trial = x.copy()

            for inner in range(inner_limit):
                vector = basis[:, inner]
                if preconditioner_input_callback is not None:
                    preconditioner_input_callback(iterations + 1, vector.copy())
                z = vector.copy() if preconditioner is None else preconditioner.apply(vector)
                preconditioned_basis[:, inner] = z
                work = np.asarray(matrix @ z, dtype=float)

                # Modified Gram-Schmidt with one reorthogonalization pass.
                for column in range(inner + 1):
                    coefficient = float(np.dot(basis[:, column], work))
                    hessenberg[column, inner] += coefficient
                    work -= coefficient * basis[:, column]
                for column in range(inner + 1):
                    correction = float(np.dot(basis[:, column], work))
                    hessenberg[column, inner] += correction
                    work -= correction * basis[:, column]

                next_norm = float(np.linalg.norm(work))
                hessenberg[inner + 1, inner] = next_norm
                if next_norm > config.breakdown_tolerance:
                    basis[:, inner + 1] = work / next_norm

                target = np.zeros(inner + 2, dtype=float)
                target[0] = beta
                coefficients, *_ = np.linalg.lstsq(
                    hessenberg[: inner + 2, : inner + 1], target, rcond=None
                )
                last_trial = cycle_base + preconditioned_basis[:, : inner + 1] @ coefficients
                trial_residual = rhs - matrix @ last_trial
                trial_norm = float(np.linalg.norm(trial_residual))
                iterations += 1
                residual_norms.append(trial_norm)
                relative_residuals.append(trial_norm / scale)
                if callback is not None:
                    callback(iterations, last_trial.copy(), trial_residual.copy())

                if trial_norm <= tolerance:
                    x = last_trial
                    converged = True
                    break
                if next_norm <= config.breakdown_tolerance:
                    x = last_trial
                    failure = "Arnoldi breakdown before convergence"
                    break

            x = last_trial
            if failure:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    solve_seconds = perf_counter() - start
    preconditioner_seconds = (
        0.0 if preconditioner is None else float(preconditioner.apply_seconds)
    )
    if not converged and not failure and iterations >= config.max_iterations:
        failure = "maximum iterations reached"
    return KrylovResult(
        x,
        converged,
        iterations,
        np.asarray(residual_norms),
        np.asarray(relative_residuals),
        solve_seconds,
        preconditioner_seconds,
        failure,
    )
