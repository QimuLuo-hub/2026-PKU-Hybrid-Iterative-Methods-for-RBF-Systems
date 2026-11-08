"""Shared benchmark helpers for global-RBF preconditioned FGMRES runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .krylov import FGMRESResult, fgmres
from .preconditioners import Preconditioner


def true_residual_norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Return ``||b-Ax||_2`` using the original, unpreconditioned system."""

    return float(np.linalg.norm(np.asarray(b) - np.asarray(A) @ np.asarray(x)))


def true_relative_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Return ``||b-Ax||_2 / ||b||_2`` for the original system."""

    numerator = true_residual_norm(A, b, x)
    denominator = float(np.linalg.norm(b))
    return numerator / denominator if denominator else numerator


@dataclass(frozen=True)
class FGMRESBenchmark:
    result: FGMRESResult
    solve_seconds_median: float
    final_true_residual_norm: float
    final_true_relative_residual: float


def benchmark_fgmres(
    A: np.ndarray,
    b: np.ndarray,
    preconditioner: Preconditioner,
    *,
    rtol: float,
    restart: int,
    max_iterations: int,
    repeats: int = 3,
    warmup: bool = True,
    capture_residual_vectors: bool = False,
) -> FGMRESBenchmark:
    """Run repeated FGMRES solves and recompute the reported true residual.

    Timing uses the median recorded solve.  Iterations and the returned
    solution come from the first recorded solve.  Crucially, both residual
    fields are recomputed from the original ``A`` and ``b`` rather than from
    a preconditioned operator or the inner GMRES residual estimate.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if warmup:
        fgmres(
            A,
            b,
            preconditioner.apply,
            rtol=rtol,
            restart=restart,
            max_iterations=max_iterations,
        )
    results = [
        fgmres(
            A,
            b,
            preconditioner.apply,
            rtol=rtol,
            restart=restart,
            max_iterations=max_iterations,
            capture_residual_vectors=capture_residual_vectors and repeat == 0,
        )
        for repeat in range(repeats)
    ]
    primary = results[0]
    return FGMRESBenchmark(
        result=primary,
        solve_seconds_median=float(np.median([result.solve_seconds for result in results])),
        final_true_residual_norm=true_residual_norm(A, b, primary.x),
        final_true_relative_residual=true_relative_residual(A, b, primary.x),
    )
