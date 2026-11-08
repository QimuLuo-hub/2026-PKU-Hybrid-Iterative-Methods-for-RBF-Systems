"""Global radial kernels and analytic negative-Laplacian actions in 2D."""

from __future__ import annotations

import numpy as np


SHAPE_KERNELS = {"gaussian", "imq", "mq"}


def values(r: np.ndarray, kernel: str, epsilon: float | None) -> np.ndarray:
    if kernel in SHAPE_KERNELS and (epsilon is None or epsilon <= 0.0):
        raise ValueError(f"{kernel} requires epsilon > 0")
    if kernel == "gaussian":
        a = (float(epsilon) * r) ** 2
        return np.exp(-a)
    if kernel == "imq":
        a = (float(epsilon) * r) ** 2
        return (1.0 + a) ** -0.5
    if kernel == "mq":
        a = (float(epsilon) * r) ** 2
        return np.sqrt(1.0 + a)
    if kernel == "phs3":
        return r**3
    raise ValueError(f"unknown kernel: {kernel}")


def negative_laplacian(r: np.ndarray, kernel: str, epsilon: float | None) -> np.ndarray:
    """Evaluate ``-Delta phi(r)`` for two spatial dimensions."""

    if kernel in SHAPE_KERNELS and (epsilon is None or epsilon <= 0.0):
        raise ValueError(f"{kernel} requires epsilon > 0")
    if kernel == "gaussian":
        eps = float(epsilon)
        a = (eps * r) ** 2
        return 4.0 * eps**2 * (1.0 - a) * np.exp(-a)
    if kernel == "imq":
        eps = float(epsilon)
        a = (eps * r) ** 2
        return eps**2 * (2.0 - a) * (1.0 + a) ** -2.5
    if kernel == "mq":
        eps = float(epsilon)
        a = (eps * r) ** 2
        return -eps**2 * (2.0 + a) * (1.0 + a) ** -1.5
    if kernel == "phs3":
        # Delta r^m = m(m+d-2) r^(m-2); m=3, d=2.
        return -9.0 * r
    raise ValueError(f"unknown kernel: {kernel}")


def gradients(delta: np.ndarray, kernel: str, epsilon: float | None) -> np.ndarray:
    """Evaluate ``grad_x phi(||x-y||)`` from pairwise displacements ``x-y``.

    ``delta`` may have any leading shape, but its final axis must contain the
    spatial coordinates.  The formulas are nonsingular at coincident points.
    """

    if kernel in SHAPE_KERNELS and (epsilon is None or epsilon <= 0.0):
        raise ValueError(f"{kernel} requires epsilon > 0")
    squared_radius = np.sum(np.asarray(delta, dtype=float) ** 2, axis=-1)
    if kernel == "gaussian":
        eps = float(epsilon)
        scale = -2.0 * eps**2 * np.exp(-eps**2 * squared_radius)
    elif kernel == "imq":
        eps = float(epsilon)
        scale = -eps**2 * (1.0 + eps**2 * squared_radius) ** -1.5
    elif kernel == "mq":
        eps = float(epsilon)
        scale = eps**2 * (1.0 + eps**2 * squared_radius) ** -0.5
    elif kernel == "phs3":
        scale = 3.0 * np.sqrt(squared_radius)
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    return scale[..., None] * delta


def pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    delta = x[:, None, :] - y[None, :, :]
    return np.linalg.norm(delta, axis=2)
