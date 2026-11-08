"""Node generators for the unit-square global collocation experiments."""

from __future__ import annotations

import numpy as np


def square_nodes(
    side: int,
    distribution: str = "tensor",
    *,
    jitter_fraction: float = 0.22,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and a boundary mask on ``[0, 1]^2``.

    Jitter is applied only to interior tensor nodes.  Boundary nodes remain on
    the boundary so that Dirichlet equations are imposed at their true
    geometric locations.
    """

    if side < 3:
        raise ValueError("side must be at least 3")
    grid = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    boundary = (
        np.isclose(nodes[:, 0], 0.0)
        | np.isclose(nodes[:, 0], 1.0)
        | np.isclose(nodes[:, 1], 0.0)
        | np.isclose(nodes[:, 1], 1.0)
    )
    if distribution == "tensor":
        return nodes, boundary
    if distribution != "jittered":
        raise ValueError(f"unknown node distribution: {distribution}")

    rng = np.random.default_rng(seed)
    h = 1.0 / (side - 1)
    perturbation = rng.uniform(
        -jitter_fraction * h,
        jitter_fraction * h,
        size=(np.count_nonzero(~boundary), 2),
    )
    nodes[~boundary] += perturbation
    nodes[~boundary] = np.clip(nodes[~boundary], 0.05 * h, 1.0 - 0.05 * h)
    return nodes, boundary
