"""Quasi-uniform node generation on the unit square."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import DiscretizationConfig


def generate_nodes(config: DiscretizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return row-major square-grid nodes and a boundary mask.

    ``n_nodes`` must be a perfect square.  ``uniform_grid`` leaves the tensor
    grid unchanged. ``jittered_grid`` independently perturbs only interior
    nodes by at most ``jitter_fraction`` grid spacings, retaining a
    quasi-uniform set and exact boundary locations.  The row-major topology is
    preserved across jitter seeds so same-size transfer tests have a defined
    node correspondence.
    """

    n_side = math.isqrt(config.n_nodes)
    if n_side * n_side != config.n_nodes or n_side < 3:
        raise ValueError("n_nodes must be a perfect square with at least 3 nodes per side")
    if config.node_distribution not in {
        "uniform_grid",
        "jittered_grid",
        "warped_grid",
        "center_clustered_grid",
    }:
        raise ValueError(f"unsupported node distribution: {config.node_distribution}")
    if not 0.0 <= config.jitter_fraction < 0.5:
        raise ValueError("jitter_fraction must lie in [0, 0.5)")

    if not 0.0 <= config.warp_strength < 1.0:
        raise ValueError("warp_strength must lie in [0, 1)")
    if config.density_exponent < 1.0:
        raise ValueError("density_exponent must be at least one")
    axis = np.linspace(0.0, 1.0, n_side)
    if config.node_distribution == "warped_grid":
        axis = axis + config.warp_strength * np.sin(2.0 * np.pi * axis) / (
            2.0 * np.pi
        )
    elif config.node_distribution == "center_clustered_grid":
        # A monotone coordinate map gives a nested, locally refined cloud near
        # the domain centre while preserving the square boundary exactly.
        centered = 2.0 * axis - 1.0
        axis = 0.5 + 0.5 * np.sign(centered) * np.abs(centered) ** config.density_exponent
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    tol = 10.0 * np.finfo(float).eps
    boundary = np.any((nodes <= tol) | (nodes >= 1.0 - tol), axis=1)

    if config.node_distribution == "jittered_grid" and config.jitter_fraction > 0.0:
        rng = np.random.default_rng(config.seed)
        h = 1.0 / (n_side - 1)
        offsets = rng.uniform(
            -config.jitter_fraction * h,
            config.jitter_fraction * h,
            size=(np.count_nonzero(~boundary), 2),
        )
        nodes[~boundary] += offsets

    return nodes, boundary


def node_irregularity_metrics(nodes: np.ndarray) -> dict:
    """Return geometry-only nearest-neighbor irregularity diagnostics."""

    nodes = np.asarray(nodes, dtype=float)
    distances, _ = cKDTree(nodes).query(nodes, k=2)
    nearest = distances[:, 1]
    mean = float(np.mean(nearest))
    minimum = float(np.min(nearest))
    return {
        "nearest_neighbor_cv": float(np.std(nearest) / mean),
        "nearest_neighbor_ratio": float(np.max(nearest) / minimum),
        "minimum_nearest_neighbor_distance": minimum,
    }
