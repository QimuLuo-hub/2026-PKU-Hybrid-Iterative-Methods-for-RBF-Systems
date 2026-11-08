"""Memory-aware utilities for scaling tensor-grid global Gaussian collocation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse.linalg as spla

from .collocation import GlobalRBFSystem, exact_solution, forcing
from .nodes import square_nodes


def assemble_tensor_gaussian_poisson_blocked(
    side: int,
    *,
    epsilon: float,
    block_rows: int = 128,
) -> GlobalRBFSystem:
    """Assemble the existing tensor-grid Gaussian system with bounded workspace.

    The formulas and row ordering match :func:`assemble_poisson_system`, but
    row blocking avoids allocating full pairwise displacement, distance,
    kernel-value, and Laplacian arrays simultaneously.
    """

    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    start = perf_counter()
    nodes, boundary = square_nodes(side, "tensor")
    dimension = side**2
    matrix = np.empty((dimension, dimension), dtype=float, order="C")
    epsilon_squared = float(epsilon) ** 2
    for first in range(0, dimension, block_rows):
        last = min(first + block_rows, dimension)
        dx = nodes[first:last, None, 0] - nodes[None, :, 0]
        dy = nodes[first:last, None, 1] - nodes[None, :, 1]
        scaled_squared_radius = epsilon_squared * (dx * dx + dy * dy)
        values = np.exp(-scaled_squared_radius)
        operator = 4.0 * epsilon_squared * (1.0 - scaled_squared_radius) * values
        matrix[first:last] = np.where(
            boundary[first:last, None],
            values,
            operator,
        )
    rhs = np.where(boundary, exact_solution(nodes), forcing(nodes))
    return GlobalRBFSystem(
        A=matrix,
        b=np.asarray(rhs),
        nodes=nodes,
        boundary=boundary,
        kernel="gaussian",
        epsilon=float(epsilon),
        polynomial_degree=None,
        assembly_seconds=perf_counter() - start,
    )


@dataclass(frozen=True)
class TensorGaussianPoissonOperator:
    """Kronecker-structured action of the same dense collocation matrix."""

    side: int
    epsilon: float

    def __post_init__(self) -> None:
        if self.side < 3:
            raise ValueError("side must be at least three")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        grid = np.linspace(0.0, 1.0, self.side)
        difference = grid[:, None] - grid[None, :]
        scaled_squared = (self.epsilon * difference) ** 2
        interpolation = np.exp(-scaled_squared)
        one_dimensional_operator = (
            2.0
            * self.epsilon**2
            * (1.0 - 2.0 * scaled_squared)
            * interpolation
        )
        boundary = np.zeros((self.side, self.side), dtype=bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        object.__setattr__(self, "_interpolation", interpolation)
        object.__setattr__(self, "_operator", one_dimensional_operator)
        object.__setattr__(self, "_boundary", boundary)

    @property
    def dimension(self) -> int:
        return self.side**2

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(vector).reshape(self.side, self.side)
        interpolation_action = (
            self._interpolation @ coefficients @ self._interpolation.T
        )
        operator_action = (
            self._operator @ coefficients @ self._interpolation.T
            + self._interpolation @ coefficients @ self._operator.T
        )
        return np.where(self._boundary, interpolation_action, operator_action).ravel()

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        equations = np.asarray(vector).reshape(self.side, self.side)
        boundary_equations = np.where(self._boundary, equations, 0.0)
        interior_equations = np.where(self._boundary, 0.0, equations)
        transpose_action = (
            self._interpolation.T @ boundary_equations @ self._interpolation
            + self._operator.T @ interior_equations @ self._interpolation
            + self._interpolation.T @ interior_equations @ self._operator
        )
        return transpose_action.ravel()

    def as_linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=float,
        )
