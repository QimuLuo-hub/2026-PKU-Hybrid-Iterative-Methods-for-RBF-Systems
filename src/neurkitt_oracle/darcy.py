"""Darcy-flow construction used by the NeurKItt reference implementation.

The paper defines ``-div(K grad h) = f`` with a GRF permeability and constant
forcing.  Its released ``data/darcy.py`` supplies the remaining details used
here: a periodic GRF with ``alpha=2`` and ``tau=3``, ``K=exp(GRF)``, a
five-point variable-coefficient stencil with arithmetic face averages, and
zero Dirichlet boundary values.  A coefficient grid of side ``s`` therefore
produces ``(s-2)^2`` interior unknowns.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


def sample_log_permeability(
    grid_size: int,
    *,
    seed: int = 0,
    alpha: float = 2.0,
    tau: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(log_K, K)`` from the paper code's periodic Gaussian RF.

    NumPy's complex standard normal is scaled so that its real and imaginary
    parts have variance 1/2, matching ``torch.randn(..., dtype=cfloat)``.
    """

    if grid_size < 4 or grid_size % 2:
        raise ValueError("grid_size must be an even integer of at least 4")
    if alpha <= 0.0 or tau <= 0.0:
        raise ValueError("alpha and tau must be positive")

    rng = np.random.default_rng(seed)
    half = grid_size // 2
    wavenumbers = np.concatenate((np.arange(half), np.arange(-half, 0)))
    k_x = wavenumbers[:, None]
    k_y = wavenumbers[None, :]
    sigma = tau ** (0.5 * (2.0 * alpha - 2.0))
    sqrt_eig = (
        grid_size**2
        * math.sqrt(2.0)
        * sigma
        * (4.0 * math.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2.0)
    )
    sqrt_eig[0, 0] = 0.0

    coefficients = (
        rng.standard_normal((grid_size, grid_size))
        + 1j * rng.standard_normal((grid_size, grid_size))
    ) / math.sqrt(2.0)
    log_permeability = np.fft.ifftn(sqrt_eig * coefficients).real
    return log_permeability, np.exp(log_permeability)


def _assemble_reference_stencil(permeability: np.ndarray) -> sp.csr_matrix:
    """Assemble the released NeurKItt Darcy stencil without PyTorch."""

    coefficient = np.asarray(permeability, dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("permeability must be a square two-dimensional array")
    grid_size = coefficient.shape[0]
    interior_side = grid_size - 2
    if interior_side < 1:
        raise ValueError("permeability grid must contain interior points")

    diagonal_blocks: list[np.ndarray] = []
    off_diagonal_blocks: list[np.ndarray] = []
    for j in range(1, grid_size - 1):
        within_column = np.concatenate(
            (
                -0.5
                * (
                    coefficient[1 : grid_size - 2, j]
                    + coefficient[2 : grid_size - 1, j]
                ),
                [0.0],
            )
        )
        main = (
            0.5
            * (
                coefficient[0 : grid_size - 2, j]
                + coefficient[1 : grid_size - 1, j]
            )
            + 0.5
            * (
                coefficient[2:grid_size, j]
                + coefficient[1 : grid_size - 1, j]
            )
            + 0.5
            * (
                coefficient[1 : grid_size - 1, j - 1]
                + coefficient[1 : grid_size - 1, j]
            )
            + 0.5
            * (
                coefficient[1 : grid_size - 1, j + 1]
                + coefficient[1 : grid_size - 1, j]
            )
        )
        diagonal_blocks.append(np.vstack((within_column, main, within_column)))

        if j != grid_size - 2:
            off_diagonal_blocks.append(
                -0.5
                * (
                    coefficient[1 : grid_size - 1, j]
                    + coefficient[1 : grid_size - 1, j + 1]
                )
            )

    diagonal_data = np.concatenate(diagonal_blocks, axis=1)
    off_diagonal_data = np.concatenate(off_diagonal_blocks)
    dimension = interior_side**2
    matrix = sp.diags(
        diagonal_data,
        offsets=(-1, 0, 1),
        shape=(dimension, dimension),
        format="csr",
    )
    matrix += sp.diags(
        (off_diagonal_data, off_diagonal_data),
        offsets=(-interior_side, interior_side),
        shape=(dimension, dimension),
        format="csr",
    )
    return (matrix * (grid_size - 1) ** 2).tocsr()


def build_darcy_system(
    grid_size: int = 100,
    *,
    seed: int = 0,
    alpha: float = 2.0,
    tau: float = 3.0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``A, b, log_K, K`` for the NeurKItt Darcy problem."""

    log_permeability, permeability = sample_log_permeability(
        grid_size,
        seed=seed,
        alpha=alpha,
        tau=tau,
    )
    matrix = _assemble_reference_stencil(permeability)
    rhs = np.ones(matrix.shape[0], dtype=float)
    return matrix, rhs, log_permeability, permeability
