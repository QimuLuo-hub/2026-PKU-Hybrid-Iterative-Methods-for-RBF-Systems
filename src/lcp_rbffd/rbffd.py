"""Sparse, polynomially augmented RBF-FD discretization of ``-Delta``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .config import DiscretizationConfig, RBFConfig
from .nodes import generate_nodes, node_irregularity_metrics
from .pde import (
    ConstantAnisotropicDiffusion,
    ExponentialCoefficient,
    LocalizedCoefficient,
    SpatialAnisotropicDiffusion,
)


class RBFWeightError(RuntimeError):
    """Raised when a local augmented RBF-FD system cannot be solved reliably."""


@dataclass
class RBFSystem:
    """An assembled full Dirichlet RBF-FD linear system."""

    matrix: sp.csr_matrix
    nodes: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray
    diagnostics: Dict[str, float]
    config: DiscretizationConfig
    coefficient: Optional[object] = None


@dataclass(frozen=True)
class OperatorCoefficients:
    """Physical coefficients in c_L(-Delta u) + c_x u_x + c_y u_y."""

    negative_laplacian: float = 1.0
    negative_xx: float = 0.0
    negative_xy: float = 0.0
    negative_yy: float = 0.0
    first_x: float = 0.0
    first_y: float = 0.0


def monomial_powers(degree: int) -> List[Tuple[int, int]]:
    """Return two-dimensional monomial exponents of total degree at most ``degree``."""

    if degree < 0:
        raise ValueError("polynomial degree must be nonnegative")
    return [(a, total - a) for total in range(degree + 1) for a in range(total + 1)]


def _polynomial_matrix(points: np.ndarray, powers: Sequence[Tuple[int, int]]) -> np.ndarray:
    return np.column_stack([points[:, 0] ** a * points[:, 1] ** b for a, b in powers])


def _negative_laplacian_polynomials_at_origin(
    powers: Sequence[Tuple[int, int]],
) -> np.ndarray:
    values = np.zeros(len(powers), dtype=float)
    for index, (a, b) in enumerate(powers):
        if (a, b) == (2, 0) or (a, b) == (0, 2):
            values[index] = -2.0
    return values


def _gradient_polynomials_at_origin(
    powers: Sequence[Tuple[int, int]],
) -> np.ndarray:
    gradient = np.zeros((len(powers), 2), dtype=float)
    for index, power in enumerate(powers):
        if power == (1, 0):
            gradient[index, 0] = 1.0
        elif power == (0, 1):
            gradient[index, 1] = 1.0
    return gradient


def _kernel_values(radii: np.ndarray, config: RBFConfig) -> np.ndarray:
    kernel = config.kernel.lower()
    epsilon = config.shape_parameter
    if kernel == "phs":
        if config.order < 3 or config.order % 2 == 0:
            raise ValueError("PHS order must be an odd integer at least three")
        return radii ** config.order
    if epsilon <= 0.0:
        raise ValueError("shape_parameter must be positive")
    scaled2 = (epsilon * radii) ** 2
    if kernel == "gaussian":
        return np.exp(-scaled2)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + scaled2)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + scaled2)
    raise ValueError(f"unsupported RBF kernel: {config.kernel}")


def _negative_laplacian_kernel(radii: np.ndarray, config: RBFConfig) -> np.ndarray:
    """Evaluate ``-Delta phi`` in two dimensions at the supplied radii."""

    kernel = config.kernel.lower()
    epsilon = config.shape_parameter
    if kernel == "phs":
        k = config.order
        return -(k**2) * radii ** (k - 2)
    scaled2 = (epsilon * radii) ** 2
    if kernel == "gaussian":
        return 4.0 * epsilon**2 * (1.0 - scaled2) * np.exp(-scaled2)
    if kernel == "multiquadric":
        base = 1.0 + scaled2
        return -(2.0 * epsilon**2 / np.sqrt(base) - epsilon**4 * radii**2 / base**1.5)
    if kernel == "inverse_multiquadric":
        base = 1.0 + scaled2
        return 2.0 * epsilon**2 / base**1.5 - 3.0 * epsilon**4 * radii**2 / base**2.5
    raise ValueError(f"unsupported RBF kernel: {config.kernel}")


def _kernel_gradient_at_origin(
    scaled_offsets: np.ndarray,
    config: RBFConfig,
) -> np.ndarray:
    """Evaluate the kernel gradient in the evaluation coordinate at the origin."""

    radii = np.linalg.norm(scaled_offsets, axis=1)
    kernel = config.kernel.lower()
    epsilon = config.shape_parameter
    if kernel == "phs":
        factor = np.zeros_like(radii)
        nonzero = radii > 0.0
        factor[nonzero] = (
            config.order * radii[nonzero] ** (config.order - 2)
        )
    else:
        scaled2 = (epsilon * radii) ** 2
        if kernel == "gaussian":
            factor = -2.0 * epsilon**2 * np.exp(-scaled2)
        elif kernel == "multiquadric":
            factor = epsilon**2 / np.sqrt(1.0 + scaled2)
        elif kernel == "inverse_multiquadric":
            factor = -epsilon**2 / (1.0 + scaled2) ** 1.5
        else:
            raise ValueError(f"unsupported RBF kernel: {config.kernel}")
    return factor[:, None] * (-scaled_offsets)


def _negative_hessian_kernel(
    scaled_offsets: np.ndarray,
    config: RBFConfig,
) -> np.ndarray:
    """Return ``(-phi_xx, -phi_xy, -phi_yy)`` at the evaluation origin."""

    radii = np.linalg.norm(scaled_offsets, axis=1)
    x = scaled_offsets[:, 0]
    y = scaled_offsets[:, 1]
    kernel = config.kernel.lower()
    epsilon = config.shape_parameter
    if kernel == "phs":
        a = np.zeros_like(radii)
        b = np.zeros_like(radii)
        nonzero = radii > 0.0
        a[nonzero] = config.order * radii[nonzero] ** (config.order - 2)
        b[nonzero] = (
            config.order
            * (config.order - 2)
            * radii[nonzero] ** (config.order - 4)
        )
    else:
        scaled2 = (epsilon * radii) ** 2
        if kernel == "gaussian":
            exponential = np.exp(-scaled2)
            a = -2.0 * epsilon**2 * exponential
            b = 4.0 * epsilon**4 * exponential
        elif kernel == "multiquadric":
            base = 1.0 + scaled2
            a = epsilon**2 / np.sqrt(base)
            b = -epsilon**4 / base**1.5
        elif kernel == "inverse_multiquadric":
            base = 1.0 + scaled2
            a = -epsilon**2 / base**1.5
            b = 3.0 * epsilon**4 / base**2.5
        else:
            raise ValueError(f"unsupported RBF kernel: {config.kernel}")
    return -np.column_stack((a + b * x**2, b * x * y, a + b * y**2))


def _negative_hessian_polynomials_at_origin(
    powers: Sequence[Tuple[int, int]],
) -> np.ndarray:
    values = np.zeros((len(powers), 3), dtype=float)
    for index, power in enumerate(powers):
        if power == (2, 0):
            values[index, 0] = -2.0
        elif power == (1, 1):
            values[index, 1] = -1.0
        elif power == (0, 2):
            values[index, 2] = -2.0
    return values


def local_weights(
    stencil_nodes: np.ndarray,
    center: np.ndarray,
    rbf: RBFConfig,
    polynomial_degree: int,
    operator: OperatorCoefficients = OperatorCoefficients(),
    compute_condition: bool = True,
) -> Tuple[np.ndarray, float, float]:
    r"""Compute augmented RBF-FD weights for a local elliptic operator.

    The stencil is translated to the center and scaled by its farthest-node
    radius. Second derivatives scale by rho^-2 and first derivatives by
    rho^-1. No regularization or silent least-squares fallback is used.
    """

    stencil_nodes = np.asarray(stencil_nodes, dtype=float)
    center = np.asarray(center, dtype=float)
    if stencil_nodes.ndim != 2 or stencil_nodes.shape[1] != 2:
        raise ValueError("stencil_nodes must have shape (s, 2)")
    offsets = stencil_nodes - center
    rho = float(np.max(np.linalg.norm(offsets, axis=1)))
    if not np.isfinite(rho) or rho <= 0.0:
        raise RBFWeightError("stencil radius is zero or non-finite")
    scaled = offsets / rho
    pairwise = scaled[:, None, :] - scaled[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    powers = monomial_powers(polynomial_degree)
    if len(powers) > stencil_nodes.shape[0]:
        raise ValueError("stencil size must be at least the polynomial basis dimension")
    phi = _kernel_values(distances, rbf)
    polynomials = _polynomial_matrix(scaled, powers)
    augmented = np.block(
        [
            [phi, polynomials],
            [polynomials.T, np.zeros((len(powers), len(powers)))],
        ]
    )
    center_distances = np.linalg.norm(scaled, axis=1)
    kernel_gradient = _kernel_gradient_at_origin(scaled, rbf)
    kernel_hessian = _negative_hessian_kernel(scaled, rbf)
    polynomial_gradient = _gradient_polynomials_at_origin(powers)
    polynomial_hessian = _negative_hessian_polynomials_at_origin(powers)
    rbf_rhs = (
        operator.negative_laplacian
        * _negative_laplacian_kernel(center_distances, rbf)
        / rho**2
        + operator.negative_xx * kernel_hessian[:, 0] / rho**2
        + operator.negative_xy * kernel_hessian[:, 1] / rho**2
        + operator.negative_yy * kernel_hessian[:, 2] / rho**2
        + operator.first_x * kernel_gradient[:, 0] / rho
        + operator.first_y * kernel_gradient[:, 1] / rho
    )
    polynomial_rhs = (
        operator.negative_laplacian
        * _negative_laplacian_polynomials_at_origin(powers)
        / rho**2
        + operator.negative_xx * polynomial_hessian[:, 0] / rho**2
        + operator.negative_xy * polynomial_hessian[:, 1] / rho**2
        + operator.negative_yy * polynomial_hessian[:, 2] / rho**2
        + operator.first_x * polynomial_gradient[:, 0] / rho
        + operator.first_y * polynomial_gradient[:, 1] / rho
    )
    rhs = np.concatenate((rbf_rhs, polynomial_rhs))
    condition = (
        float(np.linalg.cond(augmented))
        if compute_condition
        else float("nan")
    )
    try:
        solution = la.solve(augmented, rhs, assume_a="sym")
    except la.LinAlgError as exc:
        raise RBFWeightError(
            f"local augmented system failed (size={stencil_nodes.shape[0]}, cond={condition:.3e})"
        ) from exc
    weights = solution[: stencil_nodes.shape[0]]
    if not np.all(np.isfinite(weights)):
        raise RBFWeightError("local solve produced non-finite weights")
    return weights, rho, condition


def assemble_on_nodes(
    nodes: np.ndarray,
    boundary: np.ndarray,
    config: DiscretizationConfig,
    coefficient: Optional[object] = None,
) -> RBFSystem:
    """Assemble the sparse full system on supplied, consistently ordered nodes."""

    nodes = np.asarray(nodes, dtype=float)
    boundary = np.asarray(boundary, dtype=bool)
    n_nodes = nodes.shape[0]
    if nodes.shape != (config.n_nodes, 2) or boundary.shape != (config.n_nodes,):
        raise ValueError("nodes/boundary shapes must agree with config.n_nodes")
    if config.stencil_size > n_nodes:
        raise ValueError("stencil size cannot exceed the number of nodes")

    tree = cKDTree(nodes)
    rows: List[int] = []
    columns: List[int] = []
    data: List[float] = []
    radii: List[float] = []
    local_conditions: List[float] = []
    is_tensor_coefficient = isinstance(
        coefficient, (ConstantAnisotropicDiffusion, SpatialAnisotropicDiffusion)
    )
    coefficient_values = (
        np.ones(n_nodes, dtype=float)
        if coefficient is None or is_tensor_coefficient
        else coefficient.values(nodes)
    )
    coefficient_gradients = (
        np.zeros((n_nodes, 2), dtype=float)
        if coefficient is None or is_tensor_coefficient
        else coefficient.gradient(nodes)
    )
    spatial_tensors = (
        coefficient.tensor_values(nodes)
        if isinstance(coefficient, SpatialAnisotropicDiffusion)
        else None
    )
    spatial_divergence = (
        coefficient.tensor_divergence(nodes)
        if isinstance(coefficient, SpatialAnisotropicDiffusion)
        else None
    )

    for row in range(n_nodes):
        if boundary[row]:
            rows.append(row)
            columns.append(row)
            data.append(1.0)
            continue
        _, stencil = tree.query(nodes[row], k=config.stencil_size)
        stencil = np.atleast_1d(stencil).astype(int)
        if isinstance(coefficient, ConstantAnisotropicDiffusion):
            tensor = coefficient.tensor
            operator = OperatorCoefficients(
                negative_laplacian=0.0,
                negative_xx=float(tensor[0, 0]),
                negative_xy=float(2.0 * tensor[0, 1]),
                negative_yy=float(tensor[1, 1]),
            )
        elif isinstance(coefficient, SpatialAnisotropicDiffusion):
            assert spatial_tensors is not None and spatial_divergence is not None
            tensor = spatial_tensors[row]
            operator = OperatorCoefficients(
                negative_laplacian=0.0,
                negative_xx=float(tensor[0, 0]),
                negative_xy=float(2.0 * tensor[0, 1]),
                negative_yy=float(tensor[1, 1]),
                first_x=float(-spatial_divergence[row, 0]),
                first_y=float(-spatial_divergence[row, 1]),
            )
        else:
            operator = OperatorCoefficients(
                negative_laplacian=float(coefficient_values[row]),
                first_x=float(-coefficient_gradients[row, 0]),
                first_y=float(-coefficient_gradients[row, 1]),
            )
        weights, radius, condition = local_weights(
            nodes[stencil],
            nodes[row],
            config.rbf,
            config.polynomial_degree,
            operator=operator,
            compute_condition=config.compute_local_condition_numbers,
        )
        rows.extend([row] * stencil.size)
        columns.extend(stencil.tolist())
        data.extend(weights.tolist())
        radii.append(radius)
        if config.compute_local_condition_numbers:
            local_conditions.append(condition)

    matrix = sp.csr_matrix((data, (rows, columns)), shape=(n_nodes, n_nodes))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix_norm = float(sp.linalg.norm(matrix))
    symmetry = float(sp.linalg.norm(matrix - matrix.T) / matrix_norm)
    diagnostics: Dict[str, float] = {
        "n_nodes": float(n_nodes),
        "n_interior": float(np.count_nonzero(~boundary)),
        "n_boundary": float(np.count_nonzero(boundary)),
        "nnz": float(matrix.nnz),
        "density": float(matrix.nnz / (n_nodes * n_nodes)),
        "symmetry_measure": symmetry,
        "median_stencil_radius": float(np.median(radii)) if radii else 0.0,
        "max_stencil_radius": float(np.max(radii)) if radii else 0.0,
        "median_local_augmented_condition": (
            float(np.median(local_conditions)) if local_conditions else float("nan")
        ),
        "max_local_augmented_condition": (
            float(np.max(local_conditions)) if local_conditions else float("nan")
        ),
        **node_irregularity_metrics(nodes),
        "coefficient_beta": (
            0.0 if coefficient is None else float(getattr(coefficient, "beta", float("nan")))
        ),
        "coefficient_contrast": (
            1.0 if coefficient is None else float(coefficient.contrast)
        ),
    }
    return RBFSystem(
        matrix,
        nodes.copy(),
        boundary.copy(),
        ~boundary,
        diagnostics,
        config,
        coefficient,
    )


def assemble_system(
    config: DiscretizationConfig,
    coefficient: Optional[object] = None,
) -> RBFSystem:
    """Generate nodes and assemble constant- or variable-coefficient rows."""

    nodes, boundary = generate_nodes(config)
    return assemble_on_nodes(nodes, boundary, config, coefficient)
