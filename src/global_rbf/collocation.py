"""Unsymmetric global Kansa collocation for the 2D Poisson equation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.linalg as la

from .kernels import gradients, negative_laplacian, pairwise_distances, values


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def forcing(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi**2 * exact_solution(x)


def diffusion_coefficient(x: np.ndarray, beta: float) -> np.ndarray:
    """Smooth positive coefficient ``a_beta(x,y)=exp(beta*x)``."""

    return np.exp(float(beta) * np.asarray(x)[:, 0])


def variable_diffusion_forcing(
    x: np.ndarray,
    beta: float,
    *,
    kx: int = 1,
    ky: int = 1,
) -> np.ndarray:
    """Manufactured forcing for ``-div(a_beta grad u)=f``.

    The exact mode is ``sin(kx*pi*x) sin(ky*pi*y)``.
    """

    points = np.asarray(x)
    xx, yy = points[:, 0], points[:, 1]
    a = diffusion_coefficient(points, beta)
    u = np.sin(kx * np.pi * xx) * np.sin(ky * np.pi * yy)
    ux = kx * np.pi * np.cos(kx * np.pi * xx) * np.sin(ky * np.pi * yy)
    return a * ((kx**2 + ky**2) * np.pi**2 * u - float(beta) * ux)


def polynomial_matrix(x: np.ndarray) -> np.ndarray:
    xx, yy = x[:, 0], x[:, 1]
    return np.column_stack([np.ones(len(x)), xx, yy, xx**2, xx * yy, yy**2])


def negative_laplacian_polynomials(x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), 6), dtype=float)
    out[:, 3] = -2.0
    out[:, 5] = -2.0
    return out


@dataclass
class GlobalRBFSystem:
    A: np.ndarray
    b: np.ndarray
    nodes: np.ndarray
    boundary: np.ndarray
    kernel: str
    epsilon: float | None
    polynomial_degree: int | None
    assembly_seconds: float

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def dimension(self) -> int:
        return self.A.shape[0]

    @property
    def coefficient_count(self) -> int:
        return self.node_count

    def evaluate(self, solution_vector: np.ndarray, points: np.ndarray) -> np.ndarray:
        c = solution_vector[: self.node_count]
        r = pairwise_distances(np.asarray(points), self.nodes)
        out = values(r, self.kernel, self.epsilon) @ c
        if self.polynomial_degree is not None:
            out = out + polynomial_matrix(np.asarray(points)) @ solution_vector[self.node_count :]
        return out

    def solve_reference(self) -> tuple[np.ndarray, float]:
        start = perf_counter()
        x = la.solve(self.A, self.b, assume_a="gen", check_finite=False)
        return x, perf_counter() - start

    def error_metrics(self, x: np.ndarray, evaluation_side: int = 41) -> dict[str, float]:
        nodal_error = self.evaluate(x, self.nodes) - exact_solution(self.nodes)
        grid = np.linspace(0.0, 1.0, evaluation_side)
        xx, yy = np.meshgrid(grid, grid, indexing="xy")
        points = np.column_stack([xx.ravel(), yy.ravel()])
        grid_error = self.evaluate(x, points) - exact_solution(points)
        residual = self.b - self.A @ x
        denom = np.linalg.norm(self.b)
        backward_denom = (
            np.linalg.norm(self.A, ord=np.inf) * np.linalg.norm(x, ord=np.inf)
            + np.linalg.norm(self.b, ord=np.inf)
        )
        return {
            "nodal_rel_l2": float(np.linalg.norm(nodal_error) / np.linalg.norm(exact_solution(self.nodes))),
            "nodal_max": float(np.max(np.abs(nodal_error))),
            "grid_rel_l2": float(np.linalg.norm(grid_error) / np.linalg.norm(exact_solution(points))),
            "grid_max": float(np.max(np.abs(grid_error))),
            "relative_residual": float(np.linalg.norm(residual) / denom),
            "normwise_backward_error": float(np.linalg.norm(residual, ord=np.inf) / backward_denom),
            "boundary_max": float(np.max(np.abs(self.evaluate(x, self.nodes[self.boundary])))),
        }


def assemble_poisson_system(
    nodes: np.ndarray,
    boundary: np.ndarray,
    *,
    kernel: str = "gaussian",
    epsilon: float | None = 3.0,
    phs_polynomial_degree: int = 2,
) -> GlobalRBFSystem:
    """Assemble the dense unsymmetric collocation matrix.

    For PHS3, the experiment's degree-2 polynomial augmentation and moment
    constraints are included, yielding an ``(N+6) x (N+6)`` saddle system.
    """

    start = perf_counter()
    nodes = np.asarray(nodes, dtype=float)
    boundary = np.asarray(boundary, dtype=bool)
    r = pairwise_distances(nodes, nodes)
    phi = values(r, kernel, epsilon)
    lop = negative_laplacian(r, kernel, epsilon)
    top = np.where(boundary[:, None], phi, lop)
    rhs = np.where(boundary, exact_solution(nodes), forcing(nodes))

    poly_degree: int | None = None
    if kernel == "phs3":
        if phs_polynomial_degree != 2:
            raise ValueError("PHS3 currently requires degree-2 polynomial augmentation")
        poly_degree = 2
        p = polynomial_matrix(nodes)
        lp = negative_laplacian_polynomials(nodes)
        top_p = np.where(boundary[:, None], p, lp)
        A = np.block([[top, top_p], [p.T, np.zeros((p.shape[1], p.shape[1]))]])
        b = np.concatenate([rhs, np.zeros(p.shape[1])])
    else:
        A = top
        b = rhs

    return GlobalRBFSystem(
        A=np.asarray(A, order="C"),
        b=np.asarray(b),
        nodes=nodes,
        boundary=boundary,
        kernel=kernel,
        epsilon=epsilon,
        polynomial_degree=poly_degree,
        assembly_seconds=perf_counter() - start,
    )


def assemble_variable_diffusion_system(
    nodes: np.ndarray,
    boundary: np.ndarray,
    *,
    beta: float,
    kernel: str = "gaussian",
    epsilon: float | None = 3.0,
) -> GlobalRBFSystem:
    """Assemble ``-div(exp(beta*x) grad u)=f`` with Dirichlet rows.

    The coefficient is smooth and strictly positive.  This path deliberately
    varies the PDE operator rather than merely rescaling every interior row by
    one constant.  Polynomially augmented PHS systems are excluded here
    because Stage 2 compares fixed-size Gaussian coefficient spaces.
    """

    if kernel == "phs3":
        raise ValueError("variable diffusion currently supports unaugmented kernels")
    start = perf_counter()
    nodes = np.asarray(nodes, dtype=float)
    boundary = np.asarray(boundary, dtype=bool)
    delta = nodes[:, None, :] - nodes[None, :, :]
    r = np.linalg.norm(delta, axis=2)
    phi = values(r, kernel, epsilon)
    minus_laplacian = negative_laplacian(r, kernel, epsilon)
    grad_phi = gradients(delta, kernel, epsilon)
    a = diffusion_coefficient(nodes, beta)
    grad_a = np.column_stack([float(beta) * a, np.zeros_like(a)])
    operator = a[:, None] * minus_laplacian - np.einsum("id,ijd->ij", grad_a, grad_phi)
    A = np.where(boundary[:, None], phi, operator)
    rhs = np.where(boundary, exact_solution(nodes), variable_diffusion_forcing(nodes, beta))
    return GlobalRBFSystem(
        A=np.asarray(A, order="C"),
        b=np.asarray(rhs),
        nodes=nodes,
        boundary=boundary,
        kernel=kernel,
        epsilon=epsilon,
        polynomial_degree=None,
        assembly_seconds=perf_counter() - start,
    )


def matrix_diagnostics(A: np.ndarray, *, exact_svd: bool = True) -> dict[str, float]:
    fro = np.linalg.norm(A, ord="fro")
    symmetry = np.linalg.norm(A - A.T, ord="fro") / fro
    result = {
        "symmetry_defect": float(symmetry),
        "density": float(np.count_nonzero(A) / A.size),
        "matrix_memory_bytes": int(A.nbytes),
        "matrix_norm_2": float("nan"),
    }
    if exact_svd:
        singular_values = la.svdvals(A, check_finite=False)
        result.update(
            {
                "sigma_max": float(singular_values[0]),
                "sigma_min": float(singular_values[-1]),
                "condition_2": float(singular_values[0] / singular_values[-1]),
                "numerical_rank": int(np.count_nonzero(singular_values > singular_values[0] * np.finfo(float).eps * max(A.shape))),
            }
        )
        result["matrix_norm_2"] = float(singular_values[0])
    return result
