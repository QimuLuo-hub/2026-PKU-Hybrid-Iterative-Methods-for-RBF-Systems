"""Manufactured solutions and coefficients for elliptic unit-square problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


Mode = Tuple[int, int, float]


@dataclass(frozen=True)
class ManufacturedSolution:
    r"""Finite sine series with known forcing for ``-Delta u = f``.

    A mode ``(kx, ky, amplitude)`` represents
    ``amplitude*sin(kx*pi*x)*sin(ky*pi*y)``.  It is smooth and has homogeneous
    Dirichlet data, while the forcing is obtained analytically by multiplying
    each mode by ``pi**2*(kx**2 + ky**2)``.
    """

    modes: Tuple[Mode, ...] = ((1, 1, 1.0),)

    def values(self, nodes: np.ndarray) -> np.ndarray:
        x, y = nodes[:, 0], nodes[:, 1]
        result = np.zeros(nodes.shape[0], dtype=float)
        for kx, ky, amplitude in self.modes:
            result += amplitude * np.sin(kx * np.pi * x) * np.sin(ky * np.pi * y)
        return result

    def forcing(self, nodes: np.ndarray) -> np.ndarray:
        x, y = nodes[:, 0], nodes[:, 1]
        result = np.zeros(nodes.shape[0], dtype=float)
        for kx, ky, amplitude in self.modes:
            eigenvalue = np.pi**2 * (kx**2 + ky**2)
            result += (
                eigenvalue
                * amplitude
                * np.sin(kx * np.pi * x)
                * np.sin(ky * np.pi * y)
            )
        return result

    def gradient(self, nodes: np.ndarray) -> np.ndarray:
        """Return the analytic gradient of the sine-series solution."""

        x, y = nodes[:, 0], nodes[:, 1]
        gradient = np.zeros((nodes.shape[0], 2), dtype=float)
        for kx, ky, amplitude in self.modes:
            gradient[:, 0] += (
                amplitude
                * kx
                * np.pi
                * np.cos(kx * np.pi * x)
                * np.sin(ky * np.pi * y)
            )
            gradient[:, 1] += (
                amplitude
                * ky
                * np.pi
                * np.sin(kx * np.pi * x)
                * np.cos(ky * np.pi * y)
            )
        return gradient

    def hessian(self, nodes: np.ndarray) -> np.ndarray:
        """Return the analytic Hessian with component order ``(xx, xy, yy)``."""

        x, y = nodes[:, 0], nodes[:, 1]
        hessian = np.zeros((nodes.shape[0], 3), dtype=float)
        for kx, ky, amplitude in self.modes:
            sx = np.sin(kx * np.pi * x)
            sy = np.sin(ky * np.pi * y)
            cx = np.cos(kx * np.pi * x)
            cy = np.cos(ky * np.pi * y)
            hessian[:, 0] -= amplitude * (kx * np.pi) ** 2 * sx * sy
            hessian[:, 1] += amplitude * kx * ky * np.pi**2 * cx * cy
            hessian[:, 2] -= amplitude * (ky * np.pi) ** 2 * sx * sy
        return hessian

    def right_hand_side(self, nodes: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """Return forcing at interior rows and exact Dirichlet data at boundary rows."""

        rhs = self.forcing(nodes)
        rhs[boundary] = self.values(nodes)[boundary]
        return rhs

    def variable_coefficient_forcing(
        self,
        nodes: np.ndarray,
        coefficient: "ExponentialCoefficient",
    ) -> np.ndarray:
        r"""Return -div(a grad u) = a(-Delta u) - grad(a) dot grad(u)."""

        a_values = coefficient.values(nodes)
        a_gradient = coefficient.gradient(nodes)
        return a_values * self.forcing(nodes) - np.sum(
            a_gradient * self.gradient(nodes), axis=1
        )

    def variable_coefficient_right_hand_side(
        self,
        nodes: np.ndarray,
        boundary: np.ndarray,
        coefficient: "ExponentialCoefficient",
    ) -> np.ndarray:
        """Return variable-coefficient forcing and exact Dirichlet rows."""

        rhs = self.variable_coefficient_forcing(nodes, coefficient)
        rhs[boundary] = self.values(nodes)[boundary]
        return rhs

    def anisotropic_right_hand_side(
        self,
        nodes: np.ndarray,
        boundary: np.ndarray,
        coefficient: "ConstantAnisotropicDiffusion",
    ) -> np.ndarray:
        r"""Return forcing for ``-div(K grad u)`` and exact Dirichlet rows."""

        tensor = coefficient.tensor
        hessian = self.hessian(nodes)
        rhs = -(
            tensor[0, 0] * hessian[:, 0]
            + 2.0 * tensor[0, 1] * hessian[:, 1]
            + tensor[1, 1] * hessian[:, 2]
        )
        rhs[boundary] = self.values(nodes)[boundary]
        return rhs

    def spatial_anisotropic_right_hand_side(
        self,
        nodes: np.ndarray,
        boundary: np.ndarray,
        coefficient: "SpatialAnisotropicDiffusion",
    ) -> np.ndarray:
        r"""Return the exact forcing for ``-div(K(x) grad u)``.

        The product rule gives ``-K:Hess(u) - div(K) dot grad(u)``.  This
        method is deliberately analytic so that Stage 19 can check the
        discretization independently of the Krylov experiments.
        """

        tensor = coefficient.tensor_values(nodes)
        divergence = coefficient.tensor_divergence(nodes)
        hessian = self.hessian(nodes)
        rhs = -(
            tensor[:, 0, 0] * hessian[:, 0]
            + 2.0 * tensor[:, 0, 1] * hessian[:, 1]
            + tensor[:, 1, 1] * hessian[:, 2]
            + np.sum(divergence * self.gradient(nodes), axis=1)
        )
        rhs[boundary] = self.values(nodes)[boundary]
        return rhs


@dataclass(frozen=True)
class ExponentialCoefficient:
    r"""Smooth positive field a=exp(beta sin(2 pi f x) sin(2 pi f y)).

    Its exact continuous contrast is exp(2 |beta|). beta=0 recovers the
    constant-coefficient Poisson problem without a separate code path.
    """

    beta: float = 0.0
    frequency: int = 1

    def values(self, nodes: np.ndarray) -> np.ndarray:
        x, y = nodes[:, 0], nodes[:, 1]
        phase = np.sin(2.0 * np.pi * self.frequency * x) * np.sin(
            2.0 * np.pi * self.frequency * y
        )
        return np.exp(self.beta * phase)

    def gradient(self, nodes: np.ndarray) -> np.ndarray:
        x, y = nodes[:, 0], nodes[:, 1]
        frequency = float(self.frequency)
        values = self.values(nodes)
        scale = self.beta * 2.0 * np.pi * frequency
        gradient = np.empty((nodes.shape[0], 2), dtype=float)
        gradient[:, 0] = (
            values
            * scale
            * np.cos(2.0 * np.pi * frequency * x)
            * np.sin(2.0 * np.pi * frequency * y)
        )
        gradient[:, 1] = (
            values
            * scale
            * np.sin(2.0 * np.pi * frequency * x)
            * np.cos(2.0 * np.pi * frequency * y)
        )
        return gradient

    @property
    def contrast(self) -> float:
        return float(np.exp(2.0 * abs(self.beta)))


@dataclass(frozen=True)
class LocalizedCoefficient:
    r"""Smooth positive Gaussian inclusion with prescribed peak contrast."""

    peak_contrast: float = 10.0
    center: Tuple[float, float] = (0.5, 0.5)
    width: float = 0.15

    def __post_init__(self) -> None:
        if self.peak_contrast < 1.0:
            raise ValueError("peak_contrast must be at least one")
        if self.width <= 0.0:
            raise ValueError("width must be positive")

    def values(self, nodes: np.ndarray) -> np.ndarray:
        delta = nodes - np.asarray(self.center, dtype=float)
        bump = np.exp(-np.sum(delta**2, axis=1) / (2.0 * self.width**2))
        return 1.0 + (self.peak_contrast - 1.0) * bump

    def gradient(self, nodes: np.ndarray) -> np.ndarray:
        delta = nodes - np.asarray(self.center, dtype=float)
        bump = np.exp(-np.sum(delta**2, axis=1) / (2.0 * self.width**2))
        return -(
            (self.peak_contrast - 1.0)
            * bump[:, None]
            * delta
            / self.width**2
        )

    @property
    def beta(self) -> float:
        """Compatibility metadata; this field is not exponential."""

        return float("nan")

    @property
    def contrast(self) -> float:
        return float(self.peak_contrast)


@dataclass(frozen=True)
class ConstantAnisotropicDiffusion:
    r"""Constant SPD tensor with eigenvalues ``1`` and ``anisotropy_ratio``."""

    anisotropy_ratio: float = 10.0
    angle_degrees: float = 30.0

    def __post_init__(self) -> None:
        if self.anisotropy_ratio < 1.0:
            raise ValueError("anisotropy_ratio must be at least one")

    @property
    def tensor(self) -> np.ndarray:
        angle = np.deg2rad(self.angle_degrees)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return rotation @ np.diag([1.0, self.anisotropy_ratio]) @ rotation.T

    @property
    def contrast(self) -> float:
        return float(self.anisotropy_ratio)


@dataclass(frozen=True)
class SpatialAnisotropicDiffusion:
    r"""Smooth SPD tensor ``Q(theta) diag(R,1) Q(theta)^T``.

    ``log(R)`` and ``theta`` use precisely the seven-parameter Stage-19A
    family.  Angle parameters are stored in degrees for readable experiment
    metadata and converted to radians before differentiation.
    """

    a0: float
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    theta0_degrees: float = 30.0
    b1_degrees: float = 0.0
    b2_degrees: float = 0.0

    def __post_init__(self) -> None:
        # This conservative analytic bound guarantees R(x)>=1 everywhere,
        # hence eigenvalues {R(x),1} and uniform positive definiteness.
        lower_log_ratio = self.a0 - abs(self.a1) - abs(self.a2) - abs(self.a3)
        if lower_log_ratio < -1.0e-12:
            raise ValueError("coefficient range does not guarantee R(x) >= 1")

    def log_ratio(self, nodes: np.ndarray) -> np.ndarray:
        x, y = np.asarray(nodes, dtype=float).T
        return (
            self.a0
            + self.a1 * np.sin(np.pi * x)
            + self.a2 * np.cos(np.pi * y)
            + self.a3 * np.sin(np.pi * x) * np.cos(np.pi * y)
        )

    def ratio_values(self, nodes: np.ndarray) -> np.ndarray:
        return np.exp(self.log_ratio(nodes))

    def angle_radians(self, nodes: np.ndarray) -> np.ndarray:
        x, y = np.asarray(nodes, dtype=float).T
        degrees = (
            self.theta0_degrees
            + self.b1_degrees * np.sin(np.pi * x)
            + self.b2_degrees * np.cos(np.pi * y)
        )
        return np.deg2rad(degrees)

    def log_ratio_gradient(self, nodes: np.ndarray) -> np.ndarray:
        x, y = np.asarray(nodes, dtype=float).T
        gradient = np.empty((nodes.shape[0], 2), dtype=float)
        gradient[:, 0] = np.pi * np.cos(np.pi * x) * (
            self.a1 + self.a3 * np.cos(np.pi * y)
        )
        gradient[:, 1] = -np.pi * np.sin(np.pi * y) * (
            self.a2 + self.a3 * np.sin(np.pi * x)
        )
        return gradient

    def angle_gradient(self, nodes: np.ndarray) -> np.ndarray:
        x, y = np.asarray(nodes, dtype=float).T
        gradient = np.empty((nodes.shape[0], 2), dtype=float)
        gradient[:, 0] = np.deg2rad(
            np.pi * self.b1_degrees * np.cos(np.pi * x)
        )
        gradient[:, 1] = np.deg2rad(
            -np.pi * self.b2_degrees * np.sin(np.pi * y)
        )
        return gradient

    def tensor_values(self, nodes: np.ndarray) -> np.ndarray:
        ratio = self.ratio_values(nodes)
        angle = self.angle_radians(nodes)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        tensor = np.empty((nodes.shape[0], 2, 2), dtype=float)
        tensor[:, 0, 0] = 1.0 + (ratio - 1.0) * cosine**2
        tensor[:, 0, 1] = (ratio - 1.0) * cosine * sine
        tensor[:, 1, 0] = tensor[:, 0, 1]
        tensor[:, 1, 1] = 1.0 + (ratio - 1.0) * sine**2
        return tensor

    def tensor_derivatives(self, nodes: np.ndarray) -> np.ndarray:
        """Return ``dK/dx_d`` with shape ``(N, 2, 2, 2)``."""

        ratio = self.ratio_values(nodes)
        ratio_gradient = ratio[:, None] * self.log_ratio_gradient(nodes)
        angle = self.angle_radians(nodes)
        angle_gradient = self.angle_gradient(nodes)
        n = np.column_stack((np.cos(angle), np.sin(angle)))
        m = np.column_stack((-np.sin(angle), np.cos(angle)))
        nn = np.einsum("ni,nj->nij", n, n)
        rotation_derivative = np.einsum("ni,nj->nij", m, n) + np.einsum(
            "ni,nj->nij", n, m
        )
        derivatives = np.empty((nodes.shape[0], 2, 2, 2), dtype=float)
        for direction in range(2):
            derivatives[:, direction, :, :] = (
                ratio_gradient[:, direction, None, None] * nn
                + (ratio - 1.0)[:, None, None]
                * angle_gradient[:, direction, None, None]
                * rotation_derivative
            )
        return derivatives

    def tensor_divergence(self, nodes: np.ndarray) -> np.ndarray:
        derivatives = self.tensor_derivatives(nodes)
        divergence = np.empty((nodes.shape[0], 2), dtype=float)
        divergence[:, 0] = derivatives[:, 0, 0, 0] + derivatives[:, 1, 0, 1]
        divergence[:, 1] = derivatives[:, 0, 1, 0] + derivatives[:, 1, 1, 1]
        return divergence

    def sampled_ratio_bounds(self, nodes: np.ndarray) -> Tuple[float, float]:
        ratios = self.ratio_values(nodes)
        return float(np.min(ratios)), float(np.max(ratios))

    @property
    def contrast(self) -> float:
        """Conservative upper bound on the global tensor contrast."""

        return float(np.exp(self.a0 + abs(self.a1) + abs(self.a2) + abs(self.a3)))


def random_smooth_solution(seed: int, max_mode: int = 4) -> ManufacturedSolution:
    """Construct a deterministic random smooth sine-series solution."""

    if max_mode < 1:
        raise ValueError("max_mode must be positive")
    rng = np.random.default_rng(seed)
    modes = []
    for kx in range(1, max_mode + 1):
        for ky in range(1, max_mode + 1):
            scale = 1.0 / (kx**2 + ky**2) ** 1.5
            modes.append((kx, ky, float(rng.normal() * scale)))
    norm = np.sqrt(sum(amplitude**2 for _, _, amplitude in modes))
    normalized = tuple((kx, ky, amplitude / norm) for kx, ky, amplitude in modes)
    return ManufacturedSolution(normalized)


def combine_modes(modes: Iterable[Mode]) -> ManufacturedSolution:
    """Build a manufactured solution from an arbitrary mode iterable."""

    return ManufacturedSolution(tuple(modes))
