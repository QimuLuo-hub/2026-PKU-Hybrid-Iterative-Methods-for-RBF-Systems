"""Configuration objects used by discretization, solvers, and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class RBFConfig:
    """Radial kernel configuration in stencil-normalized coordinates.

    Supported kernels are ``phs``, ``gaussian``, ``multiquadric``, and
    ``inverse_multiquadric``. ``order`` is used only by ``phs`` and must be an
    odd integer at least three. ``shape_parameter`` is used by the infinitely
    smooth kernels.
    """

    kernel: str = "phs"
    order: int = 3
    shape_parameter: float = 1.0


@dataclass(frozen=True)
class DiscretizationConfig:
    """Configuration of a square-domain RBF-FD discretization."""

    n_nodes: int = 324
    stencil_size: int = 25
    polynomial_degree: int = 2
    node_distribution: str = "jittered_grid"
    jitter_fraction: float = 0.15
    warp_strength: float = 0.0
    density_exponent: float = 1.0
    seed: int = 7
    rbf: RBFConfig = field(default_factory=RBFConfig)
    compute_local_condition_numbers: bool = True


@dataclass(frozen=True)
class ILUConfig:
    """Sparse incomplete-LU controls passed to :func:`scipy.sparse.linalg.spilu`."""

    drop_tolerance: float = 1.0e-3
    fill_factor: float = 5.0
    diagonal_pivot_threshold: float = 0.0
    permutation: str = "COLAMD"


@dataclass(frozen=True)
class SolverConfig:
    """Restarted right-preconditioned GMRES controls."""

    relative_tolerance: float = 1.0e-9
    absolute_tolerance: float = 0.0
    restart: int = 40
    max_iterations: int = 500
    breakdown_tolerance: float = 1.0e-14


def to_dict(config: Any) -> Dict[str, Any]:
    """Recursively convert a dataclass configuration to JSON-ready values."""

    return asdict(config)
