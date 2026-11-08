"""Dense global RBF collocation research utilities.

This package is intentionally independent of the sparse RBF-FD prototype in
``lcp_rbffd``.  Unknowns are global RBF coefficients (plus polynomial
coefficients for conditionally positive-definite kernels).
"""

from .collocation import (
    GlobalRBFSystem,
    assemble_poisson_system,
    assemble_variable_diffusion_system,
)
from .nodes import square_nodes
from .transfer import compare_subspaces, grassmann_geodesic, smallest_right_singular_spaces

__all__ = [
    "GlobalRBFSystem",
    "assemble_poisson_system",
    "assemble_variable_diffusion_system",
    "compare_subspaces",
    "grassmann_geodesic",
    "smallest_right_singular_spaces",
    "square_nodes",
]
