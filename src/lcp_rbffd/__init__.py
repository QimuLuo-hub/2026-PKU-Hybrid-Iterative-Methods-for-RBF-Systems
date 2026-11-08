"""Numerical prototype for learned complementary RBF-FD preconditioning.

The package intentionally implements only research Stages 1-5.  POD is a
diagnostic/reference construction; there are no learned models in this package.
"""

from .config import DiscretizationConfig, ILUConfig, RBFConfig, SolverConfig
from .pde import ManufacturedSolution
from .rbffd import RBFSystem, assemble_system

__all__ = [
    "DiscretizationConfig",
    "ILUConfig",
    "ManufacturedSolution",
    "RBFConfig",
    "RBFSystem",
    "SolverConfig",
    "assemble_system",
]

