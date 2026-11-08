"""Oracle-only NeurKItt/GCRO-DR benchmark utilities."""

from .darcy import build_darcy_system, sample_log_permeability
from .gcrodr import (
    GmresResult,
    build_augmentation,
    restarted_projected_gmres,
)
from .recycle import (
    RealInvariantSubspace,
    apply_real_operator_to_complex,
    apply_to_columns,
    invariant_subspace_defect,
    principal_angles_degrees,
    real_invariant_subspace,
)

__all__ = [
    "GmresResult",
    "RealInvariantSubspace",
    "apply_real_operator_to_complex",
    "apply_to_columns",
    "build_augmentation",
    "build_darcy_system",
    "invariant_subspace_defect",
    "principal_angles_degrees",
    "real_invariant_subspace",
    "restarted_projected_gmres",
    "sample_log_permeability",
]
