"""Small learned baselines; no learned-subspace model is implemented."""

from .direct_mlp import (
    DirectMLPConfig,
    DirectMLPPredictor,
    TrainingResult,
    correction_metrics,
    make_direct_preconditioner,
    train_direct_mlp,
)

__all__ = [
    "DirectMLPConfig",
    "DirectMLPPredictor",
    "TrainingResult",
    "correction_metrics",
    "make_direct_preconditioner",
    "train_direct_mlp",
]

