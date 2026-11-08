"""Closed-loop controls for a learned missing-inverse-action predictor."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..preconditioners import InverseAction


CorrectionPredictor = Callable[[np.ndarray], np.ndarray]


def make_stabilized_direct_preconditioner(
    base: InverseAction,
    predict_correction: CorrectionPredictor,
    alpha: float,
    norm_limit_ratio: Optional[float] = None,
    name: str = "direct_learned",
) -> Tuple[InverseAction, Dict[str, float]]:
    r"""Build ``P(v)=M^-1 v + alpha d_theta(v)`` with optional norm limiting.

    The optional safeguard clips only the *applied learned term* so that
    ``||alpha d_theta(v)|| <= gamma ||M^-1 v||``.  It does not alter the
    supervised target, which remains exactly ``A^-1 v-M^-1 v``.  Returning the
    mutable audit dictionary makes clipping and prediction overhead explicit.
    """

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if norm_limit_ratio is not None and (
        not np.isfinite(norm_limit_ratio) or norm_limit_ratio <= 0.0
    ):
        raise ValueError("norm_limit_ratio must be finite and positive")

    audit: Dict[str, float] = {
        "alpha": float(alpha),
        "norm_limit_ratio": (
            float(norm_limit_ratio) if norm_limit_ratio is not None else float("nan")
        ),
        "prediction_seconds": 0.0,
        "applications": 0.0,
        "clipped_applications": 0.0,
        "raw_prediction_norm_sum": 0.0,
        "applied_prediction_norm_sum": 0.0,
    }

    def action(vector: np.ndarray) -> np.ndarray:
        classical = base.apply(vector)
        start = perf_counter()
        raw = np.asarray(predict_correction(vector), dtype=float)
        audit["prediction_seconds"] += perf_counter() - start
        if raw.shape != vector.shape or not np.all(np.isfinite(raw)):
            raise ValueError("learned correction is non-finite or has the wrong shape")
        applied = float(alpha) * raw
        audit["applications"] += 1.0
        audit["raw_prediction_norm_sum"] += float(np.linalg.norm(raw))
        if norm_limit_ratio is not None:
            limit = float(norm_limit_ratio) * float(np.linalg.norm(classical))
            applied_norm = float(np.linalg.norm(applied))
            if applied_norm > limit and applied_norm > 0.0:
                applied = applied * (limit / applied_norm)
                audit["clipped_applications"] += 1.0
        audit["applied_prediction_norm_sum"] += float(np.linalg.norm(applied))
        return classical + applied

    result = InverseAction(
        f"{base.name}+{name}_alpha{alpha:g}",
        action,
        setup_seconds=base.setup_seconds,
        is_linear=False,
        diagnostics={
            "learned_strategy": "direct_missing_action",
            "target_definition": "A^-1 v - M^-1 v",
            "alpha": float(alpha),
            "norm_limit_ratio": norm_limit_ratio,
            "audit": audit,
        },
    )
    return result, audit

