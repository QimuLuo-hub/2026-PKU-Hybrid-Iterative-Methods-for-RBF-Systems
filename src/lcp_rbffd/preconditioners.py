"""Modular classical inverse-action preconditioners for sparse RBF-FD matrices."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import DiscretizationConfig, ILUConfig
from .rbffd import RBFSystem, assemble_on_nodes


class PreconditionerFailure(RuntimeError):
    """Raised when setup or application of a preconditioner fails."""


class InverseAction:
    r"""A timed linear action approximating ``A^{-1} r``.

    Instances are callable and expose setup/application timing without changing
    the mathematical action.  ``is_linear`` is recorded because later learned
    stages may require FGMRES; every Stage 1-5 implementation here is fixed and
    linear.
    """

    def __init__(
        self,
        name: str,
        action: Callable[[np.ndarray], np.ndarray],
        setup_seconds: float = 0.0,
        is_linear: bool = True,
        diagnostics: Optional[Dict[str, Any]] = None,
        transpose_action: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.name = name
        self._action = action
        self.setup_seconds = float(setup_seconds)
        self.is_linear = is_linear
        self.apply_seconds = 0.0
        self.apply_count = 0
        self.diagnostics = diagnostics or {}
        self._transpose_action = transpose_action

    def apply(self, residual: np.ndarray) -> np.ndarray:
        start = perf_counter()
        try:
            output = np.asarray(self._action(np.asarray(residual, dtype=float)), dtype=float)
        except Exception as exc:  # record context and preserve the original cause
            raise PreconditionerFailure(f"{self.name} application failed: {exc}") from exc
        self.apply_seconds += perf_counter() - start
        self.apply_count += 1
        if output.shape != residual.shape or not np.all(np.isfinite(output)):
            raise PreconditionerFailure(f"{self.name} returned invalid output")
        return output

    __call__ = apply

    def apply_transpose(self, residual: np.ndarray) -> np.ndarray:
        r"""Apply the transpose inverse action when it is available.

        This is used only for matrix-free adjoint diagnostics.  It does not
        alter the forward preconditioner used by GMRES/FGMRES.
        """

        if self._transpose_action is None:
            raise PreconditionerFailure(
                f"{self.name} does not expose a transpose inverse action"
            )
        try:
            output = np.asarray(
                self._transpose_action(np.asarray(residual, dtype=float)),
                dtype=float,
            )
        except Exception as exc:
            raise PreconditionerFailure(
                f"{self.name} transpose application failed: {exc}"
            ) from exc
        if output.shape != residual.shape or not np.all(np.isfinite(output)):
            raise PreconditionerFailure(
                f"{self.name} transpose action returned invalid output"
            )
        return output

    def reset_timing(self) -> None:
        self.apply_seconds = 0.0
        self.apply_count = 0


def identity_preconditioner() -> InverseAction:
    """Return the identity inverse action used for unpreconditioned GMRES."""

    return InverseAction("none", lambda residual: residual.copy())


def ilu_preconditioner(
    matrix: sp.spmatrix,
    config: ILUConfig,
    name: str = "ilu_A",
) -> InverseAction:
    """Build an incomplete-LU inverse action without hiding factorization failure."""

    start = perf_counter()
    try:
        factor = spla.spilu(
            matrix.tocsc(),
            drop_tol=config.drop_tolerance,
            fill_factor=config.fill_factor,
            diag_pivot_thresh=config.diagonal_pivot_threshold,
            permc_spec=config.permutation,
        )
    except Exception as exc:
        raise PreconditionerFailure(f"{name} setup failed: {exc}") from exc
    setup_seconds = perf_counter() - start
    factor_memory_bytes = int(
        factor.L.data.nbytes
        + factor.L.indices.nbytes
        + factor.L.indptr.nbytes
        + factor.U.data.nbytes
        + factor.U.indices.nbytes
        + factor.U.indptr.nbytes
    )
    diagnostics = {
        "factor_l_nnz": int(factor.L.nnz),
        "factor_u_nnz": int(factor.U.nnz),
        "factor_nnz": int(factor.L.nnz + factor.U.nnz),
        "factor_memory_bytes": factor_memory_bytes,
        "factor_fill_ratio": float(
            (factor.L.nnz + factor.U.nnz) / matrix.nnz
        ),
        "drop_tolerance": float(config.drop_tolerance),
        "fill_factor": float(config.fill_factor),
    }
    return InverseAction(
        name,
        factor.solve,
        setup_seconds=setup_seconds,
        diagnostics=diagnostics,
        transpose_action=lambda residual: factor.solve(residual, trans="T"),
    )


def smaller_stencil_ilu_preconditioner(
    system: RBFSystem,
    smaller_stencil_size: int,
    ilu_config: ILUConfig,
) -> Tuple[InverseAction, RBFSystem]:
    r"""Build ``B`` with a smaller RBF-FD stencil and apply ``ILU(B)^{-1}``.

    ``B`` uses the same nodes, boundary rows, kernel, and polynomial degree as
    the target matrix ``A``.  Only the stencil size changes.  The returned
    :class:`RBFSystem` makes construction diagnostics available to experiments.
    """

    if smaller_stencil_size >= system.config.stencil_size:
        raise ValueError("smaller stencil must be strictly smaller than the target stencil")
    polynomial_terms = (system.config.polynomial_degree + 1) * (
        system.config.polynomial_degree + 2
    ) // 2
    if smaller_stencil_size < polynomial_terms:
        raise ValueError("smaller stencil cannot underresolve the polynomial basis")
    cheaper_config = replace(system.config, stencil_size=smaller_stencil_size)
    assembly_start = perf_counter()
    cheaper_system = assemble_on_nodes(
        system.nodes,
        system.boundary,
        cheaper_config,
        coefficient=system.coefficient,
    )
    assembly_seconds = perf_counter() - assembly_start
    action = ilu_preconditioner(
        cheaper_system.matrix,
        ilu_config,
        name=f"smaller_stencil_{smaller_stencil_size}_ilu",
    )
    action.setup_seconds += assembly_seconds
    return action, cheaper_system


def build_classical_preconditioner(
    kind: str,
    system: RBFSystem,
    ilu_config: ILUConfig,
    smaller_stencil_size: Optional[int] = None,
) -> InverseAction:
    """Factory for Stage 2 classical inverse actions."""

    if kind == "none":
        return identity_preconditioner()
    if kind == "ilu_A":
        return ilu_preconditioner(system.matrix, ilu_config, name="ilu_A")
    if kind == "smaller_stencil_ilu":
        if smaller_stencil_size is None:
            raise ValueError("smaller_stencil_size is required")
        action, _ = smaller_stencil_ilu_preconditioner(
            system, smaller_stencil_size, ilu_config
        )
        return action
    raise ValueError(f"unknown classical preconditioner: {kind}")
