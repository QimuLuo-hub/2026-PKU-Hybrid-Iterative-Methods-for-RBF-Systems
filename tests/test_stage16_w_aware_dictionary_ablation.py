from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from lcp_rbffd.coarse_space import factor_az_petrov
from lcp_rbffd.experiments.stage16_w_aware_dictionary_ablation import (
    Stage16Settings,
    _pod_basis,
    _w_optimal_dictionary_metrics,
)


class _System:
    def __init__(self, matrix: sp.csr_matrix) -> None:
        self.matrix = matrix


class _Context:
    def __init__(self, matrix: sp.csr_matrix) -> None:
        self.system = _System(matrix)


def test_stage16_keeps_stage15_dictionary_dimension() -> None:
    settings = Stage16Settings()
    assert settings.dictionary_dimension == 19
    assert settings.stage15.hidden_widths == (64, 128)
    assert settings.stage15.q == 8


def test_pod_basis_is_orthonormal_and_reports_energy() -> None:
    rng = np.random.default_rng(4)
    snapshots = rng.normal(size=(30, 12))
    basis, singular, energy = _pod_basis(snapshots, 7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1.0e-12)
    expected = np.sum(singular[:7] ** 2) / np.sum(singular**2)
    assert abs(energy - expected) <= 1.0e-14


def test_w_optimal_diagnostic_recovers_contained_target() -> None:
    rng = np.random.default_rng(8)
    matrix = sp.csr_matrix(np.diag(np.linspace(1.0, 3.0, 12)))
    dictionary, _ = la.qr(rng.normal(size=(12, 6)), mode="economic")
    coefficients = rng.normal(size=(6, 3))
    target, _ = la.qr(dictionary @ coefficients, mode="economic")
    pair = factor_az_petrov(matrix, target)
    result = _w_optimal_dictionary_metrics(
        dictionary,
        1.0,
        {1.0: _Context(matrix)},
        {1.0: target},
        {1.0: pair},
    )
    assert result["w_optimal_rank"] == 3
    assert result["w_optimal_d_w_normalized"] <= 1.0e-12
