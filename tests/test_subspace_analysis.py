"""Unit tests for basis-invariant cross-system diagnostics."""

import unittest

import numpy as np

from lcp_rbffd.subspace_analysis import (
    energy_dimension,
    generalized_subspace_metrics,
    nested_krylov_ritz_modes,
    ritz_atomic_blocks,
    standardized_parameter_distances,
    subspace_metrics,
)
from lcp_rbffd.preconditioners import InverseAction
import scipy.sparse as sp


class SubspaceAnalysisTests(unittest.TestCase):
    def test_complex_plane_is_an_indivisible_atomic_block(self) -> None:
        values = np.array([0.1, 0.2 + 0.3j, 0.2 + 0.3j, 0.8], dtype=complex)
        self.assertEqual(ritz_atomic_blocks(values), ((0, 1), (1, 3), (3, 4)))

    def test_generalized_projector_distance_penalizes_extra_direction(self) -> None:
        first = np.eye(5)[:, :1]
        second = np.eye(5)[:, :2]
        metrics = generalized_subspace_metrics(first, second)
        self.assertAlmostEqual(metrics["containment_first"], 1.0)
        self.assertAlmostEqual(metrics["containment_second"], 0.5)
        self.assertAlmostEqual(
            metrics["projector_distance_normalized"], np.sqrt(1.0 / 3.0)
        )

    def test_nested_ritz_modes_are_orthonormal_and_prefix_nested(self) -> None:
        matrix = sp.diags([1.0, 2.0, 4.0, 8.0, 16.0], format="csr")
        identity = InverseAction("identity", lambda vector: vector.copy())
        initial = np.ones(5)
        result = nested_krylov_ritz_modes(
            matrix,
            identity,
            max_modes=3,
            arnoldi_dimension=4,
            seed=3,
            initial_vector=initial,
        )
        np.testing.assert_allclose(
            result.basis.T @ result.basis, np.eye(3), atol=1.0e-10
        )
        projector_two = result.basis[:, :2] @ result.basis[:, :2].T
        np.testing.assert_allclose(
            projector_two @ result.basis[:, 0], result.basis[:, 0], atol=1.0e-10
        )
        self.assertEqual(result.eigenvalues.shape, (3,))

    def test_projector_distance_ignores_basis_rotation(self) -> None:
        rng = np.random.default_rng(4)
        basis, _ = np.linalg.qr(rng.normal(size=(30, 4)), mode="reduced")
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        metrics = subspace_metrics(basis, basis @ rotation)
        self.assertLess(metrics["projector_distance"], 1.0e-12)
        self.assertLess(metrics["principal_angle_max_degrees"], 1.0e-6)

    def test_energy_dimension_is_one_based(self) -> None:
        captured = np.array([0.5, 0.82, 0.91, 0.97, 1.0])
        self.assertEqual(energy_dimension(captured, 0.90), 3)
        self.assertEqual(energy_dimension(captured, 0.95), 4)
        self.assertEqual(energy_dimension(captured, 0.99), 5)

    def test_parameter_distance_is_symmetric(self) -> None:
        distance = standardized_parameter_distances(
            [[400, 0.0, 25], [900, 0.2, 25], [900, 0.2, 37]]
        )
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_allclose(np.diag(distance), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
