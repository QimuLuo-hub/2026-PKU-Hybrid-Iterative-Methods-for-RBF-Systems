import unittest

import numpy as np

from lcp_rbffd.experiments.stage54_missing_operator_analysis import (
    Stage54Settings,
    cross_rhs_spatial_family,
    projector_span_projection,
    residual_structure,
)


class Stage54Tests(unittest.TestCase):
    def test_protocol_is_narrow(self):
        settings = Stage54Settings()
        self.assertEqual(settings.correction_ranks, (8, 12))
        self.assertEqual(settings.base_operator_count, 33)
        self.assertLessEqual(settings.maximum_extension_families, 3)

    def test_projector_projection_is_orthogonal(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(4, 10, 10))
        basis = 0.5 * (raw + raw.transpose(0, 2, 1))
        target, _ = np.linalg.qr(rng.normal(size=(10, 3)))
        _, _, residual, energy, _ = projector_span_projection(basis, target)
        self.assertGreaterEqual(energy, 0.0)
        self.assertLess(np.max(np.abs(np.einsum("kij,ij->k", basis, residual))), 1.0e-9)

    def test_cross_rhs_family_is_symmetric_and_equivariant(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(33, 9, 9))
        basis = 0.5 * (raw + raw.transpose(0, 2, 1))
        extension = cross_rhs_spatial_family(basis)
        rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
        rotated_basis = np.stack([rotation.T @ value @ rotation for value in basis])
        rotated_extension = cross_rhs_spatial_family(rotated_basis)
        expected = np.stack([rotation.T @ value @ rotation for value in extension])
        self.assertTrue(np.allclose(extension, extension.transpose(0, 2, 1), atol=1.0e-12))
        self.assertTrue(np.allclose(rotated_extension, expected, atol=1.0e-10))

    def test_residual_categories_partition(self):
        rng = np.random.default_rng(9)
        residual = rng.normal(size=(20, 20))
        residual = 0.5 * (residual + residual.T)
        features = np.zeros((20, 47))
        fractions = rng.dirichlet(np.ones(3), size=20)
        features[:, 39:42] = fractions
        features[:, 43] = rng.uniform(size=20)
        values = residual_structure(residual, features, 64, 8, 24)
        self.assertAlmostEqual(values["diagonal_self_energy"] + values["off_diagonal_energy"], 1.0)
        self.assertAlmostEqual(values["within_rhs_energy"] + values["cross_rhs_energy"], values["off_diagonal_energy"])
        self.assertAlmostEqual(values["near_step_energy"] + values["intermediate_step_energy"] + values["long_range_step_energy"], values["off_diagonal_energy"])


if __name__ == "__main__":
    unittest.main()
