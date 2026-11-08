import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage52_basis_equivariant_raw_compression import (
    EquivariantOperatorModel,
    Stage52Settings,
    _orthogonal_rotation,
    operator_polynomial_basis,
)


class Stage52Tests(unittest.TestCase):
    def test_protocol_is_fixed(self):
        settings = Stage52Settings()
        self.assertEqual(settings.correction_ranks, (8, 12))
        self.assertEqual(settings.rotations_per_target, 3)
        self.assertEqual(settings.operator_basis_count, 33)

    def test_operator_basis_is_symmetric(self):
        rng = np.random.default_rng(2)
        raw = rng.normal(size=(6, 12, 12))
        primitives = 0.5 * (raw + raw.transpose(0, 2, 1))
        basis = operator_polynomial_basis(primitives)
        self.assertTrue(np.allclose(basis, basis.transpose(0, 2, 1), atol=1.0e-12))

    def test_polynomial_basis_is_equivariant(self):
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(6, 10, 10))
        primitives = 0.5 * (raw + raw.transpose(0, 2, 1))
        rotation = _orthogonal_rotation(10, 7)
        first = operator_polynomial_basis(primitives)
        rotated = np.stack([rotation.T @ value @ rotation for value in primitives])
        second = operator_polynomial_basis(rotated)
        expected = np.stack([rotation.T @ value @ rotation for value in first])
        self.assertTrue(np.allclose(second, expected, atol=2.0e-6))

    def test_model_has_no_matrix_output_head(self):
        settings = Stage52Settings()
        model = EquivariantOperatorModel(settings)
        self.assertEqual(model.network[-1].out_features, 2 * settings.operator_basis_count)
        invariants = torch.zeros(2, settings.invariant_dimension)
        basis = torch.eye(8).reshape(1, 1, 8, 8).repeat(2, settings.operator_basis_count, 1, 1)
        output = model(invariants, basis)
        self.assertEqual(tuple(output.shape), (2, 2, 8, 8))


if __name__ == "__main__":
    unittest.main()

