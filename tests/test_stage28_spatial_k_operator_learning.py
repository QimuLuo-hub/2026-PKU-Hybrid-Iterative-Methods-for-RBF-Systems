import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage28_spatial_k_operator_learning import (
    FNOFieldEncoder,
    ParameterMLP,
    Stage28Settings,
    normalized_w_projector_losses,
    stratified_system_split,
)


class Stage28Tests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_complete(self):
        split = stratified_system_split(range(20), range(10), 7, 0.6, 0.2)
        sets = {key: set(value) for key, value in split.items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertEqual(set(range(20)), set.union(*sets.values()))
        self.assertEqual(sum(value < 10 for value in split["test"]), 2)

    def test_normalized_loss_is_basis_invariant(self):
        torch.manual_seed(4)
        p, q = 9, 3
        coefficients = torch.randn(1, p, q)
        gram = torch.eye(p).unsqueeze(0)
        q_c = torch.linalg.qr(coefficients, mode="reduced").Q
        cross = q_c.transpose(1, 2)
        rotation = torch.linalg.qr(torch.randn(q, q)).Q
        first = normalized_w_projector_losses(
            coefficients, gram, cross, torch.tensor([float(q)]), 1e-8
        )
        second = normalized_w_projector_losses(
            coefficients @ rotation, gram, cross, torch.tensor([float(q)]), 1e-8
        )
        self.assertLess(float(first), 1e-5)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_primary_models_return_same_coefficient_shape(self):
        p, q = 11, 4
        mlp = ParameterMLP(p, q, (8, 12))
        fno = FNOFieldEncoder(p, q, width=4, modes=3, layers=2)
        self.assertEqual(tuple(mlp(torch.randn(2, 7)).shape), (2, p, q))
        self.assertEqual(tuple(fno(torch.randn(2, 5, 9, 9)).shape), (2, p, q))

    def test_frozen_stage28_protocol(self):
        settings = Stage28Settings()
        self.assertEqual(settings.teacher_q, 28)
        self.assertEqual(settings.full_q, 48)
        self.assertEqual(settings.arnoldi_dimension, 128)
        self.assertEqual(len(settings.construction_rhs_seeds), 3)
        self.assertEqual(settings.first_cycle_contraction_threshold, 1e-4)
        self.assertEqual(settings.model_seeds, (20262821, 20262822))


if __name__ == "__main__":
    unittest.main()
