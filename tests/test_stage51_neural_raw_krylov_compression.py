import unittest

import numpy as np
import scipy.sparse as sp
import torch

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import RecyclePair
from lcp_rbffd.experiments.stage51_neural_raw_krylov_compression import (
    RawCompressionModel,
    Stage51Settings,
    containment_loss,
    selected_pair,
)


class Stage51Tests(unittest.TestCase):
    def test_protocol_is_fixed(self):
        settings = Stage51Settings()
        self.assertEqual(settings.raw_depth, 64)
        self.assertEqual(len(settings.raw_rhs_seeds), 3)
        self.assertEqual(settings.correction_ranks, (8, 12))

    def test_model_outputs_only_raw_mixing(self):
        settings = Stage51Settings()
        model = RawCompressionModel(settings, use_k=True)
        fields = torch.zeros(2, 5, 71, 71)
        tokens = torch.zeros(2, 192, settings.token_feature_dimension)
        masks = torch.ones(2, 192, dtype=torch.bool)
        self.assertEqual(tuple(model(fields, tokens, masks).shape), (2, 192, 12))

    def test_containment_loss_is_basis_invariant(self):
        rng = np.random.default_rng(4)
        prediction = torch.tensor(rng.normal(size=(20, 8)), dtype=torch.float64)
        target_cross = torch.tensor(rng.normal(size=(8, 20)), dtype=torch.float64)
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        first = containment_loss(prediction, target_cross, 20, 8)
        second = containment_loss(
            prediction, torch.tensor(rotation.T @ target_cross.numpy()), 20, 8
        )
        self.assertTrue(torch.allclose(first, second, atol=1.0e-10))

    def test_selected_pair_preserves_prefix(self):
        matrix = sp.eye(10, format="csr")
        classical_u = np.eye(10)[:, :2]
        classical = RecyclePair(
            z_basis=classical_u, u_basis=classical_u, w_basis=classical_u,
            setup_seconds=0.0, diagnostics={},
        )
        candidate = np.eye(10)[:, 2:8]
        correction, hybrid, diagnostics = selected_pair(
            matrix, classical, candidate, np.eye(6)[:, :3]
        )
        self.assertEqual(correction.rank, 3)
        self.assertEqual(hybrid.rank, 5)
        self.assertLess(diagnostics["q48_correction_overlap_frobenius"], 1.0e-12)
        self.assertTrue(np.array_equal(hybrid.w_basis[:, :2], classical.w_basis))


if __name__ == "__main__":
    unittest.main()

