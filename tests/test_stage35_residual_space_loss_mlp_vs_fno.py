import unittest

import torch

from lcp_rbffd.experiments.stage35_residual_space_loss_mlp_vs_fno import residual_projector_loss


class Stage35LossTests(unittest.TestCase):
    def test_exact_action_space_has_zero_loss_and_rotation_is_invariant(self):
        torch.manual_seed(35)
        action = torch.randn(17, 9, dtype=torch.float64)
        target_coefficients = torch.randn(9, 4, dtype=torch.float64)
        target_w = torch.linalg.qr(action @ target_coefficients, mode="reduced").Q
        gram = action.T @ action
        cross = target_w.T @ action
        rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
        first = residual_projector_loss(target_coefficients, gram, cross, 4, 1.0e-12)
        second = residual_projector_loss(target_coefficients @ rotation, gram, cross, 4, 1.0e-12)
        self.assertLess(float(first), 1.0e-9)
        self.assertAlmostEqual(float(first), float(second), places=9)


if __name__ == "__main__":
    unittest.main()
