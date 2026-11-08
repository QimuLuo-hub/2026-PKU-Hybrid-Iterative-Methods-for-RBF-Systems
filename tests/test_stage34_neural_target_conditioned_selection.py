import unittest

import torch

from lcp_rbffd.experiments.stage34_neural_target_conditioned_selection import coordinate_projector_loss


class Stage34LossTests(unittest.TestCase):
    def test_loss_is_invariant_to_target_rotation(self):
        torch.manual_seed(2)
        target = torch.linalg.qr(torch.randn(12, 4)).Q
        rotation = torch.linalg.qr(torch.randn(4, 4)).Q
        first = coordinate_projector_loss(target[None, :, :], target[None, :, :], torch.tensor([4]))
        second = coordinate_projector_loss(target[None, :, :], (target @ rotation)[None, :, :], torch.tensor([4]))
        self.assertAlmostEqual(float(first), float(second), places=5)


if __name__ == "__main__":
    unittest.main()
