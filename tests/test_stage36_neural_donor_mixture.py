import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage36_neural_donor_mixture import (
    masked_soft_cross_entropy,
    ranked_label,
)


class Stage36Tests(unittest.TestCase):
    def test_ranked_label_excludes_self_and_is_normalized(self):
        label = ranked_label([4, 1, 3], donor_count=5, excluded_index=4)
        self.assertEqual(float(label[4]), 0.0)
        self.assertAlmostEqual(float(label.sum()), 1.0)
        self.assertGreater(float(label[1]), float(label[3]))

    def test_masked_loss_cannot_reward_self_selection(self):
        logits = torch.tensor([[0.0, 0.0, 20.0, 0.0]])
        labels = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        unmasked = masked_soft_cross_entropy(logits, labels, None)
        masked = masked_soft_cross_entropy(logits, labels, torch.tensor([2]))
        self.assertLess(float(masked), float(unmasked))


if __name__ == "__main__":
    unittest.main()
