import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage47_fno_complementary_correction import Stage47Settings
from lcp_rbffd.experiments.stage48_full_dictionary_complementary_correction import (
    Stage48Settings,
    _batched_losses,
)


class _FixedModel(torch.nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.register_buffer("prediction", prediction)

    def forward(self, fields):
        return self.prediction[: fields.shape[0]]


class Stage48Tests(unittest.TestCase):
    def test_only_dictionary_rank_changes(self):
        old = Stage47Settings()
        new = Stage48Settings()
        old_values = old.__dict__
        new_values = new.__dict__
        self.assertEqual(new.correction_dictionary_rank, 516)
        for key, value in old_values.items():
            if key != "correction_dictionary_rank":
                self.assertEqual(value, new_values[key])

    def test_vectorized_loss_is_finite(self):
        rng = np.random.default_rng(4)
        settings = Stage48Settings(correction_dictionary_rank=16)
        prediction = torch.from_numpy(rng.normal(size=(2, 16, 12)).astype(np.float32))
        action = rng.normal(size=(2, 30, 16)).astype(np.float32)
        target = rng.normal(size=(2, 12, 30)).astype(np.float32)
        classical = rng.normal(size=(2, 7, 30)).astype(np.float32)
        data = {
            "normalized_fields": np.zeros((2, 5, 4, 4), np.float32),
            "action_grams": np.matmul(action.transpose(0, 2, 1), action),
            "target_crosses": np.matmul(target, action),
            "classical_crosses": np.matmul(classical, action),
        }
        total, subspace, overlap = _batched_losses(
            _FixedModel(prediction), data, settings
        )
        self.assertTrue(torch.isfinite(total).all())
        self.assertEqual(total.shape, (2,))
        self.assertTrue((overlap >= 0).all())


if __name__ == "__main__":
    unittest.main()

