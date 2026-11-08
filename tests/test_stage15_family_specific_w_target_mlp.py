import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage15_family_specific_w_target_mlp import (
    Stage15Settings,
    _bracketing_training_ratios,
    _nearest_training_ratio,
    reduced_w_projector_losses,
)


class Stage15SplitTests(unittest.TestCase):
    def test_interpolation_split_is_system_level_and_disjoint(self) -> None:
        settings = Stage15Settings()
        self.assertEqual(settings.train_ratios, tuple(float(value) for value in range(90, 111, 2)))
        self.assertEqual(settings.validation_ratios, (91.0, 95.0, 99.0, 103.0, 107.0))
        self.assertEqual(settings.test_ratios, (93.0, 97.0, 101.0, 105.0, 109.0))
        self.assertFalse(set(settings.train_ratios) & set(settings.validation_ratios))
        self.assertFalse(set(settings.train_ratios) & set(settings.test_ratios))
        self.assertFalse(set(settings.validation_ratios) & set(settings.test_ratios))

    def test_nearest_and_bracketing_controls_do_not_use_target_oracle(self) -> None:
        settings = Stage15Settings()
        self.assertEqual(_nearest_training_ratio(101.0, settings), 100.0)
        self.assertEqual(_bracketing_training_ratios(101.0, settings), (100.0, 102.0, 0.5))


class Stage15ReducedLossTests(unittest.TestCase):
    def test_reduced_loss_matches_explicit_projector_distance(self) -> None:
        rng = np.random.default_rng(11)
        n, p, q = 31, 9, 3
        image = rng.normal(size=(n, p))
        target, _ = np.linalg.qr(rng.normal(size=(n, q)), mode="reduced")
        coefficients = rng.normal(size=(p, q))
        explicit, _ = np.linalg.qr(image @ coefficients, mode="reduced")
        expected = np.linalg.norm(explicit @ explicit.T - target @ target.T, ord="fro") ** 2
        grams = torch.tensor((image.T @ image)[None, :, :], dtype=torch.float64)
        crosses = torch.tensor((target.T @ image)[None, :, :], dtype=torch.float64)
        actual = reduced_w_projector_losses(
            torch.tensor(coefficients[None, :, :], dtype=torch.float64),
            grams,
            crosses,
            0.0,
        )[0].item()
        self.assertAlmostEqual(actual, expected, places=11)


if __name__ == "__main__":
    unittest.main()
