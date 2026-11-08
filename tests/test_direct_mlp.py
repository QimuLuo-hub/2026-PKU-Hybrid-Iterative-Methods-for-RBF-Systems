"""Fast unit tests for direct learned-complement plumbing."""

import unittest

import numpy as np
import torch

from lcp_rbffd.models.direct_mlp import (
    DirectComplementMLP,
    DirectMLPConfig,
    DirectMLPPredictor,
    DirectNormalizer,
    correction_metrics,
)


class DirectMLPTests(unittest.TestCase):
    def test_predictor_is_positively_homogeneous_in_residual_scale(self) -> None:
        torch.manual_seed(2)
        config = DirectMLPConfig(hidden_width=8, hidden_layers=1)
        model = DirectComplementMLP(6, 1, config)
        normalizer = DirectNormalizer(
            residual_dimension_scale=np.sqrt(6.0),
            target_scale=0.02,
            system_mean=(0.5,),
            system_scale=(0.25,),
        )
        predictor = DirectMLPPredictor(model, normalizer, ("beta",))
        residual = np.array([1.0, -2.0, 0.5, 0.25, -0.1, 0.7])
        first = predictor.predict(residual, np.array([0.8]))
        second = predictor.predict(3.5 * residual, np.array([0.8]))
        np.testing.assert_allclose(second, 3.5 * first, rtol=2.0e-6)

    def test_correction_metrics_are_exact_for_equal_arrays(self) -> None:
        target = np.array([[1.0, 2.0], [-3.0, 4.0]])
        metrics = correction_metrics(target.copy(), target)
        np.testing.assert_allclose(metrics["relative_error"], np.zeros(2))
        np.testing.assert_allclose(
            metrics["cosine_similarity"], np.ones(2)
        )
        np.testing.assert_allclose(metrics["magnitude_ratio"], np.ones(2))


if __name__ == "__main__":
    unittest.main()
