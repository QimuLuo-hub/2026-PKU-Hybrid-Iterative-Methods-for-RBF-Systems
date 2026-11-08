import unittest

import numpy as np

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import RecyclePair
from lcp_rbffd.experiments.stage49_target_local_correction_ceiling import (
    Stage49Settings,
    dense_probe_coefficients,
    factor_complementary_action,
    principal_coverage_metrics,
)


class Stage49Tests(unittest.TestCase):
    def test_requested_ranks(self):
        settings = Stage49Settings()
        self.assertEqual(settings.correction_ranks, (8, 12))
        self.assertEqual(settings.frozen_classical_rank, 48)

    def test_complementary_factor_and_dense_probe_coefficients(self):
        rng = np.random.default_rng(49)
        q, _ = np.linalg.qr(rng.normal(size=(20, 3)))
        classical = RecyclePair(q, q, q, 0.0, {})
        action = rng.normal(size=(20, 7))
        perpendicular, basis, triangular, pivots, diagnostics = factor_complementary_action(
            action, classical, 1.0e-11
        )
        self.assertLess(np.linalg.norm(q.T @ perpendicular), 1.0e-12)
        self.assertEqual(diagnostics["numerical_rank"], 7)
        probe, _ = np.linalg.qr(rng.normal(size=(20, 5)))
        coefficients, local = dense_probe_coefficients(
            basis, triangular, pivots, probe, 2
        )
        self.assertEqual(coefficients.shape, (7, 2))
        self.assertGreater(local["coefficient_density"], 0.5)

    def test_principal_tail_metrics(self):
        target = np.eye(5)[:, :3]
        metrics = principal_coverage_metrics(target, target)
        self.assertAlmostEqual(metrics["correction_containment"], 1.0)
        self.assertAlmostEqual(metrics["minimum_principal_coverage"], 1.0)
        self.assertAlmostEqual(metrics["largest_principal_angle_degrees"], 0.0)


if __name__ == "__main__":
    unittest.main()
