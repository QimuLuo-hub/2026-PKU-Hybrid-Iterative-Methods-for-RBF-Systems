import unittest

import numpy as np

from lcp_rbffd.experiments.stage53_equivariant_algebra_oracle_ceiling import (
    Stage53Settings,
    _coordinate_containment,
    _top_eigenspace,
    coverage_metrics,
)


class Stage53Tests(unittest.TestCase):
    def test_protocol_is_fixed(self):
        settings = Stage53Settings()
        self.assertEqual(settings.correction_ranks, (8, 12))
        self.assertEqual(settings.random_restarts, 5)
        self.assertEqual(settings.stage52.operator_basis_count, 33)

    def test_top_eigenspace_recovers_algebra_member(self):
        basis = np.stack([np.diag([2.0, 0.0, -1.0]), np.diag([0.0, 2.0, -1.0])])
        mixing, _ = _top_eigenspace(basis, np.array([1.0, 0.0]), rank=1)
        target = np.array([[1.0], [0.0], [0.0]])
        self.assertAlmostEqual(_coordinate_containment(target, mixing), 1.0, places=12)

    def test_coverage_metrics_are_basis_invariant(self):
        rng = np.random.default_rng(4)
        target, _ = np.linalg.qr(rng.normal(size=(20, 5)))
        prediction, _ = np.linalg.qr(rng.normal(size=(20, 5)))
        rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        first = coverage_metrics(target.T, prediction, 5)
        second = coverage_metrics((target @ rotation).T, prediction, 5)
        for key in first:
            self.assertAlmostEqual(first[key], second[key], places=11)


if __name__ == "__main__":
    unittest.main()
