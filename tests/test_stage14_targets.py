import unittest

import numpy as np

from lcp_rbffd.experiments.stage14_solver_aware_targets import (
    controlled_perturbation,
    replace_atomic_block,
    weighted_block_geometry,
)
from lcp_rbffd.subspace_analysis import subspace_metrics


class Stage14TargetTests(unittest.TestCase):
    def test_controlled_perturbation_has_prescribed_projector_distance(self):
        oracle = np.eye(12)[:, :3]
        candidate = controlled_perturbation(
            oracle, 30.0, np.random.default_rng(12)
        )
        distance = subspace_metrics(candidate, oracle)["projector_distance"]
        expected = np.sqrt(2.0 * oracle.shape[1]) * np.sin(np.radians(30.0))
        self.assertAlmostEqual(distance, expected, places=11)

    def test_block_replacement_preserves_unreplaced_oracle_directions(self):
        oracle = np.eye(12)[:, :4]
        candidate = replace_atomic_block(
            oracle, 1, 3, np.random.default_rng(21)
        )
        projector = candidate @ candidate.T
        self.assertLess(np.linalg.norm((np.eye(12) - projector) @ oracle[:, [0, 3]]), 1.0e-12)
        self.assertGreater(np.linalg.norm((np.eye(12) - projector) @ oracle[:, 1:3]), 1.0)

    def test_weighted_geometry_ignores_zero_weight_block(self):
        oracle = np.eye(8)[:, :2]
        candidate = np.column_stack((oracle[:, 0], np.eye(8)[:, 3]))
        self.assertAlmostEqual(
            weighted_block_geometry(candidate, oracle, ((0, 1), (1, 2)), (1.0, 0.0)),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            weighted_block_geometry(candidate, oracle, ((0, 1), (1, 2)), (0.0, 1.0)),
            1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
