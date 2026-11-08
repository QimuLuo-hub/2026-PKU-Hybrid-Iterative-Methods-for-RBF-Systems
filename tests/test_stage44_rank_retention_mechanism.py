import unittest

import numpy as np

from lcp_rbffd.experiments.stage44_rank_retention_mechanism import (
    Stage44Settings,
    _oracle_geometry,
)


class Stage44RankRetentionTests(unittest.TestCase):
    def test_requested_grid_is_exact(self):
        settings = Stage44Settings()
        self.assertEqual(settings.retained_ranks, (8, 16, 24, 32, 40, 48))
        self.assertEqual(settings.late_blocks, ((24, 32), (32, 40), (40, 48)))
        self.assertEqual(settings.system63_oracle_block_ranks, (4, 8, 12))

    def test_oracle_geometry_penalizes_missing_dimension(self):
        oracle = np.eye(4)
        candidate = oracle[:, :2]
        result = _oracle_geometry(candidate, oracle, oracle)
        self.assertAlmostEqual(result["oracle_containment"], 0.5)
        self.assertAlmostEqual(result["largest_principal_angle_degrees"], 90.0)
        self.assertAlmostEqual(result["largest_matched_principal_angle_degrees"], 0.0)


if __name__ == "__main__":
    unittest.main()

