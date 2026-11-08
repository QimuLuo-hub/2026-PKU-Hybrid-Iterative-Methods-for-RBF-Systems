import unittest

from lcp_rbffd.experiments.stage46_zero_solve_type3_detection import (
    Stage46Settings,
    _robust_loo_z,
)


class Stage46ZeroSolveDetectionTests(unittest.TestCase):
    def test_detector_grid_is_fixed(self):
        settings = Stage46Settings()
        self.assertEqual(settings.robust_z_threshold, 2.5)
        self.assertEqual(settings.convincing_score_margin, 1.0)

    def test_robust_loo_z_uses_only_supplied_peers(self):
        result = _robust_loo_z(4.0, [1.0, 2.0, 3.0], 0.05)
        self.assertAlmostEqual(result["median"], 2.0)
        self.assertGreater(result["z"], 0.0)


if __name__ == "__main__":
    unittest.main()

