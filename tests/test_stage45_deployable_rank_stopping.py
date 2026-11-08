import unittest

from lcp_rbffd.experiments.stage45_deployable_rank_stopping import (
    Stage45Settings,
    _select_rank,
)


class Stage45DeployableRankStoppingTests(unittest.TestCase):
    def test_requested_diagnostic_is_frozen(self):
        settings = Stage45Settings()
        self.assertEqual(settings.retained_ranks, (8, 16, 24, 32, 40, 48))
        self.assertEqual(settings.diagnostic_budget, 8)
        self.assertEqual(settings.diagnostic_rhs_seed, 20262931)

    def test_scale_normalized_rank_selection(self):
        curve = [
            {"rank": rank, "log10_relative_residual": residual}
            for rank, residual in zip((8, 16, 24, 32, 40, 48), (-1.0, -1.5, -1.8, -1.91, -1.95, -2.0))
        ]
        self.assertEqual(_select_rank(curve, 0.90), 32)


if __name__ == "__main__":
    unittest.main()

