import unittest

import numpy as np
import torch

from lcp_rbffd.experiments.stage55_short_probe_neural_selector import (
    Stage55Settings,
    ShortProbeSelector,
    pair_from_overlap,
    select_neural_depth,
)


class Stage55Tests(unittest.TestCase):
    def test_protocol(self):
        settings = Stage55Settings()
        self.assertEqual(settings.probe_depths, (4, 8, 16, 32))
        self.assertEqual(settings.requested_rank, 48)
        self.assertNotIn(settings.probe_rhs_seed, settings.stage29.stage28.evaluation_rhs_seeds)

    def test_depth_rule_uses_shortest_transition(self):
        rows = [{"method": "strong_ilu", "converged_rhs": 3, "median_iterations": 300}]
        rows += [
            {"method": "classical_m4", "converged_rhs": 3, "median_iterations": 300},
            {"method": "classical_m8", "converged_rhs": 5, "median_iterations": 295},
            {"method": "classical_m16", "converged_rhs": 9, "median_iterations": 250},
            {"method": "classical_m32", "converged_rhs": 18, "median_iterations": 100},
        ]
        depth, passed = select_neural_depth(rows, Stage55Settings())
        self.assertTrue(passed)
        self.assertEqual(depth, 8)

    def test_neural_factor_preserves_probe_prefix(self):
        settings = Stage55Settings()
        depth = 16
        model = ShortProbeSelector(settings, depth, use_k=False)
        probe = torch.randn(1, settings.expected_full_span_rank, depth)
        result = model(
            torch.zeros(1, 5, 71, 71),
            torch.zeros(1, settings.expected_full_span_rank, depth + 2),
            probe,
        )
        self.assertEqual(tuple(result.shape), (1, settings.expected_full_span_rank, 48))
        torch.testing.assert_close(result[:, :, :depth], probe)


if __name__ == "__main__":
    unittest.main()
