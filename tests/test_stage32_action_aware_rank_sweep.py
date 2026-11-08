import unittest

from lcp_rbffd.experiments.stage32_action_aware_rank_sweep import classify_sweep


class Stage32DecisionTests(unittest.TestCase):
    def test_rank_limited_when_high_rank_reaches_gate(self):
        result = classify_sweep(
            {168: {"converged_rhs": 18, "median_rho": 0.7}, 573: {"converged_rhs": 21, "median_rho": 0.94}},
            {168: {"median_dW": 0.54, "worst_angle": 79.0}, 573: {"median_dW": 0.22, "worst_angle": 42.0}},
            3,
        )
        self.assertEqual(result["conclusion"], "rank_limited")

    def test_criterion_limited_when_worst_angle_remains_poor(self):
        result = classify_sweep(
            {168: {"converged_rhs": 18, "median_rho": 0.7}, 573: {"converged_rhs": 18, "median_rho": 0.82}},
            {168: {"median_dW": 0.54, "worst_angle": 79.0}, 573: {"median_dW": 0.30, "worst_angle": 67.0}},
            0,
        )
        self.assertEqual(result["conclusion"], "criterion_limited")


if __name__ == "__main__":
    unittest.main()
