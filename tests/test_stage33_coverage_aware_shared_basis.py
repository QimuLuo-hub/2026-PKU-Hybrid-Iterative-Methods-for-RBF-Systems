import unittest

from lcp_rbffd.experiments.stage33_coverage_aware_shared_basis import classify_coverage


class Stage33DecisionTests(unittest.TestCase):
    def test_success_requires_geometry_and_convergence(self):
        action = {168: {"converged_rhs": 18, "median_rho": 0.75}}
        coverage = {168: {"converged_rhs": 20, "median_rho": 0.80}}
        action_geometry = {168: {"worst_angle": 79.0}}
        coverage_geometry = {168: {"worst_angle": 70.0}}
        self.assertEqual(
            classify_coverage(action, coverage, action_geometry, coverage_geometry)["conclusion"],
            "coverage_succeeds",
        )

    def test_failure_when_convergence_does_not_improve(self):
        action = {168: {"converged_rhs": 18, "median_rho": 0.75}}
        coverage = {168: {"converged_rhs": 18, "median_rho": 0.78}}
        action_geometry = {168: {"worst_angle": 79.0}}
        coverage_geometry = {168: {"worst_angle": 68.0}}
        self.assertEqual(
            classify_coverage(action, coverage, action_geometry, coverage_geometry)["conclusion"],
            "global_basis_insufficient",
        )


if __name__ == "__main__":
    unittest.main()
