import unittest

from lcp_rbffd.experiments.stage31_krylov_aware_shared_basis import (
    Stage31Settings,
    classify_result,
)


class Stage31DecisionTests(unittest.TestCase):
    def test_success_requires_solver_and_geometry_gates(self):
        settings = Stage31Settings()
        action = {
            "rhs_convergence_fraction": 1.0,
            "all_rhs_system_pass_fraction": 1.0,
            "median_rho": 0.9,
            "median_iteration_ratio_to_full_span": 1.2,
        }
        energy = {"rhs_convergence_fraction": 0.5, "median_rho": 0.3}
        geometry = {"d_W_to_teacher": 0.2, "mean_squared_principal_cosine": 0.95}
        energy_geometry = {"d_W_to_teacher": 0.7, "mean_squared_principal_cosine": 0.5}
        self.assertEqual(
            classify_result(action, energy, geometry, energy_geometry, settings)["classification"],
            "success",
        )

    def test_partial_when_action_basis_improves_but_misses_gate(self):
        settings = Stage31Settings()
        action = {
            "rhs_convergence_fraction": 0.8,
            "all_rhs_system_pass_fraction": 0.7,
            "median_rho": 0.6,
            "median_iteration_ratio_to_full_span": 2.0,
        }
        energy = {"rhs_convergence_fraction": 0.5, "median_rho": 0.3}
        geometry = {"d_W_to_teacher": 0.4, "mean_squared_principal_cosine": 0.8}
        energy_geometry = {"d_W_to_teacher": 0.7, "mean_squared_principal_cosine": 0.5}
        self.assertEqual(
            classify_result(action, energy, geometry, energy_geometry, settings)["classification"],
            "partial",
        )

    def test_failure_without_closed_loop_improvement(self):
        settings = Stage31Settings()
        action = {
            "rhs_convergence_fraction": 0.4,
            "all_rhs_system_pass_fraction": 0.3,
            "median_rho": 0.2,
            "median_iteration_ratio_to_full_span": 3.0,
        }
        energy = {"rhs_convergence_fraction": 0.5, "median_rho": 0.3}
        geometry = {"d_W_to_teacher": 0.8, "mean_squared_principal_cosine": 0.4}
        energy_geometry = {"d_W_to_teacher": 0.7, "mean_squared_principal_cosine": 0.5}
        self.assertEqual(
            classify_result(action, energy, geometry, energy_geometry, settings)["classification"],
            "failure",
        )


if __name__ == "__main__":
    unittest.main()
