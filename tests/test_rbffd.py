"""Unit tests for local RBF-FD weights and sparse assembly."""

import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.config import DiscretizationConfig, RBFConfig
from lcp_rbffd.pde import (
    ConstantAnisotropicDiffusion,
    ExponentialCoefficient,
    ManufacturedSolution,
    SpatialAnisotropicDiffusion,
)
from lcp_rbffd.rbffd import (
    OperatorCoefficients,
    assemble_on_nodes,
    assemble_system,
    local_weights,
    monomial_powers,
)


class RBFWeightsTests(unittest.TestCase):
    def test_augmented_weights_reproduce_quadratic_laplacian(self) -> None:
        center = np.array([0.47, 0.52])
        axis = np.linspace(-0.18, 0.18, 5)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        stencil = center + np.column_stack((xx.ravel(), yy.ravel()))
        weights, _, condition = local_weights(
            stencil,
            center,
            RBFConfig(kernel="phs", order=3),
            polynomial_degree=2,
        )
        self.assertTrue(np.isfinite(condition))
        for a, b in monomial_powers(2):
            values = stencil[:, 0] ** a * stencil[:, 1] ** b
            approximation = float(weights @ values)
            exact = 0.0
            if a >= 2:
                exact -= (
                    a
                    * (a - 1)
                    * center[0] ** (a - 2)
                    * center[1] ** b
                )
            if b >= 2:
                exact -= (
                    b
                    * (b - 1)
                    * center[0] ** a
                    * center[1] ** (b - 2)
                )
            self.assertAlmostEqual(approximation, exact, places=9)

    def test_sparse_assembly_and_boundary_rows(self) -> None:
        config = DiscretizationConfig(
            n_nodes=81,
            stencil_size=17,
            polynomial_degree=2,
            node_distribution="jittered_grid",
            jitter_fraction=0.1,
            seed=3,
        )
        system = assemble_system(config)
        self.assertTrue(sp.isspmatrix_csr(system.matrix))
        self.assertLess(system.matrix.nnz, config.n_nodes**2)
        for index in np.flatnonzero(system.boundary):
            row = system.matrix.getrow(index)
            self.assertEqual(row.nnz, 1)
            self.assertEqual(row.indices[0], index)
            self.assertAlmostEqual(row.data[0], 1.0)
        self.assertEqual(
            np.count_nonzero(system.boundary)
            + np.count_nonzero(system.interior),
            config.n_nodes,
        )

    def test_gaussian_kernel_configuration_is_operational(self) -> None:
        config = DiscretizationConfig(
            n_nodes=49,
            stencil_size=15,
            polynomial_degree=2,
            node_distribution="uniform_grid",
            rbf=RBFConfig(kernel="gaussian", shape_parameter=1.2),
        )
        system = assemble_system(config)
        self.assertTrue(np.all(np.isfinite(system.matrix.data)))

    def test_warped_nodes_remain_valid_and_nonuniform(self) -> None:
        config = DiscretizationConfig(
            n_nodes=100,
            stencil_size=17,
            polynomial_degree=2,
            node_distribution="warped_grid",
            warp_strength=0.8,
            compute_local_condition_numbers=False,
        )
        system = assemble_system(config)
        self.assertGreaterEqual(float(np.min(system.nodes)), 0.0)
        self.assertLessEqual(float(np.max(system.nodes)), 1.0)
        self.assertGreater(
            system.diagnostics["nearest_neighbor_cv"], 0.25
        )

    def test_variable_coefficient_local_operator_reproduces_quadratic(
        self,
    ) -> None:
        center = np.array([0.47, 0.52])
        axis = np.linspace(-0.18, 0.18, 5)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        stencil = center + np.column_stack((xx.ravel(), yy.ravel()))
        operator = OperatorCoefficients(
            negative_laplacian=2.0,
            first_x=3.0,
            first_y=-4.0,
        )
        weights, _, _ = local_weights(
            stencil,
            center,
            RBFConfig(kernel="phs", order=3),
            polynomial_degree=2,
            operator=operator,
        )
        x, y = stencil[:, 0], stencil[:, 1]
        polynomial = x**2 + 2.0 * x * y + 3.0 * y**2 + x - 2.0 * y + 5.0
        ux = 2.0 * center[0] + 2.0 * center[1] + 1.0
        uy = 2.0 * center[0] + 6.0 * center[1] - 2.0
        expected = 2.0 * (-8.0) + 3.0 * ux - 4.0 * uy
        self.assertAlmostEqual(float(weights @ polynomial), expected, places=8)

    def test_anisotropic_local_operator_reproduces_mixed_quadratic(self) -> None:
        center = np.array([0.47, 0.52])
        axis = np.linspace(-0.18, 0.18, 5)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        stencil = center + np.column_stack((xx.ravel(), yy.ravel()))
        operator = OperatorCoefficients(
            negative_laplacian=0.0,
            negative_xx=2.0,
            negative_xy=3.0,
            negative_yy=4.0,
        )
        weights, _, _ = local_weights(
            stencil,
            center,
            RBFConfig(kernel="phs", order=3),
            polynomial_degree=2,
            operator=operator,
        )
        x, y = stencil[:, 0], stencil[:, 1]
        polynomial = x**2 + 2.0 * x * y + 3.0 * y**2
        expected = -(2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 6.0)
        self.assertAlmostEqual(float(weights @ polynomial), expected, places=8)

    def test_center_clustered_nodes_and_anisotropic_problem_are_valid(self) -> None:
        config = DiscretizationConfig(
            n_nodes=196,
            stencil_size=25,
            polynomial_degree=2,
            node_distribution="center_clustered_grid",
            density_exponent=1.5,
            compute_local_condition_numbers=False,
        )
        coefficient = ConstantAnisotropicDiffusion(10.0, 30.0)
        system = assemble_system(config, coefficient)
        manufactured = ManufacturedSolution()
        rhs = manufactured.anisotropic_right_hand_side(
            system.nodes, system.boundary, coefficient
        )
        numerical = sp.linalg.spsolve(system.matrix, rhs)
        exact = manufactured.values(system.nodes)
        relative_error = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
        self.assertLess(relative_error, 0.15)
        self.assertGreater(system.diagnostics["nearest_neighbor_cv"], 0.1)

    def test_spatial_constant_field_matches_legacy_constant_tensor(self) -> None:
        config = DiscretizationConfig(
            n_nodes=100,
            stencil_size=25,
            polynomial_degree=2,
            node_distribution="uniform_grid",
            compute_local_condition_numbers=False,
        )
        # Legacy constant anisotropy uses Q(phi) diag(1,R) Q(phi)^T, while
        # Stage 19 uses Q(theta) diag(R,1) Q(theta)^T.
        legacy = ConstantAnisotropicDiffusion(80.0, -60.0)
        spatial = SpatialAnisotropicDiffusion(np.log(80.0), theta0_degrees=30.0)
        legacy_system = assemble_system(config, legacy)
        spatial_system = assemble_on_nodes(
            legacy_system.nodes, legacy_system.boundary, config, spatial
        )
        difference = legacy_system.matrix - spatial_system.matrix
        self.assertLess(sp.linalg.norm(difference), 1.0e-10 * sp.linalg.norm(legacy_system.matrix))

        manufactured = ManufacturedSolution()
        legacy_rhs = manufactured.anisotropic_right_hand_side(
            legacy_system.nodes, legacy_system.boundary, legacy
        )
        spatial_rhs = manufactured.spatial_anisotropic_right_hand_side(
            spatial_system.nodes, spatial_system.boundary, spatial
        )
        np.testing.assert_allclose(spatial_rhs, legacy_rhs, rtol=1e-13, atol=1e-12)

    def test_variable_coefficient_manufactured_solution_converges(self) -> None:
        config = DiscretizationConfig(
            n_nodes=196,
            stencil_size=25,
            polynomial_degree=2,
            node_distribution="uniform_grid",
            compute_local_condition_numbers=False,
        )
        coefficient = ExponentialCoefficient(beta=0.8)
        system = assemble_system(config, coefficient)
        manufactured = ManufacturedSolution()
        rhs = manufactured.variable_coefficient_right_hand_side(
            system.nodes, system.boundary, coefficient
        )
        numerical = sp.linalg.spsolve(system.matrix, rhs)
        exact = manufactured.values(system.nodes)
        relative_error = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
        self.assertLess(relative_error, 0.08)
        self.assertAlmostEqual(
            system.diagnostics["coefficient_contrast"],
            np.exp(1.6),
        )


if __name__ == "__main__":
    unittest.main()
