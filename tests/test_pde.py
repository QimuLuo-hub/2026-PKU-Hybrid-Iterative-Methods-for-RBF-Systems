"""Unit tests for the manufactured Poisson data."""

import unittest

import numpy as np

from lcp_rbffd.pde import (
    ConstantAnisotropicDiffusion,
    ExponentialCoefficient,
    LocalizedCoefficient,
    ManufacturedSolution,
    SpatialAnisotropicDiffusion,
)


class ManufacturedPDETests(unittest.TestCase):
    def test_default_forcing_is_negative_laplacian(self) -> None:
        nodes = np.array(
            [[0.2, 0.3], [0.5, 0.7], [0.0, 0.4], [1.0, 0.8]]
        )
        solution = ManufacturedSolution()
        expected = 2.0 * np.pi**2 * solution.values(nodes)
        np.testing.assert_allclose(solution.forcing(nodes), expected)

    def test_boundary_rhs_uses_dirichlet_values(self) -> None:
        nodes = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.25]])
        boundary = np.array([True, False, True])
        solution = ManufacturedSolution(((1, 1, 1.0), (2, 1, 0.25)))
        rhs = solution.right_hand_side(nodes, boundary)
        values = solution.values(nodes)
        np.testing.assert_allclose(rhs[boundary], values[boundary])
        np.testing.assert_allclose(
            rhs[~boundary], solution.forcing(nodes)[~boundary]
        )

    def test_exponential_coefficient_is_positive_with_known_contrast(self) -> None:
        coefficient = ExponentialCoefficient(beta=1.2)
        axis = np.linspace(0.0, 1.0, 101)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        nodes = np.column_stack((xx.ravel(), yy.ravel()))
        values = coefficient.values(nodes)
        self.assertGreater(float(np.min(values)), 0.0)
        self.assertAlmostEqual(coefficient.contrast, np.exp(2.4))
        self.assertLessEqual(
            float(np.max(values) / np.min(values)),
            coefficient.contrast * (1.0 + 1.0e-12),
        )

    def test_localized_coefficient_gradient_matches_finite_difference(self) -> None:
        coefficient = LocalizedCoefficient(peak_contrast=100.0, width=0.2)
        point = np.array([[0.37, 0.61]])
        step = 1.0e-6
        finite = []
        for axis in range(2):
            offset = np.zeros_like(point)
            offset[0, axis] = step
            finite.append(
                float(
                    (
                        coefficient.values(point + offset)
                        - coefficient.values(point - offset)
                    )[0]
                    / (2.0 * step)
                )
            )
        np.testing.assert_allclose(
            coefficient.gradient(point)[0], finite, rtol=1.0e-8, atol=1.0e-8
        )

    def test_anisotropic_tensor_is_spd_with_prescribed_ratio(self) -> None:
        coefficient = ConstantAnisotropicDiffusion(100.0, 37.0)
        eigenvalues = np.linalg.eigvalsh(coefficient.tensor)
        self.assertGreater(float(eigenvalues[0]), 0.0)
        self.assertAlmostEqual(float(eigenvalues[-1] / eigenvalues[0]), 100.0)

    def test_spatial_anisotropic_tensor_has_prescribed_eigenvalues(self) -> None:
        nodes = np.array([[0.17, 0.31], [0.51, 0.73], [0.89, 0.22]])
        coefficient = SpatialAnisotropicDiffusion(
            a0=np.log(100.0),
            a1=0.12,
            a2=-0.08,
            a3=0.05,
            theta0_degrees=30.0,
            b1_degrees=4.0,
            b2_degrees=-3.0,
        )
        tensors = coefficient.tensor_values(nodes)
        eigenvalues = np.linalg.eigvalsh(tensors)
        np.testing.assert_allclose(eigenvalues[:, 0], 1.0, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            eigenvalues[:, 1], coefficient.ratio_values(nodes), rtol=1e-13
        )

    def test_spatial_tensor_divergence_matches_finite_difference(self) -> None:
        point = np.array([[0.37, 0.61]])
        coefficient = SpatialAnisotropicDiffusion(
            a0=np.log(90.0), a1=0.1, a2=-0.14, a3=0.07,
            theta0_degrees=29.0, b1_degrees=3.0, b2_degrees=-4.0,
        )
        step = 1.0e-6
        derivatives = []
        for axis in range(2):
            offset = np.zeros_like(point)
            offset[0, axis] = step
            derivatives.append(
                (coefficient.tensor_values(point + offset)[0]
                 - coefficient.tensor_values(point - offset)[0]) / (2.0 * step)
            )
        finite_divergence = np.array(
            [derivatives[0][0, 0] + derivatives[1][0, 1],
             derivatives[0][1, 0] + derivatives[1][1, 1]]
        )
        np.testing.assert_allclose(
            coefficient.tensor_divergence(point)[0],
            finite_divergence,
            rtol=2.0e-9,
            atol=2.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
