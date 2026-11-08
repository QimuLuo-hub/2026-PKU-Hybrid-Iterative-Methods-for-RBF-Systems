import unittest

import numpy as np
import scipy.sparse as sp

from neurkitt_oracle.darcy import _assemble_reference_stencil, build_darcy_system
from neurkitt_oracle.gcrodr import build_augmentation, restarted_projected_gmres
from neurkitt_oracle.recycle import (
    invariant_subspace_defect,
    principal_angles_degrees,
    real_invariant_subspace,
)


class TestNeurkittOracle(unittest.TestCase):
    def test_darcy_construction_is_symmetric_positive_definite(self) -> None:
        matrix, rhs, log_k, permeability = build_darcy_system(12, seed=3)
        self.assertEqual(matrix.shape, (100, 100))
        self.assertEqual(rhs.shape, (100,))
        self.assertEqual(log_k.shape, (12, 12))
        self.assertTrue(np.all(permeability > 0.0))
        np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=0.0)
        self.assertGreater(np.linalg.eigvalsh(matrix.toarray())[0], 0.0)

    def test_constant_permeability_reduces_to_standard_five_point_laplacian(self) -> None:
        grid_size = 8
        interior_side = grid_size - 2
        matrix = _assemble_reference_stencil(np.ones((grid_size, grid_size)))
        one_dimensional = sp.diags(
            (-np.ones(interior_side - 1), 2.0 * np.ones(interior_side), -np.ones(interior_side - 1)),
            (-1, 0, 1),
            format="csr",
        )
        expected = sp.kronsum(one_dimensional, one_dimensional, format="csr")
        expected *= (grid_size - 1) ** 2
        np.testing.assert_allclose(matrix.toarray(), expected.toarray(), atol=0.0)

    def test_thin_qr_augmentation_satisfies_paper_identities(self) -> None:
        diagonal = np.linspace(1.0, 8.0, 8)
        matrix = sp.diags(diagonal, format="csr")
        basis = np.eye(8)[:, :3]
        u_basis, c_basis = build_augmentation(matrix, basis)
        np.testing.assert_allclose(matrix @ u_basis, c_basis, atol=2.0e-14)
        np.testing.assert_allclose(c_basis.T @ c_basis, np.eye(3), atol=2.0e-14)

    def test_exact_low_eigenspace_reduces_projected_arnoldi_work(self) -> None:
        diagonal = np.geomspace(1.0e-3, 10.0, 20)
        matrix = sp.diags(diagonal, format="csr")
        rhs = np.ones(20)
        basis = np.eye(20)[:, :5]
        u_basis, c_basis = build_augmentation(matrix, basis)

        baseline = restarted_projected_gmres(
            matrix,
            rhs,
            rtol=1.0e-11,
            restart=20,
            max_iterations=40,
        )
        oracle = restarted_projected_gmres(
            matrix,
            rhs,
            u_basis=u_basis,
            c_basis=c_basis,
            rtol=1.0e-11,
            restart=20,
            max_iterations=40,
        )
        self.assertTrue(baseline.converged)
        self.assertTrue(oracle.converged)
        self.assertLess(oracle.iterations, baseline.iterations)
        self.assertLessEqual(oracle.relative_residual, 1.0e-11)

    def test_right_preconditioned_projected_solver_uses_true_residual(self) -> None:
        rng = np.random.default_rng(11)
        dense = rng.standard_normal((18, 18))
        dense = dense.T @ dense + 0.2 * np.eye(18)
        matrix = sp.csr_matrix(dense)
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        order = np.argsort(np.abs(eigenvalues))
        u_basis, c_basis = build_augmentation(matrix, eigenvectors[:, order[:4]])
        inverse_diagonal = 1.0 / matrix.diagonal()
        rhs = rng.standard_normal(18)

        result = restarted_projected_gmres(
            matrix,
            rhs,
            lambda vector: inverse_diagonal * vector,
            u_basis=u_basis,
            c_basis=c_basis,
            rtol=1.0e-10,
            restart=12,
            max_iterations=120,
        )
        true_relative = np.linalg.norm(rhs - matrix @ result.x) / np.linalg.norm(rhs)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.relative_residual, true_relative, places=14)
        self.assertLessEqual(true_relative, 1.0e-10)

    def test_projected_solver_records_true_residual_and_arnoldi_identities(self) -> None:
        rng = np.random.default_rng(27)
        dense = rng.standard_normal((16, 16))
        dense += 5.0 * np.eye(16)
        matrix = sp.csr_matrix(dense)
        eigenvalues, eigenvectors = np.linalg.eig(dense)
        order = np.argsort(np.abs(eigenvalues))
        raw = np.column_stack(
            [np.real(eigenvectors[:, order[0]]), np.imag(eigenvectors[:, order[0]])]
        )
        raw = raw[:, np.linalg.norm(raw, axis=0) > 1.0e-10]
        if raw.shape[1] < 2:
            raw = np.real(eigenvectors[:, order[:2]])
        basis, _ = np.linalg.qr(raw[:, :2], mode="reduced")
        u_basis, c_basis = build_augmentation(matrix, basis)
        rhs = rng.standard_normal(16)
        inverse_diagonal = 1.0 / matrix.diagonal()

        result = restarted_projected_gmres(
            matrix,
            rhs,
            lambda vector: inverse_diagonal * vector,
            u_basis=u_basis,
            c_basis=c_basis,
            rtol=1.0e-10,
            restart=10,
            max_iterations=80,
            record_true_residuals=True,
            record_projection_diagnostics=True,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.operation_counts["preconditioner_applications"], result.iterations)
        self.assertEqual(result.true_residual_iterations[-1], result.iterations)
        self.assertAlmostEqual(result.true_residual_history[-1], result.relative_residual, places=14)
        self.assertLess(result.diagnostics["projected_arnoldi_relation_error_max"], 1.0e-12)
        self.assertLess(
            result.diagnostics["arnoldi_basis_to_w_orthogonality_error_max"],
            1.0e-11,
        )
        post = [
            row for row in result.projection_records
            if row["stage"] == "cycle_post_projection"
        ]
        self.assertTrue(post)
        self.assertLess(max(float(row["rho"]) for row in post), 1.0e-12)

    def test_complex_conjugate_pair_becomes_real_invariant_plane(self) -> None:
        rotation = np.array([[2.0, -0.5], [0.5, 2.0]])
        matrix = sp.block_diag((rotation, np.diag([4.0, 7.0])), format="csr")
        eigenvalues, eigenvectors = np.linalg.eig(matrix.toarray())
        order = np.argsort(np.abs(eigenvalues))
        subspace = real_invariant_subspace(
            eigenvalues[order],
            eigenvectors[:, order],
            2,
        )
        defect = invariant_subspace_defect(
            lambda basis: matrix @ basis,
            subspace.basis,
        )
        self.assertEqual(len(subspace.selected_indices), 2)
        self.assertLess(defect, 1.0e-14)
        np.testing.assert_allclose(
            subspace.basis.T @ subspace.basis,
            np.eye(2),
            atol=1.0e-14,
        )

    def test_principal_angles_report_largest_first_in_degrees(self) -> None:
        first = np.eye(3)[:, :2]
        theta = np.deg2rad(30.0)
        second = np.column_stack(
            (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, np.cos(theta), np.sin(theta)]),
            )
        )
        angles = principal_angles_degrees(first, second)
        np.testing.assert_allclose(angles, np.array([30.0, 0.0]), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
