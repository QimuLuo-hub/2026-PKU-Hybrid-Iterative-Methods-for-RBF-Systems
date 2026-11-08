import unittest

import numpy as np

from global_rbf.collocation import (
    assemble_poisson_system,
    assemble_variable_diffusion_system,
    variable_diffusion_forcing,
)
from global_rbf.benchmarking import benchmark_fgmres
from global_rbf.kernels import gradients, negative_laplacian, values
from global_rbf.krylov import fgmres
from global_rbf.offline_dataset import (
    align_basis_to_reference,
    coefficient_representation,
    fit_shared_basis,
    projector_reconstruction_error,
    reconstructed_basis,
    validate_parameter_splits,
)
from global_rbf.nodes import square_nodes
from global_rbf.preconditioners import (
    exact_lu_preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
    low_precision_lu_preconditioner,
    rbf_local_cardinal_preconditioner,
)
from global_rbf.scaling import (
    TensorGaussianPoissonOperator,
    assemble_tensor_gaussian_poisson_blocked,
)
from global_rbf.structure import (
    additive_galerkin_coarse_preconditioner,
    build_correction_data,
    galerkin_deflation,
    harmonic_ritz_space,
    oracle_coarse_preconditioner,
    residual_minimizing_coarse_preconditioner,
    smallest_right_singular_space,
)
from global_rbf.transfer import compare_subspaces, grassmann_geodesic


class TestGlobalRBF(unittest.TestCase):
    def test_blocked_and_structured_gaussian_actions_match_reference(self):
        nodes, boundary = square_nodes(8, "tensor")
        reference = assemble_poisson_system(
            nodes,
            boundary,
            kernel="gaussian",
            epsilon=3.2,
        )
        blocked = assemble_tensor_gaussian_poisson_blocked(
            8,
            epsilon=3.2,
            block_rows=7,
        )
        structured = TensorGaussianPoissonOperator(8, 3.2)
        rng = np.random.default_rng(71)
        vector = rng.standard_normal(64)
        np.testing.assert_allclose(blocked.A, reference.A, rtol=3.0e-15, atol=2.0e-13)
        np.testing.assert_allclose(blocked.b, reference.b, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(structured.matvec(vector), reference.A @ vector, rtol=3.0e-14, atol=2.0e-12)
        np.testing.assert_allclose(structured.rmatvec(vector), reference.A.T @ vector, rtol=3.0e-14, atol=2.0e-12)

    def test_fgmres_records_restart_cycles_for_harmonic_ritz(self):
        rng = np.random.default_rng(73)
        matrix = rng.standard_normal((12, 12)) + 4.0 * np.eye(12)
        rhs = rng.standard_normal(12)
        result = fgmres(
            matrix,
            rhs,
            rtol=0.0,
            atol=0.0,
            restart=3,
            max_iterations=8,
            capture_arnoldi=True,
        )
        self.assertEqual([cycle.hessenberg.shape[1] for cycle in result.arnoldi_cycles], [3, 3, 2])
        self.assertEqual([cycle.first_iteration for cycle in result.arnoldi_cycles], [0, 3, 6])
        space = harmonic_ritz_space(result, 2)
        self.assertEqual(space.shape, (12, 2))
        self.assertTrue(np.isrealobj(space))
        np.testing.assert_allclose(space.T @ space, np.eye(2), atol=2.0e-12)

    def test_fgmres_labels_captured_true_residual_iterations(self):
        matrix = np.diag(np.linspace(1.0, 3.0, 8))
        rhs = np.ones(8)
        result = fgmres(
            matrix,
            rhs,
            restart=3,
            max_iterations=8,
            rtol=1.0e-14,
            capture_residual_vectors=True,
        )
        self.assertEqual(
            len(result.residual_vectors),
            len(result.residual_vector_iterations),
        )
        self.assertEqual(result.residual_vector_iterations[0], 0)
        self.assertTrue(
            all(
                left <= right
                for left, right in zip(
                    result.residual_vector_iterations,
                    result.residual_vector_iterations[1:],
                )
            )
        )
        for residual in result.residual_vectors:
            self.assertEqual(residual.shape, rhs.shape)

    def test_kernel_negative_laplacians_against_finite_difference(self):
        epsilon = 2.3
        point = np.array([0.31, -0.27])
        h = 2.0e-5
        for kernel in ("gaussian", "imq", "mq", "phs3"):
            eps = None if kernel == "phs3" else epsilon

            def phi(x):
                return float(values(np.array([np.linalg.norm(x)]), kernel, eps)[0])

            center = phi(point)
            lap = 0.0
            for axis in range(2):
                step = np.zeros(2)
                step[axis] = h
                lap += (phi(point + step) - 2.0 * center + phi(point - step)) / h**2
            analytic = float(negative_laplacian(np.array([np.linalg.norm(point)]), kernel, eps)[0])
            self.assertAlmostEqual(analytic, -lap, delta=3.0e-5 * max(1.0, abs(analytic)))

    def test_kernel_gradients_against_finite_difference(self):
        epsilon = 2.3
        point = np.array([0.31, -0.27])
        h = 1.0e-6
        for kernel in ("gaussian", "imq", "mq", "phs3"):
            eps = None if kernel == "phs3" else epsilon
            analytic = gradients(point[None, :], kernel, eps)[0]
            numerical = np.empty(2)
            for axis in range(2):
                step = np.zeros(2)
                step[axis] = h
                plus = values(np.array([np.linalg.norm(point + step)]), kernel, eps)[0]
                minus = values(np.array([np.linalg.norm(point - step)]), kernel, eps)[0]
                numerical[axis] = (plus - minus) / (2.0 * h)
            np.testing.assert_allclose(analytic, numerical, rtol=2.0e-6, atol=2.0e-7)

    def test_boundary_rows_and_phs_augmentation(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="phs3", epsilon=None)
        self.assertEqual(system.A.shape, (31, 31))
        x, _ = system.solve_reference()
        metrics = system.error_metrics(x, evaluation_side=11)
        self.assertLess(metrics["relative_residual"], 1e-11)
        self.assertLess(metrics["boundary_max"], 1e-10)

    def test_reference_solution_is_accurate_in_resolved_gaussian_case(self):
        nodes, boundary = square_nodes(7)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=2.0)
        x, _ = system.solve_reference()
        metrics = system.error_metrics(x, evaluation_side=21)
        self.assertLess(metrics["grid_rel_l2"], 1e-2)
        self.assertLess(metrics["relative_residual"], 1e-10)

    def test_variable_diffusion_manufactured_system(self):
        nodes, boundary = square_nodes(9)
        beta = 0.7
        system = assemble_variable_diffusion_system(
            nodes,
            boundary,
            beta=beta,
            kernel="gaussian",
            epsilon=3.6,
        )
        x, _ = system.solve_reference()
        metrics = system.error_metrics(x, evaluation_side=25)
        self.assertLess(metrics["grid_rel_l2"], 3.0e-2)
        self.assertLess(metrics["relative_residual"], 1.0e-10)
        self.assertTrue(np.all(np.isfinite(variable_diffusion_forcing(nodes, beta))))

    def test_right_preconditioned_fgmres(self):
        nodes, boundary = square_nodes(6)
        system = assemble_poisson_system(nodes, boundary, kernel="imq", epsilon=3.0)
        preconditioner = exact_lu_preconditioner(system.A)
        result = fgmres(system.A, system.b, preconditioner.apply, rtol=1e-11, restart=20)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 2)
        self.assertLess(result.relative_residual, 1e-10)

    def test_classical_baseline_applications_preserve_dimension(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=2.5)
        preconditioners = (
            identity_preconditioner(system.dimension),
            low_precision_lu_preconditioner(system.A),
            rbf_local_cardinal_preconditioner(system.A, system.nodes, support_size=9),
        )
        residual = np.linspace(-1.0, 1.0, system.dimension)
        for preconditioner in preconditioners:
            correction = preconditioner.apply(residual)
            self.assertEqual(correction.shape, residual.shape)
            self.assertTrue(np.all(np.isfinite(correction)))

    def test_transpose_actions_match_preconditioner_adjoint_products(self):
        nodes, boundary = square_nodes(7, "tensor")
        system = assemble_poisson_system(
            nodes,
            boundary,
            kernel="gaussian",
            epsilon=2.8,
        )
        preconditioners = (
            identity_preconditioner(system.dimension),
            jacobi_preconditioner(system.A),
            low_precision_lu_preconditioner(system.A),
            rbf_local_cardinal_preconditioner(system.A, system.nodes, support_size=9),
        )
        rng = np.random.default_rng(37)
        left = rng.standard_normal(system.dimension)
        right = rng.standard_normal(system.dimension)
        for preconditioner in preconditioners:
            self.assertIsNotNone(preconditioner.apply_transpose)
            forward_product = float(left @ preconditioner.apply(right))
            transpose_product = float(preconditioner.apply_transpose(left) @ right)
            # The FP32 solve and its transpose are adjoints of the same
            # factored map only up to single-precision triangular-solve
            # rounding.  This ill-conditioned smoke system can amplify that
            # rounding to O(1e-4) in the bilinear-product check.
            tolerance = 5.0e-4 if "fp32" in preconditioner.name else 2.0e-11
            self.assertLessEqual(
                abs(forward_product - transpose_product),
                tolerance * max(abs(forward_product), abs(transpose_product), 1.0),
            )

    def test_benchmark_reports_original_system_true_residual(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=2.5)
        preconditioner = rbf_local_cardinal_preconditioner(
            system.A,
            system.nodes,
            support_size=9,
        )
        benchmark = benchmark_fgmres(
            system.A,
            system.b,
            preconditioner,
            rtol=1.0e-10,
            restart=20,
            max_iterations=50,
            repeats=1,
            warmup=False,
        )
        residual = system.b - system.A @ benchmark.result.x
        expected_norm = np.linalg.norm(residual)
        expected_relative = expected_norm / np.linalg.norm(system.b)
        self.assertAlmostEqual(benchmark.final_true_residual_norm, expected_norm, places=14)
        self.assertAlmostEqual(benchmark.final_true_relative_residual, expected_relative, places=14)

    def test_jacobi_is_well_defined_for_smooth_kernels(self):
        nodes, boundary = square_nodes(5)
        for kernel in ("gaussian", "imq", "mq"):
            system = assemble_poisson_system(nodes, boundary, kernel=kernel, epsilon=2.0)
            preconditioner = jacobi_preconditioner(system.A)
            self.assertTrue(np.all(np.isfinite(preconditioner.apply(system.b))))

    def test_exact_lu_has_negligible_missing_inverse_action(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="imq", epsilon=2.0)
        preconditioner = exact_lu_preconditioner(system.A)
        rng = np.random.default_rng(7)
        train = rng.standard_normal((system.dimension, 12))
        data = build_correction_data(
            system.A,
            preconditioner,
            train_residuals=train,
            test_residuals=train[:, :4],
        )
        exact_actions = np.linalg.solve(system.A, data.train_residuals)
        self.assertLess(np.linalg.norm(data.train_corrections) / np.linalg.norm(exact_actions), 1e-12)

    def test_petrov_galerkin_coarse_step_reduces_residual(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=3.0)
        base = jacobi_preconditioner(system.A)
        rng = np.random.default_rng(11)
        Z, _ = np.linalg.qr(rng.standard_normal((system.dimension, 4)))
        hybrid = oracle_coarse_preconditioner(
            system.A,
            base,
            Z,
            name="test",
            composition="residual_complement",
        )
        r = rng.standard_normal(system.dimension)
        base_residual = r - system.A @ base.apply(r)
        hybrid_residual = r - system.A @ hybrid.apply(r)
        self.assertLessEqual(np.linalg.norm(hybrid_residual), np.linalg.norm(base_residual) * (1.0 + 1e-12))

    def test_additive_galerkin_action_matches_requested_formula(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=3.0)
        base = jacobi_preconditioner(system.A)
        rng = np.random.default_rng(29)
        Z, _ = np.linalg.qr(rng.standard_normal((system.dimension, 4)))
        hybrid = additive_galerkin_coarse_preconditioner(
            system.A,
            base,
            Z,
            name="formula_test",
        )
        residual = rng.standard_normal(system.dimension)
        expected = base.apply(residual) + Z @ np.linalg.solve(
            Z.T @ system.A @ Z,
            Z.T @ residual,
        )
        np.testing.assert_allclose(hybrid.apply(residual), expected, rtol=2.0e-12, atol=2.0e-12)

    def test_residual_minimizing_action_uses_least_squares_coarse_term(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=3.0)
        base = jacobi_preconditioner(system.A)
        rng = np.random.default_rng(37)
        Z, _ = np.linalg.qr(rng.standard_normal((system.dimension, 4)))
        hybrid = residual_minimizing_coarse_preconditioner(
            system.A,
            base,
            Z,
            name="minimum_residual_test",
        )
        residual = rng.standard_normal(system.dimension)
        expected_y, *_ = np.linalg.lstsq(system.A @ Z, residual, rcond=None)
        expected = base.apply(residual) + Z @ expected_y
        np.testing.assert_allclose(hybrid.apply(residual), expected, rtol=2.0e-12, atol=2.0e-12)

    def test_galerkin_deflation_reconstruction_residual_identity(self):
        nodes, boundary = square_nodes(5)
        system = assemble_poisson_system(nodes, boundary, kernel="gaussian", epsilon=3.0)
        rng = np.random.default_rng(41)
        Z, _ = np.linalg.qr(rng.standard_normal((system.dimension, 4)))
        deflation = galerkin_deflation(system.A, Z)
        candidate = rng.standard_normal(system.dimension)
        reconstructed = deflation.reconstruct(system.A, system.b, candidate)
        original_residual = system.b - system.A @ reconstructed
        projected_residual = deflation.deflate_rhs(system.b) - deflation.A_deflated @ candidate
        np.testing.assert_allclose(original_residual, projected_residual, rtol=2.0e-11, atol=2.0e-11)

    def test_shared_basis_uses_smallest_energy_prefix(self):
        first = np.eye(6)[:, :2]
        second = np.eye(6)[:, 1:3]
        fit = fit_shared_basis([first, second], energy_target=0.75)
        self.assertGreaterEqual(fit.cumulative_energy[fit.retained_dimension - 1], 0.75)
        if fit.retained_dimension > 1:
            self.assertLess(fit.cumulative_energy[fit.retained_dimension - 2], 0.75)

    def test_shared_coefficients_reconstruct_a_spanned_subspace(self):
        rng = np.random.default_rng(31)
        shared, _ = np.linalg.qr(rng.standard_normal((30, 10)))
        coefficients = rng.standard_normal((10, 4))
        sample, _ = np.linalg.qr(shared @ coefficients)
        anchor = sample @ np.diag([-1.0, 1.0, -1.0, 1.0])
        aligned = align_basis_to_reference(sample, anchor)
        C = coefficient_representation(shared, aligned)
        reconstructed = reconstructed_basis(shared, C)
        error, normalized = projector_reconstruction_error(aligned, reconstructed)
        self.assertLess(error, 1.0e-12)
        self.assertLess(normalized, 1.0e-12)

    def test_parameter_splits_are_marginally_disjoint(self):
        valid = {
            "train": np.array([[5.0, 0.0], [5.5, 0.5]]),
            "validation": np.array([[5.25, 0.25]]),
            "test": np.array([[5.75, 0.75]]),
        }
        validate_parameter_splits(valid)
        invalid = dict(valid)
        invalid["test"] = np.array([[5.0, 0.75]])
        with self.assertRaises(ValueError):
            validate_parameter_splits(invalid)

    def test_subspace_metrics_are_basis_invariant(self):
        rng = np.random.default_rng(18)
        z, _ = np.linalg.qr(rng.standard_normal((30, 5)))
        rotation, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        comparison = compare_subspaces(z, z @ rotation)
        self.assertLess(comparison.normalized_projector_distance, 2.0e-8)
        self.assertLess(comparison.max_angle_degrees, 2.0e-6)

    def test_smallest_singular_space_prefix_is_nested(self):
        matrix = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        basis = smallest_right_singular_space(matrix, 3)
        expected_one = np.eye(5)[:, :1]
        expected_two = np.eye(5)[:, :2]
        self.assertLess(compare_subspaces(basis[:, :1], expected_one).normalized_projector_distance, 1.0e-8)
        self.assertLess(compare_subspaces(basis[:, :2], expected_two).normalized_projector_distance, 1.0e-8)

    def test_grassmann_geodesic_for_lines(self):
        left = np.array([[1.0], [0.0], [0.0]])
        right = np.array([[0.0], [1.0], [0.0]])
        middle = grassmann_geodesic(left, right, 0.5)
        left_to_middle = compare_subspaces(left, middle)
        middle_to_right = compare_subspaces(middle, right)
        self.assertAlmostEqual(left_to_middle.max_angle_degrees, 45.0, places=10)
        self.assertAlmostEqual(middle_to_right.max_angle_degrees, 45.0, places=10)


if __name__ == "__main__":
    unittest.main()
