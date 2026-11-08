"""Unit tests for missing actions, POD, GMRES, and coarse corrections."""

import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.coarse_space import (
    build_az_petrov_coarse_preconditioner,
    build_galerkin_coarse_preconditioner,
    build_multiplicative_coarse_preconditioner,
    build_paired_petrov_coarse_preconditioner,
    factor_az_petrov,
)
from lcp_rbffd.config import ILUConfig, SolverConfig
from lcp_rbffd.krylov import restarted_gmres
from lcp_rbffd.pod import compute_pod, relative_projection_errors
from lcp_rbffd.preconditioners import InverseAction, ilu_preconditioner
from lcp_rbffd.snapshots import compute_missing_actions


class LinearAlgebraTests(unittest.TestCase):
    def test_ilu_reports_explicit_factor_storage(self) -> None:
        matrix = sp.diags(
            [np.ones(9) * -1.0, np.ones(10) * 4.0, np.ones(9) * -1.0],
            offsets=(-1, 0, 1),
            format="csr",
        )
        action = ilu_preconditioner(matrix, ILUConfig())
        self.assertGreater(action.diagnostics["factor_memory_bytes"], 0)
        self.assertEqual(
            action.diagnostics["factor_nnz"],
            action.diagnostics["factor_l_nnz"] + action.diagnostics["factor_u_nnz"],
        )

    def test_ilu_transpose_action_solves_adjoint_system(self) -> None:
        matrix = sp.csr_matrix(
            np.array([[4.0, 1.0, 0.0], [-2.0, 5.0, 1.0], [0.0, 1.0, 3.0]])
        )
        action = ilu_preconditioner(
            matrix,
            ILUConfig(drop_tolerance=0.0, fill_factor=10.0),
        )
        residual = np.array([1.0, -0.5, 2.0])
        np.testing.assert_allclose(
            action.apply_transpose(residual),
            np.linalg.solve(matrix.toarray().T, residual),
            atol=1.0e-12,
        )

    def test_missing_action_matches_definition(self) -> None:
        matrix = sp.diags([2.0, 3.0, 5.0], format="csr")
        approximate = InverseAction(
            "diagonal_approx",
            lambda residual: np.array(
                [residual[0] / 2.2, residual[1] / 2.8, residual[2] / 5.5]
            ),
        )
        residuals = np.eye(3)
        snapshots = compute_missing_actions(
            matrix, approximate, (("synthetic", residuals),)
        )
        exact = np.linalg.solve(matrix.toarray(), residuals)
        expected = exact - np.column_stack(
            [approximate.apply(residuals[:, i]) for i in range(3)]
        )
        np.testing.assert_allclose(snapshots.corrections, expected, atol=1.0e-14)
        self.assertLess(
            float(np.max(snapshots.exact_solve_relative_residuals)), 1.0e-14
        )

    def test_pod_projection_uses_basis_invariant_projector(self) -> None:
        snapshots = np.array(
            [[3.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]
        )
        pod = compute_pod(snapshots)
        errors = relative_projection_errors(
            pod.basis[:, :2], snapshots[:, :2]
        )
        self.assertLess(float(np.max(errors)), 1.0e-12)
        self.assertTrue(np.all(np.diff(pod.captured_energy) >= 0.0))
        self.assertAlmostEqual(float(pod.captured_energy[-1]), 1.0)

    def test_restarted_gmres_solves_nonsymmetric_system(self) -> None:
        matrix = sp.csr_matrix(
            np.array(
                [
                    [4.0, -1.0, 0.0, 0.0],
                    [2.0, 5.0, 1.0, 0.0],
                    [0.0, -1.0, 3.0, 1.0],
                    [0.0, 0.0, 2.0, 4.0],
                ]
            )
        )
        exact = np.array([1.0, -2.0, 0.5, 3.0])
        rhs = matrix @ exact
        result = restarted_gmres(
            matrix,
            rhs,
            SolverConfig(
                relative_tolerance=1.0e-12,
                restart=4,
                max_iterations=20,
            ),
        )
        self.assertTrue(result.converged, msg=result.failure)
        np.testing.assert_allclose(result.solution, exact, atol=1.0e-10)
        self.assertLess(result.relative_residuals[-1], 1.0e-12)

    def test_gmres_exposes_actual_preconditioner_inputs(self) -> None:
        matrix = sp.diags([2.0, 3.0, 5.0], format="csr")
        rhs = np.array([1.0, -2.0, 0.5])
        seen = []
        action_inputs = []
        action = InverseAction(
            "recording_identity",
            lambda vector: action_inputs.append(vector.copy()) or vector.copy(),
        )
        restarted_gmres(
            matrix,
            rhs,
            SolverConfig(relative_tolerance=1.0e-12, restart=3, max_iterations=6),
            action,
            preconditioner_input_callback=lambda iteration, vector: seen.append(
                (iteration, vector.copy())
            ),
        )
        self.assertEqual(len(seen), len(action_inputs))
        for (_, callback_vector), action_vector in zip(seen, action_inputs):
            np.testing.assert_allclose(callback_vector, action_vector)
            self.assertAlmostEqual(float(np.linalg.norm(callback_vector)), 1.0, places=12)

    def test_petrov_minres_coarse_residual_is_test_space_orthogonal(self) -> None:
        matrix = sp.csr_matrix(
            np.array(
                [
                    [3.0, 1.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 4.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                ]
            )
        )
        raw_basis = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
        )
        basis, _ = np.linalg.qr(raw_basis, mode="reduced")
        zero_base = InverseAction(
            "zero",
            lambda residual: np.zeros_like(residual),
        )
        action, diagnostics = build_multiplicative_coarse_preconditioner(
            matrix,
            zero_base,
            basis,
            symmetry_threshold=1.0e-14,
        )
        self.assertEqual(diagnostics["coarse_variant"], "petrov_minres")
        residual = np.array([1.0, -2.0, 0.25, 0.75])
        correction = action.apply(residual)
        remaining = residual - matrix @ correction
        test_space = matrix @ basis
        np.testing.assert_allclose(
            test_space.T @ remaining, np.zeros(2), atol=1.0e-11
        )

    def test_az_petrov_and_recycling_actions_are_equivalent(self) -> None:
        matrix = sp.csr_matrix(
            np.array(
                [
                    [3.0, 1.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 4.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                ]
            )
        )
        basis, _ = np.linalg.qr(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]),
            mode="reduced",
        )
        zero_base = InverseAction("zero", lambda vector: np.zeros_like(vector))
        factors = factor_az_petrov(matrix, basis)
        petrov, _ = build_az_petrov_coarse_preconditioner(
            matrix, zero_base, factors
        )
        recycling, _ = build_az_petrov_coarse_preconditioner(
            matrix, zero_base, factors, recycling=True
        )
        paired, _ = build_paired_petrov_coarse_preconditioner(
            matrix, zero_base, basis, factors.test
        )
        residual = np.array([1.0, -2.0, 0.25, 0.75])
        np.testing.assert_allclose(
            petrov.apply(residual), recycling.apply(residual), atol=1.0e-12
        )
        np.testing.assert_allclose(
            petrov.apply(residual), paired.apply(residual), atol=1.0e-12
        )
        np.testing.assert_allclose(
            matrix @ factors.recycling_trial, factors.test, atol=1.0e-12
        )
        self.assertLess(
            factors.diagnostics["wtaz_equals_r_relative_error"], 1.0e-12
        )

    def test_counterfactual_galerkin_matches_requested_formula(self) -> None:
        matrix = sp.csr_matrix(
            np.array(
                [
                    [3.0, 1.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 4.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                ]
            )
        )
        basis, _ = np.linalg.qr(
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]),
            mode="reduced",
        )
        zero_base = InverseAction("zero", lambda vector: np.zeros_like(vector))
        action, _ = build_galerkin_coarse_preconditioner(
            matrix, zero_base, basis
        )
        residual = np.array([1.0, -2.0, 0.25, 0.75])
        expected = basis @ np.linalg.solve(
            basis.T @ matrix @ basis, basis.T @ residual
        )
        np.testing.assert_allclose(action.apply(residual), expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
