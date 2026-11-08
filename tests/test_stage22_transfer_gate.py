"""Protocol, descriptor, and pairing tests for Stage 22 transfer."""

import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage22_transfer_gate import (
    Stage22TransferSettings,
    _target_pair_from_donor_z,
    physical_distance_rows,
    transfer_eta,
)


class Stage22TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Stage22TransferSettings(field_grid_size=9)

    def test_frozen_stage22a_protocol(self) -> None:
        self.assertEqual(self.settings.construction_rhs_seeds, (20261001, 20261011, 20261021))
        self.assertEqual(self.settings.arnoldi_dimension, 128)
        self.assertEqual(self.settings.candidates_per_rhs, 24)
        self.assertEqual(self.settings.requested_q, 48)
        self.assertEqual(self.settings.heldout_rhs_seeds, (20261101, 20261102, 20261103))

    def test_eta_definition(self) -> None:
        self.assertAlmostEqual(transfer_eta(300, 100, 50), 0.8)
        self.assertEqual(transfer_eta(300, 300, 50), 0.0)
        self.assertLess(transfer_eta(300, 310, 50), 0.0)
        self.assertTrue(np.isnan(transfer_eta(20, 10, 20)))

    def test_target_pair_preserves_AU_equals_W(self) -> None:
        rng = np.random.default_rng(11)
        matrix = sp.csr_matrix(rng.normal(size=(30, 30)) + 5.0 * np.eye(30))
        z, _ = np.linalg.qr(rng.normal(size=(30, 7)))
        pair = _target_pair_from_donor_z(matrix, z, 1.0e-12)
        self.assertEqual(pair.rank, 7)
        self.assertTrue(np.allclose(pair.z_basis, z))
        self.assertLess(pair.diagnostics["au_equals_w_relative_error"], 1.0e-11)
        self.assertLess(pair.diagnostics["test_orthogonality_error"], 1.0e-11)

    def test_physical_distances_are_symmetric_and_orientation_periodic(self) -> None:
        rows = {
            0: {
                "a0": np.log(90.0), "a1": 0.1, "a2": 0.0, "a3": 0.0,
                "theta0_degrees": 20.0, "b1_degrees": 1.0, "b2_degrees": 0.0,
            },
            1: {
                "a0": np.log(90.0), "a1": 0.1, "a2": 0.0, "a3": 0.0,
                "theta0_degrees": 200.0, "b1_degrees": 1.0, "b2_degrees": 0.0,
            },
        }
        distances, matrices = physical_distance_rows([0, 1], rows, self.settings)
        self.assertEqual(len(distances), 4)
        self.assertTrue(np.allclose(matrices["relative_K"], matrices["relative_K"].T))
        self.assertTrue(np.allclose(np.diag(matrices["relative_K"]), 0.0))
        self.assertLess(matrices["orientation"][0, 1], 1.0e-12)
        self.assertLess(matrices["relative_K"][0, 1], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
