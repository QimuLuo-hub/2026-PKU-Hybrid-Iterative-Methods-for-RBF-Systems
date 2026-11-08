import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage19c_oracle_mechanism import _capture_arnoldi
from lcp_rbffd.experiments.stage41_multi_rhs_slow_mode_target import (
    _pairwise_diversity,
    harmonic_action_information,
)
from lcp_rbffd.preconditioners import identity_preconditioner


class Stage41SlowModeTests(unittest.TestCase):
    def test_harmonic_modes_preserve_target_action_pairing(self):
        matrix = sp.diags(np.linspace(0.2, 3.0, 20), format="csr")
        initial = np.random.default_rng(7).normal(size=20)
        arnoldi = _capture_arnoldi(matrix, identity_preconditioner(), initial, 12)
        action, family, _, _ = harmonic_action_information(matrix, arnoldi, 6)
        np.testing.assert_allclose(action, matrix @ family.basis)
        self.assertEqual(action.shape, (20, 6))
        self.assertTrue(np.all(np.diff(np.abs(family.eigenvalues)) >= -1.0e-12))

    def test_pairwise_diversity_has_all_three_probe_pairs(self):
        eye = np.eye(8)
        rows = _pairwise_diversity(63, (1, 2, 3), [eye[:, :3], eye[:, 2:5], eye[:, 5:8]])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["sample_id"] == 63 for row in rows))


if __name__ == "__main__":
    unittest.main()
