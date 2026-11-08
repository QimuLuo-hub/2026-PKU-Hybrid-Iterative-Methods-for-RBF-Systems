import unittest

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from lcp_rbffd.experiments.stage40_target_conditioned_local_basis import (
    factor_full_action,
    select_local_combinations,
)


class Stage40LocalCombinationTests(unittest.TestCase):
    def test_probe_aligned_full_span_combinations_are_recovered(self):
        matrix = sp.diags(np.arange(1.0, 9.0), format="csr")
        psi = np.eye(8)
        q_action, triangular, diagnostics = factor_full_action(matrix, psi)
        probe = np.eye(8)[:, [5, 2, 7]]
        pair, coefficients, selection = select_local_combinations(
            matrix, psi, q_action, triangular, probe, 3, 1.0e-8, True
        )
        overlap = la.norm(pair.w_basis.T @ probe, ord="fro") ** 2
        self.assertAlmostEqual(overlap, 3.0, places=10)
        self.assertEqual(coefficients.shape, (8, 3))
        self.assertEqual(selection["actual_rank"], 3)
        self.assertEqual(diagnostics["target_action_rank"], 8)


if __name__ == "__main__":
    unittest.main()
