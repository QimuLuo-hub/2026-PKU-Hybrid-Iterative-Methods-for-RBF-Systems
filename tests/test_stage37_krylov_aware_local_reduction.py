import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage37_krylov_aware_local_reduction import krylov_local_reduction


class Stage37ReductionTests(unittest.TestCase):
    def test_probe_aligned_directions_are_selected_first(self):
        matrix = sp.eye(8, format="csr")
        raw = np.eye(8)
        probe = np.eye(8)[:, [3, 1]]
        pair, diagnostics = krylov_local_reduction(matrix, raw, probe, 4, 1.0e-13)
        selected = pair.w_basis[:, :2]
        overlap = np.linalg.norm(selected.T @ probe, ord="fro") ** 2
        self.assertAlmostEqual(overlap, 2.0, places=10)
        self.assertEqual(pair.rank, 4)
        self.assertEqual(diagnostics["probe_informed_rank"], 2)


if __name__ == "__main__":
    unittest.main()
