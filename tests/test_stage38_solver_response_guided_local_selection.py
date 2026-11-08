import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import _pair_from_z
from lcp_rbffd.experiments.stage38_solver_response_guided_local_selection import (
    _next_donor_block,
    _ordered_pair_from_z,
)


class Stage38SelectionTests(unittest.TestCase):
    def test_novel_block_avoids_selected_action_direction(self):
        matrix = sp.eye(8, format="csr")
        donor = _pair_from_z(matrix, np.eye(8)[:, :5])
        selected = _ordered_pair_from_z(matrix, np.eye(8)[:, :2], 1.0e-13)
        block, diagnostics = _next_donor_block(donor, selected, 2, 1.0e-13)
        self.assertEqual(block.shape, (8, 2))
        self.assertLess(np.linalg.norm(selected.w_basis.T @ block), 1.0e-10)
        self.assertEqual(diagnostics["block_rank"], 2)

    def test_ordered_pair_preserves_prefix_span(self):
        matrix = sp.diags(np.arange(1.0, 9.0), format="csr")
        raw = np.eye(8)[:, [4, 1, 7, 2]]
        pair = _ordered_pair_from_z(matrix, raw, 1.0e-13)
        overlap = np.linalg.norm(pair.z_basis[:, :2].T @ raw[:, :2], ord="fro") ** 2
        self.assertAlmostEqual(overlap, 2.0, places=10)
        self.assertEqual(pair.rank, 4)


if __name__ == "__main__":
    unittest.main()
