import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import _pair_from_z
from lcp_rbffd.experiments.stage39_direction_level_representation_ceiling import _best_addition


class Stage39BlockOracleTests(unittest.TestCase):
    def test_best_addition_uses_uncovered_teacher_direction(self):
        matrix = sp.eye(8, format="csr")
        block = _pair_from_z(matrix, np.eye(8)[:, [0, 2, 4]])
        selected = _pair_from_z(matrix, np.eye(8)[:, [0]])
        teacher = np.eye(8)[:, [0, 4]]
        z_add, gain, actual, _ = _best_addition(
            block, selected, teacher, 1, 1.0e-13
        )
        action = np.asarray(matrix @ z_add)
        self.assertEqual(actual, 1)
        self.assertAlmostEqual(gain, 1.0, places=10)
        self.assertAlmostEqual(abs(float(action[:, 0] @ teacher[:, 1])), 1.0, places=10)

    def test_partial_final_block_respects_requested_width(self):
        matrix = sp.eye(7, format="csr")
        block = _pair_from_z(matrix, np.eye(7)[:, :3])
        teacher = np.eye(7)[:, :3]
        z_add, _, actual, _ = _best_addition(block, None, teacher, 2, 1.0e-13)
        self.assertEqual(actual, 2)
        self.assertEqual(z_add.shape, (7, 2))


if __name__ == "__main__":
    unittest.main()
