import unittest

import numpy as np

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import RecyclePair
from lcp_rbffd.experiments.stage47_fno_complementary_correction import (
    Stage47Settings,
    complementary_target,
)


class Stage47ComplementTests(unittest.TestCase):
    def test_requested_ranks_and_small_output(self):
        settings = Stage47Settings()
        self.assertEqual(settings.correction_ranks, (8, 12))
        self.assertLess(settings.correction_dictionary_rank * 12, 1456 * 48)

    def test_complement_removes_classical_action(self):
        matrix = np.eye(6)
        classical_w = np.eye(6)[:, :2]
        teacher_w = np.eye(6)[:, 2:6]
        classical = RecyclePair(classical_w, classical_w, classical_w, 0.0, {"au_equals_w_relative_error": 0.0})
        teacher = RecyclePair(teacher_w, teacher_w, teacher_w, 0.0, {"au_equals_w_relative_error": 0.0})
        target, diagnostics = complementary_target(matrix, classical, teacher, 3, 1.0e-11)
        self.assertLess(np.linalg.norm(classical.w_basis.T @ target.w_basis), 1.0e-12)
        self.assertEqual(target.rank, 3)


if __name__ == "__main__":
    unittest.main()

