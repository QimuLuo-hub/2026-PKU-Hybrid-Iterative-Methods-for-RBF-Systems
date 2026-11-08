import unittest

import numpy as np

from lcp_rbffd.experiments.stage43_missing_subspace_block_ceiling import (
    select_missing_indices,
)


class Stage43MissingBlockTests(unittest.TestCase):
    def test_threshold_and_rank_selection_use_principal_coverage(self):
        eta = np.asarray([0.8, 0.1, 0.6, 0.2, 0.4])
        self.assertEqual(select_missing_indices(eta, threshold=0.25), [1, 3])
        self.assertEqual(select_missing_indices(eta, rank=3), [1, 3, 4])

    def test_selection_requires_exactly_one_rule(self):
        eta = np.asarray([0.1, 0.2])
        with self.assertRaises(ValueError):
            select_missing_indices(eta)
        with self.assertRaises(ValueError):
            select_missing_indices(eta, threshold=0.5, rank=1)


if __name__ == "__main__":
    unittest.main()
