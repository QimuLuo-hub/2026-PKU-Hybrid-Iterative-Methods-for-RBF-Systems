import unittest

import numpy as np

from lcp_rbffd.experiments.stage50_missing_tail_information_attribution import (
    Stage50Settings,
    observable_tail_space,
    unresolved_tail,
)


class Stage50Tests(unittest.TestCase):
    def test_protocol_is_fixed(self):
        settings = Stage50Settings()
        self.assertEqual(settings.prefix_depths, (8, 16, 32, 64))
        self.assertEqual(settings.stage41.probe_depth, 64)
        self.assertEqual(len(settings.stage41.probe_rhs_seeds), 3)

    def test_unresolved_tail_is_basis_invariant(self):
        deployable = np.eye(8)[:, :2]
        oracle = np.eye(8)[:, 1:5]
        tail, diagnostics = unresolved_tail(deployable, oracle, 1.0e-11)
        self.assertEqual(diagnostics["tail_rank"], 3)
        self.assertLess(np.linalg.norm(deployable.T @ tail), 1.0e-12)
        rotation, _ = np.linalg.qr(np.ones((4, 4)) + np.eye(4))
        rotated, _ = unresolved_tail(deployable, oracle @ rotation, 1.0e-11)
        singular = np.linalg.svd(tail.T @ rotated, compute_uv=False)
        self.assertTrue(np.allclose(singular, 1.0))

    def test_observable_tail_truncates_invisible_components(self):
        information = np.eye(6)[:, :2]
        tail = np.eye(6)[:, 1:4]
        observed, diagnostics = observable_tail_space(information, tail, 1.0e-3)
        self.assertEqual(diagnostics["observable_rank"], 1)
        self.assertEqual(observed.shape[1], 1)


if __name__ == "__main__":
    unittest.main()

