"""Unit tests for Stage 20 selection and clustering utilities."""

import unittest

import numpy as np

from lcp_rbffd.experiments.stage20_spatial_k_robust_oracle import (
    Stage20Settings,
    _cluster_atomic_blocks,
    _complete_block_indices,
    _family_success,
    _orthonormalize,
)


class Stage20ProtocolTests(unittest.TestCase):
    def test_complete_block_indices_never_split_pair(self) -> None:
        blocks = ((0, 1), (1, 3), (3, 4), (4, 6))
        self.assertEqual(_complete_block_indices(blocks, 5), (0, 1, 2, 3))

    def test_gap_clusters_cover_every_atomic_block_once(self) -> None:
        values = np.array([0.01, 0.02, 0.02, 0.021, 0.08, 0.09])
        blocks = ((0, 1), (1, 3), (3, 4), (4, 5), (5, 6))
        clusters, threshold = _cluster_atomic_blocks(values, blocks)
        flattened = [index for cluster in clusters for block in cluster for index in range(*block)]
        self.assertEqual(flattened, list(range(6)))
        self.assertGreaterEqual(threshold, 0.25)

    def test_orthonormalize_drops_dependent_columns(self) -> None:
        raw = np.column_stack((np.eye(5)[:, 0], np.eye(5)[:, 0], np.eye(5)[:, 1]))
        basis = _orthonormalize(raw)
        self.assertEqual(basis.shape, (5, 2))
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)

    def test_family_success_requires_strict_majority(self) -> None:
        settings = Stage20Settings(sample_ids=(1, 2, 3, 4))
        rows = [
            {"sample_id": i, "method": "m", "system_success": i <= 2}
            for i in settings.sample_ids
        ]
        self.assertFalse(_family_success(rows, "m", settings))
        rows[2]["system_success"] = True
        self.assertTrue(_family_success(rows, "m", settings))


if __name__ == "__main__":
    unittest.main()
