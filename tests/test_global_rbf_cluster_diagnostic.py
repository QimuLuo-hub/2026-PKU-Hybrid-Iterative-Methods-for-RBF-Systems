"""Tests for the harmonic-Ritz cluster boundary diagnostic."""

import unittest

import numpy as np

from scripts.run_global_rbf_harmonic_cluster_diagnostic import (
    adjacent_relative_gaps,
    cluster_end,
    common_cluster_boundary,
)


class HarmonicClusterDiagnosticTests(unittest.TestCase):
    def test_cluster_end_extends_through_connected_small_gaps(self) -> None:
        magnitudes = np.array([1.0, 1.20, 1.22, 1.24, 1.50])
        self.assertEqual(cluster_end(magnitudes, 2), 4)

    def test_exact_five_percent_gap_is_a_boundary(self) -> None:
        magnitudes = np.array([1.0, 1.05, 1.06])
        self.assertEqual(cluster_end(magnitudes, 1), 1)

    def test_common_boundary_closes_clusters_at_both_endpoints(self) -> None:
        source = np.arange(1.0, 31.0)
        target = np.arange(1.0, 31.0)
        source[16] = source[15] * 1.01
        target[17] = target[16] * 1.01
        # Source extends 16 -> 17; target then extends 17 -> 18.
        self.assertEqual(common_cluster_boundary(source, target, start=16), 18)

    def test_relative_gap_definition(self) -> None:
        gaps = adjacent_relative_gaps(np.array([2.0, 2.1, 2.31]))
        np.testing.assert_allclose(gaps, [0.05, 0.10])


if __name__ == "__main__":
    unittest.main()
