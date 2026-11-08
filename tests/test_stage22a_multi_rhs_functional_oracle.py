"""Protocol and algebra tests for Stage 22A."""

import unittest

import numpy as np
import scipy.sparse as sp

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import (
    Stage22ASettings,
    _functional_curve_distance,
    _w_union,
    family_gate,
    system_gate,
)


class Stage22AProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Stage22ASettings()

    def test_splits_are_disjoint(self) -> None:
        construction = set(self.settings.construction_rhs_seeds)
        selection = set(self.settings.selection_rhs_seeds)
        heldout = set(self.settings.heldout_rhs_seeds)
        self.assertFalse(construction & selection)
        self.assertFalse(construction & heldout)
        self.assertFalse(selection & heldout)

    def test_system_gate(self) -> None:
        self.assertTrue(system_gate([True, True, True], [0.4, 0.5, 0.6], self.settings))
        self.assertFalse(system_gate([True, False, True], [0.8, 0.8, 0.8], self.settings))
        self.assertFalse(system_gate([True, True, True], [0.1, 0.2, 0.3], self.settings))

    def test_family_gate_uses_eighty_percent(self) -> None:
        rows = [{"system_gate_passed": index < 12} for index in range(15)]
        self.assertTrue(family_gate(rows, self.settings)["passed"])
        rows[-4]["system_gate_passed"] = False
        self.assertFalse(family_gate(rows, self.settings)["passed"])

    def test_w_union_recovers_au_equals_w(self) -> None:
        rng = np.random.default_rng(7)
        matrix = sp.csr_matrix(rng.normal(size=(12, 12)) + 4.0 * np.eye(12))
        raw = rng.normal(size=(12, 6))
        pair, details = _w_union(matrix, raw, 4, 1.0e-11)
        self.assertEqual(pair.rank, 4)
        self.assertEqual(details["dictionary_rank"], 6)
        self.assertLess(pair.diagnostics["au_equals_w_relative_error"], 1.0e-11)
        self.assertLess(pair.diagnostics["test_orthogonality_error"], 1.0e-11)

    def test_functional_curve_distance(self) -> None:
        first = [np.zeros(10), np.ones(10)]
        second = [np.zeros(10), np.ones(10) + 0.8]
        self.assertAlmostEqual(_functional_curve_distance(first, second), 0.05)


if __name__ == "__main__":
    unittest.main()
