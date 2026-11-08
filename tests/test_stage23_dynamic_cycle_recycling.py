"""Unit tests for the frozen Stage-23 protocol and cycle extraction."""

import unittest

import numpy as np

from neurkitt_oracle.gcrodr import CycleArnoldiData
from lcp_rbffd.experiments.stage23_dynamic_cycle_recycling import (
    Stage23Settings,
    compress_union,
    dynamic_system_gate,
    extract_cycle_ritz,
    population_dynamic_gate,
)


class Stage23Tests(unittest.TestCase):
    def test_splits_and_protocol_are_frozen(self) -> None:
        settings = Stage23Settings()
        self.assertEqual(settings.local_requested_q, 24)
        self.assertEqual(settings.cumulative_max_q, 48)
        self.assertEqual(len(settings.sample_ids), 8)
        self.assertEqual(len(settings.heldout_rhs_seeds), 3)

    def test_cycle_ritz_maps_solution_basis(self) -> None:
        rng = np.random.default_rng(4)
        n, m = 40, 12
        residual, _ = np.linalg.qr(rng.normal(size=(n, m + 1)))
        solution = rng.normal(size=(n, m))
        hessenberg = np.zeros((m + 1, m))
        hessenberg[:m, :] = np.diag(np.linspace(0.1, 2.0, m))
        hessenberg[m, m - 1] = 0.03
        data = CycleArnoldiData(
            cycle=1,
            start_iteration=0,
            end_iteration=m,
            residual_basis=residual,
            solution_basis=solution,
            hessenberg=hessenberg,
            coupling=np.empty((0, m)),
            recycle_u=np.empty((n, 0)),
            recycle_c=np.empty((n, 0)),
            start_residual=residual[:, 0],
            end_residual=residual[:, -1],
            breakdown=False,
        )
        result = extract_cycle_ritz(data, 6)
        self.assertEqual(result.basis.shape, (n, 6))
        self.assertLess(result.orthogonality_error, 1e-10)

    def test_compress_union_respects_cap(self) -> None:
        rng = np.random.default_rng(5)
        basis = compress_union(rng.normal(size=(80, 30)), 12, 1e-11)
        self.assertEqual(basis.shape, (80, 12))
        self.assertLess(np.linalg.norm(basis.T @ basis - np.eye(12)), 1e-10)

    def test_predeclared_gates(self) -> None:
        settings = Stage23Settings()
        gate = dynamic_system_gate([200, 210, 220], [120, 130, 140], [True] * 3, settings)
        self.assertTrue(gate["system_dynamic_gate_passed"])
        rows = [
            {
                "method": "refreshed",
                "system_dynamic_gate_passed": index < 6,
                "ratio_to_fixed": 0.70,
                "ratio_to_ilu": 0.60,
            }
            for index in range(8)
        ]
        self.assertTrue(population_dynamic_gate(rows, "refreshed", settings)["passed"])


if __name__ == "__main__":
    unittest.main()
