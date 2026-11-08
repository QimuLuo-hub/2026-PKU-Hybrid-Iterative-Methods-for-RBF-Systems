"""Protocol and decision tests for Stage 25."""

import unittest

from lcp_rbffd.experiments.stage25_spatial_k_rank_ablation import (
    Stage25Settings,
    rank_decision,
)


class Stage25Tests(unittest.TestCase):
    def test_protocol_is_frozen_and_disjoint(self) -> None:
        settings = Stage25Settings()
        self.assertEqual(settings.requested_q_values, (24, 28, 32, 36, 40, 44, 48))
        self.assertEqual(settings.archived_q_values, (24, 32, 48))
        self.assertEqual(settings.arnoldi_dimension, 128)
        self.assertEqual(settings.required_systems, 6)
        self.assertFalse(
            set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        )

    def test_rank_decision_thresholds(self) -> None:
        settings = Stage25Settings()

        def rows(first_pass):
            return [
                {"requested_q": q, "family_gate_passed": q >= first_pass}
                for q in settings.requested_q_values
            ]

        self.assertEqual(rank_decision(rows(32), settings)["category"], "substantially_smaller")
        self.assertEqual(rank_decision(rows(36), settings)["category"], "intermediate")
        self.assertEqual(rank_decision(rows(44), settings)["category"], "near_full")
        self.assertEqual(
            rank_decision(
                [
                    {"requested_q": q, "family_gate_passed": False}
                    for q in settings.requested_q_values
                ],
                settings,
            )["category"],
            "no_robust_prefix",
        )


if __name__ == "__main__":
    unittest.main()
