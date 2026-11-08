"""Protocol-level tests for the Stage 19C oracle diagnostic."""

import unittest

import numpy as np

from lcp_rbffd.experiments.stage19c_oracle_mechanism import (
    Stage19CSettings,
    _actual_prefix,
    _decision,
    _q_min_rows,
)


class Stage19CProtocolTests(unittest.TestCase):
    def test_actual_prefix_never_splits_a_complex_block(self) -> None:
        basis = np.eye(8)
        blocks = ((0, 1), (1, 3), (3, 4), (4, 6), (6, 8))
        prefix, actual = _actual_prefix(basis, blocks, requested_q=5)
        self.assertEqual(actual, 4)
        np.testing.assert_array_equal(prefix, basis[:, :4])

    def test_q_min_requires_convergence_and_acceleration(self) -> None:
        settings = Stage19CSettings(sample_ids=(3,))
        rows = []
        for definition in ("ordinary_ritz", "harmonic_ritz", "correction_pod"):
            rows.extend(
                [
                    {
                        "sample_id": 3,
                        "definition": definition,
                        "q_requested": 8,
                        "q_actual": 8,
                        "convergence_rate": 1.0,
                        "gamma_median": 0.39,
                    },
                    {
                        "sample_id": 3,
                        "definition": definition,
                        "q_requested": 16,
                        "q_actual": 16,
                        "convergence_rate": 2.0 / 3.0,
                        "gamma_median": 0.60,
                    },
                    {
                        "sample_id": 3,
                        "definition": definition,
                        "q_requested": 24,
                        "q_actual": 24,
                        "convergence_rate": 1.0,
                        "gamma_median": 0.45,
                    },
                ]
            )
        q_min = _q_min_rows(rows, settings)
        self.assertTrue(all(row["q_min"] == 24 for row in q_min))

    def test_decision_d_when_no_definition_is_robust(self) -> None:
        settings = Stage19CSettings()
        rows = []
        success_counts = {"ordinary_ritz": 2, "harmonic_ritz": 2, "correction_pod": 1}
        for definition, count in success_counts.items():
            for index, sample_id in enumerate(settings.sample_ids):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "definition": definition,
                        "success_by_q32": index < count,
                    }
                )
        decision, diagnostics = _decision(rows, settings)
        self.assertEqual(decision, "D")
        self.assertEqual(diagnostics["successful_system_counts"], success_counts)


if __name__ == "__main__":
    unittest.main()
