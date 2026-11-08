"""Unit tests for Stage-21 predeclared gates and cost accounting."""

import math
import unittest

from lcp_rbffd.experiments.stage21_population_validation import (
    Stage21Settings,
    classify_difficulty,
    estimate_break_even,
    population_gate,
    system_gate,
)


class Stage21ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Stage21Settings()

    def test_difficulty_is_archived_ilu_only(self) -> None:
        self.assertEqual(classify_difficulty(49, self.settings), "easy")
        self.assertEqual(classify_difficulty(50, self.settings), "transition")
        self.assertEqual(classify_difficulty(199, self.settings), "transition")
        self.assertEqual(classify_difficulty(200, self.settings), "hard")

    def test_system_gate_requires_all_rhs_and_forty_percent(self) -> None:
        self.assertTrue(system_gate([True, True, True], [0.40, 0.50, 0.60], self.settings))
        self.assertFalse(system_gate([True, False, True], [0.80, 0.80, 0.80], self.settings))
        self.assertFalse(system_gate([True, True, True], [0.10, 0.20, 0.30], self.settings))

    def test_population_gate_uses_only_unseen_hard_transition(self) -> None:
        rows = [
            {"provenance_cohort": "unseen", "difficulty_class": "hard", "system_gate_passed": True, "iteration_ratio_median": 0.50},
            {"provenance_cohort": "unseen", "difficulty_class": "transition", "system_gate_passed": True, "iteration_ratio_median": 0.60},
            {"provenance_cohort": "unseen", "difficulty_class": "hard", "system_gate_passed": True, "iteration_ratio_median": 0.65},
            {"provenance_cohort": "unseen", "difficulty_class": "hard", "system_gate_passed": False, "iteration_ratio_median": 0.90},
            {"provenance_cohort": "stage20_diagnostic", "difficulty_class": "hard", "system_gate_passed": False, "iteration_ratio_median": 1.00},
            {"provenance_cohort": "unseen", "difficulty_class": "easy", "system_gate_passed": False, "iteration_ratio_median": 2.00},
        ]
        result = population_gate(rows, self.settings)
        self.assertEqual(result["systems"], 4)
        self.assertAlmostEqual(result["pass_fraction"], 0.75)
        self.assertTrue(result["population_gate_passed"])

    def test_break_even_excludes_common_ilu_setup(self) -> None:
        self.assertEqual(estimate_break_even(0.8, 0.5, 0.3), 4.0)
        self.assertTrue(math.isinf(estimate_break_even(0.8, 0.2, 0.3)))


if __name__ == "__main__":
    unittest.main()
