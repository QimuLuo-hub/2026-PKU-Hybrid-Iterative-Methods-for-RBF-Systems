"""Protocol and attribution tests for Stage 24."""

import unittest

from lcp_rbffd.experiments.stage24_stage22a_attribution import (
    Stage24Settings,
    attribution_decision,
)


class Stage24Tests(unittest.TestCase):
    def test_protocol_is_disjoint_and_minimal(self) -> None:
        settings = Stage24Settings()
        self.assertFalse(
            set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        )
        self.assertEqual(settings.shallow_m, 50)
        self.assertEqual(settings.deep_m, 128)
        self.assertEqual(settings.individual_q, 24)
        self.assertEqual(settings.reference_q, 48)

    def test_attribution_priority(self) -> None:
        def row(name, passed):
            return {"variant_id": name, "family_reproduction_gate_passed": passed}

        rows = [
            row("single_m50_seed1_q24", False),
            row("single_m128_seed1_q24", False),
            row("single_m128_seed2_q24", False),
            row("single_m128_seed3_q24", False),
            row("union3_m128_q24", False),
            row("reference_union3_m128_q48", True),
        ]
        self.assertEqual(attribution_decision(rows)["category"], "C")
        rows[1]["family_reproduction_gate_passed"] = True
        self.assertEqual(attribution_decision(rows)["category"], "A")


if __name__ == "__main__":
    unittest.main()
