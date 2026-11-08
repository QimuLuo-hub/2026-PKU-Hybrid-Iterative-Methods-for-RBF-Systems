"""Policy tests for Stage 26."""

import unittest

from lcp_rbffd.experiments.stage26_spatial_k_adaptive_rank import (
    Stage26Settings,
    expansion_target,
)


class Stage26Tests(unittest.TestCase):
    def test_protocol_is_frozen(self) -> None:
        settings = Stage26Settings()
        self.assertEqual(settings.default_q, 28)
        self.assertEqual(settings.adaptive_q_schedule, (28, 32, 40, 48))
        self.assertEqual(settings.moderate_contraction_threshold, 1.0e-2)
        self.assertEqual(settings.severe_contraction_threshold, 0.5)

    def test_expansion_rule(self) -> None:
        settings = Stage26Settings()
        self.assertEqual(expansion_target(28, 0.001, settings), (28, "keep_rank"))
        self.assertEqual(expansion_target(28, 0.02, settings), (32, "moderate_step"))
        self.assertEqual(expansion_target(32, 0.02, settings), (40, "moderate_step"))
        self.assertEqual(expansion_target(28, 0.6, settings), (48, "severe_jump_to_full"))
        self.assertEqual(expansion_target(48, 0.9, settings), (48, "already_at_maximum"))


if __name__ == "__main__":
    unittest.main()
