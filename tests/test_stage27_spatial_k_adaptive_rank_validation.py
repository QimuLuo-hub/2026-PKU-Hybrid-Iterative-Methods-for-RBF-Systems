import unittest

from lcp_rbffd.experiments.stage27_spatial_k_adaptive_rank_validation import (
    Stage27Settings,
    frozen_prediction,
)


class Stage27ProtocolTests(unittest.TestCase):
    def test_frozen_threshold(self):
        settings = Stage27Settings()
        self.assertEqual(settings.first_cycle_contraction_threshold, 1.0e-4)
        self.assertFalse(frozen_prediction(9.9e-5, settings))
        self.assertTrue(frozen_prediction(1.0e-4, settings))

    def test_new_splits_and_fixed_construction(self):
        settings = Stage27Settings()
        self.assertFalse(
            set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        )
        self.assertEqual(settings.arnoldi_dimension, 128)
        self.assertEqual(settings.candidates_per_rhs, 24)
        self.assertEqual(settings.fixed_q_values, (28, 32, 48))

    def test_preregistered_gate(self):
        settings = Stage27Settings()
        self.assertEqual(settings.minimum_hard_systems, 6)
        self.assertEqual(settings.sensitivity_gate, 0.80)
        self.assertEqual(settings.specificity_gate, 0.80)
        self.assertEqual(settings.recovery_gate, 0.75)
        self.assertEqual(settings.mean_final_rank_gate, 36.0)


if __name__ == "__main__":
    unittest.main()
