import unittest

import numpy as np

from lcp_rbffd.experiments.stage42_missing_critical_mode_diagnosis import (
    _spatial_features,
    coverage_values,
)


class Stage42DiagnosticTests(unittest.TestCase):
    def test_projection_coverage_is_directionwise_squared_norm(self):
        probe = np.eye(6)[:, :2]
        directions = np.column_stack((np.eye(6)[:, 0], np.eye(6)[:, 3]))
        np.testing.assert_allclose(coverage_values(probe, directions), [1.0, 0.0])

    def test_constant_spatial_mode_is_low_frequency_and_delocalized(self):
        direction = np.ones(16) / 4.0
        tensor = np.column_stack((np.ones(16), np.zeros(16), np.ones(16)))
        features = _spatial_features(direction, tensor)
        self.assertAlmostEqual(features["low_frequency_energy_fraction"], 1.0)
        self.assertAlmostEqual(features["effective_support_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
