import unittest

import numpy as np

from lcp_rbffd.experiments.stage18_two_parameter_anisotropy import (
    Stage18Settings,
    grassmann_bilinear,
    legacy_angle,
    requested_tensor,
    tensor_conversion_error,
)


class Stage18ProtocolTests(unittest.TestCase):
    def test_requested_tensor_has_declared_eigenvalues(self) -> None:
        tensor = requested_tensor(100.0, 30.0)
        self.assertTrue(np.allclose(np.linalg.eigvalsh(tensor), [1.0, 100.0]))
        major = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))])
        self.assertTrue(np.allclose(tensor @ major, 100.0 * major))
        self.assertEqual(legacy_angle(30.0), -60.0)
        self.assertLess(tensor_conversion_error(100.0, 30.0), 1.0e-14)

    def test_system_splits_are_disjoint_and_strictly_intermediate(self) -> None:
        settings = Stage18Settings()
        train = set(settings.train_points)
        validation = set(settings.validation_points)
        test = set(settings.test_points)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train), 25)
        self.assertEqual(len(validation), 8)
        self.assertEqual(len(test), 8)
        for ratio, theta in validation | test:
            self.assertGreater(ratio, settings.training_ratios[0])
            self.assertLess(ratio, settings.training_ratios[-1])
            self.assertGreater(theta, settings.training_thetas[0])
            self.assertLess(theta, settings.training_thetas[-1])

    def test_grassmann_interpolation_preserves_a_constant_space(self) -> None:
        rng = np.random.default_rng(18)
        basis, _ = np.linalg.qr(rng.normal(size=(31, 4)), mode="reduced")
        interpolated = grassmann_bilinear(basis, basis, basis, basis, 0.37, 0.62)
        distance = np.linalg.norm(
            interpolated @ interpolated.T - basis @ basis.T, ord="fro"
        )
        self.assertLess(distance, 1.0e-10)


if __name__ == "__main__":
    unittest.main()
