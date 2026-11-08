"""Tests for damping and norm limiting of the direct complement."""

import unittest

import numpy as np

from lcp_rbffd.models.direct_control import make_stabilized_direct_preconditioner
from lcp_rbffd.preconditioners import InverseAction


class DirectControlTests(unittest.TestCase):
    def test_damping_preserves_the_classical_plus_complement_form(self) -> None:
        base = InverseAction("base", lambda vector: 2.0 * vector)
        action, audit = make_stabilized_direct_preconditioner(
            base, lambda vector: -0.5 * vector, alpha=0.25
        )
        vector = np.array([1.0, -3.0])
        np.testing.assert_allclose(action.apply(vector), 1.875 * vector)
        self.assertEqual(audit["applications"], 1.0)
        self.assertEqual(audit["clipped_applications"], 0.0)

    def test_norm_limit_caps_only_the_learned_term(self) -> None:
        base = InverseAction("base", lambda vector: vector.copy())
        action, audit = make_stabilized_direct_preconditioner(
            base, lambda vector: 10.0 * vector, alpha=1.0, norm_limit_ratio=0.5
        )
        vector = np.array([3.0, 4.0])
        result = action.apply(vector)
        np.testing.assert_allclose(result, 1.5 * vector)
        self.assertEqual(audit["clipped_applications"], 1.0)


if __name__ == "__main__":
    unittest.main()
