import unittest

import numpy as np
import scipy.linalg as la

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import _pair_from_z
from lcp_rbffd.experiments.stage29_family_level_coarse_space import (
    Stage29Settings,
    oracle_select_inside_dictionary,
    summarize,
)


class Stage29Tests(unittest.TestCase):
    def test_frozen_rank_ladder(self):
        settings = Stage29Settings()
        self.assertEqual(settings.ranks, (80, 105, 168))
        self.assertEqual(settings.rank_labels, ("energy90", "energy95", "energy99"))

    def test_oracle_selection_stays_inside_dictionary_and_pairs(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(18, 18)) + 4.0 * np.eye(18)
        psi, _ = la.qr(rng.normal(size=(18, 10)), mode="economic")
        target_z, _ = la.qr(rng.normal(size=(18, 4)), mode="economic")
        target = _pair_from_z(matrix, target_z)
        pair, diagnostics = oracle_select_inside_dictionary(matrix, psi, target.w_basis)
        self.assertEqual(pair.rank, target.rank)
        self.assertLess(pair.diagnostics["au_equals_w_relative_error"], 1e-10)
        projection_error = np.linalg.norm(
            pair.z_basis - psi @ (psi.T @ pair.z_basis), ord="fro"
        )
        self.assertLess(projection_error, 1e-10)
        self.assertGreaterEqual(diagnostics["mean_squared_principal_cosine"], 0.0)
        self.assertLessEqual(diagnostics["mean_squared_principal_cosine"], 1.0 + 1e-12)

    def test_decision_b_when_only_ceiling_passes(self):
        settings = Stage29Settings()
        methods = [f"shared_p{rank}" for rank in settings.ranks]
        methods += [f"random_p{rank}" for rank in settings.ranks]
        methods += [settings.ceiling_label]
        rows = []
        for sample_id in range(7):
            for rhs_seed in range(3):
                for method in methods:
                    ceiling = method == settings.ceiling_label
                    iterations = 40 if ceiling else 300
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "hard": True,
                            "rhs_seed": rhs_seed,
                            "method": method,
                            "converged": ceiling,
                            "iterations": iterations,
                            "eta_full": (300 - iterations) / 270,
                            "iteration_ratio_to_ilu": iterations / 300,
                            "actual_rank": 40 if ceiling else 80,
                            "method_setup_seconds": 0.1,
                            "online_solve_seconds": 0.1,
                            "total_target_online_seconds": 0.2,
                        }
                    )
        _, decision = summarize(rows, settings)
        self.assertEqual(decision["decision"], "B")


if __name__ == "__main__":
    unittest.main()
