import unittest

from lcp_rbffd.experiments.stage30_full_span_representation_ceiling import (
    Stage30Settings,
    summarize_and_decide,
)


def _rows(converged, iterations):
    rows = []
    for sample_id in Stage30Settings().hard_system_ids:
        for rhs_seed in range(3):
            for method, value in (
                ("strong_ilu", 300),
                ("teacher_full", 30),
                ("oracle_within_p168", 200),
                ("oracle_within_full_span", iterations),
            ):
                rows.append({"sample_id": sample_id, "rhs_seed": rhs_seed, "method": method, "converged": True if method in {"strong_ilu", "teacher_full"} else converged, "iterations": value, "rho": (300 - value) / 270, "iteration_ratio_to_ilu": value / 300, "actual_rank": 28, "method_setup_seconds": 0.1, "online_solve_seconds": 0.1})
    return rows


def _geometry(improved):
    rows = []
    for sample_id in Stage30Settings().hard_system_ids:
        rows.extend([
            {"sample_id": sample_id, "method": "oracle_within_p168", "d_Z_to_teacher": 0.7, "d_W_to_teacher": 0.7, "maximum_angle_Z_degrees": 80, "maximum_angle_W_degrees": 80, "mean_squared_principal_cosine": 0.4},
            {"sample_id": sample_id, "method": "oracle_within_full_span", "d_Z_to_teacher": 0.5 if improved else 0.69, "d_W_to_teacher": 0.5 if improved else 0.69, "maximum_angle_Z_degrees": 60 if improved else 79, "maximum_angle_W_degrees": 60 if improved else 79, "mean_squared_principal_cosine": 0.7 if improved else 0.41},
        ])
    return rows


class Stage30DecisionTests(unittest.TestCase):
    def test_decision_a(self):
        _, decision = summarize_and_decide(_rows(True, 50), _geometry(True), Stage30Settings())
        self.assertEqual(decision["decision"], "A")

    def test_decision_b(self):
        _, decision = summarize_and_decide(_rows(False, 300), _geometry(False), Stage30Settings())
        self.assertEqual(decision["decision"], "B")

    def test_decision_c(self):
        _, decision = summarize_and_decide(_rows(False, 300), _geometry(True), Stage30Settings())
        self.assertEqual(decision["decision"], "C")


if __name__ == "__main__":
    unittest.main()
