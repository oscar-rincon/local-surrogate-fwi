"""Checks for first-passage accounting, including nonmonotonic model error."""
import unittest

import torch

import fwi_toy as T
from run_quality_cost import TimedSolver, compare_pair, first_passage


class QualityCostTests(unittest.TestCase):
    def test_joint_requires_same_model(self):
        history = [
            dict(sims=3, seconds=2, J=0.5, err=2.0),
            dict(sims=5, seconds=4, J=2.0, err=0.5),
            dict(sims=7, seconds=6, J=0.4, err=0.4),
        ]
        self.assertEqual(first_passage(history, 1, 1, "misfit")["sims"], 3)
        self.assertEqual(first_passage(history, 1, 1, "model")["sims"], 5)
        self.assertEqual(first_passage(history, 1, 1, "joint")["sims"], 7)
        self.assertIsNone(first_passage(history, 0.1, 0.1, "joint"))
        with self.assertRaises(ValueError):
            first_passage(history, 1, 1, "invalid")

    def test_full_budget_and_first_passage_are_distinct(self):
        baseline = dict(J=1, err=1, n_sims=100, wall=80, accepted=[
            dict(sims=98, seconds=75, J=1, err=1),
        ])
        learned = dict(accepted=[dict(sims=33, seconds=30, J=0.9, err=0.9)])
        result = compare_pair(baseline, learned)["criteria"]["joint"]
        self.assertAlmostEqual(result["solves_saved_vs_full"], 0.67)
        self.assertAlmostEqual(result["solves_saved_vs_first"], 1 - 33 / 98)
        self.assertAlmostEqual(result["seconds_saved_vs_full"], 0.625)
        self.assertAlmostEqual(result["seconds_saved_vs_first"], 0.6)
        self.assertAlmostEqual(result["time_speedup"], 2.5)

    def test_missing_target_fails_explicitly(self):
        baseline = dict(J=1, err=1, n_sims=100, wall=80, accepted=[
            dict(sims=100, seconds=80, J=1, err=1),
        ])
        with self.assertRaises(RuntimeError):
            compare_pair(baseline, dict(accepted=[]))

    def test_timing_preserves_real_solver_result(self):
        reference = T.run_lbfgs(6)
        solver = TimedSolver()
        timed = T.run_lbfgs(6, solver=solver)
        self.assertEqual(solver.n_sims, 6)
        torch.testing.assert_close(timed["m"], reference["m"])
        self.assertAlmostEqual(timed["J"], reference["J"], places=6)
        times = [event["seconds"] for event in solver.events.values()]
        self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
        for entry in timed["log"][1:]:
            self.assertEqual(entry["J"], solver.events[entry["sims"]]["J"])


if __name__ == "__main__":
    unittest.main()
