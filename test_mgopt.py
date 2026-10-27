import unittest

import torch

import fwi_toy as T
import mgopt as G


class TestMGOPT(unittest.TestCase):
    def test_coarse_cost_accounting(self):
        s = G.MFSolver(T.d_obs)
        s.mode = "coarse"
        _, _, h = s.forward(T.v_init, keep=True)
        s.adjoint(h)
        self.assertEqual((s.n_fwd, s.n_adj), (0, 0))
        self.assertAlmostEqual(s.n_sims, 2 * G.CC, places=6)
        s.mode = "fine"
        s.forward(T.v_init)
        self.assertAlmostEqual(s.n_sims, 1 + 2 * G.CC, places=6)

    def test_c2f_zero_coarse_budget_starts_at_init(self):
        R = G.run_coarse_to_fine(4, 0, "C2F-0")
        self.assertEqual(R["solver"].coarse_cost, 0.0)
        self.assertAlmostEqual(R["log"][0]["err"], T.rel_err(T.v_init), places=6)

    def test_mgopt_budget_and_descent(self):
        R = G.run_mgopt(8, n_inner=3)
        self.assertLessEqual(R["solver"].n_sims, 8 + 1e-6)
        self.assertGreater(R["solver"].coarse_cost, 0.0)
        Js = [r["J"] for r in R["log"]]
        self.assertTrue(all(b <= a for a, b in zip(Js, Js[1:])))
        self.assertLess(Js[-1], Js[0])


if __name__ == "__main__":
    unittest.main()
