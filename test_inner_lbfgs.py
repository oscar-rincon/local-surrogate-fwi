"""Verify identical Adam behavior and the alternative inner optimizer."""
import unittest

import torch

import fwi_toy as T
from ml_ideas import LearnedPrecond
from inner_lbfgs import LBFGSPrecond


class InnerOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.Y = [torch.randn(T.nz * T.nx, device=T.dev) for _ in range(2)]
        self.S = [0.4 * y + 0.01 * torch.randn_like(y) for y in self.Y]

    def test_adam_refactor_matches_original_loop(self):
        actual = LearnedPrecond(epochs=3, illum_bound=3.0)
        expected = LearnedPrecond(epochs=3, illum_bound=3.0)
        actual.update(self.S, self.Y)
        expected.gamma = ((self.S[-1] @ self.Y[-1]) / (self.Y[-1] @ self.Y[-1])).item()
        Sb, Yb = torch.stack(self.S), torch.stack(self.Y)
        w = torch.tensor([0.25, 1], device=T.dev)
        opt = torch.optim.Adam([expected.theta, expected.K], lr=0.03)
        for _ in range(3):
            opt.zero_grad()
            loss = (w * ((expected._P(Yb) - Sb).norm(dim=1) / Sb.norm(dim=1)) ** 2).sum() / w.sum()
            loss = loss + expected.reg * (((expected.theta - expected.theta0) ** 2).mean()
                                         + ((expected.K - expected.K0) ** 2).sum())
            loss.backward()
            opt.step()
        torch.testing.assert_close(actual.theta, expected.theta, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(actual.K, expected.K, rtol=1e-5, atol=1e-7)
        self.assertEqual(actual.history[-1]["gradient_evaluations"], 3)

    def test_lbfgs_closure_counts_and_positivity(self):
        p = LBFGSPrecond(illum_bound=3.0)
        p.update(self.S, self.Y)
        entry = p.history[-1]
        self.assertGreater(entry["gradient_evaluations"], 0)
        self.assertLessEqual(entry["iterations"], 20)
        self.assertEqual(entry["loss_evaluations"], entry["gradient_evaluations"] + 1)
        self.assertGreater((self.Y[0] @ p.apply(self.Y[0])).item(), 0)
        self.assertTrue(torch.isfinite(p.theta).all())
        self.assertTrue(torch.isfinite(p.K).all())

    def test_real_solver_integration(self):
        p = LBFGSPrecond(illum_bound=3.0)
        result = T.run_lbfgs(8, precond=p)
        self.assertEqual(result["solver"].n_sims, 8)
        self.assertTrue(torch.isfinite(result["m"]).all())
        self.assertTrue(p.history)
        self.assertTrue(all(e["optimizer"] == "LBFGS" for e in p.history))
        values = [e["J"] for e in result["log"]]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
