"""Focused mathematical checks and a small real-solver integration test."""
import unittest

import torch

import fwi_toy as T
from elm_surrogate import ELMCorrection


class ELMTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator(device=T.dev).manual_seed(7)
        self.V = torch.eye(3, device=T.dev)
        self.U = torch.randn(8, 3, device=T.dev, generator=self.gen)
        self.r = torch.randn(8, device=T.dev, generator=self.gen)
        self.gc = torch.randn(3, device=T.dev, generator=self.gen)
        self.x = torch.zeros(3, device=T.dev)
        self.cfg = T.TRConfig()

    def objective(self, elm, pairs):
        return elm(pairs, self.V, 1, self.x, 1.0, self.cfg, self.U, self.r, self.gc)

    def test_linear_equivalence(self):
        expected = T.make_Jtilde(self.U, self.r, self.gc)
        for elm in [ELMCorrection(hidden=0), ELMCorrection()]:
            actual = self.objective(elm, [])
            for _ in range(4):
                c = torch.randn(3, device=T.dev, generator=self.gen) * 0.1
                torch.testing.assert_close(actual(c), expected(c))

    def test_nonlinear_fit_and_first_order_consistency(self):
        elm = ELMCorrection(hidden=1, ridge=1e-6)
        elm._weights = torch.tensor([[1.5, -0.8, 0.4]], device=T.dev)
        elm._bias = torch.tensor([0.7], device=T.dev)
        amplitude = torch.linspace(0.5, 2, 8, device=T.dev)

        def nonlinear(c):
            q = (elm._weights @ c).squeeze()
            b = elm._bias.squeeze()
            return amplitude * (torch.tanh(q + b) - torch.tanh(b) - q * (1 - torch.tanh(b).square()))

        pairs = []
        for i in range(12):
            start = torch.randn(3, device=T.dev, generator=self.gen) * 0.15
            step = torch.randn(3, device=T.dev, generator=self.gen) * 0.4
            dd = self.U @ step + nonlinear(start + step) - nonlinear(start)
            pairs.append(T.Pair(step, dd, "secant", 1, start))
        obj = self.objective(elm, pairs)
        zero = torch.zeros(3, device=T.dev, requires_grad=True)
        torch.testing.assert_close(obj(zero), 0.5 * self.r.square().sum())
        (grad,) = torch.autograd.grad(obj(zero), zero)
        torch.testing.assert_close(grad, self.gc)
        linear = T.make_Jtilde(self.U, self.r, self.gc)
        elm_error = linear_error = 0.0
        correction = self.gc - self.U.T @ self.r
        for _ in range(10):
            c = torch.randn(3, device=T.dev, generator=self.gen) * 0.4
            exact = 0.5 * (self.r + self.U @ c + nonlinear(c)).square().sum() + correction @ c
            elm_error += (obj(c) - exact).abs().item()
            linear_error += (linear(c) - exact).abs().item()
        self.assertLess(elm_error, linear_error * 1e-3)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ELMCorrection(ridge=0)

    def test_real_solver_budget_and_monotonic_acceptance(self):
        elm = ELMCorrection(hidden=4, min_pairs=2)
        result = T.run_trust_region(
            T.TRConfig(budget=8, born_mode="never", inner_iters=4, surrogate_factory=elm)
        )
        self.assertEqual(result["solver"].n_sims, 8)
        self.assertTrue(torch.isfinite(result["m"]).all())
        for entry in result["log"]:
            if entry["accepted"]:
                self.assertLess(entry["J_trial"], entry["J"])
        self.assertTrue(any(entry["active"] for entry in elm.history))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
