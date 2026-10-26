"""Mathematical and integration tests for fixed-filter NNLS preconditioning."""
import unittest

import numpy as np
from scipy.optimize import nnls
import torch

import fwi_toy as T
from direct_precond import DirectPrecond, nonnegative_ridge
from run_direct_experiments import compare_methods


class DirectTests(unittest.TestCase):
    def test_ridge_matches_full_design_nnls(self):
        rng = np.random.default_rng(4)
        X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
        prior = np.array([0.9, 0, 0, 0])
        ridge = 0.01
        expected, _ = nnls(np.vstack([X, np.sqrt(ridge) * np.eye(4)]),
                           np.concatenate([y, np.sqrt(ridge) * prior]))
        actual = nonnegative_ridge(X.T @ X, X.T @ y, prior, ridge)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_actions_are_symmetric_positive_and_fit_improves(self):
        torch.manual_seed(8)
        p = DirectPrecond()
        Y = [torch.randn(T.nz * T.nx, device=T.dev) for _ in range(3)]
        true_theta = torch.tensor([0.15, 1.2, 0.8, 0.3], device=T.dev)
        S = [0.5 * (p.eps * y + p.actions(y) @ true_theta) for y in Y]
        p.update(S, Y)
        self.assertTrue((p.theta >= 0).all())
        a, b = Y[:2]
        torch.testing.assert_close(a @ p.apply(b), b @ p.apply(a), atol=2e-4, rtol=2e-4)
        self.assertGreater((a @ p.apply(a)).item(), 0)
        self.assertLess((a @ (-p.apply(a))).item(), 0)
        def objective(theta):
            errors = [((p.gamma * (p.eps * y + p.actions(y) @ theta) - s).norm() / s.norm()).square()
                      for s, y in zip(S, Y)]
            w = torch.tensor([4 ** (-1), 4 ** (-0.5), 1], device=T.dev)
            return (torch.stack(errors) * w).sum() / w.sum() + p.ridge * (theta - p.prior).square().sum()
        self.assertLessEqual(objective(p.theta).item(), objective(p.prior).item() + 1e-6)

    def test_invalid_curvature_and_uninitialized_apply(self):
        p = DirectPrecond()
        y = torch.ones(T.nz * T.nx, device=T.dev)
        with self.assertRaises(RuntimeError):
            p.apply(y)
        with self.assertRaises(ValueError):
            p.update([-y], [y])
        with self.assertRaises(ValueError):
            nonnegative_ridge(np.eye(2), np.ones(2), np.zeros(2), 0)

    def test_real_solver_budget(self):
        for freeze in [False, True]:
            p = DirectPrecond(freeze=freeze)
            result = T.run_lbfgs(8, precond=p)
            self.assertEqual(result["solver"].n_sims, 8)
            self.assertTrue(torch.isfinite(result["m"]).all())
            self.assertTrue(p.history)
            values = [e["J"] for e in result["log"]]
            self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_unreached_quality_is_reported(self):
        baseline = dict(J=1, err=1, accepted=[dict(sims=99, seconds=50, J=1, err=1)])
        candidate = dict(accepted=[dict(sims=33, seconds=20, J=0.5, err=2)])
        result = compare_methods({"L-BFGS": baseline, "Direct": candidate})
        self.assertTrue(result["misfit"]["methods"]["Direct"]["reached"])
        self.assertFalse(result["joint"]["methods"]["Direct"]["reached"])
        self.assertIsNone(result["joint"]["methods"]["Direct"]["hit"])
        self.assertNotIn("speedup", result["joint"]["methods"]["Direct"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
