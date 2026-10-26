import unittest

import torch

import fwi_toy as T
import lodo as L


class LODOTest(unittest.TestCase):
    def test_initial_matrix_is_scaled_identity(self):
        G = L.LODOMatrix(103, alpha0=0.7)
        v = torch.randn(103, device=L.dev)
        with torch.no_grad():
            self.assertLess(float((G(v) - 0.7 * v).norm() / v.norm()), 1e-4)

    def test_perturbed_matrix_is_symmetric_psd(self):
        G = L.LODOMatrix(40, seed=3)
        with torch.no_grad():
            G.blocks.add_(0.3 * torch.randn_like(G.blocks))
            M = torch.stack([G(e) for e in torch.eye(40, device=L.dev)], 1)
        self.assertLess(float((M - M.T).abs().max()), 1e-4)
        self.assertGreater(float(torch.linalg.eigvalsh(0.5 * (M + M.T)).min()), -1e-4)

    def test_hypergradient_matches_finite_difference(self):
        torch.manual_seed(0)
        n = 30
        A = torch.randn(n, n, device=L.dev); H = A @ A.T / n + torch.eye(n, device=L.dev)
        f = lambda x: 0.5 * x @ H @ x
        x, m = torch.randn(n, device=L.dev), torch.randn(n, device=L.dev)
        G = L.LODOMatrix(n, alpha0=0.1, seed=1)
        step = G(m)
        x_new = (x - step).detach()
        (-(step * (H @ x_new)).sum()).backward()
        d = torch.randn_like(G.blocks)
        eps = 1e-3
        with torch.no_grad():
            G.blocks.add_(eps * d); fp = f(x - G(m))
            G.blocks.sub_(2 * eps * d); fm = f(x - G(m))
            G.blocks.add_(eps * d)
        self.assertAlmostEqual(float((G.blocks.grad * d).sum()), float((fp - fm) / (2 * eps)), delta=2e-2)

    def test_zero_meta_lr_reproduces_lbfgs(self):
        R0 = T.run_lbfgs(14)
        R1 = L.run_lbfgs_lodo(14, meta_lr=0.0)
        self.assertEqual([r["sims"] for r in R0["log"]], [r["sims"] for r in R1["log"]])
        for a, b in zip(R0["log"], R1["log"]):
            self.assertAlmostEqual(a["J"], b["J"], delta=1e-3 * abs(a["J"]))


if __name__ == "__main__":
    unittest.main()
