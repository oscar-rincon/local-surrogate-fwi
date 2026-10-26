"""Positive fixed-filter preconditioning with a small nonnegative ridge fit."""
import math

import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn.functional as Fnn

import fwi_toy as T


def nonnegative_ridge(gram, rhs, prior, ridge):
    """Solve min ||X theta - y||^2 + ridge ||theta-prior||^2, theta >= 0.

    gram=X.T@X and rhs=X.T@y; the positive ridge makes Cholesky well defined.
    """
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    G = np.asarray(gram, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    p = np.asarray(prior, dtype=np.float64)
    if G.shape != (len(p), len(p)) or b.shape != p.shape:
        raise ValueError("Incompatible normal-equation dimensions")
    if not all(np.isfinite(a).all() for a in (G, b, p)):
        raise ValueError("Nonfinite normal-equation input")
    L = np.linalg.cholesky((G + G.T) / 2 + ridge * np.eye(len(p)))
    coefficients, _ = nnls(L.T, np.linalg.solve(L, b + ridge * p))
    return coefficients


class DirectPrecond:
    """P = gamma [eps I + sum theta_l B_l B_l.T], theta_l >= 0.

    B_0=I; the other B_l are fixed Gaussian convolutions. No illumination
    map or kernel is trained. freeze=True is the equal-weight control.
    """

    def __init__(self, sigmas=(1.0, 2.0, 4.0), eps=0.1, ridge=0.01,
                 recent=4.0, freeze=False):
        if not 0 < eps < 1 or ridge <= 0 or recent <= 0:
            raise ValueError("Require 0 < eps < 1, ridge > 0 and recent > 0")
        if any(s <= 0 or not math.isfinite(s) for s in sigmas):
            raise ValueError("Filter scales must be finite and positive")
        self.eps, self.ridge, self.recent, self.freeze = eps, ridge, recent, freeze
        self.kernels = []
        for sigma in sigmas:
            radius = math.ceil(2 * sigma)
            axis = torch.arange(-radius, radius + 1, device=T.dev, dtype=T.DT)
            g = torch.exp(-0.5 * (axis / sigma).square())
            g = g / g.sum()
            self.kernels.append(torch.outer(g, g)[None, None])
        self.prior = torch.zeros(len(sigmas) + 1, device=T.dev, dtype=T.DT)
        self.prior[0] = 1 - eps
        self.theta = (torch.full_like(self.prior, (1 - eps) / len(self.prior))
                      if freeze else self.prior.clone())
        self.gamma = None
        self.history = []

    def actions(self, q):
        v = q.reshape(-1, 1, T.nz, T.nx)
        out = [q]
        for kernel in self.kernels:
            padding = kernel.shape[-1] // 2
            bt = Fnn.conv_transpose2d(v, kernel, padding=padding)
            out.append(Fnn.conv2d(bt, kernel, padding=padding).reshape(q.shape))
        return torch.stack(out, -1)

    @torch.no_grad()
    def update(self, S, Y):
        if len(S) != len(Y) or not S:
            raise ValueError("Require nonempty paired secant histories")
        s, y = S[-1], Y[-1]
        sy, yy = (s @ y).item(), (y @ y).item()
        if sy <= 0 or yy <= 0 or not math.isfinite(sy + yy):
            raise ValueError("Require finite positive secant curvature")
        self.gamma = sy / yy
        if self.freeze:
            self.history.append(dict(n_pairs=len(S), theta=self.theta.cpu().tolist()))
            return
        Sb, Yb = torch.stack(S), torch.stack(Y)
        norms = Sb.norm(dim=1)
        if not torch.isfinite(Sb).all() or not torch.isfinite(Yb).all() or (norms <= 0).any():
            raise ValueError("Require finite, nonzero model secants")
        n = len(S)
        weights = torch.tensor(
            [self.recent ** ((i - n + 1) / max(n - 1, 1)) for i in range(n)],
            device=T.dev, dtype=T.DT)
        scale = (weights / weights.sum()).sqrt() / norms
        X = (self.gamma * self.actions(Yb) * scale[:, None, None]).reshape(-1, len(self.theta))
        target = ((Sb - self.gamma * self.eps * Yb) * scale[:, None]).flatten()
        # Only a small Gram matrix and RHS cross the device boundary.
        G = (X.double().T @ X.double()).cpu().numpy()
        rhs = (X.double().T @ target.double()).cpu().numpy()
        coefficients = nonnegative_ridge(G, rhs, self.prior.cpu().numpy(), self.ridge)
        self.theta = torch.tensor(coefficients, device=T.dev, dtype=T.DT)
        loss = ((X @ self.theta - target).square().sum()).item()
        self.history.append(dict(n_pairs=n, loss=loss, theta=coefficients.tolist()))

    @torch.no_grad()
    def apply(self, q):
        if self.gamma is None:
            raise RuntimeError("Call update before applying the preconditioner")
        return self.gamma * (self.eps * q + self.actions(q) @ self.theta)
