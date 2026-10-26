"""Online random-feature correction for the local FWI trust-region model."""
from dataclasses import dataclass, field

import torch

import fwi_toy as T


@dataclass
class ELMCorrection:
    """Fit secant residuals; zero value/derivative at the current center.

    hidden=0 is an algebraically equivalent, compressed linear control.
    The output remains in the span of observed waveform residuals, avoiding
    a full waveform-sized matrix multiplication at every inner iteration.
    """

    hidden: int = 32
    seed: int = 0
    ridge: float = 0.01
    max_pairs: int = 16
    min_pairs: int = 4
    history: list = field(default_factory=list, init=False)
    _weights: torch.Tensor | None = field(default=None, init=False)
    _bias: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self):
        if self.hidden < 0 or self.ridge <= 0 or self.min_pairs < 1 or self.max_pairs < self.min_pairs:
            raise ValueError("Require hidden >= 0, ridge > 0, and max_pairs >= min_pairs >= 1")

    def __call__(self, pairs, Vhat, it, x_hat, Delta, cfg, U, r, gc):
        selected = [p for p in pairs if p.kind == "secant" and p.shat.norm() > 0][-self.max_pairs:]
        use_elm = self.hidden > 0 and len(selected) >= self.min_pairs
        k = Vhat.shape[1]
        scale = max(Delta, 1e-6)
        W = bias = dual = None
        basis = U
        if use_elm:
            if self._weights is None:
                gen = torch.Generator().manual_seed(self.seed)
                self._weights = torch.randn(self.hidden, Vhat.shape[0], generator=gen).to(Vhat)
                self._bias = torch.randn(self.hidden, generator=gen).to(Vhat)
            W = self._weights @ Vhat / scale
            bias = self._bias
            assert bias is not None

            def features(z):
                q = z @ W.T
                return torch.tanh(q + bias) - torch.tanh(bias) - q * (1 - torch.tanh(bias).square())

            starts = torch.stack([p.base_hat - x_hat for p in selected]) @ Vhat
            steps = torch.stack([p.shat for p in selected]) @ Vhat
            H = features(starts + steps) - features(starts)
            weights = []
            for p, c in zip(selected, steps):
                distance = (p.base_hat - x_hat).norm().item() / scale
                projection = (c.norm() / p.shat.norm()).item() ** 2
                weights.append(cfg.decay ** (it - p.it) * projection / (1 + distance ** 2))
            sqrt_w = torch.tensor(weights, device=U.device, dtype=U.dtype).sqrt()
            Hw = (H * sqrt_w[:, None]).double()
            G = Hw @ Hw.T
            lam = self.ridge * G.trace() / len(selected) + 1e-10
            dual = torch.linalg.solve(
                G + lam * torch.eye(len(selected), device=G.device, dtype=G.dtype), Hw
            ).to(U.dtype)
            residuals = torch.stack([p.dd for p in selected], 1) - U @ steps.T
            basis = torch.cat([U, residuals * sqrt_w], 1)

        gram = basis.T @ basis
        linear = basis.T @ r
        J0 = 0.5 * (r @ r)
        correction = gc - linear[:k]
        self.history.append(dict(it=it, pairs=len(selected), active=use_elm))

        def objective(c):
            if use_elm:
                assert W is not None and bias is not None and dual is not None
                q = W @ c
                phi = torch.tanh(q + bias) - torch.tanh(bias) - q * (1 - torch.tanh(bias).square())
                coefficients = torch.cat([c, dual @ phi])
            else:
                coefficients = c
            return J0 + linear @ coefficients + 0.5 * coefficients @ gram @ coefficients + correction @ c

        return objective
