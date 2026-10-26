"""LODO (Liao, Dangovski, Foerster & Soljačić, TMLR 2023, arXiv:2210.06171) aplicado a la FWI del juguete.

G(θ) = α0 · Eᵀ G̃ G̃ᵀ E,   G̃ = B(θ¹)P₁ · B(θ²)P₂ ⋯ B(θᴺ)P_N
(bloques k×k inicializados ortogonales ⇒ G(θ₀) = α0·I; permutaciones aleatorias fijas; E embebe Rⁿ en R^ñ).
θ se aprende en línea con el hipergradiente ∇_θ f(x_{t+1}) del paso recién dado (meta-optimizador Adam).

- `run_lodo`: algoritmo 1 del artículo (x ← x − G m, momentum β, sin búsqueda lineal).
- `run_lbfgs_lodo`: L-BFGS idéntico a `fwi_toy.run_lbfgs` pero con H0 = γ·G(θ) en el two-loop; θ se actualiza con
  el hipergradiente del paso aceptado, α·(∂p/∂θ)ᵀ ∇f(x + αp).
"""
import time

import torch
import torch.nn as nn

import fwi_toy as T

dev = T.dev


class LODOMatrix(nn.Module):
    def __init__(self, n, k=4, depth=16, alpha0=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n, self.k, self.alpha0 = n, k, alpha0
        self.nt = (2 * n // k) * k
        nb = self.nt // k
        q, _ = torch.linalg.qr(torch.randn(depth, nb, k, k, generator=g))
        self.blocks = nn.Parameter(q.to(dev))
        perms = torch.stack([torch.randperm(self.nt, generator=g) for _ in range(depth)])
        self.register_buffer("perm", perms.to(dev))
        self.register_buffer("iperm", torch.argsort(perms, 1).to(dev))

    def _Gt_T(self, u):                     # G̃ᵀ u = P_Nᵀ B_Nᵀ ⋯ P₁ᵀ B₁ᵀ u
        for i in range(self.blocks.shape[0]):
            u = torch.einsum("bji,bj->bi", self.blocks[i], u.view(-1, self.k)).reshape(-1)
            u = u[self.iperm[i]]
        return u

    def _Gt(self, w):                       # G̃ w = B₁P₁ ⋯ B_N P_N w
        for i in reversed(range(self.blocks.shape[0])):
            w = w[self.perm[i]]
            w = torch.einsum("bij,bj->bi", self.blocks[i], w.view(-1, self.k)).reshape(-1)
        return w

    def forward(self, v):
        u = torch.zeros(self.nt, device=v.device, dtype=v.dtype)
        u = torch.cat([v, u[self.n:]])
        return self.alpha0 * self._Gt(self._Gt_T(u))[:self.n]


def _evaluator(solver, param):
    def evaluate(x):
        with torch.no_grad():
            m = param.model(x)
        J, _, h = solver.forward(m, keep=True)
        return J, h

    def gradient(h, x):
        return param.pullback(x, solver.adjoint(h))
    return evaluate, gradient


def run_lodo(budget, meta_lr=1e-3, beta=0.9, step0=0.01, step_max=0.06, seed=0, label="LODO"):
    """Algoritmo 1 de LODO. α0 = step0/‖g₀‖ para que el primer paso tenga la misma norma que el de L-BFGS
    (el α0 = 1 del artículo no tiene escala en FWI). m₀ = g₀ en lugar de 0 (con m₀ = 0 el primer paso es nulo
    y desperdicia 2 simulaciones). El paso se acota a ‖·‖ ≤ step_max, como en run_lbfgs."""
    solver, param = T.ExpensiveSolver(T.d_obs), T.HatParam()
    evaluate, gradient = _evaluator(solver, param)
    x = param.x0.clone()
    J, h = evaluate(x); g = gradient(h, x); del h
    G = LODOMatrix(x.numel(), alpha0=step0 / g.norm().item(), seed=seed)
    opt = torch.optim.Adam(G.parameters(), lr=meta_lr)
    m = g.clone()
    log = [dict(it=0, sims=solver.n_sims, J=J, err=T.rel_err(param.model(x)), alpha=1.0)]
    best = (J, x.clone())
    it, t_meta = 0, 0.0
    while solver.n_sims + 2 <= budget:
        it += 1
        t0 = time.time()
        step = G(m)
        sn = step.norm()
        if sn > step_max:
            step = step * (step_max / sn.detach())
        x_new = x - step.detach()
        t_meta += time.time() - t0
        J_new, h = evaluate(x_new); g_new = gradient(h, x_new); del h
        t0 = time.time()
        opt.zero_grad()
        (-(step * g_new).sum()).backward()        # ∇_θ f(x − G(θ)m) = −(∂(Gm)/∂θ)ᵀ ∇f(x_new)
        opt.step()
        t_meta += time.time() - t0
        m = beta * m + (1 - beta) * g_new
        x, g, J = x_new, g_new, J_new
        if J < best[0]:
            best = (J, x.clone())
        log.append(dict(it=it, sims=solver.n_sims, J=J, err=T.rel_err(param.model(x)), alpha=1.0))
    with torch.no_grad():
        m_fin = param.model(x)
    return dict(label=label, m=m_fin, x=x, log=log, solver=solver, err=T.rel_err(m_fin), J=J, meta_time=t_meta,
                best_J=best[0], best_err=T.rel_err(param.model(best[1])))


def two_loop(g, Sm, Ym, H0):
    """Dirección L-BFGS p = −H g con H0(q) arbitrario (diferenciable a través de H0)."""
    q = g.clone(); al = []
    for s_, y_ in zip(reversed(Sm), reversed(Ym)):
        a = (s_ @ q) / (y_ @ s_); al.append(a); q = q - a * y_
    q = H0(q)
    for (s_, y_), a in zip(zip(Sm, Ym), reversed(al)):
        b = (y_ @ q) / (y_ @ s_); q = q + (a - b) * s_
    return -q


def run_lbfgs_lodo(budget, meta_lr=1e-3, mem=8, step0=0.01, step_max=0.06, c1=1e-4, max_ls=6, seed=0,
                   label="LODO-PC"):
    """Copia de fwi_toy.run_lbfgs (sin reg/hv/param) con H0 = γ·G(θ), γ = sᵀy/yᵀy. meta_lr = 0 ⇒ L-BFGS."""
    solver, param = T.ExpensiveSolver(T.d_obs), T.HatParam()
    evaluate, gradient = _evaluator(solver, param)
    x = param.x0.clone()
    G = LODOMatrix(x.numel(), alpha0=1.0, seed=seed)
    opt = torch.optim.Adam(G.parameters(), lr=meta_lr) if meta_lr > 0 else None
    J, h = evaluate(x); g = gradient(h, x); del h
    Sm, Ym, log, it, n_meta, t_meta = [], [], [], 0, 0, 0.0
    log.append(dict(it=0, sims=solver.n_sims, J=J, err=T.rel_err(param.model(x)), alpha=0.0))
    while solver.n_sims < budget:
        it += 1
        t0 = time.time()
        if Sm:
            gam = ((Sm[-1] @ Ym[-1]) / (Ym[-1] @ Ym[-1])).item()
            with torch.set_grad_enabled(opt is not None):
                p_g = two_loop(g, Sm, Ym, lambda q: gam * G(q))
            p = p_g.detach()
        else:
            p_g, p = None, -g / g.norm() * step0
        pn = p.norm()
        if pn > step_max:
            p = p / pn * step_max
            if p_g is not None:
                p_g = p_g * (step_max / pn)
        if g @ p >= 0:
            Sm, Ym = [], []; p_g = None; p = -g / g.norm() * step0
        t_meta += time.time() - t0
        alpha, ok = 1.0, False
        for _ in range(max_ls):
            if solver.n_sims >= budget:
                break
            J_try, h = evaluate(x + alpha * p)
            if J_try <= J + c1 * alpha * (g @ p).item():
                ok = True; break
            del h; alpha *= 0.5
        if not ok:
            Sm, Ym = [], []
            if solver.n_sims >= budget:
                break
            continue
        x_new = x + alpha * p
        log.append(dict(it=it, sims=solver.n_sims, J=J_try, alpha=alpha, err=T.rel_err(param.model(x_new))))
        if solver.n_sims >= budget:
            x, J = x_new, J_try; break
        g_new = gradient(h, x_new); del h
        t0 = time.time()
        if opt is not None and p_g is not None:   # ∇_θ f(x + α p(θ)) = α (∂p/∂θ)ᵀ ∇f(x_new)
            opt.zero_grad()
            (alpha * (p_g * g_new).sum()).backward()
            opt.step(); n_meta += 1
        t_meta += time.time() - t0
        s_, y_ = x_new - x, g_new - g
        if (s_ @ y_) > 1e-10 * s_.norm() * y_.norm():
            Sm.append(s_); Ym.append(y_)
            if len(Sm) > mem:
                Sm.pop(0); Ym.pop(0)
        x, g, J = x_new, g_new, J_try
    with torch.no_grad():
        m = param.model(x)
    return dict(label=label, m=m, x=x, log=log, solver=solver, err=T.rel_err(m), J=J, meta_time=t_meta,
                n_meta=n_meta)
