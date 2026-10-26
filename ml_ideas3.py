"""Tercera ronda: estrategias PIML combinables con la extrapolación de bajas frecuencias (ver `ml_ideas3_fwi.ipynb`).

1. Extrapolación de bajas frecuencias **adaptada con física**: una red preentrenada (posiblemente fuera de
   distribución) se ajusta durante la inversión con pares (banda alta → banda baja) generados por los forwards de
   banda ancha que la propia FWI ya ejecuta en el modelo actual (geometría, ondícula y geología cercanas a las
   reales, sin simulaciones extra). L2-SP (Li et al. 2018) ancla los pesos a los preentrenados para limitar la
   autoconfirmación. Variante opcional: re-ponderar trazas según su consistencia física ‖F_baja(m) − d̂_baja‖.
2. **Reparametrización neuronal** del modelo (red de coordenadas con Fourier features, Tancik et al. 2020;
   en FWI: Sun et al. 2023, "implicit FWI"): m = m_base · (1 + mask · δ_θ(x, z)). El gradiente adjunto se
   propaga a θ por autograd; el sesgo espectral de la red actúa como regularización sin entrenamiento previo.
"""
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.func import functional_call, jvp

import fwi_toy as T
import ml_ideas2 as M2
from fwi_toy import dev, DT

CACHE = M2.CACHE


# ================================================================================================================
# 2. Reparametrización neuronal (INR con Fourier features)
# ================================================================================================================
class INRNet(nn.Module):
    def __init__(self, sigma, nf=64, width=128, depth=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(2, nf, generator=g) * sigma)     # ciclos/km
        layers, d_in = [], 2 * nf
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.GELU()]; d_in = width
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)       # m(θ0) = m_base

    def forward(self, xy):
        z = 2 * math.pi * xy @ self.B
        return self.out(self.mlp(torch.cat([torch.sin(z), torch.cos(z)], -1)))[:, 0]


class INRParam:
    """m = base · (1 + mask · amp · tanh(NN_θ(x, z))).  x = θ aplanado; step_norm = cambio relativo RMS de m
    (linealizado con un JVP), la misma métrica que ‖x̂‖ en HatParam ⇒ mismos step0/step_max de run_lbfgs."""

    def __init__(self, base=None, sigma=3.0, amp=0.4, seed=0, init_std=0.0, **kw):
        torch.manual_seed(seed)
        self.base = T.v_init if base is None else base
        self.net = INRNet(sigma, seed=seed, **kw).to(dev)
        self.amp = amp
        xy = torch.stack([torch.tensor(T.xx, dtype=DT), torch.tensor(T.zz, dtype=DT)], -1).view(-1, 2)
        self.xy = xy.to(dev)
        if init_std > 0:            # última capa aleatoria pequeña (todas las capas reciben gradiente), m(θ0) ≈ base
            with torch.no_grad():
                self.net.out.weight.normal_(0, init_std)
                self.net.out.bias.fill_(-self.net(self.xy).mean())
        self.names = [n for n, _ in self.net.named_parameters()]
        self.shapes = [p.shape for _, p in self.net.named_parameters()]
        self.sizes = [p.numel() for _, p in self.net.named_parameters()]
        self.x0 = torch.cat([p.detach().flatten() for p in self.net.parameters()])

    def _params(self, x):
        return {n: t.view(s) for n, t, s in zip(self.names, torch.split(x, self.sizes), self.shapes)}

    def model(self, x):
        d = functional_call(self.net, self._params(x), (self.xy,)).view(T.nz, T.nx)
        return self.base * (1 + T.mask * self.amp * torch.tanh(d))

    def pullback(self, x, gm):
        x_ = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(self.model(x_), x_, gm)
        return g

    def step_norm(self, x, p):
        _, dm = jvp(self.model, (x.detach(),), (p.detach(),))
        return ((dm / self.base) ** 2).sum().div(T.n_act).sqrt()


class INRProjector:
    """Π(m): ajuste de una INR (sin entrenamiento previo) a m con un número fijo de pasos de Adam desde la misma
    inicialización ⇒ proyección determinista sobre lo que la red representa "pronto" (sesgo espectral)."""

    def __init__(self, sigma=3.0, steps=300, lr=2e-3, seed=0, width=128, depth=3):
        self.P = INRParam(sigma=sigma, seed=seed, width=width, depth=depth)
        with torch.no_grad():
            self.P.net.out.weight.normal_(0, 1e-2)
        self.x0 = torch.cat([p.detach().flatten() for p in self.P.net.parameters()])
        self.steps, self.lr = steps, lr

    def __call__(self, m):
        x = self.x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=self.lr)
        tgt = m.detach()
        for _ in range(self.steps):
            loss = (((self.P.model(x) - tgt) / self.P.base) ** 2 * T.mask).sum() / T.n_act
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return self.P.model(x)


def reg_inr(lam, sigma=3.0, steps=300):
    """RED con prior INR no entrenado: R(m) = ½λ Σ_mask (m − Π(m))² / n_act, con Π fijo en el gradiente
    (∇R ≈ λ(m − Π(m)), aproximación RED). Misma escala de λ que `ml_ideas2.reg_denoiser`."""
    proj = INRProjector(sigma=sigma, steps=steps)

    def R(m):
        return 0.5 * lam * (((m - proj(m)) ** 2) * T.mask).sum() / T.n_act
    return R


# ================================================================================================================
# 1. Extrapolación de bajas frecuencias: red "genérica" fuera de distribución + adaptación online
# ================================================================================================================
GEN_F0 = 6.0          # ondícula del preentrenamiento genérico (los datos usan 5 Hz)


def random_model_1d(rng):
    """Geología genérica: solo capas planas sobre un gradiente (sin estructura lateral, sin anomalías)."""
    z = T.zz[:, 0]
    v = rng.uniform(1.5, 1.9) + rng.uniform(0.3, 1.1) * z / z.max()
    th = rng.uniform(0.08, 0.3)
    steps = rng.uniform(-0.1, 0.2, size=30)
    v = v + (np.cumsum(steps) - steps[0])[np.clip((z / th).astype(int), 0, 29)]
    v = gaussian_filter(v, rng.uniform(0.0, 1.5), mode="nearest")
    return np.clip(np.repeat(v[:, None], T.nx, 1), 1.4, 3.0).astype(np.float32)


def _grid_with_wavelet(f0):
    g = copy.copy(T.fine)
    tt = np.arange(T.nt) * T.dt
    g.wavelet = torch.tensor(T.ricker(f0, tt, 1.2 / f0), dtype=DT, device=dev)
    return g


def _pairs(d, noise_std=0.0, gen=None):
    """Sintético de banda ancha d (ns, nt, nrec) → (X normalizado banda alta [+ ruido], Y banda baja, offsets)."""
    dn = d if noise_std == 0 else d + noise_std * torch.randn(d.shape, generator=gen).to(dev)
    hp, lp = M2._traces(M2.band(dn, "hp")), M2._traces(M2.band(d, "lp"))
    s = hp.abs().amax(1, keepdim=True) + 1e-8
    return hp / s, lp / s, M2._offsets().flatten() / 2.0


def _fit(net, X, Y, O, steps, lr, bs=256, anchor=None, mu=0.0, seed=0, sched=True):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if sched else None
    g = torch.Generator().manual_seed(seed)
    net.train()
    for it in range(steps):
        idx = torch.randint(0, X.shape[0], (bs,), generator=g)
        x, y, o = X[idx].to(dev).float(), Y[idx].to(dev).float(), O[idx].to(dev)
        loss = ((net(x, o) - y) ** 2).mean() / (y ** 2).mean().clamp_min(1e-8)
        if anchor is not None and mu > 0:
            loss = loss + mu * sum(((p - a) ** 2).sum() for p, a in zip(net.parameters(), anchor))
        opt.zero_grad(); loss.backward(); opt.step()
        if sch:
            sch.step()
    net.eval()
    return loss.item()


def train_lowfreq_generic(n_models=150, steps=4000, force=False, verbose=True):
    """Red preentrenada en otra 'familia': geología 1D y ondícula Ricker de GEN_F0 Hz (no la de los datos)."""
    path = f"{CACHE}/lowfreq_gen_{T.ACQ}.pt"
    net = M2.LowFreqNet().to(dev)
    if os.path.exists(path) and not force:
        net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
        return net
    t0 = time.time()
    rng = np.random.default_rng(11)
    grid = _grid_with_wavelet(GEN_F0)
    X, Y = [], []
    with torch.no_grad():
        for _ in range(n_models):
            d = grid.simulate(torch.tensor(random_model_1d(rng), device=dev)) / T.DSCALE
            x, y, _ = _pairs(d)
            X.append(x.half().cpu()); Y.append(y.half().cpu())
    X, Y = torch.cat(X), torch.cat(Y)
    O = (M2._offsets().flatten() / 2.0).repeat(n_models).cpu()
    if verbose:
        print(f"[gen] {X.shape[0]} trazas ({time.time() - t0:.0f}s)", flush=True)
    _fit(net, X, Y, O, steps, 1e-3)
    torch.save(net.state_dict(), path)
    if verbose:
        print(f"[gen] red genérica entrenada en {time.time() - t0:.0f}s", flush=True)
    return net


def adapt_lowfreq(net, net0, synth, steps=300, lr=3e-4, mu=1e-4, seed=0):
    """Ajuste fino con los sintéticos de banda ancha de la inversión (lista de (ns, nt, nrec)). L2-SP hacia net0.
    El ruido de los datos se imita con el mismo nivel (se supone estimado, p. ej. de ventanas previas al primer arribo)."""
    gen = torch.Generator().manual_seed(seed)
    P = [_pairs(d, T.NOISE_STD, gen) for d in synth]
    X = torch.cat([p[0] for p in P]); Y = torch.cat([p[1] for p in P]); O = torch.cat([p[2] for p in P])
    anchor = [p.detach().clone() for p in net0.parameters()] if net0 is not None else None
    return _fit(net, X, Y, O, steps, lr, anchor=anchor, mu=mu, seed=seed)


class RecordingSolver(T.ExpensiveSolver):
    """Guarda (m, d_banda_ancha) de cada forward: son datos de entrenamiento gratis para la adaptación."""

    def __init__(self, *a, keep=12, **kw):
        super().__init__(*a, **kw)
        self.mem, self.keep = [], keep

    def forward(self, m, keep=False):
        J, d, h = super().forward(m, keep)
        self.mem.append((m.detach().clone(), d.detach()))
        if len(self.mem) > self.keep:
            self.mem.pop(0)
        return J, d, h


def _w_l2_lp(w):
    return lambda d, do: 0.5 * (w[:, None, :] * (M2.band(d, "lp") - do) ** 2).sum()


def run_ms_adaptive(net0, label, mode="adapt", n_low=40, n_adapt=3, budget=100, low_param=None, high_reg=None,
                    weight=False, steps0=600, steps=300, mu=1e-4):
    """Multiescala con extrapolación de bajas frecuencias.
    mode: "none" (red fija, 1 etapa baja) | "adapt" (ajuste fino de net0 con L2-SP) | "self" (red desde cero,
          entrenada solo con los sintéticos de la inversión).
    low_param: callable(base) -> parametrización de la etapa baja (p. ej. INRParam); por defecto HatParam.
    high_reg: regularizador para la etapa de banda alta (p. ej. denoiser).
    weight: re-ponderar trazas por consistencia física con el modelo actual."""
    d_hp = M2.band(T.d_obs, "hp")
    true_lp = M2.band(T.d_obs_clean, "lp")                          # solo diagnóstico (no lo usa el método)
    solver = RecordingSolver(torch.zeros_like(d_hp), misfit=M2.l2_band("lp"))
    if mode == "self":
        net = M2.LowFreqNet().to(dev); net0 = None
    else:
        net = copy.deepcopy(net0)
    param = (low_param or T.HatParam)(T.v_init)
    x, logs, diag = None, [], []
    chunks = [n_low] if mode == "none" else list(np.linspace(0, n_low, n_adapt + 1)[1:])
    w = torch.ones(T.ns, T.nrec, device=dev)
    for c, end in enumerate(chunks):
        m_cur = param.model(x if x is not None else param.x0).detach()
        if mode != "none":
            if c == 0:
                solver.forward(m_cur)                               # 1 simulación: sintético en m_0
            adapt_lowfreq(net, net0, [d for _, d in solver.mem], steps=steps0 if c == 0 else steps, mu=mu,
                          seed=c)
        pred = M2.predict_lowfreq(net, d_hp)
        if weight and solver.mem:                                   # consistencia con el último sintético en m_cur
            i = int(np.argmin([(mm - m_cur).norm().item() for mm, _ in solver.mem]))
            r = ((M2.band(solver.mem[i][1], "lp") - pred) ** 2).sum(1)
            rr = r / (pred ** 2).sum(1).clamp_min(1e-12)
            w = 1.0 / (1.0 + (rr / rr.median()) ** 2)
            w = w / w.mean()
        solver.d_obs = pred
        solver.misfit = _w_l2_lp(w) if weight else M2.l2_band("lp")
        R = T.run_lbfgs(end, solver=solver, x0=x, param=param, label=label)
        x = R["x"]; logs += R["log"]
        with torch.no_grad():
            dm = T.simulate(R["m"].clamp(T.VMIN, T.VMAX)) / T.DSCALE           # diagnóstico, no cuenta
            cons = ((M2.band(dm, "lp") - pred).norm() / pred.norm()).item()
        diag.append(dict(sims=solver.n_sims, pred_err=((pred - true_lp).norm() / true_lp.norm()).item(),
                         cons=cons, err=R["err"]))
        print(f"[{label}] etapa baja {c}: sims {solver.n_sims:.0f} err pred {diag[-1]['pred_err']:.3f} "
              f"consistencia {cons:.3f} err modelo {R['err']:.4f}", flush=True)
    m_low = R["m"].detach()
    s_hi = T.ExpensiveSolver(d_hp, misfit=M2.l2_band("hp"))
    s_hi.n_fwd, s_hi.n_adj, s_hi.n_born, s_hi.trace = solver.n_fwd, solver.n_adj, solver.n_born, solver.trace
    R = T.run_lbfgs(budget, solver=s_hi, param=T.HatParam(m_low), reg=high_reg, label=label)
    R["log"] = logs + R["log"]; R["label"] = label
    R["lowfreq"] = dict(rel_err=diag[-1]["pred_err"], diag=diag, pred=pred[2].cpu(), true=true_lp[2].cpu(),
                        hp=d_hp[2].cpu(), m_low=m_low.cpu())
    return R
