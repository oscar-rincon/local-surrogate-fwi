"""Segunda ronda de ideas de ML para FWI (ver `ml_ideas2_fwi.ipynb`).

A. Precondicionador: control clásico (pseudo-Hessiano de Shin) y P aprendido con productos H·v exactos
   (este último en `ml_ideas.LearnedPrecond.update(..., V, HV)` + `fwi_toy.run_lbfgs(hv_every=...)`).
B. Regularización aprendida: denoiser CNN entrenado offline con modelos geológicos aleatorios, usado como
   penalización ½λ‖m − D(m)‖² (gradiente exacto por autograd, sin simulaciones). Controles: Tikhonov y TV.
C. Cycle skipping: extrapolación de bajas frecuencias con una CNN 1D por traza (Sun & Demanet 2020) entrenada
   offline; control: misfit de transporte óptimo W1 por traza (Yang & Engquist 2018) y oráculo con bajas reales.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.ndimage import gaussian_filter

import fwi_toy as T
from fwi_toy import dev, DT

CACHE = "results/models_ml"
os.makedirs(CACHE, exist_ok=True)


# ================================================================================================================
# A. Pseudo-Hessiano (Shin et al. 2001):  h(x) = Σ_s Σ_t (∂²_t u_s(x,t))²
#    En SPECFEM se acumula durante la simulación forward ⇒ no cuesta simulaciones extra.
# ================================================================================================================
@torch.no_grad()
def pseudo_hessian(v):
    g = T.fine; p = g.npml
    vp = Fnn.pad(v[None, None], (p,) * 4, mode="replicate")[0, 0]
    c2 = (vp * g.dt / g.dx) ** 2
    u0 = torch.zeros(T.ns, g.NZ, g.NX, dtype=DT, device=dev); u1 = torch.zeros_like(u0)
    acc = torch.zeros(g.NZ, g.NX, dtype=DT, device=dev)
    for it in range(g.nt):
        u2 = (2 * u1 - g.B * u0 + c2 * T.lap4(u1) + (g.dt ** 2) * g.wavelet[it] * g.S) / g.A
        acc += (((u2 - 2 * u1 + u0) / g.dt ** 2) ** 2).sum(0)
        u0, u1 = u1, u2
    h = acc[p:p + T.nz, p:p + T.nx]
    return h / h.max()


class DiagPrecond:
    """H0 = γ diag(1 / (h_hat + ε max h_hat)), con h_hat ≈ diagonal del pseudo-Hessiano en coordenadas 'hat'."""

    def __init__(self, h, eps=0.05):
        hh = T.smooth(h * T.Dsc ** 2).flatten()
        self.w = 1.0 / (hh + eps * hh.max())
        self.w = self.w / self.w.mean()
        self.gamma = None

    def update(self, S, Y, *a):
        s, y = S[-1], Y[-1]
        self.gamma = ((s @ y) / (y @ (self.w * y))).item()

    @torch.no_grad()
    def apply(self, q):
        return self.gamma * self.w * q


# ================================================================================================================
# Generador de modelos geológicos aleatorios (prior para B y C). Capas buzantes, fallas, lentes, anomalías
# gaussianas y capas delgadas sobre un gradiente lineal. Comparte la "familia" de los escenarios de prueba (el
# prior aprendido es justamente la información que el método aporta), pero nunca contiene los modelos de prueba.
# ================================================================================================================
def random_model(rng):
    zz, xx = T.zz, T.xx
    zmax = zz.max()
    v = rng.uniform(1.5, 1.8) + rng.uniform(0.4, 1.0) * zz / zmax
    if rng.random() < 0.7:                               # capas buzantes (+ falla)
        dip = rng.uniform(-0.25, 0.25)
        zt = zz - dip * xx
        if rng.random() < 0.6:
            xf = rng.uniform(0.4, 1.6); zt = zt + rng.uniform(-0.12, 0.12) * (xx > xf + rng.uniform(-0.2, 0.2) * zz)
        th = rng.uniform(0.1, 0.25)
        steps = rng.uniform(-0.08, 0.18, size=20)
        idx = np.clip(np.floor((zt - zt.min()) / th).astype(int), 0, 19)
        cs = np.cumsum(steps)
        v = v + cs[idx] - cs[0]
    for _ in range(rng.integers(0, 4)):                   # anomalías / lentes
        x0, z0 = rng.uniform(0.2, 1.8), rng.uniform(0.2, 1.0)
        sx, sz = rng.uniform(0.05, 0.2), rng.uniform(0.04, 0.15)
        v = v + rng.uniform(-0.35, 0.4) * np.exp(-(xx - x0) ** 2 / (2 * sx ** 2) - (zz - z0) ** 2 / (2 * sz ** 2))
    if rng.random() < 0.35:                               # capa delgada
        z0, h = rng.uniform(0.3, 1.05), rng.uniform(0.03, 0.08)
        xa = rng.uniform(0.0, 0.8); xb = rng.uniform(1.2, 2.0)
        v = v + rng.uniform(-0.2, 0.2) * (np.abs(zz - z0) < h) * (xx > xa) * (xx < xb)
    v = gaussian_filter(v, rng.uniform(0.0, 1.5), mode="nearest")
    return np.clip(v, 1.4, 3.0).astype(np.float32)


def model_bank(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([random_model(rng) for _ in range(n)])


# ================================================================================================================
# B. Denoiser de modelos de velocidad (residual CNN): aprende a quitar ruido correlacionado y a re-enfocar
#    modelos desenfocados (los artefactos típicos de FWI con mala iluminación).
# ================================================================================================================
class Denoiser(nn.Module):
    def __init__(self, ch=48, depth=8):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch), nn.ReLU()]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, v):                                 # v: (B, 1, nz, nx) en km/s
        x = (v - 2.2) / 0.4
        return v - 0.4 * self.net(x)


def _gauss_blur(x, sigma):
    if sigma < 0.3:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=dev, dtype=DT) ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    x = Fnn.conv2d(Fnn.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return Fnn.conv2d(Fnn.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def train_denoiser(n_models=6000, steps=12000, bs=32, force=False, verbose=True):
    path = f"{CACHE}/denoiser.pt"
    D = Denoiser().to(dev)
    if os.path.exists(path) and not force:
        D.load_state_dict(torch.load(path, map_location=dev)); D.eval()
        return D
    t0 = time.time()
    bank = torch.tensor(model_bank(n_models, seed=1), device=dev)[:, None]
    opt = torch.optim.Adam(D.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=dev).manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, n_models, (bs,), device=dev, generator=gen)
        clean = bank[idx]
        noisy = _gauss_blur(clean, float(np.random.uniform(0, 2.5)))
        nse = _gauss_blur(torch.randn(clean.shape, device=dev, generator=gen), float(np.random.uniform(0, 3)))
        nse = nse / nse.std(dim=(2, 3), keepdim=True) * torch.rand(bs, 1, 1, 1, device=dev, generator=gen) * 0.06
        noisy = noisy + nse
        keep = torch.rand(bs, 1, 1, 1, device=dev, generator=gen) < 0.2      # 20 % limpio: D(m) ≈ m en la prior
        noisy = torch.where(keep, clean, noisy)
        loss = ((D(noisy) - clean) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 3000 == 0:
            print(f"denoiser it {it} loss {loss.item():.2e}", flush=True)
    D.eval()
    torch.save(D.state_dict(), path)
    if verbose:
        print(f"denoiser entrenado en {time.time() - t0:.0f}s")
    return D


def reg_denoiser(D, lam):
    """R(m) = ½ λ Σ_mask (m − D(m))² / n_act  (gradiente exacto por autograd)."""
    for p in D.parameters():
        p.requires_grad_(False)

    def R(m):
        return 0.5 * lam * (((m - D(m[None, None])[0, 0]) ** 2) * T.mask).sum() / T.n_act
    return R


def _grad2(m):
    gz = m[1:, :-1] - m[:-1, :-1]; gx = m[:-1, 1:] - m[:-1, :-1]
    return gz, gx, T.mask[:-1, :-1]


def reg_tikhonov(lam):
    def R(m):
        gz, gx, w = _grad2(m)
        return 0.5 * lam * ((gz ** 2 + gx ** 2) * w).sum() / T.n_act
    return R


def reg_tv(lam, eps=1e-3):
    def R(m):
        gz, gx, w = _grad2(m)
        return lam * (torch.sqrt(gz ** 2 + gx ** 2 + eps ** 2) * w).sum() / T.n_act
    return R


# ================================================================================================================
# C. Bandas de frecuencia, extrapolación de bajas frecuencias y misfit W1
# ================================================================================================================
F_HP, F_LP = float(os.environ.get("FWI_FHP", "4.0")), 2.0          # datos "registrados": > F_HP Hz;  banda baja objetivo: < F_LP Hz


def _bw(fc, n=6, high=False):
    f = torch.fft.rfftfreq(2 * T.nt, T.dt).to(dev)
    lp = 1.0 / torch.sqrt(1 + (f / fc) ** (2 * n))
    return (torch.sqrt(1 - lp ** 2) if high else lp).to(DT)


_HHP, _HLP = _bw(F_HP, high=True), _bw(F_LP)


def band(d, which):
    """Filtro de fase cero en el eje temporal (dim 1) de d: (ns, nt, nrec). Diferenciable."""
    H = _HHP if which == "hp" else _HLP
    Dd = torch.fft.rfft(d, n=2 * T.nt, dim=1)
    return torch.fft.irfft(Dd * H[None, :, None], n=2 * T.nt, dim=1)[:, :T.nt]


def l2_band(which):
    return lambda d, do: 0.5 * ((band(d, which) - do) ** 2).sum()


def w1_misfit(scale=1.0):
    """W1 por traza con normalización lineal: f̃ = (f + c)/Σ(f + c), c = 3 max|d_obs| de cada traza.
    Se compara la banda alta de los sintéticos con los datos observados (ya en banda alta)."""
    def M(d, do):
        f = band(d, "hp")
        c = 3 * do.abs().amax(1, keepdim=True) + 1e-8
        fp = torch.clamp(f + c, min=1e-10); gp = do + c
        Ff = torch.cumsum(fp / fp.sum(1, keepdim=True), 1)
        Fg = torch.cumsum(gp / gp.sum(1, keepdim=True), 1)
        return scale * (Ff - Fg).abs().sum() * T.dt
    return M


class LowFreqNet(nn.Module):
    """CNN 1D dilatada: traza en banda alta (normalizada) + offset → traza en banda baja."""

    def __init__(self, ch=64, dil=(1, 2, 4, 8, 16, 32, 64, 128, 1, 4, 16, 64)):
        super().__init__()
        self.inp = nn.Conv1d(2, ch, 7, padding=3)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dil])
        self.out = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x, off):                            # x: (B, nt), off: (B,)
        h = torch.relu(self.inp(torch.stack([x, off[:, None].expand_as(x)], 1)))
        for b in self.blocks:
            h = h + torch.relu(b(h))
        return self.out(h)[:, 0]


def _offsets():
    rx = np.arange(T.nx) * T.dx; rz = np.full(T.nx, T.rec_z * T.dx)
    if T.ACQ == "wells":
        rx = np.concatenate([rx, np.full(T.nz, T.well_l * T.dx), np.full(T.nz, T.well_r * T.dx)])
        rz = np.concatenate([rz, np.arange(T.nz) * T.dx, np.arange(T.nz) * T.dx])
    off = np.hypot(rx[None, :] - (T.src_x * T.dx)[:, None], rz[None, :] - T.src_z * T.dx)
    return torch.tensor(off, dtype=DT, device=dev)       # (ns, nrec)


def _traces(d):
    """(ns, nt, nrec) -> (ns*nrec, nt)"""
    return d.permute(0, 2, 1).reshape(-1, T.nt)


def train_lowfreq(n_models=300, steps=6000, bs=256, force=False, verbose=True):
    tag = f"lowfreq_{T.ACQ}"
    path = f"{CACHE}/{tag}.pt"
    net = LowFreqNet().to(dev)
    if os.path.exists(path) and not force:
        net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
        return net
    t0 = time.time()
    bank = model_bank(n_models, seed=7)
    off = (_offsets().flatten() / 2.0)
    X, Y = [], []
    with torch.no_grad():
        for v in bank:
            d = T.simulate(torch.tensor(v, device=dev)) / T.DSCALE
            hp, lp = _traces(band(d, "hp")), _traces(band(d, "lp"))
            s = hp.abs().amax(1, keepdim=True) + 1e-8
            X.append((hp / s).half().cpu()); Y.append((lp / s).half().cpu())
    X, Y = torch.cat(X), torch.cat(Y)
    O = off.repeat(n_models).cpu()
    if verbose:
        print(f"datos de entrenamiento: {X.shape[0]} trazas ({time.time() - t0:.0f}s)", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for it in range(steps):
        idx = torch.randint(0, X.shape[0], (bs,), generator=g)
        x, y, o = X[idx].to(dev).float(), Y[idx].to(dev).float(), O[idx].to(dev)
        loss = ((net(x, o) - y) ** 2).mean() / (y ** 2).mean().clamp_min(1e-8)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 1000 == 0:
            print(f"lowfreq it {it} rel loss {loss.item():.3f}", flush=True)
    net.eval()
    torch.save(net.state_dict(), path)
    if verbose:
        print(f"red de bajas frecuencias entrenada en {time.time() - t0:.0f}s")
    return net


@torch.no_grad()
def predict_lowfreq(net, d_hp):
    tr = _traces(d_hp); s = tr.abs().amax(1, keepdim=True) + 1e-8
    off = (_offsets().flatten() / 2.0)
    out = torch.cat([net(tr[i:i + 2048] / s[i:i + 2048], off[i:i + 2048]) for i in range(0, tr.shape[0], 2048)])
    out = out * s
    return out.view(T.ns, T.nrec, T.nt).permute(0, 2, 1).contiguous()


def run_multiscale(stages, label):
    """stages: lista de (presupuesto_acumulado, misfit, d_obs_banda, kwargs de run_lbfgs)."""
    solver = None; x = None; logs = []
    for budget, misfit, dob, kw in stages:
        s_new = T.ExpensiveSolver(dob, misfit=misfit)
        if solver is not None:                            # el contador es acumulado
            s_new.n_fwd, s_new.n_adj, s_new.n_born = solver.n_fwd, solver.n_adj, solver.n_born
            s_new.trace = solver.trace
        solver = s_new
        R = T.run_lbfgs(budget, solver=solver, x0=x, label=label, **kw)
        x = R["x"]; logs += R["log"]
    R["log"] = logs; R["label"] = label
    return R
