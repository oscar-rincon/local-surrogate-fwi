"""Misfit de desfase temporal aprendido (Alfarhan et al., Geophys. Prospect. 2026, doi:10.1111/1365-2478.70160;
arXiv:2511.08134) y un control clásico de correlación cruzada (Luo & Schuster, 1991).

Red: U-Net 1D (kernel 11, sin la primera conexión de salto, capa final de suavizado fija) que recibe
(traza modelada, traza observada) y predice τ(t) tal que d_obs(t) ≈ d_mod(t − τ(t)). Se entrena de forma supervisada
con desfases sintéticos (suma de gaussianas) aplicados a los datos modelados en el modelo inicial y se ajusta
finamente con el par real (modelado, observado) usando solo la pérdida de datos. Misfit FWI: J = ½ Σ ‖τ_p‖².
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

import fwi_toy as T
import ml_ideas2 as M2

dev, DT = T.dev, T.DT
NT_PAD = int(math.ceil(T.nt / 16) * 16)


# ----------------------------------------------------------------------------------------------------------------
# Utilidades de trazas
def normalize(x, eps=1e-8):
    """Normaliza cada traza (B, nt) por su RMS. Diferenciable."""
    return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + eps)


def warp(d, tau):
    """d2(t) = d(t − τ(t)) con interpolación lineal; d, τ: (B, nt) en muestras. Diferenciable en d y τ."""
    n = d.shape[-1]
    pos = (torch.arange(n, device=d.device, dtype=d.dtype)[None] - tau).clamp(0, n - 1)
    i0 = pos.floor().long().clamp(max=n - 2)
    w = pos - i0
    return (1 - w) * d.gather(-1, i0) + w * d.gather(-1, i0 + 1)


def support(d, rel=0.02, width=15):
    """Máscara 0/1 donde la envolvente (máx. móvil de |d|) supera rel·máx de la traza."""
    env = Fnn.max_pool1d(d.abs()[:, None], width, 1, width // 2)[:, 0]
    return (env > rel * env.amax(-1, keepdim=True).clamp_min(1e-12)).to(d.dtype)


def random_shifts(B, n, gen, n_gauss=(3, 5), amp=60.0, sig=(40.0, 125.0), max_slope=0.7):
    """τ(t) = Σ αᵢ exp(−(t−μᵢ)²/2σᵢ²) (Ec. 1 del artículo); μ ∈ [0.2, 0.8]·n, |αᵢ| ≤ amp muestras.
    Se reescala para que |τ'| ≤ max_slope (el mapeo t → t − τ(t) sigue siendo monótono)."""
    t = torch.arange(n, device=dev, dtype=DT)[None, None]
    k = torch.randint(n_gauss[0], n_gauss[1] + 1, (B, 1), generator=gen).to(dev)
    on = (torch.arange(n_gauss[1])[None] < k.cpu()).to(dev, DT)[..., None]
    mu = (0.2 + 0.6 * torch.rand(B, n_gauss[1], 1, generator=gen)).to(dev) * n
    s = (sig[0] + (sig[1] - sig[0]) * torch.rand(B, n_gauss[1], 1, generator=gen)).to(dev)
    a = (amp * (2 * torch.rand(B, n_gauss[1], 1, generator=gen) - 1)).to(dev)
    tau = (on * a * torch.exp(-(t - mu) ** 2 / (2 * s ** 2))).sum(1)
    slope = (tau[:, 1:] - tau[:, :-1]).abs().amax(-1, keepdim=True)
    return tau * torch.clamp(max_slope / slope.clamp_min(1e-8), max=1.0)


# ----------------------------------------------------------------------------------------------------------------
# U-Net 1D
class _Block(nn.Sequential):
    def __init__(self, cin, cout, k, p):
        super().__init__(nn.Conv1d(cin, cout, k, padding=k // 2), nn.Dropout(p), nn.ReLU(),
                         nn.Conv1d(cout, cout, k, padding=k // 2), nn.Dropout(p), nn.ReLU())


class TimeShiftUNet(nn.Module):
    """U-Net 1D con las tres modificaciones del artículo: kernel 11, sin la primera conexión de salto y capa final
    de suavizado gaussiano fija (no entrenable)."""

    def __init__(self, ch=(16, 32, 64, 128), k=11, p=0.1, smooth_sigma=4.0):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 2
        for c in ch:
            self.enc.append(_Block(cin, c, k, p)); cin = c
        self.mid = _Block(ch[-1], 2 * ch[-1], k, p)
        self.dec = nn.ModuleList()
        cin = 2 * ch[-1]
        for i, c in reversed(list(enumerate(ch))):
            skip = c if i > 0 else 0                      # sin la primera conexión de salto
            self.dec.append(_Block(cin + skip, c, k, p)); cin = c
        self.out = nn.Conv1d(ch[0], 1, 1)
        r = int(3 * smooth_sigma)
        g = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * smooth_sigma ** 2))
        self.register_buffer("smooth", (g / g.sum())[None, None])

    def forward(self, d1, d2):
        """d1 (modelado), d2 (observado): (B, nt) ya normalizados. Devuelve τ (B, nt) en muestras."""
        n = d1.shape[-1]
        x = Fnn.pad(torch.stack([d1, d2], 1), (0, NT_PAD - n))
        skips = []
        for b in self.enc:
            x = b(x); skips.append(x); x = Fnn.max_pool1d(x, 2)
        x = self.mid(x)
        for i, b in enumerate(self.dec):
            lvl = len(self.enc) - 1 - i
            x = Fnn.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
            x = b(torch.cat([x, skips[lvl]], 1) if lvl > 0 else x)
        x = self.out(x)
        x = Fnn.conv1d(Fnn.pad(x, (self.smooth.shape[-1] // 2,) * 2, mode="replicate"), self.smooth)
        return x[:, 0, :n]


def train_timeshift(d_mod, d_obs, steps=3000, ft_steps=300, bs=256, lr=1e-3, ft_lr=1e-5, mu=1.0, noise=0.05,
                    seed=0, verbose=True):
    """d_mod, d_obs: (ns, nt, nrec) en banda alta. Etapa 1: desfases sintéticos sobre d_mod (pérdida MAE de τ +
    μ·MSE de datos). Etapa 2 (ft_steps > 0): ajuste fino con los pares reales (d_mod, d_obs), solo MSE de datos."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    X = normalize(M2._traces(d_mod).detach())
    X = X[X.abs().amax(-1) > 0]
    Xo = normalize(M2._traces(d_obs).detach())
    Xm = normalize(M2._traces(d_mod).detach())
    net = TimeShiftUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time(); hist = []
    net.train()
    for it in range(steps):
        idx = torch.randint(0, X.shape[0], (bs,), generator=gen).to(dev)
        d1 = X[idx]
        tau = random_shifts(bs, d1.shape[-1], gen)
        d2 = warp(d1, tau)
        lab = tau * torch.clamp(support(d1) + support(d2), max=1.0)
        sc = noise * torch.rand(bs, 1, generator=gen).to(dev)
        d2n = d2 + sc * torch.randn(d2.shape, generator=gen).to(dev)
        tp = net(d1, d2n)
        l_tau = (tp - lab).abs().mean()
        l_dat = (warp(d1, tp) - d2).pow(2).mean()
        loss = l_tau + mu * l_dat
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0 or it == steps - 1:
            hist.append(dict(stage="pre", it=it, l_tau=l_tau.item(), l_dat=l_dat.item()))
            if verbose:
                print(f"[ts-net] it {it:5d} MAE τ {l_tau.item():.2f} muestras  MSE datos {l_dat.item():.4f}",
                      flush=True)
    if ft_steps > 0:
        keep = (Xo.abs().amax(-1) > 0) & (Xm.abs().amax(-1) > 0)
        Xm_, Xo_ = Xm[keep], Xo[keep]
        opt = torch.optim.Adam(net.parameters(), lr=ft_lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ft_steps)
        for it in range(ft_steps):
            idx = torch.randint(0, Xm_.shape[0], (bs,), generator=gen).to(dev)
            tp = net(Xm_[idx], Xo_[idx])
            loss = (warp(Xm_[idx], tp) - Xo_[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if it % 100 == 0 or it == ft_steps - 1:
                hist.append(dict(stage="ft", it=it, l_dat=loss.item()))
                if verbose:
                    print(f"[ts-net] ajuste fino it {it:4d} MSE datos {loss.item():.4f}", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, dict(history=hist, train_time=time.time() - t0)


def ts_misfit(net, chunk=2048):
    """J(d) = ½ Σ ‖τ_p(band_hp(d), d_obs)‖² (τ en muestras). d_obs ya está en banda alta."""
    def M(d, do):
        a, b = normalize(M2._traces(M2.band(d, "hp"))), normalize(M2._traces(do))
        return sum(0.5 * net(a[i:i + chunk], b[i:i + chunk]).pow(2).sum() for i in range(0, a.shape[0], chunk))
    return M


@torch.no_grad()
def predict_tau(net, d_mod, d_obs):
    """τ_p para todas las trazas: (ns*nrec, nt)."""
    return net(normalize(M2._traces(d_mod)), normalize(M2._traces(d_obs)))


# ----------------------------------------------------------------------------------------------------------------
# Control clásico: desfase único por traza con correlación cruzada (Luo & Schuster, 1991)
def cc_shift(d, do, max_lag):
    """Δτ (muestras) que maximiza C(l) = Σ_t d(t − l)·do(t), con refinamiento parabólico. d, do: (B, nt)."""
    n = d.shape[-1]
    nf = 2 * n
    C = torch.fft.irfft(torch.fft.rfft(do, nf) * torch.conj(torch.fft.rfft(d, nf)), nf)   # C[l] = Σ d(t) do(t+l)
    lags = torch.cat([C[:, :max_lag + 1], C[:, -max_lag:]], 1)                         # l = 0..L, −L..−1
    lag_vals = torch.cat([torch.arange(max_lag + 1), torch.arange(-max_lag, 0)]).to(d.device)
    order = torch.argsort(lag_vals)
    Cl, lv = lags[:, order], lag_vals[order]
    k = Cl.argmax(-1).clamp(1, Cl.shape[-1] - 2)
    c0, cm, cp = Cl.gather(-1, k[:, None]), Cl.gather(-1, k[:, None] - 1), Cl.gather(-1, k[:, None] + 1)
    den = (cm - 2 * c0 + cp)
    frac = torch.where(den.abs() > 1e-12, 0.5 * (cm - cp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    return lv[k].to(d.dtype)[:, None] + frac


class _CCShift(torch.autograd.Function):
    """Δτ(d) con gradiente implícito: ∂Δτ/∂d(s) = ḋo(s+Δτ) / Σ_s ḋ(s)·ḋo(s+Δτ)."""

    @staticmethod
    def forward(ctx, d, do, max_lag):
        tau = cc_shift(d, do, max_lag)
        ctx.save_for_backward(d, do, tau)
        return tau

    @staticmethod
    def backward(ctx, g):
        d, do, tau = ctx.saved_tensors
        do_s = warp(do, -tau.expand_as(do))                 # do(s + Δτ)
        dd = torch.gradient(d, dim=-1)[0]
        ddo = torch.gradient(do_s, dim=-1)[0]
        den = (dd * ddo).sum(-1, keepdim=True)
        den = torch.where(den.abs() > 1e-12, den, torch.full_like(den, float("inf")))
        return g * ddo / den, None, None


def cc_misfit(max_lag=100):
    """J(d) = ½ Σ_trazas Δτ²; usa la banda alta de d contra d_obs (ya en banda alta)."""
    def M(d, do):
        a, b = M2._traces(M2.band(d, "hp")), M2._traces(do)
        keep = (a.abs().amax(-1) > 0) & (b.abs().amax(-1) > 0)
        return 0.5 * _CCShift.apply(a[keep], b[keep], max_lag).pow(2).sum()
    return M
