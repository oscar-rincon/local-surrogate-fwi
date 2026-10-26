"""Problema de juguete 2D compartido por los notebooks.

Contiene: solver acústico en diferencias finitas (fino y grueso), modelo verdadero/inicial, datos observados,
contador de simulaciones caras, parametrización precondicionada ("hat"), surrogate local de bajo rango con región
de confianza y L-BFGS de referencia. Es la misma configuración que `surrogate_tr_fwi.ipynb`.
"""
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.ndimage import gaussian_filter

torch.manual_seed(0)
np.random.seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32

# ----------------------------------------------------------------------------------------------------------------
# Malla fina y adquisición
nz, nx, dx = 60, 100, 0.020
nt, dt, f0 = 700, 0.002, 5.0
npml = 20
src_x = np.linspace(4, nx - 5, 6).round().astype(int)
src_z, rec_z = 2, 2
well_l, well_r = 2, nx - 3
ns = len(src_x)
ACQ = os.environ.get("FWI_ACQ", "wells")          # "wells": superficie + 2 pozos; "surface": solo superficie
INIT = os.environ.get("FWI_INIT", "smooth")       # "smooth": v_true suavizado; "poor": 1D, sin estructura lateral
INIT_SCALE = float(os.environ.get("FWI_INIT_SCALE", "0.88"))   # factor de velocidad del modelo 1D "poor"
nrec = nx + 2 * nz if ACQ == "wells" else nx
VMIN, VMAX = 1.2, 3.2


def ricker(f, t, t0):
    a = (np.pi * f * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def lap4(u):
    up = Fnn.pad(u, (2, 2, 2, 2))
    c = up[:, 2:-2, 2:-2]
    lz = (-1 / 12) * (up[:, :-4, 2:-2] + up[:, 4:, 2:-2]) + (4 / 3) * (up[:, 1:-3, 2:-2] + up[:, 3:-1, 2:-2])
    lx = (-1 / 12) * (up[:, 2:-2, :-4] + up[:, 2:-2, 4:]) + (4 / 3) * (up[:, 2:-2, 1:-3] + up[:, 2:-2, 3:-1])
    return lz + lx - 5.0 * c


class WaveGrid:
    """Solver acústico 2D. fac=1 es la malla 'cara'; fac=2 es una malla gruesa (dx y dt dobles, ~1/8 del costo)."""

    def __init__(self, fac=1):
        self.fac = fac
        self.nz, self.nx, self.dx = nz // fac, nx // fac, dx * fac
        self.nt, self.dt = nt // fac, dt * fac
        self.npml = npml // fac
        self.NZ, self.NX = self.nz + 2 * self.npml, self.nx + 2 * self.npml
        self.src_x = np.round(src_x / fac).astype(int)
        self.src_z, self.rec_z = max(src_z // fac, 1), max(rec_z // fac, 1)
        self.well_l, self.well_r = max(well_l // fac, 1), int(round(well_r / fac))
        tt = np.arange(self.nt) * self.dt
        self.wavelet = torch.tensor(ricker(f0, tt, 1.2 / f0), dtype=DT, device=dev)
        S = torch.zeros(ns, self.NZ, self.NX, dtype=DT, device=dev)
        for i, sx in enumerate(self.src_x):
            S[i, self.npml + self.src_z, self.npml + sx] = 1.0 / fac ** 2     # misma fuente física
        self.S = S
        p = self.npml
        sig_max = 3 * VMAX * np.log(1e3) / (2 * p * self.dx)
        prof = sig_max * (np.arange(p, 0, -1) / p) ** 2
        sig = np.zeros((self.NZ, self.NX))
        for i in range(p):
            sig[i, :] = np.maximum(sig[i, :], prof[i]); sig[-1 - i, :] = np.maximum(sig[-1 - i, :], prof[i])
            sig[:, i] = np.maximum(sig[:, i], prof[i]); sig[:, -1 - i] = np.maximum(sig[:, -1 - i], prof[i])
        sig = torch.tensor(sig, dtype=DT, device=dev)
        self.A, self.B = 1 + sig * self.dt / 2, 1 - sig * self.dt / 2
        # costo relativo a la malla fina (celdas x pasos de tiempo)
        self.cost = (self.NZ * self.NX * self.nt) / ((nz + 2 * npml) * (nx + 2 * npml) * nt)

    def simulate(self, v):
        p = self.npml
        vp = Fnn.pad(v[None, None], (p,) * 4, mode="replicate")[0, 0]
        c2 = (vp * self.dt / self.dx) ** 2
        u0 = torch.zeros(ns, self.NZ, self.NX, dtype=DT, device=dev)
        u1 = torch.zeros_like(u0)
        traces = []
        for it in range(self.nt):
            u2 = (2 * u1 - self.B * u0 + c2 * lap4(u1) + (self.dt ** 2) * self.wavelet[it] * self.S) / self.A
            if ACQ == "wells":
                traces.append(torch.cat([u2[:, p + self.rec_z, p:p + self.nx],
                                         u2[:, p:p + self.nz, p + self.well_l],
                                         u2[:, p:p + self.nz, p + self.well_r]], 1))
            else:
                traces.append(u2[:, p + self.rec_z, p:p + self.nx])
            u0, u1 = u1, u2
        return torch.stack(traces, 1)


fine = WaveGrid(1)


def simulate(v):
    return fine.simulate(v)


# ----------------------------------------------------------------------------------------------------------------
# Modelos y datos
zz, xx = np.meshgrid(np.arange(nz) * dx, np.arange(nx) * dx, indexing="ij")
SCENARIO = int(os.environ.get("FWI_SCENARIO", "1"))
if SCENARIO == 1:      # gradiente + anomalía rápida + anomalía lenta + capa delgada
    v_true_np = 1.6 + 0.9 * zz / zz.max()
    v_true_np += 0.35 * np.exp(-((xx - 0.65) ** 2 + (zz - 0.55) ** 2) / (2 * 0.11 ** 2))
    v_true_np -= 0.30 * np.exp(-((xx - 1.35) ** 2 + (zz - 0.70) ** 2) / (2 * 0.11 ** 2))
    v_true_np += 0.15 * (np.abs(zz - 0.95) < 0.04) * (xx > 0.3) * (xx < 1.7)
else:                  # capas buzantes cortadas por una falla + lente rápida
    zt = zz - 0.12 * xx + 0.08 * (xx > 1.1)
    v_true_np = 1.65 + 0.12 * np.floor(np.clip(zt, 0, None) / 0.17)
    v_true_np += 0.30 * np.exp(-((xx - 1.45) ** 2 / (2 * 0.16 ** 2) + (zz - 0.42) ** 2 / (2 * 0.06 ** 2)))
    v_true_np = gaussian_filter(v_true_np, 1.0, mode="nearest")
v_smooth_np = gaussian_filter(v_true_np, sigma=12, mode="nearest")
if INIT == "poor":     # gradiente lineal 1D con velocidades más lentas (INIT_SCALE): provoca cycle skipping
    _prof = v_true_np.mean(1)
    _cz = np.polyfit(zz[:, 0], _prof, 1)
    v_smooth_np = np.repeat((INIT_SCALE * np.polyval(_cz, zz[:, 0]))[:, None], nx, 1)


def taper1d(n, i0, i1):
    t = np.clip((np.arange(n) - i0) / (i1 - i0), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


tz = taper1d(nz, 5, 12)
tx = np.minimum(taper1d(nx, 4, 9), taper1d(nx, 4, 9)[::-1])
mask_np = tz[:, None] * tx[None, :]
v_init_np = mask_np * v_smooth_np + (1 - mask_np) * v_true_np

v_true = torch.tensor(v_true_np, dtype=DT, device=dev)
v_init = torch.tensor(v_init_np, dtype=DT, device=dev)
mask = torch.tensor(mask_np, dtype=DT, device=dev)
n_act = float(mask.sum())

with torch.no_grad():
    d_raw = simulate(v_true)
DSCALE = d_raw.abs().max().item()
d_obs = d_raw / DSCALE
del d_raw
d_obs_clean = d_obs
NOISE = float(os.environ.get("FWI_NOISE", "0"))   # ruido gaussiano blanco: σ = NOISE · RMS(d_obs)
if NOISE > 0:
    _gen = torch.Generator().manual_seed(1234)
    NOISE_STD = NOISE * d_obs.pow(2).mean().sqrt().item()
    d_obs = d_obs + NOISE_STD * torch.randn(d_obs.shape, generator=_gen).to(dev)
else:
    NOISE_STD = 0.0


def rel_err(v):
    return (((v - v_true) / v_true) ** 2 * mask).sum().div(n_act).sqrt().item()


# ----------------------------------------------------------------------------------------------------------------
# Simulaciones caras con contador
class ExpensiveSolver:
    """forward / adjoint / born cuestan 1 simulación fina cada uno; `coarse_cost` acumula simulaciones gruesas
    (en unidades de simulación fina)."""

    def __init__(self, d_obs, misfit=None):
        """misfit(d_syn, d_obs) -> escalar (tensor). Por defecto ½‖d − d_obs‖²."""
        self.d_obs = d_obs
        self.misfit = misfit or (lambda d, do: 0.5 * ((d - do) ** 2).sum())
        self.n_fwd = self.n_adj = self.n_born = 0
        self.coarse_cost = 0.0
        self.trace = []

    @property
    def n_sims(self):
        return self.n_fwd + self.n_adj + self.n_born + self.coarse_cost

    def _F(self, m):
        return simulate(m.clamp(VMIN, VMAX)) / DSCALE

    def forward(self, m, keep=False):
        self.n_fwd += 1
        if keep:
            m = m.detach().requires_grad_(True)
            d = self._F(m)
            J = self.misfit(d, self.d_obs)
            self.trace.append((self.n_sims, J.item()))
            return J.item(), d.detach(), (m, J)
        with torch.no_grad():
            d = self._F(m)
            J = self.misfit(d, self.d_obs).item()
        self.trace.append((self.n_sims, J))
        return J, d, None

    def adjoint(self, handle):
        self.n_adj += 1
        m, J = handle
        (g,) = torch.autograd.grad(J, m)
        return g

    def born(self, m, v):
        self.n_born += 1
        _, Bv = torch.func.jvp(self._F, (m.detach(),), (v.detach(),))
        return Bv

    def gn_hvp(self, m, v):
        """Producto Gauss-Newton Bᵀ B v (misfit L2): 1 Born + 1 adjunta."""
        Bv = self.born(m, v)
        self.n_adj += 1
        _, vjp = torch.func.vjp(self._F, m.detach())
        return vjp(Bv)[0]


# ----------------------------------------------------------------------------------------------------------------
# Parametrización "hat": s = D ⊙ G(ŝ),  ||ŝ|| ≈ cambio relativo RMS de velocidad
SMOOTH_SIGMA = 2.0
Dsc = mask * v_init * math.sqrt(n_act)
_PAD = int(3 * SMOOTH_SIGMA)
_k1 = torch.exp(-torch.arange(-_PAD, _PAD + 1, device=dev, dtype=DT) ** 2 / (2 * SMOOTH_SIGMA ** 2))
_K = ((_k1[:, None] * _k1[None, :]) / _k1.sum() ** 2)[None, None]


def smooth(a):
    return Fnn.conv2d(a[None, None], _K, padding=_PAD)[0, 0]


def to_hat_grad(g):
    return smooth(g * Dsc).flatten()


def from_hat(xh):
    return Dsc * smooth(xh.view(nz, nx))


class HatParam:
    """m = base + D ⊙ G(x̂). Parametrización por defecto de run_lbfgs (base = v_init)."""
    step_norm = None    # None ⇒ ‖p‖ (‖x̂‖ ≈ cambio relativo RMS de velocidad)

    def __init__(self, base=None):
        self.base = v_init if base is None else base
        self.x0 = torch.zeros(nz * nx, device=dev)

    def model(self, x):
        return self.base + from_hat(x)

    def pullback(self, x, gm):
        return to_hat_grad(gm)


def traveltime_cap(s, v):
    dtcol = 2 * (-(s / v ** 2) * dx).sum(0)
    return dtcol.abs().max().item()


def build_basis(cands, kmax, tol=1e-2):
    Q = []
    for v in cands:
        v0n = v.norm()
        if v0n == 0:
            continue
        w = v.clone()
        for _ in range(2):
            for q in Q:
                w -= (q @ w) * q
        if w.norm() > tol * v0n:
            Q.append(w / w.norm())
        if len(Q) == kmax:
            break
    return torch.stack(Q, 1)


# ----------------------------------------------------------------------------------------------------------------
# Surrogate local de bajo rango (parte lineal multi-secante / Born) + región de confianza
@dataclass
class Pair:
    shat: torch.Tensor
    dd: torch.Tensor
    kind: str
    it: int
    base_hat: torch.Tensor


@dataclass
class TRConfig:
    budget: float = 100
    Delta0: float = 0.01
    Delta_max: float = 0.06
    eta1: float = 0.1
    eta2: float = 0.75
    gamma_dec: float = 0.5
    gamma_inc: float = 2.0
    kmax: int = 10
    n_steps_basis: int = 7
    n_grads_basis: int = 2
    decay: float = 0.8
    ridge: float = 1e-3
    born_mode: str = "always"      # "never" | "always" (en g_k tras aceptar) | "adaptive" (según incertidumbre)
                                   # | "rho" (en g_k tras aceptar solo si |ρ_k − 1| > rho_tol)
    unc_tau: float = 0.3           # umbral de incertidumbre para "adaptive"
    rho_tol: float = 0.25          # umbral para "rho"
    extra_dirs: object = None      # callable(ghat, history) -> lista de direcciones hat para V
    inner_iters: int = 300
    tt_cap: float = 1 / (4 * f0)
    verbose: bool = False


def fit_linear(pairs, Vhat, it, x_hat, Delta, cfg):
    """U por mínimos cuadrados ponderados + ridge. Devuelve U y (G, lam) para la incertidumbre bayesiana."""
    k = Vhat.shape[1]
    if not pairs:
        return torch.zeros(d_obs.numel(), k, device=dev), None
    Shat = torch.stack([p.shat for p in pairs], 1)
    Dd = torch.stack([p.dd for p in pairs], 1)
    C = Vhat.T @ Shat
    w = []
    for i, p in enumerate(pairs):
        dist = (p.base_hat - x_hat).norm().item()
        proj = (C[:, i].norm() / Shat[:, i].norm()).item() ** 2
        w.append(cfg.decay ** (it - p.it) / (1 + (dist / max(Delta, 1e-6)) ** 2) * proj)
    w = torch.tensor(w, device=dev, dtype=DT)
    # pesos normalizados por ||c_i||² para que la incertidumbre sea adimensional
    wn = w / (C.norm(dim=0) ** 2).clamp_min(1e-30)
    G = (C * w) @ C.T
    lam = cfg.ridge * torch.trace(G) / k + 1e-12
    U = torch.linalg.solve(G + lam * torch.eye(k, device=dev), ((Dd * w) @ C.T).T).T
    Gn = (C * wn) @ C.T
    return U, (Gn, 1e-2 * torch.trace(Gn) / k + 1e-12)


def uncertainty(c, stats):
    """Fracción (0..1) del paso no explicada por los datos: varianza posterior de un modelo lineal bayesiano,
    u² = λ cᵀ(G+λI)⁻¹c / ||c||²  (≈ 1 en direcciones nunca observadas, ≈ 0 en direcciones bien muestreadas)."""
    if stats is None:
        return 1.0
    G, lam = stats
    k = G.shape[0]
    u2 = lam * c @ torch.linalg.solve(G + lam * torch.eye(k, device=dev), c) / (c @ c)
    return math.sqrt(max(u2.item(), 0.0))


def make_Jtilde(U, r, gc):
    corr = gc - U.T @ r
    def Jt(c):
        res = r + U @ c
        return 0.5 * (res ** 2).sum() + corr @ c
    return Jt


def solve_subproblem(Jt, gc, Delta, iters=300):
    J0 = Jt(torch.zeros_like(gc)).item()
    p = -gc / gc.norm()
    ts = torch.linspace(0, Delta, 65, device=dev)[1:]
    with torch.no_grad():
        vals = torch.stack([Jt(t * p) for t in ts])
    c_cau = ts[vals.argmin()] * p
    J_cau = vals.min().item()
    c = c_cau.clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=Delta / 20)
    best_c, best_J = c_cau.clone(), J_cau
    for _ in range(iters):
        opt.zero_grad()
        J = Jt(c)
        if J.item() < best_J:
            best_J, best_c = J.item(), c.detach().clone()
        J.backward(); opt.step()
        with torch.no_grad():
            n = c.norm()
            if n > Delta:
                c.mul_(Delta / n)
    with torch.no_grad():
        Jl = Jt(c).item()
        if Jl < best_J:
            best_J, best_c = Jl, c.detach().clone()
    return best_c, J0 - best_J


def update_radius(Delta, rho, cnorm, cfg):
    if rho < cfg.eta1:
        return Delta * cfg.gamma_dec
    if rho >= cfg.eta2 and cnorm >= 0.9 * Delta:
        return min(cfg.gamma_inc * Delta, cfg.Delta_max)
    return Delta


def run_trust_region(cfg, label="TR"):
    solver = ExpensiveSolver(d_obs)
    m = v_init.clone()
    x_hat = torch.zeros(nz * nx, device=dev)
    J, d, h = solver.forward(m, keep=True)
    g = solver.adjoint(h); del h
    ghat = to_hat_grad(g)
    Delta = cfg.Delta0
    pairs, steps, grads, log = [], [], [ghat], []
    hist = dict(steps=steps, grads=grads)
    if cfg.born_mode in ("always", "rho"):
        v_dir = ghat / ghat.norm() * Delta
        pairs.append(Pair(v_dir, solver.born(m, from_hat(v_dir)).flatten(), "born", 0, x_hat.clone()))
    it = 0
    while solver.n_sims < cfg.budget:
        it += 1
        r = (d - d_obs).flatten()
        extra = cfg.extra_dirs(ghat, hist) if cfg.extra_dirs else []
        cands = [ghat] + list(extra) + steps[::-1][: cfg.n_steps_basis] + grads[-2::-1][: cfg.n_grads_basis]
        Vhat = build_basis(cands, cfg.kmax)
        gc = Vhat.T @ ghat
        born_now = False
        for attempt in range(2):
            U, stats = fit_linear(pairs, Vhat, it, x_hat, Delta, cfg)
            Jt = make_Jtilde(U, r, gc)
            c, pred = solve_subproblem(Jt, gc, Delta, cfg.inner_iters)
            unc = uncertainty(c, stats)
            if (cfg.born_mode == "adaptive" and attempt == 0 and unc > cfg.unc_tau
                    and solver.n_sims + 2 < cfg.budget):
                # consulta activa: Born exacto en la dirección propuesta, luego se re-ajusta y re-resuelve
                shat_q = Vhat @ c
                pairs.append(Pair(shat_q, solver.born(m, from_hat(shat_q)).flatten(), "born", it, x_hat.clone()))
                born_now = True
                continue
            break
        shat = Vhat @ c
        s = from_hat(shat)
        dtmax = traveltime_cap(s, m)
        if dtmax > cfg.tt_cap:
            fac = cfg.tt_cap / dtmax
            c, shat, s = c * fac, shat * fac, s * fac
            with torch.no_grad():
                pred = Jt(torch.zeros_like(c)).item() - Jt(c).item()
        J_new, d_new, h_new = solver.forward(m + s, keep=True)
        ared = J - J_new
        rho = ared / pred if pred > 0 else -np.inf
        pairs.append(Pair(shat.detach(), (d_new - d).flatten(), "secant", it, x_hat.clone()))
        steps.append(shat.detach())
        accepted = rho >= cfg.eta1
        log.append(dict(it=it, sims=solver.n_sims, J=J, J_trial=J_new, pred=pred, ared=ared, rho=rho, Delta=Delta,
                        cnorm=c.norm().item(), accepted=accepted, err=rel_err(m), unc=unc, born=born_now))
        Delta = update_radius(Delta, rho, c.norm().item(), cfg)
        if accepted:
            m = m + s; x_hat = x_hat + shat; J, d = J_new, d_new
            if solver.n_sims >= cfg.budget:
                break
            g = solver.adjoint(h_new)
            ghat = to_hat_grad(g); grads.append(ghat)
            if (cfg.born_mode == "always" or (cfg.born_mode == "rho" and abs(rho - 1) > cfg.rho_tol)) \
                    and solver.n_sims < cfg.budget:
                v_dir = ghat / ghat.norm() * Delta
                pairs.append(Pair(v_dir, solver.born(m, from_hat(v_dir)).flatten(), "born", it, x_hat.clone()))
        del h_new
        if cfg.verbose:
            e = log[-1]
            print(f"[{label}] it {it:3d} sims {e['sims']:6.1f} J {e['J']:.4e}->{e['J_trial']:.4e} rho {e['rho']:+.2f} "
                  f"Δ {e['Delta']:.4f} unc {unc:.2f}{' BORN' if born_now else ''} {'OK' if accepted else 'rej'}")
    return dict(label=label, m=m, log=log, solver=solver, err=rel_err(m), J=J)


# ----------------------------------------------------------------------------------------------------------------
# L-BFGS de referencia (ensayos de búsqueda lineal = forward; adjunto solo en el punto aceptado)
def run_lbfgs(budget, mem=8, step0=0.01, step_max=0.06, c1=1e-4, max_ls=6, precond=None, verbose=False,
              label="L-BFGS", solver=None, x0=None, reg=None, hv_every=0, param=None):
    """precond: objeto opcional con .update(S, Y[, V, HV]) y .apply(q) que reemplaza H0 = γI en el two-loop.
    solver/x0: permiten encadenar etapas (multiescala) compartiendo el contador de simulaciones.
    reg(m) -> escalar diferenciable: regularización sin costo de simulación (J total = J_datos + reg).
    hv_every: cada cuántas iteraciones pedir un producto Gauss-Newton H·g (2 sims) para ajustar `precond`.
    param: parametrización del modelo con .x0, .model(x), .pullback(x, ∇_m J) y .step_norm (None ⇒ ‖p‖);
    por defecto HatParam(v_init)."""
    solver = solver or ExpensiveSolver(d_obs)
    param = param or HatParam()
    x = param.x0.clone() if x0 is None else x0.clone()

    def snorm(xx, p):
        return p.norm() if param.step_norm is None else param.step_norm(xx, p)

    def evaluate(xx):
        with torch.no_grad():
            m_ = param.model(xx)
        Jd, _, h_ = solver.forward(m_, keep=True)
        if reg is None:
            return Jd, Jd, h_
        mr = m_.detach().requires_grad_(True)
        R = reg(mr)
        (gR,) = torch.autograd.grad(R, mr)
        return Jd + R.item(), Jd, (h_, gR)

    def gradient(h_, xx):
        if reg is None:
            return param.pullback(xx, solver.adjoint(h_))
        return param.pullback(xx, solver.adjoint(h_[0]) + h_[1])

    J, Jd, h = evaluate(x)
    g = gradient(h, x); del h
    Sm, Ym, Sall, Yall, Vh, HVh, log, it = [], [], [], [], [], [], [], 0
    log.append(dict(it=0, sims=solver.n_sims, J=J, Jd=Jd, err=rel_err(param.model(x).detach()), alpha=0.0))
    while solver.n_sims < budget:
        it += 1
        q = g.clone(); al = []
        for s_, y_ in zip(reversed(Sm), reversed(Ym)):
            a = (s_ @ q) / (y_ @ s_); al.append(a); q -= a * y_
        if Sm:
            if precond is not None:
                q = precond.apply(q)
            else:
                q *= (Sm[-1] @ Ym[-1]) / (Ym[-1] @ Ym[-1])
            for (s_, y_), a in zip(zip(Sm, Ym), reversed(al)):
                b = (y_ @ q) / (y_ @ s_); q += (a - b) * s_
            p = -q
        else:
            p = -g / snorm(x, g) * step0
        pn = snorm(x, p)
        if pn > step_max:
            p = p / pn * step_max
        if g @ p >= 0:
            Sm, Ym = [], []; p = -g / snorm(x, g) * step0
        alpha, ok = 1.0, False
        for _ in range(max_ls):
            if solver.n_sims >= budget:
                break
            J_try, Jd_try, h = evaluate(x + alpha * p)
            if J_try <= J + c1 * alpha * (g @ p).item():
                ok = True; break
            del h; alpha *= 0.5
        if not ok:
            Sm, Ym = [], []
            if solver.n_sims >= budget:
                break
            continue
        x_new = x + alpha * p
        log.append(dict(it=it, sims=solver.n_sims, J=J_try, Jd=Jd_try, alpha=alpha,
                        err=rel_err(param.model(x_new).detach())))
        if solver.n_sims >= budget:
            x, J = x_new, J_try; break
        g_new = gradient(h, x_new); del h
        s_, y_ = x_new - x, g_new - g
        if hv_every and precond is not None and it % hv_every == 1 and solver.n_sims + 2 <= budget:
            # H_hat v = Gᵀ D Bᵀ B D G v  (Gauss-Newton en coordenadas "hat"), v = gradiente actual normalizado
            v = g_new / g_new.norm()
            Hv = to_hat_grad(solver.gn_hvp(v_init + from_hat(x_new), from_hat(v)))
            Vh.append(v); HVh.append(Hv)
        if (s_ @ y_) > 1e-10 * s_.norm() * y_.norm():
            Sm.append(s_); Ym.append(y_); Sall.append(s_); Yall.append(y_)
            if len(Sm) > mem:
                Sm.pop(0); Ym.pop(0)
            if precond is not None:
                if Vh:
                    precond.update(Sall, Yall, Vh, HVh)
                else:
                    precond.update(Sall, Yall)
        x, g, J = x_new, g_new, J_try
        if verbose:
            print(f"[{label}] it {it:3d} sims {solver.n_sims:4.0f} J {J:.4e} alpha {alpha:.3f} err {log[-1]['err']:.4f}")
    with torch.no_grad():
        m = param.model(x)
    return dict(label=label, m=m, x=x, log=log, solver=solver, err=rel_err(m), J=J)
