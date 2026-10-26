"""Ideas de ML para FWI evaluadas sobre el problema de juguete de `fwi_toy`.

1. LearnedPrecond: precondicionador SPD aprendido online (H0 de L-BFGS).
2. run_mf_trust_region: surrogate multifidelidad (malla gruesa + corrección aprendida en el espacio de datos)
   con consistencia de primer orden y región de confianza.
3. La consulta Born adaptativa por incertidumbre está en `fwi_toy.run_trust_region(born_mode="adaptive")`.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fnn

import fwi_toy as T
from fwi_toy import dev, DT


# ================================================================================================================
# 1. Precondicionador aprendido:  P q = A Aᵀ q + ε γ q,   A v = a ⊙ (K ⋆ v)
#    a(x,z) = sqrt(γ) exp(θ) con θ en una malla 6x10 (iluminación), K kernel 9x9 (desenfoque / acoplamiento).
#    SPD por construcción ⇒ -P g siempre es dirección de descenso. Se ajusta a la ecuación secante inversa P y_i ≈ s_i
#    con TODOS los pares del historial (L-BFGS solo usa los últimos `mem`).
# ================================================================================================================
class LearnedPrecond:
    def __init__(self, coarse=(6, 10), ksize=9, eps=0.1, reg=1e-2, epochs=200, lr=0.03, recent=4.0,
                 learn_illum=True, learn_kernel=True, init=None, freeze=False, illum_bound=None):
        """init: dict con 'theta' y 'K' de un P ya aprendido (transferencia). freeze: no re-entrenar (solo γ).
        illum_bound: si se da (p. ej. 3), a² queda acotada en [1/b, b] (θ = ½ ln b · tanh(θ_raw))."""
        self.illum_bound = illum_bound
        self.eps, self.reg, self.epochs, self.lr, self.recent = eps, reg, epochs, lr, recent
        self.learn_illum, self.learn_kernel, self.freeze = learn_illum, learn_kernel, freeze
        self.theta = torch.zeros(1, 1, *coarse, device=dev, requires_grad=True)
        K0 = torch.zeros(1, 1, ksize, ksize, device=dev)
        K0[0, 0, ksize // 2, ksize // 2] = math.sqrt(1 - eps)
        self.K0 = K0
        self.K = K0.clone().requires_grad_(True)
        if init is not None:
            self.theta = init["theta"].to(dev).clone().requires_grad_(True)
            self.K = init["K"].to(dev).clone().requires_grad_(True)
            self.K0 = self.K.detach().clone()
        self.theta0 = self.theta.detach().clone()
        self.gamma = None
        self.pad = ksize // 2
        self.history = []

    def _a(self):
        th = self.theta if self.illum_bound is None else 0.5 * math.log(self.illum_bound) * torch.tanh(self.theta)
        return torch.exp(Fnn.interpolate(th, size=(T.nz, T.nx), mode="bilinear", align_corners=True))[0, 0]

    def _P(self, q):
        """q: (n,) o (B, n)."""
        a = self._a() * math.sqrt(self.gamma)
        v = q.reshape(-1, 1, T.nz, T.nx)
        At = Fnn.conv_transpose2d(a * v, self.K, padding=self.pad)       # Aᵀ q
        Aq = a * Fnn.conv2d(At, self.K, padding=self.pad)                  # A (Aᵀ q)
        return (Aq + self.eps * self.gamma * v).reshape(q.shape)

    def update(self, S, Y, V=None, HV=None, w_hv=1.0):
        """V, HV: pares exactos (v, H v) de productos Gauss-Newton; se ajustan con P(Hv) ≈ v y peso w_hv
        relativo al total de pares secantes."""
        s, y = S[-1], Y[-1]
        self.gamma = ((s @ y) / (y @ y)).item()
        if self.freeze:
            return
        n = len(S)
        w = torch.tensor([self.recent ** ((i - n + 1) / max(n - 1, 1)) for i in range(n)], device=dev)
        Sb, Yb = torch.stack(S), torch.stack(Y)
        if V:
            Sb, Yb = torch.cat([Sb, torch.stack(V)]), torch.cat([Yb, torch.stack(HV)])
            w = torch.cat([w, torch.full((len(V),), w_hv * w.sum().item() / len(V), device=dev)])
        sn = Sb.norm(dim=1)
        params = [p for p, on in ((self.theta, self.learn_illum), (self.K, self.learn_kernel)) if on]
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (w * ((self._P(Yb) - Sb).norm(dim=1) / sn) ** 2).sum() / w.sum()
            loss = loss + self.reg * (((self.theta - self.theta0) ** 2).mean() + ((self.K - self.K0) ** 2).sum())
            loss.backward(); opt.step()
        with torch.no_grad():
            base = (w * ((self.gamma * Yb - Sb).norm(dim=1) / sn) ** 2).sum() / w.sum()
        self.history.append(dict(n_pairs=n, loss=loss.item(), loss_gammaI=base.item()))

    @torch.no_grad()
    def apply(self, q):
        return self._P(q)

    @torch.no_grad()
    def illumination(self):
        return (self._a() ** 2).cpu().numpy()


# ================================================================================================================
# 2. Surrogate multifidelidad
# ================================================================================================================
coarse = T.WaveGrid(2)


def _interp_matrix(n_out, n_in, pos):
    """Matriz (n_out, n_in) de interpolación lineal; pos[i] = posición fraccional en la malla de entrada."""
    pos = np.clip(pos, 0, n_in - 1)
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, n_in - 1); w = pos - i0
    M = np.zeros((n_out, n_in), dtype=np.float32)
    M[np.arange(n_out), i0] += 1 - w
    M[np.arange(n_out), i1] += w
    return torch.tensor(M, device=dev)


f = coarse.fac
_Mt = _interp_matrix(T.nt, coarse.nt, (np.arange(T.nt) + 1) / f - 1)   # traza fina i ↔ t=(i+1)dt
_Ms = _interp_matrix(T.nx, coarse.nx, np.arange(T.nx) / f)
_Mw = _interp_matrix(T.nz, coarse.nz, np.arange(T.nz) / f)
_Mwr = _interp_matrix(T.nz, coarse.nz, np.arange(T.nz) / f)


def coarse_to_fine(dc):
    """(ns, nt_c, nrec_c) -> (ns, nt, nrec) en la geometría fina."""
    ncx, ncz = coarse.nx, coarse.nz
    sur, wl, wr = dc[..., :ncx], dc[..., ncx:ncx + ncz], dc[..., ncx + ncz:]
    out = torch.cat([sur @ _Ms.T, wl @ _Mw.T, wr @ _Mwr.T], -1)
    return torch.einsum("ti,sir->str", _Mt, out)


def Fc(m):
    mc = Fnn.avg_pool2d(m.clamp(T.VMIN, T.VMAX)[None, None], f)[0, 0]
    return coarse_to_fine(coarse.simulate(mc)) / T.DSCALE


# ---- corrección aprendida en el espacio de datos: filtro temporal (conv 1D) por rango de offset -------------------
_rx = np.concatenate([np.arange(T.nx) * T.dx, np.full(T.nz, T.well_l * T.dx), np.full(T.nz, T.well_r * T.dx)])
_rz = np.concatenate([np.full(T.nx, T.rec_z * T.dx), np.arange(T.nz) * T.dx, np.arange(T.nz) * T.dx])
_off = np.hypot(_rx[None, :] - (T.src_x * T.dx)[:, None], _rz[None, :] - T.src_z * T.dx)    # (ns, nrec)


class DataCorrection:
    """δd_fino ≈ h_b ⋆ δd_grueso, un filtro de `ntaps` coeficientes por rango de offset (CNN 1D lineal).
    Se ajusta en forma cerrada (ridge hacia la identidad) con los pares (δd_grueso, δd_fino) de cada verificación."""

    def __init__(self, nbins=4, ntaps=31, ridge=1e-2, decay=0.85, enabled=True):
        self.nbins, self.ntaps, self.ridge, self.decay, self.enabled = nbins, ntaps, ridge, decay, enabled
        edges = np.quantile(_off, np.linspace(0, 1, nbins + 1)); edges[-1] += 1
        self.bin = torch.tensor(np.digitize(_off, edges[1:-1]), device=dev).flatten()     # (ns*nrec,)
        self.delta = torch.zeros(ntaps, device=dev); self.delta[ntaps // 2] = 1
        self.h = self.delta.repeat(nbins, 1)
        self.stats = []                        # (G_b, rhs_b, edad)

    def _unfold(self, D):
        Dt = D.permute(0, 2, 1).reshape(-1, T.nt)                                    # (ntr, nt)
        return Fnn.pad(Dt, (self.ntaps // 2,) * 2).unfold(-1, self.ntaps, 1)          # (ntr, nt, ntaps)

    def __call__(self, D):
        if not self.enabled:
            return D
        X = self._unfold(D)
        out = torch.einsum("rtk,rk->rt", X, self.h[self.bin])
        return out.view(T.ns, T.nrec, T.nt).permute(0, 2, 1)

    @torch.no_grad()
    def add_pair(self, dc, df):
        X = self._unfold(dc)
        y = df.permute(0, 2, 1).reshape(-1, T.nt)
        G = torch.zeros(self.nbins, self.ntaps, self.ntaps, device=dev)
        b = torch.zeros(self.nbins, self.ntaps, device=dev)
        for k in range(self.nbins):
            sel = self.bin == k
            Xk = X[sel].reshape(-1, self.ntaps)
            G[k] = Xk.T @ Xk; b[k] = Xk.T @ y[sel].reshape(-1)
        self.stats = [(Gs, bs, a + 1) for Gs, bs, a in self.stats] + [(G, b, 0)]
        if self.enabled:
            Gt = sum(self.decay ** a * Gs for Gs, _, a in self.stats)
            bt = sum(self.decay ** a * bs for _, bs, a in self.stats)
            lam = self.ridge * torch.diagonal(Gt, dim1=1, dim2=2).mean(1)[:, None, None]
            eye = torch.eye(self.ntaps, device=dev)
            self.h = torch.linalg.solve(Gt + lam * eye, (bt + lam[:, :, 0] * self.delta)[..., None])[..., 0]


@dataclass
class MFConfig:
    budget: float = 100
    Delta0: float = 0.01
    Delta_max: float = 0.06
    eta1: float = 0.1
    eta2: float = 0.75
    gamma_dec: float = 0.5
    gamma_inc: float = 2.0
    n_inner: int = 4
    inner_mem: int = 5
    learn_correction: bool = True
    tt_cap: float = 1 / (4 * T.f0)
    verbose: bool = False


def run_mf_trust_region(cfg, label="MF-TR"):
    """J̃_k(ŝ) = ½||r_k + C(Fc(m_k+s) − Fc(m_k))||² + (ĝ_k − ∇_ŝ[...](0))ᵀŝ,  s = from_hat(ŝ), ||ŝ|| ≤ Δ.
    Subproblema: Cauchy + L-BFGS proyectado, cada evaluación cuesta una simulación gruesa (fwd+adj)."""
    solver = T.ExpensiveSolver(T.d_obs)
    corr = DataCorrection(enabled=cfg.learn_correction)
    cc = coarse.cost

    def coarse_eval(m, shat, dc0, r, lin, need_grad=True):
        x = shat.detach().clone().requires_grad_(need_grad)
        with torch.set_grad_enabled(need_grad):
            dc = Fc(m + T.from_hat(x))
            res = r + corr(dc - dc0)
            val = 0.5 * (res ** 2).sum() + lin @ x
        solver.coarse_cost += cc * (2 if need_grad else 1)
        if need_grad:
            (gx,) = torch.autograd.grad(val, x)
            return val.item(), gx, dc.detach()
        return val.item(), None, dc.detach()

    m = T.v_init.clone()
    J, d, h = solver.forward(m, keep=True)
    ghat = T.to_hat_grad(solver.adjoint(h)); del h
    Delta, log, it = cfg.Delta0, [], 0
    dc0 = None
    while solver.n_sims < cfg.budget:
        it += 1
        r = d - T.d_obs
        n = T.nz * T.nx
        # término de consistencia: ∇ de la parte grueso-corregida en 0 (forward + adjunto grueso);
        # el forward da también Fc(m_k) si no se tiene de la iteración anterior
        x0 = torch.zeros(n, device=dev, requires_grad=True)
        dcx = Fc(m + T.from_hat(x0))
        if dc0 is None:
            dc0 = dcx.detach()
        q = 0.5 * ((r + corr(dcx - dc0)) ** 2).sum()
        (g_c,) = torch.autograd.grad(q, x0)
        solver.coarse_cost += 2 * cc
        lin = ghat - g_c
        f0 = 0.5 * (r ** 2).sum().item()
        # --- punto de Cauchy (backtracking sobre -ĝ desde el borde de la región)
        p = -ghat / ghat.norm()
        t, best = Delta, (0.0, torch.zeros(n, device=dev), None, None)
        for _ in range(4):
            fv, gv, dcv = coarse_eval(m, t * p, dc0, r, lin)
            if fv < f0 - 1e-4 * t * ghat.norm().item():
                best = (f0 - fv, t * p, gv, dcv); break
            t *= 0.5
        pred_cau, x, gx, dcx_best = best
        fx = f0 - pred_cau
        if gx is None:
            x, gx, fx = torch.zeros(n, device=dev), ghat.clone(), f0
        # --- L-BFGS proyectado sobre el surrogate (sin simulaciones finas)
        Sm, Ym = [], []
        best_x, best_f, best_dc = x.clone(), fx, dcx_best
        for _ in range(cfg.n_inner):
            if solver.n_sims >= cfg.budget - 1:
                break
            qv = gx.clone(); al = []
            for s_, y_ in zip(reversed(Sm), reversed(Ym)):
                a = (s_ @ qv) / (y_ @ s_); al.append(a); qv -= a * y_
            if Sm:
                qv *= (Sm[-1] @ Ym[-1]) / (Ym[-1] @ Ym[-1])
            else:
                qv *= 0.5 * Delta / gx.norm()
            for (s_, y_), a in zip(zip(Sm, Ym), reversed(al)):
                b = (y_ @ qv) / (y_ @ s_); qv += (a - b) * s_
            alpha, moved = 1.0, False
            for _ in range(2):
                xn = x - alpha * qv
                if xn.norm() > Delta:
                    xn = xn * (Delta / xn.norm())
                fn, gn, dcn = coarse_eval(m, xn, dc0, r, lin)
                if fn < fx:
                    moved = True; break
                alpha *= 0.3
            if not moved:
                break
            s_, y_ = xn - x, gn - gx
            if (s_ @ y_) > 1e-10 * s_.norm() * y_.norm():
                Sm.append(s_); Ym.append(y_)
                if len(Sm) > cfg.inner_mem:
                    Sm.pop(0); Ym.pop(0)
            x, gx, fx = xn, gn, fn
            if fx < best_f:
                best_x, best_f, best_dc = x.clone(), fx, dcn
        shat, pred = best_x, f0 - best_f
        if pred <= 0:                                  # surrogate sin descenso: paso de Cauchy mínimo
            shat = 0.25 * Delta * p
            pred = 0.25 * Delta * ghat.norm().item() * 0.5
            best_dc = None
        s = T.from_hat(shat)
        dtmax = T.traveltime_cap(s, m)
        if dtmax > cfg.tt_cap:
            fac = cfg.tt_cap / dtmax
            shat, s = shat * fac, s * fac
            fv, _, best_dc = coarse_eval(m, shat, dc0, r, lin, need_grad=False)
            pred = f0 - fv
        if best_dc is None:
            _, _, best_dc = coarse_eval(m, shat, dc0, r, lin, need_grad=False)
        if solver.n_sims >= cfg.budget:
            break
        J_new, d_new, h_new = solver.forward(m + s, keep=True)
        ared = J - J_new
        rho = ared / pred if pred > 0 else -np.inf
        corr.add_pair(best_dc - dc0, d_new - d)
        accepted = rho >= cfg.eta1
        log.append(dict(it=it, sims=solver.n_sims, J=J, J_trial=J_new, pred=pred, pred_cauchy=pred_cau, ared=ared,
                        rho=rho, Delta=Delta, cnorm=shat.norm().item(), accepted=accepted, err=T.rel_err(m)))
        if rho < cfg.eta1:
            Delta *= cfg.gamma_dec
        elif rho >= cfg.eta2 and shat.norm().item() >= 0.9 * Delta:
            Delta = min(cfg.gamma_inc * Delta, cfg.Delta_max)
        if accepted:
            m = m + s; J, d = J_new, d_new
            dc0 = best_dc
            if solver.n_sims >= cfg.budget:
                del h_new; break
            ghat = T.to_hat_grad(solver.adjoint(h_new))
        del h_new
        if cfg.verbose:
            e = log[-1]
            print(f"[{label}] it {it:3d} sims {e['sims']:6.2f} J {e['J']:.4e}->{e['J_trial']:.4e} rho {e['rho']:+.2f} "
                  f"Δ {e['Delta']:.4f} |s| {e['cnorm']:.4f} {'OK' if accepted else 'rej'}")
    return dict(label=label, m=m, log=log, solver=solver, err=T.rel_err(m), J=J, corr=corr)
