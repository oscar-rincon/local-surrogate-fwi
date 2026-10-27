"""Multifidelidad clásica (sin ML) para la FWI del juguete: MG/OPT (Nash, 2000) y malla gruesa → fina.

MG/OPT: en x_k (gradiente fino g) se minimiza aproximadamente, con simulaciones gruesas, el modelo corregido
    ψ(z) = J_c(z) + (g − ∇J_c(x_k))ᵀ (z − x_k),      ‖z − x_k‖ ≤ Δ
(consistencia de primer orden: ∇ψ(x_k) = g), y la dirección e = z − x_k se verifica con búsqueda lineal fina.
J_c usa `ml_ideas.Fc` (malla fac=2, costo 0.125 de una simulación fina por solve) contra los datos observados.
Todos los costos se cuentan en equivalentes de simulación fina en `solver.n_sims`.
"""
import torch

import fwi_toy as T
import ml_ideas as M

CC = M.coarse.cost


class MFSolver(T.ExpensiveSolver):
    """ExpensiveSolver con modo 'coarse': forward/adjoint usan Fc y cuestan CC cada uno."""

    def __init__(self, d_obs):
        super().__init__(d_obs)
        self.mode = "fine"

    def _F(self, m):
        return M.Fc(m) if self.mode == "coarse" else super()._F(m)

    def forward(self, m, keep=False):
        out = super().forward(m, keep)
        if self.mode == "coarse":
            self.n_fwd -= 1; self.coarse_cost += CC
            self.trace[-1] = (self.n_sims, self.trace[-1][1])
        return out

    def adjoint(self, handle):
        g = super().adjoint(handle)
        if self.mode == "coarse":
            self.n_adj -= 1; self.coarse_cost += CC
        return g


def run_coarse_to_fine(budget, coarse_budget, label="C2F"):
    """Control clásico: L-BFGS sobre J_c hasta `coarse_budget` equivalentes finos, luego L-BFGS fino."""
    solver = MFSolver(T.d_obs)
    rc = dict(x=None, log=[])
    if coarse_budget > 0:
        solver.mode = "coarse"
        rc = T.run_lbfgs(coarse_budget, solver=solver, label=label + "-coarse")
        solver.mode = "fine"
    rf = T.run_lbfgs(budget, solver=solver, x0=rc["x"], label=label)
    rf["coarse_log"] = rc["log"]
    return rf


def run_mgopt(budget, n_inner=10, Delta0=0.03, Delta_min=0.005, Delta_max=0.12, mem=8, c1=1e-4, max_ls=4,
              inner_ls=4, step0=0.01, switch=False, label="MG/OPT", verbose=False):
    """switch=True: híbrido; tras el primer paso grueso con retroceso (α < 1) continúa con L-BFGS fino."""
    solver = T.ExpensiveSolver(T.d_obs)
    param = T.HatParam()
    x = param.x0.clone()

    def fine_eval(xx):
        with torch.no_grad():
            m = param.model(xx)
        J_, _, h_ = solver.forward(m, keep=True)
        return J_, h_

    def coarse_eval(zz, grad=True):
        z_ = zz.detach().clone().requires_grad_(grad)
        with torch.set_grad_enabled(grad):
            Jc = 0.5 * ((M.Fc(param.model(z_)) - T.d_obs) ** 2).sum()
        solver.coarse_cost += CC * (2 if grad else 1)
        if grad:
            (gz,) = torch.autograd.grad(Jc, z_)
            return Jc.item(), gz
        return Jc.item(), None

    J, h = fine_eval(x)
    g = param.pullback(x, solver.adjoint(h)); del h
    Delta, log, it, n_coarse_acc = Delta0, [], 0, 0
    log.append(dict(it=0, sims=solver.n_sims, J=J, err=T.rel_err(param.model(x)), alpha=0.0, Delta=Delta))
    while solver.n_sims < budget:
        it += 1
        Jc0, gc0 = coarse_eval(x)
        lin = g - gc0

        def psi(zz, grad=True):
            Jc, gz = coarse_eval(zz, grad)
            val = Jc + (lin @ (zz - x)).item()
            return val, (gz + lin if grad else None)

        # L-BFGS interno sobre ψ (proyectado a la bola ‖z − x‖ ≤ Δ), sin simulaciones finas
        z, fz, gz = x.clone(), Jc0, g.clone()
        Sm, Ym = [], []
        for _ in range(n_inner):
            if solver.n_sims + 2 * CC >= budget - 2:
                break
            q = gz.clone(); al = []
            for s_, y_ in zip(reversed(Sm), reversed(Ym)):
                a = (s_ @ q) / (y_ @ s_); al.append(a); q -= a * y_
            if Sm:
                q *= (Sm[-1] @ Ym[-1]) / (Ym[-1] @ Ym[-1])
                for (s_, y_), a in zip(zip(Sm, Ym), reversed(al)):
                    b = (y_ @ q) / (y_ @ s_); q += (a - b) * s_
            else:
                q = gz / gz.norm() * min(step0, Delta)
            p = -q
            if gz @ p >= 0:
                break
            alpha, moved = 1.0, False
            for _ in range(inner_ls):
                zn = z + alpha * p
                dz = zn - x
                if dz.norm() > Delta:
                    zn = x + dz * (Delta / dz.norm())
                fn, gn = psi(zn)
                if fn <= fz + c1 * (gz @ (zn - z)).item():
                    moved = True; break
                alpha *= 0.5
            if not moved:
                break
            s_, y_ = zn - z, gn - gz
            if (s_ @ y_) > 1e-10 * s_.norm() * y_.norm():
                Sm.append(s_); Ym.append(y_)
                if len(Sm) > mem:
                    Sm.pop(0); Ym.pop(0)
            z, fz, gz = zn, fn, gn
        e = z - x
        if e.norm() < 1e-8 or g @ e >= 0:          # sin dirección gruesa útil: paso de gradiente fino
            e = -g / g.norm() * min(step0, Delta)
            coarse_used = False
        else:
            coarse_used = True
        # verificación fina por búsqueda lineal (Armijo)
        alpha, ok = 1.0, False
        for _ in range(max_ls):
            if solver.n_sims >= budget:
                break
            J_try, h = fine_eval(x + alpha * e)
            if J_try <= J + c1 * alpha * (g @ e).item():
                ok = True; break
            del h; alpha *= 0.5
        if not ok:
            Delta = max(Delta * 0.25, Delta_min)
            if solver.n_sims >= budget:
                break
            if switch:
                break
            continue
        x_new = x + alpha * e
        n_coarse_acc += int(coarse_used)
        Delta = min(2 * Delta, Delta_max) if alpha == 1.0 and coarse_used else max(0.5 * Delta, Delta_min)
        log.append(dict(it=it, sims=solver.n_sims, J=J_try, err=T.rel_err(param.model(x_new)), alpha=alpha,
                        Delta=Delta, enorm=e.norm().item(), coarse=coarse_used))
        x, J = x_new, J_try
        if solver.n_sims >= budget:
            del h; break
        g = param.pullback(x, solver.adjoint(h)); del h
        if verbose:
            r = log[-1]
            print(f"[{label}] it {it:3d} sims {r['sims']:6.2f} J {J:.4e} α {alpha:.3f} |e| {r['enorm']:.4f} "
                  f"Δ {Delta:.3f} err {r['err']:.4f}")
        if switch and alpha < 1.0:      # la malla gruesa dejó de predecir bien el paso: pasar a L-BFGS fino
            break
    switched_at = None
    if switch and solver.n_sims < budget:
        switched_at = solver.n_sims
        rf = T.run_lbfgs(budget, solver=solver, x0=x, param=param, label=label)
        log += [dict(r, it=it + r["it"]) for r in rf["log"][1:]]
        x, J = rf["x"], rf["J"]
    with torch.no_grad():
        m = param.model(x)
    return dict(label=label, m=m, x=x, log=log, solver=solver, err=T.rel_err(m), J=J, n_coarse_acc=n_coarse_acc,
                switched_at=switched_at)
