"""Ejecuta los experimentos de `ml_ideas_fwi.ipynb` y guarda resultados en results/s<escenario>/<nombre>.pt.
Uso: FWI_SCENARIO=1 [FWI_ACQ=surface] [FWI_INIT=poor] python run_ml_experiments.py NOMBRE [NOMBRE ...]
Nombres: ver EXPERIMENTS, más los regularizados  Tik_<λ>, TV_<λ>, DEN_<λ>, DEN+P_<λ>."""
import math, os, sys, time
import torch
import fwi_toy as T
import ml_ideas as M
import ml_ideas2 as M2
import ml_ideas3 as M3

BUDGET = 100
TAG = (f"s{T.SCENARIO}" + ("" if T.ACQ == "wells" else "-surf") + ("" if T.INIT == "smooth" else "-poor")
       + ("-n" if T.NOISE > 0 else ""))


def lbfgs_p(label, hv_every=0, reg=None, **kw):
    P = M.LearnedPrecond(**kw)
    R = T.run_lbfgs(BUDGET, precond=P, label=label, hv_every=hv_every, reg=reg)
    R["precond"] = dict(illum=P.illumination(), K=P.K.detach().cpu(), theta=P.theta.detach().cpu(),
                        history=P.history)
    return R


def transfer_init():
    """P aprendido en el escenario 1 (results/s1/L-BFGS+P.pt)."""
    pc = torch.load("results/s1/L-BFGS+P.pt", weights_only=False)["precond"]
    return dict(theta=pc["theta"], K=pc["K"])


def gauss_init(sigma, eps=0.1):
    """Control no aprendido: θ=0 y K gaussiano fijo (suma sqrt(1-ε)) ⇒ H0 = γ(1-ε)G_σ*G_σ + εγ."""
    r = int(math.ceil(2 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    K = torch.outer(g, g); K = K / K.sum() * math.sqrt(1 - eps)
    return dict(theta=torch.zeros(1, 1, 6, 10), K=K[None, None]), 2 * r + 1


def lbfgs_gauss(label, sigma):
    init, ks = gauss_init(sigma)
    return lbfgs_p(label, init=init, ksize=ks, freeze=True)


def lbfgs_ph(label, eps):
    P = M2.DiagPrecond(M2.pseudo_hessian(T.v_init), eps)
    R = T.run_lbfgs(BUDGET, precond=P, label=label)
    R["precond"] = dict(illum=P.w.view(T.nz, T.nx).cpu().numpy())
    return R


def regularized(name):
    kind, lam = name.rsplit("_", 1)
    lam = float(lam)
    if kind == "Tik":
        return T.run_lbfgs(BUDGET, reg=M2.reg_tikhonov(lam), label=name)
    if kind == "TV":
        return T.run_lbfgs(BUDGET, reg=M2.reg_tv(lam), label=name)
    if kind == "RED-INR":
        return T.run_lbfgs(BUDGET, reg=M3.reg_inr(lam, sigma=4.0), label=name)
    D = M2.train_denoiser()
    if kind == "DEN":
        return T.run_lbfgs(BUDGET, reg=M2.reg_denoiser(D, lam), label=name)
    if kind == "DEN+P":
        return lbfgs_p(name, reg=M2.reg_denoiser(D, lam))
    raise KeyError(name)


# ---- cycle skipping (FWI_INIT=poor): datos registrados solo en banda alta (> F_HP Hz) ----
def d_hp():
    return M2.band(T.d_obs, "hp")


def ms(label, low, n_low=40, precond=False, **pkw):
    """Multiescala: n_low simulaciones en banda baja (datos `low`) y el resto en banda alta.
    precond=True: un P aprendido nuevo en cada etapa (pkw se pasa a LearnedPrecond)."""
    kw = (lambda: dict(precond=M.LearnedPrecond(**pkw))) if precond else (lambda: {})
    R = M2.run_multiscale([(n_low, M2.l2_band("lp"), low, kw()),
                           (BUDGET, M2.l2_band("hp"), d_hp(), kw())], label)
    return R


def ms_ml(label, **kw):
    net = M2.train_lowfreq()
    pred = M2.predict_lowfreq(net, d_hp())
    R = ms(label, pred, **kw)
    true_lp = M2.band(T.d_obs, "lp")
    R["lowfreq"] = dict(rel_err=((pred - true_lp).norm() / true_lp.norm()).item(),
                        pred=pred[2].cpu(), true=true_lp[2].cpu(), hp=d_hp()[2].cpu())
    return R


def ms_adapt(label, net="gen", **kw):
    """Ronda 3: extrapolación de bajas frecuencias con adaptación online (ver ml_ideas3.run_ms_adaptive)."""
    net0 = M3.train_lowfreq_generic() if net == "gen" else M2.train_lowfreq()
    return M3.run_ms_adaptive(net0, label, budget=BUDGET, **kw)


EXPERIMENTS = {
    "INR-s3": lambda: T.run_lbfgs(BUDGET, param=M3.INRParam(sigma=3.0, init_std=0.1), label="INR-s3"),
    "INR-s6": lambda: T.run_lbfgs(BUDGET, param=M3.INRParam(sigma=6.0, init_std=0.1), label="INR-s6"),
    "MS-GEN": lambda: ms_adapt("MS-GEN", mode="none"),
    "MS-GEN-adapt": lambda: ms_adapt("MS-GEN-adapt"),
    "MS-GEN-adapt-w": lambda: ms_adapt("MS-GEN-adapt-w", weight=True),
    "MS-self": lambda: ms_adapt("MS-self", mode="self"),
    "MS-ML-adapt": lambda: ms_adapt("MS-ML-adapt", net="indist"),
    "MS-GEN-adapt+DEN": lambda: ms_adapt("MS-GEN-adapt+DEN", high_reg=M2.reg_denoiser(M2.train_denoiser(), 100.0)),
    "MS-GEN-adapt+INRlow": lambda: ms_adapt("MS-GEN-adapt+INRlow",
                                            low_param=lambda b: M3.INRParam(b, sigma=3.0, init_std=0.1)),
    "L-BFGS+PH0.02": lambda: lbfgs_ph("L-BFGS+PH0.02", 0.02),
    "L-BFGS+PH0.1": lambda: lbfgs_ph("L-BFGS+PH0.1", 0.1),
    "L-BFGS+P-b3": lambda: lbfgs_p("L-BFGS+P-b3", illum_bound=3.0),
    "L-BFGS+P-b10": lambda: lbfgs_p("L-BFGS+P-b10", illum_bound=10.0),
    "L-BFGS+P+Hv10": lambda: lbfgs_p("L-BFGS+P+Hv10", hv_every=10),
    "L-BFGS+P+Hv5": lambda: lbfgs_p("L-BFGS+P+Hv5", hv_every=5),
    "HP-L2": lambda: M2.run_multiscale([(BUDGET, M2.l2_band("hp"), d_hp(), {})], "HP-L2"),
    "HP-W1": lambda: M2.run_multiscale([(BUDGET, M2.w1_misfit(), d_hp(), {})], "HP-W1"),
    "W1-L2": lambda: M2.run_multiscale([(40, M2.w1_misfit(), d_hp(), {}),
                                        (BUDGET, M2.l2_band("hp"), d_hp(), {})], "W1-L2"),
    "MS-oracle": lambda: ms("MS-oracle", M2.band(T.d_obs, "lp")),
    "MS-ML": lambda: ms_ml("MS-ML"),
    "MS-ML+P": lambda: ms_ml("MS-ML+P", precond=True),
    "MS-ML+P-b3": lambda: ms_ml("MS-ML+P-b3", precond=True, illum_bound=3.0),
    "MS-oracle+P": lambda: ms("MS-oracle+P", M2.band(T.d_obs, "lp"), precond=True),
    "L-BFGS+G2": lambda: lbfgs_gauss("L-BFGS+G2", 2.0),
    "L-BFGS+G4": lambda: lbfgs_gauss("L-BFGS+G4", 4.0),
    "L-BFGS": lambda: T.run_lbfgs(BUDGET),
    "L-BFGS-mem50": lambda: T.run_lbfgs(BUDGET, mem=50, label="L-BFGS-mem50"),
    "L-BFGS+P": lambda: lbfgs_p("L-BFGS+P"),
    "L-BFGS+P-illum": lambda: lbfgs_p("L-BFGS+P-illum", learn_kernel=False),
    "L-BFGS+P-kernel": lambda: lbfgs_p("L-BFGS+P-kernel", learn_illum=False),
    "L-BFGS+P-eps0.3": lambda: lbfgs_p("L-BFGS+P-eps0.3", eps=0.3),
    "L-BFGS+P-k5": lambda: lbfgs_p("L-BFGS+P-k5", ksize=5),
    "L-BFGS+P-reg0.1": lambda: lbfgs_p("L-BFGS+P-reg0.1", reg=0.1),
    "L-BFGS+P-transfer": lambda: lbfgs_p("L-BFGS+P-transfer", init=transfer_init(), freeze=True),
    "L-BFGS+P-transfer-ft": lambda: lbfgs_p("L-BFGS+P-transfer-ft", init=transfer_init()),
    "TR-secante": lambda: T.run_trust_region(T.TRConfig(budget=BUDGET, born_mode="never"), "TR-secante"),
    "TR-Born": lambda: T.run_trust_region(T.TRConfig(budget=BUDGET, born_mode="always"), "TR-Born"),
    "TR-Born-adapt": lambda: T.run_trust_region(T.TRConfig(budget=BUDGET, born_mode="adaptive"), "TR-Born-adapt"),
    "TR-Born-rho": lambda: T.run_trust_region(T.TRConfig(budget=BUDGET, born_mode="rho"), "TR-Born-rho"),
    "MF-TR": lambda: M.run_mf_trust_region(M.MFConfig(budget=BUDGET, learn_correction=False), "MF-TR"),
    "MF-TR+corr": lambda: M.run_mf_trust_region(M.MFConfig(budget=BUDGET, learn_correction=True), "MF-TR+corr"),
}


def summarize(R, wall):
    s = R["solver"]
    out = dict(label=R["label"], m=R["m"].detach().cpu(), log=R["log"], err=R["err"], J=R["J"], wall=wall,
               n_fwd=s.n_fwd, n_adj=s.n_adj, n_born=s.n_born, coarse_cost=s.coarse_cost, n_sims=s.n_sims,
               trace=s.trace, err0=T.rel_err(T.v_init))
    if "corr" in R:
        out["corr_h"] = R["corr"].h.cpu()
    if "precond" in R:
        out["precond"] = R["precond"]
    if "lowfreq" in R:
        out["lowfreq"] = R["lowfreq"]
    out["noise"] = T.NOISE
    if R["log"] and "Jd" in R["log"][-1]:
        out["Jd"] = R["log"][-1]["Jd"]
    with torch.no_grad():   # misfit L2 (contra datos limpios) en banda completa y en banda alta (diagnóstico, no cuenta simulaciones)
        d = T.simulate(R["m"].detach().clamp(T.VMIN, T.VMAX)) / T.DSCALE
        out["J_full"] = 0.5 * ((d - T.d_obs_clean) ** 2).sum().item()
        out["J_hp"] = 0.5 * ((M2.band(d, "hp") - d_hp()) ** 2).sum().item()
    return out


if __name__ == "__main__":
    outdir = f"results/{TAG}"
    os.makedirs(outdir, exist_ok=True)
    for name in sys.argv[1:]:
        t0 = time.time()
        fn = EXPERIMENTS.get(name) or (lambda: regularized(name))
        out = summarize(fn(), time.time() - t0)
        torch.save(out, f"{outdir}/{name}.pt")
        print(f"[{TAG}] {name:16s} J={out['J']:.4e} err={out['err']:.4f} sims={out['n_sims']:.1f} "
              f"(fwd {out['n_fwd']}, adj {out['n_adj']}, born {out['n_born']}, coarse {out['coarse_cost']:.2f}) "
              f"wall {out['wall']:.0f}s", flush=True)
