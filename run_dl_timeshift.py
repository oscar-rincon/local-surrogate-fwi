"""Prueba del misfit de desfase temporal aprendido (Alfarhan et al., 2026) en el escenario de cycle skipping.

Uso: FWI_SCENARIO={1,2} FWI_INIT=poor python run_dl_timeshift.py [--output DIR] [NOMBRE ...]
Todos los brazos usan 100 simulaciones (forward + adjunta). Los de dos etapas usan N_LOW=40 en la primera etapa
(misfit cinemático o banda baja) y continúan con L2 en banda alta.
"""
import argparse
import json
import os
import time

import torch

import dl_timeshift as D
import fwi_toy as T
import ml_ideas2 as M2

BUDGET, N_LOW = 100, 40
assert T.INIT == "poor", "use FWI_INIT=poor (escenario de cycle skipping)"


def d_hp():
    return M2.band(T.d_obs, "hp")


def chain(stages, label, solver=None):
    """stages: lista de (presupuesto_acumulado, misfit, d_obs). Comparte contador entre etapas."""
    x, logs, ends = None, [], []
    for budget, misfit, dob in stages:
        s_new = T.ExpensiveSolver(dob, misfit=misfit)
        if solver is not None:
            s_new.n_fwd, s_new.n_adj, s_new.n_born, s_new.trace = solver.n_fwd, solver.n_adj, solver.n_born, \
                solver.trace
        solver = s_new
        R = T.run_lbfgs(budget, solver=solver, x0=x, label=label)
        x = R["x"]; logs += R["log"]
        ends.append(dict(sims=solver.n_sims, err=R["err"]))
    R["log"], R["label"], R["stage_end"] = logs, label, ends
    return R


def ts_nn(label, ft_steps=300, steps=3000):
    dob = d_hp()
    s0 = T.ExpensiveSolver(dob, misfit=M2.l2_band("hp"))
    _, d0, _ = s0.forward(T.v_init)                        # 1 simulación contada: datos modelados en m0
    d0 = M2.band(d0, "hp")
    net, info = D.train_timeshift(d0, dob, steps=steps, ft_steps=ft_steps)
    with torch.no_grad():
        tau = D.predict_tau(net, d0, dob)
        a, b = D.normalize(M2._traces(d0)), D.normalize(M2._traces(dob))
        tau_self = D.predict_tau(net, dob, dob)
        cc = D.cc_shift(a, b, 100)[:, 0]
        info.update(
            mse_before=(a - b).pow(2).mean().item(),
            mse_after=(D.warp(a, tau) - b).pow(2).mean().item(),
            tau_rms=tau.pow(2).mean().sqrt().item(),
            tau_self_rms=tau_self.pow(2).mean().sqrt().item(),
            cc_rms=cc.pow(2).mean().sqrt().item(),
            corr_tau_cc=torch.corrcoef(torch.stack([(tau * b.abs()).sum(-1) / b.abs().sum(-1), cc]))[0, 1].item())
    print(f"[{label}] diagnóstico: {json.dumps({k: v for k, v in info.items() if k != 'history'})}", flush=True)
    R = chain([(N_LOW, D.ts_misfit(net), dob), (BUDGET, M2.l2_band("hp"), dob)], label, solver=s0)
    R["ts_info"] = info
    return R


def ms_ml(label):
    pred = M2.predict_lowfreq(M2.train_lowfreq(), d_hp())
    return chain([(N_LOW, M2.l2_band("lp"), pred), (BUDGET, M2.l2_band("hp"), d_hp())], label)


EXPERIMENTS = {
    "HP-L2": lambda: chain([(BUDGET, M2.l2_band("hp"), d_hp())], "HP-L2"),
    "CC-L2": lambda: chain([(N_LOW, D.cc_misfit(), d_hp()), (BUDGET, M2.l2_band("hp"), d_hp())], "CC-L2"),
    "TS-NN-L2": lambda: ts_nn("TS-NN-L2"),
    "TS-NN-noFT-L2": lambda: ts_nn("TS-NN-noFT-L2", ft_steps=0),
    "TS-NN-1k-L2": lambda: ts_nn("TS-NN-1k-L2", steps=1000, ft_steps=100),
    "W1-L2": lambda: chain([(N_LOW, M2.w1_misfit(), d_hp()), (BUDGET, M2.l2_band("hp"), d_hp())], "W1-L2"),
    "MS-ML": lambda: ms_ml("MS-ML"),
    "MS-oracle": lambda: chain([(N_LOW, M2.l2_band("lp"), M2.band(T.d_obs, "lp")),
                                (BUDGET, M2.l2_band("hp"), d_hp())], "MS-oracle"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--output", default=f"results/dl-timeshift/s{T.SCENARIO}")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    summary_path = f"{args.output}/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for name in args.names:
        path = f"{args.output}/{name}.pt"
        if os.path.exists(path):
            raise SystemExit(f"{path} ya existe; use --output para otra réplica")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        R = EXPERIMENTS[name]()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall = time.time() - t0
        s = R["solver"]
        with torch.no_grad():
            d = T.simulate(R["m"].detach().clamp(T.VMIN, T.VMAX)) / T.DSCALE    # diagnóstico, no cuenta
            J_hp = 0.5 * ((M2.band(d, "hp") - d_hp()) ** 2).sum().item()
        row = dict(err0=T.rel_err(T.v_init), err=R["err"], J_hp=J_hp, n_sims=s.n_sims, wall=wall,
                   stage_end=R["stage_end"])
        if "ts_info" in R:
            row["ts_info"] = {k: v for k, v in R["ts_info"].items() if k != "history"}
        torch.save(dict(row, label=name, m=R["m"].detach().cpu(), log=R["log"],
                        ts_history=R.get("ts_info", {}).get("history")), path)
        summary[name] = row
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(f"[s{T.SCENARIO}-poor] {name:14s} err0 {row['err0']:.4f} etapa1 "
              f"{R['stage_end'][0]['err']:.4f} final {row['err']:.4f} J_hp {J_hp:.4e} "
              f"sims {s.n_sims:.0f} wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
