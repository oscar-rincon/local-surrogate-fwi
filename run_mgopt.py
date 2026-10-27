"""Benchmark multifidelidad: ¿ahorra simulaciones finas la malla gruesa frente a L-BFGS fino?

Uso: FWI_SCENARIO={1,2} FWI_ACQ=wells FWI_INIT=smooth python run_mgopt.py [--output DIR] [NOMBRE ...]
Presupuesto: 100 equivalentes de simulación fina (solve grueso = 0.125). Una corrida por brazo.
"""
import argparse
import json
import os
import time

import torch

import fwi_toy as T
import mgopt as G
import ml_ideas as M

BUDGET = 100


def mf_tr(n_inner, learn):
    name = f"MF-TR{'+corr' if learn else ''}-{n_inner}"
    return lambda: M.run_mf_trust_region(M.MFConfig(budget=BUDGET, n_inner=n_inner, Delta_max=0.12,
                                                    learn_correction=learn), name)


EXPERIMENTS = {
    "L-BFGS": lambda: T.run_lbfgs(BUDGET),
    "C2F-10": lambda: G.run_coarse_to_fine(BUDGET, 10, "C2F-10"),
    "C2F-25": lambda: G.run_coarse_to_fine(BUDGET, 25, "C2F-25"),
    "MGOPT-4": lambda: G.run_mgopt(BUDGET, n_inner=4, label="MGOPT-4"),
    "MGOPT-10": lambda: G.run_mgopt(BUDGET, n_inner=10, label="MGOPT-10"),
    "MGOPT-10-sw": lambda: G.run_mgopt(BUDGET, n_inner=10, switch=True, label="MGOPT-10-sw"),
    "MF-TR-10": mf_tr(10, False),
    "MF-TR+corr-10": mf_tr(10, True),
}


def fine_log(R):
    """Entradas del log evaluadas con el solver fino (MF-TR registra J en el iterado aceptado)."""
    return [dict(sims=r["sims"], J=r["J"], err=r["err"]) for r in R["log"]]


def first_passage(log, J_ref, err_ref):
    out = {}
    for key, cond in [("J", lambda r: r["J"] <= J_ref), ("err", lambda r: r["err"] <= err_ref),
                      ("joint", lambda r: r["J"] <= J_ref and r["err"] <= err_ref)]:
        out[key] = next((r["sims"] for r in log if cond(r)), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--output", default=f"results/mgopt/s{T.SCENARIO}")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    spath = f"{args.output}/summary.json"
    summary = json.load(open(spath)) if os.path.exists(spath) else {}
    for name in args.names:
        path = f"{args.output}/{name}.pt"
        if os.path.exists(path):
            raise SystemExit(f"{path} ya existe; use --output para otra réplica")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        R = EXPERIMENTS[name]()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.time() - t0
        s = R["solver"]
        lg = fine_log(R)
        best = min(lg, key=lambda r: r["J"])
        row = dict(J=R["J"], err=R["err"], err0=T.rel_err(T.v_init), n_sims=s.n_sims, n_fwd=s.n_fwd,
                   n_adj=s.n_adj, coarse_cost=s.coarse_cost, wall=wall, best_J=best["J"], best_err=best["err"],
                   n_coarse_acc=R.get("n_coarse_acc"), switched_at=R.get("switched_at"))
        torch.save(dict(row, label=name, m=R["m"].detach().cpu(), log=lg), path)
        summary[name] = row
        if "L-BFGS" in summary:
            ref = summary["L-BFGS"]
            for k, r in summary.items():
                lk = torch.load(f"{args.output}/{k}.pt", weights_only=False)["log"]
                r["passage_vs_lbfgs"] = first_passage(lk, ref["J"], ref["err"])
        json.dump(summary, open(spath, "w"), indent=1)
        print(f"[s{T.SCENARIO}] {name:14s} J {R['J']:.4e} err {R['err']:.4f} sims {s.n_sims:.1f} "
              f"(fwd {s.n_fwd} adj {s.n_adj} coarse {s.coarse_cost:.2f}) wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
