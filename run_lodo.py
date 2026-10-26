"""Benchmark de LODO (arXiv:2210.06171) como optimizador y como precondicionador de L-BFGS en la FWI del juguete.

Uso: FWI_SCENARIO={1,2} FWI_ACQ=wells FWI_INIT=smooth python run_lodo.py [--output DIR] [NOMBRE ...]
Presupuesto: 100 simulaciones (forward + adjunta, incluidos ensayos rechazados). Una corrida por brazo.
"""
import argparse
import json
import os
import time

import torch

import fwi_toy as T
import lodo as L
import ml_ideas as M

BUDGET = 100


def lbfgs_p_b3():
    return T.run_lbfgs(BUDGET, precond=M.LearnedPrecond(illum_bound=3.0), label="L-BFGS+P-b3")


EXPERIMENTS = {
    "L-BFGS": lambda: T.run_lbfgs(BUDGET),
    "L-BFGS+P-b3": lbfgs_p_b3,
    "LODO": lambda: L.run_lodo(BUDGET, meta_lr=1e-3, label="LODO"),
    "LODO-a1e-2": lambda: L.run_lodo(BUDGET, meta_lr=1e-2, label="LODO-a1e-2"),
    "LODO-PC": lambda: L.run_lbfgs_lodo(BUDGET, meta_lr=1e-3, label="LODO-PC"),
    "LODO-PC-a1e-2": lambda: L.run_lbfgs_lodo(BUDGET, meta_lr=1e-2, label="LODO-PC-a1e-2"),
}


def first_passage(log, J_ref, err_ref):
    out = {}
    for key, cond in [("J", lambda r: r["J"] <= J_ref), ("err", lambda r: r["err"] <= err_ref),
                      ("joint", lambda r: r["J"] <= J_ref and r["err"] <= err_ref)]:
        out[key] = next((r["sims"] for r in log if cond(r)), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--output", default=f"results/lodo/s{T.SCENARIO}")
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
        row = dict(J=R["J"], err=R["err"], err0=T.rel_err(T.v_init), n_sims=R["solver"].n_sims, wall=wall,
                   meta_time=R.get("meta_time"), n_meta=R.get("n_meta"), best_J=R.get("best_J"),
                   best_err=R.get("best_err"))
        torch.save(dict(row, label=name, m=R["m"].detach().cpu(), log=R["log"]), path)
        summary[name] = row
        if "L-BFGS" in summary:
            ref = summary["L-BFGS"]
            for k, r in summary.items():
                lg = torch.load(f"{args.output}/{k}.pt", weights_only=False)["log"]
                r["passage_vs_lbfgs"] = first_passage(lg, ref["J"], ref["err"])
        json.dump(summary, open(spath, "w"), indent=1)
        print(f"[s{T.SCENARIO}] {name:14s} J {R['J']:.4e} err {R['err']:.4f} sims {R['solver'].n_sims:.0f} "
              f"wall {wall:.1f}s meta {row['meta_time'] or 0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
