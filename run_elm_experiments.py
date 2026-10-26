"""Equal-budget ELM benchmark; run separately with FWI_SCENARIO=1 and 2.

Example: python run_elm_experiments.py --seeds 0 1 2
Outputs are separate from the existing notebook caches.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import torch

import fwi_toy as T
import ml_ideas as M
from elm_surrogate import ELMCorrection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--methods", nargs="+", choices=["Linear", "TR-secante", "ELM", "L-BFGS", "L-BFGS+P"],
                        default=["TR-secante", "Linear", "L-BFGS", "L-BFGS+P", "ELM"])
    parser.add_argument("--output", type=Path, default=Path("results/elm"))
    args = parser.parse_args()
    if args.budget < 3:
        parser.error("--budget must be at least 3")
    if T.ACQ != "wells" or T.INIT != "smooth" or T.NOISE != 0 or T.SCENARIO not in (1, 2):
        parser.error("This benchmark requires scenario 1/2, wells, smooth initialization and no noise")
    torch.set_num_threads(1)
    outdir = args.output / f"s{T.SCENARIO}"
    outdir.mkdir(parents=True, exist_ok=True)
    source_names = ["fwi_toy.py", "ml_ideas.py", "elm_surrogate.py", "run_elm_experiments.py"]
    metadata = dict(
        scenario=T.SCENARIO, acquisition=T.ACQ, initialization=T.INIT, noise=T.NOISE,
        budget=args.budget, torch=torch.__version__, python=platform.python_version(),
        device=str(T.dev), gpu=torch.cuda.get_device_name(0) if T.dev.type == "cuda" else None,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in source_names},
        setup_forward_solves=1, diagnostic_forward_solves=0,
    )
    for method in args.methods:
        for seed in args.seeds if method == "ELM" else [0]:
            label = f"ELM-seed{seed}" if method == "ELM" else method
            target = outdir / f"{label}.pt"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}; choose a new --output directory")
            factory = None
            if T.dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if method in ("Linear", "ELM"):
                factory = ELMCorrection(hidden=args.hidden if method == "ELM" else 0,
                                        seed=seed, ridge=args.ridge)
                cfg = T.TRConfig(budget=args.budget, born_mode="never", surrogate_factory=factory)
                result = T.run_trust_region(cfg, label)
            elif method == "TR-secante":
                result = T.run_trust_region(T.TRConfig(budget=args.budget, born_mode="never"), label)
            else:
                precond = M.LearnedPrecond() if method == "L-BFGS+P" else None
                result = T.run_lbfgs(args.budget, precond=precond, label=label)
            if T.dev.type == "cuda":
                torch.cuda.synchronize()
            wall = time.perf_counter() - start
            solver = result["solver"]
            if solver.n_sims != args.budget or not torch.isfinite(result["m"]).all():
                raise RuntimeError(f"Invalid benchmark result: {label}, solves={solver.n_sims}")
            metrics = dict(
                label=label, J=result["J"], err=result["err"], wall=wall,
                n_fwd=solver.n_fwd, n_adj=solver.n_adj, n_born=solver.n_born, n_sims=solver.n_sims,
                err0=T.rel_err(T.v_init), J0=solver.trace[0][1],
                iterations=len(result["log"]),
                rejected=sum(not entry["accepted"] for entry in result["log"])
                if method in ("Linear", "ELM", "TR-secante") else None,
                peak_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if T.dev.type == "cuda" else None,
            )
            saved = dict(
                **metrics, metadata=metadata, m=result["m"].detach().cpu(),
                log=result["log"], trace=solver.trace,
                elm_config={k: v for k, v in asdict(factory).items()
                            if k not in ("_weights", "_bias", "history")} if factory else None,
                elm_history=factory.history if factory else None,
            )
            torch.save(saved, target)
            target.with_suffix(".json").write_text(json.dumps(dict(**metrics, metadata=metadata), indent=2) + "\n")
            print(json.dumps(dict(scenario=T.SCENARIO, **metrics)), flush=True)
            del result, saved, solver


if __name__ == "__main__":
    main()
