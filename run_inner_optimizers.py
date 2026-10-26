"""One scenario, one run per method: Adam-200, Adam-50 and inner L-BFGS."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import fwi_toy as T
from ml_ideas import LearnedPrecond
from inner_lbfgs import LBFGSPrecond
from run_quality_cost import TimedSolver, synchronize
from run_direct_experiments import TimedPrecond, compare_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/inner-optimizers"))
    args = parser.parse_args()
    if T.SCENARIO != 2 or T.ACQ != "wells" or T.INIT != "smooth" or T.NOISE != 0:
        parser.error("Requires scenario 2, wells, smooth initialization and no noise")
    outdir = args.output / "s2"
    outdir.mkdir(parents=True, exist_ok=True)
    if list(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {outdir}; use --output")
    torch.set_num_threads(1)
    methods = ["L-BFGS", "Adam-200", "Adam-50", "Inner-LBFGS-20"]
    names = ["fwi_toy.py", "ml_ideas.py", "inner_lbfgs.py", "run_inner_optimizers.py",
             "run_quality_cost.py", "run_direct_experiments.py"]
    metadata = dict(
        scenario=2, acquisition=T.ACQ, initialization=T.INIT, noise=T.NOISE,
        budget=100, repeats=1, order=methods, torch=torch.__version__,
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if T.dev.type == "cuda" else None,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in names},
        illum_bound=3.0, adam_lr=0.03,
        inner_lbfgs=dict(lr=1.0, max_iter=20, max_eval=40, history_size=10,
                         line_search="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9,
                         memory_reset_each_update=True,
                         max_eval_note="PyTorch stopping budget; a line search can exceed it"),
        setup_forwards=1, warmup_forwards=1,
        timing="Synchronized forward completion and component timings; includes preceding training",
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.no_grad():
        warmup = T.simulate(T.v_init)
    del warmup
    synchronize()
    runs = {}
    for label in methods:
        torch.manual_seed(0)
        np.random.seed(0)
        synchronize()
        solver = TimedSolver()
        raw = (LBFGSPrecond(illum_bound=3.0) if label == "Inner-LBFGS-20" else
               LearnedPrecond(illum_bound=3.0, epochs=50 if label == "Adam-50" else 200)
               if label.startswith("Adam") else None)
        precond = TimedPrecond(raw) if raw is not None else None
        result = T.run_lbfgs(100, precond=precond, solver=solver, label=label)
        synchronize()
        wall = time.perf_counter() - solver.started
        if solver.n_sims != 100 or not torch.isfinite(result["m"]).all():
            raise RuntimeError(f"Invalid result for {label}")
        accepted = []
        for entry in result["log"][1:]:
            event = solver.events[entry["sims"]]
            if entry["J"] != event["J"]:
                raise RuntimeError("Accepted objective mismatches timed forward")
            accepted.append(dict(entry, seconds=event["seconds"]))
        history = raw.history if raw is not None else []
        run = dict(
            label=label, J=result["J"], err=result["err"], wall=wall, n_sims=solver.n_sims,
            n_fwd=solver.n_fwd, n_adj=solver.n_adj, n_born=solver.n_born,
            m=result["m"].detach().cpu(), metadata=metadata, accepted=accepted,
            log=result["log"], events=solver.events, trace=solver.trace,
            training_history=history,
            update_seconds=precond.update_seconds if precond is not None else 0,
            apply_seconds=precond.apply_seconds if precond is not None else 0,
            component_events=precond.events if precond is not None else [],
            gradient_evaluations=sum(e["gradient_evaluations"] for e in history),
            loss_evaluations=sum(e["loss_evaluations"] for e in history),
            inner_iterations=sum(e["iterations"] for e in history),
        )
        runs[label] = run
        # Keep audit candidates until all paired quality thresholds are known.
        run["models"] = {n: model.cpu() for n, model in solver.models.items()}
        print(json.dumps({k: run[k] for k in
                          ["label", "J", "err", "wall", "update_seconds",
                           "gradient_evaluations", "inner_iterations"]}), flush=True)
        del solver, result, precond, raw
    comparisons = compare_methods(runs)
    for label, run in runs.items():
        models = run.pop("models")
        hits = {c["methods"][label]["hit"]["sims"] for c in comparisons.values()
                if c["methods"][label]["reached"]}
        run["threshold_models"] = {n: models[n] for n in hits}
        torch.save(run, outdir / f"{label}.pt")
    (outdir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(dict(joint=comparisons["joint"])), flush=True)


if __name__ == "__main__":
    main()
