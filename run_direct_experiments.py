"""Time-to-quality benchmark for direct nonnegative preconditioner fitting."""
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
from direct_precond import DirectPrecond
from run_quality_cost import TimedSolver, first_passage, synchronize


class TimedPrecond:
    def __init__(self, precond):
        self.precond = precond
        self.update_seconds = 0.0
        self.apply_seconds = 0.0
        self.events = []

    def update(self, S, Y):
        synchronize()
        start = time.perf_counter()
        self.precond.update(S, Y)
        synchronize()
        self.update_seconds += time.perf_counter() - start
        self.events.append(dict(n_pairs=len(S), update_seconds=self.update_seconds,
                                apply_seconds=self.apply_seconds))

    def apply(self, q):
        synchronize()
        start = time.perf_counter()
        result = self.precond.apply(q)
        synchronize()
        self.apply_seconds += time.perf_counter() - start
        return result


def compare_methods(runs):
    baseline = runs["L-BFGS"]
    comparisons = {}
    for criterion in ("misfit", "model", "joint"):
        target_J, target_err = baseline["J"], baseline["err"]
        base_hit = first_passage(baseline["accepted"], target_J, target_err, criterion)
        if base_hit is None:
            raise RuntimeError("Baseline did not reach its own final quality")
        methods = {}
        for label, run in runs.items():
            hit = first_passage(run["accepted"], target_J, target_err, criterion)
            methods[label] = dict(reached=hit is not None, hit=hit)
            if hit is not None:
                methods[label].update(
                    solves_saved=1 - hit["sims"] / base_hit["sims"],
                    seconds_saved=1 - hit["seconds"] / base_hit["seconds"],
                    speedup=base_hit["seconds"] / hit["seconds"])
        comparisons[criterion] = dict(target_J=target_J, target_err=target_err, methods=methods)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/direct-fit"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if T.SCENARIO not in (1, 2) or T.ACQ != "wells" or T.INIT != "smooth" or T.NOISE != 0:
        parser.error("Requires standard scenario 1/2, wells, smooth initialization and no noise")
    outdir = args.output / f"s{T.SCENARIO}"
    outdir.mkdir(parents=True, exist_ok=True)
    if list(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {outdir}; use a new --output directory")
    torch.set_num_threads(1)
    names = ["fwi_toy.py", "ml_ideas.py", "run_quality_cost.py", "direct_precond.py", "run_direct_experiments.py"]
    metadata = dict(
        scenario=T.SCENARIO, acquisition=T.ACQ, initialization=T.INIT, noise=T.NOISE,
        budget=100, repeats=args.repeats, torch=torch.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if T.dev.type == "cuda" else None,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in names},
        direct=dict(sigmas=[1, 2, 4], eps=0.1, ridge=0.01, recent=4.0),
        fixed="same filters; equal weights summing to 0.9; gamma updated from latest secant",
        setup_forwards=1, warmup_forwards=1,
        timing="Synchronized forward completion; includes training and application; component timing synchronized",
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.no_grad():
        warmup = T.simulate(T.v_init)
    del warmup
    synchronize()
    methods = ["L-BFGS", "Bounded", "Direct", "Fixed"]
    all_comparisons = []
    for repeat in range(args.repeats):
        order = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
        runs = {}
        for label in order:
            torch.manual_seed(0)
            np.random.seed(0)
            synchronize()
            solver = TimedSolver()
            raw = (LearnedPrecond(illum_bound=3.0) if label == "Bounded" else
                   DirectPrecond(freeze=label == "Fixed") if label in ("Direct", "Fixed") else None)
            precond = TimedPrecond(raw) if raw is not None else None
            result = T.run_lbfgs(100, precond=precond, solver=solver, label=label)
            synchronize()
            wall = time.perf_counter() - solver.started
            if solver.n_sims != 100 or not torch.isfinite(result["m"]).all():
                raise RuntimeError(f"Invalid benchmark result for {label}")
            accepted = []
            for entry in result["log"][1:]:
                event = solver.events[entry["sims"]]
                if event["J"] != entry["J"]:
                    raise RuntimeError("Accepted objective mismatches timed forward")
                accepted.append(dict(entry, seconds=event["seconds"]))
            runs[label] = dict(
                label=label, repeat=repeat, J=result["J"], err=result["err"], wall=wall,
                n_sims=solver.n_sims, n_fwd=solver.n_fwd, n_adj=solver.n_adj,
                n_born=solver.n_born, m=result["m"].detach().cpu(), metadata=metadata,
                accepted=accepted, events=solver.events, trace=solver.trace, log=result["log"],
                models={n: m.cpu() for n, m in solver.models.items()},
                training_history=raw.history if raw is not None else None,
                component_events=precond.events if precond is not None else None,
                update_seconds=precond.update_seconds if precond is not None else 0,
                apply_seconds=precond.apply_seconds if precond is not None else 0,
            )
            print(json.dumps(dict(scenario=T.SCENARIO, repeat=repeat, label=label,
                                  J=result["J"], err=result["err"], wall=wall,
                                  update_seconds=runs[label]["update_seconds"])), flush=True)
            del result, solver, raw, precond
        comparisons = compare_methods(runs)
        all_comparisons.append(dict(repeat=repeat, order=order, criteria=comparisons))
        for label, run in runs.items():
            models = run.pop("models")
            hits = {criterion["methods"][label]["hit"]["sims"] for criterion in comparisons.values()
                    if criterion["methods"][label]["reached"]}
            run["threshold_models"] = {n: models[n] for n in hits}
            torch.save(run, outdir / f"{label}-rep{repeat}.pt")
        (outdir / "comparisons.json").write_text(json.dumps(all_comparisons, indent=2) + "\n")
        for criterion in ("misfit", "joint"):
            print(json.dumps(dict(scenario=T.SCENARIO, repeat=repeat, criterion=criterion,
                                  results={label: dict(reached=v["reached"],
                                           sims=v["hit"]["sims"] if v["reached"] else None,
                                           seconds=v["hit"]["seconds"] if v["reached"] else None)
                                           for label, v in comparisons[criterion]["methods"].items()})),
                  flush=True)
        del runs


if __name__ == "__main__":
    main()
