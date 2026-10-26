"""Measure cost to match L-BFGS quality with the existing bounded preconditioner.

Run with FWI_SCENARIO=1 and 2, sequentially on a single GPU.
"""
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


def synchronize():
    if T.dev.type == "cuda":
        torch.cuda.synchronize()


class TimedSolver(T.ExpensiveSolver):
    """Record forward completion times without changing the optimization loop."""

    def __init__(self):
        super().__init__(T.d_obs)
        self.started = time.perf_counter()
        self.events = {}
        self.models = {}

    def forward(self, m, keep=False):
        result = super().forward(m, keep)
        synchronize()
        elapsed = time.perf_counter() - self.started
        self.events[self.n_sims] = dict(sims=self.n_sims, seconds=elapsed, J=result[0])
        self.models[self.n_sims] = m.detach().clone()
        return result


def first_passage(history, target_J, target_err, criterion):
    """Only accepted models belong in history; require simultaneous joint quality."""
    if criterion not in ("misfit", "model", "joint"):
        raise ValueError(f"Unknown quality criterion: {criterion}")
    for entry in history:
        good_J = entry["J"] <= target_J
        good_err = entry["err"] <= target_err
        if ((criterion == "misfit" and good_J) or
                (criterion == "model" and good_err) or
                (criterion == "joint" and good_J and good_err)):
            return dict(entry)
    return None


def compare_pair(baseline, learned):
    target_J, target_err = baseline["J"], baseline["err"]
    results = {}
    for criterion in ("misfit", "model", "joint"):
        base_hit = first_passage(baseline["accepted"], target_J, target_err, criterion)
        learned_hit = first_passage(learned["accepted"], target_J, target_err, criterion)
        if base_hit is None or learned_hit is None:
            raise RuntimeError(f"Target not reached for {criterion}: baseline={base_hit}, learned={learned_hit}")
        results[criterion] = dict(
            baseline=base_hit, learned=learned_hit,
            solves_saved_vs_full=1 - learned_hit["sims"] / baseline["n_sims"],
            seconds_saved_vs_full=1 - learned_hit["seconds"] / baseline["wall"],
            solves_saved_vs_first=1 - learned_hit["sims"] / base_hit["sims"],
            seconds_saved_vs_first=1 - learned_hit["seconds"] / base_hit["seconds"],
            time_speedup=base_hit["seconds"] / learned_hit["seconds"],
        )
    return dict(target_J=target_J, target_err=target_err, criteria=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/quality-cost"))
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
    source_names = ["fwi_toy.py", "ml_ideas.py", "run_quality_cost.py"]
    metadata = dict(
        scenario=T.SCENARIO, acquisition=T.ACQ, initialization=T.INIT, noise=T.NOISE,
        budget=100, repeats=args.repeats, illum_bound=3.0,
        python=platform.python_version(), torch=torch.__version__,
        device=str(T.dev), gpu=torch.cuda.get_device_name(0) if T.dev.type == "cuda" else None,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in source_names},
        setup_forwards=1, warmup_forwards=1,
        timing="Synchronized forward completion; includes all preceding optimization/training",
        thresholds="Each paired baseline's final J and masked relative RMS model error at 100 solves",
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with torch.no_grad():
        warmup = T.simulate(T.v_init)
    del warmup
    synchronize()
    pairs = []
    for repeat in range(args.repeats):
        order = ["L-BFGS", "L-BFGS+P-b3"] if repeat % 2 == 0 else ["L-BFGS+P-b3", "L-BFGS"]
        runs = {}
        for label in order:
            torch.manual_seed(0)
            np.random.seed(0)
            synchronize()
            solver = TimedSolver()
            precond = LearnedPrecond(illum_bound=3.0) if label != "L-BFGS" else None
            result = T.run_lbfgs(100, precond=precond, solver=solver, label=label)
            synchronize()
            wall = time.perf_counter() - solver.started
            if solver.n_sims != 100 or not torch.isfinite(result["m"]).all():
                raise RuntimeError(f"Invalid result for {label}: {solver.n_sims} solves")
            accepted = []
            for entry in result["log"]:
                if entry["it"] == 0:
                    continue
                event = solver.events[entry["sims"]]
                if event["J"] != entry["J"]:
                    raise RuntimeError("Forward event does not match accepted objective")
                accepted.append(dict(entry, seconds=event["seconds"]))
            saved = dict(
                label=label, repeat=repeat, J=result["J"], err=result["err"], wall=wall,
                n_sims=solver.n_sims, n_fwd=solver.n_fwd, n_adj=solver.n_adj, n_born=solver.n_born,
                accepted=accepted, log=result["log"], events=solver.events, trace=solver.trace,
                m=result["m"].detach().cpu(), metadata=metadata,
                training_history=precond.history if precond is not None else None,
            )
            runs[label] = saved
            # Models are retained only until paired thresholds are known.
            runs[label]["models"] = {n: model.cpu() for n, model in solver.models.items()}
            print(json.dumps(dict(scenario=T.SCENARIO, repeat=repeat, label=label,
                                  J=saved["J"], err=saved["err"], wall=wall)), flush=True)
            del result, solver, precond
        comparison = compare_pair(runs["L-BFGS"], runs["L-BFGS+P-b3"])
        comparison["repeat"] = repeat
        comparison["order"] = order
        pairs.append(comparison)
        for label, run in runs.items():
            hits = {comparison["criteria"][criterion]["baseline" if label == "L-BFGS" else "learned"]["sims"]
                    for criterion in ("misfit", "model", "joint")}
            models = run.pop("models")
            run["threshold_models"] = {n: models[n] for n in hits}
            torch.save(run, outdir / f"{label}-rep{repeat}.pt")
        (outdir / "comparisons.json").write_text(json.dumps(pairs, indent=2) + "\n")
        print(json.dumps(dict(scenario=T.SCENARIO, comparison=comparison)), flush=True)
        del runs


if __name__ == "__main__":
    main()
