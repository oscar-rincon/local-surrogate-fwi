# local-surrogate-fwi
Toy 2D test of a local surrogate + trust-region scheme for FWI.

- `surrogate_tr_fwi.ipynb`: acoustic 2D finite-difference solver in PyTorch (stand-in for SPECFEM3D, with counted
  forward/adjoint/Born simulations), low-rank surrogate with first-order consistency, trust-region loop, and comparison
  against L-BFGS at equal simulation budget.
- `ml_ideas_fwi.ipynb`: controlled test of three ML ideas (online-learned SPD preconditioner for L-BFGS,
  multifidelity coarse-grid surrogate with a learned data correction, adaptive Born queries) on two scenarios at a
  budget of 100 simulations, with ablations and non-learned controls. It loads cached results from `results/`.
  - `fwi_toy.py`: the shared problem, solver, trust region and L-BFGS.
  - `ml_ideas.py`: the three ideas.
  - `run_ml_experiments.py`: runner that regenerates the cache, e.g.
    `PYTHONPATH=. FWI_SCENARIO=2 python run_ml_experiments.py L-BFGS L-BFGS+P`.
- `ml_ideas2_fwi.ipynb`: second round of ML ideas, using the same protocol:
  - pseudo-Hessian control and learned P with exact H·v products / bounded illumination;
  - a CNN denoiser regularizer vs Tikhonov/TV, with surface-only acquisition (`FWI_ACQ=surface`);
  - CNN low-frequency extrapolation vs a W1 optimal-transport misfit, with a cycle-skipping setup
    (`FWI_INIT=poor`, data > 4 Hz).

  The code is in `ml_ideas2.py`. Trained networks are cached in `results/models_ml/`.
- Requirements: `torch`, `numpy`, `scipy`, `matplotlib`, `pandas` (GPU recommended; full run ≈ 12 min on an RTX 4060).

## Online ELM benchmark

`elm_surrogate.py` adds a local extreme-learning-machine correction to the
trust-region waveform surrogate. Fixed random tanh features are fitted by
weighted ridge regression to recent secant residuals. The correction has zero
value and derivative at the current model; every trial is still checked by
the fine forward solver. No offline training simulations are used.

Run both standard, noise-free scenarios, sequentially on one GPU:

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 python run_elm_experiments.py --seeds 0 1 2
OMP_NUM_THREADS=1 FWI_SCENARIO=2 python run_elm_experiments.py --seeds 0 1 2
python -m unittest -v test_elm_surrogate
```

The runner compares the original secant trust region, an algebraically
equivalent compressed linear control, ELM, L-BFGS, and learned-preconditioned
L-BFGS. Each run has a strict 100 fine-solve budget (including adjoints and
rejected trials). Observation generation is one shared setup forward outside
that inversion budget. Wall time includes fitting and inner optimization,
but excludes setup and file serialization.

Results, configuration, source hashes, and runtime metadata are saved under
`results/elm/s1/` and `results/elm/s2/`, without replacing notebook caches.
The runner refuses to overwrite existing runs; use `--output PATH` for a
separate replication. These are acoustic 2D toy experiments, not SPECFEM3D
validation.

See [ELM_REPORT.md](./ELM_REPORT.md) for the measured comparison: the tested
ELM correction did not reliably improve the original trust-region results
and did not beat learned-preconditioned L-BFGS.

## Cost at matched quality

`run_quality_cost.py` compares ordinary L-BFGS with the existing bounded
learned preconditioner (`illum_bound=3`). Each method runs to 100 solve
equivalents; the baseline's final misfit and model error define the paired
quality thresholds. The runner measures the first accepted model meeting
each threshold, and both thresholds simultaneously.

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_quality_cost.py --repeats 3
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_quality_cost.py --repeats 3
python -m unittest -v test_quality_cost
```

Synchronized forward-completion timestamps include all preceding forward,
adjoint, line-search, and preconditioner-training work. The methods run
sequentially, with alternating order across repetitions. Observation setup,
one warmup forward per scenario, and serialization are outside inversion
timing. Results and threshold models are stored in `results/quality-cost/`;
existing results are never overwritten (use `--output` to replicate).

See [QUALITY_COST_REPORT.md](./QUALITY_COST_REPORT.md) for measured savings.
At matched waveform and model quality, the bounded preconditioner saves
66.3% of solves and 51.9% of time in scenario 1; in scenario 2 it saves
27.8% of solves but takes 4.8% longer on the measured GPU.

## Direct nonnegative preconditioner fitting

`direct_precond.py` replaces iterative Adam training with a four-coefficient
nonnegative ridge fit: identity plus three fixed Gaussian covariance filters
at scales 1, 2, and 4 cells. It preserves positive definiteness using a
positive identity floor. The constrained fit uses NNLS on a small Cholesky
factor of the normal equations; it is not an analytic closed-form solution.

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_direct_experiments.py --repeats 3
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_direct_experiments.py --repeats 3
python -m unittest -v test_direct_precond
```

The benchmark compares L-BFGS, the existing bounded learned preconditioner,
direct fitting, and an equal-weight fixed-filter control, on both standard
scenarios. Each run uses 100 solve equivalents. Thresholds come from each
paired L-BFGS final result, with misfit, model-error and simultaneous-quality
first passage reported separately. Failure to reach a target remains an
explicit result. Synchronized component timings measure fitting/application;
the time-to-quality includes those costs. Outputs are separate under
`results/direct-fit/`; use `--output` for a new replication directory.

Measured outcome: fitting time fell by about 98%, but the direct variant
did not reach baseline joint quality in 100 solves on either scenario.
See [DIRECT_FIT_REPORT.md](./DIRECT_FIT_REPORT.md) for the full negative
result, controls, timing and verification.

## Inner optimizer pilot

`inner_lbfgs.py` changes only the training optimizer of the existing bounded
learned operator. It uses a fresh inner L-BFGS with strong-Wolfe line search,
up to 20 iterations and a nominal evaluation budget of 40 per update.
Actual closure/gradient evaluations are recorded because line search may
exceed that stopping budget. Parameters are warm-started, optimizer memory
is reset, and the operator, loss and outer FWI optimizer are unchanged.

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_inner_optimizers.py
python -m unittest -v test_inner_lbfgs
```

This is a one-scenario pilot without repetitions: L-BFGS baseline, Adam-200,
Adam-50, and inner L-BFGS-20, each with 100 solve equivalents. It reuses the
first-passage and synchronized component timing protocol. Outputs are under
`results/inner-optimizers/s2/`, with no overwriting (`--output` for replication).

See [INNER_OPTIMIZER_REPORT.md](./INNER_OPTIMIZER_REPORT.md): in this single
run, inner L-BFGS reached joint quality in 45.30 s, versus 51.02 s for
Adam-200 and 46.96 s for conventional L-BFGS. The small 3.5% advantage over
the conventional baseline is not a robust speedup claim without repetitions.

## LODO optimizer / preconditioner

`lodo.py` implements LODO (Liao et al., TMLR 2023, arXiv:2210.06171): an
online-learned inverse Hessian `G(θ)`, built from random permutations and 4×4
blocks, and trained with the one-step hypergradient. Two variants are tested:
the paper's Algorithm 1 (`LODO`), and `G(θ)` as the initial inverse Hessian
`H0` inside L-BFGS (`LODO-PC`).

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 python run_lodo.py
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 python run_lodo.py
python -m unittest -v test_lodo
```

Outputs are written to `results/lodo/s{1,2}/`, with one run per arm.

See [LODO_REPORT.md](./LODO_REPORT.md): pure LODO ends with a 46–158× worse
misfit than L-BFGS, and `LODO-PC` is neutral (95 vs 98 and 97 vs 100 solves to
L-BFGS quality). Neither approaches the bounded learned preconditioner
(33 and 69 solves).

## Multifidelity test (MG/OPT)

`mgopt.py` implements classical MG/OPT (Nash 2000), with first-order gradient
correction on the coarse `WaveGrid(2)` grid. It also provides a coarse-to-fine
control and an MG/OPT → L-BFGS hybrid. The runner also includes the anchored
data-space surrogate with and without the learned correction.

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 python run_mgopt.py
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 python run_mgopt.py
python -m unittest -v test_mgopt
```

Outputs are written to `results/mgopt/s{1,2}/`, with one run per arm.

See [MGOPT_REPORT.md](./MGOPT_REPORT.md): no coarse-grid arm reaches the fine
L-BFGS quality within 100 equivalents. The coarse grid has a 32% data error
from dispersion. MG/OPT wins only during the first ~10 solves, and the hybrid
ends level with L-BFGS. A learned coarse-to-fine correction is therefore not
justified in this setting.
