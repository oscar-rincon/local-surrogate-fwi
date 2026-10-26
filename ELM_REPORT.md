# ELM surrogate benchmark

Date: 2026-10-26

## Conclusion

**The tested online ELM correction does not improve the current results
reliably, and it does not beat the existing learned-preconditioned L-BFGS.**

At 100 inversion solves, averaged over three ELM feature seeds:

- Scenario 1: effectively unchanged final misfit versus the original secant
  trust region (+0.024%), with 15.7% longer runtime.
- Scenario 2: 26.4% higher final misfit and 7.6% higher model error versus
  the original secant trust region.
- Every ELM seed has worse final misfit and model error than ordinary
  L-BFGS on both scenarios.
- Relative to freshly run learned-preconditioned L-BFGS, mean ELM misfit is
  about 47 times higher in scenario 1 and 5.3 times higher in scenario 2.

This is a negative result for this particular local ELM design and budget,
not evidence that all ELM or multifidelity approaches fail.

## Experiment

The existing acoustic 2D finite-difference solver in [fwi_toy.py](./fwi_toy.py)
is the expensive verification model. **No SPECFEM3D simulation was run.**

Both existing standard scenarios were tested:

1. Velocity gradient, fast/slow Gaussian anomalies, and a thin layer.
2. Dipping faulted layers and a fast lens.

The grid is 60 x 100, with 700 time steps, a 5 Hz Ricker source, six shots,
and surface plus two-well receivers. Runs use the existing smooth
initialization, boundary taper, no added noise, and the same parameterization
and trust-region settings. The initialization incorporates true-model values
outside the active mask; these are favorable toy conditions, not a realistic
unknown-boundary field experiment.

Each inversion receives exactly **100 fine-solve equivalents**:
one forward or adjoint costs one unit, including rejected forward trials.
These runs use no Born or coarse solves and no offline training simulations.
All trial steps are verified by the fine solver before acceptance.

Generating observations costs one shared setup forward per scenario process,
outside the inversion budget. After benchmarking, an independent audit
recomputed the final objective and model error for every saved model
(14 additional diagnostic forwards total); those evaluations did not affect
optimization or reported inversion time.

All 14 inversions ran sequentially on the same NVIDIA RTX 2000 Ada Generation
Laptop GPU, using Python 3.12 and PyTorch 2.8.0+cu128. Inversion wall times
include surrogate fitting, inner optimization, and solver work, with CUDA
synchronization. They exclude observation setup, serialization, and the
post-run audit. Total measured inversion time was 1,093 seconds, about
18.2 minutes. Timing was not replicated.

## ELM design

[elm_surrogate.py](./elm_surrogate.py) implements:

- The existing low-rank linear waveform surrogate as the baseline.
- 32 fixed random tanh features, with seeds 0, 1, and 2.
- A weighted ridge fit to waveform residuals of the most recent 16 secants,
  starting after four secants are available; ridge coefficient 0.01.
- Age, distance, and subspace-projection weighting.
- Features scaled by the current trust-region radius.
- Zero correction value and derivative at the current model, preserving the
  exact objective value and gradient.
- A compressed objective in the span of the linear response and observed
  secant residuals; no waveform-sized evaluation in each inner optimizer step.

For a projected coordinate z, the anchored features are

```text
psi(z) = tanh(W z + b) - tanh(b) - sech(b)^2 * (W z).
```

Training uses feature differences between each secant's projected endpoints,
not just the step from zero, because historical secants have different base
models. Targets are the observed waveform differences minus the current
linear prediction. Rejected trials also provide training secants.

The trust-region loop keeps its default rank limit of 10 and 300 inner Adam
iterations. No hyperparameter search or seed selection was performed.
The `Linear` ablation uses the same compressed objective with ELM disabled;
`TR-secante` runs the original, uncompressed code path.

## Results at 100 inversion solves

Lower objective and model error are better. J is the existing half-squared
waveform residual after the repository's data normalization. Model error is
100 times the masked relative RMS velocity error returned by `rel_err`.
Rejections count trust-region trials only.

### Scenario 1

Initial J: 3.35009. Initial model error: 3.04689%.

| Method | Final J | Model error (%) | Time (s) | Rejections |
|---|---:|---:|---:|---:|
| Original TR-secante | 0.00331409 | 0.71291 | 73.3 | 7 |
| Compressed Linear | 0.00363659 | 0.71915 | 78.5 | 7 |
| L-BFGS | 0.00254258 | 0.69022 | 56.5 | n/a |
| L-BFGS+P | **0.00007053** | **0.49988** | 90.3 | n/a |
| ELM seed 0 | 0.00311734 | 0.70698 | 80.4 | 7 |
| ELM seed 1 | 0.00343540 | 0.71612 | 84.1 | 6 |
| ELM seed 2 | 0.00339196 | 0.71594 | 89.9 | 12 |

ELM mean +/- sample standard deviation:

- J: 0.00331490 +/- 0.00017246.
- Model error: 0.71301% +/- 0.00523 percentage points.
- Runtime: 84.8 +/- 4.8 seconds.

ELM reduces mean J by 8.8% relative to the compressed linear ablation, but is
8.1% slower. Relative to the original trust-region implementation, its mean
misfit and model error are essentially unchanged. The best ELM seed improves
J by about 5.9% over original TR, but the other two seeds do not.

### Scenario 2

Initial J: 10.60127. Initial model error: 2.33702%.

| Method | Final J | Model error (%) | Time (s) | Rejections |
|---|---:|---:|---:|---:|
| Original TR-secante | 0.00252389 | 0.35782 | 85.4 | 7 |
| Compressed Linear | 0.00282487 | 0.36967 | 74.7 | 6 |
| L-BFGS | 0.00190509 | 0.33463 | 59.9 | n/a |
| L-BFGS+P | **0.00060500** | **0.30384** | 82.3 | n/a |
| ELM seed 0 | 0.00365104 | 0.40161 | 83.8 | 6 |
| ELM seed 1 | 0.00296958 | 0.37755 | 79.5 | 6 |
| ELM seed 2 | 0.00294800 | 0.37550 | 74.6 | 6 |

ELM mean +/- sample standard deviation:

- J: 0.00318954 +/- 0.00039982.
- Model error: 0.38489% +/- 0.01452 percentage points.
- Runtime: 79.3 +/- 4.6 seconds.

ELM increases mean J by 12.9% and model error by 4.1% relative to the
compressed linear ablation, with 6.2% longer runtime. All three seeds are
worse than both linear controls in final misfit and model error.

## Relation to the archived current results

The tables above use fresh runs, not archived timings. Scanning the existing
standard-scenario caches also found a stronger archived bounded-preconditioner
variant:

| Archived method | Scenario 1 J | Scenario 2 J |
|---|---:|---:|
| L-BFGS+P-b3 | 0.00004170 | 0.00029611 |

Those cached results were not rerun here; their timing is not compared with
the new hardware. ELM does not surpass them either.

Although the compressed and original linear objectives are algebraically
equivalent and pass a focused numerical equivalence test, their final
optimization trajectories differ in float32. This is why both controls are
reported, and the modest scenario-1 gain against only the compressed control
is not treated as robust improvement.

## Validation and reproducibility

- Four tests pass in [test_elm_surrogate.py](./test_elm_surrogate.py):
  invalid-configuration rejection; linear-objective equivalence;
  nonlinear fitting on held-out synthetic points with exact first-order
  consistency; and real-solver integration with strict budget and decreasing
  accepted objectives.
- Pylance syntax checks passed for the changed Python files.
- All 14 saved outputs were reloaded and checked for finite models, exact
  100-solve counters, matching JSON summaries, and matching source hashes.
- Independent forwards of all 14 persisted final models reproduced their
  stored objectives exactly; model errors also matched.
- The original trust-region behavior remains the default. Existing notebook
  caches and notebooks were not changed.

The tested executable was
`/home/orincon/miniconda3/envs/ml-seismic-waves-env/bin/python`.
The editor currently selects `/usr/bin/python3`; Pylance reports `torch` and
`scipy` unresolved in that environment. Runtime tests used the existing
seismic environment above, not the editor interpreter. No packages were
installed and the editor selection was not changed.

Reproduce with a CUDA-compatible environment and the existing dependencies:

```bash
OMP_NUM_THREADS=1 FWI_SCENARIO=1 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_elm_experiments.py --seeds 0 1 2 --output results/elm-replication
OMP_NUM_THREADS=1 FWI_SCENARIO=2 FWI_ACQ=wells FWI_INIT=smooth FWI_NOISE=0 \
  python run_elm_experiments.py --seeds 0 1 2 --output results/elm-replication
python -m unittest -v test_elm_surrogate
```

Raw saved models, optimization logs, feature configurations, source hashes,
runtime metadata, and JSON metrics are in [results/elm](./results/elm/).
The runner refuses to overwrite runs.

## Recommendation and limits

Keep the existing learned-preconditioned L-BFGS as the preferred method for
these standard scenarios. Do not promote this ELM correction to the default
or port it to SPECFEM3D on the basis of these results.

A possible next, **untested**, experiment is learning a coarse-to-fine
discrepancy instead of a nonlinear secant residual, where there may be more
structured error to learn. A second option is a more constrained scalar
curvature correction with fewer fitted degrees of freedom.

The current experiment cannot establish why ELM fails to improve performance:
limited local data, projection error, feature scaling, and regularization are
possible explanations, not measured causes. Only two noise-free, smooth-start
toy scenarios, one ELM configuration, and three feature seeds were tested.
There are no confidence claims about generalization to noisy surface-only
acquisition, cycle skipping, 3D elastic media, or SPECFEM3D.
