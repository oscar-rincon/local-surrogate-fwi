# LODO as an FWI optimizer and preconditioner

Method: LODO, *Learning to Optimize Quasi-Newton Methods* (Liao, Dangovski,
Foerster & Soljačić, TMLR 2023, [arXiv:2210.06171](https://arxiv.org/abs/2210.06171)).
LODO learns an inverse-Hessian approximation online,
`G(θ) = α0 Eᵀ G̃ G̃ᵀ E` with `G̃ = Π B(θᵢ) Pᵢ` (16 layers of random permutations
and 4×4 blocks, orthogonal at initialization so `G(θ0) = α0 I`). The learning
signal is the hypergradient of the loss after one step, `∇θ f(x_{t+1})`, used
with an Adam meta-optimizer. No meta-training on other tasks is required.

## Variants tested

- **LODO**: Algorithm 1 of the paper, `x ← x − G(θ)m` with momentum β = 0.9
  and no line search. Two changes were needed for FWI:
  - `α0 = step0/‖g0‖`, because the paper's α0 = 1 has no scale in FWI;
  - `m0 = g0`, because `m0 = 0` wastes the first two solves.

  Steps are capped at `step_max`, as in `run_lbfgs`.
- **LODO-PC**: identical to `run_lbfgs`, but with `H0 = γ G(θ)` in the two-loop
  recursion. After each accepted step, θ is updated with
  `α (∂p/∂θ)ᵀ ∇f(x + αp)`. With `meta_lr = 0` it reproduces `run_lbfgs`
  exactly (unit test).
- Meta learning rates: 1e-3 (the paper's default) and 1e-2.

Code: `lodo.py`, `run_lodo.py`, `test_lodo.py`. Results: `results/lodo/s{1,2}/`.

## Protocol

- Two scenarios, wells acquisition, smooth initial model, no noise.
- Budget of 100 solve equivalents per run.
- One run per arm, with no repetitions.
- "Solves to L-BFGS quality" is the first accepted iterate that reaches both
  the final misfit J and the final model error of the L-BFGS run.

## Results

| Scenario | Arm | Final J | Model err | Solves to L-BFGS quality | Wall time (s) |
|---|---|---:|---:|---:|---:|
| s1 | L-BFGS | 2.54e-3 | 0.0069 | 98 | 47.6 |
| s1 | L-BFGS+P-b3 | **4.31e-5** | **0.0050** | **33** | 63.5 |
| s1 | LODO | 1.16e-1 | 0.0096 | never | 48.7 |
| s1 | LODO, meta 1e-2 | 9.17 | 0.0383 | never | 57.7 |
| s1 | LODO-PC | 2.46e-3 | 0.0069 | 95 | 56.1 |
| s1 | LODO-PC, meta 1e-2 | 2.35e-3 | 0.0067 | 97 | 44.8 |
| s2 | L-BFGS | 1.97e-3 | 0.0034 | 100 | 50.6 |
| s2 | L-BFGS+P-b3 | **2.78e-4** | **0.0028** | **69** | 77.6 |
| s2 | LODO | 3.12e-1 | 0.0113 | never | 59.7 |
| s2 | LODO, meta 1e-2 | 43.4 | 0.1903 | never | 54.0 |
| s2 | LODO-PC | 1.87e-3 | 0.0033 | 97 | 48.8 |
| s2 | LODO-PC, meta 1e-2 | 4.22e-3 | 0.0038 | never | 44.8 |

The meta-optimizer costs under 1 s per run, so it is not a bottleneck. Wall
times are single runs with CUDA timing noise; differences of ±5 s are not
meaningful.

## Interpretation

- **Pure LODO does not work for FWI under this budget.** The final misfit is
  46× (s1) and 158× (s2) worse than L-BFGS. With meta learning rate 1e-2 it
  diverges. LODO was designed for thousands of cheap, noisy steps; here there
  are only about 50 deterministic steps, so θ barely learns. Without a line
  search, the method behaves like an untuned momentum gradient descent.
- **LODO as an L-BFGS preconditioner is neutral.** With meta 1e-3 it reaches
  L-BFGS quality 3 solves earlier in both scenarios (95 vs 98 and 97 vs 100).
  This is within run-to-run noise. With meta 1e-2 it is mixed: slightly better
  in s1 and worse in s2.
- **The online one-step hypergradient is a weak signal for FWI.** About 45
  updates of a 768k-parameter generic matrix cannot learn useful curvature.
  The structured learned preconditioner (`L-BFGS+P-b3`, about 141 parameters
  fitted to all secant pairs) remains far better: 33 vs 98 solves in s1 and
  69 vs 100 in s2.

## Conclusion

LODO is not a useful FWI preconditioner in this setting. The evidence again
favors preconditioners that combine a small, physically structured
parameterization with a data-rich fit (all secant pairs) over generic
high-capacity parameterizations trained with a one-step signal.

## Limitations

- These are 2D acoustic toy experiments with one run per arm. They are not
  SPECFEM3D evidence.
- Only two meta learning rates were tried, and the paper's architecture
  defaults (k = 4, N = 16) were not tuned.
- LODO might fare better in stochastic FWI with many cheap mini-batch
  (source-subsampled) iterations. That setting is closer to its design but
  was not tested.
