# Multifidelity test: MG/OPT and coarse-grid controls

## Question

Can a coarse grid save fine-grid solves in the toy FWI? This is the gate for a
learned coarse-to-fine correction. If plain multifidelity does not save solves,
a learned correction is not worth building.

## Setup

- Toy problem: 2D acoustic FD, `FWI_ACQ=wells`, `FWI_INIT=smooth`, no noise.
- Budget: 100 fine-solve equivalents per arm.
- Coarse grid: `WaveGrid(2)` (`ml_ideas.Fc`). Grid spacing and time step are
  doubled. One coarse solve costs 0.125 fine solve (cells × time steps), so a
  coarse forward + adjoint costs 0.25.
- Repetitions: one run per arm. CUDA nondeterminism makes single runs slightly
  noisy.

Code: [mgopt.py](./mgopt.py), [run_mgopt.py](./run_mgopt.py),
[test_mgopt.py](./test_mgopt.py) (3 tests pass). Results are in
`results/mgopt/s{1,2}/`.

| Arm | ML? | Description |
|---|---|---|
| L-BFGS | no | Fine-grid baseline |
| C2F-10 / C2F-25 | no | L-BFGS on the coarse misfit for 10 or 25 equivalents, then fine L-BFGS |
| MGOPT-4 / MGOPT-10 | no | Classical MG/OPT (Nash 2000): the coarse model plus a first-order gradient correction is minimized for 4 or 10 inner L-BFGS iterations in a ball of radius Δ ≤ 0.12, then checked with a fine Armijo line search |
| MGOPT-10-sw | no | Hybrid: MG/OPT until the first fine backtrack (α < 1), then fine L-BFGS |
| MF-TR-10 | no | Anchored data-space surrogate (`ml_ideas.run_mf_trust_region`), 10 inner iterations, Δmax 0.12 |
| MF-TR+corr-10 | yes | Same, plus the learned data-space correction (a filter per offset bin, fitted online) |

## Diagnostic: the coarse grid is inaccurate

The coarse grid has about 2.4 points per minimum wavelength. Computed at
`v_true`, its data differ from the observed data by 32% (RMS), and this does
not shrink with a better scale (0.98) or a time shift (0–1 samples). This is
numerical dispersion, not a bug.

The coarse and fine gradients also disagree. The table gives their cosine
(full band):

| Scenario | v_init | Halfway | Near v_true |
|---|---|---|---|
| s1 | 0.79 | 0.73 | 0.65 |
| s2 | 0.94 | 0.90 | 0.79 |

The coarse misfit stays about 73 even near `v_true`, while the fine misfit
drops to about 0.04–0.12.

## Results

### Final values

J and model error after 100 equivalents. The last column is the number of
solves needed to reach both the final J and the final error of L-BFGS.

| Arm | s1 J | s1 err | s2 J | s2 err | Solves to L-BFGS quality (s1 / s2) |
|---|---|---|---|---|---|
| L-BFGS | 2.55e-3 | 0.0069 | 1.95e-3 | 0.0034 | 98 / 100 |
| C2F-10 | 1.83e-1 | 0.0154 | 2.12e-1 | 0.0171 | never / never |
| C2F-25 | 1.15 | 0.0402 | 1.34 | 0.0442 | never / never |
| MGOPT-4 | 1.12e-2 | 0.0082 | 2.82e-2 | 0.0064 | never / never |
| MGOPT-10 | 5.53e-3 | 0.0073 | 1.44e-2 | 0.0050 | never / never |
| MGOPT-10-sw | 2.76e-3 | 0.0070 | 2.05e-3 | 0.0034 | never / never |
| MF-TR-10 | 7.72e-3 | 0.0079 | 2.21e-2 | 0.0063 | never / never |
| MF-TR+corr-10 | 8.79e-3 | 0.0080 | 1.62e-2 | 0.0058 | never / never |
| *L-BFGS+P-b3 (LODO run)* | *4.31e-5* | *0.0050* | *2.78e-4* | *0.0028* | *33 / 69* |

The starting error is 0.0305 (s1) and 0.0234 (s2). MG/OPT overshoots the
budget by at most 0.8 equivalents because the coarse evaluation at x_k is not
budget-checked. That overshoot favors MG/OPT, and it still loses.

### Quality at fixed budget (J / error)

| Arm | s1 @10 | s1 @30 | s1 @50 | s2 @10 | s2 @30 | s2 @50 |
|---|---|---|---|---|---|---|
| L-BFGS | 5.2e-1 / .0200 | 2.2e-2 / .0090 | 6.8e-3 / .0077 | 4.2e-1 / .0146 | 5.0e-2 / .0075 | 1.1e-2 / .0053 |
| MGOPT-4 | **2.3e-1 / .0151** | 3.6e-2 / .0094 | 1.8e-2 / .0087 | **3.8e-1 / .0130** | 9.8e-2 / .0092 | 6.4e-2 / .0081 |
| MGOPT-10-sw | 4.1e-1 / .0151 | 5.4e-2 / .0101 | 1.0e-2 / .0081 | 9.5e-1 / .0148 | 1.2e-1 / .0081 | 1.7e-2 / .0056 |
| MF-TR+corr-10 | 1.2 / .0250 | 1.1e-1 / .0127 | 1.5e-2 / .0086 | 3.0 / .0194 | 9.6e-2 / .0091 | 5.9e-2 / .0082 |

## Conclusion

**In this toy, multifidelity does not save fine solves.** No coarse-grid arm
reaches the L-BFGS final quality within 100 equivalents.

- **Coarse to fine.** The coarse optimum is biased by dispersion. After the
  coarse phase, the model error *increases* to 0.055–0.068, worse than the
  starting model, and the fine phase cannot recover it within the budget.
- **MG/OPT.** The first-order correction removes the bias at x_k, and the
  first coarse steps work. MGOPT-4 is better than L-BFGS at 10 solves (J 2.3e-1
  vs 5.2e-1 in s1). After that, the coarse model stops predicting the fine
  steps: the fine line search backtracks, and each outer iteration costs about
  2 fine solves plus several coarse ones. L-BFGS overtakes it by about 20–30
  solves.
- **Hybrid.** MGOPT-10-sw switches after 12–15 solves and ends at L-BFGS
  quality within noise (s1 2.76e-3 vs 2.55e-3; s2 2.05e-3 vs 1.95e-3). It loses
  the early advantage because L-BFGS restarts without memory.
- **Learned correction (ML).** The learned data-space correction does not
  change the picture. Against MF-TR-10, it is worse in s1 (J 8.8e-3 vs 7.7e-3)
  and better in s2 (1.6e-2 vs 2.2e-2), both within the noise of a single run.
  Since the non-ML multifidelity arms do not save solves, a learned
  coarse-to-fine correction is **not justified** in this setting.

The learned bounded preconditioner (`L-BFGS+P-b3`) remains the only method
that reaches L-BFGS quality earlier (33 and 69 solves).

## Limits

- The coarse grid is `fac=2` at the same frequency band, which makes it very
  inaccurate (32% data error). A coarse grid that is accurate for its band, as
  in standard multiscale FWI with one SPECFEM mesh per frequency band, is a
  different and well-established method. It was not tested here.
- With a 2 Hz low-pass, the coarse-fine cosine was *lower* (0.35–0.86). This
  is unexplained, and possibly related to the band filter or to the
  coarse-to-fine interpolation of the traces. It was not investigated.
- Single runs, two scenarios, smooth initial model only. MG/OPT parameters
  (Δ, number of inner iterations) were not tuned. MF-TR logs J before each
  step, which slightly delays its first passage.
- This is a 2D toy, not SPECFEM3D. In 3D the coarse/fine cost ratio would be
  smaller (about 1/16 instead of 1/8), but the accuracy problem would be the
  same if the coarse mesh breaks the points-per-wavelength rule.
