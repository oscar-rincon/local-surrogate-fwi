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
