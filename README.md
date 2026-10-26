# local-surrogate-fwi
Toy 2D test of a local surrogate + trust-region scheme for FWI.

- `surrogate_tr_fwi.ipynb`: acoustic 2D finite-difference solver in PyTorch (stand-in for SPECFEM3D, with counted
  forward/adjoint/Born simulations), low-rank surrogate with first-order consistency, trust-region loop, and comparison
  against L-BFGS at equal simulation budget.
- Requirements: `torch`, `numpy`, `scipy`, `matplotlib` (GPU recommended; full run ≈ 12 min on an RTX 4060).
