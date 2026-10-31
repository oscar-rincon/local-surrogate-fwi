# Geometric seismic-wave inversion with BEM

This repository contains an executable, educational implementation of the
geometric full-waveform inversion in Bignardi et al.,
["Geometric Seismic-Wave Inversion by the Boundary Element Method"
(2012)](https://doi.org/10.1785/0120110091).
Open [geometric_seismic_inversion.ipynb](./geometric_seismic_inversion.ipynb)
for the equations, synthetic experiments, plots, and interpretation.

The implementation solves coupled two-dimensional plane-strain elastic waves
with a boundary-element forward model and estimates one interface using
finite-difference sensitivities and regularized Gauss-Newton updates. It includes
the pressurized-cylinder benchmark, flat and irregular surfaces, both impedance
contrasts, amplitude/phase noise, vertical-only measurements, and an experiment
with 33 interface parameters. It is an independent, simplified implementation,
**not an exact reproduction of the authors' code or field results**. The
notebook spells out the differences, including the element discretization and
synthetic target geometry.

## Run the notebook

Create a Python environment, install the dependencies, and select that
environment as the notebook kernel in VS Code:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all cells from the repository root. The expanded set of inversions can take
10 minutes or longer on a CPU; fixed random seeds make the noise examples repeatable.
The notebook contains saved output from one run, but rerunning it computes
the results again.

The forward and inversion routines live in [geometric_bem.py](./geometric_bem.py);
the cylinder benchmark is in [cylinder_bem.py](./cylinder_bem.py). To run the
numerical checks:

```bash
python -m pytest -q
```