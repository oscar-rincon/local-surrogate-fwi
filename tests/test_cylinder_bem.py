import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from cylinder_bem import (
    cylinder_analytic,
    cylinder_axisymmetric_bem,
    cylinder_bem,
    cylinder_first_frequency,
)
from geometric_bem import ElasticMedium


@pytest.fixture
def cylinder_medium():
    return ElasticMedium(vp=np.sqrt(30000), vs=100, density=100, damping=0.05)


def test_cylinder_resonance_matches_paper(cylinder_medium):
    omega = cylinder_first_frequency(6, cylinder_medium)
    assert abs(omega - 59.73) < 0.02
    assert abs(cylinder_analytic(6, omega, cylinder_medium)) == pytest.approx(
        1.183e-3, rel=0.002
    )


def test_cylinder_displacement_scales_linearly_with_pressure(cylinder_medium):
    omega = cylinder_first_frequency(6, cylinder_medium)
    with_100_pa = cylinder_analytic(6, omega, cylinder_medium, pressure=100)
    with_100_squared_pa = cylinder_analytic(6, omega, cylinder_medium, pressure=100**2)
    assert with_100_squared_pa == pytest.approx(100 * with_100_pa)


@pytest.mark.parametrize("frequency_ratio", [0.25, 0.5, 1.0, 1.5, 2.0])
def test_axisymmetric_bem_matches_analytic_cylinder(cylinder_medium, frequency_ratio):
    omega = frequency_ratio * cylinder_first_frequency(6, cylinder_medium)
    analytic = cylinder_analytic(6, omega, cylinder_medium)
    numerical = cylinder_axisymmetric_bem(6, omega, cylinder_medium, elements=8)
    assert abs(numerical - analytic) / abs(analytic) < 2e-4


def test_constant_density_cylinder_converges_with_refinement(cylinder_medium):
    omega = cylinder_first_frequency(6, cylinder_medium)
    exact = cylinder_analytic(6, omega, cylinder_medium)
    with threadpool_limits(limits=1, user_api="blas"):
        coarse = cylinder_bem(6, omega, cylinder_medium, elements=8)
        fine = cylinder_bem(6, omega, cylinder_medium, elements=32)
    coarse_error = abs(coarse - exact) / abs(exact)
    fine_error = abs(fine - exact) / abs(exact)
    assert fine_error < 0.10
    assert fine_error < 0.3 * coarse_error
