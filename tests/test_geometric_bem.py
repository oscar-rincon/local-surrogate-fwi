import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from geometric_bem import (
    ElasticMedium,
    InversionOptions,
    TwoLayerBEM,
    _green_and_traction,
    arc_length_terms,
    invert_interface,
    perturb_measurements,
)


def test_traction_matches_differentiated_green_tensor():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    offset = np.array([4.3, 7.1])
    normal = np.array([0.6, 0.8])
    omega = 2 * np.pi * 8
    _, traction = _green_and_traction(offset, normal, medium, omega)
    epsilon = 1e-4
    gradient = np.stack(
        [
            (
                _green_and_traction(offset + epsilon * axis, normal, medium, omega)[0]
                - _green_and_traction(offset - epsilon * axis, normal, medium, omega)[0]
            )
            / (2 * epsilon)
            for axis in np.eye(2)
        ],
        axis=-1,
    )
    expected = np.empty((2, 2), dtype=complex)
    for source_component in range(2):
        grad_u = gradient[:, source_component, :]
        stress = medium.lam * np.trace(grad_u) * np.eye(2) + medium.mu * (
            grad_u + grad_u.T
        )
        expected[:, source_component] = stress @ normal
    np.testing.assert_allclose(traction, expected, rtol=2e-7, atol=1e-10)


def test_green_tensor_satisfies_navier_equation():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    omega = 2 * np.pi * 8
    position = np.array([4.3, 7.1])
    normal = np.array([0.0, 1.0])
    step = 0.003
    axes = np.eye(2)

    def green(point):
        return _green_and_traction(point, normal, medium, omega)[0]

    center = green(position)
    laplacian = sum(
        (green(position + step * axis) - 2 * center + green(position - step * axis))
        / step**2
        for axis in axes
    )
    gradient_divergence = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            if i == j:
                mixed = (
                    green(position + step * axes[i])
                    - 2 * center
                    + green(position - step * axes[i])
                ) / step**2
            else:
                mixed = (
                    green(position + step * axes[i] + step * axes[j])
                    - green(position + step * axes[i] - step * axes[j])
                    - green(position - step * axes[i] + step * axes[j])
                    + green(position - step * axes[i] - step * axes[j])
                ) / (4 * step**2)
            gradient_divergence[i, :] += mixed[j, :]
    terms = (
        medium.mu * laplacian,
        (medium.lam + medium.mu) * gradient_divergence,
        medium.density * omega**2 * center,
    )
    assert np.linalg.norm(sum(terms)) / sum(np.linalg.norm(t) for t in terms) < 1e-5


def test_green_tensor_is_stable_near_singularity():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    kp, ks = medium.wavenumbers(1.0)
    normal = np.array([0.0, 1.0])
    first = _green_and_traction(np.array([1e-6, 0.0]), normal, medium, 1.0)[0]
    second = _green_and_traction(np.array([2e-6, 0.0]), normal, medium, 1.0)[0]
    expected = (1 + kp**2 / ks**2) * np.log(2) / (4 * np.pi * medium.mu)
    np.testing.assert_allclose(first - second, expected * np.eye(2), rtol=1e-8, atol=1e-18)


def test_single_layer_traction_has_expected_jump():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    normal = np.array([0.0, 1.0])
    omega = 2 * np.pi * 8
    epsilon = 0.001

    nodes, weights = leggauss(64)
    limit = np.arctan(5 / epsilon)
    angles = limit * nodes
    x = epsilon * np.tan(angles)
    transformed_weights = weights * limit * epsilon / np.cos(angles) ** 2
    positive = _green_and_traction(
        np.column_stack((-x, np.full_like(x, epsilon))), normal, medium, omega
    )[1]
    negative = _green_and_traction(
        np.column_stack((-x, np.full_like(x, -epsilon))), normal, medium, omega
    )[1]
    jump = np.einsum("qij,q->ij", positive - negative, transformed_weights)
    np.testing.assert_allclose(jump, -np.eye(2), rtol=2e-3, atol=2e-3)


def test_identical_media_make_interface_invisible():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    model = TwoLayerBEM(
        surface_edges=[-15, -7, -3, 1, 3, 8, 15],
        interface_x=[-15, -8, 0, 8, 15],
        control_x=[-8, 0, 8],
        receiver_x=[-5, -1],
        source_x=2,
        upper=medium,
        lower=medium,
        omega=2 * np.pi * 8,
    )
    first = model.predict([5, 8, 6])
    second = model.predict([14, 11, 12])
    np.testing.assert_allclose(first, second, rtol=1e-11, atol=1e-15)


def test_flat_surface_default_matches_explicit_zero_topography():
    medium = ElasticMedium(vp=500, vs=150, density=1600, damping=0.05)
    edges = np.array([-15, -7, -3, 1, 3, 8, 15])
    arguments = (
        edges,
        [-15, -8, 0, 8, 15],
        [-8, 0, 8],
        [-5, -1],
        2,
        medium,
        ElasticMedium(vp=1000, vs=250, density=2000),
        1.0,
    )
    flat = TwoLayerBEM(*arguments).predict([5, 8, 6])
    explicit = TwoLayerBEM(*arguments, surface_depths=np.zeros_like(edges)).predict([5, 8, 6])
    np.testing.assert_allclose(flat, explicit, rtol=1e-13, atol=1e-18)


def test_irregular_surface_is_translation_invariant_and_interface_cannot_cross_it():
    edges = np.array([-15, -7, -3, 1, 3, 8, 15])
    surface = np.array([0.0, -0.5, -1.5, -2.0, -1.0, 0.0, 0.0])
    arguments = (
        edges,
        [-15, -8, 0, 8, 15],
        [-8, 0, 8],
        [-5, -1],
        2,
        ElasticMedium(vp=500, vs=150, density=1600),
        ElasticMedium(vp=1000, vs=250, density=2000),
        1.0,
    )
    model = TwoLayerBEM(*arguments, surface_depths=surface)
    shifted = TwoLayerBEM(*arguments, surface_depths=surface + 4)
    np.testing.assert_allclose(
        model.predict([5, 8, 6]), shifted.predict([9, 12, 10]), rtol=1e-12, atol=1e-17
    )
    crossing_surface = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0])
    crossing = TwoLayerBEM(*arguments, surface_depths=crossing_surface)
    with pytest.raises(ValueError, match="below the free surface"):
        crossing.predict([5, 5, 5])


def test_tilted_geometry_and_normal_source_rotate_the_response():
    edges = np.array([-15, -7, -3, 1, 3, 8, 15], dtype=float)
    interface_x = np.array([-15, -8, 0, 8, 15], dtype=float)
    controls = interface_x.copy()
    receivers = np.array([-5, -1], dtype=float)
    upper = ElasticMedium(vp=500, vs=150, density=1600)
    lower = ElasticMedium(vp=1000, vs=250, density=2000)
    theta = 0.1
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    flat = TwoLayerBEM(edges, interface_x, controls, receivers, 2, upper, lower, 1.0)
    tilted = TwoLayerBEM(
        edges * c,
        interface_x * c - 10 * s,
        controls * c - 10 * s,
        receivers * c,
        2 * c,
        upper,
        lower,
        1.0,
        surface_depths=edges * s,
    )
    expected = flat.predict(np.full(len(controls), 10.0)) @ rotation.T
    actual = tilted.predict(controls * s + 10 * c)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-14)


def test_inversion_with_irregular_surface_reduces_misfit():
    model = TwoLayerBEM(
        surface_edges=[-15, -7, -3, 1, 3, 8, 15],
        interface_x=[-15, -8, 0, 8, 15],
        control_x=[-8, 0, 8],
        receiver_x=[-5, -1],
        source_x=2,
        upper=ElasticMedium(vp=500, vs=150, density=1600),
        lower=ElasticMedium(vp=1000, vs=250, density=2000),
        omega=1.0,
        surface_depths=[0, -0.5, -1.5, -2, -1, 0, 0],
    )
    target = np.array([5.0, 8.0, 6.0])
    initial = np.full(3, 12.0)
    with threadpool_limits(limits=1, user_api="blas"):
        observed = model.predict(target)
        result = invert_interface(
            model, observed, initial, InversionOptions(max_iterations=8, alpha_end=1e-5)
        )
    assert result.history[-1].relative_misfit < 0.3 * result.history[0].relative_misfit
    assert np.linalg.norm(result.depths - target) < np.linalg.norm(initial - target)


def test_surface_response_approximately_obeys_reciprocity():
    central = np.r_[[-29], np.arange(-27, 0, 2), [0, 2, 4]]
    controls = np.linspace(-30, 0, 9)
    left = np.arange(-120, -30, 20)
    right = np.arange(10, 121, 20)
    surface = np.r_[left, central, right]
    interface = np.r_[left, controls, right]
    upper = ElasticMedium(vp=500, vs=150, density=1600)
    lower = ElasticMedium(vp=2000, vs=800, density=2200)
    target = np.interp(
        controls, [-30, -25, -20, -16, -12, -6, 0], [10, 10, 7.5, 11, 15, 20, 22]
    )
    with threadpool_limits(limits=1, user_api="blas"):
        first = TwoLayerBEM(
            surface, interface, controls, [-2], 1, upper, lower, 1.0
        ).predict(target)[0, 1]
        second = TwoLayerBEM(
            surface, interface, controls, [1], -2, upper, lower, 1.0
        ).predict(target)[0, 1]
    assert abs(first - second) / abs(first) < 0.005


def test_arc_length_derivatives_match_finite_differences():
    x = np.array([-4.0, -1.0, 2.0, 7.0])
    depths = np.array([4.0, 5.0, 7.0, 6.0])
    _, gradient, hessian = arc_length_terms(x, depths)
    step = 1e-5
    numerical_gradient = np.empty_like(depths)
    numerical_hessian = np.empty_like(hessian)
    for index in range(len(depths)):
        delta = np.zeros_like(depths)
        delta[index] = step
        length_plus, gradient_plus, _ = arc_length_terms(x, depths + delta)
        length_minus, gradient_minus, _ = arc_length_terms(x, depths - delta)
        numerical_gradient[index] = (length_plus - length_minus) / (2 * step)
        numerical_hessian[:, index] = (gradient_plus - gradient_minus) / (2 * step)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-8, atol=1e-9)


def test_inversion_reduces_misfit_and_geometry_error():
    model = TwoLayerBEM(
        surface_edges=[-35, -15, -9, -5, -1, 1, 3, 7, 15, 35],
        interface_x=[-35, -12, -6, 0, 6, 35],
        control_x=[-12, -6, 0, 6],
        receiver_x=[-12, -7, -3],
        source_x=2,
        upper=ElasticMedium(vp=500, vs=150, density=1600),
        lower=ElasticMedium(vp=2000, vs=800, density=2200),
        omega=1.0,
    )
    target = np.array([6.0, 7.0, 9.0, 10.0])
    initial = np.full(4, 12.0)
    data = model.predict(target)
    result = invert_interface(
        model,
        data,
        initial,
        InversionOptions(max_iterations=8, alpha_end=1e-5, max_step=2.0),
    )
    assert result.history[-1].relative_misfit < 0.25 * result.history[0].relative_misfit
    assert np.linalg.norm(result.depths - target) < np.linalg.norm(initial - target)


def test_measurement_noise_perturbs_only_requested_quantity():
    data = np.array([[1 + 1j, 1 - 2j], [3 - 1j, -2 - 1j]])
    amplitude_noise = perturb_measurements(data, 0.05, "amplitude", np.random.default_rng(1))
    phase_noise = perturb_measurements(data, 0.05, "phase", np.random.default_rng(1))
    np.testing.assert_allclose(np.angle(amplitude_noise), np.angle(data), atol=1e-15)
    np.testing.assert_allclose(np.abs(phase_noise), np.abs(data), atol=1e-15)
    assert not np.allclose(amplitude_noise, data)
    assert not np.allclose(phase_noise, data)


@pytest.mark.parametrize("kind", ["amplitude", "phase"])
def test_noise_uses_paper_three_sigma_scale_for_each_component(kind):
    data = np.array([[1 + 1j, 2 - 1j], [3 - 1j, -1 - 2j]])
    level = 0.1
    amplitude = np.abs(data)
    phase = np.angle(data)
    quantity = amplitude if kind == "amplitude" else np.abs(phase)
    sigma = level * np.max(quantity, axis=0) / 3
    noise = np.random.default_rng(42).normal(0, sigma, size=data.shape)
    expected = (
        (amplitude + noise) * np.exp(1j * phase)
        if kind == "amplitude"
        else amplitude * np.exp(1j * (phase + noise))
    )
    actual = perturb_measurements(data, level, kind, np.random.default_rng(42))
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
