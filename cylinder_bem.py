"""Radial vibration benchmark for a uniformly pressurized elastic cylinder."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv

from geometric_bem import ElasticMedium, _green_and_traction


def _validate(radius: float, omega: float, pressure: float) -> None:
    if not np.all(np.isfinite([radius, omega, pressure])):
        raise ValueError("Radius, angular frequency and pressure must be finite.")
    if radius <= 0 or omega <= 0:
        raise ValueError("Radius and angular frequency must be finite and positive.")


def cylinder_first_frequency(radius: float, medium: ElasticMedium) -> float:
    """First undamped, axisymmetric plane-strain natural frequency in rad/s."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be finite and positive.")
    mu = medium.density * medium.vs**2
    compression = medium.density * medium.vp**2

    def boundary_equation(value: float) -> float:
        return compression * value * jv(0, value) - 2 * mu * jv(1, value)

    root = brentq(boundary_equation, 1e-8, float(jn_zeros(0, 1)[0]))
    return float(root * medium.vp / radius)


def cylinder_analytic(
    radius: float, omega: float, medium: ElasticMedium, pressure: float = 100.0
) -> complex:
    """Exact boundary displacement for inward uniform pressure in plane strain."""
    _validate(radius, omega, pressure)
    kp, _ = medium.wavenumbers(omega)
    argument = kp * radius
    denominator = (
        (medium.lam + 2 * medium.mu) * kp * jv(0, argument)
        - 2 * medium.mu * jv(1, argument) / radius
    )
    if denominator == 0:
        raise ValueError("The undamped cylinder is exactly at a radial resonance.")
    return complex(-pressure * jv(1, argument) / denominator)


def cylinder_bem(
    radius: float,
    omega: float,
    medium: ElasticMedium,
    pressure: float = 100.0,
    elements: int = 32,
    quadrature_order: int = 12,
) -> complex:
    """Average radial displacement using constant densities on circular arcs."""
    _validate(radius, omega, pressure)
    if elements < 4:
        raise ValueError("The cylinder requires at least four boundary elements.")
    if quadrature_order < 4 or quadrature_order % 2:
        raise ValueError("Quadrature order must be even and at least 4.")
    angles = 2 * np.pi * (np.arange(elements) + 0.5) / elements
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    targets = radius * normals
    angle_width = 2 * np.pi / elements
    nodes, weights = leggauss(quadrature_order)
    source_angles = angles[:, None] + angle_width * nodes[None, :] / 2
    source_points = radius * np.stack((np.cos(source_angles), np.sin(source_angles)), axis=-1)
    offsets = targets[:, None, None, :] - source_points[None, :, :, :]
    green, traction = _green_and_traction(offsets, normals[:, None, None, :], medium, omega)
    panel_weights = radius * angle_width * weights / 2
    displacement_blocks = np.einsum("pqkij,k->pqij", green, panel_weights)
    traction_blocks = np.einsum("pqkij,k->pqij", traction, panel_weights)

    t = (nodes + 1) / 2
    transformed_weights = weights / 2 * radius * angle_width * t
    offset_angles = angle_width * t**2 / 2
    self_green = np.zeros((elements, 2, 2), dtype=complex)
    self_traction = np.zeros_like(self_green)
    for sign in (-1, 1):
        source_angles = angles[:, None] + sign * offset_angles[None, :]
        source_points = radius * np.stack(
            (np.cos(source_angles), np.sin(source_angles)), axis=-1
        )
        green, traction = _green_and_traction(
            targets[:, None, :] - source_points, normals[:, None, :], medium, omega
        )
        self_green += np.einsum("pkij,k->pij", green, transformed_weights)
        self_traction += np.einsum("pkij,k->pij", traction, transformed_weights)
    diagonal = np.arange(elements)
    displacement_blocks[diagonal, diagonal] = self_green
    traction_blocks[diagonal, diagonal] = self_traction

    def flatten(blocks):
        return blocks.transpose(0, 2, 1, 3).reshape(2 * elements, 2 * elements)

    displacement_matrix = flatten(displacement_blocks)
    traction_matrix = flatten(traction_blocks) + 0.5 * np.eye(2 * elements)
    density = solve(traction_matrix, (-pressure * normals).ravel())
    displacement = (displacement_matrix @ density).reshape(elements, 2)
    return complex(np.mean(np.sum(displacement * normals, axis=1)))


def cylinder_axisymmetric_bem(
    radius: float,
    omega: float,
    medium: ElasticMedium,
    pressure: float = 100.0,
    elements: int = 32,
    quadrature_order: int = 12,
) -> complex:
    """Solve the cylinder BEM using its known rotational symmetry."""
    _validate(radius, omega, pressure)
    if elements < 4:
        raise ValueError("The cylinder requires at least four boundary elements.")
    if quadrature_order < 4 or quadrature_order % 2:
        raise ValueError("Quadrature order must be even and at least 4.")
    normal = np.array([1.0, 0.0])
    target = radius * normal
    width = 2 * np.pi / elements
    nodes, weights = leggauss(quadrature_order)
    angles = np.arange(elements)[:, None] * width + width * nodes[None, :] / 2
    source_normals = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    green, traction = _green_and_traction(
        target - radius * source_normals, normal, medium, omega
    )
    weight = radius * width * weights / 2
    displacement = np.einsum("i,pqij,pqj,q->", normal, green, source_normals, weight)
    traction_pv = np.einsum("i,pqij,pqj,q->", normal, traction, source_normals, weight)
    displacement -= np.einsum(
        "i,qij,qj,q->", normal, green[0], source_normals[0], weight
    )
    traction_pv -= np.einsum(
        "i,qij,qj,q->", normal, traction[0], source_normals[0], weight
    )

    t = (nodes + 1) / 2
    self_weight = weights / 2 * radius * width * t
    for sign in (-1, 1):
        angles = sign * width * t**2 / 2
        source_normals = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        green, traction = _green_and_traction(
            target - radius * source_normals, normal, medium, omega
        )
        displacement += np.einsum("i,qij,qj,q->", normal, green, source_normals, self_weight)
        traction_pv += np.einsum("i,qij,qj,q->", normal, traction, source_normals, self_weight)
    return complex(-pressure * displacement / (0.5 + traction_pv))
