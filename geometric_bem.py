"""A small plane-strain BEM for single-frequency geometric wave inversion.

The method is an educational, constant-element implementation of the ideas in
Bignardi et al. (2012), not the quadratic-element code used in the paper.
Coordinates are (x, depth), with depth increasing downwards.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve
from scipy.special import hankel1


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ElasticMedium:
    """Homogeneous isotropic material; damping uses the exp(-i omega t) convention."""

    vp: float
    vs: float
    density: float
    damping: float = 0.05

    def __post_init__(self) -> None:
        values = np.asarray([self.vp, self.vs, self.density, self.damping])
        if not np.all(np.isfinite(values)):
            raise ValueError("Material parameters must be finite.")
        if self.vp <= 0 or self.vs <= 0 or self.density <= 0:
            raise ValueError("Density and wave speeds must be positive.")
        if self.vp**2 <= 4 * self.vs**2 / 3:
            raise ValueError("The material must have a positive bulk modulus.")
        if not 0 <= self.damping < 0.5:
            raise ValueError("Damping must be in [0, 0.5).")

    @property
    def mu(self) -> complex:
        return self.density * self.vs**2 * (1 - 2j * self.damping)

    @property
    def lam(self) -> complex:
        return self.density * (self.vp**2 - 2 * self.vs**2) * (
            1 - 2j * self.damping
        )

    def wavenumbers(self, omega: float) -> tuple[complex, complex]:
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("Angular frequency must be finite and positive.")
        kp = np.sqrt(self.density * omega**2 / (self.lam + 2 * self.mu))
        ks = np.sqrt(self.density * omega**2 / self.mu)
        return kp, ks


@dataclass(frozen=True)
class _Boundary:
    nodes: FloatArray
    centers: FloatArray
    tangents: FloatArray
    normals: FloatArray
    lengths: FloatArray

    @classmethod
    def from_nodes(cls, nodes: ArrayLike, normal_side: int) -> "_Boundary":
        points = np.asarray(nodes, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Boundary nodes must have shape (n >= 2, 2).")
        if not np.all(np.isfinite(points)) or np.any(np.diff(points[:, 0]) <= 0):
            raise ValueError("Boundary nodes must be finite and ordered by x.")
        if normal_side not in (-1, 1):
            raise ValueError("normal_side must be -1 or 1.")
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        tangents = segments / lengths[:, None]
        normals = normal_side * np.column_stack((-tangents[:, 1], tangents[:, 0]))
        return cls(points, (points[:-1] + points[1:]) / 2, tangents, normals, lengths)


def _green_and_traction(
    offsets: FloatArray,
    normals: FloatArray,
    medium: ElasticMedium,
    omega: float,
) -> tuple[ComplexArray, ComplexArray]:
    """Full-space displacement Green tensor and its observation-point traction."""
    radii = np.linalg.norm(offsets, axis=-1)
    if np.any(radii == 0):
        raise ValueError("The Green tensor is singular at zero separation.")
    direction = offsets / radii[..., None]
    kp, ks = medium.wavenumbers(omega)
    r = np.ravel(radii)
    small = np.maximum(abs(kp), abs(ks)) * r < 1e-3
    a, b, da, db = (np.empty_like(r, dtype=complex) for _ in range(4))
    factor = 1j / (4 * medium.mu)
    if np.any(~small):
        regular_r = r[~small]
        sp, ss = kp * regular_r, ks * regular_r
        h0p, h1p, h2p, h3p = (hankel1(n, sp) for n in range(4))
        h0s, h1s, h2s, h3s = (hankel1(n, ss) for n in range(4))
        f1 = -ks * h1s + kp * h1p
        f2 = (ks**2 * (h2s - h0s) - kp**2 * (h2p - h0p)) / 2
        f3 = (ks**3 * (3 * h1s - h3s) - kp**3 * (3 * h1p - h3p)) / 4
        a[~small] = factor * (h0s + f1 / (ks**2 * regular_r))
        b[~small] = factor * (f2 - f1 / regular_r) / ks**2
        da[~small] = factor * (
            -ks * h1s + (f2 / regular_r - f1 / regular_r**2) / ks**2
        )
        db[~small] = factor * (
            f3 - f2 / regular_r + f1 / regular_r**2
        ) / ks**2
    if np.any(small):
        # The small-argument expansion avoids cancellation of singular Hankel terms.
        near_r = r[small]
        ratio = kp**2 / ks**2
        log_s = np.log(ks * near_r / 2) + np.euler_gamma
        log_p = np.log(kp * near_r / 2) + np.euler_gamma
        a[small] = 1j * (1 + ratio) / (8 * medium.mu) - (
            log_s + ratio * log_p + (1 - ratio) / 2
        ) / (4 * np.pi * medium.mu)
        b[small] = (1 - ratio) / (4 * np.pi * medium.mu)
        da[small] = -(1 + ratio) / (4 * np.pi * medium.mu * near_r)
        db[small] = 0
    a, b, da, db = (value.reshape(radii.shape) for value in (a, b, da, db))

    identity = np.eye(2)
    ee = direction[..., :, None] * direction[..., None, :]
    en = direction[..., :, None] * normals[..., None, :]
    ne = normals[..., :, None] * direction[..., None, :]
    dot = np.sum(direction * normals, axis=-1)
    green = a[..., None, None] * identity + b[..., None, None] * ee
    traction = (
        medium.lam * (da + db + b / radii)[..., None, None] * ne
        + medium.mu
        * (
            da[..., None, None] * (dot[..., None, None] * identity + en)
            + 2 * db[..., None, None] * dot[..., None, None] * ee
            + (b / radii)[..., None, None]
            * (2 * ne + en + dot[..., None, None] * identity - 4 * dot[..., None, None] * ee)
        )
    )
    return green, traction


def _operators(
    targets: FloatArray,
    normals: FloatArray,
    source: _Boundary,
    medium: ElasticMedium,
    omega: float,
    order: int,
    self_interaction: bool = False,
) -> tuple[ComplexArray, ComplexArray]:
    """Integrate constant single-layer densities over straight boundary panels."""
    nodes, weights = leggauss(order)
    points = source.centers[:, None, :] + (
        source.lengths[:, None, None] * nodes[None, :, None] / 2
    ) * source.tangents[:, None, :]
    offsets = targets[:, None, None, :] - points[None, :, :, :]
    green, traction = _green_and_traction(
        offsets, normals[:, None, None, :], medium, omega
    )
    panel_weights = source.lengths[:, None] * weights[None, :] / 2
    displacement_blocks = np.einsum("tpqij,pq->tpij", green, panel_weights)
    traction_blocks = np.einsum("tpqij,pq->tpij", traction, panel_weights)

    if self_interaction:
        if targets.shape != source.centers.shape or not np.allclose(
            targets, source.centers
        ):
            raise ValueError("Self-interaction targets must be the panel centers.")
        # A quadratic change of variables resolves the logarithmic self singularity.
        t = (nodes + 1) / 2
        wt = weights / 2
        distance = source.lengths[:, None] * t[None, :] ** 2 / 2
        self_offsets = distance[..., None] * source.tangents[:, None, :]
        self_green, _ = _green_and_traction(
            self_offsets, source.normals[:, None, :], medium, omega
        )
        self_weights = 2 * source.lengths[:, None] * t[None, :] * wt[None, :]
        diagonal = np.arange(len(source.lengths))
        displacement_blocks[diagonal, diagonal] = np.einsum(
            "pqij,pq->pij", self_green, self_weights
        )
        traction_blocks[diagonal, diagonal] = 0

    def flatten(blocks: ComplexArray) -> ComplexArray:
        return blocks.transpose(0, 2, 1, 3).reshape(
            2 * len(targets), 2 * len(source.lengths)
        )

    return flatten(displacement_blocks), flatten(traction_blocks)


class TwoLayerBEM:
    """Indirect BEM for a piecewise-linear free surface and one interface."""

    def __init__(
        self,
        surface_edges: ArrayLike,
        interface_x: ArrayLike,
        control_x: ArrayLike,
        receiver_x: ArrayLike,
        source_x: float,
        upper: ElasticMedium,
        lower: ElasticMedium,
        omega: float,
        force: float = 1000.0,
        quadrature_order: int = 12,
        surface_depths: Optional[ArrayLike] = None,
    ) -> None:
        edges = np.asarray(surface_edges, dtype=float)
        self.interface_x = np.asarray(interface_x, dtype=float)
        self.control_x = np.asarray(control_x, dtype=float)
        receivers = np.asarray(receiver_x, dtype=float)
        for name, array in (
            ("surface_edges", edges),
            ("interface_x", self.interface_x),
            ("control_x", self.control_x),
        ):
            if (
                array.ndim != 1
                or len(array) < 2
                or not np.all(np.isfinite(array))
                or np.any(np.diff(array) <= 0)
            ):
                raise ValueError(f"{name} must be a finite, strictly increasing array.")
        if (
            self.control_x[0] < self.interface_x[0]
            or self.control_x[-1] > self.interface_x[-1]
        ):
            raise ValueError("Control nodes must lie inside the interface domain.")
        if receivers.ndim != 1 or len(receivers) == 0 or not np.all(np.isfinite(receivers)):
            raise ValueError("receiver_x must contain at least one finite coordinate.")
        if not np.isfinite(source_x) or not np.isfinite(force):
            raise ValueError("Source coordinate and force must be finite.")
        if quadrature_order < 4 or quadrature_order % 2:
            raise ValueError("Quadrature order must be even and at least 4.")
        upper.wavenumbers(omega)
        lower.wavenumbers(omega)

        surface_z = (
            np.zeros_like(edges)
            if surface_depths is None
            else np.asarray(surface_depths, dtype=float)
        )
        if surface_z.shape != edges.shape or not np.all(np.isfinite(surface_z)):
            raise ValueError("Surface depths must be finite and match surface_edges.")
        if max(edges[0], self.interface_x[0]) >= min(edges[-1], self.interface_x[-1]):
            raise ValueError("The surface and interface must overlap horizontally.")
        self.surface = _Boundary.from_nodes(
            np.column_stack((edges, surface_z)), normal_side=-1
        )

        def find_center(x: float) -> int:
            matches = np.flatnonzero(
                np.isclose(self.surface.centers[:, 0], x, rtol=0, atol=1e-9)
            )
            if len(matches) != 1:
                raise ValueError(f"x={x} must be the center of exactly one surface panel.")
            return int(matches[0])

        self.receiver_indices = np.array([find_center(x) for x in receivers])
        self.source_index = find_center(source_x)
        self.upper, self.lower = upper, lower
        self.omega, self.force = float(omega), float(force)
        self.order = quadrature_order
        self._surface_green, self._surface_traction = _operators(
            self.surface.centers,
            self.surface.normals,
            self.surface,
            upper,
            omega,
            quadrature_order,
            self_interaction=True,
        )

    def predict(self, depths: ArrayLike) -> ComplexArray:
        """Return horizontal and vertical complex displacements at the receivers."""
        values = np.asarray(depths, dtype=float)
        if values.shape != self.control_x.shape or not np.all(np.isfinite(values)):
            raise ValueError("Depths must be finite and match control_x.")
        if np.any(values <= 0):
            raise ValueError("All interface depths must be positive.")
        overlap_start = max(self.surface.nodes[0, 0], self.interface_x[0])
        overlap_end = min(self.surface.nodes[-1, 0], self.interface_x[-1])
        interface_depths = np.interp(self.interface_x, self.control_x, values)
        check_x = np.unique(
            np.r_[
                overlap_start,
                overlap_end,
                self.surface.nodes[
                    (self.surface.nodes[:, 0] > overlap_start)
                    & (self.surface.nodes[:, 0] < overlap_end),
                    0,
                ],
                self.interface_x[
                    (self.interface_x > overlap_start) & (self.interface_x < overlap_end)
                ],
            ]
        )
        if np.any(
            np.interp(check_x, self.interface_x, interface_depths)
            <= np.interp(check_x, self.surface.nodes[:, 0], self.surface.nodes[:, 1])
        ):
            raise ValueError("The interface must remain below the free surface.")
        interface = _Boundary.from_nodes(
            np.column_stack((self.interface_x, interface_depths)),
            normal_side=1,
        )
        surface, upper, lower, omega, order = (
            self.surface,
            self.upper,
            self.lower,
            self.omega,
            self.order,
        )
        g_si, t_si = _operators(
            surface.centers, surface.normals, interface, upper, omega, order
        )
        g_is_upper, t_is_upper = _operators(
            interface.centers, interface.normals, surface, upper, omega, order
        )
        g_is_lower, t_is_lower = _operators(
            interface.centers, interface.normals, surface, lower, omega, order
        )
        g_ii_upper, t_ii_upper = _operators(
            interface.centers, interface.normals, interface, upper, omega, order, True
        )
        g_ii_lower, t_ii_lower = _operators(
            interface.centers, interface.normals, interface, lower, omega, order, True
        )

        ns, ni = len(surface.lengths), len(interface.lengths)
        identity_s, identity_i = np.eye(2 * ns), np.eye(2 * ni)
        displacement_scale = abs(upper.mu)
        matrix = np.block(
            [
                [self._surface_traction + 0.5 * identity_s, t_si, np.zeros_like(t_si)],
                [
                    displacement_scale * (g_is_upper - g_is_lower),
                    displacement_scale * g_ii_upper,
                    -displacement_scale * g_ii_lower,
                ],
                [
                    t_is_upper - t_is_lower,
                    t_ii_upper + 0.5 * identity_i,
                    -t_ii_lower + 0.5 * identity_i,
                ],
            ]
        )
        rhs = np.zeros(2 * ns + 4 * ni, dtype=complex)
        source_slice = slice(2 * self.source_index, 2 * self.source_index + 2)
        rhs[source_slice] = (
            -self.force
            * surface.normals[self.source_index]
            / surface.lengths[self.source_index]
        )
        density = solve(matrix, rhs)
        displacement = (
            self._surface_green @ density[: 2 * ns]
            + g_si @ density[2 * ns : 2 * (ns + ni)]
        ).reshape(ns, 2)
        return displacement[self.receiver_indices]


def arc_length_terms(x: ArrayLike, depths: ArrayLike) -> tuple[float, FloatArray, FloatArray]:
    """Excess length and its first two derivatives for a piecewise-linear curve."""
    x_values = np.asarray(x, dtype=float)
    z_values = np.asarray(depths, dtype=float)
    if (
        x_values.ndim != 1
        or len(x_values) < 2
        or z_values.shape != x_values.shape
        or not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(z_values))
        or np.any(np.diff(x_values) <= 0)
    ):
        raise ValueError("Curve coordinates must be finite and strictly ordered in x.")
    dx, dz = np.diff(x_values), np.diff(z_values)
    lengths = np.hypot(dx, dz)
    slopes = dz / lengths
    curvature = dx**2 / lengths**3
    gradient = np.zeros_like(z_values)
    gradient[:-1] -= slopes
    gradient[1:] += slopes
    hessian = np.zeros((len(z_values), len(z_values)))
    indices = np.arange(len(dx))
    hessian[indices, indices] += curvature
    hessian[indices + 1, indices + 1] += curvature
    hessian[indices, indices + 1] -= curvature
    hessian[indices + 1, indices] -= curvature
    return float(np.sum(lengths - dx)), gradient, hessian


@dataclass(frozen=True)
class InversionOptions:
    max_iterations: int = 25
    finite_difference_step: float = 0.05
    alpha_start: float = 0.001
    alpha_end: float = 0.000001
    damping: float = 0.00001
    max_step: float = 2.0
    min_depth: float = 2.0
    max_depth: float = 40.0
    tolerance: float = 0.01
    max_backtracks: int = 10

    def __post_init__(self) -> None:
        numbers = np.asarray(
            [
                self.finite_difference_step,
                self.alpha_start,
                self.alpha_end,
                self.damping,
                self.max_step,
                self.min_depth,
                self.max_depth,
                self.tolerance,
            ]
        )
        if not np.all(np.isfinite(numbers)):
            raise ValueError("Inversion options must be finite.")
        if self.max_iterations < 1 or self.max_backtracks < 1:
            raise ValueError("Iteration and backtracking counts must be positive.")
        if (
            self.finite_difference_step <= 0
            or self.alpha_start <= 0
            or self.alpha_end <= 0
            or self.alpha_start < self.alpha_end
            or self.damping <= 0
            or self.max_step <= 0
            or self.min_depth <= 0
            or self.tolerance <= 0
            or self.max_depth - self.min_depth <= 2 * self.finite_difference_step
        ):
            raise ValueError("Inversion bounds, weights, and step sizes are invalid.")


@dataclass(frozen=True)
class InversionStep:
    iteration: int
    depths: FloatArray
    relative_misfit: float
    objective: float
    excess_length: float
    alpha: float
    max_update: float


@dataclass(frozen=True)
class InversionResult:
    depths: FloatArray
    predicted: ComplexArray
    history: list[InversionStep]
    reason: str


def invert_interface(
    model: TwoLayerBEM,
    observed: ArrayLike,
    initial_depths: ArrayLike,
    options: InversionOptions,
    components: tuple[int, ...] = (0, 1),
) -> InversionResult:
    """Fit real interface depths to complex receiver displacements."""
    data = np.asarray(observed, dtype=complex)
    depths = np.asarray(initial_depths, dtype=float).copy()
    if data.shape != (len(model.receiver_indices), 2) or not np.all(np.isfinite(data)):
        raise ValueError("Observed data must have shape (number of receivers, 2).")
    if not components or len(set(components)) != len(components) or any(
        component not in (0, 1) for component in components
    ):
        raise ValueError("Components must be a nonempty selection of 0 and/or 1.")
    lower = options.min_depth + options.finite_difference_step
    upper = options.max_depth - options.finite_difference_step
    if (
        depths.shape != model.control_x.shape
        or not np.all(np.isfinite(depths))
        or np.any(depths < lower)
        or np.any(depths > upper)
    ):
        raise ValueError("Initial depths must lie inside the finite-difference bounds.")

    selected = data[:, components]
    data_scale = np.linalg.norm(selected)
    if data_scale == 0:
        raise ValueError("Selected observations must have a nonzero norm.")

    def residual(predicted: ComplexArray) -> FloatArray:
        difference = (predicted[:, components] - selected).ravel() / data_scale
        return np.concatenate((difference.real, difference.imag))

    predicted = model.predict(depths)
    error = residual(predicted)
    length, _, _ = arc_length_terms(model.control_x, depths)
    history = [
        InversionStep(
            0,
            depths.copy(),
            float(np.linalg.norm(error)),
            float(error @ error / 2 + options.alpha_start * length),
            length,
            options.alpha_start,
            0.0,
        )
    ]
    weights = np.geomspace(options.alpha_start, options.alpha_end, options.max_iterations)
    reason = "maximum iterations reached"
    for iteration, alpha in enumerate(weights, start=1):
        length, gradient, regularizer_hessian = arc_length_terms(model.control_x, depths)
        objective = float(error @ error / 2 + alpha * length)
        jacobian = np.empty((len(error), len(depths)))
        step_size = options.finite_difference_step
        for index in range(len(depths)):
            perturbation = np.zeros_like(depths)
            perturbation[index] = step_size
            jacobian[:, index] = (
                residual(model.predict(depths + perturbation))
                - residual(model.predict(depths - perturbation))
            ) / (2 * step_size)
        total_gradient = jacobian.T @ error + alpha * gradient
        hessian = (
            jacobian.T @ jacobian
            + alpha * regularizer_hessian
            + options.damping * np.eye(len(depths))
        )
        update = solve(hessian, -total_gradient, assume_a="pos")
        largest_update = np.max(np.abs(update))
        if largest_update > options.max_step:
            update *= options.max_step / largest_update

        accepted = False
        for backtrack in range(options.max_backtracks):
            candidate = np.clip(depths + update * 0.5**backtrack, lower, upper)
            actual_update = candidate - depths
            candidate_predicted = model.predict(candidate)
            candidate_error = residual(candidate_predicted)
            candidate_length, _, _ = arc_length_terms(model.control_x, candidate)
            candidate_objective = float(
                candidate_error @ candidate_error / 2 + alpha * candidate_length
            )
            if candidate_objective <= objective:
                depths, predicted, error = candidate, candidate_predicted, candidate_error
                accepted = True
                break
        if not accepted:
            reason = f"line search failed at iteration {iteration}"
            break
        max_update = float(np.max(np.abs(actual_update)))
        history.append(
            InversionStep(
                iteration,
                depths.copy(),
                float(np.linalg.norm(error)),
                candidate_objective,
                candidate_length,
                float(alpha),
                max_update,
            )
        )
        if max_update < options.tolerance and alpha <= options.alpha_end:
            reason = "update tolerance reached"
            break
    return InversionResult(depths.copy(), predicted.copy(), history, reason)


def perturb_measurements(
    observed: ArrayLike,
    level: float,
    kind: Literal["amplitude", "phase"],
    rng: np.random.Generator,
) -> ComplexArray:
    """Perturb amplitudes or phases using the paper's three-sigma convention."""
    data = np.asarray(observed, dtype=complex)
    if data.ndim != 2 or data.shape[1] != 2 or not np.all(np.isfinite(data)):
        raise ValueError("Observed data must have shape (number of receivers, 2).")
    if not np.isfinite(level) or level < 0:
        raise ValueError("Noise level must be finite and nonnegative.")
    if kind not in ("amplitude", "phase"):
        raise ValueError("Noise kind must be 'amplitude' or 'phase'.")
    if len(data) == 0:
        raise ValueError("At least one receiver is required.")
    amplitude = np.abs(data)
    phase = np.angle(data)
    if kind == "amplitude":
        sigma = level * np.max(amplitude, axis=0) / 3
        amplitude = amplitude + rng.normal(0, sigma, size=data.shape)
        if np.any(amplitude < 0):
            raise ValueError("Noise produced a negative amplitude; reduce the noise level.")
    else:
        sigma = level * np.max(np.abs(phase), axis=0) / 3
        phase = phase + rng.normal(0, sigma, size=data.shape)
    return amplitude * np.exp(1j * phase)
