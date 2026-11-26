"""Planar range/azimuth measurement model — Lu et al., ICC 2024, Eqs. (4) and (7).

Paper-reported expressions (authoritative source: supplied PDF):

* Eq. (4): ``z_t = h(x_t) + v_t`` with ``h(x) = [sqrt(x^2 + y^2), atan(y/x)]^T``;
* Eq. (7): ``H_t`` is the 2 x 4 Jacobian of ``h`` with rows
  ``[x/r, y/r, 0, 0]`` and ``[-y/r^2, x/r^2, 0, 0]``, ``r = sqrt(x^2 + y^2)``.

Documented implementation conventions (M3 instruction 6):

* The printed azimuth is the fraction ``tan^-1(y_t / x_t)``, which has a 180
  degree ambiguity. Because the paper's radar scans the full 360 degree
  circle, the azimuth is evaluated with the quadrant-aware ``atan2(y, x)``,
  returning values in ``[-pi, pi)``. The analytical Jacobian is identical for
  both readings (d/dx atan(y/x) = -y/r^2 either way).
* The radar sits at the origin of the Cartesian frame (paper assumption,
  p. 2786-2787): range/azimuth geometry is singular exactly at the origin.
  No epsilon substitution is applied; a domain-specific
  :class:`MeasurementGeometryError` is raised instead.

The analytical Jacobian is validated against numerical differentiation in
``radar_bo/tests/test_measurement_model.py``.
"""

from __future__ import annotations

import numpy as np

from radar_bo.environment.target_state import MEASUREMENT_DIMENSION, STATE_DIMENSION, TargetState
from radar_bo.numerics import NumericalValidationError, as_vector


class MeasurementGeometryError(RuntimeError):
    """Singular or degenerate measurement geometry (e.g. target at radar origin).

    Raised instead of substituting epsilon denominators or producing NaN/inf
    (M3 instructions 3.3 and 6): the caller must see an identifiable,
    domain-specific failure.
    """


class PlanarRangeAzimuthModel:
    """Ideal (noise-free) measurement function and its Jacobian."""

    state_dimension = STATE_DIMENSION
    measurement_dimension = MEASUREMENT_DIMENSION

    @staticmethod
    def _check_geometry(position: np.ndarray, radius: float) -> None:
        if not np.isfinite(radius):
            raise MeasurementGeometryError(f"Non-finite range {radius!r} at position {position!r}.")
        if radius == 0.0:
            raise MeasurementGeometryError(
                "Singular measurement geometry: target at the radar origin. Range is 0 and the "
                "azimuth atan2(0, 0) is undefined; the measurement Jacobian is singular. "
                "No epsilon substitution is applied by project policy — initialize the estimator "
                "away from the origin (e.g. measurement-driven initialization)."
            )

    def measure(self, position: np.ndarray) -> np.ndarray:
        """Eq. (4) without noise: ``[range, azimuth]`` from a position (2,) vector."""
        p = as_vector(position, 2, "position")
        radius = float(np.hypot(p[0], p[1]))
        self._check_geometry(p, radius)
        azimuth = float(np.arctan2(p[1], p[0]))  # quadrant-aware; see module docstring
        return np.array([radius, azimuth], dtype=np.float64)

    def measure_from_state(self, state: TargetState) -> np.ndarray:
        """Ideal observation of a true :class:`TargetState` (velocity not observed)."""
        if not isinstance(state, TargetState):
            raise TypeError(f"measure_from_state expects TargetState; got {type(state)!r}.")
        return self.measure(state.position)

    def jacobian(self, position: np.ndarray) -> np.ndarray:
        """Eq. (7): 2 x 4 Jacobian ``H`` of ``h`` evaluated at the position.

        Geometry is validated BEFORE differentiation; the origin raises
        :class:`MeasurementGeometryError`. Velocity columns are structurally
        zero (position-only measurement).
        """
        p = as_vector(position, 2, "position")
        x, y = float(p[0]), float(p[1])
        radius = float(np.hypot(x, y))
        self._check_geometry(p, radius)
        h = np.zeros((2, 4), dtype=np.float64)
        h[0, 0] = x / radius
        h[0, 1] = y / radius
        h[1, 0] = -y / radius**2
        h[1, 1] = x / radius**2
        return h

    def numerical_jacobian(self, position: np.ndarray, step: float = 1e-6) -> np.ndarray:
        """Independent central-difference Jacobian (test/verification aid only).

        Uses symmetric finite differences of :meth:`measure`; not used inside
        the estimator.
        """
        p = as_vector(position, 2, "position")
        jacobian = np.zeros((2, 4), dtype=np.float64)
        for k in range(4):
            if k < 2:
                delta = np.zeros(2, dtype=np.float64)
                delta[k] = step
                plus = self.measure(p + delta)
                minus = self.measure(p - delta)
                jacobian[:, k] = (plus - minus) / (2.0 * step)
            else:
                jacobian[:, k] = 0.0  # velocity does not enter the measurement
        return jacobian
