"""Constant-velocity planar motion model — Lu et al., ICC 2024, Eqs. (1)-(3).

Paper-reported expressions (authoritative source: supplied PDF):

* Eq. (1): ``x_{t+1} = F_t x_t + w_t``;
* Eq. (2): ``F_t`` with the revisit interval ``T_t`` in the position-velocity
  coupling blocks;
* Eq. (3): ``Q_t = (structured matrix) * sigma_w^2`` with entries
  ``[[T^4/4, 0, T^3/2, 0], [0, T^4/4, 0, T^3/2], [T^3/2, 0, T^2, 0],
  [0, T^3/2, 0, T^2]]``.

The maneuverability-noise ambiguity is resolved through the configurable
interpretation switch on ``TrackingConfig`` (M2/M3 review):

* ``table_variance`` (default, NOT an author-confirmed correction): sigma_w
  = 25 is the acceleration-noise variance ((m/s)^2)^2, so Q is scaled by 25;
* ``equation_literal``: Eq. (3) taken literally, Q is scaled by 25^2 = 625.

Responsibilities are separated: transition-matrix construction, process-noise
covariance construction, deterministic propagation and stochastic propagation
with an explicitly supplied ``numpy.random.Generator`` (no global RNG state).

State ordering: [x, y, vx, vy] (project-wide).
"""

from __future__ import annotations

import numpy as np

from radar_bo.environment.target_state import TargetState
from radar_bo.numerics import NumericalValidationError

_MANEUVERABILITY_INTERPRETATIONS = ("table_variance", "equation_literal")


def _check_dt(dt: float) -> float:
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise NumericalValidationError(f"Revisit interval dt must be finite and > 0; got {dt!r}.")
    return value


class ConstantVelocityMotionModel:
    """Planar constant-velocity dynamics with acceleration process noise.

    ``process_noise_variance`` is the sigma_w^2 multiplier of Eq. (3) — i.e.
    25 under ``table_variance`` and 625 under ``equation_literal``. Use
    :meth:`from_config` to resolve the interpretation from the validated
    configuration; construct directly with an explicit variance when testing.
    """

    def __init__(self, process_noise_variance: float) -> None:
        value = float(process_noise_variance)
        if not np.isfinite(value) or value <= 0.0:
            raise NumericalValidationError(
                f"process_noise_variance (sigma_w^2 multiplier) must be finite and > 0; got {process_noise_variance!r}."
            )
        self.process_noise_variance = value

    @classmethod
    def from_config(cls, tracking_config) -> "ConstantVelocityMotionModel":
        """Resolve the configured maneuverability-noise interpretation.

        ``table_variance``: Q scaled by sigma_w = 25 (default; documented
        reading of Table II units and prose, not author-confirmed).
        ``equation_literal``: Q scaled by sigma_w^2 = 625 (Eq. (3) as printed).
        """
        interpretation = tracking_config.maneuverability_noise_interpretation
        if interpretation not in _MANEUVERABILITY_INTERPRETATIONS:
            raise ValueError(
                f"Unknown maneuverability_noise_interpretation {interpretation!r}; "
                f"expected one of {_MANEUVERABILITY_INTERPRETATIONS}."
            )
        sigma_w = tracking_config.maneuverability_noise
        variance = sigma_w if interpretation == "table_variance" else sigma_w**2
        return cls(process_noise_variance=variance)

    # ---- Matrix construction (Eqs. (2)-(3)) ----------------------------------
    def state_transition_matrix(self, dt: float) -> np.ndarray:
        """Eq. (2): position-velocity coupling with revisit interval ``dt``."""
        t = _check_dt(dt)
        f = np.eye(4, dtype=np.float64)
        f[0, 2] = t
        f[1, 3] = t
        return f

    def acceleration_input_matrix(self, dt: float) -> np.ndarray:
        """Low-rank factor G (4 x 2) with ``Q = G @ G.T * sigma_w^2``.

        Columns map per-axis white acceleration to state increments:
        position change ``T^2/2 * a`` and velocity change ``T * a`` per axis.
        Exposed for exact stochastic sampling and algebraic testing.
        """
        t = _check_dt(dt)
        g = np.zeros((4, 2), dtype=np.float64)
        g[0, 0] = 0.5 * t**2  # x  <- a_x
        g[1, 1] = 0.5 * t**2  # y  <- a_y
        g[2, 0] = t           # vx <- a_x
        g[3, 1] = t           # vy <- a_y
        return g

    def process_noise_covariance(self, dt: float) -> np.ndarray:
        """Eq. (3): structured process-noise covariance scaled by sigma_w^2."""
        t = _check_dt(dt)
        q = np.array(
            [
                [t**4 / 4.0, 0.0, t**3 / 2.0, 0.0],
                [0.0, t**4 / 4.0, 0.0, t**3 / 2.0],
                [t**3 / 2.0, 0.0, t**2, 0.0],
                [0.0, t**3 / 2.0, 0.0, t**2],
            ],
            dtype=np.float64,
        )
        return q * self.process_noise_variance

    # ---- Propagation ----------------------------------------------------------
    def propagate_deterministic(self, state: TargetState, dt: float) -> TargetState:
        """Eq. (1) without noise: ``x_{t+1} = F_t x_t`` (target metadata kept)."""
        if not isinstance(state, TargetState):
            raise TypeError(f"propagate_deterministic expects TargetState; got {type(state)!r}.")
        f = self.state_transition_matrix(dt)
        return TargetState(vector=f @ state.vector, target_id=state.target_id, is_active=state.is_active)

    def propagate_stochastic(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        """Eq. (1) with process noise: ``x_{t+1} = F_t x_t + G a``, ``a ~ N(0, sigma_w^2 I_2)``.

        Sampling per-axis constant accelerations through the low-rank factor G
        is exactly equivalent to sampling ``w ~ N(0, Q_t)`` (Q_t is singular
        with rank 2, so Cholesky sampling of Q_t is impossible without
        arbitrary rank choices; this construction is the unique physical one).
        Requires an explicit ``numpy.random.Generator`` — no global RNG state.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "propagate_stochastic requires an explicit numpy.random.Generator "
                f"(no global random state); got {type(rng)!r}."
            )
        f = self.state_transition_matrix(dt)
        g = self.acceleration_input_matrix(dt)
        acceleration = rng.normal(loc=0.0, scale=float(np.sqrt(self.process_noise_variance)), size=2)
        new_vector = f @ state.vector + g @ acceleration
        return TargetState(vector=new_vector, target_id=state.target_id, is_active=state.is_active)
