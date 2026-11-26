"""Estimator initialization policies (M3 instruction 9).

Policies configured on ``TrackingConfig.initialization_policy``:

* ``paper_zero_identity`` — the paper's initialization (p. 2786): posterior
  state initialized as a zero vector and covariance as the identity matrix,
  "assuming that the system has no prior knowledge of the targets".
  RETAINED AS A DOCUMENTED DIAGNOSTIC MODE: at the origin the range/azimuth
  geometry is singular, so an EKF update from this initialization raises the
  identifiable :class:`radar_bo.radar.measurement_model.MeasurementGeometryError`
  instead of silently producing NaN/inf or an artificial Jacobian.
* ``measurement_driven`` — OPERATIONAL policy (M3 review, instruction 3.3):
  initialize from the first supplied sensor observation only. Position comes
  from the polar observation; the position uncertainty is the observation
  covariance transformed to Cartesian axes (first-order polar-to-Cartesian
  covariance propagation). Velocity uses ONLY explicitly configured prior
  assumptions (:attr:`TrackingConfig.measurement_driven_initial_velocity`
  and ``..._variance``); when they are unresolved (``None``) an informative
  error is raised — nothing is invented. The simulated true state is never
  accessible here (structurally: this function's signature has no truth
  input, guarded by tests).
* ``experimental_custom`` — use the configured ``initial_state`` and
  ``initial_covariance`` exactly as supplied (experimental; values must be
  justified, never attributed to the paper).

Position and velocity uncertainty are kept conceptually separate (separate
covariance blocks, no cross-terms unless derived from the observation).
"""

from __future__ import annotations

import numpy as np

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.environment.target_state import GaussianState, Observation, STATE_DIMENSION


class EstimatorInitializationError(RuntimeError):
    """Raised when estimator initialization cannot proceed from supplied inputs."""

#: Unit strings used in unresolved-parameter errors.
_VELOCITY_UNITS = {
    "tracking.measurement_driven_initial_velocity": "(m/s, m/s) tuple",
    "tracking.measurement_driven_initial_velocity_variance": "(m/s)^2",
}


def _polar_to_cartesian_position(r: float, azimuth: float) -> np.ndarray:
    """Position from the observation: ``[r cos(azimuth), r sin(azimuth)]``."""
    return np.array([r * np.cos(azimuth), r * np.sin(azimuth)], dtype=np.float64)


def _polar_to_cartesian_jacobian(r: float, azimuth: float) -> np.ndarray:
    """Jacobian of (x, y) with respect to (r, azimuth): first-order covariance transform."""
    cos_a = np.cos(azimuth)
    sin_a = np.sin(azimuth)
    return np.array(
        [
            [cos_a, -r * sin_a],
            [sin_a, r * cos_a],
        ],
        dtype=np.float64,
    )


def initialize_estimator(
    initialization_policy: str,
    tracking_config,
    observation: Observation | None,
) -> GaussianState:
    """Create the initial posterior estimate according to the configured policy.

    ``observation`` is required by (and only used by) the
    ``measurement_driven`` policy. No true simulated state is an input.
    """
    if initialization_policy == "paper_zero_identity":
        # Diagnostic mode: config validators guarantee these equal the paper
        # defaults (zero vector, I_4). Values come from the configuration.
        return GaussianState(
            mean=np.asarray(tracking_config.initial_state, dtype=np.float64),
            covariance=np.asarray(tracking_config.initial_covariance, dtype=np.float64),
            label="paper_zero_identity_diagnostic",
        )

    if initialization_policy == "measurement_driven":
        if observation is None:
            raise EstimatorInitializationError(
                "initialization_policy='measurement_driven' requires the first sensor "
                "observation; none was supplied."
            )
        r = observation.range
        azimuth = observation.azimuth
        position = _polar_to_cartesian_position(r, azimuth)
        # First-order polar-to-Cartesian covariance transform of the observation R:
        # position uncertainty derives from the measurement itself, not from
        # invented numbers.
        j_polar = _polar_to_cartesian_jacobian(r, azimuth)
        position_covariance = j_polar @ observation.covariance @ j_polar.T
        velocity_mean = tracking_config.measurement_driven_initial_velocity
        velocity_variance = tracking_config.measurement_driven_initial_velocity_variance
        if velocity_mean is None or velocity_variance is None:
            raise UnresolvedParameterError(
                {name: unit for name, unit, value in (
                    ("tracking.measurement_driven_initial_velocity", _VELOCITY_UNITS["tracking.measurement_driven_initial_velocity"], velocity_mean),
                    ("tracking.measurement_driven_initial_velocity_variance", _VELOCITY_UNITS["tracking.measurement_driven_initial_velocity_variance"], velocity_variance),
                ) if value is None},
                guidance=(
                    "Measurement-driven initialization needs explicitly configured velocity "
                    "prior assumptions (proposed assumptions, NOT paper values). Set "
                    "measurement_driven_initial_velocity and "
                    "measurement_driven_initial_velocity_variance in TrackingConfig; "
                    "synthetic experiments must label them as synthetic settings."
                ),
            )
        velocity_mean = np.asarray(velocity_mean, dtype=np.float64)
        if velocity_mean.shape != (2,) or not np.all(np.isfinite(velocity_mean)):
            raise ValueError(
                f"measurement_driven_initial_velocity must be a finite (m/s, m/s) pair; got {velocity_mean!r}."
            )
        if not np.isfinite(velocity_variance) or velocity_variance <= 0.0:
            raise ValueError(
                f"measurement_driven_initial_velocity_variance must be finite and > 0; got {velocity_variance!r}."
            )
        mean = np.zeros(STATE_DIMENSION, dtype=np.float64)
        mean[:2] = position
        mean[2:] = velocity_mean
        covariance = np.zeros((STATE_DIMENSION, STATE_DIMENSION), dtype=np.float64)
        covariance[:2, :2] = position_covariance
        covariance[2:, 2:] = float(velocity_variance) * np.eye(2)
        return GaussianState(mean=mean, covariance=covariance, label="measurement_driven")

    if initialization_policy == "experimental_custom":
        return GaussianState(
            mean=np.asarray(tracking_config.initial_state, dtype=np.float64),
            covariance=np.asarray(tracking_config.initial_covariance, dtype=np.float64),
            label="experimental_custom",
        )

    raise ValueError(
        f"Unknown initialization_policy {initialization_policy!r}; expected "
        "'paper_zero_identity', 'measurement_driven' or 'experimental_custom'."
    )
