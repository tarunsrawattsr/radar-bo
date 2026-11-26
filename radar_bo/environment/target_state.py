"""Planar state representations for the tracking pipeline (README section 5.1).

Distinct representations are maintained for:

* true simulated state (:class:`TargetState`) — accessible only to the
  simulation and evaluation layers, never passed to the estimator;
* prior estimate and posterior estimate (:class:`GaussianState`, separate
  instances with explicit labels);
* prior / posterior covariances (the covariance attribute of each
  :class:`GaussianState`);
* sensor observation and its measurement covariance (:class:`Observation`
  bundling a validated :class:`MeasurementCovariance`).

State ordering is project-wide (M3 instruction 4)::

    STATE_ORDER = ("x_position", "y_position", "x_velocity", "y_velocity")

Units: metres and metres-per-second; azimuths in radians wrapped to [-pi, pi).
All arrays are float64 and validated at construction (interface boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from radar_bo.numerics import as_vector, validate_covariance, wrap_to_pi

STATE_DIMENSION: Final[int] = 4
MEASUREMENT_DIMENSION: Final[int] = 2
#: Project-wide state ordering: x, y, vx, vy.
STATE_ORDER: Final[tuple[str, ...]] = ("x_position", "y_position", "x_velocity", "y_velocity")
POSITION_SLICE: Final[slice] = slice(0, 2)
VELOCITY_SLICE: Final[slice] = slice(2, 4)


@dataclass(frozen=True)
class TargetState:
    """TRUE simulated planar state ``[x, y, vx, vy]`` (m, m/s).

    Identifiers and active status are stored separately from the numerical
    vector (README section 5.1). This object belongs to the simulation and
    evaluation layers; the estimator never receives it.
    """

    vector: np.ndarray
    target_id: str = "target_0"
    is_active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "vector", as_vector(self.vector, STATE_DIMENSION, f"TargetState[{self.target_id}].vector")
        )

    @property
    def position(self) -> np.ndarray:
        """Position sub-vector [x, y] in metres (copy)."""
        return self.vector[POSITION_SLICE].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Velocity sub-vector [vx, vy] in m/s (copy)."""
        return self.vector[VELOCITY_SLICE].copy()

    @property
    def x(self) -> float:
        return float(self.vector[0])

    @property
    def y(self) -> float:
        return float(self.vector[1])


@dataclass(frozen=True)
class GaussianState:
    """Gaussian estimate: mean vector (4,) and covariance matrix (4, 4).

    Used for BOTH prior (predicted) and posterior (updated) estimates as
    distinct instances; ``label`` records which role the instance plays.
    Covariance is validated symmetric positive-semidefinite at construction;
    violations raise rather than being repaired.
    """

    mean: np.ndarray
    covariance: np.ndarray
    label: str = "estimate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", as_vector(self.mean, STATE_DIMENSION, f"GaussianState[{self.label}].mean"))
        object.__setattr__(
            self,
            "covariance",
            validate_covariance(self.covariance, STATE_DIMENSION, f"GaussianState[{self.label}].covariance"),
        )

    @property
    def position_mean(self) -> np.ndarray:
        return self.mean[POSITION_SLICE].copy()

    @property
    def velocity_mean(self) -> np.ndarray:
        return self.mean[VELOCITY_SLICE].copy()


@dataclass(frozen=True)
class MeasurementCovariance:
    """Measurement-noise covariance R (2 x 2) for [range, azimuth].

    Kept as its own representation (distinct from the 4 x 4 state covariance)
    per M3 instruction 7. Validated symmetric positive-semidefinite.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            validate_covariance(self.matrix, MEASUREMENT_DIMENSION, "MeasurementCovariance.matrix"),
        )


@dataclass(frozen=True)
class Observation:
    """Sensor observation ``z = [range, azimuth]`` plus measurement covariance R.

    The azimuth is normalized to [-pi, pi) on construction (documented
    convention; quadrant-aware representation of the printed tan^-1(y/x)).
    Range must be non-negative and finite. ``step_index`` is optional
    bookkeeping linking the observation to a simulation step.
    """

    vector: np.ndarray
    covariance: np.ndarray
    step_index: int | None = None

    def __post_init__(self) -> None:
        v = as_vector(self.vector, MEASUREMENT_DIMENSION, "Observation.vector")
        if v[0] < 0.0:
            raise ValueError(f"Observation range must be non-negative; got {v[0]}.")
        v = np.array([v[0], wrap_to_pi(float(v[1]))], dtype=np.float64)
        object.__setattr__(self, "vector", v)
        object.__setattr__(self, "covariance", MeasurementCovariance(self.covariance).matrix)

    @property
    def range(self) -> float:
        return float(self.vector[0])

    @property
    def azimuth(self) -> float:
        return float(self.vector[1])
