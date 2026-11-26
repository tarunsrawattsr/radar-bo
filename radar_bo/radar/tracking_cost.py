"""Tracking-cost function — Lu et al., ICC 2024, Eqs. (14) and (15).

Paper-reported expressions (authoritative source: supplied PDF):

* Eq. (14): ``c_t(T_t, tau_t) = trace(E P_{t|t} E^T)`` — the paper interprets
  this as the sum of the posterior estimate variances of the target's current
  x-axis and y-axis position;
* Eq. (15): ``E = [[1, 0, 0, 0], [0, 1, 0, 0]]`` — the position selector.

Implemented as a standalone function, independent of the EKF, optimization,
scanning, weighting and power-allocation logic (M3 instruction 10). It
accepts any 4 x 4 posterior covariance (typically the posterior covariance of
a :class:`radar_bo.environment.target_state.GaussianState`), rejects
incompatible dimensions and non-finite results, and never modifies the input.
"""

from __future__ import annotations

import numpy as np

from radar_bo.environment.target_state import STATE_DIMENSION
from radar_bo.numerics import NumericalValidationError, validate_covariance

#: Eq. (15): position selector matrix E (2 x 4).
E_POSITION_SELECTOR = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def position_tracking_cost(posterior_covariance: np.ndarray) -> float:
    """Eq. (14): ``c_t = trace(E P_{t|t} E^T) = P[0, 0] + P[1, 1]``.

    Validates the covariance (shape, finiteness, symmetry, positive
    semidefiniteness) before extracting the position block; rejects a
    non-finite trace result.
    """
    covariance = validate_covariance(posterior_covariance, STATE_DIMENSION, "posterior_covariance")
    value = float(np.trace(E_POSITION_SELECTOR @ covariance @ E_POSITION_SELECTOR.T))
    if not np.isfinite(value):
        raise NumericalValidationError(f"Tracking cost is non-finite for covariance {covariance!r}.")
    return value
