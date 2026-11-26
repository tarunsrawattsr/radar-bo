"""Extended Kalman Filter — Lu et al., ICC 2024, Eqs. (9)-(13).

Paper-reported expressions (authoritative source: supplied PDF, p. 2787):

* Eq. (9):  ``x_{t|t-1} = F_t x_{t-1|t-1}``                          (predict mean)
* Eq. (10): ``P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q_t``              (predict covariance)
* Eq. (11): ``K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R_t)^{-1}`` (Kalman gain)
* Eq. (12): ``x_{t|t} = x_{t|t-1} + K_t (z_t - h(x_{t|t-1}))``        (update mean)
* Eq. (13): ``P_{t|t} = (I - K_t H_t) P_{t|t-1}``                     (update covariance)

Numerical implementation decisions (M3 instruction 8), all documented:

* The Kalman gain is computed with ``numpy.linalg.solve`` on the innovation
  covariance instead of forming an explicit matrix inverse.
* Angular residuals: the azimuth innovation component is wrapped to
  ``[-pi, pi)`` before use in Eq. (12) (quadrant-aware angular convention;
  see ``radar_bo.numerics.wrap_to_pi``).
* Posterior covariances are symmetrized — ``(P + P^T)/2`` — to counter
  floating-point drift. This is NOT eigenvalue clipping; positive-
  semidefiniteness is validated afterwards and violations raise
  :class:`radar_bo.numerics.NumericalValidationError`.
* Optional ``covariance_update_method='joseph'`` (numerical-stability
  improvement): ``P = (I-KH) P (I-KH)^T + K R K^T``, which is mathematically
  equivalent to Eq. (13) for the exact gain and form-preserving positive
  semidefinite. The default ``'paper_equation_13'`` reproduces Eq. (13)
  literally. Equivalence under valid numerical conditions is tested.

The EKF receives ALL numerical inputs through explicit interfaces (motion
model, measurement model, observation, process covariance). It has no
dependency on BoTorch, GPyTorch, Bayesian optimization, resource allocation,
scanning performance or corridor geometry (layer separation, guarded by
tests). Prediction and update are separate operations; initialization is
handled by ``radar_bo.radar.initialization`` outside the predict/update cycle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import POSITION_SLICE, GaussianState, Observation
from radar_bo.numerics import NumericalValidationError, symmetrize, validate_covariance, wrap_to_pi
from radar_bo.radar.measurement_model import PlanarRangeAzimuthModel

CovarianceUpdateMethod = Literal["paper_equation_13", "joseph"]

_COVARIANCE_UPDATE_METHODS = ("paper_equation_13", "joseph")


class ExtendedKalmanFilter:
    """EKF over the planar 4-D state with range/azimuth measurements.

    ``covariance_update_method``: ``'paper_equation_13'`` (default) applies
    Eq. (13) as printed; ``'joseph'`` is the documented numerical-stability
    variant, mathematically equivalent for the exact gain (tested).
    """

    def __init__(
        self,
        motion_model: ConstantVelocityMotionModel,
        measurement_model: PlanarRangeAzimuthModel,
        covariance_update_method: CovarianceUpdateMethod = "paper_equation_13",
    ) -> None:
        if not isinstance(motion_model, ConstantVelocityMotionModel):
            raise TypeError(f"motion_model must be a ConstantVelocityMotionModel; got {type(motion_model)!r}.")
        if not isinstance(measurement_model, PlanarRangeAzimuthModel):
            raise TypeError(f"measurement_model must be a PlanarRangeAzimuthModel; got {type(measurement_model)!r}.")
        if covariance_update_method not in _COVARIANCE_UPDATE_METHODS:
            raise ValueError(
                f"covariance_update_method must be one of {_COVARIANCE_UPDATE_METHODS}; "
                f"got {covariance_update_method!r}."
            )
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.covariance_update_method = covariance_update_method

    # ---- Prediction (Eqs. (9)-(10)) ---------------------------------------------
    def predict(
        self,
        posterior_estimate: GaussianState,
        revisit_interval: float,
        process_covariance: np.ndarray,
    ) -> GaussianState:
        """Advance a posterior estimate to the prior estimate at the next revisit."""
        f = self.motion_model.state_transition_matrix(revisit_interval)
        q = validate_covariance(process_covariance, 4, "process_covariance")
        prior_mean = f @ posterior_estimate.mean
        prior_covariance = symmetrize(f @ posterior_estimate.covariance @ f.T + q)
        return GaussianState(mean=prior_mean, covariance=prior_covariance, label="prior")

    # ---- Update (Eqs. (11)-(13)) --------------------------------------------------
    def update(self, prior_estimate: GaussianState, observation: Observation) -> GaussianState:
        """Fuse one observation into the prior estimate, producing the posterior.

        Geometry is validated when the Jacobian is evaluated at the prior
        mean: a prior at the radar origin raises
        :class:`radar_bo.radar.measurement_model.MeasurementGeometryError`.
        """
        if not isinstance(prior_estimate, GaussianState):
            raise TypeError(f"prior_estimate must be a GaussianState; got {type(prior_estimate)!r}.")
        if not isinstance(observation, Observation):
            raise TypeError(f"observation must be an Observation; got {type(observation)!r}.")

        prior_position = prior_estimate.mean[POSITION_SLICE]
        h_prior = self.measurement_model.measure(prior_position)
        h_jacobian = self.measurement_model.jacobian(prior_position)

        innovation = observation.vector - h_prior
        # Angular residual convention: azimuth component wrapped to [-pi, pi).
        innovation[1] = wrap_to_pi(float(innovation[1]))

        r = observation.covariance
        innovation_covariance = symmetrize(h_jacobian @ prior_estimate.covariance @ h_jacobian.T + r)
        s_eigenvalues = np.linalg.eigvalsh(innovation_covariance)
        if not np.all(np.isfinite(s_eigenvalues)) or float(s_eigenvalues.min()) <= 0.0:
            raise NumericalValidationError(
                "Innovation covariance must be positive definite for a valid update; "
                f"eigenvalues {s_eigenvalues!r}. Inputs are rejected rather than repaired."
            )

        # Eq. (11) via a linear solve (P symmetric): K^T = S^{-1} (H P).
        gain = np.linalg.solve(innovation_covariance, h_jacobian @ prior_estimate.covariance).T

        posterior_mean = prior_estimate.mean + gain @ innovation  # Eq. (12)

        if self.covariance_update_method == "paper_equation_13":
            posterior_covariance = (np.eye(4, dtype=np.float64) - gain @ h_jacobian) @ prior_estimate.covariance
        else:  # 'joseph': documented numerical-stability variant, equivalent for exact gain
            i_kh = np.eye(4, dtype=np.float64) - gain @ h_jacobian
            posterior_covariance = i_kh @ prior_estimate.covariance @ i_kh.T + gain @ r @ gain.T

        # Symmetrization counters floating-point drift only; PSD is validated
        # on GaussianState construction and violations raise (no clipping).
        posterior_covariance = symmetrize(posterior_covariance)
        return GaussianState(mean=posterior_mean, covariance=posterior_covariance, label="posterior")
