"""Tests for estimator initialization policies (M3 instruction 9)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.target_state import Observation
from radar_bo.radar.initialization import EstimatorInitializationError, initialize_estimator
from radar_bo.radar.measurement_model import MeasurementGeometryError, PlanarRangeAzimuthModel

R = np.diag([1.0, 4.0e-6])


def make_observation(range_m: float = 3000.0, azimuth: float = 0.6) -> Observation:
    return Observation(vector=[range_m, azimuth], covariance=R.copy())


def measurement_driven_config() -> TrackingConfig:
    """Synthetic-labeled velocity prior assumptions (NOT paper values)."""
    return TrackingConfig(
        initialization_policy="measurement_driven",
        measurement_driven_initial_velocity=(0.0, 0.0),
        measurement_driven_initial_velocity_variance=25.0,
    )


class TestPaperZeroIdentity:
    def test_reproduces_paper_initialization(self):
        """Diagnostic mode returns exactly x0 = 0, P0 = I_4 (paper p. 2786)."""
        config = TrackingConfig()  # default policy
        estimate = initialize_estimator(config.initialization_policy, config, observation=None)
        np.testing.assert_array_equal(estimate.mean, np.zeros(4))
        np.testing.assert_array_equal(estimate.covariance, np.eye(4))
        assert "diagnostic" in estimate.label

    def test_update_from_origin_fails_loudly(self):
        """Zero-vector init must NOT silently produce NaN/inf or an artificial Jacobian."""
        from radar_bo.radar.extended_kalman_filter import ExtendedKalmanFilter
        from radar_bo.environment.motion_model import ConstantVelocityMotionModel

        ekf = ExtendedKalmanFilter(
            motion_model=ConstantVelocityMotionModel.from_config(TrackingConfig()),
            measurement_model=PlanarRangeAzimuthModel(),
        )
        estimate = initialize_estimator("paper_zero_identity", TrackingConfig(), None)
        prior = ekf.predict(estimate, 2.5, ekf.motion_model.process_noise_covariance(2.5))
        with pytest.raises(MeasurementGeometryError):
            ekf.update(prior, make_observation())


class TestMeasurementDriven:
    def test_position_from_observation_only(self):
        """Position is the polar-to-Cartesian transform of the observation itself."""
        config = measurement_driven_config()
        obs = make_observation(range_m=3000.0, azimuth=0.6)
        estimate = initialize_estimator("measurement_driven", config, obs)
        expected_position = np.array([3000.0 * np.cos(0.6), 3000.0 * np.sin(0.6)])
        np.testing.assert_allclose(estimate.mean[:2], expected_position, rtol=1e-14)

    def test_position_covariance_from_observation_covariance(self):
        """P_position = J_p R J_p^T with J_p the polar-to-Cartesian Jacobian."""
        config = measurement_driven_config()
        r, az = 3000.0, 0.6
        obs = make_observation(r, az)
        estimate = initialize_estimator("measurement_driven", config, obs)
        j_polar = np.array(
            [[np.cos(az), -r * np.sin(az)], [np.sin(az), r * np.cos(az)]]
        )
        expected_pos_cov = j_polar @ R @ j_polar.T
        np.testing.assert_allclose(estimate.covariance[:2, :2], expected_pos_cov, rtol=1e-12)

    def test_velocity_from_configured_prior_only(self):
        """Velocity mean/variance come from the configured assumptions, cross-terms zero."""
        config = TrackingConfig(
            initialization_policy="measurement_driven",
            measurement_driven_initial_velocity=(4.0, -2.0),
            measurement_driven_initial_velocity_variance=9.0,
        )
        estimate = initialize_estimator("measurement_driven", config, make_observation())
        np.testing.assert_array_equal(estimate.mean[2:], [4.0, -2.0])
        np.testing.assert_array_equal(estimate.covariance[2:, 2:], 9.0 * np.eye(2))
        np.testing.assert_array_equal(estimate.covariance[:2, 2:], np.zeros((2, 2)))
        np.testing.assert_array_equal(estimate.covariance[2:, :2], np.zeros((2, 2)))

    def test_position_and_velocity_uncertainty_separate(self):
        """Velocity uncertainty comes from the configured variance, NOT the observation."""
        config = measurement_driven_config()  # velocity variance 25
        estimate = initialize_estimator("measurement_driven", config, make_observation())
        assert estimate.covariance[2, 2] == 25.0
        assert estimate.covariance[0, 0] != 25.0  # position block derived from R

    def test_unresolved_velocity_assumptions_raise_informatively(self):
        """No invented defaults: unresolved assumptions raise a parameter error."""
        config = TrackingConfig(initialization_policy="measurement_driven")  # both None
        with pytest.raises(UnresolvedParameterError) as excinfo:
            initialize_estimator("measurement_driven", config, make_observation())
        message = str(excinfo.value)
        assert "measurement_driven_initial_velocity" in message

    def test_partially_unresolved_assumptions_raise(self):
        config = TrackingConfig(
            initialization_policy="measurement_driven",
            measurement_driven_initial_velocity=(0.0, 0.0),
            # variance left unresolved
        )
        with pytest.raises(UnresolvedParameterError):
            initialize_estimator("measurement_driven", config, make_observation())

    def test_missing_observation_raises(self):
        config = measurement_driven_config()
        with pytest.raises(EstimatorInitializationError, match="observation"):
            initialize_estimator("measurement_driven", config, observation=None)


class TestExperimentalCustom:
    def test_supplied_values_pass_through(self):
        config = TrackingConfig(
            initialization_policy="experimental_custom",
            initial_state=(500.0, -800.0, 12.0, 3.0),
            initial_covariance=(100.0 * np.eye(4)).tolist(),
        )
        estimate = initialize_estimator("experimental_custom", config, observation=None)
        np.testing.assert_array_equal(estimate.mean, [500.0, -800.0, 12.0, 3.0])
        np.testing.assert_array_equal(estimate.covariance, 100.0 * np.eye(4))
        assert estimate.label == "experimental_custom"


class TestNoGroundTruthAccess:
    def test_initializer_signature_has_no_truth_input(self):
        """Structural guarantee: initialization cannot receive the true state."""
        params = set(inspect.signature(initialize_estimator).parameters)
        assert params == {"initialization_policy", "tracking_config", "observation"}
        assert not any("true" in p or "truth" in p or "ground" in p for p in params)

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="Unknown initialization_policy"):
            initialize_estimator("teleportation", TrackingConfig(), None)
