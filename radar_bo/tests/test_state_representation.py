"""Tests for the planar state representations (M3 instruction 4)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.environment.target_state import (
    MEASUREMENT_DIMENSION,
    POSITION_SLICE,
    STATE_DIMENSION,
    STATE_ORDER,
    VELOCITY_SLICE,
    GaussianState,
    MeasurementCovariance,
    Observation,
    TargetState,
)
from radar_bo.numerics import NumericalValidationError


class TestStateOrdering:
    def test_project_wide_ordering(self):
        """State ordering is fixed: x, y, vx, vy (instruction 4)."""
        assert STATE_ORDER == ("x_position", "y_position", "x_velocity", "y_velocity")
        assert POSITION_SLICE == slice(0, 2)
        assert VELOCITY_SLICE == slice(2, 4)
        assert STATE_DIMENSION == 4
        assert MEASUREMENT_DIMENSION == 2

    def test_position_velocity_properties_follow_ordering(self):
        s = TargetState(vector=[100.0, 200.0, 3.0, -4.0])
        np.testing.assert_array_equal(s.position, [100.0, 200.0])
        np.testing.assert_array_equal(s.velocity, [3.0, -4.0])
        assert (s.x, s.y) == (100.0, 200.0)

    def test_metadata_separate_from_numerics(self):
        """Identifiers/active status stored separately from the vector (README 5.1)."""
        s = TargetState(vector=[1.0, 2.0, 3.0, 4.0], target_id="t7", is_active=False)
        assert s.target_id == "t7"
        assert s.is_active is False
        assert s.vector.shape == (4,)


class TestTargetStateValidation:
    def test_float64_coercion(self):
        s = TargetState(vector=np.array([1, 2, 3, 4], dtype=np.int32))
        assert s.vector.dtype == np.float64

    def test_wrong_shape_rejected(self):
        with pytest.raises(NumericalValidationError, match="length 4"):
            TargetState(vector=[1.0, 2.0, 3.0])

    def test_non_finite_rejected(self):
        with pytest.raises(NumericalValidationError, match="non-finite"):
            TargetState(vector=[np.nan, 0.0, 0.0, 0.0])
        with pytest.raises(NumericalValidationError, match="non-finite"):
            TargetState(vector=[np.inf, 0.0, 0.0, 0.0])


class TestGaussianStateValidation:
    def test_prior_and_posterior_are_distinct_instances(self):
        prior = GaussianState(mean=np.zeros(4), covariance=np.eye(4), label="prior")
        posterior = GaussianState(mean=np.ones(4), covariance=2.0 * np.eye(4), label="posterior")
        assert prior.label == "prior" and posterior.label == "posterior"
        assert not np.allclose(prior.mean, posterior.mean)

    def test_non_psd_covariance_rejected(self):
        with pytest.raises(NumericalValidationError, match="positive semidefinite"):
            GaussianState(mean=np.zeros(4), covariance=np.diag([1.0, -2.0, 1.0, 1.0]))

    def test_asymmetric_covariance_rejected(self):
        bad = np.eye(4)
        bad[0, 1] = 0.3
        with pytest.raises(NumericalValidationError, match="symmetric"):
            GaussianState(mean=np.zeros(4), covariance=bad)

    def test_wrong_dimensions_rejected(self):
        with pytest.raises(NumericalValidationError):
            GaussianState(mean=np.zeros(3), covariance=np.eye(3))


class TestObservationValidation:
    def test_valid_observation(self):
        obs = Observation(vector=[3000.0, 0.5], covariance=np.diag([1.0, 4e-6]), step_index=3)
        assert obs.range == 3000.0
        assert obs.azimuth == 0.5
        assert obs.step_index == 3
        assert obs.covariance.shape == (2, 2)

    def test_azimuth_wrapped_to_half_open_interval(self):
        obs = Observation(vector=[3000.0, 3.0 * np.pi], covariance=np.eye(2))
        assert -np.pi <= obs.azimuth < np.pi
        obs2 = Observation(vector=[3000.0, -4.0 * np.pi], covariance=np.eye(2))
        assert -np.pi <= obs2.azimuth < np.pi

    def test_negative_range_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Observation(vector=[-1.0, 0.0], covariance=np.eye(2))

    def test_non_psd_measurement_covariance_rejected(self):
        with pytest.raises(NumericalValidationError):
            Observation(vector=[100.0, 0.0], covariance=np.diag([1.0, -1e-6]))

    def test_measurement_covariance_standalone(self):
        mc = MeasurementCovariance(np.diag([4.0, 1e-6]))
        assert mc.matrix[0, 0] == 4.0
        with pytest.raises(NumericalValidationError):
            MeasurementCovariance(np.zeros((2, 2)) - np.eye(2))
