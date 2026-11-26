"""Tests for the Extended Kalman Filter (Lu et al. Eqs. (9)-(13))."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import GaussianState, Observation
from radar_bo.numerics import NumericalValidationError
from radar_bo.radar.extended_kalman_filter import ExtendedKalmanFilter
from radar_bo.radar.measurement_model import MeasurementGeometryError, PlanarRangeAzimuthModel

T = 2.5
PRIOR_MEAN = np.array([1000.0, 500.0, 10.0, -5.0])
PRIOR_COV = np.diag([100.0, 100.0, 25.0, 25.0])
R = np.diag([1.0, 4.0e-6])


def build_filter(method="paper_equation_13") -> ExtendedKalmanFilter:
    return ExtendedKalmanFilter(
        motion_model=ConstantVelocityMotionModel(process_noise_variance=25.0),
        measurement_model=PlanarRangeAzimuthModel(),
        covariance_update_method=method,
    )


def make_prior() -> GaussianState:
    return GaussianState(mean=PRIOR_MEAN.copy(), covariance=PRIOR_COV.copy(), label="prior")


def make_observation(zero_mean_noise=None) -> Observation:
    model = PlanarRangeAzimuthModel()
    z_ideal = model.measure(PRIOR_MEAN[:2])
    z = z_ideal if zero_mean_noise is None else z_ideal + zero_mean_noise
    return Observation(vector=z, covariance=R.copy())


class TestPredict:
    def test_matches_equations_9_and_10(self):
        """x_prior = F x; P_prior = F P F^T + Q, independently recomputed."""
        ekf = build_filter()
        f = np.array(
            [[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        q = ekf.motion_model.process_noise_covariance(T)
        expected_mean = f @ PRIOR_MEAN
        expected_cov = f @ PRIOR_COV @ f.T + q
        prior = ekf.predict(make_prior(), T, q)
        np.testing.assert_allclose(prior.mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(prior.covariance, expected_cov, atol=1e-12)
        assert prior.label == "prior"

    def test_prior_does_not_mutate_input(self):
        ekf = build_filter()
        state = make_prior()
        ekf.predict(state, T, ekf.motion_model.process_noise_covariance(T))
        np.testing.assert_array_equal(state.mean, PRIOR_MEAN)
        np.testing.assert_array_equal(state.covariance, PRIOR_COV)

    def test_invalid_process_covariance_rejected(self):
        ekf = build_filter()
        with pytest.raises(NumericalValidationError):
            ekf.predict(make_prior(), T, np.diag([1.0, -1.0, 1.0, 1.0]))
        with pytest.raises(NumericalValidationError):
            ekf.predict(make_prior(), T, np.full((3, 3), np.eye(3)))
        with pytest.raises(NumericalValidationError):
            ekf.predict(make_prior(), T, np.full((4, 4), np.nan))

    def test_negative_dt_rejected(self):
        ekf = build_filter()
        q = ekf.motion_model.process_noise_covariance(T)
        with pytest.raises(NumericalValidationError):
            ekf.predict(make_prior(), -T, q)


class TestUpdate:
    def test_matches_equations_11_to_13(self):
        """Posterior recomputed independently with explicit matrix inverse."""
        ekf = build_filter()
        q = ekf.motion_model.process_noise_covariance(T)
        prior = ekf.predict(make_prior(), T, q)
        obs = make_observation()

        mm = PlanarRangeAzimuthModel()
        h_mat = mm.jacobian(prior.mean[:2])
        h_prior = mm.measure(prior.mean[:2])
        innovation = obs.vector - h_prior
        s = h_mat @ prior.covariance @ h_mat.T + R
        k = prior.covariance @ h_mat.T @ np.linalg.inv(s)  # explicit inverse (independent path)
        expected_mean = prior.mean + k @ innovation
        expected_cov = (np.eye(4) - k @ h_mat) @ prior.covariance

        posterior = ekf.update(prior, obs)
        np.testing.assert_allclose(posterior.mean, expected_mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(posterior.covariance, expected_cov, rtol=1e-10, atol=1e-12)
        assert posterior.label == "posterior"

    def test_angular_residual_wrapping(self):
        """Azimuth innovation crossing the branch cut wraps to [-pi, pi)."""
        ekf = build_filter()
        q = ekf.motion_model.process_noise_covariance(T)
        prior = ekf.predict(make_prior(), T, q)
        # True azimuth of prior position ~ atan2(500, 1250) ~= 0.3805 rad.
        mm = PlanarRangeAzimuthModel()
        h_az = float(mm.measure(prior.mean[:2])[1])
        # Choose observed azimuth = h_az - 2*pi + delta so the RAW innovation is < -pi.
        delta = 0.1
        obs_azimuth = h_az - 2.0 * np.pi + delta
        obs = Observation(
            vector=[float(mm.measure(prior.mean[:2])[0]), obs_azimuth],
            covariance=R.copy(),
        )
        expected_innovation = np.array([0.0, delta])
        posterior = ekf.update(prior, obs)
        # Independent expected posterior using the WRAPPED innovation:
        h_mat = mm.jacobian(prior.mean[:2])
        s = h_mat @ prior.covariance @ h_mat.T + R
        k = prior.covariance @ h_mat.T @ np.linalg.inv(s)
        expected_mean = prior.mean + k @ expected_innovation
        np.testing.assert_allclose(posterior.mean, expected_mean, rtol=1e-10, atol=1e-12)

    def test_posterior_covariance_stays_valid_across_cycles(self):
        """Symmetric PSD posterior across repeated predict/update (no clipping)."""
        ekf = build_filter()
        rng = np.random.default_rng(11)
        mm = PlanarRangeAzimuthModel()
        q = ekf.motion_model.process_noise_covariance(T)
        estimate = make_prior()
        for step in range(12):
            prior = ekf.predict(estimate, T, q)
            true_position = prior.mean[:2] + rng.normal(0.0, 5.0, size=2)
            z = mm.measure(true_position) + np.array(
                [rng.normal(0.0, 1.0), rng.normal(0.0, np.sqrt(4.0e-6))]
            )
            posterior = ekf.update(prior, Observation(vector=z, covariance=R.copy()))
            p = posterior.covariance
            np.testing.assert_allclose(p, p.T, atol=1e-10)
            assert np.linalg.eigvalsh(p).min() >= -1e-10 * max(1.0, np.abs(np.linalg.eigvalsh(p)).max())
            estimate = posterior

    def test_prior_at_origin_raises_geometry_error(self):
        """The paper's zero-vector diagnostic initialization fails loudly, not silently."""
        ekf = build_filter()
        prior = GaussianState(mean=np.zeros(4), covariance=np.eye(4), label="paper_zero_identity_diagnostic")
        with pytest.raises(MeasurementGeometryError, match="origin"):
            ekf.update(prior, make_observation())

    def test_non_finite_observation_rejected(self):
        ekf = build_filter()
        with pytest.raises(Exception, match="non-finite"):
            ekf.update(make_prior(), Observation(vector=[np.nan, 0.0], covariance=R.copy()))

    def test_update_does_not_mutate_prior(self):
        ekf = build_filter()
        prior = make_prior()
        ekf.update(prior, make_observation())
        np.testing.assert_array_equal(prior.mean, PRIOR_MEAN)
        np.testing.assert_array_equal(prior.covariance, PRIOR_COV)


class TestJosephEquivalence:
    def test_mathematically_equivalent_under_valid_conditions(self):
        """Joseph form == Eq. (13) for the exact gain, over repeated cycles."""
        paper = build_filter("paper_equation_13")
        joseph = build_filter("joseph")
        rng = np.random.default_rng(21)
        mm = PlanarRangeAzimuthModel()
        q = paper.motion_model.process_noise_covariance(T)
        estimate_paper = make_prior()
        estimate_joseph = make_prior()
        for _ in range(10):
            true_position = estimate_paper.mean[:2] + rng.normal(0.0, 4.0, size=2)
            z = mm.measure(true_position) + np.array(
                [rng.normal(0.0, 1.0), rng.normal(0.0, np.sqrt(4.0e-6))]
            )
            obs = Observation(vector=z, covariance=R.copy())
            estimate_paper = paper.update(paper.predict(estimate_paper, T, q), obs)
            estimate_joseph = joseph.update(joseph.predict(estimate_joseph, T, q), obs)
            np.testing.assert_allclose(
                estimate_paper.mean, estimate_joseph.mean, rtol=1e-9, atol=1e-12
            )
            np.testing.assert_allclose(
                estimate_paper.covariance, estimate_joseph.covariance, rtol=1e-7, atol=1e-12
            )

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="covariance_update_method"):
            ExtendedKalmanFilter(
                motion_model=ConstantVelocityMotionModel(25.0),
                measurement_model=PlanarRangeAzimuthModel(),
                covariance_update_method="rank_reduction",
            )


class TestInputValidation:
    def test_nan_prior_rejected(self):
        ekf = build_filter()
        q = ekf.motion_model.process_noise_covariance(T)
        with pytest.raises(NumericalValidationError):
            ekf.predict(
                GaussianState(mean=np.full(4, np.nan), covariance=np.eye(4)), T, q
            )

    def test_wrong_type_rejected(self):
        ekf = build_filter()
        with pytest.raises(TypeError):
            ekf.update(np.zeros(4), make_observation())
        with pytest.raises(TypeError):
            ekf.update(make_prior(), np.zeros(2))
