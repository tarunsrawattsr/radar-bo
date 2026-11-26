"""Tests for the constant-velocity motion model (Lu et al. Eqs. (1)-(3))."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import TargetState
from radar_bo.numerics import NumericalValidationError

T = 2.5  # paper revisit interval


def _expected_q(t: float, sigma_w_squared: float) -> np.ndarray:
    """Independent closed-form Eq. (3) for cross-checking."""
    return sigma_w_squared * np.array(
        [
            [t**4 / 4, 0.0, t**3 / 2, 0.0],
            [0.0, t**4 / 4, 0.0, t**3 / 2],
            [t**3 / 2, 0.0, t**2, 0.0],
            [0.0, t**3 / 2, 0.0, t**2],
        ]
    )


class TestStateTransitionMatrix:
    def test_paper_structure(self):
        model = ConstantVelocityMotionModel(process_noise_variance=25.0)
        f = model.state_transition_matrix(T)
        expected = np.array(
            [
                [1.0, 0.0, T, 0.0],
                [0.0, 1.0, 0.0, T],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(f, expected)

    def test_invalid_dt_rejected(self):
        model = ConstantVelocityMotionModel(25.0)
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(NumericalValidationError):
                model.state_transition_matrix(bad)


class TestDeterministicPropagation:
    def test_matches_independently_calculated_expected_state(self):
        model = ConstantVelocityMotionModel(25.0)
        state = TargetState(vector=[100.0, 200.0, 3.0, -4.0])
        propagated = model.propagate_deterministic(state, T)
        # Independent computation: x' = x + vx*T; y' = y + vy*T; velocities unchanged.
        expected = np.array([100.0 + 3.0 * T, 200.0 + (-4.0) * T, 3.0, -4.0])
        np.testing.assert_allclose(propagated.vector, expected, rtol=0.0, atol=1e-12)

    def test_metadata_preserved(self):
        model = ConstantVelocityMotionModel(25.0)
        state = TargetState(vector=[1.0, 2.0, 3.0, 4.0], target_id="t9", is_active=False)
        out = model.propagate_deterministic(state, T)
        assert out.target_id == "t9"
        assert out.is_active is False

    def test_rejects_non_target_state(self):
        model = ConstantVelocityMotionModel(25.0)
        with pytest.raises(TypeError):
            model.propagate_deterministic(np.zeros(4), T)


class TestProcessNoiseCovariance:
    @pytest.mark.parametrize("variance", [25.0, 625.0, 1.0])
    def test_matches_closed_form(self, variance):
        model = ConstantVelocityMotionModel(variance)
        np.testing.assert_allclose(model.process_noise_covariance(T), _expected_q(T, variance), rtol=1e-14)

    def test_symmetry_and_positive_semidefiniteness_both_interpretations(self):
        for variance in (25.0, 625.0):
            q = ConstantVelocityMotionModel(variance).process_noise_covariance(T)
            np.testing.assert_allclose(q, q.T, atol=1e-15)
            assert np.linalg.eigvalsh(q).min() >= -1e-12

    def test_low_rank_factor_identity(self):
        """Q == G G^T sigma^2 exactly (algebraic identity used for sampling)."""
        model = ConstantVelocityMotionModel(25.0)
        g = model.acceleration_input_matrix(T)
        np.testing.assert_allclose(g @ g.T * model.process_noise_variance, model.process_noise_covariance(T), atol=1e-12)

    @pytest.mark.parametrize("interp, expected_variance", [("table_variance", 25.0), ("equation_literal", 625.0)])
    def test_both_maneuverability_interpretations(self, interp, expected_variance):
        """Both sigma_w interpretations resolve to the correct Q multiplier."""
        config = TrackingConfig(maneuverability_noise_interpretation=interp)
        model = ConstantVelocityMotionModel.from_config(config)
        assert model.process_noise_variance == expected_variance
        np.testing.assert_allclose(
            model.process_noise_covariance(T), _expected_q(T, expected_variance), rtol=1e-14
        )

    def test_interpretation_ratio_is_sigma_w_squared(self):
        q25 = ConstantVelocityMotionModel.from_config(TrackingConfig()).process_noise_covariance(T)
        q625 = ConstantVelocityMotionModel.from_config(
            TrackingConfig(maneuverability_noise_interpretation="equation_literal")
        ).process_noise_covariance(T)
        np.testing.assert_allclose(q625, 25.0 * q25)  # 625 / 25 (no elementwise division: Q has exact zeros)


class TestStochasticPropagation:
    def test_explicit_generator_required(self):
        model = ConstantVelocityMotionModel(25.0)
        state = TargetState(vector=[1.0, 2.0, 3.0, 4.0])
        with pytest.raises(TypeError, match="Generator"):
            model.propagate_stochastic(state, T, None)
        with pytest.raises(TypeError, match="Generator"):
            model.propagate_stochastic(state, T, np.random.RandomState(0))

    def test_seeded_reproducibility(self):
        model = ConstantVelocityMotionModel(25.0)
        state = TargetState(vector=[100.0, 200.0, 3.0, -4.0])
        a = model.propagate_stochastic(state, T, np.random.default_rng(1234))
        b = model.propagate_stochastic(state, T, np.random.default_rng(1234))
        np.testing.assert_array_equal(a.vector, b.vector)

    def test_different_seeds_differ(self):
        model = ConstantVelocityMotionModel(25.0)
        state = TargetState(vector=[100.0, 200.0, 3.0, -4.0])
        a = model.propagate_stochastic(state, T, np.random.default_rng(1))
        b = model.propagate_stochastic(state, T, np.random.default_rng(2))
        assert not np.allclose(a.vector, b.vector)

    def test_sample_covariance_matches_q(self):
        """Empirical second moment of G a matches Q (N(0, Q) equivalence).

        atol covers Monte-Carlo noise on structurally-zero entries (sample
        cross-covariances are O(sigma^2 / sqrt(N)) ~ 1 m^2, not exactly 0).
        """
        model = ConstantVelocityMotionModel(25.0)
        g = model.acceleration_input_matrix(T)
        rng = np.random.default_rng(7)
        n = 60_000
        samples = g @ rng.normal(0.0, np.sqrt(model.process_noise_variance), size=(2, n))
        empirical = np.cov(samples)
        np.testing.assert_allclose(empirical, model.process_noise_covariance(T), rtol=0.05, atol=4.0)
