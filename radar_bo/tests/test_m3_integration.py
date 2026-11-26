"""Integration test: synthetic motion -> measurement -> estimation -> tracking cost.

Covers the full M3 pipeline on a seeded synthetic single-target scenario
(instruction 12), including reproducibility and truth-separation checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import Observation, TargetState
from radar_bo.radar.extended_kalman_filter import ExtendedKalmanFilter
from radar_bo.radar.initialization import initialize_estimator
from radar_bo.radar.measurement_model import PlanarRangeAzimuthModel
from radar_bo.radar.measurement_noise import PlanarSNRMeasurementNoiseModel
from radar_bo.radar.tracking_cost import position_tracking_cost

# Synthetic settings, explicitly labeled (NOT paper values):
SYNTHETIC_TRUE_INITIAL_STATE = (2000.0, 500.0, -10.0, 5.0)
SYNTHETIC_REFERENCE_SNR = 100.0  # linear; tracking.reference_snr stays None
SYNTHETIC_VELOCITY_PRIOR = (0.0, 0.0)
SYNTHETIC_VELOCITY_PRIOR_VARIANCE = 25.0
NUM_STEPS = 12


def synthetic_tracking_config() -> TrackingConfig:
    return TrackingConfig(
        initialization_policy="measurement_driven",
        measurement_driven_initial_velocity=SYNTHETIC_VELOCITY_PRIOR,
        measurement_driven_initial_velocity_variance=SYNTHETIC_VELOCITY_PRIOR_VARIANCE,
    )


def run_pipeline(seed: int, config: TrackingConfig | None = None):
    """Run the seeded synthetic pipeline; return recorded arrays.

    Documented decision: the measurement covariance R_t is computed by the
    simulation/measurement layer from the true operating point (paper Eqs.
    (5)-(8): SNR depends on the actual target distance and dwell) and supplied
    to the estimator through the Observation interface. The estimator never
    receives the true state.
    """
    config = config or synthetic_tracking_config()
    motion = ConstantVelocityMotionModel.from_config(config)
    measurement_model = PlanarRangeAzimuthModel()
    noise_model = PlanarSNRMeasurementNoiseModel(
        reference_range_variance=config.reference_range_variance,
        reference_angle_variance=config.reference_angle_variance,
        reference_distance=config.reference_distance,
        reference_dwell_time=config.reference_dwell_time,
        reference_snr=SYNTHETIC_REFERENCE_SNR,  # synthetic, labeled
    )
    ekf = ExtendedKalmanFilter(motion, measurement_model, config.covariance_update_method)

    rng_truth = np.random.default_rng(seed)
    rng_measurement = np.random.default_rng(seed + 1)
    t = config.revisit_interval
    dwell = config.reference_dwell_time

    truth = TargetState(vector=SYNTHETIC_TRUE_INITIAL_STATE, target_id="target_0")
    r0 = noise_model.observation_covariance(dwell, float(np.hypot(*truth.position))).matrix
    z0 = measurement_model.measure(truth.position) + np.array(
        [rng_measurement.normal(0.0, np.sqrt(r0[0, 0])), rng_measurement.normal(0.0, np.sqrt(r0[1, 1]))]
    )
    posterior = initialize_estimator(
        "measurement_driven", config, Observation(vector=z0, covariance=r0, step_index=0)
    )

    true_positions, estimates = [truth.position.copy()], [posterior.mean.copy()]
    prior_traces, posterior_traces, costs = [], [], [position_tracking_cost(posterior.covariance)]
    for step in range(1, NUM_STEPS + 1):
        truth = motion.propagate_stochastic(truth, t, rng_truth)
        true_range = float(np.hypot(*truth.position))
        r = noise_model.observation_covariance(dwell, true_range).matrix
        z = measurement_model.measure(truth.position) + np.array(
            [rng_measurement.normal(0.0, np.sqrt(r[0, 0])), rng_measurement.normal(0.0, np.sqrt(r[1, 1]))]
        )
        prior = ekf.predict(posterior, t, motion.process_noise_covariance(t))
        posterior = ekf.update(prior, Observation(vector=z, covariance=r, step_index=step))

        true_positions.append(truth.position.copy())
        estimates.append(posterior.mean.copy())
        prior_traces.append(float(np.trace(prior.covariance)))
        posterior_traces.append(float(np.trace(posterior.covariance)))
        costs.append(position_tracking_cost(posterior.covariance))

    return {
        "true_positions": np.asarray(true_positions),
        "estimates": np.asarray(estimates),
        "prior_traces": np.asarray(prior_traces),
        "posterior_traces": np.asarray(posterior_traces),
        "costs": np.asarray(costs),
    }


class TestSyntheticPipeline:
    def test_finite_estimates_and_valid_covariances(self):
        result = run_pipeline(seed=42)
        assert np.all(np.isfinite(result["estimates"]))
        assert np.all(np.isfinite(result["true_positions"]))
        assert np.all(np.isfinite(result["costs"]))
        assert np.all(result["prior_traces"] > 0.0)
        assert np.all(result["posterior_traces"] > 0.0)

    def test_tracking_cost_finite_and_positive(self):
        result = run_pipeline(seed=42)
        assert np.all(result["costs"] > 0.0)
        assert np.all(np.isfinite(result["costs"]))

    def test_reproducibility_with_same_seed(self):
        a = run_pipeline(seed=123)
        b = run_pipeline(seed=123)
        np.testing.assert_array_equal(a["estimates"], b["estimates"])
        np.testing.assert_array_equal(a["true_positions"], b["true_positions"])
        np.testing.assert_array_equal(a["costs"], b["costs"])

    def test_estimates_differ_from_truth(self):
        """Estimator output is a filtered estimate from noisy observations only."""
        result = run_pipeline(seed=77)
        errors = np.linalg.norm(result["estimates"][:, :2] - result["true_positions"], axis=1)
        assert np.all(errors > 0.0)  # never exactly the truth
        assert np.isfinite(errors).all()

    def test_both_maneuverability_interpretations_run_and_differ(self):
        """Both sigma_w interpretations complete the pipeline with different Q."""
        default_result = run_pipeline(seed=5)
        literal_config = TrackingConfig(
            maneuverability_noise_interpretation="equation_literal",
            initialization_policy="measurement_driven",
            measurement_driven_initial_velocity=SYNTHETIC_VELOCITY_PRIOR,
            measurement_driven_initial_velocity_variance=SYNTHETIC_VELOCITY_PRIOR_VARIANCE,
        )
        literal_result = run_pipeline(seed=5, config=literal_config)
        assert np.all(np.isfinite(literal_result["estimates"]))
        assert not np.allclose(default_result["estimates"], literal_result["estimates"])

    def test_joseph_method_pipeline(self):
        config = TrackingConfig(
            covariance_update_method="joseph",
            initialization_policy="measurement_driven",
            measurement_driven_initial_velocity=SYNTHETIC_VELOCITY_PRIOR,
            measurement_driven_initial_velocity_variance=SYNTHETIC_VELOCITY_PRIOR_VARIANCE,
        )
        result = run_pipeline(seed=9, config=config)
        assert np.all(np.isfinite(result["estimates"]))
