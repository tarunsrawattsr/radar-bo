"""Synthetic single-target tracking validation — M3 deliverable (instruction 11).

RESULT LABEL: synthetic M3 validation. This experiment verifies the NUMERICAL
CORRECTNESS of the motion -> measurement -> estimation -> tracking-cost
pipeline. It does NOT reproduce the numerical results of Lu et al. and does
NOT demonstrate operational UAV surveillance performance.

Synthetic settings (explicitly documented, NOT paper values):

* true initial state ``[2000, 500, -10, 5]`` (m, m/s);
* synthetic reference SNR ``100`` (linear) supplied to the noise model —
  ``TrackingConfig.reference_snr`` remains ``None`` (paper-unresolved);
* measurement-driven initialization with synthetic velocity prior
  ``(0, 0)`` m/s and variance ``25`` (m/s)^2;
* fixed dwell time ``tau_0 = 1 s`` (paper reference dwell; dwell allocation
  is NOT optimized in M3).

Paper-derived settings (Lu et al., Table II / Eqs. (1)-(15)): revisit interval
T0 = 2.5 s, reference variances 100 m^2 and 4e-4 rad^2, maneuverability-noise
default interpretation ``table_variance`` (25), Eq. (6) division semantics.

Documented separation decision: the measurement covariance R_t is computed by
the simulation/measurement layer from the true operating point (paper Eqs.
(5)-(8): SNR depends on the actual target distance and the allocated dwell)
and is supplied to the estimator through the Observation interface. The
estimator never receives the true simulated state.

Initialization/update indexing convention (M4 review, instruction 3.3): the
step-0 observation (``step_index=0``) is consumed ONLY by measurement-driven
initialization. Steps 1..N each generate a FRESH observation (``step_index=k``)
consumed by the ordinary EKF update at that step, after one predict over the
revisit interval. The initialization observation is never reused as an update
measurement; regression test:
``radar_bo/tests/test_m3_integration.py::TestInitializationSequencing``.

Statistical claim scope: the reported RMS position error is a SINGLE-RUN
numerical-validation result. It is NOT evidence of statistical filter
consistency (no consistency analysis, e.g. NEES/NIS studies, is performed
here).

Outputs (under ``results/m3_synthetic_validation/``): serialized config,
metrics JSON, trajectory data NPZ, and a 2x2 diagnostic figure (true vs
estimated trajectory; position error; prior/posterior covariance traces;
tracking cost). All artifacts carry the synthetic-validation label.

Run::

    python -m radar_bo.experiments.m3_synthetic_validation [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from radar_bo.config import SimulationConfig, build_default_provenance
from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import Observation, TargetState
from radar_bo.radar.extended_kalman_filter import ExtendedKalmanFilter
from radar_bo.radar.initialization import initialize_estimator
from radar_bo.radar.measurement_model import MeasurementGeometryError, PlanarRangeAzimuthModel
from radar_bo.radar.measurement_noise import PlanarSNRMeasurementNoiseModel
from radar_bo.radar.tracking_cost import position_tracking_cost

RESULT_LABEL = "synthetic M3 validation - numerical correctness only; not a paper reproduction"

# ---- Explicitly labeled synthetic settings -----------------------------------
SYNTHETIC_TRUE_INITIAL_STATE = (2000.0, 500.0, -10.0, 5.0)
SYNTHETIC_REFERENCE_SNR = 100.0  # linear (not dB); NOT a paper value
SYNTHETIC_VELOCITY_PRIOR = (0.0, 0.0)  # m/s; NOT a paper value
SYNTHETIC_VELOCITY_PRIOR_VARIANCE = 25.0  # (m/s)^2; NOT a paper value


def _build_config() -> SimulationConfig:
    tracking = SimulationConfig().tracking.model_copy(
        update={
            "initialization_policy": "measurement_driven",
            "measurement_driven_initial_velocity": SYNTHETIC_VELOCITY_PRIOR,
            "measurement_driven_initial_velocity_variance": SYNTHETIC_VELOCITY_PRIOR_VARIANCE,
        }
    )
    environment = SimulationConfig().environment.model_copy(
        update={
            "num_targets": 1,
            "initial_conditions": [SYNTHETIC_TRUE_INITIAL_STATE],
            "simulation_duration": 60.0,
        }
    )
    return SimulationConfig(tracking=tracking, environment=environment)


def run(output_dir: str | Path = "results/m3_synthetic_validation", seed: int = 0) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config()
    tracking = config.tracking
    motion = ConstantVelocityMotionModel.from_config(tracking)
    measurement_model = PlanarRangeAzimuthModel()
    # Synthetic SNR is supplied EXPLICITLY here; tracking.reference_snr stays None.
    noise_model = PlanarSNRMeasurementNoiseModel(
        reference_range_variance=tracking.reference_range_variance,
        reference_angle_variance=tracking.reference_angle_variance,
        reference_distance=tracking.reference_distance,
        reference_dwell_time=tracking.reference_dwell_time,
        reference_snr=SYNTHETIC_REFERENCE_SNR,
    )
    ekf = ExtendedKalmanFilter(motion, measurement_model, tracking.covariance_update_method)

    revisit = tracking.revisit_interval
    dwell = tracking.reference_dwell_time
    num_steps = int(round(config.environment.simulation_duration / revisit))

    rng_truth = np.random.default_rng(seed)
    rng_measurement = np.random.default_rng(seed + 1)

    # ---- Simulation layer: true state (inaccessible to the estimator) --------
    truth = TargetState(vector=SYNTHETIC_TRUE_INITIAL_STATE, target_id="target_0")

    def synthesize_observation(true_state: TargetState, step: int) -> Observation:
        """Measurement layer: R from the operating point, z = h(x_true) + noise."""
        true_range = float(np.hypot(*true_state.position))
        r = noise_model.observation_covariance(dwell, true_range).matrix
        noise = np.array(
            [
                rng_measurement.normal(0.0, float(np.sqrt(r[0, 0]))),
                rng_measurement.normal(0.0, float(np.sqrt(r[1, 1]))),
            ]
        )
        return Observation(
            vector=measurement_model.measure(true_state.position) + noise,
            covariance=r,
            step_index=step,
        )

    # ---- Initialization (measurement-driven, operational policy) -------------
    observation = synthesize_observation(truth, step=0)
    posterior = initialize_estimator("measurement_driven", tracking, observation)

    steps = [0]
    true_positions = [truth.position.copy()]
    estimated_positions = [posterior.mean[:2].copy()]
    prior_traces = []  # prior traces begin at step 1: no prediction precedes the first update
    posterior_traces = [float(np.trace(posterior.covariance))]
    costs = [position_tracking_cost(posterior.covariance)]

    # ---- Predict/update cycle -------------------------------------------------
    for step in range(1, num_steps + 1):
        truth = motion.propagate_stochastic(truth, revisit, rng_truth)
        observation = synthesize_observation(truth, step)
        prior = ekf.predict(posterior, revisit, motion.process_noise_covariance(revisit))
        posterior = ekf.update(prior, observation)

        steps.append(step)
        true_positions.append(truth.position.copy())
        estimated_positions.append(posterior.mean[:2].copy())
        prior_traces.append(float(np.trace(prior.covariance)))
        posterior_traces.append(float(np.trace(posterior.covariance)))
        costs.append(position_tracking_cost(posterior.covariance))

    steps_arr = np.asarray(steps)
    true_arr = np.asarray(true_positions)
    est_arr = np.asarray(estimated_positions)
    prior_traces_arr = np.asarray(prior_traces)
    posterior_traces_arr = np.asarray(posterior_traces)
    costs_arr = np.asarray(costs)
    position_errors = np.linalg.norm(est_arr - true_arr, axis=1)

    # ---- Diagnostic: the paper's zero-vector initialization fails loudly ------
    paper_init_diagnostic = "not_run"
    try:
        diagnostic_estimate = initialize_estimator("paper_zero_identity", tracking, None)
        diagnostic_prior = ekf.predict(
            diagnostic_estimate, revisit, motion.process_noise_covariance(revisit)
        )
        ekf.update(diagnostic_prior, synthesize_observation(truth, step=num_steps))
        paper_init_diagnostic = "unexpectedly_succeeded"
    except MeasurementGeometryError:
        paper_init_diagnostic = (
            "MeasurementGeometryError raised as designed: the paper's zero-vector "
            "initialization is retained as a diagnostic mode; an update at the "
            "singular origin fails clearly instead of producing NaN/inf."
        )

    # ---- Honest failure on any non-finite result ------------------------------
    for name, array in (
        ("estimates", est_arr),
        ("true positions", true_arr),
        ("tracking costs", costs_arr),
        ("prior traces", prior_traces_arr),
        ("posterior traces", posterior_traces_arr),
    ):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"Non-finite values in {name}; synthetic validation FAILED.")

    metrics = {
        "result_label": RESULT_LABEL,
        "num_steps": num_steps,
        "revisit_interval_s": revisit,
        "dwell_time_s": dwell,
        "random_seed": seed,
        "synthetic_settings": {
            "true_initial_state": list(SYNTHETIC_TRUE_INITIAL_STATE),
            "synthetic_reference_snr_linear": SYNTHETIC_REFERENCE_SNR,
            "note": "tracking.reference_snr remains None (paper-unresolved); the SNR above is a synthetic testing value",
            "velocity_prior": list(SYNTHETIC_VELOCITY_PRIOR),
            "velocity_prior_variance": SYNTHETIC_VELOCITY_PRIOR_VARIANCE,
        },
        "metrics": {
            "rms_position_error_m": float(np.sqrt(np.mean(position_errors**2))),
            "final_position_error_m": float(position_errors[-1]),
            "mean_tracking_cost": float(np.mean(costs_arr)),
            "final_tracking_cost": float(costs_arr[-1]),
            "final_posterior_covariance_trace": float(posterior_traces_arr[-1]),
        },
        "paper_zero_initialization_diagnostic": paper_init_diagnostic,
        "not_paper_reproduction": True,
        "not_operational_capability": True,
    }

    # ---- Persist artifacts ------------------------------------------------------
    config.save_json(output_dir / "config.json")
    provenance = build_default_provenance(config)
    provenance.save_json(output_dir / "provenance_register.json")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(
        output_dir / "trajectory_data.npz",
        steps=steps_arr,
        true_positions=true_arr,
        estimated_positions=est_arr,
        prior_covariance_traces=prior_traces_arr,
        posterior_covariance_traces=posterior_traces_arr,
        tracking_costs=costs_arr,
        position_errors=position_errors,
    )
    _plot(output_dir / "m3_synthetic_validation.png", steps_arr, true_arr, est_arr, position_errors,
          prior_traces_arr, posterior_traces_arr, costs_arr)
    return metrics


def _plot(path: Path, steps, true_arr, est_arr, position_errors, prior_traces, posterior_traces, costs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Synthetic M3 validation - numerical correctness only (not a paper reproduction)")

    ax = axes[0, 0]
    ax.plot(true_arr[:, 0], true_arr[:, 1], "o-", ms=3, label="true trajectory (simulation only)")
    ax.plot(est_arr[:, 0], est_arr[:, 1], "s--", ms=3, label="EKF posterior estimate")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("True vs estimated trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, position_errors, "d-", ms=3)
    ax.set_xlabel("simulation step (T0 = 2.5 s each)")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position-estimation error")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps[1:], prior_traces, ".-", ms=3, label="prior covariance trace")
    ax.plot(steps, posterior_traces, ".-", ms=3, label="posterior covariance trace")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("trace of 4x4 covariance")
    ax.set_title("Prior and posterior covariance traces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, costs, ".-", ms=3)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("tracking cost [m^2]")
    ax.set_title("Tracking cost (Eq. 14): trace(E P E^T)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> int:
    output_dir = Path(argv[1]) if len(argv) > 1 else Path("results/m3_synthetic_validation")
    metrics = run(output_dir)
    print(RESULT_LABEL)
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
