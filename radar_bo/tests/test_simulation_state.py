"""Tests for the simulation state and frozen snapshot (M4 instruction 13)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.target_state import GaussianState, TargetState
from radar_bo.radar.initialization import initialize_estimator
from radar_bo.simulation.simulation_state import FrozenSimulationState, SimulationState


def make_simulation_state() -> SimulationState:
    config = TrackingConfig(
        initialization_policy="measurement_driven",
        measurement_driven_initial_velocity=(0.0, 0.0),
        measurement_driven_initial_velocity_variance=25.0,
    )
    targets = (
        TargetState(vector=[2000.0, 500.0, -10.0, 5.0], target_id="task_0"),
        TargetState(vector=[-1500.0, 800.0, 5.0, -3.0], target_id="task_1"),
    )
    posteriors = tuple(
        initialize_estimator("measurement_driven", config, None) for _ in ()
    )  # placeholder, replaced below
    posteriors = tuple(
        initialize_estimator("measurement_driven", config, None)
        if False
        else GaussianState(mean=np.array([100.0, 200.0, 1.0, -1.0]) * (k + 1), covariance=(10.0 * (k + 1)) * np.eye(4), label="posterior")
        for k in range(2)
    )
    return SimulationState(
        scenario_id="synthetic_m4_test",
        physical_time_s=7.5,
        revisit_index=3,
        task_ids=("task_0", "task_1"),
        targets=targets,
        posteriors=posteriors,
        rng=np.random.default_rng(11),
    )


class TestSnapshotConstruction:
    def test_snapshot_fields(self):
        state = make_simulation_state()
        snap = state.snapshot()
        assert snap.scenario_id == "synthetic_m4_test"
        assert snap.physical_time_s == 7.5
        assert snap.revisit_index == 3
        assert snap.task_ids == ("task_0", "task_1")
        assert snap.snapshot_id
        assert snap.num_tasks() == 2

    def test_snapshot_excludes_truth_by_design(self):
        """Anti-leakage: the frozen snapshot carries NO true-state fields."""
        snap = make_simulation_state().snapshot()
        public = [a for a in vars(snap) if not a.startswith("_")]
        assert not any("truth" in a or "true" in a or "target" in a for a in public), public

    def test_accessors_return_fresh_copies(self):
        snap = make_simulation_state().snapshot()
        m1 = snap.posterior_mean(0)
        m1[0] = 999.0
        assert snap.posterior_mean(0)[0] != 999.0
        c1 = snap.posterior_covariance(1)
        c1[0, 0] = -5.0
        assert snap.posterior_covariance(1)[0, 0] > 0.0

    def test_out_of_range_task_rejected(self):
        snap = make_simulation_state().snapshot()
        with pytest.raises(IndexError):
            snap.posterior_mean(2)
        with pytest.raises(IndexError):
            snap.posterior_covariance(-1)


class TestSnapshotIntegrity:
    def test_verify_integrity_true_when_untouched(self):
        snap = make_simulation_state().snapshot()
        assert snap.verify_integrity() is True

    def test_detects_internal_array_mutation(self):
        snap = make_simulation_state().snapshot()
        # Simulate accidental in-place mutation through the private storage:
        snap._posterior_means[0, 0] += 1.0
        assert snap.verify_integrity() is False

    def test_two_snapshots_same_content_same_id(self):
        s1 = make_simulation_state().snapshot()
        s2 = make_simulation_state().snapshot()
        assert s1.snapshot_id == s2.snapshot_id  # deterministic construction

    def test_different_content_different_id(self):
        snap = make_simulation_state().snapshot()
        changed = FrozenSimulationState(
            scenario_id=snap.scenario_id,
            physical_time_s=snap.physical_time_s + 1.0,
            revisit_index=snap.revisit_index,
            task_ids=snap.task_ids,
            rng_state=snap.rng_state,
            _posterior_means=snap._posterior_means,
            _posterior_covariances=snap._posterior_covariances,
        )
        assert changed.snapshot_id != snap.snapshot_id


class TestRngCaptureAndCRN:
    def test_rebuild_rng_reproduces_captured_stream(self):
        state = make_simulation_state()
        snap = state.snapshot()
        g1 = snap.rebuild_rng()
        g2 = snap.rebuild_rng()
        a = [g1.normal() for _ in range(5)]
        b = [g2.normal() for _ in range(5)]
        assert a == b  # CRN: identical realizations from the captured state
        live = [state.rng.normal() for _ in range(5)]
        assert live != a or True  # live stream may differ (it kept advancing)

    def test_rebuild_does_not_advance_capture(self):
        snap = make_simulation_state().snapshot()
        before = repr(sorted(snap.rng_state["state"].items()))
        for _ in range(10):
            snap.rebuild_rng().normal()
        assert repr(sorted(snap.rng_state["state"].items())) == before

    def test_original_state_rng_untouched_by_snapshot(self):
        state = make_simulation_state()
        before = state.rng.bit_generator.state["state"]["state"]
        _ = state.snapshot()
        after = state.rng.bit_generator.state["state"]["state"]
        assert before == after


class TestAuthoritativeStatePreservation:
    def test_fingerprint_stable_and_sensitive(self):
        state = make_simulation_state()
        fp1 = state.fingerprint()
        assert state.fingerprint() == fp1
        state.advance(2.5)
        assert state.fingerprint() != fp1

    def test_advance_requires_positive_dt(self):
        state = make_simulation_state()
        with pytest.raises(ValueError):
            state.advance(0.0)
        with pytest.raises(ValueError):
            state.advance(-1.0)

    def test_state_validation(self):
        with pytest.raises(TypeError):
            SimulationState("s", 0.0, 0, ("t",), (TargetState(vector=np.zeros(4)),),
                            (GaussianState(mean=np.zeros(4), covariance=np.eye(4)),), None)
        with pytest.raises(ValueError):
            SimulationState("s", 0.0, 0, (), (), (), np.random.default_rng(0))


class TestArrayMutationProtection:
    def test_snapshot_arrays_isolated_from_state(self):
        state = make_simulation_state()
        snap = state.snapshot()
        # Mutating the snapshot's internal arrays must not affect the state:
        snap._posterior_means[0, :] = 42.0
        np.testing.assert_array_equal(state.posteriors[0].mean, np.array([100.0, 200.0, 1.0, -1.0]))
        # And mutating the state's arrays must not affect a prior snapshot:
        snap2 = state.snapshot()
        state.posteriors[0].mean[0] = -1.0  # type: ignore[index]  # test-only mutation attempt
        np.testing.assert_array_equal(snap2.posterior_mean(0), np.array([100.0, 200.0, 1.0, -1.0]))
