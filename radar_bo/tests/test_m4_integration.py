"""M4 integration tests: configuration -> snapshot -> allocation -> feasibility ->
component evaluation -> utility -> evaluation result (M4 instruction 18)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.resource_config import ResourceConfig
from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.environment.target_state import Observation, TargetState
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.constraints import ConstraintManager
from radar_bo.optimization.objective import ObjectiveEvaluator, SnapshotDiagnosticTrackingCostProvider
from radar_bo.optimization.resource_accounting import ResourceAccountant
from radar_bo.radar.initialization import initialize_estimator
from radar_bo.radar.scanning_model import SyntheticLinearScanningModel
from radar_bo.simulation.simulation_state import SimulationState

ACTION_SET = tuple(ResourceConfig().dwell_time_action_set)
T0 = 2.5
N_TASKS = 3
BETA = 1.0e6


def make_state() -> SimulationState:
    config = TrackingConfig(
        initialization_policy="measurement_driven",
        measurement_driven_initial_velocity=(0.0, 0.0),
        measurement_driven_initial_velocity_variance=25.0,
    )
    posteriors = tuple(
        initialize_estimator(
            "measurement_driven", config,
            Observation(vector=[3000.0, 0.5], covariance=np.diag([1.0, 4e-6])),
        )
        for _ in range(N_TASKS)
    )
    targets = tuple(
        TargetState(vector=[3000.0 * np.cos(0.5), 3000.0 * np.sin(0.5), 0.0, 0.0], target_id=f"task_{k}")
        for k in range(N_TASKS)
    )
    return SimulationState(
        scenario_id="synthetic_m4_integration",
        physical_time_s=0.0,
        revisit_index=0,
        task_ids=tuple(f"task_{k}" for k in range(N_TASKS)),
        targets=targets,
        posteriors=posteriors,
        rng=np.random.default_rng(2024),
    )


def make_evaluator() -> ObjectiveEvaluator:
    """PROPOSED hard per-slot BO profile (per M2/M4 review)."""
    profile = ResourceConfig(time_budget_semantics="hard_per_slot")
    return ObjectiveEvaluator(
        constraint_manager=ConstraintManager(profile, N_TASKS, T0),
        accountant=ResourceAccountant(profile, revisit_interval_s=T0),
        tracking_provider=SnapshotDiagnosticTrackingCostProvider(),
        scanning_model=SyntheticLinearScanningModel(),
        resource_config=profile,
    )


# Candidates: feasible partial, exact-boundary, over-budget, and one whose
# utility is hand-checkable below.
CANDIDATE_PARTIAL = (0, 2, 4)      # 0.1 + 0.3 + 0.5 = 0.9 s
CANDIDATE_EXACT = (4, 9, 9)        # 0.5 + 1.0 + 1.0 = 2.5 s (exact budget)
CANDIDATE_OVER = (9, 9, 9)         # 3.0 s > 2.5 s


class TestFullWorkflow:
    def test_valid_allocations_produce_finite_results(self):
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        for indices in (CANDIDATE_PARTIAL, CANDIDATE_EXACT):
            result = evaluator.evaluate_candidate(snapshot, ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET)))
            assert result.status == "success"
            assert np.isfinite(result.utility)
            assert all(np.isfinite(c) for c in result.per_task_tracking_costs)
            assert np.isfinite(result.scanning_gamma)
            assert result.feasibility.is_feasible

    def test_invalid_allocation_rejected_as_infeasible(self):
        result = make_evaluator().evaluate_candidate(make_state().snapshot(), ResourceAllocation.from_dwell_indices(list(CANDIDATE_OVER), list(ACTION_SET)))
        assert result.status == "infeasible"
        assert result.utility is None

    def test_unresolved_configuration_not_silently_replaced(self):
        """Power block present but bounds unresolved: category is unresolved_parameter."""
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        allocation = ResourceAllocation.from_dwell_indices(
            list(CANDIDATE_PARTIAL), list(ACTION_SET), continuous_power_w=(10.0, 10.0, 10.0)
        )
        result = evaluator.evaluate_candidate(snapshot, allocation)
        assert result.feasibility.category == "unresolved_parameter"
        assert not result.feasibility.is_feasible

    def test_result_corresponds_to_correct_allocation(self):
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        allocations = [ResourceAllocation.from_dwell_indices(list(i), list(ACTION_SET)) for i in
                       (CANDIDATE_PARTIAL, CANDIDATE_EXACT, (2, 2, 2))]
        results = [evaluator.evaluate_candidate(snapshot, a) for a in allocations]
        for allocation, result in zip(allocations, results):
            assert result.allocation_id == allocation.content_id()
            assert result.snapshot_id == snapshot.snapshot_id
        # Distinct allocations produce distinct recorded usage:
        assert results[0].resource_usage.total_dwell_time_s != results[1].resource_usage.total_dwell_time_s

    def test_hand_checked_utility_for_exact_budget_candidate(self):
        """Exact-budget candidate: remaining scanning time 0 -> gamma 0 ->
        utility = -(sum of diagnostic costs) exactly (beta * 0 contributes nothing)."""
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        costs = [float(snapshot.posterior_covariance(k)[0, 0] + snapshot.posterior_covariance(k)[1, 1]) for k in range(N_TASKS)]
        result = evaluator.evaluate_candidate(snapshot, ResourceAllocation.from_dwell_indices(list(CANDIDATE_EXACT), list(ACTION_SET)))
        assert result.status == "success"
        assert result.resource_usage.remaining_scanning_time_s_paper_eq24 == pytest.approx(0.0)
        assert result.scanning_gamma == pytest.approx(0.0)
        assert result.per_task_tracking_costs == pytest.approx(costs)
        assert result.utility == pytest.approx(-sum(costs))


class TestIsolationAndTime:
    def test_candidate_evaluation_does_not_affect_other_candidates_starting_state(self):
        evaluator = make_evaluator()
        state = make_state()
        snapshot_a = state.snapshot()
        fingerprint_before = state.fingerprint()
        first = evaluator.evaluate_candidate(snapshot_a, ResourceAllocation.from_dwell_indices(list(CANDIDATE_PARTIAL), list(ACTION_SET)))
        # A second candidate starts from an IDENTICAL snapshot content:
        snapshot_b = state.snapshot()
        assert snapshot_b.snapshot_id == snapshot_a.snapshot_id
        second = evaluator.evaluate_candidate(snapshot_b, ResourceAllocation.from_dwell_indices(list(CANDIDATE_EXACT), list(ACTION_SET)))
        assert state.fingerprint() == fingerprint_before
        assert first.status == "success" and second.status == "success"

    def test_candidate_evaluation_does_not_advance_simulation_time(self):
        evaluator = make_evaluator()
        state = make_state()
        snapshot = state.snapshot()
        t_before, idx_before = state.physical_time_s, state.revisit_index
        evaluator.evaluate_candidate(snapshot, ResourceAllocation.from_dwell_indices(list(CANDIDATE_PARTIAL), list(ACTION_SET)))
        assert state.physical_time_s == t_before
        assert state.revisit_index == idx_before
        assert snapshot.physical_time_s == t_before  # frozen time preserved

    def test_authoritative_state_not_mutated_by_evaluation(self):
        evaluator = make_evaluator()
        state = make_state()
        fingerprint_before = state.fingerprint()
        snapshot = state.snapshot()
        for indices in (CANDIDATE_PARTIAL, CANDIDATE_EXACT, CANDIDATE_OVER):
            evaluator.evaluate_candidate(snapshot, ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET)))
        assert state.fingerprint() == fingerprint_before


class TestOrderInvarianceAndReproducibility:
    def test_evaluation_order_does_not_change_results(self):
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        allocations = [ResourceAllocation.from_dwell_indices(list(i), list(ACTION_SET)) for i in
                       (CANDIDATE_PARTIAL, (2, 2, 2), CANDIDATE_EXACT, (1, 3, 5))]
        forward = [evaluator.evaluate_candidate(snapshot, a) for a in allocations]
        backward = [evaluator.evaluate_candidate(snapshot, a) for a in reversed(allocations)]
        for f, b in zip(forward, reversed(backward)):
            assert f.utility == pytest.approx(b.utility)
            assert f.per_task_tracking_costs == pytest.approx(b.per_task_tracking_costs)
            assert f.scanning_gamma == pytest.approx(b.scanning_gamma)
            assert f.resource_usage.total_dwell_time_s == pytest.approx(b.resource_usage.total_dwell_time_s)

    def test_repeat_evaluations_reproducible(self):
        evaluator = make_evaluator()
        snapshot = make_state().snapshot()
        allocation = ResourceAllocation.from_dwell_indices(list(CANDIDATE_PARTIAL), list(ACTION_SET))
        r1 = evaluator.evaluate_candidate(snapshot, allocation)
        r2 = evaluator.evaluate_candidate(snapshot, allocation)
        assert r1.utility == pytest.approx(r2.utility)
        assert r1.per_task_tracking_costs == r2.per_task_tracking_costs
        assert r1.scanning_gamma == pytest.approx(r2.scanning_gamma)
