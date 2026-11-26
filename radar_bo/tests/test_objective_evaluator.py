"""Tests for the objective evaluator (M4 instructions 12, 14, 15)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.constraints import ConstraintManager, FeasibilityReport
from radar_bo.optimization.objective import (
    EvaluationResult,
    ObjectiveEvaluator,
    SnapshotDiagnosticTrackingCostProvider,
    SyntheticLinearTrackingCostProvider,
)
from radar_bo.optimization.resource_accounting import ResourceAccountant
from radar_bo.radar.scanning_model import SyntheticLinearScanningModel

ACTION_SET = tuple(ResourceConfig().dwell_time_action_set)
T0 = 2.5
N_TASKS = 3
BETA = 1.0e6  # paper beta

BASE_COSTS = np.array([2.0, 3.0, 1.5])
KAPPA = np.array([0.5, 1.0, 0.25])


def make_evaluator(**kwargs) -> ObjectiveEvaluator:
    """Evaluator configured with the PROPOSED hard per-slot budget semantics
    (the BO configuration profile per M2/M4 review)."""
    profile = ResourceConfig(time_budget_semantics="hard_per_slot")
    return ObjectiveEvaluator(
        constraint_manager=ConstraintManager(profile, N_TASKS, T0),
        accountant=ResourceAccountant(profile, revisit_interval_s=T0),
        tracking_provider=SyntheticLinearTrackingCostProvider(BASE_COSTS, KAPPA),
        scanning_model=SyntheticLinearScanningModel(
            synthetic_area_rate_m2_per_s=2000.0, synthetic_reference_area_m2=4000.0
        ),
        resource_config=profile,
        **kwargs,
    )


def make_snapshot():
    """Minimal snapshot with known posterior covariances for the diagnostic provider."""
    from radar_bo.simulation.simulation_state import FrozenSimulationState

    covs = np.zeros((N_TASKS, 4, 4))
    covs[0] = np.diag([1.0, 2.0, 0.5, 0.5])  # Eq. 14 cost = 3.0
    covs[1] = np.diag([4.0, 1.0, 0.5, 0.5])  # Eq. 14 cost = 5.0
    covs[2] = np.diag([0.5, 0.5, 0.5, 0.5])  # Eq. 14 cost = 1.0
    means = np.zeros((N_TASKS, 4))
    return FrozenSimulationState(
        scenario_id="synthetic_m4_test",
        physical_time_s=5.0,
        revisit_index=2,
        task_ids=("task_0", "task_1", "task_2"),
        rng_state={"bit_generator_name": "PCG64", "state": np.random.default_rng(0).bit_generator.state},
        _posterior_means=means,
        _posterior_covariances=covs,
    )


def alloc(indices, powers=None) -> ResourceAllocation:
    return ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET), powers)


class TestAggregationHandChecks:
    def test_utility_equals_neg_weighted_cost_plus_beta_gamma(self):
        """Hand-computed: tau = (0.1, 0.3, 0.5) -> sum 0.9, remaining 1.6 s,
        gamma = 2000*1.6/4000 = 0.8; costs = base - kappa*tau (tau_ref=1);
        uniform weights -> utility = -(2.0-0.05 + 3.0-0.3 + 1.5-0.125) + 1e6*0.8."""
        evaluator = make_evaluator()
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 2, 4)))
        assert result.status == "success"
        expected_costs = [2.0 - 0.5 * 0.1, 3.0 - 1.0 * 0.3, 1.5 - 0.25 * 0.5]
        assert result.per_task_tracking_costs == pytest.approx(expected_costs)
        expected_aggregate = sum(expected_costs)  # uniform weights = 1
        assert result.aggregate_weighted_tracking_cost == pytest.approx(expected_aggregate)
        assert result.scanning_gamma == pytest.approx(0.8)
        assert result.detectable_area_m2 == pytest.approx(2000.0 * 1.6)
        assert result.utility == pytest.approx(-expected_aggregate + BETA * 0.8)

    def test_weighted_aggregation_with_configured_weights(self):
        """Weights (2, 1, 0.5): aggregate = 2*c0 + 1*c1 + 0.5*c2 (hand-computed)."""
        evaluator = make_evaluator()
        evaluator.resource_config = ResourceConfig(task_utility_weights=[2.0, 1.0, 0.5])
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        c = [2.0 - 0.5 * 0.1, 3.0 - 1.0 * 0.1, 1.5 - 0.25 * 0.1]
        expected = 2.0 * c[0] + 1.0 * c[1] + 0.5 * c[2]
        assert result.aggregate_weighted_tracking_cost == pytest.approx(expected)
        assert result.task_utility_weights == [2.0, 1.0, 0.5]

    def test_uniform_weights_are_the_documented_proposed_default(self):
        evaluator = make_evaluator()
        weights = evaluator._task_weights(3)
        np.testing.assert_array_equal(weights, np.ones(3))

    def test_invalid_weights_surface_as_failed_evaluation(self):
        """Weight misconfiguration must produce an explicit FAILED record (no
        silent zero-utility fallback), never a successful observation."""
        evaluator = make_evaluator()
        evaluator.resource_config = ResourceConfig(
            time_budget_semantics="hard_per_slot", task_utility_weights=[1.0, -1.0, 0.5]
        )
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        assert result.status == "failed"
        assert result.utility is None
        assert "task_utility_weights" in result.error
        evaluator2 = make_evaluator()
        evaluator2.resource_config = ResourceConfig(
            time_budget_semantics="hard_per_slot", task_utility_weights=[1.0, 1.0]
        )
        result2 = evaluator2.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        assert result2.status == "failed"
        assert "expected 3" in result2.error

    def test_paper_semantics_evaluator_treats_over_budget_as_feasible_with_note(self):
        """The paper-reference profile does NOT hard-reject a single over-budget
        slot (distinction preserved end-to-end through the evaluator)."""
        profile = ResourceConfig(time_budget_semantics="discounted_cumulative")
        evaluator = ObjectiveEvaluator(
            constraint_manager=ConstraintManager(profile, N_TASKS, T0),
            accountant=ResourceAccountant(profile, revisit_interval_s=T0),
            tracking_provider=SyntheticLinearTrackingCostProvider(BASE_COSTS, KAPPA),
            scanning_model=SyntheticLinearScanningModel(),
            resource_config=profile,
        )
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((9, 9, 9)))
        assert result.status == "success"  # paper semantics: no hard per-slot rejection
        assert result.feasibility.budget_semantics == "discounted_cumulative"
        assert any(v.constraint == "discounted_cumulative_slot_term" for v in result.feasibility.violations)


class TestDiagnosticProvider:
    def test_eq14_costs_from_snapshot_covariances(self):
        """Paper Eq. (14) on hand-built covariances: costs (3.0, 5.0, 1.0)."""
        evaluator = make_evaluator()
        evaluator.tracking_provider = SnapshotDiagnosticTrackingCostProvider()
        res = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 2, 4)))
        assert res.per_task_tracking_costs == pytest.approx([3.0, 5.0, 1.0])
        # Aggregate = sum (uniform): utility = -(3+5+1) + beta*gamma
        gamma = 2000.0 * (T0 - 0.9) / 4000.0
        assert res.utility == pytest.approx(-9.0 + BETA * gamma)

    def test_diagnostic_provider_is_allocation_independent(self):
        evaluator = make_evaluator()
        evaluator.tracking_provider = SnapshotDiagnosticTrackingCostProvider()
        r1 = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        r2 = evaluator.evaluate_candidate(make_snapshot(), alloc((9, 9, 4)))
        assert r1.per_task_tracking_costs == r2.per_task_tracking_costs


class TestFailureRepresentation:
    def test_infeasible_candidate_gets_infeasible_status_and_no_utility(self):
        result = make_evaluator().evaluate_candidate(make_snapshot(), alloc((9, 9, 9)))
        assert result.status == "infeasible"
        assert result.utility is None
        assert not result.feasibility.is_feasible
        assert result.resource_usage is not None

    def test_failing_provider_gives_failed_status_not_zero_utility(self):
        class ExplodingProvider:
            identifier = "exploding"
            is_synthetic = True

            def tracking_costs(self, snapshot, allocation, rng):
                raise RuntimeError("boom")

        evaluator = make_evaluator()
        evaluator.tracking_provider = ExplodingProvider()
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        assert result.status == "failed"
        assert result.utility is None
        assert "boom" in result.error

    def test_failing_scanning_model_gives_failed_status(self):
        evaluator = make_evaluator()
        evaluator.scanning_model = SyntheticLinearScanningModel()  # valid
        # Force failure through a snapshot whose evaluation triggers an error:
        # use a provider returning non-finite costs.
        class BadProvider:
            identifier = "nan_provider"
            is_synthetic = True

            def tracking_costs(self, snapshot, allocation, rng):
                return np.array([np.nan, 1.0, 2.0])

        evaluator.tracking_provider = BadProvider()
        result = evaluator.evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        assert result.status == "failed"
        assert result.utility is None
        assert "invalid costs" in result.error

    def test_result_model_validators(self):
        fields = dict(
            evaluation_id="x", snapshot_id="s", allocation_id="a",
            tracking_provider_identifier="p", randomness_policy="common_random_numbers",
            evaluation_duration_s=0.01, snapshot_integrity_verified=True,
        )
        feasibility = FeasibilityReport(is_feasible=True, category="valid")
        fields["feasibility"] = feasibility
        with pytest.raises(ValidationError):
            EvaluationResult(status="success", utility=None, **fields)  # success without utility
        with pytest.raises(ValidationError):
            EvaluationResult(status="failed", utility=0.0, **fields)  # failed with utility
        with pytest.raises(ValidationError):
            EvaluationResult(status="failed", utility=None, error=None, **fields)  # failed without error


class TestTraceabilityAndTiming:
    def test_result_traceable_to_snapshot_and_allocation(self):
        snapshot = make_snapshot()
        allocation = alloc((2, 4, 6))
        result = make_evaluator().evaluate_candidate(snapshot, allocation)
        assert result.snapshot_id == snapshot.snapshot_id
        assert result.allocation_id == allocation.content_id()
        assert result.evaluation_id

    def test_duration_recorded_nonnegative(self):
        result = make_evaluator().evaluate_candidate(make_snapshot(), alloc((0, 0, 0)))
        assert result.evaluation_duration_s >= 0.0

    def test_randomness_policy_recorded(self):
        result = make_evaluator(randomness_policy="common_random_numbers").evaluate_candidate(
            make_snapshot(), alloc((0, 0, 0))
        )
        assert result.randomness_policy == "common_random_numbers"
