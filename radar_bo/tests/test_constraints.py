"""Tests for the constraint manager and feasibility report (M4 instructions 9, 11)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.constraints import BUDGET_TOLERANCE, ConstraintManager

ACTION_SET = tuple(ResourceConfig().dwell_time_action_set)  # 0.1 .. 1.0
T0 = 2.5
N_TASKS = 3  # 3 tasks x 1.0 s = 3.0 s > T0: budget violation is reachable


def manager(semantics="hard_per_slot") -> ConstraintManager:
    return ConstraintManager(
        ResourceConfig(time_budget_semantics=semantics), num_tasks=N_TASKS, revisit_interval_s=T0
    )


def alloc(indices) -> ResourceAllocation:
    return ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET))


class TestConfigurationChecks:
    def test_valid_configuration_has_no_problems(self):
        assert manager().check_configuration() == []

    def test_invalid_configuration_reported_as_category(self):
        # model_construct bypasses construction validation to simulate a broken
        # configuration reaching the manager (the manager must catch it).
        broken = ConstraintManager(ResourceConfig.model_construct(dwell_time_action_set=[]), N_TASKS, T0)
        report = broken.check_allocation(alloc((0, 1, 2)))
        assert report.category == "invalid_configuration"
        assert not report.is_feasible

    def test_bad_manager_arguments_rejected(self):
        with pytest.raises(ValueError):
            ConstraintManager(ResourceConfig(), 0, T0)
        with pytest.raises(ValueError):
            ConstraintManager(ResourceConfig(), N_TASKS, -1.0)


def raw_allocation(dwell_times) -> ResourceAllocation:
    """Unvalidated allocation via model_construct — feeds candidates that would
    be rejected at construction through the manager's INDEPENDENT defensive
    checks (the manager must catch what other code paths might produce)."""
    return ResourceAllocation.model_construct(
        num_tasks=len(dwell_times),
        task_ids=tuple(f"task_{i}" for i in range(len(dwell_times))),
        dwell_indices=tuple(range(len(dwell_times))),
        dwell_times_s=tuple(dwell_times),
        action_set=ACTION_SET,
        continuous_power_w=None,
    )


class TestDimensionAndBounds:
    def test_task_count_mismatch_detected(self):
        report = manager().check_allocation(ResourceAllocation.from_dwell_indices([0, 1], list(ACTION_SET)))
        assert report.category == "infeasible_allocation"
        assert any(v.constraint == "task_count_consistency" for v in report.violations_only)

    def test_bounds_violation_detected(self):
        """5.0 s dwell: outside [0.1, 1.0] bounds — caught by the manager even
        though construction validation would already reject it."""
        report = manager().check_allocation(raw_allocation([0.1, 0.2, 5.0]))
        assert any(v.constraint == "dwell_bounds" for v in report.violations_only)

    def test_membership_violation_detected(self):
        """0.55 s dwell: not an action-set member — caught by the manager."""
        report = manager().check_allocation(raw_allocation([0.1, 0.2, 0.55]))
        assert any(v.constraint == "dwell_discrete_membership" for v in report.violations_only)

    def test_power_bounds_checked_when_resolved(self):
        rc = ResourceConfig(min_transmit_power=1.0, max_transmit_power=100.0)
        mgr = ConstraintManager(rc, N_TASKS, T0)
        good = ResourceAllocation.from_dwell_indices([0, 1, 2], list(ACTION_SET), continuous_power_w=(1.0, 50.0, 100.0))
        assert mgr.check_allocation(good).is_feasible
        bad = ResourceAllocation.from_dwell_indices([0, 1, 2], list(ACTION_SET), continuous_power_w=(1.0, 500.0, 100.0))
        report = mgr.check_allocation(bad)
        assert any(v.constraint == "power_bounds" for v in report.violations_only)

    def test_power_bounds_unresolved_is_a_parameter_error_not_infeasible(self):
        """Powers present but bounds None: 'unresolved_parameter', not 'infeasible_allocation'."""
        mgr = ConstraintManager(ResourceConfig(), N_TASKS, T0)  # bounds unresolved
        report = mgr.check_allocation(
            ResourceAllocation.from_dwell_indices([0, 1, 2], list(ACTION_SET), continuous_power_w=(1.0, 2.0, 3.0))
        )
        assert report.category == "unresolved_parameter"
        assert not report.is_feasible
        assert any(v.constraint == "power_bounds_resolvable" for v in report.violations_only)

    def test_power_free_allocation_unaffected_by_unresolved_bounds(self):
        """No power block: unresolved power bounds must NOT block feasibility."""
        assert manager().check_allocation(alloc((0, 1, 2))).is_feasible


class TestBudgetSemantics:
    def test_zero_consumption_feasible_and_remaining_full(self):
        report = manager().check_allocation(alloc((0, 0, 0)))  # 0.3 s total
        assert report.is_feasible
        assert report.slot_dwell_sum_s == pytest.approx(0.3)
        assert report.normalized_time_usage == pytest.approx(0.3 / T0)

    def test_partial_consumption_feasible(self):
        assert manager().check_allocation(alloc((0, 2, 4))).is_feasible  # 0.9 s

    def test_exact_budget_usage_is_feasible(self):
        """sum(tau) == Theta_max * T0 exactly: satisfied (<=), boundary behavior.
        Indices chosen so dwell times are (0.5, 1.0, 1.0) summing to 2.5 s."""
        exact = manager().check_allocation(ResourceAllocation.from_dwell_indices([4, 9, 9], list(ACTION_SET)))
        assert exact.slot_dwell_sum_s == pytest.approx(2.5)
        assert exact.normalized_time_usage == pytest.approx(1.0)
        assert exact.is_feasible  # <= Theta_max (equality allowed)

    def test_budget_exceedance_infeasible_under_hard_semantics(self):
        report = manager("hard_per_slot").check_allocation(alloc((9, 9, 9)))  # 3.0 s > 2.5 s
        assert not report.is_feasible
        assert report.category == "infeasible_allocation"
        assert any(v.constraint == "hard_per_slot_budget" for v in report.violations_only)
        assert report.normalized_time_usage == pytest.approx(1.2)

    def test_float_boundary_absorbed_by_tolerance(self):
        """Members (0.9, 0.9, 0.7) sum to mathematically exactly 2.5 s, but the
        float sum is 2.5000000000000004 (normalized 1.0000000000000002). The
        budget tolerance (1e-12) must absorb this: feasible, not violated."""
        report = manager("hard_per_slot").check_allocation(
            ResourceAllocation.from_dwell_indices([8, 8, 6], list(ACTION_SET))
        )
        assert report.slot_dwell_sum_s >= 2.5  # float sum at/above the boundary
        assert report.is_feasible

    def test_budget_exceeded_by_member_combination_is_violated(self):
        """(0.9, 0.9, 0.8) = 2.6 s: a genuine member-combination exceedance."""
        report = manager("hard_per_slot").check_allocation(
            ResourceAllocation.from_dwell_indices([8, 8, 7], list(ACTION_SET))
        )
        assert not report.is_feasible
        assert any(v.constraint == "hard_per_slot_budget" for v in report.violations_only)

    def test_budget_just_below_by_member_combination_is_feasible(self):
        """(0.9, 0.9, 0.6) = 2.4 s: just below the 2.5 s budget."""
        report = manager("hard_per_slot").check_allocation(
            ResourceAllocation.from_dwell_indices([8, 8, 5], list(ACTION_SET))
        )
        assert report.is_feasible
        assert report.normalized_time_usage == pytest.approx(0.96)

    def test_paper_semantics_records_slot_term_without_hard_infeasibility(self):
        """Discounted cumulative: single-slot exceedance recorded as a note, NOT infeasible."""
        report = manager("discounted_cumulative").check_allocation(alloc((9, 9, 9)))
        assert report.is_feasible  # paper-reference profile: horizon judgment out of M4 scope
        assert report.budget_semantics == "discounted_cumulative"
        terms = [v for v in report.violations if v.constraint == "discounted_cumulative_slot_term"]
        assert len(terms) == 1
        assert terms[0].severity == "note"
        assert terms[0].category == "valid"
        lowered = terms[0].detail.lower()
        assert "not" in lowered and "equivalent" in lowered

    def test_within_budget_under_paper_semantics_also_recorded(self):
        report = manager("discounted_cumulative").check_allocation(alloc((0, 1, 2)))
        assert report.is_feasible
        assert report.slot_dwell_sum_s == pytest.approx(0.6)

    def test_semantics_are_not_treated_as_equivalent(self):
        """Same allocation: hard semantics infeasible, paper semantics feasible."""
        over = alloc((9, 9, 9))
        assert not manager("hard_per_slot").check_allocation(over).is_feasible
        assert manager("discounted_cumulative").check_allocation(over).is_feasible


class TestReportStructure:
    def test_report_lists_specific_violations(self):
        report = manager().check_allocation(alloc((9, 9, 9)))
        assert report.violations_only
        assert all(v.detail for v in report.violations)
        assert "hard_per_slot_budget" in report.summary

    def test_report_carries_usage_quantities(self):
        report = manager().check_allocation(alloc((2, 2, 2)))
        assert report.slot_dwell_sum_s == pytest.approx(0.9)
        assert report.normalized_time_usage == pytest.approx(0.36)
        assert report.budget_threshold == 1.0

    def test_no_mutation_of_candidate(self):
        allocation = alloc((1, 3, 5))
        before = allocation.to_flat_vector().copy()
        manager().check_allocation(allocation)
        np.testing.assert_array_equal(allocation.to_flat_vector(), before)


class TestPolicyLevelFeasibility:
    """M4 review 3.1: local (per-action) feasibility distinguished from
    policy-level feasibility of the paper's cumulative discounted budget."""

    def test_hard_semantics_determines_feasibility_locally(self):
        report = manager("hard_per_slot").check_allocation(alloc((9, 9, 9)))
        assert report.policy_level_feasibility == "determined_locally"
        assert not report.is_feasible

    def test_paper_semantics_requires_horizon_for_policy_level(self):
        report = manager("discounted_cumulative").check_allocation(alloc((9, 9, 9)))
        # Locally over-budget, but NOT proof of discounted-budget infeasibility:
        assert report.is_feasible
        assert report.policy_level_feasibility == "not_assessed_horizon_required"
        detail = next(v for v in report.violations if v.constraint == "discounted_cumulative_slot_term")
        assert "does NOT establish policy-level" in detail.detail

    def test_within_budget_action_does_not_prove_discounted_feasibility(self):
        report = manager("discounted_cumulative").check_allocation(alloc((0, 1, 2)))
        assert report.is_feasible
        assert report.policy_level_feasibility == "not_assessed_horizon_required"
