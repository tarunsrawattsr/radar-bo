"""Tests for resource accounting (M4 instructions 10 and 11)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.resource_accounting import ResourceAccountant

ACTION_SET = tuple(ResourceConfig().dwell_time_action_set)
T0 = 2.5
N_TASKS = 3


def accountant(power_convention="average") -> ResourceAccountant:
    return ResourceAccountant(ResourceConfig(), revisit_interval_s=T0, power_convention=power_convention)


def alloc(indices, powers=None) -> ResourceAllocation:
    return ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET), powers)


class TestTimeAccounting:
    def test_zero_consumption(self):
        """All-minimum dwell: usage 0.3 s of 2.5 s; Eq. (24) remaining 2.2 s."""
        usage = accountant().account(alloc((0, 0, 0)))
        assert usage.total_dwell_time_s == pytest.approx(0.3)
        assert usage.normalized_time_usage == pytest.approx(0.3 / T0)
        assert usage.remaining_scanning_time_s_paper_eq24 == pytest.approx(T0 * (1 - 0.3 / T0)) == pytest.approx(2.2)
        assert usage.raw_time_deficit_s == 0.0

    def test_partial_consumption_hand_computed(self):
        usage = accountant().account(alloc((0, 2, 4)))  # 0.1 + 0.3 + 0.5 = 0.9
        assert usage.total_dwell_time_s == pytest.approx(0.9)
        assert usage.normalized_time_usage == pytest.approx(0.36)
        # Eq. (24): tau_s = T0 * max(0, 1 - 0.36) = 2.5 * 0.64 = 1.6
        assert usage.remaining_scanning_time_s_paper_eq24 == pytest.approx(1.6)
        assert usage.slot_budget_term_discounted == pytest.approx(0.36 - 1.0)

    def test_exact_budget_usage(self):
        usage = accountant().account(ResourceAllocation.from_dwell_indices([4, 9, 9], list(ACTION_SET)))  # 0.5+1+1=2.5
        assert usage.normalized_time_usage == pytest.approx(1.0)
        assert usage.remaining_scanning_time_s_paper_eq24 == pytest.approx(0.0)
        assert usage.raw_time_deficit_s == pytest.approx(0.0, abs=1e-12)
        assert usage.slot_budget_term_discounted == pytest.approx(0.0, abs=1e-12)

    def test_budget_exceedance_records_both_eq24_and_raw_deficit(self):
        """Paper Eq. (24) clips to zero; the RAW deficit is recorded so the
        exceedance is never hidden by the paper's nonnegative convention."""
        usage = accountant().account(alloc((9, 9, 9)))  # 3.0 s > 2.5 s
        assert usage.remaining_scanning_time_s_paper_eq24 == pytest.approx(0.0)  # paper convention
        assert usage.raw_time_deficit_s == pytest.approx(0.5)  # raw exceedance preserved
        assert usage.slot_budget_term_discounted == pytest.approx(0.2)
        assert usage.normalized_time_usage == pytest.approx(1.2)

    def test_invalid_negative_quantity_rejected(self):
        """The accountant independently rechecks quantities even for allocations
        built through paths that bypass construction validation."""
        bad = ResourceAllocation.model_construct(
            num_tasks=3, task_ids=("task_0", "task_1", "task_2"),
            dwell_indices=(0, 0, 0), dwell_times_s=(0.1, -0.2, 0.1),
            action_set=ACTION_SET, continuous_power_w=None,
        )
        with pytest.raises(ValueError, match="non-negative"):
            accountant().account(bad)
        nan_alloc = ResourceAllocation.model_construct(
            num_tasks=1, task_ids=("task_0",), dwell_indices=(0,), dwell_times_s=(np.nan,),
            action_set=ACTION_SET[:1], continuous_power_w=None,
        )
        with pytest.raises(ValueError):
            accountant().account(nan_alloc)

    def test_invalid_accountant_arguments_rejected(self):
        with pytest.raises(ValueError):
            ResourceAccountant(ResourceConfig(), revisit_interval_s=0.0)
        with pytest.raises(ValueError):
            ResourceAccountant(ResourceConfig(), revisit_interval_s=np.nan)


class TestEnergyAndPowerConventions:
    def test_energy_computed_under_average_convention(self):
        """sum(tau_n * p_n): 0.1*10 + 0.3*20 + 0.5*30 = 22.0 J (hand-computed)."""
        usage = accountant("average").account(alloc((0, 2, 4), powers=(10.0, 20.0, 30.0)))
        assert usage.total_energy_j == pytest.approx(22.0)
        assert usage.total_power_w == pytest.approx(60.0)
        assert usage.power_convention == "average"

    def test_energy_not_computed_under_peak_convention(self):
        usage = accountant("peak_pulse").account(alloc((0, 2, 4), powers=(10.0, 20.0, 30.0)))
        assert usage.total_energy_j is None
        assert usage.total_power_w == pytest.approx(60.0)
        assert any("PEAK PULSE" in n for n in usage.notes)

    def test_duty_cycle_convention_recorded(self):
        usage = accountant("duty_cycle_weighted").account(alloc((0, 0, 0), powers=(1.0, 1.0, 1.0)))
        assert usage.total_energy_j is None
        assert usage.power_convention == "duty_cycle_weighted"

    def test_energy_budget_unresolved_fails_clearly(self):
        usage = accountant().account(alloc((0, 2, 4), powers=(10.0, 20.0, 30.0)))
        with pytest.raises(UnresolvedParameterError) as excinfo:
            accountant().check_energy_budget(usage)
        assert "resource.energy_budget" in str(excinfo.value)

    def test_energy_budget_check_with_synthetic_budget(self):
        """Explicitly identified synthetic budget for testing."""
        rc = ResourceConfig(energy_budget=25.0)
        acct = ResourceAccountant(rc, revisit_interval_s=T0)
        usage = acct.account(alloc((0, 2, 4), powers=(10.0, 20.0, 30.0)))  # 22 J
        assert acct.check_energy_budget(usage) is True
        usage_over = acct.account(alloc((8, 8, 8), powers=(10.0, 20.0, 30.0)))  # 0.9*90 = 81 J
        assert acct.check_energy_budget(usage_over) is False

    def test_energy_check_requires_energy_in_usage(self):
        usage = accountant().account(alloc((0, 0, 0)))  # no power block
        with pytest.raises(ValueError, match="energy"):
            accountant().check_energy_budget(usage)

    def test_no_power_block_leaves_power_fields_none(self):
        usage = accountant().account(alloc((0, 1, 2)))
        assert usage.total_power_w is None
        assert usage.total_energy_j is None
        assert usage.power_convention is None
