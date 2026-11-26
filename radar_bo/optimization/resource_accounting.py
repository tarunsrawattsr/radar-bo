"""Generic resource accounting (M4 instruction 10).

Computes and records resource usage for a candidate allocation under the
DECLARED configuration conventions:

* Time accounting: total dwell, normalized usage ``sum(tau)/T0``, the paper's
  Eq. (24) remaining scanning time ``tau_s = T0 * max(0, 1 - sum(tau)/T0)``
  (the paper's own nonnegative convention), and the RAW time deficit when the
  budget is exceeded. The paper's clipped remaining time never makes an
  infeasible allocation valid: feasibility is judged separately by
  :mod:`radar_bo.optimization.constraints`, which sees the raw usage.
* Optional abstract energy accounting: ``sum(tau_n * p_n)`` in joules, ONLY
  under the declared ``average`` power convention (energy from peak pulse
  power is not computed — that would require waveform assumptions the
  project does not have). Peak-power, average-power and duty-cycle
  conventions are explicitly separated on the accountant.
* No hardware-dependent power calculations are implemented. Any operation
  that REQUIRES an unresolved configuration quantity (e.g. comparing against
  an energy budget that is still ``None``) fails clearly with
  :class:`UnresolvedParameterError`.

Synthetic integration tests use explicitly identified synthetic quantities
and budgets (labeled in notes).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from radar_bo.config.base import StrictModel, UnresolvedParameterError
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation

PowerConvention = Literal["average", "peak_pulse", "duty_cycle_weighted"]


class ResourceUsage(StrictModel):
    """Recorded resource usage for one candidate allocation."""

    total_dwell_time_s: float = Field(description="Sum of per-task dwell times (s).")
    per_task_dwell_times_s: list[float] = Field(description="Ordered per-task dwell times (s).")
    revisit_interval_s: float = Field(description="T0 used for normalization (single authoritative source).")
    normalized_time_usage: float = Field(description="sum(tau) / T0.")
    remaining_scanning_time_s_paper_eq24: float = Field(
        description="Eq. (24): tau_s = T0 * max(0, 1 - sum(tau)/T0) — the paper's nonnegative convention."
    )
    raw_time_deficit_s: float = Field(
        description="max(0, sum(tau) - Theta_max * T0); > 0 only when the slot budget is exceeded. "
        "Recorded so an over-budget allocation is never silently made valid by the Eq. (24) clipping."
    )
    slot_budget_term_discounted: float = Field(
        description="sum(tau)/T0 - Theta_max: the per-slot term of the paper's Eq. (22) cumulative budget."
    )
    total_power_w: float | None = Field(default=None, description="Convention-dependent power sum, if a power block is present.")
    total_energy_j: float | None = Field(
        default=None,
        description="sum(tau_n * p_n) in J; computed ONLY under the 'average' power convention.",
    )
    power_convention: PowerConvention | None = Field(default=None)
    notes: list[str] = Field(default_factory=list, description="Labels, e.g. synthetic-quantity markers.")


class ResourceAccountant:
    """Shared resource accounting for benchmarks and optimization methods (M4 instruction 10)."""

    def __init__(
        self,
        resource_config: ResourceConfig,
        revisit_interval_s: float,
        power_convention: PowerConvention = "average",
    ) -> None:
        if not np.isfinite(revisit_interval_s) or revisit_interval_s <= 0.0:
            raise ValueError(f"revisit_interval_s (T0) must be finite and > 0; got {revisit_interval_s!r}.")
        self.resource_config = resource_config
        self.revisit_interval_s = float(revisit_interval_s)
        self.power_convention: PowerConvention = power_convention

    def account(self, allocation: ResourceAllocation) -> ResourceUsage:
        """Compute usage; validates quantities, never repairs them."""
        rc = self.resource_config
        dwells = np.asarray(allocation.dwell_times_s, dtype=np.float64)
        if not np.all(np.isfinite(dwells)) or np.any(dwells < 0.0):
            raise ValueError(f"Dwell times must be finite and non-negative; got {allocation.dwell_times_s!r}.")
        total_dwell = float(np.sum(dwells))
        normalized = total_dwell / self.revisit_interval_s
        theta = rc.normalized_time_budget_threshold

        eq24_remaining = self.revisit_interval_s * max(0.0, 1.0 - normalized)  # paper Eq. (24), clipped by the paper
        raw_deficit = max(0.0, total_dwell - theta * self.revisit_interval_s)  # raw exceedance, never hidden
        slot_term = normalized - theta  # paper Eq. (22) per-slot term

        total_power: float | None = None
        total_energy: float | None = None
        notes: list[str] = []
        if allocation.continuous_power_w is not None:
            powers = np.asarray(allocation.continuous_power_w, dtype=np.float64)
            if not np.all(np.isfinite(powers)) or np.any(powers < 0.0):
                raise ValueError(f"Powers must be finite and non-negative; got {allocation.continuous_power_w!r}.")
            total_power = float(np.sum(powers))
            if self.power_convention == "average":
                # Energy during the revisit: only the average-power convention
                # supports sum(tau * p); peak pulse power would need waveform
                # assumptions that are unresolved (README section 14).
                total_energy = float(np.sum(dwells * powers))
                notes.append("Energy computed as sum(tau_n * p_n) under the declared 'average' power convention.")
            elif self.power_convention == "peak_pulse":
                notes.append(
                    "Power block carries PEAK PULSE power per the declared convention; energy accounting is "
                    "NOT computed (would require unresolved waveform/duty assumptions)."
                )
            else:  # duty_cycle_weighted
                notes.append(
                    "Power block carries duty-cycle-weighted power per the declared convention; energy "
                    "accounting is NOT computed in M4 (convention documented, no hardware model)."
                )
            notes.append(
                "Power quantities are abstract synthetic variables; no hardware-dependent calculation is performed."
            )

        return ResourceUsage(
            total_dwell_time_s=total_dwell,
            per_task_dwell_times_s=[float(t) for t in dwells],
            revisit_interval_s=self.revisit_interval_s,
            normalized_time_usage=normalized,
            remaining_scanning_time_s_paper_eq24=eq24_remaining,
            raw_time_deficit_s=raw_deficit,
            slot_budget_term_discounted=slot_term,
            total_power_w=total_power,
            total_energy_j=total_energy,
            power_convention=self.power_convention if allocation.continuous_power_w is not None else None,
            notes=notes,
        )

    def check_energy_budget(self, usage: ResourceUsage) -> bool:
        """Compare recorded energy against the configured budget.

        Raises :class:`UnresolvedParameterError` when ``resource.energy_budget``
        is ``None`` (no invented defaults). Returns True when within budget.
        """
        if usage.total_energy_j is None:
            raise ValueError("check_energy_budget requires a usage record with computed energy (average power convention).")
        budget = self.resource_config.energy_budget
        if budget is None:
            raise UnresolvedParameterError(
                {"resource.energy_budget": "J"},
                guidance="No paper or proposed default exists; supply a justified synthetic budget for testing.",
            )
        return bool(usage.total_energy_j <= budget)
