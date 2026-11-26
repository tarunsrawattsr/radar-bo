"""Feasibility validation and constraint management (M4 instruction 9).

The :class:`ConstraintManager` validates candidate allocations against the
configured resource rules and returns a structured
:class:`FeasibilityReport` listing every violated constraint. It
distinguishes outcome categories:

* ``valid``                 — allocation admissible under the configured semantics;
* ``invalid_configuration`` — the manager's own configuration is defective (not the candidate's fault);
* ``infeasible_allocation`` — the candidate violates an enforced constraint;
* ``unresolved_parameter``  — a required check depends on a configuration value still ``None``;
* ``numerical_failure``     — a check could not be computed numerically.

Budget semantics (never treated as equivalent; M4 instruction 9, M4 review 3.1):

* ``hard_per_slot`` (PROPOSED, BO configuration profile): the per-slot
  normalized dwell sum ``sum(tau) / T0 <= Theta_max`` is ENFORCED — a
  violating allocation is reported infeasible. Under this PROPOSED semantics,
  per-slot checking fully determines feasibility
  (``policy_level_feasibility = "determined_locally"``).
* ``discounted_cumulative`` (PAPER-REFERENCE profile, Lu et al. Eq. (22)):
  the per-slot term ``sum(tau)/T0 - Theta_max`` of the discounted cumulative
  budget is RECORDED as informational. LOCAL feasibility of an individual
  action neither establishes nor refutes POLICY-LEVEL satisfaction of the
  paper's cumulative discounted budget: that judgment spans a discounted
  horizon and is outside M4/M5 scope
  (``policy_level_feasibility = "not_assessed_horizon_required"``). An
  over-budget individual action is NOT proof of discounted-budget
  infeasibility, and a within-budget action is NOT proof of
  discounted-budget feasibility.

The revisit interval ``T0`` is received from the composed SimulationConfig
(single authoritative source: TrackingConfig). Invalid allocations are never
clipped, rounded, repaired or replaced.

M5 synthetic optimization problems use their own explicitly defined
instantaneous mathematical constraints (see
``radar_bo.optimization.bo.benchmarks``); they do not reuse or reinterpret
the radar budget semantics above.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation

ViolationCategory = Literal[
    "valid", "invalid_configuration", "infeasible_allocation", "unresolved_parameter", "numerical_failure"
]
ViolationSeverity = Literal["violation", "note"]

#: Exact-equality tolerance for the budget boundary (instruction 11: floating-point
#: boundary behavior is part of the tested contract).
BUDGET_TOLERANCE: float = 1e-12


class ViolationDetail(StrictModel):
    """One specific violated constraint (or informational note)."""

    constraint: str = Field(description="Stable constraint name, e.g. 'hard_per_slot_budget'.")
    severity: ViolationSeverity = Field(description="'violation' affects feasibility; 'note' does not.")
    category: ViolationCategory = Field(description="Outcome category this entry belongs to.")
    detail: str = Field(description="Human-readable specifics, including the offending numbers.")


class FeasibilityReport(StrictModel):
    """Structured feasibility report for one candidate allocation."""

    is_feasible: bool = Field(description="True iff no violation-severity entries are present.")
    category: ViolationCategory = Field(description="Overall outcome category.")
    violations: list[ViolationDetail] = Field(default_factory=list)
    summary: str = Field(default="")
    slot_dwell_sum_s: float | None = Field(default=None, description="Sum of per-task dwell times (s).")
    normalized_time_usage: float | None = Field(default=None, description="Sum of dwell times divided by T0.")
    budget_threshold: float | None = Field(default=None, description="Theta_max from the resource configuration.")
    budget_semantics: str | None = Field(default=None, description="Configured budget semantics.")
    policy_level_feasibility: Literal[
        # PROPOSED hard semantics: the per-slot check fully determines feasibility.
        "determined_locally",
        # Paper-reference semantics: the cumulative discounted budget spans a
        # horizon; local checks neither prove nor refute policy-level feasibility.
        "not_assessed_horizon_required",
    ] = Field(
        default="not_assessed_horizon_required",
        description=(
            "M4 review 3.1: distinguishes LOCAL (per-action) feasibility from POLICY-LEVEL "
            "feasibility of the paper's cumulative discounted budget. Local within-budget "
            "actions do not establish discounted-budget satisfaction; local over-budget "
            "actions do not by themselves establish discounted-budget infeasibility."
        ),
    )

    @property
    def violations_only(self) -> list[ViolationDetail]:
        return [v for v in self.violations if v.severity == "violation"]


class ConstraintManager:
    """Reusable feasibility validation for :class:`ResourceAllocation` candidates."""

    def __init__(self, resource_config: ResourceConfig, num_tasks: int, revisit_interval_s: float) -> None:
        if int(num_tasks) < 1:
            raise ValueError(f"num_tasks must be >= 1; got {num_tasks}.")
        if not np.isfinite(revisit_interval_s) or revisit_interval_s <= 0.0:
            raise ValueError(f"revisit_interval_s (T0) must be finite and > 0; got {revisit_interval_s!r}.")
        self.resource_config = resource_config
        self.num_tasks = int(num_tasks)
        self.revisit_interval_s = float(revisit_interval_s)

    # ---- Configuration self-check ------------------------------------------------
    def check_configuration(self) -> list[ViolationDetail]:
        """Validate the manager's own configuration (invalid_configuration category)."""
        problems: list[ViolationDetail] = []
        rc = self.resource_config
        if not rc.dwell_time_action_set:
            problems.append(
                ViolationDetail(
                    constraint="action_set_nonempty", severity="violation",
                    category="invalid_configuration", detail="dwell_time_action_set is empty.",
                )
            )
        if rc.normalized_time_budget_threshold <= 0.0:
            problems.append(
                ViolationDetail(
                    constraint="budget_threshold_positive", severity="violation",
                    category="invalid_configuration",
                    detail=f"normalized_time_budget_threshold must be > 0; got {rc.normalized_time_budget_threshold}.",
                )
            )
        return problems

    # ---- Allocation checks ---------------------------------------------------------
    def check_allocation(self, allocation: ResourceAllocation) -> FeasibilityReport:
        """Full feasibility check; never mutates or repairs the candidate."""
        config_problems = self.check_configuration()
        if config_problems:
            return FeasibilityReport(
                is_feasible=False,
                category="invalid_configuration",
                violations=config_problems,
                summary="Constraint manager configuration is invalid; no allocation can be judged.",
            )

        violations: list[ViolationDetail] = []
        rc = self.resource_config

        # Dimension consistency with the configured task count.
        if allocation.num_tasks != self.num_tasks:
            violations.append(
                ViolationDetail(
                    constraint="task_count_consistency", severity="violation",
                    category="infeasible_allocation",
                    detail=f"Allocation has {allocation.num_tasks} tasks but the manager is configured "
                    f"for {self.num_tasks}.",
                )
            )

        # Individual variable bounds and discrete membership (dwell block).
        lo, hi = min(rc.dwell_time_action_set), max(rc.dwell_time_action_set)
        for i, tau in enumerate(allocation.dwell_times_s):
            if not np.isfinite(tau):
                violations.append(
                    ViolationDetail(
                        constraint="dwell_finite", severity="violation",
                        category="numerical_failure",
                        detail=f"task_{i} dwell time is non-finite: {tau!r}.",
                    )
                )
                continue
            if not (lo - BUDGET_TOLERANCE <= tau <= hi + BUDGET_TOLERANCE):
                violations.append(
                    ViolationDetail(
                        constraint="dwell_bounds", severity="violation",
                        category="infeasible_allocation",
                        detail=f"task_{i} dwell time {tau} s outside action-set bounds [{lo}, {hi}] s.",
                    )
                )
            if not any(abs(tau - a) <= BUDGET_TOLERANCE for a in rc.dwell_time_action_set):
                violations.append(
                    ViolationDetail(
                        constraint="dwell_discrete_membership", severity="violation",
                        category="infeasible_allocation",
                        detail=f"task_{i} dwell time {tau} s is not a member of the configured action set.",
                    )
                )

        # Individual bounds for the optional continuous power block.
        power_unresolved = False
        if allocation.continuous_power_w is not None:
            if rc.min_transmit_power is None or rc.max_transmit_power is None:
                power_unresolved = True
                missing = {
                    name: "W"
                    for name, value in (
                        ("resource.min_transmit_power", rc.min_transmit_power),
                        ("resource.max_transmit_power", rc.max_transmit_power),
                    )
                    if value is None
                }
                from radar_bo.config.base import UnresolvedParameterError

                violations.append(
                    ViolationDetail(
                        constraint="power_bounds_resolvable", severity="violation",
                        category="unresolved_parameter",
                        detail=f"Power bounds are unresolved ({sorted(missing)}); power feasibility cannot be "
                        "judged. Supply justified values or omit the power block.",
                    )
                )
            else:
                for i, p in enumerate(allocation.continuous_power_w):
                    if not np.isfinite(p):
                        violations.append(
                            ViolationDetail(
                                constraint="power_finite", severity="violation",
                                category="numerical_failure", detail=f"task_{i} power is non-finite: {p!r}.",
                            )
                        )
                    elif not (rc.min_transmit_power <= p <= rc.max_transmit_power):
                        violations.append(
                            ViolationDetail(
                                constraint="power_bounds", severity="violation",
                                category="infeasible_allocation",
                                detail=f"task_{i} power {p} W outside configured bounds "
                                f"[{rc.min_transmit_power}, {rc.max_transmit_power}] W.",
                            )
                        )

        # Shared time budget under the configured semantics (instructions 9 and 11;
        # M4 review 3.1: local vs policy-level feasibility distinguished).
        slot_dwell_sum = float(np.sum(np.asarray(allocation.dwell_times_s, dtype=np.float64)))
        normalized = slot_dwell_sum / self.revisit_interval_s
        theta = rc.normalized_time_budget_threshold
        if rc.time_budget_semantics == "hard_per_slot":
            policy_level = "determined_locally"  # proposed semantics: per-slot check is decisive
            if normalized > theta + BUDGET_TOLERANCE:
                violations.append(
                    ViolationDetail(
                        constraint="hard_per_slot_budget", severity="violation",
                        category="infeasible_allocation",
                        detail=f"Per-slot dwell sum {slot_dwell_sum:.6f} s gives {normalized:.6f} = "
                        f"sum(tau)/T0 > Theta_max = {theta} (proposed hard per-slot semantics; "
                        "PROPOSED constraint, stricter than the paper's).",
                    )
                )
        else:  # 'discounted_cumulative' (paper-reference profile)
            slot_term = normalized - theta
            policy_level = "not_assessed_horizon_required"
            violations.append(
                ViolationDetail(
                    constraint="discounted_cumulative_slot_term", severity="note",
                    category="valid",
                    detail=f"Paper-reference semantics: LOCAL per-slot term sum(tau)/T0 - Theta_max = "
                    f"{slot_term:.6e} (normalized usage {normalized:.6f}). This instantaneous residual "
                    "does NOT establish policy-level satisfaction of the cumulative discounted "
                    "budget (Eq. (22)), which spans a discounted horizon; that judgment is outside "
                    "M4/M5 scope. A single slot is neither auto-infeasible nor proven feasible, and "
                    "this is NOT equivalent to the proposed hard per-slot rule.",
                )
            )

        hard = [v for v in violations if v.severity == "violation"]
        if not hard:
            category: ViolationCategory = "valid"
        elif any(v.category == "invalid_configuration" for v in hard):
            category = "invalid_configuration"
        elif any(v.category == "unresolved_parameter" for v in hard):
            category = "unresolved_parameter"
        elif any(v.category == "numerical_failure" for v in hard):
            category = "numerical_failure"
        else:
            category = "infeasible_allocation"

        summary = (
            f"Feasible under {rc.time_budget_semantics} semantics."
            if not hard
            else f"{len(hard)} violated constraint(s): " + "; ".join(v.constraint for v in hard)
        )
        return FeasibilityReport(
            is_feasible=not hard,
            category=category,
            violations=violations,
            summary=summary,
            slot_dwell_sum_s=slot_dwell_sum,
            normalized_time_usage=normalized,
            budget_threshold=theta,
            budget_semantics=rc.time_budget_semantics,
            policy_level_feasibility=policy_level,
        )
