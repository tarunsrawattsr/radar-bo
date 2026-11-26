"""ResourceConfig: dwell-time action set, time budget and trade-off parameters.

Paper sources (Lu et al., ICC 2024):

* dwell-time action set tau in {0.1, 0.2, ..., 1.0} s (paper action space);
* normalized time-budget threshold Theta_max = 1 (Eq. (22));
* tracking/scanning trade-off coefficient beta = 1e6 (Eq. (23));
* discount factor gamma = 0.9 (Eq. (22), discounted cumulative CMDP budget).

Recorded deviations and distinctions (README non-negotiables):

* ``time_budget_semantics='hard_per_slot'`` is a PROPOSED constraint that is
  STRICTER than the paper's discounted cumulative budget: it bounds the
  per-slot dwell sum directly instead of the discounted cumulative sum. The
  paper semantics remain the default.
* Peak pulse power and average transmitted power are DISTINCT quantities.
  Every power field is named explicitly and values must never be silently
  interchanged; none of them has a paper default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from radar_bo.config.base import StrictModel

TimeBudgetSemantics = Literal[
    # Paper Eq. (22): discounted cumulative dwell budget, sum_m gamma^m * (sum_i tau_i / T0 - Theta_max) <= 0.
    "discounted_cumulative",
    # PROPOSED, stronger alternative: per-slot constraint sum_i tau_i / T0 <= Theta_max every slot.
    "hard_per_slot",
]


class ResourceConfig(StrictModel):
    """Dwell-time resources, budget semantics and the tracking/scanning trade-off."""

    # ---- Paper-reported values ----------------------------------------------
    dwell_time_action_set: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        description="Paper dwell-time action set: tau in {0.1, ..., 1.0} s in 0.1 s steps.",
    )
    normalized_time_budget_threshold: float = Field(
        default=1.0, gt=0.0, description="Theta_max = 1 (paper Eq. (22)); budget is normalized by T0."
    )
    tracking_scanning_tradeoff: float = Field(
        default=1.0e6,
        gt=0.0,
        description=(
            "beta = 1e6 (paper Eq. (23)). Preserved for reproduction only: it "
            "calibrates the scanning reward to the tracking cost units and must be "
            "recalibrated if the metric definitions change."
        ),
    )
    discount_factor: float = Field(
        default=0.9, gt=0.0, le=1.0, description="gamma = 0.9 (paper Eq. (22) discounted cumulative budget)."
    )
    time_budget_semantics: TimeBudgetSemantics = Field(
        default="discounted_cumulative",
        description=(
            "Paper default: discounted cumulative budget (Eq. (22)). "
            "'hard_per_slot' is a proposed, STRICTER deviation and must be "
            "labelled as such in any experiment output."
        ),
    )

    # ---- Proposed power settings (no paper default; never invent) -----------
    # NOTE: peak pulse power and average transmitted power are distinct
    # quantities. These fields are intentionally explicit; do not merge them.
    min_transmit_power: float | None = Field(
        default=None, gt=0.0, description="PROPOSED lower bound of the transmit-power search space, W (average power unless stated otherwise)."
    )
    max_transmit_power: float | None = Field(
        default=None, gt=0.0, description="PROPOSED upper bound of the transmit-power search space, W (average power unless stated otherwise)."
    )
    reference_transmit_power: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "PROPOSED reference power for SNR normalization (analogous to tau0/r0 in "
            "Eq. (5)). Authoritative for shared use; TrackingConfig carries the same "
            "field only for the README-defined interface and SimulationConfig "
            "cross-validates equality."
        ),
    )
    scanning_transmit_power: float | None = Field(
        default=None, gt=0.0, description="PROPOSED transmit power dedicated to scanning beams, W."
    )
    energy_budget: float | None = Field(
        default=None, gt=0.0, description="PROPOSED total energy budget, J; no paper default."
    )

    # ---- Validators ----------------------------------------------------------
    @field_validator("dwell_time_action_set")
    @classmethod
    def _validate_action_set(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("dwell_time_action_set must be non-empty.")
        if any(t <= 0.0 for t in v):
            raise ValueError(f"All dwell times must be positive; got {v}.")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError(f"dwell_time_action_set must be strictly increasing; got {v}.")
        diffs = [b - a for a, b in zip(v, v[1:])]
        if len(diffs) > 1 and (max(diffs) - min(diffs)) > 1e-9:
            raise ValueError(
                "dwell_time_action_set must be a uniform grid (paper: 0.1 s steps). "
                "A non-uniform grid needs documented justification."
            )
        return v

    @model_validator(mode="after")
    def _validate_power_bounds(self) -> "ResourceConfig":
        if self.min_transmit_power is not None and self.max_transmit_power is not None:
            if self.min_transmit_power >= self.max_transmit_power:
                raise ValueError(
                    f"min_transmit_power ({self.min_transmit_power} W) must be strictly below "
                    f"max_transmit_power ({self.max_transmit_power} W)."
                )
        return self
