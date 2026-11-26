"""EnvironmentConfig: synthetic planar scenario definition.

Per README section 4.4, the framework starts with fixed-population synthetic
planar scenarios. Civilian corridor geometry and variable (arriving/departing)
populations are later research features, not needed to validate the optimizer.
All values here are EXPERIMENTAL settings, not paper-reported values.

The revisit interval is intentionally NOT redefined here: TrackingConfig.
revisit_interval is the single authoritative source (README section 4.6).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel


class EnvironmentConfig(StrictModel):
    """Fixed-population synthetic planar scenario (experimental settings)."""

    scenario_id: str = Field(
        default="synthetic_planar_fixed_population",
        description="Stable identifier recorded in every saved run configuration.",
    )
    simulation_dimension: int = Field(
        default=2, frozen=True, description="Planar scenario (x, y); matches the 4-D CV state."
    )
    num_targets: int = Field(
        default=1, ge=1, description="Number of simultaneously active synthetic targets (experimental default 1)."
    )
    initial_conditions: list[tuple[float, float, float, float]] | None = Field(
        default=None,
        description=(
            "Optional per-target initial states [x, y, vx, vy] in m and m/s, overriding "
            "TrackingConfig.initial_state (an experimental override of the paper's zero "
            "vector). When None, the tracking-config initialization policy applies."
        ),
    )
    simulation_duration: float = Field(
        default=60.0, gt=0.0, description="Total simulated physical time in seconds (experimental)."
    )
    random_seed: int = Field(default=0, description="Master seed for scenario randomness (experimental).")
    motion_model: Literal["constant_velocity"] = Field(
        default="constant_velocity", description="Planar CV motion per Lu et al. Eqs. (1)-(2)."
    )
    arrival_times: list[float] | None = Field(
        default=None, description="Optional arrival times (s); None means fixed population for the whole run."
    )
    departure_times: list[float] | None = Field(
        default=None, description="Optional departure times (s); None means fixed population for the whole run."
    )

    # ---- Validators ----------------------------------------------------------
    @model_validator(mode="after")
    def _population_list_lengths(self) -> "EnvironmentConfig":
        n = self.num_targets
        if self.initial_conditions is not None and len(self.initial_conditions) != n:
            raise ValueError(
                f"initial_conditions has {len(self.initial_conditions)} entries but num_targets is {n}."
            )
        if self.arrival_times is not None and len(self.arrival_times) != n:
            raise ValueError(f"arrival_times has {len(self.arrival_times)} entries but num_targets is {n}.")
        if self.departure_times is not None and len(self.departure_times) != n:
            raise ValueError(f"departure_times has {len(self.departure_times)} entries but num_targets is {n}.")
        return self
