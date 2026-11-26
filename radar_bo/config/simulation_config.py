"""SimulationConfig: master nested configuration with cross-model validation.

Composes TrackingConfig, ScanningConfig, ResourceConfig, EnvironmentConfig,
OptimizationConfig and ExperimentConfig (README section 4.6). Pydantic is used
for validation and serialization only, never for numerical matrix computation.

Single authoritative sources (README section 4.6):

* ``revisit_interval`` lives ONLY in TrackingConfig; every other layer must
  reference ``SimulationConfig.revisit_interval``.
* ``reference_transmit_power`` is authoritative in ResourceConfig;
  TrackingConfig carries the same field only for its README-defined interface
  and the two must be equal when both are supplied.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel
from radar_bo.config.environment_config import EnvironmentConfig
from radar_bo.config.experiment_config import ExperimentConfig
from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.config.scanning_config import ScanningConfig
from radar_bo.config.tracking_config import TrackingConfig


class SimulationConfig(StrictModel):
    """Nested master configuration; save one serialized copy with every run."""

    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    scanning: ScanningConfig = Field(default_factory=ScanningConfig)
    resource: ResourceConfig = Field(default_factory=ResourceConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    # ---- Single authoritative accessors --------------------------------------
    @property
    def revisit_interval(self) -> float:
        """Revisit interval T0 — authoritative source is TrackingConfig."""
        return self.tracking.revisit_interval

    # ---- Cross-model consistency validation ----------------------------------
    @model_validator(mode="after")
    def _cross_validate(self) -> "SimulationConfig":
        tau_max = max(self.resource.dwell_time_action_set)
        if tau_max > self.tracking.revisit_interval + 1e-12:
            raise ValueError(
                f"Cross-model inconsistency: max dwell time {tau_max} s exceeds the revisit "
                f"interval T0 = {self.tracking.revisit_interval} s."
            )
        if not any(
            math.isclose(self.tracking.reference_dwell_time, t, rel_tol=1e-9, abs_tol=1e-12)
            for t in self.resource.dwell_time_action_set
        ):
            raise ValueError(
                f"Cross-model inconsistency: reference_dwell_time "
                f"({self.tracking.reference_dwell_time} s, paper tau0) must be a member of "
                f"resource.dwell_time_action_set."
            )
        tp_tracking = self.tracking.reference_transmit_power
        tp_resource = self.resource.reference_transmit_power
        if (
            tp_tracking is not None
            and tp_resource is not None
            and not math.isclose(tp_tracking, tp_resource, rel_tol=1e-9, abs_tol=0.0)
        ):
            raise ValueError(
                "Cross-model inconsistency: reference_transmit_power is defined in both "
                "TrackingConfig and ResourceConfig (ResourceConfig is authoritative); "
                f"values differ: {tp_tracking} vs {tp_resource}."
            )
        return self

    # ---- Serialization helpers (README section 4.6: save config with each run)
    def to_json_string(self) -> str:
        """Exact serialized configuration for a run record."""
        return self.model_dump_json(indent=2)

    def save_json(self, path: str | Path) -> Path:
        """Write the exact serialized configuration to ``path`` (UTF-8 JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json_string(), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "SimulationConfig":
        """Load and re-validate a saved configuration (same validation as construction)."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
