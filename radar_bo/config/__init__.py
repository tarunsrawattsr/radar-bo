"""Configuration layer (README section 4).

Nested Pydantic v2 models + the parameter provenance register. This package
must never import the optimizer, radar or simulation layers.

Typical use::

    from radar_bo.config import SimulationConfig, build_default_provenance

    cfg = SimulationConfig()
    cfg.save_json("results/run/config.json")
    provenance = build_default_provenance(cfg)
    provenance.require_resolved("tracking.revisit_interval")  # fails clearly if None
"""

from radar_bo.config.base import StrictModel, UnresolvedParameterError, ensure_resolved
from radar_bo.config.environment_config import EnvironmentConfig
from radar_bo.config.experiment_config import ExperimentConfig
from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.config.provenance import (
    ProvenanceEntry,
    ProvenanceRegister,
    ProvenanceSource,
    ProvenanceValue,
    build_default_provenance,
)
from radar_bo.config.resource_config import ResourceConfig, TimeBudgetSemantics
from radar_bo.config.scanning_config import RESOLVABLE_FIELDS, ScanningConfig
from radar_bo.config.simulation_config import SimulationConfig
from radar_bo.config.tracking_config import (
    InitializationPolicy,
    ManeuverabilityInterpretation,
    MeasurementVarianceScaling,
    TrackingConfig,
)

__all__ = [
    "StrictModel",
    "UnresolvedParameterError",
    "ensure_resolved",
    "TrackingConfig",
    "InitializationPolicy",
    "ManeuverabilityInterpretation",
    "MeasurementVarianceScaling",
    "ScanningConfig",
    "RESOLVABLE_FIELDS",
    "ResourceConfig",
    "TimeBudgetSemantics",
    "EnvironmentConfig",
    "OptimizationConfig",
    "ExperimentConfig",
    "SimulationConfig",
    "ProvenanceEntry",
    "ProvenanceRegister",
    "ProvenanceSource",
    "ProvenanceValue",
    "build_default_provenance",
]
