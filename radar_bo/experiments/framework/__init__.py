"""M7 experimental framework (instruction-stream M7, items 5-31).

Application-independent experiment orchestration on top of the existing M5/M6
components. This layer MAY import ``radar_bo.config``, the generic
``radar_bo.optimization.bo`` interfaces and scientific/plotting libraries; it
must NOT be imported by config/environment/radar/numerics/simulation modules
(layer separation, item 32 — guarded by tests).

Everything in this framework is generic and synthetic. No physical radar
power allocation, corridor sensing model, contextual radar optimization or
application-specific surveillance optimization is introduced here (item 1).
"""

from radar_bo.experiments.framework.experiment_config import (
    AcquisitionSettings,
    CheckpointPolicy,
    ExperimentConfig,
    GPSettings,
    InitialDesignSettings,
    RuntimeMeasurementSettings,
    StoppingSettings,
    config_fingerprint,
)
from radar_bo.experiments.framework.matrix import (
    RunSpec,
    RunStatus,
    generate_manifest,
    run_identity_key,
    stable_run_id,
)
from radar_bo.experiments.framework.checkpointing import CheckpointStore

__all__ = [
    "AcquisitionSettings",
    "CheckpointPolicy",
    "ExperimentConfig",
    "GPSettings",
    "InitialDesignSettings",
    "RuntimeMeasurementSettings",
    "StoppingSettings",
    "config_fingerprint",
    "RunSpec",
    "RunStatus",
    "generate_manifest",
    "run_identity_key",
    "stable_run_id",
    "CheckpointStore",
]
