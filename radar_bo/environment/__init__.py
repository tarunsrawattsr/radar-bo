"""Environment layer: planar target states and motion models.

Contains true-state representation (for simulation/evaluation only) and the
constant-velocity motion model, Lu et al. Eqs. (1)-(3). Must not import the
optimizer or the radar layer.
"""

from radar_bo.environment.motion_model import ConstantVelocityMotionModel
from radar_bo.environment.target_state import (
    MEASUREMENT_DIMENSION,
    POSITION_SLICE,
    STATE_DIMENSION,
    STATE_ORDER,
    VELOCITY_SLICE,
    GaussianState,
    MeasurementCovariance,
    Observation,
    TargetState,
)

__all__ = [
    "ConstantVelocityMotionModel",
    "TargetState",
    "GaussianState",
    "MeasurementCovariance",
    "Observation",
    "STATE_DIMENSION",
    "MEASUREMENT_DIMENSION",
    "STATE_ORDER",
    "POSITION_SLICE",
    "VELOCITY_SLICE",
]
