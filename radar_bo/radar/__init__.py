"""Radar/estimation layer: measurement model, measurement noise, EKF,
initialization policies and tracking cost (Lu et al. Eqs. (4)-(15)).

Generic estimation components only — no scanning physics, power allocation,
corridor geometry or optimization imports (layer separation, guarded by
tests).
"""

from radar_bo.radar.extended_kalman_filter import CovarianceUpdateMethod, ExtendedKalmanFilter
from radar_bo.radar.initialization import EstimatorInitializationError, initialize_estimator
from radar_bo.radar.measurement_model import MeasurementGeometryError, PlanarRangeAzimuthModel
from radar_bo.radar.measurement_noise import PlanarSNRMeasurementNoiseModel
from radar_bo.radar.tracking_cost import E_POSITION_SELECTOR, position_tracking_cost

__all__ = [
    "PlanarRangeAzimuthModel",
    "MeasurementGeometryError",
    "PlanarSNRMeasurementNoiseModel",
    "EstimatorInitializationError",
    "initialize_estimator",
    "ExtendedKalmanFilter",
    "CovarianceUpdateMethod",
    "position_tracking_cost",
    "E_POSITION_SELECTOR",
]
