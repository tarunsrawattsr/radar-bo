"""Optimization-support layer (generic M4 interfaces).

Contains the allocation representation, constraint validation, resource
accounting and the objective evaluator — all independent of BoTorch, GP
fitting, acquisition functions and BO search algorithms (layer separation,
guarded by tests). Generic BO search modules (M6+) must not depend on radar
equations; the evaluator below is the application-side composition point.
"""

from radar_bo.optimization.allocation import DiscreteMembershipError, ResourceAllocation
from radar_bo.optimization.constraints import (
    BUDGET_TOLERANCE,
    ConstraintManager,
    FeasibilityReport,
    ViolationCategory,
    ViolationDetail,
    ViolationSeverity,
)
from radar_bo.optimization.objective import (
    EvaluationResult,
    ObjectiveEvaluator,
    RandomnessPolicy,
    SnapshotDiagnosticTrackingCostProvider,
    SyntheticLinearTrackingCostProvider,
    TrackingCostProvider,
)
from radar_bo.optimization.resource_accounting import PowerConvention, ResourceAccountant, ResourceUsage

__all__ = [
    "ResourceAllocation",
    "DiscreteMembershipError",
    "ConstraintManager",
    "FeasibilityReport",
    "ViolationDetail",
    "ViolationCategory",
    "ViolationSeverity",
    "BUDGET_TOLERANCE",
    "ResourceAccountant",
    "ResourceUsage",
    "PowerConvention",
    "ObjectiveEvaluator",
    "EvaluationResult",
    "TrackingCostProvider",
    "SnapshotDiagnosticTrackingCostProvider",
    "SyntheticLinearTrackingCostProvider",
    "RandomnessPolicy",
]
