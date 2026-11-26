"""Structured optimization-result records (M5 instruction 25).

Serializable metadata for one optimization run. Experiment metadata is kept
SEPARATE from GP training tensors: results carry plain lists/floats only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.dataset import FailureRecord, ObservationRecord

StopReason = Literal[
    "evaluation_budget_exhausted",
    "max_iterations_reached",
    "runtime_limit_reached",
    "patience_without_improvement",
    "candidate_pool_exhausted",
    "gp_fitting_failure",
    "repeated_objective_failures",
    "no_feasible_initial_design",
]


class IterationDiagnostics(StrictModel):
    """Per-BO-iteration diagnostics (instruction 21 step 13)."""

    iteration: int
    n_observations: int
    incumbent_objective: float
    gap_to_reference: float | None = None
    selected_config: list[float]
    selected_evaluation_id: str
    objective_value: float | None = Field(default=None, description="None when the evaluation failed.")
    improvement: float | None = None
    gp_fit_duration_s: float
    acquisition_duration_s: float
    n_candidates_scored: int


class OptimizationResult(StrictModel):
    """Complete, serializable record of one optimization run."""

    problem_id: str
    algorithm_id: str
    seed: int
    initial_design_size: int
    n_initial_evaluations: int
    n_bo_evaluations: int
    n_failed_attempts: int
    n_acquisition_scores: int
    # Candidate-pool accounting (instruction 19/29): pool sizes are NOT objective
    # evaluations; rejections are reported, never repaired.
    pool_requests: int = Field(default=0, description="Number of pool generations (initial design + per BO iteration).")
    pool_requested_total: int = Field(default=0, description="Sum of requested pool sizes.")
    pool_delivered_total: int = Field(default=0, description="Sum of delivered (feasible, distinct) pool sizes.")
    pool_rejected_infeasible_total: int = Field(default=0)
    pool_rejected_duplicate_within_pool_total: int = Field(default=0)
    pool_rejected_duplicate_evaluated_total: int = Field(default=0)
    best_config: list[float] | None = Field(
        default=None,
        description="Exact decision vector of the best EVALUATED feasible configuration; "
        "None when no objective evaluation succeeded (no incumbent exists — never a GP-predicted vector).",
    )
    best_objective: float | None = Field(
        default=None,
        description="Best successfully OBSERVED objective value; None when no evaluation succeeded.",
    )
    reference_objective: float | None = None
    reference_method: str | None = None
    best_gap_to_reference: float | None = None
    incumbent_history: list[float] = Field(description="Best-observed objective after each iteration (nondecreasing for deterministic problems).")
    observations: list[ObservationRecord] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)
    iteration_diagnostics: list[IterationDiagnostics] = Field(default_factory=list)
    gp_fit_runtime_s: float
    acquisition_runtime_s: float
    objective_runtime_s: float
    total_runtime_s: float
    stopping_reason: StopReason
    settings: dict = Field(description="Optimizer settings snapshot (plain values; no tensors).")
    dependency_versions: dict[str, str] = Field(default_factory=dict)
