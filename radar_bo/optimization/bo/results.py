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
    "all_blocks_exhausted_in_cycle",
    "cycle_patience_exhausted",
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
    # ---- M6 block fields (defaults keep M5 joint records unchanged) ----
    block_id: str | None = Field(default=None, description="Active block for this iteration (None in pure joint runs).")
    cycle_number: int | None = None
    within_cycle_index: int | None = None
    accepted_current_config: bool | None = Field(
        default=None, description="Whether the current configuration was updated (improvement-only acceptance)."
    )
    # ---- M7 additive fields (None in M5/M6 records; pure bookkeeping) ----
    attempted_evaluations: int | None = Field(
        default=None,
        description=(
            "Total ATTEMPTED objective evaluations (initial design + BO attempts incl. "
            "failures) at the time this diagnostic was recorded; enables exact "
            "convergence curves on the attempted-evaluation axis for runs produced "
            "after M7. None in archived M5/M6 records."
        ),
    )
    elapsed_runtime_s: float | None = Field(
        default=None,
        description="Wall-clock seconds since the start of optimize() at diagnostic time (M7; None in archived records).",
    )


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
    # ---- M6 block-architecture fields (all defaulted: M5 joint records unchanged) ----
    search_strategy: str = Field(default="joint", description="Search strategy id: joint / alternating / pairwise.")
    variable_blocks: list[dict] = Field(
        default_factory=list,
        description="Resolved block descriptions (metadata; never GP input features).",
    )
    uncovered_variable_indices: list[int] = Field(
        default_factory=list,
        description="Decision-variable indices covered by NO block (stay fixed at the current configuration).",
    )
    current_config_final: list[float] | None = Field(
        default=None,
        description="Final current configuration (search center under the acceptance policy); distinct from the incumbent.",
    )
    current_config_value_final: float | None = Field(
        default=None, description="Verified objective value of the final current configuration."
    )
    n_accepted_updates: int = Field(default=0, description="Current-configuration acceptances (improvement-only policy).")
    n_block_visits: int = Field(default=0, description="Scheduler visits (a visit may perform >=1 evaluation attempt).")
    n_cycles_completed: int = Field(default=0, description="Complete cycles (every block visited or skipped once).")
    n_skipped_blocks: int = Field(default=0, description="Block visits skipped for lack of feasible candidates.")
    skipped_block_records: list[dict] = Field(
        default_factory=list,
        description="Skipped-block log: block_id, cycle_number, reason (instruction 9).",
    )
    n_cycles_without_improvement: int = Field(
        default=0, description="Consecutive complete cycles without accepted improvement or incumbent improvement."
    )
    evaluations_per_block_visit: int = Field(
        default=1, description="Configured evaluations per block visit (instruction 13)."
    )
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
    candidate_generation_runtime_s: float = Field(
        default=0.0,
        description=(
            "Wall-clock time of candidate-pool SAMPLING+dedup (M7 item-23 boundary fix: feasibility "
            "validation EXCLUDED so stage timers are disjoint). ARCHIVED M6 records used the nested "
            "convention where this field INCLUDED feasibility validation."
        ),
    )
    constraint_validation_runtime_s: float = Field(
        default=0.0,
        description="Time spent in feasibility/constraint validation inside candidate generation (M6 instruction 24).",
    )
    total_runtime_s: float
    stopping_reason: StopReason
    settings: dict = Field(description="Optimizer settings snapshot (plain values; no tensors).")
    dependency_versions: dict[str, str] = Field(default_factory=dict)
