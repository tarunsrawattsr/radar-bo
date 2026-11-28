"""Unified experiment configuration (M7 item 5).

A Pydantic model describing a COMPLETE experiment (benchmarks, methods,
seeds, budgets, measurement and analysis settings, outputs, checkpoint
policy). Distinctness from radar configuration (item 5):

* This model lives under ``radar_bo/experiments/`` and is consumed ONLY by
  the experiment framework; it never reaches the radar/environment layers.
* Every value here is an EXPLICIT experimental value. No paper-derived radar
  parameter and no unresolved physical quantity appears in this model; the
  provenance register (M2) keeps those unresolved and untouched.
* Fields that were inherited from earlier milestones are labelled as such
  ("inherited M5/M6 setting; not proven optimal") in their descriptions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel


class GPSettings(StrictModel):
    """GP surrogate settings (fixed semantics inherited from M5/M6)."""

    surrogate: Literal["single_task_gp"] = "single_task_gp"
    kernel: Literal["matern_5_2"] = "matern_5_2"
    matern_nu: Literal[2.5] = 2.5
    ard_num_dims: bool = True
    input_normalization: bool = True
    outcome_standardization: bool = True
    precision: Literal["float64"] = "float64"
    device: Literal["cpu"] = "cpu"
    refit_per_observation: bool = Field(
        default=True,
        description="Inherited M5 policy (experimental): fresh GP refit after each successful observation.",
    )
    # Noise convention (M5): the GP likelihood noise is INFERRED, never forced
    # to zero, even for deterministic synthetic objectives.
    noise_inferred: Literal[True] = True


class AcquisitionSettings(StrictModel):
    """Acquisition + candidate-pool settings (inherited M5/M6 values)."""

    acquisition: Literal["log_expected_improvement"] = "log_expected_improvement"
    num_acquisition_candidates: int = Field(
        default=500, ge=1,
        description="Candidate-pool size per acquisition step (inherited M5/M6 setting; not proven optimal).",
    )
    # NOTE: Benzaghta xi = 0.01 has no effect with the installed LogEI API and
    # is preserved only in the provenance register (M5 decision).


class InitialDesignSettings(StrictModel):
    """Initial-design settings (inherited M5/M6: seeded uniform feasible sampler)."""

    size: int = Field(default=30, ge=1, description="Initial-design size (inherited M5/M6; not proven optimal).")
    sampler: Literal["seeded_uniform_feasible"] = "seeded_uniform_feasible"
    mode: Literal["seeded_sampler", "explicit_shared"] = Field(
        default="seeded_sampler",
        description=(
            "M8A (item 3.3), corrected in M8A-R1: 'seeded_sampler' preserves the M5/M6 "
            "behavior (same seed per group + retrospective verification from stored "
            "records). 'explicit_shared' persists ONE authoritative design per "
            "(benchmark, dimension, seed) group under initial_designs/ with a genuine "
            "SHA-256 fingerprint over the design rows, RELOADS it unchanged on resume "
            "(never silently regenerated), validates it on every load (size, "
            "dimension, finiteness, feasibility, duplicates; malformed/duplicate/"
            "incomplete/inconsistent designs are rejected), and passes the SAME "
            "immutable configuration list to every applicable optimization method "
            "through the engine's validated initial_design_configs interface (the BO "
            "architectures and the separately identified random_search_shared_start; "
            "plain random_search has no initial phase and explicitly ignores the "
            "design); identical initial objectives are then guaranteed by "
            "construction and verified against the persisted artifact."
        ),
    )
    sharing_policy: Literal["same_seed_with_stored_record_verification"] = Field(
        default="same_seed_with_stored_record_verification",
        description=(
            "Item-10 fairness policy: all methods for one (benchmark, seed) run with "
            "the same seed through the same deterministic sampler, and the runner "
            "VERIFIES identical initial configurations and observed values directly "
            "from the stored records (seeds alone are never trusted). In "
            "'explicit_shared' mode the verification additionally compares every "
            "record against the persisted authoritative design and its SHA-256 "
            "fingerprint (M8A-R1)."
        ),
    )


class StoppingSettings(StrictModel):
    """Stopping/budget settings mapped 1:1 onto the engine's OptimizationConfig."""

    evaluation_budget: int = Field(ge=1, description="TOTAL attempted objective evaluations per run (never per block).")
    max_bo_iterations: int | None = Field(default=None, ge=1, description="Optional BO-iteration cap (experimental).")
    evaluations_per_block_visit: int = Field(
        default=1, ge=1, description="Evaluations per block visit (inherited M6 default 1; experimental).",
    )
    cycle_patience: int | None = Field(default=None, ge=1, description="Complete cycles without improvement (experimental).")
    patience: int | None = Field(default=None, ge=1, description="BO iterations without improvement (experimental).")
    runtime_budget_s: float | None = Field(default=None, gt=0.0, description="Optional wall-clock budget (experimental).")
    improvement_tolerance: float | None = Field(default=None, gt=0.0, description="Acceptance tolerance (experimental).")


class RuntimeMeasurementSettings(StrictModel):
    """Stage-timer and diagnostics settings (M7 items 13/23)."""

    stage_timers: Literal[True] = True
    per_iteration_diagnostics: Literal[True] = True
    per_iteration_elapsed_and_attempts: Literal[True] = Field(
        default=True,
        description="Record attempted_evaluations + elapsed wall-clock per iteration (M7 additive engine fields).",
    )
    timer_boundary_documentation: Literal["see framework.runtime_analysis.TIMER_BOUNDARIES"] = (
        "see framework.runtime_analysis.TIMER_BOUNDARIES"
    )


class StatisticalAnalysisSettings(StrictModel):
    """Statistical-analysis settings (M7 items 18-22) — frozen BEFORE new runs."""

    threshold_closures: list[float] = Field(
        default=[0.5, 0.9, 0.99],
        description="Gap-closure thresholds (item 18): fractions of the INITIAL reference gap, defined before new comparative experiments.",
    )
    bootstrap_replications: int = Field(default=10000, ge=100, description="Bootstrap resampling count (item 21).")
    bootstrap_seed: int = Field(default=2026, description="Documented bootstrap RNG seed (item 21).")
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    paired_tie_tolerance: float = Field(
        default=1e-9,
        description="|difference| <= tolerance counts as a tie in paired comparisons (item 20; matches the engine duplicate tolerance).",
    )
    multiple_comparison_correction: Literal["holm", "none"] = Field(
        default="holm",
        description="Correction applied when a FAMILY of significance tests is presented (item 22); 'none' only for explicitly exploratory single tests.",
    )


class CheckpointPolicy(StrictModel):
    """Checkpoint/resume policy (M7 items 11-12)."""

    enabled: bool = True
    atomic_write: Literal[True] = True
    per_run_files: Literal[True] = True
    resume: bool = Field(default=True, description="Resume incomplete experiments without repeating completed compatible runs.")
    verify_compatibility_on_resume: Literal[True] = True
    reject_incompatible_records: Literal[True] = Field(
        default=True,
        description="Corrupted or incompatible records are explicitly rejected; never silently combined.",
    )


class ExperimentConfig(StrictModel):
    """One reproducible experiment definition (M7 item 5).

    DISTINCT from paper-derived radar configuration: this model lives in the
    experiment layer, never carries radar quantities, and every value is an
    explicit experimental setting (unresolved radar parameters stay resolved
    to None in the provenance register, untouched by this framework).
    """

    experiment_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_\-]*$", description="Stable experiment identifier.")
    description: str = Field(min_length=1)
    result_label: str = Field(
        description="Mandatory label, e.g. 'generic synthetic M7 validation experiment; no radar claims'.",
    )
    benchmark_ids: list[str] = Field(min_length=1, description="Registered benchmark identifiers (benchmark registry).")
    algorithm_ids: list[str] = Field(min_length=1, description="Registered algorithm identifiers (algorithm registry).")
    seeds: list[int] = Field(min_length=1, description="Random seeds; every search method runs once per seed.")
    dimensions: list[int] = Field(
        default=[],
        description=(
            "Problem dimensions for dimension-family benchmarks (separable/coupled). "
            "Fixed-dimension benchmarks ignore this list. Empty = use each registry "
            "entry's own dimension(s)."
        ),
    )
    initial_design: InitialDesignSettings = InitialDesignSettings()
    stopping: StoppingSettings
    candidate_pool: AcquisitionSettings = AcquisitionSettings()
    gp: GPSettings = GPSettings()
    acquisition: AcquisitionSettings = AcquisitionSettings()
    runtime_measurement: RuntimeMeasurementSettings = RuntimeMeasurementSettings()
    statistics: StatisticalAnalysisSettings = StatisticalAnalysisSettings()
    output_dir: str = Field(description="Experiment output directory (results/...; gitignored, regenerable).")
    checkpoint: CheckpointPolicy = CheckpointPolicy()

    @model_validator(mode="after")
    def _validate_distinct_lists(self) -> "ExperimentConfig":
        if len(set(self.benchmark_ids)) != len(self.benchmark_ids):
            raise ValueError("benchmark_ids contains duplicates.")
        if len(set(self.algorithm_ids)) != len(self.algorithm_ids):
            raise ValueError("algorithm_ids contains duplicates.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds contains duplicates.")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions contains duplicates.")
        return self

    def fingerprint(self) -> str:
        """Stable sha256 over the canonical JSON of the full configuration."""
        return config_fingerprint(self.model_dump())


def config_fingerprint(config_dict: dict) -> str:
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
