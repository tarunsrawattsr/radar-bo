"""Experiment-matrix generation (M7 item 8) and run statuses (item 12).

A run = (benchmark, dimension, algorithm, seed, hyperparameter configuration).
Every combination receives a STABLE run identifier (sha256 over the canonical
identity tuple). Duplicate configurations are detected; incompatible
combinations are validated BEFORE execution and appear in the manifest as
explicitly skipped entries with reasons.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.experiments.framework.experiment_config import ExperimentConfig, config_fingerprint


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunSpec(StrictModel):
    """One requested run (never contains results — results live in run files)."""

    run_id: str = Field(min_length=16, max_length=16, description="Stable identity hash (sha256[:16]).")
    experiment_id: str
    benchmark_id: str
    dimension: int
    algorithm_id: str
    architecture: str
    seed: int | None = Field(default=None, description="None for deterministic reference 'runs'.")
    config_fingerprint: str = Field(description="sha256 of the experiment configuration this run belongs to.")
    hyperparameters: dict = Field(description="Exact engine-facing settings for this run (budget, initial design, pool, GP, acquisition).")
    status: RunStatus = RunStatus.PENDING
    reason: str | None = Field(default=None, description="Failure/skip/interruption reason (item 12).")


def run_identity_key(experiment_id: str, benchmark_id: str, dimension: int,
                     algorithm_id: str, seed: int | None, config_fingerprint_value: str) -> str:
    canonical = json.dumps(
        [experiment_id, benchmark_id, dimension, algorithm_id, seed, config_fingerprint_value],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_run_id(identity_key: str) -> str:
    return identity_key[:16]


def generate_manifest(
    config: ExperimentConfig,
    benchmark_registry,
    algorithm_registry,
) -> list[RunSpec]:
    """Complete, validated manifest of requested runs (item 8)."""
    fingerprint = config.fingerprint()
    hyperparams = {
        "evaluation_budget": config.stopping.evaluation_budget,
        "num_initial_observations": config.initial_design.size,
        "num_acquisition_candidates": config.candidate_pool.num_acquisition_candidates,
        "evaluations_per_block_visit": config.stopping.evaluations_per_block_visit,
        "cycle_patience": config.stopping.cycle_patience,
        "max_bo_iterations": config.stopping.max_bo_iterations,
        "patience": config.stopping.patience,
        "runtime_budget_s": config.stopping.runtime_budget_s,
        "improvement_tolerance": config.stopping.improvement_tolerance,
        "gp": config.gp.model_dump(),
        "acquisition": config.acquisition.model_dump(),
        "initial_design": config.initial_design.model_dump(),
    }

    specs: list[RunSpec] = []
    seen: set[str] = set()
    for benchmark_id in config.benchmark_ids:
        benchmark = benchmark_registry.get(benchmark_id)
        dimensions = config.dimensions or list(benchmark.supported_dimensions)
        for dimension in dimensions:
            if dimension not in benchmark.supported_dimensions:
                specs.append(RunSpec(
                    run_id=stable_run_id(run_identity_key(
                        config.experiment_id, benchmark_id, dimension, "INVALID_DIMENSION", None, fingerprint)),
                    experiment_id=config.experiment_id, benchmark_id=benchmark_id, dimension=dimension,
                    algorithm_id="(all)", architecture="(all)", seed=None, config_fingerprint=fingerprint,
                    hyperparameters=hyperparams, status=RunStatus.SKIPPED,
                    reason=f"incompatible_combination: dimension {dimension} not supported by {benchmark_id}",
                ))
                continue
            for algorithm_id in config.algorithm_ids:
                algorithm = algorithm_registry.get(algorithm_id)
                problem = benchmark.problem(dimension)
                incompatible_reason = None
                if algorithm.requires_recommended_blocks:
                    recommended = getattr(problem, "recommended_blocks", None)
                    if recommended is None:
                        incompatible_reason = (
                            f"incompatible_combination: {benchmark_id} defines no recommended "
                            f"{algorithm.architecture} grouping (block-restricted search undefined)"
                        )
                    else:
                        groups = recommended["alternating" if algorithm.architecture == "alternating" else "pairwise"]
                        if algorithm.architecture == "alternating" and len(groups) < 2:
                            incompatible_reason = (
                                f"incompatible_combination: {benchmark_id} recommends <2 alternating groups"
                            )
                        if algorithm.architecture == "pairwise":
                            if len(groups) < 2:
                                incompatible_reason = (
                                    f"incompatible_combination: {benchmark_id} recommends <2 pairwise groups"
                                )
                            elif any(len(g) != 2 for g in groups):
                                incompatible_reason = (
                                    f"incompatible_combination: {benchmark_id} pairwise groups are not all 2-variable"
                                )
                if incompatible_reason is not None:
                    specs.append(RunSpec(
                        run_id=stable_run_id(run_identity_key(
                            config.experiment_id, benchmark_id, dimension, algorithm_id + ":incompatible", None, fingerprint)),
                        experiment_id=config.experiment_id, benchmark_id=benchmark_id, dimension=dimension,
                        algorithm_id=algorithm_id, architecture=algorithm.architecture, seed=None,
                        config_fingerprint=fingerprint, hyperparameters=hyperparams,
                        status=RunStatus.SKIPPED, reason=incompatible_reason,
                    ))
                    continue

                seeds = [None] if algorithm.kind == "reference" else config.seeds
                for seed in seeds:
                    identity = run_identity_key(
                        config.experiment_id, benchmark_id, dimension, algorithm_id, seed, fingerprint)
                    run_id = stable_run_id(identity)
                    if run_id in seen:
                        raise ValueError(
                            f"Duplicate run configuration detected (run_id={run_id}): "
                            f"benchmark={benchmark_id} dimension={dimension} algorithm={algorithm_id} seed={seed}"
                        )
                    seen.add(run_id)
                    specs.append(RunSpec(
                        run_id=run_id, experiment_id=config.experiment_id, benchmark_id=benchmark_id,
                        dimension=dimension, algorithm_id=algorithm_id, architecture=algorithm.architecture,
                        seed=seed, config_fingerprint=fingerprint, hyperparameters=hyperparams,
                    ))
    return specs
