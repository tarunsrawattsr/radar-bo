"""Unified experiment runner (M7 items 9, 10, 13).

Capabilities: load configuration, resolve registered problems and algorithms,
construct INDEPENDENT problem instances, execute methods, collect structured
result records, record failures, persist completed runs atomically, verify
fairness (shared initialization) directly from stored records, and produce an
experiment summary.

The runner contains NO GP-fitting and NO acquisition-function logic; it reuses
the M5/M6 engine through its public interfaces (item 9). Fairness policy
(item 10): all methods for one (benchmark, seed) use the same seed and the
same deterministic sampler; the runner then VERIFIES identical initial
configurations AND identical observed values directly from the stored records
(seeds alone are never trusted) and records the verification outcome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.experiments.framework.checkpointing import (
    CheckpointStore,
    RecordRejected,
    atomic_write_json,
    compatibility_metadata,
)
from radar_bo.experiments.framework.experiment_config import ExperimentConfig
from radar_bo.experiments.framework.matrix import RunSpec, RunStatus, generate_manifest


def _engine_config(config: ExperimentConfig, seed: int) -> OptimizationConfig:
    return OptimizationConfig(
        surrogate=config.gp.surrogate,
        kernel=config.gp.kernel,
        acquisition=config.acquisition.acquisition,
        input_normalization=config.gp.input_normalization,
        outcome_standardization=config.gp.outcome_standardization,
        precision=config.gp.precision,
        device=config.gp.device,
        refit_per_observation=config.gp.refit_per_observation,
        maximize=True,
        num_initial_observations=config.initial_design.size,
        num_acquisition_candidates=config.candidate_pool.num_acquisition_candidates,
        evaluation_budget=config.stopping.evaluation_budget,
        max_iterations=config.stopping.max_bo_iterations,
        evaluations_per_block_visit=config.stopping.evaluations_per_block_visit,
        cycle_patience=config.stopping.cycle_patience,
        patience=config.stopping.patience,
        runtime_budget_s=config.stopping.runtime_budget_s,
        improvement_tolerance=config.stopping.improvement_tolerance,
        random_seed=seed,
    )


def build_run_record(
    spec_dict: dict,
    algorithm_version: str,
    compatibility: dict,
    optimization_result,
    reference,
    wall_runtime_s: float,
) -> dict:
    """Structured run record (item 13) from an engine OptimizationResult."""
    result = optimization_result
    initial_design = [
        {"config": [float(v) for v in r.config], "objective": float(r.objective), "evaluation_id": r.evaluation_id}
        for r in result.observations
        if (r.metadata or {}).get("stage") == "initial_design"
    ]
    components = {
        "gp_fit_runtime_s": result.gp_fit_runtime_s,
        "acquisition_runtime_s": result.acquisition_runtime_s,
        "objective_runtime_s": result.objective_runtime_s,
        "candidate_generation_runtime_s": result.candidate_generation_runtime_s,
        "constraint_validation_runtime_s": result.constraint_validation_runtime_s,
    }
    component_sum = sum(components.values())
    return {
        "run_id": spec_dict["run_id"],
        "experiment_id": spec_dict["experiment_id"],
        "experiment_config_fingerprint": spec_dict["config_fingerprint"],
        "benchmark_id": spec_dict["benchmark_id"],
        "dimension": spec_dict["dimension"],
        "algorithm_id": spec_dict["algorithm_id"],
        "architecture": spec_dict["architecture"],
        "seed": spec_dict["seed"],
        "algorithm_version": algorithm_version,
        "source_commit": compatibility["source_commit"],
        "dependency_versions": compatibility["dependency_versions"],
        "hyperparameters": spec_dict["hyperparameters"],
        # ---- observations / failures (full precision) -------------------------
        "initial_design": initial_design,
        "observations": [r.model_dump() for r in result.observations],
        "failures": [f.model_dump() for f in result.failures],
        "incumbent_history": list(result.incumbent_history),
        "final_incumbent_config": result.best_config,
        "final_objective": result.best_objective,
        "reference_objective": reference.objective_value,
        "reference_method": reference.verification_method,
        "reference_is_exact": reference.is_exact,
        # ---- counts (item 13) --------------------------------------------------
        "evaluation_counts": {
            "n_initial_evaluations": result.n_initial_evaluations,
            "n_bo_evaluations": result.n_bo_evaluations,
            "n_failed_attempts": result.n_failed_attempts,
            "n_acquisition_scores": result.n_acquisition_scores,
            "pool_requests": result.pool_requests,
            "pool_requested_total": result.pool_requested_total,
            "pool_delivered_total": result.pool_delivered_total,
            "pool_rejected_infeasible_total": result.pool_rejected_infeasible_total,
            "pool_rejected_duplicate_within_pool_total": result.pool_rejected_duplicate_within_pool_total,
            "pool_rejected_duplicate_evaluated_total": result.pool_rejected_duplicate_evaluated_total,
        },
        # ---- block/cycle diagnostics (item 13) ----------------------------------
        "block_cycle_diagnostics": {
            "search_strategy": result.search_strategy,
            "variable_blocks": result.variable_blocks,
            "uncovered_variable_indices": result.uncovered_variable_indices,
            "n_block_visits": result.n_block_visits,
            "n_cycles_completed": result.n_cycles_completed,
            "n_skipped_blocks": result.n_skipped_blocks,
            "skipped_block_records": list(result.skipped_block_records),
            "n_accepted_updates": result.n_accepted_updates,
            "n_cycles_without_improvement": result.n_cycles_without_improvement,
            "current_config_final": result.current_config_final,
        },
        "iteration_diagnostics": [d.model_dump() for d in result.iteration_diagnostics],
        # ---- runtimes (item 13/23; unattributed computed, never assumed zero) ---
        "runtime_measurements": {
            **components,
            "component_sum_s": component_sum,
            "wall_runtime_s": wall_runtime_s,
            "unattributed_overhead_s": wall_runtime_s - component_sum,
        },
        "stopping_reason": result.stopping_reason,
    }


def verify_shared_initialization(records: list[dict], tie_tolerance: float = 1e-9) -> dict:
    """Item-10 fairness verification directly from stored records.

    All BO-method records of one (benchmark_id, dimension, seed) group must have
    identical initial-design CONFIGURATIONS and identical OBSERVED values (seeds
    alone are never trusted). Random search has NO initial-design phase by
    construction (its whole budget is random sampling); it is excluded from the
    initial-design comparison and the exclusion is reported explicitly.
    """
    groups: dict[tuple, dict[str, list]] = {}
    excluded_no_initial_phase: set[str] = set()
    for rec in records:
        if rec.get("seed") is None:
            continue  # reference "runs" have no seed and no initial design
        if rec.get("architecture") == "random_search":
            excluded_no_initial_phase.add(rec["algorithm_id"])
            continue
        initial = rec.get("initial_design") or []
        if not initial:
            excluded_no_initial_phase.add(rec["algorithm_id"])
            continue
        key = (rec["benchmark_id"], rec["dimension"], rec["seed"])
        groups.setdefault(key, {})[rec["algorithm_id"]] = initial
    issues: list[str] = []
    verified: list[dict] = []
    for (benchmark_id, dimension, seed), by_method in sorted(groups.items(), key=lambda kv: str(kv[0])):
        methods = sorted(by_method)
        reference_method = methods[0]
        base = by_method[reference_method]
        for method in methods[1:]:
            other = by_method[method]
            if len(other) != len(base):
                issues.append(
                    f"{benchmark_id}/d{dimension}/seed{seed}: initial-design size differs between "
                    f"{reference_method} ({len(base)}) and {method} ({len(other)})"
                )
                continue
            for i, (a, b) in enumerate(zip(base, other)):
                if any(abs(x - y) > 1e-12 for x, y in zip(a["config"], b["config"])):
                    issues.append(f"{benchmark_id}/d{dimension}/seed{seed}: initial config {i} differs ({method})")
                if abs(a["objective"] - b["objective"]) > tie_tolerance:
                    issues.append(f"{benchmark_id}/d{dimension}/seed{seed}: initial objective {i} differs ({method})")
        verified.append({
            "benchmark_id": benchmark_id, "dimension": dimension, "seed": seed,
            "methods": methods, "n_initial": len(base),
            "initial_config_hash": json.dumps([x["config"] for x in base], sort_keys=True)[:64],
            "identical": True,
        })
    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "groups": verified,
        "excluded_no_initial_design_phase": sorted(excluded_no_initial_phase),
        "policy_note": (
            "random search has no initial-design phase by construction; shared-initial-design "
            "fairness applies to the BO methods and is verified from their stored records"
        ),
    }


class ExperimentRunner:
    """Executes one ExperimentConfig end to end with checkpoint/resume."""

    def __init__(self, config: ExperimentConfig, benchmark_registry, algorithm_registry) -> None:
        self.config = config
        self.benchmarks = benchmark_registry
        self.algorithms = algorithm_registry
        self.store = CheckpointStore(config.output_dir)
        # Validate registry ids before anything executes (item 8).
        for bid in config.benchmark_ids:
            self.benchmarks.get(bid)
        for aid in config.algorithm_ids:
            self.algorithms.get(aid)

    # ---- public API ----------------------------------------------------------
    def execute(self) -> dict:
        """Run (or resume) the experiment; returns the manifest status counts."""
        specs = generate_manifest(self.config, self.benchmarks, self.algorithms)
        compatibility = compatibility_metadata(self.config, self.benchmarks, self.algorithms)
        self.store.initialize(self.config, specs, compatibility)

        pending = self.store.pending_specs()
        for spec_dict in pending:
            if spec_dict["status"] == RunStatus.SKIPPED.value:
                continue  # recorded omission with reason; never executed, never 'completed'
            self._execute_one(spec_dict, compatibility)

        records = self._load_all_search_records(compatibility)
        fairness = verify_shared_initialization(
            records, tie_tolerance=self.config.statistics.paired_tie_tolerance
        )
        summary = self.summarize(fairness)
        return summary

    def _execute_one(self, spec_dict: dict, compatibility: dict) -> None:
        run_id = spec_dict["run_id"]
        self.store.set_status(run_id, RunStatus.RUNNING)
        try:
            spec = RunSpec.model_validate(spec_dict)
            benchmark = self.benchmarks.get(spec.benchmark_id)
            algorithm = self.algorithms.get(spec.algorithm_id)
            if algorithm.kind == "reference":
                reference = benchmark.reference(spec.dimension if spec.dimension in benchmark.supported_dimensions else None)
                record = {
                    "run_id": run_id,
                    "experiment_id": spec.experiment_id,
                    "experiment_config_fingerprint": spec.config_fingerprint,
                    "benchmark_id": spec.benchmark_id,
                    "dimension": spec.dimension,
                    "algorithm_id": spec.algorithm_id,
                    "architecture": spec.architecture,
                    "seed": None,
                    "algorithm_version": algorithm.version,
                    "source_commit": compatibility["source_commit"],
                    "dependency_versions": compatibility["dependency_versions"],
                    "reference_objective": reference.objective_value,
                    "reference_method": reference.verification_method,
                    "reference_is_exact": reference.is_exact,
                    "reference_config": reference.config,
                    "stopping_reason": "not_applicable",
                    "runtime_measurements": {"wall_runtime_s": 0.0},
                }
            else:
                problem = benchmark.problem(spec.dimension)  # INDEPENDENT instance per run
                engine_config = _engine_config(self.config, spec.seed)
                reference = benchmark.reference(spec.dimension)
                started = time.perf_counter()
                out = algorithm.run(
                    benchmark=benchmark, dimension=spec.dimension, seed=spec.seed,
                    config=engine_config,
                    reference_value=reference.objective_value,
                    reference_method=reference.verification_method,
                )
                wall = time.perf_counter() - started
                if out.architecture in ("random_search",):
                    record = self._random_search_record(spec_dict, algorithm, compatibility, out, reference, wall)
                else:
                    record = build_run_record(
                        spec_dict, algorithm.version, compatibility, out.optimization_result, reference, wall
                    )
                    # Per-run validity checks (fail loudly — M6 convention):
                    budget = self.config.stopping.evaluation_budget
                    attempted = (record["evaluation_counts"]["n_initial_evaluations"]
                                 + record["evaluation_counts"]["n_bo_evaluations"])
                    if attempted != budget:
                        raise AssertionError(f"budget violation in {run_id}: {attempted} != {budget}")
            self.store.save_run_record(run_id, record)
            self.store.set_status(run_id, RunStatus.COMPLETED)
        except RecordRejected:
            raise
        except Exception as exc:  # recorded failure with reason (item 12)
            self.store.set_status(run_id, RunStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")

    def _random_search_record(self, spec_dict, algorithm, compatibility, out, reference, wall) -> dict:
        result = out.optimization_result
        # RS 'history' is the full per-evaluation sequence, NOT a best-so-far
        # curve (the M6 archive mislabeled it 'incumbent_history' — see the M7
        # review artifact). M7 records keep the raw sequence under
        # 'evaluation_sequence' AND store a TRUE running-best incumbent history.
        sequence = [entry[1] for entry in result.history]
        incumbent_history: list[float] = []
        current = float("-inf")
        for value in sequence:
            current = max(current, float(value))
            incumbent_history.append(current)
        return {
            "run_id": spec_dict["run_id"],
            "experiment_id": spec_dict["experiment_id"],
            "experiment_config_fingerprint": spec_dict["config_fingerprint"],
            "benchmark_id": spec_dict["benchmark_id"],
            "dimension": spec_dict["dimension"],
            "algorithm_id": spec_dict["algorithm_id"],
            "architecture": spec_dict["architecture"],
            "seed": spec_dict["seed"],
            "algorithm_version": algorithm.version,
            "source_commit": compatibility["source_commit"],
            "dependency_versions": compatibility["dependency_versions"],
            "hyperparameters": spec_dict["hyperparameters"],
            "initial_design": [],
            "observations": [
                {"config": config_list, "objective": objective, "evaluation_id": evaluation_id}
                for config_list, objective, evaluation_id in result.history
            ],
            "failures": [{"evaluation_id": eval_id, "error": error} for eval_id, error in result.failures],
            "evaluation_sequence": sequence,
            "incumbent_history": incumbent_history,
            "final_incumbent_config": result.best_config,
            "final_objective": result.best_objective,
            "reference_objective": reference.objective_value,
            "reference_method": reference.verification_method,
            "reference_is_exact": reference.is_exact,
            "evaluation_counts": {
                "n_objective_evaluations": result.n_objective_evaluations,
                "n_failed_attempts": result.n_failed_attempts,
                "n_rejected_infeasible": result.n_rejected_infeasible,
                "n_skipped_duplicates": result.n_skipped_duplicates,
            },
            "block_cycle_diagnostics": {"search_strategy": "random_search"},
            "iteration_diagnostics": [],
            "runtime_measurements": {"wall_runtime_s": wall},
            "stopping_reason": result.stopping_reason,
        }

    # ---- records / summary -----------------------------------------------------
    def _load_all_search_records(self, compatibility: dict) -> list[dict]:
        records = []
        for spec_dict in self.store.specs():
            if spec_dict["status"] != RunStatus.COMPLETED.value:
                continue
            if spec_dict.get("seed") is None and spec_dict["architecture"] == "reference":
                continue
            try:
                records.append(self.store.load_run_record(
                    spec_dict["run_id"], expected_fingerprint=self.config.fingerprint()
                ))
            except RecordRejected:
                raise  # explicit rejection, never silent (item 11)
        return records

    def summarize(self, fairness: dict) -> dict:
        from radar_bo.experiments.framework.metrics import describe_run_records

        records = self._collect_records()
        analyses = describe_run_records(records, self.config)
        status_counts: dict[str, int] = {}
        for spec in self.store.specs():
            status_counts[spec["status"]] = status_counts.get(spec["status"], 0) + 1
        summary = {
            "experiment_id": self.config.experiment_id,
            "description": self.config.description,
            "result_label": self.config.result_label,
            "configuration": self.config.model_dump(),
            "status_counts": status_counts,
            "shared_initialization_verification": fairness,
            "analysis": analyses,
        }
        atomic_write_json(Path(self.config.output_dir) / "summary.json", summary)
        return summary

    def _collect_records(self) -> list[dict]:
        records = []
        fingerprint = self.config.fingerprint()
        for spec_dict in self.store.specs():
            if spec_dict["status"] != RunStatus.COMPLETED.value:
                continue
            if spec_dict.get("seed") is None and spec_dict["architecture"] == "reference":
                continue
            records.append(self.store.load_run_record(spec_dict["run_id"], expected_fingerprint=fingerprint))
        return records
