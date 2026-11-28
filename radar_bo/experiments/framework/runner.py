"""Unified experiment runner (M7 items 9, 10, 13; M8A-R1 Phase B/C corrections).

Capabilities: load configuration, resolve registered problems and algorithms,
construct INDEPENDENT problem instances, execute methods, collect structured
result records, record failures, persist completed runs atomically, verify
fairness (shared initialization) directly from stored records AND against the
persisted authoritative initial designs, and produce an experiment summary.

The runner contains NO GP-fitting and NO acquisition-function logic; it reuses
the M5/M6 engine through its public interfaces (item 9). Fairness policy
(item 10): under ``explicit_shared`` mode ONE authoritative design per
(benchmark, dimension, seed) group is PERSISTED with a genuine SHA-256
fingerprint (framework.initial_design_store) and passed UNCHANGED to every
applicable method; verification compares stored records against that artifact.
Under the historical ``seeded_sampler`` mode the M5/M6 behavior is preserved
(same seed per group + retrospective verification from stored records).

M8A-R1 Phase C: evaluation-budget handling distinguishes LEGITIMATE EARLY
STOPPING (patience/cycle patience/runtime/iteration caps → COMPLETED with
``early_completion`` metadata) from ENGINE FAILURES (no feasible initial
design, repeated GP/objective failures → FAILED with reason) and from genuine
budget violations (attempted > budget → AssertionError). Resume adopts an
already-persisted valid record whose manifest status update was interrupted,
preventing repetition of completed compatible runs; failure/retry histories
are preserved in the manifest.
"""

from __future__ import annotations

import json
import time

import numpy as np
from pathlib import Path

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.experiments.framework.checkpointing import (
    CheckpointStore,
    RecordRejected,
    atomic_write_json,
    compatibility_metadata,
)
from radar_bo.experiments.framework.experiment_config import ExperimentConfig
from radar_bo.experiments.framework.initial_design_store import (
    DesignValidationError,
    InitialDesignStore,
)
from radar_bo.experiments.framework.matrix import RunSpec, RunStatus, generate_manifest

# M8A-R1 Phase C: stopping-reason classification.
# The full budget being exhausted is the NORMAL completion path.
BUDGET_EXHAUSTED_REASON = "evaluation_budget_exhausted"
# Legitimate early stops: the run completed with FEWER evaluations than the
# budget because a configured stopping condition fired (or the discrete
# candidate/block structure was exhausted). These are COMPLETED runs, never
# failures.
LEGITIMATE_EARLY_STOP_REASONS = frozenset({
    "max_iterations_reached",
    "runtime_limit_reached",
    "patience_without_improvement",
    "cycle_patience_exhausted",
    "candidate_pool_exhausted",
    "all_blocks_exhausted_in_cycle",
    "generation_attempt_limit_reached",  # random-search generation cap
})
# Engine-level failure conditions: the engine returned a result but the run
# did not produce a usable optimization trajectory. The record is persisted
# (failure history preserved) and the run status is FAILED.
ENGINE_FAILURE_STOP_REASONS = frozenset({
    "no_feasible_initial_design",
    "gp_fitting_failure",
    "repeated_objective_failures",
})


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


def verify_shared_initialization(
    records: list[dict],
    tie_tolerance: float = 1e-9,
    authoritative_designs: dict | None = None,
    objective_prover=None,
) -> dict:
    """Item-10 fairness verification directly from stored records.

    All BO-method records of one (benchmark_id, dimension, seed) group must have
    identical initial-design CONFIGURATIONS and identical OBSERVED values (seeds
    alone are never trusted). Random search has NO initial-design phase by
    construction (its whole budget is random sampling); it is excluded from the
    initial-design comparison and the exclusion is reported explicitly.

    M8A-R1 (Phase B): when ``authoritative_designs`` is provided
    ({(benchmark_id, dimension, seed): {"rows": [...], "fingerprint": sha256}}),
    each record's stored initial phase must ALSO match the persisted
    authoritative rows exactly and carry the matching fingerprint. Additionally,
    within each record the initial phase is checked for duplicates and
    completeness (malformed/duplicate/incomplete detection).
    """
    groups: dict[tuple, dict[str, list]] = {}
    excluded_no_initial_phase: set[str] = set()
    for rec in records:
        if rec.get("seed") is None:
            continue  # reference "runs" have no seed and no initial design
        initial = rec.get("initial_design") or []
        if rec.get("architecture") == "random_search" or not initial:
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
        # M8A-R1: duplicate / incomplete detection within each record's phase
        for method, phase in by_method.items():
            configs = [entry["config"] for entry in phase]
            for i in range(len(configs)):
                for j in range(i):
                    if configs[i] == configs[j] or all(
                        abs(x - y) <= 1e-12 for x, y in zip(configs[i], configs[j])
                    ):
                        issues.append(
                            f"{benchmark_id}/d{dimension}/seed{seed}: duplicate initial configs "
                            f"{j} and {i} in {method}'s record"
                        )
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
        group_result = {
            "benchmark_id": benchmark_id, "dimension": dimension, "seed": seed,
            "methods": methods, "n_initial": len(base),
            "initial_config_hash": json.dumps([x["config"] for x in base], sort_keys=True)[:64],
            "identical": True,
        }
        # M8A-R1: verify against the persisted authoritative design (fingerprint)
        if authoritative_designs is not None:
            key = (benchmark_id, dimension, seed)
            auth = authoritative_designs.get(key)
            if auth is None:
                issues.append(f"{benchmark_id}/d{dimension}/seed{seed}: no authoritative design persisted for a verified group")
                group_result["authoritative_match"] = False
            else:
                records_by_method = {}
                for rec in records:
                    if (rec.get("benchmark_id"), rec.get("dimension"), rec.get("seed")) == key:
                        records_by_method[rec["algorithm_id"]] = rec
                fp_issues = []
                for m in methods:
                    fp = records_by_method.get(m, {}).get("initial_design_fingerprint")
                    if fp != auth["fingerprint"]:
                        fp_issues.append(f"{m} recorded {str(fp)[:12]}")
                config_match = (
                    len(base) == len(auth["rows"]) and all(
                        len(entry["config"]) == len(row)
                        and all(abs(x - y) <= 1e-12 for x, y in zip(entry["config"], row))
                        for entry, row in zip(base, auth["rows"])
                    )
                )
                if fp_issues:
                    issues.append(
                        f"{benchmark_id}/d{dimension}/seed{seed}: initial-design fingerprint mismatch "
                        f"vs persisted design {auth['fingerprint'][:12]}: " + ", ".join(fp_issues)
                    )
                if not config_match:
                    issues.append(
                        f"{benchmark_id}/d{dimension}/seed{seed}: stored initial configs do not match "
                        "the persisted authoritative design rows"
                    )
                # M8A-R1: verify the OBSERVED initial objectives against freshly
                # recomputed values of the authoritative rows (deterministic
                # problems only) — detects inconsistent records even when a
                # group has a single method (no cross-method base to compare).
                objective_match = None
                if objective_prover is not None:
                    recomputed = objective_prover(benchmark_id, dimension, seed, auth["rows"])
                    if recomputed is not None:  # None = noisy problem, skipped
                        objective_match = True
                        for m in methods:
                            stored_obj = [entry["objective"] for entry in by_method[m]]
                            for i, (stored, fresh) in enumerate(zip(stored_obj, recomputed)):
                                if abs(stored - fresh) > tie_tolerance:
                                    objective_match = False
                                    issues.append(
                                        f"{benchmark_id}/d{dimension}/seed{seed}: initial objective {i} "
                                        f"of {m} ({stored:.12g}) does not match the recomputed value "
                                        f"of the authoritative design row ({fresh:.12g})"
                                    )
                group_result["authoritative_match"] = (
                    config_match and not fp_issues and objective_match is not False
                )
                group_result["authoritative_fingerprint"] = auth["fingerprint"]
                if objective_match is not None:
                    group_result["authoritative_objectives_match"] = objective_match
        verified.append(group_result)
    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "groups": verified,
        "excluded_no_initial_design_phase": sorted(excluded_no_initial_phase),
        "policy_note": (
            "plain random search has no initial-design phase by construction; shared-initial-design "
            "fairness applies to the BO methods and the shared-start random search, verified from "
            "their stored records and (in explicit_shared mode) against the persisted authoritative "
            "design and its SHA-256 fingerprint"
        ),
    }
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

        # M8A-R1 Phase B: in explicit_shared mode, ONE authoritative design per
        # compatible comparison group (benchmark, dimension, seed) is ensured
        # BEFORE any run executes — persisted with a genuine SHA-256 fingerprint
        # and RELOADED unchanged on resume. Groups are derived from the specs of
        # search-method runs (reference "runs" have no group).
        self._designs: dict[tuple, dict] = {}
        if self.config.initial_design.mode == "explicit_shared":
            store = InitialDesignStore(self.config.output_dir)
            design_groups = sorted({
                (spec["benchmark_id"], spec["dimension"], spec["seed"])
                for spec in self.store.specs()
                if spec["architecture"] != "reference"
            })
            for (bid, dim, seed) in design_groups:
                benchmark = self.benchmarks.get(bid)
                problem = benchmark.problem(dim)
                rows, fp, _created = store.load_or_create(
                    (bid, dim, seed), problem,
                    size=self.config.initial_design.size,
                    seed=seed,
                    metadata={
                        "mode": "explicit_shared",
                        "sampler": self.config.initial_design.sampler,
                        "sharing_policy": self.config.initial_design.sharing_policy,
                    },
                )
                self._designs[(bid, dim, seed)] = {"rows": rows, "fingerprint": fp}

        pending = self.store.pending_specs()
        for spec_dict in pending:
            if spec_dict["status"] == RunStatus.SKIPPED.value:
                continue  # recorded omission with reason; never executed, never 'completed'
            self._execute_one(spec_dict, compatibility)

        records = self._load_all_search_records(compatibility)
        fairness = verify_shared_initialization(
            records,
            tie_tolerance=self.config.statistics.paired_tie_tolerance,
            authoritative_designs=self._designs if self.config.initial_design.mode == "explicit_shared" else None,
            objective_prover=self._recompute_authoritative_objectives if self.config.initial_design.mode == "explicit_shared" else None,
        )
        summary = self.summarize(fairness)
        return summary

    def _recompute_authoritative_objectives(self, benchmark_id: str, dimension: int | None,
                                            seed: int | None, rows: list[list[float]]) -> list[float] | None:
        """M8A-R1: recompute the objectives of the authoritative design rows on
        an INDEPENDENT fresh problem instance. Returns None for noisy problems
        (the comparison would be meaningless there)."""
        benchmark = self.benchmarks.get(benchmark_id)
        problem = benchmark.problem(dimension)
        if problem.observation_noise_std is not None:
            return None  # noisy problem: stored objectives are stochastic samples
        return [float(problem.evaluate(np.asarray(row, dtype=np.float64))) for row in rows]

    def _execute_one(self, spec_dict: dict, compatibility: dict) -> None:
        run_id = spec_dict["run_id"]
        self.store.set_status(run_id, RunStatus.RUNNING)
        try:
            # M8A-R1: manifest entries carry runner-only bookkeeping (e.g. the
            # status history) that is not part of RunSpec — validate against
            # the model's declared fields only.
            spec = RunSpec.model_validate({
                k: v for k, v in spec_dict.items() if k in RunSpec.model_fields
            })
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
                initial_design_configs = None
                design_fingerprint_value = None
                if self.config.initial_design.mode == "explicit_shared":
                    # M8A-R1 Phase B: use the AUTHORITATIVE persisted design for this
                    # comparison group (generated/validated once per experiment in
                    # execute(), reloaded unchanged on resume). The design rows are
                    # passed unchanged to every applicable method.
                    auth = self._designs[(spec.benchmark_id, spec.dimension, spec.seed)]
                    initial_design_configs = auth["rows"]
                    design_fingerprint_value = auth["fingerprint"]
                started = time.perf_counter()
                out = algorithm.run(
                    benchmark=benchmark, dimension=spec.dimension, seed=spec.seed,
                    config=engine_config,
                    reference_value=reference.objective_value,
                    reference_method=reference.verification_method,
                    initial_design_configs=initial_design_configs,
                )
                wall = time.perf_counter() - started
                if out.architecture in ("random_search", "random_search_shared_start"):
                    record = self._random_search_record(
                        spec_dict, algorithm, compatibility, out, reference, wall,
                        design_fingerprint=design_fingerprint_value,
                    )
                    budget = self.config.stopping.evaluation_budget
                    attempted = record["evaluation_counts"]["n_objective_evaluations"]
                    record["early_completion"] = self._classify_completion(record, attempted, budget)
                    if record["early_completion"] == "violation":
                        raise AssertionError(
                            f"budget violation in {run_id}: attempted {attempted} > budget {budget}"
                        )
                else:
                    record = build_run_record(
                        spec_dict, algorithm.version, compatibility, out.optimization_result, reference, wall
                    )
                    record["initial_design_fingerprint"] = design_fingerprint_value
                    record["initial_design_source"] = (
                        "persisted_shared_design" if design_fingerprint_value else "seeded_sampler"
                    )
                    # Per-run validity checks (M8A-R1 Phase C: legitimate early
                    # stopping is NOT a budget violation; over-budget still is).
                    budget = self.config.stopping.evaluation_budget
                    attempted = (record["evaluation_counts"]["n_initial_evaluations"]
                                 + record["evaluation_counts"]["n_bo_evaluations"])
                    record["early_completion"] = self._classify_completion(record, attempted, budget)
                    if record["early_completion"] == "violation":
                        raise AssertionError(
                            f"budget violation in {run_id}: attempted {attempted} > budget {budget}"
                        )
                    if record["early_completion"] == "engine_failure":
                        self.store.save_run_record(run_id, record)
                        self.store.set_status(
                            run_id, RunStatus.FAILED,
                            reason=f"engine stopped: {record['stopping_reason']}",
                        )
                        return
            self.store.save_run_record(run_id, record)
            self.store.set_status(run_id, RunStatus.COMPLETED)
        except RecordRejected:
            raise
        except Exception as exc:  # recorded failure with reason (item 12)
            self.store.set_status(run_id, RunStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")

    def _classify_completion(self, record: dict, attempted: int, budget: int) -> str:
        """M8A-R1 Phase C: classify one finished run.

        Returns one of:
        * ``"full"``            — budget fully exhausted (normal completion);
        * ``"early"``           — legitimate early stop (configured stopping
          condition fired with evaluations left); the record keeps
          ``stopping_reason`` and stays COMPLETED;
        * ``"engine_failure"``  — the engine reported a failure-class stop
          (no feasible initial design / repeated GP or objective failures);
        * ``"violation"``       — MORE evaluations than the budget (hard error).
        """
        reason = record.get("stopping_reason")
        if attempted > budget:
            return "violation"
        if attempted == budget:
            return "full"
        if reason in ENGINE_FAILURE_STOP_REASONS:
            return "engine_failure"
        if reason in LEGITIMATE_EARLY_STOP_REASONS:
            return "early"
        # under-budget with an UNKNOWN reason: keep the run visible and
        # classified, never silently accepted as a full completion
        return "early"

    def _random_search_record(self, spec_dict, algorithm, compatibility, out, reference, wall,
                              design_fingerprint: str | None = None) -> dict:
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
        # M8A-R1: the shared-start variant records the authoritative initial
        # phase (same fields as BO records) so fairness verification can include
        # it; plain random search has an empty initial phase by construction.
        initial_design = [
            {"config": config_list, "objective": objective, "evaluation_id": evaluation_id}
            for config_list, objective, evaluation_id in getattr(result, "initial_phase", [])
        ]
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
            "initial_design": initial_design,
            "initial_design_fingerprint": design_fingerprint if initial_design else None,
            "initial_design_source": (
                "persisted_shared_design" if initial_design and design_fingerprint
                else "seeded_sampler" if initial_design else None
            ),
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
