"""Generic Bayesian-optimization loop — joint and block-restricted architectures
(M5 instructions 21-24; M6 instructions 4-18).

This is the SINGLE generic engine: search strategies are configurations of
variable blocks + a deterministic cyclic scheduler, never separate
implementations (M6 instruction 4). The M5 joint baseline is the special case
of ONE block containing every variable; with the default configuration
(``variable_blocks=None``, ``evaluations_per_block_visit=1``, no cycle
patience) the engine reproduces the M5 joint behavior exactly (regression
tested).

Logical sequence per BLOCK VISIT (instruction 21; M6 instructions 6, 13):

1. validate the problem and optimizer configuration (constructor)
2. generate and evaluate the initial design (seeded feasible full-dimension
   sampling — identical across all architectures for a given seed, M6
   instruction 21)
3. construct the successful-observation dataset (failures logged separately)
4. determine the active block (deterministic cyclic schedule) and, for block
   search, generate feasible COMPLETE candidate vectors around the CURRENT
   CONFIGURATION (fixed coordinates preserved exactly; full-vector feasibility
   validation — shared constraints may span blocks)
5. fit the shared Gaussian process on ALL successful observations
   (M x d FULL dimension regardless of block size — fresh refit per
   observation; no warm-start)
6. determine the GLOBAL INCUMBENT (best successfully OBSERVED value)
7. build LogExpectedImprovement (``best_f`` = incumbent, original units)
8. score candidates with the fitted GP (the GP NEVER calls the true objective)
9. select one candidate (argmax finite log-EI; first-index tie-break)
10. evaluate the TRUE synthetic objective
11. append the successful observation to the SHARED dataset (failures logged
    separately with block metadata)
12. update the CURRENT CONFIGURATION per the improvement-only acceptance
    policy, and the GLOBAL INCUMBENT independently (only from evaluated
    observations — never from a GP prediction)
13. record iteration diagnostics (block id, cycle number, within-cycle index)
14. check stopping conditions (recorded exactly; termination is NEVER
    described as proof of global convergence)
15. repeat when permitted.

CURRENT CONFIGURATION vs GLOBAL INCUMBENT (M6 instruction 7): the current
configuration is the feasible complete vector around which block candidates
are constructed; it changes ONLY through the improvement-only acceptance
policy (a proposed implementation choice). The global incumbent is the best
successfully evaluated feasible configuration observed during the entire run.
The two objects are maintained independently and may differ (exploratory
evaluations that improve the incumbent by less than the acceptance tolerance
update the incumbent but not the current configuration; both always refer to
actual stored observations).

Evaluation-budget accounting (M5 instruction 22 / M6 instruction 18; the
provisional synthetic budget 100 is inherited from M5): the initial
observations and all subsequent attempted evaluations consume the SAME
declared total budget; FAILED attempts also consume it (failures are real
evaluation costs and counting them prevents unlimited failure loops);
acquisition candidate scores consume NO budget; no independent
final-validation evaluations are performed. Consecutive-failure caps prevent
repeated failures from looping forever.

Block-skip semantics (M6 instruction 9): an empty feasible pool for ONE block
is recorded (skipped-block log with reasons) and the schedule moves on — an
empty individual block does NOT terminate the run while other blocks remain
searchable. Only a COMPLETE cycle in which every block was skipped stops the
engine (``all_blocks_exhausted_in_cycle``), because no searchable block
remains. For the JOINT full-space block an empty pool stops immediately
(``candidate_pool_exhausted`` — M5 semantics, regression-preserved). Block-wise
stagnation never proves global optimality.

Stopping conditions (M5 instruction 23 / M6 instruction 17): total evaluation
budget, maximum BO iteration count, configured runtime limit, verified
improvement tolerance with iteration patience, configurable cycle-level
patience, exhaustion of feasible candidates (full space or all blocks in a
cycle), GP-fitting failure, repeated objective failures.
"""

from __future__ import annotations

import time

import numpy as np

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.acquisition import (
    AcquisitionManager,
    SelectionFailureError,
    verify_installed_log_ei_signature,
)
from radar_bo.optimization.bo.block_candidates import BlockCandidateGenerator
from radar_bo.optimization.bo.blocks import (
    BlockDefinitionError,
    VariableBlockSpec,
    resolve_blocks,
    validate_blocks,
)
from radar_bo.optimization.bo.candidates import CandidateGenerator
from radar_bo.optimization.bo.dataset import ObservationDataset
from radar_bo.optimization.bo.problem import OptimizationProblem
from radar_bo.optimization.bo.results import IterationDiagnostics, OptimizationResult
from radar_bo.optimization.bo.scheduling import CyclicBlockScheduler
from radar_bo.optimization.bo.surrogate import GPSurrogate, GPSurrogateError

#: Consecutive objective-evaluation failures tolerated before stopping
#: (prevents repeated failures from creating an unlimited loop).
MAX_CONSECUTIVE_OBJECTIVE_FAILURES = 3
#: Consecutive GP-fitting failures tolerated before stopping.
MAX_CONSECUTIVE_GP_FAILURES = 3


def dependency_versions() -> dict[str, str]:
    """Installed library versions recorded with every result."""
    import botorch
    import gpytorch
    import numpy
    import scipy
    import torch

    return {
        "python_numpy": numpy.__version__,
        "torch": torch.__version__,
        "gpytorch": gpytorch.__version__,
        "botorch": botorch.__version__,
        "scipy": scipy.__version__,
    }


class BayesianOptimizer:
    """Reusable generic BO engine: block-cyclic visits, LogEI, float64 CPU.

    Joint search is the single-full-block configuration (M5 baseline).
    Alternating (two blocks) and pair-wise (multiple two-variable blocks)
    search reuse the SAME dataset, surrogate, acquisition, budget accounting
    and stopping conditions (M6 instructions 4, 10, 11, 16).
    """

    ALGORITHM_ID = "bo_single_candidate_log_ei_matern52"  # one candidate per evaluation in ALL modes; mode via search_strategy

    def __init__(
        self,
        problem: OptimizationProblem,
        settings: OptimizationConfig,
        reference_objective: float | None = None,
        reference_method: str | None = None,
        seed: int | None = None,
    ) -> None:
        # ---- 1. Validate problem and configuration ---------------------------
        if not isinstance(problem, OptimizationProblem):
            raise TypeError(f"problem must be an OptimizationProblem; got {type(problem)!r}.")
        if settings.kernel != "matern_5_2":
            raise NotImplementedError(
                f"M5 supports the explicitly configured Matern-5/2 kernel only (requested: {settings.kernel})."
            )
        if settings.surrogate != "single_task_gp":
            raise NotImplementedError(f"M5 supports SingleTaskGP only (requested: {settings.surrogate}).")
        if settings.acquisition != "log_expected_improvement":
            raise NotImplementedError(
                f"M5 uses conventional LogExpectedImprovement only (requested: {settings.acquisition}); "
                "noisy-objective acquisition is out of M5 scope."
            )
        if settings.precision != "float64" or settings.device != "cpu":
            raise NotImplementedError("M5 requires float64 precision on CPU (README section 2).")
        if not settings.input_normalization or not settings.outcome_standardization:
            raise NotImplementedError(
                "M5 always uses BoTorch Normalize/Standardize transforms; disabling them is not supported."
            )
        verify_installed_log_ei_signature()  # installed-API check (no xi parameter)
        settings.ensure_ready()  # evaluation_budget must be set (no invented defaults)

        # ---- M6: resolve and validate the variable-block structure ------------
        specs: list[VariableBlockSpec] | None = None
        if settings.variable_blocks is not None:
            specs = [VariableBlockSpec(block_id=bid, active_indices=tuple(idxs)) for bid, idxs in settings.variable_blocks]
        self.blocks = resolve_blocks(problem, specs)
        coverage = validate_blocks(self.blocks, problem, require_full_coverage=False)
        self.uncovered_variable_indices: list[int] = coverage["uncovered_indices"]

        # The DECLARED search_mode must be structurally compatible with the
        # blocks (instruction 6): joint = one full-dimension block (declared by
        # leaving variable_blocks at None); alternating = exactly two blocks of
        # any sizes; pair-wise ('coordinate') = at least two blocks, each with
        # exactly two variables. Two two-variable blocks match BOTH readings;
        # the declared mode resolves the ambiguity.
        if specs is None:
            compatible_modes = {"joint"}
        elif len(self.blocks) == 1:
            raise BlockDefinitionError(
                "variable_blocks resolves to a single block; use variable_blocks=None for joint search "
                "(a single partial block matches no search mode)."
            )
        else:
            compatible_modes = {"alternating"}
            if all(b.block_dimension == 2 for b in self.blocks):
                compatible_modes.add("coordinate")
        if settings.search_mode not in compatible_modes:
            raise BlockDefinitionError(
                f"search_mode='{settings.search_mode}' is incompatible with the variable_blocks structure "
                f"(compatible modes: {sorted(compatible_modes)})."
            )
        self.search_strategy = {"joint": "joint", "alternating": "alternating", "coordinate": "pairwise"}[
            settings.search_mode
        ]

        self.problem = problem
        self.settings = settings
        self.reference_objective = reference_objective
        self.reference_method = reference_method
        self.seed = int(seed if seed is not None else (settings.random_seed if settings.random_seed is not None else 0))

    # ---- helpers ---------------------------------------------------------------
    def _gap(self, best_value: float) -> float | None:
        """Signed gap to the reference: reference - best when maximizing (positive
        means the reference is still better)."""
        if self.reference_objective is None:
            return None
        if self.problem.maximize:
            return float(self.reference_objective - best_value)
        return float(best_value - self.reference_objective)

    def _improvement(self, previous: float, current: float) -> float:
        return float(current - previous) if self.problem.maximize else float(previous - current)

    # ---- main loop ----------------------------------------------------------------
    def optimize(self) -> OptimizationResult:
        problem = self.problem
        settings = self.settings
        started = time.perf_counter()
        budget_remaining = int(settings.evaluation_budget)
        dataset = ObservationDataset(
            problem_id=problem.problem_id,
            dimension=problem.dimension,
            maximize=problem.maximize,
        )
        allow_duplicates_in_dataset = problem.observation_noise_std is not None
        design_generator = CandidateGenerator(
            problem, num_candidates=settings.num_initial_observations, seed=self.seed, exclude_evaluated=True
        )
        candidate_generator = CandidateGenerator(
            problem,
            num_candidates=settings.num_acquisition_candidates,
            seed=self.seed + 1,
            exclude_evaluated=not allow_duplicates_in_dataset,
        )
        # M6: one block-restricted generator per PROPER block (decorrelated seeds;
        # pools re-seed per call for deterministic reproducibility, like M5).
        block_generators: dict[str, BlockCandidateGenerator] = {
            block.block_id: BlockCandidateGenerator(
                problem, block,
                num_candidates=settings.num_acquisition_candidates,
                seed=self.seed + 1 + position,
            )
            for position, block in enumerate(self.blocks)
            if not block.covers_all
        }
        scheduler = CyclicBlockScheduler(self.blocks)
        surrogate = GPSurrogate(problem.dimension, seed=self.seed)  # SHARED full-dimension GP
        manager: AcquisitionManager | None = None

        n_initial_evaluations = 0
        n_bo_evaluations = 0
        n_failed_attempts = 0
        n_acquisition_scores = 0
        gp_fit_runtime = 0.0
        acquisition_runtime = 0.0
        objective_runtime = 0.0
        candidate_generation_runtime = 0.0
        constraint_validation_runtime = 0.0
        incumbent_history: list[float] = []
        diagnostics: list[IterationDiagnostics] = []
        stopping_reason: str | None = None

        # M6 counters (instruction 17: distinguish evaluations/iterations/visits/cycles/skips)
        n_block_visits = 0
        n_cycles_completed = 0
        n_skipped_blocks = 0
        n_accepted_updates = 0
        n_cycles_without_improvement = 0
        skipped_block_records: list[dict] = []
        current_config: np.ndarray | None = None
        current_value: float | None = None

        # ---- 2-3. Initial design (consumes the same budget; instruction 22) ----
        initial_pool, pool_report = design_generator.generate_pool()
        pool_stats = {
            "pool_requests": 1,
            "pool_requested_total": pool_report.requested_size,
            "pool_delivered_total": pool_report.final_size,
            "pool_rejected_infeasible_total": pool_report.rejected_infeasible,
            "pool_rejected_duplicate_within_pool_total": pool_report.rejected_duplicate_within_pool,
            "pool_rejected_duplicate_evaluated_total": pool_report.rejected_duplicate_evaluated,
        }
        candidate_generation_runtime += pool_report.generation_duration_s
        constraint_validation_runtime += pool_report.feasibility_validation_s
        if initial_pool.shape[0] == 0:
            stopping_reason = "no_feasible_initial_design"
        for x in initial_pool:
            if n_initial_evaluations >= settings.num_initial_observations or budget_remaining <= 0:
                break
            evaluation_id = problem.next_evaluation_id()
            budget_remaining -= 1
            n_initial_evaluations += 1
            t0 = time.perf_counter()
            try:
                value = problem.evaluate(x)
            except Exception as exc:
                objective_runtime += time.perf_counter() - t0
                n_failed_attempts += 1
                dataset.record_failure(evaluation_id, f"{type(exc).__name__}: {exc}", "initial_design", x, str(self.seed))
                continue
            objective_runtime += time.perf_counter() - t0
            dataset.insert(
                x, value, evaluation_id,
                seed_info=str(self.seed),
                observation_noise_std=problem.observation_noise_std,
                metadata={"stage": "initial_design"},
                allow_duplicates=False,
            )
        if dataset.size > 0:
            _idx, best_initial = dataset.best()
            incumbent_history.append(float(best_initial.objective))
            # Current configuration starts at the best INITIAL observation
            # (identical across architectures for equal seeds — instruction 21).
            current_config = np.asarray(best_initial.config, dtype=np.float64).copy()
            current_value = float(best_initial.objective)
        if dataset.size == 0 and stopping_reason is None:
            stopping_reason = "no_feasible_initial_design"

        consecutive_objective_failures = 0
        consecutive_gp_failures = 0
        consecutive_acquisition_failures = 0
        iteration = 0
        no_improvement_count = 0

        # ---- 4-15. Block-cyclic main loop ------------------------------------------
        while stopping_reason is None:
            if budget_remaining <= 0:
                stopping_reason = "evaluation_budget_exhausted"
                break
            if settings.max_iterations is not None and iteration >= settings.max_iterations:
                stopping_reason = "max_iterations_reached"
                break
            if settings.runtime_budget_s is not None and (time.perf_counter() - started) >= settings.runtime_budget_s:
                stopping_reason = "runtime_limit_reached"
                break

            # M6: scheduling — which block is searched next (deterministic cyclic).
            ticket = scheduler.next_visit()
            block = ticket.block
            if ticket.within_cycle_index == 0:
                skipped_in_current_cycle: set[str] = set()
                cycle_start_incumbent = incumbent_history[-1] if incumbent_history else None
            n_block_visits += 1

            visit_ended_by_empty_pool = False
            # M6 instruction 13: evaluations per block visit (default 1; all inside
            # the shared global budget). One acquisition-selected evaluation is NOT
            # an exact maximization of the block.
            for _visit_attempt in range(settings.evaluations_per_block_visit):
                if budget_remaining <= 0:
                    stopping_reason = "evaluation_budget_exhausted"
                    break

                # 4. Fit the SHARED GP on all successful observations (fresh refit
                #    per observation; full M x d dimension regardless of block size).
                try:
                    t0 = time.perf_counter()
                    surrogate.fit(dataset.training_inputs(), dataset.training_outputs())
                    gp_fit_runtime += time.perf_counter() - t0
                except GPSurrogateError as exc:
                    consecutive_gp_failures += 1
                    dataset.record_failure(
                        f"{problem.problem_id}_gp_fit_{iteration}", f"{exc}", "gp_fitting", None, str(self.seed)
                    )
                    if consecutive_gp_failures >= MAX_CONSECUTIVE_GP_FAILURES:
                        stopping_reason = "gp_fitting_failure"
                        break
                    continue  # no budget consumed: no objective evaluation was attempted
                consecutive_gp_failures = 0

                # 5. Incumbent: best successfully OBSERVED value (never a GP prediction).
                incumbent_index, incumbent_record = dataset.best()
                incumbent_value = float(incumbent_record.objective)

                # 7. Feasible candidate pool for the ACTIVE block (zero objective
                #    evaluations). Joint block: full-space generator (M5 path).
                #    Proper block: fixed coordinates from the CURRENT CONFIGURATION,
                #    active coordinates sampled, COMPLETE vectors validated.
                if block.covers_all:
                    candidates, report = candidate_generator.generate_pool(dataset.configs())
                else:
                    candidates, report = block_generators[block.block_id].generate_pool(
                        current_config, dataset.configs()
                    )
                pool_stats["pool_requests"] += 1
                pool_stats["pool_requested_total"] += report.requested_size
                pool_stats["pool_delivered_total"] += report.final_size
                pool_stats["pool_rejected_infeasible_total"] += report.rejected_infeasible
                pool_stats["pool_rejected_duplicate_within_pool_total"] += report.rejected_duplicate_within_pool
                pool_stats["pool_rejected_duplicate_evaluated_total"] += report.rejected_duplicate_evaluated
                candidate_generation_runtime += report.generation_duration_s
                constraint_validation_runtime += report.feasibility_validation_s
                if candidates.shape[0] == 0:
                    if block.covers_all:
                        # M5 semantics (regression-preserved): the full-space pool is empty.
                        stopping_reason = "candidate_pool_exhausted"
                        break
                    # M6 instruction 9: skip THIS block; other blocks may remain searchable.
                    visit_ended_by_empty_pool = True
                    break

                # 6+8. Build LogEI and score the pool (GP never calls the objective).
                t0 = time.perf_counter()
                try:
                    if manager is None or manager.surrogate is not surrogate:
                        manager = AcquisitionManager(surrogate, maximize=problem.maximize)
                    acquisition = manager.build(incumbent_value)
                    scores, finite_mask = manager.score_candidates(acquisition, candidates)
                except (GPSurrogateError, SelectionFailureError) as exc:
                    acquisition_runtime += time.perf_counter() - t0
                    consecutive_acquisition_failures += 1
                    dataset.record_failure(
                        f"{problem.problem_id}_acq_{iteration}", f"{type(exc).__name__}: {exc}",
                        "acquisition_scoring", None, str(self.seed),
                    )
                    if consecutive_acquisition_failures >= MAX_CONSECUTIVE_GP_FAILURES:
                        stopping_reason = "gp_fitting_failure"
                        break
                    continue  # no budget consumed
                acquisition_runtime += time.perf_counter() - t0
                consecutive_acquisition_failures = 0
                n_acquisition_scores += int(candidates.shape[0])
                selected_index = AcquisitionManager.select_best(scores, finite_mask)
                x_selected = candidates[selected_index]

                # 9-11. Evaluate the TRUE objective (consumes budget; failures too).
                evaluation_id = problem.next_evaluation_id()
                budget_remaining -= 1
                n_bo_evaluations += 1
                t0 = time.perf_counter()
                try:
                    value = problem.evaluate(x_selected)
                except Exception as exc:
                    objective_runtime += time.perf_counter() - t0
                    n_failed_attempts += 1
                    consecutive_objective_failures += 1
                    dataset.record_failure(
                        evaluation_id, f"{type(exc).__name__}: {exc}", "objective_evaluation",
                        x_selected, str(self.seed),
                        metadata={"block_id": block.block_id, "cycle_number": str(ticket.cycle_number)},
                    )
                    diagnostics.append(
                        IterationDiagnostics(
                            iteration=iteration, n_observations=dataset.size,
                            incumbent_objective=incumbent_value, gap_to_reference=self._gap(incumbent_value),
                            selected_config=[float(v) for v in x_selected], selected_evaluation_id=evaluation_id,
                            objective_value=None, improvement=None,
                            gp_fit_duration_s=0.0, acquisition_duration_s=0.0,
                            n_candidates_scored=int(candidates.shape[0]),
                            block_id=block.block_id, cycle_number=ticket.cycle_number,
                            within_cycle_index=ticket.within_cycle_index, accepted_current_config=False,
                        )
                    )
                    if consecutive_objective_failures >= MAX_CONSECUTIVE_OBJECTIVE_FAILURES:
                        stopping_reason = "repeated_objective_failures"
                        break
                    break  # end this visit (failure); next visit is scheduled
                objective_runtime += time.perf_counter() - t0
                consecutive_objective_failures = 0

                previous_incumbent = incumbent_value
                dataset.insert(
                    x_selected, value, evaluation_id,
                    seed_info=str(self.seed),
                    observation_noise_std=problem.observation_noise_std,
                    metadata={
                        "stage": "bo_iteration", "iteration": str(iteration),
                        "block_id": block.block_id,
                        "cycle_number": str(ticket.cycle_number),
                        "within_cycle_index": str(ticket.within_cycle_index),
                    },
                    allow_duplicates=allow_duplicates_in_dataset,
                )
                new_incumbent_value = float(dataset.best()[1].objective)
                improvement = self._improvement(previous_incumbent, new_incumbent_value)
                incumbent_history.append(new_incumbent_value)

                # 12. CURRENT-CONFIGURATION acceptance (M6 instruction 14;
                #     improvement-only policy — a PROPOSED implementation choice).
                #     The global incumbent is updated independently above (dataset best).
                acceptance_tolerance = settings.improvement_tolerance if settings.improvement_tolerance is not None else 0.0
                accepted = self._improvement(current_value, float(value)) > acceptance_tolerance
                if accepted:
                    current_config = np.asarray(x_selected, dtype=np.float64).copy()
                    current_value = float(value)
                    n_accepted_updates += 1

                # 13. Iteration diagnostics.
                iteration += 1
                diagnostics.append(
                    IterationDiagnostics(
                        iteration=iteration, n_observations=dataset.size,
                        incumbent_objective=new_incumbent_value,
                        gap_to_reference=self._gap(new_incumbent_value),
                        selected_config=[float(v) for v in x_selected], selected_evaluation_id=evaluation_id,
                        objective_value=float(value), improvement=float(improvement),
                        gp_fit_duration_s=0.0, acquisition_duration_s=0.0,
                        n_candidates_scored=int(candidates.shape[0]),
                        block_id=block.block_id, cycle_number=ticket.cycle_number,
                        within_cycle_index=ticket.within_cycle_index, accepted_current_config=bool(accepted),
                    )
                )

                # 14. Verified improvement tolerance / patience (instruction 23).
                tolerance = settings.improvement_tolerance if settings.improvement_tolerance is not None else 0.0
                if improvement > tolerance:
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if settings.patience is not None and no_improvement_count >= settings.patience:
                        stopping_reason = "patience_without_improvement"
                        break

            if stopping_reason is not None:
                break  # a stopping condition fired inside the visit

            # ---- M6 instruction 9: skipped-block bookkeeping ----------------------
            if visit_ended_by_empty_pool:
                n_skipped_blocks += 1
                skipped_block_records.append({
                    "block_id": block.block_id,
                    "cycle_number": ticket.cycle_number,
                    "within_cycle_index": ticket.within_cycle_index,
                    "reason": "empty_feasible_pool_for_block",
                })
                skipped_in_current_cycle.add(block.block_id)

            # ---- M6 instruction 17: cycle-level bookkeeping -----------------------
            if ticket.within_cycle_index == len(self.blocks) - 1:
                n_cycles_completed += 1
                if len(skipped_in_current_cycle) == len(self.blocks):
                    # Every block in a full cycle had no feasible candidates:
                    # no searchable block remains (stagnation is NOT optimality).
                    stopping_reason = "all_blocks_exhausted_in_cycle"
                    break
                if settings.cycle_patience is not None:
                    cycle_improved = (
                        incumbent_history
                        and cycle_start_incumbent is not None
                        and self._improvement(cycle_start_incumbent, incumbent_history[-1]) > 0.0
                    ) or any(
                        d.accepted_current_config
                        for d in diagnostics
                        if d.cycle_number == ticket.cycle_number
                    )
                    if cycle_improved:
                        n_cycles_without_improvement = 0
                    else:
                        n_cycles_without_improvement += 1
                        if n_cycles_without_improvement >= settings.cycle_patience:
                            stopping_reason = "cycle_patience_exhausted"
                            break

        if stopping_reason is None:  # defensive: the loop must always record a reason
            stopping_reason = "evaluation_budget_exhausted"

        # ---- Result (incumbent is always an EVALUATED configuration) ------------
        if dataset.size > 0:
            _best_index, best_record = dataset.best()
            best_config: list[float] | None = [float(v) for v in best_record.config]
            best_value: float | None = float(best_record.objective)
        else:
            # No successful evaluation exists: there is NO incumbent — recorded
            # honestly as None (never replaced by an unevaluated GP prediction).
            best_config = None
            best_value = None
        return OptimizationResult(
            problem_id=problem.problem_id,
            algorithm_id=self.ALGORITHM_ID,
            seed=self.seed,
            initial_design_size=settings.num_initial_observations,
            n_initial_evaluations=n_initial_evaluations,
            n_bo_evaluations=n_bo_evaluations,
            n_failed_attempts=n_failed_attempts,
            n_acquisition_scores=n_acquisition_scores,
            pool_requests=pool_stats["pool_requests"],
            pool_requested_total=pool_stats["pool_requested_total"],
            pool_delivered_total=pool_stats["pool_delivered_total"],
            pool_rejected_infeasible_total=pool_stats["pool_rejected_infeasible_total"],
            pool_rejected_duplicate_within_pool_total=pool_stats["pool_rejected_duplicate_within_pool_total"],
            pool_rejected_duplicate_evaluated_total=pool_stats["pool_rejected_duplicate_evaluated_total"],
            best_config=best_config,
            best_objective=best_value,
            reference_objective=self.reference_objective,
            reference_method=self.reference_method,
            best_gap_to_reference=self._gap(best_value) if best_value is not None else None,
            incumbent_history=incumbent_history,
            observations=dataset.records,
            failures=dataset.failures,
            iteration_diagnostics=diagnostics,
            gp_fit_runtime_s=gp_fit_runtime,
            acquisition_runtime_s=acquisition_runtime,
            objective_runtime_s=objective_runtime,
            candidate_generation_runtime_s=candidate_generation_runtime,
            constraint_validation_runtime_s=constraint_validation_runtime,
            total_runtime_s=time.perf_counter() - started,
            stopping_reason=stopping_reason,
            settings=settings.model_dump(),
            dependency_versions=dependency_versions(),
            search_strategy=self.search_strategy,
            variable_blocks=[b.describes() for b in self.blocks],
            uncovered_variable_indices=list(self.uncovered_variable_indices),
            current_config_final=None if current_config is None else [float(v) for v in current_config],
            current_config_value_final=current_value,
            n_accepted_updates=n_accepted_updates,
            n_block_visits=n_block_visits,
            n_cycles_completed=n_cycles_completed,
            n_skipped_blocks=n_skipped_blocks,
            skipped_block_records=skipped_block_records,
            n_cycles_without_improvement=n_cycles_without_improvement,
            evaluations_per_block_visit=settings.evaluations_per_block_visit,
        )
