"""Generic single-candidate Bayesian-optimization loop (M5 instructions 21-24).

Logical sequence per iteration (instruction 21):

1. validate the problem and optimizer configuration
2. generate and evaluate the initial design (seeded feasible sampling)
3. construct the successful-observation dataset (failures logged separately)
4. fit the Gaussian process (fresh refit per observation; no warm-start)
5. determine the incumbent objective (max successfully OBSERVED value)
6. build LogExpectedImprovement (``best_f`` = incumbent, original units)
7. generate feasible candidates (500 provisional; zero objective evaluations)
8. score candidates with the fitted GP (the GP NEVER calls the true objective)
9. select one candidate (argmax finite log-EI; first-index tie-break)
10. evaluate the TRUE synthetic objective
11. append the successful observation (failures logged separately)
12. update the incumbent (only from evaluated observations — never from a
    GP prediction, even a higher one)
13. record iteration diagnostics
14. check stopping conditions (recorded exactly; termination is NEVER
    described as proof of global convergence)
15. repeat when permitted.

Evaluation-budget accounting (instruction 22, provisional synthetic budget
100): the initial observations and subsequent BO evaluations both consume the
same budget; FAILED objective-evaluation attempts also consume it (documented
policy: failures are real evaluation costs, and counting them prevents
unlimited failure loops); acquisition candidate scores consume NO budget; no
independent final-validation evaluations are performed by the engine. A
consecutive-failure cap prevents repeated failures from looping forever.

Stopping conditions (instruction 23): total evaluation budget, maximum
iteration count, configured runtime limit, verified improvement tolerance
with patience without improvement, exhaustion of feasible unevaluated
candidates, GP-fitting failure, repeated objective failures.
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
from radar_bo.optimization.bo.candidates import CandidateGenerator
from radar_bo.optimization.bo.dataset import ObservationDataset
from radar_bo.optimization.bo.problem import OptimizationProblem
from radar_bo.optimization.bo.results import IterationDiagnostics, OptimizationResult
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
    """Reusable generic BO engine: one candidate per iteration, LogEI, float64 CPU."""

    ALGORITHM_ID = "bo_single_candidate_log_ei_matern52"

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
        surrogate = GPSurrogate(problem.dimension, seed=self.seed)
        manager: AcquisitionManager | None = None

        n_initial_evaluations = 0
        n_bo_evaluations = 0
        n_failed_attempts = 0
        n_acquisition_scores = 0
        gp_fit_runtime = 0.0
        acquisition_runtime = 0.0
        objective_runtime = 0.0
        incumbent_history: list[float] = []
        diagnostics: list[IterationDiagnostics] = []
        stopping_reason: str | None = None

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
            incumbent_history.append(float(dataset.best()[1].objective))
        if dataset.size == 0 and stopping_reason is None:
            stopping_reason = "no_feasible_initial_design"

        consecutive_objective_failures = 0
        consecutive_gp_failures = 0
        consecutive_acquisition_failures = 0
        iteration = 0
        no_improvement_count = 0

        # ---- 4-15. Main loop ------------------------------------------------------
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

            # 4. Fit the GP (fresh refit per observation).
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

            # 7. Feasible candidate pool (zero objective evaluations).
            candidates, report = candidate_generator.generate_pool(dataset.configs())
            pool_stats["pool_requests"] += 1
            pool_stats["pool_requested_total"] += report.requested_size
            pool_stats["pool_delivered_total"] += report.final_size
            pool_stats["pool_rejected_infeasible_total"] += report.rejected_infeasible
            pool_stats["pool_rejected_duplicate_within_pool_total"] += report.rejected_duplicate_within_pool
            pool_stats["pool_rejected_duplicate_evaluated_total"] += report.rejected_duplicate_evaluated
            if candidates.shape[0] == 0:
                stopping_reason = "candidate_pool_exhausted"
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
                dataset.record_failure(evaluation_id, f"{type(exc).__name__}: {exc}", "objective_evaluation",
                                       x_selected, str(self.seed))
                diagnostics.append(
                    IterationDiagnostics(
                        iteration=iteration, n_observations=dataset.size,
                        incumbent_objective=incumbent_value, gap_to_reference=self._gap(incumbent_value),
                        selected_config=[float(v) for v in x_selected], selected_evaluation_id=evaluation_id,
                        objective_value=None, improvement=None,
                        gp_fit_duration_s=0.0, acquisition_duration_s=0.0,
                        n_candidates_scored=int(candidates.shape[0]),
                    )
                )
                if consecutive_objective_failures >= MAX_CONSECUTIVE_OBJECTIVE_FAILURES:
                    stopping_reason = "repeated_objective_failures"
                    break
                continue
            objective_runtime += time.perf_counter() - t0
            consecutive_objective_failures = 0

            previous_incumbent = incumbent_value
            dataset.insert(
                x_selected, value, evaluation_id,
                seed_info=str(self.seed),
                observation_noise_std=problem.observation_noise_std,
                metadata={"stage": "bo_iteration", "iteration": str(iteration)},
                allow_duplicates=allow_duplicates_in_dataset,
            )
            new_incumbent_value = float(dataset.best()[1].objective)
            improvement = self._improvement(previous_incumbent, new_incumbent_value)
            incumbent_history.append(new_incumbent_value)

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
            total_runtime_s=time.perf_counter() - started,
            stopping_reason=stopping_reason,
            settings=settings.model_dump(),
            dependency_versions=dependency_versions(),
        )
