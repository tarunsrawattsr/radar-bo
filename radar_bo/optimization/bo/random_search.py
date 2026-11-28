"""Generic feasible random-search benchmark (M5 instruction 7).

A comparison method — explicitly NOT an implementation of Bayesian
Optimization. It uses the same objective interface, respects all
mathematical constraints, accepts an explicit random seed, operates under
the same total evaluation budget as BO, records every successful objective
evaluation, and returns the best evaluated feasible configuration. There is
no separate initial design: the initial (and only) sampling is part of the
same budget, so the initial observation cost is counted in the total budget.

Documented duplicate policy (instruction 9): for deterministic problems a
sampled configuration already evaluated is SKIPPED before evaluation (it
does not consume budget); for noisy problems repeats may legitimately be
evaluated (controlled by the problem's observation_noise_std declaration).

M8A-R1 (Phase B): an OPTIONAL shared-start mode (``initial_configs``) makes
the same implementation usable as a separately identified comparison method
(``random_search_shared_start`` in the experiment framework): the
authoritative shared design is evaluated FIRST — exactly the initial phase
the BO methods see — and the remaining budget is uniform random feasible
sampling. Plain random search (``initial_configs=None``) is unchanged and
remains the independent comparator.
"""

from __future__ import annotations

import numpy as np

from radar_bo.optimization.bo.dataset import DUPLICATE_TOLERANCE
from radar_bo.optimization.bo.problem import OptimizationProblem


class RandomSearchResult:
    """Plain result container for the random-search benchmark."""

    def __init__(
        self,
        problem_id: str,
        algorithm_id: str,
        seed: int,
        best_config: list[float],
        best_objective: float,
        history: list[tuple[list[float], float, str]],
        n_objective_evaluations: int,
        n_failed_attempts: int,
        n_generated: int,
        n_rejected_infeasible: int,
        n_skipped_duplicates: int,
        stopping_reason: str,
        failures: list[tuple[str, str]],
        n_initial_evaluations: int = 0,
        initial_phase: list[tuple[list[float], float, str]] | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.algorithm_id = algorithm_id
        self.seed = seed
        self.best_config = best_config
        self.best_objective = best_objective
        self.history = history  # (config list, objective, evaluation_id)
        self.n_objective_evaluations = n_objective_evaluations
        self.n_failed_attempts = n_failed_attempts
        self.n_generated = n_generated
        self.n_rejected_infeasible = n_rejected_infeasible
        self.n_skipped_duplicates = n_skipped_duplicates
        self.stopping_reason = stopping_reason
        self.failures = failures  # (evaluation_id, exception repr)
        # M8A-R1 shared-start fields (empty for plain random search)
        self.n_initial_evaluations = n_initial_evaluations
        self.initial_phase = initial_phase if initial_phase is not None else []


class RandomSearchOptimizer:
    """Seeded uniform feasible sampling under a fixed evaluation budget."""

    ALGORITHM_ID = "random_search_uniform"

    def __init__(
        self,
        problem: OptimizationProblem,
        evaluation_budget: int,
        seed: int,
        max_generation_attempts: int = 100_000,
        initial_configs: list[list[float]] | None = None,
    ) -> None:
        if evaluation_budget < 1:
            raise ValueError(f"evaluation_budget must be >= 1; got {evaluation_budget}.")
        self.problem = problem
        self.evaluation_budget = int(evaluation_budget)
        self.seed = int(seed)
        self.max_generation_attempts = int(max_generation_attempts)
        # M8A-R1 Phase B: optional authoritative initial configs (the separately
        # identified SHARED-START random-search mode). When provided, these are
        # evaluated FIRST (same initial phase as the BO methods sharing the
        # design) and the REMAINING budget is uniform random feasible sampling.
        # Plain random search passes None (no initial phase — unchanged M5/M6
        # semantics, historical behavior preserved).
        self.initial_configs: list[list[float]] | None = None
        if initial_configs is not None:
            if len(initial_configs) > self.evaluation_budget:
                raise ValueError(
                    f"shared-start initial design ({len(initial_configs)} rows) exceeds the "
                    f"evaluation budget ({self.evaluation_budget})."
                )
            validated: list[list[float]] = []
            for i, row in enumerate(initial_configs):
                arr = np.asarray(row, dtype=np.float64).flatten()
                if arr.shape[0] != problem.dimension:
                    raise ValueError(
                        f"initial_configs[{i}] has dimension {arr.shape[0]}; problem dimension is {problem.dimension}."
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"initial_configs[{i}] contains non-finite values.")
                if not problem.is_feasible(arr):
                    raise ValueError(
                        f"initial_configs[{i}] is infeasible (shared-start designs are validated, never repaired)."
                    )
                validated.append([float(v) for v in arr])
            self.initial_configs = validated

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform sample: continuous within bounds, discrete uniform choice."""
        x = np.empty(self.problem.dimension, dtype=np.float64)
        for idx in range(self.problem.dimension):
            if self.problem.variable_types[idx] == "discrete":
                x[idx] = rng.choice(self.problem.discrete_values[idx])
            else:
                x[idx] = rng.uniform(self.problem.lower_bounds[idx], self.problem.upper_bounds[idx])
        return x

    def optimize(self) -> RandomSearchResult:
        problem = self.problem
        rng = np.random.default_rng(self.seed)
        evaluated: list[np.ndarray] = []
        history: list[tuple[list[float], float, str]] = []
        failures: list[tuple[str, str]] = []
        best_config: list[float] | None = None
        # Incumbent direction follows the problem's declared objective sense.
        best_objective = -np.inf if problem.maximize else np.inf
        n_evaluations = 0
        n_failed_attempts = 0
        n_generated = 0
        n_rejected_infeasible = 0
        n_skipped_duplicates = 0
        stopping_reason = "evaluation_budget_exhausted"
        allow_duplicates = problem.observation_noise_std is not None  # noisy problems may repeat
        # M8A-R1 shared-start bookkeeping
        n_initial_evaluations = 0
        initial_phase: list[tuple[list[float], float, str]] = []

        def _better(value: float, current: float) -> bool:
            return value > current if problem.maximize else value < current

        # ---- shared-start initial phase (authoritative design; consumes budget) ----
        if self.initial_configs is not None:
            for x_row in self.initial_configs:
                if n_evaluations >= self.evaluation_budget:
                    stopping_reason = "evaluation_budget_exhausted"
                    break
                x = np.asarray(x_row, dtype=np.float64)
                n_evaluations += 1
                n_initial_evaluations += 1
                evaluation_id = problem.next_evaluation_id()
                try:
                    value = problem.evaluate(x)
                except (ValueError, TypeError, ArithmeticError, KeyError, AttributeError, RuntimeError) as exc:
                    n_failed_attempts += 1
                    failures.append((evaluation_id, repr(exc)))
                    continue
                evaluated.append(x.copy())
                history.append(([float(v) for v in x], value, evaluation_id))
                initial_phase.append(([float(v) for v in x], value, evaluation_id))
                if _better(value, best_objective):
                    best_objective = value
                    best_config = [float(v) for v in x]

        while n_evaluations < self.evaluation_budget:
            if n_generated >= self.max_generation_attempts:
                stopping_reason = "generation_attempt_limit_reached"
                break
            n_generated += 1
            x = self._sample(rng)
            if not problem.is_feasible(x):
                n_rejected_infeasible += 1
                continue
            if not allow_duplicates and any(
                float(np.max(np.abs(x - prev))) <= DUPLICATE_TOLERANCE for prev in evaluated
            ):
                n_skipped_duplicates += 1
                continue
            # Objective evaluation: consumes budget (initial sampling included).
            # Failed attempts consume budget too (same policy as the BO loop).
            n_evaluations += 1
            evaluation_id = problem.next_evaluation_id()
            try:
                value = problem.evaluate(x)
            except (ValueError, TypeError, ArithmeticError, KeyError, AttributeError, RuntimeError) as exc:
                n_failed_attempts += 1
                failures.append((evaluation_id, repr(exc)))
                continue
            evaluated.append(x.copy())
            history.append(([float(v) for v in x], value, evaluation_id))
            if _better(value, best_objective):
                best_objective = value
                best_config = [float(v) for v in x]

        if best_config is None:
            raise RuntimeError(
                f"Random search found no feasible successful evaluation within {self.max_generation_attempts} "
                f"attempts (budget {self.evaluation_budget}); problem feasible region may be empty "
                "or measure ~0 under uniform sampling."
            )
        return RandomSearchResult(
            problem_id=problem.problem_id,
            algorithm_id=self.ALGORITHM_ID,
            seed=self.seed,
            best_config=best_config,
            best_objective=float(best_objective),
            history=history,
            n_objective_evaluations=n_evaluations,
            n_failed_attempts=n_failed_attempts,
            n_generated=n_generated,
            n_rejected_infeasible=n_rejected_infeasible,
            n_skipped_duplicates=n_skipped_duplicates,
            stopping_reason=stopping_reason,
            failures=failures,
            n_initial_evaluations=n_initial_evaluations,
            initial_phase=initial_phase,
        )
