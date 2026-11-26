"""Tests for the generic BO loop (M5 instructions 21-25)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
)
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.problem import OptimizationProblem
from radar_bo.optimization.bo.reference_solvers import benchmark_a_reference
from radar_bo.tests.test_candidates import TinyFeasibleRegion


def settings(**kwargs) -> OptimizationConfig:
    """Small initial design (10) so short-budget tests exercise BO iterations."""
    defaults: dict = {"evaluation_budget": 40, "num_initial_observations": 10}
    defaults.update(kwargs)
    return OptimizationConfig(**defaults)


class CountingQuadratic(BenchmarkAQuadratic):
    """Benchmark A with an objective-call counter (scoring-must-not-evaluate check)."""

    def __init__(self) -> None:
        super().__init__()
        self.objective_calls = 0

    def objective(self, x: np.ndarray) -> float:
        self.objective_calls += 1
        return super().objective(x)


class FailAfterProblem(OptimizationProblem):
    """Succeeds for the first K evaluations, then always fails (deterministic
    failure-injection for the consecutive-failure cap)."""

    def __init__(self, succeed_for: int) -> None:
        super().__init__(
            problem_id="fail_after", variable_names=("x",), variable_types=("continuous",),
            lower_bounds=(0.0,), upper_bounds=(10.0,), maximize=True,
        )
        self.succeed_for = succeed_for

    def objective(self, x: np.ndarray) -> float:
        if self.evaluation_count > self.succeed_for:
            raise RuntimeError("injected failure")
        return float(x[0])


class AlwaysFailsProblem(OptimizationProblem):
    def __init__(self) -> None:
        super().__init__(
            problem_id="always_fails", variable_names=("x",), variable_types=("continuous",),
            lower_bounds=(0.0,), upper_bounds=(1.0,), maximize=True,
        )

    def objective(self, x: np.ndarray) -> float:
        raise RuntimeError("always fails")


class TestFullRuns:
    def test_run_on_benchmark_a(self):
        problem = CountingQuadratic()
        result = BayesianOptimizer(problem, settings(), reference_objective=10.0,
                                   reference_method="analytic").optimize()
        # Budget respected exactly: every attempt (initial + BO) consumes one
        # evaluation; n_failed_attempts is a SUBSET of those attempts.
        assert result.n_initial_evaluations == 10
        assert result.n_initial_evaluations + result.n_bo_evaluations == 40
        assert result.n_failed_attempts == 0
        assert problem.objective_calls == result.n_initial_evaluations + result.n_bo_evaluations
        assert result.stopping_reason == "evaluation_budget_exhausted"
        # Incumbent was actually evaluated:
        dataset_configs = [record.config for record in result.observations]
        assert result.best_config in dataset_configs
        # Incumbent history nondecreasing (deterministic problem):
        history = np.asarray(result.incumbent_history)
        assert np.all(np.diff(history) >= -1e-12)
        # All successful observations feasible:
        for record in result.observations:
            assert problem.is_feasible(np.asarray(record.config))
        # Gap to the analytic reference recorded and finite:
        assert result.best_gap_to_reference is not None and np.isfinite(result.best_gap_to_reference)
        # Reference never claimed as proven by BO:
        assert result.best_objective <= 10.0 + 1e-9

    def test_run_on_mixed_benchmark_b(self):
        result = BayesianOptimizer(
            BenchmarkBMixedConstrained(), settings(evaluation_budget=60),
            reference_objective=11.0, reference_method="enumeration_exact", seed=1,
        ).optimize()
        assert result.stopping_reason == "evaluation_budget_exhausted"
        # Best configuration respects the mixed structure and the constraint:
        d, xc = result.best_config
        assert d in {0.5, 1.0, 1.5, 2.0}
        assert xc + 3.0 * d <= 8.0 + 1e-9
        assert result.best_objective <= 11.0 + 1e-9

    def test_run_on_multimodal_benchmark_c(self):
        result = BayesianOptimizer(BenchmarkCMultimodal(), settings(evaluation_budget=50), seed=2).optimize()
        assert np.isfinite(result.best_objective)
        assert result.stopping_reason == "evaluation_budget_exhausted"


class TestBudgetAndScoring:
    def test_acquisition_scoring_does_not_evaluate_objective(self):
        problem = CountingQuadratic()
        result = BayesianOptimizer(problem, settings()).optimize()
        # Objective calls == objective evaluations only (scores consumed zero):
        assert problem.objective_calls == result.n_initial_evaluations + result.n_bo_evaluations
        # Pool scoring happened at scale (500 candidates per iteration):
        assert result.n_acquisition_scores == 500 * len(result.iteration_diagnostics)

    def test_failures_consume_budget_and_are_logged(self):
        result = BayesianOptimizer(
            FailAfterProblem(succeed_for=10), settings(evaluation_budget=30), seed=0
        ).optimize()
        # The initial design (10 evaluations) succeeds; every BO evaluation then
        # fails and ALSO consumes budget (documented policy). After
        # MAX_CONSECUTIVE_OBJECTIVE_FAILURES = 3 consecutive failures the loop
        # stops instead of burning the whole budget on guaranteed failures.
        assert result.n_initial_evaluations == 10
        assert result.n_bo_evaluations == 3
        assert result.n_failed_attempts == 3
        assert result.n_initial_evaluations + result.n_bo_evaluations <= 30
        assert result.stopping_reason == "repeated_objective_failures"
        # Failures never entered the successful dataset:
        assert all(record.evaluation_id not in {f.evaluation_id for f in result.failures}
                   for record in result.observations)
        # Incumbent is the best of the successful initial design:
        assert len(result.observations) == 10
        best_observed = max(record.objective for record in result.observations)
        assert result.best_objective == pytest.approx(best_observed)
        assert result.best_objective <= 10.0  # never exceeds the true maximum

    def test_all_failing_problem_stops_without_observations(self):
        result = BayesianOptimizer(AlwaysFailsProblem(), settings(evaluation_budget=10), seed=0).optimize()
        assert result.stopping_reason == "no_feasible_initial_design"
        assert len(result.observations) == 0
        assert result.n_failed_attempts == 10  # all initial attempts failed and consumed budget
        assert result.n_initial_evaluations == 10
        # No successful evaluation => NO incumbent (recorded honestly as None):
        assert result.best_config is None and result.best_objective is None


class TestStoppingConditions:
    def test_max_iterations(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(), settings(evaluation_budget=100, max_iterations=3), seed=0
        ).optimize()
        assert result.stopping_reason == "max_iterations_reached"
        assert len(result.iteration_diagnostics) == 3
        assert result.n_bo_evaluations == 3

    def test_runtime_limit(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(), settings(evaluation_budget=100, runtime_budget_s=1e-6), seed=0
        ).optimize()
        assert result.stopping_reason == "runtime_limit_reached"

    def test_patience_without_improvement(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            settings(evaluation_budget=100, improvement_tolerance=1e-2, patience=3),
            seed=0,
        ).optimize()
        assert result.stopping_reason == "patience_without_improvement"
        history = np.asarray(result.incumbent_history)
        assert np.all(np.diff(history) >= -1e-12)  # still nondecreasing

    def test_candidate_pool_exhausted(self):
        """A problem with only two feasible configurations: after the initial
        design evaluates both, no NEW feasible candidate exists."""
        result = BayesianOptimizer(
            TinyFeasibleRegion(),
            settings(evaluation_budget=20, num_initial_observations=2, num_acquisition_candidates=500),
            seed=0,
        ).optimize()
        assert result.stopping_reason == "candidate_pool_exhausted"
        assert result.n_bo_evaluations == 0

    def test_config_validation(self):
        problem = BenchmarkAQuadratic()
        # Literal fields reject invalid values at construction (pydantic), so
        # unsupported-engine branches are reached via validated copies:
        with pytest.raises(NotImplementedError):
            BayesianOptimizer(problem, settings().model_copy(update={"kernel": "rbf"}))
        with pytest.raises(NotImplementedError):
            BayesianOptimizer(problem, settings(acquisition="q_log_noisy_expected_improvement"))
        with pytest.raises(NotImplementedError):
            BayesianOptimizer(problem, settings().model_copy(update={"precision": "float32"}))
        with pytest.raises(NotImplementedError):
            BayesianOptimizer(
                problem, OptimizationConfig(evaluation_budget=10, input_normalization=False)
            )
        with pytest.raises(TypeError):
            BayesianOptimizer("not a problem", settings())  # type: ignore[arg-type]
        with pytest.raises(Exception, match="evaluation_budget"):
            BayesianOptimizer(problem, OptimizationConfig())  # budget unset


class TestResultRecords:
    def test_serialization_round_trip(self):
        result = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=25), seed=0).optimize()
        text = result.model_dump_json()
        from radar_bo.optimization.bo.results import OptimizationResult

        restored = OptimizationResult.model_validate_json(text)
        assert restored.best_config == result.best_config
        assert restored.best_objective == result.best_objective
        assert restored.stopping_reason == result.stopping_reason
        assert restored.settings["evaluation_budget"] == 25

    def test_dependency_versions_recorded(self):
        result = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=25), seed=0).optimize()
        assert "torch" in result.dependency_versions and "botorch" in result.dependency_versions

    def test_reference_recorded(self):
        ref = benchmark_a_reference()
        result = BayesianOptimizer(
            BenchmarkAQuadratic(), settings(evaluation_budget=25),
            reference_objective=ref.objective_value, reference_method=ref.verification_method,
            seed=0,
        ).optimize()
        assert result.reference_objective == pytest.approx(10.0)
        assert result.reference_method == "analytic_exact"

    def test_result_metadata_separate_from_tensors(self):
        """Result records carry plain values only (no tensors)."""
        import json

        result = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=25), seed=0).optimize()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed["best_config"], list)


class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=30), seed=4).optimize()
        r2 = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=30), seed=4).optimize()
        assert r1.best_config == r2.best_config
        assert r1.best_objective == r2.best_objective
        assert r1.incumbent_history == r2.incumbent_history
        assert r1.stopping_reason == r2.stopping_reason

    def test_different_seeds_differ(self):
        r1 = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=30), seed=1).optimize()
        r2 = BayesianOptimizer(BenchmarkAQuadratic(), settings(evaluation_budget=30), seed=2).optimize()
        assert r1.best_config != r2.best_config or r1.best_objective != r2.best_objective
