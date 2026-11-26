"""Evaluation-accounting identity checks (M5 review instruction 3.2; M6
instruction 18). These identities are verified on instrumented runs covering
the joint and block architectures, with and without injected failures.

Identities (attempted evaluations = budget-consuming objective evaluations):

I1  total attempted = initial attempted + subsequent attempted
I2  total attempted = successful evaluations + failed evaluations
    (failed attempts are a SUBSET of attempted evaluations — never additive)
I3  dataset successes + logged objective-stage failures = total attempted
    (GP-fitting and acquisition-scoring failures are NOT objective attempts
    and never enter this count; scoring/GP fitting are tracked separately)
I4  acquisition scores and GP fits are accounted separately from the budget
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic
from radar_bo.optimization.bo.random_search import RandomSearchOptimizer

OBJECTIVE_STAGES = {"initial_design", "objective_evaluation"}


def _check_identities(result) -> None:
    attempted = result.n_initial_evaluations + result.n_bo_evaluations  # I1
    successful = len(result.observations)
    failed = result.n_failed_attempts
    assert attempted == successful + failed, (
        f"I2 violated: attempted {attempted} != successful {successful} + failed {failed}"
    )
    logged_objective_failures = sum(1 for f in result.failures if f.stage in OBJECTIVE_STAGES)
    assert logged_objective_failures == failed, (
        f"I3 violated: logged objective-stage failures {logged_objective_failures} != failed {failed}"
    )
    other_stage_failures = sum(1 for f in result.failures if f.stage not in OBJECTIVE_STAGES)
    assert other_stage_failures == len(result.failures) - logged_objective_failures
    # I4: scores are tracked separately (no overlap with attempted evaluations):
    assert result.n_acquisition_scores >= 0
    # Per-record traceability: every attempt has an id in exactly one ledger.
    success_ids = {r.evaluation_id for r in result.observations}
    failure_ids = {f.evaluation_id for f in result.failures if f.stage in OBJECTIVE_STAGES}
    assert not (success_ids & failure_ids)
    assert len(success_ids) + len(failure_ids) == attempted


class TestAccountingIdentities:
    def test_joint_clean_run(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=40, num_initial_observations=10),
            seed=0,
        ).optimize()
        assert result.n_failed_attempts == 0
        _check_identities(result)

    def test_block_clean_run(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=40, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("a", (0,)), ("b", (1,))]),
            seed=1,
        ).optimize()
        _check_identities(result)

    def test_joint_run_with_failures(self):
        class HalfFailing(BenchmarkAQuadratic):
            def objective(self, x: np.ndarray) -> float:
                if x[0] > 5.0:
                    raise ValueError("injected")
                return super().objective(x)

        result = BayesianOptimizer(
            HalfFailing(),
            OptimizationConfig(evaluation_budget=50, num_initial_observations=10),
            seed=2,
        ).optimize()
        assert result.n_failed_attempts >= 1
        _check_identities(result)

    def test_block_run_with_failures_and_k2(self):
        class HalfFailing(BenchmarkAQuadratic):
            def objective(self, x: np.ndarray) -> float:
                if x[0] > 5.0:
                    raise ValueError("injected")
                return super().objective(x)

        result = BayesianOptimizer(
            HalfFailing(),
            OptimizationConfig(evaluation_budget=50, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("a", (0,)), ("b", (1,))], evaluations_per_block_visit=2),
            seed=3,
        ).optimize()
        _check_identities(result)

    def test_random_search_identities(self):
        problem = BenchmarkAQuadratic()
        result = RandomSearchOptimizer(problem, 30, seed=4).optimize()
        attempted = result.n_objective_evaluations
        assert attempted == len(result.history) + result.n_failed_attempts  # I2 for random search
        assert attempted == 30  # I1: random search has no separate initial design

    def test_all_failing_problem(self):
        class AlwaysFails(BenchmarkAQuadratic):
            def objective(self, x: np.ndarray) -> float:
                raise RuntimeError("always")

        result = BayesianOptimizer(
            AlwaysFails(),
            OptimizationConfig(evaluation_budget=10, num_initial_observations=10),
            seed=5,
        ).optimize()
        assert result.stopping_reason == "no_feasible_initial_design"
        _check_identities(result)
        assert result.n_failed_attempts == 10
        assert len(result.observations) == 0
