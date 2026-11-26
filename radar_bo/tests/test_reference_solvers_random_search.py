"""Tests for the independent reference solvers and random search (M5 instructions 6-7)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
)
from radar_bo.optimization.bo.random_search import RandomSearchOptimizer
from radar_bo.optimization.bo.reference_solvers import (
    benchmark_a_reference,
    benchmark_b_reference,
    benchmark_c_reference,
)


class TestReferenceSolvers:
    def test_benchmark_a_analytic_reference(self):
        ref = benchmark_a_reference()
        assert ref.verification_method == "analytic_exact"
        assert ref.is_exact
        assert ref.objective_value == pytest.approx(10.0, abs=1e-12)
        assert ref.config == [3.5, 7.2]
        assert ref.feasible

    def test_benchmark_b_enumeration_reference(self):
        """Enumeration + analytic inner optimum: f* = 11 at (2.0, 2.0) exactly."""
        ref = benchmark_b_reference()
        assert ref.verification_method == "enumeration_exact"
        assert ref.objective_value == pytest.approx(11.0, abs=1e-12)
        assert ref.config[0] == pytest.approx(2.0)
        assert ref.config[1] == pytest.approx(2.0)
        assert ref.feasible

    def test_benchmark_c_numerical_reference_approximate(self):
        ref = benchmark_c_reference(seed=0)
        assert ref.verification_method == "numerical_approximate"
        assert not ref.is_exact
        assert np.isfinite(ref.objective_value)
        assert ref.feasible
        # The numerical reference must be at least as good as a dense random scan:
        problem = BenchmarkCMultimodal()
        rng = np.random.default_rng(1)
        best_random = max(
            problem.objective(rng.uniform(-2.5, 2.5, size=2)) for _ in range(20_000)
        )
        assert ref.objective_value >= best_random - 1e-9

    def test_solvers_have_no_gp_or_botorch_imports(self):
        """Reference solvers must not depend on GP predictions (instruction 6)."""
        source_path = Path("radar_bo/optimization/bo/reference_solvers.py")
        if not source_path.exists():  # run from another cwd
            source_path = Path(__file__).resolve().parents[1] / "optimization/bo/reference_solvers.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
                assert not roots & {"torch", "botorch", "gpytorch"}
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "botorch", "gpytorch"}

    def test_reference_solution_serializable(self):
        ref = benchmark_a_reference()
        text = ref.model_dump_json()
        from radar_bo.optimization.bo.reference_solvers import ReferenceSolution

        assert ReferenceSolution.model_validate_json(text) == ref


class TestRandomSearch:
    def test_respects_evaluation_budget_exactly(self):
        problem = BenchmarkAQuadratic()
        result = RandomSearchOptimizer(problem, evaluation_budget=25, seed=0).optimize()
        assert result.n_objective_evaluations == 25
        assert len(result.history) == 25

    def test_seed_reproducibility(self):
        # Fresh problem instances: evaluation ids embed a per-instance counter
        # (and a uuid), so identity of records is about configs/values, not ids.
        r1 = RandomSearchOptimizer(BenchmarkAQuadratic(), 30, seed=7).optimize()
        r2 = RandomSearchOptimizer(BenchmarkAQuadratic(), 30, seed=7).optimize()
        assert r1.best_config == r2.best_config
        assert r1.best_objective == r2.best_objective
        assert [(c, v) for c, v, _ in r1.history] == [(c, v) for c, v, _ in r2.history]

    def test_all_evaluated_configs_feasible(self):
        problem = BenchmarkBMixedConstrained()
        result = RandomSearchOptimizer(problem, 40, seed=3).optimize()
        for config, _, _ in result.history:
            assert problem.is_feasible(np.asarray(config))
        # Rejections were counted (constraint makes some samples infeasible):
        assert result.n_rejected_infeasible > 0

    def test_best_is_max_of_history(self):
        problem = BenchmarkAQuadratic()
        result = RandomSearchOptimizer(problem, 30, seed=5).optimize()
        best_from_history = max(value for _, value, _ in result.history)
        assert result.best_objective == pytest.approx(best_from_history)
        # Best config corresponds to the best objective:
        assert problem.evaluate(np.asarray(result.best_config)) == pytest.approx(result.best_objective)

    def test_initial_cost_counted_in_budget(self):
        """Random search has no separate initial design: every evaluation is in
        the same budget (instruction 7)."""
        problem = BenchmarkAQuadratic()
        result = RandomSearchOptimizer(problem, 10, seed=2).optimize()
        assert result.n_objective_evaluations == 10  # no extra unaccounted evaluations

    def test_mixed_problem_sampling(self):
        problem = BenchmarkBMixedConstrained()
        result = RandomSearchOptimizer(problem, 60, seed=11).optimize()
        discrete_values_seen = {config[0] for config, _, _ in result.history}
        assert discrete_values_seen <= set(problem.discrete_values[0].tolist())

    def test_budget_of_one(self):
        result = RandomSearchOptimizer(BenchmarkAQuadratic(), 1, seed=0).optimize()
        assert result.n_objective_evaluations == 1
        assert np.isfinite(result.best_objective)

    def test_invalid_budget_rejected(self):
        with pytest.raises(ValueError):
            RandomSearchOptimizer(BenchmarkAQuadratic(), 0, seed=0)
