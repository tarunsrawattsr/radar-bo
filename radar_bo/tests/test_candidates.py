"""Tests for feasible candidate-pool generation (M5 instruction 19)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic, BenchmarkBMixedConstrained
from radar_bo.optimization.bo.candidates import CandidateGenerator
from radar_bo.optimization.bo.problem import OptimizationProblem


class TinyFeasibleRegion(OptimizationProblem):
    """Two feasible discrete points only (for empty-pool handling tests)."""

    def __init__(self) -> None:
        super().__init__(
            problem_id="tiny_feasible",
            variable_names=("d",),
            variable_types=("discrete",),
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
            discrete_values={0: np.array([0.1, 0.9])},
            maximize=True,
        )

    def objective(self, x: np.ndarray) -> float:
        return float(x[0])


class TestPoolGeneration:
    def test_requested_size_respected(self):
        generator = CandidateGenerator(BenchmarkAQuadratic(), num_candidates=500, seed=0)
        pool, report = generator.generate_pool()
        assert pool.shape == (500, 2)
        assert report.final_size == 500
        assert pool.dtype == np.float64

    def test_all_candidates_feasible(self):
        problem = BenchmarkBMixedConstrained()
        generator = CandidateGenerator(problem, num_candidates=300, seed=1)
        pool, report = generator.generate_pool()
        for row in pool:
            assert problem.is_feasible(row)
        # The instantaneous constraint rejects some uniform samples:
        assert report.rejected_infeasible > 0

    def test_mixed_variable_support(self):
        problem = BenchmarkBMixedConstrained()
        generator = CandidateGenerator(problem, num_candidates=200, seed=2)
        pool, _ = generator.generate_pool()
        discrete_members = {0.5, 1.0, 1.5, 2.0}
        assert all(row[0] in discrete_members for row in pool)
        assert np.all((pool[:, 1] >= 0.0) & (pool[:, 1] <= 10.0))

    def test_within_bounds(self):
        problem = BenchmarkAQuadratic()
        generator = CandidateGenerator(problem, num_candidates=200, seed=3)
        pool, _ = generator.generate_pool()
        assert np.all(pool >= problem.lower_bounds) and np.all(pool <= problem.upper_bounds)

    def test_seed_reproducibility(self):
        g1 = CandidateGenerator(BenchmarkAQuadratic(), 100, seed=5)
        g2 = CandidateGenerator(BenchmarkAQuadratic(), 100, seed=5)
        p1, r1 = g1.generate_pool()
        p2, r2 = g2.generate_pool()
        np.testing.assert_array_equal(p1, p2)
        # Report equality EXCLUDES the M6 wall-clock instrumentation fields
        # (generation_duration_s / feasibility_validation_s are timings, not
        # reproducible quantities):
        def _strip(report):
            return report.model_dump(exclude={"generation_duration_s", "feasibility_validation_s"})
        assert _strip(r1) == _strip(r2)

    def test_duplicate_detection_within_pool(self):
        """With only 2 distinct feasible points, a 500-pool cannot contain 500
        unique members: duplicates within the pool are rejected and counted."""
        generator = CandidateGenerator(TinyFeasibleRegion(), num_candidates=500, seed=0)
        pool, report = generator.generate_pool()
        assert pool.shape[0] == 2  # exactly the distinct feasible points
        assert report.rejected_duplicate_within_pool > 0

    def test_exclusion_of_evaluated_configs(self):
        generator = CandidateGenerator(TinyFeasibleRegion(), num_candidates=10, seed=0)
        evaluated = np.array([[0.1]])
        pool, report = generator.generate_pool(evaluated_configs=evaluated)
        np.testing.assert_array_equal(pool, [[0.9]])  # 0.1 excluded
        assert report.rejected_duplicate_evaluated > 0

    def test_empty_pool_reported_not_hidden(self):
        generator = CandidateGenerator(TinyFeasibleRegion(), num_candidates=10, seed=0)
        evaluated = np.array([[0.1], [0.9]])  # everything already evaluated
        pool, report = generator.generate_pool(evaluated_configs=evaluated)
        assert pool.shape == (0, 1)
        assert report.final_size == 0

    def test_invalid_configuration_rejected(self):
        with pytest.raises(ValueError):
            CandidateGenerator(BenchmarkAQuadratic(), num_candidates=0, seed=0)

    def test_pool_size_distinct_from_objective_evaluations(self):
        """Scoring the pool consumes ZERO objective evaluations (documented in report notes)."""
        generator = CandidateGenerator(BenchmarkAQuadratic(), 500, seed=0)
        problem = BenchmarkAQuadratic()
        before = problem.evaluation_count
        pool, report = generator.generate_pool()
        assert problem.evaluation_count == before  # no evaluations consumed
        assert any("zero" in note for note in report.notes)
        assert len(pool) == 500
