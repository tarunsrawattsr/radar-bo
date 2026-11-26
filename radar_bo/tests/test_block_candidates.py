"""Tests for block-restricted candidate generation (M6 instruction 8)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.block_candidates import BlockCandidateGenerator
from radar_bo.optimization.bo.blocks import VariableBlock, VariableBlockSpec
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic, BenchmarkBMixedConstrained
from radar_bo.optimization.bo.benchmarks_highdim import BlockStagnationLP


def make_block(problem, indices, block_id="b"):
    return VariableBlock(VariableBlockSpec(block_id=block_id, active_indices=tuple(indices)), problem)


class TestBlockPoolGeneration:
    def test_fixed_coordinates_preserved_exactly(self):
        problem = BenchmarkAQuadratic()
        block = make_block(problem, (1,))
        current = np.array([3.5, 0.0])
        pool, _ = BlockCandidateGenerator(problem, block, num_candidates=50, seed=0).generate_pool(current)
        assert pool.shape == (50, 2)  # COMPLETE vectors, full dimension
        np.testing.assert_array_equal(pool[:, 0], np.full(50, 3.5))  # fixed coord identical everywhere
        assert np.all((pool[:, 1] >= -10.0) & (pool[:, 1] <= 10.0))

    def test_shared_constraint_validated_on_complete_vectors(self):
        """The LP's shared budget couples BOTH blocks: candidates that violate it
        through the interaction of active and FIXED coordinates are rejected."""
        problem = BlockStagnationLP()
        block_a = make_block(problem, (0, 2), "a_group")
        # Budget-tight current configuration: a1+a2+b1+b2 = 3.2 already.
        current = np.array([1.4, 0.4, 0.8, 0.6])
        pool, report = BlockCandidateGenerator(problem, block_a, num_candidates=100, seed=1).generate_pool(current)
        for row in pool:
            assert problem.is_feasible(row)  # every delivered candidate feasible
            assert row[1] == 0.4 and row[3] == 0.6  # fixed b coords preserved
            assert np.sum(row) <= problem.BUDGET + 1e-9  # shared constraint holds
        assert report.rejected_infeasible > 0  # overshooting candidates were rejected

    def test_all_rejected_returns_empty_pool_with_report(self):
        problem = BlockStagnationLP()
        block_a = make_block(problem, (0, 2), "a_group")
        # Budget-tight current configuration with the a-coordinates at ZERO:
        # the shared constraint leaves a1 + a2 <= 0, so the only feasible
        # a-block candidate is the current configuration itself, which is
        # duplicate-excluded -> the pool is EMPTY (reported, not hidden).
        current = np.array([0.0, 1.6, 0.0, 1.6])
        pool, report = BlockCandidateGenerator(problem, block_a, num_candidates=50, seed=0).generate_pool(current)
        assert pool.shape == (0, 4)
        assert report.final_size == 0
        assert report.rejected_infeasible > 0 or report.rejected_duplicate_evaluated > 0

    def test_current_config_excluded_as_duplicate(self):
        problem = BenchmarkAQuadratic()
        block = make_block(problem, (1,))
        current = np.array([0.0, 5.0])
        pool, report = BlockCandidateGenerator(problem, block, num_candidates=200, seed=2).generate_pool(current)
        for row in pool:
            assert float(np.max(np.abs(row - current))) > 1e-9  # current config never re-sampled
        _ = report

    def test_evaluated_configs_excluded(self):
        problem = BenchmarkBMixedConstrained()  # discrete d: sampling HITS evaluated values
        block = make_block(problem, (0,), "discrete")
        current = np.array([1.0, 5.0])
        evaluated = np.array([[0.5, 5.0], [2.0, 5.0]])
        pool, report = BlockCandidateGenerator(problem, block, num_candidates=100, seed=3).generate_pool(
            current, evaluated_configs=evaluated
        )
        for row in pool:
            for ev in evaluated:
                assert float(np.max(np.abs(row - ev))) > 1e-9
        assert report.rejected_duplicate_evaluated > 0  # discrete choices hit the evaluated set

    def test_discrete_block_sampling(self):
        problem = BenchmarkBMixedConstrained()
        block = make_block(problem, (0,), "discrete")
        current = np.array([1.0, 2.0])
        pool, report = BlockCandidateGenerator(problem, block, num_candidates=60, seed=4).generate_pool(current)
        assert pool.shape[1] == 2
        # Only 4 DISTINCT candidates exist (the discrete values): the pool holds
        # exactly those; further samples were rejected as within-pool duplicates.
        assert pool.shape[0] == 4
        assert all(row[0] in {0.5, 1.0, 1.5, 2.0} for row in pool)
        np.testing.assert_array_equal(pool[:, 1], np.full(4, 2.0))  # continuous coord fixed
        assert report.rejected_duplicate_within_pool > 0

    def test_reproducible_pool_for_same_seed(self):
        problem = BenchmarkAQuadratic()
        block = make_block(problem, (1,))
        current = np.array([0.0, 5.0])
        g1 = BlockCandidateGenerator(problem, block, 30, seed=7)
        g2 = BlockCandidateGenerator(problem, block, 30, seed=7)
        p1, r1 = g1.generate_pool(current)
        p2, r2 = g2.generate_pool(current)
        np.testing.assert_array_equal(p1, p2)
        assert r1 == r2

    def test_invalid_current_config_rejected(self):
        problem = BenchmarkAQuadratic()
        block = make_block(problem, (1,))
        with pytest.raises(ValueError, match="current_config"):
            BlockCandidateGenerator(problem, block, 10, seed=0).generate_pool(np.zeros(3))
        with pytest.raises(ValueError, match="non-finite"):
            BlockCandidateGenerator(problem, block, 10, seed=0).generate_pool(np.array([np.nan, 0.0]))

    def test_pool_size_is_not_evaluation_count(self):
        problem = BenchmarkAQuadratic()
        block = make_block(problem, (1,))
        before = problem.evaluation_count
        pool, report = BlockCandidateGenerator(problem, block, 500, seed=0).generate_pool(np.array([0.0, 0.0]))
        assert problem.evaluation_count == before  # zero objective evaluations consumed
        assert pool.shape[0] == 500
        assert any("zero" in note.lower() for note in report.notes)
        assert any("block-restricted" in note.lower() for note in report.notes)
