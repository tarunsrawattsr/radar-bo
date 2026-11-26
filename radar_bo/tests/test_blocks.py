"""Tests for the generic variable-block representation and scheduling
(M6 instructions 5-6)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.blocks import (
    BlockDefinitionError,
    VariableBlock,
    VariableBlockSpec,
    resolve_blocks,
    validate_blocks,
)
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic, BenchmarkBMixedConstrained
from radar_bo.optimization.bo.benchmarks_highdim import BlockStagnationLP, SeparableQuadraticHD
from radar_bo.optimization.bo.scheduling import (
    CyclicBlockScheduler,
    alternating_blocks_spec,
    joint_blocks_spec,
    pairwise_blocks_spec,
)


class TestVariableBlockValidation:
    def test_valid_block_resolves(self):
        problem = BenchmarkAQuadratic()
        block = VariableBlock(VariableBlockSpec(block_id="b0", active_indices=(0,)), problem)
        assert block.block_dimension == 1
        assert block.fixed_indices == (1,)
        assert block.variable_names == ("x1",)
        assert block.variable_types == ("continuous",)
        assert block.covers_all is False

    def test_index_outside_dimension_rejected(self):
        problem = BenchmarkAQuadratic()  # dimension 2
        with pytest.raises(BlockDefinitionError, match="outside decision-vector dimension"):
            VariableBlock(VariableBlockSpec(block_id="bad", active_indices=(0, 2)), problem)

    def test_duplicate_indices_within_block_rejected(self):
        problem = BenchmarkAQuadratic()
        # pydantic wraps the ValueError of the model validator:
        with pytest.raises(ValueError, match="duplicate"):
            VariableBlockSpec(block_id="bad", active_indices=(1, 1))

    def test_empty_block_rejected(self):
        problem = BenchmarkAQuadratic()
        with pytest.raises(Exception, match="at least 1 item|active_indices"):
            VariableBlockSpec(block_id="bad", active_indices=())

    def test_embed_preserves_fixed_coordinates_exactly(self):
        problem = BenchmarkAQuadratic()
        block = VariableBlock(VariableBlockSpec(block_id="b1", active_indices=(1,)), problem)
        base = np.array([3.0, -2.0])
        complete = block.embed(np.array([4.5]), base)
        assert complete[0] == 3.0  # fixed coordinate preserved EXACTLY
        assert complete[1] == 4.5
        # base must not be mutated:
        assert base[1] == -2.0

    def test_embed_rejects_wrong_active_shape(self):
        problem = BenchmarkAQuadratic()
        block = VariableBlock(VariableBlockSpec(block_id="b", active_indices=(0,)), problem)
        with pytest.raises(BlockDefinitionError, match="active_values"):
            block.embed(np.array([1.0, 2.0]), np.zeros(2))

    def test_extract_round_trip(self):
        problem = BenchmarkAQuadratic()
        block = VariableBlock(VariableBlockSpec(block_id="b", active_indices=(1,)), problem)
        x = np.array([2.0, 3.0])
        values = block.extract(x)
        np.testing.assert_array_equal(values, [3.0])
        np.testing.assert_array_equal(block.embed(values, x), x)


class TestBlockCollectionValidation:
    def _blocks(self, problem, groups):
        return [VariableBlock(VariableBlockSpec(block_id=f"b{i}", active_indices=tuple(g)), problem)
                for i, g in enumerate(groups)]

    def test_overlap_rejected(self):
        problem = SeparableQuadraticHD(4)
        blocks = self._blocks(problem, [(0, 1), (1, 2)])
        with pytest.raises(BlockDefinitionError, match="overlap"):
            validate_blocks(blocks, problem)

    def test_duplicate_block_ids_rejected(self):
        problem = SeparableQuadraticHD(4)
        blocks = [
            VariableBlock(VariableBlockSpec(block_id="same", active_indices=(0,)), problem),
            VariableBlock(VariableBlockSpec(block_id="same", active_indices=(1,)), problem),
        ]
        with pytest.raises(BlockDefinitionError, match="unique"):
            validate_blocks(blocks, problem)

    def test_full_coverage_requirement(self):
        problem = SeparableQuadraticHD(4)
        blocks = self._blocks(problem, [(0, 1)])
        with pytest.raises(BlockDefinitionError, match="not covered"):
            validate_blocks(blocks, problem, require_full_coverage=True)

    def test_uncovered_indices_reported(self):
        problem = SeparableQuadraticHD(4)
        blocks = self._blocks(problem, [(0, 1)])
        report = validate_blocks(blocks, problem, require_full_coverage=False)
        assert report["uncovered_indices"] == [2, 3]
        assert report["full_coverage"] is False

    def test_intended_coverage_enforced(self):
        problem = SeparableQuadraticHD(4)
        blocks = self._blocks(problem, [(0,)])
        with pytest.raises(BlockDefinitionError, match="intended"):
            validate_blocks(blocks, problem, intended_coverage={0, 1, 2, 3})

    def test_intended_coverage_satisfied(self):
        problem = SeparableQuadraticHD(4)
        blocks = self._blocks(problem, [(0, 2), (1, 3)])
        report = validate_blocks(blocks, problem, intended_coverage={0, 1, 2, 3})
        assert report["full_coverage"] is True

    def test_resolve_blocks_defaults_to_joint(self):
        problem = BenchmarkAQuadratic()
        blocks = resolve_blocks(problem, None)
        assert len(blocks) == 1 and blocks[0].covers_all
        assert blocks[0].block_id == "joint_all"


class TestScheduling:
    def test_cyclic_order_and_ticket_fields(self):
        problem = BlockStagnationLP()
        blocks = [
            VariableBlock(VariableBlockSpec(block_id="a_group", active_indices=(0, 2)), problem),
            VariableBlock(VariableBlockSpec(block_id="b_group", active_indices=(1, 3)), problem),
        ]
        scheduler = CyclicBlockScheduler(blocks)
        tickets = [scheduler.next_visit() for _ in range(5)]
        # cycle 1: a, b; cycle 2: a, b; cycle 3: a
        assert [(t.cycle_number, t.within_cycle_index, t.block.block_id) for t in tickets] == [
            (1, 0, "a_group"), (1, 1, "b_group"),
            (2, 0, "a_group"), (2, 1, "b_group"),
            (3, 0, "a_group"),
        ]

    def test_strategy_constructors(self):
        assert joint_blocks_spec(3)[0].active_indices == (0, 1, 2)
        alt = alternating_blocks_spec([0, 2], [1, 3])
        assert len(alt) == 2 and alt[0].active_indices == (0, 2)
        pairs = pairwise_blocks_spec([[0, 1], [2, 3], [4, 5]])
        assert len(pairs) == 3 and all(len(p.active_indices) == 2 for p in pairs)

    def test_pairwise_requires_two_variables_per_block(self):
        with pytest.raises(BlockDefinitionError, match="exactly two"):
            pairwise_blocks_spec([[0, 1, 2], [3, 4]])

    def test_pairwise_requires_at_least_two_blocks(self):
        with pytest.raises(BlockDefinitionError, match="at least two"):
            pairwise_blocks_spec([[0, 1]])

    def test_alternating_requires_nonempty_groups(self):
        with pytest.raises(BlockDefinitionError):
            alternating_blocks_spec([], [1])

    def test_empty_scheduler_rejected(self):
        with pytest.raises(BlockDefinitionError):
            CyclicBlockScheduler([])


class TestMixedProblemBlocks:
    def test_blocks_on_mixed_variable_problem(self):
        problem = BenchmarkBMixedConstrained()  # (discrete d, continuous x)
        block_d = VariableBlock(VariableBlockSpec(block_id="discrete", active_indices=(0,)), problem)
        block_x = VariableBlock(VariableBlockSpec(block_id="continuous", active_indices=(1,)), problem)
        assert block_d.variable_types == ("discrete",)
        assert block_x.variable_types == ("continuous",)
        report = validate_blocks([block_d, block_x], problem, require_full_coverage=True)
        assert report["full_coverage"] is True
