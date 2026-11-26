"""Tests for the resource-allocation representation (M4 instruction 8)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import DiscreteMembershipError, ResourceAllocation

ACTION_SET = tuple(ResourceConfig().dwell_time_action_set)  # 0.1 .. 1.0


def make_allocation(indices=(0, 2, 4), powers=None) -> ResourceAllocation:
    return ResourceAllocation.from_dwell_indices(list(indices), list(ACTION_SET), powers)


class TestStructureAndOrdering:
    def test_task_ids_fixed_ordering(self):
        alloc = make_allocation()
        assert alloc.task_ids == ("task_0", "task_1", "task_2")
        assert alloc.num_tasks == 3
        assert alloc.dwell_times_s == (0.1, 0.3, 0.5)  # action_set[0], [2], [4]

    def test_indices_must_be_in_range(self):
        # Out-of-range positive index: factory indexing fails first.
        with pytest.raises((IndexError, ValueError)):
            ResourceAllocation.from_dwell_indices([0, 10], list(ACTION_SET))
        # Negative index: model validator rejects (no silent wraparound).
        with pytest.raises(ValueError, match="indices"):
            ResourceAllocation.from_dwell_indices([0, -1], list(ACTION_SET))

    def test_task_count_consistency(self):
        with pytest.raises(ValueError, match="inconsistency"):
            ResourceAllocation(
                num_tasks=3, task_ids=(), dwell_indices=(0, 0), dwell_times_s=(0.1, 0.2),
                action_set=ACTION_SET,
            )

    def test_dwell_must_match_indexed_value(self):
        with pytest.raises(ValueError, match="exactly|exactly match"):
            ResourceAllocation(
                num_tasks=1, task_ids=(), dwell_indices=(0,), dwell_times_s=(0.5,),
                action_set=ACTION_SET,
            )

    def test_power_block_length_validated(self):
        with pytest.raises(ValueError, match="num_tasks"):
            make_allocation(powers=(1.0, 2.0))
        with pytest.raises(ValueError, match="continuous_power_w"):
            make_allocation(powers=(1.0, np.nan, 2.0))
        with pytest.raises(ValueError):
            make_allocation(powers=(1.0, -2.0, 2.0))


class TestFlatVectorRoundTrip:
    def test_round_trip_dwell_only(self):
        alloc = make_allocation((1, 3, 5))
        flat = alloc.to_flat_vector()
        assert flat.dtype == np.float64
        np.testing.assert_array_equal(flat, [0.2, 0.4, 0.6])
        decoded = ResourceAllocation.from_flat_vector(flat, ACTION_SET, include_power=False)
        assert decoded == alloc

    def test_round_trip_with_power(self):
        alloc = make_allocation((0, 1, 2), powers=(10.0, 20.0, 30.0))
        flat = alloc.to_flat_vector()
        assert flat.shape == (6,)
        np.testing.assert_array_equal(flat, [0.1, 0.2, 0.3, 10.0, 20.0, 30.0])
        decoded = ResourceAllocation.from_flat_vector(flat, ACTION_SET, include_power=True)
        assert decoded == alloc
        assert decoded.continuous_power_w == (10.0, 20.0, 30.0)

    def test_expected_flat_dimension(self):
        assert make_allocation().expected_flat_dimension() == 3
        assert make_allocation(powers=(1.0, 2.0, 3.0)).expected_flat_dimension() == 6

    def test_to_flat_vector_returns_fresh_array(self):
        alloc = make_allocation()
        a = alloc.to_flat_vector()
        a[0] = 999.0
        np.testing.assert_array_equal(alloc.to_flat_vector(), [0.1, 0.3, 0.5])  # unchanged

    def test_non_member_value_rejected_not_snapped(self):
        """0.55 lies between members; decode must reject, never snap."""
        with pytest.raises(DiscreteMembershipError, match="never snapped"):
            ResourceAllocation.from_flat_vector(np.array([0.55, 0.2, 0.3]), ACTION_SET)
        with pytest.raises(DiscreteMembershipError):
            ResourceAllocation.from_flat_vector(np.array([0.11, 0.2, 0.3]), ACTION_SET)

    def test_invalid_flat_vectors_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            ResourceAllocation.from_flat_vector(np.zeros((2, 3)), ACTION_SET)
        with pytest.raises(ValueError, match="non-finite"):
            ResourceAllocation.from_flat_vector(np.array([np.nan, 0.1]), ACTION_SET)
        # Odd length with include_power=True: not divisible into (dwell, power) pairs.
        with pytest.raises(ValueError, match="divisible"):
            ResourceAllocation.from_flat_vector(np.array([0.1, 0.2, 0.3]), ACTION_SET, include_power=True)


class TestSerializationAndCopying:
    def test_json_round_trip(self):
        alloc = make_allocation((2, 4, 6), powers=(5.0, 6.0, 7.0))
        text = alloc.model_dump_json()
        restored = ResourceAllocation.model_validate_json(text)
        assert restored == alloc
        assert restored.continuous_power_w == (5.0, 6.0, 7.0)

    def test_plain_json_structure(self):
        parsed = json.loads(make_allocation().model_dump_json())
        assert parsed["task_ids"] == ["task_0", "task_1", "task_2"]
        assert parsed["dwell_times_s"] == [0.1, 0.3, 0.5]

    def test_defensive_copy_independent(self):
        alloc = make_allocation(powers=(1.0, 2.0, 3.0))
        copy = alloc.defensive_copy()
        assert copy == alloc
        # Tuples are immutable; flat arrays are freshly allocated:
        a1 = alloc.to_flat_vector()
        a2 = copy.to_flat_vector()
        np.testing.assert_array_equal(a2, [0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
        a2[0] = 99.0
        np.testing.assert_array_equal(alloc.to_flat_vector(), [0.1, 0.3, 0.5, 1.0, 2.0, 3.0])

    def test_content_id_deterministic_and_sensitive(self):
        a = make_allocation((0, 1, 2))
        b = make_allocation((0, 1, 2))
        c = make_allocation((1, 1, 2))
        assert a.content_id() == b.content_id()
        assert a.content_id() != c.content_id()
        d = make_allocation((0, 1, 2), powers=(1.0, 2.0, 3.0))
        assert d.content_id() != a.content_id()

    def test_no_torch_dependency(self):
        import ast
        from pathlib import Path

        source = Path(ResourceAllocation.__module__.replace(".", "/") + ".py").resolve()
        assert "optimization/allocation.py" in str(source)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {"torch", "botorch", "gpytorch"} & {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "botorch", "gpytorch"}
