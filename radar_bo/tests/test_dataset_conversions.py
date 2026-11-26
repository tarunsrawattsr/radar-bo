"""Tests for the observation dataset and NumPy-PyTorch conversion (M5 instructions 8-9, 11)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from radar_bo.optimization.bo.conversions import numpy_to_torch, torch_to_numpy
from radar_bo.optimization.bo.dataset import (
    DUPLICATE_TOLERANCE,
    DuplicateObservationError,
    ObservationDataset,
    ObservationRecord,
)


def make_dataset(dimension: int = 2, maximize: bool = True) -> ObservationDataset:
    return ObservationDataset(problem_id="test_problem", dimension=dimension, maximize=maximize)


class TestDatasetInsertion:
    def test_insert_and_shapes(self):
        ds = make_dataset()
        for i in range(5):
            ds.insert(np.array([float(i), 1.0]), float(i * i), f"eval_{i}")
        assert ds.size == 5
        x = ds.training_inputs()
        y = ds.training_outputs()
        assert x.shape == (5, 2) and y.shape == (5, 1)
        assert x.dtype == np.float64 and y.dtype == np.float64
        np.testing.assert_array_equal(y.flatten(), [0.0, 1.0, 4.0, 9.0, 16.0])

    def test_record_fields(self):
        ds = make_dataset()
        ds.insert(
            np.array([1.0, 2.0]), 5.0, "eval_1",
            seed_info="seed=3", observation_noise_std=None, metadata={"stage": "initial"},
        )
        record = ds.records[0]
        assert record.evaluation_id == "eval_1"
        assert record.problem_id == "test_problem"
        assert record.config == [1.0, 2.0]
        assert record.objective == 5.0
        assert record.seed_info == "seed=3"
        assert record.observation_noise_std is None
        assert record.metadata == {"stage": "initial"}

    def test_invalid_insertions_rejected(self):
        ds = make_dataset(dimension=2)
        with pytest.raises(ValueError, match="2 entries"):
            ds.insert(np.array([1.0]), 1.0, "e1")
        with pytest.raises(ValueError, match="non-finite"):
            ds.insert(np.array([np.nan, 1.0]), 1.0, "e1")
        with pytest.raises(ValueError, match="finite"):
            ds.insert(np.array([1.0, 1.0]), np.inf, "e1")

    def test_failed_observations_never_enter_dataset(self):
        """Failure records stay in the separate log; training tensors unaffected."""
        ds = make_dataset()
        ds.insert(np.array([1.0, 1.0]), 4.0, "ok_1")
        ds.record_failure("fail_1", "ValueError: bad", "objective_evaluation", np.array([2.0, 2.0]))
        ds.record_failure("fail_2", "RuntimeError: boom", "gp_fitting", None)
        assert ds.size == 1
        assert ds.failure_count == 2
        np.testing.assert_array_equal(ds.training_inputs(), [[1.0, 1.0]])
        assert all(f.evaluation_id not in {r.evaluation_id for r in ds.records} for f in ds.failures)


class TestDuplicatePolicy:
    def test_exact_duplicate_rejected(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        with pytest.raises(DuplicateObservationError, match="allow_duplicates=True"):
            ds.insert(np.array([1.0, 2.0]), 5.0, "e2")

    def test_near_duplicate_within_tolerance_rejected(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        near = np.array([1.0 + DUPLICATE_TOLERANCE / 2.0, 2.0])
        with pytest.raises(DuplicateObservationError):
            ds.insert(near, 5.0, "e2")

    def test_distinct_configurations_not_merged(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        ds.insert(np.array([1.5, 2.0]), 6.0, "e2")  # clearly distinct
        assert ds.size == 2

    def test_intentional_repeat_allowed_with_flag(self):
        """Noisy objectives: repeated measurements are legitimate when flagged."""
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        ds.insert(np.array([1.0, 2.0]), 5.1, "e2", allow_duplicates=True)
        assert ds.size == 2  # both kept as independent observations

    def test_has_config(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        assert ds.has_config([1.0, 2.0])
        assert not ds.has_config([9.0, 9.0])


class TestDatasetQueriesAndSerialization:
    def test_best_respects_direction(self):
        ds_max = make_dataset(maximize=True)
        ds_max.insert(np.array([0.0, 0.0]), 1.0, "a")
        ds_max.insert(np.array([1.0, 0.0]), 3.0, "b")
        idx, record = ds_max.best()
        assert record.objective == 3.0 and idx == 1
        ds_min = make_dataset(maximize=False)
        ds_min.insert(np.array([0.0, 0.0]), 1.0, "a")
        ds_min.insert(np.array([1.0, 0.0]), 3.0, "b")
        assert ds_min.best()[1].objective == 1.0

    def test_best_on_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            make_dataset().best()

    def test_training_arrays_are_fresh_copies(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1")
        x = ds.training_inputs()
        x[0, 0] = 99.0
        np.testing.assert_array_equal(ds.training_inputs()[0], [1.0, 2.0])

    def test_json_round_trip(self):
        ds = make_dataset()
        ds.insert(np.array([1.0, 2.0]), 5.0, "e1", seed_info="0", metadata={"k": "v"})
        ds.record_failure("f1", "err", "objective_evaluation")
        text = ds.model_dump_json()
        restored = ObservationDataset.model_validate_json(text)
        assert restored == ds
        assert restored.failure_count == 1

    def test_inconsistent_record_dimension_rejected(self):
        bad = ObservationDataset(problem_id="p", dimension=3)
        bad.records.append(
            ObservationRecord(evaluation_id="x", problem_id="p", config=[1.0, 2.0], objective=0.0)
        )
        with pytest.raises(ValueError, match="dimension"):
            bad.model_validate(bad.model_dump())


class TestConversions:
    def test_numpy_to_torch_dtype_device_shape(self):
        t = numpy_to_torch(np.arange(6, dtype=np.float64).reshape(3, 2))
        assert t.dtype == torch.float64
        assert t.device == torch.device("cpu")
        assert t.shape == (3, 2)

    def test_non_float64_input_coerced(self):
        t = numpy_to_torch(np.arange(4, dtype=np.float32).reshape(2, 2))
        assert t.dtype == torch.float64

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            numpy_to_torch(np.array([[np.nan, 1.0]]))

    def test_1d_rejected_when_2d_required(self):
        with pytest.raises(ValueError, match="2-D"):
            numpy_to_torch(np.array([1.0, 2.0]))

    def test_no_shared_memory_numpy_side(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        t = numpy_to_torch(arr)
        t[0, 0] = 99.0
        assert arr[0, 0] == 1.0  # source unchanged

    def test_no_shared_memory_torch_side(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        t = numpy_to_torch(arr)
        arr[0, 0] = 77.0
        assert float(t[0, 0]) == 1.0  # tensor unchanged

    def test_torch_to_numpy_copies(self):
        t = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        arr = torch_to_numpy(t)
        arr[0, 0] = -5.0
        assert float(t[0, 0]) == 1.0

    def test_round_trip_preserves_values(self):
        arr = np.random.default_rng(0).normal(size=(7, 3))
        np.testing.assert_array_equal(torch_to_numpy(numpy_to_torch(arr)), arr)
