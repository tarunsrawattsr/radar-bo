"""M7 item 30: experiment-configuration unit tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radar_bo.experiments.framework.experiment_config import (
    AcquisitionSettings,
    CheckpointPolicy,
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
    config_fingerprint,
)


def make_config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_id="test_experiment",
        description="unit-test experiment",
        result_label="generic synthetic unit test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["joint_bo"],
        seeds=[0, 1],
        stopping=StoppingSettings(evaluation_budget=20, **{}),
        output_dir="results/test_experiment",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


class TestExperimentConfigValidation:
    def test_valid_config_builds(self):
        config = make_config()
        assert config.checkpoint.enabled is True
        assert config.candidate_pool.num_acquisition_candidates == 500
        assert config.statistics.threshold_closures == [0.5, 0.9, 0.99]

    def test_experiment_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            make_config(experiment_id="Bad Id!")

    def test_duplicate_benchmark_ids_rejected(self):
        with pytest.raises(ValidationError, match="duplicates"):
            make_config(benchmark_ids=["benchmark_a", "benchmark_a"])

    def test_duplicate_seeds_rejected(self):
        with pytest.raises(ValidationError, match="duplicates"):
            make_config(seeds=[0, 0])

    def test_budget_required(self):
        with pytest.raises(ValidationError):
            make_config(stopping=StoppingSettings())

    def test_distinct_from_radar_config(self):
        """Experiment settings must never carry radar configuration (item 5)."""
        fields = set(ExperimentConfig.model_fields)
        forbidden = {"radar", "dwell", "burst", "prf", "beam", "sinr", "corridor"}
        assert not forbidden & {f.lower() for f in fields}
        # and the model docstring/labels declare the separation
        assert "radar" in ExperimentConfig.__doc__.lower()


class TestFingerprint:
    def test_fingerprint_stable_across_instances(self):
        assert make_config().fingerprint() == make_config().fingerprint()

    def test_fingerprint_changes_with_any_setting(self):
        base = make_config()
        changed_budget = make_config(stopping=StoppingSettings(evaluation_budget=21))
        changed_seed = make_config(seeds=[0, 2])
        changed_pool = make_config(candidate_pool=AcquisitionSettings(num_acquisition_candidates=250))
        changed_output = make_config(output_dir="results/other")
        changed_checkpoint = make_config(checkpoint=CheckpointPolicy(enabled=False))
        changed_initial = make_config(initial_design=InitialDesignSettings(size=5))
        for other in (changed_budget, changed_seed, changed_pool, changed_output, changed_checkpoint, changed_initial):
            assert base.fingerprint() != other.fingerprint()

    def test_config_fingerprint_function_canonical_ordering(self):
        a = config_fingerprint({"x": 1, "y": [1, 2]})
        b = config_fingerprint({"y": [1, 2], "x": 1})
        assert a == b
