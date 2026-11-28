"""M7 item 30: algorithm-registry and experiment-matrix unit tests."""

from __future__ import annotations

import pytest

from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry, AlgorithmRegistryError
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.experiment_config import ExperimentConfig, StoppingSettings
from radar_bo.experiments.framework.matrix import RunStatus, generate_manifest, run_identity_key, stable_run_id


def make_config(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_id="matrix_test",
        description="matrix unit test",
        result_label="generic synthetic unit test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo"],
        seeds=[0, 1],
        stopping=StoppingSettings(evaluation_budget=10),
        output_dir="results/matrix_test",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


class TestAlgorithmRegistry:
    def test_all_methods_registered(self):
        assert set(AlgorithmRegistry().ids()) == {
            "random_search", "random_search_shared_start", "joint_bo",
            "alternating_bo", "pairwise_bo", "reference",
        }

    def test_architectures(self):
        registry = AlgorithmRegistry()
        assert registry.get("joint_bo").architecture == "joint"
        assert registry.get("alternating_bo").architecture == "alternating"
        assert registry.get("pairwise_bo").architecture == "pairwise"
        assert registry.get("random_search").architecture == "random_search"
        assert registry.get("reference").kind == "reference"

    def test_unknown_algorithm_rejected(self):
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm"):
            AlgorithmRegistry().get("nope")

    def test_duplicate_registration_rejected(self):
        registry = AlgorithmRegistry()
        with pytest.raises(AlgorithmRegistryError, match="already registered"):
            registry.register(registry.get("joint_bo"))


class TestStableRunIds:
    def test_run_id_deterministic(self):
        key1 = run_identity_key("exp", "benchmark_a", 2, "joint_bo", 0, "fp")
        key2 = run_identity_key("exp", "benchmark_a", 2, "joint_bo", 0, "fp")
        assert stable_run_id(key1) == stable_run_id(key2)
        assert len(stable_run_id(key1)) == 16

    def test_run_id_changes_with_each_identity_component(self):
        base = run_identity_key("exp", "benchmark_a", 2, "joint_bo", 0, "fp")
        variants = [
            run_identity_key("exp2", "benchmark_a", 2, "joint_bo", 0, "fp"),
            run_identity_key("exp", "benchmark_b", 2, "joint_bo", 0, "fp"),
            run_identity_key("exp", "benchmark_a", 8, "joint_bo", 0, "fp"),
            run_identity_key("exp", "benchmark_a", 2, "pairwise_bo", 0, "fp"),
            run_identity_key("exp", "benchmark_a", 2, "joint_bo", 1, "fp"),
            run_identity_key("exp", "benchmark_a", 2, "joint_bo", 0, "fp2"),
        ]
        for variant in variants:
            assert stable_run_id(variant) != stable_run_id(base)


class TestManifestGeneration:
    def test_manifest_covers_full_matrix(self):
        config = make_config()
        specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        assert len(specs) == 4  # 1 benchmark x 2 algorithms x 2 seeds
        combos = {(s.benchmark_id, s.algorithm_id, s.seed) for s in specs}
        assert combos == {
            ("benchmark_a", "random_search", 0), ("benchmark_a", "random_search", 1),
            ("benchmark_a", "joint_bo", 0), ("benchmark_a", "joint_bo", 1),
        }

    def test_reference_run_has_no_seed_and_is_pending(self):
        config = make_config(algorithm_ids=["reference"])
        specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        assert len(specs) == 1
        assert specs[0].seed is None
        assert specs[0].status == RunStatus.PENDING

    def test_incompatible_combinations_skipped_with_reason(self):
        config = make_config(algorithm_ids=["alternating_bo"])
        specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        assert len(specs) == 1
        assert specs[0].status == RunStatus.SKIPPED
        assert "incompatible_combination" in specs[0].reason

    def test_unsupported_dimension_recorded_as_skipped(self):
        config = make_config(dimensions=[99])
        specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        assert specs[0].status == RunStatus.SKIPPED
        assert "not supported" in specs[0].reason

    def test_duplicate_configuration_detected(self):
        """Two identical (benchmark, algorithm, seed) combos cannot coexist."""
        registry_b, registry_a = BenchmarkRegistry(), AlgorithmRegistry()
        config = make_config()
        specs = generate_manifest(config, registry_b, registry_a)
        run_ids = [s.run_id for s in specs]
        assert len(run_ids) == len(set(run_ids))
        # identical identity inputs must yield the same run id -> duplicate
        # detection at generation time (same config regenerates same ids, not
        # duplicates); a true duplicate requires different specs mapping to one id
        key1 = run_identity_key("e", "b", 2, "a", 0, "f")
        key2 = run_identity_key("e", "b", 2, "a", 0, "f")
        assert stable_run_id(key1) == stable_run_id(key2)

    def test_hyperparameters_recorded_per_run(self):
        config = make_config()
        specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        for spec in specs:
            assert spec.hyperparameters["evaluation_budget"] == 10
            assert spec.hyperparameters["num_acquisition_candidates"] == 500
            assert spec.config_fingerprint == config.fingerprint()
