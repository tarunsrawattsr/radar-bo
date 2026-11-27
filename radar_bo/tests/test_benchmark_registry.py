"""M7 item 30: benchmark-registry unit tests."""

from __future__ import annotations

import pytest

from radar_bo.experiments.framework.benchmark_registry import (
    BenchmarkRegistry,
    BenchmarkRegistryEntry,
    BenchmarkRegistryError,
)


class TestRegistry:
    def test_all_existing_benchmarks_registered(self):
        registry = BenchmarkRegistry()
        assert set(registry.ids()) == {
            "benchmark_a", "benchmark_b", "benchmark_c",
            "separable_4d", "separable_8d", "separable_16d",
            "coupled_4d", "coupled_8d", "constrained_mixed_8d", "block_stagnation_4d",
        }

    def test_unknown_id_rejected(self):
        with pytest.raises(BenchmarkRegistryError, match="Unknown benchmark"):
            BenchmarkRegistry().get("does_not_exist")

    def test_duplicate_registration_rejected(self):
        registry = BenchmarkRegistry()
        entry = registry.get("benchmark_a")
        with pytest.raises(BenchmarkRegistryError, match="already registered"):
            registry.register(entry)

    def test_problem_returns_fresh_independent_instances(self):
        registry = BenchmarkRegistry()
        p1 = registry.get("benchmark_a").problem()
        p2 = registry.get("benchmark_a").problem()
        assert p1 is not p2
        # independent mutable state (e.g. evaluation counters) does not leak
        p1.evaluate(__import__("numpy").asarray([1.0, 2.0]))
        p1.evaluate(__import__("numpy").asarray([1.0, 2.0]))
        assert p2.evaluate(__import__("numpy").asarray([1.0, 2.0])) == p2.evaluate(
            __import__("numpy").asarray([1.0, 2.0])
        )

    def test_unsupported_dimension_rejected(self):
        with pytest.raises(BenchmarkRegistryError, match="does not support dimension"):
            BenchmarkRegistry().get("separable_4d").problem(8)

    def test_registry_metadata_complete(self):
        registry = BenchmarkRegistry()
        for benchmark_id in registry.ids():
            entry = registry.get(benchmark_id)
            assert entry.implementation_reference
            assert entry.mathematical_definition
            assert entry.constraints
            assert entry.variable_names and len(entry.variable_names) == entry.dimension
            assert all(t in ("continuous", "discrete") for t in entry.variable_types)
            assert len(entry.bounds) == entry.dimension
            assert entry.objective_direction == "maximize"
            assert entry.noise_assumption
            assert entry.reference_verification_method in (
                "analytic_exact", "enumeration_exact", "numerical_approximate",
            )

    def test_reference_kinds_distinguished(self):
        registry = BenchmarkRegistry()
        assert registry.get("benchmark_a").reference().verification_method == "analytic_exact"
        assert registry.get("benchmark_b").reference().verification_method == "enumeration_exact"
        assert registry.get("constrained_mixed_8d").reference().verification_method == "numerical_approximate"

    def test_reference_matches_recorded_values(self):
        """Reference values must match the durable archives exactly."""
        registry = BenchmarkRegistry()
        assert registry.get("benchmark_a").reference().objective_value == 10.0
        assert registry.get("separable_4d").reference().objective_value == 40.0
        assert registry.get("block_stagnation_4d").reference().objective_value == pytest.approx(9.6)

    def test_custom_entry_registration(self):
        registry = BenchmarkRegistry()
        entry = registry.get("benchmark_a")
        custom = BenchmarkRegistryEntry(
            benchmark_id="custom_benchmark", dimension=entry.dimension,
            supported_dimensions=entry.supported_dimensions, factory=entry.factory,
            reference_factory=entry.reference_factory,
            implementation_reference="custom", mathematical_definition="custom",
            variable_names=entry.variable_names, variable_types=entry.variable_types,
            bounds=entry.bounds, constraints=entry.constraints,
            objective_direction=entry.objective_direction,
            noise_assumption=entry.noise_assumption,
            reference_verification_method=entry.reference_verification_method,
        )
        registry.register(custom)
        assert registry.get("custom_benchmark") is custom
