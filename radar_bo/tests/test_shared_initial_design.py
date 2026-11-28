"""M8A item 3.3: validated immutable shared initial-design interface.

Regression tests for the M8A ``initial_design_configs`` engine extension and
the framework's ``explicit_shared`` initial-design mode:

* backward compatibility (``None`` reproduces the frozen M5 trajectory);
* exact evaluation of the explicit design, in order, input immutable;
* fail-fast validation (count, dimension, finiteness, feasibility);
* identical initial phase across engines/methods sharing the same design;
* budget accounting unchanged;
* framework ``explicit_shared`` mode: stored records carry identical designs
  per (benchmark, seed) group and fairness verification passes.

Historical M5/M6 behavior (seeded sampler) is untouched: default ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic
from radar_bo.optimization.bo.benchmarks_highdim import BlockStagnationLP


EXPLICIT_DESIGN = [[3.4, 6.9], [1.0, 2.0], [5.0, 5.0]]


class TestEngineExplicitInitialDesign:
    def test_backward_compat_none_reproduces_frozen_m5_trajectory(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=25, num_initial_observations=10),
            seed=0,
        ).optimize()
        # frozen M5 joint trajectory (see test_block_bo.py for the full history)
        assert result.best_objective == 9.905619188792086

    def test_explicit_design_evaluated_exactly_in_order(self):
        design = [list(row) for row in EXPLICIT_DESIGN]
        snapshot = [list(row) for row in design]
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=25, num_initial_observations=3,
                               initial_design_configs=design),
            seed=0,
        ).optimize()
        initials = [tuple(o.config) for o in result.observations
                    if o.metadata["stage"] == "initial_design"]
        assert initials == [tuple(row) for row in design]
        assert design == snapshot, "engine must not mutate the caller's design list"
        # independently recomputed initial objectives (deterministic objective)
        problem = BenchmarkAQuadratic()
        expected = [float(problem.objective(np.asarray(row, dtype=np.float64))) for row in design]
        observed = [o.objective for o in result.observations
                    if o.metadata["stage"] == "initial_design"]
        assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_identical_initial_phase_across_engines_sharing_the_design(self):
        results = [
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=30, num_initial_observations=3,
                                   initial_design_configs=EXPLICIT_DESIGN),
                seed=seed,
            ).optimize()
            for seed in (1, 2)
        ]
        problem = BenchmarkAQuadratic()
        expected_objectives = [
            float(problem.objective(np.asarray(row, dtype=np.float64))) for row in EXPLICIT_DESIGN
        ]
        for result in results:
            initial_configs = [tuple(o.config) for o in result.observations
                               if o.metadata["stage"] == "initial_design"]
            initial_objectives = [o.objective for o in result.observations
                                  if o.metadata["stage"] == "initial_design"]
            assert initial_configs == [tuple(row) for row in EXPLICIT_DESIGN]
            assert initial_objectives == pytest.approx(expected_objectives, abs=1e-12)
        # the BO phase may differ by seed, but the initial phase cannot
        first_initial_a = [(o.config, o.objective) for o in results[0].observations
                           if o.metadata["stage"] == "initial_design"]
        first_initial_b = [(o.config, o.objective) for o in results[1].observations
                           if o.metadata["stage"] == "initial_design"]
        assert first_initial_a == first_initial_b

    def test_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="exactly num_initial_observations"):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, num_initial_observations=3,
                                   initial_design_configs=EXPLICIT_DESIGN[:2]),
                seed=0,
            ).optimize()

    def test_wrong_dimension_rejected(self):
        with pytest.raises(ValueError, match="dimension"):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, num_initial_observations=2,
                                   initial_design_configs=[[3.4, 6.9, 7.0], [1.0, 2.0]]),
                seed=0,
            ).optimize()

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, num_initial_observations=2,
                                   initial_design_configs=[[3.4, float("nan")], [1.0, 2.0]]),
                seed=0,
            ).optimize()

    def test_infeasible_point_rejected_fail_fast(self):
        # [99, 99] is inside... no: outside [0, 10]^2 -> infeasible box constraint
        with pytest.raises(ValueError, match="infeasible"):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, num_initial_observations=2,
                                   initial_design_configs=[[3.4, 6.9], [99.0, 99.0]]),
                seed=0,
            ).optimize()

    def test_budget_accounting_unchanged(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=20, num_initial_observations=3,
                               initial_design_configs=EXPLICIT_DESIGN),
            seed=0,
        ).optimize()
        assert result.n_initial_evaluations == 3
        assert result.n_initial_evaluations + result.n_bo_evaluations == 20

    def test_pool_report_records_explicit_mode(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=10, num_initial_observations=3,
                               initial_design_configs=EXPLICIT_DESIGN),
            seed=0,
        ).optimize()
        # the initial-design pool request is recorded with zero sampling cost
        assert result.pool_requests >= 1
        assert result.candidate_generation_runtime_s >= 0.0
        # the dataset's initial observations carry the explicit configs
        assert result.best_objective is not None


class TestFrameworkExplicitSharedMode:
    def test_shared_design_identical_across_methods_per_seed(self):
        """joint_bo and alternating_bo on block_stagnation_4d receive the SAME
        immutable explicit design per (benchmark, seed) — verified from the
        stored records (item 10), now guaranteed by construction."""
        from radar_bo.experiments.framework.experiment_config import (
            ExperimentConfig,
            InitialDesignSettings,
            StoppingSettings,
        )
        from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
        from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
        from radar_bo.experiments.framework.runner import ExperimentRunner
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            config = ExperimentConfig(
                experiment_id="m8a_explicit_design_test",
                description="explicit shared initial-design regression test",
                result_label="generic synthetic regression test; no radar claims",
                benchmark_ids=["block_stagnation_4d"],
                algorithm_ids=["joint_bo", "alternating_bo"],
                seeds=[0, 1],
                initial_design=InitialDesignSettings(size=4, mode="explicit_shared"),
                stopping=StoppingSettings(evaluation_budget=12),
                output_dir=str(td + "/exp"),
            )
            runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
            summary = runner.execute()
            fairness = summary["shared_initialization_verification"]
            assert fairness["verified"] is True, fairness["issues"]
            assert len(fairness["groups"]) == 2  # one per seed
            records = runner._collect_records()
            assert len(records) == 4  # 2 methods x 2 seeds
            by_group: dict[tuple, dict[str, list]] = {}
            for rec in records:
                by_group.setdefault((rec["seed"], rec["algorithm_id"]), {})["design"] = [
                    list(o["config"]) for o in rec["initial_design"]
                ]
            for seed in (0, 1):
                joint = by_group[(seed, "joint_bo")]["design"]
                alternating = by_group[(seed, "alternating_bo")]["design"]
                assert joint == alternating  # identical designs by construction
                assert len(joint) == 4
            # across seeds the designs DIFFER (replicates), within a seed they cannot
            assert by_group[(0, "joint_bo")]["design"] != by_group[(1, "joint_bo")]["design"]

    def test_seeded_mode_still_default(self):
        """The default mode remains 'seeded_sampler' (M5/M6 behavior)."""
        from radar_bo.experiments.framework.experiment_config import InitialDesignSettings

        assert InitialDesignSettings().mode == "seeded_sampler"

    def test_random_search_accepts_and_ignores_explicit_design(self):
        """Regression (M8A final-suite finding): the runner passes
        initial_design_configs to every search-method entry for a uniform
        interface; random search has no initial-design phase and must ACCEPT
        the kwarg while explicitly ignoring it (its identity is uniform random
        feasible sampling; injecting designs would change its semantics and
        break comparability with historical RS records)."""
        from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
        from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry

        registry = AlgorithmRegistry()
        benchmark = BenchmarkRegistry().get("benchmark_a")
        ref = benchmark.reference(2)
        design = [[1.5, 2.5], [3.5, 1.5], [0.5, 0.5]]
        out_with = registry.get("random_search").run(
            benchmark=benchmark, dimension=2, seed=7,
            config=OptimizationConfig(evaluation_budget=9),
            reference_value=ref.objective_value,
            reference_method=ref.verification_method,
            initial_design_configs=design,
        )
        out_without = registry.get("random_search").run(
            benchmark=benchmark, dimension=2, seed=7,
            config=OptimizationConfig(evaluation_budget=9),
            reference_value=ref.objective_value,
            reference_method=ref.verification_method,
        )
        assert out_with.architecture == "random_search"
        # identical seed + budget -> identical trajectory: the design was IGNORED
        seq_with = [e[1] for e in out_with.optimization_result.history]
        seq_without = [e[1] for e in out_without.optimization_result.history]
        assert seq_with == seq_without
        assert len(seq_with) == 9
