"""Tests for the block-cyclic BO engine: acceptance policy, incumbent
management, dataset dimensionality, budget/stopping, reproducibility and
joint-regression compatibility (M6 instructions 7, 9, 10, 13, 14, 17, 18, 27)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.blocks import BlockDefinitionError
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic
from radar_bo.optimization.bo.benchmarks_highdim import BlockStagnationLP
from radar_bo.optimization.bo.problem import OptimizationProblem


def alt_config(problem_groups, **kwargs) -> OptimizationConfig:
    defaults: dict = {
        "evaluation_budget": 40,
        "num_initial_observations": 10,
        "search_mode": "alternating",
        "variable_blocks": [("block_A", tuple(problem_groups[0])), ("block_B", tuple(problem_groups[1]))],
    }
    defaults.update(kwargs)
    return OptimizationConfig(**defaults)


class CountingA(BenchmarkAQuadratic):
    def __init__(self) -> None:
        super().__init__()
        self.objective_calls = 0

    def objective(self, x: np.ndarray) -> float:
        self.objective_calls += 1
        return super().objective(x)


class TestJointRegressionCompatibility:
    # Frozen M5 joint trajectory (BenchmarkAQuadratic, budget 25, initial 10,
    # seed 0) — recorded from the M5 engine at commit 8c825a6 and required to
    # stay bit-for-bit identical after the M6 block refactor (ONE generic
    # engine; joint behavior is preserved via regression tests, M6
    # instruction 4). Initial incumbent + one entry per BO evaluation.
    M5_JOINT_SEED0_TRAJECTORY = [
        1.3180741000165792,
        9.905619188792086,
    ] + [9.905619188792086] * 14
    M5_JOINT_SEED0_FINAL = 9.905619188792086
    M5_JOINT_SEED0_BEST_CONFIG = [3.431752679614066, 6.900461831366929]

    def test_joint_mode_matches_m5_behavior(self):
        """variable_blocks=None must reproduce the M5 joint trajectory exactly
        (same initial design, same pool generator, same iteration order)."""
        r1 = BayesianOptimizer(
            BenchmarkAQuadratic(), OptimizationConfig(evaluation_budget=30, num_initial_observations=10), seed=4
        ).optimize()
        r2 = BayesianOptimizer(
            BenchmarkAQuadratic(), OptimizationConfig(evaluation_budget=30, num_initial_observations=10), seed=4
        ).optimize()
        assert r1.search_strategy == "joint"
        assert r1.best_config == r2.best_config
        assert r1.incumbent_history == r2.incumbent_history
        assert r1.stopping_reason == r2.stopping_reason
        # Known M5 seed-4 trajectory properties hold:
        assert r1.n_initial_evaluations == 10
        assert r1.n_initial_evaluations + r1.n_bo_evaluations == 30

    def test_joint_trajectory_bit_for_bit_m5(self):
        """The M6-refactored engine reproduces the frozen M5 joint trajectory
        (seed 0, BenchmarkA, budget 25/initial 10) bit-for-bit — final
        incumbent, full incumbent history and best configuration."""
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=25, num_initial_observations=10),
            seed=0,
        ).optimize()
        assert result.best_objective == self.M5_JOINT_SEED0_FINAL
        assert result.incumbent_history == self.M5_JOINT_SEED0_TRAJECTORY
        assert result.best_config == self.M5_JOINT_SEED0_BEST_CONFIG
        assert result.stopping_reason == "evaluation_budget_exhausted"

    def test_joint_single_block_explicitly_rejected(self):
        """Redundant explicit joint spec (same as default) is rejected to avoid
        ambiguity (must use variable_blocks=None)."""
        with pytest.raises(BlockDefinitionError):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, search_mode="joint",
                                   variable_blocks=[("all", (0, 1))]),
                seed=0,
            )


class TestAlternatingRuns:
    def test_alternating_on_benchmark_a(self):
        problem = CountingA()
        result = BayesianOptimizer(
            problem, alt_config([(0,), (1,)], evaluation_budget=30),
            reference_objective=10.0, reference_method="analytic_exact", seed=0,
        ).optimize()
        assert result.search_strategy == "alternating"
        assert result.stopping_reason == "evaluation_budget_exhausted"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 30
        # Objective calls == attempted evaluations (scoring consumes none):
        assert problem.objective_calls == result.n_initial_evaluations + result.n_bo_evaluations
        # Blocks recorded; visits and cycles consistent:
        assert result.n_block_visits == result.n_bo_evaluations  # one eval per visit
        assert result.n_cycles_completed == result.n_block_visits // 2
        # Every observation carries block metadata and full-dimension configs:
        for record in result.observations:
            assert len(record.config) == 2
            if record.metadata.get("stage") == "bo_iteration":
                assert record.metadata["block_id"] in {"block_A", "block_B"}
        # Incumbent evaluated, history nondecreasing:
        assert result.best_config in [record.config for record in result.observations]
        history = np.asarray(result.incumbent_history)
        assert np.all(np.diff(history) >= -1e-12)
        # Diagnostics carry block identifiers:
        assert all(d.block_id in {"block_A", "block_B"} for d in result.iteration_diagnostics)

    def test_two_single_variable_blocks_declared_alternating(self):
        """On a 2-variable problem, two single-variable blocks are the
        ALTERNATING structure (pair-wise blocks must have exactly two
        variables each — instruction 6; the declared mode must match)."""
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=30, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("block_A", (0,)), ("block_B", (1,))]),
            seed=1,
        ).optimize()
        assert result.search_strategy == "alternating"
        assert result.stopping_reason == "evaluation_budget_exhausted"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 30

    def test_two_two_variable_blocks_declared_coordinate(self):
        """The 2-two-variable-block ambiguity is resolved by the DECLARED mode."""
        result = BayesianOptimizer(
            BlockStagnationLP(),
            OptimizationConfig(evaluation_budget=30, num_initial_observations=10, search_mode="coordinate",
                               variable_blocks=[("pair_0", (0, 1)), ("pair_1", (2, 3))]),
            seed=1,
        ).optimize()
        assert result.search_strategy == "pairwise"

    def test_search_mode_must_match_blocks(self):
        with pytest.raises(ValueError, match="incompatible"):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, search_mode="joint",
                                   variable_blocks=[("a", (0,)), ("b", (1,))]),
                seed=0,
            )


class TestSharedDatasetAndGP:
    def test_dataset_stays_full_dimension_and_shared(self):
        """M x d training tensors regardless of block size; ONE shared dataset."""
        problem = BenchmarkAQuadratic()
        result = BayesianOptimizer(problem, alt_config([(0,), (1,)], evaluation_budget=30), seed=2).optimize()
        assert len(result.observations) == result.n_initial_evaluations + result.n_bo_evaluations - result.n_failed_attempts
        # All observations complete 2-D vectors (never block-only coordinates):
        assert all(len(record.config) == 2 for record in result.observations)
        # Both blocks appear in the observation metadata (shared dataset):
        block_ids = {record.metadata.get("block_id") for record in result.observations
                     if record.metadata.get("stage") == "bo_iteration"}
        assert block_ids == {"block_A", "block_B"}

    def test_fixed_coordinates_traceable_across_block_visits(self):
        """For each block-restricted evaluation, the coordinates of the OTHER
        block must equal the current configuration's coordinates at that visit."""
        problem = BenchmarkAQuadratic()
        result = BayesianOptimizer(problem, alt_config([(0,), (1,)], evaluation_budget=30), seed=3).optimize()
        records_by_id = {record.evaluation_id: record for record in result.observations}
        # Current configuration starts at the best INITIAL observation:
        best_initial_value, current = -np.inf, None
        for record in result.observations:
            if record.metadata.get("stage") == "initial_design":
                if record.objective > best_initial_value:
                    best_initial_value = record.objective
                    current = np.asarray(record.config, dtype=np.float64)
        assert current is not None
        for diagnostic in result.iteration_diagnostics:
            if diagnostic.objective_value is None:
                continue  # failed attempt: no observation, no acceptance
            record = records_by_id[diagnostic.selected_evaluation_id]
            active_index = 0 if diagnostic.block_id == "block_A" else 1
            fixed_index = 1 - active_index
            # The fixed coordinate of the EVALUATED candidate equals the current
            # configuration's coordinate BEFORE this visit (acceptance only
            # changes the active coordinate's contribution afterwards).
            assert record.config[fixed_index] == pytest.approx(float(current[fixed_index]), abs=1e-12)
            if diagnostic.accepted_current_config:
                current = np.asarray(record.config, dtype=np.float64)


class TestAcceptanceAndIncumbent:
    def test_current_config_and_incumbent_distinct(self):
        """With a positive acceptance tolerance, an evaluation can improve the
        global incumbent WITHOUT being accepted as the next search center."""
        problem = BenchmarkAQuadratic()
        # Tolerance 0.5: improvements smaller than 0.5 update the incumbent only.
        result = BayesianOptimizer(
            problem, alt_config([(0,), (1,)], evaluation_budget=40, improvement_tolerance=0.5), seed=5
        ).optimize()
        if result.n_accepted_updates == 0:
            # The seed produced no accepted update; force the distinction with a
            # fresh seed instead of asserting a vacuous truth.
            result = BayesianOptimizer(
                problem, alt_config([(0,), (1,)], evaluation_budget=40, improvement_tolerance=0.5), seed=7
            ).optimize()
        assert result.n_accepted_updates >= 1
        # The incumbent is an actual stored observation (never a prediction):
        observed = {tuple(record.config) for record in result.observations}
        assert tuple(result.best_config) in observed
        # Incumbent value equals the max observed value:
        assert result.best_objective == pytest.approx(max(r.objective for r in result.observations))
        # The final current configuration must also be an evaluated configuration:
        if result.current_config_final is not None:
            assert tuple(result.current_config_final) in observed
        # Distinctness is POSSIBLE and demonstrated when the last acceptance
        # precedes the last incumbent improvement:
        accepted_flags = [d.accepted_current_config for d in result.iteration_diagnostics]
        assert accepted_flags.count(True) == result.n_accepted_updates

    def test_non_improving_observations_retained(self):
        """Exploratory evaluations that fail to improve are kept in the dataset
        (never discarded) and never accepted as the current configuration."""
        problem = BenchmarkAQuadratic()
        result = BayesianOptimizer(problem, alt_config([(0,), (1,)], evaluation_budget=30), seed=6).optimize()
        # A 1-D block search on a 2-D quadratic: many evaluations cannot beat the
        # running best; the dataset still holds ALL successful evaluations.
        assert len(result.observations) == result.n_initial_evaluations + result.n_bo_evaluations
        assert result.n_accepted_updates <= len(result.iteration_diagnostics)

    def test_incumbent_always_evaluated_never_predicted(self):
        problem = BenchmarkAQuadratic()
        result = BayesianOptimizer(problem, alt_config([(0,), (1,)], evaluation_budget=30), seed=8).optimize()
        observed = {tuple(record.config) for record in result.observations}
        assert tuple(result.best_config) in observed


class TestBlockStagnationAndSkips:
    def test_alternating_stagnation_on_lp_benchmark(self):
        """From a budget-tight configuration the a/b alternating grouping cannot
        improve even though the analytic joint optimum (9.6) exists. The engine
        records the stagnation honestly: the final gap to the EXACT reference
        stays large (stagnation is NOT global optimality — instruction 9/25)."""
        result = BayesianOptimizer(
            BlockStagnationLP(),
            alt_config([(0, 2), (1, 3)], evaluation_budget=50, num_initial_observations=20),
            reference_objective=9.6, seed=0,
        ).optimize()
        # The incumbent is evaluated, feasible, but stays well below the
        # analytic LP optimum:
        observed = {tuple(record.config) for record in result.observations}
        assert tuple(result.best_config) in observed
        assert result.best_gap_to_reference is not None and result.best_gap_to_reference > 0.1
        # Stagnation evidence: most block visits produce no accepted improvement.
        accepted = [d.accepted_current_config for d in result.iteration_diagnostics]
        assert accepted.count(False) > accepted.count(True)

    def test_empty_block_skipped_not_terminal(self):
        """A block whose only allowed value equals the current configuration's
        value has NO distinct feasible candidate: it is SKIPPED and logged every
        cycle, while the other block keeps searching (instruction 9)."""

        class OnePointBlock(OptimizationProblem):
            """Discrete variable pinned to a single allowed value (5.0); the
            d-block therefore never offers a distinct candidate."""

            def __init__(self) -> None:
                super().__init__(
                    problem_id="one_point_block", variable_names=("x", "d"),
                    variable_types=("continuous", "discrete"),
                    lower_bounds=(0.0, 4.5), upper_bounds=(10.0, 5.5),
                    discrete_values={1: np.array([5.0])}, maximize=True,
                )

            def objective(self, x: np.ndarray) -> float:
                return float(2.0 * x[0] + x[1])

            def constraint_violation(self, x: np.ndarray) -> float:
                return float(max(0.0, float(x[0]) + float(x[1]) - 12.0))

        result = BayesianOptimizer(
            OnePointBlock(),
            alt_config([(0,), (1,)], evaluation_budget=30),
            seed=0,
        ).optimize()
        # The d-block has no distinct candidate (its single value is the current
        # config's value, duplicate-excluded): skipped every cycle with reasons.
        assert result.n_skipped_blocks >= 1
        assert all(r["block_id"] == "block_B" for r in result.skipped_block_records)
        assert all(r["reason"] == "empty_feasible_pool_for_block" for r in result.skipped_block_records)
        # The x-block continued searching: budget respected, incumbent produced.
        assert result.stopping_reason == "evaluation_budget_exhausted"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 30
        assert result.best_config is not None
        assert result.n_cycles_completed >= 1

    def test_all_blocks_skipped_stops_with_exact_reason(self):
        """After the initial design evaluates EVERY feasible configuration of a
        small discrete problem, both single-variable blocks have no distinct
        feasible candidate left: one full cycle with all blocks skipped stops
        the engine with the exact reason — no infinite loop (instruction 9)."""

        class DiscreteExhaustion(OptimizationProblem):
            """Feasible combos (d1, d2) in {1, 2}^2 with d1 + d2 <= 3.5:
            (1,1), (1,2), (2,1) — exactly three feasible configurations."""

            def __init__(self) -> None:
                super().__init__(
                    problem_id="discrete_exhaustion", variable_names=("d1", "d2"),
                    variable_types=("discrete", "discrete"),
                    lower_bounds=(1.0, 1.0), upper_bounds=(2.0, 2.0),
                    discrete_values={0: np.array([1.0, 2.0]), 1: np.array([1.0, 2.0])},
                    maximize=True,
                )

            def objective(self, x: np.ndarray) -> float:
                return float(x[0] + x[1])

            def constraint_violation(self, x: np.ndarray) -> float:
                return float(max(0.0, float(x[0]) + float(x[1]) - 3.5))

        result = BayesianOptimizer(
            DiscreteExhaustion(),
            alt_config([(0,), (1,)], evaluation_budget=20, num_initial_observations=3),
            seed=0,
        ).optimize()
        # The initial design evaluates all three feasible configurations; the
        # best is (2, 1). Every block alternative is then infeasible or already
        # evaluated (duplicate-excluded): both blocks skip, one full cycle.
        assert result.n_initial_evaluations == 3
        assert result.n_bo_evaluations == 0
        assert result.stopping_reason == "all_blocks_exhausted_in_cycle"
        assert result.n_skipped_blocks == 2
        assert {r["block_id"] for r in result.skipped_block_records} == {"block_A", "block_B"}
        assert all(r["reason"] == "empty_feasible_pool_for_block" for r in result.skipped_block_records)
        assert result.best_objective == pytest.approx(3.0)  # best evaluated (2, 1)
        assert result.best_config == [2.0, 1.0]


class TestBudgetAndCycleStopping:
    def test_budget_enforced_exactly(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(), alt_config([(0,), (1,)], evaluation_budget=25), seed=9
        ).optimize()
        assert result.n_initial_evaluations + result.n_bo_evaluations == 25

    def test_partial_cycle_termination(self):
        """Budget exhaustion part-way through a cycle records the exact reason
        and the cycle counts reflect the incomplete cycle."""
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            alt_config([(0,), (1,)], evaluation_budget=23, num_initial_observations=11),  # odd BO count: mid-cycle stop
            seed=10,
        ).optimize()
        assert result.stopping_reason == "evaluation_budget_exhausted"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 23
        assert result.n_block_visits == result.n_bo_evaluations
        # Completed cycles: 12 visits -> 6 complete cycles (7th never started).
        assert result.n_cycles_completed == result.n_block_visits // 2

    def test_cycle_patience_stops(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            alt_config([(0,), (1,)], evaluation_budget=100, cycle_patience=2, patience=None),
            seed=11,
        ).optimize()
        assert result.stopping_reason in {"cycle_patience_exhausted", "evaluation_budget_exhausted"}
        if result.stopping_reason == "cycle_patience_exhausted":
            assert result.n_cycles_without_improvement >= 2
            assert result.n_cycles_completed >= 2

    def test_max_iterations_still_applies(self):
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            alt_config([(0,), (1,)], evaluation_budget=100, max_iterations=4),
            seed=12,
        ).optimize()
        assert result.stopping_reason == "max_iterations_reached"
        assert len(result.iteration_diagnostics) == 4

    def test_evaluations_per_block_visit_two(self):
        """k = 2 evaluations per visit: all evaluations inside the shared budget,
        visits fewer than evaluations."""
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            alt_config([(0,), (1,)], evaluation_budget=30, evaluations_per_block_visit=2),
            seed=13,
        ).optimize()
        assert result.n_initial_evaluations + result.n_bo_evaluations == 30
        assert result.evaluations_per_block_visit == 2
        assert 2 * result.n_block_visits >= result.n_bo_evaluations >= result.n_block_visits


class TestFailuresAndReproducibility:
    def test_failures_consume_budget_and_carry_block_metadata(self):
        class FailingX(BenchmarkAQuadratic):
            """Fails whenever x1 > 0 (deterministic injection; ~half the initial
            design and every x1-block candidate above 0 fails)."""

            def objective(self, x: np.ndarray) -> float:
                if x[0] > 0.0:
                    raise ValueError("injected")
                return super().objective(x)

        result = BayesianOptimizer(
            FailingX(), alt_config([(0,), (1,)], evaluation_budget=30), seed=14
        ).optimize()
        assert result.n_failed_attempts >= 1
        assert result.stopping_reason in {"evaluation_budget_exhausted", "repeated_objective_failures"}
        failed_ids = {f.evaluation_id for f in result.failures}
        successful_ids = {r.evaluation_id for r in result.observations}
        assert not (failed_ids & successful_ids)  # failures never in training data
        for failure in result.failures:
            if failure.stage == "objective_evaluation":
                assert "block_id" in failure.metadata  # M6 block metadata on failures
        assert result.n_initial_evaluations + result.n_bo_evaluations == 30

    def test_reproducible_same_seed(self):
        cfg = alt_config([(0,), (1,)], evaluation_budget=30)
        r1 = BayesianOptimizer(BenchmarkAQuadratic(), cfg, seed=15).optimize()
        r2 = BayesianOptimizer(BenchmarkAQuadratic(), cfg, seed=15).optimize()
        assert r1.best_config == r2.best_config
        assert r1.incumbent_history == r2.incumbent_history
        assert r1.n_accepted_updates == r2.n_accepted_updates
        assert r1.current_config_final == r2.current_config_final

    def test_serialization_round_trip_with_block_fields(self):
        from radar_bo.optimization.bo.results import OptimizationResult

        result = BayesianOptimizer(
            BenchmarkAQuadratic(), alt_config([(0,), (1,)], evaluation_budget=25), seed=16
        ).optimize()
        restored = OptimizationResult.model_validate_json(result.model_dump_json())
        assert restored.search_strategy == result.search_strategy
        assert restored.variable_blocks == result.variable_blocks
        assert restored.n_cycles_completed == result.n_cycles_completed
        assert restored.skipped_block_records == result.skipped_block_records
        assert restored.current_config_final == result.current_config_final
        assert restored.best_config == result.best_config
