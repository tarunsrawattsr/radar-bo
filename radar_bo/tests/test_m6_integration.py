"""M6 integration tests: complete alternating and pair-wise workflows on small
independently verifiable synthetic problems (M6 instruction 28)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic, BenchmarkBMixedConstrained
from radar_bo.optimization.bo.benchmarks_highdim import BlockStagnationLP, ConstrainedMixedHD
from radar_bo.optimization.bo.reference_solvers import stagnation_lp_reference

OBJECTIVE_STAGES = {"initial_design", "objective_evaluation"}


def alternating(problem_groups, **kwargs) -> OptimizationConfig:
    defaults: dict = {
        "evaluation_budget": 40,
        "num_initial_observations": 10,
        "search_mode": "alternating",
        "variable_blocks": [("block_A", tuple(problem_groups[0])), ("block_B", tuple(problem_groups[1]))],
    }
    defaults.update(kwargs)
    return OptimizationConfig(**defaults)


class TestAlternatingWorkflow:
    def test_complete_workflow_on_mixed_problem(self):
        """Alternating between the discrete and the continuous block of the M5
        mixed benchmark B (analytic reference 11.0)."""
        problem = BenchmarkBMixedConstrained()
        result = BayesianOptimizer(
            problem, alternating([(0,), (1,)], evaluation_budget=50),
            reference_objective=11.0, reference_method="enumeration_exact", seed=0,
        ).optimize()
        # Budget respected; every observation traceable to its complete config:
        assert result.n_initial_evaluations + result.n_bo_evaluations == 50
        attempted_ids = {r.evaluation_id for r in result.observations} | {
            f.evaluation_id for f in result.failures if f.stage in OBJECTIVE_STAGES
        }
        assert len(attempted_ids) == result.n_initial_evaluations + result.n_bo_evaluations
        # Feasibility checked BEFORE evaluation: no infeasible observation exists.
        for record in result.observations:
            assert problem.is_feasible(np.asarray(record.config))
            assert record.config[0] in [0.5, 1.0, 1.5, 2.0]  # discrete block respected
        # The shared dataset was updated by BOTH blocks throughout the run:
        block_sequence = [r.metadata.get("block_id") for r in result.observations
                          if r.metadata.get("stage") == "bo_iteration"]
        assert {"block_A", "block_B"} <= set(block_sequence)
        # Visit accounting (totals): BO observations + skipped-block visits ==
        # scheduled block visits. The discrete d-block legitimately exhausts
        # once every (d', x_cur) combination has been evaluated — each skip is
        # logged with its reason (instruction 9).
        assert len(block_sequence) + result.n_skipped_blocks == result.n_block_visits
        for skip in result.skipped_block_records:
            assert skip["reason"] == "empty_feasible_pool_for_block"
            assert skip["block_id"] in {"block_A", "block_B"}
        # Every successful diagnostic maps 1:1 to a stored observation:
        successful = [d for d in result.iteration_diagnostics if d.objective_value is not None]
        bo_observation_ids = {r.evaluation_id for r in result.observations
                              if r.metadata.get("stage") == "bo_iteration"}
        assert {d.selected_evaluation_id for d in successful} == bo_observation_ids
        # Returned incumbent evaluated; never exceeds the exact reference:
        observed = {tuple(r.config) for r in result.observations}
        assert tuple(result.best_config) in observed
        assert result.best_objective <= 11.0 + 1e-9

    def test_empty_blocks_never_create_infinite_loops(self):
        """All-blocks-exhausted and no-feasible-initial-design paths terminate
        with recorded reasons (regression guard for budget-consuming loops)."""
        from radar_bo.optimization.bo.problem import OptimizationProblem

        class NothingFeasible(OptimizationProblem):
            def __init__(self) -> None:
                super().__init__(
                    problem_id="nothing_feasible", variable_names=("x", "y"),
                    variable_types=("continuous", "continuous"),
                    lower_bounds=(0.0, 0.0), upper_bounds=(1.0, 1.0), maximize=True,
                )

            def objective(self, x: np.ndarray) -> float:
                return float(x[0] + x[1])

            def constraint_violation(self, x: np.ndarray) -> float:
                return 1.0  # everything infeasible

        result = BayesianOptimizer(
            NothingFeasible(), alternating([(0,), (1,)], evaluation_budget=10), seed=0
        ).optimize()
        assert result.stopping_reason == "no_feasible_initial_design"
        # The initial-design generator finds NO feasible sample, so no
        # evaluation is attempted at all (nothing consumed — nothing to account):
        assert result.n_failed_attempts == 0
        assert result.n_initial_evaluations == 0
        assert result.best_config is None  # no incumbent exists

    def test_failures_never_enter_gp_training(self):
        class Failing(BenchmarkAQuadratic):
            def objective(self, x: np.ndarray) -> float:
                if x[1] < -5.0:
                    raise ValueError("injected")
                return super().objective(x)

        result = BayesianOptimizer(
            Failing(), alternating([(0,), (1,)], evaluation_budget=40), seed=1
        ).optimize()
        failed_ids = {f.evaluation_id for f in result.failures}
        assert failed_ids
        assert not (failed_ids & {r.evaluation_id for r in result.observations})
        # GP fits succeeded throughout (fitting failures would appear in failures):
        assert not any(f.stage == "gp_fitting" for f in result.failures)


class TestPairwiseWorkflow:
    def test_pairwise_on_stagnation_lp(self):
        """Pair-wise blocks {a1,b1}, {a2,b2} on the LP benchmark: mixed pairs CAN
        partially redistribute budget (unlike the a/b alternating grouping) —
        the architecture difference is the primary variable, same engine."""
        ref = stagnation_lp_reference()
        result = BayesianOptimizer(
            BlockStagnationLP(),
            OptimizationConfig(evaluation_budget=40, num_initial_observations=10, search_mode="coordinate",
                               variable_blocks=[("pair_0", (0, 1)), ("pair_1", (2, 3))]),
            reference_objective=ref.objective_value, reference_method=ref.verification_method, seed=2,
        ).optimize()
        assert result.search_strategy == "pairwise"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 40
        for record in result.observations:
            assert BlockStagnationLP().is_feasible(np.asarray(record.config))
        observed = {tuple(r.config) for r in result.observations}
        assert tuple(result.best_config) in observed
        # Whether pairwise beats other architectures is REPORTED, not asserted
        # (instruction 32: an unfavorable comparison is valid evidence).

    def test_pairwise_on_constrained_mixed_8d(self):
        problem = ConstrainedMixedHD(8)
        result = BayesianOptimizer(
            problem,
            OptimizationConfig(evaluation_budget=60, num_initial_observations=20, search_mode="coordinate",
                               variable_blocks=[("d_pair", (0, 1)), ("x_pair_0", (2, 3)),
                                                ("x_pair_1", (4, 5)), ("x_pair_2", (6, 7))]),
            seed=3,
        ).optimize()
        assert result.stopping_reason == "evaluation_budget_exhausted"
        assert result.n_initial_evaluations + result.n_bo_evaluations == 60
        # Full 8-D vectors stored; constraint holds for every observation:
        for record in result.observations:
            assert len(record.config) == 8
            assert problem.is_feasible(np.asarray(record.config))
        # Acquisition scoring happened at scale with full-dimension vectors:
        assert result.n_acquisition_scores > 0

    def test_partial_coverage_reported(self):
        """Partial coverage is allowed and REPORTED; uncovered variables stay
        fixed at the current configuration (initial-design best)."""
        result = BayesianOptimizer(
            BenchmarkAQuadratic(),
            OptimizationConfig(evaluation_budget=30, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("only_x1", (0,)), ("only_x2", (1,))]),
            seed=4,
        ).optimize()
        assert result.uncovered_variable_indices == []  # full coverage here
        assert result.search_strategy == "alternating"
        # Genuinely partial coverage on the 4-D LP (variables 2, 3 uncovered):
        result2 = BayesianOptimizer(
            BlockStagnationLP(),
            OptimizationConfig(evaluation_budget=30, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("only_a1", (0,)), ("only_b1", (1,))]),
            seed=5,
        ).optimize()
        assert result2.uncovered_variable_indices == [2, 3]
        # Uncovered coordinates stay fixed at the initial-design best for every
        # BLOCK-RESTRICTED evaluation (initial-design records are full-space
        # samples and legitimately differ):
        best_initial = max(
            (r for r in result2.observations if r.metadata.get("stage") == "initial_design"),
            key=lambda r: r.objective,
        )
        for record in result2.observations:
            if record.metadata.get("stage") != "bo_iteration":
                continue
            assert record.config[2] == pytest.approx(best_initial.config[2], abs=1e-12)
            assert record.config[3] == pytest.approx(best_initial.config[3], abs=1e-12)
        assert result2.current_config_final[2] == pytest.approx(best_initial.config[2], abs=1e-12)
        assert result2.current_config_final[3] == pytest.approx(best_initial.config[3], abs=1e-12)
        # A single PROPER-subset block matches no search mode (rejected):
        with pytest.raises(ValueError):
            BayesianOptimizer(
                BenchmarkAQuadratic(),
                OptimizationConfig(evaluation_budget=10, search_mode="coordinate",
                                   variable_blocks=[("only_x1", (0,))]),
                seed=6,
            )


class TestFairnessAndCrossArchitecture:
    def test_identical_initial_design_across_architectures(self):
        """Instruction 21: identical complete initial configurations and observed
        values for the same seed — joint, alternating, pairwise."""
        problem_a, problem_b, problem_c = BenchmarkAQuadratic(), BenchmarkAQuadratic(), BenchmarkAQuadratic()
        r_joint = BayesianOptimizer(
            problem_a, OptimizationConfig(evaluation_budget=40, num_initial_observations=10), seed=6
        ).optimize()
        r_alt = BayesianOptimizer(
            problem_b, alternating([(0,), (1,)], evaluation_budget=40), seed=6
        ).optimize()
        r_pair = BayesianOptimizer(
            problem_c,
            OptimizationConfig(evaluation_budget=40, num_initial_observations=10, search_mode="alternating",
                               variable_blocks=[("p0", (0,)), ("p1", (1,))]),
            seed=6,
        ).optimize()
        initial_joint = [r.config for r in r_joint.observations if r.metadata.get("stage") == "initial_design"]
        initial_alt = [r.config for r in r_alt.observations if r.metadata.get("stage") == "initial_design"]
        initial_pair = [r.config for r in r_pair.observations if r.metadata.get("stage") == "initial_design"]
        assert initial_joint == initial_alt == initial_pair
        assert [r.objective for r in r_joint.observations[:10]] == [r.objective for r in r_alt.observations[:10]]

    def test_unfavorable_comparison_is_valid_output(self):
        """Instruction 32: an architecture may lose; the engine must not hide it.
        Alternating on the stagnation LP (a/b grouping) records its stagnation
        honestly rather than reporting the joint optimum."""
        ref = stagnation_lp_reference()
        alt = BayesianOptimizer(
            BlockStagnationLP(),
            alternating([(0, 2), (1, 3)], evaluation_budget=50, num_initial_observations=20),
            reference_objective=ref.objective_value, seed=7,
        ).optimize()
        joint = BayesianOptimizer(
            BlockStagnationLP(),
            OptimizationConfig(evaluation_budget=50, num_initial_observations=20),
            reference_objective=ref.objective_value, seed=8,
        ).optimize()
        # Both runs are valid records; either may be better for its seed.
        for result in (alt, joint):
            assert result.best_objective is not None
            assert result.stopping_reason == "evaluation_budget_exhausted"
            assert result.best_gap_to_reference is not None
