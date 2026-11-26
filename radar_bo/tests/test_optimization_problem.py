"""Tests for the generic optimization problem interface (M5 instruction 4)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic, BenchmarkBMixedConstrained
from radar_bo.optimization.bo.problem import FEASIBILITY_TOLERANCE, ProblemDefinitionError, OptimizationProblem


class Unconstrained2D(OptimizationProblem):
    """Minimal concrete problem for interface tests."""

    def __init__(self) -> None:
        super().__init__(
            problem_id="interface_test",
            variable_names=("u", "v"),
            variable_types=("continuous", "continuous"),
            lower_bounds=(0.0, -1.0),
            upper_bounds=(2.0, 1.0),
            maximize=True,
        )

    def objective(self, x: np.ndarray) -> float:
        return float(-(x[0] - 1.0) ** 2 - x[1] ** 2)


class _RawShell(OptimizationProblem):
    """Concrete shell used to drive the base __init__ with explicit kwargs."""

    def objective(self, x: np.ndarray) -> float:
        return 0.0


class TestDefinitionValidation:
    def test_valid_problem_constructs(self):
        p = Unconstrained2D()
        assert p.dimension == 2
        assert p.variable_names == ("u", "v")
        assert p.maximize is True
        assert p.observation_noise_std is None

    def _init_raw(self, **kwargs):
        defaults = dict(
            problem_id="raw",
            variable_names=("u",),
            variable_types=("continuous",),
            lower_bounds=(0.0,),
            upper_bounds=(1.0,),
        )
        defaults.update(kwargs)
        shell = _RawShell.__new__(_RawShell)
        OptimizationProblem.__init__(shell, **defaults)
        return shell

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="equal length"):
            self._init_raw(
                problem_id="bad", variable_names=("u",),
                variable_types=("continuous", "continuous"),
                lower_bounds=(0.0, 0.0), upper_bounds=(1.0, 1.0),
            )

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="lower bound"):
            self._init_raw(lower_bounds=(2.0,), upper_bounds=(1.0,))

    def test_nonfinite_bounds_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="finite"):
            self._init_raw(lower_bounds=(np.nan,), upper_bounds=(1.0,))

    def test_invalid_variable_type_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="variable_types"):
            self._init_raw(variable_types=("fuzzy",))

    def test_discrete_values_outside_bounds_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="outside its bounds"):
            self._init_raw(
                lower_bounds=(0.5,), upper_bounds=(2.0,),
                variable_types=("discrete",), discrete_values={0: np.array([0.0, 5.0])},
            )

    def test_discrete_type_requires_values(self):
        with pytest.raises(ProblemDefinitionError, match="no allowed values"):
            self._init_raw(variable_types=("discrete",))

    def test_continuous_with_discrete_values_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="not declared discrete"):
            self._init_raw(discrete_values={0: np.array([0.5, 1.0])})

    def test_invalid_noise_std_rejected(self):
        with pytest.raises(ProblemDefinitionError, match="observation_noise_std"):
            self._init_raw(observation_noise_std=-0.5)


class TestFeasibility:
    def test_bounds_check(self):
        p = Unconstrained2D()
        assert p.is_feasible([1.0, 0.0])
        assert not p.is_feasible([3.0, 0.0])  # outside bounds
        assert not p.is_feasible([1.0, -2.0])

    def test_discrete_membership_check(self):
        p = BenchmarkBMixedConstrained()
        assert p.is_feasible([1.5, 2.0])
        assert not p.is_feasible([1.55, 2.0])  # not a member

    def test_instantaneous_constraint_check(self):
        p = BenchmarkBMixedConstrained()
        assert p.constraint_violation([2.0, 2.0]) == pytest.approx(0.0, abs=FEASIBILITY_TOLERANCE)  # 2+6=8 exactly
        assert p.constraint_violation([2.0, 3.0]) == pytest.approx(1.0)  # 3+6-8
        assert not p.is_feasible([2.0, 3.0])

    def test_nonfinite_vector_rejected(self):
        p = Unconstrained2D()
        with pytest.raises(ProblemDefinitionError, match="non-finite"):
            p.validate_vector([np.nan, 0.0])

    def test_wrong_shape_rejected(self):
        p = Unconstrained2D()
        with pytest.raises(ProblemDefinitionError, match="2 entries"):
            p.validate_vector([1.0, 0.0, 3.0])


class TestEvaluation:
    def test_evaluate_validates_and_returns_scalar(self):
        p = BenchmarkAQuadratic()
        value = p.evaluate([3.5, 7.2])
        assert value == pytest.approx(10.0)

    def test_evaluate_rejects_infeasible_vector(self):
        p = BenchmarkBMixedConstrained()
        with pytest.raises(ProblemDefinitionError, match="infeasible"):
            p.evaluate([2.0, 5.0])  # violates x + 3d <= 8

    def test_evaluation_ids_and_counter(self):
        p = Unconstrained2D()
        id1 = p.next_evaluation_id()
        id2 = p.next_evaluation_id()
        assert id1 != id2
        assert p.evaluation_count == 2
        assert "interface_test" in id1

    def test_objective_direction_recorded(self):
        assert BenchmarkAQuadratic().maximize is True
