"""Generic, application-independent optimization problem interface (M5 instruction 4).

The abstraction defines: decision-vector dimensionality, variable names and
ordering, continuous/discrete variable types, bounds, allowed discrete values,
feasibility validation, scalar objective evaluation, objective direction,
observation-noise assumptions and evaluation identifiers.

It contains NO radar equations and NO physical simulation parameters; the same
BO engine can optimize any problem built on this interface.

Constraint semantics (M4 review 3.1, M5 instruction 3.1): the
``constraint_violation`` contract is an INSTANTANEOUS mathematical constraint
on a single decision vector. It says nothing about any cumulative,
discounted or policy-level budget: those notions belong to the radar
resource layer and are not used by the M5 synthetic benchmarks.

State ordering is inherited from the interface contract: variable order is
``variable_names``, fixed for the problem's lifetime; flat vectors follow it.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

import numpy as np

VARIABLE_TYPES = ("continuous", "discrete")

#: Exact-membership tolerance for discrete variables and feasibility boundary.
FEASIBILITY_TOLERANCE: float = 1e-9


class ProblemDefinitionError(ValueError):
    """The problem definition itself is inconsistent (bounds, types, values)."""


class OptimizationProblem(ABC):
    """Abstract base class for generic scalar optimization problems."""

    def __init__(
        self,
        problem_id: str,
        variable_names: tuple[str, ...],
        variable_types: tuple[str, ...],
        lower_bounds: tuple[float, ...] | np.ndarray,
        upper_bounds: tuple[float, ...] | np.ndarray,
        discrete_values: dict[int, np.ndarray] | None = None,
        maximize: bool = True,
        observation_noise_std: float | None = None,
    ) -> None:
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        d = len(variable_names)
        if d < 1:
            raise ProblemDefinitionError("A problem needs at least one decision variable.")
        if len(variable_types) != d or lower.shape != (d,) or upper.shape != (d,):
            raise ProblemDefinitionError(
                f"variable_names ({d}), variable_types ({len(variable_types)}), lower_bounds "
                f"({lower.shape[0]}) and upper_bounds ({upper.shape[0]}) must all have equal length."
            )
        if any(t not in VARIABLE_TYPES for t in variable_types):
            raise ProblemDefinitionError(
                f"variable_types must be a subset of {VARIABLE_TYPES}; got {variable_types}."
            )
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ProblemDefinitionError("Bounds must be finite.")
        if np.any(lower >= upper):
            raise ProblemDefinitionError(f"Every lower bound must be < its upper bound; got {lower} vs {upper}.")
        discrete_values = discrete_values or {}
        for idx, values in discrete_values.items():
            if not (0 <= idx < d):
                raise ProblemDefinitionError(f"discrete_values index {idx} out of range for dimension {d}.")
            if variable_types[idx] != "discrete":
                raise ProblemDefinitionError(f"Variable {idx} ({variable_names[idx]}) is not declared discrete.")
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ProblemDefinitionError(f"Discrete values for variable {idx} must be non-empty and finite.")
            if np.any(values < lower[idx] - FEASIBILITY_TOLERANCE) or np.any(values > upper[idx] + FEASIBILITY_TOLERANCE):
                raise ProblemDefinitionError(
                    f"Discrete values for variable {idx} lie outside its bounds [{lower[idx]}, {upper[idx]}]."
                )
            if np.any(np.diff(values) <= 0):
                raise ProblemDefinitionError(f"Discrete values for variable {idx} must be ascending and distinct.")
        for idx, t in enumerate(variable_types):
            if t == "discrete" and idx not in discrete_values:
                raise ProblemDefinitionError(
                    f"Discrete variable {idx} ({variable_names[idx]}) has no allowed values."
                )
        if observation_noise_std is not None and (not np.isfinite(observation_noise_std) or observation_noise_std < 0.0):
            raise ProblemDefinitionError(
                f"observation_noise_std must be finite and >= 0 (None = deterministic); got {observation_noise_std!r}."
            )

        self.problem_id = problem_id
        self.variable_names = tuple(variable_names)
        self.variable_types = tuple(variable_types)
        self.lower_bounds = lower
        self.upper_bounds = upper
        self.discrete_values: dict[int, np.ndarray] = {
            idx: np.asarray(values, dtype=np.float64) for idx, values in discrete_values.items()
        }
        self.maximize = maximize
        self.observation_noise_std = observation_noise_std
        self.dimension = d
        self.evaluation_count = 0

    # ---- Problem-definition helpers ------------------------------------------------
    def validate_vector(self, x: np.ndarray, name: str = "x") -> np.ndarray:
        """Coerce and validate a decision vector: shape, dtype, finiteness."""
        arr = np.asarray(x, dtype=np.float64).flatten()
        if arr.shape != (self.dimension,):
            raise ProblemDefinitionError(
                f"{name} must have {self.dimension} entries (order: {self.variable_names}); got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ProblemDefinitionError(f"{name} contains non-finite values: {arr!r}.")
        return arr

    def next_evaluation_id(self) -> str:
        """Fresh evaluation identifier; also increments the evaluation counter."""
        self.evaluation_count += 1
        return f"{self.problem_id}_eval_{self.evaluation_count:05d}_{uuid.uuid4().hex[:8]}"

    # ---- Feasibility (instantaneous mathematical constraints) ------------------------
    def constraint_violation(self, x: np.ndarray) -> float:
        """Instantaneous constraint violation (>= 0; 0 = feasible).

        Default: unconstrained. Benchmarks override with explicit mathematical
        constraints. This is a LOCAL property of one decision vector only —
        it never represents cumulative, discounted or policy-level budgets
        (M4 review 3.1).
        """
        return 0.0

    def is_feasible(self, x: np.ndarray) -> bool:
        """Bounds + discrete membership + instantaneous constraint check."""
        arr = self.validate_vector(x)
        if np.any(arr < self.lower_bounds - FEASIBILITY_TOLERANCE) or np.any(
            arr > self.upper_bounds + FEASIBILITY_TOLERANCE
        ):
            return False
        for idx, values in self.discrete_values.items():
            if not np.any(np.abs(arr[idx] - values) <= FEASIBILITY_TOLERANCE):
                return False
        return self.constraint_violation(arr) <= FEASIBILITY_TOLERANCE

    # ---- Objective -------------------------------------------------------------------
    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """Scalar objective at a feasible decision vector (synthetic, deterministic
        unless observation_noise_std is declared — noise applies to OBSERVATIONS,
        not to this deterministic function)."""

    def evaluate(self, x: np.ndarray) -> float:
        """Validate, check feasibility, and evaluate the objective.

        Infeasible vectors are REJECTED (never evaluated, never repaired).
        """
        arr = self.validate_vector(x)
        if not self.is_feasible(arr):
            raise ProblemDefinitionError(
                f"evaluate() called with an infeasible vector {arr.tolist()}; infeasible candidates "
                "must be rejected before evaluation, never repaired."
            )
        return float(self.objective(arr))
