"""Higher-dimensional synthetic benchmarks for block-architecture comparisons
(M6 instruction 19). ALL parameters are SYNTHETIC and self-defined; none
derives from the reference papers. The M5 low-dimensional benchmarks (A/B/C)
remain in ``benchmarks.py`` for regression testing.

Categories (instruction 19):

* A. Separable objective — variable groups optimize largely independently;
     analytic exact optimum (per-coordinate completion of the square).
* B. Coupled objective — cross-variable interactions through a fixed
     positive-definite mixing matrix; analytic exact optimum (the coupling
     shifts the SURROUNDINGS, not the location, of the optimum).
* C. Constrained mixed-variable objective — two discrete + several continuous
     variables with a SHARED instantaneous resource constraint; independent
     reference by enumeration over the discrete combinations with a
     one-dimensional KKT multiplier solved by bisection per combination
     (numerical_approximate, tight tolerance; never assumed).
* D. Block-stagnation example — a linear program in which, from any
     constraint-tight configuration, NO single two-variable block of the
     alternating grouping can improve the objective while a better JOINT
     configuration exists (the optimum shifts budget between groups).
     Analytic exact reference via the fractional-knapsack/LP greedy argument.

Each benchmark class exposes ``recommended_blocks`` for the alternating and
pair-wise strategies so tests and experiments share one definition.
"""

from __future__ import annotations

import numpy as np

from radar_bo.optimization.bo.problem import OptimizationProblem


def _center_values(dimension: int) -> np.ndarray:
    """Deterministic synthetic per-variable centers in [-3, 3] (fixed constants,
    documented — not random and not tuned to any method)."""
    indices = np.arange(1, dimension + 1, dtype=np.float64)
    return np.round(3.0 * np.cos(1.3 * indices), 6)


def _mixing_matrix(dimension: int) -> np.ndarray:
    """Fixed synthetic PD matrix M = L L^T + 0.5 I with a deterministic band
    structure (couples each variable with its neighbors; positive definite by
    construction)."""
    L = np.zeros((dimension, dimension))
    for i in range(dimension):
        L[i, i] = 0.9 + 0.05 * ((i % 4) + 1)
        if i + 1 < dimension:
            L[i, i + 1] = 0.35 + 0.03 * (i % 3)
        if i + 2 < dimension:
            L[i, i + 2] = 0.15
    return L @ L.T + 0.5 * np.eye(dimension)


class SeparableQuadraticHD(OptimizationProblem):
    """Category A: f(x) = sum_i [10 - (x_i - c_i)^2] on [-5, 5]^d, maximize.

    Analytic exact optimum x* = c (per-coordinate), f* = 10 d. The variables
    are INDEPENDENT: block-restricted search loses (almost) nothing here —
    this is the control problem for the dimension study.
    """

    def __init__(self, dimension: int = 4) -> None:
        if dimension not in (4, 8, 16):
            raise ValueError(f"dimension must be one of 4/8/16 (study dimensions); got {dimension}.")
        self.centers = _center_values(dimension)
        super().__init__(
            problem_id=f"synthetic_separable_{dimension}d",
            variable_names=tuple(f"x{i + 1}" for i in range(dimension)),
            variable_types=tuple(["continuous"] * dimension),
            lower_bounds=tuple([-5.0] * dimension),
            upper_bounds=tuple([5.0] * dimension),
            maximize=True,
            observation_noise_std=None,
        )
        self.reference_objective: float | None = 10.0 * dimension
        self.reference_method = "analytic exact (per-coordinate completion of the square)"

    def objective(self, x: np.ndarray) -> float:
        return float(np.sum(10.0 - (np.asarray(x, dtype=np.float64) - self.centers) ** 2))

    @property
    def recommended_blocks(self) -> dict[str, list[list[int]]]:
        d = self.dimension
        return {
            "alternating": [list(range(0, d // 2)), list(range(d // 2, d))],
            "pairwise": [[2 * j, 2 * j + 1] for j in range(d // 2)],
        }


class CoupledQuadraticHD(OptimizationProblem):
    """Category B: f(x) = 20 - (x - c)^T M (x - c) on [-5, 5]^d, maximize.

    M is a fixed synthetic positive-definite matrix (see ``_mixing_matrix``),
    so the objective couples NEIGHBORING variables. Analytic exact optimum:
    x* = c, f* = 20 (the quadratic form vanishes at x = c because M is PD).
    Coupling makes single-block moves less informative: improving one block
    can be undone by interactions with the fixed remainder.
    """

    def __init__(self, dimension: int = 4) -> None:
        if dimension not in (4, 8, 16):
            raise ValueError(f"dimension must be one of 4/8/16 (study dimensions); got {dimension}.")
        self.centers = _center_values(dimension)
        self.M = _mixing_matrix(dimension)
        super().__init__(
            problem_id=f"synthetic_coupled_{dimension}d",
            variable_names=tuple(f"x{i + 1}" for i in range(dimension)),
            variable_types=tuple(["continuous"] * dimension),
            lower_bounds=tuple([-5.0] * dimension),
            upper_bounds=tuple([5.0] * dimension),
            maximize=True,
            observation_noise_std=None,
        )
        self.reference_objective: float | None = 20.0
        self.reference_method = "analytic exact (x* = c: PD quadratic form vanishes at the center)"

    def objective(self, x: np.ndarray) -> float:
        delta = np.asarray(x, dtype=np.float64) - self.centers
        return float(20.0 - float(delta @ self.M @ delta))

    @property
    def recommended_blocks(self) -> dict[str, list[list[int]]]:
        d = self.dimension
        return {
            "alternating": [list(range(0, d // 2)), list(range(d // 2, d))],
            "pairwise": [[2 * j, 2 * j + 1] for j in range(d // 2)],
        }


class ConstrainedMixedHD(OptimizationProblem):
    """Category C: constrained mixed-variable problem (dim 8 or 16).

    Decision vector: the FIRST TWO variables are discrete in
    {0.5, 1.0, 1.5, 2.0}; the remaining ``d - 2`` variables are continuous
    in [0, 4]. Objective (maximize)::

        f(x) = 2 (d0 + d1) + sum_i [6 - 0.5 (x_i - m_i)^2]

    Shared instantaneous resource constraint::

        sum_i x_i + 3 (d0 + d1) <= C   (C = 14, synthetic)

    Larger discrete values add objective but consume shared budget. Because
    every continuous term shares the SAME curvature (0.5), the inner
    continuous optimum for a fixed budget R = C - 3 (d0 + d1) has the KKT
    form x_i = clamp(m_i - lambda, 0, 4) with a single multiplier lambda >= 0
    chosen so that sum x_i = R (binding case) — the reference solver
    enumerates the 16 discrete combinations and solves the one-dimensional
    multiplier by bisection (numerical_approximate; the value is COMPUTED,
    never assumed, and BO results exceeding it are reported and investigated).
    """

    DISCRETE_D: np.ndarray = np.array([0.5, 1.0, 1.5, 2.0])
    BUDGET_C = 14.0

    def __init__(self, dimension: int = 8) -> None:
        if dimension not in (8, 16):
            raise ValueError(f"dimension must be 8 or 16 for the constrained mixed benchmark; got {dimension}.")
        n_cont = dimension - 2
        self.cont_centers = np.round(1.2 + 0.35 * (np.arange(n_cont) % 5), 6)  # in [1.2, 2.6]
        super().__init__(
            problem_id=f"synthetic_constrained_mixed_{dimension}d",
            variable_names=tuple(["d0", "d1"] + [f"x{i}" for i in range(n_cont)]),
            variable_types=tuple(["discrete", "discrete"] + ["continuous"] * n_cont),
            lower_bounds=tuple([0.5, 0.5] + [0.0] * n_cont),
            upper_bounds=tuple([2.0, 2.0] + [4.0] * n_cont),
            discrete_values={0: self.DISCRETE_D, 1: self.DISCRETE_D},
            maximize=True,
            observation_noise_std=None,
        )
        self.reference_objective: float | None = None  # computed by the reference solver
        self.reference_method = (
            "enumeration over 16 discrete combinations + KKT multiplier bisection "
            "(numerical_approximate; computed by the independent solver)"
        )

    def objective(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float64)
        d0, d1 = float(arr[0]), float(arr[1])
        cont = arr[2:]
        return float(2.0 * (d0 + d1) + np.sum(6.0 - 0.5 * (cont - self.cont_centers) ** 2))

    def constraint_violation(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float64)
        return float(max(0.0, float(np.sum(arr[2:])) + 3.0 * (float(arr[0]) + float(arr[1])) - self.BUDGET_C))

    @staticmethod
    def inner_continuous_optimum(centers: np.ndarray, budget_r: float, upper: float = 4.0) -> tuple[np.ndarray, bool]:
        """KKT inner optimum: x_i = clamp(m_i - lambda, 0, upper), sum x_i = R.

        Returns (x, binding). Lambda solved by bisection to 1e-12 (independent
        of any BO component)."""
        centers = np.asarray(centers, dtype=np.float64)
        unconstrained = np.clip(centers, 0.0, upper)
        if float(unconstrained.sum()) <= budget_r:
            return unconstrained, False

        def total(lam: float) -> float:
            return float(np.clip(centers - lam, 0.0, upper).sum())

        lo, hi = 0.0, float(centers.max()) + 1.0  # g(hi) < 0 guaranteed: total(hi)=0 <= R (R >= 0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid) > budget_r:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        return np.clip(centers - lam, 0.0, upper), True

    @property
    def recommended_blocks(self) -> dict[str, list[list[int]]]:
        d = self.dimension
        return {
            "alternating": [[0, 1], list(range(2, d))],
            "pairwise": [[0, 1]] + [[2 + 2 * j, 3 + 2 * j] for j in range((d - 2) // 2)],
        }


class BlockStagnationLP(OptimizationProblem):
    """Category D: block-stagnation example (dim 4).

    Decision vector [a1, b1, a2, b2], each in [0, 1.6]; shared constraint
    a1 + b1 + a2 + b2 <= 3.2; objective (maximize)::

        f = 3 a1 + b1 + 3 a2 + b2

    The 'a' variables are worth 3 per unit, the 'b' variables 1. The LP
    optimum (fractional-knapsack greedy: fill highest-weight coordinates to
    their bounds first) is a1 = a2 = 1.6, b1 = b2 = 0, f* = 9.6 (analytic
    exact). From ANY constraint-tight configuration, the ALTERNATING grouping
    {a1, a2} / {b1, b2} cannot improve: the a-group's total is capped by the
    fixed b-group consumption (equal within-group weights make redistribution
    worthless) and vice versa — a better JOINT configuration exists (moving
    budget from the b-group to the a-group) but NO single-block improvement
    does. This demonstrates block-wise stagnation WITHOUT proving global
    optimality of the stagnation point.
    """

    UPPER = 1.6
    BUDGET = 3.2

    def __init__(self) -> None:
        super().__init__(
            problem_id="synthetic_block_stagnation_4d",
            variable_names=("a1", "b1", "a2", "b2"),
            variable_types=tuple(["continuous"] * 4),
            lower_bounds=tuple([0.0] * 4),
            upper_bounds=tuple([self.UPPER] * 4),
            maximize=True,
            observation_noise_std=None,
        )
        self.reference_objective: float | None = 9.6
        self.reference_method = (
            "analytic exact (LP greedy / fractional knapsack: fill weight-3 coordinates to their bounds)"
        )
        self.analytic_optimum: tuple[float, ...] = (self.UPPER, 0.0, self.UPPER, 0.0)

    def objective(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=np.float64)
        return float(3.0 * arr[0] + arr[1] + 3.0 * arr[2] + arr[3])

    def constraint_violation(self, x: np.ndarray) -> float:
        return float(max(0.0, float(np.sum(np.asarray(x, dtype=np.float64))) - self.BUDGET))

    @property
    def recommended_blocks(self) -> dict[str, list[list[int]]]:
        return {
            "alternating": [[0, 2], [1, 3]],   # a-group / b-group: stagnates when budget-tight
            "pairwise": [[0, 1], [2, 3]],      # mixed pairs: can partially redistribute
        }


def all_m6_benchmarks() -> dict[str, OptimizationProblem]:
    """Fresh M6 benchmark instances keyed by study name (synthetic, labeled)."""
    return {
        "separable_4d": SeparableQuadraticHD(4),
        "separable_8d": SeparableQuadraticHD(8),
        "separable_16d": SeparableQuadraticHD(16),
        "coupled_4d": CoupledQuadraticHD(4),
        "coupled_8d": CoupledQuadraticHD(8),
        "constrained_mixed_8d": ConstrainedMixedHD(8),
        "block_stagnation_4d": BlockStagnationLP(),
    }
