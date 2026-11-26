"""Synthetic mathematical benchmark problems (M5 instruction 5).

ALL parameters are SYNTHETIC (labeled); none derives from the reference
papers. These benchmarks validate the generic BO engine only. They do not
use the M4 allocation-independent diagnostic tracking provider.

* Benchmark A (continuous quadratic): smooth bounded maximization with a
  known ANALYTIC optimum — validates the basic GP + acquisition workflow.
* Benchmark B (constrained mixed-variable): one discrete and one continuous
  variable with group interaction and a shared instantaneous resource
  constraint; the optimum is determined INDEPENDENTLY by enumeration over
  the 4 discrete values with the analytic clamped inner optimum.
* Benchmark C (multimodal continuous): multiple local optima to investigate
  exploration behavior; its global optimum is documented as an APPROXIMATE
  numerical reference (SciPy differential evolution + dense grid), not an
  analytic proof.
"""

from __future__ import annotations

import numpy as np

from radar_bo.optimization.bo.problem import FEASIBILITY_TOLERANCE, OptimizationProblem


class BenchmarkAQuadratic(OptimizationProblem):
    """Benchmark A: maximize ``f(x) = c - (x1 - a)^2 - (x2 - b)^2`` on [-10, 10]^2.

    Analytic optimum: ``x* = (a, b)``, ``f* = c`` — proven by completion of the
    square (independent verification, exact).
    Synthetic parameters: a = 3.5, b = 7.2, c = 10.0.
    """

    reference_objective: float | None = 10.0
    reference_method: str = "analytic (completion of the square): x* = (3.5, 7.2), f* = 10.0 (exact)"
    analytic_optimum: tuple[float, ...] = (3.5, 7.2)

    def __init__(self) -> None:
        super().__init__(
            problem_id="synthetic_quadratic_A",
            variable_names=("x1", "x2"),
            variable_types=("continuous", "continuous"),
            lower_bounds=(-10.0, -10.0),
            upper_bounds=(10.0, 10.0),
            maximize=True,
            observation_noise_std=None,  # deterministic observations
        )
        self.a, self.b, self.c = 3.5, 7.2, 10.0

    def objective(self, x: np.ndarray) -> float:
        return float(self.c - (x[0] - self.a) ** 2 - (x[1] - self.b) ** 2)


class BenchmarkBMixedConstrained(OptimizationProblem):
    """Benchmark B: constrained mixed-variable problem.

    Decision vector ``[d, x]`` with:
    * discrete ``d`` in {0.5, 1.0, 1.5, 2.0} (synthetic action values);
    * continuous ``x`` in [0, 10];
    * objective ``f(d, x) = -(x - (d + 1))^2 + 6 d`` (group interaction: the
      continuous optimum moves with the discrete value);
    * instantaneous shared-resource constraint ``g(d, x) = x + 3 d - 8 <= 0``.

    INDEPENDENT optimum by enumeration + analytic clamped inner solution:
    for each d, the unconstrained inner optimum x0 = d + 1 is clamped to the
    feasible interval [0, 8 - 3 d]:
      d = 0.5: x0 = 1.5 (feasible)        -> f = 3.0
      d = 1.0: x0 = 2.0 (feasible)        -> f = 6.0
      d = 1.5: x0 = 2.5 (feasible)        -> f = 9.0
      d = 2.0: x0 = 3.0 INFEASIBLE (3 + 6 = 9 > 8); clamped to x = 2 -> f = -(2-3)^2 + 12 = 11.0
    Global optimum: f* = 11.0 at (d, x) = (2.0, 2.0), with the constraint
    BINDING (x = 8 - 3 d) — verified exactly.
    """

    reference_objective: float | None = 11.0
    reference_method: str = "enumeration over 4 discrete values + analytic clamped inner optimum (exact)"
    analytic_optimum: tuple[float, ...] = (2.0, 2.0)

    DISCRETE_D: np.ndarray = np.array([0.5, 1.0, 1.5, 2.0])

    def __init__(self) -> None:
        super().__init__(
            problem_id="synthetic_mixed_constrained_B",
            variable_names=("d_discrete", "x_continuous"),
            variable_types=("discrete", "continuous"),
            lower_bounds=(0.5, 0.0),
            upper_bounds=(2.0, 10.0),
            discrete_values={0: self.DISCRETE_D},
            maximize=True,
            observation_noise_std=None,
        )

    def objective(self, x: np.ndarray) -> float:
        d, xc = float(x[0]), float(x[1])
        return float(-(xc - (d + 1.0)) ** 2 + 6.0 * d)

    def constraint_violation(self, x: np.ndarray) -> float:
        """Instantaneous shared-resource constraint: max(0, x + 3 d - 8)."""
        return float(max(0.0, float(x[1]) + 3.0 * float(x[0]) - 8.0))

    def analytic_inner_optimum(self, d: float) -> tuple[float, float]:
        """Exact inner optimum over x for a given discrete value (verification aid)."""
        x_unconstrained = d + 1.0
        x_feasible_upper = 8.0 - 3.0 * d
        x_star = min(max(x_unconstrained, 0.0), x_feasible_upper)
        return float(d), x_star


class BenchmarkCMultimodal(OptimizationProblem):
    """Benchmark C: multimodal continuous maximization on [-2.5, 2.5]^2.

    Objective (synthetic)::

        f(x1, x2) = 3 cos(4 x1) cos(4 x2) + sin(3 x1) + cos(2 x2) + 0.3 x1

    The trigonometric product term creates multiple separated local optima;
    the purpose is to investigate BO exploration behavior. The global optimum
    is NOT analytically proven: the reference is an APPROXIMATE numerical
    solution (SciPy differential evolution polished, cross-checked on a dense
    grid; see ``radar_bo.optimization.bo.reference_solvers``). The reference
    value is computed by the solver, never assumed.
    """

    reference_objective: float | None = None  # filled by the numerical reference solver
    reference_method: str = "numerical (SciPy differential evolution, polished; dense-grid cross-check; APPROXIMATE)"

    def __init__(self) -> None:
        super().__init__(
            problem_id="synthetic_multimodal_C",
            variable_names=("x1", "x2"),
            variable_types=("continuous", "continuous"),
            lower_bounds=(-2.5, -2.5),
            upper_bounds=(2.5, 2.5),
            maximize=True,
            observation_noise_std=None,
        )

    def objective(self, x: np.ndarray) -> float:
        x1, x2 = float(x[0]), float(x[1])
        return float(
            3.0 * np.cos(4.0 * x1) * np.cos(4.0 * x2) + np.sin(3.0 * x1) + np.cos(2.0 * x2) + 0.3 * x1
        )


def all_benchmarks() -> dict[str, OptimizationProblem]:
    """The three M5 benchmark instances (fresh objects; synthetic, labeled)."""
    return {
        "A": BenchmarkAQuadratic(),
        "B": BenchmarkBMixedConstrained(),
        "C": BenchmarkCMultimodal(),
    }
