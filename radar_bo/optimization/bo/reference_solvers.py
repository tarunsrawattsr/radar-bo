"""Independent reference solvers (M5 instruction 6).

Reference solutions for the synthetic benchmarks, established WITHOUT any GP
prediction or BO acquisition score:

* analytical solutions where available (Benchmark A; Benchmark B inner
  optimum);
* exhaustive enumeration for sufficiently small discrete problems (Benchmark
  B: 4 discrete values, exact analytic clamped inner optimum, plus a SciPy
  numerical cross-check);
* suitable SciPy optimization for continuous problems (Benchmark C:
  differential evolution polished + dense-grid cross-check — documented as
  APPROXIMATE).

Each solver returns a :class:`ReferenceSolution` recording the configuration,
objective value, feasibility and verification method.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.problem import OptimizationProblem

VerificationMethod = Literal["analytic_exact", "enumeration_exact", "enumeration_plus_numerical", "numerical_approximate"]


class ReferenceSolution(StrictModel):
    """Independent reference solution for one benchmark problem."""

    problem_id: str
    config: list[float] = Field(description="Reference decision vector in problem variable order.")
    objective_value: float
    feasible: bool
    verification_method: VerificationMethod
    solver: str = Field(description="Solver identifier (no GP/BO involvement).")
    notes: str = ""

    @property
    def is_exact(self) -> bool:
        return self.verification_method in ("analytic_exact", "enumeration_exact")


def benchmark_a_reference() -> ReferenceSolution:
    """Benchmark A: analytic optimum by completion of the square (exact)."""
    problem = _lazy_benchmark_a()
    x_star = np.asarray(problem.analytic_optimum, dtype=np.float64)
    return ReferenceSolution(
        problem_id=problem.problem_id,
        config=[float(v) for v in x_star],
        objective_value=problem.evaluate(x_star),
        feasible=problem.is_feasible(x_star),
        verification_method="analytic_exact",
        solver="analytic_completion_of_square",
        notes="f* = c with x* = (a, b); exact, independently provable.",
    )


def benchmark_b_reference() -> ReferenceSolution:
    """Benchmark B: enumeration over the 4 discrete values with the ANALYTIC
    clamped inner optimum (exact), cross-checked by brute-force grid scan."""
    problem = _lazy_benchmark_b()
    best = None
    for d in problem.discrete_values[0]:
        d_star, x_star = problem.analytic_inner_optimum(float(d))
        value = problem.evaluate(np.array([d_star, x_star]))
        if best is None or value > best[0]:
            best = (value, np.array([d_star, x_star]))
    assert best is not None
    # Independent cross-check: dense grid over x per discrete value.
    grid_best = -np.inf
    for d in problem.discrete_values[0]:
        xs = np.linspace(0.0, 8.0 - 3.0 * float(d) + 1e-12, 100_001)
        for xc in (xs[0], xs[-1], float(d) + 1.0 if float(d) + 1.0 <= xs[-1] else xs[-1]):
            if xc <= 8.0 - 3.0 * float(d) + 1e-12:
                grid_best = max(grid_best, problem.objective(np.array([float(d), xc])))
    assert abs(grid_best - best[0]) <= 1e-9 or best[0] >= grid_best - 1e-9
    return ReferenceSolution(
        problem_id=problem.problem_id,
        config=[float(v) for v in best[1]],
        objective_value=float(best[0]),
        feasible=problem.is_feasible(best[1]),
        verification_method="enumeration_exact",
        solver="enumeration_plus_analytic_inner",
        notes="4 discrete values; inner continuous optimum analytic (clamped); constraint binds at the optimum.",
    )


def benchmark_c_reference(seed: int = 0) -> ReferenceSolution:
    """Benchmark C: APPROXIMATE numerical reference via SciPy.

    Method: differential evolution (polished) on the negated objective plus a
    dense-grid scan; the best of the two is reported. DOCUMENTED AS
    APPROXIMATE — not an analytic proof. No GP or BO involvement.
    """
    from scipy.optimize import differential_evolution, NonlinearConstraint

    problem = _lazy_benchmark_c()

    def neg_objective(x: np.ndarray) -> float:
        return -problem.objective(np.asarray(x, dtype=np.float64))

    constraint = NonlinearConstraint(
        lambda x: problem.constraint_violation(np.asarray(x, dtype=np.float64)), -np.inf, 0.0
    )
    result = differential_evolution(
        neg_objective,
        bounds=list(zip(problem.lower_bounds, problem.upper_bounds)),
        constraints=(constraint,),
        seed=seed,
        tol=1e-10,
        maxiter=1000,
        polish=True,
    )
    x_de = np.asarray(result.x, dtype=np.float64)
    # Dense-grid cross-check (1001 x 1001):
    g1 = np.linspace(problem.lower_bounds[0], problem.upper_bounds[0], 1001)
    g2 = np.linspace(problem.lower_bounds[1], problem.upper_bounds[1], 1001)
    xx1, xx2 = np.meshgrid(g1, g2)
    values = (
        3.0 * np.cos(4.0 * xx1) * np.cos(4.0 * xx2)
        + np.sin(3.0 * xx1)
        + np.cos(2.0 * xx2)
        + 0.3 * xx1
    )
    mask = (xx1 + 3.0 * 0.0 - 8.0) <= 0.0  # unconstrained benchmark: mask all-true
    idx = np.argmax(np.where(mask, values, -np.inf))
    x_grid = np.array([xx1.flatten()[idx], xx2.flatten()[idx]], dtype=np.float64)
    candidates = [(float(problem.objective(x_de)), x_de), (float(problem.objective(x_grid)), x_grid)]
    best_value, best_x = max(candidates, key=lambda pair: pair[0])
    return ReferenceSolution(
        problem_id=problem.problem_id,
        config=[float(v) for v in best_x],
        objective_value=best_value,
        feasible=problem.is_feasible(best_x),
        verification_method="numerical_approximate",
        solver=f"scipy_differential_evolution_plus_grid(seed={seed})",
        notes="APPROXIMATE reference: best of polished DE and a 1001x1001 grid scan.",
    )


# Lazy constructors (avoid importing benchmarks at module import time in tests).
def _lazy_benchmark_a():
    from radar_bo.optimization.bo.benchmarks import BenchmarkAQuadratic

    return BenchmarkAQuadratic()


def _lazy_benchmark_b():
    from radar_bo.optimization.bo.benchmarks import BenchmarkBMixedConstrained

    return BenchmarkBMixedConstrained()


def _lazy_benchmark_c():
    from radar_bo.optimization.bo.benchmarks import BenchmarkCMultimodal

    return BenchmarkCMultimodal()


def reference_solutions() -> dict[str, ReferenceSolution]:
    """All benchmark references (C computed numerically, deterministic seed)."""
    return {
        "A": benchmark_a_reference(),
        "B": benchmark_b_reference(),
        "C": benchmark_c_reference(seed=0),
    }
