"""Feasible candidate-pool generation (M5 instruction 19).

Generates a finite pool (provisional experimental setting: 500) of candidate
configurations for acquisition scoring. Every candidate is validated against
the problem's bounds, discrete membership and instantaneous constraints
BEFORE scoring; invalid candidates are REJECTED and counted, never repaired.

Sampling: seeded uniform (continuous within bounds; discrete uniform choice
from allowed values). Sobol sampling was considered and deliberately NOT
included in M5: its treatment of discrete variables under constraints needs
its own documented design and is not required for the deterministic baseline.

Clearly distinguished quantities: the pool size (500 configurations SCORED
per iteration) is not the objective-evaluation count (scoring consumes ZERO
objective evaluations).
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.dataset import DUPLICATE_TOLERANCE
from radar_bo.optimization.bo.problem import OptimizationProblem


class CandidatePoolReport(StrictModel):
    """Statistics of one candidate-pool generation (rejection accounting)."""

    requested_size: int
    final_size: int
    n_generated: int
    rejected_infeasible: int
    rejected_duplicate_within_pool: int
    rejected_duplicate_evaluated: int
    seed: int
    excluded_evaluated: bool
    notes: list[str] = Field(default_factory=list)


class CandidateGenerationError(RuntimeError):
    """Raised when no feasible candidate can be generated (empty pool)."""


class CandidateGenerator:
    """Seeded feasible sampler for acquisition candidate pools."""

    def __init__(
        self,
        problem: OptimizationProblem,
        num_candidates: int = 500,
        seed: int = 0,
        exclude_evaluated: bool = True,
        max_attempts_multiplier: int = 50,
    ) -> None:
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1; got {num_candidates}.")
        self.problem = problem
        self.num_candidates = int(num_candidates)
        self.seed = int(seed)
        # For deterministic problems, exclude already-evaluated configurations
        # (do not spend objective evaluations on known points). For noisy
        # problems, re-evaluating sampled configurations may be legitimate.
        self.exclude_evaluated = bool(exclude_evaluated)
        self.max_attempts_multiplier = int(max_attempts_multiplier)

    def _sample(self, rng: np.random.Generator) -> np.ndarray:
        x = np.empty(self.problem.dimension, dtype=np.float64)
        for idx in range(self.problem.dimension):
            if self.problem.variable_types[idx] == "discrete":
                x[idx] = rng.choice(self.problem.discrete_values[idx])
            else:
                x[idx] = rng.uniform(self.problem.lower_bounds[idx], self.problem.upper_bounds[idx])
        return x

    def generate_pool(self, evaluated_configs: np.ndarray | None = None) -> tuple[np.ndarray, CandidatePoolReport]:
        """Generate the feasible candidate pool.

        Returns ``(pool (k, d) float64, report)``. The pool may be EMPTY when
        the feasible region is tiny; the caller decides how to proceed (the
        BO loop treats an empty pool as a stopping condition). Raises nothing
        for emptiness here — emptiness is REPORTED, not hidden.
        """
        problem = self.problem
        rng = np.random.default_rng(self.seed)
        allow_duplicates_of_evaluated = not self.exclude_evaluated
        pool: list[np.ndarray] = []
        n_generated = 0
        n_infeasible = 0
        n_dup_pool = 0
        n_dup_evaluated = 0
        max_attempts = self.num_candidates * self.max_attempts_multiplier

        evaluated_list = (
            [] if evaluated_configs is None else [np.asarray(c, dtype=np.float64).flatten() for c in evaluated_configs]
        )

        while len(pool) < self.num_candidates and n_generated < max_attempts:
            n_generated += 1
            x = self._sample(rng)
            if not problem.is_feasible(x):
                n_infeasible += 1
                continue
            if any(float(np.max(np.abs(x - p))) <= DUPLICATE_TOLERANCE for p in pool):
                n_dup_pool += 1
                continue
            if not allow_duplicates_of_evaluated and any(
                float(np.max(np.abs(x - p))) <= DUPLICATE_TOLERANCE for p in evaluated_list
            ):
                n_dup_evaluated += 1
                continue
            pool.append(x)

        report = CandidatePoolReport(
            requested_size=self.num_candidates,
            final_size=len(pool),
            n_generated=n_generated,
            rejected_infeasible=n_infeasible,
            rejected_duplicate_within_pool=n_dup_pool,
            rejected_duplicate_evaluated=n_dup_evaluated,
            seed=self.seed,
            excluded_evaluated=self.exclude_evaluated,
            notes=[
                "Pool size is the number of acquisition-SCORED candidates; it consumes zero "
                "objective evaluations.",
                "Invalid candidates are rejected and counted, never repaired.",
            ],
        )
        if pool:
            return np.stack(pool).astype(np.float64), report
        return np.zeros((0, problem.dimension), dtype=np.float64), report
