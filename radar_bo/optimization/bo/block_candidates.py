"""Block-restricted candidate generation (M6 instruction 8).

For a selected block the generator obtains the CURRENT COMPLETE CONFIGURATION,
samples candidate values ONLY for the block's active coordinates (uniform
within their bounds; discrete variables choose from allowed values), embeds
them into the current configuration with every fixed coordinate preserved
EXACTLY, and validates each COMPLETE candidate against the generic problem
interface (bounds + discrete membership + instantaneous constraints).

The full-vector validation is essential: shared mathematical constraints may
depend on variables belonging to DIFFERENT blocks, so validating only the
active coordinates would be unsound (instruction 8). Infeasible candidates
are REJECTED and counted — never repaired, and fixed variables are never
silently modified to make a candidate feasible.

An empty feasible pool is REPORTED (empty array + statistics), not hidden:
block-wise search can legitimately run out of feasible neighbors while other
blocks remain searchable (instruction 9; stagnation does not prove global
optimality).
"""

from __future__ import annotations

import time

import numpy as np
from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.blocks import VariableBlock
from radar_bo.optimization.bo.candidates import CandidatePoolReport
from radar_bo.optimization.bo.dataset import DUPLICATE_TOLERANCE
from radar_bo.optimization.bo.problem import OptimizationProblem


class BlockCandidateGenerator:
    """Seeded feasible sampler over ONE block's active coordinates."""

    def __init__(
        self,
        problem: OptimizationProblem,
        block: VariableBlock,
        num_candidates: int = 500,
        seed: int = 0,
        max_attempts_multiplier: int = 50,
    ) -> None:
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1; got {num_candidates}.")
        self.problem = problem
        self.block = block
        self.num_candidates = int(num_candidates)
        self.seed = int(seed)
        self.max_attempts_multiplier = int(max_attempts_multiplier)

    def _sample_active(self, rng: np.random.Generator) -> np.ndarray:
        values = np.empty(self.block.block_dimension, dtype=np.float64)
        for pos, idx in enumerate(self.block.active_indices):
            if self.problem.variable_types[idx] == "discrete":
                values[pos] = rng.choice(self.problem.discrete_values[idx])
            else:
                values[pos] = rng.uniform(self.problem.lower_bounds[idx], self.problem.upper_bounds[idx])
        return values

    def generate_pool(
        self,
        current_config: np.ndarray,
        evaluated_configs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, CandidatePoolReport]:
        """Generate feasible COMPLETE candidate vectors (pool shape (k, d_full)).

        Returns ``(pool, report)``; the pool may be EMPTY when the block has
        no feasible distinct neighbor of the current configuration.
        """
        problem = self.problem
        base = np.asarray(current_config, dtype=np.float64).flatten()
        if base.shape != (problem.dimension,):
            raise ValueError(
                f"current_config must have {problem.dimension} entries; got shape {base.shape}."
            )
        if not np.all(np.isfinite(base)):
            raise ValueError("current_config contains non-finite values.")
        rng = np.random.default_rng(self.seed)
        pool: list[np.ndarray] = []
        n_generated = 0
        n_infeasible = 0
        n_dup_pool = 0
        n_dup_evaluated = 0
        max_attempts = self.num_candidates * self.max_attempts_multiplier
        generation_started = time.perf_counter()
        feasibility_seconds = 0.0

        evaluated_list = (
            [] if evaluated_configs is None
            else [np.asarray(c, dtype=np.float64).flatten() for c in evaluated_configs]
        )

        while len(pool) < self.num_candidates and n_generated < max_attempts:
            n_generated += 1
            x = self.block.embed(self._sample_active(rng), base)  # fixed coords preserved exactly
            t_feas = time.perf_counter()
            feasible = problem.is_feasible(x)  # FULL-vector validation (shared constraints)
            feasibility_seconds += time.perf_counter() - t_feas
            if not feasible:
                n_infeasible += 1
                continue
            if any(float(np.max(np.abs(x - p))) <= DUPLICATE_TOLERANCE for p in pool):
                n_dup_pool += 1
                continue
            if any(float(np.max(np.abs(x - p))) <= DUPLICATE_TOLERANCE for p in evaluated_list):
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
            excluded_evaluated=True,
            generation_duration_s=time.perf_counter() - generation_started,
            feasibility_validation_s=feasibility_seconds,
            notes=[
                f"block-restricted generation: block '{self.block.block_id}' varies indices "
                f"{list(self.block.active_indices)} (block dimension {self.block.block_dimension} of "
                f"full dimension {problem.dimension}); fixed coordinates preserved exactly from the "
                "current configuration; feasibility validated on COMPLETE vectors.",
                "Acquisition scoring of this pool consumes ZERO objective evaluations.",
            ],
        )
        if pool:
            return np.vstack(pool), report
        return np.empty((0, problem.dimension), dtype=np.float64), report
