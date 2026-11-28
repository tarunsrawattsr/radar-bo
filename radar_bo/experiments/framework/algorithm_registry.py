"""Algorithm registry (M7 item 7).

Registers the EXISTING methods — random feasible search, joint BO
(variable_blocks=None, the M5 baseline), alternating BO, pair-wise
coordinate BO — and the independent direct reference solvers. All methods
reuse the existing generic M5/M6 engine through its public interfaces; the
registry does NOT duplicate any optimization algorithm.

The reference "algorithm" produces the registry's independent reference
solution for a benchmark (analytic / enumeration / numerical, no BO) and is
recorded per run so every BO run's gap refers to exactly that value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistryEntry
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.random_search import RandomSearchOptimizer


class AlgorithmRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class AlgorithmRunResult:
    """Structured output of one algorithm execution (conversion handled by the runner)."""

    kind: str  # "search" | "reference"
    architecture: str
    optimization_result: Any | None  # OptimizationResult for search methods
    reference_solution: Any | None  # ReferenceSolution for reference "runs"
    algorithm_version: str


@dataclass(frozen=True)
class AlgorithmRegistryEntry:
    algorithm_id: str
    kind: str  # "search" | "reference"
    architecture: str  # "joint" | "alternating" | "pairwise" | "random_search" | "reference"
    version: str
    runner: Callable[..., AlgorithmRunResult] = None  # type: ignore[assignment]
    requires_recommended_blocks: bool = False

    def run(self, benchmark: BenchmarkRegistryEntry, dimension: int | None, seed: int,
            config: OptimizationConfig, reference_value: float | None,
            reference_method: str | None, initial_design_configs: list[list[float]] | None = None
            ) -> AlgorithmRunResult:
        return self.runner(
            benchmark=benchmark, dimension=dimension, seed=seed, config=config,
            reference_value=reference_value, reference_method=reference_method,
            initial_design_configs=initial_design_configs,
        )


def _recommended(problem, mode: str) -> list[tuple[str, tuple[int, ...]]] | None:
    """Recommended block grouping from the benchmark (M6 convention)."""
    recommended = getattr(problem, "recommended_blocks", None)
    if recommended is None:
        return None
    if mode == "alternating":
        return [(f"block_{i}", tuple(g)) for i, g in enumerate(recommended["alternating"])]
    if mode == "pairwise":
        return [(f"pair_{i}", tuple(p)) for i, p in enumerate(recommended["pairwise"])]
    raise AlgorithmRegistryError(f"Unknown recommended-blocks mode: {mode}")


def _run_random_search(benchmark, dimension, seed, config, reference_value, reference_method,
                       initial_design_configs=None) -> AlgorithmRunResult:
    # M8A: the runner passes initial_design_configs to every search-method entry for a
    # uniform interface. Random search has NO initial-design phase (its identity is
    # uniform random feasible sampling), so the design is explicitly IGNORED here —
    # injecting it would change RS semantics and break comparability with historical
    # RS records. Fairness vs BO rests on equal budgets, which the runner enforces.
    del initial_design_configs
    problem = benchmark.problem(dimension)
    budget = config.evaluation_budget
    assert budget is not None
    result = RandomSearchOptimizer(problem, budget, seed=seed).optimize()
    return AlgorithmRunResult("search", "random_search", result, None, ALGORITHM_VERSIONS["random_search"])


def _run_bo(architecture: str):
    def _runner(benchmark, dimension, seed, config, reference_value, reference_method,
                initial_design_configs=None) -> AlgorithmRunResult:
        problem = benchmark.problem(dimension)
        if architecture == "joint":
            specs = None
            mode = "joint"
        else:
            specs = _recommended(problem, architecture)
            if specs is None:
                raise AlgorithmRegistryError(
                    f"Benchmark '{benchmark.benchmark_id}' provides no recommended {architecture} grouping."
                )
            mode = "alternating" if architecture == "alternating" else "coordinate"
        update = {"search_mode": mode, "variable_blocks": specs, "random_seed": seed}
        if initial_design_configs is not None:
            # M8A item 3.3: validated immutable shared design (the engine re-validates
            # dimension/finiteness/feasibility and fails fast on any violation).
            update["initial_design_configs"] = [list(map(float, row)) for row in initial_design_configs]
        run_config = config.model_copy(update=update)
        result = BayesianOptimizer(
            problem, run_config,
            reference_objective=reference_value,
            reference_method=reference_method,
            seed=seed,
        ).optimize()
        return AlgorithmRunResult("search", architecture, result, None, ALGORITHM_VERSIONS["bo_engine"])

    return _runner


def _run_reference(benchmark, dimension, seed, config, reference_value, reference_method) -> AlgorithmRunResult:
    reference = benchmark.reference(dimension)
    return AlgorithmRunResult("reference", "reference", None, reference, ALGORITHM_VERSIONS["reference"])


ALGORITHM_VERSIONS = {
    "random_search": "random_search/m6-engine-companion/1.0",
    "bo_engine": "bo_single_candidate_log_ei_matern52 (block-cyclic engine, M6 refactor; joint preserved bit-for-bit vs M5)",
    "reference": "independent reference solvers (analytic/enumeration/numerical; no BO)",
}


def _default_registry() -> dict[str, AlgorithmRegistryEntry]:
    entries: dict[str, AlgorithmRegistryEntry] = {
        "random_search": AlgorithmRegistryEntry(
            "random_search", "search", "random_search", ALGORITHM_VERSIONS["random_search"], _run_random_search,
        ),
        "joint_bo": AlgorithmRegistryEntry(
            "joint_bo", "search", "joint", ALGORITHM_VERSIONS["bo_engine"], _run_bo("joint"),
        ),
        "alternating_bo": AlgorithmRegistryEntry(
            "alternating_bo", "search", "alternating", ALGORITHM_VERSIONS["bo_engine"], _run_bo("alternating"),
            requires_recommended_blocks=True,
        ),
        "pairwise_bo": AlgorithmRegistryEntry(
            "pairwise_bo", "search", "pairwise", ALGORITHM_VERSIONS["bo_engine"], _run_bo("pairwise"),
            requires_recommended_blocks=True,
        ),
        "reference": AlgorithmRegistryEntry(
            "reference", "reference", "reference", ALGORITHM_VERSIONS["reference"], _run_reference,
        ),
    }
    return entries


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._entries = _default_registry()

    def register(self, entry: AlgorithmRegistryEntry) -> None:
        if entry.algorithm_id in self._entries:
            raise AlgorithmRegistryError(f"Algorithm id already registered: {entry.algorithm_id}")
        self._entries[entry.algorithm_id] = entry

    def get(self, algorithm_id: str) -> AlgorithmRegistryEntry:
        if algorithm_id not in self._entries:
            raise AlgorithmRegistryError(
                f"Unknown algorithm id '{algorithm_id}'; registered: {sorted(self._entries)}"
            )
        return self._entries[algorithm_id]

    def ids(self) -> list[str]:
        return sorted(self._entries)
