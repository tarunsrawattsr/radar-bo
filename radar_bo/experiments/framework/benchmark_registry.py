"""Benchmark registry (M7 item 6).

Every mathematical benchmark used by the experiment framework is registered
with its identifier, implementation reference, variable metadata, constraint
definition, objective direction, noise assumption and independent reference
solution. The runner accesses benchmarks ONLY through registered interfaces —
no problem construction is hard-coded in the runner.

Existing M5/M6 benchmarks are registered here unchanged (their definitions
and references are versioned implementations, preserved bit-for-bit; the
durable archives were produced from exactly these classes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
)
from radar_bo.optimization.bo.benchmarks_highdim import (
    BlockStagnationLP,
    ConstrainedMixedHD,
    CoupledQuadraticHD,
    SeparableQuadraticHD,
)
from radar_bo.optimization.bo.problem import OptimizationProblem
from radar_bo.optimization.bo.reference_solvers import (
    ReferenceSolution,
    benchmark_a_reference,
    benchmark_b_reference,
    benchmark_c_reference,
    constrained_mixed_hd_reference,
    coupled_hd_reference,
    separable_hd_reference,
    stagnation_lp_reference,
)


class BenchmarkRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkRegistryEntry:
    """One registered benchmark (item 6 field list)."""

    benchmark_id: str
    dimension: int
    supported_dimensions: tuple[int, ...]
    factory: Callable[[int], OptimizationProblem] = field(compare=False)
    reference_factory: Callable[[int], ReferenceSolution] = field(compare=False)
    implementation_reference: str  # versioned implementation reference
    mathematical_definition: str
    variable_names: tuple[str, ...]
    variable_types: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    constraints: str
    objective_direction: str  # "maximize"
    noise_assumption: str
    reference_verification_method: str  # analytic_exact | enumeration_exact | numerical_approximate

    def problem(self, dimension: int | None = None) -> OptimizationProblem:
        """A FRESH, independent problem instance (runner must never share one)."""
        d = self.dimension if dimension is None else dimension
        if d not in self.supported_dimensions:
            raise BenchmarkRegistryError(
                f"Benchmark '{self.benchmark_id}' does not support dimension {d} "
                f"(supported: {list(self.supported_dimensions)})."
            )
        return self.factory(d)

    def reference(self, dimension: int | None = None) -> ReferenceSolution:
        d = self.dimension if dimension is None else dimension
        if d not in self.supported_dimensions:
            raise BenchmarkRegistryError(f"Benchmark '{self.benchmark_id}' does not support dimension {d}.")
        return self.reference_factory(d)


def _entry_from_problem(
    benchmark_id: str,
    problem: OptimizationProblem,
    factory: Callable[[int], OptimizationProblem],
    reference_factory: Callable[[int], ReferenceSolution],
    implementation_reference: str,
    mathematical_definition: str,
    constraints: str,
    supported_dimensions: tuple[int, ...] | None = None,
) -> BenchmarkRegistryEntry:
    """Build a registry entry by introspecting a fresh problem instance."""
    dims = supported_dimensions if supported_dimensions is not None else (problem.dimension,)
    noise = problem.observation_noise_std
    return BenchmarkRegistryEntry(
        benchmark_id=benchmark_id,
        dimension=problem.dimension,
        supported_dimensions=dims,
        factory=factory,
        reference_factory=reference_factory,
        implementation_reference=implementation_reference,
        mathematical_definition=mathematical_definition,
        variable_names=tuple(problem.variable_names),
        variable_types=tuple(problem.variable_types),
        bounds=tuple(zip(problem.lower_bounds, problem.upper_bounds)),
        constraints=constraints,
        objective_direction="maximize" if problem.maximize else "minimize",
        noise_assumption=(
            "deterministic synthetic objective (GP likelihood noise still inferred, M5 convention)"
            if noise in (None, 0.0)
            else f"synthetic observation noise std={noise}"
        ),
        reference_verification_method=reference_factory(problem.dimension).verification_method,
    )


def _default_registry() -> dict[str, BenchmarkRegistryEntry]:
    entries: dict[str, BenchmarkRegistryEntry] = {}

    entries["benchmark_a"] = _entry_from_problem(
        "benchmark_a", BenchmarkAQuadratic(), lambda d: BenchmarkAQuadratic(),
        lambda d: benchmark_a_reference(),
        "radar_bo.optimization.bo.benchmarks.BenchmarkAQuadratic (M5, unchanged since commit 8c825a6)",
        "f(x) = 10 - (x0 - 3.5)^2 - (x1 - 7.2)^2 on [0, 10]^2; analytic optimum f* = 10 at (3.5, 7.2)",
        "x0, x1 in [0, 10] (box constraints only)",
        supported_dimensions=(2,),
    )
    entries["benchmark_b"] = _entry_from_problem(
        "benchmark_b", BenchmarkBMixedConstrained(), lambda d: BenchmarkBMixedConstrained(),
        lambda d: benchmark_b_reference(),
        "radar_bo.optimization.bo.benchmarks.BenchmarkBMixedConstrained (M5, unchanged since commit 8c825a6)",
        "mixed discrete/continuous quadratic with a budget constraint; enumeration-exact optimum f* = 11 "
        "(analytic clamped inner optimum per discrete value)",
        "one discrete variable {0.5, 1, 2, 3, 4} + one continuous [0, 10]; budget constraint (see class docstring)",
        supported_dimensions=(2,),
    )
    entries["benchmark_c"] = _entry_from_problem(
        "benchmark_c", BenchmarkCMultimodal(), lambda d: BenchmarkCMultimodal(),
        lambda d: benchmark_c_reference(),
        "radar_bo.optimization.bo.benchmarks.BenchmarkCMultimodal (M5, unchanged since commit 8c825a6)",
        "multimodal sum-of-bumps objective on [-5, 5]^2; numerical_approximate reference f* = 4.5295 (DE + grid)",
        "x0, x1 in [-5, 5] (box constraints only)",
        supported_dimensions=(2,),
    )
    entries["separable_4d"] = _entry_from_problem(
        "separable_4d", SeparableQuadraticHD(4), lambda d: SeparableQuadraticHD(d),
        lambda d: separable_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.SeparableQuadraticHD (M6, unchanged since commit ff787bf)",
        "f(x) = sum_i [10 - (x_i - c_i)^2], c_i = round(3 cos(1.3 i), 6); analytic optimum f* = 10 d",
        "box constraints [0, 10]^d",
        supported_dimensions=(4,),
    )
    entries["separable_8d"] = _entry_from_problem(
        "separable_8d", SeparableQuadraticHD(8), lambda d: SeparableQuadraticHD(d),
        lambda d: separable_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.SeparableQuadraticHD (M6)",
        "f(x) = sum_i [10 - (x_i - c_i)^2]; analytic optimum f* = 80",
        "box constraints [0, 10]^d",
        supported_dimensions=(8,),
    )
    entries["separable_16d"] = _entry_from_problem(
        "separable_16d", SeparableQuadraticHD(16), lambda d: SeparableQuadraticHD(d),
        lambda d: separable_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.SeparableQuadraticHD (M6)",
        "f(x) = sum_i [10 - (x_i - c_i)^2]; analytic optimum f* = 160",
        "box constraints [0, 10]^d",
        supported_dimensions=(16,),
    )
    entries["coupled_4d"] = _entry_from_problem(
        "coupled_4d", CoupledQuadraticHD(4), lambda d: CoupledQuadraticHD(d),
        lambda d: coupled_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.CoupledQuadraticHD (M6)",
        "f(x) = 20 - (x - c)^T M (x - c) with fixed PD band matrix M = LL^T + 0.5 I; analytic optimum f* = 20",
        "box constraints [-5, 5]^d",
        supported_dimensions=(4,),
    )
    entries["coupled_8d"] = _entry_from_problem(
        "coupled_8d", CoupledQuadraticHD(8), lambda d: CoupledQuadraticHD(d),
        lambda d: coupled_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.CoupledQuadraticHD (M6)",
        "f(x) = 20 - (x - c)^T M (x - c), fixed PD band matrix; analytic optimum f* = 20",
        "box constraints [-5, 5]^d",
        supported_dimensions=(8,),
    )
    entries["constrained_mixed_8d"] = _entry_from_problem(
        "constrained_mixed_8d", ConstrainedMixedHD(8), lambda d: ConstrainedMixedHD(d),
        lambda d: constrained_mixed_hd_reference(d),
        "radar_bo.optimization.bo.benchmarks_highdim.ConstrainedMixedHD (M6)",
        "2 discrete {0.5, 1.0, 1.5, 2.0} + 6 continuous [0, 4]; f = 2(d0+d1) + sum [6 - 0.5 (x_i - m_i)^2]; "
        "sum(x) + 3(d0+d1) <= 14; reference numerical_approximate (enumeration + KKT bisection) f* = 39.53",
        "budget constraint sum(x) + 3(d0 + d1) <= 14 plus box constraints",
        supported_dimensions=(8,),
    )
    entries["block_stagnation_4d"] = _entry_from_problem(
        "block_stagnation_4d", BlockStagnationLP(), lambda d: BlockStagnationLP(),
        lambda d: stagnation_lp_reference(),
        "radar_bo.optimization.bo.benchmarks_highdim.BlockStagnationLP (M6)",
        "LP f = 3a1 + b1 + 3a2 + b2 on [0, 1.6]^4 with a1+b1+a2+b2 <= 3.2; analytic optimum f* = 9.6 at "
        "(1.6, 0, 1.6, 0); from budget-tight configurations the alternating a/b grouping provably stagnates",
        "budget constraint a1 + b1 + a2 + b2 <= 3.2 plus box constraints",
        supported_dimensions=(4,),
    )
    return entries


class BenchmarkRegistry:
    """Registry with a fixed default content and optional additions (M7 item 6)."""

    def __init__(self) -> None:
        self._entries: dict[str, BenchmarkRegistryEntry] = _default_registry()

    def register(self, entry: BenchmarkRegistryEntry) -> None:
        if entry.benchmark_id in self._entries:
            raise BenchmarkRegistryError(f"Benchmark id already registered: {entry.benchmark_id}")
        self._entries[entry.benchmark_id] = entry

    def get(self, benchmark_id: str) -> BenchmarkRegistryEntry:
        if benchmark_id not in self._entries:
            raise BenchmarkRegistryError(
                f"Unknown benchmark id '{benchmark_id}'; registered: {sorted(self._entries)}"
            )
        return self._entries[benchmark_id]

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def entries(self) -> dict[str, BenchmarkRegistryEntry]:
        return dict(self._entries)
