"""M7 targeted M6 review, item 3.1: runtime-attribution profiling evidence.

The M5 report claimed "GP fitting dominates runtime". The archived M5
measurements contradict that claim: GP fitting occupied ~12-14 s of a
57-61 s run (~22 %); the remaining ~44 s per run was UNINSTRUMENTED at the
time. This script produces supporting profiling evidence on the CURRENT
instrumented engine (M6 refactor, joint mode — one engine, M6 instruction 4):

* stage timers recorded by the engine: GP fit, acquisition scoring, objective
  evaluation, candidate generation, feasibility validation;
* total wall-clock;
* unattributed overhead = wall - sum(components) (never assumed zero);
* a cProfile pass (one run) listing the top cumulative-time functions as
  descriptive supporting evidence — WITHOUT claiming an exact attribution.

This is a NEW analysis artifact (research_results/m7_m6_review_profiling.json).
Historical M5/M6 values are NOT modified anywhere.

Label: generic synthetic benchmarks (M5 A/B/C), no radar claims.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import statistics
import subprocess
import time

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
)
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.reference_solvers import (
    benchmark_a_reference,
    benchmark_b_reference,
    benchmark_c_reference,
)

BUDGET = 100
INITIAL = 30
SEEDS = [0, 1, 2]

PROBLEMS = [
    ("benchmark_a", BenchmarkAQuadratic, benchmark_a_reference),
    ("benchmark_b", BenchmarkBMixedConstrained, benchmark_b_reference),
    ("benchmark_c", BenchmarkCMultimodal, benchmark_c_reference),
]


def run_one(name: str, cls, reference_factory, seed: int, profile: bool = False) -> dict:
    problem = cls()
    config = OptimizationConfig(evaluation_budget=BUDGET, num_initial_observations=INITIAL)
    reference = reference_factory()
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    started = time.perf_counter()
    result = BayesianOptimizer(
        problem, config,
        reference_objective=reference.objective_value,
        reference_method=reference.verification_method,
        seed=seed,
    ).optimize()
    wall = time.perf_counter() - started
    if profile:
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(28)
        profile_text = stream.getvalue()
    else:
        profile_text = None
    components = {
        "gp_fit_runtime_s": result.gp_fit_runtime_s,
        "acquisition_runtime_s": result.acquisition_runtime_s,
        "objective_runtime_s": result.objective_runtime_s,
        "candidate_generation_runtime_s": result.candidate_generation_runtime_s,
        "constraint_validation_runtime_s": result.constraint_validation_runtime_s,
    }
    component_sum = sum(components.values())
    return {
        "benchmark": name,
        "problem_id": problem.problem_id,
        "seed": seed,
        "budget": BUDGET,
        "initial_design": INITIAL,
        "wall_runtime_s": wall,
        **components,
        "component_sum_s": component_sum,
        "unattributed_overhead_s": wall - component_sum,
        "unattributed_fraction": (wall - component_sum) / wall if wall > 0 else None,
        "gp_fit_fraction": components["gp_fit_runtime_s"] / wall if wall > 0 else None,
        "stopping_reason": result.stopping_reason,
        "best_objective": result.best_objective,
    }, profile_text


def main() -> int:
    runs = []
    profile_text = None
    for name, cls, ref_factory in PROBLEMS:
        for seed in SEEDS:
            rec, prof = run_one(name, cls, ref_factory, seed, profile=(seed == 0))
            if prof is not None:
                profile_text = prof
            print(
                f"[{name} seed={seed}] wall={rec['wall_runtime_s']:.1f}s gp={rec['gp_fit_runtime_s']:.1f} "
                f"acq={rec['acquisition_runtime_s']:.2f} obj={rec['objective_runtime_s'] * 1000:.0f}ms "
                f"gen={rec['candidate_generation_runtime_s']:.1f} feas={rec['constraint_validation_runtime_s']:.1f} "
                f"unattributed={rec['unattributed_overhead_s']:.1f}s ({rec['unattributed_fraction'] * 100:.0f}%)",
                flush=True,
            )
            runs.append(rec)

    summary = {}
    for name, _, _ in PROBLEMS:
        rows = [r for r in runs if r["benchmark"] == name]
        summary[name] = {
            "n_runs": len(rows),
            "wall_mean_s": statistics.fmean(r["wall_runtime_s"] for r in rows),
            "gp_fit_mean_s": statistics.fmean(r["gp_fit_runtime_s"] for r in rows),
            "acquisition_mean_s": statistics.fmean(r["acquisition_runtime_s"] for r in rows),
            "objective_mean_s": statistics.fmean(r["objective_runtime_s"] for r in rows),
            "candidate_generation_mean_s": statistics.fmean(r["candidate_generation_runtime_s"] for r in rows),
            "constraint_validation_mean_s": statistics.fmean(r["constraint_validation_runtime_s"] for r in rows),
            "unattributed_mean_s": statistics.fmean(r["unattributed_overhead_s"] for r in rows),
            "gp_fit_fraction_mean": statistics.fmean(r["gp_fit_fraction"] for r in rows),
            "unattributed_fraction_mean": statistics.fmean(r["unattributed_fraction"] for r in rows),
        }

    artifact = {
        "result_label": (
            "M7 targeted M6 review (item 3.1): runtime-attribution profiling on the "
            "current instrumented engine, joint mode, generic synthetic M5 benchmarks "
            "A/B/C; supporting evidence only, no radar claims"
        ),
        "purpose": (
            "Correct the M5 report claim that GP fitting dominates runtime; quantify "
            "the previously uninstrumented overhead with per-stage timers and a "
            "cProfile pass; no historical values are modified"
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "budget": BUDGET,
        "initial_design": INITIAL,
        "seeds": SEEDS,
        "summary_per_benchmark": summary,
        "runs": runs,
        "cProfile_top_functions_benchmark_a_seed0": profile_text,
        "notes": [
            "Timer boundaries: gp_fit = GP construction+fitting; acquisition = LogEI "
            "evaluation over the scored pool; objective = problem.objective calls; "
            "candidate_generation = pool sampling; constraint_validation = feasibility "
            "checks on complete candidates. Unattributed = wall - sum(components); it "
            "includes torch tensor conversion, dataset insertion, incumbent tracking, "
            "GP training-tensor preparation, diagnostics construction, loop overhead "
            "and serialization into the result model — these are NOT separately "
            "timed, so no exact attribution is claimed.",
            "cProfile output is descriptive supporting evidence (cumulative-time top "
            "functions under profiler overhead), not an exact decomposition.",
        ],
    }
    with open("research_results/m7_m6_review_profiling.json", "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1)
    print("\nArtifact written: research_results/m7_m6_review_profiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
