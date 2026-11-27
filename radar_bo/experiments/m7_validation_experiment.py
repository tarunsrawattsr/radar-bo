"""M7 limited validation experiment (item 26).

PURPOSE (reported BEFORE execution): validate the NEW experimental
infrastructure — experiment configuration, manifest generation, registered
benchmarks/algorithms, checkpointed runner with resume, shared-initialization
verification, metrics, report and figure generation — on a SMALL synthetic
matrix. These runs validate INFRASTRUCTURE, not algorithm superiority.

Validation matrix (explicit):
- benchmarks: benchmark_a (low-dimensional 2-D), constrained_mixed_8d (mixed-variable, constrained);
- algorithms: random_search, joint_bo, alternating_bo, pairwise_bo, reference;
- seeds: 0, 1, 2;
- budget: 40 total attempted evaluations (initial design 10) — deliberately small;
- settings otherwise inherited from M5/M6 conventions (pool 500, Matérn-5/2, LogEI).

Additionally demonstrates RESUME: after completion the runner is invoked again
and must not repeat any completed run (verified via objective-call counters).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.experiment_config import (
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.figures import generate_all_figures
from radar_bo.experiments.framework.reporting import generate_report
from radar_bo.experiments.framework.runner import ExperimentRunner

OUTPUT_DIR = "results/m7_validation_experiment"


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="m7_infrastructure_validation",
        description=(
            "Limited M7 validation matrix: validates the new experiment framework "
            "(runner, checkpoint/resume, metrics, reporting) on synthetic benchmarks; "
            "NOT an algorithm-comparison experiment"
        ),
        result_label=(
            "generic synthetic M7 infrastructure-validation experiment; no radar claims; "
            "descriptive statistics only; no superiority claims"
        ),
        benchmark_ids=["benchmark_a", "constrained_mixed_8d"],
        algorithm_ids=["random_search", "joint_bo", "alternating_bo", "pairwise_bo", "reference"],
        seeds=[0, 1, 2],
        initial_design=InitialDesignSettings(size=10),
        stopping=StoppingSettings(evaluation_budget=40),
        output_dir=OUTPUT_DIR,
    )


def main(argv: list[str]) -> int:
    config = build_config()
    benchmarks = BenchmarkRegistry()
    algorithms = AlgorithmRegistry()

    # Item 26: report the matrix and its purpose BEFORE execution.
    manifest_preview = json.dumps({
        "purpose": "infrastructure validation only (not algorithm superiority)",
        "benchmarks": config.benchmark_ids,
        "algorithms": config.algorithms if hasattr(config, "algorithms") else config.algorithm_ids,
        "seeds": config.seeds,
        "budget": config.stopping.evaluation_budget,
        "initial_design": config.initial_design.size,
        "output_dir": config.output_dir,
    }, indent=1)
    print("=== M7 validation matrix ===")
    print(manifest_preview, flush=True)

    started = time.perf_counter()
    runner = ExperimentRunner(config, benchmarks, algorithms)
    summary = runner.execute()
    wall = time.perf_counter() - started
    print(f"=== execution wall: {wall:.0f}s | status counts: {summary['status_counts']} ===")
    print(f"shared-initialization verification: verified={summary['shared_initialization_verification']['verified']}")
    if not summary["shared_initialization_verification"]["verified"]:
        for issue in summary["shared_initialization_verification"]["issues"]:
            print("  ISSUE:", issue)
        return 1

    # Figures + report from STORED data only (item 27/28).
    records_path = Path(config.output_dir)
    from radar_bo.experiments.framework.figures import load_records_from_store

    records = load_records_from_store(records_path)
    generate_all_figures(records, Path(config.output_dir) / "figures",
                         fractions=config.statistics.threshold_closures)
    report = generate_report(records_path)
    print(f"figures + report written: {report}")

    # RESUME demonstration: invoking the runner again must not repeat runs.
    # A counting registry wraps benchmark_a's problem factory so every objective
    # evaluation increments a counter; a true resume adds ZERO calls.
    import dataclasses

    class CountingBenchmarkRegistry(BenchmarkRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.evaluate_calls = 0
            entry = self._entries["benchmark_a"]
            real_factory = entry.factory
            registry_ref = self

            def counting_factory(d):
                problem = real_factory(d)
                original_evaluate = problem.evaluate

                def counting_evaluate(x):
                    registry_ref.evaluate_calls += 1
                    return original_evaluate(x)

                problem.evaluate = counting_evaluate
                return problem

            self._entries["benchmark_a"] = dataclasses.replace(entry, factory=counting_factory)

    counting = CountingBenchmarkRegistry()
    resume_runner = ExperimentRunner(config, counting, algorithms)
    resume_summary = resume_runner.execute()
    calls = counting.evaluate_calls
    print(f"=== RESUME CHECK: objective evaluations during resume = {calls} (expected 0) ===")
    if calls != 0:
        print("FAIL: resume repeated completed runs")
        return 1
    print("M7 validation experiment COMPLETE (infrastructure validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
