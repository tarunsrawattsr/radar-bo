"""M7 sensitivity-analysis pilot (item 25).

Explicit small scenario list (NO unrestricted Cartesian product): one base
configuration plus three single-factor scenarios on one low-dimensional
benchmark, two seeds, small budget. Purpose: validate the sensitivity
INTERFACE (apply -> diff -> run -> compare), not to draw algorithm conclusions.

Scenarios (everything else fixed; every changed field recorded):
- base:        pool 500, initial design 8, budget 25
- pool_100:    candidate pool 100 (factor: candidate-pool size)
- initial_4:   initial design 4 (factor: initial-design size)
- budget_15:   budget 15 (factor: total evaluation budget)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.experiment_config import (
    AcquisitionSettings,
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.figures import fig_sensitivity, load_records_from_store
from radar_bo.experiments.framework.runner import ExperimentRunner
from radar_bo.experiments.framework.sensitivity import (
    SensitivityScenario,
    build_sensitivity_experiment,
)


def base_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="m7_sensitivity_base",
        description="M7 sensitivity pilot base configuration (benchmark_a, joint_bo + random_search)",
        result_label="generic synthetic M7 sensitivity pilot; no radar claims; infrastructure validation",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo"],
        seeds=[0, 1],
        initial_design=InitialDesignSettings(size=8),
        stopping=StoppingSettings(evaluation_budget=25),
        output_dir="results/m7_sensitivity_base",
    )


SCENARIOS = [
    SensitivityScenario(
        scenario_id="pool_100",
        description="candidate pool reduced to 100 (factor: candidate-pool size)",
        overrides={"candidate_pool.num_acquisition_candidates": 100},
    ),
    SensitivityScenario(
        scenario_id="initial_4",
        description="initial design reduced to 4 (factor: initial-design size)",
        overrides={"initial_design.size": 4},
    ),
    SensitivityScenario(
        scenario_id="budget_15",
        description="total budget reduced to 15 (factor: total evaluation budget)",
        overrides={"stopping.evaluation_budget": 15},
    ),
]


def summarize_scenario(config: ExperimentConfig) -> dict:
    records = load_records_from_store(config.output_dir)
    bo_records = [r for r in records if r["algorithm_id"] == "joint_bo"]
    finals = [r["final_objective"] for r in bo_records if r.get("final_objective") is not None]
    import numpy as np

    return {
        "scenario_id": config.experiment_id.split("-", 1)[1] if "-" in config.experiment_id else config.experiment_id,
        "n_bo_runs": len(bo_records),
        "final_objective_mean": float(np.mean(finals)) if finals else None,
        "final_objective_stdev": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "wall_mean_s": float(np.mean([r["runtime_measurements"]["wall_runtime_s"] for r in bo_records])) if bo_records else None,
    }


def main() -> int:
    base = base_config()
    matrix = build_sensitivity_experiment(base, SCENARIOS)
    print("=== M7 sensitivity pilot: explicit scenario list (no Cartesian product) ===")
    for scenario, config, changed in matrix:
        print(f"  scenario {scenario.scenario_id}: changes {list(changed.keys())} | everything else FIXED", flush=True)
    summaries = []
    for scenario, config, changed in [(("base", "base configuration"), base, {})] + [
        (s, c, d) for s, c, d in matrix
    ]:
        started = time.perf_counter()
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        summary = summarize_scenario(config)
        summary["changed_fields"] = list(changed.keys())
        summary["wall_total_s"] = time.perf_counter() - started
        summaries.append(summary)
        print(f"  [{summary['scenario_id']}] joint_bo final mean: {summary['final_objective_mean']}", flush=True)

    artifact = {
        "result_label": "generic synthetic M7 sensitivity pilot; infrastructure validation; no radar claims",
        "purpose": "validate the sensitivity-analysis interface (apply -> diff -> run -> compare)",
        "factor_space": "explicit scenario list; every changed field recorded; no unrestricted sweep",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scenarios": summaries,
    }
    out = Path("research_results/m7_sensitivity_pilot.json")
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    fig_path = Path("results/m7_sensitivity_base/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_sensitivity(summaries, fig_path)
    print(f"Artifact: {out} | figure: {fig_path / 'sensitivity_results.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
