"""M8A-R1 targeted validation experiment (Phase E).

SMALL synthetic experiment validating the R1 corrections end-to-end (NOT a
repeat of the M5/M6 matrices):

1. explicit_shared mode: ONE authoritative design per (benchmark, dimension,
   seed), persisted under initial_designs/ with a genuine SHA-256 fingerprint;
2. the same design reaches joint_bo (initial phase), random_search_shared_start
   (shared-start prefix) and is explicitly ignored by random_search;
3. legitimate early stopping (patience) is COMPLETED (early), never failed;
4. resume adopts an interrupted (record-present, status-not-updated) run
   without re-execution;
5. shared-initialization verification passes against the persisted design,
   including recomputed objectives on deterministic problems;
6. design reload-on-resume: the second execute() reuses the stored designs.

Output: results/m8a_r1_validation/summary.json (gitignored, regenerable).
No radar claims; generic synthetic benchmark only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.checkpointing import CheckpointStore, compatibility_metadata
from radar_bo.experiments.framework.experiment_config import (
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.initial_design_store import InitialDesignStore, design_fingerprint
from radar_bo.experiments.framework.matrix import RunStatus, generate_manifest
from radar_bo.experiments.framework.runner import ExperimentRunner


def build_config(output_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="m8a_r1_validation",
        description="M8A-R1 targeted validation of shared-design, early-stop and resume corrections",
        result_label="generic synthetic correction-validation experiment; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "random_search_shared_start", "joint_bo", "reference"],
        seeds=[0, 1],
        dimensions=[],
        initial_design=InitialDesignSettings(size=4, mode="explicit_shared"),
        stopping=StoppingSettings(evaluation_budget=12, patience=3),
        output_dir=output_dir,
    )


def main() -> int:
    out = Path("results/m8a_r1_validation")
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"experiment": "m8a_r1_validation", "checks": {}}

    config = build_config(str(out / "experiment"))
    runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
    summary = runner.execute()
    checks = report["checks"]

    # 1-2-5: designs persisted with genuine sha256, verification green
    store = InitialDesignStore(str(out / "experiment"))
    designs_ok = True
    for group in [( "benchmark_a", 2, 0), ("benchmark_a", 2, 1)]:
        rows, fp = store.load(group, BenchmarkRegistry().get("benchmark_a").problem(2), 4)
        if fp != design_fingerprint(rows):
            designs_ok = False
    checks["designs_persisted_with_genuine_sha256"] = designs_ok
    fairness = summary["shared_initialization_verification"]
    checks["verification_green"] = fairness["verified"]
    checks["verification_groups"] = [
        {k: g[k] for k in ("benchmark_id", "dimension", "seed", "methods", "authoritative_match",
                           "authoritative_fingerprint", "authoritative_objectives_match")}
        for g in fairness["groups"]
    ]
    checks["excluded_no_initial_phase"] = fairness["excluded_no_initial_design_phase"]

    # 3: early stopping completed, not failed
    statuses = {r["run_id"]: r for r in CheckpointStore(str(out / "experiment")).specs()}
    early = [
        {"algorithm_id": r["algorithm_id"], "seed": r["seed"], "status": r["status"]}
        for rid, r in statuses.items() if r["algorithm_id"] in ("joint_bo", "random_search_shared_start")
    ]
    checks["early_stop_statuses"] = early
    record_probe = runner._collect_records()
    early_records = [
        {"algorithm_id": r["algorithm_id"], "seed": r["seed"],
         "early_completion": r.get("early_completion"), "stopping_reason": r.get("stopping_reason"),
         "attempted": (r["evaluation_counts"].get("n_initial_evaluations", 0)
                       + r["evaluation_counts"].get("n_bo_evaluations", r["evaluation_counts"].get("n_objective_evaluations", 0)))}
        for r in record_probe if r["algorithm_id"] in ("joint_bo", "random_search_shared_start")
    ]
    checks["early_records"] = early_records

    # 4 + 6: resume adoption + design reload (no re-execution, no regeneration)
    joint_ids = [r["run_id"] for r in statuses.values() if r["algorithm_id"] == "joint_bo"]
    store2 = CheckpointStore(str(out / "experiment"))
    store2.set_status(joint_ids[0], RunStatus.RUNNING, reason="interrupted: R1 validation simulation")
    design_files_before = sorted(p.name for p in (out / "experiment" / "initial_designs").iterdir())
    resume = ExperimentRunner(build_config(str(out / "experiment")), BenchmarkRegistry(), AlgorithmRegistry())
    resume.execute()
    statuses2 = {r["run_id"]: r for r in store2.specs()}
    adopted = statuses2[joint_ids[0]]
    checks["resume_adoption"] = {
        "status": adopted["status"],
        "history_tail": adopted["history"][-2:],
    }
    design_files_after = sorted(p.name for p in (out / "experiment" / "initial_designs").iterdir())
    checks["design_files_stable_on_resume"] = design_files_before == design_files_after
    checks["all_completed_after_resume"] = all(r["status"] == "completed" for r in statuses2.values())

    checks["status_counts"] = summary["status_counts"]
    (out / "summary.json").write_text(json.dumps(report, indent=1))

    ok = (checks["designs_persisted_with_genuine_sha256"] and checks["verification_green"]
          and checks["resume_adoption"]["status"] == "completed"
          and checks["design_files_stable_on_resume"] and checks["all_completed_after_resume"])
    print(json.dumps({k: v for k, v in checks.items() if k != "verification_groups"}, indent=1)[:2000])
    print("M8A-R1 VALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
