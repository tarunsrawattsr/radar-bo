"""M7 item 31: integration tests — the complete experimental workflow.

config -> manifest -> registered problem/method -> execution -> persisted
result -> archive validation -> statistical analysis -> report generation,
plus the interrupted-experiment scenario, resume-without-repetition, and
corrupted-record rejection.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
from radar_bo.experiments.framework.archive_integrity import validate_experiment_store
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.checkpointing import CheckpointStore, RecordRejected
from radar_bo.experiments.framework.experiment_config import (
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.matrix import RunStatus
from radar_bo.experiments.framework.reporting import generate_report
from radar_bo.experiments.framework.runner import ExperimentRunner


def make_config(output_dir: str, **overrides) -> ExperimentConfig:
    base = dict(
        experiment_id="m7_integration_test",
        description="integration test of the M7 experimental workflow",
        result_label="generic synthetic integration test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo", "reference"],
        seeds=[0, 1],
        initial_design=InitialDesignSettings(size=3),
        stopping=StoppingSettings(evaluation_budget=7),
        output_dir=output_dir,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


class CountingBenchmarkRegistry(BenchmarkRegistry):
    """Registry whose benchmark_a problems count objective evaluations."""

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


class TestFullWorkflow:
    def test_config_to_report(self, tmp_path):
        config = make_config(str(tmp_path / "experiment"))
        benchmarks = CountingBenchmarkRegistry()
        algorithms = AlgorithmRegistry()
        runner = ExperimentRunner(config, benchmarks, algorithms)
        summary = runner.execute()

        # execution + persistence
        assert summary["status_counts"].get("completed") == 5  # 2 algorithms x 2 seeds + 1 reference
        store = CheckpointStore(config.output_dir)
        statuses = {r["run_id"]: r["status"] for r in store.specs()}
        assert set(statuses.values()) <= {"completed", "skipped"}
        run_files = list((Path(config.output_dir) / "runs").glob("*.json"))
        assert len(run_files) == summary["status_counts"]["completed"]

        # shared-initialization verification from STORED records (item 10)
        fairness = summary["shared_initialization_verification"]
        assert fairness["verified"] is True
        assert fairness["groups"], "joint_bo groups must be verified"
        for group in fairness["groups"]:
            assert group["n_initial"] == 3

        # archive validation of the whole store (item 14)
        result = validate_experiment_store(
            Path(config.output_dir),
            feasibility_checker=BenchmarkRegistry().get("benchmark_a").problem().is_feasible,
            expected_reference_by_benchmark={"benchmark_a": 10.0},
        )
        assert result["valid"] is True, result["issues"]

        # analysis present (item 19-22 wiring)
        analysis = summary["analysis"]["benchmark_a"]
        assert "joint_bo" in analysis["per_method"]
        assert "random_search" in analysis["per_method"]
        js = analysis["per_method"]["joint_bo"]
        assert js["final_objective"]["n"] == 2
        assert js["threshold_analysis"], "threshold metrics must be computed"
        assert "0.9" in {str(t["fraction"]) for t in js["threshold_analysis"]}

        # report generated from stored data only
        report = generate_report(Path(config.output_dir))
        text = report.read_text(encoding="utf-8")
        assert "m7_integration_test" in text
        assert "shared initialization" in text.lower() or "verified" in text.lower()

        # objective-call accounting: budget x seeds x 2 search algorithms (exact)
        expected_calls = 7 * 2 * 2  # budget x seeds x {random_search, joint_bo}
        assert benchmarks.evaluate_calls == expected_calls

    def test_evaluations_per_run_match_budget(self, tmp_path):
        config = make_config(str(tmp_path / "experiment2"), seeds=[0])
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(config.output_dir)
        for spec in store.specs():
            if spec["status"] != "completed" or spec["algorithm_id"] == "reference":
                continue
            record = store.load_run_record(spec["run_id"])
            if spec["algorithm_id"] == "random_search":
                assert record["evaluation_counts"]["n_objective_evaluations"] == 7
            else:
                attempted = (record["evaluation_counts"]["n_initial_evaluations"]
                             + record["evaluation_counts"]["n_bo_evaluations"])
                assert attempted == 7
                assert len(record["observations"]) + len(record["failures"]) == 7


class TestInterruptionAndResume:
    def test_interrupted_run_detected_and_redone_once(self, tmp_path):
        """Simulate an interruption: one run marked 'running' without a record.
        Resuming completes exactly that run; nothing else re-executes."""
        config = make_config(str(tmp_path / "experiment3"), seeds=[0])
        counting = CountingBenchmarkRegistry()
        algorithms = AlgorithmRegistry()

        runner = ExperimentRunner(config, counting, algorithms)
        summary1 = runner.execute()
        calls_full = counting.evaluate_calls

        # simulate an interrupted experiment: fresh directory, manifest with a
        # 'running' entry and no record for it, others completed
        fresh = tmp_path / "experiment3_interrupted"
        config2 = make_config(str(fresh), seeds=[0])
        store = CheckpointStore(str(fresh))
        from radar_bo.experiments.framework.matrix import generate_manifest
        from radar_bo.experiments.framework.checkpointing import compatibility_metadata

        specs = generate_manifest(config2, BenchmarkRegistry(), AlgorithmRegistry())
        store.initialize(config2, specs, compatibility_metadata(config2, BenchmarkRegistry(), AlgorithmRegistry()))
        joint_ids = [s.run_id for s in specs if s.algorithm_id == "joint_bo"]
        rs_ids = [s.run_id for s in specs if s.algorithm_id == "random_search"]
        # pretend the first joint run was interrupted; everything else completed
        store.set_status(joint_ids[0], RunStatus.RUNNING, reason="interrupted: simulated kill")
        spec_by_id = {s.run_id: s for s in specs}
        for run_id in rs_ids + joint_ids[1:]:
            spec = spec_by_id[run_id]
            record = {"run_id": run_id, "algorithm_id": spec.algorithm_id,
                      "benchmark_id": spec.benchmark_id, "dimension": spec.dimension,
                      "seed": spec.seed, "initial_design": [],
                      "experiment_config_fingerprint": config2.fingerprint()}
            store.save_run_record(run_id, record)
            store.set_status(run_id, RunStatus.COMPLETED)
        # NOTE: those 'completed' records lack the accounting fields, so the
        # counting registry counts only the redone joint run.

        counting2 = CountingBenchmarkRegistry()
        resume_runner = ExperimentRunner(config2, counting2, algorithms)
        resume_summary = resume_runner.execute()
        # exactly ONE interrupted joint run was redone: budget 7 evaluations
        assert counting2.evaluate_calls == 7, (
            f"resume must redo ONLY the interrupted run; got {counting2.evaluate_calls} calls"
        )
        store2 = CheckpointStore(str(fresh))
        statuses = {r["run_id"]: r["status"] for r in store2.specs()}
        assert all(s == "completed" for s in statuses.values())

    def test_resume_does_not_repeat_completed_runs(self, tmp_path):
        config = make_config(str(tmp_path / "experiment4"), seeds=[0])
        counting = CountingBenchmarkRegistry()
        runner = ExperimentRunner(config, counting, AlgorithmRegistry())
        runner.execute()
        calls_before = counting.evaluate_calls
        assert calls_before > 0

        counting2 = CountingBenchmarkRegistry()
        runner2 = ExperimentRunner(config, counting2, AlgorithmRegistry())
        runner2.execute()
        assert counting2.evaluate_calls == 0, "resume must not repeat completed runs"

    def test_corrupted_record_rejected_explicitly(self, tmp_path):
        config = make_config(str(tmp_path / "experiment5"), seeds=[0])
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(config.output_dir)
        completed = [s["run_id"] for s in store.specs() if s["status"] == "completed"]
        target = next(r for r in completed
                      if store.load_run_record(r).get("algorithm_id") == "joint_bo")
        store.run_path(target).write_text("{corrupted", encoding="utf-8")

        runner2 = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        with pytest.raises(RecordRejected, match="Corrupted JSON"):
            runner2.execute()

    def test_incompatible_stored_record_rejected_on_resume(self, tmp_path):
        config = make_config(str(tmp_path / "experiment6"), seeds=[0])
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(config.output_dir)
        completed = [s["run_id"] for s in store.specs() if s["status"] == "completed"]
        target = next(r for r in completed
                      if store.load_run_record(r).get("algorithm_id") == "joint_bo")
        record = store.load_run_record(target)
        record["experiment_config_fingerprint"] = "different_experiment"
        store.save_run_record(target, record)

        runner2 = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        with pytest.raises(RecordRejected, match="different experiment configuration"):
            runner2.execute()


class TestOmittedRunsReportedHonestly:
    def test_skipped_combinations_listed_never_completed(self, tmp_path):
        config = make_config(
            str(tmp_path / "experiment7"),
            algorithm_ids=["alternating_bo"],  # benchmark_a has no recommended blocks
        )
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        summary = runner.execute()
        assert summary["status_counts"].get("skipped") == 1
        assert summary["status_counts"].get("completed", 0) == 0
        store = CheckpointStore(config.output_dir)
        skipped = [r for r in store.specs() if r["status"] == "skipped"]
        assert skipped and "incompatible_combination" in skipped[0]["reason"]
        assert not (Path(config.output_dir) / "runs").exists() or \
            not list((Path(config.output_dir) / "runs").glob("*.json"))
