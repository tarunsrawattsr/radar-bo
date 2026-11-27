"""M7 item 30: reporting, sensitivity, and simulator-interface adapter tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar_bo.experiments.framework.checkpointing import atomic_write_json
from radar_bo.experiments.framework.reporting import generate_report
from radar_bo.experiments.framework.sensitivity import (
    SensitivityScenario,
    apply_scenario,
    build_sensitivity_experiment,
    scenario_diff,
)
from radar_bo.experiments.framework.simulator_adapter import (
    MockEvaluationRecord,
    MockFrozenSnapshot,
    MockSimulatedEvaluator,
    SimulatorBackedProblem,
    evaluate_through_adapter,
)


def make_config(**overrides):
    from radar_bo.experiments.framework.experiment_config import ExperimentConfig, StoppingSettings

    base = dict(
        experiment_id="report_test",
        description="report unit test",
        result_label="generic synthetic unit test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo", "reference"],
        seeds=[0, 1],
        stopping=StoppingSettings(evaluation_budget=8),
        output_dir="results/report_test",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def build_store(tmp_path) -> Path:
    """A minimal persisted experiment store with two completed runs."""
    from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
    from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
    from radar_bo.experiments.framework.matrix import RunStatus, generate_manifest
    from radar_bo.experiments.framework.checkpointing import CheckpointStore, compatibility_metadata

    config = make_config(output_dir=str(tmp_path / "experiment"))
    specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
    compatibility = compatibility_metadata(config, BenchmarkRegistry(), AlgorithmRegistry())
    store = CheckpointStore(config.output_dir)
    store.initialize(config, specs, compatibility)
    ref = BenchmarkRegistry().get("benchmark_a").reference()
    joint = next(s for s in specs if s.algorithm_id == "joint_bo")
    rs = next(s for s in specs if s.algorithm_id == "random_search")
    ref_spec = next(s for s in specs if s.algorithm_id == "reference")
    joint_record = {
        "run_id": joint.run_id, "experiment_id": config.experiment_id,
        "experiment_config_fingerprint": config.fingerprint(),
        "benchmark_id": "benchmark_a", "dimension": 2, "algorithm_id": "joint_bo",
        "architecture": "joint", "seed": 0, "algorithm_version": "test",
        "source_commit": compatibility["source_commit"],
        "dependency_versions": compatibility["dependency_versions"],
        "initial_design": [{"config": [1.0, 2.0], "objective": 5.0, "evaluation_id": "e0"}],
        "observations": [
            {"config": [1.0, 2.0], "objective": 5.0, "evaluation_id": "e0", "metadata": {"stage": "initial_design"}},
            {"config": [3.0, 2.0], "objective": 9.0, "evaluation_id": "e1", "metadata": {"stage": "bo_iteration"}},
        ],
        "failures": [],
        "incumbent_history": [5.0, 9.0],
        "final_objective": 9.0, "final_incumbent_config": [3.0, 2.0],
        "reference_objective": ref.objective_value, "reference_method": ref.verification_method,
        "reference_is_exact": ref.is_exact,
        "evaluation_counts": {"n_initial_evaluations": 1, "n_bo_evaluations": 1, "n_failed_attempts": 0},
        "block_cycle_diagnostics": {"search_strategy": "joint"},
        "iteration_diagnostics": [],
        "runtime_measurements": {"wall_runtime_s": 5.0, "gp_fit_runtime_s": 4.0, "acquisition_runtime_s": 0.5,
                                 "objective_runtime_s": 0.001, "candidate_generation_runtime_s": 0.4,
                                 "constraint_validation_runtime_s": 0.05},
        "stopping_reason": "evaluation_budget_exhausted",
    }
    rs_record = dict(joint_record)
    rs_record.update({
        "run_id": rs.run_id, "algorithm_id": "random_search", "architecture": "random_search",
        "initial_design": [], "incumbent_history": [5.0], "final_objective": 5.0,
        "final_incumbent_config": [1.0, 2.0], "stopping_reason": "evaluation_budget_exhausted",
        "evaluation_counts": {"n_objective_evaluations": 1, "n_failed_attempts": 0},
        "block_cycle_diagnostics": {"search_strategy": "random_search"},
    })
    ref_record = {
        "run_id": ref_spec.run_id, "experiment_id": config.experiment_id,
        "benchmark_id": "benchmark_a", "dimension": 2, "algorithm_id": "reference",
        "architecture": "reference", "seed": None, "algorithm_version": "test",
        "source_commit": compatibility["source_commit"],
        "dependency_versions": compatibility["dependency_versions"],
        "reference_objective": ref.objective_value, "reference_method": ref.verification_method,
        "reference_is_exact": ref.is_exact, "reference_config": [3.5, 7.2],
        "stopping_reason": "not_applicable",
        "runtime_measurements": {"wall_runtime_s": 0.0},
    }
    store.save_run_record(joint.run_id, joint_record)
    store.set_status(joint.run_id, RunStatus.COMPLETED)
    store.save_run_record(rs.run_id, rs_record)
    store.set_status(rs.run_id, RunStatus.COMPLETED)
    store.save_run_record(ref_spec.run_id, ref_record)
    store.set_status(ref_spec.run_id, RunStatus.COMPLETED)
    # summary (what the runner writes before reporting)
    atomic_write_json(Path(config.output_dir) / "summary.json", {
        "experiment_id": config.experiment_id,
        "description": config.description,
        "result_label": config.result_label,
        "configuration": config.model_dump(),
        "status_counts": {"completed": 3},
        "shared_initialization_verification": {"verified": True, "issues": [], "groups": []},
        "analysis": {},
    })
    return Path(config.output_dir)


class TestReportGeneration:
    def test_report_generated_from_stored_data_only(self, tmp_path):
        store = build_store(tmp_path)
        report = generate_report(store)
        text = report.read_text(encoding="utf-8")
        assert "# Experiment report: report_test" in text
        assert "generic synthetic" in text
        assert "evaluation_budget" in text or "budget" in text
        assert "Reference" in text or "reference" in text
        assert "unattributed" in text
        assert "Limitations" in text
        assert "research_results/" in text  # evidence locations

    def test_report_regeneration_does_not_evaluate_objectives(self, tmp_path):
        """Reports are regenerated from stored records; no optimizer runs."""
        store = build_store(tmp_path)
        report1 = generate_report(store)
        report2 = generate_report(store)
        assert report1.read_text(encoding="utf-8") == report2.read_text(encoding="utf-8")

    def test_report_lists_skipped_runs_with_reasons(self, tmp_path):
        store = build_store(tmp_path)
        manifest_path = store / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runs"].append({
            "run_id": "skipped00000000001", "experiment_id": "report_test",
            "benchmark_id": "benchmark_a", "dimension": 2, "algorithm_id": "alternating_bo",
            "architecture": "alternating", "seed": 0, "status": "skipped",
            "reason": "incompatible_combination: no recommended grouping",
            "config_fingerprint": "x", "hyperparameters": {},
        })
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = generate_report(store)
        text = report.read_text(encoding="utf-8")
        assert "incompatible_combination" in text


class TestSensitivity:
    def test_scenario_overrides_recorded(self):
        base = make_config()
        scenario = SensitivityScenario(
            scenario_id="small_initial", description="initial design 5",
            overrides={"initial_design.size": 5},
        )
        changed = apply_scenario(base, scenario)
        assert changed.initial_design.size == 5
        assert changed.stopping.evaluation_budget == base.stopping.evaluation_budget  # fixed
        diff = scenario_diff(base, changed)
        assert "initial_design" in diff

    def test_unknown_path_rejected(self):
        with pytest.raises(ValueError, match="Unknown override path"):
            apply_scenario(make_config(), SensitivityScenario(
                scenario_id="bad1", description="x", overrides={"nonexistent.field": 1}))

    def test_no_change_rejected(self):
        base = make_config()
        with pytest.raises(ValueError, match="does not change"):
            apply_scenario(base, SensitivityScenario(
                scenario_id="noop1", description="no change", overrides={}))

    def test_matrix_is_explicit_not_cartesian(self):
        base = make_config()
        scenarios = [
            SensitivityScenario(scenario_id="scn1", description="small initial", overrides={"initial_design.size": 5}),
            SensitivityScenario(scenario_id="scn2", description="smaller pool", overrides={"candidate_pool.num_acquisition_candidates": 100}),
        ]
        matrix = build_sensitivity_experiment(base, scenarios)
        assert len(matrix) == 2  # exactly the listed scenarios — no cross-product
        ids = [c.experiment_id for _, c, _ in matrix]
        assert ids == ["report_test-scn1", "report_test-scn2"]
        for _, config, changed in matrix:
            assert changed  # each scenario records what changed


class TestSimulatorInterfaceAdapter:
    """Item 29: software-interface validation ONLY (mock providers, no physics)."""

    def _setup(self):
        snapshot = MockFrozenSnapshot(snapshot_id="snap_1", values=(1.0, 2.0))
        evaluator = MockSimulatedEvaluator(snapshot, objective=lambda v: -(v[0] - 3.5) ** 2 - (v[1] - 7.2) ** 2 + 10.0)
        problem = SimulatorBackedProblem(
            problem_id="mock_problem", dimension=2,
            lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0],
            variable_types=["continuous", "continuous"],
        )
        return evaluator, problem

    def test_structured_result_conversion_success(self):
        evaluator, problem = self._setup()
        record = evaluate_through_adapter(evaluator, problem, [3.5, 7.2])
        assert record["kind"] == "observation"
        assert record["objective"] == pytest.approx(10.0)
        assert record["evaluation_id"].startswith("mock_eval_")
        assert record["snapshot_id"] == "snap_1"

    def test_snapshot_isolation_no_mutation(self):
        evaluator, problem = self._setup()
        first = evaluate_through_adapter(evaluator, problem, [3.5, 7.2])
        second = evaluate_through_adapter(evaluator, problem, [3.5, 7.2])
        assert first["objective"] == second["objective"]  # deterministic, no state advance
        assert len(evaluator.calls) == 2
        assert evaluator.snapshot.values == (1.0, 2.0)  # untouched
        assert evaluator.snapshot.integrity_sentinel == (True,)  # sentinel intact

    def test_failure_propagation_as_failure_record(self):
        snapshot = MockFrozenSnapshot(snapshot_id="snap_1", values=(1.0, 2.0))
        evaluator = MockSimulatedEvaluator(snapshot, objective=lambda v: 0.0,
                                           fail_on={(9.0, 9.0)})
        problem = SimulatorBackedProblem(
            problem_id="mock_problem", dimension=2,
            lower_bounds=[0.0, 0.0], upper_bounds=[10.0, 10.0],
            variable_types=["continuous", "continuous"],
        )
        record = evaluate_through_adapter(evaluator, problem, [9.0, 9.0])
        assert record["kind"] == "failure"
        assert "synthetic evaluator failure" in record["error"]
        assert "objective" not in record or record["objective"] is None

    def test_adapter_is_synthetic_only(self):
        """The adapter must not import radar/environment modules (layer separation)."""
        import radar_bo.experiments.framework.simulator_adapter as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "radar_bo.radar" not in source
        assert "radar_bo.environment" not in source
