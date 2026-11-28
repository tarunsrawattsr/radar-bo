"""M8A-R1 regression tests (Phases B and C).

Covers the defects identified by the independent review of commit 7e447be:

Phase B (shared initial designs):
* genuine SHA-256 fingerprints over persisted authoritative designs;
* reload-on-resume (a stored design is never silently regenerated);
* malformed / non-finite / infeasible / duplicate / incomplete designs
  are DETECTED and rejected;
* the insufficient-feasible-design error path works (no TypeError);
* the separately identified shared-start random search evaluates exactly
  the authoritative initial phase, while plain random search ignores it.

Phase C (experiment framework):
* legitimate early stopping is COMPLETED (early), not FAILED;
* over-budget is a hard violation (unit-level classification);
* resume ADOPTS a persisted record whose manifest update was interrupted
  (no repetition of completed compatible runs);
* failure and retry histories are preserved in the manifest;
* shared-initialization verification detects inconsistent records against
  the persisted authoritative design;
* checkpoint compatibility checks reject configuration/identity drift.

No historical behavior is changed: seeded_sampler mode and plain random
search keep their M5/M6 semantics (covered by the pre-existing suites).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.checkpointing import (
    CheckpointStore,
    RecordRejected,
    compatibility_metadata,
)
from radar_bo.experiments.framework.experiment_config import (
    ExperimentConfig,
    InitialDesignSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.initial_design_store import (
    DesignValidationError,
    InitialDesignStore,
    design_fingerprint,
    validate_design_rows,
)
from radar_bo.experiments.framework.matrix import RunStatus, generate_manifest
from radar_bo.experiments.framework.runner import ExperimentRunner, verify_shared_initialization


def make_config(output_dir: str, **overrides) -> ExperimentConfig:
    base = dict(
        experiment_id="m8a_r1_corrections_test",
        description="M8A-R1 correction regression test",
        result_label="generic synthetic correction test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo", "reference"],
        seeds=[0],
        initial_design=InitialDesignSettings(size=3),
        stopping=StoppingSettings(evaluation_budget=7),
        output_dir=output_dir,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# ---------------------------------------------------------------------------
# Phase B — initial-design store and fingerprints
# ---------------------------------------------------------------------------

class TestDesignFingerprint:
    def test_fingerprint_is_genuine_sha256(self):
        """The recorded digest is a real SHA-256 over the canonical row JSON,
        recomputable with plain hashlib from the stored rows."""
        rows = [[0.5, 1.5], [1.25, 2.0]]
        fp = design_fingerprint(rows)
        canonical = json.dumps([[0.5, 1.5], [1.25, 2.0]], separators=(",", ":"))
        assert fp == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_changes_with_content(self):
        assert design_fingerprint([[0.5, 1.5]]) != design_fingerprint([[0.5, 1.500001]])

    def test_float_reordering_stability(self):
        # canonical form is order-preserving per row and stable for equal inputs
        assert design_fingerprint([[1.0, 2.0], [3.0, 4.0]]) == design_fingerprint([[1.0, 2.0], [3.0, 4.0]])


class TestDesignValidation:
    @pytest.fixture()
    def problem(self):
        return BenchmarkRegistry().get("benchmark_a").problem(2)

    def test_incomplete_design_detected(self, problem):
        issues = validate_design_rows([[0.5, 1.5]], problem, expected_size=3)
        assert any("incomplete" in i for i in issues)

    def test_non_finite_detected(self, problem):
        rows = [[0.5, float("nan")], [1.0, 2.0], [2.0, 1.0]]
        issues = validate_design_rows(rows, problem, expected_size=3)
        assert any("non-finite" in i for i in issues)

    def test_infeasible_detected(self, problem):
        rows = [[100.0, 100.0], [1.0, 2.0], [2.0, 1.0]]
        issues = validate_design_rows(rows, problem, expected_size=3)
        assert any("infeasible" in i for i in issues)

    def test_duplicate_rows_detected(self, problem):
        rows = [[0.5, 1.5], [0.5, 1.5], [2.0, 1.0]]
        issues = validate_design_rows(rows, problem, expected_size=3)
        assert any("duplicate" in i for i in issues)

    def test_wrong_dimension_detected(self, problem):
        rows = [[0.5, 1.5, 9.9], [1.0, 2.0], [2.0, 1.0]]
        issues = validate_design_rows(rows, problem, expected_size=3)
        assert any("dimension" in i for i in issues)

    def test_malformed_non_numeric_detected(self, problem):
        rows = [["bad", 1.5], [1.0, 2.0], [2.0, 1.0]]
        issues = validate_design_rows(rows, problem, expected_size=3)
        assert any("malformed" in i for i in issues)

    def test_valid_design_has_no_issues(self, problem):
        rows = [[0.5, 1.5], [1.0, 2.0], [2.0, 1.0]]
        assert validate_design_rows(rows, problem, expected_size=3) == []


class TestDesignStorePersistence:
    def test_create_persists_and_reload_matches(self, tmp_path):
        store = InitialDesignStore(str(tmp_path))
        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        rows, fp, created = store.load_or_create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        assert created is True
        rows2, fp2, created2 = store.load_or_create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        # reload path: identical rows, identical genuine fingerprint, no regeneration
        assert created2 is False
        assert rows2 == rows and fp2 == fp == design_fingerprint(rows)

    def test_tampered_rows_rejected_on_load(self, tmp_path):
        store = InitialDesignStore(str(tmp_path))
        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        rows, fp, _ = store.load_or_create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        # tamper: change one row without updating the digest
        path = store._path(("benchmark_a", 2, 0))
        payload = json.loads(path.read_text())
        payload["rows"][0][0] = payload["rows"][0][0] + 0.125
        path.write_text(json.dumps(payload))
        with pytest.raises(DesignValidationError, match="fingerprint mismatch"):
            store.load(("benchmark_a", 2, 0), problem, expected_size=3)

    def test_tampered_digest_rejected_on_load(self, tmp_path):
        store = InitialDesignStore(str(tmp_path))
        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        store.load_or_create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        path = store._path(("benchmark_a", 2, 0))
        payload = json.loads(path.read_text())
        payload["design_sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
        with pytest.raises(DesignValidationError, match="fingerprint mismatch"):
            store.load(("benchmark_a", 2, 0), problem, expected_size=3)

    def test_stored_rows_edited_consistently_still_validated_for_feasibility(self, tmp_path):
        """If both rows AND digest are edited, validation still catches
        infeasible/duplicate content — the digest alone is not trusted."""
        store = InitialDesignStore(str(tmp_path))
        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        store.load_or_create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        path = store._path(("benchmark_a", 2, 0))
        payload = json.loads(path.read_text())
        payload["rows"][1] = list(payload["rows"][0])  # duplicate
        payload["design_sha256"] = design_fingerprint(payload["rows"])
        path.write_text(json.dumps(payload))
        with pytest.raises(DesignValidationError, match="duplicate"):
            store.load(("benchmark_a", 2, 0), problem, expected_size=3)

    def test_insufficient_feasible_design_raises_runtime_error(self, tmp_path, monkeypatch):
        """The insufficient-feasible-design error path: a RuntimeError with the
        group identity (no TypeError from wrong attribute access)."""
        store = InitialDesignStore(str(tmp_path))
        problem = BenchmarkRegistry().get("benchmark_a").problem(2)

        from radar_bo.optimization.bo import candidates as candidates_module

        class SmallPool(CandidateGeneratorStub):
            pass

        # monkeypatch generate_pool on the real class
        original = candidates_module.CandidateGenerator.generate_pool

        def small_pool(self):
            return np.zeros((1, problem.dimension)), None

        monkeypatch.setattr(candidates_module.CandidateGenerator, "generate_pool", small_pool)
        with pytest.raises(RuntimeError, match="benchmark 'benchmark_a' d=2 seed=0"):
            store.create(("benchmark_a", 2, 0), problem, size=3, seed=0, metadata={})
        monkeypatch.setattr(candidates_module.CandidateGenerator, "generate_pool", original)


class CandidateGeneratorStub:
    pass


# ---------------------------------------------------------------------------
# Phase B — shared-start random search vs plain random search
# ---------------------------------------------------------------------------

class TestSharedStartRandomSearch:
    def test_framework_run_shared_start_matches_bo_initial_phase(self, tmp_path):
        config = make_config(
            str(tmp_path / "exp"),
            algorithm_ids=["joint_bo", "random_search_shared_start", "reference"],
            initial_design=InitialDesignSettings(size=3, mode="explicit_shared"),
        )
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        summary = runner.execute()
        assert summary["shared_initialization_verification"]["verified"] is True, \
            summary["shared_initialization_verification"]["issues"]
        # the shared-start RS participates in the verification group (it HAS an
        # initial phase); plain random search would be excluded instead
        group = summary["shared_initialization_verification"]["groups"][0]
        assert "random_search_shared_start" in group["methods"]
        assert group["authoritative_match"] is True

    def test_plain_random_search_ignores_design_trajectorily(self):
        from radar_bo.optimization.bo.random_search import RandomSearchOptimizer

        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        design = [[0.5, 1.5], [1.0, 2.0], [2.0, 1.0]]
        from radar_bo.experiments.framework.algorithm_registry import _run_random_search

        benchmark = BenchmarkRegistry().get("benchmark_a")
        ref = benchmark.reference(2)
        cfg = OptimizationConfig(evaluation_budget=9)
        out_with = _run_random_search(
            benchmark=benchmark, dimension=2, seed=11, config=cfg,
            reference_value=ref.objective_value, reference_method=ref.verification_method,
            initial_design_configs=design,
        )
        out_without = _run_random_search(
            benchmark=benchmark, dimension=2, seed=11, config=cfg,
            reference_value=ref.objective_value, reference_method=ref.verification_method,
        )
        assert [e[1] for e in out_with.optimization_result.history] == \
               [e[1] for e in out_without.optimization_result.history]
        assert out_with.architecture == "random_search"

    def test_shared_start_evaluates_authoritative_phase_then_random(self, tmp_path):
        from radar_bo.optimization.bo.random_search import RandomSearchOptimizer

        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        store = InitialDesignStore(str(tmp_path))
        rows, fp, _ = store.load_or_create(("benchmark_a", 2, 3), problem, size=3, seed=3, metadata={})
        result = RandomSearchOptimizer(problem, 7, seed=5, initial_configs=rows).optimize()
        assert result.n_initial_evaluations == 3
        assert result.n_objective_evaluations == 7  # 3 shared + 4 random, budget respected
        initial_configs = [c for c, _v, _e in result.initial_phase]
        assert initial_configs == [[float(v) for v in row] for row in rows]
        # the random-phase samples must not repeat the initial rows (deterministic problem)
        random_phase = result.history[3:]
        for config_list, _v, _e in random_phase:
            for row in rows:
                assert max(abs(a - b) for a, b in zip(config_list, row)) > 1e-9

    def test_shared_start_rejects_bad_designs(self):
        from radar_bo.optimization.bo.random_search import RandomSearchOptimizer

        problem = BenchmarkRegistry().get("benchmark_a").problem(2)
        with pytest.raises(ValueError, match="infeasible"):
            RandomSearchOptimizer(problem, 9, seed=1, initial_configs=[[1000.0, 1000.0]])
        with pytest.raises(ValueError, match="non-finite"):
            RandomSearchOptimizer(problem, 9, seed=1, initial_configs=[[float("inf"), 1.0]])
        with pytest.raises(ValueError, match="dimension"):
            RandomSearchOptimizer(problem, 9, seed=1, initial_configs=[[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="exceeds"):
            RandomSearchOptimizer(problem, 2, seed=1, initial_configs=[[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]])


# ---------------------------------------------------------------------------
# Phase C — budget classification, adoption, history, compatibility
# ---------------------------------------------------------------------------

class TestCompletionClassification:
    def _record(self, reason):
        return {"stopping_reason": reason}

    def _runner(self, tmp_path):
        return ExperimentRunner(make_config(str(tmp_path / "exp")), BenchmarkRegistry(), AlgorithmRegistry())

    def test_full_budget_is_full(self, tmp_path):
        assert self._runner(tmp_path)._classify_completion(self._record("evaluation_budget_exhausted"), 7, 7) == "full"

    def test_patience_is_early_not_failure(self, tmp_path):
        assert self._runner(tmp_path)._classify_completion(self._record("patience_without_improvement"), 5, 7) == "early"

    def test_runtime_limit_is_early(self, tmp_path):
        assert self._runner(tmp_path)._classify_completion(self._record("runtime_limit_reached"), 4, 7) == "early"

    def test_engine_failure_reasons_classified(self, tmp_path):
        r = self._runner(tmp_path)
        assert r._classify_completion(self._record("no_feasible_initial_design"), 0, 7) == "engine_failure"
        assert r._classify_completion(self._record("gp_fitting_failure"), 3, 7) == "engine_failure"
        assert r._classify_completion(self._record("repeated_objective_failures"), 2, 7) == "engine_failure"

    def test_over_budget_is_violation(self, tmp_path):
        assert self._runner(tmp_path)._classify_completion(self._record("evaluation_budget_exhausted"), 8, 7) == "violation"

    def test_unknown_under_budget_reason_is_early(self, tmp_path):
        assert self._runner(tmp_path)._classify_completion(self._record("mystery"), 6, 7) == "early"


class TestEarlyStoppingEndToEnd:
    def test_patience_run_completes_early_not_failed(self, tmp_path):
        config = make_config(
            str(tmp_path / "exp"),
            algorithm_ids=["joint_bo", "reference"],
            stopping=StoppingSettings(evaluation_budget=25, patience=2),
        )
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        summary = runner.execute()
        manifest = {r["run_id"]: r for r in CheckpointStore(str(tmp_path / "exp")).specs()}
        joint = [r for r in manifest.values() if r["algorithm_id"] == "joint_bo"][0]
        assert joint["status"] == "completed", joint.get("reason")
        record = runner.store.load_run_record(joint["run_id"], expected_fingerprint=config.fingerprint())
        attempted = (record["evaluation_counts"]["n_initial_evaluations"]
                     + record["evaluation_counts"]["n_bo_evaluations"])
        assert attempted < 25
        assert record["early_completion"] == "early"
        assert record["stopping_reason"] == "patience_without_improvement"
        # completed runs: no 'failed' status anywhere in the summary
        assert summary["status_counts"].get("failed", 0) == 0


class TestResumeAdoption:
    def test_persisted_record_adopted_not_reexecuted(self, tmp_path):
        """Interruption between result persistence and manifest update: the
        record file exists, the manifest still says RUNNING. Resume must ADOPT
        the record (zero re-execution), not repeat the run."""
        out_dir = str(tmp_path / "exp")
        config = make_config(out_dir, seeds=[0])
        counting = CountingBenchmarkRegistry()
        runner = ExperimentRunner(config, counting, AlgorithmRegistry())
        runner.execute()
        calls_full = counting.evaluate_calls

        # simulate the interruption window: status RUNNING, record present
        store = CheckpointStore(out_dir)
        joint_ids = [r["run_id"] for r in store.specs() if r["algorithm_id"] == "joint_bo"]
        store.set_status(joint_ids[0], RunStatus.RUNNING, reason="interrupted: simulated kill")

        counting2 = CountingBenchmarkRegistry()
        resume = ExperimentRunner(config, counting2, AlgorithmRegistry())
        resume.execute()
        assert counting2.evaluate_calls == 0, "adopted run must not re-execute"
        statuses = {r["run_id"]: r for r in store.specs()}
        assert statuses[joint_ids[0]]["status"] == "completed"
        assert any("adopted" in (h.get("reason") or "") for h in statuses[joint_ids[0]]["history"])

    def test_identity_mismatch_rejected(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(out_dir, seeds=[0])
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(out_dir)
        joint_ids = [r["run_id"] for r in store.specs() if r["algorithm_id"] == "joint_bo"]
        store.set_status(joint_ids[0], RunStatus.RUNNING, reason="interrupted")
        # tamper the record's identity
        record = store.load_run_record(joint_ids[0])
        record["benchmark_id"] = "benchmark_b"
        store.save_run_record(joint_ids[0], record)
        with pytest.raises(RecordRejected, match="benchmark_id"):
            ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry()).execute()

    def test_failure_history_preserved_across_retry(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(out_dir, seeds=[0])
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(out_dir)
        joint_ids = [r["run_id"] for r in store.specs() if r["algorithm_id"] == "joint_bo"]
        # simulate a historical failure without a record (retryable): FAILED, no record
        record = store.load_run_record(joint_ids[0])
        store.set_status(joint_ids[0], RunStatus.FAILED, reason="simulated historical failure")
        store.run_path(joint_ids[0]).unlink()
        ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry()).execute()
        specs = {r["run_id"]: r for r in store.specs()}
        hist = specs[joint_ids[0]]["history"]
        statuses_in_history = [h["status"] for h in hist]
        assert "failed" in statuses_in_history and statuses_in_history[-1] == "completed"
        assert hist[[i for i, h in enumerate(hist) if h["status"] == "failed"][0]]["reason"] == "simulated historical failure"


class TestCheckpointCompatibility:
    def test_config_drift_rejected(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(out_dir, seeds=[0])
        ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry()).execute()
        drifted = make_config(out_dir, seeds=[0], stopping=StoppingSettings(evaluation_budget=9))
        with pytest.raises(RecordRejected, match="fingerprint"):
            ExperimentRunner(drifted, BenchmarkRegistry(), AlgorithmRegistry()).execute()

    def test_compatibility_component_reported(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(out_dir, seeds=[0])
        ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry()).execute()
        store = CheckpointStore(out_dir)
        # corrupt the stored compatibility block
        stored = json.loads(store.config_path.read_text())
        stored["compatibility"]["dependency_versions"]["numpy"] = "0.0.0"
        store.config_path.write_text(json.dumps(stored))
        with pytest.raises(RecordRejected, match="dependency_versions"):
            ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry()).execute()


class TestVerificationAgainstPersistedDesign:
    def test_tampered_initial_phase_detected(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(
            out_dir,
            algorithm_ids=["joint_bo", "reference"],
            initial_design=InitialDesignSettings(size=3, mode="explicit_shared"),
        )
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        summary = runner.execute()
        assert summary["shared_initialization_verification"]["verified"] is True

        store = CheckpointStore(out_dir)
        joint_id = [r["run_id"] for r in store.specs() if r["algorithm_id"] == "joint_bo"][0]
        record = store.load_run_record(joint_id, expected_fingerprint=config.fingerprint())
        record["initial_design"][1]["objective"] += 1e-6  # inconsistent observed value
        store.save_run_record(joint_id, record)
        records = runner._collect_records()
        designs = runner._designs
        fairness = verify_shared_initialization(
            records, authoritative_designs=designs,
            objective_prover=runner._recompute_authoritative_objectives,
        )
        assert fairness["verified"] is False
        assert any("objective" in i for i in fairness["issues"])

    def test_missing_fingerprint_detected(self, tmp_path):
        out_dir = str(tmp_path / "exp")
        config = make_config(
            out_dir,
            algorithm_ids=["joint_bo", "reference"],
            initial_design=InitialDesignSettings(size=3, mode="explicit_shared"),
        )
        runner = ExperimentRunner(config, BenchmarkRegistry(), AlgorithmRegistry())
        runner.execute()
        store = CheckpointStore(out_dir)
        joint_id = [r["run_id"] for r in store.specs() if r["algorithm_id"] == "joint_bo"][0]
        record = store.load_run_record(joint_id, expected_fingerprint=config.fingerprint())
        record["initial_design_fingerprint"] = None
        store.save_run_record(joint_id, record)
        records = runner._collect_records()
        fairness = verify_shared_initialization(records, authoritative_designs=runner._designs)
        assert fairness["verified"] is False
        assert any("fingerprint" in i for i in fairness["issues"])


class CountingBenchmarkRegistry(BenchmarkRegistry):
    """Benchmark registry counting objective evaluations (mirrors the M7 test)."""

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
