"""M7 item 30: archive-integrity validator unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radar_bo.experiments.framework.archive_integrity import (
    build_checksum_manifest,
    sha256_file,
    validate_experiment_store,
    validate_run_record,
    verify_checksum_manifest,
    write_checksum_manifest,
)


def feasible_true(_config) -> bool:
    return True


def make_record(**overrides) -> dict:
    record = {
        "run_id": "run000000000000001",
        "experiment_id": "exp",
        "experiment_config_fingerprint": "fp0000",
        "algorithm_id": "joint_bo",
        "architecture": "joint",
        "seed": 0,
        "source_commit": "abcdef123456",
        "dependency_versions": {"torch": "2"},
        "observations": [
            {"config": [1.0, 2.0], "objective": 5.0, "evaluation_id": "eval_1"},
            {"config": [3.0, 4.0], "objective": 9.0, "evaluation_id": "eval_2"},
        ],
        "failures": [],
        "incumbent_history": [5.0, 9.0],
        "final_objective": 9.0,
        "final_incumbent_config": [3.0, 4.0],
        "reference_objective": 10.0,
        "evaluation_counts": {
            "n_initial_evaluations": 1, "n_bo_evaluations": 1, "n_failed_attempts": 0,
        },
        "runtime_measurements": {"wall_runtime_s": 10.0, "gp_fit_runtime_s": 8.0},
    }
    record.update(overrides)
    return record


class TestRunRecordValidation:
    def test_clean_record_passes(self):
        issues = validate_run_record(make_record(), feasibility_checker=feasible_true,
                                     expected_reference=10.0)
        assert issues == []

    def test_budget_accounting_violation_detected(self):
        record = make_record(evaluation_counts={
            "n_initial_evaluations": 2, "n_bo_evaluations": 2, "n_failed_attempts": 0,
        })
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("accounting violated" in i for i in issues)

    def test_infeasible_observation_detected(self):
        issues = validate_run_record(make_record(), feasibility_checker=lambda c: False)
        assert any("INFEASIBLE" in i for i in issues)

    def test_incumbent_not_best_observation_detected(self):
        record = make_record(final_objective=5.0)  # best observation is 9.0
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("does not correspond to the best observation" in i for i in issues)

    def test_incumbent_config_not_in_observations_detected(self):
        record = make_record(final_incumbent_config=[99.0, 99.0])
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("not among observations" in i for i in issues)

    def test_non_monotone_history_detected(self):
        record = make_record(incumbent_history=[5.0, 3.0, 9.0])
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("decreases" in i for i in issues)

    def test_duplicate_evaluation_ids_detected(self):
        record = make_record(observations=[
            {"config": [1.0, 2.0], "objective": 5.0, "evaluation_id": "eval_1"},
            {"config": [3.0, 4.0], "objective": 9.0, "evaluation_id": "eval_1"},
        ])
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("duplicate evaluation ids" in i for i in issues)

    def test_reference_mismatch_detected(self):
        issues = validate_run_record(make_record(), feasibility_checker=feasible_true,
                                     expected_reference=11.0)
        assert any("recorded reference" in i for i in issues)

    def test_double_counted_timers_detected(self):
        record = make_record(runtime_measurements={
            "wall_runtime_s": 10.0, "gp_fit_runtime_s": 8.0,
            "candidate_generation_runtime_s": 20.0,  # sums above wall
        })
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("exceeds" in i and "wall" in i for i in issues)

    def test_missing_metadata_detected(self):
        record = make_record(source_commit=None)
        issues = validate_run_record(record, feasibility_checker=feasible_true)
        assert any("missing metadata field 'source_commit'" in i for i in issues)

    def test_minimization_history_supported(self):
        record = make_record(observations=[
            {"config": [1.0, 2.0], "objective": 5.0, "evaluation_id": "eval_1"},
            {"config": [3.0, 4.0], "objective": 2.0, "evaluation_id": "eval_2"},
        ], incumbent_history=[5.0, 2.0], final_objective=2.0)
        issues = validate_run_record(record, feasibility_checker=feasible_true, maximize=False)
        assert issues == []


class TestChecksums:
    def test_sha256_deterministic(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"x": 1}', encoding="utf-8")
        assert sha256_file(f) == sha256_file(f)

    def test_checksum_manifest_detects_corruption(self, tmp_path):
        f = tmp_path / "archive.json"
        f.write_text('{"value": 42}', encoding="utf-8")
        manifest_path = write_checksum_manifest([f], out_path=tmp_path / "checksums.json")
        result = verify_checksum_manifest(manifest_path)
        assert result["all_ok"] is True
        f.write_text('{"value": 43}', encoding="utf-8")  # corrupt the archive
        result = verify_checksum_manifest(manifest_path)
        assert result["all_ok"] is False
        assert result["files"][str(f)]["status"] == "CHECKSUM_MISMATCH"

    def test_checksum_manifest_detects_missing_file(self, tmp_path):
        f = tmp_path / "gone.json"
        f.write_text("{}", encoding="utf-8")
        manifest_path = write_checksum_manifest([f], out_path=tmp_path / "checksums.json")
        f.unlink()
        result = verify_checksum_manifest(manifest_path)
        assert result["files"][str(f)]["status"] == "MISSING"
        assert result["all_ok"] is False

    def test_manifest_distinguishes_derived_artifacts(self, tmp_path):
        original = tmp_path / "archive.json"
        derived = tmp_path / "derived_correction.json"
        original.write_text("{}", encoding="utf-8")
        derived.write_text("{}", encoding="utf-8")
        manifest = build_checksum_manifest([original, derived])
        assert manifest["note"] and "derived" in manifest["note"]


class TestExperimentStoreValidation:
    def _build_store(self, tmp_path):
        store_dir = tmp_path / "experiment"
        (store_dir / "runs").mkdir(parents=True)
        manifest = {"runs": [
            {"run_id": "r1" + "0" * 14, "benchmark_id": "benchmark_a", "dimension": 2,
             "algorithm_id": "joint_bo", "seed": 0, "status": "completed", "reason": None},
        ]}
        (store_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        record = make_record(run_id="r1" + "0" * 14)
        (store_dir / "runs" / f"r1{'0' * 14}.json").write_text(json.dumps(record), encoding="utf-8")
        return store_dir

    def test_valid_store_passes(self, tmp_path):
        store_dir = self._build_store(tmp_path)
        result = validate_experiment_store(store_dir, feasibility_checker=feasible_true,
                                           expected_reference_by_benchmark={"benchmark_a": 10.0})
        assert result["valid"] is True
        assert result["n_completed"] == 1

    def test_missing_expected_combination_reported(self, tmp_path):
        store_dir = self._build_store(tmp_path)
        expected = {("benchmark_a", 2, "pairwise_bo", 0)}
        result = validate_experiment_store(store_dir, feasibility_checker=feasible_true,
                                           expected_combinations=expected)
        assert result["valid"] is False
        assert any("missing run" in i for i in result["issues"])

    def test_failed_run_without_reason_reported(self, tmp_path):
        store_dir = self._build_store(tmp_path)
        manifest = {"runs": [
            {"run_id": "f2" + "0" * 14, "benchmark_id": "benchmark_a", "dimension": 2,
             "algorithm_id": "pairwise_bo", "seed": 1, "status": "failed", "reason": None},
        ]}
        (store_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = validate_experiment_store(store_dir, feasibility_checker=feasible_true)
        assert result["valid"] is False
        assert any("without a recorded reason" in i for i in result["issues"])

    def test_corrupted_run_file_reported(self, tmp_path):
        store_dir = self._build_store(tmp_path)
        (store_dir / "runs" / ("r1" + "0" * 14 + ".json")).write_text("corrupt{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            validate_experiment_store(store_dir, feasibility_checker=feasible_true)
