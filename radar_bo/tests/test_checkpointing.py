"""M7 item 30: checkpoint persistence, run statuses, resume compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar_bo.experiments.framework.checkpointing import (
    CheckpointError,
    CheckpointStore,
    RecordRejected,
    atomic_write_json,
    compatibility_metadata,
)
from radar_bo.experiments.framework.experiment_config import ExperimentConfig, StoppingSettings
from radar_bo.experiments.framework.matrix import RunSpec, RunStatus, generate_manifest
from radar_bo.experiments.framework.benchmark_registry import BenchmarkRegistry
from radar_bo.experiments.framework.algorithm_registry import AlgorithmRegistry


def make_config(tmp_path, **overrides) -> ExperimentConfig:
    base = dict(
        experiment_id="ckpt_test",
        description="checkpoint unit test",
        result_label="generic synthetic unit test; no radar claims",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["random_search", "joint_bo"],
        seeds=[0],
        stopping=StoppingSettings(evaluation_budget=8),
        output_dir=str(tmp_path / "experiment"),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def make_store(tmp_path):
    config = make_config(tmp_path)
    specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
    compatibility = compatibility_metadata(config, BenchmarkRegistry(), AlgorithmRegistry())
    store = CheckpointStore(config.output_dir)
    store.initialize(config, specs, compatibility)
    return config, store, specs, compatibility


class TestAtomicPersistence:
    def test_atomic_write_leaves_valid_json(self, tmp_path):
        target = tmp_path / "nested" / "record.json"
        atomic_write_json(target, {"a": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        leftovers = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_initialize_writes_config_and_manifest(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        stored = json.loads((Path(config.output_dir) / "experiment_config.json").read_text(encoding="utf-8"))
        assert stored["config"]["fingerprint"] == config.fingerprint()
        manifest = json.loads((Path(config.output_dir) / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["runs"]) == len(specs)

    def test_resume_with_same_config_is_compatible(self, tmp_path):
        config, store, specs, compatibility = make_store(tmp_path)
        new_specs = generate_manifest(config, BenchmarkRegistry(), AlgorithmRegistry())
        CheckpointStore(config.output_dir).initialize(config, new_specs, compatibility)  # must not raise

    def test_resume_with_different_config_rejected(self, tmp_path):
        config, store, specs, compatibility = make_store(tmp_path)
        changed = make_config(tmp_path, stopping=StoppingSettings(evaluation_budget=9))
        with pytest.raises(RecordRejected, match="Resume rejected"):
            CheckpointStore(changed.output_dir).initialize(
                changed, generate_manifest(changed, BenchmarkRegistry(), AlgorithmRegistry()), compatibility
            )


class TestStatuses:
    def test_status_transitions_and_reasons(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        store.set_status(run_id, RunStatus.RUNNING)
        store.set_status(run_id, RunStatus.FAILED, reason="synthetic failure")
        stored = {r["run_id"]: r for r in store.specs()}
        assert stored[run_id]["status"] == "failed"
        assert stored[run_id]["reason"] == "synthetic failure"
        with pytest.raises(CheckpointError, match="Unknown run_id"):
            store.set_status("nonexistent000000", RunStatus.COMPLETED)

    def test_completed_requires_and_loads_record(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        store.save_run_record(run_id, {"run_id": run_id, "payload": 42})
        store.set_status(run_id, RunStatus.COMPLETED)
        loaded = store.load_run_record(run_id)
        assert loaded["payload"] == 42
        assert store.completed_run_ids() == {run_id}

    def test_missing_record_for_completed_run_repaired_on_resume(self, tmp_path):
        """A completed status without a record means interrupted mid-write: a
        pending-spec pass must NOT treat it as completed (no record -> retried)."""
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        other_id = specs[1].run_id
        store.set_status(run_id, RunStatus.COMPLETED)  # status without record
        assert run_id not in store.completed_run_ids()  # record missing
        pending_ids = [p["run_id"] for p in store.pending_specs()]
        assert run_id in pending_ids  # the unverified 'completed' run is retried
        assert other_id in pending_ids  # the genuinely pending run as well

    def test_interrupted_running_run_detected(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        store.set_status(run_id, RunStatus.RUNNING, reason="interrupted: sandbox reset")
        store.record_interruption(run_id, "simulated kill")
        pending = store.pending_specs()
        assert run_id in {p["run_id"] for p in pending}


class TestResumeCompatibility:
    def test_corrupted_json_record_rejected(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        store.save_run_record(run_id, {"run_id": run_id})
        path = store.run_path(run_id)
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(RecordRejected, match="Corrupted JSON"):
            store.load_run_record(run_id)

    def test_fingerprint_mismatch_rejected(self, tmp_path):
        config, store, specs, _ = make_store(tmp_path)
        run_id = specs[0].run_id
        store.save_run_record(run_id, {"run_id": run_id, "experiment_config_fingerprint": "other"})
        with pytest.raises(RecordRejected, match="different experiment configuration"):
            store.load_run_record(run_id, expected_fingerprint=config.fingerprint())

    def test_compatibility_metadata_captures_required_fields(self, tmp_path):
        config = make_config(tmp_path)
        compat = compatibility_metadata(config, BenchmarkRegistry(), AlgorithmRegistry())
        for key in ("config_fingerprint", "benchmark_versions", "algorithm_versions",
                    "dependency_versions", "source_commit", "seeds", "python"):
            assert key in compat
        assert "torch" in compat["dependency_versions"]
