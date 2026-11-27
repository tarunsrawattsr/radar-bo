"""Checkpoint store, manifest persistence and resume compatibility (M7 items 11-12).

Layout of one experiment directory::

    <output_dir>/
        experiment_config.json     # full configuration + fingerprint
        manifest.json              # requested runs + statuses + reasons
        runs/<run_id>.json         # one atomic file per completed/failed run
        summary.json               # experiment summary (after execution)
        reports/report.md          # generated report (item 27)
        figures/*.png              # generated figures (item 28)

Writes are atomic (temporary file + ``os.replace`` in the same directory) so
an interrupted process never leaves a partially written run record behind.
Resume compatibility is verified against configuration fingerprint, benchmark
and algorithm versions, dependency versions, seeds and run identifiers;
incompatible or corrupted records are explicitly rejected (never silently
combined).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from radar_bo.experiments.framework.experiment_config import ExperimentConfig, config_fingerprint
from radar_bo.experiments.framework.matrix import RunSpec, RunStatus


class CheckpointError(RuntimeError):
    pass


class RecordRejected(CheckpointError):
    """A stored record failed compatibility or integrity validation (item 11)."""


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class CheckpointStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.dir = Path(output_dir)
        self.runs_dir = self.dir / "runs"
        self.config_path = self.dir / "experiment_config.json"
        self.manifest_path = self.dir / "manifest.json"

    # ---- creation / resume entry points -------------------------------------
    def initialize(self, config: ExperimentConfig, specs: list[RunSpec],
                   compatibility: dict) -> None:
        """Create the experiment directory (or resume an existing compatible one)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            stored = _read_json(self.config_path)
            if stored["config"]["fingerprint"] != config.fingerprint():
                raise RecordRejected(
                    "Resume rejected: stored experiment configuration fingerprint "
                    f"{stored['config']['fingerprint'][:12]} != current {config.fingerprint()[:12]}"
                )
            if stored.get("compatibility") != compatibility:
                raise RecordRejected("Resume rejected: stored compatibility metadata differs from current environment.")
            stored_manifest = _read_json(self.manifest_path)
            self._merge_manifest(specs, stored_manifest)
        else:
            atomic_write_json(self.config_path, {
                "config": {"model_dump": config.model_dump(), "fingerprint": config.fingerprint()},
                "compatibility": compatibility,
            })
            self._write_manifest(specs, compatibility)

    def _manifest_specs(self) -> list[dict]:
        manifest = _read_json(self.manifest_path)
        return manifest["runs"]

    def _write_manifest(self, specs: list[RunSpec], compatibility: dict) -> None:
        atomic_write_json(self.manifest_path, {
            "runs": [s.model_dump() for s in specs],
            "compatibility": compatibility,
        })

    def _merge_manifest(self, specs: list[RunSpec], stored_manifest: dict) -> None:
        stored_by_id = {r["run_id"]: r for r in stored_manifest["runs"]}
        merged: list[dict] = []
        for spec in specs:
            s = spec.model_dump()
            prior = stored_by_id.get(spec.run_id)
            if prior is not None:
                s["status"] = prior.get("status", spec.status.value)
                s["reason"] = prior.get("reason")
            merged.append(s)
        extra = [r for r in stored_manifest["runs"] if r["run_id"] not in {x["run_id"] for x in merged}]
        for r in extra:
            raise RecordRejected(
                f"Resume rejected: stored manifest contains run_id {r['run_id']} that is not "
                "in the regenerated manifest (configuration or registry drift)."
            )
        atomic_write_json(self.manifest_path, {"runs": merged, "compatibility": stored_manifest["compatibility"]})

    # ---- statuses ------------------------------------------------------------
    def specs(self) -> list[dict]:
        return self._manifest_specs()

    def set_status(self, run_id: str, status: RunStatus, reason: str | None = None) -> None:
        runs = self._manifest_specs()
        for r in runs:
            if r["run_id"] == run_id:
                r["status"] = status.value
                r["reason"] = reason
                break
        else:
            raise CheckpointError(f"Unknown run_id in manifest: {run_id}")
        atomic_write_json(self.manifest_path, {
            "runs": runs,
            "compatibility": _read_json(self.manifest_path)["compatibility"],
        })

    # ---- run records ----------------------------------------------------------
    def run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def save_run_record(self, run_id: str, record: dict) -> None:
        atomic_write_json(self.run_path(run_id), record)

    def load_run_record(self, run_id: str, expected_fingerprint: str | None = None) -> dict:
        path = self.run_path(run_id)
        if not path.exists():
            raise CheckpointError(f"No stored record for run_id {run_id}.")
        try:
            record = _read_json(path)
        except json.JSONDecodeError as exc:
            raise RecordRejected(f"Corrupted JSON record for run_id {run_id}: {exc}") from exc
        if expected_fingerprint is not None:
            stored = record.get("experiment_config_fingerprint")
            if stored != expected_fingerprint:
                raise RecordRejected(
                    f"Record {run_id} belongs to a different experiment configuration "
                    f"(stored fingerprint {str(stored)[:12]}; expected {expected_fingerprint[:12]})."
                )
        return record

    def completed_run_ids(self) -> set[str]:
        return {
            r["run_id"] for r in self._manifest_specs()
            if r["status"] == RunStatus.COMPLETED.value and self.run_path(r["run_id"]).exists()
        }

    def pending_specs(self) -> list[dict]:
        """Runs still to execute: pending, interrupted 'running', or FAILED
        (a failed run never completed and is retried on resume; its prior
        failure reason is retained in the manifest history)."""
        completed = self.completed_run_ids()
        out = []
        for r in self._manifest_specs():
            if r["status"] in (RunStatus.PENDING.value, RunStatus.RUNNING.value, RunStatus.FAILED.value):
                if r["run_id"] in completed:
                    self.set_status(r["run_id"], RunStatus.COMPLETED)
                    continue
                out.append(r)
            elif r["status"] == RunStatus.COMPLETED.value and r["run_id"] not in completed:
                # completed status WITHOUT a stored record: the run is not
                # verifiably complete -> treat as pending (re-executed).
                out.append(r)
        return out

    def record_interruption(self, run_id: str, reason: str) -> None:
        self.set_status(run_id, RunStatus.RUNNING, reason=f"interrupted: {reason}")


def compatibility_metadata(config: ExperimentConfig, benchmark_registry, algorithm_registry) -> dict:
    """The resume-compatibility block (item 11 checks)."""
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    return {
        "config_fingerprint": config.fingerprint(),
        "benchmark_versions": {
            bid: benchmark_registry.get(bid).implementation_reference for bid in config.benchmark_ids
        },
        "algorithm_versions": {
            aid: algorithm_registry.get(aid).version for aid in config.algorithm_ids
        },
        "dependency_versions": {
            "numpy": __import__("numpy").__version__,
            "torch": __import__("torch").__version__,
            "gpytorch": __import__("gpytorch").__version__,
            "botorch": __import__("botorch").__version__,
            "scipy": __import__("scipy").__version__,
            "pydantic": __import__("pydantic").VERSION,
        },
        "source_commit": commit,
        "seeds": config.seeds,
        "python": __import__("sys").version.split()[0],
    }
