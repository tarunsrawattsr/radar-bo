"""Archive integrity verification (M7 items 4 and 14).

A reusable validator for durable evidence archives:

* sha256 checksums for evidence files (``research_results/ARCHIVE_CHECKSUMS.json``);
* structural validation of M5-style and M7-runner-style run records:
  unique run/evaluation identifiers, expected experiment combinations,
  objective-evaluation accounting, feasibility of successful observations,
  final-incumbent correspondence, monotone deterministic incumbent
  histories, reference consistency, valid runtime fields, missing or
  duplicate records, configuration/version metadata;
* explicit separation of ORIGINAL run records from DERIVED analysis
  artifacts (``derived`` flag / directory naming ``derived_*``).

Historical M5/M6 archives are read-only inputs here: this module NEVER
modifies them (item 14). All corrections are new ``derived_*`` artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

FEASIBILITY_TOLERANCE = 1e-9
INCUMBENT_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checksum_manifest(paths: list[str | Path], base_dir: str | Path = ".") -> dict:
    entries = {}
    for p in paths:
        path = Path(p)
        entries[str(path)] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "purpose": "integrity identifiers for durable evidence archives (M7 item 14)",
        "note": "original run records only; derived analysis artifacts carry a derived_ name prefix",
        "entries": entries,
    }


def write_checksum_manifest(paths: list[str | Path], out_path: str | Path = "research_results/ARCHIVE_CHECKSUMS.json") -> Path:
    manifest = build_checksum_manifest(paths)
    out = Path(out_path)
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def verify_checksum_manifest(manifest_path: str | Path = "research_results/ARCHIVE_CHECKSUMS.json") -> dict:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    results = {}
    for rel, entry in data["entries"].items():
        path = Path(rel)
        if not path.exists():
            results[rel] = {"status": "MISSING"}
            continue
        if sha256_file(path) != entry["sha256"]:
            results[rel] = {"status": "CHECKSUM_MISMATCH"}
        else:
            results[rel] = {"status": "OK", "bytes": path.stat().st_size}
    return {"all_ok": all(v["status"] == "OK" for v in results.values()), "files": results}


# ---- record-level checks ------------------------------------------------------------
def _validate_incumbent_correspondence(record: dict, maximize: bool = True, issues: list[str] | None = None) -> None:
    observations = record.get("observations") or []
    best = record.get("final_objective")
    if best is None:
        if observations:
            assert issues is not None
            issues.append(f"{record.get('run_id', '?')}: final_objective is None but observations exist")
        return
    if not observations:
        assert issues is not None
        issues.append(f"{record.get('run_id', '?')}: final_objective set without observations")
        return
    values = [o["objective"] for o in observations]
    best_from_observations = max(values) if maximize else min(values)
    if abs(best_from_observations - best) > INCUMBENT_TOLERANCE:
        assert issues is not None
        issues.append(
            f"{record.get('run_id', '?')}: final_objective {best} does not correspond to the best "
            f"observation {best_from_observations} (incumbent must be an OBSERVED value, never predicted)"
        )
    final_config = record.get("final_incumbent_config")
    if final_config is not None:
        matching = [o for o in observations
                    if all(abs(a - b) <= 1e-12 for a, b in zip(o["config"], final_config))]
        if not matching:
            assert issues is not None
            issues.append(f"{record.get('run_id', '?')}: final incumbent configuration not among observations")


def _validate_history_monotone(record: dict, maximize: bool = True, issues: list[str] | None = None) -> None:
    history = record.get("incumbent_history") or []
    for a, b in zip(history, history[1:]):
        if maximize and b < a - INCUMBENT_TOLERANCE:
            assert issues is not None
            issues.append(f"{record.get('run_id', '?')}: incumbent history decreases ({a} -> {b})")
            return


def _validate_runtime_fields(record: dict, issues: list[str] | None = None) -> None:
    rt = record.get("runtime_measurements") or {}
    wall = rt.get("wall_runtime_s")
    if wall is None:
        return
    if wall < 0:
        assert issues is not None
        issues.append(f"{record.get('run_id', '?')}: negative wall runtime")
    components = [rt.get(k) for k in (
        "gp_fit_runtime_s", "acquisition_runtime_s", "objective_runtime_s",
        "candidate_generation_runtime_s", "constraint_validation_runtime_s") if rt.get(k) is not None]
    if components and sum(components) > wall + 1e-6:
        assert issues is not None
        issues.append(
            f"{record.get('run_id', '?')}: component timers sum ({sum(components):.3f}s) exceeds "
            f"wall ({wall:.3f}s) — double counting or timer-boundary error"
        )
    for k, v in rt.items():
        if isinstance(v, float) and v < -1e-12:
            assert issues is not None
            issues.append(f"{record.get('run_id', '?')}: negative runtime field {k}={v}")


def _validate_metadata(record: dict, issues: list[str] | None = None) -> None:
    required = ("run_id", "experiment_id", "algorithm_id", "architecture", "source_commit", "dependency_versions")
    for key in required:
        if record.get(key) in (None, ""):
            assert issues is not None
            issues.append(f"{record.get('run_id', '?')}: missing metadata field '{key}'")
    if record.get("experiment_config_fingerprint") is None and record.get("algorithm_id") != "reference":
        assert issues is not None
        issues.append(f"{record.get('run_id', '?')}: missing experiment_config_fingerprint")


def validate_run_record(
    record: dict,
    feasibility_checker: Callable[[list[float]], bool] | None = None,
    expected_reference: float | None = None,
    maximize: bool = True,
    issues: list[str] | None = None,
) -> list[str]:
    """Validate ONE run record; returns the accumulated issues list (item 4)."""
    if issues is None:
        issues = []
    # unique evaluation ids
    evaluation_ids = [o.get("evaluation_id") for o in (record.get("observations") or []) if o.get("evaluation_id")]
    duplicates = {e for e in evaluation_ids if evaluation_ids.count(e) > 1}
    if duplicates:
        issues.append(f"{record.get('run_id', '?')}: duplicate evaluation ids {sorted(duplicates)[:3]}")
    # evaluation accounting (BO records)
    counts = record.get("evaluation_counts") or {}
    if "n_initial_evaluations" in counts:
        attempted = counts["n_initial_evaluations"] + counts["n_bo_evaluations"]
        successful = len(record.get("observations") or [])
        failures = len(record.get("failures") or [])
        if successful + failures != attempted:
            issues.append(
                f"{record.get('run_id', '?')}: accounting violated: successful ({successful}) + "
                f"failed ({failures}) != attempted ({attempted})"
            )
        if counts.get("n_failed_attempts") != failures:
            issues.append(f"{record.get('run_id', '?')}: n_failed_attempts != len(failures)")
    elif "n_objective_evaluations" in counts:  # random-search records
        successful = len(record.get("observations") or [])
        if successful != counts["n_objective_evaluations"]:
            issues.append(f"{record.get('run_id', '?')}: RS observations != n_objective_evaluations")
    # feasibility of successful observations (recomputed, independent of the run)
    if feasibility_checker is not None:
        for o in record.get("observations") or []:
            if not feasibility_checker(list(o["config"])):
                issues.append(f"{record.get('run_id', '?')}: observation {o.get('evaluation_id')} is INFEASIBLE")
    # incumbent correspondence + monotone history + runtimes + metadata
    _validate_incumbent_correspondence(record, maximize, issues)
    _validate_history_monotone(record, maximize, issues)
    _validate_runtime_fields(record, issues)
    _validate_metadata(record, issues)
    # reference consistency
    if expected_reference is not None and record.get("reference_objective") is not None:
        if abs(record["reference_objective"] - expected_reference) > 1e-9:
            issues.append(
                f"{record.get('run_id', '?')}: recorded reference {record['reference_objective']} != "
                f"registry reference {expected_reference}"
            )
    return issues


def validate_experiment_store(
    store_dir: str | Path,
    feasibility_checker: Callable[[list[float]], bool] | None = None,
    expected_reference_by_benchmark: dict[str, float] | None = None,
    expected_combinations: set[tuple] | None = None,
    maximize: bool = True,
) -> dict[str, Any]:
    """Validate a whole M7 runner experiment store (manifest + run files)."""
    store_dir = Path(store_dir)
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    issues: list[str] = []
    run_ids: set[str] = set()
    completed_combinations: set[tuple] = set()
    n_completed = 0
    for spec in manifest["runs"]:
        run_id = spec["run_id"]
        if run_id in run_ids:
            issues.append(f"duplicate run_id in manifest: {run_id}")
        run_ids.add(run_id)
        if spec["status"] == "completed":
            n_completed += 1
            record = json.loads((store_dir / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
            if record.get("run_id") != run_id:
                issues.append(f"run file {run_id}.json carries a different run_id ({record.get('run_id')})")
            expected_reference = (expected_reference_by_benchmark or {}).get(spec["benchmark_id"])
            validate_run_record(record, feasibility_checker, expected_reference, maximize, issues)
            if spec.get("seed") is not None:
                completed_combinations.add(
                    (spec["benchmark_id"], spec["dimension"], spec["algorithm_id"], spec["seed"])
                )
        elif spec["status"] in ("failed", "skipped") and not spec.get("reason"):
            issues.append(f"run {run_id} is {spec['status']} without a recorded reason (item 12)")
    missing = expected = None
    if expected_combinations is not None:
        missing = sorted(expected_combinations - completed_combinations)
        extra = sorted(completed_combinations - expected_combinations)
        for combo in missing:
            issues.append(f"missing run (not completed): {combo}")
        for combo in extra:
            issues.append(f"unexpected completed run: {combo}")
    return {
        "n_manifest_runs": len(manifest["runs"]),
        "n_completed": n_completed,
        "issues": issues,
        "missing_combinations": missing,
        "duplicate_run_ids": len(run_ids) != len(manifest["runs"]),
        "valid": len(issues) == 0,
    }
