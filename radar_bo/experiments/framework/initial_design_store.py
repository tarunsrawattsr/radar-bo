"""Authoritative shared initial designs (M8A-R1 Phase B).

One PERSISTED initial design per compatible comparison group
``(benchmark_id, dimension, seed)``. The design file is the single
authoritative artifact:

* created once per experiment from a FRESH problem instance via the
  registry's deterministic sampler (explicit_shared mode);
* fingerprinted with a GENUINE SHA-256 digest over the canonical JSON of
  the design rows (the digest is recomputed from the stored rows at every
  load — a mismatch is an explicit rejection, never a silent regeneration);
* reloaded unchanged when resuming an experiment (deterministic
  regeneration is NOT used to "repair" a stored design);
* validated on every load: shape/size, finiteness, feasibility against a
  fresh problem instance, and within-design duplicates.

Malformed, duplicate, incomplete and inconsistent designs are detected and
REJECTED with :class:`DesignValidationError` (M8A-R1 requirement). The
historical ``seeded_sampler`` mode does NOT touch this store.

File layout inside the experiment output directory::

    initial_designs/<design_id>.json   design_id = sha256(group key)[:16]

The stored JSON contains the group key, the exact rows, the genuine
SHA-256 over the rows, generation metadata (mode/sampler/seed/size,
problem reference) and the source commit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from radar_bo.experiments.framework.checkpointing import RecordRejected, atomic_write_json
from radar_bo.optimization.bo.candidates import CandidateGenerator
from radar_bo.optimization.bo.dataset import DUPLICATE_TOLERANCE


class DesignValidationError(RecordRejected):
    """An initial design is malformed, duplicate, incomplete or inconsistent."""


def canonical_design_json(rows: list[list[float]]) -> str:
    """Canonical JSON of the design rows (deterministic float formatting)."""
    return json.dumps([[float(v) for v in row] for row in rows],
                      separators=(",", ":"), sort_keys=False)


def design_fingerprint(rows: list[list[float]]) -> str:
    """Genuine SHA-256 over the canonical JSON of the rows (hex digest)."""
    return hashlib.sha256(canonical_design_json(rows).encode("utf-8")).hexdigest()


def design_id_for(group_key: tuple) -> str:
    """Stable short id for a (benchmark_id, dimension, seed) group."""
    key_json = json.dumps(list(group_key), separators=(",", ":"))
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]


def validate_design_rows(
    rows: list[list[float]],
    problem,
    expected_size: int,
    *,
    check_feasible: bool = True,
) -> list[str]:
    """Detect malformed, incomplete, infeasible and duplicate designs.

    Returns the list of issue strings (empty = valid). Raises nothing here;
    callers decide whether to raise :class:`DesignValidationError`.
    """
    issues: list[str] = []
    if not isinstance(rows, list) or not rows:
        return [f"design is empty or not a list (got {type(rows).__name__})."]
    if len(rows) != expected_size:
        issues.append(f"incomplete design: {len(rows)} rows; expected {expected_size}.")
    n_cols = None
    for i, row in enumerate(rows):
        try:
            arr = np.asarray(row, dtype=np.float64).flatten()
        except (TypeError, ValueError) as exc:
            issues.append(f"malformed row {i}: cannot convert to float64 ({exc}).")
            continue
        if arr.shape[0] != problem.dimension:
            issues.append(
                f"malformed row {i}: dimension {arr.shape[0]}; problem dimension is {problem.dimension}."
            )
            continue
        if not np.all(np.isfinite(arr)):
            issues.append(f"malformed row {i}: contains non-finite values.")
            continue
        if check_feasible and not problem.is_feasible(arr):
            issues.append(f"infeasible row {i} (explicit shared designs are validated, never repaired).")
        if n_cols is None:
            n_cols = arr.shape[0]
        # duplicate detection against EARLIER rows (tolerance = engine duplicate tolerance)
        for j in range(i):
            try:
                prev = np.asarray(rows[j], dtype=np.float64).flatten()
            except (TypeError, ValueError):
                continue  # already reported as malformed
            if prev.shape[0] == arr.shape[0] and float(np.max(np.abs(prev - arr))) <= DUPLICATE_TOLERANCE:
                issues.append(f"duplicate rows {j} and {i} (max-abs diff <= {DUPLICATE_TOLERANCE}).")
                break
    return issues


class InitialDesignStore:
    """Persists and reloads one authoritative design per comparison group."""

    def __init__(self, output_dir: str | Path) -> None:
        self.dir = Path(output_dir) / "initial_designs"

    def _path(self, group_key: tuple) -> Path:
        return self.dir / f"{design_id_for(group_key)}.json"

    def load(self, group_key: tuple, problem, expected_size: int) -> tuple[list[list[float]], str]:
        """Load a stored design; REJECT on any inconsistency (never regenerate)."""
        path = self._path(group_key)
        if not path.exists():
            raise FileNotFoundError(f"No stored design for group {group_key} at {path}.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecordRejected(f"Corrupted design JSON for group {group_key}: {exc}") from exc
        rows = payload.get("rows")
        stored_fp = payload.get("design_sha256")
        if not isinstance(rows, list):
            raise DesignValidationError(f"Stored design for group {group_key} has no 'rows' list.")
        issues = validate_design_rows(rows, problem, expected_size)
        if issues:
            raise DesignValidationError(
                f"Stored design for group {group_key} failed validation: " + "; ".join(issues)
            )
        recomputed = design_fingerprint(rows)
        if stored_fp != recomputed:
            raise DesignValidationError(
                f"Stored design fingerprint mismatch for group {group_key}: "
                f"stored {str(stored_fp)[:16]}, recomputed from rows {recomputed[:16]} "
                "(the file content and its digest are inconsistent)."
            )
        if tuple(payload.get("group_key") or ()) != tuple(group_key):
            raise DesignValidationError(
                f"Stored design group_key {payload.get('group_key')} does not match requested {group_key}."
            )
        return [[float(v) for v in row] for row in rows], stored_fp

    def create(self, group_key: tuple, problem, size: int, seed: int, metadata: dict) -> tuple[list[list[float]], str]:
        """Generate, validate and PERSIST one authoritative design."""
        pool, _report = CandidateGenerator(
            problem, num_candidates=size, seed=seed, exclude_evaluated=True,
        ).generate_pool()
        if pool.shape[0] < size:
            raise RuntimeError(
                f"explicit_shared initial design could not be generated for "
                f"benchmark '{group_key[0]}' d={group_key[1]} seed={group_key[2]}: "
                f"{pool.shape[0]} feasible points found, {size} required "
                "(the feasible region is too small for the requested design size)."
            )
        rows = [[float(v) for v in pool[i]] for i in range(size)]
        issues = validate_design_rows(rows, problem, size)
        if issues:  # defensive: the sampler must deliver distinct feasible rows
            raise DesignValidationError(
                f"Freshly generated design for group {group_key} failed validation: " + "; ".join(issues)
            )
        fingerprint = design_fingerprint(rows)
        payload = {
            "group_key": list(group_key),
            "rows": rows,
            "design_sha256": fingerprint,
            "metadata": {
                **metadata,
                "size": size,
                "seed": seed,
                "problem_id": problem.problem_id,
            },
        }
        atomic_write_json(self._path(group_key), payload)
        return rows, fingerprint

    def load_or_create(self, group_key: tuple, problem, size: int, seed: int, metadata: dict) -> tuple[list[list[float]], str, bool]:
        """Authoritative accessor: reload the persisted design if present
        (validated against its stored SHA-256), otherwise create and persist it.
        Returns (rows, fingerprint, created_now)."""
        try:
            rows, fp = self.load(group_key, problem, size)
            return rows, fp, False
        except FileNotFoundError:
            rows, fp = self.create(group_key, problem, size, seed, metadata)
            return rows, fp, True
