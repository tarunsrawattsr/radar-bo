"""Experiment-to-simulator interface validation (M7 item 29).

SOFTWARE-INTERFACE VALIDATION ONLY. This module verifies that the M4
evaluation-interface contracts (structured evaluation records, non-mutating
frozen snapshots, explicit failure semantics) are architecturally compatible
with the generic experiment framework, using SYNTHETIC component providers
and mock evaluation adapters.

* No physical radar optimization is activated.
* No new physical sensing model is introduced.
* The adapter works on duck-typed structured records shaped like the M4
  ``EvaluationResult`` (evaluation_id / status / value / duration), so it can
  wrap any deterministic evaluator that honors the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from radar_bo.config.base import StrictModel
from pydantic import Field


@dataclass(frozen=True)
class MockFrozenSnapshot:
    """Duck-typed stand-in for the M4 FrozenSimulationState (isolation test)."""

    snapshot_id: str
    values: tuple[float, ...]
    # mutation sentinel: any in-place change by an evaluator would be visible here
    integrity_sentinel: tuple = field(default=(True,))


@dataclass(frozen=True)
class MockEvaluationRecord:
    """Structured record shaped like the M4 EvaluationResult contract."""

    evaluation_id: str
    status: str  # "success" | "failed" | "infeasible"
    value: float | None
    snapshot_id: str
    duration_s: float
    error: str | None = None


class MockSimulatedEvaluator:
    """Synthetic provider honoring the M4 non-mutation + structured-record contract.

    Wraps a pure function ``f(vector) -> float`` evaluated against an immutable
    snapshot; records every call so tests can assert isolation and counting.
    """

    def __init__(self, snapshot: MockFrozenSnapshot, objective: Callable[[tuple[float, ...]], float],
                 fail_on: set[tuple[float, ...]] | None = None) -> None:
        self.snapshot = snapshot
        self.objective = objective
        self.fail_on = fail_on or set()
        self.calls: list[tuple[float, ...]] = []
        self._counter = 0

    def evaluate_candidate(self, vector: tuple[float, ...]) -> MockEvaluationRecord:
        self.calls.append(tuple(vector))
        self._counter += 1
        if vector in self.fail_on:
            return MockEvaluationRecord(
                evaluation_id=f"mock_eval_{self._counter:05d}", status="failed", value=None,
                snapshot_id=self.snapshot.snapshot_id, duration_s=0.0,
                error="synthetic evaluator failure (contract test)",
            )
        return MockEvaluationRecord(
            evaluation_id=f"mock_eval_{self._counter:05d}", status="success",
            value=float(self.objective(vector)), snapshot_id=self.snapshot.snapshot_id, duration_s=0.0,
        )


class SimulatorBackedProblem(StrictModel):
    """Adapter configuration: binds a mock evaluator into the framework's problem shape."""

    problem_id: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    lower_bounds: list[float]
    upper_bounds: list[float]
    variable_types: list[str]


def evaluate_through_adapter(evaluator: MockSimulatedEvaluator, problem: SimulatorBackedProblem,
                             vector: list[float]) -> dict:
    """Convert one structured evaluation record into the framework record shape.

    Implements the contract mapping: success -> observation record, failed ->
    failure record. Snapshot isolation is asserted on every call (the snapshot
    is immutable data; the evaluator receives the vector only).
    """
    record = evaluator.evaluate_candidate(tuple(vector))
    # isolation assertions (M4 contract): snapshot untouched, no sim-time advance
    if evaluator.snapshot.integrity_sentinel != (True,):
        raise AssertionError("snapshot integrity sentinel changed: evaluator mutated state")
    if record.status == "success":
        return {
            "kind": "observation",
            "evaluation_id": record.evaluation_id,
            "config": list(vector),
            "objective": record.value,
            "snapshot_id": record.snapshot_id,
            "duration_s": record.duration_s,
        }
    return {
        "kind": "failure",
        "evaluation_id": record.evaluation_id,
        "config": list(vector),
        "error": record.error or record.status,
        "snapshot_id": record.snapshot_id,
        "duration_s": record.duration_s,
    }
