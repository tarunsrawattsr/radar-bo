"""BO observation dataset (M5 instructions 8 and 9).

Each SUCCESSFUL observation carries: the complete decision vector, the
observed scalar objective, an evaluation identifier, the problem identifier,
random-seed information, observation-noise information and metadata. Failed
or infeasible evaluations NEVER enter the successful-observation dataset;
they are kept in a separate failure log.

Training-tensor contract:

* ``training_inputs()`` -> float64 array of shape ``M x d``;
* ``training_outputs()`` -> float64 array of shape ``M x 1``
  (M = number of successful observations, d = full decision-vector
  dimension). Fresh copies are returned (mutation protection).

Duplicate policy (instruction 9): accidental duplicates of already-observed
configurations are REJECTED on insert unless ``allow_duplicates=True``
(intentional repeated measurements, legitimate for noisy objectives).
Tolerance: max-abs difference <= 1e-9 (``DUPLICATE_TOLERANCE``). Distinct
configurations are never merged.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel

DUPLICATE_TOLERANCE: float = 1e-9


class DuplicateObservationError(ValueError):
    """An insert would duplicate an already-observed configuration (rejected,
    not merged)."""


class ObservationRecord(StrictModel):
    """One successful objective observation."""

    evaluation_id: str
    problem_id: str
    config: list[float] = Field(description="Complete decision vector in problem variable order.")
    objective: float
    seed_info: str | None = Field(default=None, description="Random-seed information where applicable.")
    observation_noise_std: float | None = Field(
        default=None, description="Declared observation-noise std (None = deterministic observations)."
    )
    metadata: dict[str, str] = Field(default_factory=dict)


class FailureRecord(StrictModel):
    """One failed or infeasible evaluation attempt (kept OUT of training data)."""

    evaluation_id: str
    problem_id: str
    config: list[float] | None = Field(default=None, description="Config if one existed at failure time.")
    error: str
    stage: str = Field(description="Where it failed: 'objective_evaluation', 'initial_design', ...")
    seed_info: str | None = None


class ObservationDataset(StrictModel):
    """Successful-observation dataset with a separate failure log."""

    problem_id: str
    dimension: int = Field(ge=1)
    maximize: bool = True
    records: list[ObservationRecord] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)

    # ---- Insertion ---------------------------------------------------------------
    def insert(
        self,
        config: np.ndarray,
        objective: float,
        evaluation_id: str,
        seed_info: str | None = None,
        observation_noise_std: float | None = None,
        metadata: dict[str, str] | None = None,
        allow_duplicates: bool = False,
    ) -> ObservationRecord:
        """Append a successful observation (validated; duplicates rejected).

        ``allow_duplicates=True`` marks an INTENTIONAL repeated measurement
        (noisy-objective design); accidental duplicates are always rejected.
        """
        arr = np.asarray(config, dtype=np.float64).flatten()
        if arr.shape != (self.dimension,):
            raise ValueError(f"config must have {self.dimension} entries; got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"config contains non-finite values: {arr!r}.")
        if not np.isfinite(objective):
            raise ValueError(f"objective must be finite; got {objective!r}.")
        if not allow_duplicates:
            duplicate_index = self.find_duplicate_index(arr)
            if duplicate_index is not None:
                raise DuplicateObservationError(
                    f"config {arr.tolist()} duplicates observation {duplicate_index} "
                    f"(tolerance {DUPLICATE_TOLERANCE}); intentional repeats require "
                    "allow_duplicates=True (noisy objectives only)."
                )
        record = ObservationRecord(
            evaluation_id=evaluation_id,
            problem_id=self.problem_id,
            config=[float(v) for v in arr],
            objective=float(objective),
            seed_info=seed_info,
            observation_noise_std=observation_noise_std,
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    def record_failure(
        self, evaluation_id: str, error: str, stage: str, config: np.ndarray | None = None, seed_info: str | None = None
    ) -> None:
        """Append to the SEPARATE failure log (never enters training data)."""
        self.failures.append(
            FailureRecord(
                evaluation_id=evaluation_id,
                problem_id=self.problem_id,
                config=None if config is None else [float(v) for v in np.asarray(config).flatten()],
                error=error,
                stage=stage,
                seed_info=seed_info,
            )
        )

    # ---- Queries -------------------------------------------------------------------
    def find_duplicate_index(self, config: np.ndarray) -> int | None:
        """Index of the first observed configuration within duplicate tolerance."""
        arr = np.asarray(config, dtype=np.float64).flatten()
        for i, record in enumerate(self.records):
            if float(np.max(np.abs(arr - np.asarray(record.config, dtype=np.float64)))) <= DUPLICATE_TOLERANCE:
                return i
        return None

    def has_config(self, config: np.ndarray) -> bool:
        return self.find_duplicate_index(config) is not None

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def objectives(self) -> np.ndarray:
        return np.asarray([r.objective for r in self.records], dtype=np.float64)

    def configs(self) -> np.ndarray:
        """(M, d) array of observed configurations (fresh copy)."""
        if not self.records:
            return np.zeros((0, self.dimension), dtype=np.float64)
        return np.asarray([r.config for r in self.records], dtype=np.float64).copy()

    def best_index(self) -> int:
        if not self.records:
            raise ValueError("The dataset is empty; no best observation exists.")
        values = self.objectives()
        return int(np.argmax(values)) if self.maximize else int(np.argmin(values))

    def best(self) -> tuple[int, ObservationRecord]:
        idx = self.best_index()
        return idx, self.records[idx]

    # ---- Training-tensor contract (M x d / M x 1) -------------------------------------
    def training_inputs(self) -> np.ndarray:
        """float64 array of shape (M, d); fresh copy."""
        if not self.records:
            return np.zeros((0, self.dimension), dtype=np.float64)
        return np.asarray([r.config for r in self.records], dtype=np.float64).copy()

    def training_outputs(self) -> np.ndarray:
        """float64 array of shape (M, 1); fresh copy."""
        return self.objectives().reshape(-1, 1).copy()

    # ---- Validation --------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_records(self) -> "ObservationDataset":
        for record in self.records:
            if len(record.config) != self.dimension:
                raise ValueError(
                    f"Observation {record.evaluation_id} has {len(record.config)} entries; "
                    f"dataset dimension is {self.dimension}."
                )
        return self
