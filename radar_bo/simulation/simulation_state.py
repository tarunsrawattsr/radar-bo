"""Simulation-state and frozen evaluation-snapshot representation (M4 instruction 13).

Authoritative state vs snapshot:

* :class:`SimulationState` — the authoritative, mutable simulation state:
  physical simulation time, revisit index, ordered task identifiers, TRUE
  simulated states (simulation/evaluation layer only), estimator posteriors
  and the experimental random generator.
* :class:`FrozenSimulationState` — an immutable estimator-facing snapshot for
  independent hypothetical (candidate) evaluations.

Anti-leakage design: the snapshot carries ONLY estimator-facing information
(posterior means/covariances, time, task ids, captured generator state). The
true simulated states are deliberately NOT copied into the snapshot, so no
candidate-evaluation component can receive ground truth through the snapshot
interface; truth lives in the authoritative SimulationState for simulation and
evaluation bookkeeping.

Immutability and independence:

* Snapshot arrays are stored privately; every accessor returns a fresh copy
  (mutation protection, tested).
* ``snapshot_id`` is a sha256 content hash; ``verify_integrity()`` recomputes
  it so accidental mutation is detectable. A candidate evaluation must not
  permanently alter its input snapshot or the authoritative state.
* The generator state is CAPTURED at snapshot time; ``rebuild_rng()``
  constructs a FRESH generator from the captured state on every call. Under
  the declared COMMON RANDOM NUMBERS policy every candidate evaluation rebuilt
  from the same snapshot therefore sees identical random realizations,
  independent of evaluation order (tested). An independent-realizations policy
  would instead draw fresh entropy per evaluation; M4 defaults to CRN and
  records the policy in every evaluation result.

Physical simulation time is preserved: candidate evaluation never advances the
clock (time advancement belongs to the future simulation runner, M5+).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import numpy as np

from radar_bo.environment.target_state import GaussianState, STATE_DIMENSION, TargetState


def _content_digest(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSimulationState:
    """Immutable, estimator-facing snapshot for candidate evaluations."""

    scenario_id: str
    physical_time_s: float
    revisit_index: int
    task_ids: tuple[str, ...]
    rng_state: dict
    snapshot_id: str = ""
    _posterior_means: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    _posterior_covariances: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.task_ids)
        if n < 1:
            raise ValueError("A snapshot must contain at least one task.")
        if not np.isfinite(self.physical_time_s) or self.physical_time_s < 0.0:
            raise ValueError(f"physical_time_s must be finite and >= 0; got {self.physical_time_s!r}.")
        if self.revisit_index < 0:
            raise ValueError(f"revisit_index must be >= 0; got {self.revisit_index}.")
        means = np.asarray(self._posterior_means, dtype=np.float64)
        covs = np.asarray(self._posterior_covariances, dtype=np.float64)
        if means.shape != (n, STATE_DIMENSION):
            raise ValueError(f"posterior_means must have shape ({n}, {STATE_DIMENSION}); got {means.shape}.")
        if covs.shape != (n, STATE_DIMENSION, STATE_DIMENSION):
            raise ValueError(
                f"posterior_covariances must have shape ({n}, {STATE_DIMENSION}, {STATE_DIMENSION}); got {covs.shape}."
            )
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covs)):
            raise ValueError("Snapshot posterior arrays contain non-finite values.")
        for k in range(n):
            cov = covs[k]
            if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10) or float(np.linalg.eigvalsh(cov).min()) < -1e-10:
                raise ValueError(f"Posterior covariance for {self.task_ids[k]} is not symmetric PSD.")
        object.__setattr__(self, "_posterior_means", means)
        object.__setattr__(self, "_posterior_covariances", covs)
        if not self.snapshot_id:
            object.__setattr__(self, "snapshot_id", self._compute_id())

    # ---- Estimator-facing accessors (fresh copies every call) -------------------
    def num_tasks(self) -> int:
        return len(self.task_ids)

    def posterior_mean(self, task_index: int) -> np.ndarray:
        """Fresh copy of one task's posterior mean (4,)."""
        if not (0 <= task_index < len(self.task_ids)):
            raise IndexError(f"task_index {task_index} out of range for {len(self.task_ids)} tasks.")
        return self._posterior_means[task_index].copy()

    def posterior_covariance(self, task_index: int) -> np.ndarray:
        """Fresh copy of one task's posterior covariance (4, 4)."""
        if not (0 <= task_index < len(self.task_ids)):
            raise IndexError(f"task_index {task_index} out of range for {len(self.task_ids)} tasks.")
        return self._posterior_covariances[task_index].copy()

    def all_posterior_covariances(self) -> np.ndarray:
        """Fresh stacked copy of all posterior covariances (n, 4, 4)."""
        return self._posterior_covariances.copy()

    def rebuild_rng(self) -> np.random.Generator:
        """Fresh generator rebuilt from the CAPTURED state (common random numbers).

        Every call returns an independent generator producing the SAME stream,
        so candidate evaluations are order-invariant under the CRN policy.
        """
        bit_generator = getattr(np.random, self.rng_state["bit_generator_name"])()
        bit_generator.state = copy.deepcopy(self.rng_state["state"])
        return np.random.Generator(bit_generator)

    # ---- Integrity ---------------------------------------------------------------
    def _compute_id(self) -> str:
        rng_repr = repr(sorted(self.rng_state["state"].items())).encode("utf-8")
        return _content_digest(
            self.scenario_id.encode("utf-8"),
            np.asarray([self.physical_time_s], dtype=np.float64).tobytes(),
            np.asarray([self.revisit_index], dtype=np.int64).tobytes(),
            ";".join(self.task_ids).encode("utf-8"),
            self._posterior_means.tobytes(),
            self._posterior_covariances.tobytes(),
            rng_repr,
        )[:32]

    def verify_integrity(self) -> bool:
        """True iff the snapshot content still matches the construction-time hash."""
        return self._compute_id() == self.snapshot_id


class SimulationState:
    """Authoritative mutable simulation state (truth stays here, not in snapshots)."""

    def __init__(
        self,
        scenario_id: str,
        physical_time_s: float,
        revisit_index: int,
        task_ids: tuple[str, ...],
        targets: tuple[TargetState, ...],
        posteriors: tuple[GaussianState, ...],
        rng: np.random.Generator,
    ) -> None:
        if not (len(task_ids) == len(targets) == len(posteriors)) or len(task_ids) < 1:
            raise ValueError("task_ids, targets and posteriors must be non-empty and of equal length.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an explicit numpy.random.Generator (no global random state).")
        self.scenario_id = scenario_id
        self.physical_time_s = float(physical_time_s)
        self.revisit_index = int(revisit_index)
        self.task_ids = tuple(task_ids)
        self.targets = tuple(targets)  # TRUE states: simulation/evaluation layer only
        self.posteriors = tuple(posteriors)
        self.rng = rng

    def capture_rng_state(self) -> dict:
        """Deep-copied capture of the generator state (frozen into snapshots)."""
        return {
            "bit_generator_name": type(self.rng.bit_generator).__name__,
            "state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def snapshot(self) -> FrozenSimulationState:
        """Frozen estimator-facing snapshot (truth deliberately excluded)."""
        means = np.stack([p.mean for p in self.posteriors]).astype(np.float64)
        covs = np.stack([p.covariance for p in self.posteriors]).astype(np.float64)
        base = FrozenSimulationState(
            scenario_id=self.scenario_id,
            physical_time_s=self.physical_time_s,
            revisit_index=self.revisit_index,
            task_ids=self.task_ids,
            rng_state=self.capture_rng_state(),
            _posterior_means=means,
            _posterior_covariances=covs,
        )
        return base

    def fingerprint(self) -> str:
        """Content fingerprint of the AUTHORITATIVE state (truth included) for
        preservation checks by tests/experiments."""
        truth_bytes = b"".join(t.vector.tobytes() for t in self.targets)
        posterior_bytes = b"".join(p.mean.tobytes() + p.covariance.tobytes() for p in self.posteriors)
        rng_repr = repr(sorted(self.rng.bit_generator.state.items())).encode("utf-8")
        return _content_digest(
            np.asarray([self.physical_time_s], dtype=np.float64).tobytes(),
            np.asarray([self.revisit_index], dtype=np.int64).tobytes(),
            truth_bytes,
            posterior_bytes,
            rng_repr,
        )[:32]

    def advance(self, dt: float) -> None:
        """Advance the physical clock (simulation runner's responsibility, NOT the
        candidate evaluator's). Provided for completeness; M4 candidate
        evaluation never calls it."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0; got {dt!r}.")
        self.physical_time_s += float(dt)
        self.revisit_index += 1
