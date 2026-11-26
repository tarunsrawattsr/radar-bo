"""Objective evaluator — Lu et al., ICC 2024, Eq. (23) aggregation structure.

Paper-reported expression (authoritative source: supplied PDF):

* Eq. (23): ``U_t = - sum_n w_n c^n_t + beta * Gamma(tau_s,t)`` — utility as
  the negative weighted aggregate tracking cost plus the trade-off
  coefficient times scanning performance. ``beta`` is the paper's 1e6
  (preserved for reproduction only; see ResourceConfig); per-task weights
  ``w_n`` are PROPOSED uniform (1.0) when
  ``ResourceConfig.task_utility_weights`` is None.

M4 interface specification (instructions 12 and 14):

* The evaluator receives its component performance quantities through
  EXPLICIT interfaces (:class:`TrackingCostProvider`,
  :class:`ScanningPerformanceModel`) — it contains NO GP training, NO
  acquisition optimization and NO BO candidate generation.
* :meth:`ObjectiveEvaluator.evaluate_candidate` accepts a FROZEN simulation
  state, a candidate allocation and (implicitly, through the snapshot's
  captured generator state) the experiment's random-number source. It returns
  a structured :class:`EvaluationResult` WITHOUT committing the candidate as
  an actual action and WITHOUT advancing physical simulation time.
* Randomness policy (instruction 15): the declared default is COMMON RANDOM
  NUMBERS — the generator is rebuilt from the snapshot's captured state on
  every evaluation, so all candidates see identical random realizations and
  evaluation order cannot change outcomes. The policy is recorded in every
  result.
* Failed evaluations are represented EXPLICITLY (status ``failed`` with the
  error, utility ``None``); a utility of zero is never substituted. Infeasible
  candidates get status ``infeasible`` (also never a zero utility): they are
  not admissible objective observations.
* The snapshot's content hash is re-verified after evaluation and the outcome
  is recorded (``snapshot_integrity_verified``); a detected mutation fails the
  evaluation loudly.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.optimization.allocation import ResourceAllocation
from radar_bo.optimization.constraints import ConstraintManager, FeasibilityReport
from radar_bo.optimization.resource_accounting import ResourceAccountant, ResourceUsage
from radar_bo.radar.scanning_model import ScanningPerformanceModel
from radar_bo.radar.tracking_cost import position_tracking_cost
from radar_bo.simulation.simulation_state import FrozenSimulationState

RandomnessPolicy = Literal["common_random_numbers", "independent_realizations", "deterministic"]


@runtime_checkable
class TrackingCostProvider(Protocol):
    """Explicit interface supplying per-task tracking costs (Eq. (23) c^n_t).

    Implementations must be deterministic for identical (snapshot, allocation,
    generator) inputs unless randomness is explicitly declared. The generator
    is rebuilt from the snapshot's captured state (common random numbers) and
    passed for future stochastic providers; deterministic providers ignore it.
    """

    identifier: str
    is_synthetic: bool

    def tracking_costs(
        self, snapshot: FrozenSimulationState, allocation: ResourceAllocation, rng: np.random.Generator
    ) -> np.ndarray: ...


class SnapshotDiagnosticTrackingCostProvider:
    """Paper Eq. (14) tracking cost on the snapshot's posterior covariances.

    DIAGNOSTIC and ALLOCATION-INDEPENDENT: it returns the snapshot's current
    per-task position-tracking costs. A candidate's effect on FUTURE tracking
    requires simulation advancement (M5+) and is deliberately out of M4 scope.
    """

    identifier: str = "snapshot_diagnostic_eq14"
    is_synthetic: bool = False

    def tracking_costs(
        self, snapshot: FrozenSimulationState, allocation: ResourceAllocation, rng: np.random.Generator
    ) -> np.ndarray:
        if allocation.num_tasks != snapshot.num_tasks():
            raise ValueError(
                f"Allocation has {allocation.num_tasks} tasks but the snapshot has {snapshot.num_tasks()}."
            )
        return np.asarray(
            [position_tracking_cost(snapshot.posterior_covariance(k)) for k in range(snapshot.num_tasks())],
            dtype=np.float64,
        )


class SyntheticLinearTrackingCostProvider:
    """SYNTHETIC allocation-dependent tracking-cost response (testing only).

    Response (documented, independently verifiable)::

        c_n(tau_n) = base_n - kappa_n * (tau_n / tau_ref_s)

    with finite ``base_n >= 0``, ``kappa_n >= 0`` and ``tau_ref_s > 0``.
    Clearly labeled synthetic: NOT a validated tracking-performance model.
    """

    identifier: str = "synthetic_linear_tracking_v1"
    is_synthetic: bool = True

    def __init__(
        self,
        base_costs: np.ndarray,
        kappa_per_task: np.ndarray,
        reference_dwell_s: float = 1.0,
    ) -> None:
        base = np.asarray(base_costs, dtype=np.float64)
        kappa = np.asarray(kappa_per_task, dtype=np.float64)
        if base.ndim != 1 or kappa.ndim != 1 or base.shape != kappa.shape:
            raise ValueError(f"base_costs and kappa_per_task must be equal-shaped 1-D arrays; got {base.shape}, {kappa.shape}.")
        if not np.all(np.isfinite(base)) or np.any(base < 0.0):
            raise ValueError(f"base_costs must be finite and >= 0; got {base_costs!r}.")
        if not np.all(np.isfinite(kappa)) or np.any(kappa < 0.0):
            raise ValueError(f"kappa_per_task must be finite and >= 0; got {kappa_per_task!r}.")
        if not np.isfinite(reference_dwell_s) or reference_dwell_s <= 0.0:
            raise ValueError(f"reference_dwell_s must be finite and > 0; got {reference_dwell_s!r}.")
        self.base_costs = base
        self.kappa_per_task = kappa
        self.reference_dwell_s = float(reference_dwell_s)

    def tracking_costs(
        self, snapshot: FrozenSimulationState, allocation: ResourceAllocation, rng: np.random.Generator
    ) -> np.ndarray:
        if allocation.num_tasks != self.base_costs.shape[0]:
            raise ValueError(
                f"Allocation has {allocation.num_tasks} tasks; provider configured for {self.base_costs.shape[0]}."
            )
        taus = np.asarray(allocation.dwell_times_s, dtype=np.float64)
        costs = self.base_costs - self.kappa_per_task * (taus / self.reference_dwell_s)
        if not np.all(np.isfinite(costs)):
            raise ValueError(f"Synthetic tracking costs are non-finite: {costs!r}.")
        return costs


class EvaluationResult(StrictModel):
    """Structured result of one candidate evaluation (instruction 12)."""

    evaluation_id: str
    status: Literal["success", "infeasible", "failed"]
    utility: float | None = None
    per_task_tracking_costs: list[float] | None = None
    task_utility_weights: list[float] | None = None
    aggregate_weighted_tracking_cost: float | None = None
    scanning_gamma: float | None = None
    detectable_area_m2: float | None = None
    scanning_model_identifier: str | None = None
    scanning_is_synthetic: bool | None = None
    resource_usage: ResourceUsage | None = None
    feasibility: FeasibilityReport
    snapshot_id: str
    allocation_id: str
    tracking_provider_identifier: str
    randomness_policy: RandomnessPolicy
    evaluation_duration_s: float
    snapshot_integrity_verified: bool
    error: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> "EvaluationResult":
        if self.status == "success":
            required = (self.utility, self.per_task_tracking_costs, self.aggregate_weighted_tracking_cost,
                        self.scanning_gamma, self.detectable_area_m2, self.resource_usage)
            if any(v is None for v in required):
                raise ValueError("A successful evaluation must carry utility, costs, scanning performance and usage.")
            if not np.isfinite(self.utility):
                raise ValueError(f"Utility must be finite; got {self.utility!r}.")
            if self.error is not None:
                raise ValueError("A successful evaluation must not carry an error.")
        elif self.status == "infeasible":
            if self.utility is not None:
                raise ValueError("An infeasible evaluation must NOT carry a utility (it is not an objective observation).")
        else:  # failed
            if self.utility is not None:
                raise ValueError("A failed evaluation must NOT carry a utility.")
            if self.error is None:
                raise ValueError("A failed evaluation must record the error.")
        if self.evaluation_duration_s < 0.0:
            raise ValueError("evaluation_duration_s must be >= 0.")
        return self


class ObjectiveEvaluator:
    """Eq. (23) aggregation over explicit component interfaces; non-mutating."""

    def __init__(
        self,
        constraint_manager: ConstraintManager,
        accountant: ResourceAccountant,
        tracking_provider: TrackingCostProvider,
        scanning_model: ScanningPerformanceModel,
        resource_config: ResourceConfig,
        randomness_policy: RandomnessPolicy = "common_random_numbers",
    ) -> None:
        self.constraint_manager = constraint_manager
        self.accountant = accountant
        self.tracking_provider = tracking_provider
        self.scanning_model = scanning_model
        self.resource_config = resource_config
        self.randomness_policy = randomness_policy

    # ---- Weights (proposed uniform default when unconfigured) --------------------
    def _task_weights(self, num_tasks: int) -> np.ndarray:
        configured = self.resource_config.task_utility_weights
        if configured is None:
            return np.ones(num_tasks, dtype=np.float64)  # PROPOSED uniform default
        weights = np.asarray(configured, dtype=np.float64)
        if weights.shape != (num_tasks,):
            raise ValueError(f"task_utility_weights has {weights.shape[0]} entries; expected {num_tasks}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"task_utility_weights must be finite and >= 0; got {configured!r}.")
        return weights

    def _failed(self, snapshot, allocation, feasibility, started, error, integrity=True) -> EvaluationResult:
        return EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:16],
            status="failed",
            utility=None,
            feasibility=feasibility,
            snapshot_id=snapshot.snapshot_id,
            allocation_id=allocation.content_id(),
            tracking_provider_identifier=self.tracking_provider.identifier,
            randomness_policy=self.randomness_policy,
            evaluation_duration_s=time.perf_counter() - started,
            snapshot_integrity_verified=integrity,
            error=error,
        )

    # ---- Candidate evaluation ------------------------------------------------------
    def evaluate_candidate(
        self,
        snapshot: FrozenSimulationState,
        allocation: ResourceAllocation,
    ) -> EvaluationResult:
        """Evaluate one candidate against a frozen state without committing it.

        Non-mutating contract: the snapshot's arrays, physical time and the
        authoritative simulation state are untouched; the returned result is a
        pure record. The candidate is NOT applied as an actual action.
        """
        started = time.perf_counter()
        feasibility = self.constraint_manager.check_allocation(allocation)
        try:
            usage = self.accountant.account(allocation)
        except (ValueError, TypeError) as exc:
            return self._failed(snapshot, allocation, feasibility, started, f"Resource accounting failed: {exc}")

        if not feasibility.is_feasible:
            return EvaluationResult(
                evaluation_id=uuid.uuid4().hex[:16],
                status="infeasible",
                utility=None,
                resource_usage=usage,
                feasibility=feasibility,
                snapshot_id=snapshot.snapshot_id,
                allocation_id=allocation.content_id(),
                tracking_provider_identifier=self.tracking_provider.identifier,
                randomness_policy=self.randomness_policy,
                evaluation_duration_s=time.perf_counter() - started,
                snapshot_integrity_verified=snapshot.verify_integrity(),
            )

        try:
            rng = snapshot.rebuild_rng()  # common random numbers from the captured state
            costs = np.asarray(
                self.tracking_provider.tracking_costs(snapshot, allocation, rng), dtype=np.float64
            )
            if costs.shape != (allocation.num_tasks,) or not np.all(np.isfinite(costs)):
                raise ValueError(f"Tracking provider returned invalid costs: {costs!r}.")
            scanning = self.scanning_model.evaluate(usage.remaining_scanning_time_s_paper_eq24)
            weights = self._task_weights(allocation.num_tasks)
            aggregate = float(np.dot(weights, costs))
            beta = self.resource_config.tracking_scanning_tradeoff
            utility = -aggregate + beta * scanning.gamma
            if not np.isfinite(utility) or not np.isfinite(aggregate) or not np.isfinite(scanning.gamma):
                raise ValueError("Non-finite utility components; evaluation rejected rather than returned.")
        except Exception as exc:  # any component failure must yield an explicit FAILED record
            return self._failed(snapshot, allocation, feasibility, started, f"Component evaluation failed: {exc}")

        integrity_ok = snapshot.verify_integrity()
        if not integrity_ok:
            return self._failed(
                snapshot, allocation, feasibility, started,
                "Snapshot integrity verification FAILED after evaluation: the frozen state was mutated "
                "during candidate evaluation (non-mutation contract violated).",
                integrity=False,
            )

        return EvaluationResult(
            evaluation_id=uuid.uuid4().hex[:16],
            status="success",
            utility=float(utility),
            per_task_tracking_costs=[float(c) for c in costs],
            task_utility_weights=[float(w) for w in weights],
            aggregate_weighted_tracking_cost=aggregate,
            scanning_gamma=float(scanning.gamma),
            detectable_area_m2=float(scanning.detectable_area_m2),
            scanning_model_identifier=scanning.model_identifier,
            scanning_is_synthetic=bool(scanning.is_synthetic),
            resource_usage=usage,
            feasibility=feasibility,
            snapshot_id=snapshot.snapshot_id,
            allocation_id=allocation.content_id(),
            tracking_provider_identifier=self.tracking_provider.identifier,
            randomness_policy=self.randomness_policy,
            evaluation_duration_s=time.perf_counter() - started,
            snapshot_integrity_verified=integrity_ok,
        )
