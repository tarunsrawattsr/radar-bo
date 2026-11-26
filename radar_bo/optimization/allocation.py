"""Structured resource-allocation representation (M4 instruction 8).

A complete allocation over a fixed number of synthetic tasks, compatible with
Pydantic validation and NumPy conversion:

* discrete block: one dwell time per task, a member of the configured
  dwell-time action set (paper action space, Lu et al.);
* optional continuous block: one abstract power variable per task (no
  hardware-dependent interpretation; conventions are handled by
  :mod:`radar_bo.optimization.resource_accounting`);
* conversion between the structured representation and a flat float64 NumPy
  vector ``[dwell_0, ..., dwell_{n-1}, (power_0, ..., power_{n-1})]``;
* exact discrete membership on decode — near-members are REJECTED, never
  snapped, rounded or clipped;
* JSON serialization/deserialization and defensive copying (no shared
  mutable state);
* task ordering ``task_0 .. task_{n-1}`` is fixed and preserved everywhere.

No BoTorch or PyTorch dependency (layer separation, guarded by tests).
"""

from __future__ import annotations

import hashlib

import numpy as np
from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel
from radar_bo.numerics import NumericalValidationError

_MEMBERSHIP_TOLERANCE: float = 1e-12


class DiscreteMembershipError(ValueError):
    """A flat-vector dwell value is not an exact member of the action set.

    Raised instead of snapping to the nearest discrete value (M4 instruction
    9: no silent clip, round, repair or replace).
    """


class ResourceAllocation(StrictModel):
    """Complete allocation over ``num_tasks`` ordered synthetic tasks."""

    num_tasks: int = Field(ge=1, description="Fixed number of synthetic tasks; ordering is task_0..task_{n-1}.")
    task_ids: tuple[str, ...] = Field(
        default=(),
        description="Ordered task identifiers; auto-derived as task_0..task_{n-1} when empty, validated otherwise.",
    )
    dwell_indices: tuple[int, ...] = Field(description="Per-task index into the configured dwell-time action set.")
    dwell_times_s: tuple[float, ...] = Field(description="Per-task dwell times in seconds (members of the action set).")
    action_set: tuple[float, ...] = Field(description="Configured discrete dwell-time action set (ascending, from ResourceConfig).")
    continuous_power_w: tuple[float, ...] | None = Field(
        default=None,
        description="Optional per-task continuous power block (abstract W); conventions are separate (see resource_accounting).",
    )

    # ---- Construction helpers -------------------------------------------------
    @classmethod
    def from_dwell_indices(
        cls,
        dwell_indices: list[int] | tuple[int, ...],
        action_set: list[float] | tuple[float, ...],
        continuous_power_w: list[float] | tuple[float, ...] | None = None,
    ) -> "ResourceAllocation":
        """Build from discrete indices (dwell times derived exactly)."""
        action = tuple(float(a) for a in action_set)
        dwells = tuple(action[int(i)] for i in dwell_indices)
        return cls(
            num_tasks=len(dwell_indices),
            task_ids=tuple(f"task_{i}" for i in range(len(dwell_indices))),
            dwell_indices=tuple(int(i) for i in dwell_indices),
            dwell_times_s=dwells,
            action_set=action,
            continuous_power_w=None if continuous_power_w is None else tuple(float(p) for p in continuous_power_w),
        )

    # ---- Validators -----------------------------------------------------------
    @model_validator(mode="after")
    def _validate_structure(self) -> "ResourceAllocation":
        n = self.num_tasks
        if len(self.dwell_indices) != n or len(self.dwell_times_s) != n:
            raise ValueError(
                f"Allocation inconsistency: num_tasks={n} but {len(self.dwell_indices)} indices and "
                f"{len(self.dwell_times_s)} dwell times."
            )
        expected_ids = tuple(f"task_{i}" for i in range(n))
        if self.task_ids != expected_ids:
            if len(self.task_ids) == 0:
                object.__setattr__(self, "task_ids", expected_ids)
            else:
                raise ValueError(f"task_ids must be the ordered tuple {expected_ids}; got {self.task_ids}.")
        if len(self.action_set) == 0 or any(b <= a for a, b in zip(self.action_set, self.action_set[1:])):
            raise ValueError(f"action_set must be non-empty and strictly ascending; got {self.action_set}.")
        if any(not (0 <= i < len(self.action_set)) for i in self.dwell_indices):
            raise ValueError(f"All dwell_indices must be valid indices into action_set; got {self.dwell_indices}.")
        for i, (idx, tau) in enumerate(zip(self.dwell_indices, self.dwell_times_s)):
            if not np.isfinite(tau) or tau <= 0.0:
                raise ValueError(f"dwell_times_s[{i}] must be finite and > 0; got {tau!r}.")
            if abs(tau - self.action_set[idx]) > _MEMBERSHIP_TOLERANCE:
                raise ValueError(
                    f"dwell_times_s[{i}]={tau} does not exactly match action_set[{idx}]={self.action_set[idx]}; "
                    "discrete membership is validated exactly (no snapping)."
                )
        if self.continuous_power_w is not None:
            if len(self.continuous_power_w) != n:
                raise ValueError(
                    f"continuous_power_w has {len(self.continuous_power_w)} entries but num_tasks is {n}."
                )
            for i, p in enumerate(self.continuous_power_w):
                if not np.isfinite(p) or p < 0.0:
                    raise ValueError(f"continuous_power_w[{i}] must be finite and >= 0; got {p!r}.")
        return self

    # ---- Flat-vector conversion -------------------------------------------------
    def includes_power(self) -> bool:
        return self.continuous_power_w is not None

    def expected_flat_dimension(self) -> int:
        return self.num_tasks * (2 if self.includes_power() else 1)

    def to_flat_vector(self) -> np.ndarray:
        """Flat float64 vector ``[dwell_0..dwell_{n-1}, (power_0..power_{n-1})]``.

        A fresh array is returned on every call (mutation protection).
        """
        parts = [np.asarray(self.dwell_times_s, dtype=np.float64)]
        if self.includes_power():
            parts.append(np.asarray(self.continuous_power_w, dtype=np.float64))
        return np.concatenate(parts).astype(np.float64).copy()

    @classmethod
    def from_flat_vector(
        cls,
        vector: np.ndarray,
        action_set: list[float] | tuple[float, ...],
        include_power: bool = False,
    ) -> "ResourceAllocation":
        """Decode a flat vector; discrete membership is validated EXACTLY.

        A dwell value that is not an exact member of ``action_set`` raises
        :class:`DiscreteMembershipError` (nearest values are reported for
        diagnosis, but the vector is never snapped).
        """
        v = np.asarray(vector, dtype=np.float64)
        action = tuple(float(a) for a in action_set)
        n = v.shape[0] // (2 if include_power else 1)
        block = 2 if include_power else 1
        if v.ndim != 1:
            raise NumericalValidationError(f"Flat allocation vector must be 1-D; got shape {v.shape}.")
        if not np.all(np.isfinite(v)):
            raise NumericalValidationError(f"Flat allocation vector contains non-finite values: {v!r}.")
        if v.shape[0] % block != 0 or n < 1:
            raise NumericalValidationError(
                f"Flat vector length {v.shape[0]} not divisible by block size {block} "
                f"(include_power={include_power})."
            )
        dwells = v[:n]
        indices = []
        for i, tau in enumerate(dwells):
            matches = [k for k, a in enumerate(action) if abs(tau - a) <= _MEMBERSHIP_TOLERANCE]
            if len(matches) != 1:
                nearest = min(action, key=lambda a: abs(a - tau))
                raise DiscreteMembershipError(
                    f"Flat vector dwell value {tau!r} (task {i}) is not an exact member of the action set "
                    f"{action}; nearest member {nearest!r}. Values are never snapped — supply a member "
                    "of the configured discrete set."
                )
            indices.append(matches[0])
        powers = None
        if include_power:
            powers = tuple(float(p) for p in v[n:])
        return cls(
            num_tasks=n,
            dwell_indices=tuple(indices),
            dwell_times_s=tuple(float(t) for t in dwells),
            action_set=action,
            continuous_power_w=powers,
        )

    # ---- Identity, copying, serialization ---------------------------------------
    def content_id(self) -> str:
        """Deterministic content identifier (sha256 of the flat vector bytes)."""
        digest = hashlib.sha256()
        digest.update(np.asarray(self.dwell_times_s, dtype=np.float64).tobytes())
        if self.continuous_power_w is not None:
            digest.update(np.asarray(self.continuous_power_w, dtype=np.float64).tobytes())
        digest.update(str(self.action_set).encode("utf-8"))
        return f"alloc_{digest.hexdigest()[:16]}"

    def defensive_copy(self) -> "ResourceAllocation":
        """Copy with no shared mutable state (underlying values are immutable tuples;
        the flat-vector arrays are always freshly allocated)."""
        return ResourceAllocation.model_validate(self.model_dump())
