"""Generic variable-block representation (M6 instruction 5).

A block names an ordered subset of decision-variable indices that a
block-restricted search may change, while all other coordinates stay FIXED at
the current configuration. Blocks are application-free: they reference
indices of the generic problem interface only, never radar quantities.

Validation (instruction 5):

* every index within the complete decision-vector dimension;
* no duplicate indices WITHIN a block;
* blocks used together must not OVERLAP (checked by ``validate_blocks``);
* partition-based strategies may require full coverage of the decision
  vector (``required_coverage="all"``) — coverage of the INTENDED variables
  is validated, and uncovered indices are reported explicitly.

The block dimensionality, associated variable names and variable types are
derived from the problem so that "block dimension" is never confused with
the full decision-vector dimension (M6 instruction 12: the GP keeps the
FULL dimension; only the active acquisition-search dimension shrinks).
"""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.problem import OptimizationProblem


class BlockDefinitionError(ValueError):
    """Inconsistent variable-block definition (indices, duplicates, coverage)."""


class VariableBlockSpec(StrictModel):
    """Serializable block specification (block id + active variable indices)."""

    block_id: str = Field(min_length=1, description="Unique block identifier.")
    active_indices: tuple[int, ...] = Field(min_length=1, description="Ordered indices this block may change.")

    @model_validator(mode="after")
    def _unique_indices(self) -> "VariableBlockSpec":
        if len(set(self.active_indices)) != len(self.active_indices):
            raise BlockDefinitionError(
                f"Block '{self.block_id}' contains duplicate variable indices: {self.active_indices}."
            )
        return self


class VariableBlock:
    """Resolved block bound to one concrete problem (derived metadata included)."""

    def __init__(self, spec: VariableBlockSpec, problem: OptimizationProblem) -> None:
        d = problem.dimension
        for idx in spec.active_indices:
            if not (0 <= idx < d):
                raise BlockDefinitionError(
                    f"Block '{spec.block_id}' index {idx} outside decision-vector dimension {d}."
                )
        self.spec = spec
        self.problem = problem
        self.block_id = spec.block_id
        self.active_indices: tuple[int, ...] = tuple(int(i) for i in spec.active_indices)
        fixed = sorted(set(range(d)) - set(self.active_indices))
        self.fixed_indices: tuple[int, ...] = tuple(fixed)
        self.block_dimension = len(self.active_indices)
        self.variable_names: tuple[str, ...] = tuple(problem.variable_names[i] for i in self.active_indices)
        self.variable_types: tuple[str, ...] = tuple(problem.variable_types[i] for i in self.active_indices)
        self.covers_all = self.block_dimension == d

    def embed(self, active_values: np.ndarray, base_config: np.ndarray) -> np.ndarray:
        """Reconstruct a COMPLETE decision vector: fixed coordinates are copied
        EXACTLY from ``base_config`` (never modified, never repaired); active
        coordinates take ``active_values`` in block order."""
        base = np.asarray(base_config, dtype=np.float64).flatten()
        if base.shape != (self.problem.dimension,):
            raise BlockDefinitionError(
                f"base_config must have shape ({self.problem.dimension},); got {base.shape}."
            )
        values = np.asarray(active_values, dtype=np.float64).flatten()
        if values.shape != (self.block_dimension,):
            raise BlockDefinitionError(
                f"active_values must have {self.block_dimension} entries; got {values.shape}."
            )
        complete = base.copy()  # fixed coordinates preserved exactly
        complete[list(self.active_indices)] = values
        return complete

    def extract(self, complete_config: np.ndarray) -> np.ndarray:
        """Active-coordinate values (block order) of a complete vector."""
        arr = np.asarray(complete_config, dtype=np.float64).flatten()
        return np.asarray(arr[list(self.active_indices)], dtype=np.float64)

    def describes(self) -> dict:
        """Serializable description (metadata; never a GP input feature)."""
        return {
            "block_id": self.block_id,
            "active_indices": list(self.active_indices),
            "fixed_indices": list(self.fixed_indices),
            "block_dimension": self.block_dimension,
            "variable_names": list(self.variable_names),
            "variable_types": list(self.variable_types),
            "covers_all": self.covers_all,
        }


def resolve_blocks(
    problem: OptimizationProblem,
    specs: list[VariableBlockSpec] | None,
) -> list[VariableBlock]:
    """Resolve block specs against a problem; empty/None -> single joint block."""
    if not specs:
        return [VariableBlock(VariableBlockSpec(block_id="joint_all", active_indices=tuple(range(problem.dimension))), problem)]
    blocks = [VariableBlock(spec, problem) for spec in specs]
    validate_blocks(blocks, problem, require_full_coverage=False)
    return blocks


def validate_blocks(
    blocks: list[VariableBlock],
    problem: OptimizationProblem,
    require_full_coverage: bool = False,
    intended_coverage: set[int] | None = None,
) -> dict:
    """Validate a block collection; returns a coverage report.

    Checks (instruction 5):
    * at least one block; unique block ids;
    * no index appears in more than one block (overlap rejection);
    * every index within the problem dimension (already enforced per block);
    * ``require_full_coverage``: the blocks must form an exact partition of
      ALL decision variables;
    * ``intended_coverage``: the union of blocks must cover exactly these
      indices (coverage of the INTENDED decision variables).
    """
    if not blocks:
        raise BlockDefinitionError("At least one variable block is required.")
    ids = [b.block_id for b in blocks]
    if len(set(ids)) != len(ids):
        raise BlockDefinitionError(f"Block identifiers must be unique; got {ids}.")
    seen: set[int] = set()
    for block in blocks:
        overlap = seen & set(block.active_indices)
        if overlap:
            raise BlockDefinitionError(
                f"Blocks must not overlap; indices {sorted(overlap)} appear in multiple blocks."
            )
        seen |= set(block.active_indices)
    d = problem.dimension
    uncovered = sorted(set(range(d)) - seen)
    out_of_range_intent = None
    if intended_coverage is not None:
        out_of_range_intent = sorted(intended_coverage - set(range(d)))
        if out_of_range_intent:
            raise BlockDefinitionError(f"Intended coverage references invalid indices: {out_of_range_intent}.")
        missing_intent = sorted(intended_coverage - seen)
        if missing_intent:
            raise BlockDefinitionError(
                f"Blocks do not cover the intended decision variables {sorted(intended_coverage)}; "
                f"missing indices: {missing_intent}."
            )
        extra = sorted(seen - intended_coverage)
        if extra:
            raise BlockDefinitionError(f"Blocks cover indices outside the intended set: {extra}.")
    if require_full_coverage and uncovered:
        raise BlockDefinitionError(
            f"Full partition required but indices {uncovered} are not covered by any block."
        )
    return {
        "covered_indices": sorted(seen),
        "uncovered_indices": uncovered,
        "full_coverage": not uncovered,
        "n_blocks": len(blocks),
    }
