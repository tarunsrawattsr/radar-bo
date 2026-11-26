"""Generic block scheduling (M6 instruction 6).

A scheduler determines WHICH block is searched next. M6 implements the
deterministic CYCLIC scheduler: blocks are visited in their given order,
repeating; a COMPLETE CYCLE consists of one visit (or recorded skip) of every
block. Search strategies are configurations of blocks + scheduler, never
changes to the mathematical objective:

* joint search       — one block containing every variable;
* alternating search — two user-defined variable blocks;
* pair-wise search   — multiple two-variable blocks.

Each visit ticket records the active block, the cycle number and the
within-cycle iteration number so logs can be traced (instruction 6).
Scheduling is independent of radar quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

from radar_bo.optimization.bo.blocks import BlockDefinitionError, VariableBlock, VariableBlockSpec


@dataclass(frozen=True)
class VisitTicket:
    """Identifies one scheduling step (instruction 6 log fields)."""

    cycle_number: int          # 1-based: number of the cycle this visit belongs to
    within_cycle_index: int    # 0-based position within the cycle
    block: VariableBlock       # the active block for this visit


class BlockScheduler:
    """Scheduling interface: which block is searched next."""

    def next_visit(self) -> VisitTicket:
        raise NotImplementedError

    def n_blocks(self) -> int:
        raise NotImplementedError


class CyclicBlockScheduler(BlockScheduler):
    """Deterministic cyclic scheduling: block[0], block[1], ..., block[0], ..."""

    def __init__(self, blocks: list[VariableBlock]) -> None:
        if not blocks:
            raise BlockDefinitionError("CyclicBlockScheduler needs at least one block.")
        self.blocks = list(blocks)
        self._cursor = 0          # absolute visit counter (0-based)
        self._cycle = 0           # 0-based internal; tickets expose 1-based

    def n_blocks(self) -> int:
        return len(self.blocks)

    def next_visit(self) -> VisitTicket:
        cycle, within = divmod(self._cursor, len(self.blocks))
        ticket = VisitTicket(
            cycle_number=cycle + 1,
            within_cycle_index=within,
            block=self.blocks[within],
        )
        self._cursor += 1
        return ticket


# ---- Strategy constructors (configurations, not new algorithms) ----------------

def joint_blocks_spec(dimension: int) -> list[VariableBlockSpec]:
    """One block containing every variable (joint search)."""
    return [VariableBlockSpec(block_id="joint_all", active_indices=tuple(range(dimension)))]


def alternating_blocks_spec(group_a: list[int], group_b: list[int], ids: tuple[str, str] = ("block_A", "block_B")) -> list[VariableBlockSpec]:
    """Two user-defined variable blocks (alternating search)."""
    if not group_a or not group_b:
        raise BlockDefinitionError("Alternating search needs two non-empty variable groups.")
    return [
        VariableBlockSpec(block_id=ids[0], active_indices=tuple(group_a)),
        VariableBlockSpec(block_id=ids[1], active_indices=tuple(group_b)),
    ]


def pairwise_blocks_spec(pairs: list[list[int]], id_prefix: str = "pair") -> list[VariableBlockSpec]:
    """Multiple two-variable blocks (pair-wise coordinate search)."""
    if len(pairs) < 2:
        raise BlockDefinitionError(
            f"Pair-wise search needs at least two two-variable blocks; got {len(pairs)}."
        )
    specs = []
    for i, pair in enumerate(pairs):
        if len(pair) != 2:
            raise BlockDefinitionError(
                f"Pair-wise blocks must contain exactly two variable indices; block {i} got {pair}."
            )
        specs.append(VariableBlockSpec(block_id=f"{id_prefix}_{i}", active_indices=(int(pair[0]), int(pair[1]))))
    return specs
