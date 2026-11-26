"""Generic Bayesian-optimization engine (M5+M6).

Application-independent infrastructure: optimization problem interface,
synthetic mathematical benchmarks (low- and high-dimensional), independent
reference solvers, dataset management, NumPy<->PyTorch conversion, GP
surrogate, LogExpectedImprovement acquisition, candidate generation (full
space and block-restricted), variable blocks with a deterministic cyclic
scheduler, and the generic single-candidate BO loop (joint / alternating /
pair-wise configurations of ONE engine) with budget accounting, block-visit
bookkeeping and stopping conditions.

Layer separation (M5 instruction 30): this subpackage MAY import
PyTorch/GPyTorch/BoTorch; it must NOT import radar, environment or simulation
modules (guarded by tests). Radar budget semantics are not used here — M5
synthetic problems carry their own explicitly defined instantaneous
mathematical constraints (M4 review 3.1), and the M6 block machinery is
index-based and application-free.
"""

from radar_bo.optimization.bo.blocks import (
    BlockDefinitionError,
    VariableBlock,
    VariableBlockSpec,
    resolve_blocks,
    validate_blocks,
)
from radar_bo.optimization.bo.scheduling import (
    BlockScheduler,
    CyclicBlockScheduler,
    VisitTicket,
    alternating_blocks_spec,
    joint_blocks_spec,
    pairwise_blocks_spec,
)
from radar_bo.optimization.bo.block_candidates import BlockCandidateGenerator

__all__ = [
    "BlockDefinitionError",
    "VariableBlock",
    "VariableBlockSpec",
    "resolve_blocks",
    "validate_blocks",
    "BlockScheduler",
    "CyclicBlockScheduler",
    "VisitTicket",
    "joint_blocks_spec",
    "alternating_blocks_spec",
    "pairwise_blocks_spec",
    "BlockCandidateGenerator",
]

