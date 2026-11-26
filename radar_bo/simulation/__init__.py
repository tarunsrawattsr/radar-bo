"""Simulation layer: authoritative state and frozen snapshots (M4).

The candidate evaluator consumes frozen snapshots only; physical time
advancement and truth management stay in this layer.
"""

from radar_bo.simulation.simulation_state import FrozenSimulationState, SimulationState

__all__ = ["SimulationState", "FrozenSimulationState"]
