"""Runtime decomposition analysis (M7 item 23).

Aggregates per-run runtime decompositions into per-benchmark/method summaries.
Timer boundaries and the double-counting policy live in ``metrics``
(TIMER_BOUNDARIES / DOUBLE_COUNTING_NOTE); the summary NEVER claims that
component times sum exactly to the wall time — the unattributed remainder is
reported explicitly (item 23).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from radar_bo.experiments.framework.metrics import (
    DOUBLE_COUNTING_NOTE,
    TIMER_BOUNDARIES,
    runtime_decomposition,
)

COMPONENT_KEYS = (
    "gp_fit_runtime_s",
    "acquisition_runtime_s",
    "objective_runtime_s",
    "candidate_generation_runtime_s",
    "constraint_validation_runtime_s",
)


def aggregate_runtime_decomposition(records: list[dict]) -> dict[str, Any]:
    decomps = [runtime_decomposition(r) for r in records]
    if not decomps:
        return {"n_runs": 0}
    components = {}
    for key in COMPONENT_KEYS:
        values = [d["components_s"].get(key) for d in decomps]
        values = [v for v in values if v is not None]
        components[key] = float(np.mean(values)) if values else None
    walls = [d["wall_runtime_s"] for d in decomps if d["wall_runtime_s"] is not None]
    unattributed = [d["unattributed_overhead_s"] for d in decomps if d["unattributed_overhead_s"] is not None]
    return {
        "n_runs": len(decomps),
        "component_means_s": components,
        "wall_runtime_mean_s": float(np.mean(walls)) if walls else None,
        "unattributed_overhead_mean_s": float(np.mean(unattributed)) if unattributed else None,
        "unattributed_overhead_max_s": float(np.max(unattributed)) if unattributed else None,
        "timer_boundaries": TIMER_BOUNDARIES,
        "double_counting_note": DOUBLE_COUNTING_NOTE,
        "interpretation": (
            "component sums are a PARTIAL attribution; the unattributed remainder "
            "(tensor conversion, dataset insertion, incumbent tracking, GP training-tensor "
            "preparation, diagnostics construction, loop overhead, serialization) is NOT "
            "identified component-by-component by these timers alone"
        ),
    }
