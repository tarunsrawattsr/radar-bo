"""Candidate-generation profiling (M7 item 24).

Analyzes the existing pool reports recorded in run records (no new
instrumentation required): sampling rejection rates, duplicate rates, empty
pools, skipped blocks, validation time and time per delivered candidate.

Exhaustion classification (item 24):
* "proven finite-space exhaustion" — when the searched space is PROVABLY
  finite (an all-discrete block with k discrete values each has exactly
  k^m distinct combinations), reaching every distinct combination proves
  exhaustion;
* "sampling failure" — for continuous spaces, an empty pool only shows that
  random sampling found no feasible candidate within the attempt budget; it
  does NOT prove the feasible set is empty.

The historical M6 algorithm is preserved unchanged as the versioned baseline
(ALGORITHM_BASELINE); any future more-efficient generic generator must be a
new, explicitly versioned implementation (item 24).
"""

from __future__ import annotations

from typing import Any

import numpy as np

ALGORITHM_BASELINE = "block-cyclic seeded feasible sampler (M6, unchanged; see radar_bo.optimization.bo.block_candidates)"


def _pool_counts(record: dict) -> dict[str, int] | None:
    counts = (record.get("evaluation_counts") or {})
    if "pool_requested_total" not in counts:
        return None
    return {
        "pool_requests": counts.get("pool_requests", 0),
        "requested": counts.get("pool_requested_total", 0),
        "delivered": counts.get("pool_delivered_total", 0),
        "rejected_infeasible": counts.get("pool_rejected_infeasible_total", 0),
        "duplicate_within_pool": counts.get("pool_rejected_duplicate_within_pool_total", 0),
        "duplicate_evaluated": counts.get("pool_rejected_duplicate_evaluated_total", 0),
    }


def classify_exhaustion(benchmark_entry, record: dict) -> str:
    """Proven finite-space exhaustion vs sampling failure (item 24)."""
    blocks = ((record.get("block_cycle_diagnostics") or {}).get("variable_blocks")) or []
    if not blocks:
        return "not_block_restricted"
    problem = benchmark_entry.problem(record["dimension"])
    finite = True
    total_distinct = 1
    for block in blocks:
        active = block["active_indices"]
        for idx in active:
            if problem.variable_types[idx] != "discrete":
                finite = False
                break
            total_distinct *= len(problem.discrete_values[idx])
        if not finite:
            break
    if finite:
        return f"proven_finite_space: {total_distinct} distinct combinations per {len(blocks)}-block cycle"
    return "continuous_or_mixed_space: empty pools indicate sampling failure within the attempt budget, NOT an empty feasible set"


def profile_candidate_generation(records: list[dict], benchmark_entry) -> dict[str, Any]:
    out: dict[str, Any] = {"algorithm_baseline": ALGORITHM_BASELINE, "per_method": {}}
    by_method: dict[str, list[dict]] = {}
    for rec in records:
        by_method.setdefault(rec["algorithm_id"], []).append(rec)
    for method, recs in sorted(by_method.items()):
        pool_rows = [_pool_counts(r) for r in recs]
        pool_rows = [p for p in pool_rows if p is not None]
        rt = [r.get("runtime_measurements") or {} for r in recs]
        gen = [x.get("candidate_generation_runtime_s") for x in rt if x.get("candidate_generation_runtime_s") is not None]
        feas = [x.get("constraint_validation_runtime_s") for x in rt if x.get("constraint_validation_runtime_s") is not None]
        skips = [ (r.get("block_cycle_diagnostics") or {}).get("n_skipped_blocks", 0) for r in recs ]
        if pool_rows:
            requested = float(np.mean([p["requested"] for p in pool_rows]))
            delivered = float(np.mean([p["delivered"] for p in pool_rows]))
            rejected = float(np.mean([p["rejected_infeasible"] for p in pool_rows]))
            infeasible_rate_per_requested = rejected / requested if requested > 0 else None
            time_per_delivered = (float(np.mean(gen)) / delivered) if delivered > 0 and gen else None
        else:
            requested = delivered = rejected = infeasible_rate_per_requested = time_per_delivered = None
        out["per_method"][method] = {
            "n_runs": len(recs),
            "pool_requested_mean": requested,
            "pool_delivered_mean": delivered,
            "pool_rejected_infeasible_mean": rejected,
            "infeasible_rejections_per_requested": infeasible_rate_per_requested,
            "duplicate_within_pool_mean": float(np.mean([p["duplicate_within_pool"] for p in pool_rows])) if pool_rows else None,
            "duplicate_evaluated_mean": float(np.mean([p["duplicate_evaluated"] for p in pool_rows])) if pool_rows else None,
            "empty_pool_skipped_blocks_mean": float(np.mean(skips)) if skips else None,
            "candidate_generation_time_mean_s": float(np.mean(gen)) if gen else None,
            "constraint_validation_time_mean_s": float(np.mean(feas)) if feas else None,
            "generation_time_per_delivered_candidate_s": time_per_delivered,
            "exhaustion_classification": classify_exhaustion(benchmark_entry, recs[0]),
        }
    return out
