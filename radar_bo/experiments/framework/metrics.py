"""Objective-gap, progress, threshold and statistical-analysis metrics (M7 items 15-22).

All functions are pure and independently testable (item 30: unit tests use
independently calculated expected values). Numerical conventions:

* Maximization: simple regret = reference - best observed feasible objective.
  Minimization: reversed. Exact references -> "simple_regret"; approximate
  numerical references -> "reference_gap" (NOT proven optimality regret).
* Negative gaps (an optimizer exceeding an approximate reference) are NEVER
  clipped; they are retained and flagged for investigation.
* Normalized gap closure = (best - initial incumbent) / (reference - initial
  incumbent), maximization convention; zero/near-zero and negative
  denominators return None with an explicit reason (item 16).
* Paired statistics preserve seed-level pairing (item 20); ties use a
  documented tolerance; bootstrap resamples complete seed-level pairs with a
  documented seed and replication count (item 21); Wilcoxon tests document
  their hypotheses and are corrected (Holm) when presented as a family
  (item 22).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

DENOMINATOR_TOLERANCE = 1e-12
TIE_TOLERANCE_DEFAULT = 1e-9


# ---- item 15: objective gaps -------------------------------------------------
def objective_gap(best: float, reference: float, maximize: bool = True,
                  reference_is_exact: bool = True) -> dict[str, Any]:
    gap = float(reference) - float(best) if maximize else float(best) - float(reference)
    return {
        "value": gap,
        "label": "simple_regret" if reference_is_exact else "reference_gap",
        "reference_is_exact": reference_is_exact,
        "exceeds_reference": gap < 0.0,  # retained and flagged, never clipped
    }


# ---- item 16: normalized gap closure ------------------------------------------
def normalized_gap_closure(best: float, initial_incumbent: float, reference: float,
                           maximize: bool = True) -> dict[str, Any]:
    denominator = (float(reference) - float(initial_incumbent)) if maximize \
        else (float(initial_incumbent) - float(reference))
    numerator = (float(best) - float(initial_incumbent)) if maximize \
        else (float(initial_incumbent) - float(best))
    if abs(denominator) <= DENOMINATOR_TOLERANCE:
        return {"value": None, "reason": "degenerate_gap: initial incumbent equals the reference; closure undefined"}
    if denominator < 0:
        return {"value": None, "reason": "negative_gap: initial incumbent is better than the reference; closure undefined"}
    return {"value": float(numerator / denominator), "reason": None}


# ---- item 18: thresholds --------------------------------------------------------
def gap_closure_threshold_value(initial_incumbent: float, reference: float, fraction: float,
                                maximize: bool = True) -> float | None:
    """Objective level that corresponds to closing `fraction` of the initial gap.

    Identical formula for both directions: level = initial + fraction * (reference - initial).
    """
    del maximize  # the level formula is direction-independent
    return float(initial_incumbent + fraction * (reference - initial_incumbent))


def evaluations_to_level(history: list[float], level: float) -> int | None:
    """First index (1-based) at which the incumbent history reaches `level`.

    The history is per-successful-evaluation in evaluation order (entry 0 after
    the initial design). Returns None when the level is never reached — an
    unreached threshold is reported explicitly, never interpolated (item 18).
    """
    for i, value in enumerate(history):
        if value is not None and float(value) >= level:
            return i + 1
    return None


def threshold_analysis(history: list[float], initial_incumbent: float, reference: float,
                       fractions: list[float], wall_timestamps: list[float] | None = None,
                       attempted_axis: list[int] | None = None) -> list[dict]:
    """Threshold metrics for the documented gap-closure fractions (item 18)."""
    out = []
    for fraction in fractions:
        level = gap_closure_threshold_value(initial_incumbent, reference, fraction)
        index = evaluations_to_level(history, level) if level is not None else None
        entry: dict[str, Any] = {
            "fraction": fraction,
            "level": level,
            "reached": index is not None,
            "successful_evaluations_required": index,
            "attempted_evaluations_required": None,
            "wall_clock_s_required": None,
        }
        if index is not None:
            if attempted_axis is not None and len(attempted_axis) >= index:
                entry["attempted_evaluations_required"] = int(attempted_axis[index - 1])
            if wall_timestamps is not None and len(wall_timestamps) >= index:
                entry["wall_clock_s_required"] = float(wall_timestamps[index - 1])
            if float(history[0]) >= level:
                entry["initial_design_already_meets_threshold"] = True
        out.append(entry)
    return out


# ---- item 19: descriptive statistics ---------------------------------------------
def describe(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "note": "no valid runs"}
    q1, q3 = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "stdev": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "iqr": float(q3 - q1),
    }


# ---- item 20: paired comparisons ---------------------------------------------------
def paired_differences(values_a: dict[int, float], values_b: dict[int, float],
                       tie_tolerance: float = TIE_TOLERANCE_DEFAULT) -> dict[str, Any]:
    """Per-seed paired differences (a - b); seeds must match (item 20)."""
    seeds = sorted(set(values_a) & set(values_b))
    missing = sorted(set(values_a) ^ set(values_b))
    diffs = np.asarray([values_a[s] - values_b[s] for s in seeds], dtype=np.float64)
    wins = int(np.sum(diffs > tie_tolerance))
    losses = int(np.sum(diffs < -tie_tolerance))
    ties = int(np.sum(np.abs(diffs) <= tie_tolerance))
    return {
        "seeds": seeds,
        "missing_seeds": missing,
        "differences": diffs,
        "n_pairs": int(diffs.size),
        "mean_difference": float(diffs.mean()) if diffs.size else None,
        "median_difference": float(np.median(diffs)) if diffs.size else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "tie_tolerance": tie_tolerance,
    }


def paired_effect_size(diffs: np.ndarray) -> float | None:
    """Cohen's d_z for paired differences (mean / std of differences)."""
    if diffs.size < 2:
        return None
    sd = float(diffs.std(ddof=1))
    if sd <= DENOMINATOR_TOLERANCE:
        return None  # all differences (near-)identical: effect undefined
    return float(diffs.mean() / sd)


# ---- item 21: bootstrap ------------------------------------------------------------
def bootstrap_ci(diffs: np.ndarray, replications: int, seed: int,
                 confidence_level: float = 0.95) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples COMPLETE seed-level pairs (with replacement) — the pairing is
    preserved under resampling (item 21). Documented seed + replication count.
    """
    rng = np.random.default_rng(seed)
    n = diffs.size
    if n == 0:
        return {"lower": None, "upper": None, "note": "no pairs"}
    idx = rng.integers(0, n, size=(replications, n))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "lower": float(np.percentile(means, 100 * alpha)),
        "upper": float(np.percentile(means, 100 * (1 - alpha))),
        "confidence_level": confidence_level,
        "replications": replications,
        "bootstrap_seed": seed,
        "note": (
            "uncertainty from ten runs is descriptive only; a narrow interval is NOT "
            "evidence of universal superiority (item 21 limitation)"
        ),
    }


# ---- item 22: statistical tests ------------------------------------------------------
WILCOXON_DOCUMENTATION = {
    "null_hypothesis": "the median of the paired per-seed differences is zero",
    "alternative_hypothesis": "the median of the paired per-seed differences is not zero (two-sided)",
    "zero_difference_handling": "zero differences are DROPPED (scipy zero_method='wilcox') and reduce the effective sample size",
    "tie_handling": "equal absolute differences receive averaged ranks (scipy normal-approximation fallback when ties/zeroes are present)",
    "assumptions": "paired differences are treated as (approximately) symmetrically distributed around the median; n is small (10), so exact-vs-approximate mode follows scipy 'auto'",
    "policy": "p-values COMPLEMENT effect sizes and bootstrap intervals; no practical significance is inferred from p-values alone (item 22)",
}


def wilcoxon_signed_rank(diffs: np.ndarray) -> dict[str, Any]:
    if diffs.size == 0:
        return {"statistic": None, "p_value": None, "n_used": 0, "valid": False}
    nonzero = diffs[diffs != 0.0]
    if nonzero.size == 0:
        return {"statistic": None, "p_value": None, "n_used": 0, "valid": False,
                "note": "all differences zero: test undefined"}
    result = scipy_stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", mode="auto")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_used": int(nonzero.size),
        "n_zero_differences": int(diffs.size - nonzero.size),
        "valid": True,
        "documentation": WILCOXON_DOCUMENTATION,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = float(running_max)
    return adjusted


# ---- convergence axes (item 17) --------------------------------------------------------
def convergence_curves(record: dict) -> dict[str, Any]:
    """Convergence points on the documented axes (item 17).

    * attempted evaluations: EXACT for M7 runner records (per-iteration
      attempted_evaluations counter); archived M6 records lack the counter, so
      the attempted axis is unavailable there and the successful axis is used.
    * successful evaluations: entry k of the incumbent history = k-th
      successful evaluation (exact for every record).
    * wall-clock: from per-iteration elapsed_runtime_s (M7 records; None for
      archived records).
    * complete block cycles: cycle numbers of improving iterations (secondary
      axis; NOT a primary cross-architecture axis because architectures have
      different numbers of blocks per cycle — item 17).

    No evaluations are manufactured after termination (item 17): curves end at
    the recorded stopping point and the remaining horizon is documented as
    truncated.
    """
    history = record.get("incumbent_history") or []
    diagnostics = record.get("iteration_diagnostics") or []
    successful_axis = list(range(1, len(history) + 1))
    attempted_axis = [d.get("attempted_evaluations") for d in diagnostics if d.get("objective_value") is not None]
    wall_axis = [d.get("elapsed_runtime_s") for d in diagnostics if d.get("objective_value") is not None]
    improving = []
    prev = None
    for d in diagnostics:
        if d.get("objective_value") is None:
            continue
        v = d["incumbent_objective"]
        if prev is None or v > prev + 1e-12:
            improving.append(d.get("cycle_number"))
        prev = v
    return {
        "successful_evaluations": {"x": successful_axis, "y": history},
        "attempted_evaluations": {
            "x": attempted_axis if attempted_axis and all(a is not None for a in attempted_axis) else None,
            "y": history[1:1 + len(attempted_axis)] if attempted_axis else [],
            "note": None if attempted_axis and all(a is not None for a in attempted_axis)
            else "attempted-eval counter not archived (pre-M7 record); use the successful axis",
        },
        "wall_clock": {
            "x": wall_axis if wall_axis and all(w is not None for w in wall_axis) else None,
            "y": history[1:1 + len(wall_axis)] if wall_axis else [],
        },
        "block_cycles": {"cycle_numbers_of_improvements": improving},
        "truncated_horizon_note": (
            "curves end at the recorded stopping point; no objective values are "
            "manufactured for the unexecuted remainder of the budget"
        ),
    }


# ---- runtime decomposition (item 23) ----------------------------------------------------
TIMER_BOUNDARIES = {
    "gp_fit_runtime_s": "GP construction + fitting (per successful observation; refit-per-observation policy).",
    "acquisition_runtime_s": "LogExpectedImprovement evaluation over the scored candidate pool (selection excluded).",
    "objective_runtime_s": "problem.objective() calls only (true objective evaluation).",
    "candidate_generation_runtime_s": "candidate-pool sampling (random number generation + assembly).",
    "constraint_validation_runtime_s": "feasibility checks on complete candidate vectors.",
    "wall_runtime_s": "total wall-clock of the optimize() call.",
}
# The five stage timers are DISJOINT by construction (pool generation is not
# nested in acquisition scoring; GP fitting is not nested in acquisition
# evaluation), so their sum is a valid partial attribution. Serialization,
# tensor conversion, dataset insertion, incumbent tracking, GP training-tensor
# preparation and loop overhead are NOT separately timed and appear only in
# `unattributed_overhead_s` — the decomposition is NOT claimed to sum exactly
# to the wall time (item 23).
DOUBLE_COUNTING_NOTE = (
    "stage timers are disjoint by construction; no component timer nests another; "
    "unattributed = wall - sum(components) >= 0 is expected and reported, never hidden"
)


def runtime_decomposition(record: dict) -> dict[str, Any]:
    rt = record.get("runtime_measurements") or {}
    components = {k: rt[k] for k in (
        "gp_fit_runtime_s", "acquisition_runtime_s", "objective_runtime_s",
        "candidate_generation_runtime_s", "constraint_validation_runtime_s") if k in rt}
    component_sum = sum(components.values())
    wall = rt.get("wall_runtime_s")
    return {
        "components_s": components,
        "component_sum_s": component_sum,
        "wall_runtime_s": wall,
        "unattributed_overhead_s": (wall - component_sum) if wall is not None else None,
        "timer_boundaries": TIMER_BOUNDARIES,
        "double_counting_note": DOUBLE_COUNTING_NOTE,
    }


# ---- record-level aggregation (items 19/20 wiring into runner records) ---------------
def run_objective_gap(record: dict) -> dict[str, Any]:
    return objective_gap(
        record["final_objective"], record["reference_objective"],
        maximize=True, reference_is_exact=record.get("reference_is_exact", True),
    )


def run_normalized_closure(record: dict) -> dict[str, Any]:
    history = record.get("incumbent_history") or []
    if not history:
        return {"value": None, "reason": "no incumbent history (no successful evaluation)"}
    return normalized_gap_closure(record["final_objective"], history[0], record["reference_objective"], maximize=True)


def describe_run_records(records: list[dict], config) -> dict:
    """Per-benchmark x method descriptive + paired analysis (items 19-22).

    Records missing required analysis fields (e.g. minimal stub records from an
    interrupted-experiment scenario) are EXCLUDED from aggregation and counted
    explicitly under ``incomplete_records_excluded`` — never silently pooled.
    """
    by_bm_method: dict[str, dict[str, list[dict]]] = {}
    excluded = 0
    for rec in records:
        if rec.get("final_objective") is None or rec.get("reference_objective") is None:
            excluded += 1
            continue
        by_bm_method.setdefault(rec["benchmark_id"], {}).setdefault(rec["algorithm_id"], []).append(rec)
    if excluded:
        pass  # reported per benchmark block below via the shared note

    stats_settings = config.statistics
    fractions = stats_settings.threshold_closures
    result: dict[str, Any] = {"incomplete_records_excluded": excluded}
    for benchmark_id, by_method in sorted(by_bm_method.items()):
        block: dict[str, Any] = {"per_method": {}, "paired_comparisons": {}}
        reference_value = None
        for method, recs in sorted(by_method.items()):
            finals = [r["final_objective"] for r in recs if r.get("final_objective") is not None]
            gaps = [run_objective_gap(r) for r in recs if r.get("final_objective") is not None]
            closures = [run_normalized_closure(r) for r in recs]
            threshold_rows = []
            for r in recs:
                history = r.get("incumbent_history") or []
                if not history:
                    continue
                attempted_axis = None
                diagnostics = r.get("iteration_diagnostics") or []
                att = [d.get("attempted_evaluations") for d in diagnostics if d.get("objective_value") is not None]
                if att and all(a is not None for a in att):
                    attempted_axis = att
                wall_ts = [d.get("elapsed_runtime_s") for d in diagnostics if d.get("objective_value") is not None]
                wall_ts = wall_ts if wall_ts and all(w is not None for w in wall_ts) else None
                threshold_rows.append(threshold_analysis(
                    history, history[0], r["reference_objective"], fractions,
                    wall_timestamps=wall_ts, attempted_axis=attempted_axis,
                ))
            threshold_summary = []
            for fi, fraction in enumerate(fractions):
                reached = [rows[fi] for rows in threshold_rows if rows[fi]["reached"]]
                threshold_summary.append({
                    "fraction": fraction,
                    "runs_reached": f"{len(reached)}/{len(threshold_rows)}",
                    "successful_evaluations_required_mean": (
                        float(np.mean([x["successful_evaluations_required"] for x in reached])) if reached else None
                    ),
                    "attempted_evaluations_required_mean": (
                        float(np.mean([x["attempted_evaluations_required"] for x in reached]))
                        if reached and all(x["attempted_evaluations_required"] is not None for x in reached) else None
                    ),
                    "wall_clock_s_required_mean": (
                        float(np.mean([x["wall_clock_s_required"] for x in reached]))
                        if reached and all(x["wall_clock_s_required"] is not None for x in reached) else None
                    ),
                })
            failures_total = sum(len(r.get("failures") or []) for r in recs)
            decomp = [runtime_decomposition(r) for r in recs]
            block["per_method"][method] = {
                "n_valid_runs": len(recs),
                "final_objective": describe(finals),
                "gap": {
                    "label": gaps[0]["label"] if gaps else None,
                    "mean": float(np.mean([g["value"] for g in gaps])) if gaps else None,
                    "median": float(np.median([g["value"] for g in gaps])) if gaps else None,
                    "exceeds_reference_runs": sum(1 for g in gaps if g["exceeds_reference"]),
                },
                "normalized_gap_closure": {
                    "values": [c["value"] for c in closures],
                    "undefined_reasons": sorted({c["reason"] for c in closures if c["reason"] is not None}),
                    "mean": float(np.mean([c["value"] for c in closures if c["value"] is not None]))
                    if any(c["value"] is not None for c in closures) else None,
                },
                "threshold_analysis": threshold_summary,
                "evaluation_failure_frequency": failures_total / max(1, len(recs)),
                "runtime_decomposition_mean_s": {
                    k: float(np.mean([d["components_s"][k] for d in decomp if d["components_s"].get(k) is not None]))
                    for k in decomp[0]["components_s"]
                } if decomp else {},
                "unattributed_overhead_mean_s": float(np.mean([
                    d["unattributed_overhead_s"] for d in decomp if d["unattributed_overhead_s"] is not None
                ])) if decomp and any(d["unattributed_overhead_s"] is not None for d in decomp) else None,
                "wall_runtime_mean_s": float(np.mean([
                    d["wall_runtime_s"] for d in decomp if d["wall_runtime_s"] is not None
                ])) if decomp and any(d["wall_runtime_s"] is not None for d in decomp) else None,
                "stopping_reasons": sorted({r.get("stopping_reason") or "unknown" for r in recs}),
            }
            if reference_value is None:
                reference_value = recs[0]["reference_objective"]

        # paired comparisons vs the first non-random method? Convention: every
        # method pair vs joint_bo when present, else vs the alphabetically
        # first BO method (documented in the report).
        methods = sorted(by_method.keys())
        baseline = "joint_bo" if "joint_bo" in methods else methods[0]
        base_by_seed = {r["seed"]: r["final_objective"] for r in by_method[baseline]}
        p_values: list[float] = []
        pair_keys: list[str] = []
        for method in methods:
            if method == baseline:
                continue
            other_by_seed = {r["seed"]: r["final_objective"] for r in by_method[method]}
            paired = paired_differences(other_by_seed, base_by_seed, stats_settings.paired_tie_tolerance)
            diffs = paired.pop("differences")
            paired["effect_size_dz"] = paired_effect_size(diffs)
            paired["bootstrap_ci"] = bootstrap_ci(
                diffs, stats_settings.bootstrap_replications, stats_settings.bootstrap_seed,
                stats_settings.confidence_level,
            )
            test = wilcoxon_signed_rank(diffs)
            paired["wilcoxon"] = test
            if test.get("valid"):
                p_values.append(test["p_value"])
                pair_keys.append(method)
            block["paired_comparisons"][f"{method}_vs_{baseline}"] = paired
        if p_values and stats_settings.multiple_comparison_correction == "holm":
            adjusted = holm_correction(p_values)
            for key, p in zip(pair_keys, adjusted):
                block["paired_comparisons"][f"{key}_vs_{baseline}"]["wilcoxon"]["p_value_holm_adjusted"] = p
        result[benchmark_id] = block
    return result
