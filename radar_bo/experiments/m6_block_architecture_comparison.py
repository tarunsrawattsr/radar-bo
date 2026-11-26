"""M6 block-architecture comparison experiment (M6 instructions 21-26).

RESULT LABEL: generic synthetic M6 block-architecture comparison — synthetic
benchmarks ONLY. This experiment makes NO radar claims and NO claim that any
architecture is superior in general; unfavorable comparisons are reported as
measured (instruction 32).

Design:

* Benchmarks (all synthetic, self-defined; independent references):
  - separable_4d / separable_8d / separable_16d (analytic exact, f* = 10 d)
  - coupled_4d / coupled_8d (analytic exact, f* = 20)
  - constrained_mixed_8d (numerical_approximate via enumeration + KKT bisection)
  - block_stagnation_4d (analytic exact LP, f* = 9.6)
* Methods: random_search (same budget/seed), joint BO (M5 baseline,
  variable_blocks=None), alternating BO (recommended two-block grouping),
  pair-wise coordinate BO (recommended two-variable blocks).
* Primary comparison: budget 100, initial design 30, pool 500, seeds 0-9
  (settings inherited from M5; experimental, not proven optimal).
* Dimension study: separable_8d / separable_16d with joint + pairwise (+RS).

Fairness (instruction 21): identical initial configurations and observed
values per seed (same seeded sampler), equivalent GP/acquisition/precision/
feasibility settings across architectures; per-run requested/feasible/scored
candidate counts are recorded (rejection rates differ by architecture and are
NOT normalized away).

Phases (instruction 22): run --pilot first (budget 40, initial 10, seeds 0-1)
with validation checks and cost estimate; the full matrix only after the
pilot passes.

Run::

    python -m radar_bo.experiments.m6_block_architecture_comparison [output_dir] [--pilot]
        [--benchmarks=a,b,...] [--methods=m1,m2,...] [--seeds=0,1,...]

The compact durable evidence archive (per-seed histories, incumbents,
runtimes, references, versions, commit id) is written to
``research_results/`` (tracked in Git) in addition to the full artifacts
under ``results/`` (gitignored, regenerable).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from radar_bo.config import build_default_provenance
from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.benchmarks_highdim import (
    BlockStagnationLP,
    ConstrainedMixedHD,
    CoupledQuadraticHD,
    SeparableQuadraticHD,
)
from radar_bo.optimization.bo.random_search import RandomSearchOptimizer
from radar_bo.optimization.bo.reference_solvers import (
    constrained_mixed_hd_reference,
    coupled_hd_reference,
    separable_hd_reference,
    stagnation_lp_reference,
)

RESULT_LABEL = (
    "generic synthetic M6 block-architecture comparison - synthetic benchmarks "
    "only; no radar, no physical UAV claims; descriptive statistics over seeds, "
    "no superiority claims"
)

EVALUATION_BUDGET = 100   # provisional synthetic budget (inherited from M5)
INITIAL_DESIGN = 30       # experimental (inherited from M5)
NUM_CANDIDATES = 500      # Benzaghta Nc (inherited from M5)
PILOT_BUDGET = 40
PILOT_INITIAL = 10

# Objective-quality thresholds (instruction 23): frozen BEFORE any full-matrix
# result was examined (i.e. during the pilot phase). threshold = fraction x the
# independent reference value f*; "evaluations_to_threshold" is the number of
# objective evaluations until the running incumbent first reaches it.
THRESHOLD_FRACTIONS = {
    "separable_4d": 0.90,
    "separable_8d": 0.90,
    "separable_16d": 0.90,
    "coupled_4d": 0.90,
    "coupled_8d": 0.90,
    "constrained_mixed_8d": 0.95,
    "block_stagnation_4d": 0.90,
}
THRESHOLD_RULE = (
    "frozen during the pilot phase, before any full-matrix comparative result "
    "was examined: threshold = fraction * independent reference value f* "
    "(0.90 for separable/coupled/block-stagnation, 0.95 for constrained_mixed)"
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable (not a git checkout at run time)"


def benchmark_factories() -> dict[str, dict]:
    return {
        "separable_4d": {
            "factory": lambda: SeparableQuadraticHD(4),
            "reference": lambda: separable_hd_reference(4),
        },
        "separable_8d": {
            "factory": lambda: SeparableQuadraticHD(8),
            "reference": lambda: separable_hd_reference(8),
        },
        "separable_16d": {
            "factory": lambda: SeparableQuadraticHD(16),
            "reference": lambda: separable_hd_reference(16),
        },
        "coupled_4d": {
            "factory": lambda: CoupledQuadraticHD(4),
            "reference": lambda: coupled_hd_reference(4),
        },
        "coupled_8d": {
            "factory": lambda: CoupledQuadraticHD(8),
            "reference": lambda: coupled_hd_reference(8),
        },
        "constrained_mixed_8d": {
            "factory": lambda: ConstrainedMixedHD(8),
            "reference": lambda: constrained_mixed_hd_reference(8),
        },
        "block_stagnation_4d": {
            "factory": lambda: BlockStagnationLP(),
            "reference": lambda: stagnation_lp_reference(),
        },
    }


def blocks_for(problem, method: str):
    recommended = getattr(problem, "recommended_blocks", None)
    if method == "joint":
        return None, "joint"
    if method == "alternating":
        groups = recommended["alternating"]
        return [(f"block_{i}", tuple(g)) for i, g in enumerate(groups)], "alternating"
    if method == "pairwise":
        pairs = recommended["pairwise"]
        return [(f"pair_{i}", tuple(p)) for i, p in enumerate(pairs)], "coordinate"
    raise ValueError(f"Unknown BO method: {method}")


def _evaluations_to_threshold(history: list[float], threshold: float | None) -> int | None:
    """Number of objective evaluations until the running incumbent first
    reaches ``threshold`` (1-based; None when never reached). ``history`` has
    per-evaluation resolution: one entry after the initial design plus one per
    successful BO evaluation, in evaluation order."""
    if threshold is None:
        return None
    for i, value in enumerate(history):
        if value is not None and float(value) >= threshold:
            return i + 1
    return None


def _out_of_block_evidence(result) -> dict:
    """Instruction-25 stagnation metric: per-block attribution of incumbent
    improvements plus evidence of how much each improvement event depends on
    coordinates OUTSIDE the visited block.

    For every incumbent-improving observation made under block ``b`` we record
    how many coordinates outside ``b`` differ from the previous incumbent.
    Block candidates only vary active coordinates, so differences outside
    ``b`` come from the inherited search center (not from this visit's
    sampling); the count documents how much of each improvement is inherited
    rather than attributable to the visited block.
    """
    blocks = {b["block_id"]: set(int(i) for i in b["active_indices"]) for b in result.variable_blocks}
    improvements_by_block: dict[str, int] = {}
    events: list[dict] = []
    prev_val: float | None = None
    prev_cfg: list[float] | None = None
    for rec in result.observations:
        value = float(rec.objective)
        block_id = rec.metadata.get("block_id") if rec.metadata else None
        if prev_val is None or value > prev_val + 1e-12:
            if prev_cfg is not None and block_id in blocks:
                active = blocks[block_id]
                outside = [
                    i for i in range(len(rec.config))
                    if i not in active and abs(float(prev_cfg[i]) - float(rec.config[i])) > 1e-9
                ]
                if outside:
                    events.append({
                        "block_id": block_id,
                        "evaluation_id": rec.evaluation_id,
                        "n_outside_changed": len(outside),
                    })
            key = block_id if block_id is not None else "initial_design"
            improvements_by_block[key] = improvements_by_block.get(key, 0) + 1
            prev_val, prev_cfg = value, [float(x) for x in rec.config]
    return {
        "incumbent_improvements_by_block": improvements_by_block,
        "n_improvement_events_outside_visited_block": len(events),
        "events": events,
    }


def run_bo(problem, method: str, seed: int, budget: int, initial: int, reference, threshold: float | None = None) -> dict:
    specs, mode = blocks_for(problem, method)
    config = OptimizationConfig(
        evaluation_budget=budget,
        num_initial_observations=initial,
        num_acquisition_candidates=NUM_CANDIDATES,
        search_mode=mode,
        variable_blocks=specs,
    )
    started = time.perf_counter()
    result = BayesianOptimizer(
        problem, config,
        reference_objective=reference.objective_value,
        reference_method=reference.verification_method,
        seed=seed,
    ).optimize()
    wall = time.perf_counter() - started
    # ---- per-run validity checks (fail loudly; instruction 22/33) ------------
    attempted = result.n_initial_evaluations + result.n_bo_evaluations
    assert attempted == budget, f"budget violation: {attempted} != {budget}"
    assert len(result.observations) + result.n_failed_attempts == attempted
    for record in result.observations:
        assert problem.is_feasible(np.asarray(record.config)), f"infeasible observation {record.config}"
    if result.best_config is not None:
        assert tuple(result.best_config) in {tuple(r.config) for r in result.observations}
    # Block-visit accounting (BO observations + skips == visits):
    bo_obs = sum(1 for r in result.observations if r.metadata.get("stage") == "bo_iteration")
    assert bo_obs + result.n_skipped_blocks == result.n_block_visits, "block-visit accounting violated"
    # Per-evaluation incumbent history resolution (initial design + one entry
    # per successful BO evaluation, in evaluation order):
    assert len(result.incumbent_history) == 1 + bo_obs, "incumbent_history resolution violated"
    return {
        "seed": seed,
        "method": method,
        "best_config": result.best_config,
        "best_objective": result.best_objective,
        "gap_to_reference": result.best_gap_to_reference,
        "n_initial_evaluations": result.n_initial_evaluations,
        "n_bo_evaluations": result.n_bo_evaluations,
        "n_failed_attempts": result.n_failed_attempts,
        "n_acquisition_scores": result.n_acquisition_scores,
        "n_block_visits": result.n_block_visits,
        "n_cycles_completed": result.n_cycles_completed,
        "n_skipped_blocks": result.n_skipped_blocks,
        "n_accepted_updates": result.n_accepted_updates,
        "n_cycles_without_improvement": result.n_cycles_without_improvement,
        "pool_requests": result.pool_requests,
        "pool_requested_total": result.pool_requested_total,
        "pool_delivered_total": result.pool_delivered_total,
        "pool_rejected_infeasible_total": result.pool_rejected_infeasible_total,
        "pool_rejected_duplicate_within_pool_total": result.pool_rejected_duplicate_within_pool_total,
        "pool_rejected_duplicate_evaluated_total": result.pool_rejected_duplicate_evaluated_total,
        "objective_threshold": threshold,
        "evaluations_to_threshold": _evaluations_to_threshold(result.incumbent_history, threshold),
        "improvement_over_initial_incumbent": (
            float(result.incumbent_history[-1]) - float(result.incumbent_history[0])
        ) if result.incumbent_history else None,
        "out_of_block_improvement_evidence": _out_of_block_evidence(result),
        "skipped_block_records": list(result.skipped_block_records),
        "iteration_diagnostics": [d.model_dump() for d in result.iteration_diagnostics],
        "gp_fit_runtime_s": result.gp_fit_runtime_s,
        "acquisition_runtime_s": result.acquisition_runtime_s,
        "objective_runtime_s": result.objective_runtime_s,
        "candidate_generation_runtime_s": result.candidate_generation_runtime_s,
        "constraint_validation_runtime_s": result.constraint_validation_runtime_s,
        "wall_runtime_s": wall,
        "stopping_reason": result.stopping_reason,
        "incumbent_history": result.incumbent_history,
        "current_config_final": result.current_config_final,
        "search_strategy": result.search_strategy,
        "validity_checks_passed": True,
    }


def run_random_search(problem, seed: int, budget: int, threshold: float | None = None) -> dict:
    started = time.perf_counter()
    result = RandomSearchOptimizer(problem, budget, seed=seed).optimize()
    wall = time.perf_counter() - started
    assert result.n_objective_evaluations == budget, "RS budget violation"
    reference_value = None  # filled by caller
    history = [entry[1] for entry in result.history]
    return {
        "seed": seed,
        "method": "random_search",
        "best_config": result.best_config,
        "best_objective": result.best_objective,
        "gap_to_reference": None,
        "n_objective_evaluations": result.n_objective_evaluations,
        "n_failed_attempts": result.n_failed_attempts,
        "n_rejected_infeasible": result.n_rejected_infeasible,
        "n_skipped_duplicates": result.n_skipped_duplicates,
        "objective_threshold": threshold,
        "evaluations_to_threshold": _evaluations_to_threshold(history, threshold),
        "improvement_over_initial_incumbent": (
            float(history[-1]) - float(history[0])
        ) if history else None,
        "wall_runtime_s": wall,
        "stopping_reason": result.stopping_reason,
        "incumbent_history": history,
        "validity_checks_passed": True,
    }


def _simple_regret(best_objective: float, reference_value: float, maximize: bool) -> float:
    """Simple regret relative to the reference (nonnegative when the reference
    is better; a numerical reference may be exceeded -> small negative values
    are reported and flagged by the caller)."""
    return float(reference_value - best_objective) if maximize else float(best_objective - reference_value)


def run_matrix(benchmark_names: list[str], methods: list[str], seeds: list[int],
               budget: int, initial: int, progress: bool = True) -> dict:
    factories = benchmark_factories()
    out: dict[str, dict] = {}
    for name in benchmark_names:
        spec = factories[name]
        problem = spec["factory"]()
        reference = spec["reference"]()
        maximize = problem.maximize
        rows: list[dict] = []
        threshold = float(reference.objective_value) * THRESHOLD_FRACTIONS.get(name, 0.90)
        # Shared initial incumbent per seed across methods (instruction 23:
        # improvement over the shared initial incumbent).
        for seed in seeds:
            for method in methods:
                if method == "random_search":
                    row = run_random_search(spec["factory"](), seed, budget, threshold=threshold)
                    row["gap_to_reference"] = _simple_regret(row["best_objective"], reference.objective_value, maximize)
                else:
                    row = run_bo(spec["factory"](), method, seed, budget, initial, reference, threshold=threshold)
                    row["regret_vs_reference"] = _simple_regret(row["best_objective"], reference.objective_value, maximize)
                    if maximize and row["best_objective"] > reference.objective_value and reference.verification_method == "numerical_approximate":
                        row["exceeds_numerical_reference"] = True  # reported, investigated, never clipped
                rows.append(row)
                if progress:
                    print(
                        f"  [{name} {method} seed={seed}] best={row['best_objective']:.6f} "
                        f"wall={row['wall_runtime_s']:.1f}s stop={row['stopping_reason']}",
                        flush=True,
                    )
        out[name] = {
            "reference": json.loads(reference.model_dump_json()),
            "budget": budget,
            "initial_design": initial,
            "num_candidates": NUM_CANDIDATES,
            "objective_threshold": threshold,
            "threshold_fraction": THRESHOLD_FRACTIONS.get(name, 0.90),
            "threshold_rule": THRESHOLD_RULE,
            "seeds": seeds,
            "methods": methods,
            "runs": rows,
        }
    return out


def aggregate(benchmark_block: dict) -> dict:
    """Descriptive aggregates per method (never a superiority claim)."""
    rows = benchmark_block["runs"]
    reference_value = benchmark_block["reference"]["objective_value"]
    maximize = True
    methods = benchmark_block["methods"]
    agg: dict = {"reference_value": reference_value, "per_method": {}}
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        best = [r["best_objective"] for r in mrows]
        regrets = [
            r.get("regret_vs_reference", r.get("gap_to_reference"))
            for r in mrows
        ]
        agg["per_method"][method] = {
            "n_runs": len(mrows),
            "best_mean": float(np.mean(best)),
            "best_median": float(np.median(best)),
            "best_stdev": float(np.std(best, ddof=1)) if len(best) > 1 else 0.0,
            "best_min": float(np.min(best)),
            "best_max": float(np.max(best)),
            "regret_mean": float(np.mean(regrets)),
            "regret_median": float(np.median(regrets)),
            "failed_attempts_total": sum(r.get("n_failed_attempts", 0) for r in mrows),
            "wall_runtime_mean_s": float(np.mean([r["wall_runtime_s"] for r in mrows])),
            "gp_fit_runtime_mean_s": float(np.mean([r.get("gp_fit_runtime_s", 0.0) for r in mrows])),
            "acquisition_runtime_mean_s": float(np.mean([r.get("acquisition_runtime_s", 0.0) for r in mrows])),
            "objective_runtime_mean_s": float(np.mean([r.get("objective_runtime_s", 0.0) for r in mrows])),
            "candidate_generation_runtime_mean_s": float(np.mean([r.get("candidate_generation_runtime_s", 0.0) for r in mrows])),
            "constraint_validation_runtime_mean_s": float(np.mean([r.get("constraint_validation_runtime_s", 0.0) for r in mrows])),
            "pool_rejection_rate_infeasible": float(
                np.mean([
                    r.get("pool_rejected_infeasible_total", r.get("n_rejected_infeasible", 0))
                    / max(1, r.get("pool_requested_total", 500 * 70))
                    for r in mrows
                ])
            ),
        }
        if method != "random_search":
            agg["per_method"][method].update({
                "accepted_updates_mean": float(np.mean([r["n_accepted_updates"] for r in mrows])),
                "block_visits_mean": float(np.mean([r["n_block_visits"] for r in mrows])),
                "skipped_blocks_total": sum(r["n_skipped_blocks"] for r in mrows),
                "cycles_completed_mean": float(np.mean([r["n_cycles_completed"] for r in mrows])),
                "cycles_without_improvement_mean": float(np.mean([r["n_cycles_without_improvement"] for r in mrows])),
                "scores_total": sum(r["n_acquisition_scores"] for r in mrows),
                "improvement_over_initial_incumbent_mean": float(np.mean([
                    r["improvement_over_initial_incumbent"] for r in mrows
                    if r.get("improvement_over_initial_incumbent") is not None
                ])),
                "out_of_block_improvement_events_total": int(np.sum([
                    r.get("out_of_block_improvement_evidence", {}).get(
                        "n_improvement_events_outside_visited_block", 0)
                    for r in mrows
                ])),
            })
        thresholds = [r.get("evaluations_to_threshold") for r in mrows]
        reached = [t for t in thresholds if t is not None]
        agg["per_method"][method]["evaluations_to_threshold_mean"] = (
            float(np.mean(reached)) if reached else None
        )
        agg["per_method"][method]["evaluations_to_threshold_reached"] = f"{len(reached)}/{len(thresholds)}"
    # Paired descriptive comparisons vs joint (same seed): counts only.
    if "joint" in methods:
        for method in methods:
            if method == "joint":
                continue
            wins = ties = losses = 0
            for seed_row_a, seed_row_b in zip(
                [r for r in rows if r["method"] == method], [r for r in rows if r["method"] == "joint"]
            ):
                if seed_row_a["seed"] != seed_row_b["seed"]:
                    continue
                if seed_row_a["best_objective"] > seed_row_b["best_objective"]:
                    wins += 1
                elif seed_row_a["best_objective"] == seed_row_b["best_objective"]:
                    ties += 1
                else:
                    losses += 1
            agg["per_method"][method]["vs_joint_same_seed"] = {"better": wins, "ties": ties, "worse": losses}
    return agg


def write_evidence_archive(output_dir: Path, data: dict) -> Path:
    """Compact durable archive under research_results/ (tracked in Git):
    per-seed histories, incumbents, runtimes, references, versions, commit."""
    archive_dir = Path("research_results")
    archive_dir.mkdir(exist_ok=True)
    compact = {
        "result_label": RESULT_LABEL,
        "phase": data.get("phase"),
        "configuration": {
            "evaluation_budget": data.get("evaluation_budget"),
            "initial_design": data.get("initial_design"),
            "num_acquisition_candidates": NUM_CANDIDATES,
            "seeds": data.get("seeds"),
        },
        "git_commit": git_commit(),
        "dependency_versions": {k: str(v) for k, v in data.get("dependency_versions", {}).items()},
        "benchmarks": {},
    }
    for name, block in data["benchmarks"].items():
        compact["benchmarks"][name] = {
            "reference": block["reference"],
            "budget": block["budget"],
            "initial_design": block["initial_design"],
            "objective_threshold": block.get("objective_threshold"),
            "threshold_fraction": block.get("threshold_fraction"),
            "threshold_rule": block.get("threshold_rule"),
            # Full per-run records (including iteration diagnostics, skipped
            # blocks, pool accounting, out-of-block evidence) — durable
            # evidence over brevity; still plain JSON, no binaries.
            "runs": [dict(r) for r in block["runs"]],
            "aggregate": block.get("aggregate"),
        }
    archive_path = archive_dir / f"m6_block_architecture_{data.get('phase', 'full')}_evidence.json"
    archive_path.write_text(json.dumps(compact, indent=1), encoding="utf-8")
    return archive_path


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    output_dir = Path(args[0]) if args else Path("results/m6_block_architecture_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot = "--pilot" in flags
    benchmark_filter = None
    method_filter = None
    seed_filter = None
    for flag in flags:
        if flag.startswith("--benchmarks="):
            benchmark_filter = flag.split("=", 1)[1].split(",")
        elif flag.startswith("--methods="):
            method_filter = flag.split("=", 1)[1].split(",")
        elif flag.startswith("--seeds="):
            seed_filter = [int(s) for s in flag.split("=", 1)[1].split(",")]

    if pilot:
        budget, initial, seeds = PILOT_BUDGET, PILOT_INITIAL, [0, 1]
        benchmark_names = ["separable_4d", "coupled_8d", "constrained_mixed_8d", "block_stagnation_4d"]
        methods = ["random_search", "joint", "alternating", "pairwise"]
        phase = "pilot"
    else:
        budget, initial, seeds = EVALUATION_BUDGET, INITIAL_DESIGN, list(range(10))
        benchmark_names = ["separable_4d", "coupled_8d", "constrained_mixed_8d", "block_stagnation_4d",
                           "separable_8d", "separable_16d"]
        methods = ["random_search", "joint", "alternating", "pairwise"]
        phase = "full"
    # Dimension study needs only joint/pairwise/RS (primary set covers alternating):
    if not pilot:
        study = {"separable_8d", "separable_16d"}
    else:
        study = set()
    if benchmark_filter:
        benchmark_names = [b for b in benchmark_names if b in benchmark_filter]
    if method_filter:
        methods = [m for m in methods if m in method_filter]
    if seed_filter:
        seeds = seed_filter

    data: dict = {
        "result_label": RESULT_LABEL,
        "phase": phase,
        "evaluation_budget": budget,
        "initial_design": initial,
        "seeds": seeds,
        "benchmarks": {},
    }
    for name in benchmark_names:
        block_methods = [m for m in methods if not (name in study and m == "alternating")]
        print(f"=== benchmark {name} (methods: {block_methods}) ===", flush=True)
        started = time.perf_counter()
        block = run_matrix([name], block_methods, seeds, budget, initial)
        block_runtime = time.perf_counter() - started
        data["benchmarks"][name] = block[name]
        data["benchmarks"][name]["aggregate"] = aggregate(block[name])
        data["benchmarks"][name]["matrix_wall_runtime_s"] = block_runtime
        print(f"    -> block wall: {block_runtime:.0f}s", flush=True)

    # Dependency versions (recorded by the engine in every run result as well):
    data["dependency_versions"] = {
        "numpy": __import__("numpy").__version__,
        "torch": __import__("torch").__version__,
        "gpytorch": __import__("gpytorch").__version__,
        "botorch": __import__("botorch").__version__,
        "scipy": __import__("scipy").__version__,
    }

    (output_dir / f"m6_{phase}_results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (output_dir / "provenance_register.json").write_text(
        build_default_provenance().model_dump_json(indent=2), encoding="utf-8"
    )
    archive = write_evidence_archive(output_dir, data)
    print(f"Results written to {output_dir}; durable archive: {archive}", flush=True)

    print(json.dumps({
        name: {
            "reference": block["reference"]["objective_value"],
            **{m: block["aggregate"]["per_method"][m]["best_mean"] for m in block["methods"]},
        }
        for name, block in data["benchmarks"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
