"""Generic synthetic BO validation across seeds (M5 instructions 28-29).

RESULT LABEL: generic synthetic BO validation — engine behavior on synthetic
benchmarks ONLY. This experiment makes NO radar claims, NO physical UAV
detection claims, and NO claim that Bayesian optimization is superior in
general. Unfavorable BO results are reported as measured (honest reporting).

Design (instruction 28):

* Benchmarks A (continuous quadratic, analytic exact reference 10.0),
  B (constrained mixed-variable, enumeration exact reference 11.0),
  C (multimodal, numerical approximate reference computed by the reference
  solver — never assumed).
* 10 seeds (0-9) per benchmark; BO and seeded random search run on the SAME
  evaluation budget (100 provisional synthetic budget) with the SAME seed.
  NOTE on comparability: BO's initial design and random search share the same
  seeded uniform feasible sampler, so with equal seeds RS's evaluation sequence
  begins with the BO initial design (documented deliberate choice; exact ties
  between methods are reported, not hidden).
* BO initial design: 30 seeded feasible random samples (experimental setting,
  not paper-derived); acquisition candidate pool 500 (Benzaghta Nc).
* Every run is reported individually; aggregates are reported alongside, never
  instead of the per-run records.

Outputs (instruction 29): best-objective-vs-evaluations curves, gap to the
independent reference, final distributions across seeds, feasibility
statistics, runtimes, and rejection statistics. All parameters are labeled
synthetic/experimental in the provenance register.

Run::

    python -m radar_bo.experiments.m5_synthetic_bo_validation [output_dir]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from radar_bo.config import build_default_provenance
from radar_bo.config.optimization_config import OptimizationConfig
from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
)
from radar_bo.optimization.bo.bo_loop import BayesianOptimizer
from radar_bo.optimization.bo.random_search import RandomSearchOptimizer
from radar_bo.optimization.bo.reference_solvers import (
    benchmark_a_reference,
    benchmark_b_reference,
    benchmark_c_reference,
)

RESULT_LABEL = (
    "generic synthetic BO validation - synthetic benchmarks only; no radar, "
    "no physical UAV, no corridor-surveillance claims; comparison statistics "
    "are descriptive (10 seeds), not proof of superiority"
)

EVALUATION_BUDGET = 100   # provisional synthetic budget (instruction 22)
INITIAL_DESIGN = 30       # experimental setting (instruction 10)
NUM_CANDIDATES = 500      # Benzaghta et al. Nc = 500
SEEDS = list(range(10))   # instruction 28: seeds 0-9


def _incumbent_curve(result) -> list[dict]:
    """Best-observed-objective-vs-evaluations points (x = cumulative evaluation
    attempts consumed so far)."""
    points = []
    eval_index = result.n_initial_evaluations
    if result.incumbent_history:
        points.append({"evaluations": eval_index, "best_objective": result.incumbent_history[0]})
    for i, value in enumerate(result.incumbent_history[1:]):
        points.append({"evaluations": eval_index + i + 1, "best_objective": value})
    return points


def _validate_run(problem, bo_result, rs_result, reference_value) -> dict:
    """Comparison-validity guards (instruction 33): same budget, feasible
    observations, incumbent evaluated. Raises on violation (fail loudly)."""
    if bo_result.n_initial_evaluations + bo_result.n_bo_evaluations != EVALUATION_BUDGET:
        raise AssertionError(
            f"BO budget violation: {bo_result.n_initial_evaluations} + {bo_result.n_bo_evaluations} "
            f"!= {EVALUATION_BUDGET}"
        )
    if rs_result.n_objective_evaluations != EVALUATION_BUDGET:
        raise AssertionError(
            f"Random-search budget violation: {rs_result.n_objective_evaluations} != {EVALUATION_BUDGET}"
        )
    for record in bo_result.observations:
        if not problem.is_feasible(np.asarray(record.config)):
            raise AssertionError(f"BO observation {record.evaluation_id} infeasible: {record.config}")
    for config, _, _ in rs_result.history:
        if not problem.is_feasible(np.asarray(config)):
            raise AssertionError(f"Random-search observation infeasible: {config}")
    if bo_result.observations:
        observed = {tuple(r.config) for r in bo_result.observations}
        if tuple(bo_result.best_config) not in observed:
            raise AssertionError("BO incumbent was never evaluated (contract violation).")
    return {
        "bo_all_observations_feasible": True,
        "rs_all_observations_feasible": True,
        "bo_incumbent_was_evaluated": bo_result.observations != [],
    }


def run_benchmark(spec: dict) -> dict:
    problem_factory = spec["factory"]
    reference = spec["reference"]()
    benchmark_rows: list[dict] = []
    for seed in SEEDS:
        problem = problem_factory()
        config = OptimizationConfig(
            evaluation_budget=EVALUATION_BUDGET,
            num_initial_observations=INITIAL_DESIGN,
            num_acquisition_candidates=NUM_CANDIDATES,
        )
        bo_result = BayesianOptimizer(
            problem,
            config,
            reference_objective=reference.objective_value,
            reference_method=reference.verification_method,
            seed=seed,
        ).optimize()

        rs_problem = problem_factory()  # fresh instance: its own evaluation counter
        rs_started = time.perf_counter()
        rs_result = RandomSearchOptimizer(rs_problem, EVALUATION_BUDGET, seed=seed).optimize()
        rs_runtime = time.perf_counter() - rs_started

        validity = _validate_run(problem, bo_result, rs_result, reference.objective_value)
        gap_bo = reference.objective_value - bo_result.best_objective  # maximize convention
        gap_rs = reference.objective_value - rs_result.best_objective
        if reference.verification_method != "analytic_exact":
            note = "reference is numerical_approximate: gaps may be negative if BO exceeds the approximate reference"
        else:
            note = "reference is analytic_exact (exact optimum)"
        benchmark_rows.append({
            "seed": seed,
            "bo": {
                "best_config": bo_result.best_config,
                "best_objective": bo_result.best_objective,
                "gap_to_reference": gap_bo,
                "n_initial_evaluations": bo_result.n_initial_evaluations,
                "n_bo_evaluations": bo_result.n_bo_evaluations,
                "n_failed_attempts": bo_result.n_failed_attempts,
                "n_acquisition_scores": bo_result.n_acquisition_scores,
                "pool_requests": bo_result.pool_requests,
                "pool_rejected_infeasible_total": bo_result.pool_rejected_infeasible_total,
                "pool_rejected_duplicate_evaluated_total": bo_result.pool_rejected_duplicate_evaluated_total,
                "pool_rejected_duplicate_within_pool_total": bo_result.pool_rejected_duplicate_within_pool_total,
                "gp_fit_runtime_s": bo_result.gp_fit_runtime_s,
                "acquisition_runtime_s": bo_result.acquisition_runtime_s,
                "objective_runtime_s": bo_result.objective_runtime_s,
                "total_runtime_s": bo_result.total_runtime_s,
                "stopping_reason": bo_result.stopping_reason,
                "incumbent_curve": _incumbent_curve(bo_result),
                "full_result": json.loads(bo_result.model_dump_json()),
            },
            "random_search": {
                "best_config": rs_result.best_config,
                "best_objective": rs_result.best_objective,
                "gap_to_reference": gap_rs,
                "n_objective_evaluations": rs_result.n_objective_evaluations,
                "n_generated": rs_result.n_generated,
                "n_rejected_infeasible": rs_result.n_rejected_infeasible,
                "n_skipped_duplicates": rs_result.n_skipped_duplicates,
                "runtime_s": rs_runtime,
                "stopping_reason": rs_result.stopping_reason,
            },
            "validity_checks": validity,
            "gap_note": note,
        })
        print(
            f"  [{spec['name']} seed={seed}] BO best={bo_result.best_objective:.6f} "
            f"(gap {gap_bo:+.4f}, {bo_result.stopping_reason}, {bo_result.total_runtime_s:.1f}s) | "
            f"RS best={rs_result.best_objective:.6f} (gap {gap_rs:+.4f}, {rs_runtime:.3f}s)",
            flush=True,
        )
    return {"reference": json.loads(reference.model_dump_json()), "runs": benchmark_rows}


def _aggregate(rows: list[dict]) -> dict:
    bo_best = [r["bo"]["best_objective"] for r in rows]
    rs_best = [r["random_search"]["best_objective"] for r in rows]
    bo_gap = [r["bo"]["gap_to_reference"] for r in rows]
    rs_gap = [r["random_search"]["gap_to_reference"] for r in rows]
    bo_wins = sum(1 for b, s in zip(bo_best, rs_best) if b > s)
    ties = sum(1 for b, s in zip(bo_best, rs_best) if b == s)
    try:
        from scipy.stats import wilcoxon

        if all(b == s for b, s in zip(bo_best, rs_best)):
            stat_p = {"statistic": None, "p_value": None, "note": "all paired differences are zero"}
        else:
            stat = wilcoxon(bo_best, rs_best)  # descriptive statistic, NOT a superiority claim
            stat_p = {"statistic": float(stat.statistic), "p_value": float(stat.pvalue),
                      "note": "descriptive paired Wilcoxon over 10 seeds; not a claim of superiority"}
    except Exception as exc:  # pragma: no cover - defensive
        stat_p = {"statistic": None, "p_value": None, "note": f"not computable: {exc}"}
    return {
        "n_seeds": len(rows),
        "bo_best_objective": {"mean": statistics.mean(bo_best), "median": statistics.median(bo_best),
                              "stdev": statistics.stdev(bo_best) if len(bo_best) > 1 else 0.0,
                              "min": min(bo_best), "max": max(bo_best)},
        "random_search_best_objective": {"mean": statistics.mean(rs_best), "median": statistics.median(rs_best),
                                         "stdev": statistics.stdev(rs_best) if len(rs_best) > 1 else 0.0,
                                         "min": min(rs_best), "max": max(rs_best)},
        "bo_gap_to_reference": {"mean": statistics.mean(bo_gap), "median": statistics.median(bo_gap)},
        "random_search_gap_to_reference": {"mean": statistics.mean(rs_gap), "median": statistics.median(rs_gap)},
        "bo_better_in_n_seeds": bo_wins,
        "ties_in_n_seeds": ties,
        "random_search_better_in_n_seeds": len(rows) - bo_wins - ties,
        "wilcoxon_paired": stat_p,
        "bo_total_runtime_s": {"mean": statistics.mean([r["bo"]["total_runtime_s"] for r in rows])},
        "random_search_runtime_s": {"mean": [r["random_search"]["runtime_s"] for r in rows] and
                                    statistics.mean([r["random_search"]["runtime_s"] for r in rows])},
        "failed_attempts_total_bo": sum(r["bo"]["n_failed_attempts"] for r in rows),
        "bo_pool_rejections_infeasible_total": sum(r["bo"]["pool_rejected_infeasible_total"] for r in rows),
        "rs_rejections_infeasible_total": sum(r["random_search"]["n_rejected_infeasible"] for r in rows),
        "stopping_reasons_bo": sorted({r["bo"]["stopping_reason"] for r in rows}),
    }


def write_evidence_archive(data: dict) -> Path:
    """Compact DURABLE evidence archive (tracked in Git under research_results/):
    per-seed incumbent histories, final incumbents, references, runtimes,
    dependency versions and the producing commit id. The full artifacts under
    results/ are gitignored and regenerable; this archive preserves the
    essential numerical evidence durably (M6 instruction 3.3/31.16)."""
    import subprocess

    archive_dir = Path("research_results")
    archive_dir.mkdir(exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    from radar_bo.optimization.bo.bo_loop import dependency_versions

    compact = {
        "result_label": RESULT_LABEL,
        "git_commit": commit,
        "dependency_versions": dependency_versions(),
        "configuration": {
            "evaluation_budget": EVALUATION_BUDGET,
            "initial_design_size": INITIAL_DESIGN,
            "num_acquisition_candidates": NUM_CANDIDATES,
            "seeds": SEEDS,
        },
        "benchmarks": {},
    }
    for name, block in data["benchmarks"].items():
        compact["benchmarks"][name] = {
            "reference": block["reference"],
            "runs": [
                {
                    "seed": r["seed"],
                    "best_config": r["bo"]["best_config"],
                    "best_objective": r["bo"]["best_objective"],
                    "gap_to_reference": r["bo"]["gap_to_reference"],
                    "stopping_reason": r["bo"]["stopping_reason"],
                    "n_failed_attempts": r["bo"]["n_failed_attempts"],
                    "n_acquisition_scores": r["bo"]["n_acquisition_scores"],
                    "incumbent_history": [p["best_objective"] for p in r["bo"]["incumbent_curve"]],
                    "gp_fit_runtime_s": r["bo"]["gp_fit_runtime_s"],
                    "acquisition_runtime_s": r["bo"]["acquisition_runtime_s"],
                    "objective_runtime_s": r["bo"]["objective_runtime_s"],
                    "total_runtime_s": r["bo"]["total_runtime_s"],
                    "random_search": {
                        "best_config": r["random_search"]["best_config"],
                        "best_objective": r["random_search"]["best_objective"],
                        "gap_to_reference": r["random_search"]["gap_to_reference"],
                        "runtime_s": r["random_search"]["runtime_s"],
                    },
                }
                for r in block["runs"]
            ],
            "aggregate": block["aggregate"],
        }
    archive_path = archive_dir / "m5_synthetic_bo_validation.json"
    archive_path.write_text(json.dumps(compact, indent=1), encoding="utf-8")
    return archive_path


def _summary_md(data: dict) -> str:
    lines = [
        "# M5 generic synthetic BO validation (seeds 0-9)",
        "",
        f"**Result label:** {RESULT_LABEL}",
        "",
        f"- Evaluation budget: {EVALUATION_BUDGET} objective evaluations (provisional synthetic; failures consume budget)",
        f"- BO initial design: {INITIAL_DESIGN} seeded feasible samples (experimental, not paper-derived)",
        f"- Acquisition candidate pool: {NUM_CANDIDATES} per iteration (Benzaghta Nc; pool scoring consumes NO budget)",
        "- Comparison method: seeded uniform random search on the identical budget and seed",
        "",
        "## Comparison design (read before interpreting exact ties)",
        "",
        "BO's initial design and random search use the SAME seeded uniform feasible sampler",
        " (`CandidateGenerator._sample` == `RandomSearchOptimizer._sample` logic) with the SAME",
        " seed per run. Consequence: with equal seeds, random search's evaluation sequence BEGINS",
        " with the BO initial design and continues with further uniform samples, while BO continues",
        " with EI-selected candidates instead. Exact BO/RS ties therefore occur when neither",
        " method's additional evaluations beat the shared initial design's best — a deliberate",
        " comparability choice (instruction 28: comparable initial designs), NOT a bug and NOT",
        " evidence that BO equals random search in general.",
        "",
    ]
    for name, block in data["benchmarks"].items():
        ref = block["reference"]
        lines += [
            f"## Benchmark {name}",
            "",
            f"- Reference: method=`{ref['verification_method']}`, value={ref['objective_value']:.9f}, "
            f"config={[round(v, 6) for v in ref['config']]}, feasible={ref['feasible']}",
            "",
            "| seed | BO best | BO gap | BO stop | BO s | RS best | RS gap | RS infeasible rejected | RS s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in block["runs"]:
            lines.append(
                f"| {r['seed']} | {r['bo']['best_objective']:.6f} | {r['bo']['gap_to_reference']:+.4f} "
                f"| {r['bo']['stopping_reason']} | {r['bo']['total_runtime_s']:.1f} "
                f"| {r['random_search']['best_objective']:.6f} | {r['random_search']['gap_to_reference']:+.4f} "
                f"| {r['random_search']['n_rejected_infeasible']} | {r['random_search']['runtime_s']:.3f} |"
            )
        agg = block["aggregate"]
        lines += [
            "",
            f"- BO best objective across seeds: mean={agg['bo_best_objective']['mean']:.6f}, "
            f"median={agg['bo_best_objective']['median']:.6f}, stdev={agg['bo_best_objective']['stdev']:.6f}, "
            f"range=[{agg['bo_best_objective']['min']:.6f}, {agg['bo_best_objective']['max']:.6f}]",
            f"- Random search best objective across seeds: mean={agg['random_search_best_objective']['mean']:.6f}, "
            f"median={agg['random_search_best_objective']['median']:.6f}, "
            f"stdev={agg['random_search_best_objective']['stdev']:.6f}",
            f"- Paired outcome: BO better in {agg['bo_better_in_n_seeds']}/{agg['n_seeds']} seeds, "
            f"ties {agg['ties_in_n_seeds']}, random search better {agg['random_search_better_in_n_seeds']} "
            f"(descriptive; no superiority claim)",
            f"- BO failed objective attempts (total): {agg['failed_attempts_total_bo']}; "
            f"BO pool infeasible rejections (total): {agg['bo_pool_rejections_infeasible_total']}; "
            f"RS infeasible rejections (total): {agg['rs_rejections_infeasible_total']}",
            f"- BO mean total runtime: {agg['bo_total_runtime_s']['mean']:.1f} s/run; "
            f"RS mean runtime: {agg['random_search_runtime_s']['mean']:.4f} s/run",
            "",
        ]
    lines += [
        "## Interpretation guard",
        "",
        "- Termination at the budget is NOT proof of global convergence.",
        "- For benchmark C the reference is numerical_approximate: BO values may legitimately exceed it.",
        "- All settings are synthetic/experimental; see provenance_register.json in this directory.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    output_dir = Path(argv[1]) if len(argv) > 1 else Path("results/m5_synthetic_bo_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_c_reference_value = benchmark_c_reference(seed=0)  # computed by the solver, not assumed
    specs: dict[str, dict] = {
        "A_continuous_quadratic": {
            "name": "A_continuous_quadratic",
            "factory": BenchmarkAQuadratic,
            "reference": benchmark_a_reference,
        },
        "B_mixed_constrained": {
            "name": "B_mixed_constrained",
            "factory": BenchmarkBMixedConstrained,
            "reference": benchmark_b_reference,
        },
        "C_multimodal": {
            "name": "C_multimodal",
            "factory": BenchmarkCMultimodal,
            "reference": lambda: benchmark_c_reference_value,
        },
    }

    data: dict = {
        "result_label": RESULT_LABEL,
        "evaluation_budget": EVALUATION_BUDGET,
        "initial_design_size": INITIAL_DESIGN,
        "num_acquisition_candidates": NUM_CANDIDATES,
        "seeds": SEEDS,
        "benchmarks": {},
    }
    for name, spec in specs.items():
        print(f"Running benchmark {name} ...", flush=True)
        block = run_benchmark(spec)
        block["aggregate"] = _aggregate(block["runs"])
        data["benchmarks"][name] = block

    (output_dir / "m5_synthetic_bo_validation_results.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_summary_md(data), encoding="utf-8")
    (output_dir / "provenance_register.json").write_text(
        build_default_provenance().model_dump_json(indent=2), encoding="utf-8"
    )
    write_evidence_archive(data)

    print(json.dumps({
        name: {
            "reference_value": block["reference"]["objective_value"],
            "bo_mean_best": block["aggregate"]["bo_best_objective"]["mean"],
            "rs_mean_best": block["aggregate"]["random_search_best_objective"]["mean"],
            "bo_better_seeds": block["aggregate"]["bo_better_in_n_seeds"],
        }
        for name, block in data["benchmarks"].items()
    }, indent=2))
    print(f"Results written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
