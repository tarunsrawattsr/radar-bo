"""Paired M5/M6 comparative summaries (M7 deliverable 10; items 19-22).

DERIVED analysis over the durable archives (no optimizer runs):

* M6 archive: per-benchmark paired comparisons vs joint_bo (seed-level pairing
  preserved): per-seed differences, wins/losses/ties (tie tol 1e-9), paired
  d_z, bootstrap CIs (resampling seed-level pairs, B=10000, seed 2026),
  Wilcoxon signed-rank with Holm correction across the per-benchmark family;
* M5 archive: paired BO-vs-RS per-seed differences on benchmarks A/B/C.

Written to research_results/m7_paired_comparative_summary.json. Historical
archives are read-only inputs. All statistics are DESCRIPTIVE (10 seeds); no
superiority claims.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from radar_bo.experiments.framework.experiment_config import (
    ExperimentConfig,
    StatisticalAnalysisSettings,
    StoppingSettings,
)
from radar_bo.experiments.framework.metrics import (
    bootstrap_ci,
    describe,
    holm_correction,
    objective_gap,
    paired_differences,
    paired_effect_size,
    wilcoxon_signed_rank,
)


def analysis_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="m7_paired_analysis",
        description="derived paired statistical analysis of archived M5/M6 results",
        result_label="generic synthetic derived analysis; no radar claims; descriptive only",
        benchmark_ids=["benchmark_a"],
        algorithm_ids=["joint_bo"],
        seeds=[0],
        stopping=StoppingSettings(evaluation_budget=1),
        statistics=StatisticalAnalysisSettings(
            bootstrap_replications=10000,
            bootstrap_seed=2026,
            paired_tie_tolerance=1e-9,
            multiple_comparison_correction="holm",
        ),
        output_dir="results/m7_paired_analysis",
    )


def paired_block(by_seed_a: dict[int, float], by_seed_b: dict[int, float], settings) -> dict:
    paired = paired_differences(by_seed_a, by_seed_b, settings.paired_tie_tolerance)
    diffs = paired.pop("differences")
    paired["effect_size_dz"] = paired_effect_size(diffs)
    paired["bootstrap_ci"] = bootstrap_ci(
        diffs, settings.bootstrap_replications, settings.bootstrap_seed, settings.confidence_level
    )
    paired["wilcoxon"] = wilcoxon_signed_rank(diffs)
    return paired


def analyze_m6() -> dict:
    archive = json.loads(Path("research_results/m6_block_architecture_full_evidence.json").read_text(encoding="utf-8"))
    settings = analysis_config().statistics
    result: dict = {}
    p_values: list[float] = []
    test_keys: list[str] = []
    for benchmark_id, block in sorted(archive["benchmarks"].items()):
        by_method: dict[str, dict[int, float]] = {}
        for run in block["runs"]:
            by_method.setdefault(run["method"], {})[run["seed"]] = run["best_objective"]
        if "joint" not in by_method:
            continue
        per_benchmark: dict = {"reference": block["reference"]["objective_value"], "pairs": {}}
        family_p: list[float] = []
        family_keys: list[str] = []
        for method in sorted(by_method):
            if method == "joint":
                continue
            pair = paired_block(by_method[method], by_method["joint"], settings)
            # gap statistics per method (item 15: reference-gap labeling)
            ref = block["reference"]["objective_value"]
            approx = block["reference"]["verification_method"] == "numerical_approximate"
            gaps = [objective_gap(v, ref, maximize=True, reference_is_exact=not approx)
                    for v in by_method[method].values()]
            pair["gap_label"] = gaps[0]["label"]
            pair["gap_mean"] = sum(g["value"] for g in gaps) / len(gaps)
            pair["exceeds_reference_runs"] = sum(1 for g in gaps if g["exceeds_reference"])
            per_benchmark["pairs"][f"{method}_vs_joint"] = pair
            if pair["wilcoxon"].get("valid"):
                family_p.append(pair["wilcoxon"]["p_value"])
                family_keys.append(f"{method}_vs_joint")
        if family_p and settings.multiple_comparison_correction == "holm":
            adjusted = holm_correction(family_p)
            per_benchmark["holm_adjusted"] = dict(zip(family_keys, adjusted))
            for key, p in zip(family_keys, adjusted):
                per_benchmark["pairs"][key]["wilcoxon"]["p_value_holm_adjusted"] = p
            p_values.extend(family_p)
            test_keys.extend(f"{benchmark_id}:{k}" for k in family_keys)
        result[benchmark_id] = per_benchmark
    return {
        "archive": "research_results/m6_block_architecture_full_evidence.json",
        "per_benchmark": result,
        "family": {
            "n_tests": len(p_values),
            "correction": "Holm-Bonferroni across the per-benchmark family of same-seed paired tests",
            "note": "exploratory descriptive analysis; p-values complement effect sizes and CIs, never replace them",
        },
    }


def analyze_m5() -> dict:
    archive = json.loads(Path("research_results/m5_synthetic_bo_validation.json").read_text(encoding="utf-8"))
    settings = analysis_config().statistics
    result: dict = {}
    for benchmark_id, block in sorted(archive["benchmarks"].items()):
        bo_by_seed: dict[int, float] = {}
        rs_by_seed: dict[int, float] = {}
        for run in block["runs"]:
            bo_by_seed[run["seed"]] = run["best_objective"]
            rs = run.get("random_search") or {}
            if rs.get("best_objective") is not None:
                rs_by_seed[run["seed"]] = rs["best_objective"]
        ref = block["reference"]["objective_value"]
        approx = block["reference"]["verification_method"] == "numerical_approximate"
        pair = paired_block(bo_by_seed, rs_by_seed, settings)
        pair["gap_label"] = "reference_gap" if approx else "simple_regret"
        result[benchmark_id] = {
            "reference": ref,
            "reference_verification_method": block["reference"]["verification_method"],
            "bo_vs_random_search": pair,
            "bo_final_objective": describe(list(bo_by_seed.values())),
            "rs_final_objective": describe(list(rs_by_seed.values())),
        }
    return {
        "archive": "research_results/m5_synthetic_bo_validation.json",
        "per_benchmark": result,
        "note": "paired BO-vs-RS differences per seed (the M5 runs share seeds and budgets)",
    }


def main() -> int:
    artifact = {
        "result_label": (
            "M7 derived paired comparative summaries over the durable M5/M6 archives; "
            "descriptive statistics over 10 seeds; NO superiority claims; seed-level "
            "pairing preserved (item 20); bootstrap resamples complete seed pairs (item 21)"
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "methods": {
            "tie_tolerance": 1e-9,
            "bootstrap": "percentile CI on the mean paired difference; B=10000; seed 2026; seed-level pairs resampled as units",
            "wilcoxon": "two-sided signed-rank on non-zero paired differences (zero_method wilcox); Holm-corrected across the presented family",
        },
        "m6_comparisons": analyze_m6(),
        "m5_comparisons": analyze_m5(),
    }
    out = Path("research_results/m7_paired_comparative_summary.json")
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    # console digest
    for benchmark_id, block in artifact["m6_comparisons"]["per_benchmark"].items():
        print(f"== {benchmark_id} (ref {block['reference']}) ==")
        for pair_name, pair in block["pairs"].items():
            wil = pair["wilcoxon"]
            ci = pair["bootstrap_ci"]
            print(f"  {pair_name}: mean diff {pair['mean_difference']:.4f} | W/L/T {pair['wins']}/{pair['losses']}/{pair['ties']} "
                  f"| d_z {pair['effect_size_dz']} | CI [{ci['lower']:.3f}, {ci['upper']:.3f}] "
                  f"| p {wil.get('p_value'):.4f} holm {wil.get('p_value_holm_adjusted', 0):.4f}")
    for benchmark_id, block in artifact["m5_comparisons"]["per_benchmark"].items():
        pair = block["bo_vs_random_search"]
        print(f"== M5 {benchmark_id}: BO vs RS mean diff {pair['mean_difference']:.4f} "
              f"(W/L/T {pair['wins']}/{pair['losses']}/{pair['ties']})")
    print(f"Artifact: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
