"""Reproducible report generation (M7 item 27).

Generates a human-readable Markdown report FROM STORED NUMERICAL DATA ONLY
(experiment store summary + run records + configuration). Regenerating a
report never evaluates an objective (integration-tested). The report contains:
experiment scope, configuration, benchmark definitions, algorithms, budgets,
reference solutions, results, statistical uncertainty, runtime analysis,
failures and omitted runs, limitations, and evidence-file locations.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def generate_report(store_dir: str | Path) -> Path:
    """Build reports/report.md from the persisted experiment store (no re-evaluation)."""
    store_dir = Path(store_dir)
    config_data = json.loads((store_dir / "experiment_config.json").read_text(encoding="utf-8"))
    config = config_data["config"]["model_dump"]
    compatibility = config_data["compatibility"]
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((store_dir / "summary.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# Experiment report: {config['experiment_id']}")
    lines.append("")
    lines.append(f"**Label:** {config['result_label']}")
    lines.append("")
    lines.append(f"**Description:** {config['description']}")
    lines.append("")
    lines.append("## Scope and configuration")
    lines.append("")
    lines.append(f"- Experiment id: `{config['experiment_id']}` (config fingerprint `{config_data['config']['fingerprint'][:16]}`)")
    lines.append(f"- Benchmarks: {', '.join(f'`{b}`' for b in config['benchmark_ids'])}")
    lines.append(f"- Algorithms: {', '.join(f'`{a}`' for a in config['algorithm_ids'])}")
    lines.append(f"- Seeds: {config['seeds']}")
    lines.append(f"- Total objective-evaluation budget per run: {config['stopping']['evaluation_budget']} (ATTEMPTED evaluations; never per block)")
    lines.append(f"- Initial design: {config['initial_design']['size']} (shared across methods per seed; verified from stored records: "
                 f"{'PASS' if summary['shared_initialization_verification']['verified'] else 'FAILED — see issues'})")
    lines.append(f"- Candidate pool: {config['candidate_pool']['num_acquisition_candidates']} per acquisition step")
    lines.append(f"- GP: {config['gp']['surrogate']} / matern_5_2 (nu 2.5) / float64 CPU / inferred noise")
    lines.append(f"- Source commit: `{compatibility['source_commit'][:12]}`; torch {compatibility['dependency_versions']['torch']}, "
                 f"botorch {compatibility['dependency_versions']['botorch']}")
    lines.append("")
    lines.append("## Benchmark definitions and reference solutions")
    lines.append("")
    lines.append("| benchmark | dim | definition | constraints | reference | verification |")
    lines.append("|---|---|---|---|---|---|")
    for spec in manifest["runs"]:
        if spec["algorithm_id"] != "reference":
            continue
        rec_path = store_dir / "runs" / f"{spec['run_id']}.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        lines.append(
            f"| {spec['benchmark_id']} | {spec['dimension']} | (see benchmark registry — versioned implementation) "
            f"| (see benchmark registry) | {_fmt(rec.get('reference_objective'))} | {rec.get('reference_method')} |"
        )
    lines.append("")
    lines.append("## Algorithms")
    lines.append("")
    for aid in config["algorithm_ids"]:
        lines.append(f"- `{aid}` — registered in `algorithm_registry` (version recorded per run; existing M5/M6 implementations, no duplication)")
    lines.append("")
    lines.append("## Run statuses (item 12: omissions are explicit)")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---|")
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    skipped = [r for r in manifest["runs"] if r["status"] in ("failed", "skipped")]
    if skipped:
        lines.append("| run | benchmark | algorithm | seed | status | reason |")
        lines.append("|---|---|---|---|---|---|")
        for r in skipped:
            lines.append(f"| {r['run_id'][:12]} | {r['benchmark_id']} | {r['algorithm_id']} | {r.get('seed')} | {r['status']} | {r.get('reason')} |")
    lines.append("")
    lines.append("## Results (per benchmark x method)")
    lines.append("")
    analysis = summary["analysis"]
    excluded = analysis.get("incomplete_records_excluded", 0)
    if excluded:
        lines.append(f"**Note:** {excluded} incomplete record(s) excluded from aggregation (reported, not silently pooled).")
        lines.append("")
    benchmark_blocks = {k: v for k, v in analysis.items() if isinstance(v, dict) and "per_method" in v}
    for benchmark_id, block in sorted(benchmark_blocks.items()):
        lines.append(f"### {benchmark_id}")
        lines.append("")
        lines.append("| method | n | final mean | final median | final stdev | IQR | gap mean | gap label | closure mean | threshold summary | failures/run | wall mean s |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for method, stats in sorted(block["per_method"].items()):
            thr = "; ".join(f"closure {t['fraction']}: {t['runs_reached']}" for t in stats["threshold_analysis"])
            lines.append(
                f"| {method} | {stats['final_objective']['n']} | {_fmt(stats['final_objective']['mean'])} "
                f"| {_fmt(stats['final_objective']['median'])} | {_fmt(stats['final_objective']['stdev'])} "
                f"| {_fmt(stats['final_objective']['iqr'])} | {_fmt(stats['gap']['mean'])} | {stats['gap']['label']} "
                f"| {_fmt(stats['normalized_gap_closure']['mean'])} | {thr} "
                f"| {_fmt(stats['evaluation_failure_frequency'])} | {_fmt(stats['wall_runtime_mean_s'], 2)} |"
            )
        lines.append("")
        if block["paired_comparisons"]:
            lines.append("**Paired comparisons (seed-level pairing preserved; item 20-22):**")
            lines.append("")
            for pair_name, pair in sorted(block["paired_comparisons"].items()):
                wil = pair.get("wilcoxon") or {}
                ci = pair.get("bootstrap_ci") or {}
                lines.append(
                    f"- `{pair_name}`: n={pair['n_pairs']}, mean diff {_fmt(pair['mean_difference'])}, "
                    f"median diff {_fmt(pair['median_difference'])}, wins/losses/ties "
                    f"{pair['wins']}/{pair['losses']}/{pair['ties']} (tie tol {pair['tie_tolerance']:.0e}), "
                    f"d_z {_fmt(pair.get('effect_size_dz'))}, bootstrap 95% CI "
                    f"[{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}] (B={ci.get('replications')}, seed {ci.get('bootstrap_seed')}), "
                    f"Wilcoxon p={_fmt(wil.get('p_value'))} (Holm-adjusted {_fmt(wil.get('p_value_holm_adjusted'))}, n used {wil.get('n_used')})"
                )
            lines.append("")
    lines.append("## Runtime analysis (item 23: partial attribution, unattributed reported)")
    lines.append("")
    lines.append("| benchmark | method | GP fit s | acquisition s | objective s | candidate gen s | feasibility s | wall s | unattributed s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for benchmark_id, block in sorted(benchmark_blocks.items()):
        for method, stats in sorted(block["per_method"].items()):
            rt = stats.get("runtime_decomposition_mean_s") or {}
            lines.append(
                f"| {benchmark_id} | {method} | {_fmt(rt.get('gp_fit_runtime_s'), 2)} | {_fmt(rt.get('acquisition_runtime_s'), 2)} "
                f"| {_fmt(rt.get('objective_runtime_s'), 4)} | {_fmt(rt.get('candidate_generation_runtime_s'), 2)} "
                f"| {_fmt(rt.get('constraint_validation_runtime_s'), 2)} | {_fmt(stats.get('wall_runtime_mean_s'), 2)} "
                f"| {_fmt(stats.get('unattributed_overhead_mean_s'), 2)} |"
            )
    lines.append("")
    lines.append("Component timers are disjoint by construction; their sum is a PARTIAL attribution. "
                 "The unattributed remainder (tensor conversion, dataset insertion, incumbent tracking, GP "
                 "training-tensor preparation, diagnostics, serialization) is reported and NOT claimed to be "
                 "identified (item 23).")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Descriptive statistics over a small seed sample; uncertainty intervals from bootstrap are NOT evidence of universal superiority (item 21).")
    lines.append("- Threshold metrics depend on the frozen threshold definitions in the configuration (item 18).")
    lines.append("- Archived pre-M7 records lack per-iteration attempted-evaluation/wall timestamps; attempted-axis and wall-clock convergence are exact only for M7 runner records (item 17).")
    lines.append("- Reference gaps on numerical_approximate references are NOT proven optimality regret; values exceeding such references are retained and investigated (item 15).")
    lines.append("")
    lines.append("## Evidence files")
    lines.append("")
    lines.append(f"- Experiment store: `{store_dir}` (config, manifest, per-run records, summary)")
    lines.append(f"- This report: `{store_dir / 'reports' / 'report.md'}` (regenerable from stored data only)")
    lines.append(f"- Figures: `{store_dir / 'figures'}` (if generated)")
    lines.append("- Durable tracked archives: `research_results/` (checksum manifest `ARCHIVE_CHECKSUMS.json`)")

    reports_dir = store_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
