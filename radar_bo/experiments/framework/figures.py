"""Standardized research figures (M7 item 28).

Reads STORED run records (experiment store or M6 durable archive) and produces
the standardized figure set with units, legends, sample sizes and explicit
uncertainty definitions. Every figure is labeled as a generic synthetic
experiment (no radar claims). Approximate reference values are marked
"(numerical_approximate reference)".
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = "generic synthetic experiment (no radar claims)"
METHOD_COLORS = {
    "random_search": "#7f7f7f",
    "joint_bo": "#1f77b4",
    "alternating_bo": "#ff7f0e",
    "pairwise_bo": "#2ca02c",
    "joint": "#1f77b4",
    "alternating": "#ff7f0e",
    "pairwise": "#2ca02c",
}


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#9467bd")


def _reference_caption(block_or_record: dict) -> str:
    if block_or_record.get("reference_method") == "numerical_approximate":
        return " (numerical_approximate reference)"
    return ""


# ---- loading helpers ------------------------------------------------------------
def load_records_from_store(store_dir: str | Path) -> list[dict]:
    store_dir = Path(store_dir)
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for spec in manifest["runs"]:
        if spec["status"] == "completed" and spec.get("seed") is not None:
            records.append(json.loads((store_dir / "runs" / f"{spec['run_id']}.json").read_text(encoding="utf-8")))
    return records


def load_records_from_m6_archive(archive_path: str | Path) -> list[dict]:
    """Load + normalize M6 durable-archive rows into the framework record shape.

    Normalizations (derived; documented in m7_m6_review_corrections.json):
    * flat runtime fields -> runtime_measurements;
    * pool fields -> evaluation_counts;
    * method -> algorithm_id;
    * random-search rows: their stored sequence is converted to a running-max
      best-so-far curve (the archive mislabels it 'incumbent_history'; see the
      review artifact) so convergence figures use consistent semantics.
    """
    data = json.loads(Path(archive_path).read_text(encoding="utf-8"))
    records = []
    for benchmark_id, block in data["benchmarks"].items():
        for run in block["runs"]:
            row = dict(run)
            row["benchmark_id"] = benchmark_id
            row["algorithm_id"] = row.pop("method", None)
            # map archived M6 field names onto the framework record shape
            if "best_objective" in row:
                row["final_objective"] = row.pop("best_objective")
            if "best_config" in row:
                row["final_incumbent_config"] = row.pop("best_config")
            if row["algorithm_id"] == "random_search":
                row["evaluation_counts"] = {"n_objective_evaluations": len(row.get("incumbent_history") or [])}
            row.setdefault("reference_objective", block["reference"]["objective_value"])
            row.setdefault("reference_method", block["reference"]["verification_method"])
            row["runtime_measurements"] = {
                "gp_fit_runtime_s": row.get("gp_fit_runtime_s"),
                "acquisition_runtime_s": row.get("acquisition_runtime_s"),
                "objective_runtime_s": row.get("objective_runtime_s"),
                "candidate_generation_runtime_s": row.get("candidate_generation_runtime_s"),
                "constraint_validation_runtime_s": row.get("constraint_validation_runtime_s"),
                "wall_runtime_s": row.get("wall_runtime_s"),
            }
            pool_counts = {
                "pool_requested_total": row.get("pool_requested_total"),
                "pool_delivered_total": row.get("pool_delivered_total"),
                "pool_rejected_infeasible_total": row.get("pool_rejected_infeasible_total"),
            }
            if row["algorithm_id"] == "random_search":
                row["evaluation_counts"] = {
                    "n_objective_evaluations": len(row.get("incumbent_history") or []),
                    **pool_counts,
                }
            else:
                row["evaluation_counts"] = pool_counts
            if row["algorithm_id"] == "random_search":
                sequence = row.get("incumbent_history") or []
                running_best: list[float] = []
                current = float("-inf")
                for v in sequence:
                    current = max(current, float(v))
                    running_best.append(current)
                row["incumbent_history"] = running_best
                row["iteration_diagnostics"] = []
            records.append(row)
    return records


# ---- figures --------------------------------------------------------------------
def fig_convergence(records: list[dict], benchmark_id: str, out_dir: Path,
                    axis: str = "successful") -> None:
    """Convergence vs successful evaluations (exact for all records).

    For M7 records the attempted axis is also available (item 17); pre-M7
    records only support the successful axis (documented).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["benchmark_id"] == benchmark_id:
            groups.setdefault(r.get("algorithm_id", r.get("method")), []).append(r)
    for method, rows in sorted(groups.items()):
        curves = []
        for r in rows:
            if axis == "attempted":
                diag = [d for d in (r.get("iteration_diagnostics") or []) if d.get("objective_value") is not None]
                xs = [d.get("attempted_evaluations") for d in diag]
                if not xs or any(x is None for x in xs):
                    continue
                ys = r["incumbent_history"][1:1 + len(xs)]
            else:
                xs = list(range(1, len(r["incumbent_history"]) + 1))
                ys = r["incumbent_history"]
            if not xs:
                continue
            curves.append((np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
        for xs, ys in curves:
            ax.plot(xs, ys, color=_method_color(method), alpha=0.25, linewidth=0.8)
        if curves:
            grid = np.linspace(min(c[0].min() for c in curves), max(c[0].max() for c in curves), 50)
            interp = [np.interp(grid, c[0], c[1]) for c in curves]
            ax.plot(grid, np.mean(interp, axis=0), color=_method_color(method),
                    label=f"{method} (n={len(curves)})", linewidth=2)
    if groups:
        ref = next(iter(groups.values()))[0].get("reference_objective")
        approx = _reference_caption(next(iter(groups.values()))[0])
        if ref is not None:
            ax.axhline(ref, color="k", linestyle="--", linewidth=1, label=f"reference {ref}{approx}")
    ax.set_xlabel(f"{axis} objective evaluations (count)")
    ax.set_ylabel("best observed objective (benchmark units)")
    ax.set_title(f"Convergence — {benchmark_id} ({axis}-evaluation axis)\n{LABEL}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"convergence_vs_{axis}_evals_{benchmark_id}.png", dpi=150)
    plt.close(fig)


def fig_convergence_wall(records: list[dict], benchmark_id: str, out_dir: Path) -> None:
    """Convergence vs wall-clock (exact for M7 records via per-iteration timestamps)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["benchmark_id"] == benchmark_id:
            groups.setdefault(r.get("algorithm_id", r.get("method")), []).append(r)
    plotted_methods: dict[str, int] = {}
    for method, rows in sorted(groups.items()):
        for r in rows:
            diag = [d for d in (r.get("iteration_diagnostics") or []) if d.get("objective_value") is not None]
            xs = [d.get("elapsed_runtime_s") for d in diag]
            if not xs or any(x is None for x in xs):
                continue  # pre-M7 record: wall-clock convergence not archived
            ys = r["incumbent_history"][1:1 + len(xs)]
            ax.plot(xs, ys, color=_method_color(method), alpha=0.5, linewidth=1.0)
            plotted_methods[method] = plotted_methods.get(method, 0) + 1
    if plotted_methods:
        handles = [plt.Line2D([0], [0], color=_method_color(m)) for m in plotted_methods]
        ax.legend(handles, [f"{m} (n={n})" for m, n in plotted_methods.items()], fontsize=8)
    ax.set_xlabel("wall-clock time (seconds)")
    ax.set_ylabel("best observed objective (benchmark units)")
    ax.set_title(f"Convergence — {benchmark_id} (wall-clock axis, M7 records only)\n{LABEL}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"convergence_vs_wall_{benchmark_id}.png", dpi=150)
    plt.close(fig)


def fig_gap_distributions(records: list[dict], out_dir: Path) -> None:
    """Final reference-gap distributions per benchmark x method (sample sizes labeled)."""
    benchmarks = sorted({r["benchmark_id"] for r in records})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark_id in zip(axes[0], benchmarks):
        methods = sorted({r.get("algorithm_id", r.get("method")) for r in records if r["benchmark_id"] == benchmark_id})
        data, labels = [], []
        for method in methods:
            gaps = [
                r["reference_objective"] - r["final_objective"]
                for r in records
                if r["benchmark_id"] == benchmark_id
                and r.get("algorithm_id", r.get("method")) == method
                and r.get("final_objective") is not None
            ]
            if gaps:
                data.append(gaps)
                labels.append(f"{method}\n(n={len(gaps)})")
        if data:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel("final reference gap (benchmark units)")
        ax.set_title(benchmark_id, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"Final gap distributions across seeds — {LABEL}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "gap_distributions.png", dpi=150)
    plt.close(fig)


def fig_paired_differences(records: list[dict], out_dir: Path, tie_tolerance: float = 1e-9) -> None:
    """Per-seed paired differences vs joint_bo (pairing preserved; item 20)."""
    benchmarks = sorted({r["benchmark_id"] for r in records})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark_id in zip(axes[0], benchmarks):
        by_method: dict[str, dict[int, float]] = {}
        for r in records:
            if r["benchmark_id"] == benchmark_id and r.get("final_objective") is not None:
                by_method.setdefault(r.get("algorithm_id", r.get("method")), {})[r["seed"]] = r["final_objective"]
        baseline = "joint_bo" if "joint_bo" in by_method else sorted(by_method)[0]
        for method in sorted(by_method):
            if method == baseline:
                continue
            seeds = sorted(set(by_method[method]) & set(by_method[baseline]))
            diffs = [by_method[method][s] - by_method[baseline][s] for s in seeds]
            ax.scatter(seeds, diffs, label=f"{method} - {baseline} (n={len(diffs)})", alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
        ax.axhline(tie_tolerance, color="k", linewidth=0.5, linestyle="--")
        ax.axhline(-tie_tolerance, color="k", linewidth=0.5, linestyle="--")
        ax.set_xlabel("seed")
        ax.set_ylabel(f"paired difference vs {baseline} (benchmark units)")
        ax.set_title(benchmark_id, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Paired per-seed differences (tie tol {tie_tolerance:.0e}) — {LABEL}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_differences.png", dpi=150)
    plt.close(fig)


def fig_threshold_success(records: list[dict], out_dir: Path, fractions: list[float]) -> None:
    """Threshold success rates with explicit unreached handling (item 18)."""
    benchmarks = sorted({r["benchmark_id"] for r in records})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5), squeeze=False)
    for ax, benchmark_id in zip(axes[0], benchmarks):
        by_method: dict[str, list[dict]] = {}
        for r in records:
            if r["benchmark_id"] == benchmark_id:
                by_method.setdefault(r.get("algorithm_id", r.get("method")), []).append(r)
        width = 0.8 / max(1, len(fractions))
        for fi, fraction in enumerate(fractions):
            methods = sorted(by_method)
            rates, labels = [], []
            for method in methods:
                reached = 0
                total = 0
                for r in by_method[method]:
                    history = r.get("incumbent_history") or []
                    if not history:
                        continue
                    total += 1
                    level = history[0] + fraction * (r["reference_objective"] - history[0])
                    if any(v >= level for v in history):
                        reached += 1
                rates.append(reached / total if total else 0.0)
                labels.append(method)
            xs = np.arange(len(labels)) + fi * width
            ax.bar(xs, rates, width=width, label=f"closure {fraction:.2f}")
            ax.set_xticks(np.arange(len(labels)) + width * (len(fractions) - 1) / 2)
            ax.set_xticklabels(labels, fontsize=7, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of runs reaching threshold")
        ax.set_title(benchmark_id, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Gap-closure threshold success rates — {LABEL}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_success_rates.png", dpi=150)
    plt.close(fig)


def fig_runtime_decomposition(records: list[dict], out_dir: Path) -> None:
    """Stacked component runtimes + explicit unattributed remainder (item 23/28)."""
    benchmarks = sorted({r["benchmark_id"] for r in records})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4.5), squeeze=False)
    components = [
        ("gp_fit_runtime_s", "GP fit"), ("candidate_generation_runtime_s", "candidate gen"),
        ("constraint_validation_runtime_s", "feasibility"), ("acquisition_runtime_s", "acquisition"),
        ("objective_runtime_s", "objective"),
    ]
    for ax, benchmark_id in zip(axes[0], benchmarks):
        groups: dict[str, list[dict]] = {}
        for r in records:
            if r["benchmark_id"] == benchmark_id:
                groups.setdefault(r.get("algorithm_id", r.get("method")), []).append(r)
        methods = sorted(groups)
        means = {k: [] for k, _ in components}
        unattr = []
        for method in methods:
            rows = [r for r in groups[method] if (r.get("runtime_measurements") or {}).get("wall_runtime_s") is not None]
            if not rows:
                for k, _ in components:
                    means[k].append(0.0)
                unattr.append(0.0)
                continue
            for k, _ in components:
                vals = [r["runtime_measurements"].get(k) for r in rows]
                vals = [v for v in vals if v is not None]
                means[k].append(float(np.mean(vals)) if vals else 0.0)
            comp_sums = [sum(filter(None, [r["runtime_measurements"].get(k) for k, _ in components])) for r in rows]
            unattr.append(float(np.mean([r["runtime_measurements"]["wall_runtime_s"] - cs for r, cs in zip(rows, comp_sums)])))
        bottom = np.zeros(len(methods))
        xs = np.arange(len(methods))
        for k, label in components:
            vals = np.asarray(means[k])
            ax.bar(xs, vals, bottom=bottom, label=label, width=0.6)
            bottom += vals
        ax.bar(xs, np.asarray(unattr), bottom=bottom, label="unattributed (wall - sum)", width=0.6, color="#d9d9d9", hatch="//")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{m}\n(n={len(groups[m])})" for m in methods], fontsize=7)
        ax.set_ylabel("time per run (seconds, mean over seeds)")
        ax.set_title(benchmark_id, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper right")
    fig.suptitle(f"Runtime decomposition (partial attribution; unattributed explicit) — {LABEL}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.88, 0.94])
    fig.savefig(out_dir / "runtime_decomposition.png", dpi=150)
    plt.close(fig)


def fig_candidate_efficiency(records: list[dict], out_dir: Path) -> None:
    """Candidate-generation efficiency: infeasible rejections per requested + time per delivered."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    benchmarks = sorted({r["benchmark_id"] for r in records if "pool_requested_total" in (r.get("evaluation_counts") or {}) and r["evaluation_counts"].get("pool_requested_total") is not None})
    if not benchmarks:
        plt.close(fig)
        return
    width = 0.35
    for ax, (title, fn, ylabel) in zip(axes, [
        ("infeasible rejections per requested candidate", lambda r: (
            (r["evaluation_counts"].get("pool_rejected_infeasible_total") or 0)) /
            max(1, r["evaluation_counts"].get("pool_requested_total") or 1), "rejections / requested"),
        ("candidate-generation time per delivered candidate (s)", lambda r: (
            (r.get("runtime_measurements") or {}).get("candidate_generation_runtime_s") /
            max(1, r["evaluation_counts"].get("pool_delivered_total") or 1)
        ) if (r.get("runtime_measurements") or {}).get("candidate_generation_runtime_s") is not None else None,
         "seconds per delivered candidate"),
    ]):
        for bi, benchmark_id in enumerate(benchmarks):
            groups: dict[str, list[dict]] = {}
            for r in records:
                if r["benchmark_id"] == benchmark_id:
                    groups.setdefault(r.get("algorithm_id", r.get("method")), []).append(r)
            xs, ys = [], []
            for mi, method in enumerate(sorted(groups)):
                vals = [fn(r) for r in groups[method]]
                vals = [v for v in vals if v is not None]
                if vals:
                    xs.append(bi + (mi - len(groups) / 2) * 0.2)
                    ys.append(float(np.mean(vals)))
            ax.bar(xs, ys, width=0.18, label=benchmark_id if mi == 0 else None)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels(benchmarks, fontsize=7, rotation=20)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Candidate-generation efficiency (M6 baseline algorithm, unchanged) — {LABEL}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_generation_efficiency.png", dpi=150)
    plt.close(fig)


def fig_performance_vs_dimension(records: list[dict], out_dir: Path) -> None:
    """Performance (gap closure of f*) versus problem dimension (item 28)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    family: dict[str, dict[int, dict[str, list[float]]]] = {}
    for r in records:
        benchmark_id = r["benchmark_id"]
        if not benchmark_id.startswith("separable_"):
            continue
        dim = r.get("dimension") or int(benchmark_id.split("_")[-1].replace("d", ""))
        method = r.get("algorithm_id", r.get("method"))
        ref = r["reference_objective"]
        if r.get("final_objective") is not None:
            family.setdefault("separable", {}).setdefault(dim, {}).setdefault(method, []).append(
                r["final_objective"] / ref
            )
    for method in ("random_search", "joint_bo", "pairwise_bo", "joint", "pairwise"):
        dims_sorted = sorted({d for fam in family.values() for d in fam if method in fam[d]})
        if not dims_sorted:
            continue
        means = [float(np.mean(family["separable"][d][method])) for d in dims_sorted]
        stds = [float(np.std(family["separable"][d][method], ddof=1)) if len(family["separable"][d][method]) > 1 else 0.0
                for d in dims_sorted]
        ax.errorbar(dims_sorted, means, yerr=stds, marker="o", capsize=3,
                    label=f"{method} (mean ± sd over seeds)")
    ax.set_xlabel("problem dimension (separable quadratic family)")
    ax.set_ylabel("best objective / analytic optimum (dimensionless)")
    ax.set_title(f"Performance versus problem dimension\n{LABEL}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_vs_dimension.png", dpi=150)
    plt.close(fig)


def fig_sensitivity(sensitivity_summaries: list[dict], out_dir: Path) -> None:
    """Sensitivity-analysis results (item 25/28): final objective vs scenario."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scenarios = [s["scenario_id"] for s in sensitivity_summaries]
    means = [s["final_objective_mean"] for s in sensitivity_summaries]
    stds = [s["final_objective_stdev"] for s in sensitivity_summaries]
    xs = np.arange(len(scenarios))
    ax.bar(xs, means, yerr=stds, capsize=4, color="#1f77b4", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("final objective (benchmark units, mean ± sd over seeds)")
    ax.set_title(f"Sensitivity-analysis pilot (explicit scenario list, everything else fixed)\n{LABEL}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_results.png", dpi=150)
    plt.close(fig)


def generate_all_figures(records: list[dict], out_dir: str | Path,
                         fractions: list[float] | None = None) -> Path:
    """Standardized figure set for one experiment (item 28)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = fractions or [0.5, 0.9, 0.99]
    benchmarks = sorted({r["benchmark_id"] for r in records})
    for benchmark_id in benchmarks:
        fig_convergence(records, benchmark_id, out_dir, axis="successful")
        fig_convergence(records, benchmark_id, out_dir, axis="attempted")
        fig_convergence_wall(records, benchmark_id, out_dir)
    fig_gap_distributions(records, out_dir)
    fig_paired_differences(records, out_dir)
    fig_threshold_success(records, out_dir, fractions)
    fig_runtime_decomposition(records, out_dir)
    fig_candidate_efficiency(records, out_dir)
    fig_performance_vs_dimension(records, out_dir)
    return out_dir
