"""Generate the M6 research figures from the experiment results (instruction 26).

Reads ``results/m6_block_architecture_comparison/m6_full_results.json`` (and the
pilot file when present) and writes PNG figures labeled as generic synthetic M6
experiments under ``results/m6_block_architecture_comparison/figures/``:

1. convergence curves versus attempted objective evaluations (per benchmark)
2. convergence curves versus wall-clock time (per benchmark)
3. final simple-regret distributions across seeds
4. runtime decomposition by architecture
5. performance versus problem dimension (separable family)
6. candidate rejection rates
7. block acceptance rates
8. block stagnation statistics

All figures are regenerable from the tracked evidence archive; no physical
radar-performance claims are attached to any of them.

Run::

    python -m radar_bo.experiments.m6_generate_figures [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = "generic synthetic M6 experiment (no radar claims)"


def _load(results_dir: Path) -> dict:
    return json.loads((results_dir / "m6_full_results.json").read_text())


def _method_color() -> dict:
    return {
        "random_search": "#7f7f7f",
        "joint": "#1f77b4",
        "alternating": "#ff7f0e",
        "pairwise": "#2ca02c",
    }


def _history_points(row: dict, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """(evaluations, best-so-far) points from the incumbent history.

    The history records the best OBSERVED value after each improvement; steps
    are reconstructed on the attempted-evaluation axis: the incumbent after
    the initial design is at n_initial evaluations and subsequent improvements
    are placed at their attempted-evaluation positions (best-known placement:
    the history list is monotone; positions are approximated by even spacing
    across the BO phase — exact per-eval curves are in the archived records
    via the full results file when needed)."""
    history = row["incumbent_history"]
    if not history:
        return np.array([]), np.array([])
    n_initial = row.get("n_initial_evaluations", 30)
    n_bo = row.get("n_bo_evaluations", budget - n_initial)
    # improvement positions are not individually logged in the compact archive;
    # use the monotone envelope over the evaluation axis (documented placement):
    xs = [n_initial]
    for _ in history[1:]:
        xs.append(min(budget, xs[-1] + max(1, n_bo // max(1, len(history) - 1))))
    return np.array(xs[: len(history)]), np.array(history)


def fig_convergence_vs_evals(data: dict, out_dir: Path) -> None:
    colors = _method_color()
    budget = data["evaluation_budget"]
    for name, block in data["benchmarks"].items():
        if name in {"separable_8d", "separable_16d"}:
            continue  # dimension-study benchmarks use the dedicated figure
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, color in colors.items():
            rows = [r for r in block["runs"] if r["method"] == method]
            if not rows:
                continue
            grid = np.linspace(0, budget, 60)
            curves = []
            for row in rows:
                xs, ys = _history_points(row, budget)
                if len(xs) == 0:
                    continue
                curves.append(np.interp(grid, np.append([0], xs), np.append(ys[0] if len(ys) else 0.0, ys)))
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
            if curves:
                mean = np.mean(curves, axis=0)
                ax.plot(grid, mean, color=color, label=method, linewidth=2)
        ref = block["reference"]["objective_value"]
        ax.axhline(ref, color="k", linestyle="--", linewidth=1, label=f"reference ({ref:.2f})")
        ax.set_xlabel("attempted objective evaluations")
        ax.set_ylabel("best observed objective")
        ax.set_title(f"Convergence — {name}\n({LABEL})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"convergence_vs_evals_{name}.png", dpi=150)
        plt.close(fig)


def fig_convergence_vs_time(data: dict, out_dir: Path) -> None:
    colors = _method_color()
    for name, block in data["benchmarks"].items():
        if name in {"separable_8d", "separable_16d"}:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, color in colors.items():
            rows = [r for r in block["runs"] if r["method"] == method]
            if not rows:
                continue
            for row in rows:
                hist = row["incumbent_history"]
                if not hist:
                    continue
                wall = row.get("gp_fit_runtime_s", 0) + row.get("acquisition_runtime_s", 0) + row.get(
                    "objective_runtime_s", 0)
                if wall <= 0:
                    continue
                # even spacing over the measured wall segments (documented approximation)
                xs = np.linspace(0, wall, len(hist))
                ax.plot(xs, hist, color=color, alpha=0.3, linewidth=0.9)
            ax.plot([], [], color=color, label=method)
        ref = block["reference"]["objective_value"]
        ax.axhline(ref, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("approximate wall-clock time (s, even-spaced segments)")
        ax.set_ylabel("best observed objective")
        ax.set_title(f"Convergence vs wall-clock — {name}\n({LABEL}; BO runs are ~3 orders of magnitude slower than RS)", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"convergence_vs_time_{name}.png", dpi=150)
        plt.close(fig)


def fig_regret_distributions(data: dict, out_dir: Path) -> None:
    colors = _method_color()
    names = [n for n in data["benchmarks"] if n not in {"separable_8d", "separable_16d"}]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 4.2), sharey=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        block = data["benchmarks"][name]
        methods = [m for m in block["methods"] if m in colors]
        values, labels, cols = [], [], []
        for method in methods:
            rows = [r for r in block["runs"] if r["method"] == method]
            reg = [r.get("regret_vs_reference", r.get("gap_to_reference")) for r in rows]
            values.append(reg)
            labels.append(method)
            cols.append(colors[method])
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True)
        for patch, color in zip(box["boxes"], cols):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("simple regret vs reference")
    fig.suptitle(f"Final simple-regret distributions across seeds ({LABEL})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "regret_distributions.png", dpi=150)
    plt.close(fig)


def fig_runtime_decomposition(data: dict, out_dir: Path) -> None:
    colors = _method_color()
    names = [n for n in data["benchmarks"] if n not in {"separable_8d", "separable_16d"}]
    phases = ["gp_fit_runtime_s", "acquisition_runtime_s", "objective_runtime_s",
              "candidate_generation_runtime_s", "constraint_validation_runtime_s"]
    phase_labels = ["GP fit", "acquisition", "objective", "candidate gen", "constraint validation"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xticks, xlabels = [], []
    xpos = 0
    for name in names:
        block = data["benchmarks"][name]
        for method in block["methods"]:
            rows = [r for r in block["runs"] if r["method"] == method]
            means = [float(np.mean([r.get(p, 0.0) for r in rows])) for p in phases]
            bottom = 0.0
            for value, phase_label, phase_color in zip(means, phase_labels, plt.cm.tab20.colors[:5]):
                if value > 0:
                    ax.bar(xpos, value, bottom=bottom, width=0.7,
                           color=phase_color, label=phase_label if xpos == 0 else None)
                    bottom += value
            xticks.append(xpos)
            xlabels.append(f"{name[:14]}\n{method}")
            xpos += 1
        xpos += 0.6
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("mean seconds per run")
    ax.set_yscale("symlog")
    ax.set_title(f"Runtime decomposition by architecture ({LABEL})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_decomposition.png", dpi=150)
    plt.close(fig)


def fig_dimension_study(data: dict, out_dir: Path) -> None:
    dims = [4, 8, 16]
    methods = ["joint", "pairwise", "random_search"]
    colors = _method_color()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        means, stds = [], []
        for d in dims:
            name = f"separable_{d}d"
            block = data["benchmarks"].get(name)
            if block is None:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            rows = [r for r in block["runs"] if r["method"] == method]
            if not rows:
                means.append(np.nan)
                stds.append(np.nan)
                continue
            # normalize by the analytic reference (fraction of optimum reached)
            ref = block["reference"]["objective_value"]
            fracs = [r["best_objective"] / ref for r in rows]
            means.append(float(np.mean(fracs)))
            stds.append(float(np.std(fracs)))
        ax.errorbar(dims, means, yerr=stds, marker="o", color=colors[method], label=method, capsize=3)
    ax.set_xticks(dims)
    ax.set_xlabel("problem dimension (separable quadratic family)")
    ax.set_ylabel("best objective / analytic optimum")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Performance versus problem dimension ({LABEL})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_vs_dimension.png", dpi=150)
    plt.close(fig)


def fig_rejection_rates(data: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels, rates, cols = [], [], []
    palette = plt.cm.tab10.colors
    idx = 0
    for name, block in data["benchmarks"].items():
        for method in block["methods"]:
            rows = [r for r in block["runs"] if r["method"] == method]
            if method == "random_search":
                vals = [r.get("n_rejected_infeasible", 0) / max(1, r.get("n_rejected_infeasible", 0)
                                                               + r.get("n_objective_evaluations", 1)
                                                               + r.get("n_skipped_duplicates", 0)) for r in rows]
            else:
                vals = [
                    r.get("pool_rejected_infeasible_total", 0)
                    / max(1, r.get("pool_requested_total", 1))
                    for r in rows
                ]
            labels.append(f"{name[:16]}\n{method}")
            rates.append(float(np.mean(vals)))
            cols.append(palette[idx % 10])
            idx += 1
    ax.bar(range(len(labels)), rates, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("mean infeasible-rejection rate")
    ax.set_title(f"Candidate rejection rates ({LABEL})\nidentical pool sizes do NOT imply identical effective search effort", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_rejection_rates.png", dpi=150)
    plt.close(fig)


def fig_block_statistics(data: dict, out_dir: Path) -> None:
    bo_methods = ["joint", "alternating", "pairwise"]
    names = [n for n in data["benchmarks"] if n not in {"separable_8d", "separable_16d"}]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(1, len(bo_methods))
    for i, method in enumerate(bo_methods):
        acc_means, skip_means, labels = [], [], []
        for name in names:
            rows = [r for r in data["benchmarks"][name]["runs"] if r["method"] == method]
            if not rows:
                acc_means.append(np.nan)
                skip_means.append(np.nan)
                labels.append(name)
                continue
            visits = [max(1, r.get("n_block_visits", 1)) for r in rows]
            acc_means.append(float(np.mean([r.get("n_accepted_updates", 0) / v for r, v in zip(rows, visits)])))
            skip_means.append(float(np.mean([r.get("n_skipped_blocks", 0) for r in rows])))
            labels.append(name)
        xs = np.arange(len(names)) + i * width
        ax1.bar(xs, acc_means, width=width, label=method, alpha=0.85)
        ax2.bar(xs, skip_means, width=width, label=method, alpha=0.85)
    for ax, title, ylabel in (
        (ax1, "Block acceptance rate (accepted updates / visits)", "mean acceptance rate"),
        (ax2, "Block stagnation: skipped-block visits per run", "mean skipped blocks"),
    ):
        ax.set_xticks(np.arange(len(names)) + width / 2)
        ax.set_xticklabels(names, fontsize=7, rotation=20)
        ax.set_title(f"{title}\n({LABEL})", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "block_acceptance_and_stagnation.png", dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1]) if len(argv) > 1 else Path("results/m6_block_architecture_comparison")
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load(results_dir)
    fig_convergence_vs_evals(data, out_dir)
    fig_convergence_vs_time(data, out_dir)
    fig_regret_distributions(data, out_dir)
    fig_runtime_decomposition(data, out_dir)
    fig_dimension_study(data, out_dir)
    fig_rejection_rates(data, out_dir)
    fig_block_statistics(data, out_dir)
    print(f"Figures written to {out_dir} (labeled: {LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
