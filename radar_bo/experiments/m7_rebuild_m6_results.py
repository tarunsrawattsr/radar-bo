"""Rebuild the M6 merged results working file FROM the durable archive (M7 item 14).

The gitignored ``results/`` artifacts were lost in a sandbox reset; the durable
TRACKED archive ``research_results/m6_block_architecture_full_evidence.json``
holds the same per-run records at full precision. This script derives a working
copy of ``results/m6_block_architecture_comparison/m6_full_results.json`` from
the archive so the M6 figure script can run again. The reconstruction is
labeled DERIVED; the archive itself is read-only. ``matrix_wall_runtime_s`` is
not archived and is set to null with an explicit note (never fabricated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    archive_path = Path("research_results/m6_block_architecture_full_evidence.json")
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    config = archive["configuration"]

    data = {
        "result_label": archive["result_label"],
        "phase": archive["phase"],
        "evaluation_budget": config["evaluation_budget"],
        "initial_design": config["initial_design"],
        "seeds": config["seeds"],
        "dependency_versions": archive["dependency_versions"],
        "derived_from": (
            "results file rebuilt by m7_rebuild_m6_results.py from the durable archive "
            f"{archive_path} (commit {archive.get('git_commit')}); matrix_wall_runtime_s was "
            "not archived and is null; per-benchmark blocks are the archived records unchanged"
        ),
        "matrix_wall_runtime_s": None,
        "benchmarks": {},
    }
    for name, block in archive["benchmarks"].items():
        data["benchmarks"][name] = {
            "reference": block["reference"],
            "budget": block["budget"],
            "initial_design": block["initial_design"],
            "num_candidates": config["num_acquisition_candidates"],
            "objective_threshold": block.get("objective_threshold"),
            "threshold_fraction": block.get("threshold_fraction"),
            "threshold_rule": block.get("threshold_rule"),
            "seeds": sorted({r["seed"] for r in block["runs"]}),
            "methods": sorted({r["method"] for r in block["runs"]}),
            "runs": block["runs"],
            "aggregate": block.get("aggregate"),
        }
    out_dir = Path("results/m6_block_architecture_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m6_full_results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Derived results file: {out_dir / 'm6_full_results.json'} "
          f"({sum(len(b['runs']) for b in data['benchmarks'].values())} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
