"""Merge the M6 experiment partials into the final result set and regenerate
the durable evidence archive (M6 instruction 31.16).

Merges ``results/m6_primary/m6_full_results.json`` and
``results/m6_dimension_study/m6_full_results.json`` (produced by two parallel
processes of ``m6_block_architecture_comparison``) into
``results/m6_block_architecture_comparison/m6_full_results.json`` and rewrites
``research_results/m6_block_architecture_full_evidence.json``.

Run::

    python -m radar_bo.experiments.m6_merge_results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from radar_bo.experiments.m6_block_architecture_comparison import write_evidence_archive


def main(argv: list[str]) -> int:
    primary_dir = Path("results/m6_primary")
    study_dir = Path("results/m6_dimension_study")
    out_dir = Path("results/m6_block_architecture_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = json.loads((primary_dir / "m6_full_results.json").read_text())
    study = json.loads((study_dir / "m6_full_results.json").read_text())

    merged = dict(primary)
    for name, block in study["benchmarks"].items():
        merged["benchmarks"][name] = block
    merged["merge_note"] = (
        "Full matrix assembled from two parallel runs (results/m6_primary + "
        "results/m6_dimension_study); per-benchmark blocks are unchanged."
    )

    (out_dir / "m6_full_results.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
    for partial_dir in (primary_dir, study_dir):
        prov = partial_dir / "provenance_register.json"
        if prov.exists():
            (out_dir / "provenance_register.json").write_text(prov.read_text(encoding="utf-8"), encoding="utf-8")
            break
    archive = write_evidence_archive(out_dir, merged)
    print(f"Merged {len(merged['benchmarks'])} benchmarks -> {out_dir / 'm6_full_results.json'}")
    print(f"Durable evidence archive: {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
