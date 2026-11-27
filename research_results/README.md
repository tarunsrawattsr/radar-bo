# Durable research-results archive

This directory is TRACKED in Git. It preserves the essential numerical evidence
of the synthetic experiments compactly and durably (M5 review instruction 3.3,
M6 instruction 31.16): benchmark definitions, experiment configurations,
per-seed incumbent/convergence histories, final incumbents, independent
reference solutions, runtime measurements, statistical summaries, dependency
versions and the producing Git commit id.

The FULL experiment artifacts (complete serialized result records, per-run
configs, provenance registers) live under `results/`, which is gitignored and
regenerable from the committed experiment scripts:

```bash
.venv/bin/python -m radar_bo.experiments.m5_synthetic_bo_validation   # ~30 min
.venv/bin/python -m radar_bo.experiments.m6_block_architecture_comparison results/m6_block_architecture_comparison   # full matrix
.venv/bin/python -m radar_bo.experiments.m6_block_architecture_comparison results/m6_pilot --pilot                   # pilot
.venv/bin/python -m radar_bo.experiments.m7_rebuild_m6_results        # DERIVED: rebuild results/m6_* from the archive below
.venv/bin/python -m radar_bo.experiments.m6_generate_figures results/m6_block_architecture_comparison   # figures
.venv/bin/python -m radar_bo.experiments.m7_validation_experiment     # M7 infrastructure validation (~11 min)
.venv/bin/python -m radar_bo.experiments.m7_sensitivity_pilot         # M7 sensitivity pilot
.venv/bin/python -m radar_bo.experiments.m7_m6_review                 # M7 review + archive validation (derived artifacts)
.venv/bin/python -m radar_bo.experiments.m7_paired_summary            # M7 paired statistical summaries (derived)
```

Figures themselves are regenerable from the archived per-seed histories and
are therefore NOT committed (no large binary files in Git).

## Files

| File | Producer | Content |
|---|---|---|
| `m5_synthetic_bo_validation.json` | `m5_synthetic_bo_validation` | Benchmarks A/B/C x seeds 0-9 x BO vs random search: references, per-seed incumbent histories, final incumbents, gaps, runtimes (GP fit / acquisition / objective / total), aggregates, dependency versions, commit `8c825a6...`. Regenerated after a sandbox reset LOST the original `results/` artifacts; the regenerated run reproduces the original numbers deterministically (engine unchanged, commit `8c825a6`). |
| `m6_block_architecture_pilot_evidence.json` | `m6_block_architecture_comparison --pilot` | Pilot phase (budget 40, initial 10, seeds 0-1) with validation checks. |
| `m6_block_architecture_full_evidence.json` | `m6_block_architecture_comparison` | Full matrix: 4 primary benchmarks x {random search, joint, alternating, pair-wise} x seeds 0-9 + separable 8d/16d dimension study; per-seed histories, incumbents, regret, runtimes, block-acceptance/stagnation statistics. ORIGINAL record — preserved unchanged during M7. |
| `ARCHIVE_CHECKSUMS.json` | `framework.archive_integrity` (M7 item 14) | sha256 integrity identifiers for every archive above; verify with `framework.archive_integrity.verify_checksum_manifest()`. |
| `m7_m6_review_profiling.json` | `m7_runtime_profiling` (M7 item 3.1) | DERIVED analysis: runtime-attribution profiling on the current instrumented engine (M5 benchmarks, joint mode, seeds 0-2) + cProfile pass. Corrects the M5 "GP fitting dominates" claim: GP fit ~20-23%, candidate generation ~55-59 s (~75-80%), acquisition <1%, small NEGATIVE unattributed remainder exposing the nested-timer convention. |
| `m7_m6_review_corrections.json` | `m7_m6_review` (M7 items 3.1-3.6, 4) | DERIVED analysis: targeted M6 review corrections (runtime attribution, the 251.5 s trace, matrix omissions, stagnation definitions, chronology) + archive validation results (M5: 30 records valid; M6: 220 records valid with the documented nested-timer finding; RS 'incumbent_history' is actually an evaluation sequence — data intact, label wrong). Historical archives unchanged. |
| `m7_paired_comparative_summary.json` | `m7_paired_summary` (M7 deliverable 10) | DERIVED analysis: paired seed-level comparisons (M6: per benchmark vs joint; M5: BO vs RS) with per-seed differences, wins/losses/ties (tol 1e-9), paired d_z, bootstrap CIs (B=10000, seed 2026), Wilcoxon + Holm. Descriptive only. |
| `m7_sensitivity_pilot.json` | `m7_sensitivity_pilot` (M7 item 25) | Sensitivity-interface validation: explicit 3-scenario pilot on benchmark_a (pool 100 / initial 4 / budget 15), everything else fixed; infrastructure validation, no algorithm conclusions. |

## Provenance of the M5 archive

The original M5 run (2026-11-26, commit `8c825a6`, engine
`bo_single_candidate_log_ei_matern52`) was lost with the sandbox's `results/`
directory. The archived data was produced by RE-EXECUTING the same committed
experiment script on the same commit with the same seeds — the engine is
deterministic and the re-run reproduced the original per-seed results exactly
(verified against the recorded console log of the original run for benchmarks
A and B). No value was reconstructed from summary statistics.
