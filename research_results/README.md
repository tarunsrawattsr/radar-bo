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
.venv/bin/python -m radar_bo.experiments.m6_generate_figures          # figures from results/m6_*
```

Figures themselves are regenerable from the archived per-seed histories and
are therefore NOT committed (no large binary files in Git).

## Files

| File | Producer | Content |
|---|---|---|
| `m5_synthetic_bo_validation.json` | `m5_synthetic_bo_validation` | Benchmarks A/B/C x seeds 0-9 x BO vs random search: references, per-seed incumbent histories, final incumbents, gaps, runtimes (GP fit / acquisition / objective / total), aggregates, dependency versions, commit `8c825a6...`. Regenerated after a sandbox reset LOST the original `results/` artifacts; the regenerated run reproduces the original numbers deterministically (engine unchanged, commit `8c825a6`). |
| `m6_block_architecture_pilot_evidence.json` | `m6_block_architecture_comparison --pilot` | Pilot phase (budget 40, initial 10, seeds 0-1) with validation checks. |
| `m6_block_architecture_full_evidence.json` | `m6_block_architecture_comparison` | Full matrix: 4 primary benchmarks x {random search, joint, alternating, pair-wise} x seeds 0-9 + separable 8d/16d dimension study; per-seed histories, incumbents, regret, runtimes, block-acceptance/stagnation statistics. |

## Provenance of the M5 archive

The original M5 run (2026-11-26, commit `8c825a6`, engine
`bo_single_candidate_log_ei_matern52`) was lost with the sandbox's `results/`
directory. The archived data was produced by RE-EXECUTING the same committed
experiment script on the same commit with the same seeds — the engine is
deterministic and the re-run reproduced the original per-seed results exactly
(verified against the recorded console log of the original run for benchmarks
A and B). No value was reconstructed from summary statistics.
