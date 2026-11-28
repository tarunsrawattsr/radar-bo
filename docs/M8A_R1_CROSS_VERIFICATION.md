# M8A-R1 — Corrective Implementation and Cross-Verification Report

_Status: COMPLETE — stopped for independent cross-verification before M8B, per the
milestone gate. This document reports, for every finding of the independent review
of commit `7e447be`, whether it is **RESOLVED**, **PARTIALLY RESOLVED**,
**UNRESOLVED**, or **NOT APPLICABLE**. Understandable independently of the project
conversation._

**Review-access note (honest):** at execution time (2026-11-28) the GitHub
repository contained NO review artifacts accessible through the API (no pull
requests, issues, commit comments, statuses, or review branches for `7e447be` —
verified via `gh pr list`, `gh issue list`, `gh api .../commits/7e447be/comments`,
`git ls-remote`). The authoritative finding list was therefore taken from the
M8A-R1 instruction itself (Phases A–E); every named finding was independently
re-verified against the actual code/documents BEFORE fixing, and the confirmed
defects are marked as such below. If a published review contains findings beyond
the instruction's list, they are not addressed here and this should be flagged in
cross-verification.

Base state: M8A final commit `7e447be` (recovered after sandbox reset #5 via
`git fetch origin` + `git reset origin/arena/7ed2bf69-radar-bo` — the established
procedure; HEAD verified `7e447be`, research archives checksum-verified). The two
original research PDFs (uploaded by the user at the branch-point commit `321e167`
and tracked since) were re-extracted before the reset recovery so nothing was lost.

---

## Phase A — Scientific source verification

**Method:** both PDFs extracted to `notes/paper_extracts/` (pypdf 7.2.0 installed to
a scratch directory OUTSIDE the project venv and lock — documented requirement; no
PDF tool existed in the environment; `requirements-lock.txt` unchanged) and read
line-by-line. Verified: ICC Eqs. (1)–(32), Table I (10 symbols, no values), Table II
(14 values), reference list (14 entries), authors (Ziyang Lu, M. Cenk Gursoy,
Chilukuri K. Mohan, Pramod K. Varshney — Syracuse University), venue, pp. 2785–2790,
DOI 10.1109/ICC51166.2024.10622225 (footer); Benzaghta authors/affiliations, venue,
pp. 4552–4557, DOI 10.1109/GLOBECOM54140.2023.10437768 (footer), Matérn-5/2 + BoTorch
+ max-a-posteriori fitting, ξ = 0.01, N_c = 500 (10 batches × 50), tilts+powers
objective, stochastic evaluations, block-wise BS-by-BS structure.

| # | Finding (review) | Status | What was done |
|---|---|---|---|
| A1 | Re-extract both original research PDFs available in the repository | **RESOLVED** | Both PDFs extracted; extracts gitignored at `notes/paper_extracts/{lu2024_icc,benzaghta2023_globecom}_fulltext.txt`; recorded in provenance entry `experiment.m8a_r1_source_verification` |
| A2 | Verify equations, parameter tables, references, author names, publication details, DOIs directly against the PDFs | **RESOLVED** | All verified; DOIs footer-confirmed (the M8A literature register had marked the Benzaghta DOI "unavailable" — corrected; the ICC reference list had been reconstructed from the extended version — now grounded in the verified 14-entry ICC list) |
| A3 | Correct the provenance register and all seven M8A documents | **RESOLVED** | Register regenerated (84 entries); corrections applied to all seven `docs/` files (see A4–A9 and Phase D rows for the specifics); correction notes are in-place and marked "M8A-R1" |
| A4 | Record the original paper's training weights and testing-weight definitions | **RESOLVED** | NEW register entries: `paper_lu_2024.training_weight_set` = {1, 3, 5} (ICC Sec. VI) and `paper_lu_2024.testing_weight_rule` = Eqs. (31)–(32) (distance-to-radar/asset rule min{10, 4+10000/d₀}, approaching-target +5 via Aⁿ = xⁿ·ẋⁿ < 0, cap 10, from ESTIMATED quantities). Recorded as paper FACTS; NOT adopted (`resource.task_utility_weights` stays None/uniform-labeled) |
| A5 | Correct scanning-parameter counts | **RESOLVED** | All documents corrected "eleven" → **TWELVE** unresolved `scanning.*` parameters (7 ICC Table-I hardware symbols + beam spacing φ + reference area S₀ + detection triple P_d/P_f/SNR_min); the count error is explained in the generated register (§4) |
| A6 | Correct the dimensional interpretation of the utility tradeoff coefficient | **RESOLVED** | β is NOT dimensionless: Γ = S_max/S₀ is dimensionless (m²/m²), so β carries the tracking-term dimension (weight·m²) per unit Γ. Register unit corrected ("(m^2 * weight) per unit Gamma"); math spec §9/§12, assumptions §13, and the decision register (DR-07) updated; Fig. 5's ×10⁶ axis noted as consistent |
| A7 | Preserve all accepted mathematical ambiguity resolutions | **RESOLVED (preserved)** | All five accepted resolutions preserved unchanged; two were STRENGTHENED with source evidence: (a) `table_variance=25` — the ICC Table II row literally reads "σ_w ((m/s²)²) 25", the paper's own unit annotation; (b) `inverse_snr` — the ICC PRINTS Eq. (6) as multiplication (σ² = σ²₀·SNR) contradicting its own prose, while the extended version divides; the discrepancy is documented, not reconciled |
| A8 | Keep extended-version parameters separate from the original conference-paper configuration | **RESOLVED** | Version-explicit throughout: generated register §5 table extended (solver DQN vs DDPG, dwell actions discrete vs continuous, α 15000 vs 5000, benchmarks EA+Random vs EA/DA/OA, weights definitions, Eq. (6) printed form, P₀ = I vs diag); Benzaghta tilts+powers (GLOBECOM) vs tilts+HPBW (journal) noted; ICC values reproduced, never silently switched |
| A9 | Preserve unresolved physical parameters as None | **RESOLVED (verified)** | 17 entries remain None (see deliverable 8 below); a Phase E check asserts this programmatically |
| A10 | Do not introduce new numerical assumptions | **RESOLVED** | No new numerical value entered any config/register/code default; the only new numeric register values are the paper's own recorded facts ({1,3,5} weights) and they are explicitly non-operational |

## Phase B — Shared initial-design corrections

Confirmed defects in `7e447be`: the "insufficient feasible design" error path used
`spec['benchmark_id']` (dict subscript on a pydantic model → TypeError instead of the
intended RuntimeError); designs were regenerated per run (deterministic but NOT
persisted as authoritative artifacts, with no design fingerprint) and were therefore
not reloaded on resume; shared-initialization verification compared records only
against each other (silent for single-method groups); there was no
malformed/duplicate/incomplete-design detection; random search could not participate
in shared-initial-phase comparisons.

| # | Finding | Status | What was done |
|---|---|---|---|
| B1 | Generate one authoritative initial design per compatible comparison group | **RESOLVED** | NEW `radar_bo/experiments/framework/initial_design_store.py`: `InitialDesignStore.load_or_create` ensures exactly one persisted design per (benchmark, dimension, seed) group, generated from a fresh problem before any run executes |
| B2 | Persist the design with a genuine SHA-256 fingerprint | **RESOLVED** | `design_fingerprint()` = real SHA-256 over canonical row JSON; stored as `design_sha256` and RECOMPUTED from the rows at every load (mismatch → rejection); unit-tested against plain `hashlib` |
| B3 | Pass the same immutable design to all applicable optimization methods | **RESOLVED** | Runner passes the authoritative rows to every BO architecture AND the new shared-start RS; plain RS explicitly ignores them; run records carry `initial_design_fingerprint` + `initial_design_source` |
| B4 | Reload the persisted design when resuming experiments | **RESOLVED** | `load()` revalidates (size/dimension/finiteness/feasibility/duplicates + digest) and returns the stored rows; regeneration is NEVER used to "repair" a stored design; targeted experiment: design files stable on resume |
| B5 | Correct the shared-initialization verification logic | **RESOLVED** | `verify_shared_initialization` now also checks each record against the PERSISTED design (config match + fingerprint match) and recomputes the authoritative rows' objectives on a fresh deterministic problem (catches inconsistent records even in single-method groups); duplicate/incomplete detection within each record added |
| B6 | Detect malformed, duplicate, incomplete and inconsistent designs | **RESOLVED** | `validate_design_rows()` detects all four classes (unit tests for each); `DesignValidationError` (a RecordRejected subclass) — explicit, never silent |
| B7 | Correct the insufficient-feasible-design error path | **RESOLVED (confirmed defect)** | Error path moved into `InitialDesignStore.create()` with attribute access and the full group identity in the message; unit test asserts `RuntimeError` matching "benchmark 'benchmark_a' d=2 seed=0" |
| B8 | Preserve historical seeded-sampler behavior | **RESOLVED (verified)** | `seeded_sampler` remains the default mode; the M7 integration + M8A shared-design suites pass unchanged (frozen trajectory preserved) |
| B9 | Retain independent random search and introduce a separately identified shared-start random-search mode | **RESOLVED** | `random_search` unchanged (explicitly ignores designs — trajectory-identity test added); NEW `random_search_shared_start` (separate algorithm_id/architecture/version) evaluates the authoritative design first, then spends the remaining budget on uniform random sampling; `RandomSearchOptimizer` gained the optional `initial_configs` prefix in the SAME implementation (no duplicated algorithm) |

## Phase C — Experiment framework corrections

Confirmed defects in `7e447be`: the runner asserted `attempted == budget` for every
BO record, making any legitimate early stop (patience/runtime/iteration caps) a
FAILED run — early completion and failure were conflated; a run interrupted between
result persistence and manifest update was re-executed on resume (repetition of
completed work); manifest status changes overwrote each other (no failure/retry
history); resume compatibility diagnostics were coarse.

| # | Finding | Status | What was done |
|---|---|---|---|
| C1 | Correct evaluation-budget handling to permit legitimate early stopping | **RESOLVED (confirmed defect)** | `_classify_completion`: attempted == budget → full; attempted < budget with a configured-stopping reason (patience, cycle patience, runtime, max iterations, candidate-pool/block exhaustion, generation cap) → **early COMPLETED**; over-budget → hard AssertionError |
| C2 | Distinguish early completion from failure | **RESOLVED** | Engine-level failure reasons (`no_feasible_initial_design`, `gp_fitting_failure`, `repeated_objective_failures`) → record persisted + status FAILED with the engine reason; legitimate early stops → COMPLETED with `early_completion: "early"`; both visible in records and summaries |
| C3 | Strengthen checkpoint compatibility checks | **RESOLVED** | `initialize()` reports WHICH compatibility components differ (strict whole-dict equality retained); NEW `_validate_record_identity` checks fingerprint + benchmark/algorithm/architecture/experiment_id/seed/dimension before any record is trusted (adoption or analysis) |
| C4 | Handle interruptions between result persistence and manifest updates | **RESOLVED (confirmed defect)** | `pending_specs()` ADOPTS a persisted, valid, identity-matching record for a run whose manifest status was not updated (history note "adopted existing record (manifest update was interrupted)") instead of re-executing |
| C5 | Prevent repetition of completed compatible runs | **RESOLVED** | Adoption covers the PENDING/RUNNING/FAILED-with-record cases; the targeted experiment shows zero re-executions after a simulated interruption; corrupted or mismatched records are explicitly REJECTED (RecordRejected), never silently accepted |
| C6 | Preserve failure and retry histories | **RESOLVED** | `set_status` appends an append-only `history` list (status/reason/timestamp) per run; `_merge_manifest` carries histories across resumes; a FAILED-without-record run is retried and its failure stays in the history (regression-tested) |
| C7 | Add independent regression tests for the discovered defects | **RESOLVED** | NEW `radar_bo/tests/test_m8a_r1_corrections.py` — 33 tests covering every Phase B/C row above (fingerprint genuineness, tamper rejection, the four design-defect classes, error-path identity, shared-start semantics, early-stop end-to-end, adoption, history, compatibility, verification) |

## Phase D — Scientific interpretation and documentation corrections

| # | Finding | Status | What was done |
|---|---|---|---|
| D1 | Suitability of BO for inexpensive deterministic objectives | **RESOLVED** | Assumptions §14 rewritten: [Benz2023] motivates BO for expensive/STOCHASTIC black-boxes (PDF-verified quote) — our benchmarks are the OPPOSITE regime (cheap, deterministic); the previous "the regime BO is designed for" phrasing removed; missing deterministic control recorded (DR-11); no BO-superiority implication remains |
| D2 | Process-noise variance versus realized estimation error | **RESOLVED** | Assumptions §8 addition: σ_w is a MODEL INPUT to Q; the tracking cost (Eqs. (14)–(15)) is the filter's SELF-ASSESSED posterior uncertainty — neither is the realized error versus ground truth; validation-plan V2.4 wording aligned ("model spread", direction check distinct from realized accuracy) |
| D3 | Partial equation reproduction versus complete experimental reproduction | **RESOLVED** | Reproduction-scope statement added (assumptions §6): equations (1)–(24) + Table II parameters are reproduced; the papers' experiments, solvers (DQN/CDRL), scenarios and results are NOT; every "reproduces [Lu2024]" claim means equations/parameters only |
| D4 | Paper-reference versus operational initialization profiles | **RESOLVED** | Profile naming made explicit (assumptions §9.4, math spec §5): `paper_zero_identity` (zero state + IDENTITY covariance — ICC PDF-verified) is the paper-reference/diagnostic profile; measurement-driven is operational; extended diag([1e4,1e4,1e2,1e2]) stays a candidate requiring approval |
| D5 | Historical runtime measurements | **RESOLVED (verified consistent)** | Re-checked: the M7 profiling findings and the nested-vs-disjoint timer-convention documentation were already consistent across README/docs; the corrected derived alternative-generation mean (184.12 s over 10 seeds vs archived 251.5 per-run-maxima mean) is stated with both conventions; no changes were needed beyond re-verification — historical values untouched |
| D6 | Original conference-paper facts versus extended-version findings | **RESOLVED** | Every document now attributes facts to a version (ICC/GLOBECOM = in-repo PDFs; extended = arXiv); the literature register separates them structurally; the version-difference table lives in the generated register §5 |

## Phase E — Final verification (results)

| Check | Result |
|---|---|
| Full regression suite | **713 tests: 712 passed + 1 registry-content test updated for the new algorithm entry, re-run to green** — final machine-readable report `research_results/m8a_r1_test_report.json` (totals, per-test outcomes, wall time; junit xml at `results/m8a_r1_junit.xml`, gitignored). First full run: 712 passed + 1 failed — `test_experiment_matrix.py::TestAlgorithmRegistry::test_all_methods_registered` asserts the exact registered-algorithm set and did not know `random_search_shared_start`; the test's expected set was extended (a content test, not a behavioral invariant) and the complete suite re-run to green |
| Historical archive checksums | 7/7 pre-existing archives verified OK (see deliverable 7) |
| Historical records unchanged | `git status` on `research_results/` shows only the NEW R1 artifacts (`m8a_r1_test_report.json`, regenerated `ARCHIVE_CHECKSUMS.json` including it); no existing file modified |
| Unresolved physical parameters | 17 entries remain None (programmatic assertion in the checksum/report step) |
| Targeted synthetic experiments | `results/m8a_r1_validation/summary.json`: **PASS** — genuine SHA-256 designs, verification green vs persisted designs (config + fingerprint + recomputed objectives), resume adoption with history, design files stable on resume, all runs completed; small matrices only (benchmark_a × 2 seeds × 3 methods, budget 12) |

## Deliverables index

1. This document — `docs/M8A_R1_CROSS_VERIFICATION.md`.
2. M8B decision register — `docs/M8B_DECISION_REGISTER.md` (15 decisions DR-01…DR-15; nothing adopted).
3. Corrected scientific documentation — all seven `docs/` files (in-place, marked M8A-R1) + regenerated `PARAMETER_RESOLUTION_REGISTER.md`.
4. Updated provenance register — `provenance_register.json` (84 entries / 17 unresolved) + `radar_bo/config/provenance.py`.
5. Complete list of code changes — see the commit message and the Phase B/C tables (8 modified code/test files + 3 new modules/files + 1 new experiment + 1 new test suite + the registry-content test update).
6. Full regression-test report — `research_results/m8a_r1_test_report.json` (machine-readable).
7. Historical archive-integrity report — checksum verification output (below and in the commit message).
8. Remaining unresolved parameters — the 17-entry list (below).
9. Decisions requiring user approval — `docs/M8B_DECISION_REGISTER.md` (all of DR-01…DR-15 require approval; DR-12 usage and DR-13 policy are the operational ones).
10. Git commit SHA and repository status — in the final commit + push output.

## Remaining unresolved parameters (17 — all by design, none invented)

`tracking.reference_snr`; `scanning.peak_transmit_power`, `scanning.transmit_antenna_gain`,
`scanning.receive_antenna_gain`, `scanning.wavelength`, `scanning.radar_cross_section`,
`scanning.loss_factor`, `scanning.system_temperature`, `scanning.beam_angular_spacing`,
`scanning.reference_area`, `scanning.detection_probability`, `scanning.false_alarm_probability`,
`scanning.required_detection_snr`; `tracking.measurement_driven_initial_velocity`,
`tracking.measurement_driven_initial_velocity_variance`; `resource.task_utility_weights`;
`optimization.evaluation_budget` (unresolved by design).

## STOP

M8A-R1 ends here. M8B requires explicit user approval of the scientific assumptions
and proposed model specifications (decision register DR-01…DR-15). No proposed
decision was adopted automatically; no value was fabricated; historical evidence is
unchanged.
