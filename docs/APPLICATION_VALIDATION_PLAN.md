# Application Validation Plan (M8A item 16)

_A staged plan for scientifically validating the models and assumptions BEFORE any
application-specific use. This is a SPECIFICATION — nothing in it is implemented or
executed in M8A. It separates software validation (does the code compute what the
specification says) from scientific validation (does the specification describe
anything meaningful). Understandable independently of the project conversation._

Companion documents: `MATHEMATICAL_MODEL_SPECIFICATION.md` (what is implemented),
`ASSUMPTIONS_AND_LIMITATIONS.md` (assumption inventory, §4–§6),
`PARAMETER_RESOLUTION_REGISTER.md` (which parameters are resolved),
`RESEARCH_GAP_ASSESSMENT.md` (what the literature does and does not establish).

---

## 0. Gate rule

No stage may start until the previous stage's exit criteria are met AND any parameter
the stage requires is resolved through the register's approval process. A stage that
discovers a failed expectation produces a model-rejection report (§7), not a parameter
patch.

## Stage V1 — Parameter resolution and provenance closure (scientific)

**Purpose:** convert unresolved parameters into either (a) approved paper values with
version identified, (b) approved scenario values explicitly labeled synthetic, or
(c) documented deferral with scope consequences.

Required inputs:
1. SNR₀ (`tracking.reference_snr`): re-verify against an accessed copy of the ICC
   paper, or approve a synthetic scenario value. Blocks: absolute measurement-noise
   scale (Eq. (5)–(6)–(8) chain).
2. `resource.task_utility_weights`: confirm from source or approve uniform-with-label.
   Blocks: any multi-target utility comparison other than equal-priority scenarios.
3. Initialization prior (`tracking.measurement_driven_initial_velocity[_variance]`):
   approve per-scenario values, or approve the extended paper's candidate
   diag([1e4, 1e4, 1e2, 1e2]), or approve a first-two-observations estimator. The
   ICC paper's own profile (zero state, IDENTITY covariance — PDF-verified in
   M8A-R1) remains the diagnostic paper-reference profile. Blocks: realistic EKF
   transients.
4. Physical scanning chain (`scanning.*`): deferred unless the physical model is
   approved for implementation; deferral is recorded, not hidden.

Independent checks:
- Every approved value enters the register with source + version + approval date; the
  generated document must regenerate byte-identically (deterministic generator).
- Cross-version consistency: ICC vs extended values never mixed in one configuration.

Exit criteria: register shows no `None` value REQUIRED by any activated code path;
classification of every active value is unambiguous.

## Stage V2 — Estimator correctness against the specification (software + scientific)

**Purpose:** verify the EKF chain computes the specified quantities on known cases
BEFORE any optimization claim.

Synthetic integration tests (deterministic, seeded):
1. Analytic covariance propagation: for noise-free motion, propagated covariance must
   match F P Fᵀ + Q in closed form (already unit-tested; keep as regression).
2. Known-observation limit: with R → tiny and correct association, EKF posterior must
   converge to the true state at a documented rate; failure indicates model/geometry bugs.
3. Noise-consistency check: empirical innovation covariance over many Monte Carlo
   runs must match H P Hᵀ + R within stated tolerance ( chi-square-style consistency
   test, NEES reported, thresholds stated in advance).
4. Both σ_w readings (`table_variance`/`equation_literal`) exercised end-to-end; the
   625-reading must show LARGER model spread (posterior covariance) — a falsifiable
   direction check on the MODEL INPUT, distinct from realized error (see assumptions
   §8). M8A-R1: the noise-model direction check also distinguishes the accepted
   inverse-SNR semantics (variance decreases with dwell) from the ICC paper's PRINTED
   Eq. (6) multiplication form — the discrepancy is documented, the resolution
   preserved.
5. Joseph-form vs literal-form posterior covariance equality within float tolerance
   across a seeded scenario battery.

Independent checks: none of these tests may use optimizer outputs; estimator tests
stand alone.

Exit criteria: all tests pass with thresholds fixed BEFORE running; NEES consistency
documented.

## Stage V3 — Utility and constraint semantics (software)

**Purpose:** prove the evaluator computes Eq. (23) and both budget semantics exactly.

1. Utility decomposition: run evaluator with w_n known; compare against hand-computed
   −Σw_n c^n + βΓ for ≥ 100 seeded configurations (property test, not spot check).
2. Constraint boundary: enumerate allocations at Στ/T₀ = Θ_max ± ε and verify
   feasibility classification; for discounted-cumulative, construct a case where local
   feasible ≠ discounted-feasible and one where local infeasible ≠ discounted-
   infeasible, and verify the reported semantics match §9.5 of the assumptions doc.
3. Remaining-time accounting: τ_s = T₀·max(0, 1 − Στ/T₀) verified including the
   saturation branch.
4. Failure semantics: forced exceptions produce status="failed" with utility None —
   never zero; infeasible never counted as evaluated successes.

Exit criteria: property tests pass; semantics table published in the experiment docs.

## Stage V4 — Optimization-behavior validation on synthetic problems (scientific)

**Purpose:** establish that each BO architecture behaves as claimed on problems with
KNOWN structure — the scientific basis for any later application claim.

Requirements:
1. Shared immutable initial designs for every comparison (M8A interface), ≥ 10 seeds,
   budgets and pool settings archived in advance (pre-registration style: settings
   committed before results).
2. Controls: random search (already archived) AND at least one deterministic
   direct-search method (e.g., coordinate/pattern search on the same budget) — the
   currently missing control identified by the BO-suitability review (assumptions §14).
3. Comparisons reported as paired per-seed differences with effect sizes, seed
   win/tie/loss counts, and significance only after multiple-comparison correction
   (Holm) — and even then described as evidence about THESE benchmarks, never as
   operational superiority.
4. Structure-sensitivity check: repeat the headline comparison with benchmark
   structure perturbed (rotated/sealed alternatives) so results can be attributed to
   structure, not to a specific function. No favorable-optimizer-driven coefficient or
   benchmark selection — any β or metric choice is fixed BEFORE results are seen.
5. Statistical requirements: 10 seeds minimum for descriptive statements; ≥ 30 seeds
   for any claim of distribution-level difference; report all three of effect size,
   dispersion, and seed-level records in the archive.

Exit criteria: pre-registered archive + paired analysis + controls present; claims
scoped to synthetic problems.

## Stage V5 — Model rejection conditions (scientific)

The following outcomes would REJECT the current model or an assumption and must be
reported if observed:
1. NEES consistency failure in V2 (estimator inconsistent with its own noise model).
2. Discounted-cumulative vs hard-per-slot feasibility disagreement observed on
   allocations that any approved scenario would actually generate — if the two
   semantics disagree materially in practice, per-slot results may not be presented
   under a cumulative-budget narrative.
3. Utility decomposition mismatch in V3 (evaluator ≠ specification).
4. An architecture's advantage disappearing under V4's structure perturbation — the
   "structure-specific" label then replaces any general claim.
5. β sweep (assumptions §13) showing qualitative ranking changes across β decades —
   any single-β conclusion is then invalid and must be reported as such.
6. Any resolved-parameter provenance contradiction (e.g., an approved value found
   inconsistent with its cited source version).

Rejection handling: a rejection produces a documented finding and a scope decision;
it never produces silent re-tuning of parameters to make the failure disappear.

## Stage V6 — Application-scenario validation (explicitly OUT OF M8A SCOPE)

Recorded for completeness only: physical radar power allocation, corridor sensing, or
large-scale application experiments require (a) V1–V5 complete, (b) explicit approval
of the scientific assumptions and model specifications (the M8B gate), and (c) scenario
data or a documented synthetic scenario definition with its own provenance. Nothing in
this repository may be presented as validated for any application until V6 has
actually run.

## Reproducibility requirements (all stages)

- Seeds, configurations, and archive checksums recorded per experiment (existing
  convention).
- Deterministic re-execution is NEVER described as independent replication.
- Every number quoted in documents traces to an archived record or is labeled as a
  recomputed derived quantity (M8A reconciliation convention).
- Missing/failed/skipped runs are reported as such — never substituted or invented.

## Software vs scientific validation — summary table

| Concern | Kind | Where |
|---|---|---|
| EKF computes Eqs. (9)–(13) | software | V2 tests (existing suite + V2 additions) |
| Utility computes Eq. (23) | software | V3 |
| Constraint semantics match §9.5 | software | V3 |
| Shared-design fairness | software | existing M8A tests (11) |
| σ_w reading direction (25 vs 625) | scientific | V2.4 |
| Estimator noise consistency | scientific | V2.3 |
| BO architecture behavior claims | scientific | V4 |
| β transferability | scientific | V5.5 |
| Application readiness | scientific | V6 (blocked on M8B approval) |
