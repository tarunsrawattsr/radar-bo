# M8B Decision Register (M8A-R1 deliverable)

_One entry per outstanding scientific decision. NOTHING here is adopted: every entry
requires explicit user approval before any code change or experiment uses it. Each
entry states the source-paper evidence (version-explicit), the available alternatives,
their advantages/limitations, a recommendation ONLY where the evidence supports one,
dependencies, required validation, and whether approval is required. Understandable
independently of the project conversation._

Statuses: `UNRESOLVED` (no value; stays None), `RESOLVED-PRESERVED` (accepted
ambiguity resolution, listed for completeness — NOT reopenable without new evidence),
`PROPOSED` (a documented proposal awaiting approval).

---

## DR-01 — Reference SNR (SNR₀)

- **Parameter/assumption:** `tracking.reference_snr` — reference linear SNR in Eq. (5).
- **Status:** UNRESOLVED (None in the register).
- **Source-paper evidence:** required by ICC Eq. (5) and used by the extended version;
  absent from ICC (verified against the PDF, M8A-R1) AND from extended Table III
  (verified 2026-11-26). Genuinely missing in both versions.
- **Alternatives:** (a) leave unresolved (current) — synthetic paths needing Eq. (5)
  supply an explicitly synthetic SNR0; (b) derive from a link-budget using the
  Table I symbols — impossible without hardware values (DR-06); (c) approve a
  synthetic scenario value with provenance.
- **Advantages/limitations:** (a) honest, blocks absolute-noise claims; (b) not
  derivable; (c) enables reproduction-flavored experiments but the value remains a
  research assumption, never "the paper's value".
- **Recommendation:** (a) until a physical scenario definition exists; (c) only with
  an explicit labeled value.
- **Dependencies:** independent of DR-06.
- **Required validation:** if (c): sensitivity sweep (log-spaced) before any
  comparative use.
- **Approval required:** YES (any numerical resolution).

## DR-02 — Per-task utility weights w_n

- **Parameter/assumption:** `resource.task_utility_weights` (Eq. (23) aggregate).
- **Status:** UNRESOLVED (None ⇒ uniform default, labeled proposed).
- **Source-paper evidence:** symbolic in the model; TRAINING weights drawn from
  {1, 3, 5} (ICC Sec. VI, PDF-verified); TESTING weights time-varying per Eqs. (31)–(32)
  (distance-to-radar/asset rule + approaching-target bonus, cap 10, from ESTIMATED
  quantities) — recorded as register facts (`paper_lu_2024.training_weight_set`,
  `paper_lu_2024.testing_weight_rule`), not adopted.
- **Alternatives:** (a) uniform default (current); (b) reproduce the testing-weight
  rules Eqs. (31)–(32) inside the estimator-facing snapshot layer; (c) sample weights
  from {1,3,5} per training-style scenario; (d) fixed scenario-specific priority vector.
- **Advantages/limitations:** (a) neutral, no false specificity, but no priority
  structure; (b) faithful to the paper's TEST conditions, requires asset definition +
  estimator plumbing; (c) faithful to TRAINING conditions only; (d) arbitrary unless
  scenario-derived.
- **Recommendation:** (b) IF a reproduction-flavored milestone is approved (it is the
  paper's own testing definition and uses estimated quantities, consistent with our
  truth-inaccessibility guarantee); otherwise (a).
- **Dependencies:** (b) requires the asset-location concept (scenario definition).
- **Required validation:** weight-sensitivity study before any comparative claim;
  verify Eq. (31)–(32) implementation against hand-computed cases.
- **Approval required:** YES.

## DR-03 — Measurement-driven initialization velocity prior (mean)

- **Parameter/assumption:** `tracking.measurement_driven_initial_velocity`.
- **Status:** UNRESOLVED (None; code raises rather than inventing).
- **Source-paper evidence:** ICC inits state to zero (diagnostic profile); extended
  version estimates initial velocity from the last two valid measurements (may be
  non-consecutive, e.g. VVXV patterns) — PDF-verified both.
- **Alternatives:** (a) per-scenario synthetic prior (current operational path);
  (b) first-two-observations finite-difference estimator (the extended version's own
  approach); (c) zero-velocity prior with large variance.
- **Advantages/limitations:** (a) simple, honest, but scenario-invented; (b)
  paper-aligned and measurement-driven (no scenario invention), but biased under
  missed detections and noise; (c) robust but slow transients.
- **Recommendation:** (b), mirroring the extended version's documented method.
- **Dependencies:** DR-04 (variance consistency).
- **Required validation:** initialization-transient sensitivity study (assumptions
  §8/V2); bias analysis under missed-detection patterns.
- **Approval required:** YES.

## DR-04 — Initialization covariance P₀ for the operational profile

- **Parameter/assumption:** `tracking.measurement_driven_initial_velocity_variance`
  (+ position-variance companion).
- **Status:** UNRESOLVED.
- **Source-paper evidence:** ICC uses identity P₀ (paper-reference profile,
  PDF-verified); extended version uses empirical diag([10000, 10000, 100, 100])
  (extended Eq. (14), "initialized empirically").
- **Alternatives:** (a) per-scenario values (current); (b) adopt the extended
  diag([1e4, 1e4, 1e2, 1e2]) candidate; (c) tie to the first-observation covariance
  ratio (proposed); (d) identity (paper-reference).
- **Advantages/limitations:** (b) author-sourced for THEIR scenario and unit regime —
  plausible but scenario-dependent; (c) self-consistent with the observation noise
  (no invented constants), needs derivation care; (d) paper-faithful diagnostic only.
- **Recommendation:** (c) for the operational profile (dimensionally self-consistent);
  (b) acceptable as an explicitly labeled reproduction-flavored alternative.
- **Dependencies:** DR-03; note β/cost-scale coupling only through estimator
  transients.
- **Required validation:** V2 estimator-consistency battery (NEES) with the chosen P₀.
- **Approval required:** YES.

## DR-05 — False-alarm probability P_f (candidate)

- **Parameter/assumption:** `scanning.false_alarm_probability`.
- **Status:** UNRESOLVED (detector model NOT implemented).
- **Source-paper evidence:** no value in the ICC paper; extended version uses
  P_f = 1e-3 in ITS simulations (Sec. III-D, PDF-verified 2026-11-26) with
  uniformly-distributed false-alarm measurements.
- **Alternatives:** (a) stay unresolved (current — no detector implemented);
  (b) adopt P_f = 1e-3 when (and only when) the detector chain (DR-06) is approved.
- **Advantages/limitations:** (a) honest; (b) author-sourced but tied to their
  detector/lookup-table design and scenario.
- **Recommendation:** (a) now; (b) bundled with DR-06 approval.
- **Dependencies:** DR-06.
- **Required validation:** Shnidman-lookup implementation tests; false-track rate
  sensitivity.
- **Approval required:** YES.

## DR-06 — Physical scanning chain activation (Eqs. (16)–(20) + Shnidman detector)

- **Parameter/assumption:** the twelve `scanning.*` parameters + detector model.
- **Status:** UNRESOLVED; chain NOT implemented (synthetic provider only, labeled).
- **Source-paper evidence:** ICC Table I lists 10 symbols without values (PDF-verified);
  ICC Eqs. (16)–(20) define the chain; Shnidman's equation [12] gives SNR_min for
  (P_d, P_f); Γ = S_max/S_0 (Eq. (21)) with Γ≈1 meaning the intended coverage is met.
- **Alternatives:** (a) keep synthetic provider (current); (b) implement the chain
  with author-sourced values where available (extended version has NONE either) plus
  approved scenario values; (c) implement parameterized chain, all parameters
  unresolved-None (runs would fail-fast until resolved).
- **Advantages/limitations:** (a) honest, no physical claims possible; (b) enables
  physical Γ semantics but requires ~12 approvals and risks invented hardware values;
  (c) structurally ready but non-runnable until resolution.
- **Recommendation:** none — this is a scope decision, not a parameter choice.
- **Dependencies:** DR-05; every scanning.* resolution.
- **Required validation:** radar-range-equation unit checks against hand-computed
  cases; Γ monotonicity in τ_s; separate synthetic/physical output labeling.
- **Approval required:** YES (M8B gate).

## DR-07 — β normalization plan

- **Parameter/assumption:** `resource.tracking_scanning_tradeoff` (β = 1e6,
  paper-reported, reproduction-only). Dimension (M8A-R1): weight·m² per unit Γ — NOT
  dimensionless.
- **Status:** RESOLVED-PRESERVED as the reproduction value; the NORMALIZATION plan is
  PROPOSED (not implemented).
- **Source-paper evidence:** ICC β = 1e6 with Fig. 5 utility axis ×10⁶ (PDF-verified);
  extended β = 3e4 with the N_miss metric — β is version- and metric-dependent.
- **Alternatives:** (a) keep literal β for reproduction only (current); (b) normalized
  utility (wᵀc)/c_ref + Γ/Γ_ref with explicit reference scales, making the coefficient
  dimensionless and transferable; (c) per-scenario re-tuned β sweeps.
- **Advantages/limitations:** (a) faithful, non-transferable; (b) transferable and
  scale-robust, but introduces reference-scale choices (c_ref, Γ_ref) that are
  themselves assumptions; (c) honest per-scenario calibration, expensive.
- **Recommendation:** (b) for any NEW scenario family; (a) strictly for ICC
  reproduction. No favorable-optimizer-driven coefficient selection (prohibited).
- **Dependencies:** DR-10 (metric definition), DR-06 (physical Γ semantics).
- **Required validation:** show ranking invariance under scale changes; β
  log-decade sensitivity reported.
- **Approval required:** YES.

## DR-08 — Budget-constraint semantics for future experiments

- **Parameter/assumption:** `discounted_cumulative` (ICC Eq. (22), paper default,
  locally non-decidable) vs `hard_per_slot` (proposed BO-profile, locally decidable).
- **Status:** RESOLVED-PRESERVED (both implemented, non-equivalent, never conflated).
- **Source-paper evidence:** ICC Eq. (22) is the discounted form with γ = 0.9
  (PDF-verified); the extended version retains the CMDP framing.
- **Alternatives:** (a) discounted default (current) — paper-faithful, but BO
  candidates cannot be locally feasibility-checked; (b) hard_per_slot — decidable,
  stricter; (c) report both for every future comparative experiment.
- **Advantages/limitations:** (b) enables the M8A hard-per-slot profile but excludes
  allocations Eq. (22) would allow; (c) transparent but doubles cost.
- **Recommendation:** (c) whenever comparisons are reported operationally; (b) only
  with the labeling "proposed stricter profile".
- **Dependencies:** none new.
- **Required validation:** V3 boundary cases (already specified) per semantics.
- **Approval required:** YES (for any switch of the operational default).

## DR-09 — Tracking-state extension (fixed-altitude / 3-D)

- **Parameter/assumption:** planar 4-D state (current) vs fixed-altitude 6-D vs
  general 3-D.
- **Status:** PROPOSED requirements recorded (assumptions §10); nothing implemented.
- **Source-paper evidence:** ICC model is explicitly planar (PDF-verified §II); the
  extended UCA model sweeps 360° azimuth — elevation unmeasured.
- **Alternatives:** (a) stay planar (current); (b) fixed-altitude extension (slant
  range + azimuth); (c) general 3-D (requires elevation measurement — observability
  constraint).
- **Advantages/limitations:** (b) moderate change, scenario-relevant for UAV
  altitudes; (c) partially unobservable without elevation sensors — structural risk.
- **Recommendation:** (b) only if a UAV-altitude scenario is approved, AFTER the
  recorded requirements (measurement model first, observability check, Q/R
  dimensional redo, new references — no reuse of planar references).
- **Dependencies:** scenario definition; new benchmark references.
- **Required validation:** observability analysis; full dimensional re-analysis;
  fresh synthetic validation battery.
- **Approval required:** YES.

## DR-10 — Scanning-metric definition for any future corridor-style problem

- **Parameter/assumption:** Γ = S_max/S_0 (ICC Eq. (21), implemented, ICC-faithful)
  vs missed-count N_miss (extended Eq. (19)) vs corridor SINR-statistics
  ([Benz2023] domain: median/bottom-percentile dB).
- **Status:** Γ implemented + synthetic provider (labeled); others NOT implemented.
- **Source-paper evidence:** ICC Γ with Γ≈1 semantics (PDF-verified); the AUTHORS
  THEMSELVES replaced Γ with N_miss in the extended version — direct evidence the
  metric is scenario-definitional; [Benz2023] uses mean-SINR mixes in dB.
- **Alternatives:** (a) Γ (current); (b) N_miss reproduction (needs a detection
  model → DR-06); (c) corridor percentile-SINR metric (different units, needs its own
  normalization + β-compatibility statement per assumptions §11).
- **Advantages/limitations:** (a) implemented and ICC-faithful but physically
  meaningless until DR-06; (b) operationally meaningful, heavier machinery; (c)
  matches the BO paper's domain, incompatible units with the radar utility.
- **Recommendation:** keep (a) for the reproduced problem; any corridor metric (c)
  must define quantity/statistic/normalization/β-compatibility in its specification
  before implementation.
- **Dependencies:** DR-06 for (b); DR-07 for any β coupling.
- **Required validation:** units/normalization statement review; synthetic outputs
  never labeled as detection performance.
- **Approval required:** YES.

## DR-11 — Deterministic direct-search control for future comparisons

- **Parameter/assumption:** the missing control method identified by the honest
  BO-suitability review (assumptions §14): cheap deterministic benchmarks are the
  regime where direct methods compete.
- **Status:** PROPOSED (not implemented).
- **Source-paper evidence:** [Benz2023] motivates BO for expensive/stochastic
  objectives (PDF-verified); our benchmarks are cheap/deterministic — the paper's
  own framing does NOT cover our regime. README non-negotiable: "BO's benefit
  relative to direct optimization must be measured, especially for inexpensive
  objectives."
- **Alternatives:** (a) coordinate/pattern search on the same budget; (b) enumeration
  where the discrete space is small; (c) scipy Nelder-Mead/L-BFGS-B adapted to
  constraints (complete-vector feasibility).
- **Advantages/limitations:** (a) budget-fair, structure-agnostic, cheap; (b) exact
  but only for tiny spaces; (c) established but constraint handling needs care and
  it is not budget-comparable per iteration.
- **Recommendation:** (a), implemented against the shared problem interface with the
  shared-initial-design harness (it can consume the shared design as its starting
  point — separately identified, like random_search_shared_start).
- **Dependencies:** none (reuses existing harness).
- **Required validation:** its own determinism/reproducibility tests; pre-registered
  comparison protocol (V4).
- **Approval required:** YES (any new algorithm entering comparisons).

## DR-12 — Shared-start random search as a standard comparator

- **Parameter/assumption:** `random_search_shared_start` (M8A-R1: authoritative
  shared design evaluated first, remaining budget uniform random; separately
  identified from plain `random_search`, which stays independent).
- **Status:** IMPLEMENTED (M8A-R1) — usage in future comparisons is the decision.
- **Source-paper evidence:** none — this is OUR fairness methodology (proposed),
  built so comparisons isolate the search strategy from the initial phase.
- **Alternatives:** (a) plain RS only (independent comparator, current default);
  (b) shared-start RS in every BO comparison (isolates strategy from initial design);
  (c) both reported side by side.
- **Advantages/limitations:** (b) removes initial-design confounding but makes RS
  "helped" by BO-designed starts — no longer a fully independent baseline; (a) keeps
  independence but conflates initial-phase quality; (c) transparent, slightly more
  compute.
- **Recommendation:** (c) — report both, with (b) labeled as shared-start and (a) as
  the independent baseline.
- **Dependencies:** none (implemented).
- **Required validation:** done (12 R1 regression tests; identical initial phase
  verified vs BO).
- **Approval required:** YES (for making it a standard comparison arm).

## DR-13 — Evaluation-budget policy

- **Parameter/assumption:** `optimization.evaluation_budget` (unresolved by design;
  per-experiment setting).
- **Status:** UNRESOLVED by design; M8A-R1 fixed the framework so legitimate early
  stopping (patience/runtime/iteration caps) is classified COMPLETED-early, not
  failed.
- **Source-paper evidence:** papers use solver-specific budgets (DQN episodes /
  BO iterations ≤ 170 for Benzaghta convergence, PDF-verified).
- **Alternatives:** (a) fixed per-experiment budgets, pre-registered (current);
  (b) compute-budget matching (equal wall-clock instead of equal evaluations) —
  rejected for comparability reasons unless separately justified; (c) convergence-
  adaptive stopping (patience) with budget caps.
- **Advantages/limitations:** (a) clean attribution, arbitrary magnitudes;
  (b) resource-realistic but conflates algorithm and implementation speed; (c)
  realistic but complicates paired statistics (different attempted counts per arm —
  now measurable thanks to the R1 classification).
- **Recommendation:** (a) for comparative archives; (c) allowed with
  `early_completion` metadata and descriptive-only statements.
- **Dependencies:** none.
- **Required validation:** attempted-vs-budget audit per run (now automated).
- **Approval required:** YES (per experiment family).

## DR-14 — Dwell action set for synthetic scenarios

- **Parameter/assumption:** `resource.dwell_time_action_set` = {0.1 … 1.0} s
  (ICC Sec. V-A.2, PDF-verified; register was always correct — the M8A DOCUMENTS
  wrongly said "{0, 0.5, …, 25}" and were corrected in M8A-R1).
- **Status:** RESOLVED-PRESERVED (paper-verified set).
- **Source-paper evidence:** ICC action definition, ten levels in [0.1, 1] s;
  extended version uses continuous τ ∈ [0, T₀] (DDPG).
- **Alternatives:** (a) keep the ICC discrete set (current); (b) add a continuous
  relaxation for extended-style experiments.
- **Advantages/limitations:** (a) paper-faithful; discrete granularity bounds the
  achievable allocation; (b) matches the extended solver's space but departs from the
  ICC formulation and changes feasibility structure.
- **Recommendation:** (a); (b) only under an explicitly labeled extended-version-style
  configuration.
- **Dependencies:** none.
- **Required validation:** feasibility/coverage tests already exist for the discrete
  set.
- **Approval required:** YES (for (b)).
