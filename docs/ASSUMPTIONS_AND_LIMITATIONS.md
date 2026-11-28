# Assumptions and Limitations (M8A items 9, 10–13, 15)

_Every assumption this repository makes, separated into (A) facts taken from the
primary papers, (B) accepted ambiguity resolutions that are PRESERVED and must not be
reopened without new evidence, (C) proposed research assumptions (ours, never
attributed to a paper), (D) experimental settings (explicit, not proven optimal), and
(E) synthetic constructs. Plus the M8A reviews of the tracking-state representation,
scanning metric, resource representations, and β scaling. Understandable independently
of the project conversation._

Sources: [Lu2024] (ICC 2024) and [Benz2023] (GLOBECOM 2023) as documented in
`LITERATURE_AND_SOURCE_REGISTER.md`; parameter-level provenance in
`PARAMETER_RESOLUTION_REGISTER.md`; equations in `MATHEMATICAL_MODEL_SPECIFICATION.md`.

---

## 1. Paper facts (class: paper-reported)

These are reproduced as printed in the ICC 2024 paper's Table II and equations; the
authors' extended arXiv version reports DIFFERENT values for five of them (documented
in `PARAMETER_RESOLUTION_REGISTER.md` §5 — we reproduce the ICC values):

| Quantity | Value | Paper anchor |
|---|---|---|
| σ²_{r,0} reference range-measurement variance | 100 m² | Table II |
| σ²_{θ,0} reference azimuth-measurement variance | 4e-4 rad² | Table II |
| σ_w maneuverability-noise quantity | 25 (see §9.1 for the two accepted readings) | Table II |
| r₀ reference distance | 3000 m | Table II |
| τ₀ reference dwell time | 1 s | Table II |
| T₀ revisit interval | 2.5 s | Table II |
| Θ_max budget fraction | 1 | Table II |
| β trade-off coefficient | 1e6 | Table II |
| γ discount factor | 0.9 | Table II |
| Dwell action set | τ ∈ {0.1, 0.2, ..., 1.0} s (ten levels, 0.1 s steps) | Sec. V-A.2 'Action' (M8A-R1 PDF-verified; the M8A draft's "0–25 s" claim was a documentation error) |
| Training weights (recorded fact) | drawn from {1, 3, 5} during DQN training | Sec. VI |
| Testing weights (recorded fact) | time-varying rules Eqs. (31)–(32): distance-to-radar/asset rule + approaching-target +5, cap 10, from ESTIMATED quantities | Sec. VI |
| k_B Boltzmann constant | SI exact | universal |

From [Benz2023] (all PDF-verified in M8A-R1): Matérn-5/2 kernel with max-a-posteriori
hyperparameter fitting on BoTorch, candidate pool N_c = 500 (10 batches × 50), ξ = 0.01
(ξ preserved as a register value only; the installed BoTorch LogEI has no ξ parameter),
block-wise optimization one base station at a time (2-D tilt+power subproblems), N₀
random initial observations, STOCHASTIC evaluations (GP noise σ²I).

## 2. Missing paper parameters (class: missing-paper) — stay unresolved

`tracking.reference_snr` (SNR₀) and the TWELVE unresolved `scanning.*` parameters
(7 ICC Table-I hardware symbols — P_t, G_t, G_r, λ, σ, L, T_s — plus beam spacing φ,
reference area S_0, and the detection triple P_d/P_f/SNR_min; M8A-R1 count correction:
the universal Boltzmann constant is resolved and R_t/R_r are per-target problem inputs,
not configuration parameters) plus per-task weights (`resource.task_utility_weights`)
have no value in EITHER paper version (ICC PDF re-verified line-by-line 2026-11-28;
extended version checked 2026-11-26). Policy: they stay `None` in the register;
code paths that need them raise `UnresolvedParameterError`; nothing is invented.
Candidate values surfaced by the extended version (P_f = 1e-3; P₀ = diag([1e4, 1e4,
1e2, 1e2])) are recorded in the resolution paths but are NOT adopted — adoption
requires separate review and approval.

## 3. The paper-facts vs proposed-assumptions separation rule

Any document, docstring, or report in this repository must be classifiable by this rule:
a statement is a **paper fact** only if it can be pointed to in an accessible source
(with the version identified); everything else is labeled **proposed**, **synthetic**,
or **experimental setting**. "Commonly used", "standard", or "typical" is NEVER used to
describe a value we chose. This rule is the audit standard of M8A and applies to all
future milestones.

## 4. Proposed research assumptions (class: proposed) — the honest inventory

| # | Assumption | Where it lives | Justification | Risk if wrong |
|---|---|---|---|---|
| A1 | Uniform per-task utility weights w_n as software default | `optimization/objective.py` default | Needed to run multi-task synthetic evaluations; paper uses w_n symbolically | Priority structure differs from any real scenario |
| A2 | Synthetic linear scanning model (S_det ∝ τ_s) as Γ provider | `radar/scanning_model.py` (labeled synthetic) | Exercises the resource→metric chain without the unimplemented physical model | Γ dynamics unrelated to real detectable-area build-up |
| A3 | `hard_per_slot` budget semantics as a BO-profile constraint | `optimization/constraints.py` (proposed) | Local feasibility must be decidable for BO candidates; stricter than paper's Eq. (22) | Excludes allocations the paper's constraint would allow |
| A4 | Joseph-form EKF update as numerical alternative | `radar/extended_kalman_filter.py` (literal Eq. (13) remains default) | Numerical robustness; equivalence tested | None known (equivalence under valid conditions) |
| A5 | Explicit shared initial designs for fair algorithm comparison | M8A engine/framework interface | Removes sampler-seed confounding between methods | None; backward compatible, historical trajectories unchanged |
| A6 | BO architectures (joint / alternating / pair-wise) over one engine | `optimization/bo/blocks*.py` | Investigation of block-decision structures — proposed, motivated by synthetic results, PROVEN nothing | Block choices may simply suit the synthetic benchmarks |
| A7 | Synthetic benchmarks A/B/C and 4/8/16-D families as test problems | `optimization/bo/benchmarks*.py` | Controlled, reproducible algorithm comparison | Conclusions may not transfer to real problems |

## 5. Experimental settings (class: experimental-setting)

Explicit experiment-design values (inherited from earlier milestones; documented in
their archives): BO budget (e.g., 25/40/50 per experiment), initial-design size,
SequentialHalving/LogEI configuration, candidate pool size 500, benchmark reference
values (A 10.0, B 11.0, C 4.5295…, separable 40/80/160 analytic, coupled 20,
block_stagnation 9.6, constrained_mixed 39.53). These are **settings, not findings**;
inheritance is labeled, never described as tuned or optimal. The evaluation budget is
deliberately unresolved in the register (`optimization.evaluation_budget`).

## 6. Synthetic constructs (class: synthetic) — kept separate from validated detection

**Reproduction scope statement (M8A-R1):** this repository reproduces the ICC paper's
MODEL EQUATIONS (1)–(24) and its Table II model parameters; it does NOT reproduce the
paper's EXPERIMENTS — no DQN/CDRL solver, no continuous action space, no
Lagrangian-dual training, none of the paper's scenarios/figures, and no claims about
the paper's reported performance comparisons (CDRL vs Equal/Random Allocation). Every
statement of the form "reproduces [Lu2024]" means the equations and parameters, never
the experimental results. Conversely the papers' experimental apparatus (DQN training
weights {1,3,5}, testing-weight rules Eqs. (31)–(32), DQN architecture) is recorded in
the register as reproduction context only.

- Benchmarks A/B/C and the 4/8/16-D families: analytic test functions.
- `SyntheticLinearScanningModel`: labeled synthetic; its outputs are NEVER presented as
  detection performance. No synthetic output may be described with operational radar
  vocabulary (detection, tracking accuracy in a real scenario) — synthetic outputs
  measure algorithm behavior on synthetic problems, nothing else.
- All ground truth is inaccessible to estimator/optimizer (guarded by tests).

## 7. Dimensional conventions

SI units; radians (never degrees) for angles; linear (never dB) SNR; float64; angles
wrapped to [-π, π) at the measurement residual. The one known dimensional tension —
the two terms of utility Eq. (23) — is treated in §13.

## 8. Known code-level limitations (reviewed, accepted, documented)

- Profiling: candidate generation dominates wall time (75–80%), GP fit 20–23%,
  acquisition <1% (M7 findings, preserved; the earlier "GP fitting dominates" claim is
  RETRACTED). No efficiency conclusion may be drawn from acquisition-search
  dimensionality alone.
- Feasibility evaluation of explicit initial designs happens fail-fast at engine
  entry; framework-level infeasible/insufficient designs → RuntimeError before any
  run starts (M8A-R1: the error path is unit-tested and reports the group identity).
- **Process noise vs realized error (M8A-R1 clarification):** the process-noise
  variance σ_w (Table II) is a MODEL INPUT to Q (Eq. (3)) — it bounds how much
  maneuvering the model tolerates, it is NOT the realized estimation error. The
  tracking cost (Eqs. (14)–(15)) is the filter's SELF-ASSESSED posterior position
  uncertainty, likewise not the realized error versus ground truth; realized error
  requires truth-based evaluation, which the layer-separation guards keep out of the
  estimator/optimizer. Any statement comparing σ_w choices must therefore speak of
  model spread/covariance, never of measured accuracy.
- EKF literal (13) can lose positive-definiteness in extreme regimes; Joseph form is
  available; equivalence tests exist (not a guarantee under all numerical conditions).

## 9. Accepted ambiguity resolutions — PRESERVED, do not reopen (M8A item 9)

These were resolved with user approval in earlier milestones. M8A reviewed them for
consistency and FOUND them consistently implemented; they are restated here as the
authoritative record. **Do not reopen without new evidence.**

1. **σ_w semantics (§1 anchor):** Table II's 25 is read as the variance (`table_variance`,
   default, Q multiplier 25); the literal Eq. (3) reading σ_w = 25 m/s² ⇒ Q multiplier
   625 is implemented as `equation_literal`. Both tested; choice recorded per experiment.
2. **Measurement-variance semantics:** `inverse_snr` (σ²_{·,t} = σ²_{·,0}/SNR_t) is the
   operational choice — matches Eq. (6) and the paper's stated monotonicity.
3. **Zero-vector EKF initialization:** diagnostic configuration only (paper-faithful);
   NOT the operational default.
4. **Measurement-driven initialization is operational:** position from observation,
   velocity prior from `tracking.measurement_driven_initial_velocity[_variance]`
   (unresolved by design). **Profile naming (M8A-R1):** the zero-vector +
   IDENTITY-covariance initialization (ICC PDF-verified: x̂=0, P=I, "assuming that the
   system has no prior knowledge") is the PAPER-REFERENCE profile (`paper_zero_identity`,
   diagnostic); the measurement-driven profile is OPERATIONAL. The extended paper's
   empirical diag([1e4,1e4,1e2,1e2]) is a recorded candidate requiring approval, not
   an adopted value.
5. **Budget semantics:** `discounted_cumulative` (paper Eq. (22)) is the default and is
   NOT locally decidable; `hard_per_slot` is the proposed BO-profile constraint. They
   are never conflated, and no result from one is claimed for the other.

## 10. Tracking-state representation review (M8A item 10) — spec only, nothing implemented

Current state: planar 4-D [x, y, vx, vy] — faithful to the paper and sufficient for
the implemented planar range–azimuth measurement model.

- **Fixed-altitude 3-D (planar slant range):** would require state [x, y, z, vx, vy, vz]
  with altitude fixed by configuration; measurement model becomes slant range + azimuth
  (elevation derivable but usually unmeasured in 2-D scan patterns); Q gains a third
  per-axis block (no new structure); H Jacobian changes (range includes z²); the
  tracking cost E selector extends to 3 position axes. Dimensional consistency holds;
  the substantive questions are scenario-level (min elevation, ground clutter) and
  OUT OF M8A SCOPE.
- **General 3-D:** additionally requires elevation measurement or 3-D scan pattern,
  plus data association in 3-D; the papers' UCA model (360° azimuth sweep) suggests
  azimuth-only scans, so a general 3-D state would be partially unobservable without
  elevation measurements — a structural observability constraint, not a coding task.
- **Requirement recorded:** any 3-D extension MUST (a) define the measurement model
  first, (b) verify observability of the extended state, (c) re-do dimensional analysis
  of Q and R, (d) never reuse planar test references as 3-D references. None of this
  exists in code today; no capability is claimed.

## 11. Scanning metric review (M8A item 11) — circular-area vs corridor

- **Implemented (ICC-faithful):** Γ = S_det/S_0 — dimensionless area ratio, uncapped,
  normalized by reference area S_0 (unresolved, scenario definition). With the synthetic
  provider, Γ is proportional to scanning time; with the physical chain unimplemented,
  Γ has NO physical interpretation yet.
- **The radar paper's own evolution is the key evidence:** the authors' extended version
  REPLACED the area-ratio metric with the missed-target count N_miss (utility
  −β·N_miss). Two different metrics in two versions of the same work is direct evidence
  that the scanning-metric choice is scenario-definitional, not canonical. Our
  reproduction stays ICC-faithful; both metrics are documented; no hybrid is created.
- **Corridor-style metrics (from [Benz2023]'s domain):** median/bottom-percentile SINR
  along a corridor — different units (dB), different normalization (none — raw SNR
  aggregation), different failure modes (tail vs mean). If a corridor metric is ever
  introduced it must define: quantity aggregated, aggregation statistic, normalization,
  and its own β-scale compatibility (§13). It is NOT implemented; the repo's corridor
  vocabulary extends only to benchmark naming conventions.
- **Normalization/assumption requirements recorded for any future metric:** units,
  normalization reference, cap-or-no-cap semantics, and β compatibility must be stated
  in the same sentence that defines the metric; synthetic outputs stay separate from
  validated detection (§6).

## 12. Resource representations review (M8A item 12) — what exists and what each extension requires

**Implemented: dwell TIME only.** Total dwell Στ [s], normalized usage Στ/T₀
[dimensionless], remaining scanning time τ_s = T₀·max(0, 1 − Στ/T₀) [s] (Eq. (24)),
per-slot vs discounted-cumulative budget semantics (§9.5). Complete for the reproduced
problem.

Not modeled — each with its dimensional requirements recorded for any future decision:

| Representation | Requires | Interaction with existing model |
|---|---|---|
| Peak transmit power | P_t hardware value (unresolved); per-dwell power envelope | Radar equation (18) needs it; would change SNR model inputs |
| Average power / duty cycle | Duty factor per slot; thermal limits | Duty cycle interacts with dwell set granularity; needs its own constraint |
| Energy per slot | Energy = ∫P dt; per-slot energy budget | Adds a second budget dimension; Eq. (22) analogue needed for it |
| Waveform-dependent resources | Waveform library (bandwidth, modulation) | Changes dwell semantics (dwell no longer sufficient statistic) |
| Shared/multi-radar resources | Multi-platform topology | Out of both papers' single-radar scope |

Rule recorded: any added resource dimension needs its own constraint semantics, its own
unresolved-parameter entries, and its own dimensional analysis BEFORE any optimization
run; none may be activated with invented values.

## 13. β = 1e6 scaling review (M8A item 13)

- **Status:** β = 1e6 is a paper-reported value (ICC Table II), preserved **for
  reproduction only**. The extended version uses β = 3e4 — direct evidence that β is
  scenario/scale-dependent even for the authors.
- **Why β exists and what its DIMENSION is (M8A-R1 correction):** the tracking term
  has the dimension weight·m²; Γ = S_max/S_0 (Eq. (21)) is dimensionless (m²/m²), so
  β carries the tracking-term dimension (weight·m²) per unit Γ — β is NOT dimensionless
  (the M8A register unit said "dimensionless"; corrected). With ICC values, tracking
  costs near r₀ are O(10–100) m², so β·Γ with β = 1e6 makes scanning dominate by ~4
  orders of magnitude at Γ = O(1) (the paper's Fig. 5 utility axis is scaled by 10⁶,
  consistent with this reading). In the extended version's N_miss metric the scanning
  term is a COUNT, and β's dimension changes accordingly — evidence that β is
  metric-definition-dependent, not a transferable constant.
- **Is it still meaningful under scale changes?** NO, not automatically. If (a) the
  tracking-cost scale changes (different σ², r₀, or E selector), (b) the scanning
  metric changes (§11), or (c) the budget Θ_max changes (0.9 vs 1 between versions!),
  the SAME β produces a DIFFERENT effective trade-off. β = 1e6 is therefore
  reproduction-specific and must not be carried into any new configuration as a
  "sensible default".
- **Plan recorded (no implementation in M8A):** (1) any future scenario change must
  re-derive or re-tune β with a documented sweep; (2) a normalization study should
  express the utility as (wᵀc)/c_ref vs Γ/Γ_ref with explicit reference scales, so the
  trade-off coefficient becomes dimensionless and transferable; (3) sensitivity of
  results to β over log-decades should be reported whenever β is not paper-matched;
  (4) NO coefficient may be chosen to favor a particular optimizer's results —
  favorable-optimizer-driven coefficient selection is prohibited.

## 14. BO suitability assessment (M8A item 15) — honest evaluation

**Where the evidence genuinely supports BO here:**
- Evaluations are deterministic and CHEAP on our synthetic benchmarks, but the
  candidate-generation stage (75–80% of wall time) dominates runtime, so the
  effective per-observation cost is the largest term; 10–25 evaluation budgets make
  any search method's sample efficiency the deciding factor.
- Mixed/discrete-continuous decision spaces and complete-vector feasibility checks are
  handled by the generic problem interface (dummy-index-free; one engine, no duplicated
  algorithms).
- M5 descriptive results: joint BO beat random search by +0.76/+0.38/+0.27 on
  benchmarks A/B/C at budget 25 (descriptive, 10 seeds, no significance claims).

**Where the evidence does NOT support BO-superiority claims:**
- **BO is motivated for EXPENSIVE or STOCHASTIC black-boxes — [Benz2023] itself states
  the setting ("non-convex, non-linear, stochastic, high-dimensional and/or
  computationally expensive to evaluate", M8A-R1 PDF-verified) and its evaluations are
  noisy system-level simulations. Our synthetic benchmarks are the OPPOSITE regime:
  cheap and deterministic. For cheap deterministic objectives, direct methods
  (coordinate/pattern search, enumeration where feasible) are typically competitive
  or better, and BO's surrogate overhead can dominate. Our runtime profile is
  candidate-generation-dominated, NOT GP-dominated — no claim that BO is the right
  tool for these benchmarks is made; BO's benefit relative to direct optimization
  "must be measured, especially for inexpensive objectives" (README non-negotiable).
  The M5/M6 archives compare RS and BO variants only; no deterministic direct-search
  control exists yet (recorded for validation-plan V4).
- The pairwise architecture's separable_16d advantage (+59.00, 10/0/0 records, Holm p
  .0039) is a DESCRIPTIVE synthetic result — motivation for investigation (§4 A6), not
  proof of architecture superiority; the same architecture is NOT favored on
  constrained_mixed_8d (+0.21, p_holm .0547 — indistinguishable from 0).
- Cheap-deterministic, expensive-stochastic, many-duplicate observations, or
  hard-rejection regimes would each demand different BO treatment; none is claimed
  solved.
- Redundant variables (e.g., separable benchmarks' structure) make ARD kernels carry
  load they need not; nothing is claimed about that.

**Suitable direct numerical benchmark categories recorded for future use:** analytic
separable/non-separable functions, constrained mixed-integer problems with decidable
feasibility, cheap-deterministic simulations with structured block interactions, and
expensive-stochastic simulators (requiring replication-aware BO — out of current
scope). Each new category requires its own validation before any performance claim.

## 15. Architecture evidence summary (M8A item 15, preserved)

All three BO architectures (joint / alternating / pair-wise) are RETAINED on one
engine with no duplicated algorithms. M7 paired, shared-design evidence (descriptive
only): pairwise ≥ joint on separable_16d (+59.00; 10/0/0); pairwise and alternating ≥
joint on coupled_8d (+22.30/+17.54; both 10/0/0); alternating = pairwise on
separable_4d (+1.16); all statistically indistinguishable on constrained_mixed_8d.
These results MOTIVATE the retained architectures; none justifies deleting an
architecture or claiming operational superiority (synthetic results only, §6).
