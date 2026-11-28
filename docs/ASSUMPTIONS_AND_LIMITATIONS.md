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
| Dwell action set | discrete values with 0.5 s granularity up to 25 s | Table II |
| k_B Boltzmann constant | SI exact | universal |

From [Benz2023]: Matérn-5/2 kernel, candidate pool N_c = 500, ξ = 0.01 (ξ preserved as
a register value only; the installed BoTorch LogEI has no ξ parameter).

## 2. Missing paper parameters (class: missing-paper) — stay unresolved

`tracking.reference_snr` (SNR₀) and the eleven physical scanning parameters
(`scanning.*` except the universal constant) plus per-task weights
(`resource.task_utility_weights`) have no value in EITHER paper version (re-verified
2026-11-26 against the extended version). Policy: they stay `None` in the register;
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
  entry; framework-level infeasible design → `RuntimeError` before any run starts.
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
   (unresolved by design). The extended paper's empirical diag([1e4,1e4,1e2,1e2]) is a
   recorded candidate requiring approval, not an adopted value.
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
- **Why β exists:** the two utility terms have different natural scales — tracking term
  in m² (weighted), scanning term dimensionless (Γ) or count (N_miss). β spans that
  gap. With ICC values, tracking costs near r₀ are O(10–100) m², so β·Γ with β = 1e6
  makes scanning dominate by ~4 orders of magnitude at Γ = O(1).
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
- Evaluations are deterministic on synthetic benchmarks and moderately expensive
  (candidate generation 75–80% of wall time) — the regime BO is designed for; 10–25
  eval budgets make model-based search sensible vs enumeration.
- Mixed/discrete-continuous decision spaces and complete-vector feasibility checks are
  handled by the generic problem interface (dummy-index-free; one engine, no duplicated
  algorithms).
- M5 descriptive results: joint BO beat random search by +0.76/+0.38/+0.27 on
  benchmarks A/B/C at budget 25 (descriptive, 10 seeds, no significance claims).

**Where the evidence does NOT support BO-superiority claims:**
- The deterministic benchmarks could partially be attacked by deterministic
  direct-search methods; no such comparison exists in our archives for most settings
  (M5/M6 compare RS and BO variants only). No claim is made beyond the archived
  comparisons.
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
