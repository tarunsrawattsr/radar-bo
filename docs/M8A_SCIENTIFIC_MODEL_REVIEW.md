# M8A — Scientific Model Review and Parameter Resolution (review summary)

_The umbrella review document for milestone M8A. It records what was verified, what
was reconciled, what was changed (and what was deliberately NOT changed), and indexes
the companion deliverables. Understandable independently of the project conversation.
Status: COMPLETE — stopped for cross-verification before M8B, per the milestone gate.

**Revised in M8A-R1 (2026-11-28):** the source PDFs were re-extracted and verified
line-by-line, and factual errors found by the independent review were corrected
throughout the seven M8A documents (dwell action set, scanning-parameter count,
β dimension, Eq. (6) printed form, initialization profiles, BO-suitability framing).
The correction log with per-finding status is `docs/M8A_R1_CROSS_VERIFICATION.md`;
the outstanding scientific decisions are consolidated in `docs/M8B_DECISION_REGISTER.md`._

Environment at review time: commit `371952a` (M7 final) base; Python 3.11.2;
torch 2.15.0+cu132, botorch 0.18.1, gpytorch 1.15.2, numpy 2.4.6, scipy 1.17.1,
pydantic 2.14.4 (venv rebuilt from lock, 115 s; all 7 research archives
checksum-verified after recovery).

---

## 1. Scope and non-scope

M8A covers: final M7 regression verification, experimental-evidence reconciliation,
scientific-assumption review, source-paper equation verification, parameter
provenance, dimensional analysis, literature review, consolidated mathematical
specification, limitations documentation, application-validation plan, and research
gap assessment.

Explicitly NOT in M8A (and not done): application-specific optimization, physical
radar power allocation, corridor sensing optimization, large-scale application
experiments, or any M8B work. M8B requires explicit user approval of the scientific
assumptions and proposed model specifications documented here.

## 2. Evidence reconciliation (instruction item 3)

### 2.1 Final M7 regression result — CONFIRMED (item 3.1)

The M7 full test suite was re-run on the committed M7 tree (HEAD `371952a`, clean):
**668 passed** in 676 s. This is the second independent clean confirmation (the
earlier combined full-run + fixed-file rerun also totaled 668). Item closed.

### 2.2 Run-count reconciliation — RESOLVED as a reporting error (item 3.2)

The durable M6 full-matrix archive contains **exactly 220 records**:
160 primary (4 benchmarks × 4 methods × 10 seeds) + 60 dimension-study
(separable_8d/16d × 3 methods × 10 seeds; the alternating architecture was
DELIBERATELY omitted from the dimension study — reported as such, never as a gap
in execution). The pilot archive holds 32 records separately (budget 40, seeds 0–1,
8 benchmark×method combos). The "230 runs" figure that appeared in M6/M7 narrative
summaries was an arithmetic/reporting error; the archives themselves were always
internally consistent (validated by M7 independent checks). Correction made via
documentation and the new provenance entry `experiment.m8a_run_count_reconciliation`;
historical archive values untouched (standing policy).

### 2.3 Fair-comparison interface — IMPLEMENTED, backward compatible (item 3.3)

Preference: validated immutable shared initial designs over seed-matching plus
retrospective verification.

- Engine: `OptimizationConfig.initial_design_configs` (default `None` = unchanged
  seeded behavior; frozen-trajectory regression value 9.905619188792086 reproduced).
  Fail-fast validation (exact count, dimension, finiteness, feasibility), immutable
  tuple snapshot, designs evaluated exactly in provided order, budget accounting
  unchanged (n_initial + n_bo = budget), candidate-pool report records explicit mode
  with zero sampling cost.
- Framework: `InitialDesignSettings.mode='explicit_shared'` — one design per
  (benchmark, dimension, seed) generated from a fresh problem via the registry
  sampler and passed UNCHANGED to every applicable method; infeasible design →
  RuntimeError before any run.
- Regression tests: 11 new tests in `radar_bo/tests/test_shared_initial_design.py`,
  all passing (68.6 s). Historical archives unaffected.

### 2.4 Profiling findings — PRESERVED (item 3.4)

M7 profiling stands: candidate generation dominates wall time (75–80%), GP fit
20–23%, acquisition <1%; the earlier "GP fitting dominates" claim remains RETRACTED;
archived M5/M6 generator-timer nesting is documented (corrected derived alternative:
constrained_mixed_8d alt-generation mean 184.12 s over 10 seeds, range 170.8–318.3,
vs archived 251.5 mean of per-run maxima). No efficiency inference from
acquisition-search dimensionality alone.

## 3. Provenance audit (items 5–6)

The register grew from 79 to **81 entries** (two new M8A entries) — **17 remain
UNRESOLVED (all by design, none invented)**:
`tracking.reference_snr`; `scanning.{peak_transmit_power, transmit_antenna_gain,
receive_antenna_gain, wavelength, radar_cross_section, loss_factor,
system_temperature, beam_angular_spacing, reference_area, detection_probability,
false_alarm_probability, required_detection_snr}`;
`tracking.measurement_driven_initial_velocity{,_variance}`;
`resource.task_utility_weights`; `optimization.evaluation_budget`.

Full classification (8 classes: paper-reported / missing-paper / universal constant /
hardware / scenario / proposed assumption / experimental setting / derived) and a
resolution-path analysis for every unresolved entry (universal value? hardware? paper
sufficient? derivable? extra reference? sensitivity?) are in
`docs/PARAMETER_RESOLUTION_REGISTER.md` (generated deterministically by
`radar_bo/experiments/m8a_generate_parameter_register.py` from the register JSON —
the JSON is authoritative).

New during M8A: the authors' extended journal version (arXiv 2606.05526) was fetched
and cross-checked. SNR₀ is absent there too (unresolved status re-confirmed); its
Table III values differ from the ICC Table II values we reproduce (full table in the
register §5); its P_f = 1e-3 and P₀ = diag([1e4, 1e4, 1e2, 1e2]) are recorded as
CANDIDATE values requiring separate review and approval — NOT adopted.

**Parameters requiring user approval before resolution** (register §6):
1. `tracking.reference_snr`;
2. `resource.task_utility_weights`;
3. `tracking.measurement_driven_initial_velocity{,_variance}`;
4. the TWELVE unresolved `scanning.*` parameters (7 ICC Table-I hardware symbols +
   beam spacing + S_0 + the P_d/P_f/SNR_min detection triple — only if the physical
   chain is ever approved for implementation);
5. `optimization.evaluation_budget` (per experiment, by design).

## 4. Literature review (item 7) — summary

Full register: `docs/LITERATURE_AND_SOURCE_REGISTER.md`. Highlights:
- [Lu2024] ICC DOI verified (10.1109/ICC51166.2024.10622225) from the authors' own
  extended-version bibliography; [Benz2023] GLOBECOM citation verified (pp. 4552–4557,
  DOI unavailable).
- Item-7 review of the radar paper's cited references (tracking / measurement-noise /
  detector support) completed from the extended version's bibliography with the
  paper's own "cited by" mappings: [Schöpe2021] (motion, SNR, cost), [Koch1999]
  (SNR model), [Meikle2008] (inverse-SNR variance), [Welch & Bishop 1995] (EKF),
  [Shnidman2002] (detection SNR→P_d). All full texts UNAVAILABLE — attributed content
  only, nothing inferred beyond citation context; no citation fabricated.
- The ICC reference list itself is UNAVAILABLE (paywall) — stated, not worked around.

## 5. Mathematical specification and dimensional analysis (item 8)

`docs/MATHEMATICAL_MODEL_SPECIFICATION.md` consolidates every implemented component
with paper-equation mapping and the [paper]/[proposed]/[synthetic] labeling. Findings
(§12 of that document, nothing changed silently): the Eq. (23) term-scale gap (m² vs
β·dimensionless — paper-resolved via β, reproduction-specific; see assumptions §13);
uncapped Γ semantics; SNR₀-missing chain; Eq. (22) vs hard-per-slot non-equivalence;
uniform-weights default labeling; Joseph/literal EKF equivalence status. No undefined
quantities beyond the documented unresolved parameters.

## 6. Reviews of items 10–13 (spec only)

`docs/ASSUMPTIONS_AND_LIMITATIONS.md`: §10 tracking-state requirements (planar
implemented; fixed-altitude/3-D requirements recorded, nothing implemented);
§11 scanning-metric review (circular-area Γ vs corridor metrics; units,
normalization, assumptions; synthetic outputs kept separate from validated
detection); §12 resource-representation requirements (time modeled; power/energy/
duty/shared recorded with their dimensional requirements); §13 β = 1e6 scaling review
(reproduction-specific; extended version uses 3e4; normalization/sensitivity plan;
favorable-optimizer-driven coefficients prohibited).

## 7. BO suitability and architecture evidence (items 14–15)

Assumptions §14 (honest assessment: deterministic cheap evaluations favor BO; no
superiority claims; missing deterministic direct-search control recorded for V4) and
§15 (all three architectures retained on one engine; paired synthetic results are
motivation, not proof).

## 8. Staged validation plan (item 16)

`docs/APPLICATION_VALIDATION_PLAN.md`: V1 parameter closure → V2 estimator
correctness (incl. NEES consistency and the falsifiable 25-vs-625 direction check) →
V3 utility/constraint semantics → V4 synthetic optimization-behavior validation
(pre-registered, shared designs, added deterministic control) → V5 model-rejection
conditions → V6 application scenarios (blocked on M8B approval). Software vs
scientific validation separated in a summary table.

## 9. Research gap assessment (item 17)

`docs/RESEARCH_GAP_ASSESSMENT.md`: established lines identified (CDRL radar line;
BO-for-engineering-design line); the BO×radar combination is OUR construction with an
explicit scope boundary (no systematic search performed); a table of potential
contributions and what each would require; explicit non-claims (no novelty from
differing optimizers; no improvement claims without validated evidence).

## 10. Proposed mathematical decisions (deliverable; NONE applied)

Decisions DOCUMENTED but requiring M8B approval before any implementation:
1. β normalization plan (assumptions §13) — not implemented.
2. First-two-observations velocity estimator alternative (register resolution path) —
   not implemented.
3. Adoption candidates from the extended version (P_f = 1e-3; P₀ diag) — not adopted.
4. Fixed-altitude/3-D state requirements — recorded only.
5. Corridor-metric definition requirements — recorded only.
6. Deterministic direct-search control for future comparisons — recorded for V4.
No code change in M8A embodies any of these decisions; the ONLY code change is the
backward-compatible shared-initial-design interface (item 3.3) plus the two new
provenance entries and documentation.

## 11. Deliverables index

| Deliverable | Path |
|---|---|
| Parameter resolution register (generated) | `docs/PARAMETER_RESOLUTION_REGISTER.md` |
| Literature and source register | `docs/LITERATURE_AND_SOURCE_REGISTER.md` |
| Mathematical model specification | `docs/MATHEMATICAL_MODEL_SPECIFICATION.md` |
| Assumptions and limitations | `docs/ASSUMPTIONS_AND_LIMITATIONS.md` |
| Application validation plan | `docs/APPLICATION_VALIDATION_PLAN.md` |
| Research gap assessment | `docs/RESEARCH_GAP_ASSESSMENT.md` |
| This review summary | `docs/M8A_SCIENTIFIC_MODEL_REVIEW.md` |
| M7 regression result | 668 passed / 676 s on `371952a` (this milestone's rerun adds the shared-design suite; see §12) |
| Reconciliation record | §2.2 here + `experiment.m8a_run_count_reconciliation` provenance entry |
| Provenance register (authoritative) | `provenance_register.json` (81 entries / 17 unresolved) |
| Shared-design interface | engine + framework changes, `radar_bo/tests/test_shared_initial_design.py` (11 tests) |
| Archive integrity | 7/7 checksums verified after recovery reset |

## 12. Final verification status

Full-suite rerun after all M8A changes: see README test-count note (679 tests — 668
M7 + 11 shared-design). Result recorded in the final commit message; the suite is the
authoritative gate for cross-verification.

## 13. STOP — cross-verification gate

M8A ends here. Per the milestone gate, work STOPS before M8B, which requires explicit
user approval of the scientific assumptions and proposed model specifications above.
Unresolved parameters REMAIN unresolved — no value was fabricated to satisfy
completion, and none of the acceptance criteria was satisfied by assertion alone.
