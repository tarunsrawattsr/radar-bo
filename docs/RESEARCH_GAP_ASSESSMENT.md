# Research Gap Assessment (M8A item 17)

_A preliminary, honest assessment of where this repository's work sits relative to
established literature. It states what is established, what is our construction, and
which potential contributions would require verification before they could be claimed.
It makes NO novelty claims and NO improvement claims. Understandable independently of
the project conversation._

Basis: the two primary sources and their bibliographies as verified in
`LITERATURE_AND_SOURCE_REGISTER.md` (IEEE full texts unavailable; arXiv extended
versions fetched 2026-11-26). This is a PRELIMINARY assessment — a systematic
literature review beyond the cited references was NOT performed in M8A (out of scope),
so absence-of-evidence statements below are explicitly bounded by that scope.

---

## 1. Established: constrained-DRL radar dwell-time allocation (the radar paper's line)

The radar resource-management problem — allocating dwell time between tracking N
maneuvering targets and scanning under a time budget — is established in the works of
the Lu/Gursoy/Mohan/Varshney group and their cited lineage. M8A-R1 (source-PDF
verification) makes the attribution chain version-explicit: the ICC paper's own
14-entry reference list cites [Schöpe2021] (motion model, SNR model, tracking cost),
[Koch1999] (SNR), [Meikle2008] (inverse-SNR variance), [Welch & Bishop 1995]
(Kalman filter), [Shnidman2002] (detection SNR), [Altman1999] and [Tessler2018]
(CMDP theory), plus cognitive-radar/DRL context ([Charlish2020], [Haykin2006],
[Mnih2015], [Thornton2020], [Selvi2020], [Meng2021]). The ICC solver is a constrained
DQN; the extended journal version ([Lu2024-ext], DDPG) adds the Lagrangian-relaxation
and multi-target-allocation lineage in its larger bibliography. The ICC paper's
CDRL line: [LuGursoy2023] TCCN is cited by the extended version; the ICC list itself
does not cite it. The model equations this repository reproduces
(CV motion, SNR-dependent noise, EKF, trace-of-position-covariance cost, utility with
trade-off coefficient) come from this line and are established within it.

Gap status: NO gap claimed here — we reproduce these equations and change nothing
about the radar model itself.

## 2. Established: BO for high-dimensional engineering design (the second paper's line)

Bayesian optimization for cellular-network/UAV-corridor antenna design is established
by [Benz2023] and its extended version (arXiv 2504.05176): GP surrogate with Matérn-5/2
ARD kernel, EI-type acquisition, candidate pools, and comparisons of HD-BO variants.
BO methodology in general (surrogate + acquisition + pools) is a mature field; the
specific configuration we adopt (Matérn-5/2 kernel with max-a-posteriori
hyperparameter fitting, N_c = 500 candidate pools, ξ = 0.01) is that paper's and —
M8A-R1 correction — is verified IN the GLOBECOM PDF itself (BoTorch implementation,
EI over 500 candidates in 10 batches × 50, block-wise BS-by-BS 2-D subproblems,
STOCHASTIC evaluations). Its BO lineage ([Maggi2021] JSAC, [Shahriari2016] Proc. IEEE
review, [Huang2006] sequential kriging, [Balandat2020] BoTorch) establishes the
method's maturity independently of our adaptation.

Gap status: NO gap claimed here — we adapt established BO methodology to a generic
engine; no BO-method novelty is claimed.

## 3. Our construction: BO applied to the radar dwell-time problem

The combination — using Bayesian optimization (rather than constrained DRL,
Lagrangian relaxation, or heuristic allocation) to choose per-target dwell times
against the [Lu2024] utility and budget constraints — is OUR construction. Neither
paper does this: the radar paper's bibliography (reviewed in the literature register)
contains no BO sources; [Benz2023] addresses cellular antennas, not radar dwell time.

**Boundary of this statement:** bounded by the sources actually reviewed — now
including the ICC paper's own 14-entry reference list (PDF-verified in M8A-R1, which
contains NO Bayesian-optimization source) and the GLOBECOM paper's 31-entry list
(BO-for-cellular-networks, not radar). A broader search might still find
BO-radar-resource work that neither paper cites. Any novelty claim about this combination would therefore
require a systematic literature search FIRST — which M8A deliberately did not perform.
No novelty claim is made.

**Why this is not automatically a contribution:** the radar paper's choice of CDRL is
itself justified in that line of work by partial observability, sequential
decision-making across slots, and adaptation to dynamic target populations. Our
current implementation optimizes per-slot allocations on synthetic benchmarks with
full evaluator feedback. The differing optimizer choice (BO vs CDRL) is, by itself, a
difference of method — NOT evidence of superiority and NOT a novelty claim.

## 4. Contributions that would require verification before any claim

| Potential contribution | What it would need | Status |
|---|---|---|
| BO as a competitive alternative for radar dwell-time allocation | Systematic literature search (BO × radar resource management); comparison against the papers' own baselines (Equal/Random Allocation of the ICC paper; EA/DA/OA of the extended version) on comparable scenarios; staged validation V1–V5 of `APPLICATION_VALIDATION_PLAN.md` | NOT STARTED — no claim made |
| Block/structured BO architectures (joint/alternating/pair-wise) for grouped decision variables | Evidence beyond synthetic separable/coupled benchmarks; structure-perturbation robustness (V4.4); comparison with established block-BO literature (a search also not yet performed) | Synthetic descriptive results only (M6/M7 archives); motivation, not proof |
| Validated immutable shared initial designs for fair algorithm comparison | Assessment against existing fair-comparison practice in the BO literature (not yet searched); peer review of the interface | Implemented + 11 regression tests; engineering contribution at most, unverified as novel |
| Reproducible radar-model reference implementation | The ICC paper's own artifact availability would have to be checked; our implementation currently lacks SNR₀ (missing in BOTH paper versions), so even "reproduction" is PARTIAL and EQUATION-LEVEL (M8A-R1 scope statement: equations + Table II parameters are reproduced; the papers' experiments, solvers and results are not) | Partial reproduction documented; no completeness claim |
| β-normalized utility formulation (assumptions §13 plan) | A study demonstrating that the normalized formulation preserves rankings under scale changes, plus literature check whether equivalent normalization exists | PROPOSED ONLY — not implemented, no claim |

## 5. Gaps in the primary sources that our work did NOT fill (recorded honestly)

1. SNR₀ missing in both versions → absolute noise scale unresolved; any absolute-noise
   statement in our synthetic runs is synthetic-labeled.
2. Physical scanning chain (radar equation, detector) symbol-only in both versions →
   our scanning model is a labeled synthetic provider; no physical scanning claim.
3. Per-task weights w_n symbolic in the papers → uniform default is a software default.
4. Data association (GNN) and M-of-N track initiation appear in the extended version;
   our repository implements neither → our "tracking" is single-target-per-EKF with
   assumed association, a documented scope difference (literature register §3.5).
5. The ICC vs extended parameter differences (five values + scanning metric change)
   mean "the paper's values" is version-dependent; our reproduction pins the ICC
   version and documents the divergence rather than silently following the extension.

## 6. Explicit non-claims

- No novelty claim for the BO–radar combination (§3 boundary).
- No improvement claim over CDRL, EA, DA, or OA — we have never run those baselines.
- No tracking, detection, energy, or operational improvement claims of any kind — no
  validated experimental evidence exists for any such claim in this repository.
- No claim that pairwise or alternating architectures are superior — synthetic
  descriptive results are motivation for investigation (assumptions §15).
- No efficiency claims — profiling findings are descriptive; no inference from
  acquisition-search dimensionality.

## 7. What a future contribution-claiming milestone must add

1. Systematic literature search: BO applied to radar resource management (and
   block/structured BO), with search protocol and inclusion criteria documented.
2. Baseline implementations of the radar line's own methods (at minimum EA/DA;
   ideally the OA Lagrangian approach) under the shared-design comparison harness.
3. The staged validation plan V1–V5 completed with pre-registered settings.
4. If any positive claim survives: effect sizes with confidence statements, seeds and
   archives published, and claims scoped exactly to what was measured.
