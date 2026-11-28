# Literature and Source Register (M8A item 7)

_Every citation in this document was verified against a source actually fetched during
this review (access method and date recorded per entry). No citation is fabricated;
where a source could NOT be accessed, it is explicitly marked **UNAVAILABLE** and no
content is inferred beyond what the citing source itself attributes to it._

**Access context.** The original M2/M3 paper extracts (plain-text, under
`notes/paper_extracts/`) were lost in a sandbox reset; they were gitignored and are
not recoverable from git history. This review therefore re-established source access
via the platform web tools on 2026-11-26. IEEE Xplore full texts remain inaccessible
from this environment (paywalled); both primary works have arXiv versions that WERE
accessible and were fetched. Equation numbering below follows the ICC paper
conventions used throughout this repository's documentation.

---

## 1. Primary sources (authoritative for this repository)

### 1.1 [Lu2024] — radar tracking/scanning model and time-allocation problem

- **Citation (verified):** Z. Lu, M. Cenk Gursoy, Chilukuri K. Mohan, Pramod K. Varshney,
  *Learning-based cognitive radar resource management for scanning and multi-target tracking*,
  ICC 2024 — IEEE International Conference on Communications, 2024, pp. 2785–2790.
- **DOI:** 10.1109/ICC51166.2024.10622225 — verified 2026-11-26 from entry [25] of the
  authors' own bibliography in their extended journal version (arXiv 2606.05526,
  `https://arxiv.org/html/2606.05526v1`, References section).
- **Full text access:** ICC version **UNAVAILABLE** in this environment (IEEE paywall);
  previously accessed early in the project (extracts lost — see access context). Model
  content used in this repository was re-confirmed against the authors' extended version
  (§1.3), whose Eqs. (2)–(13) match the ICC numbering for motion, measurement, and EKF.
- **What this repository reproduces:** state/motion/process-noise (Eqs. (2)–(3)),
  planar range–azimuth measurement model (Eq. (4) geometry), SNR-dependent noise
  (Eqs. (5)/(6)/(8)), EKF (Eqs. (9)–(13)), tracking cost (Eqs. (14)–(15)), scanning
  metric Γ (Eq. (21)), budget constraint (Eq. (22)), utility (Eq. (23)), remaining-time
  accounting (Eq. (24)), Table II parameter values.
- **Reproduction status:** reproduction of the ICC model with documented extensions
  (three proposed BO architectures replace the paper's CDRL solver; synthetic scanning
  provider replaces the unimplemented physical chain). Our extensions are attributed to
  neither [Lu2024] nor [Benz2023].
- **Limitations of our extraction:** Table II values are from the lost M2/M3 extracts
  (recorded in `provenance_register.json`); per-task utility weights w_n were not
  preserved — `resource.task_utility_weights` stays unresolved.

### 1.2 [Benz2023] — BO methodology (kernel, acquisition, candidate pools)

- **Citation (verified):** M. Benzaghta, G. Geraci, D. López-Pérez, A. Valcarce,
  *Designing cellular networks for UAV corridors via Bayesian optimization*,
  Proc. IEEE GLOBECOM, 2023, pp. 4552–4557.
- **DOI:** **UNAVAILABLE / not verified** — no DOI was present in any accessible source;
  the page numbers and venue were verified 2026-11-26 from the bibliography of the
  authors' extended journal version (arXiv 2504.05176) and its ResearchGate record.
- **Full text access:** GLOBECOM version **UNAVAILABLE** (paywall). Extended journal
  version: arXiv 2504.05176, *Cellular Network Design for UAV Corridors via Data-driven
  High-dimensional Bayesian Optimization* (April 2025) — citation entry and abstract
  verified; full text not fetched in this review session (M2-era extraction facts for
  Matérn-5/2 kernel, N_c = 500 candidate pool, ξ = 0.01 are preserved in
  `provenance_register.json` as register-only values).
- **What this repository adapts:** BO methodology only — Gaussian-process surrogate
  with Matérn-5/2 ARD kernel, EI-type acquisition, candidate-pool-based optimization
  (pool size 500). The application domain (cellular antennas / UAV corridors) is
  **not** reproduced; no corridor optimization is implemented or claimed.
- **Limitations of our extraction:** ξ = 0.01 is preserved as a provenance value only;
  the installed BoTorch 0.18.1 `LogExpectedImprovement` has no ξ parameter, so the
  implemented acquisition is the library default.

### 1.3 [Lu2024-ext] — extended journal version of [Lu2024] (fetched in full during this review)

- **Citation (verified):** Z. Lu, M. C. Gursoy, C. K. Mohan, P. K. Varshney,
  *Constrained Deep Reinforcement Learning for Cognitive Radar Resource Management*,
  arXiv:2606.05526 [eess.SY], submitted 2026-06-04.
  DOI: 10.48550/arXiv.2606.05526 (arXiv-issued).
- **Access:** `https://arxiv.org/html/2606.05526v1` fetched 2026-11-26 (16 chunks;
  Sections II–VII and the complete bibliography reviewed).
- **Role:** independent re-verification of the model equations and the source of the
  item-7 literature review below (the ICC reference list itself is unavailable).
- **Version differences that matter (full table in
  `PARAMETER_RESOLUTION_REGISTER.md` §5):** extended Table III uses σ²_{r,0} = 16 m²,
  σ²_{θ,0} = 1e-6 rad², σ²_w = 16 (m/s²)², β = 3e4, Θ_max = 0.9 — **different from the
  ICC Table II values (100, 4e-4, 25, 1e6, 1) reproduced here**; the scanning metric is
  changed from Γ = S_det/S_0 (ICC Eq. (21), utility +β·Γ) to the missed-target count
  N_miss (extended Eq. (19), utility −β·N_miss); EKF initialization is changed from the
  zero vector to measurement-driven initialization with an empirical
  P₀ = diag([10000, 10000, 100, 100]) (extended Eq. (14)); the physical scanning chain
  (extended Eq. (18) radar range equation) still reports symbols without values; SNR₀
  is absent from extended Table III as well — `tracking.reference_snr` therefore stays
  unresolved. **This repository reproduces the ICC version and does not silently
  switch values.** The extended version's P_f = 1e-3 and P₀ diagonal are recorded as
  candidate values requiring separate review and approval (never adopted into code).

---

## 2. References cited BY the radar paper — tracking / measurement-noise / detector support (item 7)

Source of this review: the complete bibliography of [Lu2024-ext] (fetched above), with
the paper's own "Cited by" section/equation annotations. The ICC paper's own reference
list is **UNAVAILABLE** (paywall); where the ICC and extended bibliographies could
differ, entries below are stated as "cited by the extended version". None of these
full texts was accessible from this environment; for each we record what the citing
paper attributes to it — nothing beyond that is claimed.

### 2.1 Tracking and measurement-noise support

**[Schöpe2021]** M. Schöpe, H. Driessen, A. Yarovoy, *A constrained POMDP formulation and
algorithmic solution for radar resource management in multi-target tracking*, ISIF Journal
of Advances in Information Fusion, 16(1), p. 31, 2021.

- **DOI:** not provided in the accessed bibliography — **unavailable**.
- **Full text:** **UNAVAILABLE** (not accessible from this environment).
- **Cited by [Lu2024-ext] for:** the target motion model — revisit interval determination
  (§II-A, after Eqs. (1)–(2)); the maneuverability-noise covariance σ_w² (§II-A, Eq. (3));
  the SNR model Eq. (5) (§II-B, jointly with [Koch1999]); and the tracking cost function
  (§II-D: "a cost function similar to the one defined in [Schöpe2021]").
- **Attributed role in our reproduction:** this is the paper's own source for the CV
  motion model, Q structure, SNR model, and the trace-of-position-covariance cost that
  our `environment/`, `radar/measurement_noise.py`, and `radar/tracking_cost.py`
  implement as [Lu2024] equations. Our repository does not cite or depend on
  [Schöpe2021] directly; the attribution chain is Lu → Schöpe.
- **Assumptions (as characterized by the citing paper):** constrained-POMDP view of
  multi-target radar resource allocation.
- **Applicability:** supports the mathematical validity of the equations we reproduce.
- **Limitations:** content beyond the citing paper's attributions is unknown here; the
  "similar to" phrasing in §II-D means our Eq. (14)–(15) reproduction is faithful to
  [Lu2024], not necessarily to [Schöpe2021]'s original cost definition.

**[Koch1999]** W. Koch, *Adaptive parameter control for phased-array tracking*, Signal and
Data Processing of Small Targets 1999, SPIE Vol. 3809, pp. 444–455, 1999.

- **DOI:** not provided in the accessed bibliography — **unavailable**.
- **Full text:** **UNAVAILABLE**.
- **Cited by [Lu2024-ext] for:** the SNR model Eq. (5) (§II-B, jointly with [Schöpe2021]).
- **Attributed role:** co-support for the dwell-time/range dependence of echo SNR — the
  (τ/τ₀)·(r/r₀)⁻⁴ structure our `PlanarSNRMeasurementNoiseModel` implements.
- **Applicability:** classic phased-array tracking literature; consistent with the
  dwell-time-driven SNR semantics we use.
- **Limitations:** not accessed; only the citation context is recorded.

**[Meikle2008]** H. Meikle, *Modern Radar Systems*, Artech House, 2008 (book).

- **DOI/ISBN:** not provided in the accessed bibliography — **unavailable**.
- **Full text:** **UNAVAILABLE**.
- **Cited by [Lu2024-ext] for:** the measurement-variance-from-SNR relation Eq. (6)
  (§II-B): σ²_{·,t} = σ²_{·,0}/SNR_t — exactly the `inverse_snr` semantics that is this
  repository's accepted operational resolution (preserved; not reopened).
- **Applicability:** textbook-level support for inverse-SNR measurement variance.
- **Limitations:** not accessed; the relation is reproduced from [Lu2024], and its
  being textbook-supported is recorded on the citing paper's authority alone.

**[WelchBishop1995]** G. Welch, G. Bishop et al., *An Introduction to the Kalman Filter*,
1995 (technical report; no venue/DOI given in the accessed bibliography — **unavailable**).

- **Full text:** **UNAVAILABLE** (not re-fetched; well-known but not independently
  verified here).
- **Cited by [Lu2024-ext] for:** Kalman-filter background (§II-C, before Eqs. (9)–(13)).
- **Attributed role:** general KF/EKF foundation for the filter equations we implement
  literally, including the Joseph-form alternative (our proposed numerical variant).

### 2.2 Detector support (scanning phase)

**[Shnidman2002]** D. Shnidman, *Determination of required SNR values [radar detection]*,
IEEE Transactions on Aerospace and Electronic Systems, 38(3), pp. 1059–1064, 2002.

- **DOI:** not provided in the accessed bibliography — **unavailable**.
- **Full text:** **UNAVAILABLE**.
- **Cited by [Lu2024-ext] for:** §III-B (measurement acquisition model during scanning):
  computing detection probability P_d from received SNR and fixed false-alarm probability
  P_f via "Shnidman's equation", implemented there as a precomputed SNR→P_d lookup table
  (P_d ∈ [0.01, 0.99], step 0.01, nearest-neighbor lookup).
- **Attributed role:** the detector mapping that would connect the (unimplemented)
  physical scanning chain to detection probabilities. This repository implements NO
  detector; `scanning.detection_probability`, `scanning.false_alarm_probability`, and
  `scanning.required_detection_snr` remain unresolved, and the extended version's
  concrete choices (P_f = 1e-3; lookup-table detector) are recorded as candidates only.
- **Applicability:** defines a concrete, implementable detection path should a future
  milestone approve the physical scanning chain.
- **Limitations:** not accessed; applicability of Shnidman's equation to specific
  waveform/integration regimes cannot be assessed from here.

### 2.3 Resource-management context (survey and precursors)

**[Charlish2020]** A. Charlish, F. Hoffmann, C. Degen, I. Schlangen, *The development from
adaptive to cognitive radar resource management*, IEEE Aerospace and Electronic Systems
Magazine, 35(6), pp. 8–19, 2020.

- **DOI:** not provided in the accessed bibliography — **unavailable**. **Full text:**
  **UNAVAILABLE**.
- **Cited by [Lu2024-ext] for:** §I-B survey of cognitive-radar resource management.
- **Role for this repository:** context only; no equation or parameter of ours derives
  from it.

**[LuGursoy2023]** Z. Lu, M. C. Gursoy, *Resource allocation for multi-target radar
tracking via constrained deep reinforcement learning*, IEEE Transactions on Cognitive
Communications and Networking, 9(6), pp. 1677–1690, 2023.

- **DOI:** 10.1109/TCCN.2023.3304634 — verified 2026-11-26 from entry [24] of the
  [Lu2024-ext] bibliography. **Full text:** **UNAVAILABLE** (paywall).
- **Cited by [Lu2024-ext] for:** §I-C — the group's precursor work on constrained-DRL
  dwell-time allocation.
- **Role for this repository:** lineage only; our model equations come from the ICC/extended
  versions. Recorded because it is the radar paper's closest predecessor.

---

## 3. What the radar paper does NOT provide (honest gaps)

1. **No Bayesian-optimization literature.** [Lu2024]/[Lu2024-ext] solve the allocation
   problem with constrained DRL (DQL/DDPG, Lagrangian duality). The BO methodology in
   this repository comes solely from [Benz2023]; the radar paper's bibliography contains
   no BO sources to review. The combination (BO for radar dwell-time allocation) is
   therefore OUR construction, not a reproduction of either paper (see
   `RESEARCH_GAP_ASSESSMENT.md`).
2. **No values for the physical scanning chain** in either version (P_t, G_t, G_r, λ, σ,
   L, T_s, beam spacing/φ, S_0, P_d/P_fa/SNR requirements all valueless) — the eleven
   `scanning.*` unresolved entries are faithful to both sources.
3. **No SNR₀** in either version — confirmed twice (M2/M3 extraction; extended Table III
   re-verified 2026-11-26).
4. **Per-task utility weights:** the papers use w_n symbolically; no per-task values are
   available in any accessible text — `resource.task_utility_weights` stays unresolved.
5. **Data association:** the extended version names GNN association and an M-of-N
   (3-of-4) track-initiation model (§II-D, §III-C) with FCFS admission; the ICC paper's
   treatment (as extracted) did not include the M-of-N model. This repository implements
   neither GNN association nor track initiation — noted as a scope difference, not an
   error.

## 4. Verification-method note

Citation details above were transcribed from fetched primary text (arXiv HTML, 2026-11-26)
without paraphrasing bibliographic fields. Where the same fact appeared in multiple
independent accessible sources (e.g., the ICC citation appears in three arXiv bibliographies
and the GLOBECOM citation in two), it was accepted; single-source facts are marked as such
by their "verified from" note. Anything not verifiable this way is marked **unavailable**
rather than estimated.
