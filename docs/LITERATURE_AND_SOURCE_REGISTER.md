# Literature and Source Register (M8A item 7; revised in M8A-R1)

_Every citation in this document is verified against a source that exists either IN
this repository (the two original research PDFs uploaded by the user at commit
`321e167` and re-extracted during M8A-R1) or was fetched from arXiv during this
review. No citation is fabricated; where a source could NOT be accessed, it is
explicitly marked **UNAVAILABLE** and no content is inferred beyond what the citing
source itself attributes to it._

**Access context (M8A-R1, 2026-11-28).** The original M2/M3 paper extracts were lost
in a sandbox reset. Both ORIGINAL research PDFs are now available in the repository
root (uploaded by the user, tracked since the branch-point commit) and were
re-extracted to `notes/paper_extracts/` with pypdf 7.2.0 (installed to a scratch
directory OUTSIDE the project venv and lock — documented requirement: no PDF tool
existed in the environment; the lock is unchanged). Both PDFs were verified
line-by-line: equations, parameter tables, reference lists, author names, venues,
pages and DOIs. The PDF footers carry the IEEE Xplore download stamps
("Authorized licensed use limited to: Indian Institute of Technology - Jodhpur";
ICC downloaded 2026-10-10, GLOBECOM 2023 downloaded 2026-05-26). Equation numbering
below follows the ICC paper.

---

## 1. Primary sources (authoritative for this repository)

### 1.1 [Lu2024] — radar tracking/scanning model and time-allocation problem

- **Citation (PDF-verified, M8A-R1):** Ziyang Lu, M. Cenk Gursoy, Chilukuri K. Mohan,
  Pramod K. Varshney, *Learning-Based Cognitive Radar Resource Management for Scanning
  and Multi-target Tracking*, 2024 IEEE International Conference on Communications
  (ICC): Cognitive Radio and AI-Enabled Networks, pp. 2785–2790 (6 pages).
- **DOI (footer-verified):** 10.1109/ICC51166.2024.10622225. ISBN 978-1-7281-9054-9/24.
- **Full text:** AVAILABLE — in-repo PDF, re-extracted 2026-11-28 (`notes/paper_extracts/lu2024_icc_fulltext.txt`).
- **What this repository reproduces:** state/motion/process-noise (Eqs. (1)–(3)),
  planar range–azimuth measurement model (Eq. (4)), SNR-dependent noise (Eqs. (5)/(6)/(8)),
  EKF (Eqs. (9)–(13)), tracking cost (Eqs. (14)–(15)), scanning time/power/noise/SNR/range
  chain and metric Γ (Eqs. (16)–(21), Table I), budget constraint (Eq. (22)), utility
  (Eq. (23)), remaining-time accounting (Eq. (24)), Table II parameter values.
  **Equation-level reproduction only** — the paper's CDRL/DQN solver (Eqs. (25)–(30)),
  its training/testing weight apparatus, and its experimental scenarios/figures are NOT
  reproduced (see `ASSUMPTIONS_AND_LIMITATIONS.md` §6).
- **Table II (PDF-verified):** σ²_{r,0}=100 m²; σ²_{θ,0}=4e-4 rad²; σ_w ((m/s²)²)=25
  (the unit annotation itself declares the tabulated quantity a variance — evidence
  supporting the accepted `table_variance` resolution, which is PRESERVED);
  r₀=3000 m; τ₀=1 s; T₀=2.5 s; β=1e6; Θ_max=1; γ=0.9; DQN: mini-batch 32, replay
  buffer 1e6, ε=0.1, λ₀=5000, α=15000. The dwell action set is defined in the text
  (Section V-A.2): τ ∈ {0.1, 0.2, …, 1.0} s (ten levels).
- **Weights:** symbolic w_n in the model (Eq. (23)); TRAINING weights drawn from
  {1, 3, 5} (Section VI); TESTING weights time-varying per Eqs. (31)–(32)
  (min{10, 4+10000/d₀} distance rule with d₀ = min(dⁿ_r, dⁿ_a), then +5 if
  Aⁿ = xⁿ·ẋⁿ < 0, cap 10; computed from ESTIMATED quantities). Recorded as register
  facts (`paper_lu_2024.training_weight_set`, `paper_lu_2024.testing_weight_rule`);
  NOT adopted as our operational default (`resource.task_utility_weights` stays None
  with the labeled uniform default).
- **Internal inconsistency found (M8A-R1):** ICC Eq. (6) is PRINTED as
  σ²_{·,t} = σ²_{·,0}·SNR_t (multiplication) while the paper's own prose states the
  variance DECREASES with longer dwell time; the extended version prints the division
  form. The accepted `inverse_snr` resolution (division) is preserved unchanged; the
  printed-form discrepancy is documented, not silently reconciled.
- **Reference list:** 14 entries (verified; item-7 review in §2 below).

### 1.2 [Benz2023] — BO methodology (kernel, acquisition, candidate pools)

- **Citation (PDF-verified, M8A-R1):** Mohamed Benzaghta (Univ. Pompeu Fabra),
  Giovanni Geraci (UPF), David López-Pérez (Univ. Politècnica de València),
  Alvaro Valcarce (Nokia Bell Labs), *Designing Cellular Networks for UAV Corridors
  via Bayesian Optimization*, 2023 IEEE Global Communications Conference: Selected
  Areas in Communications: Aerial Communications, pp. 4552–4557 (6 pages).
- **DOI (footer-verified):** 10.1109/GLOBECOM54140.2023.10437768.
  ISBN 979-8-3503-1090-0/23. (The M8A draft marked this DOI "unavailable";
  corrected — it is printed in the PDF footer.)
- **Full text:** AVAILABLE — in-repo PDF, re-extracted 2026-11-28
  (`notes/paper_extracts/benzaghta2023_globecom_fulltext.txt`).
- **What this repository adapts:** BO methodology only — GP surrogate with Matérn-5/2
  kernel (PDF-verified: "We use the Matern-5/2 kernel for K(X) and fit the GP
  hyperparameters using maximum posterior estimation", on BoTorch), EI-type
  acquisition with ξ = 0.01 (PDF-verified "we set ξ = 0.01"), candidate pools
  N_c = 500 (PDF-verified, "10 batches each consisting of 50 points"), and the
  paper's own block-wise structure (optimization one base station at a time, each a
  2-D tilt+power subproblem, other blocks fixed — the direct analogue of our
  variable-block architectures). The application domain (cellular antennas / UAV
  corridors) is NOT reproduced; no corridor optimization is implemented or claimed.
- **Regime (M8A-R1 correction):** the paper's objective evaluations are STOCHASTIC
  system-level simulations ("a noisy sample f̃(x)", GP noise σ²I) and the paper
  motivates BO for "non-convex, non-linear, stochastic, high-dimensional and/or
  computationally expensive" black-boxes. Our synthetic benchmarks are cheap and
  deterministic — a different regime; see `ASSUMPTIONS_AND_LIMITATIONS.md` §14.
- **Version differences:** the GLOBECOM paper optimizes antenna TILTS + TRANSMIT
  POWERS with a mean-SINR-mix objective (λ ∈ [0,1], dB); the extended journal version
  (arXiv 2504.05176) optimizes tilts + HPBW. Initial design N₀ random; initial
  current point tilts 0°/powers 46 dBm; convergence < 170 iterations.
- **Limitations of our extraction:** ξ = 0.01 is preserved as a provenance value only;
  the installed BoTorch 0.18.1 `LogExpectedImprovement` has no ξ parameter, so the
  implemented acquisition is the library default.

### 1.3 [Lu2024-ext] — extended journal version of [Lu2024] (fetched 2026-11-26)

- **Citation (verified):** Z. Lu, M. C. Gursoy, C. K. Mohan, P. K. Varshney,
  *Constrained Deep Reinforcement Learning for Cognitive Radar Resource Management*,
  arXiv:2606.05526 [eess.SY], submitted 2026-06-04. DOI: 10.48550/arXiv.2606.05526.
- **Access:** `https://arxiv.org/html/2606.05526v1` fetched 2026-11-26 (Sections
  II–VII and the complete 32-entry bibliography reviewed).
- **Version differences that matter (kept SEPARATE from the ICC configuration; full
  table in `PARAMETER_RESOLUTION_REGISTER.md` §5):** extended Table III uses
  σ²_{r,0} = 16 m², σ²_{θ,0} = 1e-6 rad², σ²_w = 16 (m/s²)², β = 3e4, Θ_max = 0.9 —
  different from the ICC Table II values reproduced here; the scanning metric is
  changed from Γ (ICC Eq. (21), utility +β·Γ) to the missed-target count N_miss
  (extended Eq. (19), utility −β·N_miss); the solver moves from DQN to DDPG
  (continuous actions τ ∈ [0, T₀]); the dual-variable step size changes (α = 5000
  extended vs 15000 ICC); EKF initialization is measurement-driven with empirical
  P₀ = diag([10000, 10000, 100, 100]) (extended Eq. (14)) vs the ICC's zero vector +
  identity covariance; SNR₀ is absent from BOTH versions — `tracking.reference_snr`
  stays unresolved. **This repository reproduces the ICC version and does not
  silently switch values.** The extended version's P_f = 1e-3 and P₀ diagonal are
  recorded as candidate values requiring separate review and approval (never adopted).

---

## 2. References cited BY the two papers — item-7 review (M8A-R1: from the verified ICC list)

The ICC paper's own reference list (14 entries) is now verified against the source
PDF. Full texts of the cited works remain **UNAVAILABLE** in this environment (not
fetched); for each entry we record what the citing paper attributes to it — with the
ICC paper's own citation-context mapping — and nothing beyond that. (The earlier
M8A draft reconstructed this review from the EXTENDED version's bibliography; the
attribution chain is now the ICC list itself, with extended-version supplements
marked.)

### 2.1 Tracking and measurement-noise support

**[8] [Schöpe2021]** M. Schöpe, H. Driessen, A. Yarovoy, *A constrained POMDP
formulation and algorithmic solution for radar resource management in multi-target
tracking*, ISIF Journal of Advances in Information Fusion, 16(1), p. 31, 2021.

- **DOI:** not printed in the ICC reference list — **unavailable**. **Full text: UNAVAILABLE.**
- **Cited by [Lu2024] for:** the revisit-interval determination (§II-A, after Eq. (2));
  the maneuverability-noise covariance σ_w² (§II-A, Eq. (3)); the SNR model Eq. (5)
  (§II-B, jointly with [9]); the tracking cost ("a similar cost function as defined
  in [8]", §II-D, Eq. (14)).
- **Attributed role in our reproduction:** the ICC paper's own source for the CV
  motion model, Q structure, SNR model, and the position-covariance-trace cost that
  `environment/`, `radar/measurement_noise.py`, and `radar/tracking_cost.py`
  implement as [Lu2024] equations. Our repository does not cite [Schöpe2021]
  directly; the attribution chain is Lu → Schöpe.
- **Limitations:** the "similar to" phrasing means our Eqs. (14)–(15) reproduction is
  faithful to [Lu2024], not necessarily to [Schöpe2021]'s original definition.

**[9] [Koch1999]** W. Koch, *Adaptive parameter control for phased-array tracking*,
Signal and Data Processing of Small Targets 1999, SPIE Vol. 3809, pp. 444–455, 1999.

- **DOI:** not printed in the ICC list — **unavailable**. **Full text: UNAVAILABLE.**
- **Cited by [Lu2024] for:** the SNR model Eq. (5) (§II-B, jointly with [8]).
- **Attributed role:** co-support for the (τ/τ₀)·(r/r₀)⁻⁴ SNR structure implemented
  in `PlanarSNRMeasurementNoiseModel`.

**[10] [Meikle2008]** H. Meikle, *Modern Radar Systems*, Artech House, 2008 (book).

- **ISBN/DOI:** not printed in the ICC list — **unavailable**. **Full text: UNAVAILABLE.**
- **Cited by [Lu2024] for:** the measurement-variance-from-SNR relation Eq. (6)
  (§II-B) — the `inverse_snr` semantics that is this repository's accepted
  operational resolution (preserved; not reopened). Note the ICC printed-form
  discrepancy documented in §1.1.

**[11] [WelchBishop1995]** G. Welch, G. Bishop et al., *An Introduction to the Kalman
Filter*, 1995 (technical report; no venue in the ICC list — **unavailable**).

- **Full text:** UNAVAILABLE (not re-fetched).
- **Cited by [Lu2024] for:** Kalman-filter background (§II-C, Eqs. (9)–(13)) — the
  filter equations we implement literally, plus our Joseph-form variant (proposed).

### 2.2 Detector support (scanning phase)

**[12] [Shnidman2002]** D. Shnidman, *Determination of required SNR values [radar
detection]*, IEEE Transactions on Aerospace and Electronic Systems, 38(3),
pp. 1059–1064, 2002.

- **DOI:** not printed in the ICC list — **unavailable**. **Full text: UNAVAILABLE.**
- **Cited by [Lu2024] for:** §III-B — determining the minimum required SNR_scan for
  given detection probability P_d and false-alarm probability P_f ("according to
  Shnidman's equation"), which feeds the maximum-detectable-range Eq. (20) and the
  Γ metric Eq. (21).
- **Attributed role:** the detector mapping that would connect the (unimplemented)
  physical scanning chain to detection probabilities. This repository implements NO
  detector; `scanning.detection_probability`, `scanning.false_alarm_probability`, and
  `scanning.required_detection_snr` remain unresolved.
- **Applicability:** defines a concrete path should a future milestone approve the
  physical scanning chain. **Limitations:** not accessed; applicability to specific
  waveform/integration regimes cannot be assessed from here.

### 2.3 Problem-formulation and context references (ICC list, verified)

- **[1] [Charlish2020]** A. Charlish et al., *The development from adaptive to
  cognitive radar resource management*, IEEE AESM 35(6), pp. 8–19, 2020 — §I
  cognitive-RRM overview. Full text UNAVAILABLE.
- **[2] [JiLiu2007]** Z. Ji, K. R. Liu, *Cognitive radios for dynamic spectrum
  access…*, IEEE Communications Magazine 45(5), pp. 88–94, 2007 — related work.
- **[3] [Haykin2006]** S. Haykin, *Cognitive radar: a way of the future*, IEEE Signal
  Processing Magazine 23(1), pp. 30–40, 2006 — cognitive-radar overview.
- **[4] [Mnih2015]** V. Mnih et al., *Human-level control through deep reinforcement
  learning*, Nature 518(7540), pp. 529–533, 2015 — DRL background.
- **[5] [Thornton2020]** C. E. Thornton et al., *Deep reinforcement learning control
  for radar detection and tracking in congested spectral environments*, IEEE TCCN
  6(4), pp. 1335–1349, 2020 — DRL-for-radar related work.
- **[6] [Selvi2020]** E. Selvi et al., *Reinforcement learning for adaptable bandwidth
  tracking radars*, IEEE TAES 56(5), pp. 3904–3921, 2020.
- **[7] [Meng2021]** F. Meng, K. Tian, C. Wu, *Deep reinforcement learning-based radar
  network target assignment*, IEEE Sensors Journal 21(14), pp. 16315–16327, 2021.
- **[13] [Altman1999]** E. Altman, *Constrained Markov decision processes*, CRC Press,
  vol. 7, 1999 — CMDP formulation (§IV-A, Eq. (22) constraint structure).
- **[14] [Tessler2018]** C. Tessler et al., *Reward constrained policy optimization*,
  arXiv:1805.11074, 2018 — constrained-RL formulation (§IV-B, "Following [14] and [13]").
  All full texts UNAVAILABLE; none of our equations derives from them directly — they
  ground the CMDP framing that Eq. (22) implements.

### 2.4 BO-lineage references cited by [Benz2023] (PDF-verified list)

- **[24] [Maggi2021]** L. Maggi et al., *Bayesian optimization for radio resource
  management: Open loop power control*, IEEE JSAC 39(7), pp. 1858–1871, 2021 —
  BO-for-wireless lineage + EI definition source.
- **[29] [Shahriari2016]** B. Shahriari et al., *Taking the human out of the loop: A
  review of Bayesian optimization*, Proc. IEEE 104(1), pp. 148–175 — BO review
  (the ICC paper cites NO BO source; the two method lines are disjoint — see §3.1).
- **[30] [Huang2006]** D. Huang et al., *Sequential kriging optimization using
  multiple-fidelity evaluations*, Structural and Multidisciplinary Optimization 32,
  pp. 369–382, 2006 — EI formulation source.
- **[31] [Balandat2020]** M. Balandat et al., *BoTorch: A framework for efficient
  Monte-carlo Bayesian optimization*, NeurIPS 33, 2020 — the library our engine uses.

---

## 3. What the papers do NOT provide (honest gaps)

1. **No Bayesian-optimization literature in the radar paper.** The ICC reference list
   (14 entries, PDF-verified) contains no BO source; [Lu2024] solves the problem with
   a constrained DQN. The BO methodology in this repository comes solely from
   [Benz2023]. The combination (BO for radar dwell-time allocation) is OUR
   construction, not a reproduction of either paper (see `RESEARCH_GAP_ASSESSMENT.md`).
2. **No values for the physical scanning chain** in either version: ICC Table I lists
   exactly 10 symbols (P_t, G_t, G_r, λ, σ, L, R_t, R_r, k, T_s) without values; the
   detection triple (P_d, P_f, SNR_min via Shnidman) is likewise valueless. The
   TWELVE unresolved `scanning.*` register entries are faithful to the sources
   (k_B is resolved as a universal constant; R_t/R_r are per-target problem inputs).
3. **No SNR₀** in either version — confirmed against the ICC PDF (M8A-R1) and the
   extended version (2026-11-26).
4. **Per-task utility weights:** symbolic in the model; the EXPERIMENTAL definitions
   (training {1,3,5}; testing Eqs. (31)–(32)) are now recorded as facts, but no
   canonical w_n vector exists — `resource.task_utility_weights` stays unresolved
   with the labeled uniform default.
5. **Data association and track initiation:** the ICC paper does not describe the
   M-of-N initiation model (that appears only in the extended version, §II-D/§III-C,
   with GNN association). This repository implements neither — a documented scope
   difference, not an error.

## 4. Verification-method note (updated M8A-R1)

Primary-source facts were transcribed directly from the two in-repository PDFs
(extracted text) without paraphrasing bibliographic fields; footers provided DOIs,
ISBNs, page ranges and the IEEE Xplore download stamps. arXiv extended versions were
fetched 2026-11-26 for version-difference analysis. Anything not verifiable this way
(the full texts of the cited references) is marked **unavailable** rather than
estimated.
