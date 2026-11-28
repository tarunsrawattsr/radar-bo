# Mathematical Model Specification (M8A item 8)

_Consolidated specification of EVERY mathematical component currently implemented
in this repository, mapped to its source-paper equation or explicitly labeled as a
proposed extension. This document is understandable independently of the project
conversation. Equations marked **[paper]** reproduce a primary reference; **[proposed]**
marks our own methodological extensions (never attributed to a paper); **[synthetic]**
marks self-defined synthetic constructs. SI units, radians, linear (non-dB) quantities,
float64 throughout._

Primary references (see `LITERATURE_AND_SOURCE_REGISTER.md` for access status):

- **[Lu2024]** Z. Lu et al., *Learning-Based Cognitive Radar Resource Management for
  Scanning and Multi-Target Tracking*, IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225.
- **[Benz2023]** M. Benzaghta et al., *Designing Cellular Networks for UAV Corridors via
  Bayesian Optimization*, IEEE GLOBECOM 2023, DOI 10.1109/GLOBECOM54140.2023.10437768.

---

## 1. State representation

**[paper, Lu2024]** Planar target state (4-D, ordered):

```
x = [x, y, vx, vy]^T   — position [m, m], velocity [m/s, m/s]
```

Implemented by `environment/target_state.py` (`TargetState`), which also carries a
2×2 covariance block and enforces finiteness/shape. Angular quantities are wrapped
to `[-pi, pi)` at the measurement-residual boundary (quadrant-aware convention;
`numerics.wrap_angle`).

**No 3-D state is implemented.** Section 10 of `ASSUMPTIONS_AND_LIMITATIONS.md`
specifies what a fixed-altitude or general 3-D representation would require; nothing
of it exists in code.

## 2. Constant-velocity motion model (Eqs. (2)-(3) of [Lu2024]) — **[paper]**

Implemented in `environment/motion_model.py` (`ConstantVelocityMotionModel`).

- **Eq. (2), state transition** with revisit interval `T = dt` (s):

```
F = [[1, 0, T, 0],
     [0, 1, 0, T],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],        x_{t|t-1} = F x_{t-1|t-1}
```

- **Eq. (3), process-noise covariance** (structured, per-axis white acceleration):

```
Q = q * [[T^4/4, 0,     T^3/2, 0     ],
         [0,     T^4/4, 0,     T^3/2 ],
         [T^3/2, 0,     T^2,   0     ],
         [0,     T^3/2, 0,     T^2   ]]       [m^2, m^2/s, m^2/s^2 units by block]
```

with multiplier `q` from `tracking.maneuverability_noise` under the documented
ambiguity resolution (item 9 of `ASSUMPTIONS_AND_LIMITATIONS.md`):

- `table_variance` (accepted default): `q = 25` (Table II value read as the variance);
- `equation_literal`: `q = 625` (= 25², Eq. (3) read literally with σ_w = 25 m/s²).

Both interpretations are implemented and tested; the choice is recorded per
experiment. An equivalent low-rank form `Q = G G^T q` with
`G = [[T²/2, 0], [0, T²/2], [T, 0], [0, T]]` is exposed for exact stochastic sampling.

Propagation: `propagate_deterministic` (mean only) and `propagate_stochastic`
(samples acceleration noise via `G`), both non-mutating.

**Dimensional note:** `q` carries (m/s²)² so `Q` blocks carry m² (position),
m²/s² (velocity), m²/s (cross terms) — dimensionally consistent.

## 3. Measurement model (planar range–azimuth) — **[paper]**

Implemented in `radar/measurement_model.py` (`PlanarRangeAzimuthModel`), sensor at
the planar origin:

```
h(x) = [r, θ]^T,   r = sqrt(x² + y²)  [m],   θ = atan2(y, x)  [rad],  r > 0 enforced
H = ∂h/∂x |_{x}    (2 x 4 analytic Jacobian; a numerical Jacobian is provided for tests)
```

`measure_from_state` applies `h` to a full state; `measure` takes a position.
The EKF update calls `measure` internally — observation counting in tests must
wrap accordingly (known M3/M4 test note).

## 4. Measurement-noise model (Eqs. (5), (6), (8) of [Lu2024]) — **[paper]**

Implemented in `radar/measurement_noise.py` (`PlanarSNRMeasurementNoiseModel`).

- **Eq. (5), SNR** (linear, not dB):

```
SNR_t(τ_t, r_t) = SNR0 * (τ_t / τ_0) * (r_t / r_0)^(-4)      [dimensionless]
```

with dwell time `τ_t` [s], range `r_t` [m], reference dwell `τ_0` [s]
(= 1.0 s, Table II), reference range `r_0` [= 3000 m, Table II], and reference SNR
`SNR0` — **UNRESOLVED (missing paper parameter)**, preserved as `None`.

- **Eqs. (6)/(8), measurement covariance** (operational `inverse_snr` semantics):

```
σ²_{·,t} = σ²_{·,0} / SNR_t,        R_t = diag(σ²_{r,t}, σ²_{θ,t})
σ²_{r,0} = 100 m²,  σ²_{θ,0} = 4e-4 rad²   (Table II)
```

This matches the paper's statement that measurement variance decreases with longer
dwell time. Alternative semantics exist behind `tracking.measurement_variance_semantics`
but `inverse_snr` is the operational choice. `SNR0` being unresolved, any synthetic
run that needs an absolute noise scale must supply an explicitly **[synthetic]**-labeled
SNR0; nothing is invented in the model itself.

## 5. Extended Kalman Filter (Eqs. (9)-(13) of [Lu2024]) — **[paper]**

Implemented in `radar/extended_kalman_filter.py` (`ExtendedKalmanFilter`):

```
Eq. (9):  x̂_{t|t-1} = F_t x̂_{t-1|t-1}
Eq. (10): P_{t|t-1}  = F_t P_{t-1|t-1} F_t^T + Q_t
Eq. (11): K_t        = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R_t)^{-1}
Eq. (12): x̂_{t|t}   = x̂_{t|t-1} + K_t (z_t − h(x̂_{t|t-1}))    (residual angle wrapped to [-π, π))
Eq. (13): P_{t|t}   = (I − K_t H_t) P_{t|t-1}
```

Covariance-update convention: `'paper_equation_13'` (literal default) and `'joseph'`
(numerical-stability form); equivalence under valid numerical conditions is tested.
`SNR0` unresolved ⇒ a run needing Eq. (5) requires a supplied (labeled) value.

**Initialization (M3-resolved ambiguity, item 9):** the paper's zero-vector
initialization is retained strictly as a **diagnostic configuration**;
operational synthetic estimation uses **measurement-driven initialization**
(first observation → position; velocity prior from
`tracking.measurement_driven_initial_velocity[_variance]`, both unresolved by
default — the code raises an informative error rather than inventing values).

## 6. Tracking-cost definition (Eqs. (14)-(15) of [Lu2024]) — **[paper]**

Implemented in `radar/tracking_cost.py`:

```
Eq. (14): c_t(T_t, τ_t) = trace(E P_{t|t} E^T) = P_{t|t}[0,0] + P_{t|t}[1,1]   [m²]
Eq. (15): E = [[1,0,0,0],[0,1,0,0]]   (position selector, 2 x 4)
```

Pure function of the posterior covariance; no scanning, weighting or power logic.
Per-task costs `c^n_t` for n = 1..N tasks feed the utility (Eq. (23)).

## 7. Scanning-performance abstraction (Eq. (21) of [Lu2024]) — **[paper] + [synthetic] provider**

Implemented in `radar/scanning_model.py`:

- **Eq. (21) metric [paper]:** `Γ = S_max / S_0` — modeled detectable area over
  reference area, dimensionless, **uncapped** (Γ > 1 is reported, never clipped);
  `AreaRatioMetric` normalizes with a configured `S_0` (itself unresolved —
  scenario-dependent parameter).
- **`ScanningPerformanceModel` protocol [proposed]:** maps available scanning time
  `τ_s` [s] → `ScanningPerformance(gamma, detectable_area_m2)`; deterministic.
- **`SyntheticLinearScanningModel` [synthetic]:** a clearly labeled synthetic linear
  provider (`S_det ∝ τ_s` up to configured limits) so the resource-accounting →
  metric chain can be exercised WITHOUT the physical model. It is NOT validated
  detection physics (see item 11 of `ASSUMPTIONS_AND_LIMITATIONS.md`).
- **Physical Eqs. (16)-(20) (radar range equation / detectable-area build-up): NOT
  implemented** — the paper lists the symbols (Table I) without values; all twelve
  associated parameters are unresolved (`scanning.*`), and implementing that chain
  is out of M8A scope.

## 8. Resource allocation, accounting and constraints — **[paper] structure + [proposed] semantics**

Implemented in `optimization/allocation.py`, `optimization/resource_accounting.py`,
`optimization/constraints.py`.

- **Allocation [paper-shaped, proposed representation]:** `ResourceAllocation` =
  ordered per-task dwell times `τ_i` [s], each a member of the discrete action set
  (Table II: `{0, 0.5, 1.0, ..., 25.0}` values per our extraction), plus indices.
  Content-addressed (`content_id`) for provenance.
- **Time accounting [paper Eq. (24) convention]:**
  `total_dwell = Σ_i τ_i` [s]; normalized usage `Σ_i τ_i / T0` [dimensionless];
  **Eq. (24) remaining scanning time:** `τ_s = T0 · max(0, 1 − Σ_i τ_i / T0)` [s].
- **Budget constraint — two explicitly non-equivalent semantics [paper vs proposed]:**
  - `discounted_cumulative` **[paper Eq. (22)]**: `Σ_m γ^m (Σ_i τ_i^m / T0 − Θ_max) ≤ 0`
    over slots m with discount γ = 0.9 (Table II). A LOCAL per-slot feasibility check
    CANNOT decide this constraint (explicitly documented in `FeasibilityReport`:
    local over-budget ≠ discounted-budget infeasibility and vice versa).
  - `hard_per_slot` **[proposed]**: `Σ_i τ_i / T0 ≤ Θ_max = 1` every slot — stricter,
    BO-configuration profile; local feasibility IS decidable.
  The default `ResourceConfig.time_budget_semantics` is `discounted_cumulative`;
  proposed-profile experiments must opt in explicitly. The two are never conflated.
- **Peak power / average power / energy / waveform-dependent duty cycle: NOT
  modeled.** The only modeled resource is dwell TIME (item 12 of
  `ASSUMPTIONS_AND_LIMITATIONS.md` documents what each extension would require).

## 9. Combined utility (Eq. (23) of [Lu2024]) — **[paper] structure + [proposed] defaults**

Implemented in `optimization/objective.py` (`ObjectiveEvaluator`, non-mutating
candidate evaluation against frozen snapshots):

```
Eq. (23):  U_t = − Σ_n w_n c^n_t + β · Γ(τ_s,t)
           [utility dimension: tracking term in m² · w_n; scanning term in β · (dimensionless)]
```

- `c^n_t` from a `TrackingCostProvider` (snapshot-diagnostic Eq. (14) provider or the
  synthetic linear provider — explicit interfaces).
- `β = 1e6` (paper value, preserved **for reproduction only**; see item 13 of
  `ASSUMPTIONS_AND_LIMITATIONS.md` for the scaling review).
- `w_n` per-task weights: **unresolved in the register**; the code default is
  UNIFORM weights, explicitly a proposed software default, NOT a paper value.
- Failure semantics: evaluation errors yield `status="failed"` with utility `None`
  (zero is never substituted); infeasible candidates yield `status="infeasible"`;
  strict mode re-raises unexpected exception types (M4 review decision).
- **Truth inaccessibility [proposed guarantee]:** the evaluator sees frozen
  estimator-facing snapshots only; ground-truth state never reaches estimator or
  optimizer (guarded by tests).

**Dimensional-consistency finding (reviewed, unchanged code):** the two Eq. (23)
terms have different natural dimensions (m²·w vs β·dimensionless). The paper's β
spans this gap by construction; the correctness of that trade-off for CHANGED cost
scales or metric definitions is exactly the item-13 review concern. The implemented
evaluator preserves the paper structure literally; no rescaling has been applied.

## 10. Optimization problem interface (generic BO) — **[proposed]**

Implemented in `optimization/bo/problem.py` (`OptimizationProblem`), radar-free:

```
maximize f(x) over x ∈ X ⊂ R^d (mixed continuous/discrete bounds)
subject to feasibility predicate is_feasible(x)     [complete-vector checks]
```

Components (all **[proposed]** unless noted): observation dataset (complete M×d
vectors; duplicates tolerated only under declared observation noise), seeded feasible
sampler OR validated explicit shared initial design (M8A), float64-CPU `SingleTaskGP`
with explicit Matérn-5/2 ARD kernel + Normalize/Standardize inputs + standardized
outcomes (kernel choice **[paper, Benz2023]**), conventional `LogExpectedImprovement`
(no xi parameter in the installed API; Benzaghta's xi = 0.01 preserved in provenance
only), candidate pools of 500 (**[paper, Benz2023]** Nc), single-candidate
block-cyclic loop with joint/alternating/pair-wise configurations of ONE engine
(**[proposed]** architectures), budget accounting on ATTEMPTED evaluations (failures
included), incumbent = best observed value (never GP-predicted), and stopping
conditions distinguishing evaluations/iterations/visits/cycles/skips.

## 11. Component-to-source mapping summary

| Component | Code | Source equation(s) | Status |
|---|---|---|---|
| State, motion, process noise | `environment/` | [Lu2024] Eqs. (2)-(3) | implemented [paper] |
| Measurement geometry | `radar/measurement_model.py` | [Lu2024] (planar r/θ; Eq. (7) residual form) | implemented [paper] |
| Measurement noise | `radar/measurement_noise.py` | [Lu2024] Eqs. (5)/(6)/(8) | implemented [paper]; SNR0 missing |
| EKF | `radar/extended_kalman_filter.py` | [Lu2024] Eqs. (9)-(13) | implemented [paper] (+ Joseph [proposed]) |
| Tracking cost | `radar/tracking_cost.py` | [Lu2024] Eqs. (14)-(15) | implemented [paper] |
| Scanning metric | `radar/scanning_model.py` | [Lu2024] Eq. (21) | metric [paper]; provider [synthetic]; Eqs. (16)-(20) NOT implemented |
| Time accounting | `optimization/resource_accounting.py` | [Lu2024] Eq. (24) | implemented [paper] |
| Budget constraint | `optimization/constraints.py` | [Lu2024] Eq. (22) + [proposed] hard-per-slot | both implemented, non-equivalent, documented |
| Utility | `optimization/objective.py` | [Lu2024] Eq. (23) | implemented [paper] structure; w_n default [proposed] |
| BO surrogate/acquisition/pools | `optimization/bo/` | [Benz2023] (Matérn-5/2, EI, Nc=500) | methodological adaptation [proposed implementation] |
| Block architectures | `optimization/bo/{blocks,scheduling,block_candidates,bo_loop}.py` | none (proposed) | [proposed] |
| Synthetic benchmarks A/B/C + 4/8/16-D families | `optimization/bo/benchmarks*.py` | none (synthetic) | [synthetic] |

## 12. Identified dimensional, definitional and assumption issues (review output; nothing changed silently)

1. **Eq. (23) term-scale gap** (m² vs β·dimensionless): resolved by the paper's β =
   1e6 for reproduction; any change of tracking-cost scale or scanning definition
   reopens the issue → item-13 normalization plan. No code change made.
2. **`Γ` uncapped** [paper]: Γ > 1 is meaningful only if `S_0` is a true reference
   (not the maximum); with the synthetic provider this is moot but documented.
3. **`SNR0` missing**: the Eq. (5)-(6)-(8) chain is implemented but cannot produce an
   absolute noise scale without a supplied (labeled) value — flagged, not worked around.
4. **Eq. (22) vs hard-per-slot**: mathematically different constraints (cumulative
   discounted vs per-slot); both are retained with explicit non-equivalence notes.
5. **Uniform `w_n` default** is a software default, not paper-reported; labeled.
6. **Joseph-form update** is a numerically equivalent alternative to Eq. (13); the
   literal form is default; equivalence is tested, never assumed.
7. **Mixed-variable decision space** (discrete dwell set × future continuous
   extensions) is handled by the generic interface; no continuous dwell relaxation
   exists.
8. **Undefined quantities**: none found beyond the documented unresolved parameters
   (register §3); no silent defaults were discovered in the audit.
