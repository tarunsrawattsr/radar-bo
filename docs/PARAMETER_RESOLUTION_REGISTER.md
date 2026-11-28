# Parameter Resolution Register (M8A)

_Generated from `provenance_register.json` at commit `7e447be64fea` (84 entries; 17 unresolved). This register is the single source of truth for parameter provenance; the JSON register is authoritative and this document is its review companion. No missing value has been replaced by an arbitrary constant; illustrative synthetic values in code are always labeled synthetic._

## 1. Classification taxonomy (M8A item 5)

| Class | Meaning |
|---|---|
| Paper-reported parameter | Value printed in a primary reference (table or prose); reproducible from the source. |
| Missing paper parameter | Symbol REQUIRED by a paper equation but never assigned a value in the paper. |
| Universal physical constant | Exact SI-defined constant. |
| Hardware-dependent parameter | Determined by radar/transmitter/receiver hardware specification. |
| Scenario-dependent parameter | Determined by the research scenario definition (targets, geometry, scan plan). |
| Proposed research assumption | Our own methodological/design decision — never attributed to a paper. |
| Experimental setting | Explicit experiment-design value (inherited settings are labeled as such, not proven optimal). |
| Derived quantity | Computed from other independently known quantities (e.g. analytic references). |

## 2. Full inventory (all entries)

Columns: entry (dotted config path), symbol/meaning, unit, source class, current value, status, affected model components, M8A classification.

| Entry | Symbol / meaning | Unit | Source | Current value | Status | Affected components | M8A classification |
|---|---|---|---|---|---|---|---|
| `tracking.reference_range_variance` | sigma^2_{r,0}: reference range-measurement variance [m^2] | m^2 | paper_lu_2024 | `100.0` | RESOLVED | measurement-noise model (Eqs. (5)/(6)/(8)) | Paper-reported parameter |
| `tracking.reference_angle_variance` | sigma^2_{theta,0}: reference azimuth-measurement variance [rad^2] | rad^2 | paper_lu_2024 | `0.0004` | RESOLVED | measurement-noise model (Eqs. (5)/(6)/(8)) | Paper-reported parameter |
| `tracking.maneuverability_noise` | sigma_w: acceleration-noise quantity of Table II (multiplier semantics per M3 decision) | (m/s^2)^2 | paper_lu_2024 | `25.0` | RESOLVED | process-noise covariance (Eq. (3)) | Paper-reported parameter |
| `tracking.reference_distance` | r_0: reference range [m] | m | paper_lu_2024 | `3000.0` | RESOLVED | SNR model (Eq. (5)) | Paper-reported parameter |
| `tracking.reference_dwell_time` | tau_0: reference dwell time [s] | s | paper_lu_2024 | `1.0` | RESOLVED | SNR model (Eq. (5)) | Paper-reported parameter |
| `tracking.revisit_interval` | T0: revisit interval [s] | s | paper_lu_2024 | `2.5` | RESOLVED | motion model (Eqs. (2)-(3)), budget normalization (Eq. (22)) | Paper-reported parameter |
| `tracking.reference_snr` | SNR0: reference linear SNR — required by Eq. (5), never reported in the paper | linear SNR | paper_lu_2024 | `None` | UNRESOLVED (None) | SNR model (Eq. (5)) | Missing paper parameter |
| `resource.dwell_time_action_set` | discrete dwell-time action set [s] (Table II) | s | paper_lu_2024 | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` | RESOLVED | allocation representation, constraints, BO decision variables | Paper-reported parameter |
| `resource.normalized_time_budget_threshold` | Theta_max = 1 (normalized by T0) | dimensionless | paper_lu_2024 | `1.0` | RESOLVED | budget constraints (Eq. (22)) | Paper-reported parameter |
| `resource.tracking_scanning_tradeoff` | beta = 1e6 trade-off coefficient | (m^2 * weight) per unit Gamma | paper_lu_2024 | `1000000.0` | RESOLVED | combined utility (Eq. (23)) | Paper-reported parameter |
| `resource.discount_factor` | gamma discount factor | dimensionless | paper_lu_2024 | `0.9` | RESOLVED | discounted cumulative budget (Eq. (22)) | Paper-reported parameter |
| `scanning.peak_transmit_power` | P_t: peak transmit power [W]; hardware/scenario specification required | W (peak pulse power) | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.transmit_antenna_gain` | G_t: transmit antenna gain (linear) | linear (dimensionless) | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.receive_antenna_gain` | G_r: receive antenna gain (linear) | linear (dimensionless) | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.wavelength` | lambda: carrier wavelength [m] (hardware/frequency plan) | m | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.radar_cross_section` | sigma_RCS: target radar cross-section [m^2] (target- and aspect-dependent) | m^2 | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Scenario-dependent parameter |
| `scanning.loss_factor` | L: system loss factor (linear, >= 1) | linear, >= 1 | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.system_temperature` | T_sys: system noise temperature [K] | K | paper_lu_2024 | `None` | UNRESOLVED (None) | radar range equation (Eqs. (16)-(20), NOT implemented) | Hardware-dependent parameter |
| `scanning.beam_angular_spacing` | angular spacing between beams [deg] (scan plan) | deg | paper_lu_2024 | `None` | UNRESOLVED (None) | detectable-area geometry (Eq. (16)-(21) chain, NOT implemented) | Scenario-dependent parameter |
| `scanning.reference_area` | S_0: reference area [m^2] (scenario definition) | m^2 | paper_lu_2024 | `None` | UNRESOLVED (None) | scanning metric normalization (Eq. (21)) | Scenario-dependent parameter |
| `scanning.detection_probability` | P_d: design detection probability | probability | paper_lu_2024 | `None` | UNRESOLVED (None) | detector model (NOT implemented) | Missing paper parameter |
| `scanning.false_alarm_probability` | P_fa: design false-alarm probability | probability | paper_lu_2024 | `None` | UNRESOLVED (None) | detector model (NOT implemented) | Missing paper parameter |
| `scanning.required_detection_snr` | SNR_min: minimum detectable SNR (linear) | linear SNR | paper_lu_2024 | `None` | UNRESOLVED (None) | detector model (NOT implemented) | Missing paper parameter |
| `scanning.boltzmann_constant` | k_B = 1.380649e-23 J/K (SI 2019 exact) | J/K | standard_constant | `1.380649e-23` | RESOLVED | noise power (Eqs. (16)-(20) chain, NOT implemented) | Universal physical constant |
| `tracking.initialization_policy` | zero-vector initialization (diagnostic) vs measurement-driven (operational synthetic) | — | paper_lu_2024 | `paper_zero_identity` | RESOLVED | estimator initialization | Paper-reported parameter |
| `tracking.measurement_driven_initial_velocity` | velocity prior mean for measurement-driven initialization [m/s] — no paper value; per-scenario | (m/s, m/s) | proposed_assumption | `None` | UNRESOLVED (None) | estimator initialization (operational synthetic) | Scenario-dependent parameter |
| `tracking.measurement_driven_initial_velocity_variance` | velocity prior variance per axis [(m/s)^2] — no paper value; per-scenario | (m/s)^2 | proposed_assumption | `None` | UNRESOLVED (None) | estimator initialization (operational synthetic) | Scenario-dependent parameter |
| `tracking.covariance_update_method` | 'paper_equation_13' literal vs 'joseph' form (numerical-stability alternative; equivalence tested) | — | proposed_assumption | `paper_equation_13` | RESOLVED | EKF update (Eq. (13)) | Proposed research assumption |
| `tracking.measurement_variance_semantics` | 'inverse_snr' (paper) vs alternatives; inverse-SNR is operational | — | paper_lu_2024 | `sigma^2_{r|theta,t} = sigma^2_{r|theta,0} / SNR_t (reference variance divided by linear SNR)` | RESOLVED | measurement-noise model (Eqs. (5)/(6)/(8)) | Proposed research assumption |
| `optimization.kernel` | Matern-5/2 (Benzaghta et al.) | — | paper_benzaghta_2023 | `matern_5_2` | RESOLVED | BO surrogate | Paper-reported parameter |
| `optimization.acquisition_xi` | xi = 0.01 (Benzaghta); no effect with installed LogEI API — preserved in register only | dimensionless | paper_benzaghta_2023 | `0.01` | RESOLVED | BO acquisition (provenance only) | Paper-reported parameter |
| `optimization.num_acquisition_candidates` | Nc = 500 (Benzaghta) | count | paper_benzaghta_2023 | `500` | RESOLVED | BO candidate pools | Paper-reported parameter |
| `optimization.search_mode` | joint/alternating/coordinate configurations of ONE engine | — | proposed_assumption | `joint` | RESOLVED | BO search architecture | Proposed research assumption |
| `scanning.area_ratio_metric` | Gamma = S_max / S_0, uncapped | — | paper_lu_2024 | `Gamma = S_max / S_0 (uncapped)` | RESOLVED | scanning metric (Eq. (21)) | Paper-reported parameter |
| `resource.remaining_scanning_time_convention` | tau_s = T0 * max(0, 1 - sum(tau)/T0) | s | paper_lu_2024 | `tau_s = T0 * max(0, 1 - sum(tau)/T0)` | RESOLVED | time accounting (Eq. (24)) | Paper-reported parameter |
| `objective.eq23_aggregation` | U_t = -sum_n w_n c^n_t + beta * Gamma | — | paper_lu_2024 | `U = -sum_n w_n c^n_t + beta * Gamma(tau_s)` | RESOLVED | combined utility (Eq. (23)) | Paper-reported parameter |
| `resource.task_utility_weights` | w_n per-task weights; None = uniform (proposed software default, NOT a paper value) | dimensionless | proposed_assumption | `None` | UNRESOLVED (None) | combined utility (Eq. (23)) | Proposed research assumption |
| `scanning.synthetic_linear_scanning_model` | synthetic linear S_det(tau) — clearly labeled synthetic; NOT validated detection physics | m^2/s, m^2 | experimental_setting | `{"area_rate_m2_per_s": 2000.0, "reference_area_m2": 4000.0}` | RESOLVED | scanning abstraction (synthetic provider) | Experimental setting |
| `objective.randomness_policy` | deterministic candidate evaluation; no hidden RNG | — | proposed_assumption | `common_random_numbers` | RESOLVED | objective evaluator determinism | Proposed research assumption |
| `allocation.discrete_membership_policy` | dwell indices must map into the discrete action set | — | proposed_assumption | `exact_match_no_snapping` | RESOLVED | allocation validation | Proposed research assumption |
| `objective.snapshot_truth_exclusion` | estimator/optimizer never see ground-truth state | — | proposed_assumption | `frozen_snapshot_is_estimator_facing` | RESOLVED | truth-inaccessibility guarantee | Proposed research assumption |
| `objective.tracking_provider_m4_scope` | snapshot diagnostic provider vs synthetic linear provider (separated interfaces) | — | proposed_assumption | `snapshot_diagnostic_eq14 (allocation-independent)` | RESOLVED | tracking-cost providers | Proposed research assumption |
| `bo.problem_interface.scope` | application-free OptimizationProblem (M5) | — | proposed_assumption | `application_free` | RESOLVED | generic BO problem interface | Proposed research assumption |
| `bo.benchmarks.a_b_c.origin` | self-defined A/B/C benchmarks | — | proposed_assumption | `synthetic_self_defined_for_verification` | RESOLVED | M5 synthetic benchmarks | Proposed research assumption |
| `bo.benchmark_a.reference` | analytic exact f* = 10 at (3.5, 7.2) | dimensionless | proposed_assumption | `{"f_star": 10.0, "u_star": 3.5, "v_star": 7.2}` | RESOLVED | M5 references | Derived quantity |
| `bo.benchmark_b.reference` | enumeration exact f* = 11 (analytic clamped inner optimum) | dimensionless | proposed_assumption | `{"f_star": 11.0, "d_star": 2.0, "x_star": 2.0}` | RESOLVED | M5 references | Derived quantity |
| `bo.benchmark_c.reference` | numerical approximate f* = 4.5295 (DE + grid; no BO) | — | proposed_assumption | `numerical_approximate (computed by the reference solver at run time)` | RESOLVED | M5 references | Derived quantity |
| `bo.duplicate_tolerance` | 1e-9 absolute tolerance | dimensionless | proposed_assumption | `1e-09` | RESOLVED | dataset/candidate dedup | Proposed research assumption |
| `bo.feasibility_tolerance` | 1e-9 absolute tolerance | dimensionless | proposed_assumption | `1e-09` | RESOLVED | feasibility checks | Proposed research assumption |
| `bo.consecutive_failure_cap` | 3 consecutive objective/GP failures | count | proposed_assumption | `3.0` | RESOLVED | loop stopping | Proposed research assumption |
| `bo.initial_design_sampling` | seeded uniform feasible sampler; M8A adds validated explicit designs | — | experimental_setting | `seeded_uniform_feasible_random` | RESOLVED | BO initial design | Experimental setting |
| `bo.refit_policy` | fresh refit per successful observation (experimental) | — | experimental_setting | `fresh_refit_per_observation` | RESOLVED | BO GP fitting | Experimental setting |
| `bo.selection_policy` | argmax LogEI over feasible pool; ties deterministic | — | proposed_assumption | `argmax_finite_log_ei_first_index_tiebreak` | RESOLVED | BO candidate selection | Proposed research assumption |
| `bo.failures_policy` | failed attempts consume budget; never enter the success dataset | — | proposed_assumption | `consume_budget_log_separately` | RESOLVED | budget accounting | Proposed research assumption |
| `bo.incumbent_policy` | best OBSERVED value only; never GP-predicted | — | proposed_assumption | `best_observed_never_predicted` | RESOLVED | incumbent definition | Proposed research assumption |
| `experiment.m5_synthetic_validation_budget` | budget 100, initial design 30, pool 500, seeds 0-9 | count | experimental_setting | `{"evaluation_budget": 100.0, "initial_design": 30.0}` | RESOLVED | M5 experiment | Experimental setting |
| `bo.blocks.representation` | index-based VariableBlock (application-free) | — | proposed_assumption | `variable_blocks_index_based` | RESOLVED | M6 block architecture | Proposed research assumption |
| `bo.scheduling.strategy` | deterministic cyclic; visit tickets | — | proposed_assumption | `deterministic_cyclic` | RESOLVED | M6 scheduler | Proposed research assumption |
| `bo.current_config_vs_incumbent` | current configuration (search center) distinct from global incumbent | — | proposed_assumption | `two_distinct_objects` | RESOLVED | M6 engine semantics | Proposed research assumption |
| `bo.acceptance_policy` | improvement-only current-configuration acceptance | — | proposed_assumption | `improvement_only_current_config` | RESOLVED | M6 acceptance | Proposed research assumption |
| `bo.evaluations_per_block_visit` | 1 evaluation per block visit (default) | count | experimental_setting | `1.0` | RESOLVED | M6 engine | Experimental setting |
| `bo.full_gp_dimension_vs_block_dimension` | ONE shared full-dimension GP for all blocks | — | proposed_assumption | `gp_trains_full_d` | RESOLVED | M6 surrogate | Proposed research assumption |
| `bo.benchmark_stagnation_lp.reference` | analytic LP optimum f* = 9.6 at (1.6, 0, 1.6, 0) | dimensionless | proposed_assumption | `{"f_star": 9.6, "a_star": 1.6, "b_star": 0.0}` | RESOLVED | M6 references | Derived quantity |
| `bo.highdim_benchmarks.origin` | self-defined separable/coupled/constrained-mixed/stagnation families | — | proposed_assumption | `synthetic_self_defined_4_8_16` | RESOLVED | M6 benchmarks 4/8/16-D | Proposed research assumption |
| `experiment.m6_block_architecture_budget` | budget 100 total, initial 30, pool 500, seeds 0-9 | count | experimental_setting | `{"evaluation_budget": 100.0, "initial_design": 30.0}` | RESOLVED | M6 experiment | Experimental setting |
| `optimization.num_initial_observations` | 30 (inherited M5/M6; not proven optimal) | count | experimental_setting | `30` | RESOLVED | BO initial design size | Experimental setting |
| `optimization.evaluation_budget` | UNRESOLVED by design: each experiment sets it explicitly (no paper default) | count | experimental_setting | `None` | UNRESOLVED (None) | BO budget | Experimental setting |
| `environment.num_targets` | number of synthetic targets | count | experimental_setting | `1` | RESOLVED | synthetic scenario | Experimental setting |
| `environment.random_seed` | scenario RNG seed | — | experimental_setting | `0` | RESOLVED | reproducibility | Experimental setting |
| `experiment.repetitions` | seeds 0-9 replicates | count | experimental_setting | `5` | RESOLVED | experiment design | Experimental setting |
| `resource.time_budget_semantics` | 'discounted_cumulative' (paper) vs 'hard_per_slot' (proposed, stricter) — NOT equivalent | — | proposed_assumption | `discounted_cumulative` | RESOLVED | budget semantics | Proposed research assumption |
| `experiment.m7_runner_architecture` | atomic per-run records, statuses, resume compatibility | policy | proposed_assumption | `{"checkpoint_per_run_files": 1.0, "atomic_write": 1.0, "resume_compatibility_checks": 7.0}` | RESOLVED | M7 experiment runner | Proposed research assumption |
| `experiment.m7_shared_initialization_policy` | same seed + stored-record verification; M8A adds validated explicit shared designs | policy | proposed_assumption | `{"policy": 1.0}` | RESOLVED | comparison fairness | Proposed research assumption |
| `experiment.m7_statistical_conventions` | paired stats, bootstrap (B=10000, seed 2026), Wilcoxon+Holm, tie tol 1e-9 | setting | proposed_assumption | `{"bootstrap_replications": 10000.0, "bootstrap_seed": 2026.0, "paired_tie_tolerance": 1e-09, "confidence_level": 0.95}` | RESOLVED | statistical analysis | Proposed research assumption |
| `experiment.m7_threshold_definitions` | gap-closure fractions frozen before comparative experiments | fraction | proposed_assumption | `{"gap_closure_fractions": 0.9}` | RESOLVED | threshold metrics | Proposed research assumption |
| `experiment.m7_runtime_accounting` | disjoint stage timers + explicit unattributed overhead (M8A: historical nested convention documented) | policy | proposed_assumption | `{"stage_timers": 5.0, "unattributed_reported": 1.0}` | RESOLVED | runtime decomposition | Proposed research assumption |
| `experiment.m7_candidate_profiling_baseline` | M6 algorithm preserved as versioned baseline | policy | proposed_assumption | `{"algorithm_baseline": 1.0}` | RESOLVED | candidate generation | Proposed research assumption |
| `experiment.m7_validation_matrix` | 2 benchmarks x 5 methods x 3 seeds, budget 40 | count | experimental_setting | `{"benchmarks": 2.0, "seeds": 3.0, "evaluation_budget": 40.0, "initial_design": 10.0}` | RESOLVED | M7 validation | Experimental setting |
| `experiment.m7_milestone_reconciliation` | 8-milestone roadmap vs 9-milestone instruction stream | count | proposed_assumption | `{"instruction_stream_milestones": 9.0, "original_roadmap_milestones": 8.0}` | RESOLVED | documentation | Proposed research assumption |
| `experiment.m8a_run_count_reconciliation` | 220 archived full-matrix records (the '230' figure was a narrative arithmetic error, corrected) | count | proposed_assumption | `{"archived_full_matrix_records": 220.0, "pilot_records": 32.0}` | RESOLVED | M8A documentation | Derived quantity |
| `experiment.m8a_r1_source_verification` | both source PDFs re-extracted and verified line-by-line (equations, tables, references, DOIs) | count | proposed_assumption | `{"pdfs_reextracted": 2.0, "icc_pages": 6.0, "globecom_pages": 6.0}` | RESOLVED | M8A-R1 Phase A | Documentation |
| `paper_lu_2024.training_weight_set` | training weights drawn from {1, 3, 5}; recorded fact, NOT adopted as our default | dimensionless | paper_lu_2024 | `[1.0, 3.0, 5.0]` | RESOLVED | CDRL training (not reproduced here) | Paper-reported parameter |
| `paper_lu_2024.testing_weight_rule` | time-varying test weights via Eqs. (31)-(32) from ESTIMATED distances/velocities, cap 10 | definition | paper_lu_2024 | `Eq. (31): w_n = min{10, 4 + 10000/d0} if d0 <= 10000 else 1, with d0 = min(d_n_r, d_n_a) (estimated distances to radar and to asset); Eq. (32): w_n = min{10, w_n + 5} if A_n < 0 else w_n, with A_n = x_n . xdot_n (position dot velocity; A_n < 0 = target approaching the radar). Weights are TIME-VARYING during tests, computed from ESTIMATED positions/velocities, capped at 10.` | RESOLVED | CDRL testing (not reproduced here) | Paper-reported parameter |
| `experiment.m8a_shared_initial_design_interface` | validated immutable initial_design_configs + framework explicit_shared mode (backward compatible) | policy | proposed_assumption | `{"engine_parameter": 1.0, "framework_mode": 1.0}` | RESOLVED | engine + framework | Proposed research assumption |

## 3. Unresolved parameters — resolution-path analysis (M8A item 6)

### `tracking.reference_snr`

- **Meaning/unit:** SNR0: reference linear SNR — required by Eq. (5), never reported in the paper [linear SNR]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eq. (5)
- **Universal value exists:** No — platform/radar/waveform specific.
- **Hardware dependent:** Yes (emitter and receiver chain).
- **Paper provides sufficient information:** No — the paper uses SNR0 but reports no value (verified during M2/M3 extraction for the ICC paper; re-verified 2026-11-26 against the authors' extended journal version arXiv 2606.05526, whose Table III also omits SNR0).
- **Derivable from known quantities:** Not from other paper quantities; SNR0 anchors the absolute noise scale.
- **Additional reference / calibration required:** Yes — either the paper's own experimental setup section (re-verify from source PDF) or a calibrated scenario definition.
- **Sensitivity analysis appropriate:** Yes — SNR0 linearly scales all measurement variances (inverse-SNR), directly affecting tracking cost magnitude; a log-spaced sweep is appropriate once resolved.
- **Needed for:** BASELINE REPRODUCTION of the paper's tracking-noise scale (the operational synthetic path can substitute an explicitly synthetic SNR0, clearly labeled).

### `scanning.peak_transmit_power`

- **Meaning/unit:** P_t: peak transmit power [W]; hardware/scenario specification required [W (peak pulse power)]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** No.
- **Hardware dependent:** Yes — transmitter specification.
- **Paper provides sufficient information:** No — Table I lists the symbol without a value (verified during M2 extraction).
- **Derivable from known quantities:** No.
- **Additional reference / calibration required:** Yes — radar hardware datasheet or scenario specification.
- **Sensitivity analysis appropriate:** Yes — enters the radar range equation linearly (in power terms); however the ENTIRE physical chain Eqs. (16)-(20) is NOT implemented, so sensitivity analysis is deferred until activation is approved.
- **Needed for:** FUTURE EXTENSION ONLY (physical scanning model).

### `scanning.transmit_antenna_gain`

- **Meaning/unit:** G_t: transmit antenna gain (linear) [linear (dimensionless)]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.receive_antenna_gain`

- **Meaning/unit:** G_r: receive antenna gain (linear) [linear (dimensionless)]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.wavelength`

- **Meaning/unit:** lambda: carrier wavelength [m] (hardware/frequency plan) [m]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.radar_cross_section`

- **Meaning/unit:** sigma_RCS: target radar cross-section [m^2] (target- and aspect-dependent) [m^2]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.loss_factor`

- **Meaning/unit:** L: system loss factor (linear, >= 1) [linear, >= 1]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.system_temperature`

- **Meaning/unit:** T_sys: system noise temperature [K] [K]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.beam_angular_spacing`

- **Meaning/unit:** angular spacing between beams [deg] (scan plan) [deg]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.reference_area`

- **Meaning/unit:** S_0: reference area [m^2] (scenario definition) [m^2]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** PHYSICAL SCANNING METRIC (the synthetic provider does not use S_0).

### `scanning.detection_probability`

- **Meaning/unit:** P_d: design detection probability [probability]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.false_alarm_probability`

- **Meaning/unit:** P_fa: design false-alarm probability [probability]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `scanning.required_detection_snr`

- **Meaning/unit:** SNR_min: minimum detectable SNR (linear) [linear SNR]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eqs. (16)-(21), Table I (symbols only, no values)
- **Universal value exists:** see below
- **Hardware dependent:** see below
- **Paper provides sufficient information:** No (see below)
- **Derivable from known quantities:** No
- **Additional reference / calibration required:** see below
- **Sensitivity analysis appropriate:** Yes — after resolution
- **Needed for:** FUTURE EXTENSION ONLY.

### `tracking.measurement_driven_initial_velocity`

- **Meaning/unit:** velocity prior mean for measurement-driven initialization [m/s] — no paper value; per-scenario [(m/s, m/s)]
- **Paper reference:** M3 review instruction 9 (no paper value)
- **Universal value exists:** No.
- **Hardware dependent:** No — scenario dependent.
- **Paper provides sufficient information:** No — the paper's zero-vector initialization is retained as the diagnostic configuration only (M3 decision); no operational prior is given.
- **Derivable from known quantities:** Partially — could be estimated from the first two observations in a fully measurement-driven design; that variant is a PROPOSED future change requiring approval.
- **Additional reference / calibration required:** No external reference required; a per-scenario value (clearly labeled synthetic) is acceptable.
- **Sensitivity analysis appropriate:** Yes — initialization bias affects transient tracking cost; recommend a small initialization-sensitivity study before any paper-style experiment.
- **Needed for:** OPERATIONAL SYNTHETIC ESTIMATION (currently supplied per-scenario with explicit synthetic labels; unresolved in the register by design).

### `tracking.measurement_driven_initial_velocity_variance`

- **Meaning/unit:** velocity prior variance per axis [(m/s)^2] — no paper value; per-scenario [(m/s)^2]
- **Paper reference:** M3 review instruction 9 (no paper value)
- **Universal value exists:** No.
- **Hardware dependent:** No — scenario/estimator-tuning dependent.
- **Paper provides sufficient information:** No value in the ICC paper (zero-vector init only, retained as diagnostic). CANDIDATE VALUE from the authors' extended journal version (arXiv 2606.05526, Eq. (14)): P_0 = diag([10000, 10000, 100, 100]) — position variance 1e4 m^2, velocity variance 1e2 m^2/s^2 per axis, described by the authors as 'initialized empirically'. NOT adopted; adoption requires separate review + approval.
- **Derivable from known quantities:** Partially — could be tied to the first-observation covariance ratio (proposed; requires approval).
- **Additional reference / calibration required:** No; per-scenario with sensitivity study.
- **Sensitivity analysis appropriate:** Yes — controls estimator transient; include with the initialization study.
- **Needed for:** OPERATIONAL SYNTHETIC ESTIMATION.

### `resource.task_utility_weights`

- **Meaning/unit:** w_n per-task weights; None = uniform (proposed software default, NOT a paper value) [dimensionless]
- **Paper reference:** Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225; Eq. (23) weights w_n
- **Universal value exists:** No.
- **Hardware dependent:** No — research-preference/scenario dependent.
- **Paper provides sufficient information:** No — the paper reports the weights symbolically; per-task values were not preserved in our extractions (verified M2).
- **Derivable from known quantities:** No.
- **Additional reference / calibration required:** Yes — re-verify from the source PDF (possibly a table exists) or define a research scenario (e.g. uniform) with explicit labeling.
- **Sensitivity analysis appropriate:** Yes — weights change the relative importance of tasks in the utility; a weight-sensitivity design is part of the M8A normalization plan.
- **Needed for:** BASELINE REPRODUCTION IF the paper's values exist; uniform default is a PROPOSED software default for synthetic work.

### `optimization.evaluation_budget`

- **Meaning/unit:** UNRESOLVED by design: each experiment sets it explicitly (no paper default) [count]
- **Paper reference:** none
- **Universal value exists:** No — experiment-design choice.
- **Hardware dependent:** Only via compute budget (wall-clock), not physically.
- **Paper provides sufficient information:** N/A — Benzaghta's settings were adopted for the synthetic studies (100/30/500); no application budget exists yet.
- **Derivable from known quantities:** No.
- **Additional reference / calibration required:** No — must be set explicitly per experiment by design (UnresolvedParameterError if absent).
- **Sensitivity analysis appropriate:** Yes — the M7 sensitivity infrastructure supports budget scenarios; a budget-sensitivity study is planned before any application claim.
- **Needed for:** EVERY EXPERIMENT (explicit by design).

## 4. Baseline reproduction vs future extensions (M8A item 5)

**Required for baseline reproduction of Lu et al.:** the Table II parameters (all RESOLVED paper-reported) plus `tracking.reference_snr` (MISSING paper parameter — the only gap in the tracking chain) and, if the paper's per-task weights exist, `resource.task_utility_weights` (re-verification from the source PDF required). The physical scanning chain (Eqs. (16)-(20)) is NOT required for the tracking/utility baseline because it is not implemented; `scanning.reference_area` is required only when a physical Eq. (21) normalization is activated.

**Required only for future extensions:** the TWELVE unresolved `scanning.*` parameters of the unimplemented radar-range/detector chain — 7 hardware symbols of ICC Table I (P_t, G_t, G_r, lambda, sigma_RCS, L, T_s), the beam angular spacing phi (Eq. (16)), the reference area S_0 (Eq. (21)), and the detection triple P_d / P_f / SNR_min (Sec. III-B, Shnidman) — plus any proposed corridor-specific scanning metric parameters (none defined yet). The universal Boltzmann constant is resolved and the per-target ranges R_t/R_r are problem inputs, not configuration parameters (M8A-R1 count correction).

## 5. Extended-version cross-reference (added 2026-11-26)

The authors of [Lu2024] have an extended journal version on arXiv (arXiv 2606.05526, 'Constrained Deep Reinforcement Learning for Cognitive Radar Resource Management', submitted 2026-06-04; fetched 2026-11-26). It confirms the ICC paper's model equations (motion (2)-(3), measurement (5)-(8), EKF (9)-(13), utility structure) and adds track-initiation content. It differs from the ICC paper in ways that matter for parameter provenance:

| Quantity | ICC 2024 (reproduced here) | Extended arXiv version |
|---|---|---|
| sigma^2_{r,0} | 100 m^2 | 16 m^2 |
| sigma^2_{theta,0} | 4e-4 rad^2 | 1e-6 rad^2 |
| sigma^2_w (maneuverability) | 25 (m/s^2)^2 | 16 m^2/s^4 |
| beta | 1e6 | 3e4 |
| Theta_max | 1 | 0.9 |
| Scanning metric | Gamma = S_det/S_0 (Eq. (21)); U = -Sum w c + beta*Gamma | N_miss (undetected-target count); U = -Sum w c - beta*N_miss (Eq. (19)) |
| Scanning physics | Eqs. (16)-(20) symbols, no values | Eq. (18) radar range equation, same symbols, no values; Shnidman SNR->P_d lookup |
| EKF initialization | zero vector | measurement-driven, empirical P_0 = diag([1e4, 1e4, 1e2, 1e2]) (Eq. (14)) |
| SNR0 | absent | absent (Table III omits it) |
| Solver | DQN (discrete actions) | DDPG (continuous actions) |
| Dwell actions | discrete {0.1, ..., 1.0} s (Sec. V-A.2) | continuous tau in [0, T_0] |
| Dual-variable step size alpha | 15000 (Table II) | 5000 (Table III) |
| Benchmarks | Equal Allocation + Random Allocation (Fig. 5) | EA + DA + OA |
| Training weights | drawn from {1, 3, 5} (Sec. VI) | (case-based, Sec. VII) |
| Testing weights | time-varying rules Eqs. (31)-(32), cap 10 | case II position/direction weights |
| Noise Eq. (6) | PRINTED as sigma^2 = sigma^2_0 * SNR (contradicts own prose) | sigma^2 = sigma^2_0 / SNR (division) |

Benzaghta version note (M8A-R1, verified against the GLOBECOM PDF): the GLOBECOM paper optimizes antenna TILTS + TRANSMIT POWERS with a mean-SINR-mix objective (lambda) on STOCHASTIC system-level simulations (noisy f-tilde; GP with sigma^2*I noise), while the extended journal version optimizes tilts + HPBW on deterministic extraction. Matern-5/2, xi = 0.01, N_c = 500 (10 batches x 50), BoTorch and max-a-posteriori hyperparameter fitting are verified in the GLOBECOM paper itself; block-wise BS-by-BS 2-D subproblems are the paper's own structure (one BS at a time, others fixed at their previous values).

Implications: (a) 'paper-reported' parameter values are VERSION-DEPENDENT — this repository reproduces the ICC 2024 values and does not switch; (b) SNR0 remains missing in both versions, so `tracking.reference_snr` stays unresolved; (c) the extended version's P_f = 1e-3 and P_0 diag values are recorded as CANDIDATE values in the resolution paths above, requiring separate review and approval before any adoption; (d) the extended version's N_miss metric is a DIFFERENT scanning metric from the ICC Gamma — our Eq. (21)/(23) reproduction is ICC-faithful, and the difference is documented, not silently reconciled.

## 6. Parameters requiring USER APPROVAL before resolution (M8A deliverable)

Resolving any of these requires an explicit human decision (value + source + approval recorded in the register). None is resolved by this milestone:

1. `tracking.reference_snr` — supply SNR0 (paper re-verification or an explicitly synthetic scenario value).
2. `resource.task_utility_weights` — confirm whether the paper reports per-task values; otherwise approve the uniform default as the labeled research assumption.
3. `tracking.measurement_driven_initial_velocity` + `..._variance` — approve per-scenario synthetic values (or the proposed first-two-observations estimator) for operational synthetic runs.
4. The TWELVE unresolved `scanning.*` parameters (7 Table-I hardware symbols + beam spacing + S_0 + P_d/P_f/SNR_min) — required only if the physical Eqs. (16)-(20) chain is ever approved for implementation (currently out of scope).
5. `optimization.evaluation_budget` — set per experiment (by design unresolved in the register).

## 7. Policy reminders (unchanged by M8A)

- Missing paper values stay `None` in the register; code paths that need them raise `UnresolvedParameterError`.
- Illustrative/synthetic values in code are always labeled synthetic and never enter the register as paper values.
- Unvalidated physical configurations are not activated.
- Any proposed numerical selection requires separate review and approval (M8A item 6).
