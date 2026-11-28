"""Generate docs/PARAMETER_RESOLUTION_REGISTER.md from the live provenance register (M8A items 5-6).

Every one of the 79 register entries is inventoried with symbol/meaning, unit,
source, value, resolution status, affected components and an M8A classification.
The 17 unresolved entries receive the full resolution-path analysis (item 6).
The register JSON is never modified; this is a documentation generator.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Per-entry M8A classification (item 5 categories) and affected components.
CLASSIFICATION: dict[str, tuple[str, str, str]] = {
    # name -> (classification, affected components, notes/meaning-symbol)
    "tracking.reference_range_variance": ("Paper-reported parameter", "measurement-noise model (Eqs. (5)/(6)/(8))",
        "sigma^2_{r,0}: reference range-measurement variance [m^2]"),
    "tracking.reference_angle_variance": ("Paper-reported parameter", "measurement-noise model (Eqs. (5)/(6)/(8))",
        "sigma^2_{theta,0}: reference azimuth-measurement variance [rad^2]"),
    "tracking.maneuverability_noise": ("Paper-reported parameter", "process-noise covariance (Eq. (3))",
        "sigma_w: acceleration-noise quantity of Table II (multiplier semantics per M3 decision)"),
    "tracking.reference_distance": ("Paper-reported parameter", "SNR model (Eq. (5))",
        "r_0: reference range [m]"),
    "tracking.reference_dwell_time": ("Paper-reported parameter", "SNR model (Eq. (5))",
        "tau_0: reference dwell time [s]"),
    "tracking.revisit_interval": ("Paper-reported parameter", "motion model (Eqs. (2)-(3)), budget normalization (Eq. (22))",
        "T0: revisit interval [s]"),
    "tracking.reference_snr": ("Missing paper parameter", "SNR model (Eq. (5))",
        "SNR0: reference linear SNR — required by Eq. (5), never reported in the paper"),
    "resource.dwell_time_action_set": ("Paper-reported parameter", "allocation representation, constraints, BO decision variables",
        "discrete dwell-time action set [s] (Table II)"),
    "resource.normalized_time_budget_threshold": ("Paper-reported parameter", "budget constraints (Eq. (22))",
        "Theta_max = 1 (normalized by T0)"),
    "resource.tracking_scanning_tradeoff": ("Paper-reported parameter", "combined utility (Eq. (23))",
        "beta = 1e6 trade-off coefficient"),
    "resource.discount_factor": ("Paper-reported parameter", "discounted cumulative budget (Eq. (22))",
        "gamma discount factor"),
    "scanning.peak_transmit_power": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "P_t: peak transmit power [W]; hardware/scenario specification required"),
    "scanning.transmit_antenna_gain": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "G_t: transmit antenna gain (linear)"),
    "scanning.receive_antenna_gain": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "G_r: receive antenna gain (linear)"),
    "scanning.wavelength": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "lambda: carrier wavelength [m] (hardware/frequency plan)"),
    "scanning.radar_cross_section": ("Scenario-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "sigma_RCS: target radar cross-section [m^2] (target- and aspect-dependent)"),
    "scanning.loss_factor": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "L: system loss factor (linear, >= 1)"),
    "scanning.system_temperature": ("Hardware-dependent parameter", "radar range equation (Eqs. (16)-(20), NOT implemented)",
        "T_sys: system noise temperature [K]"),
    "scanning.beam_angular_spacing": ("Scenario-dependent parameter", "detectable-area geometry (Eq. (16)-(21) chain, NOT implemented)",
        "angular spacing between beams [deg] (scan plan)"),
    "scanning.reference_area": ("Scenario-dependent parameter", "scanning metric normalization (Eq. (21))",
        "S_0: reference area [m^2] (scenario definition)"),
    "scanning.detection_probability": ("Missing paper parameter", "detector model (NOT implemented)",
        "P_d: design detection probability"),
    "scanning.false_alarm_probability": ("Missing paper parameter", "detector model (NOT implemented)",
        "P_fa: design false-alarm probability"),
    "scanning.required_detection_snr": ("Missing paper parameter", "detector model (NOT implemented)",
        "SNR_min: minimum detectable SNR (linear)"),
    "scanning.boltzmann_constant": ("Universal physical constant", "noise power (Eqs. (16)-(20) chain, NOT implemented)",
        "k_B = 1.380649e-23 J/K (SI 2019 exact)"),
    "tracking.initialization_policy": ("Paper-reported parameter", "estimator initialization",
        "zero-vector initialization (diagnostic) vs measurement-driven (operational synthetic)"),
    "tracking.measurement_driven_initial_velocity": ("Scenario-dependent parameter", "estimator initialization (operational synthetic)",
        "velocity prior mean for measurement-driven initialization [m/s] — no paper value; per-scenario"),
    "tracking.measurement_driven_initial_velocity_variance": ("Scenario-dependent parameter", "estimator initialization (operational synthetic)",
        "velocity prior variance per axis [(m/s)^2] — no paper value; per-scenario"),
    "tracking.covariance_update_method": ("Proposed research assumption", "EKF update (Eq. (13))",
        "'paper_equation_13' literal vs 'joseph' form (numerical-stability alternative; equivalence tested)"),
    "tracking.measurement_variance_semantics": ("Proposed research assumption", "measurement-noise model (Eqs. (5)/(6)/(8))",
        "'inverse_snr' (paper) vs alternatives; inverse-SNR is operational"),
    "optimization.kernel": ("Paper-reported parameter", "BO surrogate",
        "Matern-5/2 (Benzaghta et al.)"),
    "optimization.acquisition_xi": ("Paper-reported parameter", "BO acquisition (provenance only)",
        "xi = 0.01 (Benzaghta); no effect with installed LogEI API — preserved in register only"),
    "optimization.num_acquisition_candidates": ("Paper-reported parameter", "BO candidate pools",
        "Nc = 500 (Benzaghta)"),
    "optimization.search_mode": ("Proposed research assumption", "BO search architecture",
        "joint/alternating/coordinate configurations of ONE engine"),
    "scanning.area_ratio_metric": ("Paper-reported parameter", "scanning metric (Eq. (21))",
        "Gamma = S_max / S_0, uncapped"),
    "resource.remaining_scanning_time_convention": ("Paper-reported parameter", "time accounting (Eq. (24))",
        "tau_s = T0 * max(0, 1 - sum(tau)/T0)"),
    "objective.eq23_aggregation": ("Paper-reported parameter", "combined utility (Eq. (23))",
        "U_t = -sum_n w_n c^n_t + beta * Gamma"),
    "resource.task_utility_weights": ("Proposed research assumption", "combined utility (Eq. (23))",
        "w_n per-task weights; None = uniform (proposed software default, NOT a paper value)"),
    "scanning.synthetic_linear_scanning_model": ("Experimental setting", "scanning abstraction (synthetic provider)",
        "synthetic linear S_det(tau) — clearly labeled synthetic; NOT validated detection physics"),
    "objective.randomness_policy": ("Proposed research assumption", "objective evaluator determinism",
        "deterministic candidate evaluation; no hidden RNG"),
    "allocation.discrete_membership_policy": ("Proposed research assumption", "allocation validation",
        "dwell indices must map into the discrete action set"),
    "objective.snapshot_truth_exclusion": ("Proposed research assumption", "truth-inaccessibility guarantee",
        "estimator/optimizer never see ground-truth state"),
    "objective.tracking_provider_m4_scope": ("Proposed research assumption", "tracking-cost providers",
        "snapshot diagnostic provider vs synthetic linear provider (separated interfaces)"),
    "bo.problem_interface.scope": ("Proposed research assumption", "generic BO problem interface",
        "application-free OptimizationProblem (M5)"),
    "bo.benchmarks.a_b_c.origin": ("Proposed research assumption", "M5 synthetic benchmarks",
        "self-defined A/B/C benchmarks"),
    "bo.benchmark_a.reference": ("Derived quantity", "M5 references",
        "analytic exact f* = 10 at (3.5, 7.2)"),
    "bo.benchmark_b.reference": ("Derived quantity", "M5 references",
        "enumeration exact f* = 11 (analytic clamped inner optimum)"),
    "bo.benchmark_c.reference": ("Derived quantity", "M5 references",
        "numerical approximate f* = 4.5295 (DE + grid; no BO)"),
    "bo.duplicate_tolerance": ("Proposed research assumption", "dataset/candidate dedup",
        "1e-9 absolute tolerance"),
    "bo.feasibility_tolerance": ("Proposed research assumption", "feasibility checks",
        "1e-9 absolute tolerance"),
    "bo.consecutive_failure_cap": ("Proposed research assumption", "loop stopping",
        "3 consecutive objective/GP failures"),
    "bo.initial_design_sampling": ("Experimental setting", "BO initial design",
        "seeded uniform feasible sampler; M8A adds validated explicit designs"),
    "bo.refit_policy": ("Experimental setting", "BO GP fitting",
        "fresh refit per successful observation (experimental)"),
    "bo.selection_policy": ("Proposed research assumption", "BO candidate selection",
        "argmax LogEI over feasible pool; ties deterministic"),
    "bo.failures_policy": ("Proposed research assumption", "budget accounting",
        "failed attempts consume budget; never enter the success dataset"),
    "bo.incumbent_policy": ("Proposed research assumption", "incumbent definition",
        "best OBSERVED value only; never GP-predicted"),
    "experiment.m5_synthetic_validation_budget": ("Experimental setting", "M5 experiment",
        "budget 100, initial design 30, pool 500, seeds 0-9"),
    "bo.blocks.representation": ("Proposed research assumption", "M6 block architecture",
        "index-based VariableBlock (application-free)"),
    "bo.scheduling.strategy": ("Proposed research assumption", "M6 scheduler",
        "deterministic cyclic; visit tickets"),
    "bo.current_config_vs_incumbent": ("Proposed research assumption", "M6 engine semantics",
        "current configuration (search center) distinct from global incumbent"),
    "bo.acceptance_policy": ("Proposed research assumption", "M6 acceptance",
        "improvement-only current-configuration acceptance"),
    "bo.evaluations_per_block_visit": ("Experimental setting", "M6 engine",
        "1 evaluation per block visit (default)"),
    "bo.full_gp_dimension_vs_block_dimension": ("Proposed research assumption", "M6 surrogate",
        "ONE shared full-dimension GP for all blocks"),
    "bo.benchmark_stagnation_lp.reference": ("Derived quantity", "M6 references",
        "analytic LP optimum f* = 9.6 at (1.6, 0, 1.6, 0)"),
    "bo.highdim_benchmarks.origin": ("Proposed research assumption", "M6 benchmarks 4/8/16-D",
        "self-defined separable/coupled/constrained-mixed/stagnation families"),
    "experiment.m6_block_architecture_budget": ("Experimental setting", "M6 experiment",
        "budget 100 total, initial 30, pool 500, seeds 0-9"),
    "optimization.num_initial_observations": ("Experimental setting", "BO initial design size",
        "30 (inherited M5/M6; not proven optimal)"),
    "optimization.evaluation_budget": ("Experimental setting", "BO budget",
        "UNRESOLVED by design: each experiment sets it explicitly (no paper default)"),
    "environment.num_targets": ("Experimental setting", "synthetic scenario",
        "number of synthetic targets"),
    "environment.random_seed": ("Experimental setting", "reproducibility",
        "scenario RNG seed"),
    "experiment.repetitions": ("Experimental setting", "experiment design",
        "seeds 0-9 replicates"),
    "resource.time_budget_semantics": ("Proposed research assumption", "budget semantics",
        "'discounted_cumulative' (paper) vs 'hard_per_slot' (proposed, stricter) — NOT equivalent"),
    "experiment.m7_runner_architecture": ("Proposed research assumption", "M7 experiment runner",
        "atomic per-run records, statuses, resume compatibility"),
    "experiment.m7_shared_initialization_policy": ("Proposed research assumption", "comparison fairness",
        "same seed + stored-record verification; M8A adds validated explicit shared designs"),
    "experiment.m7_statistical_conventions": ("Proposed research assumption", "statistical analysis",
        "paired stats, bootstrap (B=10000, seed 2026), Wilcoxon+Holm, tie tol 1e-9"),
    "experiment.m7_threshold_definitions": ("Proposed research assumption", "threshold metrics",
        "gap-closure fractions frozen before comparative experiments"),
    "experiment.m7_runtime_accounting": ("Proposed research assumption", "runtime decomposition",
        "disjoint stage timers + explicit unattributed overhead (M8A: historical nested convention documented)"),
    "experiment.m7_candidate_profiling_baseline": ("Proposed research assumption", "candidate generation",
        "M6 algorithm preserved as versioned baseline"),
    "experiment.m7_validation_matrix": ("Experimental setting", "M7 validation",
        "2 benchmarks x 5 methods x 3 seeds, budget 40"),
    "experiment.m7_milestone_reconciliation": ("Proposed research assumption", "documentation",
        "8-milestone roadmap vs 9-milestone instruction stream"),
    "experiment.m8a_run_count_reconciliation": ("Derived quantity", "M8A documentation",
        "220 archived full-matrix records (the '230' figure was a narrative arithmetic error, corrected)"),
    "experiment.m8a_shared_initial_design_interface": ("Proposed research assumption", "engine + framework",
        "validated immutable initial_design_configs + framework explicit_shared mode (backward compatible)"),
}


# Item-6 resolution-path analysis for the 17 UNRESOLVED entries.
RESOLUTION_PATH: dict[str, dict] = {
    "tracking.reference_snr": {
        "question": "SNR0 (reference linear SNR), Eq. (5)",
        "universal_value": "No — platform/radar/waveform specific.",
        "hardware_dependent": "Yes (emitter and receiver chain).",
        "paper_sufficient": "No — the paper uses SNR0 but reports no value (verified during M2/M3 extraction for the ICC paper; re-verified 2026-11-26 against the authors' extended journal version arXiv 2606.05526, whose Table III also omits SNR0).",
        "derivable": "Not from other paper quantities; SNR0 anchors the absolute noise scale.",
        "additional_reference": "Yes — either the paper's own experimental setup section (re-verify from source PDF) or a calibrated scenario definition.",
        "sensitivity": "Yes — SNR0 linearly scales all measurement variances (inverse-SNR), directly affecting tracking cost magnitude; a log-spaced sweep is appropriate once resolved.",
        "needed_for": "BASELINE REPRODUCTION of the paper's tracking-noise scale (the operational synthetic path can substitute an explicitly synthetic SNR0, clearly labeled).",
    },
    "scanning.peak_transmit_power": {
        "question": "P_t (peak transmit power), Eqs. (16)-(20)",
        "universal_value": "No.",
        "hardware_dependent": "Yes — transmitter specification.",
        "paper_sufficient": "No — Table I lists the symbol without a value (verified during M2 extraction).",
        "derivable": "No.",
        "additional_reference": "Yes — radar hardware datasheet or scenario specification.",
        "sensitivity": "Yes — enters the radar range equation linearly (in power terms); however the ENTIRE physical chain Eqs. (16)-(20) is NOT implemented, so sensitivity analysis is deferred until activation is approved.",
        "needed_for": "FUTURE EXTENSION ONLY (physical scanning model).",
    },
    "scanning.transmit_antenna_gain": {"question": "G_t — hardware-dependent; not in paper (Table I symbol only); not derivable; datasheet required; deferred with the unimplemented Eqs. (16)-(20) chain.", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.receive_antenna_gain": {"question": "G_r — hardware-dependent; not in paper; datasheet required; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.wavelength": {"question": "lambda — hardware/frequency-plan dependent; not in paper; derivable from carrier frequency if a frequency plan is specified; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.radar_cross_section": {"question": "sigma_RCS — scenario/target dependent (aspect, frequency); not in paper; not derivable; target-definition document required; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.loss_factor": {"question": "L — hardware dependent (>= 1); not in paper; link-budget analysis required; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.system_temperature": {"question": "T_sys — hardware dependent; not in paper; receiver specification required; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.beam_angular_spacing": {"question": "beam angular spacing — scan-plan/scenario dependent; not in paper; scan-design document required; deferred with Eqs. (16)-(20).", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.reference_area": {"question": "S_0 (Eq. (21) normalization) — scenario definition dependent; not in paper (symbol reported, value absent); NOT derivable without a scenario; scenario definition required. Sensitivity: YES and cheap — Gamma is linear in S_det for fixed S_0, so S_0 rescales the scanning term; a normalization study is part of the M8A objective-scaling plan.", "needed_for": "PHYSICAL SCANNING METRIC (the synthetic provider does not use S_0)."},
    "scanning.detection_probability": {"question": "P_d — detector design parameter; no value in EITHER paper version; the authors' extended journal version (arXiv 2606.05526, Sec. III-B) obtains P_d from SNR via Shnidman's equation (Shnidman 2002, IEEE TAES 38(3)) using a precomputed SNR->P_d lookup table (0.01-0.99, step 0.01). Adoption of that detection chain would be a NEW implementation decision requiring separate review; detector model remains NOT implemented.", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.false_alarm_probability": {"question": "P_fa — detector design parameter; no value in the ICC paper. CANDIDATE VALUE from the authors' extended journal version (arXiv 2606.05526, Sec. III-D): P_f = 1e-3 (used in ITS simulations; false-alarm measurements assumed uniformly distributed over the area of interest). NOT adopted here; any adoption requires separate review + approval. Detector model remains NOT implemented.", "needed_for": "FUTURE EXTENSION ONLY."},
    "scanning.required_detection_snr": {"question": "SNR_min — detector/waveform dependent; not in paper; detection-curve reference required; deferred (detector model NOT implemented).", "needed_for": "FUTURE EXTENSION ONLY."},
    "tracking.measurement_driven_initial_velocity": {
        "question": "velocity prior mean for measurement-driven initialization",
        "universal_value": "No.",
        "hardware_dependent": "No — scenario dependent.",
        "paper_sufficient": "No — the paper's zero-vector initialization is retained as the diagnostic configuration only (M3 decision); no operational prior is given.",
        "derivable": "Partially — could be estimated from the first two observations in a fully measurement-driven design; that variant is a PROPOSED future change requiring approval.",
        "additional_reference": "No external reference required; a per-scenario value (clearly labeled synthetic) is acceptable.",
        "sensitivity": "Yes — initialization bias affects transient tracking cost; recommend a small initialization-sensitivity study before any paper-style experiment.",
        "needed_for": "OPERATIONAL SYNTHETIC ESTIMATION (currently supplied per-scenario with explicit synthetic labels; unresolved in the register by design).",
    },
    "tracking.measurement_driven_initial_velocity_variance": {
        "question": "velocity prior variance per axis (measurement-driven initialization)",
        "universal_value": "No.",
        "hardware_dependent": "No — scenario/estimator-tuning dependent.",
        "paper_sufficient": "No value in the ICC paper (zero-vector init only, retained as diagnostic). CANDIDATE VALUE from the authors' extended journal version (arXiv 2606.05526, Eq. (14)): P_0 = diag([10000, 10000, 100, 100]) — position variance 1e4 m^2, velocity variance 1e2 m^2/s^2 per axis, described by the authors as 'initialized empirically'. NOT adopted; adoption requires separate review + approval.",
        "derivable": "Partially — could be tied to the first-observation covariance ratio (proposed; requires approval).",
        "additional_reference": "No; per-scenario with sensitivity study.",
        "sensitivity": "Yes — controls estimator transient; include with the initialization study.",
        "needed_for": "OPERATIONAL SYNTHETIC ESTIMATION.",
    },
    "resource.task_utility_weights": {
        "question": "w_n per-task weights of the Eq. (23) tracking aggregate",
        "universal_value": "No.",
        "hardware_dependent": "No — research-preference/scenario dependent.",
        "paper_sufficient": "No — the paper reports the weights symbolically; per-task values were not preserved in our extractions (verified M2).",
        "derivable": "No.",
        "additional_reference": "Yes — re-verify from the source PDF (possibly a table exists) or define a research scenario (e.g. uniform) with explicit labeling.",
        "sensitivity": "Yes — weights change the relative importance of tasks in the utility; a weight-sensitivity design is part of the M8A normalization plan.",
        "needed_for": "BASELINE REPRODUCTION IF the paper's values exist; uniform default is a PROPOSED software default for synthetic work.",
    },
    "optimization.evaluation_budget": {
        "question": "total attempted objective evaluations per BO run",
        "universal_value": "No — experiment-design choice.",
        "hardware_dependent": "Only via compute budget (wall-clock), not physically.",
        "paper_sufficient": "N/A — Benzaghta's settings were adopted for the synthetic studies (100/30/500); no application budget exists yet.",
        "derivable": "No.",
        "additional_reference": "No — must be set explicitly per experiment by design (UnresolvedParameterError if absent).",
        "sensitivity": "Yes — the M7 sensitivity infrastructure supports budget scenarios; a budget-sensitivity study is planned before any application claim.",
        "needed_for": "EVERY EXPERIMENT (explicit by design).",
    },
}


def render() -> str:
    reg = json.loads(Path("provenance_register.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    lines: list[str] = []
    lines.append("# Parameter Resolution Register (M8A)")
    lines.append("")
    lines.append(f"_Generated from `provenance_register.json` at commit `{commit}` "
                 f"({len(reg['entries'])} entries; {sum(1 for e in reg['entries'] if e['value'] is None)} unresolved). "
                 "This register is the single source of truth for parameter provenance; the JSON "
                 "register is authoritative and this document is its review companion. "
                 "No missing value has been replaced by an arbitrary constant; illustrative "
                 "synthetic values in code are always labeled synthetic._")
    lines.append("")
    lines.append("## 1. Classification taxonomy (M8A item 5)")
    lines.append("")
    lines.append("| Class | Meaning |")
    lines.append("|---|---|")
    lines.append("| Paper-reported parameter | Value printed in a primary reference (table or prose); reproducible from the source. |")
    lines.append("| Missing paper parameter | Symbol REQUIRED by a paper equation but never assigned a value in the paper. |")
    lines.append("| Universal physical constant | Exact SI-defined constant. |")
    lines.append("| Hardware-dependent parameter | Determined by radar/transmitter/receiver hardware specification. |")
    lines.append("| Scenario-dependent parameter | Determined by the research scenario definition (targets, geometry, scan plan). |")
    lines.append("| Proposed research assumption | Our own methodological/design decision — never attributed to a paper. |")
    lines.append("| Experimental setting | Explicit experiment-design value (inherited settings are labeled as such, not proven optimal). |")
    lines.append("| Derived quantity | Computed from other independently known quantities (e.g. analytic references). |")
    lines.append("")
    lines.append("## 2. Full inventory (all entries)")
    lines.append("")
    lines.append("Columns: entry (dotted config path), symbol/meaning, unit, source class, "
                 "current value, status, affected model components, M8A classification.")
    lines.append("")
    lines.append("| Entry | Symbol / meaning | Unit | Source | Current value | Status | Affected components | M8A classification |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in reg["entries"]:
        name = e["name"]
        classification, components, meaning = CLASSIFICATION.get(
            name, ("Proposed research assumption", "(see entry rationale)", "(see entry rationale)"))
        value = e["value"]
        if isinstance(value, dict):
            value_str = "`" + json.dumps(value) + "`"
        elif isinstance(value, list):
            value_str = "`" + json.dumps(value) + "`"
        else:
            value_str = "`" + str(value) + "`"
        status = "RESOLVED" if value is not None else "UNRESOLVED (None)"
        unit = e.get("unit") or "—"
        source = e["source"]
        meaning_esc = meaning.replace("|", "/")
        comp_esc = components.replace("|", "/")
        lines.append(f"| `{name}` | {meaning_esc} | {unit} | {source} | {value_str} | {status} | {comp_esc} | {classification} |")
    lines.append("")
    lines.append("## 3. Unresolved parameters — resolution-path analysis (M8A item 6)")
    lines.append("")
    unresolved = [e for e in reg["entries"] if e["value"] is None]
    for e in unresolved:
        name = e["name"]
        path = RESOLUTION_PATH.get(name, {})
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- **Meaning/unit:** {CLASSIFICATION[name][2]} [{e.get('unit') or 'dimensionless'}]")
        lines.append(f"- **Paper reference:** {e.get('reference') or 'none'}")
        lines.append(f"- **Universal value exists:** {path.get('universal_value', 'see below')}")
        lines.append(f"- **Hardware dependent:** {path.get('hardware_dependent', 'see below')}")
        lines.append(f"- **Paper provides sufficient information:** {path.get('paper_sufficient', 'No (see below)')}")
        lines.append(f"- **Derivable from known quantities:** {path.get('derivable', 'No')}")
        lines.append(f"- **Additional reference / calibration required:** {path.get('additional_reference', 'see below')}")
        lines.append(f"- **Sensitivity analysis appropriate:** {path.get('sensitivity', 'Yes — after resolution')}")
        lines.append(f"- **Needed for:** {path.get('needed_for', 'see below')}")
        lines.append("")
    lines.append("## 4. Baseline reproduction vs future extensions (M8A item 5)")
    lines.append("")
    lines.append("**Required for baseline reproduction of Lu et al.:** the Table II parameters "
                 "(all RESOLVED paper-reported) plus `tracking.reference_snr` (MISSING paper "
                 "parameter — the only gap in the tracking chain) and, if the paper's per-task "
                 "weights exist, `resource.task_utility_weights` (re-verification from the source "
                 "PDF required). The physical scanning chain (Eqs. (16)-(20)) is NOT required for "
                 "the tracking/utility baseline because it is not implemented; `scanning.reference_area` "
                 "is required only when a physical Eq. (21) normalization is activated.")
    lines.append("")
    lines.append("**Required only for future extensions:** the eleven hardware/scenario parameters "
                 "of the unimplemented radar-range/detector chain (`scanning.*` except the Boltzmann "
                 "constant and the metric definition), plus any proposed corridor-specific scanning "
                 "metric parameters (none defined yet).")
    lines.append("")
    lines.append("## 5. Extended-version cross-reference (added 2026-11-26)")
    lines.append("")
    lines.append("The authors of [Lu2024] have an extended journal version on arXiv "
                 "(arXiv 2606.05526, 'Constrained Deep Reinforcement Learning for Cognitive Radar "
                 "Resource Management', submitted 2026-06-04; fetched 2026-11-26). It confirms the "
                 "ICC paper's model equations (motion (2)-(3), measurement (5)-(8), EKF (9)-(13), "
                 "utility structure) and adds track-initiation content. It differs from the ICC "
                 "paper in ways that matter for parameter provenance:")
    lines.append("")
    lines.append("| Quantity | ICC 2024 (reproduced here) | Extended arXiv version |")
    lines.append("|---|---|---|")
    lines.append("| sigma^2_{r,0} | 100 m^2 | 16 m^2 |")
    lines.append("| sigma^2_{theta,0} | 4e-4 rad^2 | 1e-6 rad^2 |")
    lines.append("| sigma^2_w (maneuverability) | 25 (m/s^2)^2 | 16 m^2/s^4 |")
    lines.append("| beta | 1e6 | 3e4 |")
    lines.append("| Theta_max | 1 | 0.9 |")
    lines.append("| Scanning metric | Gamma = S_det/S_0 (Eq. (21)); U = -Sum w c + beta*Gamma | N_miss (undetected-target count); U = -Sum w c - beta*N_miss (Eq. (19)) |")
    lines.append("| Scanning physics | Eqs. (16)-(20) symbols, no values | Eq. (18) radar range equation, same symbols, no values; Shnidman SNR->P_d lookup |")
    lines.append("| EKF initialization | zero vector | measurement-driven, empirical P_0 = diag([1e4, 1e4, 1e2, 1e2]) (Eq. (14)) |")
    lines.append("| SNR0 | absent | absent (Table III omits it) |")
    lines.append("")
    lines.append("Implications: (a) 'paper-reported' parameter values are VERSION-DEPENDENT — "
                 "this repository reproduces the ICC 2024 values and does not switch; (b) SNR0 "
                 "remains missing in both versions, so `tracking.reference_snr` stays unresolved; "
                 "(c) the extended version's P_f = 1e-3 and P_0 diag values are recorded as "
                 "CANDIDATE values in the resolution paths above, requiring separate review and "
                 "approval before any adoption; (d) the extended version's N_miss metric is a "
                 "DIFFERENT scanning metric from the ICC Gamma — our Eq. (21)/(23) reproduction is "
                 "ICC-faithful, and the difference is documented, not silently reconciled.")
    lines.append("")
    lines.append("## 6. Parameters requiring USER APPROVAL before resolution (M8A deliverable)")
    lines.append("")
    lines.append("Resolving any of these requires an explicit human decision (value + source + "
                 "approval recorded in the register). None is resolved by this milestone:")
    lines.append("")
    lines.append("1. `tracking.reference_snr` — supply SNR0 (paper re-verification or an explicitly synthetic scenario value).")
    lines.append("2. `resource.task_utility_weights` — confirm whether the paper reports per-task values; otherwise approve the uniform default as the labeled research assumption.")
    lines.append("3. `tracking.measurement_driven_initial_velocity` + `..._variance` — approve per-scenario synthetic values (or the proposed first-two-observations estimator) for operational synthetic runs.")
    lines.append("4. The eleven `scanning.*` hardware/scenario parameters — required only if the physical Eqs. (16)-(20) chain is ever approved for implementation (currently out of scope).")
    lines.append("5. `optimization.evaluation_budget` — set per experiment (by design unresolved in the register).")
    lines.append("")
    lines.append("## 7. Policy reminders (unchanged by M8A)")
    lines.append("")
    lines.append("- Missing paper values stay `None` in the register; code paths that need them raise `UnresolvedParameterError`.")
    lines.append("- Illustrative/synthetic values in code are always labeled synthetic and never enter the register as paper values.")
    lines.append("- Unvalidated physical configurations are not activated.")
    lines.append("- Any proposed numerical selection requires separate review and approval (M8A item 6).")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    out = Path("docs/PARAMETER_RESOLUTION_REGISTER.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
