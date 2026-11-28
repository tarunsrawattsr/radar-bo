"""Parameter provenance register.

One authoritative record of every parameter: name, value, units, source,
rationale, sensitivity range and notes (README section 4.6). Sources are
strictly separated:

* ``paper_lu_2024``      — values reported in Lu et al., ICC 2024.
* ``paper_benzaghta_2023`` — methodological settings in Benzaghta et al., GLOBECOM 2023.
* ``standard_constant``  — physics constants with exact SI definitions.
* ``proposed_assumption`` — our documented proposals (incl. ambiguity defaults).
* ``experimental_setting`` — experimental settings with no paper basis.

Unresolved parameters keep ``value=None`` and are preserved, never invented;
experiments call :meth:`ProvenanceRegister.require_resolved` and fail clearly
when they need an unresolved value.

Regenerate the JSON snapshot at the repository root::

    python -m radar_bo.config.provenance [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from radar_bo.config.base import StrictModel, UnresolvedParameterError
from radar_bo.config.simulation_config import SimulationConfig

ProvenanceSource = Literal[
    "paper_lu_2024",
    "paper_benzaghta_2023",
    "standard_constant",
    "proposed_assumption",
    "experimental_setting",
]

ProvenanceValue = float | int | str | list[float] | dict[str, float] | None


class ProvenanceEntry(StrictModel):
    """One parameter: value, units, source, rationale, sensitivity range, notes."""

    name: str = Field(description="Dotted config path, e.g. 'tracking.reference_range_variance'.")
    value: ProvenanceValue = Field(default=None, description="Parameter value; None while unresolved.")
    unit: str | None = Field(default=None, description="SI unit string, e.g. 'm^2', 's', 'rad^2', 'W'.")
    source: ProvenanceSource = Field(description="Strictly separated source category.")
    reference: str | None = Field(default=None, description="Paper location, e.g. 'Lu et al. Table II, p. 2789'.")
    rationale: str | None = Field(default=None, description="Why this value / why unresolved.")
    sensitivity_range: tuple[float, float] | None = Field(
        default=None, description="PROPOSED sweep range for sensitivity studies, when meaningful."
    )
    notes: list[str] = Field(default_factory=list, description="Deviations, ambiguities, caveats.")

    @property
    def resolved(self) -> bool:
        return self.value is not None


class ProvenanceRegister(StrictModel):
    """Ordered, JSON-serializable register of :class:`ProvenanceEntry` items."""

    entries: list[ProvenanceEntry] = Field(default_factory=list)

    def register(self, entry: ProvenanceEntry) -> None:
        if any(e.name == entry.name for e in self.entries):
            raise ValueError(f"Duplicate provenance entry: {entry.name!r}.")
        self.entries.append(entry)

    def get(self, name: str) -> ProvenanceEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(f"No provenance entry named {name!r}. Registered names: {[e.name for e in self.entries]}")

    def require_resolved(self, *names: str) -> dict[str, ProvenanceValue]:
        """Return ``{name: value}`` or raise :class:`UnresolvedParameterError`.

        The error lists every unresolved name with its unit and the register's
        rationale, so a missing paper parameter fails clearly instead of being
        silently fabricated.
        """
        missing: dict[str, str] = {}
        rationale: list[str] = []
        for name in names:
            entry = self.get(name)  # KeyError for unknown names: no silent pass
            if not entry.resolved:
                missing[name] = entry.unit or "unspecified"
                if entry.rationale:
                    rationale.append(f"  - {name}: {entry.rationale}")
        if missing:
            guidance = "See the provenance register (provenance_register.json)." + (
                "\nWhy unresolved:\n" + "\n".join(rationale) if rationale else ""
            )
            raise UnresolvedParameterError(missing, guidance=guidance)
        return {name: self.get(name).value for name in names}

    def unresolved_names(self) -> list[str]:
        return [e.name for e in self.entries if not e.resolved]

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def _entry(
    name: str,
    value: ProvenanceValue,
    unit: str | None,
    source: ProvenanceSource,
    reference: str | None = None,
    rationale: str | None = None,
    sensitivity: tuple[float, float] | None = None,
    notes: list[str] | None = None,
) -> ProvenanceEntry:
    return ProvenanceEntry(
        name=name,
        value=value,
        unit=unit,
        source=source,
        reference=reference,
        rationale=rationale,
        sensitivity_range=sensitivity,
        notes=notes or [],
    )


_LU = "Z. Lu et al., IEEE ICC 2024, DOI 10.1109/ICC51166.2024.10622225"
_BENZ = "M. Benzaghta et al., IEEE GLOBECOM 2023, DOI 10.1109/GLOBECOM54140.2023.10437768"


def build_default_provenance(config: SimulationConfig | None = None) -> ProvenanceRegister:
    """Build the default register; values are read from ``config`` (defaults if None).

    Sensitivity ranges are PROPOSED (documented experimental choices), never
    paper-reported: the papers provide no sensitivity analyses.
    """
    c = config if config is not None else SimulationConfig()
    t, s, r, o, e = c.tracking, c.scanning, c.resource, c.optimization, c.environment
    reg = ProvenanceRegister()

    # ---- Lu et al. Table II (p. 2789), paper-reported ------------------------
    reg.register(_entry(
        "tracking.reference_range_variance", t.reference_range_variance, "m^2", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789",
        rationale="Variance of range measurement at reference SNR (Eq. (6) mapping).",
        sensitivity=(10.0, 1000.0), notes=["Already a variance; never squared again."]))
    reg.register(_entry(
        "tracking.reference_angle_variance", t.reference_angle_variance, "rad^2", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789",
        rationale="Variance of azimuth measurement at reference SNR (Eq. (6) mapping); radians.",
        sensitivity=(1e-5, 1e-2)))
    reg.register(_entry(
        "tracking.maneuverability_noise", t.maneuverability_noise, "(m/s^2)^2", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789; Eq. (3)",
        rationale="TERMINOLOGY (M4 review): the interpretations are distinguished by the "
                  "process-noise covariance MULTIPLIER. 'table_variance' -> multiplier 25 "
                  "(sigma_w = 25 read as the acceleration-noise variance per Table II units "
                  "and prose); 'equation_literal' -> multiplier 625 (= 25 squared, Eq. (3) as "
                  "printed). The multiplier is applied to the structured Eq. (3) matrix "
                  "directly; no further squaring occurs. M8A-R1 (source-PDF verification): the "
                  "ICC Table II row reads literally 'sigma_w ((m/s^2)^2) 25' — the paper's own "
                  "unit annotation declares the tabulated quantity a VARIANCE (units "
                  "(m/s^2)^2), which supports the table_variance reading; the accepted "
                  "resolution and both implementations are PRESERVED unchanged.",
        sensitivity=(5.0, 625.0),
        notes=["Switchable: tracking.maneuverability_noise_interpretation in "
               "{'table_variance', 'equation_literal'}; both implemented, default documented, "
               "sensitivity to be tested (README section 4.1).",
               "M3 review decision: 'table_variance' accepted as the default based on Table II "
               "units and prose; the default is NOT an author-confirmed correction of the paper."]))
    reg.register(_entry(
        "tracking.reference_distance", t.reference_distance, "m", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789; Eq. (5)", rationale="Reference range r0 in the SNR model.",
        sensitivity=(1000.0, 10000.0)))
    reg.register(_entry(
        "tracking.reference_dwell_time", t.reference_dwell_time, "s", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789; Eq. (5)", rationale="Reference dwell time tau0 in the SNR model.",
        sensitivity=(0.1, 1.0)))
    reg.register(_entry(
        "tracking.revisit_interval", t.revisit_interval, "s", "paper_lu_2024",
        reference=f"{_LU}; Table II, p. 2789",
        rationale="Revisit interval T0; single authoritative source is TrackingConfig (README 4.6).",
        sensitivity=(0.5, 5.0)))
    reg.register(_entry(
        "tracking.reference_snr", t.reference_snr, "linear SNR", "paper_lu_2024",
        reference=f"{_LU}; Eq. (5)",
        rationale="UNRESOLVED: SNR0 is required by the SNR model Eq. (5) but is not reported "
                  "anywhere in the paper. Preserved as None; never invented.",
        notes=["Experiments requiring Eq. (5) with absolute SNRs must resolve this first."]))
    reg.register(_entry(
        "resource.dwell_time_action_set", r.dwell_time_action_set, "s", "paper_lu_2024",
        reference=f"{_LU}; Section V-A.2 'Action' (M8A-R1: verified against the source PDF)",
        rationale="Dwell-time actions tau^n_t in {0.1, 0.2, ..., 1.0} s (ten levels, 0.1 s steps, "
                  "range [0.1, 1] quantized), as stated in the ICC paper's action definition. "
                  "M8A-R1 correction: an earlier M8A document wrongly attributed a "
                  "'{0, 0.5, ..., 25.0}' set to Table II; the register value was always "
                  "{0.1..1.0} and the DOCUMENTS were corrected, not the register."))
    reg.register(_entry(
        "resource.normalized_time_budget_threshold", r.normalized_time_budget_threshold, "dimensionless",
        "paper_lu_2024", reference=f"{_LU}; Eq. (22)",
        rationale="Theta_max = 1, normalized time-budget threshold."))
    reg.register(_entry(
        "resource.tracking_scanning_tradeoff", r.tracking_scanning_tradeoff,
        "(m^2 * weight) per unit Gamma", "paper_lu_2024",
        reference=f"{_LU}; Eq. (23); M8A-R1: dimensional interpretation verified against Eq. (21) and the Table II value",
        rationale="beta = 1e6 weights scanning reward against tracking cost. Reproduction only; "
                  "recalibrate if metric definitions change.",
        notes=[
            "M8A-R1 dimensional correction: Gamma = S_max/S_0 (Eq. (21)) is DIMENSIONLESS "
            "(m^2/m^2), so the scanning term beta*Gamma carries whatever dimension beta has. "
            "For the Eq. (23) sum to be coherent, beta must carry the tracking-term dimension "
            "(weight x m^2) per unit Gamma — beta is NOT dimensionless. Its magnitude is "
            "scale-relative: with tracking costs O(10-100) m^2 near r_0, beta=1e6 makes the "
            "scanning term dominate by ~4 orders of magnitude at Gamma=O(1) (consistent with "
            "the paper's Fig. 5 utility axis being scaled by 1e6). Any change of cost scale, "
            "metric, or Theta_max changes the effective trade-off; beta=1e6 must not be "
            "carried into new configurations as a default.",
            "Units of the two weighted terms are not matched by construction in the paper."]))
    reg.register(_entry(
        "resource.discount_factor", r.discount_factor, "dimensionless", "paper_lu_2024",
        reference=f"{_LU}; Eq. (22)", rationale="gamma = 0.9 in the discounted cumulative budget.",
        sensitivity=(0.5, 0.99)))

    # ---- Lu et al. scanning model: unspecified (Table I symbols only) --------
    _scan_ref = f"{_LU}; Eqs. (16)-(21), Table I (symbols only, no values)"
    _scan_note = "Unspecified in the paper; preserved as None, never invented (README non-negotiables)."
    for name, unit, extra in [
        ("peak_transmit_power", "W (peak pulse power)", None),
        ("transmit_antenna_gain", "linear (dimensionless)", None),
        ("receive_antenna_gain", "linear (dimensionless)", None),
        ("wavelength", "m", None),
        ("radar_cross_section", "m^2", None),
        ("loss_factor", "linear, >= 1", None),
        ("system_temperature", "K", None),
        ("beam_angular_spacing", "deg",
         "phi in Eq. (16) is called 'phase delay between adjacent beams'; interpreted as angular "
         "spacing in degrees (number of beams = 360/phi). Interpretational assumption, flagged."),
        ("reference_area", "m^2", "S0 in Eq. (21); must be nonzero."),
        ("detection_probability", "probability", "Pd, Shnidman-equation input of Eq. (21)."),
        ("false_alarm_probability", "probability", "Pf, Shnidman-equation input of Eq. (21)."),
        ("required_detection_snr", "linear SNR",
         "Required detection SNR; the paper cites Shnidman's equation without all implementation "
         "settings (pulse count, integration mode)."),
    ]:
        notes = [_scan_note] + ([extra] if extra else [])
        reg.register(_entry(f"scanning.{name}", getattr(s, name), unit, "paper_lu_2024",
                            reference=_scan_ref, rationale="Unspecified by the paper.", notes=notes))

    # ---- Standard constants ---------------------------------------------------
    reg.register(_entry(
        "scanning.boltzmann_constant", s.boltzmann_constant, "J/K", "standard_constant",
        reference="SI 2019 exact definition",
        rationale="Boltzmann constant k = 1.380649e-23 J/K; a standard physical constant, not a fitted parameter."))

    # ---- Initialization: paper-reported + flagged degenerate geometry --------
    reg.register(_entry(
        "tracking.initialization_policy", t.initialization_policy, None, "paper_lu_2024",
        reference=f"{_LU}; p. 2786 (x0 = 0 vector, P0 = I_4)",
        rationale="DEGENERATE GEOMETRY: the paper initializes the estimate at the origin, where the "
                  "range/azimuth measurement is singular (atan2(0, 0) undefined; measurement Jacobian "
                  "degenerate). 'paper_zero_identity' reproduces the paper and is retained as a "
                  "documented DIAGNOSTIC mode; it must never silently produce NaN/inf or an "
                  "artificial Jacobian (an update at the origin raises MeasurementGeometryError).",
        notes=["M3 review (instruction 3.3): the OPERATIONAL estimator uses the "
               "'measurement_driven' policy, initializing from the first supplied observation "
               "plus explicitly configured velocity prior assumptions; the true state is never "
               "accessible to initialization.",
               "No numerical initial-covariance values are attributed to the paper beyond x0 = 0, "
               "P0 = I_4."]))
    reg.register(_entry(
        "tracking.measurement_driven_initial_velocity", t.measurement_driven_initial_velocity,
        "(m/s, m/s)", "proposed_assumption",
        reference="M3 review instruction 9 (no paper value)",
        rationale="Velocity prior MEAN for measurement-driven initialization. Unresolved by "
                  "default (None): initialization raises an informative error rather than "
                  "inventing a default. Synthetic experiments supply values labeled as "
                  "synthetic settings."))
    reg.register(_entry(
        "tracking.measurement_driven_initial_velocity_variance", t.measurement_driven_initial_velocity_variance,
        "(m/s)^2", "proposed_assumption",
        reference="M3 review instruction 9 (no paper value)",
        rationale="Velocity prior VARIANCE per axis (isotropic) for measurement-driven "
                  "initialization; kept conceptually separate from the position uncertainty, "
                  "which is derived from the first observation's covariance. Unresolved by "
                  "default (None)."))
    reg.register(_entry(
        "tracking.covariance_update_method", t.covariance_update_method, None, "proposed_assumption",
        reference=f"{_LU}; Eq. (13)",
        rationale="NUMERICAL STABILITY option: default 'paper_equation_13' applies Eq. (13) as "
                  "printed, P_{t|t} = (I - K_t H_t) P_{t|t-1}. 'joseph' uses the "
                  "form-preserving PSD update P = (I-KH) P (I-KH)^T + K R K^T, mathematically "
                  "equivalent for the exact Kalman gain; equivalence is tested under valid "
                  "numerical conditions.",
        notes=["Posterior covariances are symmetrized against floating-point drift; covariance "
               "eigenvalues are never clipped to conceal defects."]))

    # ---- Eq. (6): consolidated semantics (M3 review, instruction 3.2) --------
    reg.register(_entry(
        "tracking.measurement_variance_semantics",
        "sigma^2_{r|theta,t} = sigma^2_{r|theta,0} / SNR_t (reference variance divided by linear SNR)",
        None, "paper_lu_2024",
        reference=f"{_LU}; Eq. (6) and surrounding prose, p. 2787",
        rationale="Owner-reviewed reading of Eq. (6). Evidence: (a) the paper prose directly below "
                  "Eq. (6) states the variance DECREASES with longer dwell time or closer target "
                  "'according to equations (5) and (6)'; (b) SNR grows with dwell time and "
                  "proximity per Eq. (5); (c) standard radar behaviour. Only the inverse reading "
                  "satisfies all three.",
        notes=["EXTRACTION HISTORY (settled): an early plain-text PDF extraction juxtaposed "
               "sigma^2_{.,0} and SNR_t on one line, producing a transient multiplication "
               "reading that contradicted the same paper's prose. Re-inspection with "
               "layout-aware extraction and the M3/M4 reviews identified this as a PDF-"
               "EXTRACTION ARTIFACT, not an unresolved mathematical ambiguity; the division "
               "semantics is the accepted paper reading and inverse-SNR scaling is NOT "
               "classified as a proposed deviation.",
               "Implemented in radar_bo/radar/measurement_noise.py; requires the unresolved SNR0 "
               "(see tracking.reference_snr)."]))

    # ---- Benzaghta et al.: methodological settings ---------------------------
    reg.register(_entry(
        "optimization.kernel", o.kernel, None, "paper_benzaghta_2023",
        reference=_BENZ,
        rationale="Matern-5/2 kernel; our harness is a methodological adaptation, not an exact "
                  "reproduction of that paper's algorithm (README research foundations)."))
    reg.register(_entry(
        "optimization.acquisition_xi", 0.01, "dimensionless", "paper_benzaghta_2023",
        reference=_BENZ,
        rationale="Expected-improvement exploration parameter xi = 0.01 reported by Benzaghta et al. "
                  "NOT carried as a configuration field: the installed BoTorch "
                  "LogExpectedImprovement (verified at runtime) accepts NO exploration-offset "
                  "parameter, so the M5 engine uses conventional LogEI without xi. This entry "
                  "preserves the paper value in the provenance record only (methodological "
                  "adaptation); no no-effect configuration parameter is retained.",
        sensitivity=(0.0, 0.1)))
    reg.register(_entry(
        "optimization.num_acquisition_candidates", o.num_acquisition_candidates, "count",
        "paper_benzaghta_2023", reference=_BENZ,
        rationale="Acquisition candidate pool Nc = 500 per iteration."))
    reg.register(_entry(
        "optimization.search_mode", o.search_mode, None, "proposed_assumption",
        reference=_BENZ,
        rationale="'joint' is the default harness mode. Benzaghta et al. update one BS at a time "
                  "(block-wise); 'alternating' mirrors that scheme and 'coordinate' is experimental.",
        notes=["Adaptation, not reproduction: the paper's block-wise scheme is per-BS in their "
               "problem, not a generic harness mode."]))

    # ---- M4: scanning metric, remaining time, aggregation design -------------
    reg.register(_entry(
        "scanning.area_ratio_metric", "Gamma = S_max / S_0 (uncapped)", None, "paper_lu_2024",
        reference=f"{_LU}; Eq. (21)",
        rationale="Eq. (21) implemented as a pure metric accepting a modeled detectable area and a "
                  "strictly positive reference area; the ratio is NOT capped at one. Physical "
                  "scanning calculations (Eqs. (16)-(20)) are NOT implemented: their parameters are "
                  "unspecified (Table I).",
        notes=["The paper's S0 remains unresolved (scanning.reference_area); the metric raises a "
               "parameter-resolution error rather than substituting a default."]))
    reg.register(_entry(
        "resource.remaining_scanning_time_convention",
        "tau_s = T0 * max(0, 1 - sum(tau)/T0)", "s", "paper_lu_2024",
        reference=f"{_LU}; Eq. (24)",
        rationale="Paper's own nonnegative remaining-time convention implemented verbatim in the "
                  "resource accountant. The RAW time deficit is recorded alongside so an over-budget "
                  "allocation is never silently made valid by the clipping; feasibility is judged "
                  "separately by the constraint manager.",
        notes=["M4: an allocation violating the proposed hard per-slot budget is reported infeasible; "
               "the Eq. (24) clipping does not rescue it."]))
    reg.register(_entry(
        "objective.eq23_aggregation",
        "U = -sum_n w_n c^n_t + beta * Gamma(tau_s)", None, "paper_lu_2024",
        reference=f"{_LU}; Eq. (23)",
        rationale="Aggregation structure implemented by the objective evaluator over explicit "
                  "component interfaces (tracking-cost provider, scanning-performance model). No GP "
                  "training, acquisition optimization or BO candidate generation inside the evaluator.",
        notes=["Single-slot objective; the paper's long-term discounted CMDP objective remains a "
               "recorded deviation (README section 14)."]))
    reg.register(_entry(
        "resource.task_utility_weights", r.task_utility_weights, "dimensionless", "proposed_assumption",
        reference=f"{_LU}; Eq. (23) weights w_n",
        rationale="Per-task weights of the weighted tracking aggregate. None (default) means UNIFORM "
                  "weights - a proposed software default, NOT a paper value. When supplied, weights "
                  "must be finite, non-negative and match the task count."))

    # ---- M4: synthetic testing components (clearly labeled) -------------------
    reg.register(_entry(
        "scanning.synthetic_linear_scanning_model",
        {"area_rate_m2_per_s": 2000.0, "reference_area_m2": 4000.0},
        "m^2/s, m^2", "experimental_setting",
        rationale="SYNTHETIC scanning response S_det(tau_s) = k_synth * tau_s with gamma = S_det / "
                  "S0_synth, for software integration tests ONLY. Constants are synthetic testing "
                  "values, not radar physics; outputs must never be presented as validated coverage, "
                  "detectable range or detection probability.",
        notes=["Deterministic; documented numerical domain tau_s >= 0 finite; invalid inputs rejected."]))
    reg.register(_entry(
        "objective.randomness_policy", "common_random_numbers", None, "proposed_assumption",
        rationale="Candidate evaluations rebuild the NumPy generator from the snapshot's CAPTURED "
                  "state, so all candidates see identical random realizations and evaluation order "
                  "cannot change outcomes. An independent-realizations policy would draw fresh "
                  "entropy per evaluation; CRN is the documented M4 default and is recorded in "
                  "every evaluation result."))

    # ---- M4: software design decisions ---------------------------------------
    reg.register(_entry(
        "allocation.discrete_membership_policy", "exact_match_no_snapping", None, "proposed_assumption",
        rationale="Flat-vector dwell values must match the configured action set EXACTLY "
                  "(tolerance 1e-12); near-members are rejected with the nearest member reported "
                  "for diagnosis. No snapping, rounding, clipping or repair (M4 instruction 9)."))
    reg.register(_entry(
        "objective.snapshot_truth_exclusion", "frozen_snapshot_is_estimator_facing", None, "proposed_assumption",
        rationale="The frozen snapshot carries posterior means/covariances, time, task ids and the "
                  "captured generator state; TRUE simulated states are deliberately NOT copied into "
                  "snapshots, so candidate-evaluation components cannot receive ground truth through "
                  "the snapshot interface. Truth stays in the authoritative SimulationState.",
        notes=["Snapshot integrity is content-hashed (sha256) and re-verified after every evaluation."]))
    reg.register(_entry(
        "objective.tracking_provider_m4_scope",
        "snapshot_diagnostic_eq14 (allocation-independent)", None, "proposed_assumption",
        rationale="M4 tracking-cost retrieval uses the paper's Eq. (14) on the snapshot's CURRENT "
                  "posteriors - allocation-independent by design. A candidate's effect on FUTURE "
                  "tracking requires simulation advancement (M5+); a synthetic allocation-dependent "
                  "provider exists for aggregation testing only (clearly labeled synthetic)."))

    # ---- M5: generic BO engine design decisions (application-free) -----------
    reg.register(_entry(
        "bo.problem_interface.scope", "application_free", None, "proposed_assumption",
        rationale="The generic problem interface (dimension, variable names/types/bounds, discrete "
                  "values, instantaneous feasibility, scalar objective, direction, noise declaration, "
                  "evaluation ids) imports NOTHING from the radar/environment/simulation layers "
                  "(enforced by test_layer_separation). Evaluation ids have the format "
                  "'{problem_id}_eval_{counter:05d}_{uuid8}'."))
    reg.register(_entry(
        "bo.benchmarks.a_b_c.origin", "synthetic_self_defined_for_verification", None, "proposed_assumption",
        rationale="Benchmarks A/B/C are synthetic problems defined in M5 to VERIFY the generic BO "
                  "engine (analytic/enumeration-checkable optima). They are NOT taken from "
                  "Benzaghta et al. or Lu et al.; the M4 diagnostic tracking provider is NOT used "
                  "as a BO benchmark (M5 instruction 5)."))
    reg.register(_entry(
        "bo.benchmark_a.reference", {"f_star": 10.0, "u_star": 3.5, "v_star": 7.2}, "dimensionless",
        "proposed_assumption",
        rationale="Synthetic continuous quadratic benchmark A: f(u,v) = 10 - (u-3.5)^2 - (v-7.2)^2 "
                  "on [-10,10]^2, maximize. Analytic exact reference derived independently "
                  "(vertex of the concave quadratic); verified by tests with math.isclose.",
        notes=["verification_method = analytic_exact; used as the BO gap reference in the M5 "
               "validation experiment."]))
    reg.register(_entry(
        "bo.benchmark_b.reference", {"f_star": 11.0, "d_star": 2.0, "x_star": 2.0}, "dimensionless",
        "proposed_assumption",
        rationale="Synthetic constrained mixed-variable benchmark B: discrete d in {0.5,1.0,1.5,2.0}, "
                  "continuous x in [0,10], constraint x + 3d <= 8, objective f = 4d + x^2/4 "
                  "(maximize). Enumeration over the 4 discrete values with the analytic clamped "
                  "inner optimum (x* = min(10, 8 - 3d)) gives per-d optima 3.0/6.0/9.0/11.0; "
                  "f* = 11.0 at (d=2.0, x=2.0).",
        notes=["verification_method = enumeration_exact (per-d values independently derived)."]))
    reg.register(_entry(
        "bo.benchmark_c.reference", "numerical_approximate (computed by the reference solver at run time)", None,
        "proposed_assumption",
        rationale="Synthetic multimodal benchmark C on [-2.5,2.5]^2 (sum of Gaussian bumps). NO "
                  "analytic optimum is claimed: the reference is computed numerically (differential "
                  "evolution plus a dense grid scan) by an independent solver and recorded with the "
                  "result; it may be exceeded by BO because it is approximate.",
        notes=["verification_method = numerical_approximate; the reference VALUE is never assumed, "
               "only computed and recorded."]))
    reg.register(_entry(
        "bo.duplicate_tolerance", 1e-9, "dimensionless", "proposed_assumption",
        rationale="Exact-match tolerance for duplicate-configuration detection (dataset insert and "
                  "pool generation). Deterministic problems never re-evaluate a known configuration; "
                  "noisy problems may repeat deliberately (allow_duplicates). Duplicates are never "
                  "silently merged."))
    reg.register(_entry(
        "bo.feasibility_tolerance", 1e-9, "dimensionless", "proposed_assumption",
        rationale="Numerical tolerance of the instantaneous constraint check in the generic problem "
                  "interface. Synthetic problems define their OWN explicit constraints (M4 review "
                  "3.1: per-action feasibility is distinct from discounted-budget feasibility)."))
    reg.register(_entry(
        "bo.consecutive_failure_cap", 3.0, "count", "proposed_assumption",
        rationale="Maximum consecutive objective/GP-fitting failures before the loop stops "
                  "(prevents unlimited failure loops; failures consume budget by documented policy)."))
    reg.register(_entry(
        "bo.initial_design_sampling", "seeded_uniform_feasible_random", None, "experimental_setting",
        rationale="The initial design is seeded uniform random sampling over the feasible region "
                  "(provisional size 30, README BOConfig experimental). Deliberately NOT Latin "
                  "hypercube or Sobol (explicit M5 instruction: initial design is random sampling, "
                  "distinct from the acquisition search)."))
    reg.register(_entry(
        "bo.refit_policy", "fresh_refit_per_observation", None, "experimental_setting",
        rationale="M5 initial GP-fitting policy: a fresh fit_gpytorch_mll refit after each successful "
                  "new observation (no warm-start; warm-starting would be a different, explicitly "
                  "identified and tested strategy)."))
    reg.register(_entry(
        "bo.selection_policy", "argmax_finite_log_ei_first_index_tiebreak", None, "proposed_assumption",
        rationale="Candidates are scored as a batch of single candidates (k, 1, d) against the "
                  "installed single-candidate LogExpectedImprovement (q=1 verified at runtime); "
                  "selection is argmax over FINITE log-domain scores with deterministic first-index "
                  "tie-breaking; non-finite scores are masked and reported, never repaired."))
    reg.register(_entry(
        "bo.failures_policy", "consume_budget_log_separately", None, "proposed_assumption",
        rationale="Failed objective-evaluation attempts consume the evaluation budget (real "
                  "evaluation cost), are recorded as explicit failure records with preserved "
                  "tracebacks, and NEVER enter the training dataset (M4 review 3.2 semantics)."))
    reg.register(_entry(
        "bo.incumbent_policy", "best_observed_never_predicted", None, "proposed_assumption",
        rationale="The incumbent is the best successfully OBSERVED feasible configuration (exact "
                  "vector preserved); it is NEVER replaced by an unevaluated GP prediction. "
                  "Termination is never described as proof of global convergence."))
    reg.register(_entry(
        "experiment.m5_synthetic_validation_budget", {"evaluation_budget": 100.0, "initial_design": 30.0},
        "count", "experimental_setting",
        rationale="Provisional synthetic budgets for the M5 generic-validation experiment only "
                  "(benchmarks A/B/C x seeds 0-9 x BO vs random search). Not paper-derived; the "
                  "general evaluation_budget remains UNRESOLVED and is set per experiment.",
        sensitivity=(30.0, 300.0)))

    # ---- M6: block architectures (application-free) ---------------------------
    reg.register(_entry(
        "bo.blocks.representation", "variable_blocks_index_based", None, "proposed_assumption",
        rationale="A block names an ordered subset of decision-variable indices; block metadata "
                  "(ids, active/fixed indices) is recorded as evaluation metadata, NEVER as GP "
                  "input features. Blocks must not overlap; uncovered variables stay fixed at the "
                  "current configuration and are reported. Full coverage is validated only when "
                  "required."))
    reg.register(_entry(
        "bo.scheduling.strategy", "deterministic_cyclic", None, "proposed_assumption",
        rationale="Blocks are visited in given order, repeating; one complete cycle = every block "
                  "visited or skipped once. Joint = one full-dimension block (M5 baseline); "
                  "alternating = two user-defined blocks; pair-wise ('coordinate') = at least two "
                  "two-variable blocks (two two-variable blocks match both readings; the declared "
                  "search_mode resolves the ambiguity)."))
    reg.register(_entry(
        "bo.current_config_vs_incumbent", "two_distinct_objects", None, "proposed_assumption",
        rationale="The CURRENT CONFIGURATION is the feasible complete vector around which block "
                  "candidates are built; the GLOBAL INCUMBENT is the best successfully evaluated "
                  "feasible configuration of the whole run. Both always refer to actual stored "
                  "observations; neither is ever replaced by an unevaluated GP prediction; they "
                  "are maintained independently and may differ."))
    reg.register(_entry(
        "bo.acceptance_policy", "improvement_only_current_config", None, "proposed_assumption",
        rationale="PROPOSED implementation choice: the current configuration is updated only when "
                  "a verified evaluation improves its value by more than the configured tolerance "
                  "improvement_tolerance (default 0). Exploratory evaluations that do not improve "
                  "remain in the shared dataset. Non-improving-acceptance policies are separate "
                  "experiments, not mixed with the primary comparison."))
    reg.register(_entry(
        "bo.evaluations_per_block_visit", 1.0, "count", "experimental_setting",
        rationale="Primary M6 strategy: ONE acquisition-selected evaluation per block visit "
                  "(which is NOT an exact block maximization). Configurable for later "
                  "experiments; all evaluations share the global budget.",
        sensitivity=(1.0, 5.0)))
    reg.register(_entry(
        "bo.full_gp_dimension_vs_block_dimension", "gp_trains_full_d", None, "proposed_assumption",
        rationale="ONE shared GP trains on ALL successful COMPLETE configurations (M x d, full "
                  "dimension) regardless of active block size; only the acquisition-SEARCH "
                  "dimension shrinks per block. Reducing the block dimension does not reduce GP "
                  "training cost. Separate per-block GPs would be a distinct, unapproved method."))
    reg.register(_entry(
        "bo.benchmark_stagnation_lp.reference", {"f_star": 9.6, "a_star": 1.6, "b_star": 0.0}, "dimensionless",
        "proposed_assumption",
        rationale="Synthetic block-stagnation LP (dim 4): max 3a1 + b1 + 3a2 + b2 s.t. "
                  "a1+b1+a2+b2 <= 3.2, all in [0, 1.6]. Analytic exact LP optimum 9.6 at "
                  "(1.6, 0, 1.6, 0) via the fractional-knapsack greedy argument. From budget-tight "
                  "configurations the alternating a/b grouping provably cannot improve while the "
                  "joint optimum exists (demonstrated in tests)."))
    reg.register(_entry(
        "bo.highdim_benchmarks.origin", "synthetic_self_defined_4_8_16", None, "proposed_assumption",
        rationale="M6 synthetic benchmarks in dimensions 4/8/16: separable quadratic (analytic "
                  "exact f* = 10d), coupled quadratic via fixed PD mixing matrix (analytic exact "
                  "f* = 20), constrained mixed 2-discrete + continuous with shared budget "
                  "(numerical_approximate reference via enumeration + KKT multiplier bisection), "
                  "and the block-stagnation LP (analytic exact). All parameters synthetic; none "
                  "from the reference papers."))
    reg.register(_entry(
        "experiment.m6_block_architecture_budget", {"evaluation_budget": 100.0, "initial_design": 30.0},
        "count", "experimental_setting",
        rationale="Primary M6 comparison settings inherited from M5: budget 100 total ATTEMPTED "
                  "objective evaluations (NOT per block), initial design 30, pool 500 per search, "
                  "seeds 0-9. The budget is shared across architectures for fairness.",
        sensitivity=(30.0, 300.0)))

    # ---- Experimental settings ------------------------------------------------
    reg.register(_entry(
        "optimization.num_initial_observations", o.num_initial_observations, "count", "experimental_setting",
        rationale="Initial space-filling design size; no paper default (README BOConfig proposes 30).",
        sensitivity=(10, 60)))
    reg.register(_entry(
        "optimization.evaluation_budget", o.evaluation_budget, "count", "experimental_setting",
        rationale="UNRESOLVED by design: no paper default; each experiment must set it explicitly "
                  "(OptimizationConfig.ensure_ready fails clearly otherwise)."))
    reg.register(_entry(
        "environment.num_targets", e.num_targets, "count", "experimental_setting",
        rationale="Fixed-population synthetic scenario (README 4.4); paper's population is not "
                  "specified for reproduction."))
    reg.register(_entry(
        "environment.random_seed", e.random_seed, None, "experimental_setting",
        rationale="Master seed for reproducibility; per-repetition seeds derived deterministically."))
    reg.register(_entry(
        "experiment.repetitions", c.experiment.repetitions, "count", "experimental_setting",
        rationale="Independent repetitions for statistical reporting; no paper default."))
    reg.register(_entry(
        "resource.time_budget_semantics", r.time_budget_semantics, None, "proposed_assumption",
        reference=f"{_LU}; Eq. (22)",
        rationale="Default reproduces the paper's discounted cumulative budget. The proposed "
                  "'hard_per_slot' alternative is STRICTER (per-slot bound vs discounted cumulative "
                  "bound); recorded as a deviation whenever used.",
        notes=["M3 review (instruction 3.4): 'hard_per_slot' is reserved for the proposed BO "
               "configuration profile (M4+); no resource allocation or budget optimization is "
               "implemented in M3."]))

    # ---- M7: experimental framework and scientific evaluation -----------------
    reg.register(_entry(
        "experiment.m7_runner_architecture",
        {"checkpoint_per_run_files": 1.0, "atomic_write": 1.0, "resume_compatibility_checks": 7.0},
        "policy", "proposed_assumption",
        rationale="M7 experiment runner: per-run atomic JSON records (tmp + os.replace), manifest with "
                  "explicit statuses (pending/running/completed/failed/skipped + reasons), resume "
                  "compatibility verified against configuration fingerprint, benchmark/algorithm versions, "
                  "dependency versions, seeds and run ids; corrupted/incompatible records are REJECTED, "
                  "never silently combined.",
        notes=["Layer separation: the experiment layer never carries radar configuration."]))
    reg.register(_entry(
        "experiment.m7_shared_initialization_policy",
        {"policy": 1.0},
        "policy", "proposed_assumption",
        rationale="Fairness (item 10): all methods for one (benchmark, seed) run with the same seed "
                  "through the same deterministic sampler; the runner then VERIFIES identical initial "
                  "configurations AND identical observed values directly from the stored records "
                  "(seeds alone are never trusted). Random search has no initial-design phase by "
                  "construction and is excluded from the comparison, reported explicitly.",
        notes=["Deviation documented: initial designs are NOT injected into the engine; the engine's "
               "seeded sampler IS the shared design, verified post hoc from records."]))
    reg.register(_entry(
        "experiment.m7_statistical_conventions",
        {"bootstrap_replications": 10000.0, "bootstrap_seed": 2026.0, "paired_tie_tolerance": 1e-9,
         "confidence_level": 0.95},
        "setting", "proposed_assumption",
        rationale="M7 statistics: paired comparisons preserve seed-level pairing; ties use the engine "
                  "duplicate tolerance 1e-9; bootstrap resamples COMPLETE seed-level pairs with a "
                  "documented seed (2026) and B=10000; Wilcoxon two-sided signed-rank on non-zero "
                  "differences, Holm-corrected when presented as a family; p-values complement effect "
                  "sizes and CIs and never imply practical significance alone.",
        notes=["Descriptive over 10 seeds; a narrow interval is NOT evidence of universal superiority."]))
    reg.register(_entry(
        "experiment.m7_threshold_definitions",
        {"gap_closure_fractions": 0.9},
        "fraction", "proposed_assumption",
        rationale="M7 threshold metrics: fractions of the INITIAL REFERENCE GAP (0.5/0.9/0.99 by "
                  "default), frozen before new comparative experiments; unreached thresholds are "
                  "reported explicitly (never interpolated); the initial-design-already-meets case "
                  "is flagged. Distinct from the M6 threshold convention (fraction of f*).",
        sensitivity=(0.5, 0.99)))
    reg.register(_entry(
        "experiment.m7_runtime_accounting",
        {"stage_timers": 5.0, "unattributed_reported": 1.0},
        "policy", "proposed_assumption",
        rationale="M7 runtime decomposition: five disjoint stage timers (GP fit, candidate generation, "
                  "feasibility validation, acquisition scoring, objective evaluation) plus explicit "
                  "unattributed overhead = wall - sum(components). FINDING: the archived M6 generation "
                  "timer INCLUDED nested feasibility validation; the boundary is fixed for M7+ records "
                  "(disjoint), historical values are preserved unchanged and corrected only in derived "
                  "artifacts. The decomposition is never claimed to sum exactly to wall time.",
        notes=["Serialization, tensor conversion, dataset insertion, incumbent tracking and GP "
               "training-tensor preparation are NOT separately timed."]))
    reg.register(_entry(
        "experiment.m7_candidate_profiling_baseline",
        {"algorithm_baseline": 1.0},
        "policy", "proposed_assumption",
        rationale="M6 candidate-generation algorithm is preserved UNCHANGED as the versioned baseline; "
                  "profiling distinguishes PROVEN finite-space exhaustion (all-discrete blocks with "
                  "enumerable combinations) from sampling failure (continuous spaces: an empty pool "
                  "does not prove an empty feasible set). Any future faster generator must be a new "
                  "explicitly versioned implementation.",
        notes=["M6 measured (mean across 10 seeds, constrained_mixed_8d): alternating generation "
               "251.5 s incl. nested feasibility 67.4 s (corrected derived value 184.1 s); joint 77.4 s; "
               "pairwise 103.9 s."]))
    reg.register(_entry(
        "experiment.m7_validation_matrix",
        {"benchmarks": 2.0, "seeds": 3.0, "evaluation_budget": 40.0, "initial_design": 10.0},
        "count", "experimental_setting",
        rationale="M7 limited validation matrix (item 26): benchmark_a (2-D) + constrained_mixed_8d, "
                  "methods RS/joint/alternating/pairwise/reference, seeds 0-2, budget 40 total "
                  "(initial 10). Purpose: validate the INFRASTRUCTURE (runner, checkpoint/resume, "
                  "metrics, reporting), NOT algorithm superiority. The 230 archived M6 runs are "
                  "NOT repeated.",
        notes=["Manifest records 2 incompatible combinations as skipped with reasons "
               "(benchmark_a defines no recommended block groupings)."]))
    reg.register(_entry(
        "experiment.m7_milestone_reconciliation",
        {"instruction_stream_milestones": 9.0, "original_roadmap_milestones": 8.0},
        "count", "proposed_assumption",
        rationale="Milestone numbering reconciliation: the original README roadmap has EIGHT milestones "
                  "(M1 repo/env, M2 config/provenance, M3 estimation, M4 interfaces, M5 generic BO, "
                  "M6 verification of BO components, M7 architectures validation, M8 runner package + "
                  "analysis, M9 final docs in the instruction stream). The instruction stream's M7 "
                  "(experimental framework) maps onto the README roadmap's M8 'runner package' plus "
                  "parts of M7 'analysis extensions'; the instruction stream carries one additional "
                  "milestone (M9 final documentation) beyond the roadmap's eight. Every deviation is "
                  "documented in the README.",
        notes=["No milestone renumbering was applied to existing artifacts; existing archives keep "
               "their original M5/M6 labels."]))

    # ---- M8A: scientific model review and parameter resolution -----------------
    reg.register(_entry(
        "experiment.m8a_run_count_reconciliation",
        {"archived_full_matrix_records": 220.0, "pilot_records": 32.0},
        "count", "proposed_assumption",
        rationale="M8A item 3.2 reconciliation: the durable M6 full-matrix archive contains exactly "
                  "220 records (160 primary = 4 benchmarks x 4 methods x 10 seeds + 60 dimension-study "
                  "= 2 x 3 x 10) and the pilot archive 32 records. The '230 runs' figure quoted in the "
                  "M6/M7 narrative summaries was an ARITHMETIC/REPORTING ERROR (correct count 220); "
                  "the archives themselves were always internally consistent and were validated as "
                  "such by the M7 independent checks. No records are missing or invented.",
        notes=["Corrected via documentation only; historical archive values untouched."]))
    reg.register(_entry(
        "experiment.m8a_r1_source_verification",
        {"pdfs_reextracted": 2.0, "icc_pages": 6.0, "globecom_pages": 6.0},
        "count", "proposed_assumption",
        rationale="M8A-R1 Phase A: both original research PDFs (uploaded to the repository root at "
                  "commit 321e167 and tracked since) were re-extracted to notes/paper_extracts/ "
                  "(pypdf 7.2.0 installed to a scratch directory OUTSIDE the project venv and lock; "
                  "documented requirement: no PDF tool existed in the environment) and verified "
                  "line-by-line: equations (1)-(32), Table I (10 symbols, no values), Table II "
                  "(14 values), the reference list (14 entries), author names, venue, pages and "
                  "DOIs. Footer-verified DOIs: 10.1109/ICC51166.2024.10622225 (ICC 2024) and "
                  "10.1109/GLOBECOM54140.2023.10437768 (GLOBECOM 2023, pp. 4552-4557). IEEE Xplore "
                  "download stamps in the PDF footers: IIT Jodhpur, 2026-10-10 (ICC) and "
                  "2026-05-26 (GLOBECOM). All M8A documents were corrected accordingly; no "
                  "numerical assumptions were introduced.",
        notes=["Finding: the ICC paper's own Table II row reads 'sigma_w ((m/s^2)^2) 25' — the "
               "unit annotation supports the table_variance reading; the accepted resolution is "
               "preserved unchanged.",
               "Finding: ICC Eq. (6) is PRINTED as sigma^2 = sigma^2_0 * SNR (multiplication) "
               "while the paper's own prose says variance DECREASES with dwell time and the "
               "extended version divides; the inverse_snr operational resolution is preserved.",
               "Finding: the ICC dwell action set is {0.1..1.0} s (Section V-A.2); the M8A "
               "documents that said otherwise were corrected (register value was always right)."]))

    # ---- M8A-R1 Phase A: the radar paper's weight definitions (recorded FACTS;
    # ---- NOT adopted as operational defaults — task_utility_weights stays None) --
    reg.register(_entry(
        "paper_lu_2024.training_weight_set", [1.0, 3.0, 5.0], "dimensionless", "paper_lu_2024",
        reference=f"{_LU}; Section VI (Numerical Results), M8A-R1 verified against the source PDF",
        rationale="TRAINING weights: 'During the training, the weights w^n_t assigned for tracking "
                  "the targets are selected from the set {1, 3, 5}, and DQN is trained using "
                  "targets with different trajectories and weights.' Recorded as a paper fact "
                  "for reproduction purposes; this repository's operational default remains the "
                  "labeled uniform default (resource.task_utility_weights = None) — NOT changed "
                  "by this record.",
        notes=["Adopting {1,3,5} (or any sampled rule) as an operational default would be a "
               "proposed numerical selection requiring separate review + approval."]))
    reg.register(_entry(
        "paper_lu_2024.testing_weight_rule",
        "Eq. (31): w_n = min{10, 4 + 10000/d0} if d0 <= 10000 else 1, with d0 = min(d_n_r, d_n_a) "
        "(estimated distances to radar and to asset); Eq. (32): w_n = min{10, w_n + 5} if "
        "A_n < 0 else w_n, with A_n = x_n . xdot_n (position dot velocity; A_n < 0 = target "
        "approaching the radar). Weights are TIME-VARYING during tests, computed from ESTIMATED "
        "positions/velocities, capped at 10.",
        "definition", "paper_lu_2024",
        reference=f"{_LU}; Section VI, Eqs. (31)-(32), M8A-R1 verified against the source PDF",
        rationale="TESTING weights: the ICC paper defines time-varying test weights from estimated "
                  "distances (radar/asset) and the approaching-target velocity rule. Recorded as "
                  "a paper fact; the definition uses ESTIMATED quantities (compatible with our "
                  "truth-inaccessibility guarantee). Not adopted as an operational default.",
        notes=["Both definitions use estimated (not true) quantities, so a faithful reproduction "
               "would compute weights inside the estimator-facing snapshot layer."]))
    reg.register(_entry(
        "experiment.m8a_shared_initial_design_interface",
        {"engine_parameter": 1.0, "framework_mode": 1.0},
        "policy", "proposed_assumption",
        rationale="M8A item 3.3: BayesianOptimizer accepts initial_design_configs (validated immutable "
                  "explicit shared design: exact count, dimension, finiteness and feasibility enforced "
                  "fail-fast; input never mutated; pool report records the explicit mode with zero "
                  "sampling cost). The experiment framework gains initial_design.mode='explicit_shared': "
                  "one deterministic design per (benchmark, dimension, seed) is generated from the "
                  "registry sampler and passed UNCHANGED to every applicable method, guaranteeing "
                  "identical initial configurations AND observed values by construction (still verified "
                  "from stored records). Default None preserves M5/M6 seeded behavior bit-for-bit "
                  "(frozen-trajectory regression test passes unchanged).",
        notes=["Historical optimization trajectories and archives are unaffected."]))
    return reg


def _main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("provenance_register.json")
    reg = build_default_provenance()
    reg.save_json(out)
    unresolved = reg.unresolved_names()
    print(f"Wrote {len(reg.entries)} provenance entries to {out}")
    print(f"Unresolved ({len(unresolved)}): {unresolved}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv))

