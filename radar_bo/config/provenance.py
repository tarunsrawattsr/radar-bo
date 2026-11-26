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
                  "directly; no further squaring occurs.",
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
        reference=f"{_LU}; action space, Sections II-III",
        rationale="Dwell-time actions tau in {0.1, ..., 1.0} s in 0.1 s steps."))
    reg.register(_entry(
        "resource.normalized_time_budget_threshold", r.normalized_time_budget_threshold, "dimensionless",
        "paper_lu_2024", reference=f"{_LU}; Eq. (22)",
        rationale="Theta_max = 1, normalized time-budget threshold."))
    reg.register(_entry(
        "resource.tracking_scanning_tradeoff", r.tracking_scanning_tradeoff, "dimensionless", "paper_lu_2024",
        reference=f"{_LU}; Eq. (23)",
        rationale="beta = 1e6 weights scanning reward against tracking cost. Reproduction only; "
                  "recalibrate if metric definitions change.",
        notes=["Units of the two weighted terms are not matched by construction in the paper."]))
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
        "optimization.acquisition_xi", o.acquisition_xi, "dimensionless", "paper_benzaghta_2023",
        reference=_BENZ, rationale="Expected-improvement exploration parameter xi = 0.01.",
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
