"""TrackingConfig: planar target dynamics, measurement model and EKF parameters.

Authoritative source: Z. Lu et al., "Learning-Based Cognitive Radar Resource
Management for Scanning and Multi-Target Tracking", IEEE ICC 2024,
DOI 10.1109/ICC51166.2024.10622225 — Table II (p. 2789) and Eqs. (1)-(15).

Conventions
-----------
* SI units; angles in radians; linear (not dB) quantities; double precision.
* Variance fields already ARE variances; they are never squared a second time
  (README section 4.1).
* Values not reported by the paper are ``None`` and must never be invented;
  see :mod:`radar_bo.config.provenance`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from radar_bo.config.base import StrictModel

InitializationPolicy = Literal[
    # x0 = zero vector, P0 = I_4 (Lu et al., p. 2786)
    "paper_zero_identity",
    # initialize the filter from the first noisy observation (proposed fix
    # for the singular origin geometry; must be justified before use)
    "proposed_measurement_driven",
    # use initial_state / initial_covariance exactly as supplied (experimental)
    "experimental_custom",
]

ManeuverabilityInterpretation = Literal[
    # sigma_w = 25 is the acceleration variance ((m/s^2)^2): Table II units and
    # prose say "variance", so the process-noise covariance is scaled by 25.
    "table_variance",
    # Eq. (3) taken literally: Q_t is scaled by sigma_w^2 = 25^2 = 625.
    "equation_literal",
]

MeasurementVarianceScaling = Literal[
    # Eq. (6) as printed: sigma^2_{.,t} = sigma^2_{.,0} * SNR_t, i.e. the
    # measurement variance GROWS with SNR. Flagged as a printed deviation
    # from standard radar behaviour (README: record deviations).
    "paper_as_printed",
    # Standard radar behaviour: measurement variance SHRINKS with SNR.
    # Proposed alternative, enabled for sensitivity studies.
    "proposed_inverse_snr",
]

_IDENTITY_4: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class TrackingConfig(StrictModel):
    """Paper-faithful defaults for the planar CV + range/azimuth EKF baseline.

    Paper-reported values reproduce Lu et al. Table II exactly. Known
    ambiguities in the paper are exposed as explicit, switchable fields with a
    documented default (user decision: implement both interpretations, never a
    single silent choice).
    """

    # ---- Paper-reported values (Lu et al., Table II, p. 2789) ---------------
    reference_range_variance: float = Field(
        default=100.0,
        gt=0.0,
        description=(
            "sigma^2_{r,0} = 100 m^2 (paper-reported variance of the range "
            "measurement at reference SNR; do not square again)."
        ),
    )
    reference_angle_variance: float = Field(
        default=4.0e-4,
        gt=0.0,
        description=(
            "sigma^2_{theta,0} = 4e-4 rad^2 (paper-reported variance of the "
            "azimuth measurement at reference SNR; radians)."
        ),
    )
    maneuverability_noise: float = Field(
        default=25.0,
        gt=0.0,
        description=(
            "sigma_w = 25 from Table II. The paper is ambiguous (Eq. (3) "
            "squares sigma_w; prose and Table II units call it a variance in "
            "(m/s^2)^2). Resolve via maneuverability_noise_interpretation."
        ),
    )
    maneuverability_noise_interpretation: ManeuverabilityInterpretation = Field(
        default="table_variance",
        description=(
            "'table_variance' (default): sigma_w = 25 is the acceleration "
            "variance ((m/s^2)^2), so the process-noise covariance is scaled "
            "by 25. 'equation_literal': Eq. (3) as printed, Q is scaled by "
            "sigma_w^2 = 625. Both are kept switchable; sensitivity to this "
            "choice must be tested (README section 4.1)."
        ),
    )
    reference_distance: float = Field(
        default=3000.0, gt=0.0, description="r0 = 3000 m, reference range in the SNR model, Eq. (5)."
    )
    reference_dwell_time: float = Field(
        default=1.0, gt=0.0, description="tau0 = 1.0 s, reference dwell time in the SNR model, Eq. (5)."
    )
    revisit_interval: float = Field(
        default=2.5,
        gt=0.0,
        description=(
            "T0 = 2.5 s (Table II). Single authoritative source for the revisit "
            "interval: other layers reference this field, they do not redefine "
            "it (README section 4.6)."
        ),
    )

    # ---- Unresolved parameters (not reported by the paper) ------------------
    reference_snr: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "SNR0 in Eq. (5): NOT reported anywhere in Lu et al. Preserved as "
            "None with a provenance record; linear (not dB) when resolved."
        ),
    )
    reference_transmit_power: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Extension, not a paper default. ResourceConfig.reference_transmit_power "
            "is authoritative for shared use; SimulationConfig cross-validates "
            "that both are equal when both are supplied."
        ),
    )

    # ---- Structural dimensions (planar CV model, Eqs. (1)-(2)) --------------
    state_dimension: int = Field(
        default=4, frozen=True, description="Planar constant-velocity state [x, y, vx, vy]."
    )
    measurement_dimension: int = Field(
        default=2, frozen=True, description="Measurement vector [range, azimuth] per Eq. (4)."
    )

    # ---- Initialization (paper-reported, with documented alternatives) ------
    initial_state: tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 0.0),
        description=(
            "x0 = zero vector (paper-reported, p. 2786). NOTE: with the target "
            "at the origin the range/azimuth geometry is singular "
            "(atan2(0, 0) undefined, measurement Jacobian degenerate); this is "
            "why an explicit initialization policy exists."
        ),
    )
    initial_covariance: list[list[float]] = Field(
        default=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        description=(
            "P0 = I_4 (paper-reported). Must be symmetric positive "
            "semidefinite. This is an initial GUESS covariance, not a physical "
            "variance with units."
        ),
    )
    initialization_policy: InitializationPolicy = Field(
        default="paper_zero_identity",
        description=(
            "'paper_zero_identity' (default) reproduces the paper (x0 = 0, "
            "P0 = I) including its degenerate origin geometry. "
            "'proposed_measurement_driven' initializes the filter from the "
            "first noisy observation instead (proposed fix, implemented with "
            "the estimator; must be justified before use). "
            "'experimental_custom' uses the supplied initial_state and "
            "initial_covariance as-is."
        ),
    )
    measurement_variance_scaling: MeasurementVarianceScaling = Field(
        default="paper_as_printed",
        description=(
            "How Eq. (6) maps SNR to measurement variances. 'paper_as_printed' "
            "reproduces sigma^2_{.,t} = sigma^2_{.,0} * SNR_t even though it "
            "makes the variance grow with SNR (flagged deviation). "
            "'proposed_inverse_snr' implements the standard inverse relation "
            "for sensitivity studies."
        ),
    )

    # ---- Validators ----------------------------------------------------------
    @field_validator("initial_covariance")
    @classmethod
    def _validate_initial_covariance(cls, v: list[list[float]]) -> list[list[float]]:
        a = np.asarray(v, dtype=np.float64)
        if a.shape != (4, 4):
            raise ValueError(f"initial_covariance must be 4x4 for the planar model; got shape {a.shape}.")
        if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
            raise ValueError("initial_covariance must be symmetric.")
        eig_min = float(np.linalg.eigvalsh(a).min())
        if eig_min < -1e-10:
            raise ValueError(
                f"initial_covariance must be positive semidefinite; min eigenvalue {eig_min:.3e}."
            )
        return v

    @model_validator(mode="after")
    def _policy_matches_supplied_initialization(self) -> "TrackingConfig":
        if self.initialization_policy == "paper_zero_identity":
            is_zero = all(s == 0.0 for s in self.initial_state)
            is_identity = bool(np.allclose(np.asarray(self.initial_covariance), np.eye(4), atol=1e-12))
            if not (is_zero and is_identity):
                raise ValueError(
                    "initialization_policy='paper_zero_identity' requires the paper defaults "
                    "(initial_state = zero vector, initial_covariance = I_4). Use "
                    "'experimental_custom' when overriding either of them."
                )
        return self

    @model_validator(mode="after")
    def _dwell_within_revisit(self) -> "TrackingConfig":
        if self.reference_dwell_time > self.revisit_interval:
            raise ValueError(
                f"reference_dwell_time ({self.reference_dwell_time} s) must not exceed "
                f"revisit_interval ({self.revisit_interval} s)."
            )
        return self
