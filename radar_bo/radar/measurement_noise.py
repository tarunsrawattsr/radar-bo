"""SNR-dependent planar measurement-noise model — Lu et al., ICC 2024, Eqs. (5), (6), (8).

Paper-reported expressions (authoritative source: supplied PDF):

* Eq. (5): ``SNR_t(tau_t, r_t) = SNR0 * (tau_t / tau_0) * (r_t / r_0)^-4`` —
  linear (not dB) SNR depending on dwell time and range;
* Eq. (6): measurement-noise variances follow the SNR;
* Eq. (8): ``R_t = diag(sigma^2_{r,t}, sigma^2_{theta,t})``.

Eq. (6) semantics — settled (M3 review, reaffirmed M4)
------------------------------------------------------
The accepted relationship is

    sigma^2_{.,t} = sigma^2_{.,0} / SNR_t

(reference measurement variance divided by the linear SNR). This is the
paper's intended semantics: the prose directly below Eq. (6) states that
"the variance of the measurement noise decreases when a longer dwell time is
allocated to the target or when the target moves closer to the radar
according to equations (5) and (6)". The transient multiplication reading in
early extractions was a PDF-EXTRACTION ARTIFACT (fraction/glyph layout in
the two-column layout), not an unresolved mathematical ambiguity of the
paper; it is preserved only as extraction history in the provenance entry
``tracking.measurement_variance_semantics``. Inverse-SNR scaling is the
paper semantics and is NOT classified as a proposed mathematical deviation.

``reference_snr`` (SNR0) is NOT reported anywhere in the paper: it stays
``None`` in the configuration and any operation requiring it raises
:class:`UnresolvedParameterError` — the value is never invented (README
non-negotiable). Synthetic experiments may construct this model with an
explicitly labeled synthetic SNR value.

No transmit-power-dependent measurement model is introduced in M3.
"""

from __future__ import annotations

import numpy as np

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.environment.target_state import MeasurementCovariance
from radar_bo.numerics import NumericalValidationError

_SNR0_UNIT = "linear SNR (not dB)"


class PlanarSNRMeasurementNoiseModel:
    """Eqs. (5), (6), (8): SNR model and the resulting 2 x 2 measurement covariance."""

    def __init__(
        self,
        reference_range_variance: float,
        reference_angle_variance: float,
        reference_distance: float,
        reference_dwell_time: float,
        reference_snr: float | None,
    ) -> None:
        for name, value in (
            ("reference_range_variance", reference_range_variance),
            ("reference_angle_variance", reference_angle_variance),
            ("reference_distance", reference_distance),
            ("reference_dwell_time", reference_dwell_time),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise NumericalValidationError(f"{name} must be finite and > 0; got {value!r}.")
        if reference_snr is not None and (not np.isfinite(reference_snr) or reference_snr <= 0.0):
            raise NumericalValidationError(f"reference_snr must be finite and > 0 when supplied; got {reference_snr!r}.")
        self.reference_range_variance = float(reference_range_variance)
        self.reference_angle_variance = float(reference_angle_variance)
        self.reference_distance = float(reference_distance)
        self.reference_dwell_time = float(reference_dwell_time)
        self.reference_snr = None if reference_snr is None else float(reference_snr)

    @classmethod
    def from_config(cls, tracking_config) -> "PlanarSNRMeasurementNoiseModel":
        """Build from the validated TrackingConfig (``reference_snr`` may be None)."""
        return cls(
            reference_range_variance=tracking_config.reference_range_variance,
            reference_angle_variance=tracking_config.reference_angle_variance,
            reference_distance=tracking_config.reference_distance,
            reference_dwell_time=tracking_config.reference_dwell_time,
            reference_snr=tracking_config.reference_snr,
        )

    # ---- Eq. (5) ---------------------------------------------------------------
    def snr(self, dwell_time: float, range_m: float) -> float:
        """Linear ``SNR_t = SNR0 * (tau_t / tau_0) * (r_t / r_0)^-4``.

        Raises :class:`UnresolvedParameterError` when the paper-unresolved
        ``reference_snr`` (SNR0) is still ``None`` — values are never invented.
        """
        if self.reference_snr is None:
            raise UnresolvedParameterError(
                {"tracking.reference_snr": _SNR0_UNIT},
                guidance=(
                    "SNR0 is required by Eq. (5) but is not reported anywhere in Lu et al. "
                    "Supply a justified value in TrackingConfig.reference_snr, or (synthetic "
                    "experiments only) construct this model with an explicitly labeled synthetic SNR."
                ),
            )
        tau = float(dwell_time)
        r = float(range_m)
        for name, value in (("dwell_time", tau), ("range_m", r)):
            if not np.isfinite(value) or value <= 0.0:
                raise NumericalValidationError(f"{name} must be finite and > 0; got {value!r}.")
        return self.reference_snr * (tau / self.reference_dwell_time) * (self.reference_distance / r) ** 4

    # ---- Eq. (6) (division semantics; see module docstring) ---------------------
    def measurement_covariance_from_snr(self, snr_value: float) -> MeasurementCovariance:
        """``R = diag(sigma^2_{r,0} / SNR, sigma^2_{theta,0} / SNR)`` for a given linear SNR."""
        s = float(snr_value)
        if not np.isfinite(s) or s <= 0.0:
            raise NumericalValidationError(f"snr_value must be finite and > 0; got {snr_value!r}.")
        matrix = np.diag(
            [
                self.reference_range_variance / s,
                self.reference_angle_variance / s,
            ]
        )
        return MeasurementCovariance(matrix)

    def observation_covariance(self, dwell_time: float, range_m: float) -> MeasurementCovariance:
        """Eqs. (5)+(6)+(8) combined: covariance for a dwell time and target range."""
        return self.measurement_covariance_from_snr(self.snr(dwell_time, range_m))
