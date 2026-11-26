"""Scanning-performance abstraction — Lu et al., ICC 2024, Eqs. (16)-(21), Table I.

Paper-reported expressions (authoritative source: supplied PDF):

* Eq. (21): ``Gamma = S_max / S_0`` — the scanning performance is the ratio of
  the maximum detectable area ``S_max`` to a reference area ``S_0``.
* Eqs. (16)-(20): the physical chain that PRODUCES ``S_max`` (scanning time,
  radar equation, noise, Shnidman-required SNR). The paper does not provide
  the numerical parameters for this chain (Table I lists symbols only), so the
  physical chain is NOT implemented in M4.

Design (M4 instructions 5-6):

* ``AreaRatioMetric`` implements Eq. (21) alone: it accepts a modeled
  detectable area and a strictly positive reference area, and returns the
  ratio WITHOUT capping at one (the paper applies no cap).
* ``ScanningPerformanceModel`` is the substitution interface: different
  scanning models can be plugged into the objective evaluator without
  modifying it. The interface is independent of BO, GP fitting, acquisition
  functions and optimization algorithms.
* ``SyntheticLinearScanningModel`` is a clearly labeled SYNTHETIC response
  used for software integration tests only (M4 instruction 7). Its outputs
  are NOT physically validated radar coverage, detectable range or
  probability-of-detection results.
* Operations requiring unresolved physical parameters (e.g. the paper's
  ``S0``, still ``None`` in ScanningConfig) raise
  :class:`radar_bo.config.base.UnresolvedParameterError` — no defaults are
  substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.numerics import NumericalValidationError

_UNIT_REFERENCE_AREA = "m^2 (paper symbol S0 in Eq. (21))"


def area_ratio(detectable_area_m2: float, reference_area_m2: float) -> float:
    """Eq. (21): ``Gamma = S_max / S_0`` as a pure function.

    The ratio is NOT capped at one. Validates finiteness and a strictly
    positive reference area; raises :class:`NumericalValidationError`
    otherwise.
    """
    s_max = float(detectable_area_m2)
    s0 = float(reference_area_m2)
    if not np.isfinite(s_max) or s_max < 0.0:
        raise NumericalValidationError(f"detectable_area_m2 must be finite and >= 0; got {detectable_area_m2!r}.")
    if not np.isfinite(s0) or s0 <= 0.0:
        raise NumericalValidationError(f"reference_area_m2 must be finite and > 0; got {reference_area_m2!r}.")
    return s_max / s0


@dataclass(frozen=True)
class ScanningPerformance:
    """Output contract of a scanning-performance model.

    ``gamma`` is the Eq. (21) normalized performance computed by the model's
    metric against its configured reference area (uncapped). ``is_synthetic``
    marks synthetic models so results can never be mistaken for physically
    validated coverage.
    """

    detectable_area_m2: float
    reference_area_m2: float
    gamma: float
    model_identifier: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("detectable_area_m2", self.detectable_area_m2),
            ("reference_area_m2", self.reference_area_m2),
            ("gamma", self.gamma),
        ):
            if not np.isfinite(value):
                raise NumericalValidationError(f"ScanningPerformance.{name} must be finite; got {value!r}.")


@runtime_checkable
class ScanningPerformanceModel(Protocol):
    """Substitution interface for scanning models (M4 instruction 5).

    Implementations convert an available scanning-resource quantity (seconds
    of scanning time, per Eq. (24) accounting) into a modeled detectable area
    and its Eq. (21) normalized ratio. Implementations MUST be deterministic
    for identical inputs unless randomness is explicitly declared and an
    explicit ``numpy.random.Generator`` is threaded through.
    """

    model_identifier: str
    is_synthetic: bool

    def evaluate(self, available_scanning_time_s: float) -> ScanningPerformance: ...


class AreaRatioMetric:
    """Eq. (21) metric bound to a configured reference area ``S0``.

    ``reference_area_m2=None`` represents the paper-unresolved ``S0``; any
    use then raises :class:`UnresolvedParameterError` (no default value is
    substituted). Physical scanning calculations live in the models, not in
    this metric.
    """

    metric_identifier: Final[str] = "paper_eq21_area_ratio"

    def __init__(self, reference_area_m2: float | None) -> None:
        if reference_area_m2 is not None:
            value = float(reference_area_m2)
            if not np.isfinite(value) or value <= 0.0:
                raise NumericalValidationError(
                    f"reference_area_m2 must be finite and > 0 when supplied; got {reference_area_m2!r}."
                )
            self.reference_area_m2: float | None = value
        else:
            self.reference_area_m2 = None

    def normalize(self, detectable_area_m2: float) -> float:
        """Eq. (21) with the configured reference area; uncapped."""
        if self.reference_area_m2 is None:
            raise UnresolvedParameterError(
                {"scanning.reference_area": _UNIT_REFERENCE_AREA},
                guidance=(
                    "S0 is unspecified in Lu et al. (Table I lists symbols only). Supply a "
                    "justified value in ScanningConfig.reference_area, or use a synthetic "
                    "scanning model with an explicitly labeled synthetic reference area "
                    "(software integration testing only)."
                ),
            )
        return area_ratio(detectable_area_m2, self.reference_area_m2)


class SyntheticLinearScanningModel:
    """SYNTHETIC scanning response for software integration tests (not physics).

    Response (documented numerical domain)::

        S_det(tau_s) = k_synth * tau_s        [m^2],  tau_s >= 0, finite
        gamma        = S_det / S0_synth        (Eq. (21) via AreaRatioMetric, uncapped)

    with strictly positive synthetic constants ``k_synth`` (m^2/s) and
    ``S0_synth`` (m^2). Deterministic for identical inputs; no stochastic
    component. Outputs are finite for finite inputs; invalid inputs (negative
    or non-finite time) raise :class:`NumericalValidationError`. All constants
    are SYNTHETIC testing values recorded in the provenance register — they
    are not radar physics and must never be presented as validated coverage,
    detectable range or detection probability.
    """

    model_identifier: Final[str] = "synthetic_linear_scanning_v1"
    is_synthetic: Final[bool] = True

    def __init__(
        self,
        synthetic_area_rate_m2_per_s: float = 2000.0,
        synthetic_reference_area_m2: float = 4000.0,
    ) -> None:
        for name, value in (
            ("synthetic_area_rate_m2_per_s", synthetic_area_rate_m2_per_s),
            ("synthetic_reference_area_m2", synthetic_reference_area_m2),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise NumericalValidationError(f"{name} must be finite and > 0; got {value!r}.")
        self.synthetic_area_rate_m2_per_s = float(synthetic_area_rate_m2_per_s)
        self._metric = AreaRatioMetric(reference_area_m2=float(synthetic_reference_area_m2))
        self.synthetic_reference_area_m2 = float(synthetic_reference_area_m2)

    def detectable_area(self, available_scanning_time_s: float) -> float:
        """Synthetic response ``S_det = k_synth * tau_s`` (documented, verifiable)."""
        tau_s = float(available_scanning_time_s)
        if not np.isfinite(tau_s) or tau_s < 0.0:
            raise NumericalValidationError(
                f"available_scanning_time_s must be finite and >= 0; got {available_scanning_time_s!r}."
            )
        return self.synthetic_area_rate_m2_per_s * tau_s

    def evaluate(self, available_scanning_time_s: float) -> ScanningPerformance:
        s_det = self.detectable_area(available_scanning_time_s)
        return ScanningPerformance(
            detectable_area_m2=s_det,
            reference_area_m2=self.synthetic_reference_area_m2,
            gamma=self._metric.normalize(s_det),
            model_identifier=self.model_identifier,
            is_synthetic=self.is_synthetic,
        )
