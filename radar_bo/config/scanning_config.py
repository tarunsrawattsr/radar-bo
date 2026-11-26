"""ScanningConfig: UCA scanning radar interface (Lu et al., Eqs. (16)-(21), Table I).

Every scanning parameter below is UNSPECIFIED in the paper: Table I lists
symbols only, with no numeric values. Each is therefore preserved as ``None``
with a provenance record (README non-negotiable: never invent missing paper
parameters). Values may be supplied later, individually, when an experiment
requires them and a justification exists; partially populated configurations
are valid.

The paper assumes full 360-degree azimuth coverage and a monostatic radar.
Boltzmann's constant is a standard physical constant (SI 2019 exact
definition), not a fitted parameter, so it is populated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pydantic import Field, field_validator

from radar_bo.config.base import StrictModel, UnresolvedParameterError

#: Human-readable units for every resolvable (possibly-None) scanning field.
#: Used to produce clear, unit-aware "unresolved parameter" errors.
_UNITS: Final[dict[str, str]] = {
    "peak_transmit_power": "W (peak pulse power, not average power)",
    "transmit_antenna_gain": "linear gain (dimensionless, not dBi)",
    "receive_antenna_gain": "linear gain (dimensionless, not dBi)",
    "wavelength": "m",
    "radar_cross_section": "m^2 (target RCS, paper symbol sigma)",
    "loss_factor": "linear loss (>= 1, not dB)",
    "system_temperature": "K (paper symbol Ts)",
    "beam_angular_spacing": "deg (phi in Eq. (16): tau_s = (360 deg / phi) * tau_beam)",
    "reference_area": "m^2 (paper symbol S0 in Eq. (21))",
    "detection_probability": "probability in (0, 1) (paper symbol Pd)",
    "false_alarm_probability": "probability in (0, 1) (paper symbol Pf)",
    "required_detection_snr": "linear SNR (Shnidman-equation requirement in Eq. (21))",
}

#: Field names that an experiment may require from ScanningConfig.
RESOLVABLE_FIELDS: Final[tuple[str, ...]] = tuple(_UNITS)

_POSITIVE_FIELDS: Final[tuple[str, ...]] = (
    "peak_transmit_power",
    "transmit_antenna_gain",
    "receive_antenna_gain",
    "wavelength",
    "radar_cross_section",
    "system_temperature",
    "reference_area",
    "required_detection_snr",
)

_PROBABILITY_FIELDS: Final[tuple[str, ...]] = ("detection_probability", "false_alarm_probability")


class ScanningConfig(StrictModel):
    """Partially populated scanning configuration; unresolved fields stay ``None``."""

    # ---- Unspecified in the paper (Table I symbols without values) ----------
    peak_transmit_power: float | None = Field(
        default=None, description="Peak transmit power; distinct from average transmitted power."
    )
    transmit_antenna_gain: float | None = Field(default=None, description="Transmit antenna gain, linear.")
    receive_antenna_gain: float | None = Field(default=None, description="Receive antenna gain, linear (monostatic).")
    wavelength: float | None = Field(default=None, description="Carrier wavelength.")
    radar_cross_section: float | None = Field(default=None, description="Target radar cross section (paper symbol sigma).")
    loss_factor: float | None = Field(default=None, description="System loss factor, linear, >= 1.")
    system_temperature: float | None = Field(default=None, description="System noise temperature Ts.")
    beam_angular_spacing: float | None = Field(
        default=None,
        description=(
            "phi in Eq. (16). The paper calls it the 'phase delay between adjacent "
            "radar beams'; it is treated here as the angular spacing in degrees "
            "(so the number of beams is 360/phi). Flagged in the provenance "
            "register as an interpretational assumption."
        ),
    )
    reference_area: float | None = Field(default=None, description="S0 in Eq. (21), nonzero reference area.")
    detection_probability: float | None = Field(default=None, description="Pd, strict (0, 1).")
    false_alarm_probability: float | None = Field(default=None, description="Pf, strict (0, 1).")
    required_detection_snr: float | None = Field(
        default=None,
        description=(
            "Required detection SNR (linear) per the Shnidman equation referenced "
            "in Eq. (21). The paper does not give the Shnidman implementation "
            "settings (number of pulses, integration mode, etc.), so this stays "
            "unresolved until independently justified."
        ),
    )

    # ---- Standard physical constant (SI 2019 exact, not a fitted value) -----
    boltzmann_constant: float = Field(
        default=1.380649e-23,
        frozen=True,
        description="Boltzmann constant k in J/K; standard constant (SI 2019 exact definition).",
    )

    # ---- Validators ----------------------------------------------------------
    @field_validator(*_POSITIVE_FIELDS)
    @classmethod
    def _positive_when_supplied(cls, v: float | None, info) -> float | None:
        if v is not None and v <= 0.0:
            raise ValueError(f"{info.field_name} must be positive when supplied; got {v}.")
        return v

    @field_validator("loss_factor")
    @classmethod
    def _loss_at_least_one(cls, v: float | None) -> float | None:
        if v is not None and v < 1.0:
            raise ValueError(f"loss_factor must be >= 1 (linear); got {v}.")
        return v

    @field_validator(*_PROBABILITY_FIELDS)
    @classmethod
    def _strict_probability(cls, v: float | None, info) -> float | None:
        if v is not None and not (0.0 < v < 1.0):
            raise ValueError(f"{info.field_name} must lie strictly in (0, 1); got {v}.")
        return v

    @field_validator("beam_angular_spacing")
    @classmethod
    def _spacing_within_circle(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 < v <= 360.0):
            raise ValueError(f"beam_angular_spacing must lie in (0, 360] degrees; got {v}.")
        return v

    # ---- Resolution helpers ---------------------------------------------------
    def ensure_resolved(self, names: Iterable[str]) -> None:
        """Fail clearly if any requested scanning parameter is still ``None``.

        Experiments that need scanning physics must call this with the exact
        fields they use, so missing paper parameters surface as a single,
        unit-aware error instead of silent fabricated values.
        """
        missing: dict[str, str] = {}
        for name in names:
            if name not in _UNITS:
                raise AttributeError(f"Unknown ScanningConfig field: {name!r}. Resolvable fields: {RESOLVABLE_FIELDS}.")
            if getattr(self, name) is None:
                missing[f"scanning.{name}"] = _UNITS[name]
        if missing:
            raise UnresolvedParameterError(
                missing,
                guidance=(
                    "These scanning parameters are unspecified in Lu et al. (Table I lists "
                    "symbols only). Supply justified values in ScanningConfig, or extend the "
                    "provenance register with a documented proposed assumption."
                ),
            )
