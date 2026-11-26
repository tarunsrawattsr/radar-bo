"""Tests for ScanningConfig (unresolved preservation) and ResourceConfig (budgets)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.resource_config import ResourceConfig
from radar_bo.config.scanning_config import RESOLVABLE_FIELDS, ScanningConfig


class TestScanningConfig:
    def test_all_paper_parameters_unresolved_by_default(self):
        """Table I lists symbols only; every resolvable field starts as None."""
        s = ScanningConfig()
        for name in RESOLVABLE_FIELDS:
            assert getattr(s, name) is None, f"{name} should be None (unspecified in paper)"

    def test_boltzmann_constant_is_standard_and_frozen(self):
        s = ScanningConfig()
        assert s.boltzmann_constant == 1.380649e-23  # SI 2019 exact
        with pytest.raises(ValidationError):
            s.boltzmann_constant = 2.0

    def test_partially_populated_config_is_valid(self):
        s = ScanningConfig(wavelength=0.03, radar_cross_section=1.0)
        assert s.wavelength == 0.03
        assert s.peak_transmit_power is None  # still unresolved

    def test_negative_power_rejected(self):
        with pytest.raises(ValidationError):
            ScanningConfig(peak_transmit_power=-100.0)

    def test_probability_bounds_strict(self):
        with pytest.raises(ValidationError):
            ScanningConfig(detection_probability=1.5)
        with pytest.raises(ValidationError):
            ScanningConfig(false_alarm_probability=0.0)
        with pytest.raises(ValidationError):
            ScanningConfig(detection_probability=1.0)

    def test_loss_factor_at_least_one(self):
        with pytest.raises(ValidationError):
            ScanningConfig(loss_factor=0.5)
        assert ScanningConfig(loss_factor=1.0).loss_factor == 1.0

    def test_beam_spacing_within_full_circle(self):
        with pytest.raises(ValidationError):
            ScanningConfig(beam_angular_spacing=400.0)
        with pytest.raises(ValidationError):
            ScanningConfig(beam_angular_spacing=0.0)

    def test_reference_area_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScanningConfig(reference_area=0.0)

    def test_ensure_resolved_fails_clearly_with_units(self):
        s = ScanningConfig()
        with pytest.raises(UnresolvedParameterError) as excinfo:
            s.ensure_resolved(["peak_transmit_power", "beam_angular_spacing"])
        msg = str(excinfo.value)
        assert "scanning.peak_transmit_power" in msg
        assert "scanning.beam_angular_spacing" in msg
        assert "W" in msg  # unit-aware error

    def test_ensure_resolved_passes_when_supplied(self):
        s = ScanningConfig(peak_transmit_power=1000.0)
        s.ensure_resolved(["peak_transmit_power"])  # no raise

    def test_ensure_resolved_unknown_field_rejected(self):
        s = ScanningConfig()
        with pytest.raises(AttributeError):
            s.ensure_resolved(["not_a_field"])


class TestResourceConfig:
    def test_paper_action_set(self):
        r = ResourceConfig()
        assert r.dwell_time_action_set == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def test_paper_scalar_defaults(self):
        r = ResourceConfig()
        assert r.normalized_time_budget_threshold == 1.0
        assert r.tracking_scanning_tradeoff == 1.0e6
        assert r.discount_factor == 0.9
        assert r.time_budget_semantics == "discounted_cumulative"

    def test_power_fields_proposed_and_none(self):
        """Proposed power settings have no paper default."""
        r = ResourceConfig()
        for name in ("min_transmit_power", "max_transmit_power", "reference_transmit_power",
                     "scanning_transmit_power", "energy_budget"):
            assert getattr(r, name) is None

    def test_budget_semantics_switch(self):
        alt = ResourceConfig(time_budget_semantics="hard_per_slot")
        assert alt.time_budget_semantics == "hard_per_slot"

    def test_non_increasing_action_set_rejected(self):
        with pytest.raises(ValidationError, match="strictly increasing"):
            ResourceConfig(dwell_time_action_set=[0.1, 0.5, 0.3])

    def test_non_uniform_grid_rejected(self):
        with pytest.raises(ValidationError, match="uniform grid"):
            ResourceConfig(dwell_time_action_set=[0.1, 0.2, 0.5])

    def test_nonpositive_dwell_rejected(self):
        with pytest.raises(ValidationError):
            ResourceConfig(dwell_time_action_set=[0.0, 0.1])

    def test_discount_factor_bounds(self):
        with pytest.raises(ValidationError):
            ResourceConfig(discount_factor=1.5)
        assert ResourceConfig(discount_factor=1.0).discount_factor == 1.0

    def test_inverted_power_bounds_rejected(self):
        with pytest.raises(ValidationError, match="strictly below"):
            ResourceConfig(min_transmit_power=10.0, max_transmit_power=5.0)

    def test_equal_power_bounds_rejected(self):
        with pytest.raises(ValidationError):
            ResourceConfig(min_transmit_power=5.0, max_transmit_power=5.0)
