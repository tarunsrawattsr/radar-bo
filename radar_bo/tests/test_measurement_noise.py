"""Tests for the SNR-dependent measurement-noise model (Lu et al. Eqs. (5), (6), (8))."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.tracking_config import TrackingConfig
from radar_bo.radar.measurement_noise import PlanarSNRMeasurementNoiseModel


SYNTHETIC_SNR0 = 100.0  # explicitly synthetic testing value (NOT a paper value)


def synthetic_model() -> PlanarSNRMeasurementNoiseModel:
    """Model with a synthetic reference SNR, clearly labeled for testing only."""
    return PlanarSNRMeasurementNoiseModel(
        reference_range_variance=100.0,
        reference_angle_variance=4.0e-4,
        reference_distance=3000.0,
        reference_dwell_time=1.0,
        reference_snr=SYNTHETIC_SNR0,
    )


class TestSNRModel:
    def test_eq5_formula(self):
        """SNR_t = SNR0 * (tau/tau0) * (r/r0)^-4, independent recomputation."""
        model = synthetic_model()
        tau, r = 1.0, 3000.0
        expected = 100.0 * (tau / 1.0) * (3000.0 / r) ** -4
        assert model.snr(tau, r) == pytest.approx(expected)

    def test_snr_increases_with_dwell_and_proximity(self):
        model = synthetic_model()
        base = model.snr(1.0, 3000.0)
        assert model.snr(2.0, 3000.0) > base  # longer dwell
        assert model.snr(1.0, 1500.0) > base  # closer target

    def test_invalid_dwell_or_range_rejected(self):
        model = synthetic_model()
        for tau in (0.0, -1.0, np.nan):
            with pytest.raises(Exception, match="dwell_time"):
                model.snr(tau, 3000.0)
        for r in (0.0, -1.0, np.nan):
            with pytest.raises(Exception, match="range_m"):
                model.snr(1.0, r)


class TestEq6DivisionSemantics:
    def test_variance_decreases_with_snr(self):
        """Consolidated Eq. (6): sigma^2 = sigma^2_0 / SNR (M3 review 3.2)."""
        model = synthetic_model()
        r_low = model.measurement_covariance_from_snr(50.0).matrix
        r_high = model.measurement_covariance_from_snr(200.0).matrix
        assert r_high[0, 0] < r_low[0, 0]
        assert r_high[1, 1] < r_low[1, 1]

    def test_covariance_values(self):
        model = synthetic_model()
        r = model.measurement_covariance_from_snr(100.0).matrix
        assert r[0, 0] == pytest.approx(100.0 / 100.0)  # sigma^2_r0 / SNR
        assert r[1, 1] == pytest.approx(4.0e-4 / 100.0)
        assert r[0, 1] == 0.0 and r[1, 0] == 0.0  # Eq. (8) diagonal

    def test_combined_observation_covariance(self):
        model = synthetic_model()
        r = model.observation_covariance(dwell_time=1.0, range_m=3000.0).matrix
        expected_snr = 100.0
        assert r[0, 0] == pytest.approx(100.0 / expected_snr)
        assert r[1, 1] == pytest.approx(4.0e-4 / expected_snr)

    def test_invalid_snr_rejected(self):
        model = synthetic_model()
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(Exception, match="snr_value"):
                model.measurement_covariance_from_snr(bad)


class TestUnresolvedReferenceSNR:
    def test_paper_default_config_has_no_snr0(self):
        assert TrackingConfig().reference_snr is None

    def test_unresolved_snr_fails_clearly(self):
        """Operations requiring SNR0 raise a clear parameter-resolution error."""
        model = PlanarSNRMeasurementNoiseModel.from_config(TrackingConfig())
        assert model.reference_snr is None
        with pytest.raises(UnresolvedParameterError) as excinfo:
            model.snr(1.0, 3000.0)
        assert "tracking.reference_snr" in str(excinfo.value)
        with pytest.raises(UnresolvedParameterError):
            model.observation_covariance(1.0, 3000.0)

    def test_from_config_copies_paper_values(self):
        model = PlanarSNRMeasurementNoiseModel.from_config(TrackingConfig())
        assert model.reference_range_variance == 100.0
        assert model.reference_angle_variance == 4.0e-4
        assert model.reference_distance == 3000.0
        assert model.reference_dwell_time == 1.0

    def test_invalid_supplied_snr_rejected(self):
        for bad in (0.0, -5.0, np.nan):
            with pytest.raises(Exception, match="reference_snr"):
                PlanarSNRMeasurementNoiseModel(100.0, 4e-4, 3000.0, 1.0, bad)
