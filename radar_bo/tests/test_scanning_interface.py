"""Tests for the scanning-performance abstraction (M4 instructions 5-7)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.scanning_config import ScanningConfig
from radar_bo.numerics import NumericalValidationError
from radar_bo.radar.scanning_model import (
    AreaRatioMetric,
    ScanningPerformance,
    ScanningPerformanceModel,
    SyntheticLinearScanningModel,
    area_ratio,
)


class TestEq21AreaRatio:
    def test_hand_computed_ratios(self):
        """Gamma = S_max / S0, manually computed examples."""
        assert area_ratio(4000.0, 4000.0) == pytest.approx(1.0)
        assert area_ratio(2000.0, 4000.0) == pytest.approx(0.5)
        assert area_ratio(10000.0, 4000.0) == pytest.approx(2.5)  # NOT capped at one
        assert area_ratio(0.0, 4000.0) == 0.0

    def test_not_capped_at_one(self):
        """The original paper's area ratio is not automatically capped."""
        assert area_ratio(1.0e6, 100.0) == pytest.approx(1.0e4)

    def test_invalid_inputs_rejected(self):
        with pytest.raises(NumericalValidationError, match="reference_area"):
            area_ratio(100.0, 0.0)
        with pytest.raises(NumericalValidationError, match="reference_area"):
            area_ratio(100.0, -5.0)
        with pytest.raises(NumericalValidationError, match="detectable_area"):
            area_ratio(-1.0, 100.0)
        with pytest.raises(NumericalValidationError, match="non-finite|finite"):
            area_ratio(np.nan, 100.0)

    def test_metric_with_configured_reference(self):
        metric = AreaRatioMetric(reference_area_m2=2500.0)
        assert metric.normalize(1250.0) == pytest.approx(0.5)
        assert metric.normalize(5000.0) == pytest.approx(2.0)  # uncapped

    def test_metric_with_unresolved_reference_raises(self):
        """Paper S0 is None: use raises a parameter-resolution error, no default."""
        metric = AreaRatioMetric(reference_area_m2=None)
        with pytest.raises(UnresolvedParameterError) as excinfo:
            metric.normalize(100.0)
        assert "scanning.reference_area" in str(excinfo.value)

    def test_paper_scanning_config_s0_remains_none(self):
        assert ScanningConfig().reference_area is None

    def test_invalid_configured_reference_rejected(self):
        with pytest.raises(NumericalValidationError):
            AreaRatioMetric(reference_area_m2=0.0)
        with pytest.raises(NumericalValidationError):
            AreaRatioMetric(reference_area_m2=-1.0)


class TestSyntheticScanningModel:
    def test_implements_interface(self):
        model = SyntheticLinearScanningModel()
        assert isinstance(model, ScanningPerformanceModel)
        assert model.is_synthetic is True
        assert "synthetic" in model.model_identifier

    def test_deterministic_and_hand_verifiable(self):
        """S_det = k * tau_s; gamma = S_det / S0 — hand-computed example."""
        model = SyntheticLinearScanningModel(
            synthetic_area_rate_m2_per_s=2000.0, synthetic_reference_area_m2=4000.0
        )
        perf = model.evaluate(1.0)
        assert perf.detectable_area_m2 == pytest.approx(2000.0)
        assert perf.gamma == pytest.approx(0.5)
        assert model.evaluate(1.0).gamma == perf.gamma  # deterministic
        assert model.evaluate(0.0).gamma == 0.0
        assert model.evaluate(2.5).gamma == pytest.approx(2000.0 * 2.5 / 4000.0)

    def test_gamma_uncapped_for_long_times(self):
        model = SyntheticLinearScanningModel()
        assert model.evaluate(10.0).gamma > 1.0  # metric stays uncapped

    def test_invalid_time_rejected(self):
        model = SyntheticLinearScanningModel()
        for bad in (-0.1, np.nan, np.inf):
            with pytest.raises(NumericalValidationError):
                model.evaluate(bad)

    def test_invalid_constants_rejected(self):
        with pytest.raises(NumericalValidationError):
            SyntheticLinearScanningModel(synthetic_area_rate_m2_per_s=0.0)
        with pytest.raises(NumericalValidationError):
            SyntheticLinearScanningModel(synthetic_reference_area_m2=-1.0)

    def test_outputs_finite_across_domain(self):
        model = SyntheticLinearScanningModel()
        for tau in (0.0, 1e-9, 0.1, 1.0, 10.0, 1e6):
            perf = model.evaluate(tau)
            assert np.isfinite(perf.detectable_area_m2)
            assert np.isfinite(perf.gamma)
            assert perf.is_synthetic is True

    def test_result_carries_synthetic_marker(self):
        perf = SyntheticLinearScanningModel().evaluate(1.0)
        assert isinstance(perf, ScanningPerformance)
        assert perf.is_synthetic is True
        assert "synthetic" in perf.model_identifier
