"""Tests for the planar range/azimuth measurement model (Lu et al. Eqs. (4), (7))."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.environment.target_state import TargetState
from radar_bo.radar.measurement_model import MeasurementGeometryError, PlanarRangeAzimuthModel


@pytest.fixture
def model():
    return PlanarRangeAzimuthModel()


class TestMeasurementFunction:
    def test_on_positive_x_axis(self, model):
        np.testing.assert_allclose(model.measure([3000.0, 0.0]), [3000.0, 0.0], atol=1e-12)

    def test_quadrant_awareness(self, model):
        """atan2 evaluation distinguishes all four quadrants (vs printed tan^-1(y/x))."""
        cases = {
            (3000.0, 0.0): 0.0,
            (0.0, 3000.0): np.pi / 2,
            (-3000.0, 0.0): np.pi,  # wrapped to [-pi, pi) by Observation, but measure() returns atan2
            (0.0, -3000.0): -np.pi / 2,
            (1500.0, 1500.0): np.pi / 4,
            (-1500.0, 1500.0): 3.0 * np.pi / 4,
            (-1500.0, -1500.0): -3.0 * np.pi / 4,
            (1500.0, -1500.0): -np.pi / 4,
        }
        for (x, y), expected_azimuth in cases.items():
            z = model.measure([x, y])
            assert z[0] == pytest.approx(np.hypot(x, y))
            assert z[1] == pytest.approx(expected_azimuth, abs=1e-12)

    def test_measure_from_state_ignores_velocity(self, model):
        a = model.measure_from_state(TargetState(vector=[1000.0, 500.0, 50.0, -50.0]))
        b = model.measure_from_state(TargetState(vector=[1000.0, 500.0, 0.0, 0.0]))
        np.testing.assert_array_equal(a, b)

    def test_non_finite_position_rejected(self, model):
        with pytest.raises(Exception, match="non-finite|finite"):
            model.measure([np.nan, 1000.0])


class TestSingularGeometry:
    def test_origin_raises_on_measure(self, model):
        """No epsilon substitution: the origin raises the domain-specific error."""
        with pytest.raises(MeasurementGeometryError, match="origin"):
            model.measure([0.0, 0.0])

    def test_origin_raises_on_jacobian(self, model):
        with pytest.raises(MeasurementGeometryError, match="origin"):
            model.jacobian([0.0, 0.0])

    def test_error_is_domain_specific_not_value_error(self, model):
        with pytest.raises(MeasurementGeometryError) as excinfo:
            model.measure([0.0, 0.0])
        assert not isinstance(excinfo.value, ValueError)


class TestJacobian:
    @pytest.mark.parametrize(
        "position",
        [[1500.0, 800.0], [-2200.0, 600.0], [300.0, -900.0], [-100.0, -2500.0], [4000.0, 0.0], [0.001, 4000.0]],
    )
    def test_analytical_matches_numerical_differentiation(self, model, position):
        """Analytical Eq. (7) Jacobian validated against central differences."""
        analytic = model.jacobian(position)
        numeric = model.numerical_jacobian(position, step=1e-5)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_jacobian_shape_and_velocity_columns(self, model):
        h = model.jacobian([1000.0, 2000.0])
        assert h.shape == (2, 4)
        np.testing.assert_array_equal(h[:, 2:], np.zeros((2, 2)))  # velocity unobserved

    def test_jacobian_row1_matches_closed_form(self, model):
        x, y = 1200.0, -500.0
        r = np.hypot(x, y)
        h = model.jacobian([x, y])
        np.testing.assert_allclose(h[0], [x / r, y / r, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(h[1], [-y / r**2, x / r**2, 0.0, 0.0], atol=1e-15)
