"""Tests for the tracking-cost function (Lu et al. Eqs. (14)-(15))."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.numerics import NumericalValidationError
from radar_bo.radar.tracking_cost import E_POSITION_SELECTOR, position_tracking_cost


class TestEquation14:
    def test_identity_covariance(self):
        """trace(E I E^T) = 2 for P0 = I_4."""
        assert position_tracking_cost(np.eye(4)) == pytest.approx(2.0)

    def test_equals_sum_of_position_variances(self):
        """c_t = P[0,0] + P[1,1] (paper's interpretation of Eq. (14))."""
        p = np.diag([4.0, 9.0, 100.0, 25.0])
        assert position_tracking_cost(p) == pytest.approx(13.0)

    def test_selector_matrix_matches_eq15(self):
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(E_POSITION_SELECTOR, expected)
        # Manual trace(E P E^T) vs the implemented shortcut:
        p = np.array(
            [
                [2.0, 0.5, 0.1, 0.0],
                [0.5, 3.0, 0.0, 0.2],
                [0.1, 0.0, 7.0, 0.0],
                [0.0, 0.2, 0.0, 8.0],
            ]
        )
        manual = float(np.trace(E_POSITION_SELECTOR @ p @ E_POSITION_SELECTOR.T))
        assert position_tracking_cost(p) == pytest.approx(manual)
        assert position_tracking_cost(p) == pytest.approx(2.0 + 3.0)

    def test_velocity_block_does_not_enter_cost(self):
        p_a = np.diag([1.0, 1.0, 10.0, 10.0])
        p_b = np.diag([1.0, 1.0, 500.0, 500.0])
        assert position_tracking_cost(p_a) == pytest.approx(position_tracking_cost(p_b))


class TestRejections:
    def test_incompatible_dimensions_rejected(self):
        with pytest.raises(NumericalValidationError):
            position_tracking_cost(np.eye(3))
        with pytest.raises(NumericalValidationError):
            position_tracking_cost(np.eye(2))

    def test_non_finite_result_rejected(self):
        with pytest.raises(NumericalValidationError, match="non-finite"):
            position_tracking_cost(np.full((4, 4), np.nan))

    def test_asymmetric_rejected(self):
        bad = np.eye(4)
        bad[0, 1] = 0.5
        with pytest.raises(NumericalValidationError, match="symmetric"):
            position_tracking_cost(bad)

    def test_negative_definite_rejected(self):
        with pytest.raises(NumericalValidationError, match="positive semidefinite"):
            position_tracking_cost(-np.eye(4))


class TestIndependence:
    def test_module_has_no_optimizer_or_scanning_imports(self):
        """Eq. (14) stays independent of optimization/scanning/weighting logic."""
        import ast
        from pathlib import Path

        source = Path(position_tracking_cost.__module__.replace(".", "/") + ".py")
        module_path = Path(__file__).resolve().parents[1] / "radar" / "tracking_cost.py"
        assert source.name == module_path.name  # located where expected
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert not imported & {"torch", "botorch", "gpytorch", "radar_bo.optimization"}
