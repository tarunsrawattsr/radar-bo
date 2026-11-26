"""Tests for the synthetic benchmarks and their independent verification (M5 instruction 5)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.benchmarks import (
    BenchmarkAQuadratic,
    BenchmarkBMixedConstrained,
    BenchmarkCMultimodal,
    all_benchmarks,
)


class TestBenchmarkA:
    def test_analytic_optimum_exact(self):
        """f(3.5, 7.2) = 10 exactly (independent derivation: completion of square)."""
        p = BenchmarkAQuadratic()
        assert p.evaluate([3.5, 7.2]) == pytest.approx(10.0, abs=1e-12)
        assert p.analytic_optimum == (3.5, 7.2)
        assert p.reference_method.startswith("analytic")

    def test_off_optimum_lower_and_quadratic_decay(self):
        p = BenchmarkAQuadratic()
        # Independent expected value: c - 1^2 - 0^2 = 9 at (4.5, 7.2)
        assert p.evaluate([4.5, 7.2]) == pytest.approx(9.0, abs=1e-12)
        assert p.evaluate([4.5, 7.2]) < p.evaluate([3.5, 7.2])

    def test_synthetic_labeling(self):
        p = BenchmarkAQuadratic()
        assert p.problem_id.startswith("synthetic_")
        assert p.observation_noise_std is None  # deterministic


class TestBenchmarkB:
    def test_enumeration_with_analytic_inner_optimum(self):
        """Independent verification: per-d analytic clamped inner optimum."""
        p = BenchmarkBMixedConstrained()
        expected = {0.5: 3.0, 1.0: 6.0, 1.5: 9.0, 2.0: 11.0}
        best_value = -np.inf
        for d in p.discrete_values[0]:
            d_star, x_star = p.analytic_inner_optimum(float(d))
            value = p.evaluate([d_star, x_star])
            assert value == pytest.approx(expected[float(d)], abs=1e-12)
            best_value = max(best_value, value)
        assert best_value == pytest.approx(11.0)

    def test_constraint_binds_at_optimum(self):
        p = BenchmarkBMixedConstrained()
        d, x = p.analytic_inner_optimum(2.0)
        assert (d, x) == (2.0, 2.0)
        # Constraint x + 3d = 8 exactly at the optimum (binding):
        assert x + 3.0 * d == pytest.approx(8.0, abs=1e-12)
        assert p.constraint_violation([d, x]) <= 1e-12

    def test_group_interaction(self):
        """The continuous optimum moves with the discrete value (x* = d + 1)."""
        p = BenchmarkBMixedConstrained()
        for d in p.discrete_values[0]:
            _, x_star = p.analytic_inner_optimum(float(d))
            expected_x = min(float(d) + 1.0, 8.0 - 3.0 * float(d))
            assert x_star == pytest.approx(expected_x)

    def test_infeasible_high_interaction_point(self):
        p = BenchmarkBMixedConstrained()
        # (d=2, x=3): unconstrained inner optimum, but 3 + 6 = 9 > 8:
        assert not p.is_feasible([2.0, 3.0])
        assert p.constraint_violation([2.0, 3.0]) == pytest.approx(1.0)

    def test_synthetic_labeling(self):
        assert BenchmarkBMixedConstrained().problem_id.startswith("synthetic_")


class TestBenchmarkC:
    def test_multimodality_grid_evidence(self):
        """Multiple separated local optima exist (grid scan finds >= 4 peaks)."""
        p = BenchmarkCMultimodal()
        g = np.linspace(-2.5, 2.5, 401)
        xx1, xx2 = np.meshgrid(g, g)
        vals = np.asarray(
            [[p.objective(np.array([a, b])) for a in g] for b in g]
        )
        # Count local maxima on the grid (8-neighborhood):
        peaks = 0
        for i in range(1, vals.shape[0] - 1):
            for j in range(1, vals.shape[1] - 1):
                window = vals[i - 1 : i + 2, j - 1 : j + 2].copy()
                center = window[1, 1]
                window[1, 1] = -np.inf
                if center > window.max():
                    peaks += 1
        assert peaks >= 4  # multiple local optima

    def test_reference_is_numerical_approximate(self):
        p = BenchmarkCMultimodal()
        assert p.reference_objective is None  # computed by the solver, never assumed
        assert "APPROXIMATE" in p.reference_method

    def test_finite_on_domain(self):
        p = BenchmarkCMultimodal()
        rng = np.random.default_rng(0)
        for _ in range(200):
            x = rng.uniform(-2.5, 2.5, size=2)
            assert np.isfinite(p.objective(x))

    def test_synthetic_labeling(self):
        assert BenchmarkCMultimodal().problem_id.startswith("synthetic_")


class TestBenchmarkRegistry:
    def test_three_benchmarks_available(self):
        registry = all_benchmarks()
        assert set(registry) == {"A", "B", "C"}
        assert registry["A"].dimension == 2
        assert registry["B"].variable_types == ("discrete", "continuous")
