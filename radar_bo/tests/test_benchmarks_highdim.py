"""Tests for the higher-dimensional M6 benchmarks and their independent
references (M6 instructions 19-20)."""

from __future__ import annotations

import numpy as np
import pytest

from radar_bo.optimization.bo.benchmarks_highdim import (
    BlockStagnationLP,
    ConstrainedMixedHD,
    CoupledQuadraticHD,
    SeparableQuadraticHD,
)
from radar_bo.optimization.bo.reference_solvers import (
    constrained_mixed_hd_reference,
    coupled_hd_reference,
    separable_hd_reference,
    stagnation_lp_reference,
)


class TestSeparable:
    def test_analytic_optimum(self):
        for d in (4, 8, 16):
            problem = SeparableQuadraticHD(d)
            x_star = problem.centers
            assert problem.objective(x_star) == pytest.approx(10.0 * d)
            assert problem.is_feasible(x_star)
            # Any deviation strictly decreases the objective:
            perturbed = x_star.copy()
            perturbed[0] += 0.1
            assert problem.objective(perturbed) < problem.objective(x_star)

    def test_reference(self):
        for d in (4, 8, 16):
            ref = separable_hd_reference(d)
            assert ref.verification_method == "analytic_exact"
            assert ref.objective_value == pytest.approx(10.0 * d)

    def test_invalid_dimension_rejected(self):
        with pytest.raises(ValueError, match="dimension"):
            SeparableQuadraticHD(5)

    def test_blocks_partition(self):
        problem = SeparableQuadraticHD(8)
        alt = problem.recommended_blocks["alternating"]
        assert sorted(alt[0] + alt[1]) == list(range(8))
        pairs = problem.recommended_blocks["pairwise"]
        assert sorted(i for pair in pairs for i in pair) == list(range(8))


class TestCoupled:
    def test_analytic_optimum_and_pd_matrix(self):
        for d in (4, 8):
            problem = CoupledQuadraticHD(d)
            assert float(np.linalg.eigvalsh(problem.M).min()) > 0.0  # PD
            assert problem.objective(problem.centers) == pytest.approx(20.0)
            assert problem.is_feasible(problem.centers)

    def test_coupling_present(self):
        """The objective genuinely couples variables: moving x1 changes the
        gradient w.r.t. x2 (off-diagonal M entries are nonzero)."""
        problem = CoupledQuadraticHD(8)
        off_diagonal = problem.M - np.diag(np.diag(problem.M))
        assert np.abs(off_diagonal).max() > 0.0

    def test_reference(self):
        ref = coupled_hd_reference(8)
        assert ref.verification_method == "analytic_exact"
        assert ref.objective_value == pytest.approx(20.0)


class TestConstrainedMixed:
    def test_structure_and_constraint(self):
        problem = ConstrainedMixedHD(8)
        assert problem.dimension == 8
        assert problem.variable_types[:2] == ("discrete", "discrete")
        x_tight = np.array([2.0, 2.0] + [4.0] * 6)  # sum 24 + 12 = 36 > 14
        assert not problem.is_feasible(x_tight)
        assert problem.constraint_violation(x_tight) == pytest.approx(22.0)

    def test_inner_optimum_kkt_properties(self):
        centers = np.array([1.2, 1.55, 1.9, 2.25, 2.6, 1.2])
        # Slack budget: unconstrained center point (clamped to bounds).
        x, binding = ConstrainedMixedHD.inner_continuous_optimum(centers, budget_r=100.0)
        assert not binding
        np.testing.assert_allclose(x, np.clip(centers, 0.0, 4.0))
        # Tight budget: sum equals the budget exactly, all values in bounds.
        x2, binding2 = ConstrainedMixedHD.inner_continuous_optimum(centers, budget_r=6.5)
        assert binding2
        assert abs(float(x2.sum()) - 6.5) <= 1e-9
        assert np.all(x2 >= 0.0) and np.all(x2 <= 4.0)
        # KKT form: x_i = clamp(m_i - lambda) with ONE common lambda.
        shifted = centers - x2  # equals lambda where interior
        interior = x2 > 0.0
        assert np.ptp(shifted[interior]) < 1e-9  # uniform shift on the interior

    def test_reference_numerical_approximate(self):
        ref = constrained_mixed_hd_reference(8)
        assert ref.verification_method == "numerical_approximate"
        assert ref.feasible
        # The reference is COMPUTED from the problem, not assumed:
        problem = ConstrainedMixedHD(8)
        assert problem.objective(np.asarray(ref.config)) == pytest.approx(ref.objective_value)
        # Independent sanity: no discrete combination can beat the enumeration.
        best_by_rescan = -np.inf
        for d0 in problem.DISCRETE_D:
            for d1 in problem.DISCRETE_D:
                r = problem.BUDGET_C - 3.0 * (float(d0) + float(d1))
                x, _ = ConstrainedMixedHD.inner_continuous_optimum(problem.cont_centers, r)
                best_by_rescan = max(best_by_rescan, problem.objective(np.concatenate([[d0, d1], x])))
        assert ref.objective_value == pytest.approx(best_by_rescan)

    def test_blocks_partition(self):
        problem = ConstrainedMixedHD(8)
        for grouping in problem.recommended_blocks.values():
            assert sorted(i for group in grouping for i in group) == list(range(8))


class TestBlockStagnationLP:
    def test_analytic_reference(self):
        ref = stagnation_lp_reference()
        assert ref.verification_method == "analytic_exact"
        assert ref.objective_value == pytest.approx(9.6)
        assert ref.config == [1.6, 0.0, 1.6, 0.0]

    def test_no_single_block_improvement_at_tight_config(self):
        """From the balanced budget-tight configuration (0.8, 0.8, 0.8, 0.8),
        NO feasible single-block change improves the objective — the formal
        core of the stagnation demonstration."""
        problem = BlockStagnationLP()
        tight = np.array([0.8, 0.8, 0.8, 0.8])
        assert problem.is_feasible(tight)
        f_current = problem.objective(tight)
        block_a = [0, 2]  # a-group
        block_b = [1, 3]  # b-group
        rng = np.random.default_rng(0)
        for indices in (block_a, block_b):
            for _ in range(4000):
                candidate = tight.copy()
                values = rng.uniform(0.0, problem.UPPER, size=len(indices))
                candidate[indices] = values
                if problem.is_feasible(candidate):
                    assert problem.objective(candidate) <= f_current + 1e-12

    def test_joint_improvement_exists(self):
        """The joint optimum beats the tight configuration — stagnation is a
        block-restriction artifact, not global optimality."""
        problem = BlockStagnationLP()
        assert problem.objective(np.array([1.6, 0.0, 1.6, 0.0])) > problem.objective(np.array([0.8, 0.8, 0.8, 0.8]))
