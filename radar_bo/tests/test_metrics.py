"""M7 item 30: metric unit tests with INDEPENDENTLY calculated expected values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from radar_bo.experiments.framework.metrics import (
    bootstrap_ci,
    describe,
    evaluations_to_level,
    gap_closure_threshold_value,
    holm_correction,
    normalized_gap_closure,
    objective_gap,
    paired_differences,
    paired_effect_size,
    runtime_decomposition,
    threshold_analysis,
    wilcoxon_signed_rank,
)


class TestObjectiveGap:
    def test_maximization_simple_regret(self):
        # independent calculation: reference 10.0 - best 9.4 = 0.6
        result = objective_gap(9.4, 10.0, maximize=True, reference_is_exact=True)
        assert result["value"] == pytest.approx(0.6)
        assert result["label"] == "simple_regret"

    def test_minimization_reversed(self):
        # minimization: best 0.3 - reference 0.2 = 0.1
        result = objective_gap(0.3, 0.2, maximize=False, reference_is_exact=True)
        assert result["value"] == pytest.approx(0.1)
        assert result["label"] == "simple_regret"

    def test_numerical_reference_labeled_reference_gap(self):
        result = objective_gap(39.4605, 39.53, maximize=True, reference_is_exact=False)
        assert result["label"] == "reference_gap"
        assert result["value"] == pytest.approx(0.0695)

    def test_negative_gap_retained_never_clipped(self):
        result = objective_gap(40.1, 40.0, maximize=True, reference_is_exact=False)
        assert result["value"] == pytest.approx(-0.1)  # NOT clipped to zero
        assert result["exceeds_reference"] is True


class TestNormalizedGapClosure:
    def test_standard_maximization(self):
        # initial 4.0, reference 10.0, best 7.0 -> (7-4)/(10-4) = 0.5
        result = normalized_gap_closure(7.0, 4.0, 10.0, maximize=True)
        assert result["value"] == pytest.approx(0.5)
        assert result["reason"] is None

    def test_minimization(self):
        # initial 1.0, reference 0.2, best 0.5 -> (1.0-0.5)/(1.0-0.2) = 0.625
        result = normalized_gap_closure(0.5, 1.0, 0.2, maximize=False)
        assert result["value"] == pytest.approx(0.625)

    def test_zero_denominator_explicit(self):
        result = normalized_gap_closure(7.0, 10.0, 10.0, maximize=True)
        assert result["value"] is None
        assert result["reason"] == "degenerate_gap: initial incumbent equals the reference; closure undefined"

    def test_near_zero_denominator_explicit(self):
        result = normalized_gap_closure(7.0, 10.0 - 1e-15, 10.0, maximize=True)
        assert result["value"] is None

    def test_negative_denominator_initial_better_than_reference(self):
        result = normalized_gap_closure(7.0, 11.0, 10.0, maximize=True)
        assert result["value"] is None
        assert "negative_gap" in result["reason"]

    def test_closure_above_one_when_reference_exceeded(self):
        result = normalized_gap_closure(10.5, 4.0, 10.0, maximize=True)
        # independent calculation: (10.5 - 4.0) / (10.0 - 4.0) = 6.5 / 6 = 1.08333...
        assert result["value"] == pytest.approx(6.5 / 6.0)


class TestThresholds:
    def test_threshold_level_independent_calculation(self):
        # initial 4.0, reference 10.0, fraction 0.9 -> level 4 + 0.9*6 = 9.4
        assert gap_closure_threshold_value(4.0, 10.0, 0.9) == pytest.approx(9.4)
        assert gap_closure_threshold_value(4.0, 10.0, 0.5) == pytest.approx(7.0)
        assert gap_closure_threshold_value(4.0, 10.0, 0.99) == pytest.approx(9.94)

    def test_evaluations_to_level(self):
        history = [4.0, 6.0, 9.5, 9.9]
        assert evaluations_to_level(history, 9.4) == 3  # first index reaching 9.4
        assert evaluations_to_level(history, 99.0) is None  # unreached -> explicit None

    def test_initial_design_already_meets_threshold(self):
        # initial incumbent 9.5 EXCEEDS the (approximate) reference 9.0, so the
        # 0.9-closure level is 9.5 + 0.9*(9.0 - 9.5) = 9.05 and the initial
        # design already meets it (exceeded-reference edge case):
        rows = threshold_analysis([9.5, 9.9], 9.5, 9.0, [0.9])
        assert rows[0]["reached"] is True
        assert rows[0]["successful_evaluations_required"] == 1
        assert rows[0].get("initial_design_already_meets_threshold") is True

    def test_unreached_threshold_not_artificial(self):
        rows = threshold_analysis([4.0, 5.0], 4.0, 10.0, [0.9])
        assert rows[0]["reached"] is False
        assert rows[0]["successful_evaluations_required"] is None
        assert rows[0]["attempted_evaluations_required"] is None

    def test_attempted_and_wall_axes(self):
        history = [4.0, 9.5]
        diagnostics_axis = [31, 33]  # attempted evaluations per improving eval
        wall = [12.5, 20.0]
        rows = threshold_analysis(history, 4.0, 10.0, [0.9],
                                  wall_timestamps=wall, attempted_axis=diagnostics_axis)
        assert rows[0]["successful_evaluations_required"] == 2
        assert rows[0]["attempted_evaluations_required"] == 33
        assert rows[0]["wall_clock_s_required"] == pytest.approx(20.0)


class TestDescriptiveStatistics:
    def test_independently_calculated(self):
        values = [2.0, 4.0, 4.0, 10.0]
        stats = describe(values)
        assert stats["n"] == 4
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(4.0)
        # sample std (ddof=1): sqrt(((2-5)^2+(4-5)^2+(4-5)^2+(10-5)^2)/3) = sqrt(36/3)
        assert stats["stdev"] == pytest.approx(math.sqrt(12.0))
        assert stats["min"] == 2.0
        assert stats["max"] == 10.0
        assert stats["iqr"] == pytest.approx(np.percentile([2.0, 4.0, 4.0, 10.0], 75)
                                             - np.percentile([2.0, 4.0, 4.0, 10.0], 25))

    def test_empty_input(self):
        stats = describe([])
        assert stats["n"] == 0


class TestPairedComparisons:
    def test_paired_diffs_wins_losses_ties(self):
        a = {0: 10.0, 1: 9.0, 2: 8.0, 3: 7.0}
        b = {0: 9.0, 1: 9.0, 2: 8.0 + 1e-10, 3: 6.0}
        paired = paired_differences(a, b, tie_tolerance=1e-9)
        # independently calculated diffs: [1.0, 0.0, -1e-10, 1.0]
        assert paired["n_pairs"] == 4
        assert paired["wins"] == 2  # seeds 0 and 3
        assert paired["ties"] == 2  # seed 1 (exact 0) and seed 2 (1e-10 < 1e-9 tolerance)
        assert paired["losses"] == 0
        assert paired["wins"] + paired["losses"] + paired["ties"] == 4

    def test_missing_seeds_reported(self):
        paired = paired_differences({0: 1.0, 1: 2.0}, {0: 0.5})
        assert paired["missing_seeds"] == [1]
        assert paired["n_pairs"] == 1

    def test_tie_tolerance_documented(self):
        paired = paired_differences({0: 1.0}, {0: 1.0 + 1e-12}, tie_tolerance=1e-9)
        assert paired["ties"] == 1

    def test_effect_size_dz(self):
        diffs = np.asarray([1.0, -1.0, 1.0, -1.0])
        assert paired_effect_size(diffs) == 0.0  # mean 0
        diffs2 = np.asarray([2.0, 2.0, 2.0, 2.0])
        assert paired_effect_size(diffs2) is None  # zero std -> undefined

    def test_bootstrap_reproducible_with_documented_seed(self):
        rng = np.random.default_rng(7)
        diffs = rng.normal(size=10)
        ci1 = bootstrap_ci(diffs, replications=2000, seed=2026)
        ci2 = bootstrap_ci(diffs, replications=2000, seed=2026)
        assert ci1 == ci2
        ci3 = bootstrap_ci(diffs, replications=2000, seed=2027)
        assert (ci1["lower"], ci1["upper"]) != (ci3["lower"], ci3["upper"]) or ci1["bootstrap_seed"] != ci3["bootstrap_seed"]

    def test_bootstrap_resamples_seed_level_pairs(self):
        """Pairs are resampled as units: a dataset of identical pairs yields a
        degenerate but CORRECT interval at the pair mean."""
        diffs = np.asarray([5.0] * 10)
        ci = bootstrap_ci(diffs, replications=1000, seed=1)
        assert ci["lower"] == pytest.approx(5.0)
        assert ci["upper"] == pytest.approx(5.0)


class TestWilcoxonAndCorrection:
    def test_wilcoxon_basic(self):
        diffs = np.asarray([1.0, 2.0, 0.5, 1.5, -0.2, 0.9, 2.2, 1.1, 0.3, 1.8])
        result = wilcoxon_signed_rank(diffs)
        assert result["valid"] is True
        assert result["n_used"] == 10
        assert 0.0 <= result["p_value"] <= 1.0
        # symmetric-ish positive shifts should be significant for this sample
        assert result["p_value"] < 0.05

    def test_zero_differences_dropped_and_counted(self):
        diffs = np.asarray([1.0, 0.0, 2.0, 0.0, 1.5])
        result = wilcoxon_signed_rank(diffs)
        assert result["n_zero_differences"] == 2
        assert result["n_used"] == 3

    def test_all_zero_differences(self):
        result = wilcoxon_signed_rank(np.zeros(5))
        assert result["valid"] is False

    def test_documentation_present(self):
        result = wilcoxon_signed_rank(np.asarray([1.0, 2.0]))
        assert result["documentation"]["null_hypothesis"].startswith("the median")
        assert "DROPPED" in result["documentation"]["zero_difference_handling"]

    def test_holm_correction_independently_calculated(self):
        # classic example: p = [0.01, 0.04, 0.03]
        # sorted: 0.01 -> *3 = 0.03; 0.03 -> *2 = 0.06 -> running max 0.06;
        # 0.04 -> *1 = 0.04 -> running max 0.06
        adjusted = holm_correction([0.01, 0.04, 0.03])
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[1] == pytest.approx(0.06)
        assert adjusted[2] == pytest.approx(0.06)

    def test_holm_bounds_at_one(self):
        adjusted = holm_correction([0.9, 0.95])
        assert all(p <= 1.0 for p in adjusted)


class TestRuntimeDecomposition:
    def test_unattributed_computed_never_assumed(self):
        record = {
            "runtime_measurements": {
                "gp_fit_runtime_s": 10.0, "acquisition_runtime_s": 1.0,
                "objective_runtime_s": 0.5, "candidate_generation_runtime_s": 20.0,
                "constraint_validation_runtime_s": 5.0, "wall_runtime_s": 50.0,
            }
        }
        decomp = runtime_decomposition(record)
        assert decomp["component_sum_s"] == pytest.approx(36.5)
        assert decomp["unattributed_overhead_s"] == pytest.approx(13.5)

    def test_documentation_fields(self):
        decomp = runtime_decomposition({"runtime_measurements": {"wall_runtime_s": 1.0}})
        assert "GP" in decomp["timer_boundaries"]["gp_fit_runtime_s"]
        assert "NOT" in decomp["double_counting_note"] or "not" in decomp["double_counting_note"].lower()


class TestConvergenceAxes:
    def test_attempted_axis_exact_for_m7_records(self):
        record = {
            "incumbent_history": [4.0, 7.0, 9.0],
            "iteration_diagnostics": [
                {"objective_value": 6.0, "attempted_evaluations": 31, "elapsed_runtime_s": 5.0, "incumbent_objective": 7.0, "cycle_number": 1},
                {"objective_value": 9.0, "attempted_evaluations": 33, "elapsed_runtime_s": 9.0, "incumbent_objective": 9.0, "cycle_number": 2},
            ],
        }
        from radar_bo.experiments.framework.metrics import convergence_curves

        curves = convergence_curves(record)
        assert curves["attempted_evaluations"]["x"] == [31, 33]
        assert curves["attempted_evaluations"]["y"] == [7.0, 9.0]
        assert curves["wall_clock"]["x"] == [5.0, 9.0]
        assert curves["block_cycles"]["cycle_numbers_of_improvements"] == [1, 2]

    def test_pre_m7_records_report_unavailable_axis(self):
        record = {
            "incumbent_history": [4.0, 7.0],
            "iteration_diagnostics": [],
        }
        from radar_bo.experiments.framework.metrics import convergence_curves

        curves = convergence_curves(record)
        assert curves["attempted_evaluations"]["x"] is None
        assert "not archived" in curves["attempted_evaluations"]["note"]
