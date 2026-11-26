"""Tests for OptimizationConfig and ExperimentConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radar_bo.config.base import UnresolvedParameterError
from radar_bo.config.experiment_config import ExperimentConfig
from radar_bo.config.optimization_config import OptimizationConfig


class TestOptimizationConfig:
    def test_experimental_defaults(self):
        o = OptimizationConfig()
        assert o.num_initial_observations == 30
        assert o.num_acquisition_candidates == 500
        assert o.surrogate == "single_task_gp"
        assert o.kernel == "matern_5_2"
        assert o.acquisition == "log_expected_improvement"
        assert o.acquisition_xi == 0.01
        assert o.search_mode == "joint"
        assert o.input_normalization is True
        assert o.outcome_standardization is True
        assert o.precision == "float64"
        assert o.device == "cpu"
        assert o.maximize is True

    def test_no_budgets_by_default(self):
        """Budgets have no paper default; they must be set per experiment."""
        o = OptimizationConfig()
        assert o.evaluation_budget is None
        assert o.patience is None
        assert o.runtime_budget_s is None

    def test_ensure_ready_fails_without_budget(self):
        with pytest.raises(UnresolvedParameterError, match="evaluation_budget"):
            OptimizationConfig().ensure_ready()

    def test_ensure_ready_passes_with_budget(self):
        OptimizationConfig(evaluation_budget=50).ensure_ready()  # no raise

    def test_search_mode_switch(self):
        for mode in ("joint", "alternating", "coordinate"):
            assert OptimizationConfig(search_mode=mode).search_mode == mode
        with pytest.raises(ValidationError):
            OptimizationConfig(search_mode="blockwise")

    def test_acquisition_switch(self):
        o = OptimizationConfig(acquisition="q_log_noisy_expected_improvement")
        assert o.acquisition == "q_log_noisy_expected_improvement"

    def test_negative_xi_rejected(self):
        with pytest.raises(ValidationError):
            OptimizationConfig(acquisition_xi=-0.01)


class TestExperimentConfig:
    def test_defaults(self):
        e = ExperimentConfig()
        assert e.experiment_type == "static"
        assert e.repetitions == 5
        assert e.random_seed == 0
        assert e.output_directory == "results"
        assert e.save_configuration is True

    def test_repetitions_positive(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(repetitions=0)

    def test_experiment_type_literal(self):
        for et in ("static", "dynamic", "comparison", "sensitivity"):
            assert ExperimentConfig(experiment_type=et).experiment_type == et
        with pytest.raises(ValidationError):
            ExperimentConfig(experiment_type="field_trial")
