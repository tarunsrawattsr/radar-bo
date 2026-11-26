"""Tests for the master SimulationConfig: nesting, cross-validation, serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from radar_bo.config import (
    EnvironmentConfig,
    ExperimentConfig,
    OptimizationConfig,
    ResourceConfig,
    ScanningConfig,
    SimulationConfig,
    TrackingConfig,
)


class TestNesting:
    def test_all_sections_composed(self):
        cfg = SimulationConfig()
        assert isinstance(cfg.tracking, TrackingConfig)
        assert isinstance(cfg.scanning, ScanningConfig)
        assert isinstance(cfg.resource, ResourceConfig)
        assert isinstance(cfg.environment, EnvironmentConfig)
        assert isinstance(cfg.optimization, OptimizationConfig)
        assert isinstance(cfg.experiment, ExperimentConfig)

    def test_revisit_interval_single_source(self):
        """revisit_interval is exposed from TrackingConfig only (README 4.6)."""
        cfg = SimulationConfig()
        assert cfg.revisit_interval == cfg.tracking.revisit_interval == 2.5

    def test_extra_section_forbidden(self):
        with pytest.raises(ValidationError):
            SimulationConfig(nonexistent_section=1)


class TestCrossValidation:
    def test_dwell_actions_within_revisit(self):
        SimulationConfig()  # max dwell 1.0 s <= T0 2.5 s: valid

    def test_revisit_below_max_dwell_rejected(self):
        with pytest.raises(ValidationError, match="exceed"):
            SimulationConfig(tracking=TrackingConfig(revisit_interval=0.5))

    def test_reference_dwell_must_be_in_action_set(self):
        with pytest.raises(ValidationError, match="action_set"):
            SimulationConfig(tracking=TrackingConfig(reference_dwell_time=0.55))

    def test_mismatched_transmit_power_rejected(self):
        with pytest.raises(ValidationError, match="reference_transmit_power"):
            SimulationConfig(
                tracking=TrackingConfig(reference_transmit_power=100.0),
                resource=ResourceConfig(reference_transmit_power=200.0),
            )

    def test_matching_transmit_power_accepted(self):
        cfg = SimulationConfig(
            tracking=TrackingConfig(reference_transmit_power=100.0),
            resource=ResourceConfig(reference_transmit_power=100.0),
        )
        assert cfg.tracking.reference_transmit_power == 100.0


class TestEnvironmentPopulation:
    def test_initial_conditions_length_must_match(self):
        with pytest.raises(ValidationError, match="num_targets"):
            SimulationConfig(
                environment=EnvironmentConfig(
                    num_targets=2,
                    initial_conditions=[(100.0, 0.0, 5.0, 0.0)],
                )
            )

    def test_matching_population_accepted(self):
        cfg = SimulationConfig(
            environment=EnvironmentConfig(
                num_targets=2,
                initial_conditions=[(100.0, 0.0, 5.0, 0.0), (-500.0, 200.0, -3.0, 1.0)],
            )
        )
        assert len(cfg.environment.initial_conditions) == 2


class TestSerialization:
    def test_json_round_trip_preserves_everything(self):
        cfg = SimulationConfig(
            tracking=TrackingConfig(reference_snr=None),
            environment=EnvironmentConfig(num_targets=2),
            optimization=OptimizationConfig(evaluation_budget=42),
        )
        restored = SimulationConfig.model_validate_json(cfg.to_json_string())
        assert restored == cfg

    def test_saved_json_is_plain_json(self):
        cfg = SimulationConfig()
        parsed = json.loads(cfg.to_json_string())
        assert parsed["tracking"]["reference_range_variance"] == 100.0
        assert parsed["scanning"]["peak_transmit_power"] is None

    def test_save_and_load_file(self, tmp_path):
        cfg = SimulationConfig()
        path = cfg.save_json(tmp_path / "run1" / "config.json")
        assert path.exists()
        assert SimulationConfig.load_json(path) == cfg

    def test_load_revalidates(self, tmp_path):
        """A hand-edited config with an invalid value must fail on load."""
        cfg = SimulationConfig()
        path = cfg.save_json(tmp_path / "config.json")
        text = path.read_text().replace('"reference_range_variance": 100.0', '"reference_range_variance": -3.0')
        path.write_text(text)
        with pytest.raises(ValidationError):
            SimulationConfig.load_json(path)
