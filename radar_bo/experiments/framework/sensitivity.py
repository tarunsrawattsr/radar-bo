"""Sensitivity-analysis infrastructure (M7 item 25).

Explicit, named scenarios that override specific fields of a base
ExperimentConfig. NO unrestricted Cartesian product: the scenario list IS the
experiment matrix, every changed field is recorded, everything else stays
fixed, and the convention is pilot-first (small seeds before any larger
sweep).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.experiments.framework.experiment_config import ExperimentConfig


class SensitivityScenario(StrictModel):
    scenario_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    description: str
    overrides: dict[str, Any] = Field(
        description="Dotted-path overrides, e.g. {'initial_design.size': 10, 'stopping.evaluation_budget': 40}.",
    )


def apply_scenario(base: ExperimentConfig, scenario: SensitivityScenario) -> ExperimentConfig:
    """Apply dotted-path overrides to a deep copy of the base configuration."""
    data = base.model_dump()
    for dotted, value in scenario.overrides.items():
        keys = dotted.split(".")
        node = data
        for key in keys[:-1]:
            if key not in node:
                raise ValueError(f"Unknown override path '{dotted}' (segment '{key}' missing).")
            node = node[key]
        if keys[-1] not in node:
            raise ValueError(f"Unknown override path '{dotted}' (final segment missing).")
        node[keys[-1]] = value
    candidate = ExperimentConfig.model_validate(data)
    if candidate.fingerprint() == base.fingerprint():
        raise ValueError(f"Scenario '{scenario.scenario_id}' does not change the configuration.")
    return candidate


def scenario_diff(base: ExperimentConfig, scenario_config: ExperimentConfig) -> dict[str, dict]:
    """Changed vs fixed fields (recorded per scenario; item 25)."""
    base_data = base.model_dump()
    scenario_data = scenario_config.model_dump()
    changed: dict[str, dict] = {}
    for key in base_data:
        if base_data[key] != scenario_data[key]:
            changed[key] = {"base": _summarize(base_data[key]), "scenario": _summarize(scenario_data[key])}
    return changed


def _summarize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _summarize(v) for k, v in value.items() if not isinstance(v, (dict, list))}
    if isinstance(value, list):
        return value
    return value


def build_sensitivity_experiment(
    base: ExperimentConfig,
    scenarios: list[SensitivityScenario],
) -> list[tuple[SensitivityScenario, ExperimentConfig, dict]]:
    """The explicit sensitivity matrix: (scenario, config, changed-fields)."""
    out = []
    for scenario in scenarios:
        config = apply_scenario(base, scenario)
        config = config.model_copy(update={
            "experiment_id": f"{base.experiment_id}-{scenario.scenario_id}",
            "description": f"{base.description} [sensitivity scenario: {scenario.description}]",
            "output_dir": f"{base.output_dir}-{scenario.scenario_id}",
        })
        out.append((scenario, config, scenario_diff(base, config)))
    return out
