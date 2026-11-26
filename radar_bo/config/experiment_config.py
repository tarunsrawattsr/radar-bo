"""ExperimentConfig: repetition and output bookkeeping (experimental settings).

All values here are EXPERIMENTAL settings with no paper basis; they are
recorded in every saved run configuration together with the full nested
:meth:`SimulationConfig` serialization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from radar_bo.config.base import StrictModel


class ExperimentConfig(StrictModel):
    """Repetition, seeding and output settings for one experiment campaign."""

    experiment_type: Literal["static", "dynamic", "comparison", "sensitivity"] = Field(
        default="static",
        description=(
            "'static': fixed population, frozen scenario; 'dynamic': variable conditions; "
            "'comparison': baseline vs BO under identical seeds; 'sensitivity': parameter sweeps."
        ),
    )
    repetitions: int = Field(default=5, ge=1, description="Independent repetitions per configuration (experimental).")
    random_seed: int = Field(
        default=0,
        description="Master seed; per-repetition seeds are derived deterministically from it (experimental).",
    )
    output_directory: str = Field(
        default="results", description="Directory for per-run configurations, metrics and figures."
    )
    save_configuration: bool = Field(
        default=True, description="Save the exact serialized SimulationConfig with every run."
    )
