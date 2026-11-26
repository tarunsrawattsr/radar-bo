"""OptimizationConfig: generic constrained-BO harness settings.

Methodological sources:

* M. Benzaghta et al., "Designing Cellular Networks for UAV Corridors via
  Bayesian Optimization", IEEE GLOBECOM 2023,
  DOI 10.1109/GLOBECOM54140.2023.10437768 — GP surrogate with Matern-5/2
  kernel, expected improvement with xi = 0.01, 500 acquisition candidates.
  Our harness is a METHODOLOGICAL ADAPTATION, not an exact reproduction of
  that paper's algorithm (README research foundations).
* Fields marked (experimental) are proposed settings with no paper basis;
  they must be labelled as experimental in any experiment output.

The optimizer layer is generic: this config must never mention radar
quantities (README layer separation).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from radar_bo.config.base import StrictModel, UnresolvedParameterError

SearchMode = Literal[
    # All decision variables updated jointly each BO iteration.
    "joint",
    # Block-wise updates: one decision block per iteration, others fixed
    # (methodological analogue of the Benzaghta per-BS scheme).
    "alternating",
    # One coordinate per iteration (experimental).
    "coordinate",
]


class OptimizationConfig(StrictModel):
    """Settings for the generic BO harness (surrogate, acquisition, budget)."""

    # ---- Surrogate and acquisition ------------------------------------------
    surrogate: Literal["single_task_gp"] = Field(
        default="single_task_gp", description="BoTorch SingleTaskGP surrogate (float64)."
    )
    kernel: Literal["matern_5_2"] = Field(
        default="matern_5_2", description="Explicit Matern-5/2 kernel, following Benzaghta et al."
    )
    acquisition: Literal["log_expected_improvement", "q_log_noisy_expected_improvement"] = Field(
        default="log_expected_improvement",
        description=(
            "LogExpectedImprovement for the low-noise synthetic baseline; "
            "qLogNoisyExpectedImprovement only when a noisy objective is justified. "
            "M5: the installed LogExpectedImprovement accepts NO exploration-offset "
            "parameter, so Benzaghta's xi = 0.01 is preserved only in the provenance "
            "register (conventional LogEI without an offset is a methodological "
            "adaptation); no no-effect configuration parameter is retained."
        ),
    )
    search_mode: SearchMode = Field(
        default="joint",
        description=(
            "'joint' is the default harness mode; 'alternating' mirrors the "
            "block-wise per-BS scheme of Benzaghta et al.; 'coordinate' is "
            "experimental."
        ),
    )

    # ---- Normalization and precision ----------------------------------------
    input_normalization: bool = Field(
        default=True, description="Normalize inputs to the unit cube using consistent physical bounds."
    )
    outcome_standardization: bool = Field(default=True, description="Standardize outcomes before GP fitting.")
    precision: Literal["float64"] = Field(
        default="float64", description="Double precision for GP tensors (README section 2)."
    )
    device: Literal["cpu"] = Field(default="cpu", description="CPU execution first (README section 2).")
    maximize: bool = Field(
        default=True, description="The harness maximizes the objective (Benzaghta et al. maximize mean SINR)."
    )

    # ---- Budgets: experimental, no paper default -----------------------------
    num_initial_observations: int = Field(
        default=30, ge=1, description="Number of initial (space-filling) observations (experimental)."
    )
    num_acquisition_candidates: int = Field(
        default=500, ge=1, description="Acquisition-optimization candidate pool size (Benzaghta et al. use 500)."
    )
    evaluation_budget: int | None = Field(
        default=None, ge=1, description="Total number of objective evaluations per optimization run; no paper default. Must be set before an experiment runs."
    )
    max_iterations: int | None = Field(
        default=None, ge=1, description="Optional maximum number of BO iterations (experimental; distinct from the evaluation budget)."
    )
    refit_per_observation: bool = Field(
        default=True,
        description=(
            "M5 initial GP-fitting policy (experimental): a fresh refit after each successful "
            "new observation. Warm-starting or incremental updates would be a different, "
            "explicitly identified and tested strategy."
        ),
    )
    patience: int | None = Field(
        default=None, ge=1, description="Optional early-stop patience in iterations without improvement (experimental)."
    )
    improvement_tolerance: float | None = Field(
        default=None, gt=0.0, description="Optional minimum improvement considered progress (experimental)."
    )
    runtime_budget_s: float | None = Field(
        default=None, gt=0.0, description="Optional wall-clock runtime budget in seconds (experimental)."
    )
    random_seed: int | None = Field(
        default=None, description="Seed for candidate sampling and acquisition optimization; falls back to the experiment seed."
    )

    # ---- Validators ----------------------------------------------------------
    @model_validator(mode="after")
    def _noisy_acquisition_needs_justification_note(self) -> "OptimizationConfig":
        # Documented, not enforced: choosing qLogNoisyExpectedImprovement is
        # allowed but the choice must be justified in the experiment record
        # (README BOConfig: 'if justified').
        return self

    def ensure_ready(self) -> None:
        """Fail clearly if the harness is run without the required budgets.

        ``evaluation_budget`` has no paper default, so it stays ``None`` in the
        default config and must be set by the experiment configuration.
        """
        missing: dict[str, str] = {}
        if self.evaluation_budget is None:
            missing["optimization.evaluation_budget"] = "count of objective evaluations"
        if missing:
            raise UnresolvedParameterError(
                missing,
                guidance=(
                    "No paper default exists for the evaluation budget (README BOConfig). "
                    "Set it explicitly in the experiment configuration."
                ),
            )
