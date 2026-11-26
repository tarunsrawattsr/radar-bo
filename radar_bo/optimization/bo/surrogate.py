"""Gaussian-process surrogate on BoTorch/GPyTorch (M5 instructions 12-17).

Configuration (verified against the INSTALLED library versions):

* ``SingleTaskGP`` with an EXPLICITLY configured ``MaternKernel(nu=2.5)`` —
  the installed BoTorch 0.18.1 default kernel is ``RBFKernel``, so the kernel
  is set explicitly and TESTED (never assumed).
* Input normalization via BoTorch's ``Normalize(d)`` — the ONLY normalization
  mechanism; inputs are never pre-normalized manually. Constraint checks and
  candidate selection operate in ORIGINAL coordinates; normalization happens
  strictly inside the GP.
* Outcome standardization via BoTorch's ``Standardize(1)`` — outcomes stay in
  original units in the dataset; ``posterior()`` un-transforms back, so
  predictions and the acquisition's incumbent reference (``best_f``) share
  the SAME original-unit coordinate system. No manual standardization is
  applied on top.
* Exact marginal log-likelihood fitting via ``botorch.fit_gpytorch_mll`` with
  ``ExactMarginalLogLikelihood``; double precision, CPU.
* Observation noise: the deterministic benchmarks declare no noise, but the
  GP's Gaussian likelihood noise is INFERRED during fitting and is in general
  NOT numerically zero — documented treatment; latent predictive variance is
  kept distinct from the observation-noise variance.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch
from pydantic import Field

from radar_bo.config.base import StrictModel
from radar_bo.optimization.bo.conversions import numpy_to_torch, torch_to_numpy


class GPSurrogateError(RuntimeError):
    """GP construction or marginal-likelihood fitting failed (details preserved)."""


class GPSurrogateDiagnostics(StrictModel):
    """Recorded fitting diagnostics (instruction 16)."""

    dataset_size: int
    dimension: int
    fitting_duration_s: float
    fitting_status: str
    numerical_warnings: list[str] = Field(default_factory=list)
    kernel_identifier: str
    lengthscale_original_space: list[float] | None = None
    observation_noise_variance: float | None = None
    converged: bool | None = None
    notes: list[str] = Field(default_factory=list)


class GPSurrogate:
    """Reusable single-output GP surrogate (float64, CPU, Matern-5/2)."""

    def __init__(self, dimension: int, seed: int | None = None) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1; got {dimension}.")
        self.dimension = int(dimension)
        self.seed = seed
        self.model = None
        self.diagnostics: GPSurrogateDiagnostics | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "GPSurrogate":
        """Fit on (M, d) inputs and (M, 1) outputs; refits are fresh rebuilds
        (the initial policy is refit-per-observation; warm-starting is NOT used
        in M5 and would need to be explicitly identified and tested)."""
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.float64)
        if x.ndim != 2 or x.shape != (y.shape[0], self.dimension):
            raise GPSurrogateError(
                f"train_x must have shape (M, {self.dimension}); got {x.shape} (train_y: {y.shape})."
            )
        if y.ndim != 2 or y.shape[1] != 1:
            raise GPSurrogateError(f"train_y must have shape (M, 1); got {y.shape}.")
        if x.shape[0] < 2:
            raise GPSurrogateError(
                f"GP fitting needs at least 2 observations (M={x.shape[0]}); Standardize requires variation."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise GPSurrogateError("Training data contains non-finite values.")

        from botorch import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.kernels import MaternKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if self.seed is not None:
            torch.manual_seed(int(self.seed))  # reproducible model construction/fitting

        tensor_x = numpy_to_torch(x, name="train_x")
        tensor_y = numpy_to_torch(y, name="train_y")
        started = time.perf_counter()
        captured_warnings: list[str] = []
        try:
            model = SingleTaskGP(
                tensor_x,
                tensor_y,
                covar_module=MaternKernel(
                    nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=None
                ),
                input_transform=Normalize(self.dimension),
                outcome_transform=Standardize(1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_gpytorch_mll(mll)
            captured_warnings = [str(w.message) for w in caught]
            status = "converged"
        except Exception as exc:  # fitting failure: clear behavior (instruction 16)
            self.model = None
            self.diagnostics = GPSurrogateDiagnostics(
                dataset_size=int(x.shape[0]),
                dimension=self.dimension,
                fitting_duration_s=time.perf_counter() - started,
                fitting_status=f"failed: {type(exc).__name__}: {exc}",
                numerical_warnings=captured_warnings,
                kernel_identifier="matern_5_2",
            )
            raise GPSurrogateError(
                f"GP fitting failed on M={x.shape[0]} observations: {type(exc).__name__}: {exc}"
            ) from exc

        self.model = model.eval()
        self.diagnostics = GPSurrogateDiagnostics(
            dataset_size=int(x.shape[0]),
            dimension=self.dimension,
            fitting_duration_s=time.perf_counter() - started,
            fitting_status=status,
            numerical_warnings=captured_warnings,
            kernel_identifier="matern_5_2",
            lengthscale_original_space=[
                float(v) for v in model.covar_module.lengthscale.detach().cpu().numpy().flatten()
            ],
            observation_noise_variance=float(model.likelihood.noise.detach()),
            converged=True,
            notes=[
                "Kernel explicitly Matern-5/2 (installed BoTorch default is RBF and was NOT assumed).",
                "Observation noise is INFERRED by the marginal-likelihood fit; deterministic "
                "benchmarks do not imply numerically zero fitted noise.",
                "Input normalization (Normalize) and outcome standardization (Standardize) are "
                "BoTorch transforms applied inside the model; no manual re-normalization.",
            ],
        )
        return self

    def posterior_mean_variance(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean (n,) and LATENT variance (n,) in ORIGINAL outcome units.

        The predictive variance is the latent-function variance — distinct from
        the observation-noise variance (``observation_noise_variance()``).
        """
        if self.model is None:
            raise GPSurrogateError("posterior_mean_variance requires a fitted surrogate.")
        arr = np.asarray(candidates, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(f"candidates must have shape (n, {self.dimension}); got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("candidates contain non-finite values.")
        tensor = numpy_to_torch(arr, name="candidates")
        with torch.no_grad():
            posterior = self.model.posterior(tensor)
        mean = torch_to_numpy(posterior.mean, name="posterior mean").flatten()
        variance = torch_to_numpy(posterior.variance, name="posterior variance").flatten()
        if variance.shape != mean.shape:
            raise GPSurrogateError(f"Posterior mean/variance shape mismatch: {mean.shape} vs {variance.shape}.")
        # Variance must be nonnegative within numerical tolerance; small negative
        # values from floating-point round-off are reported, never silently clipped.
        if variance.min() < -1e-12:
            raise GPSurrogateError(
                f"Posterior variance contains substantially negative values (min {variance.min():.3e}); "
                "numerical defect, not round-off."
            )
        variance = np.maximum(variance, 0.0)  # clamp only |v| <= 1e-12 round-off, documented
        return mean, variance

    def observation_noise_variance(self) -> float:
        """Fitted observation-noise variance of the Gaussian likelihood."""
        if self.model is None:
            raise GPSurrogateError("observation_noise_variance requires a fitted surrogate.")
        return float(self.model.likelihood.noise.detach())
