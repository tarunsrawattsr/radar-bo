"""Tests for the GP surrogate and acquisition manager (M5 instructions 12-20)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from radar_bo.optimization.bo.acquisition import (
    AcquisitionError,
    AcquisitionManager,
    SelectionFailureError,
    verify_installed_log_ei_signature,
)
from radar_bo.optimization.bo.surrogate import GPSurrogate, GPSurrogateError


def quadratic_data(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-D quadratic observations: y = 10 - (x1-3.5)^2 - (x2-7.2)^2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, size=(n, 2))
    y = (10.0 - (x[:, 0] - 3.5) ** 2 - (x[:, 1] - 7.2) ** 2).reshape(-1, 1)
    return x, y


@pytest.fixture(scope="module")
def fitted_surrogate():
    x, y = quadratic_data()
    return GPSurrogate(dimension=2, seed=0).fit(x, y)


class TestSurrogateConfiguration:
    def test_kernel_is_explicit_matern_5_2(self, fitted_surrogate):
        """The installed BoTorch default is RBF; the kernel must be explicitly
        Matern-5/2 (instruction 14) — verified on the instantiated model."""
        from gpytorch.kernels import MaternKernel

        kernel = fitted_surrogate.model.covar_module
        assert isinstance(kernel, MaternKernel)
        assert float(kernel.nu) == 2.5

    def test_default_kernel_would_have_been_rbf(self):
        """Documents WHY explicit configuration is required (instructed check)."""
        from botorch.models import SingleTaskGP

        x, y = quadratic_data(n=6)
        default_model = SingleTaskGP(
            torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        )
        assert type(default_model.covar_module).__name__ == "RBFKernel"

    def test_input_and_outcome_transforms(self, fitted_surrogate):
        from botorch.models.transforms import Normalize, Standardize

        assert isinstance(fitted_surrogate.model.input_transform, Normalize)
        assert isinstance(fitted_surrogate.model.outcome_transform, Standardize)

    def test_float64_cpu(self, fitted_surrogate):
        p = next(fitted_surrogate.model.parameters())
        assert p.dtype == torch.float64
        assert p.device == torch.device("cpu")

    def test_installed_log_ei_signature_has_no_xi(self):
        """Instruction 18: verify the installed API before passing parameters."""
        params = verify_installed_log_ei_signature()
        assert "xi" not in params
        assert "best_f" in params


class TestSurrogatePosterior:
    def test_output_dimensions(self, fitted_surrogate):
        x = np.array([[3.5, 7.2], [0.0, 0.0], [10.0, 10.0]])
        mean, variance = fitted_surrogate.posterior_mean_variance(x)
        assert mean.shape == (3,)
        assert variance.shape == (3,)

    def test_finite_means_and_nonnegative_variance(self, fitted_surrogate):
        rng = np.random.default_rng(2)
        x = rng.uniform(-10, 10, size=(25, 2))
        mean, variance = fitted_surrogate.posterior_mean_variance(x)
        assert np.all(np.isfinite(mean))
        assert np.all(variance >= 0.0)

    def test_latent_variance_distinct_from_observation_noise(self, fitted_surrogate):
        noise_variance = fitted_surrogate.observation_noise_variance()
        assert noise_variance > 0.0  # inferred noise is NOT numerically zero (documented)
        _, variance = fitted_surrogate.posterior_mean_variance(np.array([[0.0, 0.0]]))
        # The latent variance is a separate quantity (may be smaller or larger):
        assert variance[0] != noise_variance

    def test_original_unit_predictions(self, fitted_surrogate):
        """Posterior means are in ORIGINAL units (un-standardized): near the
        analytic peak they are close to the observed range of the data."""
        x = np.array([[3.5, 7.2]])
        mean, _ = fitted_surrogate.posterior_mean_variance(x)
        assert 0.0 <= float(mean[0]) <= 12.0  # original units, not z-scores

    def test_predictions_at_training_points_reasonable(self, fitted_surrogate):
        """Diagnostic (NOT an exact-interpolation requirement, instruction 17):
        predictions at observed points track the observations within the
        data range."""
        x, y = quadratic_data(20)
        mean, _ = fitted_surrogate.posterior_mean_variance(x)
        y_range = float(y.max() - y.min())
        residuals = np.abs(mean - y.flatten())
        assert float(residuals.max()) < 0.5 * y_range

    def test_shape_and_finiteness_violations_rejected(self, fitted_surrogate):
        with pytest.raises(ValueError, match="shape"):
            fitted_surrogate.posterior_mean_variance(np.zeros((3, 5)))
        with pytest.raises(ValueError, match="non-finite"):
            fitted_surrogate.posterior_mean_variance(np.array([[np.nan, 0.0]]))
        with pytest.raises(GPSurrogateError, match="fitted"):
            GPSurrogate(2).posterior_mean_variance(np.zeros((1, 2)))


class TestSurrogateFitting:
    def test_fit_diagnostics_recorded(self):
        x, y = quadratic_data(15)
        surrogate = GPSurrogate(2, seed=1).fit(x, y)
        d = surrogate.diagnostics
        assert d.dataset_size == 15
        assert d.dimension == 2
        assert d.fitting_duration_s >= 0.0
        assert d.fitting_status == "converged"
        assert d.kernel_identifier == "matern_5_2"
        assert d.observation_noise_variance is not None and d.observation_noise_variance > 0
        assert len(d.lengthscale_original_space) == 2
        assert any("Matern-5/2" in n for n in d.notes)

    def test_fit_failure_raises_with_details(self, monkeypatch):
        """A marginal-likelihood fitting crash is converted into GPSurrogateError
        with details preserved and a 'failed' diagnostics status (instruction 16)."""
        import botorch

        x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        y = np.array([[1.0], [2.0], [3.0]])
        surrogate = GPSurrogate(2, seed=0)

        def _crash(_mll):
            raise RuntimeError("simulated optimizer crash")

        monkeypatch.setattr(botorch, "fit_gpytorch_mll", _crash)
        with pytest.raises(GPSurrogateError, match="GP fitting failed"):
            surrogate.fit(x, y)
        assert surrogate.diagnostics.fitting_status.startswith("failed")
        assert not surrogate.fitted

    def test_nonfinite_training_data_rejected(self):
        x = np.array([[0.0, 0.0], [np.nan, 1.0]])
        y = np.array([[1.0], [2.0]])
        with pytest.raises(GPSurrogateError, match="non-finite"):
            GPSurrogate(2).fit(x, y)

    def test_too_few_observations_rejected(self):
        x = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])
        with pytest.raises(GPSurrogateError, match="at least 2"):
            GPSurrogate(2).fit(x, y)

    def test_reproducible_construction_and_fitting(self):
        """Same seed + same data -> identical posteriors (controlled settings)."""
        x, y = quadratic_data(15, seed=5)
        s1 = GPSurrogate(2, seed=3).fit(x, y)
        s2 = GPSurrogate(2, seed=3).fit(x, y)
        probe = np.array([[2.0, 6.0], [-5.0, 1.0]])
        m1, v1 = s1.posterior_mean_variance(probe)
        m2, v2 = s2.posterior_mean_variance(probe)
        np.testing.assert_allclose(m1, m2, rtol=1e-10)
        np.testing.assert_allclose(v1, v2, rtol=1e-10)
        assert s1.observation_noise_variance() == pytest.approx(s2.observation_noise_variance())

    def test_refit_updates_diagnostics(self):
        x, y = quadratic_data(10)
        s = GPSurrogate(2, seed=0).fit(x, y)
        assert s.diagnostics.dataset_size == 10
        x2 = np.vstack([x, [[3.5, 7.2]]])
        y2 = np.vstack([y, [[10.0]]])
        s.fit(x2, y2)
        assert s.diagnostics.dataset_size == 11


class TestAcquisitionManager:
    def test_build_requires_fitted_surrogate(self):
        with pytest.raises(AcquisitionError, match="fitted"):
            AcquisitionManager(GPSurrogate(2))

    def test_build_with_incumbent_reference(self, fitted_surrogate):
        """best_f must be the incumbent in ORIGINAL units."""
        manager = AcquisitionManager(fitted_surrogate, maximize=True)
        acquisition = manager.build(9.0)
        assert float(acquisition.best_f) == 9.0

    def test_score_shapes_and_correspondence(self, fitted_surrogate):
        manager = AcquisitionManager(fitted_surrogate)
        acquisition = manager.build(9.0)
        candidates = np.array([[3.5, 7.2], [0.0, 0.0], [-10.0, -10.0]])
        scores, mask = manager.score_candidates(acquisition, candidates)
        assert scores.shape == (3,)
        assert mask.all()
        # Ordering sanity: near-peak candidates score higher than far corners
        # (log-EI is monotone in EI, so ranking matches EI ranking):
        assert scores[0] > scores[2]

    def test_scores_are_log_domain(self, fitted_surrogate):
        """Verify scores equal log(EI) computed INDEPENDENTLY from the posterior
        mean/variance with scipy's normal CDF/PDF (instruction 20; no naive
        EI-value interpretation)."""
        from scipy.stats import norm

        manager = AcquisitionManager(fitted_surrogate)
        best_f = 9.0
        acquisition = manager.build(best_f)
        candidates = np.array([[3.0, 7.0], [-2.0, 4.0]])
        scores, _ = manager.score_candidates(acquisition, candidates)
        mean, variance = fitted_surrogate.posterior_mean_variance(candidates)
        sigma = np.sqrt(variance)
        z = (mean - best_f) / sigma
        ei_manual = (mean - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
        np.testing.assert_allclose(scores, np.log(ei_manual), rtol=1e-6, atol=1e-10)

    def test_selection_tie_break_first_index(self):
        scores = np.array([1.0, 2.0, 2.0, 1.0])
        mask = np.ones(4, dtype=bool)
        assert AcquisitionManager.select_best(scores, mask) == 1  # first maximal index

    def test_nonfinite_scores_masked(self):
        scores = np.array([np.nan, 0.5, -np.inf, 0.2])
        mask = np.array([False, True, False, True])
        assert AcquisitionManager.select_best(scores, mask) == 1

    def test_all_nonfinite_raises_selection_failure(self):
        with pytest.raises(SelectionFailureError):
            AcquisitionManager.select_best(np.array([np.nan, np.nan]), np.array([False, False]))

    def test_invalid_inputs_rejected(self, fitted_surrogate):
        manager = AcquisitionManager(fitted_surrogate)
        acquisition = manager.build(9.0)
        with pytest.raises(AcquisitionError, match="shape"):
            manager.score_candidates(acquisition, np.zeros((3, 7)))
        with pytest.raises(AcquisitionError, match="empty"):
            manager.score_candidates(acquisition, np.zeros((0, 2)))
        with pytest.raises(AcquisitionError, match="non-finite"):
            manager.score_candidates(acquisition, np.array([[np.nan, 0.0]]))
        with pytest.raises(AcquisitionError, match="finite"):
            manager.build(np.nan)
