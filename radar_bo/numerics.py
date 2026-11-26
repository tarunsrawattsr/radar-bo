"""Shared numerical validation helpers for the estimation layer.

Conventions (README section 1/2): SI units, radians, linear quantities,
float64 precision. This module is imported by the environment and radar
layers only; it must never import the optimizer (layer separation, guarded
by radar_bo/tests/test_layer_separation.py).
"""

from __future__ import annotations

import numpy as np

#: Project-wide floating-point dtype for state and covariance arrays.
FLOAT_DTYPE = np.float64


class NumericalValidationError(ValueError):
    """Raised when numerical inputs violate shape, finiteness or covariance contracts.

    Deliberately raised instead of silently repairing inputs: covariance
    eigenvalues are never clipped to conceal defects (M3 instruction 8).
    """


def as_vector(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    """Coerce to a finite float64 vector of exact length ``dim``."""
    arr = np.asarray(value, dtype=FLOAT_DTYPE)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise NumericalValidationError(f"{name} must be a vector of length {dim}; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalValidationError(f"{name} contains non-finite values: {arr!r}.")
    return arr


def as_matrix(value: np.ndarray, rows: int, cols: int, name: str) -> np.ndarray:
    """Coerce to a finite float64 matrix of exact shape ``(rows, cols)``."""
    arr = np.asarray(value, dtype=FLOAT_DTYPE)
    if arr.shape != (rows, cols):
        raise NumericalValidationError(f"{name} must have shape ({rows}, {cols}); got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalValidationError(f"{name} contains non-finite values.")
    return arr


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return ``(M + M.T) / 2``.

    Documented numerical hygiene against floating-point drift only; PSD
    violations are still detected and raised afterwards, never clipped.
    """
    return 0.5 * (matrix + matrix.T)


def validate_covariance(
    matrix: np.ndarray,
    dim: int,
    name: str,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    psd_tolerance: float = 1e-10,
) -> np.ndarray:
    """Validate a symmetric positive-semidefinite float64 covariance matrix.

    Returns the coerced matrix. Raises :class:`NumericalValidationError` on
    wrong shape, non-finite entries, asymmetry or a negative eigenvalue beyond
    a scale-relative tolerance. Eigenvalues are never modified.
    """
    arr = as_matrix(matrix, dim, dim, name)
    if not np.allclose(arr, arr.T, rtol=rtol, atol=atol):
        max_asym = float(np.abs(arr - arr.T).max())
        raise NumericalValidationError(
            f"{name} must be symmetric; max |M - M.T| = {max_asym:.3e}."
        )
    eigvals = np.linalg.eigvalsh(symmetrize(arr))
    scale = max(1.0, float(np.abs(eigvals).max()))
    min_eig = float(eigvals.min())
    if min_eig < -psd_tolerance * scale:
        raise NumericalValidationError(
            f"{name} must be positive semidefinite; min eigenvalue {min_eig:.6e} "
            f"(scale-relative tolerance {-psd_tolerance * scale:.3e}). Eigenvalues are never clipped."
        )
    return arr


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to the interval [-pi, pi).

    Angular convention for azimuths and angular residuals (M3 instruction 6:
    quadrant-aware angular representation). Note the half-open interval:
    ``wrap_to_pi(pi) == -pi``.
    """
    value = np.asarray(angle, dtype=FLOAT_DTYPE)
    if not np.all(np.isfinite(value)):
        raise NumericalValidationError(f"Cannot wrap non-finite angle(s): {value!r}.")
    wrapped = (value + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped
