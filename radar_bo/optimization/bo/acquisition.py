"""Acquisition-function manager (M5 instructions 18 and 20).

Uses BoTorch's conventional ``LogExpectedImprovement`` for deterministic
single-candidate maximization. The installed signature (verified against
BoTorch 0.18.1 at runtime and in tests) is
``LogExpectedImprovement(model, best_f, posterior_transform=None, maximize=True)``:
it accepts NO exploration-offset parameter, so Benzaghta's reported
``xi = 0.01`` cannot be applied here — the conventional LogEI WITHOUT an
additional improvement offset is classified as a METHODOLOGICAL ADAPTATION
(recorded in the provenance register). No no-effect configuration parameter
is retained.

Scores are LOG-domain expected improvements: ``max log(EI) == max EI`` (the
logarithm is monotone), but a log score must never be reported as an ordinary
EI value. Non-finite scores are masked out explicitly; selection is the
argmax over finite scores with deterministic first-index tie-breaking.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from radar_bo.optimization.bo.conversions import numpy_to_torch, torch_to_numpy
from radar_bo.optimization.bo.surrogate import GPSurrogate, GPSurrogateError


class AcquisitionError(RuntimeError):
    """Acquisition construction or scoring failed."""


class SelectionFailureError(RuntimeError):
    """No candidate received a finite acquisition score."""


def verify_installed_log_ei_signature() -> dict[str, str]:
    """Return the installed LogExpectedImprovement parameter names/types.

    Verifies at runtime that no exploration-offset (``xi``) parameter exists;
    the result is recorded by tests (instruction 18: verify the installed API
    before passing acquisition parameters).
    """
    from botorch.acquisition import LogExpectedImprovement

    params = dict(inspect.signature(LogExpectedImprovement.__init__).parameters)
    if "xi" in params:
        raise AcquisitionError(
            "The installed LogExpectedImprovement accepts an 'xi' parameter; the M5 design "
            "assumption (no offset) no longer matches the installed API. Re-verify before use."
        )
    return {name: (p.default if p.default is not inspect._empty else "<required>") for name, p in params.items()}


class AcquisitionManager:
    """Builds LogEI against the fitted surrogate and scores candidate pools."""

    def __init__(self, surrogate: GPSurrogate, maximize: bool = True) -> None:
        if not surrogate.fitted:
            raise AcquisitionError("AcquisitionManager requires a fitted surrogate.")
        self.surrogate = surrogate
        self.maximize = maximize

    def build(self, incumbent_value: float) -> "object":
        """Build LogExpectedImprovement with ``best_f`` = the maximum successfully
        OBSERVED objective value (original outcome units — the same coordinate
        system as the un-standardized GP posterior)."""
        from botorch.acquisition import LogExpectedImprovement

        verify_installed_log_ei_signature()
        if not np.isfinite(incumbent_value):
            raise AcquisitionError(f"Incumbent must be finite; got {incumbent_value!r}.")
        return LogExpectedImprovement(self.surrogate.model, best_f=float(incumbent_value), maximize=self.maximize)

    def score_candidates(self, acquisition, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score a (k, d) candidate pool; returns ``(scores, finite_mask)``.

        The installed ``LogExpectedImprovement`` is a single-candidate (q = 1)
        acquisition: it REJECTS q > 1 inputs (verified at runtime: "Expected X
        to be ``batch_shape x q=1 x d``", shape (k, d) is parsed as q = k).
        The pool is therefore submitted as a BATCH of single candidates with
        shape ``(k, 1, d)`` — batch k of q=1 evaluations — which the installed
        version accepts and scores identically to per-candidate calls
        (equivalence verified by :func:`radar_bo.tests` and at design time:
        max |per-row - batched| ~ 1e-15). Row order is preserved by the
        batched call, so score ``i`` belongs to candidate ``i`` (the
        correspondence contract, instruction 20). Scores are in the LOG domain
        (log EI); non-finite scores are reported through the mask, never
        silently dropped.
        """
        arr = np.asarray(candidates, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.surrogate.dimension:
            raise AcquisitionError(f"Candidates must have shape (k, {self.surrogate.dimension}); got {arr.shape}.")
        if arr.shape[0] < 1:
            raise AcquisitionError("Cannot score an empty candidate pool.")
        if not np.all(np.isfinite(arr)):
            raise AcquisitionError("Candidate pool contains non-finite values.")
        tensor = numpy_to_torch(arr, name="candidates")
        with torch.no_grad():
            raw = acquisition(tensor.unsqueeze(1))  # (k, 1, d): batch of q=1 candidates
        scores = torch_to_numpy(raw.detach(), name="acquisition scores").flatten()
        if scores.shape != (arr.shape[0],):
            raise AcquisitionError(
                f"Acquisition returned {scores.shape[0]} scores for {arr.shape[0]} candidates; "
                "score-to-candidate correspondence could not be verified."
            )
        finite_mask = np.isfinite(scores)
        return scores, finite_mask

    @staticmethod
    def select_best(scores: np.ndarray, finite_mask: np.ndarray) -> int:
        """Index of the best candidate: argmax over FINITE log-EI scores with
        deterministic first-index tie-breaking."""
        scores = np.asarray(scores)
        if not finite_mask.any():
            raise SelectionFailureError("All acquisition scores are non-finite; no candidate can be selected.")
        masked = np.where(finite_mask, scores, -np.inf)
        return int(np.argmax(masked))  # argmax returns the FIRST maximal index (documented tie-break)
