"""NumPy-to-PyTorch conversion boundaries (M5 instruction 11).

NumPy is the numerical representation of the generic objective and simulator
interfaces; PyTorch float64 CPU tensors are used in the GP and acquisition
layers. Every conversion COPIES (never shares mutable memory between dataset
storage and external arrays/tensors) and validates shape, dtype, device and
finiteness. Pydantic configuration objects are never passed to the GP.
"""

from __future__ import annotations

import numpy as np
import torch

TORCH_DTYPE = torch.float64
TORCH_DEVICE = torch.device("cpu")


def numpy_to_torch(array: np.ndarray, *, require_2d: bool = True, name: str = "array") -> torch.Tensor:
    """Convert a NumPy array to a float64 CPU tensor via an explicit COPY.

    ``require_2d=True`` enforces the GP-layer contract (M x d inputs, M x 1
    outputs). The returned tensor never shares memory with the source array.
    """
    arr = np.asarray(array)
    if require_2d and arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D for the GP layer; got shape {arr.shape}.")
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    tensor = torch.tensor(arr, dtype=TORCH_DTYPE, device=TORCH_DEVICE)  # copies by construction
    if tensor.dtype != TORCH_DTYPE or tensor.device != TORCH_DEVICE:
        raise RuntimeError("Conversion produced an unexpected dtype/device.")
    return tensor


def torch_to_numpy(tensor: torch.Tensor, *, name: str = "tensor") -> np.ndarray:
    """Convert a tensor to a detached float64 NumPy array via an explicit COPY."""
    arr = tensor.detach().cpu().to(torch.float64).numpy().copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values after conversion.")
    return arr
