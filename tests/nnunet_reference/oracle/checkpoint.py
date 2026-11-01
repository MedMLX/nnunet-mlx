"""Load trusted upstream Torch checkpoints for source-only numerical comparisons."""

from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from medmlx_core.conversion import load_torch_checkpoint

from nnunet_mlx.contracts import parse_mapping


class _Serialization(Protocol):
    def safe_globals(self, safe_globals: list[object]) -> AbstractContextManager[object]:
        """Temporarily allow the specified globals during restricted Torch loading."""
        ...


def load_reference_checkpoint(path: Path) -> dict[str, object]:
    """Load an upstream reference checkpoint with restricted NumPy scalar support."""
    import torch

    scalar = import_module("numpy._core.multiarray").scalar
    allowed: list[object] = [
        scalar,
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        *(type(np.dtype(dtype)) for dtype in (
            np.bool_, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64, np.complex64, np.complex128,
        )),
    ]
    with cast(_Serialization, torch.serialization).safe_globals(allowed):
        return parse_mapping(load_torch_checkpoint(path), name="reference checkpoint")
