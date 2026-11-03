"""Blosc2's compressed-array and chunk surfaces used by nnunet-mlx."""

from enum import Enum

import numpy as np
import numpy.typing as npt

class Codec(Enum):
    ZSTD = 5

class Filter(Enum):
    SHUFFLE = 1
    BYTEDELTA = 35

class SChunk:
    def __getitem__(self, item: int | slice) -> str | bytes: ...

class NDArray:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[np.generic]: ...
    @property
    def schunk(self) -> SChunk: ...
    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> npt.NDArray[np.generic]: ...
