"""NumPy conversion boundary for native-array qualification and test observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from mlx.core import array

type NumericScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
)
type NumericArray = npt.NDArray[NumericScalar]


@overload
def host_array(value: array | Sequence[array], *, copy: bool | None = None) -> NumericArray:
    """Observe a value as a host NumPy array with optional copy behavior."""
    ...
@overload
def host_array[T: np.generic](
    value: npt.NDArray[T], *, copy: bool | None = None
) -> npt.NDArray[T]:
    """Observe a value as a host NumPy array with optional copy behavior."""
    ...
@overload
def host_array(value: object, *, copy: bool | None = None) -> npt.NDArray[np.generic]:
    """Observe a value as a host NumPy array with optional copy behavior."""
    ...


def host_array(value: object, *, copy: bool | None = None) -> npt.NDArray[np.generic]:
    """Observe a value on the host without requesting dtype conversion.

    NumPy infers a dtype for non-array inputs. ``None`` uses ``np.asarray``;
    an explicit copy flag is forwarded to ``np.array`` and can raise if
    ``copy=False`` cannot be honored.
    """
    result = np.asarray(value) if copy is None else np.array(value, copy=copy)
    return cast(npt.NDArray[np.generic], result)
