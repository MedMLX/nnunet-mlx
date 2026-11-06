"""Consumed subset of the pinned MLX namespace APIs, preserving native array and dtype identities.

The runtime owns lazy imports and Metal admission; these structural interfaces
carry the installed native signatures through that boundary. This is a partial
interface, not a replacement for upstream MLX typing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, overload

import numpy as np
import numpy.typing as npt

from nnunet_mlx.kernels.types import MetalKernel

if TYPE_CHECKING:
    from mlx.core import (
        Device,
        DeviceType,
        Dtype,
        StreamOrDevice,
        array,
    )
    from mlx.nn import (
        AvgPool2d,
        AvgPool3d,
        Conv2d,
        Conv3d,
        ConvTranspose2d,
        ConvTranspose3d,
        Module,
    )

type Scalar = int | float | bool | complex


class RandomRuntime(Protocol):
    """Declare the seeded MLX random operations exposed by the core runtime."""

    @staticmethod
    def key(seed: int) -> array:
        """Create an explicit MLX random key from a seed."""
        ...

    @staticmethod
    def seed(seed: int) -> None:
        """Set the default MLX random seed."""
        ...

    @staticmethod
    def uniform(
        low: Scalar | array = 0,
        high: Scalar | array = 1,
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Generate uniformly distributed values with the requested shape."""
        ...

    @staticmethod
    def normal(
        shape: Sequence[int] = [],
        dtype: Dtype | None = ...,
        loc: Scalar | array | None = None,
        scale: Scalar | array | None = None,
        key: array | None = None,
        stream: StreamOrDevice = None,
    ) -> array:
        """Generate normally distributed values with the requested shape."""
        ...


class FastRuntime(Protocol):
    """Declare the MLX fast Metal-kernel constructor."""

    @staticmethod
    def metal_kernel(
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
        compile_options: object | None = None,
    ) -> MetalKernel:
        """Construct an MLX Metal kernel with the supplied source and options."""
        ...


type ArrayInput = (
    Scalar | array | npt.NDArray[np.generic] | list[ArrayInput] | tuple[ArrayInput, ...]
)


class ArrayConstructor(Protocol):
    """Create an MLX array from host values and an optional dtype."""

    # The real array class admits host numeric buffers; its pinned upstream
    # DLPack annotation does not describe NumPy 2.4's keyword signature.
    def __call__(self, value: ArrayInput, dtype: Dtype | None = None) -> array:
        """Create an MLX array from the given values and optional dtype."""
        ...


class CoreRuntime(Protocol):
    """Describe the pinned ``mlx.core`` namespace subset consumed at inference."""

    array: ArrayConstructor
    float16: Dtype
    float32: Dtype
    uint16: Dtype
    uint32: Dtype
    int32: Dtype
    bool_: Dtype
    gpu: DeviceType
    random: RandomRuntime
    fast: FastRuntime
    __version__: str
    __file__: str

    @staticmethod
    def eval(*args: object) -> None:
        """Evaluate the supplied lazy MLX values."""
        ...
    @staticmethod
    def async_eval(*args: object) -> None:
        """Schedule evaluation of the supplied lazy MLX values."""
        ...
    @staticmethod
    def compile[**P, O](
        fun: Callable[P, O], inputs: object = None, outputs: object = None, shapeless: bool = False
    ) -> Callable[P, O]:
        """Compile a callable using the supplied state and shape options."""
        ...
    @staticmethod
    @overload
    def load(
        file: str | Path | BinaryIO,
        /,
        format: Literal["safetensors"],
        return_metadata: Literal[False] = False,
        *,
        stream: StreamOrDevice = None,
    ) -> dict[str, array]:
        """Load arrays or metadata from an MLX-supported file."""
        ...
    @staticmethod
    @overload
    def load(
        file: str | Path | BinaryIO,
        /,
        format: str | None = None,
        return_metadata: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array | dict[str, array] | tuple[dict[str, array], dict[str, str]]:
        """Load arrays or metadata from an MLX-supported file."""
        ...

    @staticmethod
    def default_device() -> Device:
        """Return the current default MLX device."""
        ...

    @staticmethod
    def get_peak_memory() -> int:
        """Return peak active MLX memory in bytes."""
        ...

    @staticmethod
    def reset_peak_memory() -> None:
        """Reset the peak active-memory counter."""
        ...

    @staticmethod
    def reshape(a: array, /, shape: Sequence[int], *, stream: StreamOrDevice = None) -> array:
        """Reshape an array without changing its element count."""
        ...

    @staticmethod
    def flip(
        a: array, /, axis: int | Sequence[int] | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Reverse array elements along the selected axes."""
        ...

    @staticmethod
    def matmul(a: array, b: array, /, *, stream: StreamOrDevice = None) -> array:
        """Multiply arrays using matrix multiplication semantics."""
        ...

    @staticmethod
    def square(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise square."""
        ...

    @staticmethod
    def rsqrt(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise reciprocal square root."""
        ...

    @staticmethod
    def sigmoid(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Compute the elementwise logistic sigmoid."""
        ...

    @overload
    @staticmethod
    def arange(
        start: int | float,
        stop: int | float | None,
        step: int | float | None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create a regularly spaced sequence with the requested dtype."""
        ...

    @overload
    @staticmethod
    def arange(
        stop: int | float,
        step: int | float | None = None,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create a regularly spaced sequence with the requested dtype."""
        ...

    @staticmethod
    def take(
        a: array, /, indices: int | array, axis: int | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Select array elements by index along an axis."""
        ...

    @staticmethod
    def full(
        shape: int | Sequence[int],
        vals: Scalar | array,
        dtype: Dtype | None = None,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Create an array filled with the supplied value."""
        ...

    @staticmethod
    def zeros(
        shape: int | Sequence[int], dtype: Dtype | None = ..., *, stream: StreamOrDevice = None
    ) -> array:
        """Create an array filled with zeros."""
        ...

    @staticmethod
    def ones(
        shape: int | Sequence[int], dtype: Dtype | None = ..., *, stream: StreamOrDevice = None
    ) -> array:
        """Create an array filled with ones."""
        ...

    @staticmethod
    def ones_like(
        a: array, /, dtype: Dtype | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Create an array of ones matching the input shape."""
        ...

    @staticmethod
    def all(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reduce boolean truth across the selected axes."""
        ...

    @staticmethod
    def any(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Reduce boolean truth across the selected axes."""
        ...

    @staticmethod
    def maximum(
        a: Scalar | array, b: Scalar | array, /, *, stream: StreamOrDevice = None
    ) -> array:
        """Return the elementwise maximum of the operands."""
        ...

    @staticmethod
    def floor(a: array, /, *, stream: StreamOrDevice = None) -> array:
        """Round array values down to their elementwise floor."""
        ...

    @staticmethod
    def isfinite(a: array, stream: StreamOrDevice = None) -> array:
        """Mark elements that are neither infinite nor NaN."""
        ...

    @staticmethod
    def transpose(
        a: array, /, axes: Sequence[int] | None = None, *, stream: StreamOrDevice = None
    ) -> array:
        """Reorder the axes of an array."""
        ...

    @staticmethod
    def sum(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Sum values across the selected axes."""
        ...

    @staticmethod
    def mean(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Average values across the selected axes."""
        ...

    @staticmethod
    def argmax(
        a: array,
        /,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return indices of maximum values along an axis."""
        ...

    @staticmethod
    def softmax(
        a: array,
        /,
        axis: int | Sequence[int] | None = None,
        *,
        precise: bool = False,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply softmax along the selected axes."""
        ...

    @staticmethod
    def concatenate(
        arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along an existing axis."""
        ...

    @staticmethod
    def conv3d(
        input: array,
        weight: array,
        /,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Apply a channels-last 3-D convolution."""
        ...

    @staticmethod
    def stack(
        arrays: Sequence[array], axis: int | None = 0, *, stream: StreamOrDevice = None
    ) -> array:
        """Join arrays along a new axis."""
        ...

    @staticmethod
    def pad(
        a: array,
        pad_width: int | tuple[int] | tuple[int, int] | list[tuple[int, int]],
        mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
        constant_values: Scalar | array = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Pad array axes using the requested boundary mode."""
        ...

    @staticmethod
    def where(
        condition: Scalar | array,
        x: Scalar | array,
        y: Scalar | array,
        /,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Select elements from two operands using a condition."""
        ...

    @staticmethod
    def contiguous(
        a: array, /, allow_col_major: bool = False, *, stream: StreamOrDevice = None
    ) -> array:
        """Return an array with a contiguous memory layout."""
        ...


class NeuralRuntime(Protocol):
    """Describe the pinned ``mlx.nn`` modules and activations consumed at inference."""

    Module: type[Module]
    Conv2d: type[Conv2d]
    Conv3d: type[Conv3d]
    ConvTranspose2d: type[ConvTranspose2d]
    ConvTranspose3d: type[ConvTranspose3d]
    AvgPool2d: type[AvgPool2d]
    AvgPool3d: type[AvgPool3d]

    @staticmethod
    def leaky_relu(x: array, negative_slope: float = 0.01) -> array:
        """Apply leaky ReLU with the selected negative slope."""
        ...


class ScatterUpdate(Protocol):
    """Describe the update returned by ``array.at[index]`` for scattered addition."""

    def add(self, value: array) -> array:
        """Add values at selected indices and return the resulting array."""
        ...
