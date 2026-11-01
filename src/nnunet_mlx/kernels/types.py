"""Typed callable contracts for the public MLX custom Metal kernel API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mlx.core import Dtype, StreamOrDevice, array

type TemplateArgument = tuple[str, int | bool | Dtype]


class MetalKernel(Protocol):
    """Declare the callable surface of a compiled MLX Metal kernel."""

    def __call__(
        self,
        *,
        inputs: Sequence[array],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[TemplateArgument] | None = None,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Launch the Metal kernel with the supplied arrays, grid and output contracts."""
        ...


class ShapedArray(Protocol):
    """Describe the shape and rank needed to validate arrays before MLX is imported."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array dimensions."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of array dimensions."""
        ...
