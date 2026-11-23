"""Structural type for validating arrays before MLX is imported."""

from __future__ import annotations

from typing import Protocol


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
