"""Structural type for the indexed scatter update used by sliding-window prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mlx.core import array


class ScatterUpdate(Protocol):
    """Describe the update returned by ``array.at[index]`` for scattered addition."""

    def add(self, value: array) -> array:
        """Add values at selected indices and return the resulting array."""
        ...
