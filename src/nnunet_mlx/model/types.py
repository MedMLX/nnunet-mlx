"""Structural inference contract shared by resident and checkpoint-backed networks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec

if TYPE_CHECKING:
    from mlx.core import array

type NetworkOutput = array | list[array]
type PredictionModel = Callable[[array], object]


class InferenceModel(Protocol):
    """Declare the structural interface shared by resident and checkpoint-backed networks."""

    @property
    def spec(self) -> PlainConvUNetSpec | ResidualEncoderUNetSpec:
        """Return the model topology specification."""
        ...

    @property
    def training(self) -> bool:
        """Report whether the model is in training mode."""
        ...

    def __call__(self, values: array, /) -> NetworkOutput:
        """Run the model on its input array."""
        ...
