"""Seeded upstream PlainConvUNet of the upstream runner recording, and its API contracts.

The runner recorder and the runner tests build the same network from the recorded
plans and seeds. Only the pinned upstream packages run here, never nnunet_mlx.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, Self, TypedDict, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from nnunet_reference.fixture_types import RunnerFixture

type HostArray = npt.NDArray[np.generic]


class ArchitectureArguments(TypedDict):
    """Record the dynamic-network-architectures U-Net keyword arguments in the fixture plans."""

    features_per_stage: list[int]
    kernel_sizes: list[list[int]]
    strides: list[list[int]]
    n_conv_per_stage: list[int]
    n_conv_per_stage_decoder: list[int]
    conv_bias: bool
    norm_op_kwargs: dict[str, float | bool]
    nonlin_kwargs: dict[str, float]


class RecorderArchitecture(TypedDict):
    """Record the architecture node of one fixture plans configuration."""

    arch_kwargs: ArchitectureArguments


class RecorderConfiguration(TypedDict):
    """Record one named fixture plans configuration."""

    architecture: RecorderArchitecture


class RecorderPlans(TypedDict):
    """Describe the minimal fixture plans mapping needed to build the source network."""

    configurations: dict[str, RecorderConfiguration]


class ReferenceNetwork(Protocol):
    """Describe the dynamic-network-architectures v0.4.4 U-Net surface used as reference."""

    def __call__(self, data: Tensor, /) -> Tensor:
        """Run the upstream network on a Torch tensor."""
        ...
    def eval(self) -> Self:
        """Set the upstream network to evaluation mode."""
        ...
    def state_dict(self) -> dict[str, Tensor]:
        """Return the upstream network parameter mapping."""
        ...
    def load_state_dict(self, state: Mapping[str, Tensor]) -> object:
        """Load the supplied parameter mapping into the upstream network."""
        ...


class TorchNeuralApi(Protocol):
    """Describe the ``torch.nn`` modules used by the nnU-Net v2.8.1 reference network."""

    Conv3d: type[torch.nn.Conv3d]
    InstanceNorm3d: type[torch.nn.InstanceNorm3d]
    LeakyReLU: type[torch.nn.LeakyReLU]

    def Identity(self) -> ReferenceNetwork:
        """Create the Torch identity module used for fold rounding."""
        ...


class TorchApi(Protocol):
    """Describe the ``torch`` namespace subset used to run the reference on CPU."""

    nn: TorchNeuralApi

    def manual_seed(self, seed: int) -> torch.Generator:
        """Set the Torch random seed and return its generator."""
        ...
    def set_num_threads(self, count: int) -> None:
        """Set the Torch reference CPU thread count."""
        ...
    def use_deterministic_algorithms(self, mode: bool) -> None:
        """Enable or disable deterministic Torch algorithms."""
        ...
    def from_numpy(self, data: HostArray) -> Tensor:
        """Create a Torch tensor from a NumPy array."""
        ...
    def inference_mode(self) -> AbstractContextManager[object]:
        """Enter Torch inference mode."""
        ...
    def device(self, device: str) -> torch.device:
        """Construct a Torch device descriptor."""
        ...
    def zeros(self, size: tuple[int, ...]) -> Tensor:
        """Create a zero-filled Torch tensor."""
        ...


class NetworkApi(Protocol):
    """Describe the dynamic-network-architectures U-Net constructor used as reference."""

    def PlainConvUNet(
        self,
        *,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: list[int],
        conv_op: type[torch.nn.Conv3d],
        kernel_sizes: list[list[int]],
        strides: list[list[int]],
        n_conv_per_stage: list[int],
        n_conv_per_stage_decoder: list[int],
        conv_bias: bool,
        norm_op: type[torch.nn.InstanceNorm3d],
        norm_op_kwargs: dict[str, float | bool],
        dropout_op: None,
        dropout_op_kwargs: dict[str, object],
        nonlin: type[torch.nn.LeakyReLU],
        nonlin_kwargs: dict[str, float],
        deep_supervision: bool,
        nonlin_first: bool,
    ) -> ReferenceNetwork:
        """Construct the upstream plain convolution U-Net from its topology."""
        ...


def build_network(runner: RunnerFixture, seed: int) -> ReferenceNetwork:
    """Build a seeded source plain U-Net from the runner fixture plans."""
    torch = cast(TorchApi, import_module("torch"))
    unet = cast(NetworkApi, import_module("dynamic_network_architectures.architectures.unet"))
    kwargs = cast(RecorderPlans, runner["plans"])["configurations"]["base"]["architecture"][
        "arch_kwargs"
    ]
    torch.manual_seed(seed)
    return unet.PlainConvUNet(
        input_channels=1,
        num_classes=2,
        n_stages=len(kwargs["features_per_stage"]),
        features_per_stage=kwargs["features_per_stage"],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=kwargs["kernel_sizes"],
        strides=kwargs["strides"],
        n_conv_per_stage=kwargs["n_conv_per_stage"],
        n_conv_per_stage_decoder=kwargs["n_conv_per_stage_decoder"],
        conv_bias=kwargs["conv_bias"],
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs=kwargs["norm_op_kwargs"],
        dropout_op=None,
        dropout_op_kwargs={},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs=kwargs["nonlin_kwargs"],
        deep_supervision=False,
        nonlin_first=False,
    ).eval()
