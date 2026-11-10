"""Explicit MLX runtime boundary and metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, cast

from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec
from nnunet_mlx.prediction.source import SMILE_SOURCE_PROFILE
from nnunet_mlx.runtime_types import ArrayConstructor, CoreRuntime, NeuralRuntime

if TYPE_CHECKING:
    from mlx.core import Dtype, array

type Tree[T] = T | list[Tree[T]] | tuple[Tree[T], ...] | dict[str, Tree[T]]
type ArrayTree = Tree[array]

MLX_VERSION = "0.32.4"


class InferenceRuntime(Protocol):
    """Declare the MLX operations used by sliding-window prediction."""

    @property
    def array(self) -> ArrayConstructor:
        """Return the MLX array constructor used by inference."""
        ...

    @property
    def float32(self) -> Dtype:
        """Return the MLX float32 dtype."""
        ...

    def eval(self, *args: object) -> None:
        """Evaluate the supplied lazy MLX values."""
        ...

    def async_eval(self, *args: object) -> None:
        """Schedule evaluation of the supplied lazy MLX values without waiting."""
        ...

    def contiguous(self, value: array) -> array:
        """Return a contiguous MLX array."""
        ...

    def flip(self, value: array, *, axis: Sequence[int]) -> array:
        """Reverse the requested array axes."""
        ...


class CompilableModel[I, O](Protocol):
    """Describe a model with fixed state that MLX can compile."""

    @property
    def state(self) -> object:
        """Return the model state used by MLX compilation."""
        ...

    def __call__(self, values: I, /) -> O:
        """Run the compilable model on its input."""
        ...


class InferenceExecutionState(Protocol):
    """Expose the training flag that gates ordered residual inference."""

    @property
    def training(self) -> bool:
        """Report whether the model is in training mode."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Record active MLX, Python and platform versions and the Metal device.

    Compilation is chosen per model by ``inference_compilation_eligible``, so
    this process-wide record carries no compile flag.
    """

    backend: Literal["metal"]
    mlx_version: str
    python_version: str
    platform: str
    machine: str
    device: str
    dtype: str = "float32"
    module_version: str = ""

    def to_payload(self) -> dict[str, object]:
        """Serialize the active runtime settings."""
        return asdict(self)


def import_mlx() -> tuple[CoreRuntime, NeuralRuntime]:
    """Import MLX on Apple Silicon with Metal, or fail before model execution."""

    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise ContractError("MLX Metal requires Apple-silicon macOS")
    return _mlx_dependencies()


@lru_cache(maxsize=1)
def _mlx_dependencies() -> tuple[CoreRuntime, NeuralRuntime]:
    """Import the pinned MLX core and nn modules, requiring exactly ``MLX_VERSION``."""
    from medmlx_core.errors import MissingDependencyError
    from medmlx_core.runtime import import_mlx as import_metal

    try:
        import_metal()
        import mlx.core as core
        import mlx.nn as nn
    except MissingDependencyError as error:
        raise ContractError(str(error)) from error
    except ModuleNotFoundError as error:
        raise ContractError(f"MLX {MLX_VERSION} is required; run `make env`") from error
    version = _mlx_version()
    if version != MLX_VERSION:
        raise ContractError(f"MLX version {MLX_VERSION} is required, found {version}")
    return cast(tuple[CoreRuntime, NeuralRuntime], (core, nn))


@lru_cache(maxsize=1)
def _mlx_version() -> str:
    """Return the installed MLX distribution version."""
    return importlib.metadata.version("mlx")


def runtime_info() -> RuntimeInfo:
    """Describe the active MLX runtime; raise unless the default device is a Metal GPU."""
    mx, _ = import_mlx()
    device = str(mx.default_device())
    if "gpu" not in device.lower():
        raise ContractError(f"nnU-Net MLX requires a Metal GPU device, found {device}")
    return RuntimeInfo(
        backend="metal",
        mlx_version=_mlx_version(),
        module_version=str(mx.__version__),
        python_version=platform.python_version(),
        platform=sys.platform,
        machine=platform.machine(),
        device=device,
    )


def compile_inference_model[I, O](model: Callable[[I], O]) -> Callable[[I], O]:
    """Compile a fixed-state inference model for the selected MLX accelerator."""

    if not inference_compilation_eligible(model):
        raise ContractError(f"inference compilation is not qualified for {type(model).__name__}")
    mx, _ = import_mlx()
    if isinstance(getattr(model, "spec", None), ResidualEncoderUNetSpec):
        from nnunet_mlx.model import network

        return cast(
            Callable[[I], O],
            network.compile_residual_inference_model(cast(network.ResidualEncoderUNet, model)),
        )
    compiled = cast(CompilableModel[I, O], model)
    return mx.compile(compiled, inputs=compiled.state)


def inference_compilation_eligible(model: object) -> bool:
    """Return whether exact compiled inference is qualified for this class scale."""

    if getattr(model, "inference_arithmetic", "native") != "native":
        return False
    spec = getattr(model, "spec", None)
    return hasattr(model, "state") and (
        (isinstance(spec, PlainConvUNetSpec) and spec.num_classes <= 3)
        or _residual_inference_eligible(spec)
    )


def default_inference_compilation(model: object) -> bool:
    """Compile by default only within the qualified inference envelope."""
    return inference_compilation_eligible(model)


def validate_inference_execution(model: object) -> None:
    """Reject residual configurations outside the qualified inference envelopes."""

    spec = getattr(model, "spec", None)
    if (
        isinstance(spec, ResidualEncoderUNetSpec)
        and getattr(model, "inference_arithmetic", "native") != "native"
    ):
        execution_state = cast(InferenceExecutionState, model)
        if (
            getattr(model, "inference_arithmetic", None) == "torch-order-fp32"
            and getattr(model, "inference_source_profile", None) == SMILE_SOURCE_PROFILE
            and getattr(model, "inference_input_shape", None) == (1, 640, 640, 1)
            and not execution_state.training
            and residual_ordered_arithmetic_topology_eligible(spec)
        ):
            return
        raise ContractError(
            "ordered residual inference requires the qualified Metal SMILE envelope"
        )
    if isinstance(spec, ResidualEncoderUNetSpec):
        # Both block types pass conversion and repeated mirrored-window inference.
        # Compilation has its own, narrower envelope below.
        maximum_outputs = 26 if spec.spatial_dims == 2 else 32
        if spec.num_classes > maximum_outputs:
            raise ContractError(
                "model-folder inference is qualified only for BasicBlockD or BottleneckD "
                "ResidualEncoderUNet models with at most 32 outputs in 3-D native mode, "
                "or at most 26 outputs in 2-D native mode"
            )


def _residual_inference_eligible(spec: object) -> bool:
    """Identify the bounded 3D basic residual topology for the optimized path."""
    return (
        isinstance(spec, ResidualEncoderUNetSpec)
        and spec.spatial_dims == 3
        and spec.bottleneck_channels is None
        and spec.num_classes <= 27
    )


def residual_ordered_arithmetic_topology_eligible(spec: object) -> bool:
    """Bound 2-D parity arithmetic to the demonstrated SMILE BasicBlockD topology."""
    return isinstance(spec, ResidualEncoderUNetSpec) and spec == ResidualEncoderUNetSpec(
        input_channels=1,
        num_classes=26,
        features=(32, 64, 128, 256, 512, 512, 512, 512),
        kernel_sizes=((3, 3),) * 8,
        strides=((1, 1),) + ((2, 2),) * 7,
        encoder_blocks=(1, 3, 4, 6, 6, 6, 6, 6),
        decoder_convs=(1,) * 7,
    )


def tree_flatten[T](tree: Tree[T]) -> list[tuple[str, T]]:
    """Flatten a nested MLX parameter tree into named leaves."""
    from mlx import utils

    return utils.tree_flatten(tree)
