"""Repository-used MLX 0.32.4 neural-network APIs, from the installed source."""

from collections.abc import Iterator, Sequence
from typing import Self

from mlx.core import array
from mlx.utils import Tree

type Parameters = dict[str, Tree[array]]
type ModuleTree = Module | list[ModuleTree] | dict[str, ModuleTree]
type Shape = int | Sequence[int]

class Module:
    # Partial interface: Module.update intentionally differs from dict.update.
    def __contains__(self, key: object) -> bool: ...
    def __getitem__(self, key: str) -> object: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __init__(self) -> None: ...
    @property
    def state(self) -> Self: ...
    @property
    def training(self) -> bool: ...
    def parameters(self) -> Parameters: ...
    def update(self, parameters: Parameters, strict: bool = True) -> Self: ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self: ...
    def save_weights(self, file: str) -> None: ...
    def eval(self) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def named_modules(self) -> list[tuple[str, Module]]: ...
    def children(self) -> dict[str, ModuleTree]: ...
    @staticmethod
    def is_module(value: object) -> bool: ...
    def update_modules(self, modules: dict[str, ModuleTree], strict: bool = True) -> Self: ...

class Conv2d(Module):
    weight: array
    bias: array
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Conv3d(Module):
    weight: array
    bias: array
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class ConvTranspose2d(Module):
    weight: array
    bias: array
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    output_padding: tuple[int, ...]
    dilation: tuple[int, ...]
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        output_padding: Shape = 0,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class ConvTranspose3d(Module):
    weight: array
    bias: array
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    output_padding: tuple[int, ...]
    dilation: tuple[int, ...]
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        output_padding: Shape = 0,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class AvgPool2d(Module):
    def __init__(
        self, kernel_size: Shape, stride: Shape | None = None, padding: Shape = 0
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class AvgPool3d(Module):
    def __init__(
        self, kernel_size: Shape, stride: Shape | None = None, padding: Shape = 0
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

def leaky_relu(x: array, negative_slope: float = 0.01) -> array: ...
