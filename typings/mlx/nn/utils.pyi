"""MLX 0.32.5 module-aware activation checkpointing."""

from collections.abc import Callable

from mlx.nn import Module

def checkpoint[**P, R](module: Module, fn: Callable[P, R]) -> Callable[P, R]: ...
