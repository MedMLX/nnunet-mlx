"""MLX 0.32.4 tree utilities used by nnunet-mlx."""

from collections.abc import Callable
from typing import overload

type Tree[T] = T | list[Tree[T]] | tuple[Tree[T], ...] | dict[str, Tree[T]]

def tree_map[T, U](
    fn: Callable[[T], U], tree: Tree[T], *,
    is_leaf: Callable[[object], bool] | None = None,
) -> Tree[U]: ...

@overload
def tree_flatten[T](
    tree: Tree[T],
    prefix: str = "",
    is_leaf: Callable[[object], bool] | None = None,
    destination: None = None,
) -> list[tuple[str, T]]: ...
@overload
def tree_flatten[T](
    tree: Tree[T],
    prefix: str = "",
    is_leaf: Callable[[object], bool] | None = None,
    *,
    destination: dict[str, T],
) -> dict[str, T]: ...
@overload
def tree_flatten[T](
    tree: Tree[T],
    prefix: str,
    is_leaf: Callable[[object], bool] | None,
    destination: dict[str, T],
) -> dict[str, T]: ...
def tree_unflatten[T](tree: list[tuple[str, T]] | dict[str, T]) -> Tree[T]: ...
