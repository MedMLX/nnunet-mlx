"""Shared fail-fast parsers for untyped external contract payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from nnunet_mlx.errors import ContractError


def parse_mapping(value: object, *, name: str) -> dict[str, object]:
    """Return a shallow copy of a mapping with string keys.

    Raises:
        ContractError: If ``value`` is not a mapping or has a non-string key.
    """
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ContractError(f"{name} keys must be strings")
        result[key] = item
    return result


def parse_sequence(value: object, *, name: str) -> tuple[object, ...]:
    """Return a JSON-style array (``list`` or ``tuple``) as a tuple.

    Raises:
        ContractError: If ``value`` is not a list or tuple. Strings are rejected.
    """
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must be an array")
    return tuple(cast(Sequence[object], value))


def parse_string(value: object, *, name: str) -> str:
    """Return ``value`` if it is a non-empty string.

    Raises:
        ContractError: If ``value`` is not a string or is empty.
    """
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def parse_boolean(value: object, *, name: str) -> bool:
    """Return ``value`` if it is a ``bool``; integers such as 0 and 1 are rejected."""
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be boolean")
    return value


def parse_integer(value: object, *, name: str) -> int:
    """Return ``value`` if it is an ``int``; ``bool`` and floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def parse_number(value: object, *, name: str) -> float:
    """Return a finite ``int`` or ``float`` as ``float``; ``bool``, NaN and infinity raise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result
