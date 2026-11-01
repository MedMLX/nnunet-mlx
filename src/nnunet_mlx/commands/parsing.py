"""Shared command-line value parsing without runtime imports."""

from __future__ import annotations

from collections.abc import Sequence

from nnunet_mlx.errors import ContractError


def parse_fold(value: str) -> int | str:
    """Parse a ``--fold`` value: ``"all"`` or a canonical non-negative integer.

    Non-canonical integers such as ``"01"`` or ``"+1"`` are rejected so the value
    matches the fold directory name ``fold_<n>`` exactly.

    Raises:
        ContractError: If ``value`` is neither ``"all"`` nor a canonical integer.
    """
    if value == "all":
        return value
    try:
        fold = int(value)
    except ValueError as error:
        raise ContractError(f"fold must be a non-negative integer or 'all': {value!r}") from error
    if fold < 0 or str(fold) != value:
        raise ContractError(f"fold must be a non-negative integer or 'all': {value!r}")
    return fold


def parse_selected_folds(
    folds: Sequence[str] | None,
) -> tuple[int | str, ...] | None:
    """Parse repeated ``--fold`` values; ``None`` (option not given) stays ``None``.

    This parses each value only; downstream model loading checks for duplicates
    and requires a nonempty selection when one is explicitly supplied.
    """
    return None if folds is None else tuple(parse_fold(value) for value in folds)
