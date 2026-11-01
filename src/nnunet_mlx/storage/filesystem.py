"""Shared filesystem transactions for new artifact directories."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from nnunet_mlx.errors import ContractError


def write_new_directory(target: str | Path, writer: Callable[[Path], None]) -> None:
    """Create ``target`` by staging a new directory beside it.

    ``writer`` fills a temporary sibling directory that is renamed to ``target`` only
    if it returns normally; on any exception the temporary directory is removed.
    The target-existence check and rename are not a concurrency-safe no-clobber
    transaction, and parent directories created before staging are retained.

    Raises:
        ContractError: If ``target`` already exists.
    """
    destination = Path(target)
    if destination.exists():
        raise ContractError(f"target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        writer(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
