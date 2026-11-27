"""Discover cases whose image modalities are stored in separate files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nnunet_mlx.errors import ContractError


@dataclass(frozen=True, slots=True)
class RawInputCase:
    """One case identifier and its image files in channel order."""

    identifier: str
    image_files: tuple[Path, ...]


def discover_input_cases(input_folder: str | Path, file_ending: str) -> tuple[RawInputCase, ...]:
    """Group split-dataset files by case and numeric modality suffix.

    Files are named ``<case>_<NNNN><file_ending>`` with a four-digit modality index.
    Each case's indices must be contiguous from ``0000``.

    Raises:
        ContractError: If the folder is missing or empty, a name lacks the suffix, a
            channel is duplicated, or a case has a gap in its channels.
    """

    folder = Path(input_folder)
    if not folder.is_dir():
        raise ContractError(f"prediction input folder does not exist: {folder}")
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.endswith(file_ending):
            continue
        stem = path.name[: -len(file_ending)]
        if len(stem) < 6 or stem[-5] != "_" or not stem[-4:].isdigit():
            raise ContractError(f"prediction input {path.name!r} lacks a four-digit channel suffix")
        identifier = stem[:-5]
        channel = int(stem[-4:])
        if not identifier or channel in grouped.setdefault(identifier, {}):
            raise ContractError(f"prediction input channel is ambiguous: {path.name}")
        grouped[identifier][channel] = path
    if not grouped:
        raise ContractError(f"prediction input folder contains no {file_ending!r} channel files")
    return tuple(
        _raw_input_case(identifier, channels) for identifier, channels in sorted(grouped.items())
    )


def _raw_input_case(identifier: str, channels: Mapping[int, Path]) -> RawInputCase:
    """Require contiguous channel indices and build one ordered input case."""
    indices = sorted(channels)
    if indices != list(range(len(indices))):
        raise ContractError(f"prediction case {identifier!r} channels must be contiguous from 0000")
    return RawInputCase(identifier, tuple(channels[index] for index in indices))
