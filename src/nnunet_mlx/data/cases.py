"""Host loading and validation of persisted nnU-Net preprocessed cases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import blosc2
import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError
from nnunet_mlx.storage.properties import load_properties_pickle

type HostArray = npt.NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class PreprocessedCase:
    """One preprocessed nnU-Net case, as stored on disk (plans axis order).

    ``data`` is (C, X, Y, Z) or (C, X, Y) floating point, already normalized and
    resampled. ``segmentation`` has the same spatial shape with integer labels
    (-1 outside the nonzero crop mask, as nnU-Net stores it). ``previous_stage`` is
    the cascade's previous-stage label map with shape ``data.shape[1:]``, or
    ``None``. ``properties`` is the case's ``.pkl`` mapping (empty if not loaded).
    """

    identifier: str
    data: HostArray
    segmentation: HostArray
    previous_stage: HostArray | None
    properties: dict[str, object]


class PreprocessedDataset:
    """Read the pinned nnU-Net NumPy case layout without importing nnU-Net/Torch.

    A case is discovered by its ``<id>.npz`` or ``<id>.b2nd`` archive. Data and
    segmentation are loaded independently, each preferring its unpacked ``.npy``,
    then ``.b2nd``, then the corresponding key (``data`` or ``seg``) in ``.npz``.
    Thus a case can use mixed formats. ``.npy`` and ``.b2nd`` arrays are
    memory-mapped read-only. ``<id>.pkl`` holds the case properties.
    """

    def __init__(
        self,
        folder: str | Path,
        identifiers: Sequence[str] | None = None,
        *,
        previous_stage_folder: str | Path | None = None,
    ) -> None:
        """Discover and validate selected case archives and any previous-stage folder."""
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise ContractError(f"preprocessed dataset folder does not exist: {self.folder}")
        discovered = _discover_case_identifiers(self.folder)
        selected = discovered if identifiers is None else list(identifiers)
        if not selected or len(selected) != len(set(selected)):
            raise ContractError("preprocessed case identifiers must be non-empty and unique")
        missing = sorted(set(selected) - set(discovered))
        if missing:
            raise ContractError(f"preprocessed cases are missing archives: {missing}")
        self.identifiers = tuple(sorted(selected))
        self.previous_stage_folder = (
            None if previous_stage_folder is None else Path(previous_stage_folder)
        )
        if self.previous_stage_folder is not None and not self.previous_stage_folder.is_dir():
            raise ContractError(
                f"previous-stage folder does not exist: {self.previous_stage_folder}"
            )
        self._finite_validated_identifiers: set[str] = set()

    def load_case(
        self,
        identifier: str,
        *,
        load_properties: bool = True,
    ) -> PreprocessedCase:
        """Load and validate one case; omitted properties are an empty dict, never a cached pickle.

        Raises:
            ContractError: If the case is unknown, files are missing, or the arrays
                violate ``validate_case_arrays``. Finiteness of ``data`` is checked
                only the first time a case is loaded by this dataset object.
        """
        if identifier not in self.identifiers:
            raise ContractError(f"case {identifier!r} is not part of this dataset")
        data = _load_case_array(self.folder, identifier, suffix="", key="data")
        segmentation = _load_case_array(self.folder, identifier, suffix="_seg", key="seg")
        properties = (
            read_case_properties(self.folder / f"{identifier}.pkl") if load_properties else {}
        )
        previous = (
            None
            if self.previous_stage_folder is None
            else _load_case_array(self.previous_stage_folder, identifier, suffix="", key="seg")
        )
        check_finite = identifier not in self._finite_validated_identifiers
        validate_case_arrays(
            data,
            segmentation,
            previous,
            check_finite=check_finite,
        )
        self._finite_validated_identifiers.add(identifier)
        return PreprocessedCase(
            identifier=identifier,
            data=data,
            segmentation=segmentation,
            previous_stage=previous,
            properties=properties,
        )

    def case_shape(self, identifier: str) -> tuple[int, ...]:
        """Read spatial dimensions without loading voxels, segmentations, or properties."""
        if identifier not in self.identifiers:
            raise ContractError(f"case {identifier!r} is not part of this dataset")
        if any((self.folder / f"{identifier}{suffix}").is_file() for suffix in (".npy", ".b2nd")):
            data = _load_case_array(self.folder, identifier, suffix="", key="data")
            shape, dtype = data.shape, data.dtype
        else:
            with ZipFile(self.folder / f"{identifier}.npz") as archive:
                if "data.npy" not in archive.namelist():
                    raise ContractError(f"case archive {identifier}.npz is missing 'data'")
                with archive.open("data.npy") as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version in ((2, 0), (3, 0)):
                        # v3 changes structured field-name encoding, not floating-point headers.
                        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ContractError(f"unsupported NumPy array header version: {version}")
        if len(shape) not in (3, 4) or any(size < 1 for size in shape):
            raise ContractError(f"preprocessed data must be non-empty CXY or CXYZ, got {shape}")
        if not np.issubdtype(dtype, np.floating):
            raise ContractError("preprocessed data must contain floating-point values")
        return tuple(int(size) for size in shape[1:])


def _discover_case_identifiers(folder: Path) -> list[str]:
    """List case IDs from NPZ and non-segmentation B2ND archives."""
    archives = {path.stem for path in folder.glob("*.npz")}
    b2nd = {path.stem for path in folder.glob("*.b2nd") if not path.stem.endswith("_seg")}
    return sorted(archives | b2nd)


def _load_case_array(folder: Path, identifier: str, *, suffix: str, key: str) -> HostArray:
    """Open a case array, preferring ``.npy``, then ``.b2nd``, then ``<id>.npz``.

    The first two are memory-mapped read-only; the archive is read eagerly from
    its ``key`` entry. Pickled content is never loaded.
    """
    npy = folder / f"{identifier}{suffix}.npy"
    if npy.is_file():
        return cast(HostArray, np.load(npy, mmap_mode="r", allow_pickle=False))
    b2nd = folder / f"{identifier}{suffix}.b2nd"
    if b2nd.is_file():
        open_b2nd = cast(Callable[..., object], blosc2.__dict__["open"])
        return cast(
            HostArray,
            open_b2nd(
                urlpath=str(b2nd),
                mode="r",
                mmap_mode="r",
                dparams={"nthreads": 1},
            ),
        )
    archive_path = folder / f"{identifier}.npz"
    if not archive_path.is_file():
        raise ContractError(
            f"case {identifier!r} is missing {npy.name}, {b2nd.name}, and {archive_path.name}"
        )
    with np.load(archive_path, allow_pickle=False) as archive:
        if key not in archive:
            raise ContractError(f"case archive {archive_path.name} is missing {key!r}")
        return np.asarray(archive[key])


def read_case_properties(path: Path) -> dict[str, object]:
    """Load a case ``.pkl`` as restricted plain/NumPy metadata.

    The restricted loader rejects executable globals, object arrays, oversized
    payloads, and non-metadata values. It does not establish file provenance.

    Raises:
        ContractError: If the file is missing or does not contain a string-keyed mapping.
    """
    if not path.is_file():
        raise ContractError(f"case properties file does not exist: {path.name}")
    return load_properties_pickle(path)


def validate_case_arrays(
    data: HostArray,
    segmentation: HostArray,
    previous: HostArray | None,
    *,
    check_finite: bool = True,
) -> None:
    """Check the preprocessed array contract, raising ``ContractError`` on violation.

    ``data`` must be floating point with shape (C, X, Y) or (C, X, Y, Z);
    ``segmentation`` integer-valued with the same spatial shape; ``previous``, if
    given, integer-valued with exactly ``data.shape[1:]``. ``check_finite=False``
    skips the full-array finiteness scan of ``data``.
    """
    if data.ndim not in (3, 4) or int(data.shape[0]) < 1:
        raise ContractError(f"preprocessed data must be CXY or CXYZ, got {data.shape}")
    if segmentation.ndim != data.ndim or segmentation.shape[1:] != data.shape[1:]:
        raise ContractError("segmentation spatial shape does not match preprocessed data")
    if int(segmentation.shape[0]) < 1 or not np.issubdtype(segmentation.dtype, np.integer):
        raise ContractError("preprocessed segmentation must have integer channels")
    if not np.issubdtype(data.dtype, np.floating):
        raise ContractError("preprocessed data must contain finite floating-point values")
    if check_finite and not np.isfinite(data).all():
        raise ContractError("preprocessed data must contain finite floating-point values")
    if previous is not None and (
        previous.shape != data.shape[1:] or not np.issubdtype(previous.dtype, np.integer)
    ):
        raise ContractError("previous-stage segmentation must match the spatial case shape")
