"""Connected-component postprocessing for exported nnU-Net segmentations.

Mirrors the inference side of upstream
``nnunetv2.postprocessing.remove_connected_components``:
``remove_all_but_largest_component_from_segmentation`` and
``apply_postprocessing``, together with the one connected-component helper
they call (``acvl_utils.morphology.morphology_helper
.remove_all_but_largest_component``, full connectivity), in pure NumPy/SciPy
with no Torch or ``acvl_utils`` dependency.

Upstream *determines* which postprocessing steps help (``determine_postprocessing``)
from ground-truth labels during training/export; that is training/evaluation
tooling and out of scope here. This module only *applies* the
``postprocessing.pkl`` a released model already shipped, which is the
inference-time counterpart of ``nnUNetv2_apply_postprocessing``.
"""

from __future__ import annotations

import pickle
import pickletools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from nnunet_mlx.contracts import parse_string
from nnunet_mlx.errors import ContractError
from nnunet_mlx.storage.imageio import (
    read_exported_segmentation,
    resolve_image_reader,
    write_segmentation,
)

type IntArray = npt.NDArray[np.integer]
type BoolArray = npt.NDArray[np.bool_]
type LabelId = int
type RegionId = tuple[int, ...]
type LabelOrRegion = LabelId | RegionId

_SUPPORTED_GLOBAL = (
    "nnunetv2.postprocessing.remove_connected_components",
    "remove_all_but_largest_component_from_segmentation",
)
_SUPPORTED_STEP_KEYS = frozenset({"labels_or_regions", "background_label"})
_MAX_POSTPROCESSING_BYTES = 1024 * 1024
_UNSAFE_PICKLE_OPCODES = frozenset({"BINPERSID", "EXT1", "EXT2", "EXT4", "PERSID"})


@dataclass(frozen=True, slots=True)
class PostprocessingStep:
    """One ``remove_all_but_largest_component_from_segmentation`` call.

    ``labels_or_regions`` is always the already-normalized sequence of items
    to merge into one binary mask before keeping its largest component(s);
    upstream's ambiguous ``int | tuple[int, ...] | list[...]`` calling
    convention is resolved once, here, by :func:`load_postprocessing_pkl`.
    """

    labels_or_regions: tuple[LabelOrRegion, ...]
    background_label: int = 0


def _postprocessing_function_marker() -> None:
    """Stand-in identity for the one supported upstream postprocessing callable."""


def _reject_unsafe_opcodes(serialized: bytes) -> None:
    """Reject pickle operations that can bypass restricted global lookup."""

    for opcode, _argument, _position in pickletools.genops(serialized):
        if opcode.name in _UNSAFE_PICKLE_OPCODES:
            raise pickle.UnpicklingError(
                f"postprocessing.pkl uses unsupported pickle opcode {opcode.name}"
            )


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only plain containers/scalars and one supported function reference.

    Upstream's ``determine_postprocessing`` pickles ``(pp_fns, pp_fn_kwargs)``
    where every entry of ``pp_fns`` is
    ``remove_all_but_largest_component_from_segmentation``. Any other global
    lookup -- the usual arbitrary-code-execution vector for ``pickle.load``
    -- is rejected rather than silently accepted.
    """

    def find_class(self, module: str, name: str) -> object:
        """Resolve only the supported connected-component postprocessing marker."""
        if (module, name) == _SUPPORTED_GLOBAL:
            return _postprocessing_function_marker
        raise pickle.UnpicklingError(
            f"postprocessing.pkl references unsupported callable {module}.{name}; "
            "only remove_all_but_largest_component_from_segmentation is supported"
        )


def _parse_pp_fns_and_kwargs(
    payload: object, *, source: Path
) -> tuple[list[object], list[object]]:
    """Validate the serialized function and keyword lists as a pair."""
    pair_error = ContractError(f"{source} must pickle a (pp_fns, pp_fn_kwargs) pair")
    if not isinstance(payload, tuple):
        raise pair_error
    # `isinstance(payload, tuple)` alone narrows to `tuple[Unknown, ...]`; an
    # explicit `object`-element cast keeps every following `len`/indexing
    # operation's argument type fully known.
    items = cast(tuple[object, ...], payload)
    if len(items) != 2:
        raise pair_error
    first, second = items
    if not isinstance(first, list) or not isinstance(second, list):
        raise pair_error
    functions = cast(list[object], first)
    keyword_arguments = cast(list[object], second)
    if len(functions) != len(keyword_arguments):
        raise pair_error
    return functions, keyword_arguments


def load_postprocessing_pkl(path: str | Path) -> tuple[PostprocessingStep, ...]:
    """Load a size-bounded ``postprocessing.pkl`` without executable globals.

    Extension and persistent-ID opcodes are rejected before unpickling because
    Python's process-global extension cache can otherwise bypass ``find_class``.
    Only one complete payload of at most 1 MiB is accepted.

    Args:
        path: ``postprocessing.pkl`` from a pinned upstream release (shipped
            alongside ``plans.json``/``dataset.json`` in exported model
            folders).

    Returns:
        Ordered steps to apply, via :func:`apply_postprocessing`.

    Raises:
        OSError: If the file cannot be opened or read.
        ContractError: If unpickling raises a handled format error, the result is not a
            ``(pp_fns, pp_fn_kwargs)`` pair of equal-length lists, references
            an unsupported callable, or has a malformed keyword set.
    """

    source = Path(path)
    try:
        with source.open("rb") as handle:
            serialized = handle.read(_MAX_POSTPROCESSING_BYTES + 1)
        if len(serialized) > _MAX_POSTPROCESSING_BYTES:
            raise ContractError(f"postprocessing.pkl exceeds 1 MiB: {source}")
        _reject_unsafe_opcodes(serialized)
        stream = BytesIO(serialized)
        payload = cast(object, _RestrictedUnpickler(stream).load())
        if stream.read(1):
            raise pickle.UnpicklingError("trailing data after postprocessing.pkl payload")
    except ContractError:
        raise
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        MemoryError,
        RecursionError,
    ) as error:
        raise ContractError(f"{source} is not a valid postprocessing.pkl: {error}") from error
    functions, keyword_arguments = _parse_pp_fns_and_kwargs(payload, source=source)
    steps: list[PostprocessingStep] = []
    for function, kwargs in zip(functions, keyword_arguments, strict=True):
        if function is not _postprocessing_function_marker:
            raise ContractError(f"{source} references an unsupported postprocessing function")
        steps.append(_parse_step(kwargs, source=source))
    return tuple(steps)


def _parse_step(kwargs: object, *, source: Path) -> PostprocessingStep:
    """Validate the supported postprocessing keywords and construct one step."""
    if not isinstance(kwargs, dict):
        raise ContractError(f"{source} postprocessing keywords must be a mapping")
    kwargs = cast(dict[object, object], kwargs)
    unknown = {key for key in kwargs if key not in _SUPPORTED_STEP_KEYS}
    if unknown or "labels_or_regions" not in kwargs:
        raise ContractError(f"{source} has an unsupported postprocessing keyword set")
    labels_or_regions = _parse_labels_or_regions(kwargs["labels_or_regions"], source=source)
    background_label = kwargs.get("background_label", 0)
    if isinstance(background_label, bool) or not isinstance(background_label, int):
        raise ContractError(f"{source} background_label must be an integer")
    return PostprocessingStep(labels_or_regions, background_label)


def _parse_labels_or_regions(raw: object, *, source: Path) -> tuple[LabelOrRegion, ...]:
    # Upstream wraps a bare label or region (not itself a list) into one item.
    """Normalize one label or region, or a list of them, to a tuple."""
    items = cast(list[object], raw) if isinstance(raw, list) else [raw]
    if not items:
        raise ContractError(f"{source} postprocessing step has no labels_or_regions")
    return tuple(_parse_label_or_region(item, source=source) for item in items)


def _parse_label_or_region(item: object, *, source: Path) -> LabelOrRegion:
    """Parse one integer label or a nonempty integer region."""
    if isinstance(item, bool) or not isinstance(item, (int, tuple, list)):
        raise ContractError(f"{source} has a malformed labels_or_regions entry")
    if isinstance(item, int):
        return item
    if not item:
        raise ContractError(f"{source} has an empty region")
    values: list[int] = []
    for value in cast("tuple[object, ...] | list[object]", item):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{source} has a non-integer region label")
        values.append(value)
    return tuple(values)


def region_or_label_to_mask(segmentation: IntArray, region_or_label: LabelOrRegion) -> BoolArray:
    """``nnunetv2.evaluation.evaluate_predictions.region_or_label_to_mask``."""

    if isinstance(region_or_label, int):
        return cast(BoolArray, segmentation == region_or_label)
    mask = np.zeros(segmentation.shape, dtype=np.bool_)
    for label in region_or_label:
        mask[segmentation == label] = True
    return mask


def remove_all_but_largest_component(binary_image: BoolArray) -> BoolArray:
    """``acvl_utils.morphology.morphology_helper.remove_all_but_largest_component``.

    Keep every component tied for largest size, with full connectivity
    (26 neighbors in 3D), matching
    ``skimage.measure.label``'s ``connectivity=None`` default, which is the
    undecorated call upstream's postprocessing makes. An all-background mask
    returns an all-``False`` result.
    """

    structure = ndimage.generate_binary_structure(binary_image.ndim, binary_image.ndim)
    labels, count = ndimage.label(binary_image, structure=structure)
    if count == 0:
        return np.zeros_like(binary_image, dtype=np.bool_)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    keep = sizes == sizes.max()
    keep[0] = False
    return keep[labels]


def remove_all_but_largest_component_from_segmentation(
    segmentation: IntArray,
    items: Sequence[LabelOrRegion],
    background_label: int = 0,
) -> IntArray:
    """``nnunetv2.postprocessing.remove_connected_components
    .remove_all_but_largest_component_from_segmentation``.

    ``items`` merges every named label/region into one binary mask; voxels
    the mask selects but that are not in the mask's largest connected
    component(s) (ties keep every tied component, as upstream does) become
    ``background_label``.
    """

    mask = np.zeros(segmentation.shape, dtype=np.bool_)
    for item in items:
        mask |= region_or_label_to_mask(segmentation, item)
    mask_keep = remove_all_but_largest_component(mask)
    result = segmentation.copy()
    result[mask & ~mask_keep] = background_label
    return result


def apply_postprocessing(
    segmentation: IntArray, steps: Sequence[PostprocessingStep]
) -> IntArray:
    """Apply each connected-component step in order.

    With no steps the original array is returned unchanged, without a copy.
    """

    result = segmentation
    for step in steps:
        result = remove_all_but_largest_component_from_segmentation(
            result, step.labels_or_regions, step.background_label
        )
    return result


def resolve_exported_segmentation_reader(
    *, plans_json: Mapping[str, object], dataset_json: Mapping[str, object]
) -> tuple[str, str]:
    """Return ``(reader_name, file_ending)`` for files ``model predict`` already exported.

    Reads only the two ``plans.json``/``dataset.json`` fields the inference-time
    ``nnUNetv2_apply_postprocessing`` needs, without the full per-configuration
    plan validation :func:`nnunet_mlx.model.folder.load_model_bundle` performs
    to build a network -- postprocessing has no model/configuration dependency.
    """

    reader_name = parse_string(
        plans_json.get("image_reader_writer"), name="plans image_reader_writer"
    )
    file_ending = parse_string(dataset_json.get("file_ending"), name="dataset file_ending")
    if not file_ending.startswith("."):
        raise ContractError("dataset file_ending must start with a dot")
    return resolve_image_reader(reader_name, file_ending=file_ending), file_ending


def apply_postprocessing_to_file(
    reader_name: str,
    segmentation_file: str | Path,
    output_file: str | Path,
    steps: Sequence[PostprocessingStep],
    *,
    overwrite: bool = False,
) -> None:
    """``nnunetv2.postprocessing.remove_connected_components.load_postprocess_save``."""

    segmentation, properties = read_exported_segmentation(reader_name, segmentation_file)
    # The reader's leading channel axis is always a singleton for a
    # segmentation (validated by read_exported_segmentation); drop it to the
    # plain (Z, Y, X) -- or (1, Y, X) for a 2-D case -- array the writer expects.
    result = apply_postprocessing(cast(IntArray, segmentation[0]), steps)
    write_segmentation(reader_name, result, output_file, properties, overwrite=overwrite)


def apply_postprocessing_to_folder(
    reader_name: str,
    input_folder: str | Path,
    output_folder: str | Path,
    steps: Sequence[PostprocessingStep],
    *,
    file_ending: str,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Process matching files in sorted order and return their output paths.

    Files are written one at a time. If a later file fails, earlier outputs
    remain in place; the folder operation is not transactional. Existing
    outputs require ``overwrite=True``.
    """

    source = Path(input_folder)
    if not source.is_dir():
        raise ContractError(f"input folder does not exist: {source}")
    files = sorted(
        path for path in source.iterdir() if path.is_file() and path.name.endswith(file_ending)
    )
    if not files:
        raise ContractError(f"{source} contains no {file_ending!r} segmentation files")
    destination = Path(output_folder)
    written: list[Path] = []
    for path in files:
        output_file = destination / path.name
        apply_postprocessing_to_file(
            reader_name, path, output_file, steps, overwrite=overwrite
        )
        written.append(output_file)
    return tuple(written)


__all__ = [
    "LabelOrRegion",
    "PostprocessingStep",
    "apply_postprocessing",
    "apply_postprocessing_to_file",
    "apply_postprocessing_to_folder",
    "load_postprocessing_pkl",
    "region_or_label_to_mask",
    "remove_all_but_largest_component",
    "remove_all_but_largest_component_from_segmentation",
    "resolve_exported_segmentation_reader",
]
