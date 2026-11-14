"""Torch-free nnU-Net v2 default raw-case preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.contracts import parse_number as _number
from nnunet_mlx.data.resampling import (
    compute_new_shape,
    resample_probabilities_or_data,
    resample_segmentation,
)
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import PlansContract

type FloatArray = npt.NDArray[np.float32]
type HostArray = npt.NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class PreprocessedRawCase:
    """Preprocessed raw case: ``data`` (C, X, Y, Z) float32 in the plans' axis order and spacing.

    ``previous_stage`` is the resampled cascade label map (1, X, Y, Z), or ``None``.
    ``properties`` extends the input properties with ``shape_before_cropping``,
    ``bbox_used_for_cropping`` and ``shape_after_cropping_and_before_resampling``,
    all in the plans' axis order, which restoration needs to invert the crop.
    """

    data: FloatArray
    previous_stage: HostArray | None
    properties: dict[str, object]


def preprocess_raw_case(
    data: npt.NDArray[np.generic],
    properties: Mapping[str, object],
    plans: PlansContract,
    *,
    segmentation_previous_stage: HostArray | None = None,
) -> PreprocessedRawCase:
    """Apply transpose, nonzero crop, normalization, and plan resampling.

    Follows upstream's ``DefaultPreprocessor.run_case_npy`` step order.

    Args:
        data: Raw modalities, (C, X, Y, Z) in the array axis order produced by
            ``nnunet_mlx.storage.imageio`` (the reverse of SimpleITK's physical
            x, y, z order). 2-D inputs use a singleton leading spatial axis.
            Real integer or floating-point values must remain finite when cast to
            float32; complex data is rejected.
        properties: Must contain ``"spacing"``, three positive voxel sizes in mm in
            the same axis order as ``data``'s spatial axes (before the transpose).
        plans: Parsed plans for the selected configuration. Null spacing entries
            keep this image's spacing on those axes after transposition.
        segmentation_previous_stage: Cascade input, (1, X, Y, Z) non-negative
            integer labels representable in int32, in the same axis order as ``data``.

    Returns:
        The preprocessed case. ``data`` is normalized per channel according to
        ``plans.configuration.normalization_schemes`` after cropping.

    Raises:
        ContractError: If shapes, spacing or dtypes violate the contract, the image
            has no nonzero voxels, or a normalization scheme is unsupported.
    """

    values = np.asarray(data)
    if values.ndim != 4 or int(values.shape[0]) < 1:
        raise ContractError("raw nnU-Net cases must have CXYZ layout")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ContractError("raw nnU-Net cases must contain finite real numeric values")
    previous = _validate_previous_stage(segmentation_previous_stage, values.shape[1:])
    configuration = plans.configuration
    if len(configuration.normalization_schemes) != int(values.shape[0]):
        raise ContractError("normalization schemes do not match the input channels")
    result_properties = dict(properties)
    spacing = _float_tuple(result_properties.get("spacing"), name="image spacing")
    if len(spacing) != 3:
        raise ContractError("raw medical-image spacing must contain three values")
    transpose = (0, *(axis + 1 for axis in plans.transpose_forward))
    with np.errstate(over="ignore"):
        transposed = np.asarray(values.transpose(transpose), dtype=np.float32)
    if not np.isfinite(transposed).all():
        raise ContractError("raw nnU-Net cases must remain finite after float32 conversion")
    transposed_previous = None if previous is None else np.asarray(previous.transpose(transpose))
    original_spacing = tuple(spacing[axis] for axis in plans.transpose_forward)
    result_properties["shape_before_cropping"] = tuple(transposed.shape[1:])
    # CT ignores the mask; filling enclosed holes cannot alter the crop bounds.
    mask_required = transposed_previous is not None or any(
        scheme != "CTNormalization" for scheme in configuration.normalization_schemes
    )
    cropped, mask, cropped_previous, bounding_box = crop_to_nonzero(
        transposed, transposed_previous, mask_required=mask_required,
    )
    result_properties["bbox_used_for_cropping"] = [list(pair) for pair in bounding_box]
    result_properties["shape_after_cropping_and_before_resampling"] = tuple(cropped.shape[1:])
    normalized = _normalize(cropped, mask if cropped_previous is None else cropped_previous, plans)
    configured_spacing = (
        configuration.spacing
        if len(configuration.spacing) == 3
        else (original_spacing[0], *configuration.spacing)
    )
    target_spacing = tuple(
        original if configured is None else configured
        for configured, original in zip(configured_spacing, original_spacing, strict=True)
    )
    new_shape = compute_new_shape(normalized.shape[1:], original_spacing, target_spacing)
    resampled = resample_probabilities_or_data(
        normalized,
        new_shape,
        original_spacing,
        target_spacing,
        configuration.data_resampling,
    )
    resampled_previous = (
        None
        if cropped_previous is None
        else resample_segmentation(
            cropped_previous,
            new_shape,
            original_spacing,
            target_spacing,
            configuration.segmentation_resampling,
        )
    )
    return PreprocessedRawCase(
        data=resampled,
        previous_stage=resampled_previous,
        properties=result_properties,
    )


def crop_to_nonzero(
    data: FloatArray,
    previous_stage: HostArray | None,
    *,
    mask_required: bool = True,
) -> tuple[
    FloatArray,
    npt.NDArray[np.int8],
    HostArray | None,
    tuple[tuple[int, int], ...],
]:
    """Crop to the shared nonzero bounds, retaining mask semantics when needed.

    The nonzero mask is true where any channel of ``data`` (C, X, Y, Z) is nonzero,
    with enclosed holes filled, as in upstream's ``crop_to_nonzero``.
    ``mask_required=False`` returns an unused zero mask for CT normalization
    without previous-stage labels. Enclosed holes do not affect crop bounds.

    Returns:
        The cropped float32 data; an int8 mask (1, X, Y, Z) that is 0 inside the
        nonzero region and -1 outside it; the cropped ``previous_stage`` with
        background (0) outside the nonzero region set to -1, or ``None``; and the
        half-open ``(start, stop)`` bounds per spatial axis.

    Raises:
        ContractError: If ``data`` has no nonzero voxel.
    """
    mask = np.any(data != 0, axis=0)
    if mask_required:
        mask = np.asarray(_binary_fill_holes()(mask), dtype=bool)
    projections = tuple(
        np.flatnonzero(
            np.any(mask, axis=tuple(other for other in range(mask.ndim) if other != axis))
        )
        for axis in range(mask.ndim)
    )
    if not projections[0].size:
        raise ContractError("raw nnU-Net case has no nonzero voxels")
    bounding_box = tuple((int(indices[0]), int(indices[-1]) + 1) for indices in projections)
    slices = tuple(slice(start, stop) for start, stop in bounding_box)
    cropped = np.ascontiguousarray(data[(slice(None), *slices)], dtype=np.float32)
    cropped_mask = mask[slices]
    segmentation_mask = (
        np.where(cropped_mask, np.int8(0), np.int8(-1))[None]
        if mask_required else np.zeros((1, *cropped.shape[1:]), dtype=np.int8)
    )
    cropped_previous = None
    if previous_stage is not None:
        # Sentinels belong to this crop, never to a reusable caller-owned array.
        cropped_previous = np.array(previous_stage[(slice(None), *slices)], order="C", copy=True)
        outside = (cropped_previous == 0) & ~cropped_mask[None]
        cropped_previous[outside] = -1
    return cropped, segmentation_mask, cropped_previous, bounding_box


def _validate_previous_stage(
    segmentation: HostArray | None, spatial_shape: Sequence[int]
) -> HostArray | None:
    """Validate a cascade previous-stage label map of shape ``(1, *spatial_shape)``.

    Returns contiguous int32 labels; non-integer, negative or int32-overflowing
    values raise ``ContractError``.
    """
    if segmentation is None:
        return None
    values = np.asarray(segmentation)
    if values.shape != (1, *spatial_shape):
        raise ContractError("previous-stage segmentation shape does not match the image")
    if not np.issubdtype(values.dtype, np.integer):
        raise ContractError("previous-stage segmentation must contain integer labels")
    integer_values = cast(npt.NDArray[np.integer], values)
    if np.any(integer_values < 0):
        raise ContractError("raw previous-stage segmentation labels must be non-negative")
    if np.any(integer_values > np.iinfo(np.int32).max):
        raise ContractError("raw previous-stage segmentation labels must fit in int32")
    return np.ascontiguousarray(integer_values, dtype=np.int32)


def _normalize(
    data: FloatArray,
    segmentation_mask: HostArray,
    plans: PlansContract,
) -> FloatArray:
    """Apply each channel's plans normalization scheme to a copy of ``data``.

    ``data`` is ``(C, *spatial)``; the mask is the ``(1, *spatial)`` non-zero crop
    segmentation used by schemes with ``use_mask_for_norm`` (labels >= 0).
    Previous-stage foreground remains eligible even outside image support.
    Non-finite output raises.
    """
    result = data.copy()
    configuration = plans.configuration
    for channel, scheme in enumerate(configuration.normalization_schemes):
        intensity = plans.intensity_properties.get(str(channel), {})
        result[channel] = _normalize_channel(
            result[channel],
            segmentation_mask[0],
            scheme=scheme,
            use_mask=configuration.use_mask_for_norm[channel],
            intensity=intensity,
        )
    if not np.isfinite(result).all():
        raise ContractError("normalization produced non-finite values")
    return result


def _normalize_channel(
    image: FloatArray,
    segmentation_mask: HostArray,
    *,
    scheme: str,
    use_mask: bool,
    intensity: Mapping[str, object],
) -> FloatArray:
    """Normalize one cropped channel according to the selected plans scheme.

    CT uses stored percentile clipping and mean/std. Z-score uses the crop mask
    when requested and leaves values outside that mask unchanged. RescaleTo01
    uses the channel's extrema; RGBTo01 divides values in [0, 255] by 255.
    """
    result = np.asarray(image, dtype=np.float32).copy()
    if scheme == "CTNormalization":
        mean = _number(intensity.get("mean"), name="CT mean")
        standard_deviation = _positive_number(intensity.get("std"), name="CT std")
        lower = _number(intensity.get("percentile_00_5"), name="CT lower percentile")
        upper = _number(intensity.get("percentile_99_5"), name="CT upper percentile")
        if lower > upper:
            raise ContractError("CT intensity percentiles are reversed")
        np.clip(result, lower, upper, out=result)
        result -= mean
        result /= max(standard_deviation, 1e-8)
        return result
    if scheme == "ZScoreNormalization":
        mask = (cast(npt.NDArray[np.signedinteger], segmentation_mask) >= 0
                if use_mask else np.ones(result.shape, dtype=bool))
        selected = result[mask]
        if not len(selected):
            raise ContractError("z-score normalization mask is empty")
        mean = float(selected.mean())
        standard_deviation = float(selected.std())
        result[mask] = (result[mask] - mean) / max(standard_deviation, 1e-8)
        return result
    if scheme == "NoNormalization":
        return result
    if scheme == "RescaleTo01Normalization":
        result -= float(result.min())
        result /= max(float(result.max()), 1e-8)
        return result
    if scheme == "RGBTo01Normalization":
        if float(result.min()) < 0 or float(result.max()) > 255:
            raise ContractError("RGB normalization requires values in [0, 255]")
        result /= 255
        return result
    raise ContractError(f"unsupported nnU-Net normalization scheme {scheme!r}")


def _float_tuple(value: object, *, name: str) -> tuple[float, ...]:
    """Parse a sequence of positive numeric values as floats."""
    if not isinstance(value, list | tuple):
        raise ContractError(f"{name} must be an array")
    result: list[float] = []
    for item in cast(Sequence[object], value):
        result.append(_positive_number(item, name=name))
    return tuple(result)


def _positive_number(value: object, *, name: str) -> float:
    """Require a positive numeric value."""
    result = _number(value, name=name)
    if result <= 0:
        raise ContractError(f"{name} must be positive")
    return result


def _binary_fill_holes() -> Callable[..., npt.NDArray[np.generic]]:
    """Resolve the SciPy hole-filling operation for nonzero masks."""
    function = import_module("scipy.ndimage").binary_fill_holes
    return cast(Callable[..., npt.NDArray[np.generic]], function)


def labelmap_to_one_hot(segmentation: HostArray, labels: Sequence[int]) -> npt.NDArray[np.float32]:
    """Convert one integer label map to upstream cascade foreground channels.

    Returns a float32 array (n_foreground, *spatial) with one binary channel per
    nonzero entry of ``labels``, in the order given.
    """

    values = np.asarray(segmentation)
    if values.ndim not in (2, 3) or not np.issubdtype(values.dtype, np.integer):
        raise ContractError("cascade input must be one integer 2-D/3-D label map")
    foreground = tuple(label for label in labels if label != 0)
    if not foreground:
        raise ContractError("cascade input requires at least one foreground label")
    return np.ascontiguousarray(
        np.stack([values == label for label in foreground]), dtype=np.float32
    )
