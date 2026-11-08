"""Torch-free nnU-Net v2 spline and torch-style host resampling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.contracts import parse_number
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import InterpolationMode, ResamplingContract, SegmentationTieBreak

type FloatArray = npt.NDArray[np.float32]
type HostArray = npt.NDArray[np.generic]

ANISOTROPY_THRESHOLD = 3.0


def compute_new_shape(
    old_shape: Sequence[int],
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
) -> tuple[int, ...]:
    """Round ``old_size * old_spacing / new_spacing`` for each 2-D or 3-D axis.

    Spacings must be finite and positive, and input sizes positive. Python's
    rounding is used; an extreme spacing ratio can produce a zero output size,
    which the resampling functions subsequently reject.
    """
    if not (len(old_shape) == len(old_spacing) == len(new_spacing) and len(old_shape) in (2, 3)):
        raise ContractError("shape and spacing ranks must match in 2-D or 3-D")
    for spacing in (*old_spacing, *new_spacing):
        parse_number(spacing, name="spacing")
    if any(size < 1 for size in old_shape) or any(
        spacing <= 0 for spacing in (*old_spacing, *new_spacing)
    ):
        raise ContractError("resampling shape and spacing values must be positive")
    return tuple(
        round(old / new * size)
        for old, new, size in zip(old_spacing, new_spacing, old_shape, strict=True)
    )


def resample_probabilities_or_data(
    data: npt.NDArray[np.generic],
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> FloatArray:
    """Resample finite floating ``(C, X, Y, Z)`` data to a contiguous float32 array.

    Spacing and the plan's separate-axis policy choose whole-volume or separate-axis
    interpolation. No anti-aliasing is applied; an unchanged shape still casts to FP32.
    The function name selects spline interpolation or the Torch sampling grid.
    A cast or interpolation producing non-finite values raises ``ContractError``.

    Spline interpolation is computed in float64 and rounded once to the input's
    precision, as upstream stores it into a buffer of the input dtype: float16
    logits from nnU-Net's half-precision accumulation come back rounded to
    float16 (returned as float32), which decides near-tie labels. The Torch
    resampler interpolates in float32 for any input, so it does not round to half.
    """

    values = np.asarray(data)
    target_shape = tuple(int(value) for value in new_shape)
    if values.ndim != 4 or len(target_shape) != 3:
        raise ContractError("default nnU-Net resampling requires CXYZ arrays")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ContractError("resampling requires finite floating-point values")
    if any(size < 1 for size in target_shape):
        raise ContractError("resampling target shape must be positive")
    source_spacing = validate_spacing(current_spacing, name="current spacing")
    target_spacing = validate_spacing(new_spacing, name="new spacing")
    if values.shape[1:] == target_shape:
        with np.errstate(over="ignore"):
            result = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ContractError("resampling produced non-finite values")
        return result
    separate_z, axis = separate_z_axis(
        source_spacing,
        target_spacing,
        force=contract.force_separate_z,
        threshold=contract.separate_z_anisotropy_threshold,
    )
    output_dtype = np.float16 if values.dtype == np.float16 else np.float32
    if contract.function_name == "resample_torch_fornnunet":
        result = _resample_torch(values, target_shape, axis=axis, contract=contract, is_seg=False)
    elif separate_z:
        if axis is None:
            raise AssertionError("separate-z resampling unexpectedly lacks an axis")
        result = _resample_separate_z(
            values,
            target_shape,
            axis=axis,
            order=contract.order,
            order_z=contract.order_z,
            dtype=output_dtype,
        )
    else:
        result = np.stack(
            [
                _resize_channel(channel, target_shape, order=contract.order, dtype=output_dtype)
                for channel in values
            ]
        )
    if not np.isfinite(result).all():
        raise ContractError("resampling produced non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def resample_segmentation(
    segmentation: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> HostArray:
    """Resample integer ``(C, X, Y, Z)`` labels, retaining their dtype.

    The plan selects interpolation and separate-axis behavior; the returned array is
    contiguous. Equal shapes bypass interpolation but not spacing validation.
    For the spline resampler, explicit ``seg_tiebreak`` modes select label-indicator
    maxima in every pass. Without that field, in-plane ties use ``nearest`` and
    separate-axis interpolation with ``order_z > 0`` retains the pinned ``> 0.5``
    threshold; unassigned positions remain 0, even when 0 was absent from the input.
    Torch-style plans instead compare scale-1000, float16-quantized indicators and
    default to ``nearest`` in both passes; labels must be in [-1, 32767].
    """

    values = np.asarray(segmentation)
    target_shape = tuple(int(value) for value in new_shape)
    if values.ndim != 4 or len(target_shape) != 3:
        raise ContractError("default segmentation resampling requires CXYZ arrays")
    if not np.issubdtype(values.dtype, np.integer):
        raise ContractError("segmentation resampling requires integer labels")
    integer_values = cast(npt.NDArray[np.int8 | np.int16 | np.int32 | np.int64 |
                                     np.uint8 | np.uint16 | np.uint32 | np.uint64], values)
    if any(size < 1 for size in target_shape):
        raise ContractError("resampling target shape must be positive")
    source_spacing = validate_spacing(current_spacing, name="current spacing")
    target_spacing = validate_spacing(new_spacing, name="new spacing")
    if contract.function_name == "resample_torch_fornnunet" and (
        integer_values.min() < -1 or integer_values.max() > 32767
    ):
        raise ContractError("resample_torch_fornnunet labels must be in [-1, 32767]")
    if values.shape[1:] == target_shape:
        return np.ascontiguousarray(values)
    separate_z, axis = separate_z_axis(
        source_spacing,
        target_spacing,
        force=contract.force_separate_z,
        threshold=contract.separate_z_anisotropy_threshold,
    )
    if contract.function_name == "resample_torch_fornnunet":
        result = _resample_torch(values, target_shape, axis=axis, contract=contract, is_seg=True)
    elif separate_z:
        if axis is None:
            raise AssertionError("separate-z resampling unexpectedly lacks an axis")
        result = _resample_segmentation_separate_z(
            values,
            target_shape,
            axis=axis,
            order=contract.order,
            order_z=contract.order_z,
            seg_tiebreak=contract.seg_tiebreak,
        )
    else:
        result = np.stack(
            [
                _resize_segmentation(
                    channel, target_shape, order=contract.order,
                    seg_tiebreak=contract.seg_tiebreak or "nearest",
                )
                for channel in values
            ]
        )
    return np.ascontiguousarray(result, dtype=values.dtype)


def _resample_torch(
    data: HostArray, shape: tuple[int, int, int], *, axis: int | None,
    contract: ResamplingContract, is_seg: bool,
) -> HostArray:
    """Apply the 17cb295 torch resampler's passes using host interpolation.

    In the separate-axis path each source plane becomes a channel, exactly as
    upstream's rearrange does. Label decisions are made between the two passes.
    """
    if not is_seg:
        with np.errstate(over="ignore"):
            data = np.asarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ContractError("resampling requires values representable as finite float32")
    tiebreak = contract.seg_tiebreak or "nearest"
    if axis is None:
        return _torch_resize(data, shape, contract.mode, is_seg=is_seg, tiebreak=tiebreak)
    planes = np.moveaxis(data, axis + 1, 1)
    plane_shape = tuple(shape[index] for index in range(3) if index != axis)
    resized = _torch_resize(
        planes.reshape(-1, *planes.shape[2:]), plane_shape, contract.mode,
        is_seg=is_seg, tiebreak=tiebreak,
    )
    intermediate = np.moveaxis(resized.reshape(*planes.shape[:2], *plane_shape), 1, axis + 1)
    return _torch_resize(intermediate, shape, contract.aniso_axis_mode,
                         is_seg=is_seg, tiebreak=tiebreak)


def _torch_resize(
    data: HostArray, shape: tuple[int, ...], mode: InterpolationMode, *,
    is_seg: bool, tiebreak: SegmentationTieBreak,
) -> HostArray:
    """Resize CXY/CXYZ arrays with half-pixel linear or Torch nearest sampling.

    SciPy supplies the interpolation kernel; float32 coordinates reproduce
    Torch's size-based align_corners=False grid. No antialiasing or clipping of
    interpolated image values is applied. Label dtype is retained by our API.
    """
    if data.shape[1:] == shape:
        return data
    if mode != "linear":
        result = data
        for axis, (old, new) in enumerate(zip(data.shape[1:], shape, strict=True), start=1):
            offset = 0.5 if mode == "nearest-exact" else 0.0
            indices = np.floor(np.float32(old / new) *
                               (np.arange(new, dtype=np.float32) + offset)).astype(np.intp)
            result = np.take(result, np.minimum(indices, old - 1), axis=axis)
        return result
    coordinates = np.asarray(np.meshgrid(*(
        np.maximum(np.float32(old / new) * (np.arange(new, dtype=np.float32) + 0.5) - 0.5, 0)
        for old, new in zip(data.shape[1:], shape, strict=True)
    ), indexing="ij"), dtype=np.float32)

    def interpolate(values: HostArray) -> FloatArray:
        """Interpolate each channel on the precomputed half-pixel coordinate grid."""
        return np.stack([
            np.asarray(_map_coordinates()(np.asarray(channel, dtype=np.float32), coordinates,
                                          order=1, mode="nearest", prefilter=False),
                       dtype=np.float32)
            for channel in values
        ])

    if not is_seg:
        return interpolate(data)
    # Both upstream memory modes compare these quantized scores, not the original
    # FP32 indicators. Retaining that rounding is essential for near-tie labels.
    best = np.full((len(data), *shape), -np.inf, dtype=np.float32)
    result = np.zeros(best.shape, dtype=data.dtype)
    nearest = (_torch_resize(data, shape, "nearest-exact", is_seg=True, tiebreak=tiebreak)
               if tiebreak == "nearest" else None)
    nearest_score = np.full(best.shape, -np.inf, dtype=np.float32)
    for label in np.unique(data):
        scores = interpolate((data == label).astype(np.float32) * 1000)
        scores = scores.astype(np.float16).astype(np.float32)
        better = scores >= best if tiebreak == "highest" else scores > best
        np.copyto(result, label, where=better)
        np.maximum(best, scores, out=best)
        if nearest is not None:
            np.copyto(nearest_score, scores, where=nearest == label)
    if nearest is not None:
        np.copyto(result, nearest, where=nearest_score == best)
    return result


def _resample_separate_z(
    data: npt.NDArray[np.generic],
    new_shape: tuple[int, int, int],
    *,
    axis: int,
    order: int,
    order_z: int,
    dtype: type[np.float16] | type[np.float32] = np.float32,
) -> npt.NDArray[np.float16 | np.float32]:
    """Resample ``(C, X, Y, Z)`` data in-plane slice by slice, then along ``axis``.

    The low-resolution ``axis`` is kept at its source size during the in-plane
    pass and only resized afterwards with ``order_z``, matching nnU-Net's
    separate-z strategy for anisotropic spacing. As upstream, the in-plane result
    stays float64 and is rounded to ``dtype`` once, after the out-of-plane pass.
    """
    source_shape = tuple(int(value) for value in data.shape[1:])
    in_plane_axes = tuple(index for index in range(3) if index != axis)
    in_plane_shape = tuple(new_shape[index] for index in in_plane_axes)
    intermediate_shape = list(new_shape)
    intermediate_shape[axis] = source_shape[axis]
    result = np.empty((len(data), *new_shape), dtype=dtype)
    for channel_index, channel in enumerate(data):
        # Rounding planes before an interpolating out-of-plane pass would round twice.
        intermediate = np.empty(tuple(intermediate_shape), dtype=np.float64)
        for slice_index in range(source_shape[axis]):
            source_slice = np.take(channel, slice_index, axis=axis)
            resized = _resize_channel(source_slice, in_plane_shape, order=order, dtype=np.float64)
            selector: list[slice | int] = [slice(None)] * 3
            selector[axis] = slice_index
            intermediate[tuple(selector)] = resized
        result[channel_index] = resize_out_of_plane(
            intermediate,
            new_shape,
            order=order_z,
            dtype=dtype,
        )
    return result


def _resample_segmentation_separate_z(
    data: HostArray,
    new_shape: tuple[int, int, int],
    *,
    axis: int,
    order: int,
    order_z: int,
    seg_tiebreak: SegmentationTieBreak | None,
) -> HostArray:
    """Separate-axis resampling of integer labels, preserving the input dtype.

    Same two-pass scheme as :func:`_resample_separate_z`, but each pass picks
    labels with :func:`_resize_segmentation` instead of interpolating values.
    """
    source_shape = tuple(int(value) for value in data.shape[1:])
    in_plane_axes = tuple(index for index in range(3) if index != axis)
    in_plane_shape = tuple(new_shape[index] for index in in_plane_axes)
    intermediate_shape = list(new_shape)
    intermediate_shape[axis] = source_shape[axis]
    result = np.empty((len(data), *new_shape), dtype=data.dtype)
    for channel_index, channel in enumerate(data):
        intermediate = np.empty(tuple(intermediate_shape), dtype=data.dtype)
        for slice_index in range(source_shape[axis]):
            source_slice = np.take(channel, slice_index, axis=axis)
            resized = _resize_segmentation(
                source_slice, in_plane_shape, order=order,
                seg_tiebreak=seg_tiebreak or "nearest",
            )
            selector: list[slice | int] = [slice(None)] * 3
            selector[axis] = slice_index
            intermediate[tuple(selector)] = resized
        result[channel_index] = _resize_segmentation_out_of_plane(
            intermediate,
            new_shape,
            order=order_z,
            seg_tiebreak=seg_tiebreak,
        )
    return result


def _resize_segmentation_out_of_plane(
    array: HostArray,
    new_shape: tuple[int, int, int],
    *,
    order: int,
    seg_tiebreak: SegmentationTieBreak | None,
) -> HostArray:
    """Resize labels along the single changed axis of an intermediate volume.

    Order zero is nearest-axis indexing. Otherwise the plans' ``seg_tiebreak``
    selects the exact-tie rules; when it is absent the per-label 0.5 threshold of
    v2.8.1 plans is used.
    """
    if order == 0:
        return nearest_axis_resize(array, new_shape)
    if seg_tiebreak is not None:
        if array.shape == new_shape:
            return array
        return _resize_segmentation(array, new_shape, order=order, seg_tiebreak=seg_tiebreak)
    # Plans without the new field retain v2.8.1's separate-axis threshold rule.
    coordinates = coordinate_map(array.shape, new_shape)
    result = np.zeros(new_shape, dtype=array.dtype)
    for label in np.sort(np.unique(array)):
        interpolated = _map_coordinates()(
            (array == label).astype(float), coordinates, order=order, mode="nearest"
        )
        result[np.asarray(interpolated, dtype=np.float64) > 0.5] = label
    return result


def resize_out_of_plane(
    array: npt.NDArray[np.generic],
    new_shape: tuple[int, int, int],
    *,
    order: int,
    dtype: type[np.float16] | type[np.float32] = np.float32,
) -> npt.NDArray[np.float16 | np.float32]:
    """Resize a 3-D intermediate to ``new_shape`` as ``dtype`` with nearest boundaries.

    Order zero uses nearest-axis indexing and permits at most one changed axis;
    higher orders use the half-pixel coordinate map and SciPy interpolation. An
    unchanged shape is only cast, as upstream skips this pass.
    """
    if order == 0 or array.shape == new_shape:
        return np.asarray(nearest_axis_resize(array, new_shape), dtype=dtype)
    coordinates = coordinate_map(array.shape, new_shape)
    return np.asarray(
        _map_coordinates()(array, coordinates, order=order, mode="nearest"),
        dtype=dtype,
    )


def nearest_axis_indices(old: int, new: int) -> npt.NDArray[np.intp]:
    """Return the pinned half-pixel nearest indices for one resized axis."""

    coordinates = float(old) / new * (np.arange(new, dtype=np.float64) + 0.5) - 0.5
    indices = np.floor(coordinates + 0.5).astype(np.intp)
    np.clip(indices, 0, old - 1, out=indices)
    return indices


def nearest_axis_resize(
    array: HostArray,
    new_shape: tuple[int, int, int],
) -> HostArray:
    """Nearest-resize one changed axis; return the original array if none changed.

    More than one changed axis raises ``ContractError``. Index locations follow
    :func:`nearest_axis_indices`.
    """
    changed = [axis for axis, size in enumerate(new_shape) if array.shape[axis] != size]
    if not changed:
        return array
    if len(changed) != 1:
        raise ContractError("separate-axis nearest resizing requires only one changed dimension")
    axis = changed[0]
    old, new = array.shape[axis], new_shape[axis]
    return np.take(array, nearest_axis_indices(old, new), axis=axis)


def coordinate_map(
    old_shape: Sequence[int], new_shape: tuple[int, int, int]
) -> npt.NDArray[np.float64]:
    """Return a float64 ``(3, *new_shape)`` half-pixel map into the old 3-D grid.

    Axis ``i`` uses ``old[i] / new[i] * (output_index + 0.5) - 0.5``; boundary
    handling is left to the sampling operation.
    """
    scales = tuple(float(old) / new for old, new in zip(old_shape, new_shape, strict=True))
    grids = np.mgrid[: new_shape[0], : new_shape[1], : new_shape[2]]
    return np.asarray(
        [scale * (grid + 0.5) - 0.5 for scale, grid in zip(scales, grids, strict=True)],
        dtype=np.float64,
    )


def _resize_channel(
    channel: npt.NDArray[np.generic],
    shape: Sequence[int],
    *,
    order: int,
    dtype: type[np.float16] | type[np.float32] | type[np.float64] = np.float32,
) -> npt.NDArray[np.float16 | np.float32 | np.float64]:
    """Resize one 2-D or 3-D channel to ``dtype`` with edge padding, no anti-aliasing.

    SciPy interpolates in float64 and returns the input's float type (float32 for
    half input), so anything but float32 to float32 runs on a float64 copy; the
    cast to ``dtype`` is then the only rounding, as in upstream's float64 path.
    """
    values = np.asarray(channel)
    if not (values.dtype == np.float32 and dtype is np.float32):
        values = np.asarray(values, dtype=np.float64)
    return np.asarray(
        _resize()(
            values,
            tuple(shape),
            order=order,
            mode="edge",
            anti_aliasing=False,
        ),
        dtype=dtype,
    )


def resize_plane(
    plane: npt.NDArray[np.generic],
    shape: Sequence[int],
    *,
    order: int,
) -> FloatArray:
    """Resize one in-plane slice with the pinned order-N interpolation, as float32.

    Half-precision planes are rounded to float16 first, as upstream does.
    """

    dtype = np.float16 if np.asarray(plane).dtype == np.float16 else np.float32
    return np.asarray(_resize_channel(plane, shape, order=order, dtype=dtype), dtype=np.float32)


def _resize_segmentation(
    segmentation: HostArray,
    shape: Sequence[int],
    *,
    order: int,
    seg_tiebreak: SegmentationTieBreak,
) -> HostArray:
    """Select label-indicator maxima with batchgenerators 0.25.5's exact-tie rules."""
    if order == 0:
        return np.asarray(
            _resize()(
                segmentation.astype(float),
                tuple(shape),
                order=0,
                mode="edge",
                anti_aliasing=False,
            ),
            dtype=segmentation.dtype,
        )
    result = np.zeros(tuple(shape), dtype=segmentation.dtype)
    labels = np.sort(np.unique(segmentation))
    if not len(labels):
        return result
    nearest = None
    if seg_tiebreak == "nearest":
        nearest = np.asarray(
            _resize()(
                segmentation.astype(float),
                tuple(shape),
                order=0,
                mode="edge",
                anti_aliasing=False,
            ),
            dtype=segmentation.dtype,
        )
    best: npt.NDArray[np.float64] | None = None
    nearest_top = np.zeros(tuple(shape), dtype=bool)
    for label in labels:
        interpolated = np.asarray(
            _resize()(
                (segmentation == label).astype(float),
                tuple(shape),
                order=order,
                mode="edge",
                anti_aliasing=False,
            ),
            dtype=np.float64,
        )
        is_nearest = None if nearest is None else nearest == label
        if best is None:
            best = interpolated
            result.fill(label)
            if is_nearest is not None:
                nearest_top = is_nearest & (interpolated > 0)
            continue
        # Labels are ascending: >= gives exact ties to the largest, > to the smallest.
        better = interpolated >= best if seg_tiebreak == "highest" else interpolated > best
        if is_nearest is not None:
            # Nearest labels resolve exact ties only when they share the top score.
            nearest_top |= (interpolated == best) & (interpolated > 0) & is_nearest
            np.copyto(nearest_top, is_nearest, where=better)
        np.copyto(result, label, where=better)
        np.maximum(best, interpolated, out=best)
    if nearest is not None:
        np.copyto(result, nearest, where=nearest_top)
    return result


def separate_z_axis(
    current_spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
    *,
    force: bool | None,
    threshold: float = ANISOTROPY_THRESHOLD,
) -> tuple[bool, int | None]:
    """Choose a unique coarsest axis for separate resampling, if the policy permits it.

    With ``force=None``, a max/min spacing ratio above ``threshold`` enables separation, using
    current spacing when it is anisotropic and otherwise target spacing. A forced
    true value uses current spacing; false disables separation. A tie for the
    coarsest axis disables it. Inputs must already be valid positive spacings.
    """
    if force is None:
        current_anisotropic = max(current_spacing) / min(current_spacing) > threshold
        separate = current_anisotropic or max(new_spacing) / min(new_spacing) > threshold
        basis = current_spacing if current_anisotropic else new_spacing
    else:
        separate = force
        basis = current_spacing
    if not separate:
        return False, None
    maximum = max(basis)
    axes = tuple(index for index, spacing in enumerate(basis) if maximum / spacing == 1)
    if len(axes) != 1:
        return False, None
    return True, axes[0]


def validate_spacing(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    """Require three finite, positive numeric spacings, rejecting booleans."""
    result = tuple(parse_number(item, name=name) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ContractError(f"{name} must contain three positive values")
    return result


def _map_coordinates() -> Callable[..., npt.NDArray[np.generic]]:
    """Resolve SciPy coordinate interpolation for host preprocessing."""
    function = import_module("scipy.ndimage").map_coordinates
    return cast(Callable[..., npt.NDArray[np.generic]], function)


def _resize() -> Callable[..., npt.NDArray[np.generic]]:
    """Resolve scikit-image resize for host preprocessing."""
    function = import_module("skimage.transform").resize
    return cast(Callable[..., npt.NDArray[np.generic]], function)
