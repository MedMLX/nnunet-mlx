"""Memory-bounded restoration helpers for class-based nnU-Net outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.resampling import (
    nearest_axis_indices,
    resample_probabilities_or_data,
    resize_plane,
    resize_worker_count,
    separate_z_axis,
    validate_spacing,
)
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import ResamplingContract

type HostArray = npt.NDArray[np.generic]

# Thread count for per-plane and per-channel restoration (scipy/skimage release the GIL).
PARALLEL_SEPARATE_Z_WORKERS = 4


def resample_class_logits_argmax(
    logits: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> npt.NDArray[np.uint16]:
    """Resample each class channel to ``new_shape`` and return the per-voxel argmax label.

    Args:
        logits: Finite floating-point ``(C, X, Y, Z)`` class logits at ``current_spacing``.
        new_shape: Target spatial shape, in the same axis order.
        current_spacing: Voxel spacing (mm) of ``logits``, one entry per spatial axis.
        new_spacing: Voxel spacing (mm) of the target grid, same axis order.
        contract: The plans' probability-resampling settings.

    Returns:
        ``(X', Y', Z')`` uint16 channel indices. Ties go to the lowest channel index.

    One channel is resampled at a time and folded into a running maximum, so the
    full-resolution ``(C, X', Y', Z')`` volume is never materialized. This follows
    the upstream label-only path, which also takes argmax on resampled logits.
    The probability-saving path takes argmax after float32 softmax and can select
    a different label when rounding turns distinct logits into a tie.

    Raises:
        ContractError: If the logits are not finite floating-point ``(C, X, Y, Z)`` or
            have more than 65536 classes.
    """

    values, target_shape = _validated_inputs(logits, new_shape)
    channels = (
        _resample_channel(
            values,
            channel_index,
            target_shape,
            current_spacing,
            new_spacing,
            contract,
        )
        for channel_index in range(int(values.shape[0]))
    )
    return _ordered_argmax(channels)


def resample_class_logits_argmax_parallel(
    logits: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> npt.NDArray[np.uint16]:
    """Resample with bounded resize workers and preserve ordered argmax semantics.

    Same arguments and result as ``resample_class_logits_argmax``. When the plans
    resample the separate axis with order 0 and that axis is not enlarged, each target
    plane is built from one nearest source plane (``separate_plane_argmax``) and
    planes are processed in parallel; otherwise each channel's resampler owns the
    worker budget. Channel and slice thread pools are never nested.
    """

    values, target_shape = _validated_inputs(logits, new_shape)
    # Plane streaming uses the spline resampler's grid, not Torch's FP32 grid.
    if (contract.function_name == "resample_data_or_seg_to_shape"
            and contract.order_z == 0 and values.shape[1:] != target_shape):
        separate, axis = separate_z_axis(
            validate_spacing(current_spacing, name="current spacing"),
            validate_spacing(new_spacing, name="new spacing"),
            force=contract.force_separate_z,
            threshold=contract.separate_z_anisotropy_threshold,
        )
        if separate and axis is not None and target_shape[axis] <= int(values.shape[axis + 1]):
            return separate_plane_argmax(values, target_shape, axis, order=contract.order)
    return resample_class_logits_argmax(
        values, target_shape, current_spacing, new_spacing, contract,
    )


def separate_plane_argmax(
    values: HostArray,
    target_shape: tuple[int, ...],
    axis: int,
    *,
    order: int,
) -> npt.NDArray[np.uint16]:
    """Resize and reduce one target plane at a time for a separate axis.

    ``values`` is ``(C, X, Y, Z)``; ``axis`` is the spatial axis resampled separately
    with nearest-neighbor indices (``order_z == 0``) while the other two axes are
    resized with spline ``order``. Returns uint16 argmax labels of shape ``target_shape``.
    """

    in_plane_axes = tuple(index for index in range(3) if index != axis)
    in_plane_shape = tuple(target_shape[index] for index in in_plane_axes)
    indices = nearest_axis_indices(int(values.shape[axis + 1]), target_shape[axis])
    output = np.empty(target_shape, dtype=np.uint16)

    def restore_plane(target_index: int) -> None:
        """Resize one source plane and select the winning class per pixel."""
        best: npt.NDArray[np.float32] | None = None
        labels: npt.NDArray[np.uint16] | None = None
        source_index = int(indices[target_index])
        for channel_index in range(int(values.shape[0])):
            plane = np.take(values[channel_index], source_index, axis=axis)
            resized = resize_plane(plane, in_plane_shape, order=order)
            if not np.isfinite(resized).all():
                raise ContractError("resampling produced non-finite values")
            if best is None:
                best = resized
                labels = np.zeros(resized.shape, dtype=np.uint16)
                continue
            if labels is None:
                raise AssertionError("streaming plane argmax lost its label buffer")
            better = resized > best
            np.copyto(labels, np.uint16(channel_index), where=better)
            np.maximum(best, resized, out=best)
        if labels is None:
            raise AssertionError("streaming plane argmax produced no labels")
        selector: list[slice | int] = [slice(None)] * 3
        selector[axis] = target_index
        output[tuple(selector)] = labels

    working_set = (24 * (values[0].size // values.shape[axis + 1])
                   + 64 * int(np.prod(in_plane_shape)))
    worker_count = min(PARALLEL_SEPARATE_Z_WORKERS,
                       resize_worker_count(target_shape[axis], working_set,
                                           retained_bytes=output.nbytes))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _ in executor.map(restore_plane, range(target_shape[axis])):
            pass
    return output


def _validated_inputs(
    logits: HostArray, new_shape: Sequence[int]
) -> tuple[HostArray, tuple[int, ...]]:
    """Require finite CXYZ logits and a valid target shape for restoration."""
    values = np.asarray(logits)
    if values.ndim != 4 or int(values.shape[0]) < 1:
        raise ContractError("class-logit restoration requires non-empty CXYZ logits")
    floating = np.issubdtype(values.dtype, np.floating)
    if not floating or not np.isfinite(values).all():
        raise ContractError("class-logit restoration requires finite floating-point values")
    if int(values.shape[0]) > np.iinfo(np.uint16).max + 1:
        raise ContractError("class-logit restoration exceeds the uint16 label range")
    return values, tuple(int(value) for value in new_shape)


def _resample_channel(
    values: HostArray,
    channel_index: int,
    target_shape: tuple[int, ...],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> npt.NDArray[np.float32]:
    """Resample one class-logit channel to the target grid."""
    return resample_probabilities_or_data(
        values[channel_index : channel_index + 1],
        target_shape,
        current_spacing,
        new_spacing,
        contract,
    )[0]


def _ordered_argmax(
    channels: Iterable[npt.NDArray[np.float32]],
) -> npt.NDArray[np.uint16]:
    # Strict ">" keeps the earliest channel on ties, like ``np.argmax``.
    """Select the first maximum class at each voxel across channel arrays."""
    best_values: npt.NDArray[np.float32] | None = None
    segmentation: npt.NDArray[np.uint16] | None = None

    for channel_index, current in enumerate(channels):
        if best_values is None:
            best_values = current.copy()
            segmentation = np.zeros(current.shape, dtype=np.uint16)
            continue
        if segmentation is None:
            raise AssertionError("streaming argmax unexpectedly lacks a label buffer")
        better = current > best_values
        best_values[better] = current[better]
        segmentation[better] = np.uint16(channel_index)

    if segmentation is None:
        raise AssertionError("streaming argmax unexpectedly produced no labels")
    return np.ascontiguousarray(segmentation, dtype=np.uint16)
