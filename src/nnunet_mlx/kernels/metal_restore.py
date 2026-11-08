"""Metal-fused class-logit restoration with explicit reference dispatch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.data.resampling import separate_z_axis, validate_spacing
from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_numeric import DOUBLE_SINGLE_HEADER
from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.plans import ResamplingContract
from nnunet_mlx.prediction.restore import (
    resample_class_logits_argmax,
    resample_class_logits_argmax_parallel,
)
from nnunet_mlx.runtime import import_mlx

type HostArray = npt.NDArray[np.generic]

_THREADGROUP_SIZE = 256
if TYPE_CHECKING:
    from mlx.core import array

_KERNEL_NAME = "nnunet_mlx_fused_restore_half_pixel_ds_v3"
_MAXIMUM_FUSED_MAGNITUDE = float(np.finfo(np.float32).max) / 16
# This threshold includes cold compile and the small-workload crossover.
MINIMUM_FUSED_CLASS_VOXELS = 500_000
MINIMUM_PARALLEL_SEPARATE_Z_CLASSES = 8
MINIMUM_PARALLEL_SEPARATE_Z_WORK = 4_000_000

# SciPy uses float64 interpolation products.
# Match that on Metal with double-single math.
_DOUBLE_SINGLE_HEADER = (
    DOUBLE_SINGLE_HEADER
    + r"""
inline DoubleSingle weighted_value(
    float value,
    DoubleSingle weight_z,
    DoubleSingle weight_y,
    DoubleSingle weight_x) {
    DoubleSingle result = {value, 0.0f};
    result = ds_multiply(result, weight_z);
    result = ds_multiply(result, weight_y);
    return ds_multiply(result, weight_x);
}

// Round to the nearest half, ties to even, as upstream's float64-to-float16
// store does. Halfway points between halves are float32 values, so the low
// part can change the result only when the high part lies exactly on one.
inline float ds_round_to_half(DoubleSingle value) {
    float high = value.high + value.low;
    float low = value.low - (high - value.high);
    float rounded = float(half(high));
    float other = rounded + 2.0f * (high - rounded);
    if (high != rounded && low != 0.0f && float(half(other)) == other
            && (low > 0.0f) == (high > rounded)) {
        return other;
    }
    return rounded;
}
"""
)

_FUSED_ARGMAX_SOURCE = r"""
    uint element = thread_position_in_grid.x;
    uint output_size = uint(parameters[7]);
    if (element >= output_size) return;

    int classes = parameters[0];
    int source_z = parameters[1];
    int source_y = parameters[2];
    int source_x = parameters[3];
    int target_y = parameters[5];
    int target_x = parameters[6];
    int z_offset = parameters[8];
    int y_offset = parameters[9];
    int x_offset = parameters[10];

    int x = int(element % uint(target_x));
    int y = int((element / uint(target_x)) % uint(target_y));
    int z = int(element / uint(target_x * target_y));

    int z_index = z_offset + z;
    int y_index = y_offset + y;
    int x_index = x_offset + x;
    int z0 = lower_indices[z_index];
    int y0 = lower_indices[y_index];
    int x0 = lower_indices[x_index];
    int z1 = upper_indices[z_index];
    int y1 = upper_indices[y_index];
    int x1 = upper_indices[x_index];
    DoubleSingle wz0 = {lower_weights_high[z_index], lower_weights_low[z_index]};
    DoubleSingle wy0 = {lower_weights_high[y_index], lower_weights_low[y_index]};
    DoubleSingle wx0 = {lower_weights_high[x_index], lower_weights_low[x_index]};
    DoubleSingle wz1 = {upper_weights_high[z_index], upper_weights_low[z_index]};
    DoubleSingle wy1 = {upper_weights_high[y_index], upper_weights_low[y_index]};
    DoubleSingle wx1 = {upper_weights_high[x_index], upper_weights_low[x_index]};

    long plane = long(source_y) * long(source_x);
    // Channel-first CXYZ logits, or channels-last XYZC logits kept on Metal.
    long channel_stride = CHANNELS_LAST ? 1 : long(source_z) * plane;
    long voxel_stride = CHANNELS_LAST ? long(classes) : 1;
    long i000 = (long(z0) * plane + long(y0) * source_x + x0) * voxel_stride;
    long i001 = (long(z0) * plane + long(y0) * source_x + x1) * voxel_stride;
    long i010 = (long(z0) * plane + long(y1) * source_x + x0) * voxel_stride;
    long i011 = (long(z0) * plane + long(y1) * source_x + x1) * voxel_stride;
    long i100 = (long(z1) * plane + long(y0) * source_x + x0) * voxel_stride;
    long i101 = (long(z1) * plane + long(y0) * source_x + x1) * voxel_stride;
    long i110 = (long(z1) * plane + long(y1) * source_x + x0) * voxel_stride;
    long i111 = (long(z1) * plane + long(y1) * source_x + x1) * voxel_stride;

    float best = -INFINITY;
    ushort label = 0;
    for (int channel = 0; channel < classes; ++channel) {
        long offset = long(channel) * channel_stride;
        DoubleSingle sum = {0.0f, 0.0f};
        sum = ds_add(sum, weighted_value(logits[offset + i000], wz0, wy0, wx0));
        sum = ds_add(sum, weighted_value(logits[offset + i001], wz0, wy0, wx1));
        sum = ds_add(sum, weighted_value(logits[offset + i010], wz0, wy1, wx0));
        sum = ds_add(sum, weighted_value(logits[offset + i011], wz0, wy1, wx1));
        sum = ds_add(sum, weighted_value(logits[offset + i100], wz1, wy0, wx0));
        sum = ds_add(sum, weighted_value(logits[offset + i101], wz1, wy0, wx1));
        sum = ds_add(sum, weighted_value(logits[offset + i110], wz1, wy1, wx0));
        sum = ds_add(sum, weighted_value(logits[offset + i111], wz1, wy1, wx1));
        float value = ROUND_TO_HALF ? ds_round_to_half(sum) : sum.high + sum.low;
        if (value > best) {
            best = value;
            label = ushort(channel);
        }
    }
    output[element] = label;
"""


@dataclass(frozen=True, slots=True)
class FusedRestoreSupport:
    """Whether the fused path is admitted and, if not, the reason it was declined."""

    eligible: bool
    reason: str | None = None


def assess_fused_class_restore(
    logits: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> FusedRestoreSupport:
    """Return the strict support decision without attempting the Metal kernel.

    The supported case is finite float32 or float16 CXYZ class logits, order-1
    interpolation, and no separate-axis resampling when the shape changes. This checks numerical
    eligibility, not the workload-size threshold used for automatic dispatch.
    Invalid target shape, spacing, index range or non-finite logits raise
    ``ContractError`` rather than returning an ineligible result.
    """

    values = np.asarray(logits)
    target_shape = _target_shape(new_shape)
    if (
        values.ndim != 4
        or int(values.shape[0]) < 1
        or any(int(value) < 1 for value in values.shape[1:])
    ):
        return FusedRestoreSupport(False, "fused restoration requires non-empty CXYZ logits")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        return FusedRestoreSupport(False, "fused restoration requires float32 or float16 logits")
    if int(values.shape[0]) > np.iinfo(np.uint16).max + 1:
        return FusedRestoreSupport(False, "fused restoration exceeds the uint16 label range")
    if contract.function_name != "resample_data_or_seg_to_shape" or contract.order != 1:
        return FusedRestoreSupport(False, "fused restoration requires pinned order-1 resampling")
    _validate_index_range(values.shape, target_shape)
    source_spacing = validate_spacing(current_spacing, name="current spacing")
    target_spacing = validate_spacing(new_spacing, name="new spacing")
    if tuple(int(value) for value in values.shape[1:]) != target_shape:
        separate, _ = separate_z_axis(
            source_spacing,
            target_spacing,
            force=contract.force_separate_z,
        )
        if separate:
            return FusedRestoreSupport(False, "fused restoration does not support separate-Z")
    if _finite_magnitude(values) > _MAXIMUM_FUSED_MAGNITUDE:
        return FusedRestoreSupport(
            False, "fused restoration logits exceed the stable magnitude range"
        )
    return FusedRestoreSupport(True)


def can_use_fused_class_restore(
    logits: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> FusedRestoreSupport:
    """Return semantic support plus the measured automatic-selection threshold."""

    support = assess_fused_class_restore(
        logits,
        new_shape,
        current_spacing,
        new_spacing,
        contract,
    )
    if not support.eligible:
        return support
    values = np.asarray(logits)
    target_voxels = math.prod(_target_shape(new_shape))
    if int(values.shape[0]) * target_voxels < MINIMUM_FUSED_CLASS_VOXELS:
        return FusedRestoreSupport(
            False, "fused restoration is below the qualified performance threshold"
        )
    return support


def resample_class_logits_argmax_optimized(
    logits: HostArray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> npt.NDArray[np.uint16]:
    """Use the admitted fused Metal path, or a host reference restoration.

    Ineligible separate-axis cases may use the parallel host implementation when
    its class-count and work thresholds are met; other cases use the streaming path.
    """

    support = can_use_fused_class_restore(
        logits,
        new_shape,
        current_spacing,
        new_spacing,
        contract,
    )
    if support.eligible:
        return _run_metal_kernel(np.asarray(logits), _target_shape(new_shape))
    # The fused and plane-streaming kernels reproduce the spline resampler's
    # coordinates. Torch-style plans require their own host interpolation grid.
    if contract.function_name == "resample_torch_fornnunet":
        return resample_class_logits_argmax(
            logits, new_shape, current_spacing, new_spacing, contract,
        )
    values = np.asarray(logits)
    separate_z, separate_axis = separate_z_axis(
        validate_spacing(current_spacing, name="current spacing"),
        validate_spacing(new_spacing, name="new spacing"),
        force=contract.force_separate_z,
    )
    target_shape = _target_shape(new_shape)
    parallel_work = (
        0
        if separate_axis is None
        else (
            int(values.shape[0])
            * int(values.shape[separate_axis + 1])
            * math.prod(target_shape[index] for index in range(3) if index != separate_axis)
        )
    )
    if (
        separate_z
        and int(values.shape[0]) >= MINIMUM_PARALLEL_SEPARATE_Z_CLASSES
        and parallel_work >= MINIMUM_PARALLEL_SEPARATE_Z_WORK
    ):
        return resample_class_logits_argmax_parallel(
            values,
            target_shape,
            current_spacing,
            new_spacing,
            contract,
        )
    return resample_class_logits_argmax(
        logits,
        new_shape,
        current_spacing,
        new_spacing,
        contract,
    )


def resample_class_logits_argmax_metal(
    logits: HostArray,
    new_shape: Sequence[int],
) -> npt.NDArray[np.uint16]:
    """Run synchronized edge-clamped trilinear interpolation and argmax on Metal.

    Input is finite float32 or float16 CXYZ with at most 65536 classes; ``new_shape``
    is a three-element target spatial shape. Returns a host uint16 target-grid
    array; ties select the first class. Interpolated float16 logits are rounded to
    float16 before the argmax, as upstream's resampling of half logits does. This
    direct entry point does not apply spacing or separate-axis policy, which
    callers must establish separately.
    """

    values = np.asarray(logits)
    target_shape = _target_shape(new_shape)
    if (
        values.ndim != 4
        or int(values.shape[0]) < 1
        or any(int(value) < 1 for value in values.shape[1:])
        or values.dtype not in (np.dtype(np.float32), np.dtype(np.float16))
    ):
        raise ContractError("fused restoration requires non-empty float32 or float16 CXYZ logits")
    if int(values.shape[0]) > np.iinfo(np.uint16).max + 1:
        raise ContractError("fused restoration exceeds the uint16 label range")
    if _finite_magnitude(values) > _MAXIMUM_FUSED_MAGNITUDE:
        raise ContractError("fused restoration logits exceed the stable magnitude range")
    _validate_index_range(values.shape, target_shape)

    return _run_metal_kernel(values, target_shape)


def _run_metal_kernel(
    values: HostArray,
    target_shape: tuple[int, int, int],
) -> npt.NDArray[np.uint16]:
    """Launch after the caller has validated the fused-path contract."""

    mx, _ = import_mlx()
    output = _launch(mx.array(values), int(values.shape[0]), target_shape, channels_last=False)
    mx.eval(output)
    return np.ascontiguousarray(np.asarray(output, dtype=np.uint16))


def resample_device_logits_argmax(
    logits: array,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    contract: ResamplingContract,
) -> npt.NDArray[np.uint16] | None:
    """Restore labels from Metal-resident channels-last ``(X,Y,Z,C)`` float32/16 logits.

    Returns ``None`` for unsupported dtype, non-scalar rank other than four, class
    count, magnitude, or resampling policy so the caller can use host restoration.
    A scalar lacks a channel dimension and is not admitted. It does not apply the host-transfer
    performance threshold. Equal source and target shapes need
    no resampling, so identity interpolation reduces to first-maximum argmax
    without checking the resampling policy or spacing. The caller must supply
    a nonempty array and has already rejected non-finite logits. Float16 logits
    are rounded to float16 after interpolation, as on the host.
    """
    import mlx.core as core

    mx, _ = import_mlx()
    target_shape = _target_shape(new_shape)
    classes = int(logits.shape[-1])
    if (
        logits.ndim != 4
        or logits.dtype not in (mx.float32, mx.float16)
        or classes > np.iinfo(np.uint16).max + 1
    ):
        return None
    source_shape = tuple(int(value) for value in logits.shape[:3])
    if source_shape != target_shape:
        if contract.function_name != "resample_data_or_seg_to_shape" or contract.order != 1:
            return None
        separate, _ = separate_z_axis(
            validate_spacing(current_spacing, name="current spacing"),
            validate_spacing(new_spacing, name="new spacing"),
            force=contract.force_separate_z,
        )
        if separate:
            return None
    _validate_index_range((classes, *source_shape), target_shape)
    stable = mx.all(core.abs(logits) <= _MAXIMUM_FUSED_MAGNITUDE)
    mx.eval(stable)
    if not bool(stable.item()):
        return None
    output = _launch(logits, classes, target_shape, channels_last=True)
    mx.eval(output)
    return np.ascontiguousarray(np.asarray(output, dtype=np.uint16))


def _launch(
    logits: array, classes: int, target_shape: tuple[int, int, int], *, channels_last: bool,
) -> array:
    """Launch fused class-logit resampling and label reduction for the target grid."""
    mx, _ = import_mlx()
    round_to_half = logits.dtype == mx.float16
    source_shape = cast(
        "tuple[int, int, int]",
        tuple(int(value) for value in (logits.shape[:3] if channels_last else logits.shape[1:])),
    )
    output_size = int(np.prod(target_shape, dtype=np.int64))
    coordinate_arrays, offsets = _coordinate_metadata(source_shape, target_shape)
    parameters = mx.array(
        [classes, *source_shape, *target_shape, output_size, *offsets],
        dtype=mx.int32,
    )
    return _metal_kernel()(
        inputs=[logits, parameters, *(mx.array(item) for item in coordinate_arrays)],
        template=[("CHANNELS_LAST", channels_last), ("ROUND_TO_HALF", round_to_half)],
        grid=(output_size, 1, 1),
        threadgroup=(min(_THREADGROUP_SIZE, output_size), 1, 1),
        output_shapes=[target_shape],
        output_dtypes=[mx.uint16],
    )[0]


def _finite_magnitude(values: HostArray) -> float:
    """Require finite float32 logits and return their maximum absolute magnitude."""
    float_values = np.asarray(values, dtype=np.float32)
    minimum = float(float_values.min())
    maximum = float(float_values.max())
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ContractError("fused restoration requires finite float32 logits")
    return max(abs(minimum), abs(maximum))


@lru_cache(maxsize=1)
def _metal_kernel() -> MetalKernel:
    """Build the fused class restoration and argmax Metal kernel."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name=_KERNEL_NAME,
        input_names=[
            "logits",
            "parameters",
            "lower_indices",
            "upper_indices",
            "lower_weights_high",
            "lower_weights_low",
            "upper_weights_high",
            "upper_weights_low",
        ],
        output_names=["output"],
        header=_DOUBLE_SINGLE_HEADER,
        source=_FUSED_ARGMAX_SOURCE,
    )


def _coordinate_metadata(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[tuple[npt.NDArray[np.generic], ...], tuple[int, int, int]]:
    """Prepare per-axis interpolation metadata for source and target grids."""
    metadata = tuple(
        _axis_metadata(source, target)
        for source, target in zip(source_shape, target_shape, strict=True)
    )
    arrays = tuple(np.concatenate(tuple(axis[item] for axis in metadata)) for item in range(6))
    offsets = (0, target_shape[0], target_shape[0] + target_shape[1])
    return arrays, offsets


def _axis_metadata(
    source: int, target: int
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Construct interpolation indices and weights for one spatial axis."""
    coordinates = (np.arange(target, dtype=np.float64) + 0.5) * (source / target) - 0.5
    coordinates = np.clip(coordinates, 0, source - 1)
    lower = np.floor(coordinates).astype(np.int32)
    upper = np.minimum(lower + 1, source - 1).astype(np.int32)
    upper_weight = coordinates - lower
    lower_weight = 1.0 - upper_weight
    lower_high, lower_low = _split_float64(lower_weight)
    upper_high, upper_low = _split_float64(upper_weight)
    return lower, upper, lower_high, lower_low, upper_high, upper_low


def _split_float64(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Represent float64 coordinates as high and residual float32 components."""
    high = values.astype(np.float32)
    low = (values - high.astype(np.float64)).astype(np.float32)
    return high, low


def _target_shape(new_shape: Sequence[int]) -> tuple[int, int, int]:
    """Require a three-dimensional positive restoration target shape."""
    result = tuple(int(value) for value in new_shape)
    if len(result) != 3 or any(value < 1 for value in result):
        raise ContractError("fused restoration target shape must contain three positive values")
    return result


def _validate_index_range(source_shape: Sequence[int], target_shape: tuple[int, int, int]) -> None:
    """Reject restoration dimensions and index products outside int32 limits."""
    limit = np.iinfo(np.int32).max
    dimensions = tuple(int(value) for value in source_shape)
    target_plane = target_shape[1] * target_shape[2]
    target_volume = target_shape[0] * target_plane
    if any(value > limit for value in (*dimensions, *target_shape)):
        raise ContractError("fused restoration dimensions exceed the int32 kernel range")
    if target_plane > limit or target_volume > limit:
        raise ContractError("fused restoration target volume exceeds the int32 kernel range")
