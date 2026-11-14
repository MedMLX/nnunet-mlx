"""Torch-grid resampling with coordinates and label scores local to each Metal thread."""

from __future__ import annotations

from functools import cache
from math import prod
from typing import TYPE_CHECKING

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_numeric import DOUBLE_SINGLE_HEADER
from nnunet_mlx.plans import InterpolationMode, ResamplingContract, SegmentationTieBreak
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.kernels.types import MetalKernel

# NumPy rounds the coordinate multiply before subtracting 0.5. Contraction
# would change sample locations, and reassociation breaks double-single math.
_HEADER = "#pragma clang fp contract(off)\n" + DOUBLE_SINGLE_HEADER + r"""
// Metal flushes floating subnormal operands. Decode their integer mantissas
// before scaling; power-of-two normalization also protects compensated sums
// near FLT_MAX from overflowing their temporary high components.
inline float scaled_sample(float value, int exponent) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0) {
        float mantissa = float(bits & 0x7fffffu);
        return metal::ldexp((bits >> 31) ? -mantissa : mantissa, -149 - exponent);
    }
    return metal::ldexp(value, -exponent);
}

inline float restored_sample(DoubleSingle value, int exponent) {
    float rounded = value.high + value.low;
    uint bits = as_type<uint>(rounded);
    if ((bits & 0x7fffffffu) == 0) return rounded;
    int output_exponent = int((bits >> 23) & 255) + exponent;
    if (output_exponent > 0) return metal::ldexp(rounded, exponent);
    // Store subnormals as bits, rounding the compensated value to a mantissa
    // once instead of allowing Metal arithmetic to flush the final result.
    float sign = (bits >> 31) ? -1.0f : 1.0f;
    float high = metal::ldexp(value.high * sign, exponent + 149);
    float low = metal::ldexp(value.low * sign, exponent + 149);
    float integer = metal::rint(high);
    if (metal::abs(high - integer) == 0.5f && low != 0.0f)
        integer = low > 0.0f ? metal::ceil(high) : metal::floor(high);
    return as_type<float>((bits & 0x80000000u) | uint(integer));
}
"""
_SOURCE = r"""
ulong element = ulong(thread_position_in_grid.x) + ulong(thread_position_in_grid.y) * ROW;
if (element >= N) return;
uint position[3] = {uint(element / (ulong(OY) * OZ) % OX),
                    uint(element / OZ % OY), uint(element % OZ)};
uint sizes[3] = {IX, IY, IZ};
ulong strides[3] = {ulong(IY) * IZ, IZ, 1};
ulong base = (element / (ulong(OX) * OY * OZ)) * (ulong(IX) * IY * IZ);
uint lower[3], upper[3];
DoubleSingle weights[3][2];
ulong nearest = base;
for (uint axis = 0; axis < 3; ++axis) {
    float shifted = float(position[axis]) + (MODE == 1 ? 0.0f : 0.5f);
    float scaled = scales[axis] * shifted;
    uint index = min(uint(scaled), sizes[axis] - 1);
    nearest += index * strides[axis];
    if constexpr (MODE == 0) {
        float coordinate = max(scaled - 0.5f, 0.0f);
        lower[axis] = min(uint(coordinate), sizes[axis] - 1);
        upper[axis] = min(lower[axis] + 1, sizes[axis] - 1);
        float fraction = coordinate - metal::floor(coordinate);
        weights[axis][0] = ds_add({1.0f, 0.0f}, {-fraction, 0.0f});
        weights[axis][1] = {fraction, 0.0f};
    }
}
if constexpr (MODE != 0) {
    output[element] = data[nearest];
    return;
}
float samples[8];
DoubleSingle products[8];
DoubleSingle sum = {0.0f, 0.0f};
uint maximum_exponent = 0;
for (uint corner = 0; corner < 8; ++corner) {
    ulong index = base;
    for (uint axis = 0; axis < 3; ++axis) {
        uint side = (corner >> (2 - axis)) & 1;
        index += (side ? upper[axis] : lower[axis]) * strides[axis];
    }
    samples[corner] = float(data[index]);
    maximum_exponent = max(maximum_exponent, (as_type<uint>(samples[corner]) >> 23) & 255);
}
int exponent = int(max(maximum_exponent, 1u)) - 127;
for (uint corner = 0; corner < 8; ++corner) {
    // SciPy multiplies the input value by each axis weight in axis order.
    DoubleSingle term = {SEG ? 1000.0f : scaled_sample(samples[corner], exponent), 0.0f};
    for (uint axis = 0; axis < 3; ++axis)
        term = ds_multiply(term, weights[axis][(corner >> (2 - axis)) & 1]);
    products[corner] = term;
    if constexpr (!SEG) sum = ds_add(sum, term);
}
if constexpr (!SEG) {
    output[element] = restored_sample(sum, exponent);
} else {
    float best = -1.0f;
    int winner = 0;
    int nearest_label = int(data[nearest]);
    // Only labels in this stencil can have a positive score. At least one
    // does, so globally present labels with zero support cannot win.
    for (uint candidate = 0; candidate < 8; ++candidate) {
        int label = int(samples[candidate]);
        bool seen = false;
        for (uint previous = 0; previous < candidate; ++previous)
            seen |= samples[previous] == samples[candidate];
        if (seen) continue;
        DoubleSingle score = {0.0f, 0.0f};
        for (uint corner = 0; corner < 8; ++corner)
            if (samples[corner] == samples[candidate]) score = ds_add(score, products[corner]);
        // Preserve the original float64 -> float32 -> float16 score stores.
        float rounded = float(half(score.high + score.low));
        bool tie = rounded == best;
        bool preferred = TIE == 1 ? label > winner : label < winner;
        if constexpr (TIE == 2)
            preferred = label == nearest_label || (winner != nearest_label && label < winner);
        if (rounded > best || (tie && preferred)) {
            best = rounded;
            winner = label;
        }
    }
    output[element] = winner;
}
"""


@cache
def _kernel() -> MetalKernel:
    """Compile once with IEEE-safe arithmetic and explicit noncontracted coordinates."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name="nnunet_mlx_torch_resize", input_names=["data", "scales"],
        output_names=["output"], source=_SOURCE, header=_HEADER,
        compile_options={"math_mode": "safe"},
    )


def torch_resize(
    data: array, shape: tuple[int, ...], mode: InterpolationMode, *,
    is_seg: bool, tiebreak: SegmentationTieBreak,
) -> array:
    """Return lazy CXY/CXYZ samples on Metal without spatial work buffers.

    Image inputs are finite float32; integer labels are in [-1, 32767]. The
    caller validates values. Linear sampling uses float32 half-pixel coordinates,
    edge replication and compensated products before a float32 store. Label
    scores additionally round through float16 after scale-1000 interpolation.
    Noncontiguous inputs are packed once by MLX; output retains the input dtype.
    Built-in Upsample uses a different coordinate expression and rounding order.
    """
    import mlx.core as mx

    import_mlx()
    if len(shape) not in (2, 3) or data.ndim != len(shape) + 1:
        raise ContractError("Torch Metal resampling requires CXY or CXYZ arrays")
    if min(*data.shape, *shape) < 1:
        raise ContractError("Torch Metal resampling requires positive dimensions")
    if mode not in ("linear", "nearest", "nearest-exact"):
        raise ContractError("unsupported Torch Metal interpolation mode")
    if tiebreak not in ("lowest", "highest", "nearest"):
        raise ContractError("unsupported Torch Metal label tie rule")
    if (not is_seg and data.dtype != mx.float32) or (is_seg and data.dtype not in (
        mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32, mx.uint64,
    )):
        raise ContractError("Torch Metal resampling requires float32 images or integer labels")
    if data.shape[1:] == shape:
        return data
    source = (1,) * (3 - len(shape)) + data.shape[1:]
    target = (1,) * (3 - len(shape)) + shape
    if max(*source, *target) >= 2**31:
        raise ContractError("Torch Metal resampling exceeds supported axis indexing")
    size = data.shape[0] * prod(shape)
    if size >= 2**31:
        raise ContractError("Torch Metal resampling exceeds supported output indexing")
    row = min(size, 32768)
    scales = mx.array([old / new for old, new in zip(source, target, strict=True)], mx.float32)
    return _kernel()(
        inputs=[data, scales], output_shapes=[(data.shape[0], *shape)],
        output_dtypes=[data.dtype], grid=(row, (size + row - 1) // row, 1),
        threadgroup=(min(row, 256), 1, 1),
        template=[("N", size), ("ROW", row), ("IX", source[0]), ("IY", source[1]),
                  ("IZ", source[2]), ("OX", target[0]), ("OY", target[1]), ("OZ", target[2]),
                  ("MODE", ("linear", "nearest", "nearest-exact").index(mode)),
                  ("SEG", is_seg), ("TIE", ("lowest", "highest", "nearest").index(tiebreak))],
        stream=mx.gpu,
    )[0]


def resample_torch(
    data: array, shape: tuple[int, int, int], *, axis: int | None,
    contract: ResamplingContract, is_seg: bool,
) -> array:
    """Keep both Torch resampling passes on Metal, deciding labels between passes."""
    import mlx.core as mx

    tiebreak = contract.seg_tiebreak or "nearest"
    if axis is None:
        return torch_resize(data, shape, contract.mode, is_seg=is_seg, tiebreak=tiebreak)
    planes = mx.moveaxis(data, axis + 1, 1)
    plane_shape = tuple(shape[index] for index in range(3) if index != axis)
    resized = torch_resize(
        mx.reshape(planes, (-1, *planes.shape[2:])), plane_shape, contract.mode,
        is_seg=is_seg, tiebreak=tiebreak,
    )
    intermediate = mx.moveaxis(mx.reshape(resized, (*planes.shape[:2], *plane_shape)), 1, axis + 1)
    return torch_resize(intermediate, shape, contract.aniso_axis_mode,
                        is_seg=is_seg, tiebreak=tiebreak)
