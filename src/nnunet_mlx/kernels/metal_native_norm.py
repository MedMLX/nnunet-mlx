"""Metal instance normalization for native float32 inference.

Three bandwidth-bound passes replace the eager graph's materialized centered and
squared tensors: a spatial sum, a fused centered sum and sum of squares, and the
affine output. Each thread accumulates one channel over a 256-row spatial
partition, so float32 chains stay as short as the reduction they replace; the
partitions are then summed with MLX. ``torch-order-fp32`` keeps its own
reduction-order kernels in ``metal_instance_norm``.

A preceding convolution's bias and a following LeakyReLU can run inside these
kernels. The output kernel computes the affine result, adds an optional
residual tensor after it and then applies ``max(slope * r, r)``, the same
float32 operations as the separate bias addition, residual addition and
``mlx.nn.leaky_relu``. MLX compiles custom kernels in its default ``safe`` math
mode, so nothing is reassociated and the output is bitwise equal to the
unfused sequence while skipping full-tensor reads and writes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

_ROWS = 256

_SUMS_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= B * P * C) return;
    uint c = index % C;
    uint p = (index / C) % P;
    uint b = index / (P * C);
    uint end = min((p + 1) * ROWS, uint(S));
    float offset = INPUT_BIAS ? input_bias[c] : 0.0f;
    if (CENTERED) {
        float shift = mean[b * C + c];
        float sum = 0.0f;
        float squares = 0.0f;
        for (uint i = p * ROWS; i < end; ++i) {
            float value = values[(b * S + i) * C + c];
            float centered = (INPUT_BIAS ? value + offset : value) - shift;
            sum += centered;
            squares = metal::fma(centered, centered, squares);
        }
        sums[index * 2] = sum;
        sums[index * 2 + 1] = squares;
    } else {
        float sum = 0.0f;
        for (uint i = p * ROWS; i < end; ++i) {
            float value = values[(b * S + i) * C + c];
            sum += INPUT_BIAS ? value + offset : value;
        }
        sums[index] = sum;
    }
"""

_AFFINE_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= B * S * C) return;
    uint c = i % C;
    uint j = (i / (S * C)) * C + c;
    float value = INPUT_BIAS ? values[i] + input_bias[c] : values[i];
    float normalized = metal::fma(value - mean[j], scale[j], bias[c]);
    if (RESIDUAL) {
        normalized = normalized + residual[i];
    }
    result[i] = LEAKY ? metal::max(slope[0] * normalized, normalized) : normalized;
"""


@lru_cache(maxsize=1)
def _kernels() -> tuple[MetalKernel, MetalKernel]:
    """Build the native instance-normalization reduction and output kernels."""
    mx, _ = import_mlx()
    sums = mx.fast.metal_kernel(
        name="nnunet_native_norm_sums",
        input_names=["values", "mean", "input_bias"],
        output_names=["sums"],
        source=_SUMS_SOURCE,
    )
    affine = mx.fast.metal_kernel(
        name="nnunet_native_norm_affine",
        input_names=["values", "mean", "scale", "bias", "input_bias", "slope", "residual"],
        output_names=["result"],
        source=_AFFINE_SOURCE,
    )
    return sums, affine


def instance_norm(
    values: array,
    weight: array | None,
    bias: array | None,
    eps: float,
    *,
    input_bias: array | None = None,
    negative_slope: float | None = None,
    residual: array | None = None,
) -> array:
    """Normalize float32 channels-last ``values`` per sample and channel.

    ``input_bias`` (per channel) is added to ``values`` first, as a convolution
    bias would be; ``negative_slope`` applies LeakyReLU to the result.
    ``residual`` is added after the affine output and before LeakyReLU. It must
    be float32 with exactly the same shape as ``values``; no broadcasting is
    applied. When absent, its placeholder buffer is not read.

    The first pass gives a provisional mean ``m``. The second sums ``d = x - m``
    and ``d * d`` in one read; with ``D = mean(d)``, the corrected mean is
    ``m + D`` and the biased variance ``mean(d * d) - D * D``. This is the
    centered two-pass result without a third full read, and keeps the eager
    implementation's correction of the first float32 mean. Rounding can make the
    variance slightly negative for constant channels, so it is clamped at zero.
    Missing ``weight``/``bias`` mean a non-affine layer.
    """
    mx, _ = import_mlx()
    if values.dtype != mx.float32:
        raise ContractError("native instance normalization requires float32 input")
    if residual is not None:
        if residual.dtype != mx.float32:
            raise ContractError("native instance normalization requires float32 residual")
        if residual.shape != values.shape:
            raise ContractError(
                "native instance normalization residual must exactly match the values shape"
            )
    sums_kernel, affine_kernel = _kernels()
    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    partitions = (spatial + _ROWS - 1) // _ROWS
    grid = (batch * partitions * channels, 1, 1)
    # Unused inputs still need a buffer; the INPUT_BIAS, LEAKY and RESIDUAL flags
    # compile their reads out.
    offsets = mx.zeros((1,), dtype=mx.float32) if input_bias is None else input_bias
    slope = mx.array([0.0 if negative_slope is None else negative_slope], dtype=mx.float32)
    residual_values = slope if residual is None else residual

    def partial_sums(mean: array, *, centered: bool) -> array:
        """Launch partial sums for raw values or values centered on the mean."""
        return sums_kernel(
            inputs=[values, mean, offsets],
            template=[
                ("B", batch),
                ("S", spatial),
                ("C", channels),
                ("P", partitions),
                ("ROWS", _ROWS),
                ("CENTERED", centered),
                ("INPUT_BIAS", input_bias is not None),
            ],
            grid=grid,
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, partitions, channels, 2 if centered else 1)],
            output_dtypes=[mx.float32],
        )[0]

    unused = mx.zeros((batch, channels), dtype=mx.float32)
    provisional = mx.sum(partial_sums(unused, centered=False), axis=(1, 3)) / spatial
    totals = mx.sum(partial_sums(provisional, centered=True), axis=1) / spatial
    correction = totals[..., 0]
    mean = provisional + correction
    variance = mx.maximum(totals[..., 1] - correction * correction, 0.0)
    scale = mx.rsqrt(variance + eps)
    if weight is not None:
        scale = weight * scale
    shift = mx.zeros((channels,), dtype=mx.float32) if bias is None else bias
    return affine_kernel(
        inputs=[values, mean, scale, shift, offsets, slope, residual_values],
        template=[
            ("B", batch),
            ("S", spatial),
            ("C", channels),
            ("INPUT_BIAS", input_bias is not None),
            ("LEAKY", negative_slope is not None),
            ("RESIDUAL", residual is not None),
        ],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
