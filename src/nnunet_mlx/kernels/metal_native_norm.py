"""Metal instance normalization for native float32 inference.

Three bandwidth-bound passes replace the eager graph's materialized centered and
squared tensors: a spatial sum, a fused centered sum and sum of squares, and the
affine output. Each thread accumulates one channel over a 256-row spatial
partition, so float32 chains stay as short as the reduction they replace; the
partitions are then summed with MLX. ``torch-order-fp32`` keeps its own
reduction-order kernels in ``metal_instance_norm``.
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
    if (CENTERED) {
        float shift = mean[b * C + c];
        float sum = 0.0f;
        float squares = 0.0f;
        for (uint i = p * ROWS; i < end; ++i) {
            float centered = values[(b * S + i) * C + c] - shift;
            sum += centered;
            squares = metal::fma(centered, centered, squares);
        }
        sums[index * 2] = sum;
        sums[index * 2 + 1] = squares;
    } else {
        float sum = 0.0f;
        for (uint i = p * ROWS; i < end; ++i) sum += values[(b * S + i) * C + c];
        sums[index] = sum;
    }
"""

_AFFINE_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= B * S * C) return;
    uint c = i % C;
    uint j = (i / (S * C)) * C + c;
    result[i] = metal::fma(values[i] - mean[j], scale[j], bias[c]);
"""


@lru_cache(maxsize=1)
def _kernels() -> tuple[MetalKernel, MetalKernel]:
    """Build the native instance-normalization reduction and output kernels."""
    mx, _ = import_mlx()
    sums = mx.fast.metal_kernel(
        name="nnunet_native_norm_sums",
        input_names=["values", "mean"],
        output_names=["sums"],
        source=_SUMS_SOURCE,
    )
    affine = mx.fast.metal_kernel(
        name="nnunet_native_norm_affine",
        input_names=["values", "mean", "scale", "bias"],
        output_names=["result"],
        source=_AFFINE_SOURCE,
    )
    return sums, affine


def instance_norm(values: array, weight: array | None, bias: array | None, eps: float) -> array:
    """Normalize float32 channels-last ``values`` per sample and channel.

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
    sums_kernel, affine_kernel = _kernels()
    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    partitions = (spatial + _ROWS - 1) // _ROWS
    grid = (batch * partitions * channels, 1, 1)

    def partial_sums(mean: array, *, centered: bool) -> array:
        """Launch partial sums for raw values or values centered on the mean."""
        return sums_kernel(
            inputs=[values, mean],
            template=[
                ("B", batch),
                ("S", spatial),
                ("C", channels),
                ("P", partitions),
                ("ROWS", _ROWS),
                ("CENTERED", centered),
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
        inputs=[values, mean, scale, shift],
        template=[("B", batch), ("S", spatial), ("C", channels)],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
