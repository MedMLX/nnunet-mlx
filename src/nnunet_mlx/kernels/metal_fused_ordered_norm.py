"""Queueable ordered statistics and fused affine/LeakyReLU for CT inference.

The variance kernel preserves separate float32 subtraction and multiplication
before the existing double-single reduction. Outputs keep logical NHWC/NDHWC
axes with dense channels-first storage, or an explicitly padded NCDHW array.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.kernels.types import MetalKernel


_VARIANCE_PARTIAL_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= B * P * C) return;
    uint c = index % C;
    uint p = (index / C) % P;
    uint b = index / (P * C);
    DoubleSingle sum = {0.0f, 0.0f};
    for (uint i = p * 128; i < min((p+1)*128, uint(S)); ++i) {
        // Preserve the two float32 roundings before double-single accumulation.
        volatile float centered = values[(b*S+i)*C+c] - average[b*C+c];
        volatile float square = centered * centered;
        sum = ds_add(sum, {square, 0.0f});
    }
    sums[index*2] = sum.high;
    sums[index*2+1] = sum.low;
"""


_AFFINE_ACTIVATION_SOURCE = r"""
uint tid = thread_position_in_threadgroup.x;
uint spatial0 = threadgroup_position_in_grid.y * 32;
uint channel0 = (threadgroup_position_in_grid.z % ((C+31)/32)) * 32;
uint b = threadgroup_position_in_grid.z / ((C+31)/32);
threadgroup float tile[32 * 33];
for (uint i = tid; i < 1024; i += 256) {
    uint s = spatial0 + i/32;
    uint c = channel0 + i%32;
    float value = 0;
    if (s < S && c < C) {
        uint j = b*C+c;
        float alpha = invstd[j]*weight[c];
        float beta = metal::fma(-mean[j], alpha, bias[c]);
        if (OUTPUT_FMA) {
            value = metal::fma(values[(b*S+s)*C+c], alpha, beta);
        } else {
            volatile float product = values[(b*S+s)*C+c]*alpha;
            value = product+beta;
        }
        // Match MLX Maximum<float>: preserve the first NaN and take y on ties.
        volatile float leaky_product = slope[0]*value;
        value = metal::isnan(leaky_product) ? leaky_product :
            (leaky_product > value ? leaky_product : value);
    }
    tile[(i/32)*33+i%32] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 1024; i += 256) {
    uint c = channel0 + i/32;
    uint s = spatial0 + i%32;
    if (s < S && c < C) result[(b*C+c)*S+s] = tile[(i%32)*33+i/32];
}
"""


@lru_cache(maxsize=1)
def _variance_partial_kernel() -> MetalKernel:
    """Build the partial variance reduction kernel."""
    import mlx.core as mx

    from nnunet_mlx.kernels.metal_numeric import DOUBLE_SINGLE_HEADER

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_ordered_variance_partial",
            input_names=["values", "average"],
            output_names=["sums"],
            header=DOUBLE_SINGLE_HEADER,
            source=_VARIANCE_PARTIAL_SOURCE,
            compile_options={"math_mode": "safe"},
        ),
    )


def spatial_statistics(values: array, eps: float) -> tuple[array, array]:
    """Return per-batch, per-channel mean and inverse standard deviation.

    Accept nonempty float32 NHWC or NDHWC values with at least two spatial
    elements and a finite positive ``eps``. Both outputs have broadcast shape
    ``(N, 1, ..., 1, C)``; the variance is population variance. Squared values
    are reduced in partitions without materializing a full squared array.
    """
    import mlx.core as mx

    from nnunet_mlx.errors import ContractError
    from nnunet_mlx.kernels.metal_instance_norm import (
        inverse_standard_deviation,
        spatial_mean,
        statistics_kernels,
    )

    if values.ndim not in (4, 5) or values.dtype != mx.float32 or any(s < 1 for s in values.shape):
        raise ContractError("ordered normalization requires nonempty float32 NHWC or NDHWC input")
    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    if spatial < 2:
        raise ContractError("ordered normalization requires spatial size >= 2")
    average = spatial_mean(values)
    partitions = (spatial + 127) // 128
    template = [
        ("B", batch), ("S", spatial), ("C", channels), ("P", partitions), ("ROUND_SUM", True),
    ]
    partial = _variance_partial_kernel()(
        inputs=[values, average],
        template=template,
        grid=(batch * partitions * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, partitions, channels, 2)],
        output_dtypes=[mx.float32],
    )[0]
    _, reduce_kernel, _, _ = statistics_kernels()
    variance = reduce_kernel(
        inputs=[partial],
        template=template,
        grid=(batch * channels * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[average.shape],
        output_dtypes=[mx.float32],
    )[0]
    return average, inverse_standard_deviation(variance, eps)


@lru_cache(maxsize=1)
def _affine_activation_kernel() -> MetalKernel:
    """Build the fused normalization affine and activation kernel."""
    import mlx.core as mx

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_ordered_affine_activation",
            input_names=["values", "mean", "invstd", "weight", "bias", "slope"],
            output_names=["result"], source=_AFFINE_ACTIVATION_SOURCE,
            compile_options={"math_mode": "safe"},
        ),
    )


@lru_cache(maxsize=16)
def _slope_parameter(slope: float) -> array:
    """Materialize the activation slope as a float32 MLX array."""
    import mlx.core as mx

    return mx.array([slope], dtype=mx.float32)


def affine_activation_from_statistics(
    values: array, average: array, inverse: array, weight: array, bias: array,
    negative_slope: float, *, output_fma: bool = False,
) -> array:
    """Apply ordered affine and LeakyReLU to channels-last values.

    ``average`` and ``inverse`` contain one statistic per batch and channel;
    ``weight`` and ``bias`` contain one float32 value per channel. These shapes
    are caller preconditions, not validated here. ``output_fma`` selects fused
    multiply-add for the final affine operation. The returned logical NHWC/NDHWC
    array has dense channels-first storage.
    """
    import mlx.core as mx

    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    result = _affine_activation_kernel()(
        inputs=[values, average, inverse, weight, bias, _slope_parameter(negative_slope)],
        template=[("B", batch), ("S", spatial), ("C", channels), ("OUTPUT_FMA", output_fma)],
        grid=(256, (spatial + 31) // 32, batch * ((channels + 31) // 32)),
        threadgroup=(256, 1, 1), output_shapes=[(batch, channels, *values.shape[1:-1])],
        output_dtypes=[mx.float32],
    )[0]
    # Dense channels-first storage avoids the next ordered convolution's input copy.
    return mx.transpose(result, (0, *range(2, result.ndim), 1))


def instance_norm_activation(
    values: array, weight: array, bias: array, eps: float, negative_slope: float,
    *, output_fma: bool = False,
) -> array:
    """Fuse ordered affine and LeakyReLU, returning a logical NHWC/NDHWC view.

    The output uses dense channels-first storage to match the next ordered
    convolution. The arithmetic follows the pinned engine's statistics and
    affine order followed by MLX's LeakyReLU; parity is qualified by the
    selected numerical tests, rather than guaranteed for arbitrary inputs.
    """
    average, inverse = spatial_statistics(values, eps)
    return affine_activation_from_statistics(
        values, average, inverse, weight, bias, negative_slope, output_fma=output_fma,
    )


_AFFINE_PAD_SOURCE = r"""
uint tid = thread_position_in_threadgroup.x;
uint p0 = threadgroup_position_in_grid.y * 32;
uint c0 = (threadgroup_position_in_grid.z % ((C + 31) / 32)) * 32;
uint b = threadgroup_position_in_grid.z / ((C + 31) / 32);
threadgroup float tile[32 * 33];
for (uint i = tid; i < 1024; i += 256) {
    uint p = p0 + i / 32;
    uint c = c0 + i % 32;
    int z = int(p / (HP * WP)) - PD;
    int y = int((p / WP) % HP) - PH;
    int x = int(p % WP) - PW;
    float value = 0.0f;
    if (p < DP * HP * WP && c < C && z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) {
        uint s = (uint(z) * H + uint(y)) * W + uint(x);
        uint j = b * C + c;
        float alpha = invstd[j] * weight[c];
        float beta = metal::fma(-mean[j], alpha, bias[c]);
        if (OUTPUT_FMA) {
            value = metal::fma(values[(b * S + s) * C + c], alpha, beta);
        } else {
            volatile float product = values[(b * S + s) * C + c] * alpha;
            value = product + beta;
        }
        volatile float leaky_product = slope[0] * value;
        value = metal::isnan(leaky_product) ? leaky_product :
            (leaky_product > value ? leaky_product : value);
    }
    // Halo values stay positive zero without receiving affine or activation.
    tile[(i / 32) * 33 + i % 32] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 1024; i += 256) {
    uint c = c0 + i / 32;
    uint p = p0 + i % 32;
    if (c < C && p < DP * HP * WP)
        packed[(b * C + c) * (DP * HP * WP) + p] = tile[(i % 32) * 33 + i / 32];
}
"""


@lru_cache(maxsize=1)
def _affine_pad_kernel() -> MetalKernel:
    """Build the fused normalization affine and padding kernel."""
    import mlx.core as mx

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_ordered_affine_activation_pad",
            input_names=["values", "mean", "invstd", "weight", "bias", "slope"],
            output_names=["packed"], source=_AFFINE_PAD_SOURCE,
            compile_options={"math_mode": "safe"},
        ),
    )


def affine_activation_padded_from_statistics(
    values: array, average: array, inverse: array, weight: array, bias: array,
    negative_slope: float, padding: tuple[int, ...], *, output_fma: bool = False,
) -> array:
    """Apply affine and LeakyReLU to NDHWC values into padded NCDHW storage.

    ``padding`` supplies nonnegative symmetric padding for each spatial axis;
    the halo is positive zero. Statistics and channel parameter shapes are
    caller preconditions as in ``affine_activation_from_statistics``.
    """
    import mlx.core as mx

    from nnunet_mlx.errors import ContractError

    batch, depth, height, width, channels = values.shape
    pd, ph, pw = padding
    if min(padding) < 0:
        raise ContractError("affine padding requires nonnegative padding")
    dp, hp, wp = depth + 2 * pd, height + 2 * ph, width + 2 * pw
    return _affine_pad_kernel()(
        inputs=[values, average, inverse, weight, bias, _slope_parameter(negative_slope)],
        template=list(dict(
            B=batch, D=depth, H=height, W=width, C=channels, S=depth * height * width,
            DP=dp, HP=hp, WP=wp, PD=pd, PH=ph, PW=pw, OUTPUT_FMA=output_fma,
        ).items()),
        grid=(256, (dp * hp * wp + 31) // 32, batch * ((channels + 31) // 32)),
        threadgroup=(256, 1, 1), output_shapes=[(batch, channels, dp, hp, wp)],
        output_dtypes=[mx.float32],
    )[0]
