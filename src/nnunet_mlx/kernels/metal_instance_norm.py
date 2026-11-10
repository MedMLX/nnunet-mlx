"""Metal-only instance normalization with pinned Torch CPU rounding order."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_numeric import DOUBLE_SINGLE_HEADER
from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

_PARTIAL_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= B * P * C) return;
    uint c = index % C;
    uint p = (index / C) % P;
    uint b = index / (P * C);
    DoubleSingle sum = {0.0f, 0.0f};
    for (uint i = p * 128; i < min((p+1)*128, uint(S)); ++i)
        sum = ds_add(sum, {values[(b*S+i)*C+c], 0.0f});
    sums[index*2] = sum.high;
    sums[index*2+1] = sum.low;
"""

_REDUCE_SOURCE = r"""
uint lane=thread_position_in_threadgroup.x;
uint index=threadgroup_position_in_grid.x;
uint c=index%C,b=index/C;
DoubleSingle sum={0.0f,0.0f};
for(uint part=lane;part<P;part+=256){
 uint i=((b*P+part)*C+c)*2;
 sum=ds_add(sum,{sums[i],sums[i+1]});
}
threadgroup DoubleSingle shared[256];
shared[lane]=sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint step=128;step>0;step/=2){
 if(lane<step) shared[lane]=ds_add(shared[lane],shared[lane+step]);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(lane==0) {
    if (ROUND_SUM) {
        // CPU BatchNorm stores the variance sum in float32 before division.
        result[index]=(shared[0].high+shared[0].low)/float(S);
    } else {
        // Mean division retains the low part until the final float32 rounding.
        float quotient=shared[0].high/float(S);
        DoubleSingle remainder=ds_add(shared[0],ds_multiply({-quotient,0.0f},{float(S),0.0f}));
        result[index]=quotient+(remainder.high+remainder.low)/float(S);
    }
}
"""

_INVSTD_SOURCE = r"""
    uint i=thread_position_in_grid.x;
    if (i >= N) return;
    DoubleSingle v=ds_add({variance[i],0.0f},{epsilon[0],epsilon[1]});
    float r=metal::precise::rsqrt(v.high);
    DoubleSingle corrected={r,0.0f};
    for (int step=0; step<2; ++step) {
        DoubleSingle square=ds_multiply(corrected,corrected);
        DoubleSingle product=ds_multiply(v,square);
        DoubleSingle error=ds_add({1.0f,0.0f},{-product.high,-product.low});
        DoubleSingle correction=ds_multiply(ds_multiply(corrected,error),{0.5f,0.0f});
        corrected=ds_add(corrected,correction);
    }
    result[i]=corrected.high+corrected.low;
"""

_AFFINE_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= B*S*C) return;
    uint c = i%C;
    uint j = (i/(S*C))*C+c;
    float alpha = invstd[j]*weight[c];
    float beta = metal::fma(-mean[j], alpha, bias[c]);
    if (OUTPUT_FMA) {
        result[i] = metal::fma(values[i], alpha, beta);
    } else {
        volatile float product = values[i]*alpha;
        result[i] = product+beta;
    }
"""


@lru_cache(maxsize=1)
def statistics_kernels() -> tuple[MetalKernel, MetalKernel, MetalKernel, MetalKernel]:
    """Share the ordered partial, reduction, inverse, and affine kernel instances."""
    mx, _ = import_mlx()
    specs = (
        ("partial", ["values"], "sums", _PARTIAL_SOURCE),
        ("reduce", ["sums"], "result", _REDUCE_SOURCE),
        ("invstd", ["variance", "epsilon"], "result", _INVSTD_SOURCE),
        ("affine", ["values", "mean", "invstd", "weight", "bias"], "result", _AFFINE_SOURCE),
    )
    kernels = [
        mx.fast.metal_kernel(
            name=f"nnunet_ordered_norm_{name}",
            input_names=inputs,
            output_names=[output],
            header=DOUBLE_SINGLE_HEADER,
            source=source,
        )
        for name, inputs, output, source in specs
    ]
    return kernels[0], kernels[1], kernels[2], kernels[3]


def spatial_mean(values: array) -> array:
    """Reduce spatial axes with the retained 128-element double-single partitions.

    Callers must supply nonempty float32 channels-last values; this helper does
    not independently validate shape or dtype. The variance path reduces with
    the same kernels and ``ROUND_SUM``, which rounds the sum before division.
    """
    mx, _ = import_mlx()
    partial_kernel, reduce_kernel, _, _ = statistics_kernels()
    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    partitions = (spatial + 127) // 128
    template = [
        ("B", batch),
        ("S", spatial),
        ("C", channels),
        ("P", partitions),
        ("ROUND_SUM", False),
    ]
    partial = partial_kernel(
        inputs=[values],
        template=template,
        grid=(batch * partitions * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, partitions, channels, 2)],
        output_dtypes=[mx.float32],
    )[0]
    return reduce_kernel(
        inputs=[partial],
        template=template,
        grid=(batch * channels * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, *((1,) * (values.ndim - 2)), channels)],
        output_dtypes=[mx.float32],
    )[0]


def spatial_statistics(values: array, eps: float) -> tuple[array, array, array]:
    """Return float32 mean, biased variance, and corrected inverse standard deviation.

    The variance reduces float32-rounded squared deviations in the same order
    as ``spatial_mean`` of the materialized squares, without forming them.
    """
    from nnunet_mlx.kernels.metal_fused_ordered_norm import spatial_variance

    _validate_values(values)
    batch, channels = values.shape[0], values.shape[-1]
    spatial = values.size // (batch * channels)
    if spatial < 2:
        raise ContractError("ordered normalization requires spatial size >= 2")
    average = spatial_mean(values)
    variance = spatial_variance(values, average)
    return average, variance, inverse_standard_deviation(variance, eps)


def _validate_values(values: array) -> None:
    """Require nonempty float32 NHWC or NDHWC input for ordered normalization."""
    mx, _ = import_mlx()
    if values.ndim not in (4, 5) or values.dtype != mx.float32 or any(s < 1 for s in values.shape):
        raise ContractError("ordered normalization requires nonempty float32 NHWC or NDHWC input")


def inverse_standard_deviation(variance: array, eps: float) -> array:
    """Round the double-single inverse square root to float32 after two corrections."""
    if not np.isfinite(eps) or eps <= 0:
        raise ContractError("ordered normalization requires finite positive epsilon")
    mx, _ = import_mlx()
    epsilon = mx.array(np.array([eps, eps - float(np.float32(eps))], dtype=np.float32))
    _, _, inverse_kernel, _ = statistics_kernels()
    inverse = inverse_kernel(
        inputs=[variance, epsilon],
        template=[("N", variance.size)],
        grid=(variance.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[variance.shape],
        output_dtypes=[mx.float32],
    )[0]
    return inverse


def instance_norm(
    values: array,
    weight: array,
    bias: array,
    eps: float,
    *,
    output_fma: bool = False,
) -> array:
    """Compute ordered statistics on Metal and apply the qualified oracle's output contraction."""
    average, _, inverse = spatial_statistics(values, eps)
    return affine_from_statistics(
        values,
        average,
        inverse,
        weight,
        bias,
        output_fma=output_fma,
    )


def affine_from_statistics(
    values: array,
    average: array,
    inverse: array,
    weight: array,
    bias: array,
    *,
    output_fma: bool = False,
) -> array:
    """Keep affine arithmetic independently testable with oracle-supplied statistics."""
    mx, _ = import_mlx()
    _validate_values(values)
    batch, channels = values.shape[0], values.shape[-1]
    if (
        any(v.dtype != mx.float32 for v in (average, inverse, weight, bias))
        or average.size != batch * channels
        or inverse.size != batch * channels
        or weight.shape != (channels,)
        or bias.shape != (channels,)
    ):
        raise ContractError(
            "ordered affine requires matching float32 statistics and channel vectors"
        )
    _, _, _, affine_kernel = statistics_kernels()
    return affine_kernel(
        inputs=[values, average, inverse, weight, bias],
        template=[
            ("B", batch),
            ("S", values.size // (batch * channels)),
            ("C", channels),
            ("OUTPUT_FMA", output_fma),
        ],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
