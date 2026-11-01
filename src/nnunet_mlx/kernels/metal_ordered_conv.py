"""Metal implicit GEMM with channel-major, bias-first float32 accumulation."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

_SOURCE = r"""
uint tid = thread_position_in_threadgroup.x;
uint sg = tid / 32;
uint row0 = threadgroup_position_in_grid.y * 128;
uint col0 = threadgroup_position_in_grid.z * TN;
threadgroup float aa[4096];
threadgroup float bb[32 * TN];
metal::simdgroup_float8x8 c00, c01, c10, c11;
c00.thread_elements()[0] = 0; c00.thread_elements()[1] = 0;
c01.thread_elements()[0] = 0; c01.thread_elements()[1] = 0;
c10.thread_elements()[0] = 0; c10.thread_elements()[1] = 0;
c11.thread_elements()[0] = 0; c11.thread_elements()[1] = 0;
// The first K block adds bias before any input product.
for (int base = -32; base < C * KD * KH * KW; base += 32) {
    for (uint i = tid; i < 4096; i += 16 * TN) {
        uint row = row0 + i / 32;
        int k = base + int(i % 32);
        float av = 0;
        if (row < B * OD * OH * OW) {
            if (k == -32) av = 1;
            else if (k >= 0 && k < C * KD * KH * KW) {
                uint ch = (k / (CB * KD * KH * KW)) * CB + k % CB;
                uint rem = (k / CB) % (KD * KH * KW);
                int z = int((row / (OH * OW)) % OD) * SD + int(rem / (KH * KW)) - PD;
                int y = int((row / OW) % OH) * SH + int((rem / KW) % KH) - PH;
                int x = int(row % OW) * SW + int(rem % KW) - PW;
                uint b = row / (OD * OH * OW);
                if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W)
                    av = values[(((b * C + ch) * D + z) * H + y) * W + x];
            }
        }
        aa[i] = av;
        if (i < 32 * TN) {
            uint oc = col0 + i % TN;
            int wk = base + int(i / TN);
            float bv = 0;
            if (oc < O) {
                if (wk == -32) bv = bias[oc];
                else if (wk >= 0 && wk < C * KD * KH * KW) bv = weights[wk * O + oc];
            }
            bb[i] = bv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
        metal::simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, aa + (sg / (TN / 16)) * 16 * 32 + kk, 32);
        simdgroup_load(a1, aa + ((sg / (TN / 16)) * 16 + 8) * 32 + kk, 32);
        simdgroup_load(b0, bb + kk * TN + (sg % (TN / 16)) * 16, TN);
        simdgroup_load(b1, bb + kk * TN + (sg % (TN / 16)) * 16 + 8, TN);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Complete N64 tiles can store directly without reusing the input staging buffer.
if constexpr (TN == 64) {
    uint row = row0 + (sg / (TN / 16)) * 16;
    uint oc = col0 + (sg % (TN / 16)) * 16;
    simdgroup_store(c00, result + row * O + oc, O);
    simdgroup_store(c01, result + row * O + oc + 8, O);
    simdgroup_store(c10, result + (row + 8) * O + oc, O);
    simdgroup_store(c11, result + (row + 8) * O + oc + 8, O);
} else {
    // All groups have finished reading aa; reuse it for the output tile.
    simdgroup_store(c00, aa + (sg / (TN / 16)) * 16 * 32 + (sg % (TN / 16)) * 16, TN);
    simdgroup_store(c01, aa + (sg / (TN / 16)) * 16 * 32 + (sg % (TN / 16)) * 16 + 8, TN);
    simdgroup_store(c10, aa + ((sg / (TN / 16)) * 16 + 8) * 32 + (sg % (TN / 16)) * 16, TN);
    simdgroup_store(c11, aa + ((sg / (TN / 16)) * 16 + 8) * 32 + (sg % (TN / 16)) * 16 + 8, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 4096; i += 16 * TN) {
        uint row = row0 + i / 32, oc = col0 + i % 32;
        if (row < B * OD * OH * OW && oc < O) result[row * O + oc] = aa[i];
    }
}
"""


@lru_cache(maxsize=1)
def _kernel() -> MetalKernel:
    """Build the ordered FP32 3D convolution Metal kernel."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name="nnunet_ordered_conv3d",
        input_names=["values", "weights", "bias"],
        output_names=["result"],
        header="#include <metal_simdgroup_matrix>\n",
        source=_SOURCE,
    )


def ordered_conv3d(
    values: array,
    weight: array,
    bias: array,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    *,
    channel_block: int = 1,
) -> array:
    """Convolve channels-last 3-D values using the specified channel-block order.

    ``values`` is float32 NDHWC, ``weight`` is float32 ``(O, KD, KH, KW, C)``,
    and ``bias`` has one entry per output channel. ``stride`` and ``padding``
    are three spatial integers. The return is float32 NDHWC. Inputs must have
    compatible shapes and positive strides; these preconditions are not fully
    checked here. Input windows use threadgroup memory instead of an im2col array.

    ``channel_block`` is 1 or 8; the actual block is capped at the input channel
    count and must divide it. A nonpositive output extent or fewer than two output
    positions raises ``ContractError``.
    """
    mx, _ = import_mlx()
    batch, depth, height, width, channels = values.shape
    block = min(channels, channel_block)
    if channel_block not in (1, 8) or channels % block:
        raise ContractError("ordered convolution requires complete channel blocks of 1 or 8")
    outputs, kd, kh, kw, _ = weight.shape
    sd, sh, sw = stride
    pd, ph, pw = padding
    od, oh, ow = (
        (depth + 2 * pd - kd) // sd + 1,
        (height + 2 * ph - kh) // sh + 1,
        (width + 2 * pw - kw) // sw + 1,
    )
    if min(od, oh, ow) < 1 or batch * min(4, od) * oh * ow < 2:
        raise ContractError("ordered convolution requires at least two output positions")
    positions = batch * od * oh * ow
    # Direct stores require complete row and column tiles; tails retain masked writes.
    tile_columns = 64 if positions % 128 == 0 and outputs % 64 == 0 else 32
    threads = 16 * tile_columns
    dimensions = dict(
        TN=tile_columns,
        B=batch,
        CB=block,
        D=depth,
        H=height,
        W=width,
        C=channels,
        O=outputs,
        KD=kd,
        KH=kh,
        KW=kw,
        SD=sd,
        SH=sh,
        SW=sw,
        PD=pd,
        PH=ph,
        PW=pw,
        OD=od,
        OH=oh,
        OW=ow,
    )
    # Pack the selected CPU channel-block reduction order with coalesced weights.
    packed_values = mx.transpose(values, (0, 4, 1, 2, 3))
    packed_weights = mx.reshape(weight, (outputs, kd, kh, kw, channels // block, block))
    packed_weights = mx.reshape(mx.transpose(packed_weights, (4, 1, 2, 3, 5, 0)), (-1, outputs))
    return _kernel()(
        inputs=[packed_values, packed_weights, bias],
        template=list(dimensions.items()),
        grid=(threads, (positions + 127) // 128, (outputs + tile_columns - 1) // tile_columns),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, od, oh, ow, outputs)],
        output_dtypes=[mx.float32],
    )[0]
