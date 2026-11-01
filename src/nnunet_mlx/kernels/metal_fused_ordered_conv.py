"""Register-blocked ordered FP32 convolution for the TotalSegmentator CT profile.

Derived from the Apache-2.0 metal_ordered_conv.py implementation at a01d237.
Each output retains bias-first SIMD8 accumulation in ascending K8 chunks. Packing preserves float32
values and writes positive-zero halos; logical network axes remain NDHWC.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.kernels.types import MetalKernel


_PACK_SOURCE = r"""
uint tid = thread_position_in_threadgroup.x;
if (C == 1) {
    // A single channel needs no transpose tile; one thread per padded voxel.
    uint p = thread_position_in_grid.x;
    uint b = threadgroup_position_in_grid.z;
    if (p >= DP * HP * WP) return;
    int z = int(p / (HP * WP)) - PD;
    int y = int((p / WP) % HP) - PH;
    int x = int(p % WP) - PW;
    float value = 0;
    if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W)
        value = values[b*values_strides[0]+z*values_strides[1]+y*values_strides[2]
                       +x*values_strides[3]];
    packed[b * (DP * HP * WP) + p] = value;
    return;
}
uint p0 = threadgroup_position_in_grid.x * 32;
uint c0 = threadgroup_position_in_grid.y * 32;
uint b = threadgroup_position_in_grid.z;
threadgroup float tile[32 * 33];
if (values_strides[4] != 1) {
    // Fused normalization already emits dense channels-first storage.
    for (uint i = tid; i < 1024; i += 256) {
        uint c = c0 + i / 32, p = p0 + i % 32;
        int z = int(p / (HP * WP)) - PD;
        int y = int((p / WP) % HP) - PH;
        int x = int(p % WP) - PW;
        float value = 0;
        if (p < DP * HP * WP && c < C && z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W)
            value = values[b*values_strides[0]+z*values_strides[1]+y*values_strides[2]
                           +x*values_strides[3]+c*values_strides[4]];
        if (c < C && p < DP * HP * WP)
            packed[(b * C + c) * (DP * HP * WP) + p] = value;
    }
} else {
    for (uint i = tid; i < 1024; i += 256) {
        uint p = p0 + i / 32, c = c0 + i % 32;
        int z = int(p / (HP * WP)) - PD;
        int y = int((p / WP) % HP) - PH;
        int x = int(p % WP) - PW;
        float value = 0;
        if (p < DP * HP * WP && c < C && z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W)
            value = values[b*values_strides[0]+z*values_strides[1]+y*values_strides[2]
                           +x*values_strides[3]+c*values_strides[4]];
        tile[(i / 32) * 33 + i % 32] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 256) {
        uint c = c0 + i / 32, p = p0 + i % 32;
        if (c < C && p < DP * HP * WP)
            packed[(b * C + c) * (DP * HP * WP) + p] = tile[(i % 32) * 33 + i / 32];
    }
}
"""


_CONV_HEADER = r"""
#include <metal_simdgroup_matrix>
"""


# Rows are output positions (b, z, y, x; x fastest) and columns are output
# channels. Each simdgroup owns TM x TN 8x8 fragments and gathers its own A
# operands straight from the packed NCDHW input: row r and reduction index
# k = ch * KT + tap address values[rowbase(r) + offset(k)], so no im2col tile,
# threadgroup staging or barriers are needed. Fragments stay in float2
# registers (simdgroup_matrix arrays spill) using MLX's 8x8 lane layout.
# Every output keeps the original arithmetic: bias + 0 first, then one
# simdgroup MMA per K8 chunk in ascending k, with the same zero-padded chunk
# count as the earlier K32-staged kernel.
_CONV_SOURCE = r"""
constexpr int KT = KD * KH * KW;
constexpr int K = C * KT;
constexpr int CS = DP * HP * WP;
constexpr int ROWS = B * OD * OH * OW;
constexpr int CHUNKS = ((K + 31) / 32) * 4;
constexpr int FULL_CHUNKS = K / 8;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint qid = lane / 4;
uint fm = (qid & 4) + ((lane / 2) % 4);
uint fn = (qid & 2) * 2 + (lane % 2) * 2;
uint row0 = threadgroup_position_in_grid.y * (WM * TM * 8) + sg * (TM * 8);
uint col0 = threadgroup_position_in_grid.z * (TN * 8);
uint rowbase[TM];
bool rowok[TM];
#pragma unroll
for (int i = 0; i < TM; ++i) {
    uint row = row0 + i * 8 + fm;
    rowok[i] = row < ROWS;
    // Tail rows read a valid position; rows never mix in an MMA and are not stored.
    row = rowok[i] ? row : 0;
    uint x = row % OW, y = (row / OW) % OH, z = (row / (OW * OH)) % OD;
    uint b = row / (OW * OH * OD);
    rowbase[i] = ((b * C * DP + z * SD) * HP + y * SH) * WP + x * SW;
}
float2 acc[TM][TN];
#pragma unroll
for (int j = 0; j < TN; ++j) {
    uint oc = col0 + j * 8 + fn;
    float2 init = float2(
        (FULL_OUTPUT || oc < O) ? bias[oc] + 0.0f : 0.0f,
        (FULL_OUTPUT || oc + 1 < O) ? bias[oc + 1] + 0.0f : 0.0f);
    #pragma unroll
    for (int i = 0; i < TM; ++i) acc[i][j] = init;
}
auto offset_of = [](int k) -> uint {
    uint ch = uint(k) / KT, tap = uint(k) % KT;
    return ch * CS + (tap / (KH * KW)) * (HP * WP) + ((tap / KW) % KH) * WP + tap % KW;
};
// Integer division by KT/KH/KW per chunk dominated the gather cost. Instead each
// lane advances its two reduction offsets by 8 using per-tap tables: moving from
// tap t to tap (t + 8) % KT adds the spatial delta plus any channel wraps.
threadgroup int delta_table[KT];
threadgroup ushort next_table[KT];
{
    uint t = thread_position_in_threadgroup.x;
    if (t < uint(KT)) {
        auto tap_offset = [](uint tap) -> int {
            return int((tap / (KH * KW)) * (HP * WP) + ((tap / KW) % KH) * WP + tap % KW);
        };
        uint next = (t + 8) % KT;
        delta_table[t] = int(((t + 8) / KT) * CS) + tap_offset(next) - tap_offset(t);
        next_table[t] = ushort(next);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint offset0 = offset_of(fn), offset1 = offset_of(fn + 1);
uint tap0 = fn % KT, tap1 = (fn + 1) % KT;
for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    int k0 = chunk * 8;
    float2 a[TM], w[TN];
    if (chunk < FULL_CHUNKS) {
        uint o0 = offset0, o1 = offset1;
        offset0 += delta_table[tap0];
        offset1 += delta_table[tap1];
        tap0 = next_table[tap0];
        tap1 = next_table[tap1];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            a[i] = float2(values[rowbase[i] + o0], values[rowbase[i] + o1]);
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint oc = col0 + j * 8 + fn;
            w[j] = float2(
                (FULL_OUTPUT || oc < O) ? weights[(k0 + fm) * O + oc] : 0.0f,
                (FULL_OUTPUT || oc + 1 < O) ? weights[(k0 + fm) * O + oc + 1] : 0.0f);
        }
    } else {
        // K tail: both operands are exact zeros past K, as in the staged kernel.
        int ka = k0 + fn, kb = ka + 1, kw = k0 + fm;
        uint o0 = offset_of(min(ka, K - 1)), o1 = offset_of(min(kb, K - 1));
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            a[i] = float2(ka < K ? values[rowbase[i] + o0] : 0.0f,
                          kb < K ? values[rowbase[i] + o1] : 0.0f);
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            uint oc = col0 + j * 8 + fn;
            w[j] = float2(
                (kw < K && (FULL_OUTPUT || oc < O)) ? weights[kw * O + oc] : 0.0f,
                (kw < K && (FULL_OUTPUT || oc + 1 < O)) ? weights[kw * O + oc + 1] : 0.0f);
        }
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            metal::simdgroup_float8x8 am, wm, cm;
            am.thread_elements()[0] = a[i].x;
            am.thread_elements()[1] = a[i].y;
            wm.thread_elements()[0] = w[j].x;
            wm.thread_elements()[1] = w[j].y;
            cm.thread_elements()[0] = acc[i][j].x;
            cm.thread_elements()[1] = acc[i][j].y;
            simdgroup_multiply_accumulate(cm, am, wm, cm);
            acc[i][j] = float2(cm.thread_elements()[0], cm.thread_elements()[1]);
        }
    }
}
#pragma unroll
for (int i = 0; i < TM; ++i) {
    uint row = row0 + i * 8 + fm;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        uint oc = col0 + j * 8 + fn;
        if (rowok[i] && (FULL_OUTPUT || oc < O)) result[row * O + oc] = acc[i][j].x;
        if (rowok[i] && (FULL_OUTPUT || oc + 1 < O)) result[row * O + oc + 1] = acc[i][j].y;
    }
}
"""


# One thread per padded NCDHW element, x fastest. Channels below CU come from
# the stride-matched transpose matmul rows (low-resolution voxel, CU x KT
# columns ordered channel then kernel tap) plus their bias in one float32 add,
# exactly like the separate reshape/transpose/bias path; the remaining channels
# copy the strided NDHWC skip. Halo positions stay positive zero.
_UPSAMPLE_CONCAT_PACK_SOURCE = r"""
uint p = thread_position_in_grid.x;
uint c = thread_position_in_grid.y;
uint b = thread_position_in_grid.z;
if (p >= DP * HP * WP) return;
int z = int(p / (HP * WP)) - PD;
int y = int((p / WP) % HP) - PH;
int x = int(p % WP) - PW;
float value = 0.0f;
if (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) {
    if (c < CU) {
        uint row = ((b * (D / KD) + uint(z) / KD) * (H / KH) + uint(y) / KH) * (W / KW)
            + uint(x) / KW;
        uint tap = ((uint(z) % KD) * KH + uint(y) % KH) * KW + uint(x) % KW;
        value = projected[row * projected_strides[0]
                          + (c * (KD * KH * KW) + tap) * projected_strides[1]] + bias[c];
    } else {
        value = skip[b * skip_strides[0] + z * skip_strides[1] + y * skip_strides[2]
                     + x * skip_strides[3] + (c - CU) * skip_strides[4]];
    }
}
packed[((b * (CU + CS) + c) * DP * HP + (p / WP)) * WP + p % WP] = value;
"""


@lru_cache(maxsize=1)
def _upsample_concat_pack_kernel() -> MetalKernel:
    """Build the fused upsample, skip concatenation and packing kernel."""
    import mlx.core as mx

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_upsample_concat_pack", input_names=["projected", "bias", "skip"],
            output_names=["packed"], source=_UPSAMPLE_CONCAT_PACK_SOURCE,
            ensure_row_contiguous=False, compile_options={"math_mode": "safe"},
        ),
    )


def upsample_concat_padded(
    projected: array, bias: array, skip: array, kernel: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> array:
    """Pack ``concat(transpose_conv + bias, skip)`` with a positive-zero halo.

    ``projected`` is the ``(B*D/KD*H/KH*W/KW, CU*KD*KH*KW)`` matmul output of a
    stride-equals-kernel transpose convolution before its spatial shuffle and
    bias; ``skip`` is logical NDHWC at the upsampled resolution. Returns dense
    ``(B, CU + CS, D + 2PD, H + 2PH, W + 2PW)`` storage for the next convolution.
    Callers must supply float32 tensors, positive kernel dimensions and matching
    channel shapes; these preconditions are not fully checked here.
    """
    import mlx.core as mx

    from nnunet_mlx.errors import ContractError

    batch, depth, height, width, skip_channels = skip.shape
    kd, kh, kw = kernel
    pd, ph, pw = padding
    upsampled = bias.shape[0]
    if (
        min(padding) < 0
        or depth % kd or height % kh or width % kw
        or projected.shape != (batch * depth * height * width // (kd * kh * kw),
                               upsampled * kd * kh * kw)
    ):
        raise ContractError("fused upsampling requires matching transpose and skip shapes")
    dp, hp, wp = depth + 2 * pd, height + 2 * ph, width + 2 * pw
    return _upsample_concat_pack_kernel()(
        inputs=[projected, bias, skip],
        template=list(dict(
            CU=upsampled, CS=skip_channels, D=depth, H=height, W=width,
            KD=kd, KH=kh, KW=kw, DP=dp, HP=hp, WP=wp, PD=pd, PH=ph, PW=pw,
        ).items()),
        grid=(dp * hp * wp, upsampled + skip_channels, batch), threadgroup=(256, 1, 1),
        output_shapes=[(batch, upsampled + skip_channels, dp, hp, wp)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _pack_kernel() -> MetalKernel:
    """Build the padded input packing kernel for ordered convolution."""
    import mlx.core as mx

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_ordered_pack_pad", input_names=["values"], output_names=["packed"],
            source=_PACK_SOURCE, ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        ),
    )


@lru_cache(maxsize=1)
def _conv_kernel() -> MetalKernel:
    """Build the register-blocked ordered convolution kernel."""
    import mlx.core as mx

    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="totalseg_padded_ordered_conv3d", input_names=["values", "weights", "bias"],
            output_names=["result"], header=_CONV_HEADER,
            source=_CONV_SOURCE, compile_options={"math_mode": "safe"},
        ),
    )


def padded_ordered_conv3d(
    values: array, weight: array, bias: array, stride: tuple[int, ...], padding: tuple[int, ...],
) -> array:
    """Pack strided float32 NDHWC input and its halo before ordered convolution.

    ``weight`` is ``(O, KD, KH, KW, C)`` and ``bias`` has ``O`` entries;
    ``stride`` and nonnegative symmetric ``padding`` follow the spatial axes.
    The result is float32 NDHWC. Callers must provide compatible weight and
    bias shapes and positive strides; these are not fully checked here.
    """
    import mlx.core as mx

    from nnunet_mlx.errors import ContractError

    batch, depth, height, width, channels = values.shape
    pd, ph, pw = padding
    if min(padding) < 0:
        raise ContractError("padded convolution requires nonnegative padding")
    dp, hp, wp = depth + 2 * pd, height + 2 * ph, width + 2 * pw
    if any(padding):
        packed_values = _pack_kernel()(
            inputs=[values],
            template=list(dict(
                B=batch, C=channels, D=depth, H=height, W=width,
                DP=dp, HP=hp, WP=wp, PD=pd, PH=ph, PW=pw,
            ).items()),
            grid=(
                dp * hp * wp if channels == 1 else 256 * ((dp * hp * wp + 31) // 32),
                (channels + 31) // 32, batch,
            ),
            threadgroup=(256, 1, 1), output_shapes=[(batch, channels, dp, hp, wp)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        packed_values = mx.transpose(values, (0, 4, 1, 2, 3))
    return ordered_conv3d_from_packed(packed_values, weight, bias, stride)


def ordered_conv3d_from_packed(
    packed_values: array, weight: array, bias: array, stride: tuple[int, ...],
) -> array:
    """Convolve dense NCDHW values whose positive-zero halo is already applied.

    ``weight`` is ``(O, KD, KH, KW, C)`` and ``bias`` has ``O`` entries; output
    is float32 NDHWC. Compatible arrays and positive strides are caller
    preconditions. A nonpositive output extent or fewer than two output positions
    raises ``ContractError``.
    """
    import mlx.core as mx

    from nnunet_mlx.errors import ContractError

    batch, channels, dp, hp, wp = packed_values.shape
    outputs, kd, kh, kw, _ = weight.shape
    sd, sh, sw = stride
    od, oh, ow = (dp - kd) // sd + 1, (hp - kh) // sh + 1, (wp - kw) // sw + 1
    if min(od, oh, ow) < 1 or batch * min(4, od) * oh * ow < 2:
        raise ContractError("ordered convolution requires at least two output positions")
    positions = batch * od * oh * ow
    # 4x4 fragments per simdgroup fit in registers; tiny bottleneck maps use
    # 2x4 so enough threadgroups stay resident.
    row_fragments = 4 if positions > 512 else 2
    column_fragments = min(4, (outputs + 7) // 8)
    simdgroups = 4 if positions > 512 else 2
    block_rows, block_columns = simdgroups * row_fragments * 8, column_fragments * 8
    dimensions = dict(
        FULL_OUTPUT=outputs % block_columns == 0,
        TM=row_fragments, TN=column_fragments, WM=simdgroups,
        B=batch, DP=dp, HP=hp, WP=wp, C=channels, O=outputs,
        KD=kd, KH=kh, KW=kw, SD=sd, SH=sh, SW=sw, OD=od, OH=oh, OW=ow,
    )
    packed_weights = mx.reshape(mx.transpose(weight, (4, 1, 2, 3, 0)), (-1, outputs))
    return _conv_kernel()(
        inputs=[packed_values, packed_weights, bias], template=list(dimensions.items()),
        grid=(
            32 * simdgroups,
            (positions + block_rows - 1) // block_rows,
            (outputs + block_columns - 1) // block_columns,
        ),
        threadgroup=(32 * simdgroups, 1, 1), output_shapes=[(batch, od, oh, ow, outputs)],
        output_dtypes=[mx.float32],
    )[0]
