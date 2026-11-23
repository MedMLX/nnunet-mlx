"""Float32 3x3x3 convolution for the native 32-channel, full-resolution U-Net layers."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from nnunet_mlx.kernels.types import MetalKernel
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array


_SOURCE = r"""
    // One SIMD group computes 16 neighboring X positions and all 32 output channels.
    uint strip_start = (thread_position_in_grid.x / 32) * 16;
    uint x0 = strip_start % W;
    uint yz = strip_start / W;
    uint y0 = yz % H, z0 = (yz / H) % D, batch = yz / (H * D);
    uint lane = thread_index_in_simdgroup;
    // Coordinates of this lane's two adjacent elements in an Apple 8x8 MMA fragment.
    uint row = ((lane / 4) & 4) + ((lane / 2) % 4);
    uint col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    metal::simdgroup_float8x8 acc[2][4];
    for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n)
            acc[m][n] = metal::simdgroup_float8x8(0.0f);

    // MLX 0.32.4 processes all spatial taps for each block of 16 input channels.
    for (int block = 0; block < 32; block += 16) {
        for (int kz = 0; kz < 3; ++kz) {
            int z = int(z0) + kz - 1;
            for (int ky = 0; ky < 3; ++ky) {
                int y = int(y0) + ky - 1;
                for (int kx = 0; kx < 3; ++kx) {
                    int tap = (kz * 3 + ky) * 3 + kx;
                    #pragma clang loop unroll(full)
                    for (int k = block; k < block + 16; k += 8) {
                        metal::simdgroup_float8x8 a[2], b[4];
                        #pragma clang loop unroll(full)
                        for (int m = 0; m < 2; ++m) {
                            int x = int(x0 + 8 * m + row) + kx - 1;
                            float2 v(0.0f);
                            if (uint(z) < D && uint(y) < H && uint(x) < W) {
                                uint voxel = ((batch * D + uint(z)) * H + uint(y)) * W + uint(x);
                                v = *(device const float2*)(values + voxel * 32 + k + col);
                            }
                            a[m].thread_elements()[0] = v.x;
                            a[m].thread_elements()[1] = v.y;
                        }
                        #pragma clang loop unroll(full)
                        for (int n = 0; n < 4; ++n) {
                            float2 w = *(device const float2*)(
                                packed + (tap * 32 + k + row) * 32 + 8 * n + col);
                            b[n].thread_elements()[0] = w.x;
                            b[n].thread_elements()[1] = w.y;
                        }
                        #pragma clang loop unroll(full)
                        for (int m = 0; m < 2; ++m) {
                            #pragma clang loop unroll(full)
                            for (int n = 0; n < 4; ++n)
                                simdgroup_multiply_accumulate(acc[m][n], a[m], b[n], acc[m][n]);
                        }
                    }
                }
            }
        }
    }
    for (int m = 0; m < 2; ++m) {
        for (int n = 0; n < 4; ++n) {
            uint base = (strip_start + 8 * m) * 32 + 8 * n;
            simdgroup_store(acc[m][n], result + base, 32);
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel() -> MetalKernel:
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name="nnunet_conv3d_32",
        input_names=["values", "packed"],
        output_names=["result"],
        source=_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def conv3d_32(values: array, weight: array) -> array:
    """Convolve 32 NDHWC channels to 32 without bias.

    The caller supplies float32 tensors, a width divisible by 16, and 3³ weights.
    The kernel uses unit stride, dilation and padding. Packing weights with output
    channels last lets each lane fetch two adjacent columns of its matrix fragment.
    """
    mx, _ = import_mlx()
    _, depth, height, width, _ = values.shape
    packed = mx.transpose(weight, (1, 2, 3, 4, 0))
    return _kernel()(
        inputs=[values, packed],
        template=[("D", depth), ("H", height), ("W", width)],
        grid=(values.size // 16, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
