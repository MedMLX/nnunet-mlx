"""Metal 3-D convolution of one input channel with a 3x3x3 kernel.

MLX 0.32.4 sends a 3-D convolution whose input channel count is not a multiple
of 16 through ``pad_and_slice_conv_3D_gpu``: it zero-pads the input and the
weight up to a multiple of 16 channels and runs the implicit GEMM, so a
one-channel input costs a padded copy sixteen times its size and sixteen
products per kernel tap, of which fifteen are zero. This kernel computes only
the 27 real products of each output value.

One thread per voxel reads the 27 input taps once, with zeros for taps in the
padding, and reuses them for every output channel. Each output starts from
zero and accumulates its taps with ``fma`` in weight memory order (depth,
height, width). MLX compiles custom kernels in ``safe`` math mode, so that
chain is not reassociated. For one input channel and 32 output channels the
result has the same bits as ``mx.conv3d``. That equality is measured against
MLX 0.32.4, not guaranteed by its API, so the dispatch in
``nnunet_mlx.model.network`` is limited to the compared configuration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from medmlx_core.typing import MetalKernel

from nnunet_mlx.errors import ContractError
from nnunet_mlx.runtime import import_mlx

if TYPE_CHECKING:
    from mlx.core import array

_SOURCE = r"""
    uint voxel = thread_position_in_grid.x;
    if (voxel >= B * D * H * W) return;    // dispatchThreads may over-provision
    int x = int(voxel % W);
    int y = int((voxel / W) % H);
    int z = int((voxel / (W * H)) % D);
    uint plane = (voxel / (W * H * D)) * D;
    float taps[27];
    for (int kd = 0; kd < 3; ++kd) {
        int d = z + kd - 1;
        for (int kh = 0; kh < 3; ++kh) {
            int h = y + kh - 1;
            for (int kw = 0; kw < 3; ++kw) {
                int w = x + kw - 1;
                bool inside = d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
                taps[(kd * 3 + kh) * 3 + kw] =
                    inside ? values[((plane + uint(d)) * H + uint(h)) * W + uint(w)] : 0.0f;
            }
        }
    }
    for (uint c = 0; c < O; ++c) {
        float sum = 0.0f;
        for (int tap = 0; tap < 27; ++tap) {
            sum = metal::fma(taps[tap], weights[c * 27 + tap], sum);
        }
        result[voxel * O + c] = sum;
    }
"""


@lru_cache(maxsize=1)
def _kernel() -> MetalKernel:
    """Build the single-channel 3x3x3 convolution kernel."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name="nnunet_single_channel_conv3d",
        input_names=["values", "weights"],
        output_names=["result"],
        source=_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def single_channel_conv3d(values: array, weight: array) -> array:
    """Convolve ``(B, D, H, W, 1)`` float32 ``values`` with ``(O, 3, 3, 3, 1)`` ``weight``.

    Stride and dilation are 1 and each spatial axis is zero-padded by 1, so the
    lazy float32 output is ``(B, D, H, W, O)``. There is no bias.
    """
    mx, _ = import_mlx()
    if values.ndim != 5 or values.shape[-1] != 1 or tuple(weight.shape[1:]) != (3, 3, 3, 1):
        raise ContractError(
            "single-channel convolution requires (B, D, H, W, 1) values and (O, 3, 3, 3, 1) weight"
        )
    if values.dtype != mx.float32 or weight.dtype != mx.float32:
        raise ContractError("single-channel convolution requires float32 values and weight")
    batch, depth, height, width, _ = values.shape
    outputs = weight.shape[0]
    return _kernel()(
        inputs=[values, weight],
        template=[("B", batch), ("D", depth), ("H", height), ("W", width), ("O", outputs)],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, depth, height, width, outputs)],
        output_dtypes=[mx.float32],
    )[0]
