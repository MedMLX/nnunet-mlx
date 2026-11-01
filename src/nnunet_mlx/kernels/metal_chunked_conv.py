"""Metal conv2d with the pinned Torch Slow2d / oneMKL SGEMM reduction order.

The qualified 5x5, 512-channel convolutions use im2col SGEMM: channel-major
256-product blocks each start from zero, then add their partial to a total
initialized with bias. These boundaries cut across channels, so the existing
convolution's channel slicing cannot express this reduction order.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from numbers import Integral
from typing import TYPE_CHECKING, cast

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.types import MetalKernel, ShapedArray
from nnunet_mlx.runtime import import_mlx

CHUNK = 256  # oneMKL kc for the qualified small-spatial convolution

if TYPE_CHECKING:
    from mlx.core import array

_SOURCE = r"""
uint index = thread_position_in_grid.x;
if (index >= B * OH * OW * O) return;      // dispatchThreads may over-provision
uint oc = index % O;
uint position = index / O;
uint b = position / (OH * OW);
uint oy = (position / OW) % OH;
uint ox = position % OW;
float total = bias[oc];
float acc = 0;
uint count = 0;
// Im2col columns order: input channel major, kernel tap minor.
for (uint c = 0; c < C; ++c) {
    for (uint tap = 0; tap < KH * KW; ++tap) {
        int iy = int(oy) * int(SH) + int(tap / KW) - int(PH);
        int ix = int(ox) * int(SW) + int(tap % KW) - int(PW);
        float sample = 0;
        if (iy >= 0 && iy < int(H) && ix >= 0 && ix < int(W))
            sample = values[(((b * H + uint(iy)) * W + uint(ix)) * C) + c];
        float coefficient = weights[((oc * KH + (tap / KW)) * KW + (tap % KW)) * C + c];
        acc = metal::fma(sample, coefficient, acc);
        if (++count == CHUNK) { total = total + acc; acc = 0; count = 0; }
    }
}
if (count > 0) { total = total + acc; }    // partial trailing block
result[index] = total;
"""


@lru_cache(maxsize=1)
def _kernel() -> MetalKernel:
    """Build the chunked 2D convolution Metal kernel."""
    mx, _ = import_mlx()
    return mx.fast.metal_kernel(
        name="nnunet_chunked_conv2d",
        input_names=["values", "weights", "bias"],
        output_names=["result"],
        source=_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def _pair(value: object, name: str, *, minimum: int) -> tuple[int, int]:
    """Require a pair of integral parameters at or above the supplied minimum."""
    items = cast("Sequence[object]", value) if isinstance(value, (tuple, list)) else ()
    if len(items) != 2 or any(not isinstance(item, Integral) or item < minimum for item in items):
        raise ContractError(f"chunked convolution {name} must be two integers >= {minimum}")
    return int(cast(Integral, items[0])), int(cast(Integral, items[1]))


def validated_geometry(
    values: ShapedArray, weight: ShapedArray, bias: ShapedArray, stride: object, padding: object
) -> dict[str, int]:
    """Validate shapes, stride and padding before importing MLX.

    Dtype is checked separately by ``chunked_conv2d``. The returned dimensions
    describe an NHWC output and the Metal template constants.
    """
    if values.ndim != 4 or weight.ndim != 4 or bias.ndim != 1:
        raise ContractError("chunked convolution requires NHWC values, OHWI weight, 1-D bias")
    batch, height, width, channels = (int(size) for size in values.shape)
    outputs, kheight, kwidth, weights_channels = (int(size) for size in weight.shape)
    if channels != weights_channels:
        raise ContractError("chunked convolution input channels do not match the weight")
    if int(bias.shape[0]) != outputs:
        raise ContractError("chunked convolution bias length does not match output channels")
    if min(batch, height, width, channels, outputs, kheight, kwidth) < 1:
        raise ContractError("chunked convolution requires non-empty values and weight")
    step_h, step_w = _pair(stride, "stride", minimum=1)
    pad_h, pad_w = _pair(padding, "padding", minimum=0)
    out_h = (height + 2 * pad_h - kheight) // step_h + 1
    out_w = (width + 2 * pad_w - kwidth) // step_w + 1
    if out_h < 1 or out_w < 1:
        raise ContractError("chunked convolution kernel exceeds the padded input")
    return dict(
        B=batch,
        C=channels,
        H=height,
        W=width,
        O=outputs,
        KH=kheight,
        KW=kwidth,
        SH=step_h,
        SW=step_w,
        PH=pad_h,
        PW=pad_w,
        OH=out_h,
        OW=out_w,
        CHUNK=CHUNK,
    )


def chunked_conv2d(
    values: array, weight: array, bias: array, stride: object, padding: object
) -> array:
    """Convolve NHWC float32 ``values`` with OHWI ``weight`` on the Metal GPU.

    ``bias`` is a required float32 vector with one value per output channel.
    ``stride`` and symmetric ``padding`` each contain two integers. The lazy
    output is NHWC float32; this path has no dilation or grouped convolution.
    """
    template = validated_geometry(values, weight, bias, stride, padding)
    mx, _ = import_mlx()
    if values.dtype != mx.float32 or weight.dtype != mx.float32 or bias.dtype != mx.float32:
        raise ContractError("chunked convolution requires float32 values, weight and bias")
    out_b, out_h, out_w, outputs = template["B"], template["OH"], template["OW"], template["O"]
    return _kernel()(
        inputs=[values, weight, bias],
        template=list(template.items()),
        grid=(out_b * out_h * out_w * outputs, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(out_b, out_h, out_w, outputs)],
        output_dtypes=[mx.float32],
    )[0]
