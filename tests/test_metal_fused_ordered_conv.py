"""Bitwise layout, K-tail, and bias parity for prefetched padded convolution."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_mlx.runtime import import_mlx

pytestmark = pytest.mark.metal


_GEOMETRIES = [
    (2, 1, 8, (5, 7, 3), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (2, 4, 64, (8, 8, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (2, 8, 7, (7, 6, 9), (3, 1, 3), (2, 1, 2), (1, 0, 1)),
    (2, 8, 65, (6, 5, 7), (3, 3, 3), (1, 1, 1), (0, 0, 0)),
    (2, 8, 64, (8, 8, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1)),
    (2, 3, 3, (4, 4, 8), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
    (2, 1, 32, (4, 4, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (2, 32, 32, (4, 4, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (2, 32, 64, (8, 8, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1)),
    (2, 32, 65, (4, 4, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    (2, 3, 7, (3, 3, 4), (1, 1, 11), (1, 1, 1), (0, 0, 5)),
    (1, 1, 2, (1, 1, 2), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
    (1, 1, 32, (4, 4, 8), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
]
_STRIDED_LAYOUTS = ["channels_first", "width_slice", "channel_slice", "spatial_transpose"]


# Every geometry runs contiguously; strided layouts run on a tail and a strided geometry.
@pytest.mark.parametrize(
    ("batch", "channels", "outputs", "spatial", "kernel", "stride", "padding", "layout"),
    [(*geometry, "contiguous") for geometry in _GEOMETRIES]
    + [(*_GEOMETRIES[index], layout) for layout in _STRIDED_LAYOUTS for index in (2, 9)],
)
def test_padded_convolution_matches_original_bits(
    batch: int,
    channels: int,
    outputs: int,
    spatial: tuple[int, int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    layout: str,
) -> None:
    """Preserve ordered convolution results when packing padding."""
    mx, nn = import_mlx()
    from nnunet_mlx.kernels.metal_fused_ordered_conv import padded_ordered_conv3d
    from nnunet_mlx.prediction.arithmetic import OrderedConv3d

    rng = np.random.default_rng(715)
    native = nn.Conv3d(channels, outputs, kernel, stride=stride, padding=padding)
    native.weight = mx.array(rng.normal(size=(outputs, *kernel, channels)).astype(np.float32))
    native.bias = mx.array(rng.normal(size=outputs).astype(np.float32))
    original = OrderedConv3d(native)
    original.eval()
    depth, height, width = spatial
    input_width = width * 2 if layout == "width_slice" else width
    input_channels = channels * 2 if layout == "channel_slice" else channels
    values = mx.array(
        rng.normal(size=(batch, depth, height, input_width, input_channels)).astype(np.float32)
    )
    if layout == "channels_first":
        values = mx.transpose(mx.contiguous(mx.transpose(values, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))
    elif layout == "width_slice":
        values = values[:, :, :, ::2, :]
    elif layout == "channel_slice":
        values = values[..., ::2]
    elif layout == "spatial_transpose":
        storage = mx.contiguous(mx.transpose(values, (0, 3, 2, 1, 4)))
        values = mx.transpose(storage, (0, 3, 2, 1, 4))
    expected = original(values)
    actual = padded_ordered_conv3d(values, native.weight, native.bias, stride, padding)
    mx.eval(expected, actual)
    expected_host, actual_host = (
        np.array(expected, dtype=np.float32),
        np.array(actual, dtype=np.float32),
    )
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.float32
    assert np.isfinite(expected_host).all()
    np.testing.assert_array_equal(actual_host.view(np.uint32), expected_host.view(np.uint32))


def test_padded_convolution_preserves_bias_rounding_with_zero_products() -> None:
    """Preserve bias-first rounding when convolution products are zero."""
    mx, nn = import_mlx()
    from nnunet_mlx.kernels.metal_fused_ordered_conv import padded_ordered_conv3d
    from nnunet_mlx.prediction.arithmetic import OrderedConv3d

    native = nn.Conv3d(3, 8, 3, padding=1)
    native.weight = mx.ones_like(native.weight)
    native.bias = mx.array([0.0, -0.0, 1e-30, -1e-30, 1e20, -1e20, 1.0, -1.0], dtype=mx.float32)
    original = OrderedConv3d(native)
    original.eval()
    values = mx.zeros((1, 4, 4, 8, 3), dtype=mx.float32)
    expected = original(values)
    actual = padded_ordered_conv3d(
        values,
        native.weight,
        native.bias,
        (1, 1, 1),
        (1, 1, 1),
    )
    mx.eval(expected, actual)
    np.testing.assert_array_equal(
        np.array(actual).view(np.uint32), np.array(expected).view(np.uint32)
    )
