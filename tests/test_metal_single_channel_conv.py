"""Single-channel Metal convolution compared with MLX's native convolution."""

from __future__ import annotations

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.kernels.metal_single_channel_conv import single_channel_conv3d
from nnunet_mlx.runtime import import_mlx


@pytest.mark.metal
def test_single_channel_conv_reproduces_native_convolution_bits() -> None:
    """Compare the kernel with ``mx.conv3d`` bit for bit in the dispatched configuration.

    The network replaces the native convolution with this kernel on the strength
    of that equality, so a changed accumulation order on either side must fail here.
    """
    mx, _ = import_mlx()
    rng = np.random.default_rng(1117)
    values = rng.normal(size=(2, 9, 12, 10, 1)).astype(np.float32)
    values[0, :2] = 0.0  # exact zeros next to the zero padding
    weight = rng.uniform(-0.2, 0.2, (32, 3, 3, 3, 1)).astype(np.float32)
    inputs, weights = mx.array(values), mx.array(weight)
    got = host_array(single_channel_conv3d(inputs, weights), copy=True)
    expected = host_array(mx.conv3d(inputs, weights, 1, 1, 1), copy=True)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
