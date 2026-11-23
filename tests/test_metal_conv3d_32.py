"""Check the float32 reduction order of the full-resolution convolution on Metal."""

from __future__ import annotations

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.kernels.metal_conv3d_32 import conv3d_32
from nnunet_mlx.runtime import import_mlx


@pytest.mark.metal
def test_conv3d_32_matches_native_bits_for_strided_inputs_and_weights() -> None:
    """Cover strip boundaries, zero padding, and both weight slices of a decoder layer."""
    mx, _ = import_mlx()
    rng = np.random.default_rng(291)
    features = mx.array(rng.standard_normal((2, 5, 7, 32, 64), dtype=np.float32))
    weights = mx.array(rng.standard_normal((32, 3, 3, 3, 64), dtype=np.float32))
    for start in (0, 32):
        values = features[..., start:start + 32]
        weight = weights[..., start:start + 32]
        actual = host_array(conv3d_32(values, weight), copy=True)
        expected = host_array(mx.conv3d(values, weight, 1, 1), copy=True)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
