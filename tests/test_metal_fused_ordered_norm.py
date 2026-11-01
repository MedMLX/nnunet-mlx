"""Ordered statistics, fused affine/activation, and positive-zero halo parity."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_mlx.runtime import import_mlx

pytestmark = pytest.mark.metal


# Padding applies the positive-zero halo through the precomputed-statistics entry point.
@pytest.mark.parametrize(
    ("shape", "distribution", "output_fma", "padding"),
    [
        ((1, 3, 5, 7, 4), "random", False, None),
        ((1, 16, 16, 16, 16), "random", True, None),
        ((1, 2, 129, 8), "constant", False, None),
        ((2, 7, 3, 13, 3), "large_offset", True, None),
        ((1, 2, 129, 8), "mixed_scales", False, None),
        ((1, 16, 16, 16, 16), "mixed_scales", True, (2, 0, 1)),
        ((1, 3, 5, 7, 1), "random", True, (2, 0, 1)),
        ((1, 3, 5, 7, 31), "random", False, (2, 0, 1)),
        ((2, 5, 7, 9, 33), "random", True, None),
        ((1, 8, 8, 8, 64), "random", False, (2, 0, 1)),
    ],
)
def test_fused_normalization_activation_matches_ordered_engine_bits(
    shape: tuple[int, ...],
    distribution: str,
    output_fma: bool,
    padding: tuple[int, ...] | None,
) -> None:
    """Match queued statistics, normalization and activation to ordered arithmetic bits."""
    mx, nn = import_mlx()
    from nnunet_mlx.kernels.metal_fused_ordered_norm import (
        affine_activation_padded_from_statistics,
        instance_norm_activation,
    )
    from nnunet_mlx.kernels.metal_instance_norm import instance_norm as original_instance_norm
    from nnunet_mlx.kernels.metal_instance_norm import spatial_statistics

    rng = np.random.default_rng(621)
    if distribution == "random":
        data = rng.normal(size=shape).astype(np.float32)
    elif distribution == "constant":
        data = np.full(shape, 3.25, dtype=np.float32)
    elif distribution == "large_offset":
        data = np.float32(1e6) + rng.normal(size=shape).astype(np.float32)
    else:
        scales = np.array([1e17, -1e17, 1e-30, -1e-30, 0], dtype=np.float32)
        data = scales[rng.integers(0, len(scales), size=shape)]
    values = mx.array(data)
    weight = mx.array(rng.normal(size=shape[-1]).astype(np.float32))
    bias = mx.array(rng.normal(size=shape[-1]).astype(np.float32))
    expected = nn.leaky_relu(
        original_instance_norm(values, weight, bias, 1e-5, output_fma=output_fma),
        negative_slope=0.01,
    )
    if padding is None:
        actual = instance_norm_activation(values, weight, bias, 1e-5, 0.01, output_fma=output_fma)
    else:
        average, _, inverse = spatial_statistics(values, 1e-5)
        actual = affine_activation_padded_from_statistics(
            values, average, inverse, weight, bias, 0.01, padding, output_fma=output_fma,
        )
    mx.eval(expected, actual)
    assert actual.dtype == mx.float32
    expected_host = np.array(expected, dtype=np.float32)
    assert np.isfinite(expected_host).all()
    if padding is not None:
        expected_host = np.pad(
            expected_host.transpose(0, 4, 1, 2, 3),
            ((0, 0), (0, 0), *((p, p) for p in padding)),
        )
    assert actual.shape == expected_host.shape
    np.testing.assert_array_equal(np.array(actual).view(np.uint32), expected_host.view(np.uint32))


@pytest.mark.parametrize("slope", [0.0, 0.01, -0.5])
@pytest.mark.parametrize("output_fma", [False, True])
@pytest.mark.parametrize("negative_bias", [False, True])
@pytest.mark.parametrize("padding", [None, (1, 0, 2)])
def test_fused_affine_activation_preserves_zero_ties_and_nan_payloads(
    slope: float,
    output_fma: bool,
    negative_bias: bool,
    padding: tuple[int, ...] | None,
) -> None:
    """Preserve zero-tie and NaN behavior in fused affine activation."""
    mx, nn = import_mlx()
    from nnunet_mlx.kernels.metal_fused_ordered_norm import (
        affine_activation_from_statistics,
        affine_activation_padded_from_statistics,
    )
    from nnunet_mlx.kernels.metal_instance_norm import affine_from_statistics

    # Signed zeros, +-1, +-inf, quiet NaN payloads, subnormals and the smallest normals.
    bits = np.array(
        [0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000,
         0x7FC12345, 0x00000001, 0x80000001, 0x00800000, 0x80800000],
        dtype=np.uint32,
    )
    values = mx.array(bits.view(np.float32).reshape(1, 1, 1, -1, 1))
    average = mx.zeros((1,), dtype=mx.float32)
    inverse, weight = mx.ones((1,), dtype=mx.float32), mx.ones((1,), dtype=mx.float32)
    bias = mx.array(np.array([-0.0 if negative_bias else 0.0], dtype=np.float32))
    expected = nn.leaky_relu(
        affine_from_statistics(values, average, inverse, weight, bias, output_fma=output_fma),
        negative_slope=slope,
    )
    arguments = (values, average, inverse, weight, bias, slope)
    actual = (
        affine_activation_from_statistics(*arguments, output_fma=output_fma)
        if padding is None
        else affine_activation_padded_from_statistics(*arguments, padding, output_fma=output_fma)
    )
    mx.eval(expected, actual)
    expected_host = np.array(expected, dtype=np.float32)
    if padding is not None:
        expected_host = np.pad(
            expected_host.transpose(0, 4, 1, 2, 3),
            ((0, 0), (0, 0), *((p, p) for p in padding)),
        )
    np.testing.assert_array_equal(np.array(actual).view(np.uint32), expected_host.view(np.uint32))
