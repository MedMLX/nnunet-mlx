"""Residual inference arithmetic and upsampling against independent operation sequences."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.network import (
    BottleneckResidualBlock,
    NonOverlappingConvTranspose3d,
    ResidualBlock,
    ResidualEncoderUNet,
    SpatialInstanceNorm,
)
from nnunet_mlx.model.topology import ResidualEncoderUNetSpec
from nnunet_mlx.runtime import import_mlx


@pytest.mark.metal
@pytest.mark.parametrize("bottleneck", [False, True])
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_residual_block_preserves_separate_fp32_operations(
    tiny_residual_spec: ResidualEncoderUNetSpec,
    bottleneck: bool,
    project: bool,
    affine: bool,
) -> None:
    """Keep residual addition after normalization and before activation, bit for bit.

    Whole-network Torch tolerances cannot detect a changed FP32 rounding order.
    These blocks also cover identity and pooled/projected skip paths, batches,
    noncontiguous inputs, and non-affine normalization.
    """
    mx, nn = import_mlx()
    mx.random.seed(93)
    spec = replace(tiny_residual_spec, norm_affine=affine, conv_bias=affine, leaky_slope=0.1)
    stride = (2, 2, 2) if project else (1, 1, 1)
    output_channels = 8 if project else 4
    block = (
        BottleneckResidualBlock(
            4, 2, output_channels, kernel_size=(3, 3, 3), stride=stride, spec=spec,
        )
        if bottleneck else
        ResidualBlock(4, output_channels, kernel_size=(3, 3, 3), stride=stride, spec=spec)
    )
    block.eval()
    last = block.conv3 if isinstance(block, BottleneckResidualBlock) else block.conv2
    if affine:
        last.norm.weight = mx.array(np.linspace(-1.25, 1.75, output_channels, dtype=np.float32))
        last.norm.bias = mx.array(np.linspace(-0.4, 0.3, output_channels, dtype=np.float32))
    values = mx.array(np.random.default_rng(93).normal(size=(2, 8, 6, 20, 4)).astype(np.float32))
    values = values[:, :, :, ::2, :]
    residual = block.pool(values) if block.pool is not None else values
    residual = block.proj(residual) if block.proj is not None else residual
    hidden = block.conv1(values)
    if isinstance(block, BottleneckResidualBlock):
        hidden = block.conv2(hidden)
    expected = nn.leaky_relu(last(hidden) + residual, negative_slope=spec.leaky_slope)
    actual = block(values)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(
        host_array(actual).view(np.uint32), host_array(expected).view(np.uint32),
    )


@pytest.mark.metal
@pytest.mark.parametrize("wrong_dtype", [False, True])
def test_residual_normalization_rejects_incompatible_buffer(wrong_dtype: bool) -> None:
    """Reject broadcasting or half precision before the FP32 kernel indexes the skip buffer."""
    mx, _ = import_mlx()
    values = mx.ones((1, 2, 2, 2, 4))
    residual = (
        values.astype(mx.float16) if wrong_dtype else mx.ones((1, 2, 2, 2, 1))
    )
    layer = SpatialInstanceNorm(4, eps=1e-5, affine=True)
    with pytest.raises(ContractError):
        layer.normalize(values, residual=residual, negative_slope=0.01)


@pytest.mark.oracle
@pytest.mark.metal
def test_intermediate_residual_upsample_repeats_at_cads_shape(
    tiny_residual_spec: ResidualEncoderUNetSpec,
) -> None:
    """Check repeated intermediate residual upsampling at the CADS shape."""
    torch = pytest.importorskip("torch")
    mx, _ = import_mlx()
    spec = replace(tiny_residual_spec, features=(32, 64, 128), encoder_blocks=(1, 1, 1))
    model = ResidualEncoderUNet(spec)
    layer = cast(NonOverlappingConvTranspose3d, model["up0"])
    source = torch.nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2).eval()
    torch.manual_seed(83)
    with torch.no_grad():
        source.weight.uniform_(-0.05, 0.05)
        source.bias.uniform_(-0.05, 0.05)
    layer.weight = mx.array(source.weight.detach().numpy().transpose(1, 2, 3, 4, 0))
    layer.bias = mx.array(source.bias.detach().numpy())
    values = np.random.default_rng(83).normal(size=(1, 128, 64, 40, 40)).astype(np.float32)
    with torch.inference_mode():
        expected = np.moveaxis(source(torch.from_numpy(values)).numpy(), 1, -1)
    inputs = mx.array(np.moveaxis(values, 1, -1))
    reference = None
    for _ in range(3):
        output = layer(inputs)
        mx.eval(output)
        actual = host_array(output, copy=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        if reference is not None:
            np.testing.assert_array_equal(actual, reference)
        reference = actual
