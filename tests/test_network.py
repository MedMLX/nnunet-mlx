"""Residual upsampling compared with the pinned Torch transpose convolution."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.model.network import NonOverlappingConvTranspose3d, ResidualEncoderUNet
from nnunet_mlx.model.topology import ResidualEncoderUNetSpec
from nnunet_mlx.runtime import import_mlx


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
