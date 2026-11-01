"""Chunked Metal convolution compared with the pinned Torch CPU operator."""

from __future__ import annotations

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.kernels.metal_chunked_conv import chunked_conv2d
from nnunet_mlx.runtime import import_mlx


@pytest.mark.oracle
@pytest.mark.metal
def test_chunked_conv_matches_torch_cpu_operator_within_tight_tolerance() -> None:
    """Compare chunked convolution with the Torch CPU operator within tolerance."""
    torch = pytest.importorskip("torch")
    mx, _ = import_mlx()
    rng = np.random.default_rng(8)
    scale = np.float32(2.0**-5)
    values = (rng.uniform(-1.0, 1.0, (1, 5, 5, 512)) * scale).astype(np.float32)
    weight = (rng.uniform(-1.0, 1.0, (512, 3, 3, 512)) * scale).astype(np.float32)
    bias = (rng.uniform(-1.0, 1.0, 512) * scale).astype(np.float32)
    got = host_array(
        chunked_conv2d(mx.array(values), mx.array(weight), mx.array(bias), (1, 1), (1, 1)),
        copy=True,
    )
    with torch.inference_mode():
        expected = torch.nn.functional.conv2d(
            torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, -1, 1))),
            torch.from_numpy(np.ascontiguousarray(np.moveaxis(weight, -1, 1))),
            torch.from_numpy(bias), stride=[1, 1], padding=[1, 1],
        )
    expected_array = np.moveaxis(expected.numpy(), 1, -1)
    assert np.max(np.abs(got.astype(np.float64) - expected_array.astype(np.float64))) < 1e-6
