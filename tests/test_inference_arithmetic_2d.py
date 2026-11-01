"""Ordered 2-D inference arithmetic against the pinned CPU oracle."""

from __future__ import annotations

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.prediction.arithmetic import NonOverlappingConvTranspose2d, OrderedConv2d
from nnunet_mlx.runtime import import_mlx, tree_flatten


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize(
    ("shape", "outputs", "kernel"),
    [
        ((1, 5, 5, 512), 512, (2, 2)),
        ((1, 10, 10, 512), 512, (2, 2)),
        ((1, 20, 20, 512), 512, (2, 2)),
        ((1, 40, 40, 512), 256, (2, 2)),
        ((1, 80, 80, 256), 128, (2, 2)),
        ((1, 160, 160, 128), 64, (2, 2)),
        ((1, 320, 320, 64), 32, (2, 2)),
        ((2, 3, 5, 257), 17, (2, 3)),
        ((1, 7, 11, 129), 3, (3, 2)),
        ((2, 1, 3, 5), 7, (1, 2)),
    ],
)
def test_nonoverlapping_2d_transpose_matches_pinned_cpu(
    shape: tuple[int, ...],
    outputs: int,
    kernel: tuple[int, int],
) -> None:
    """Compare nonoverlapping 2D transpose convolution with the pinned CPU reference."""
    torch = pytest.importorskip("torch")
    mx, nn = import_mlx()
    torch.manual_seed(604)
    source = torch.nn.ConvTranspose2d(shape[-1], outputs, kernel, stride=kernel).eval()
    layer = nn.ConvTranspose2d(shape[-1], outputs, kernel, stride=kernel)
    layer.weight = mx.array(source.weight.detach().numpy().transpose(1, 2, 3, 0))
    layer.bias = mx.array(source.bias.detach().numpy())
    candidate = NonOverlappingConvTranspose2d(layer)
    candidate.eval()
    data = np.random.default_rng(409).normal(size=shape).astype(np.float32)
    with torch.inference_mode():
        expected = source(torch.from_numpy(np.ascontiguousarray(np.moveaxis(data, -1, 1))))
    actual = host_array(candidate(mx.array(data)))
    np.testing.assert_allclose(actual, np.moveaxis(expected.numpy(), 1, -1), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual, host_array(candidate(mx.array(data))))
    for name, value in tree_flatten(layer.parameters()):
        np.testing.assert_array_equal(
            host_array(value), host_array(dict(tree_flatten(candidate.parameters()))[name])
        )


def _blocked_convolution_reference(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    padding: int,
) -> np.ndarray:
    """Compute the fixture convolution with the recorded blocked reduction order."""
    batch, height, width, channels = values.shape
    outputs, kh, kw, _ = weight.shape
    padded = np.pad(values, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    result = np.zeros((batch, height, width, outputs), dtype=np.float32)
    chunk = 128 if (kh, kw) == (1, 1) else channels
    block = min(8, channels)
    for start in range(0, channels, chunk):
        partial = np.broadcast_to(bias if start == 0 else np.zeros_like(bias), result.shape).copy()
        for base in range(start, min(start + chunk, channels), block):
            for y in range(kh):
                for x in range(kw):
                    for channel in range(base, base + block):
                        value = padded[:, y : y + height, x : x + width, channel, None].astype(
                            np.float64
                        )
                        partial = (
                            partial.astype(np.float64)
                            + value * weight[:, y, x, channel].astype(np.float64)
                        ).astype(np.float32)
        result = partial if start == 0 else result + partial
    return result


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize(
    ("channels", "outputs", "kernel", "bias", "spatial"),
    [
        (1, 8, 3, True, (5, 7)),
        (8, 16, 3, True, (5, 7)),
        (128, 24, 3, True, (5, 7)),
        (256, 32, 1, False, (5, 7)),
        (512, 26, 1, True, (5, 7)),
        (1, 64, 3, True, (8, 8)),
        (8, 64, 3, True, (8, 8)),
        (256, 64, 1, True, (8, 8)),
        (8, 64, 3, True, (8, 9)),
    ],
)
def test_ordered_2d_convolution_preserves_block_and_projection_rounding(
    channels: int,
    outputs: int,
    kernel: int,
    bias: bool,
    spatial: tuple[int, int],
) -> None:
    """Preserve CPU block and projection rounding in ordered 2D convolution."""
    torch = pytest.importorskip("torch")
    mx, nn = import_mlx()
    torch.manual_seed(906)
    source = torch.nn.Conv2d(channels, outputs, kernel, padding=kernel // 2, bias=bias).eval()
    layer = nn.Conv2d(channels, outputs, kernel, padding=kernel // 2, bias=bias)
    weight = source.weight.detach().numpy().transpose(0, 2, 3, 1)
    offset = source.bias.detach().numpy() if bias else np.zeros(outputs, np.float32)
    layer.weight = mx.array(weight)
    if bias:
        layer.bias = mx.array(offset)
    candidate = OrderedConv2d(layer)
    candidate.eval()
    values = np.random.default_rng(291).normal(size=(2, *spatial, channels)).astype(np.float32)
    actual = host_array(candidate(mx.array(values)))
    expected = _blocked_convolution_reference(values, weight, offset, kernel // 2)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, host_array(candidate(mx.array(values))))
    with torch.inference_mode():
        cpu = source(torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, -1, 1))))
    np.testing.assert_allclose(actual, np.moveaxis(cpu.numpy(), 1, -1), rtol=1e-4, atol=1e-4)
    assert set(dict(tree_flatten(layer.parameters()))) == set(
        dict(tree_flatten(candidate.parameters()))
    )
