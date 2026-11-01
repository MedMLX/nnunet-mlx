"""Ordered 3-D convolution and normalization against the pinned CPU oracle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from nnunet_reference.arrays import host_array

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_instance_norm import (
    affine_from_statistics,
    instance_norm,
    inverse_standard_deviation,
    spatial_statistics,
)
from nnunet_mlx.model.network import PlainConvUNet
from nnunet_mlx.model.topology import PlainConvUNetSpec
from nnunet_mlx.prediction.arithmetic import OrderedConv3d, apply_inference_arithmetic
from nnunet_mlx.runtime import import_mlx, tree_flatten


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize(
    ("channels", "outputs", "stride"),
    [
        (1, 8, (1, 1, 1)),
        (32, 8, (1, 1, 1)),
        (128, 8, (1, 1, 1)),
        (1, 64, (1, 1, 1)),
        (3, 64, (1, 1, 1)),
        (32, 64, (2, 1, 2)),
    ],
)
def test_ordered_convolution_matches_bias_first_cpu_reduction(
    channels: int,
    outputs: int,
    stride: tuple[int, int, int],
) -> None:
    """Match ordered convolution to the recorded bias-first CPU reduction."""
    torch = pytest.importorskip("torch")
    mx, nn = import_mlx()
    torch.manual_seed(813)
    source = torch.nn.Conv3d(channels, outputs, 3, padding=1, stride=stride).eval()
    native = nn.Conv3d(channels, outputs, 3, padding=1, stride=stride)
    native.weight = mx.array(np.moveaxis(source.weight.detach().numpy(), 1, -1))
    native.bias = mx.array(source.bias.detach().numpy())
    candidate = OrderedConv3d(native)
    candidate.eval()
    data = np.random.default_rng(34).normal(size=(2, channels, 8, 8, 8)).astype(np.float32)
    with torch.inference_mode():
        expected = source(torch.from_numpy(data)).numpy()
    actual = candidate(mx.array(np.moveaxis(data, 1, -1)))
    mx.eval(actual)
    np.testing.assert_array_equal(np.moveaxis(host_array(actual), -1, 1), expected)


@pytest.mark.oracle
@pytest.mark.metal
def test_ordered_normalization_matches_cpu_statistics_and_affine() -> None:
    """Compare ordered normalization statistics and affine output with CPU results."""
    torch = pytest.importorskip("torch")
    mx, _ = import_mlx()
    rng = np.random.default_rng(39)
    data = rng.normal(size=(2, 5, 8, 8, 8)).astype(np.float32)
    weight = rng.normal(size=5).astype(np.float32)
    bias = rng.normal(size=5).astype(np.float32)
    with torch.inference_mode():
        expected = torch.nn.functional.instance_norm(
            torch.from_numpy(data), weight=torch.from_numpy(weight), bias=torch.from_numpy(bias),
            eps=1e-5,
        ).numpy()
    actual = instance_norm(
        mx.array(np.moveaxis(data, 1, -1)), mx.array(weight), mx.array(bias), 1e-5,
    )
    mx.eval(actual)
    np.testing.assert_array_equal(np.moveaxis(host_array(actual), -1, 1), expected)


def test_inverse_square_root_resolves_near_float32_midpoint() -> None:
    """Check inverse-square-root rounding near an FP32 midpoint."""
    mx, _ = import_mlx()
    # One correction rounded down; the CPU oracle rounds to the upper neighbour.
    variance = mx.array([0.0005881802062503994], dtype=mx.float32)
    actual = inverse_standard_deviation(variance, 1e-5)
    mx.eval(actual)
    np.testing.assert_array_equal(host_array(actual), np.array([40.88688278198242], np.float32))


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize("output_fma", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 5, 5, 512),
        (1, 10, 10, 512),
        (1, 20, 20, 512),
        (1, 40, 40, 512),
        (1, 80, 80, 256),
        (1, 160, 160, 128),
        (1, 320, 320, 64),
        (1, 640, 640, 32),
        (2, 17, 19, 7),
        (2, 3, 5, 257),
        (1, 1, 3, 5),
        (1, 3, 3, 3, 2),
        (1, 5, 7, 11, 7),
    ],
)
def test_ordered_normalization_arbitrary_spatial_statistics_and_affine(
    shape: tuple[int, ...],
    output_fma: bool,
) -> None:
    """Compare ordered statistics and affine output across spatial sizes."""
    torch = pytest.importorskip("torch")
    mx, _ = import_mlx()
    rng = np.random.default_rng(604)
    values = rng.normal(size=shape).astype(np.float32)
    weight = rng.normal(size=shape[-1]).astype(np.float32)
    bias = rng.normal(size=shape[-1]).astype(np.float32)
    axes = tuple(range(1, len(shape) - 1))
    source = torch.from_numpy(values)
    mean = source.double().mean(dim=axes, keepdim=True).float()
    variance = ((source - mean) ** 2).double().sum(dim=axes, keepdim=True).float()
    variance = variance / int(np.prod(shape[1:-1]))
    inverse = (variance.double() + 1e-5).rsqrt().float()
    alpha = inverse * torch.from_numpy(weight)
    beta = (-mean.double() * alpha.double() + torch.from_numpy(bias).double()).float()
    affine = (
        (source.double() * alpha.double() + beta.double()).float()
        if output_fma
        else source * alpha + beta
    ).numpy()
    arguments = (mx.array(values), mx.array(weight), mx.array(bias), 1e-5)
    actual_stats = spatial_statistics(mx.array(values), 1e-5)
    actual = instance_norm(*arguments, output_fma=output_fma)
    mx.eval(*actual_stats, actual)
    for observed, expected in zip(actual_stats, (mean, variance, inverse), strict=True):
        assert observed.shape == (shape[0], *((1,) * len(axes)), shape[-1])
        np.testing.assert_array_equal(host_array(observed), expected.numpy())
    np.testing.assert_array_equal(host_array(actual), affine)
    with torch.inference_mode():
        nchw = torch.from_numpy(np.ascontiguousarray(np.moveaxis(values, -1, 1)))
        _, saved_mean, saved_inverse = torch.ops.aten.native_batch_norm.default(
            nchw.reshape(1, shape[0] * shape[-1], *shape[1:-1]),
            torch.from_numpy(weight).repeat(shape[0]), torch.from_numpy(bias).repeat(shape[0]),
            None, None, True, 0.0, 1e-5,
        )
        expected_output = torch.nn.functional.instance_norm(
            nchw, weight=torch.from_numpy(weight), bias=torch.from_numpy(bias), eps=1e-5,
        ).numpy()
    np.testing.assert_allclose(
        np.moveaxis(host_array(actual), -1, 1), expected_output, rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_array_equal(host_array(actual_stats[0]).reshape(-1), saved_mean.numpy())
    np.testing.assert_array_equal(host_array(actual_stats[2]).reshape(-1), saved_inverse.numpy())
    injected = affine_from_statistics(
        mx.array(values), mx.array(saved_mean.numpy()), mx.array(saved_inverse.numpy()),
        mx.array(weight), mx.array(bias), output_fma=output_fma,
    )
    np.testing.assert_array_equal(host_array(actual), host_array(injected))
    repeated = instance_norm(*arguments, output_fma=output_fma)
    np.testing.assert_array_equal(host_array(actual), host_array(repeated))


@pytest.mark.parametrize(
    ("encoder_convs", "decoder_convs"),
    [((2, 2), (2,)), ((1, 3), (1,)), ((3, 1), (3,))],
)
def test_totalsegmentator_fusions_preserve_logits_and_checkpoint_updates(
    tiny_spec: PlainConvUNetSpec,
    encoder_convs: tuple[int, ...],
    decoder_convs: tuple[int, ...],
) -> None:
    """Preserve logits and checkpoint changes through qualified fusions."""
    from nnunet_mlx.prediction.source import TOTALSEGMENTATOR_CT_SOURCE_PROFILE

    mx, _ = import_mlx()
    mx.random.seed(412)
    spec = replace(tiny_spec, encoder_convs=encoder_convs, decoder_convs=decoder_convs)
    generic, optimized = PlainConvUNet(spec), PlainConvUNet(spec)
    for model in (generic, optimized):
        model.eval()
    parameters = tree_flatten(generic.parameters())
    optimized.load_weights(parameters)
    apply_inference_arithmetic(generic, "torch-order-fp32")
    apply_inference_arithmetic(
        optimized, "torch-order-fp32", source_profile=TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
    )
    values = mx.random.normal((1, 8, 8, 8, 1))
    expected, actual = host_array(generic(values)), host_array(optimized(values))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    updated = [(name, parameter + 0.125) for name, parameter in parameters]
    generic.load_weights(updated)
    optimized.load_weights(updated)
    expected_updated = host_array(generic(values))
    actual_updated = host_array(optimized(values))
    assert np.any(expected_updated != expected)
    np.testing.assert_array_equal(actual_updated.view(np.uint32), expected_updated.view(np.uint32))
    optimized.train()
    with pytest.raises(ContractError):
        optimized(values)


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0000001192092896, 0.9999999403953552],
        [100000000.0, 1.0, -100000000.0],
        [0.125] * 127 + [-0.125, 0.25],
    ],
)
def test_ordered_norm_nonpower_reduction_edges(
    values: list[float],
) -> None:
    """Check ordered normalization at non-power-of-two reduction sizes."""
    torch = pytest.importorskip("torch")
    mx, _ = import_mlx()
    source = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
    mean = source.double().mean(dim=(1, 2), keepdim=True).float()
    variance = ((source - mean) ** 2).double().sum(dim=(1, 2), keepdim=True).float()
    variance = variance / source.numel()
    inverse = (variance.double() + 1e-5).rsqrt().float()
    actual_stats = spatial_statistics(mx.array(source.numpy()), 1e-5)
    for actual, expected in zip(actual_stats, (mean, variance, inverse), strict=True):
        np.testing.assert_array_equal(host_array(actual), expected.numpy())
    with torch.inference_mode():
        expected_output = torch.nn.functional.instance_norm(source.movedim(-1, 1)).numpy()
    actual_output = instance_norm(mx.array(source.numpy()), mx.ones((1,)), mx.zeros((1,)), 1e-5)
    np.testing.assert_array_equal(np.moveaxis(host_array(actual_output), -1, 1), expected_output)
