"""Network oracle using installed Torch and dynamic-network-architectures.

The caller must establish the pinned oracle environment before running these
helpers; importing the installed modules alone does not verify their revisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nnunet_reference.arrays import host_array

if TYPE_CHECKING:
    from nnunet_mlx.model.network import PlainConvUNet, ResidualEncoderUNet


if TYPE_CHECKING:
    from torch.nn import Module as TorchModule

import os
from importlib import import_module
from pathlib import Path

import numpy as np

from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.storage.weights import convert_state_dict, load_converted_state
from nnunet_reference.oracle.metrics import (
    FloatArray,
    ForwardParityResult,
    measure_forward_parity,
)


def real_image_input(shape: tuple[int, ...], channels: int) -> FloatArray:
    """Read a normalized CT or X-ray crop in channel-first network order."""
    if len(shape) == 3:
        default = Path.home() / ".cache" / "medmlx" / "test-data" / "hcc_063_ct.nii.gz"
        path = os.environ.get("MEDMLX_TEST_CT", str(default))
        nibabel = import_module("nibabel")
        ct = np.asarray(nibabel.load(path).dataobj, dtype=np.float32)
        image = ct[200:200 + shape[0], 200:200 + shape[1], 40:40 + shape[2]] / np.float32(1024)
    elif len(shape) == 2:
        default = Path.home() / ".cache" / "medmlx" / "test-data" / "chest_xray.jpg"
        path = os.environ.get("MEDMLX_TEST_XRAY", str(default))
        image_io = import_module("skimage.io")
        xray = image_io.imread(path)
        if xray.ndim == 3:
            xray = xray[..., 0]
        image = (
            np.asarray(xray[200:200 + shape[0], 200:200 + shape[1]], dtype=np.float32)
            / np.float32(255)
        )
    else:
        raise ValueError(f"Expected 2D or 3D image shape, got {shape}")
    if image.shape != shape:
        raise ValueError(f"real-image crop {image.shape} does not match {shape}")
    return np.broadcast_to(image, (channels, *shape)).copy()


def build_torch_plain_conv_unet(spec: PlainConvUNetSpec, *, seed: int) -> TorchModule:
    """Construct an untrained evaluation network, resetting Torch's global RNG to ``seed``."""
    torch = import_module("torch")
    unet = import_module("dynamic_network_architectures.architectures.unet")
    torch.manual_seed(seed)
    conv_op = torch.nn.Conv2d if spec.spatial_dims == 2 else torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm2d if spec.spatial_dims == 2 else torch.nn.InstanceNorm3d
    return unet.PlainConvUNet(
        input_channels=spec.input_channels,
        n_stages=len(spec.features),
        features_per_stage=spec.features,
        conv_op=conv_op,
        kernel_sizes=spec.kernel_sizes,
        strides=spec.strides,
        n_conv_per_stage=spec.encoder_convs,
        num_classes=spec.num_classes,
        n_conv_per_stage_decoder=spec.decoder_convs,
        conv_bias=spec.conv_bias,
        norm_op=norm_op,
        norm_op_kwargs={"eps": spec.norm_eps, "affine": spec.norm_affine},
        dropout_op=None,
        dropout_op_kwargs={},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": spec.leaky_slope, "inplace": False},
        deep_supervision=False,
        nonlin_first=False,
    ).eval()


def run_forward_parity(
    spec: PlainConvUNetSpec,
    *,
    seed: int = 17,
    spatial_shape: tuple[int, ...] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> ForwardParityResult:
    """Compare a seeded, untrained Torch network and its converted MLX counterpart.

    Input is real-image float32 data, with one sample and the supplied
    spatial shape (eight voxels per axis by default). The result's pass flag is
    elementwise allclose; argmax agreement is reported separately. Neither
    pretrained checkpoint compatibility nor clinical performance is tested.
    """
    source = build_torch_plain_conv_unet(spec, seed=seed)
    from nnunet_mlx.model.network import PlainConvUNet

    candidate = PlainConvUNet(spec)
    return _run_network_forward_parity(
        source,
        candidate,
        spatial_dims=spec.spatial_dims,
        input_channels=spec.input_channels,
        seed=seed,
        spatial_shape=spatial_shape,
        rtol=rtol,
        atol=atol,
    )


def build_torch_residual_encoder_unet(spec: ResidualEncoderUNetSpec, *, seed: int) -> TorchModule:
    """Construct an untrained residual evaluation network, resetting Torch's global RNG."""
    torch = import_module("torch")
    unet = import_module("dynamic_network_architectures.architectures.unet")
    residual = import_module("dynamic_network_architectures.building_blocks.residual")
    torch.manual_seed(seed)
    conv_op = torch.nn.Conv2d if spec.spatial_dims == 2 else torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm2d if spec.spatial_dims == 2 else torch.nn.InstanceNorm3d
    block = residual.BasicBlockD if spec.bottleneck_channels is None else residual.BottleneckD
    return unet.ResidualEncoderUNet(
        input_channels=spec.input_channels,
        n_stages=len(spec.features),
        features_per_stage=spec.features,
        conv_op=conv_op,
        kernel_sizes=spec.kernel_sizes,
        strides=spec.strides,
        n_blocks_per_stage=spec.encoder_blocks,
        num_classes=spec.num_classes,
        n_conv_per_stage_decoder=spec.decoder_convs,
        conv_bias=spec.conv_bias,
        norm_op=norm_op,
        norm_op_kwargs={"eps": spec.norm_eps, "affine": spec.norm_affine},
        dropout_op=None,
        dropout_op_kwargs={},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": spec.leaky_slope, "inplace": False},
        stem_channels=spec.stem_channels,
        block=block,
        bottleneck_channels=spec.bottleneck_channels,
    ).eval()


def run_residual_forward_parity(
    spec: ResidualEncoderUNetSpec,
    *,
    seed: int = 29,
    spatial_shape: tuple[int, ...] | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> ForwardParityResult:
    """Compare residual-network logits using the same gate as plain UNet."""
    source = build_torch_residual_encoder_unet(spec, seed=seed)
    from nnunet_mlx.model.network import ResidualEncoderUNet

    candidate = ResidualEncoderUNet(spec)
    return _run_network_forward_parity(
        source,
        candidate,
        spatial_dims=spec.spatial_dims,
        input_channels=spec.input_channels,
        seed=seed,
        spatial_shape=spatial_shape,
        rtol=rtol,
        atol=atol,
    )


def _run_network_forward_parity(
    source: TorchModule,
    candidate: PlainConvUNet | ResidualEncoderUNet,
    *,
    spatial_dims: int,
    input_channels: int,
    seed: int,
    spatial_shape: tuple[int, ...] | None,
    rtol: float,
    atol: float,
) -> ForwardParityResult:
    """Compare source and native forward outputs for a constructed network."""
    torch = import_module("torch")
    mx, _ = import_mlx()
    shape = spatial_shape or (8,) * spatial_dims
    if len(shape) != spatial_dims:
        raise ValueError(f"spatial shape {shape} does not match {spatial_dims}-D architecture")
    load_converted_state(candidate, convert_state_dict(source.state_dict()))
    # Exercise inference-only stage boundaries, as the model-folder loader does.
    candidate.eval()
    values = real_image_input(shape, input_channels)[None]
    with torch.inference_mode():
        expected = source(torch.from_numpy(values)).numpy()
    channels_last = np.transpose(values, (0, *range(2, values.ndim), 1))
    actual = candidate(mx.array(channels_last))
    mx.eval(actual)
    actual_host = np.transpose(
        host_array(actual), (0, actual.ndim - 1, *range(1, actual.ndim - 1))
    ).astype(np.float32)
    return ForwardParityResult(
        expected=expected.astype(np.float32),
        actual=actual_host,
        metrics=measure_forward_parity(expected, actual_host, seed=seed, rtol=rtol, atol=atol),
    )


def converted_candidate(spec: PlainConvUNetSpec, source: TorchModule) -> PlainConvUNet:
    """Convert and load a source state dict; return the candidate without setting eval mode."""
    from nnunet_mlx.model.network import PlainConvUNet

    candidate = PlainConvUNet(spec)
    load_converted_state(candidate, convert_state_dict(source.state_dict()))
    return candidate
