"""Plan patch geometry from training images when no plans file is supplied."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from math import ceil, prod
from pathlib import Path
from typing import cast

import numpy as np

from nnunet_mlx.contracts import parse_number, parse_sequence
from nnunet_mlx.data.cases import RawInputCase
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.storage.imageio import read_images, resolve_image_reader

_DEFAULT_READERS = {
    ".nii.gz": "SimpleITKIO",
    ".nii": "NibabelIO",
    ".mha": "SimpleITKIO",
    ".nrrd": "SimpleITKIO",
    ".gipl": "SimpleITKIO",
    ".png": "NaturalImage2DIO",
    ".bmp": "NaturalImage2DIO",
    ".tif": "Tiff3DIO",
    ".tiff": "Tiff3DIO",
}


def select_reader(file_ending: str, override: object) -> str:
    """Select an existing reader from the dataset override or file extension."""
    if override is not None and not isinstance(override, str):
        raise ContractError("overwrite_image_reader_writer must be a reader name")
    reader = override if override and override != "None" else _DEFAULT_READERS.get(
        file_ending.casefold()
    )
    if reader is None:
        raise ContractError(f"no image reader for {file_ending!r}")
    return resolve_image_reader(reader, file_ending=file_ending)


def native_plans(
    cases: Sequence[RawInputCase],
    *,
    dataset_name: str,
    image_channels: int,
    reader: str,
    configuration: str | None,
    patch_size: tuple[int, ...] | None,
) -> PlansContract:
    """Choose median training spacing and an anisotropy-aware plain U-Net topology."""
    geometry = [_image_geometry(case.image_files[0], reader) for case in cases]
    shapes, spacings = zip(*geometry, strict=True)
    spacing = tuple(float(value) for value in np.median(spacings, axis=0))
    flat_axis = next((axis for axis in range(3) if all(shape[axis] == 1 for shape in shapes)), None)
    native_2d = flat_axis is not None
    name = configuration or ("2d" if native_2d else "3d_fullres")
    if name not in ("2d", "3d_fullres") or (native_2d and name == "3d_fullres"):
        raise ContractError(f"cannot generate configuration {name!r} for this dataset")
    rank = 2 if name == "2d" else 3
    leading = flat_axis if flat_axis is not None else max(range(3), key=spacing.__getitem__)
    forward = (leading, *(axis for axis in range(3) if axis != leading))
    axes = forward[-rank:]
    network_spacing = tuple(spacing[axis] for axis in axes)
    extents = np.asarray(shapes) * np.asarray(spacings) / np.asarray(spacing)
    if patch_size is None:
        cap = 256 if rank == 2 else 64
        median = tuple(float(value) for value in np.median(extents, axis=0))
        patch_size = tuple(min(cap, max(16, 16 * ceil(median[axis] / 16))) for axis in axes)
    if len(patch_size) != rank:
        raise ContractError(f"{name} requires a {rank}-D patch size")
    strides, kernels = _topology(patch_size, network_spacing)
    stages = len(strides)
    config: dict[str, object] = {
        "data_identifier": f"MLX_{name}",
        "patch_size": patch_size,
        "spacing": network_spacing,
        "batch_size": 2 if rank == 2 else 1,
        "batch_dice": True,
        "normalization_schemes": ["ZScoreNormalization"] * image_channels,
        "use_mask_for_norm": [True] * image_channels,
        "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": {
                "features_per_stage": [min(16 << level, 192) for level in range(stages)],
                "conv_op": f"torch.nn.Conv{rank}d",
                "norm_op": f"torch.nn.InstanceNorm{rank}d",
                "norm_op_kwargs": {"affine": True, "eps": 1e-5},
                "nonlin": "torch.nn.LeakyReLU",
                "kernel_sizes": kernels,
                "strides": strides,
                "n_conv_per_stage": [2] * stages,
                "n_conv_per_stage_decoder": [2] * (stages - 1),
                "conv_bias": True,
            },
        },
    }
    for kind in ("data", "seg", "probabilities"):
        field = f"resampling_fn_{kind}"
        is_seg = kind == "seg"
        order = 0 if is_seg else 1
        config[field] = "resample_data_or_seg_to_shape"
        config[f"{field}_kwargs"] = {
            "is_seg": is_seg,
            "order": order,
            "order_z": order,
            "force_separate_z": None,
        }
    return PlansContract.from_payload(
        {
            "dataset_name": dataset_name,
            "image_reader_writer": reader,
            "transpose_forward": forward,
            "transpose_backward": [forward.index(axis) for axis in range(3)],
            "configurations": {name: config},
        },
        configuration=name,
    )


def _topology(
    patch: tuple[int, ...], spacing: tuple[float, ...]
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Downsample adequately sampled axes while retaining a spatial bottleneck."""
    if any(size < 2 for size in patch):
        raise ContractError("each network patch dimension must be at least two voxels")
    shape = patch
    current_spacing = spacing
    strides = [(1,) * len(patch)]
    kernels = [_kernel(current_spacing)]
    for _ in range(4):
        eligible = [axis for axis, size in enumerate(shape) if size >= 8 and size % 2 == 0]
        if not eligible:
            break
        finest = min(current_spacing[axis] for axis in eligible)
        stride = tuple(
            2 if axis in eligible and value <= 2 * finest else 1
            for axis, value in enumerate(current_spacing)
        )
        shape = tuple(size // step for size, step in zip(shape, stride, strict=True))
        current_spacing = tuple(
            value * step for value, step in zip(current_spacing, stride, strict=True)
        )
        strides.append(stride)
        kernels.append(_kernel(current_spacing))
    if len(strides) == 1 or prod(shape) < 2:
        raise ContractError("patch size must admit at least one downsampling stage")
    return strides, kernels


def _kernel(spacing: tuple[float, ...]) -> tuple[int, ...]:
    finest = min(spacing)
    return tuple(3 if value <= 2 * finest else 1 for value in spacing)


def _image_geometry(path: Path, reader: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Read geometry in array order, from headers for the default volume readers."""
    if reader == "SimpleITKIO":
        header = import_module("SimpleITK").ImageFileReader()
        header.SetFileName(str(path))
        header.ReadImageInformation()
        shape = cast(tuple[int, ...], header.GetSize())[::-1]
        spacing = cast(tuple[float, ...], header.GetSpacing())[::-1]
        if len(shape) == 2:
            return (1, *shape), (max(spacing) * 999, *spacing)
        if len(shape) == 4:
            # The reader exposes the fourth SimpleITK axis as channels in CZYX order.
            return shape[1:], spacing[1:]
        if len(shape) == 3:
            return shape, spacing
    elif reader == "NibabelIO":
        image = import_module("nibabel").load(str(path))
        shape = cast(tuple[int, ...], image.shape)[::-1]
        if len(shape) == 3:
            return shape, cast(tuple[float, ...], image.header.get_zooms())[::-1]
    else:
        data, properties = read_images(reader, (path,))
        spacing = tuple(
            parse_number(value, name="image spacing")
            for value in parse_sequence(properties.get("spacing"), name="image spacing")
        )
        return data.shape[1:], spacing
    raise ContractError(f"unsupported spatial dimensions in {path.name}")
