"""Pinned nnU-Net oracle: resampling."""

from __future__ import annotations

from importlib import import_module

import numpy as np

from nnunet_reference.oracle.metrics import (
    ContractParityResult,
    ExactParityResult,
    array_metrics,
    exact_array_parity,
)
from nnunet_reference.oracle.network import real_image_input
from nnunet_reference.real_cases import liver_crop, tumor_crop


def run_resampling_parity(*, anisotropic: bool) -> ContractParityResult:
    """Compare an HCC-TACE CT crop resampled with pinned nnU-Net v2.8.1.

    Uses linear in-plane interpolation and nearest out-of-plane interpolation;
    anisotropy selects the spacing fixture. The array gate uses rtol=atol=1e-6.
    """

    from nnunet_mlx.data.resampling import resample_probabilities_or_data
    from nnunet_mlx.plans import ResamplingContract

    source = import_module("nnunetv2.preprocessing.resampling.default_resampling")
    data = real_image_input((4, 5, 6), 2)
    current_spacing = (5.0, 1.0, 1.0) if anisotropic else (1.2, 1.0, 0.8)
    new_spacing = (2.5, 0.8, 0.8) if anisotropic else (0.9, 1.1, 1.0)
    new_shape = (7, 6, 8)
    expected = np.asarray(
        source.resample_data_or_seg_to_shape(
            data,
            new_shape,
            current_spacing,
            new_spacing,
            is_seg=False,
            order=1,
            order_z=0,
            force_separate_z=None,
        ),
        dtype=np.float32,
    )
    actual = resample_probabilities_or_data(
        data,
        new_shape,
        current_spacing,
        new_spacing,
        ResamplingContract(
            function_name="resample_data_or_seg_to_shape",
            order=1,
            order_z=0,
            force_separate_z=None,
        ),
    )
    return ContractParityResult(
        arrays=(
            array_metrics(
                expected,
                actual,
                name=f"resampling_{'anisotropic' if anisotropic else 'isotropic'}",
                rtol=1e-6,
                atol=1e-6,
            ),
        )
    )


def run_segmentation_resampling_parity(
    *, anisotropic: bool, order_z: int = 0
) -> ExactParityResult:
    """Compare HCC-TACE label resampling exactly with nnU-Net v2.8.1.

    In-plane interpolation is order 1. ``order_z`` selects out-of-plane interpolation
    when the anisotropic fixture uses a separate axis.
    """

    from nnunet_mlx.data.resampling import resample_segmentation
    from nnunet_mlx.plans import ResamplingContract

    source = import_module("nnunetv2.preprocessing.resampling.default_resampling")
    _, liver = liver_crop((4, 5, 6))
    tumor = tumor_crop((4, 5, 6))
    labels = np.where(tumor > 0, 2, liver).astype(np.int16)
    segmentation = labels[None]
    current_spacing = (5.0, 1.0, 1.0) if anisotropic else (1.2, 1.0, 0.8)
    new_spacing = (2.5, 0.8, 0.8) if anisotropic else (0.9, 1.1, 1.0)
    new_shape = (7, 6, 8)
    expected = np.asarray(
        source.resample_data_or_seg_to_shape(
            segmentation,
            new_shape,
            current_spacing,
            new_spacing,
            is_seg=True,
            order=1,
            order_z=order_z,
            force_separate_z=None,
        )
    )
    actual = resample_segmentation(
        segmentation,
        new_shape,
        current_spacing,
        new_spacing,
        ResamplingContract(
            function_name="resample_data_or_seg_to_shape",
            order=1,
            order_z=order_z,
            force_separate_z=None,
        ),
    )
    return exact_array_parity(
        name=(
            f"segmentation_resampling_{'anisotropic' if anisotropic else 'isotropic'}"
            f"_order_z_{order_z}"
        ),
        expected=expected,
        actual=actual,
    )
