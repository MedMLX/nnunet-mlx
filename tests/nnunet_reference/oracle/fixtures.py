"""Pinned nnU-Net oracle: fixtures."""

from __future__ import annotations

from typing import cast


def raw_inference_fixture() -> tuple[dict[str, object], dict[str, object]]:
    """Return the shared modern/inherited raw-inference oracle fixture."""

    data_resampling = {
        "is_seg": False,
        "order": 3,
        "order_z": 0,
        "force_separate_z": None,
    }
    plans: dict[str, object] = {
        "dataset_name": "Dataset999_Proof",
        "plans_name": "nnUNetPlans",
        "image_reader_writer": "SimpleITKIO",
        "label_manager": "LabelManager",
        "transpose_forward": [2, 0, 1],
        "transpose_backward": [1, 2, 0],
        "foreground_intensity_properties_per_channel": {"0": {}},
        "configurations": {
            "base": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "batch_size": 2,
                "batch_dice": False,
                "patch_size": [8, 8, 8],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": ["NoNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": data_resampling,
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_seg_kwargs": {
                    **data_resampling,
                    "is_seg": True,
                    "order": 1,
                },
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {
                    **data_resampling,
                    "order": 1,
                },
                "architecture": {
                    "network_class_name": (
                        "dynamic_network_architectures.architectures.unet.PlainConvUNet"
                    ),
                    "arch_kwargs": {
                        "features_per_stage": [4, 8],
                        "kernel_sizes": [[3, 3, 3], [3, 3, 3]],
                        "strides": [[1, 1, 1], [2, 2, 2]],
                        "n_conv_per_stage": [2, 2],
                        "n_conv_per_stage_decoder": [2],
                        "conv_bias": True,
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                        "nonlin": "torch.nn.modules.activation.LeakyReLU",
                        "nonlin_kwargs": {"negative_slope": 0.01},
                    },
                },
            },
            "3d_fullres": {"inherits_from": "base"},
        },
    }
    dataset_json: dict[str, object] = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "file_ending": ".nii.gz",
    }
    return plans, dataset_json


def plan_configuration_fixture() -> tuple[dict[str, object], tuple[str, ...]]:
    """Return direct, inherited, and nested-cascade plan configurations."""

    plans, _ = raw_inference_fixture()
    configurations = cast(dict[str, object], plans["configurations"])
    configurations["3d_cascade_fullres"] = {
        "inherits_from": "3d_fullres",
        "data_identifier": "nnUNetPlans_3d_cascade_fullres",
        "batch_size": 3,
        "batch_dice": True,
        "previous_stage": "3d_lowres",
    }
    return plans, ("base", "3d_fullres", "3d_cascade_fullres")


def restoration_properties() -> dict[str, object]:
    """Provide synthetic crop and geometry metadata for restoration comparisons."""
    return {
        "sitk_stuff": {
            "spacing": (1.5, 0.8, 0.7),
            "origin": (2.0, 3.0, 4.0),
            "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        },
        "spacing": (1.5, 0.8, 0.7),
        "shape_before_cropping": (6, 7, 8),
        "bbox_used_for_cropping": [[1, 5], [1, 6], [1, 7]],
        "shape_after_cropping_and_before_resampling": (4, 5, 6),
    }
