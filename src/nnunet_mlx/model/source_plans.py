"""Source-bound plan metadata for released inference profiles."""

from __future__ import annotations

from dataclasses import replace

from nnunet_mlx.contracts import parse_integer, parse_mapping, parse_sequence, parse_string
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.source import (
    TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
    resolve_inference_source_profile,
    validate_inference_model_identity,
)


def normalize_inference_plans(
    plans: PlansContract,
    *,
    source_profile: str,
    trainer_name: str | None = None,
    arithmetic: str | None = None,
) -> PlansContract:
    """Rebuild the ``architecture`` block that TotalSegmentator v2's legacy plans lack.

    Released TotalSegmentator v2 plans predate nnU-Net's ``architecture`` field and
    store ``UNet_*``, ``conv_kernel_sizes`` and ``pool_op_kernel_sizes`` instead.
    For that profile only, and only after its model identity validates, the legacy
    fields are replaced in memory by the equivalent ``PlainConvUNet`` architecture
    (3-D); the plans file on disk is not changed. Any other profile, or plans that
    already have ``architecture``, are returned unchanged.

    Raises:
        ContractError: If the identity does not match the profile or the legacy plans
            are not a 3-D ``PlainConvUNet``.
    """
    if (
        source_profile != TOTALSEGMENTATOR_CT_SOURCE_PROFILE
        or "architecture" in plans.configuration.raw
    ):
        return plans
    validate_inference_model_identity(
        resolve_inference_source_profile(source_profile),
        plans_json=plans.raw,
        trainer_name=parse_string(trainer_name, name="trainer name"),
        configuration=plans.configuration.name,
        arithmetic=arithmetic,
    )
    payload = plans.topology_payload()
    configurations = parse_mapping(payload["configurations"], name="configurations")
    configuration = parse_mapping(configurations[plans.configuration.name], name="configuration")
    if configuration.get("UNet_class_name") != "PlainConvUNet" or len(
        plans.configuration.patch_size
    ) != 3:
        raise ContractError("TotalSegmentator legacy plans require a 3-D PlainConvUNet")
    stages = len(parse_sequence(
        configuration.get("n_conv_per_stage_encoder"), name="encoder stages"
    ))
    base = parse_integer(configuration.get("UNet_base_num_features"), name="base features")
    maximum = parse_integer(configuration.get("unet_max_num_features"), name="maximum features")
    # Pinned nnU-Net 2.8.1 ConfigurationManager reconstructs precisely these
    # defaults for the released pre-architecture plans; source files stay intact.
    architecture = {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": stages,
            "features_per_stage": [min(base * 2**index, maximum) for index in range(stages)],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": configuration["conv_kernel_sizes"],
            "strides": configuration["pool_op_kernel_sizes"],
            "n_conv_per_stage": configuration["n_conv_per_stage_encoder"],
            "n_conv_per_stage_decoder": configuration["n_conv_per_stage_decoder"],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    for field in (
        "UNet_class_name", "UNet_base_num_features", "n_conv_per_stage_encoder",
        "n_conv_per_stage_decoder", "num_pool_per_axis", "pool_op_kernel_sizes",
        "conv_kernel_sizes", "unet_max_num_features",
    ):
        del configuration[field]
    configuration["architecture"] = architecture
    configurations[plans.configuration.name] = configuration
    payload["configurations"] = configurations
    return replace(
        plans, raw=payload, configuration=replace(plans.configuration, raw=configuration),
    )
