"""Source-bound plan metadata for released inference profiles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from nnunet_mlx.contracts import parse_integer, parse_mapping, parse_sequence, parse_string
from nnunet_mlx.errors import ContractError
from nnunet_mlx.plans import PlansContract, resolve_configuration
from nnunet_mlx.prediction.source import (
    TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
    resolve_inference_source_profile,
    validate_inference_model_identity,
)

# Configuration fields that pre-2.4 nnU-Net plans use instead of ``architecture``.
LEGACY_ARCHITECTURE_FIELDS = (
    "UNet_class_name", "UNet_base_num_features", "n_conv_per_stage_encoder",
    "n_conv_per_stage_decoder", "num_pool_per_axis", "pool_op_kernel_sizes",
    "conv_kernel_sizes", "unet_max_num_features",
)


def legacy_architecture(configuration: Mapping[str, object]) -> dict[str, object]:
    """Build the ``architecture`` block for a pre-2.4 plans configuration.

    ``configuration`` must be inheritance-resolved. This reproduces nnU-Net
    2.8.1's ``ConfigurationManager`` backwards-compatibility branch exactly: the
    class follows ``UNet_class_name``, features double per stage up to
    ``unet_max_num_features``, and the remaining options take the defaults those
    plans were trained with (biased convolutions, affine instance norm with eps
    1e-5, no dropout, LeakyReLU).

    Raises:
        ContractError: If ``UNet_class_name`` is neither ``PlainConvUNet`` nor
            ``ResidualEncoderUNet`` or a legacy field is malformed.
    """
    class_name = configuration.get("UNet_class_name")
    if class_name == "PlainConvUNet":
        network_class = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        blocks_key = "n_conv_per_stage"
    elif class_name == "ResidualEncoderUNet":
        network_class = (
            "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"
        )
        blocks_key = "n_blocks_per_stage"
    else:
        raise ContractError(f"unsupported legacy nnU-Net architecture {class_name!r}")
    stages = len(parse_sequence(
        configuration.get("n_conv_per_stage_encoder"), name="encoder stages"
    ))
    dims = len(parse_sequence(configuration.get("patch_size"), name="patch_size"))
    if dims not in (2, 3):
        raise ContractError("legacy nnU-Net plans must be 2-D or 3-D")
    base = parse_integer(configuration.get("UNet_base_num_features"), name="base features")
    maximum = parse_integer(configuration.get("unet_max_num_features"), name="maximum features")
    return {
        "network_class_name": network_class,
        "arch_kwargs": {
            "n_stages": stages,
            "features_per_stage": [min(base * 2**index, maximum) for index in range(stages)],
            "conv_op": f"torch.nn.modules.conv.Conv{dims}d",
            "kernel_sizes": configuration["conv_kernel_sizes"],
            "strides": configuration["pool_op_kernel_sizes"],
            blocks_key: configuration["n_conv_per_stage_encoder"],
            "n_conv_per_stage_decoder": configuration["n_conv_per_stage_decoder"],
            "conv_bias": True,
            "norm_op": f"torch.nn.modules.instancenorm.InstanceNorm{dims}d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }


def upgrade_legacy_plans(plans_json: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of ``plans_json`` with every configuration in the current format.

    Each configuration whose resolved form lacks ``architecture`` gets the block
    ``legacy_architecture`` builds from it, and the legacy fields are removed, as
    nnU-Net 2.8.1 does when it loads such plans. Current-format plans come back
    unchanged.
    """
    upgraded = deepcopy(dict(plans_json))
    configurations = parse_mapping(upgraded.get("configurations"), name="configurations")
    resolved = {name: resolve_configuration(upgraded, name) for name in configurations}
    for name, configuration in resolved.items():
        if "architecture" in configuration:
            continue
        raw = parse_mapping(configurations[name], name=f"configuration {name}")
        raw["architecture"] = legacy_architecture(configuration)
        configurations[name] = raw
    for name, raw_value in configurations.items():
        raw = parse_mapping(raw_value, name=f"configuration {name}")
        for field in LEGACY_ARCHITECTURE_FIELDS:
            raw.pop(field, None)
        configurations[name] = raw
    upgraded["configurations"] = configurations
    return upgraded


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
    configuration["architecture"] = legacy_architecture(configuration)
    for field in LEGACY_ARCHITECTURE_FIELDS:
        del configuration[field]
    configurations[plans.configuration.name] = configuration
    payload["configurations"] = configurations
    return replace(
        plans, raw=payload, configuration=replace(plans.configuration, raw=configuration),
    )
