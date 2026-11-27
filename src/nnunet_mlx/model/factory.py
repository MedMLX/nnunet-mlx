"""Plan-bound construction of MLX nnU-Net networks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from nnunet_mlx.contracts import parse_mapping as _mapping
from nnunet_mlx.contracts import parse_string as _string
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec

if TYPE_CHECKING:
    from nnunet_mlx.model.network import Network


def build_model_from_plans(
    plans: Mapping[str, object],
    *,
    configuration: str,
    input_channels: int,
    output_channels: int,
) -> Network:
    """Build the MLX network named by a plans configuration, without loading weights.

    The network's layers have their normal MLX initial parameters until a
    converted checkpoint is loaded.

    Args:
        plans: Plans payload whose configuration is already inheritance-resolved
            (see ``PlansContract.topology_payload``).
        configuration: Configuration name inside ``plans["configurations"]``.
        input_channels: Network input channels (image modalities, plus one-hot
            previous-stage channels for cascades).
        output_channels: Number of segmentation output channels (logits).

    Raises:
        ContractError: If the architecture class is neither ``PlainConvUNet`` nor
            ``ResidualEncoderUNet``, or its options are unsupported.
    """
    spec = network_spec_from_plans(
        plans,
        configuration=configuration,
        input_channels=input_channels,
        output_channels=output_channels,
    )
    from nnunet_mlx.model.network import PlainConvUNet, ResidualEncoderUNet

    if isinstance(spec, PlainConvUNetSpec):
        return PlainConvUNet(spec)
    return ResidualEncoderUNet(spec)


def network_spec_from_plans(
    plans: Mapping[str, object],
    *,
    configuration: str,
    input_channels: int,
    output_channels: int,
) -> PlainConvUNetSpec | ResidualEncoderUNetSpec:
    """Resolve the network topology without allocating model weights."""
    configurations = _mapping(plans.get("configurations"), name="configurations")
    config = _mapping(configurations.get(configuration), name=configuration)
    architecture = _mapping(config.get("architecture"), name="architecture")
    class_name = _string(architecture.get("network_class_name"), name="network_class_name")
    if class_name.endswith("PlainConvUNet"):
        return PlainConvUNetSpec.from_plans(
            plans,
            configuration=configuration,
            input_channels=input_channels,
            num_classes=output_channels,
        )
    if class_name.endswith("ResidualEncoderUNet"):
        return ResidualEncoderUNetSpec.from_plans(
            plans,
            configuration=configuration,
            input_channels=input_channels,
            num_classes=output_channels,
        )
    raise ContractError(f"unsupported nnU-Net architecture {class_name!r}")
