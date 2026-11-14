"""Topology specs for the supported nnU-Net v2 networks, parsed from plans.

Only ``PlainConvUNet`` and ``ResidualEncoderUNet`` (2-D or 3-D, InstanceNorm and
LeakyReLU, no dropout, no batch-norm running statistics) are accepted; anything
else raises ``ContractError`` rather than being approximated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nnunet_mlx.contracts import (
    parse_boolean as _boolean,
)
from nnunet_mlx.contracts import (
    parse_mapping as _mapping,
)
from nnunet_mlx.contracts import (
    parse_number as _number,
)
from nnunet_mlx.contracts import (
    parse_sequence as _sequence,
)
from nnunet_mlx.errors import ContractError

SpatialShape = tuple[int, ...]


def _positive_int(value: object, *, name: str) -> int:
    """Require a positive integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _positive_ints(value: object, *, name: str) -> tuple[int, ...]:
    """Require a nonempty sequence of positive integers."""
    result = tuple(_positive_int(item, name=name) for item in _sequence(value, name=name))
    if not result:
        raise ContractError(f"{name} must not be empty")
    return result


def _stage_positive_ints(
    value: object, *, stages: int, name: str
) -> tuple[int, ...]:
    """Expand one stage count or validate one positive count per stage."""
    if isinstance(value, int) and not isinstance(value, bool):
        return (_positive_int(value, name=name),) * stages
    result = _positive_ints(value, name=name)
    if len(result) != stages:
        raise ContractError(f"{name} must have one entry per encoder stage")
    return result


def _spatial_rows(value: object, *, name: str) -> tuple[SpatialShape, ...]:
    """Parse consistently two- or three-dimensional positive spatial rows."""
    rows: list[SpatialShape] = []
    for raw_row in _sequence(value, name=name):
        row = _positive_ints(raw_row, name=name)
        if len(row) not in (2, 3):
            raise ContractError(f"{name} rows must contain two or three integers")
        rows.append(row)
    dimensions = {len(row) for row in rows}
    if len(dimensions) != 1:
        raise ContractError(f"{name} rows must use one spatial dimensionality")
    return tuple(rows)


def _validate_architecture_ops(
    kwargs: Mapping[str, object], *, spatial_dims: int
) -> None:
    """Require matching convolution, instance normalization and LeakyReLU operations."""
    conv_op = kwargs.get("conv_op")
    norm_op = kwargs.get("norm_op")
    nonlin = kwargs.get("nonlin")
    expected_conv = f".Conv{spatial_dims}d"
    expected_norm = f".InstanceNorm{spatial_dims}d"
    if not isinstance(conv_op, str) or not conv_op.endswith(expected_conv):
        raise ContractError(
            f"{spatial_dims}-D nnU-Net requires Conv{spatial_dims}d, got {conv_op!r}"
        )
    if not isinstance(norm_op, str) or not norm_op.endswith(expected_norm):
        raise ContractError(
            f"{spatial_dims}-D nnU-Net requires InstanceNorm{spatial_dims}d, "
            f"got {norm_op!r}"
        )
    if not isinstance(nonlin, str) or not nonlin.endswith(".LeakyReLU"):
        raise ContractError(
            f"{spatial_dims}-D nnU-Net requires LeakyReLU, got {nonlin!r}"
        )


def _validate_architecture_semantics(kwargs: Mapping[str, object]) -> None:
    """Reject unsupported normalization, activation and dropout settings."""
    if _boolean(kwargs.get("nonlin_first", False), name="nonlin_first"):
        raise ContractError("unsupported nonlin_first activation ordering")
    stages = len(_sequence(kwargs.get("features_per_stage"), name="features_per_stage"))
    if "n_stages" in kwargs and _positive_int(kwargs["n_stages"], name="n_stages") != stages:
        raise ContractError("n_stages does not match features_per_stage")
    dropout = kwargs.get("dropout_op")
    options = kwargs.get("dropout_op_kwargs")
    if dropout is not None:
        if not isinstance(dropout, str) or dropout not in {
            "torch.nn.Dropout", "torch.nn.Dropout2d", "torch.nn.Dropout3d",
            "torch.nn.modules.dropout.Dropout", "torch.nn.modules.dropout.Dropout2d",
            "torch.nn.modules.dropout.Dropout3d",
        }:
            raise ContractError("unsupported dropout_op")
        dropout_kwargs = _mapping(options, name="dropout_op_kwargs")
        if _number(dropout_kwargs.get("p", 0.5), name="dropout p") != 0:
            raise ContractError("unsupported nonzero dropout")
        if set(dropout_kwargs) - {"p", "inplace"}:
            raise ContractError("unsupported dropout options")
        _boolean(dropout_kwargs.get("inplace", False), name="dropout inplace")
    norm = _mapping(kwargs.get("norm_op_kwargs", {}), name="norm_op_kwargs")
    if set(norm) - {"eps", "affine", "momentum", "track_running_stats"}:
        raise ContractError("unsupported normalization options")
    if _boolean(norm.get("track_running_stats", False), name="track_running_stats"):
        raise ContractError("unsupported normalization running statistics")
    if "momentum" in norm and norm["momentum"] is not None:
        momentum = _number(norm["momentum"], name="normalization momentum")
        if not 0 <= momentum <= 1:
            raise ContractError("normalization momentum must be in [0, 1]")
    nonlin = _mapping(kwargs.get("nonlin_kwargs", {}), name="nonlin_kwargs")
    if set(nonlin) - {"negative_slope", "inplace"}:
        raise ContractError("unsupported activation options")
    _boolean(nonlin.get("inplace", False), name="activation inplace")


@dataclass(frozen=True, slots=True)
class PlainConvUNetSpec:
    """Topology of upstream's ``PlainConvUNet`` (2-D or 3-D), read from nnU-Net plans.

    Per-stage fields are ordered from the highest-resolution encoder stage down to
    the bottleneck. ``kernel_sizes`` and ``strides`` hold one spatial row per
    encoder stage, in the plans' axis order. ``decoder_convs`` has one entry per
    skip connection (``len(features) - 1``).
    """

    input_channels: int
    num_classes: int
    features: tuple[int, ...]
    kernel_sizes: tuple[SpatialShape, ...]
    strides: tuple[SpatialShape, ...]
    encoder_convs: tuple[int, ...]
    decoder_convs: tuple[int, ...]
    conv_bias: bool = True
    norm_eps: float = 1e-5
    # True matches nnU-Net v2's default_experiment_planner, which always writes
    # norm_op_kwargs={"eps": 1e-5, "affine": True} explicitly for every admitted norm
    # op (InstanceNorm); this is the value a hand-built spec should use. It is not the
    # same thing as torch.nn.InstanceNorm{2,3}d's own class default (affine=False),
    # which ``from_plans`` falls back to only for the edge case of plans that omit the
    # key outright.
    norm_affine: bool = True
    leaky_slope: float = 0.01

    def __post_init__(self) -> None:
        """Validate stage lengths, spatial dimensions and normalization parameters."""
        stages = len(self.features)
        if self.input_channels < 1 or self.num_classes < 1:
            raise ContractError("PlainConvUNet needs input channels and at least one output")
        if stages < 2:
            raise ContractError("PlainConvUNet needs at least two stages")
        lengths = (len(self.kernel_sizes), len(self.strides), len(self.encoder_convs))
        if any(length != stages for length in lengths):
            raise ContractError("encoder topology lengths do not match")
        if len(self.decoder_convs) != stages - 1:
            raise ContractError("decoder topology must have one entry per skip connection")
        dimensions = {len(row) for row in (*self.kernel_sizes, *self.strides)}
        if len(dimensions) != 1 or next(iter(dimensions)) not in (2, 3):
            raise ContractError("PlainConvUNet requires consistent 2-D or 3-D spatial rows")
        _number(self.norm_eps, name="norm eps")
        _number(self.leaky_slope, name="leaky slope")
        if any(value < 1 or value % 2 == 0 for row in self.kernel_sizes for value in row):
            raise ContractError("only positive odd convolution kernels are supported")
        if self.norm_eps <= 0:
            raise ContractError("normalization epsilon must be positive")
        # The fused maximum expression equals piecewise LeakyReLU only in this range.
        if not 0 <= self.leaky_slope <= 1:
            raise ContractError("LeakyReLU negative_slope must be in [0, 1]")

    @property
    def spatial_dims(self) -> int:
        """Return the number of spatial dimensions in the topology."""
        return len(self.kernel_sizes[0])

    @classmethod
    def from_plans(
        cls,
        plans: Mapping[str, object],
        *,
        configuration: str,
        input_channels: int,
        num_classes: int,
    ) -> PlainConvUNetSpec:
        """Build the spec from ``plans["configurations"][configuration]["architecture"]``.

        ``input_channels`` and ``num_classes`` come from the dataset, not the plans.
        The configuration is looked up directly: ``inherits_from`` is not followed,
        so pass plans whose configuration is already resolved
        (``PlansContract.topology_payload``).

        Raises:
            ContractError: If the architecture class, ops or options are unsupported.
        """
        configurations = _mapping(plans.get("configurations"), name="configurations")
        raw = configurations.get(configuration)
        if raw is None:
            raise ContractError(f"nnU-Net plans lack configuration {configuration!r}")
        config = _mapping(raw, name=configuration)
        architecture = _mapping(config.get("architecture"), name="architecture")
        class_name = architecture.get("network_class_name")
        if not isinstance(class_name, str) or not class_name.endswith(".PlainConvUNet"):
            raise ContractError(f"unsupported nnU-Net architecture {class_name!r}")
        kwargs = _mapping(architecture.get("arch_kwargs"), name="arch_kwargs")
        _validate_architecture_semantics(kwargs)
        kernel_sizes = _spatial_rows(kwargs.get("kernel_sizes"), name="kernel_sizes")
        _validate_architecture_ops(kwargs, spatial_dims=len(kernel_sizes[0]))
        norm = _mapping(kwargs.get("norm_op_kwargs", {}), name="norm_op_kwargs")
        nonlin = _mapping(kwargs.get("nonlin_kwargs", {}), name="nonlin_kwargs")
        return cls(
            input_channels=input_channels,
            num_classes=num_classes,
            features=_positive_ints(kwargs.get("features_per_stage"), name="features"),
            kernel_sizes=kernel_sizes,
            strides=_spatial_rows(kwargs.get("strides"), name="strides"),
            encoder_convs=_positive_ints(kwargs.get("n_conv_per_stage"), name="encoder"),
            decoder_convs=_positive_ints(
                kwargs.get("n_conv_per_stage_decoder"), name="decoder"
            ),
            conv_bias=_boolean(kwargs.get("conv_bias", False), name="conv_bias"),
            norm_eps=_number(norm.get("eps", 1e-5), name="norm eps"),
            norm_affine=_boolean(norm.get("affine", False), name="norm affine"),
            leaky_slope=_number(nonlin.get("negative_slope", 0.01), name="leaky slope"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the validated plain-convolution topology."""
        return {
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "features": self.features,
            "kernel_sizes": self.kernel_sizes,
            "strides": self.strides,
            "encoder_convs": self.encoder_convs,
            "decoder_convs": self.decoder_convs,
            "conv_bias": self.conv_bias,
            "norm_eps": self.norm_eps,
            "norm_affine": self.norm_affine,
            "leaky_slope": self.leaky_slope,
        }


@dataclass(frozen=True, slots=True)
class ResidualEncoderUNetSpec:
    """Topology of upstream's ``ResidualEncoderUNet`` (2-D or 3-D), read from nnU-Net plans.

    The encoder uses ``BasicBlockD`` blocks, or ``BottleneckD`` when
    ``bottleneck_channels`` is set (one entry per encoder stage). ``encoder_blocks``
    is the number of residual blocks per stage; all other per-stage fields follow
    the ordering described for ``PlainConvUNetSpec``. ``stem_channels`` of ``None``
    means the stem uses ``features[0]`` channels.
    """

    input_channels: int
    num_classes: int
    features: tuple[int, ...]
    kernel_sizes: tuple[SpatialShape, ...]
    strides: tuple[SpatialShape, ...]
    encoder_blocks: tuple[int, ...]
    decoder_convs: tuple[int, ...]
    conv_bias: bool = True
    norm_eps: float = 1e-5
    # True matches nnU-Net v2's default_experiment_planner, which always writes
    # norm_op_kwargs={"eps": 1e-5, "affine": True} explicitly for every admitted norm
    # op (InstanceNorm); this is the value a hand-built spec should use. It is not the
    # same thing as torch.nn.InstanceNorm{2,3}d's own class default (affine=False),
    # which ``from_plans`` falls back to only for the edge case of plans that omit the
    # key outright.
    norm_affine: bool = True
    leaky_slope: float = 0.01
    stem_channels: int | None = None
    bottleneck_channels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate residual stage lengths, spatial dimensions and parameters."""
        stages = len(self.features)
        if self.input_channels < 1 or self.num_classes < 1:
            raise ContractError(
                "ResidualEncoderUNet needs input channels and at least one output"
            )
        if stages < 2:
            raise ContractError("ResidualEncoderUNet needs at least two stages")
        lengths = (len(self.kernel_sizes), len(self.strides), len(self.encoder_blocks))
        if any(length != stages for length in lengths):
            raise ContractError("residual encoder topology lengths do not match")
        if len(self.decoder_convs) != stages - 1:
            raise ContractError("decoder topology must have one entry per skip connection")
        dimensions = {len(row) for row in (*self.kernel_sizes, *self.strides)}
        if len(dimensions) != 1 or next(iter(dimensions)) not in (2, 3):
            raise ContractError(
                "ResidualEncoderUNet requires consistent 2-D or 3-D spatial rows"
            )
        _number(self.norm_eps, name="norm eps")
        _number(self.leaky_slope, name="leaky slope")
        if any(value < 1 or value % 2 == 0 for row in self.kernel_sizes for value in row):
            raise ContractError("only positive odd convolution kernels are supported")
        if self.norm_eps <= 0:
            raise ContractError("normalization epsilon must be positive")
        if not 0 <= self.leaky_slope <= 1:
            raise ContractError("LeakyReLU negative_slope must be in [0, 1]")
        if self.stem_channels is not None and self.stem_channels < 1:
            raise ContractError("stem channels must be positive")
        if self.bottleneck_channels is not None:
            if len(self.bottleneck_channels) != stages:
                raise ContractError(
                    "bottleneck channels must have one entry per encoder stage"
                )
            if any(channels < 1 for channels in self.bottleneck_channels):
                raise ContractError("bottleneck channels must be positive")

    @property
    def spatial_dims(self) -> int:
        """Return the number of spatial dimensions in the topology."""
        return len(self.kernel_sizes[0])

    @property
    def effective_stem_channels(self) -> int:
        """Return the configured stem width or the first stage width."""
        return self.stem_channels or self.features[0]

    @classmethod
    def from_plans(
        cls,
        plans: Mapping[str, object],
        *,
        configuration: str,
        input_channels: int,
        num_classes: int,
    ) -> ResidualEncoderUNetSpec:
        """Build the spec from ``plans["configurations"][configuration]["architecture"]``.

        See ``PlainConvUNetSpec.from_plans`` for the arguments and lookup rules.

        Raises:
            ContractError: If the architecture class, block type, ops or options are
                unsupported.
        """
        configurations = _mapping(plans.get("configurations"), name="configurations")
        raw = configurations.get(configuration)
        if raw is None:
            raise ContractError(f"nnU-Net plans lack configuration {configuration!r}")
        config = _mapping(raw, name=configuration)
        architecture = _mapping(config.get("architecture"), name="architecture")
        class_name = architecture.get("network_class_name")
        if not isinstance(class_name, str) or not class_name.endswith(
            ".ResidualEncoderUNet"
        ):
            raise ContractError(f"unsupported nnU-Net architecture {class_name!r}")
        kwargs = _mapping(architecture.get("arch_kwargs"), name="arch_kwargs")
        _validate_architecture_semantics(kwargs)
        kernel_sizes = _spatial_rows(kwargs.get("kernel_sizes"), name="kernel_sizes")
        _validate_architecture_ops(kwargs, spatial_dims=len(kernel_sizes[0]))
        features = _positive_ints(kwargs.get("features_per_stage"), name="features")
        block = kwargs.get("block")
        raw_bottlenecks = kwargs.get("bottleneck_channels")
        if block is None or (
            isinstance(block, str) and block.endswith(".BasicBlockD")
        ):
            if raw_bottlenecks is not None:
                raise ContractError(
                    "BasicBlockD plans must not define bottleneck channels"
                )
            bottleneck_channels = None
        elif isinstance(block, str) and block.endswith(".BottleneckD"):
            if raw_bottlenecks is None:
                raise ContractError("BottleneckD requires bottleneck channels")
            bottleneck_channels = _stage_positive_ints(
                raw_bottlenecks,
                stages=len(features),
                name="bottleneck channels",
            )
        else:
            raise ContractError(f"unsupported residual block {block!r}")
        norm = _mapping(kwargs.get("norm_op_kwargs", {}), name="norm_op_kwargs")
        nonlin = _mapping(kwargs.get("nonlin_kwargs", {}), name="nonlin_kwargs")
        raw_stem = kwargs.get("stem_channels")
        stem_channels = (
            None if raw_stem is None else _positive_int(raw_stem, name="stem channels")
        )
        return cls(
            input_channels=input_channels,
            num_classes=num_classes,
            features=features,
            kernel_sizes=kernel_sizes,
            strides=_spatial_rows(kwargs.get("strides"), name="strides"),
            encoder_blocks=_positive_ints(
                kwargs.get("n_blocks_per_stage"), name="encoder blocks"
            ),
            decoder_convs=_positive_ints(
                kwargs.get("n_conv_per_stage_decoder"), name="decoder"
            ),
            conv_bias=_boolean(kwargs.get("conv_bias", False), name="conv_bias"),
            norm_eps=_number(norm.get("eps", 1e-5), name="norm eps"),
            norm_affine=_boolean(norm.get("affine", False), name="norm affine"),
            leaky_slope=_number(nonlin.get("negative_slope", 0.01), name="leaky slope"),
            stem_channels=stem_channels,
            bottleneck_channels=bottleneck_channels,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the validated residual-encoder topology."""
        return {
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            "features": self.features,
            "kernel_sizes": self.kernel_sizes,
            "strides": self.strides,
            "encoder_blocks": self.encoder_blocks,
            "decoder_convs": self.decoder_convs,
            "conv_bias": self.conv_bias,
            "norm_eps": self.norm_eps,
            "norm_affine": self.norm_affine,
            "leaky_slope": self.leaky_slope,
            "stem_channels": self.stem_channels,
            "bottleneck_channels": self.bottleneck_channels,
        }
