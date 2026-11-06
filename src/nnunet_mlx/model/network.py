"""nnU-Net v2 2-D/3-D PlainConvUNet and ResidualEncoderUNet in MLX channels-last layout.

Activations are channels-last: (B, D, H, W, C) for 3-D and (B, H, W, C) for 2-D,
unlike upstream's channels-first (B, C, *spatial). Convolution weights use MLX's
(C_out, *kernel, C_in) layout; native safetensors checkpoints use the
parameter names defined here. Modules return logits for the
highest-resolution decoder head only (upstream's ``deep_supervision=False``).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING, cast

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_native_norm import instance_norm
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec, SpatialShape
from nnunet_mlx.runtime import import_mlx

_, _nn = import_mlx()

if TYPE_CHECKING:
    from mlx.core import array
    from mlx.nn import AvgPool2d, AvgPool3d, Conv3d, ModuleTree
    from mlx.nn import Module as MlxModule
    from mlx.utils import Tree
else:
    MlxModule = _nn.Module


type NetworkSpec = PlainConvUNetSpec | ResidualEncoderUNetSpec
type Network = PlainConvUNet | ResidualEncoderUNet
type NetworkOutput = array
type ArrayTransform = Callable[[array], array]


def _convolution(spatial_dims: int):
    """Select the native MLX convolution for two or three spatial dimensions."""
    return _nn.Conv2d if spatial_dims == 2 else _nn.Conv3d


def _transposed_convolution(spatial_dims: int):
    """Select the native MLX transposed convolution for the spatial dimensionality."""
    return _nn.ConvTranspose2d if spatial_dims == 2 else _nn.ConvTranspose3d


def _average_pool(spatial_dims: int):
    """Select the native MLX average pool for the spatial dimensionality."""
    return _nn.AvgPool2d if spatial_dims == 2 else _nn.AvgPool3d


class SpatialInstanceNorm(MlxModule):
    """Instance normalization over the spatial axes of a (B, *spatial, C) tensor.

    Statistics are per sample and per channel, with biased variance, as in
    ``torch.nn.InstanceNorm{2,3}d``; there are no running statistics. ``weight`` and
    ``bias`` exist only when ``affine`` is true. Float32 input is normalized by the
    Metal kernels in ``nnunet_mlx.kernels.metal_native_norm``.
    """

    def __init__(self, dims: int, *, eps: float, affine: bool) -> None:
        """Configure channel count, epsilon and optional affine parameters."""
        super().__init__()
        if eps <= 0:
            raise ValueError("instance-normalization epsilon must be positive")
        self.dims = dims
        self.eps = eps
        if affine:
            mx = _mx()
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, values: array) -> array:
        """Validate channel layout and normalize each sample over its spatial axes."""
        return self.normalize(values)

    def normalize(
        self,
        values: array,
        *,
        input_bias: array | None = None,
        negative_slope: float | None = None,
        residual: array | None = None,
    ) -> array:
        """Normalize ``values + input_bias``, add ``residual``, then apply LeakyReLU.

        ``residual`` is added after the affine output and before the activation;
        it must be float32 with exactly the shape of ``values``. All extras run
        inside the normalization kernels and give the same bits as separate
        operations in the same order.
        """
        if values.ndim < 3:
            raise ValueError("instance normalization requires a spatial dimension")
        if values.shape[-1] != self.dims:
            raise ValueError(
                f"instance normalization expected {self.dims} channels, got {values.shape[-1]}"
            )
        return instance_norm(
            values,
            getattr(self, "weight", None),
            getattr(self, "bias", None),
            self.eps,
            input_bias=input_bias,
            negative_slope=negative_slope,
            residual=residual,
        )


class NonOverlappingConvTranspose3d(MlxModule):
    """3-D transposed convolution whose stride equals its kernel, computed as one matmul.

    With stride == kernel the output windows do not overlap, so each input voxel
    expands independently to a ``kernel_size`` block. Input (B, D, H, W, C_in) gives
    output (B, D * kD, H * kH, W * kW, C_out); the weight has MLX's transposed-conv
    layout (C_out, kD, kH, kW, C_in). Used instead of ``mlx.nn.ConvTranspose3d``
    because the native Metal transposed convolution drifts from the reference at
    intermediate stages (see ``_build_decoder_upsample``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        bias: bool,
    ) -> None:
        """Initialize transposed-convolution weights for disjoint output blocks."""
        super().__init__()
        initialized = _nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
        )
        self.weight = initialized.weight
        if bias:
            self.bias = initialized.bias

    def __call__(self, values: array) -> array:
        """Expand input voxels by matmul and add optional channel bias."""
        output = self._matmul(values)
        bias = getattr(self, "bias", None)
        return output if bias is None else output + bias

    def _matmul(self, values: array) -> array:
        # With stride == kernel every input voxel owns a disjoint output block, so the
        # transpose convolution is one matmul against the flattened (C_out*K, C_in)
        # weight followed by interleaving the kernel offsets into each spatial axis.
        """Project each voxel and interleave its nonoverlapping spatial kernel block."""
        mx = _mx()
        batch, depth, height, width, channels = values.shape
        channels_out, kernel_depth, kernel_height, kernel_width, _ = self.weight.shape
        projected = mx.matmul(
            mx.reshape(values, (-1, channels)),
            mx.transpose(mx.reshape(self.weight, (-1, channels))),
        )
        projected = mx.reshape(
            projected,
            (
                batch,
                depth,
                height,
                width,
                channels_out,
                kernel_depth,
                kernel_height,
                kernel_width,
            ),
        )
        projected = mx.transpose(projected, (0, 1, 5, 2, 6, 3, 7, 4))
        return mx.reshape(
            projected,
            (
                batch,
                depth * kernel_depth,
                height * kernel_height,
                width * kernel_width,
                channels_out,
            ),
        )


class ConvNorm(MlxModule):
    """Convolution (padding ``kernel // 2`` per axis) followed by instance norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        stride: SpatialShape,
        spec: NetworkSpec,
        bias: bool | None = None,
    ) -> None:
        """Build a padded convolution followed by spatial instance normalization."""
        super().__init__()
        self.conv = _convolution(spec.spatial_dims)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=tuple(value // 2 for value in kernel_size),
            bias=spec.conv_bias if bias is None else bias,
        )
        self.norm = SpatialInstanceNorm(
            out_channels,
            eps=spec.norm_eps,
            affine=spec.norm_affine,
        )

    def __call__(self, values: array) -> array:
        """Apply convolution and instance normalization to channels-last features."""
        return self._conv_norm(values, negative_slope=None)

    def concatenated(self, parts: tuple[array, array]) -> array:
        """Apply ``self`` to the channel concatenation of ``parts``."""
        return self._conv_norm(parts, negative_slope=None)

    def residual_activation(
        self, values: array, residual: array, *, negative_slope: float
    ) -> array:
        """Apply conv-norm, add ``residual`` to its affine output, then LeakyReLU.

        Native 3-D layers add the residual and activation in the normalization
        output kernel. Layers whose convolution or normalization was replaced add
        the residual and activate eagerly in the same order.
        """
        return self._conv_norm(values, negative_slope=negative_slope, residual=residual)

    def _conv_norm(
        self,
        values: array | tuple[array, array],
        *,
        negative_slope: float | None,
        residual: array | None = None,
    ) -> array:
        """Convolve ``values`` (or a concatenated pair), normalize and optionally activate.

        Native 3-D layers pass the convolution bias, ``residual`` and LeakyReLU to
        the normalization kernels, which gives the same bits as separate operations
        in the same order. Layers whose convolution or normalization was replaced
        (ordered arithmetic, compiled normalization) run each operation separately.
        """
        conv, norm = self.conv, self.norm
        # Ordered arithmetic swaps in a SpatialInstanceNorm subclass, hence the exact type.
        if isinstance(conv, _nn.Conv3d) and type(norm) is SpatialInstanceNorm:
            return norm.normalize(
                _unbiased_convolution(conv, values),
                input_bias=conv.bias if "bias" in conv else None,
                negative_slope=negative_slope,
                residual=residual,
            )
        if isinstance(values, tuple):
            values = _mx().concatenate(values, axis=-1)
        normalized = cast(ArrayTransform, norm)(cast(ArrayTransform, conv)(values))
        if residual is not None:
            normalized = normalized + residual
        if negative_slope is None:
            return normalized
        return _nn.leaky_relu(normalized, negative_slope=negative_slope)


def _unbiased_convolution(conv: Conv3d, values: array | tuple[array, array]) -> array:
    """Apply ``conv`` without its bias to ``values`` or to a channel-concatenated pair.

    MLX 0.32.4's implicit-GEMM 3-D convolution uses 64-wide output tiles once
    the input has at least 64 channels and 8192 output voxels, so a decoder
    convolution from two 32-channel inputs to 32 outputs computes twice the
    needed products. For that shape (see ``_split_concatenated_input``) the
    convolution is the sum of one convolution per part over the matching
    input-channel slice of the weight, each on 32-wide tiles, and the
    concatenated tensor is never formed. The float32 result differs from the
    single convolution only by the rounding of the two partial sums.
    """
    mx = _mx()
    weight = conv.weight
    if not isinstance(values, tuple):
        return mx.conv3d(values, weight, conv.stride, conv.padding, conv.dilation)
    if not _split_concatenated_input(conv, values):
        concatenated = mx.concatenate(values, axis=-1)
        return mx.conv3d(concatenated, weight, conv.stride, conv.padding, conv.dilation)
    first, second = values
    split = first.shape[-1]
    return mx.conv3d(
        first, weight[..., :split], conv.stride, conv.padding, conv.dilation
    ) + mx.conv3d(second, weight[..., split:], conv.stride, conv.padding, conv.dilation)


def _split_concatenated_input(conv: Conv3d, parts: tuple[array, array]) -> bool:
    """Return whether per-part convolution avoids MLX's oversized 3-D output tiles.

    Mirrors the tile choice in MLX 0.32.4 ``implicit_gemm_conv_3D_gpu``: 64-row
    tiles (and therefore 64-column tiles) when the input has at least 64
    channels and the output at least 8192 voxels. With 17 to 32 output channels
    and parts of fewer than 64 channels (multiples of 16, so each part stays on
    the implicit-GEMM path), every part runs on 32-wide tiles instead.
    """
    channels = [int(part.shape[-1]) for part in parts]
    output_channels = int(conv.weight.shape[0])
    voxels = int(parts[0].size // (parts[0].shape[0] * channels[0]))
    return (
        sum(channels) >= 64
        and 16 < output_channels <= 32
        and all(value < 64 and value % 16 == 0 for value in channels)
        and voxels >= 8192
    )


class ConvNormAct(ConvNorm):
    """``ConvNorm`` followed by LeakyReLU with the spec's negative slope."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        stride: SpatialShape,
        spec: NetworkSpec,
        bias: bool | None = None,
    ) -> None:
        """Configure convolution, normalization and the plans negative slope."""
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            spec=spec,
            bias=bias,
        )
        self._leaky_slope = spec.leaky_slope

    def __call__(self, values: array) -> array:
        """Apply convolution and normalization followed by LeakyReLU."""
        return self._conv_norm(values, negative_slope=self._leaky_slope)

    def concatenated(self, parts: tuple[array, array]) -> array:
        """Apply ``self`` to the channel concatenation of ``parts``."""
        return self._conv_norm(parts, negative_slope=self._leaky_slope)


class StackedConvs(MlxModule):
    """``count`` ``ConvNormAct`` blocks; only the first applies ``initial_stride``."""

    def __init__(
        self,
        count: int,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        initial_stride: SpatialShape,
        spec: NetworkSpec,
    ) -> None:
        """Build the requested convolution blocks with stride on the first block."""
        super().__init__()
        self._count = count
        for index in range(count):
            setattr(
                self,
                f"block{index}",
                ConvNormAct(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=initial_stride if index == 0 else (1,) * spec.spatial_dims,
                    spec=spec,
                ),
            )

    def __call__(self, values: array) -> array:
        """Apply the configured convolution blocks in order."""
        for index in range(self._count):
            values = getattr(self, f"block{index}")(values)
        return values

    def concatenated(self, parts: tuple[array, array]) -> array:
        """Apply the blocks to the channel concatenation of ``parts``.

        A first block replaced by an inference-arithmetic subclass receives the
        concatenated tensor through its own ``__call__``.
        """
        first = self._block(0)
        if type(first) is ConvNormAct:
            values = first.concatenated(parts)
        else:
            values = first(_mx().concatenate(parts, axis=-1))
        for index in range(1, self._count):
            values = self._block(index)(values)
        return values

    def _block(self, index: int) -> ConvNormAct:
        """Return the convolution block at ``index``."""
        return getattr(self, f"block{index}")


class PlainConvUNet(MlxModule):
    """Plan-driven ``PlainConvUNet`` following dynamic-network-architectures v0.4.4.

    Maps channels-last ``(B, *spatial, input_channels)`` to channels-last logits
    with ``num_classes`` channels. The highest-resolution decoder follows the
    first encoder stage; its output need not match the input spatial size for
    unusual admitted specs, such as a first-stage stride greater than one.
    """

    def __init__(self, spec: PlainConvUNetSpec) -> None:
        """Build the plain encoder and decoder from a validated topology."""
        super().__init__()
        self.spec = spec
        self.inference_arithmetic = "native"
        self._build_encoder(spec)
        self._build_decoder(spec)

    def _build_encoder(self, spec: PlainConvUNetSpec) -> None:
        """Attach the plain encoder stages from the topology."""
        in_channels = spec.input_channels
        for index, out_channels in enumerate(spec.features):
            setattr(
                self,
                f"enc{index}",
                StackedConvs(
                    spec.encoder_convs[index],
                    in_channels,
                    out_channels,
                    kernel_size=spec.kernel_sizes[index],
                    initial_stride=spec.strides[index],
                    spec=spec,
                ),
            )
            in_channels = out_channels
        self._stages = len(spec.features)

    def _build_decoder(self, spec: PlainConvUNetSpec) -> None:
        """Attach decoder stages and record their count."""
        self._decoders = _build_decoder(self, spec)

    def __call__(self, values: array) -> NetworkOutput:
        """Encode channels-last inputs and decode the highest-resolution logits."""
        skips: list[array] = []
        hidden = values
        for index in range(self._stages):
            hidden = getattr(self, f"enc{index}")(hidden)
            skips.append(hidden)
        return _decode(
            self,
            skips,
            self._decoders,
        )


class ResidualBlock(MlxModule):
    """MLX ``BasicBlockD``: conv-norm-act, conv-norm, plus a skip path, then LeakyReLU.

    The skip path average-pools when ``stride`` is not 1 and applies a bias-free
    1x1 conv-norm projection when the channel count changes. For native 3-D
    convolutions the final normalization kernel adds the skip and applies the
    activation.
    The convolution and pooled skip must have compatible spatial shapes; this
    block does not crop or pad them before addition. For example, an odd spatial
    size with stride 2 and a 3-wide kernel produces incompatible shapes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        stride: SpatialShape,
        spec: ResidualEncoderUNetSpec,
    ) -> None:
        """Build the basic residual branch and optional downsample or projection."""
        super().__init__()
        unit_stride = (1,) * spec.spatial_dims
        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            spec=spec,
        )
        self.conv2 = ConvNorm(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=unit_stride,
            spec=spec,
        )
        self.pool, self.proj = _build_residual_skip(
            in_channels,
            out_channels,
            stride=stride,
            spec=spec,
        )
        self._leaky_slope = spec.leaky_slope

    def __call__(self, values: array) -> array:
        """Add the residual branch and shortcut, then apply LeakyReLU."""
        residual = self.pool(values) if self.pool is not None else values
        residual = self.proj(residual) if self.proj is not None else residual
        hidden = self.conv1(values)
        return self.conv2.residual_activation(hidden, residual, negative_slope=self._leaky_slope)


class BottleneckResidualBlock(MlxModule):
    """MLX ``BottleneckD``: 1x1, kxk (strided) and 1x1 convolutions plus the same skip path.

    As in ``ResidualBlock``, the final normalization adds the skip and applies the
    activation for native 3-D convolutions. The caller must supply spatial sizes
    compatible with both the strided convolution and pooled skip; no alignment is
    applied before addition.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        stride: SpatialShape,
        spec: ResidualEncoderUNetSpec,
    ) -> None:
        """Build bottleneck convolutions and the optional residual shortcut."""
        super().__init__()
        unit_stride = (1,) * spec.spatial_dims
        self.conv1 = ConvNormAct(
            in_channels,
            bottleneck_channels,
            kernel_size=unit_stride,
            stride=unit_stride,
            spec=spec,
        )
        self.conv2 = ConvNormAct(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=stride,
            spec=spec,
        )
        self.conv3 = ConvNorm(
            bottleneck_channels,
            out_channels,
            kernel_size=unit_stride,
            stride=unit_stride,
            spec=spec,
        )
        self.pool, self.proj = _build_residual_skip(
            in_channels,
            out_channels,
            stride=stride,
            spec=spec,
        )
        self._leaky_slope = spec.leaky_slope

    def __call__(self, values: array) -> array:
        """Apply bottleneck convolutions and activate their residual sum."""
        residual = self.pool(values) if self.pool is not None else values
        residual = self.proj(residual) if self.proj is not None else residual
        hidden = self.conv2(self.conv1(values))
        return self.conv3.residual_activation(hidden, residual, negative_slope=self._leaky_slope)


def _build_residual_skip(
    in_channels: int,
    out_channels: int,
    *,
    stride: SpatialShape,
    spec: ResidualEncoderUNetSpec,
) -> tuple[AvgPool2d | AvgPool3d | None, ConvNorm | None]:
    """Create the pooled or projected shortcut required by stride and channel changes."""
    unit_stride = (1,) * spec.spatial_dims
    pool = (
        _average_pool(spec.spatial_dims)(kernel_size=stride, stride=stride)
        if stride != unit_stride
        else None
    )
    projection = (
        ConvNorm(
            in_channels,
            out_channels,
            kernel_size=unit_stride,
            stride=unit_stride,
            spec=spec,
            bias=False,
        )
        if in_channels != out_channels
        else None
    )
    return pool, projection


class ResidualStack(MlxModule):
    """``count`` residual blocks; only the first applies ``initial_stride``."""

    def __init__(
        self,
        count: int,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: SpatialShape,
        initial_stride: SpatialShape,
        spec: ResidualEncoderUNetSpec,
        bottleneck_channels: int | None,
    ) -> None:
        """Build basic or bottleneck residual blocks with stride in the first block."""
        super().__init__()
        self._count = count
        for index in range(count):
            block_input_channels = in_channels if index == 0 else out_channels
            block_stride = initial_stride if index == 0 else (1,) * spec.spatial_dims
            if bottleneck_channels is None:
                block: MlxModule = ResidualBlock(
                    block_input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    spec=spec,
                )
            else:
                block = BottleneckResidualBlock(
                    block_input_channels,
                    bottleneck_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    spec=spec,
                )
            setattr(
                self,
                f"block{index}",
                block,
            )

    def __call__(self, values: array) -> array:
        """Apply the residual blocks in order."""
        for index in range(self._count):
            values = getattr(self, f"block{index}")(values)
        return values


class ResidualEncoderUNet(MlxModule):
    """nnU-Net ``ResidualEncoderUNet``: residual stem and stages, plain decoder.

    ``inference_arithmetic``, ``inference_input_shape`` and ``inference_source_profile``
    are set by ``nnunet_mlx.prediction.arithmetic`` when ordered ``torch-order-fp32``
    arithmetic is applied; when ``inference_input_shape`` is set, other input shapes
    are rejected.
    """

    def __init__(self, spec: ResidualEncoderUNetSpec) -> None:
        """Build the residual encoder, stem and plain decoder from the topology."""
        super().__init__()
        self.spec = spec
        self.inference_arithmetic = "native"
        self.inference_input_shape: tuple[int, ...] | None = None
        self.inference_source_profile: str | None = None
        self.stem = ConvNormAct(
            spec.input_channels,
            spec.effective_stem_channels,
            kernel_size=spec.kernel_sizes[0],
            stride=(1,) * spec.spatial_dims,
            spec=spec,
        )
        in_channels = spec.effective_stem_channels
        for index, out_channels in enumerate(spec.features):
            setattr(
                self,
                f"res{index}",
                ResidualStack(
                    spec.encoder_blocks[index],
                    in_channels,
                    out_channels,
                    kernel_size=spec.kernel_sizes[index],
                    initial_stride=spec.strides[index],
                    spec=spec,
                    bottleneck_channels=(
                        None
                        if spec.bottleneck_channels is None
                        else spec.bottleneck_channels[index]
                    ),
                ),
            )
            in_channels = out_channels
        self._stages = len(spec.features)
        self._decoders = _build_decoder(self, spec)

    def __call__(self, values: array) -> NetworkOutput:
        """Run the residual encoder and decoder with its normal final upsample."""
        return self.forward_with_final_upsample(values, final_upsample=None)

    def forward_with_final_upsample(
        self, values: array, *, final_upsample: ArrayTransform | None
    ) -> NetworkOutput:
        """Run residual inference with an optional replacement final upsample."""
        if (
            self.inference_input_shape is not None
            and tuple(values.shape) != self.inference_input_shape
        ):
            raise ContractError("ordered residual inference requires its qualified input shape")
        materialize = _metal_residual_inference_materialization(self)
        hidden = self.stem(values)
        _eval_stage(hidden, enabled=materialize)
        skips: list[array] = []
        for index in range(self._stages):
            hidden = getattr(self, f"res{index}")(hidden)
            _eval_stage(hidden, enabled=materialize)
            skips.append(hidden)
        return _decode(
            self,
            skips,
            self._decoders,
            final_upsample=final_upsample,
            materialize=materialize,
        )

    def final_upsample(self) -> MlxModule:
        """Return the final decoder upsample module."""
        return cast("MlxModule", getattr(self, f"up{self._decoders - 1}"))


class _CompiledInstanceNorm(MlxModule):
    """Compile one native normalization, preserving its corrected FP32 reductions."""

    def __init__(self, layer: SpatialInstanceNorm) -> None:
        """Compile one native instance-normalization layer with its state."""
        super().__init__()
        self.layer = layer
        self._call = _mx().compile(cast(ArrayTransform, layer), inputs=layer.state)

    def __call__(self, values: array) -> array:
        """Run the compiled instance-normalization layer."""
        return self._call(values)


class _CompiledResidualInference:
    """Callable wrapper that runs a residual model with compiled norms and final upsample."""

    compilation_scope = "normalization_and_final_upsample"

    def __init__(self, model: ResidualEncoderUNet) -> None:
        # Copy module ownership, sharing parameter arrays. This leaves
        # the caller's reference model untouched and avoids copying kernel callables.
        """Copy the model tree and compile normalization and the final upsample."""
        self.model = cast(ResidualEncoderUNet, _compile_normalization_tree(model))
        self.spec = model.spec
        final_upsample = self.model.final_upsample()
        self._final_upsample = _mx().compile(
            cast(ArrayTransform, final_upsample),
            inputs=final_upsample.state,
        )

    def __call__(self, values: array) -> NetworkOutput:
        """Run the copied model with the compiled final upsample."""
        return self.model.forward_with_final_upsample(
            values,
            final_upsample=self._final_upsample,
        )


def _compile_normalization_tree(layer: MlxModule) -> MlxModule:
    """Copy a module tree and replace instance norms with compiled wrappers."""
    from mlx.utils import tree_map

    copied = copy(layer)
    if isinstance(copied, SpatialInstanceNorm):
        return _CompiledInstanceNorm(copied)
    children = cast("Tree[MlxModule]", layer.children())
    compiled = tree_map(_compile_normalization_tree, children, is_leaf=MlxModule.is_module)
    copied.update_modules(cast("dict[str, ModuleTree]", compiled))
    return copied


def compile_residual_inference_model(model: ResidualEncoderUNet) -> _CompiledResidualInference:
    """Compile native normalization and the final upsample of a fixed-state residual model.

    Stage materialization and convolution dispatch remain unchanged. The caller's
    model stays unmodified. Whole-network compilation is unsupported; the README's
    supported configurations list the compilation and architecture limits.
    """

    return _CompiledResidualInference(model)


def _build_decoder(owner: MlxModule, spec: NetworkSpec) -> int:
    """Attach ``up``/``dec``/``seg`` modules per decoder stage to ``owner``; return the count."""
    # A segmentation head is created for every decoder stage so that upstream
    # checkpoints (which store all ``seg_layers``) load with ``strict=True``;
    # ``_decode`` applies only the last one.
    segmentation_conv = _convolution(spec.spatial_dims)
    for decoder_index in range(len(spec.features) - 1):
        source_index = len(spec.features) - 1 - decoder_index
        below = spec.features[source_index]
        skip = spec.features[source_index - 1]
        stride = spec.strides[source_index]
        setattr(
            owner,
            f"up{decoder_index}",
            _build_decoder_upsample(
                spec,
                in_channels=below,
                out_channels=skip,
                stride=stride,
            ),
        )
        setattr(
            owner,
            f"dec{decoder_index}",
            StackedConvs(
                spec.decoder_convs[decoder_index],
                2 * skip,
                skip,
                kernel_size=spec.kernel_sizes[source_index - 1],
                initial_stride=(1,) * spec.spatial_dims,
                spec=spec,
            ),
        )
        setattr(
            owner,
            f"seg{decoder_index}",
            segmentation_conv(skip, spec.num_classes, 1),
        )
    return len(spec.features) - 1


def _build_decoder_upsample(
    spec: NetworkSpec,
    *,
    in_channels: int,
    out_channels: int,
    stride: SpatialShape,
) -> MlxModule:
    """Select the decoder upsample implementation for the spatial dimensionality."""
    if spec.spatial_dims == 3:
        # Native Metal transpose convolution also drifts at intermediate CADS stages.
        return NonOverlappingConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            bias=spec.conv_bias,
        )
    return _transposed_convolution(spec.spatial_dims)(
        in_channels,
        out_channels,
        kernel_size=stride,
        stride=stride,
        bias=spec.conv_bias,
    )


def _decode(
    owner: MlxModule,
    skips: list[array],
    decoders: int,
    *,
    final_upsample: ArrayTransform | None = None,
    materialize: bool = False,
) -> NetworkOutput:
    """Pop encoder ``skips`` from deepest to shallowest and return the final head.

    Intermediate segmentation heads are not evaluated (no deep supervision).
    ``final_upsample`` optionally replaces the last stage's upsample module, and
    ``materialize`` evaluates each stage to bound the Metal graph size.
    Consumed skips leave the caller-owned list before evaluation, so Python
    references do not retain their buffers after the graph's last use.
    """
    hidden = skips.pop()
    outputs: list[array] = []
    for index in range(decoders):
        upsample = (
            final_upsample
            if final_upsample is not None and index == decoders - 1
            else getattr(owner, f"up{index}")
        )
        skip = skips.pop()
        decoder = getattr(owner, f"dec{index}")
        # Inference arithmetic may fuse upsampling, concatenation and padding.
        fused = getattr(decoder, "upsample_concat", None)
        if fused is not None and upsample is not final_upsample:
            hidden = fused(upsample, hidden, skip)
        else:
            hidden = _match_spatial(upsample(hidden), skip)
            if type(decoder) is StackedConvs:
                hidden = decoder.concatenated((hidden, skip))
            else:
                hidden = decoder(_mx().concatenate((hidden, skip), axis=-1))
        del skip
        _eval_stage(hidden, enabled=materialize)
        if index == decoders - 1:
            outputs.append(getattr(owner, f"seg{index}")(hidden))
    return outputs[0]


def _metal_residual_inference_materialization(model: ResidualEncoderUNet) -> bool:
    # Use inference-only boundaries for repeatable large 3-D residual graphs on Metal.
    """Enable stage materialization for 3D residual inference."""
    return not bool(getattr(model, "training", True)) and model.spec.spatial_dims == 3


def _eval_stage(value: array, *, enabled: bool) -> None:
    """Evaluate a stage output when materialization is enabled."""
    if enabled:
        _mx().eval(value)


def _mx():
    """Resolve the admitted MLX runtime for graph operations."""
    mx, _ = import_mlx()
    return mx


def _match_spatial(values: array, reference: array) -> array:
    # Centered crop (or zero-pad) of the upsampled tensor to the skip tensor's size.
    # A no-op when every spatial size is divisible by its stride, the normal case for
    # nnU-Net patch sizes.
    """Center-crop or zero-pad an upsampled tensor to match its skip connection."""
    mx = _mx()
    for axis in range(1, values.ndim - 1):
        delta = int(values.shape[axis] - reference.shape[axis])
        if delta > 0:
            start = delta // 2
            slices: list[slice] = [slice(None)] * values.ndim
            slices[axis] = slice(start, start + int(reference.shape[axis]))
            values = values[tuple(slices)]
        elif delta < 0:
            before = (-delta) // 2
            padding: list[tuple[int, int]] = [(0, 0)] * values.ndim
            padding[axis] = (before, -delta - before)
            values = mx.pad(values, padding)
    return values
