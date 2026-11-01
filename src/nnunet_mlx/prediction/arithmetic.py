"""Explicit MLX Metal arithmetic that follows the pinned Torch float32 operation order."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from nnunet_mlx.errors import ContractError
from nnunet_mlx.kernels.metal_chunked_conv import chunked_conv2d
from nnunet_mlx.kernels.metal_fused_ordered_conv import (
    ordered_conv3d_from_packed,
    padded_ordered_conv3d,
    upsample_concat_padded,
)
from nnunet_mlx.kernels.metal_fused_ordered_norm import (
    affine_activation_padded_from_statistics,
    instance_norm_activation,
)
from nnunet_mlx.kernels.metal_fused_ordered_norm import (
    spatial_statistics as queued_spatial_statistics,
)
from nnunet_mlx.kernels.metal_instance_norm import instance_norm
from nnunet_mlx.kernels.metal_ordered_conv import ordered_conv3d
from nnunet_mlx.model.network import (
    ConvNormAct,
    MlxModule,
    NonOverlappingConvTranspose3d,
    PlainConvUNet,
    ResidualEncoderUNet,
    SpatialInstanceNorm,
    StackedConvs,
)
from nnunet_mlx.prediction.source import (
    SMILE_SOURCE_PROFILE,
    TOTALSEGMENTATOR_CT_SOURCE_PROFILE,
)
from nnunet_mlx.runtime import (
    import_mlx,
    residual_ordered_arithmetic_topology_eligible,
    tree_flatten,
)

if TYPE_CHECKING:
    from mlx.core import array
    from mlx.nn import Conv2d, Conv3d, ConvTranspose2d, ModuleTree

TORCH_ORDER_FLOAT32 = "torch-order-fp32"


class OrderedConv3d(MlxModule):
    """Channel-first reduction with bias as the first FMA accumulator value."""

    def __init__(self, layer: Conv3d) -> None:
        """Retain a compatible convolution layer for source-ordered FP32 inference."""
        super().__init__()
        if layer.dilation not in (1, (1, 1, 1)):
            raise ContractError("ordered convolution requires dilation=1")
        if layer.weight.shape[0] < 2:
            raise ContractError("ordered convolution requires at least two output channels")
        self.weight = layer.weight
        self.bias = layer.bias
        self._stride = tuple(layer.stride)
        self._padding = tuple(layer.padding)

    def __call__(self, values: array) -> array:
        """Validate NDHWC inference input and run ordered 3D convolution."""
        mx, _ = import_mlx()
        if self.training or values.dtype != mx.float32 or values.ndim != 5:
            raise ContractError("ordered convolution requires float32 NDHWC inference")
        if values.shape[-1] != self.weight.shape[-1] or any(s < 1 for s in values.shape):
            raise ContractError("ordered convolution input channels or shape do not match")
        return self._convolve(values)

    def _convolve(self, values: array) -> array:
        """Apply ordered 3D convolution with the retained weights and padding."""
        return ordered_conv3d(values, self.weight, self.bias, self._stride, self._padding)


class PaddedOrderedConv3d(OrderedConv3d):
    """Preserve ordered products while packing the positive-zero halo once."""

    def _convolve(self, values: array) -> array:
        """Apply ordered 3D convolution with a packed padding halo."""
        return padded_ordered_conv3d(values, self.weight, self.bias, self._stride, self._padding)

    @property
    def padding(self) -> tuple[int, ...]:
        """Return the convolution padding used for its packed halo."""
        return self._padding

    def from_packed(self, values: array) -> array:
        """Consume a dense NCDHW array with this convolution's halo already applied."""
        return ordered_conv3d_from_packed(values, self.weight, self.bias, self._stride)


class OrderedConv2d(MlxModule):
    """Reuse the Metal GEMM with the retained oneDNN AVX2 channel-block order."""

    def __init__(self, layer: Conv2d) -> None:
        """Retain a compatible 2D convolution for source-ordered inference."""
        super().__init__()
        if layer.dilation not in (1, (1, 1)) or layer.groups != 1:
            raise ContractError("ordered 2-D convolution requires dilation=1 and groups=1")
        channels = layer.weight.shape[-1]
        if channels != 1 and channels % 8:
            raise ContractError("ordered 2-D convolution requires 1 or a multiple of 8 inputs")
        self.weight = layer.weight
        if "bias" in layer:
            self.bias = layer.bias
        self._stride, self._padding = tuple(layer.stride), tuple(layer.padding)

    def __call__(self, values: array) -> array:
        """Validate NHWC FP32 inference input and apply ordered convolution."""
        mx, _ = import_mlx()
        bias = getattr(self, "bias", None)
        if (
            self.training
            or values.dtype != mx.float32
            or values.ndim != 4
            or self.weight.dtype != mx.float32
            or (bias is not None and bias.dtype != mx.float32)
        ):
            raise ContractError("ordered convolution requires float32 NHWC inference")
        if values.shape[-1] != self.weight.shape[-1] or any(s < 1 for s in values.shape):
            raise ContractError("ordered convolution input channels or shape do not match")
        zeros = mx.zeros((self.weight.shape[0],), dtype=mx.float32)
        bias = bias if bias is not None else zeros
        # The retained CPU switches these small maps from oneDNN to im2col/SGEMM.
        if (
            values.shape == (1, 5, 5, 512)
            and self.weight.shape == (512, 3, 3, 512)
            and self._stride == (1, 1)
            and self._padding == (1, 1)
        ):
            return chunked_conv2d(values, self.weight, bias, self._stride, self._padding)
        # oneDNN's blocked 1x1 kernel adds separately accumulated 128-channel chunks.
        chunk = 128 if self.weight.shape[1:3] == (1, 1) else values.shape[-1]
        result = None
        for start in range(0, values.shape[-1], chunk):
            partial = ordered_conv3d(
                values[:, None, ..., start : start + chunk],
                self.weight[:, None, ..., start : start + chunk],
                bias if start == 0 else zeros,
                (1, *self._stride),
                (0, *self._padding),
                channel_block=8,
            )[:, 0]
            result = partial if result is None else result + partial
        # Nonempty input channels above guarantee at least one chunk.
        return cast("array", result)


class NonOverlappingConvTranspose2d(MlxModule):
    """Channels-last, stride-equals-kernel transpose convolution using public MLX."""

    def __init__(self, layer: ConvTranspose2d) -> None:
        """Require the supported nonoverlapping transposed-convolution layout."""
        super().__init__()
        if (
            tuple(layer.stride) != tuple(layer.weight.shape[1:3])
            or layer.padding not in (0, (0, 0))
            or layer.output_padding not in (0, (0, 0))
            or layer.dilation not in (1, (1, 1))
        ):
            raise ContractError(
                "ordered transpose requires stride=kernel, dilation=1 and no padding"
            )
        self.weight = layer.weight
        if "bias" in layer:
            self.bias = layer.bias

    def __call__(self, values: array) -> array:
        """Apply nonoverlapping 2D transpose convolution to NHWC inference input."""
        mx, _ = import_mlx()
        bias = getattr(self, "bias", None)
        if (
            self.training
            or values.dtype != mx.float32
            or values.ndim != 4
            or self.weight.dtype != mx.float32
            or (bias is not None and bias.dtype != mx.float32)
        ):
            raise ContractError("ordered transpose requires float32 NHWC inference")
        batch, height, width, channels = values.shape
        outputs, kh, kw, inputs = self.weight.shape
        if channels != inputs or any(s < 1 for s in values.shape):
            raise ContractError("ordered transpose input channels or shape do not match")
        projected = mx.matmul(
            mx.reshape(values, (-1, channels)), mx.reshape(self.weight, (-1, channels)).T
        )
        result = mx.reshape(projected, (batch, height, width, outputs, kh, kw))
        result = mx.reshape(
            mx.transpose(result, (0, 1, 4, 2, 5, 3)), (batch, height * kh, width * kw, outputs)
        )
        return result + bias if bias is not None else result


class OrderedInstanceNorm(SpatialInstanceNorm):
    """Instance norm over the spatial axes of channels-last float32 input, inference only.

    Statistics and the affine output follow the torch-order CPU arithmetic via
    `instance_norm`. `output_fma` selects a fused multiply-add for the final
    `x * scale + shift` step instead of a separately rounded product and sum.
    `__call__` raises `ContractError` in training mode or when the last axis
    differs from `dims`.
    """

    def __init__(self, dims: int, *, eps: float, affine: bool, output_fma: bool = False) -> None:
        """Configure ordered instance normalization and optional fused output arithmetic."""
        super().__init__(dims, eps=eps, affine=affine)
        self.output_fma = output_fma

    def __call__(self, values: array) -> array:
        """Validate inference mode and apply ordered instance normalization."""
        if self.training:
            raise ContractError("ordered normalization supports inference only")
        if values.shape[-1] != self.dims:
            raise ContractError("ordered normalization input channels do not match")
        return instance_norm(values, self.weight, self.bias, self.eps, output_fma=self.output_fma)


class FusedOrderedConvNormAct(ConvNormAct):
    """Keep child parameter paths while fusing ordered affine and LeakyReLU."""

    def __init__(self, layer: ConvNormAct) -> None:
        """Reuse a convolution, normalization and activation slope for fused inference."""
        MlxModule.__init__(self)
        self.conv, self.norm = layer.conv, layer.norm
        self._leaky_slope = layer._leaky_slope

    @property
    def negative_slope(self) -> float:
        """Return the LeakyReLU negative slope."""
        return self._leaky_slope

    def __call__(self, values: array) -> array:
        """Apply ordered convolution and fused instance normalization with activation."""
        if self.training:
            raise ContractError("ordered normalization supports inference only")
        values = self.conv(values)
        norm = cast(OrderedInstanceNorm, self.norm)
        mx, _ = import_mlx()
        if values.dtype != mx.float32 or values.shape[-1] != norm.dims:
            raise ContractError("ordered normalization input dtype or channels do not match")
        return instance_norm_activation(
            values, norm.weight, norm.bias, norm.eps, self._leaky_slope,
            output_fma=norm.output_fma,
        )


class AffinePaddedPair(StackedConvs):
    """Write a two-block stack's middle affine/activation into its next halo."""

    def __init__(self, layer: StackedConvs) -> None:
        """Reuse the two ordered convolution blocks for fused intermediate padding."""
        MlxModule.__init__(self)
        self._count = layer._count
        self.block0, self.block1 = (
            cast(FusedOrderedConvNormAct, getattr(layer, f"block{index}")) for index in range(2)
        )

    @staticmethod
    def supports(layer: StackedConvs) -> bool:
        """Report whether a convolution stack contains exactly two blocks."""
        return layer._count == 2

    def __call__(self, values: array) -> array:
        """Run the fused two-block inference path."""
        if self.training:
            raise ContractError("fused affine padding supports inference only")
        return self._finish(self.block0.conv(values))

    def upsample_concat(
        self, upsample: MlxModule, values: array, skip: array,
    ) -> array:
        """Run ``self(concat(upsample(values), skip))`` without the shuffled copies.

        The transpose matmul is unchanged; one kernel adds its bias, applies the
        spatial shuffle and concatenation, and writes this stack's padded input.
        """
        first_conv = cast(PaddedOrderedConv3d, self.block0.conv)
        if (
            self.training
            or not isinstance(upsample, NonOverlappingConvTranspose3d)
            or "bias" not in upsample
            or values.ndim != 5
        ):
            raise ContractError("fused upsampling requires a biased 3-D stride-matched transpose")
        kernel = cast("tuple[int, int, int]", tuple(upsample.weight.shape[1:4]))
        expected = tuple(
            size * factor for size, factor in zip(values.shape[1:4], kernel, strict=True)
        )
        if skip.shape[1:4] != expected or skip.shape[0] != values.shape[0]:
            raise ContractError("fused upsampling requires the skip at the upsampled shape")
        mx, _ = import_mlx()
        channels = values.shape[-1]
        projected = mx.matmul(
            mx.reshape(values, (-1, channels)),
            mx.transpose(mx.reshape(upsample.weight, (-1, channels))),
        )
        packed = upsample_concat_padded(
            projected, upsample.bias, skip, kernel,
            cast("tuple[int, int, int]", first_conv.padding),
        )
        return self._finish(first_conv.from_packed(packed))

    def _finish(self, values: array) -> array:
        """Normalize the first block's conv output, then run the second block.

        The first norm's affine, activation and the second conv's padding are fused
        into one packed tensor; ``values`` must be float32 channels-last.
        """
        first, second = self.block0, self.block1
        first_norm = cast(OrderedInstanceNorm, first.norm)
        mx, _ = import_mlx()
        if values.dtype != mx.float32 or values.shape[-1] != first_norm.dims:
            raise ContractError("ordered normalization input dtype or channels do not match")
        average, inverse = queued_spatial_statistics(values, first_norm.eps)
        second_conv = cast(PaddedOrderedConv3d, second.conv)
        packed = affine_activation_padded_from_statistics(
            values, average, inverse, first_norm.weight, first_norm.bias,
            first.negative_slope, second_conv.padding,
            output_fma=first_norm.output_fma,
        )
        values = second_conv.from_packed(packed)
        second_norm = cast(OrderedInstanceNorm, second.norm)
        return instance_norm_activation(
            values, second_norm.weight, second_norm.bias, second_norm.eps,
            second.negative_slope, output_fma=second_norm.output_fma,
        )


def _fuse_totalsegmentator_blocks(model: PlainConvUNet) -> None:
    """Swap eligible ordered conv/norm blocks for fused modules, in place.

    First each ``ConvNormAct`` with an ordered norm is fused; then stacked pairs
    that support it become :class:`AffinePaddedPair`. Leaves the model in eval mode.
    """
    from mlx import utils

    blocks = [
        (name, FusedOrderedConvNormAct(layer)) for name, layer in model.named_modules()
        if isinstance(layer, ConvNormAct) and isinstance(layer.norm, OrderedInstanceNorm)
    ]
    model.update_modules(cast("dict[str, ModuleTree]", utils.tree_unflatten(blocks)))
    pairs: list[tuple[str, MlxModule]] = []
    for name, layer in model.named_modules():
        if not isinstance(layer, StackedConvs) or not AffinePaddedPair.supports(layer):
            continue
        children = (
            cast(ConvNormAct, getattr(layer, f"block{index}")) for index in range(2)
        )
        if any(
            not isinstance(block.conv, PaddedOrderedConv3d)
            or not isinstance(block.norm, OrderedInstanceNorm)
            for block in children
        ):
            raise ContractError(
                "fused affine padding requires ordered convolution and norm children"
            )
        pairs.append((name, AffinePaddedPair(layer)))
    model.update_modules(cast("dict[str, ModuleTree]", utils.tree_unflatten(pairs)))
    model.eval()


def apply_inference_arithmetic(
    model: object,
    arithmetic: str,
    *,
    source_profile: str | None = None,
    patch_size: tuple[int, ...] | None = None,
) -> None:
    """Swap one loaded model's operators for torch-order float32 Metal kernels, in place.

    Only the given instance is changed; global MLX classes are never patched.
    A `ResidualEncoderUNet` is accepted only for the SMILE profile topology with
    a `(640, 640)` patch, and is then pinned to a `(1, 640, 640, 1)` NHWC input.
    Otherwise the model must be a 3-D `PlainConvUNet`; for the TotalSegmentator
    CT profile its convolution blocks are additionally fused. The model is left
    in evaluation mode with `inference_arithmetic` set.

    Args:
        model: Loaded network, mutated in place.
        arithmetic: Must be `TORCH_ORDER_FLOAT32`.
        source_profile: Compatibility profile name recorded at conversion, if any.
        patch_size: Plans patch size in voxels, required to match for residual models.

    Raises:
        ContractError: If the arithmetic, topology, profile, patch size, training
            mode, bias/affine settings or float32 parameter dtype is unsupported.
    """
    if arithmetic != TORCH_ORDER_FLOAT32:
        raise ContractError(f"unsupported inference arithmetic {arithmetic!r}")
    residual = isinstance(model, ResidualEncoderUNet)
    totalsegmentator = source_profile == TOTALSEGMENTATOR_CT_SOURCE_PROFILE
    if isinstance(model, ResidualEncoderUNet):
        if (
            not residual_ordered_arithmetic_topology_eligible(model.spec)
            or source_profile != SMILE_SOURCE_PROFILE
            or patch_size != (640, 640)
            or model.training
        ):
            raise ContractError(
                "torch-order-fp32 requires the qualified SMILE topology, "
                "source and patch in evaluation mode"
            )
    elif not isinstance(model, PlainConvUNet) or model.spec.spatial_dims != 3 or model.training:
        raise ContractError("torch-order-fp32 requires a 3-D PlainConvUNet in evaluation mode")
    if not model.spec.conv_bias or not model.spec.norm_affine:
        raise ContractError("torch-order-fp32 requires convolution bias and affine normalization")
    mx, _ = import_mlx()
    import mlx.nn as nn

    if any(value.dtype != mx.float32 for _, value in tree_flatten(model.parameters())):
        raise ContractError("torch-order-fp32 requires float32 model parameters")
    replacements: list[tuple[str, MlxModule]] = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv3d):
            ordered_conv = PaddedOrderedConv3d if totalsegmentator else OrderedConv3d
            replacements.append((name, ordered_conv(layer)))
        elif residual and isinstance(layer, nn.Conv2d):
            replacements.append((name, OrderedConv2d(layer)))
        elif residual and isinstance(layer, nn.ConvTranspose2d):
            replacements.append((name, NonOverlappingConvTranspose2d(layer)))
        elif isinstance(layer, SpatialInstanceNorm):
            replacement = OrderedInstanceNorm(
                layer.dims,
                eps=layer.eps,
                affine=True,
                output_fma=residual,
            )
            replacement.weight, replacement.bias = layer.weight, layer.bias
            replacements.append((name, replacement))
    from mlx import utils

    model.update_modules(cast("dict[str, ModuleTree]", utils.tree_unflatten(replacements)))
    model.inference_arithmetic = arithmetic
    if isinstance(model, ResidualEncoderUNet):
        model.inference_source_profile = source_profile
        model.inference_input_shape = (1, 640, 640, 1)
    model.eval()
    if totalsegmentator:
        _fuse_totalsegmentator_blocks(cast(PlainConvUNet, model))
