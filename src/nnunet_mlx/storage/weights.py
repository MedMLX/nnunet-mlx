"""Map upstream nnU-Net state-dict keys and tensor layouts to MLX network weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError

if TYPE_CHECKING:
    from mlx.nn import Module

type HostArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ConvertedState:
    """Checkpoint weights renamed and re-laid-out for the MLX network.

    ``weights`` maps MLX parameter names to contiguous float32 host arrays;
    ``dropped_keys`` lists the sorted upstream keys intentionally not carried over.
    """

    weights: dict[str, HostArray]
    dropped_keys: tuple[str, ...]


def _conv_layout(array: HostArray) -> HostArray:
    # Torch (C_out, C_in, *kernel) -> MLX (C_out, *kernel, C_in).
    """Transpose Torch convolution weights into MLX channels-last kernel layout."""
    if array.ndim not in (4, 5):
        raise ContractError(f"expected a 2-D or 3-D convolution weight, got {array.shape}")
    return np.transpose(array, (0, *range(2, array.ndim), 1))


def _transposed_conv_layout(array: HostArray) -> HostArray:
    # Torch ConvTranspose (C_in, C_out, *kernel) -> MLX (C_out, *kernel, C_in).
    """Transpose Torch transposed-convolution weights into MLX layout."""
    if array.ndim not in (4, 5):
        raise ContractError(f"expected a 2-D or 3-D transposed weight, got {array.shape}")
    return np.transpose(array, (1, *range(2, array.ndim), 0))


def map_key_and_layout(torch_key: str, array: HostArray) -> tuple[str, HostArray] | None:
    """Map one upstream parameter name and array to the MLX name and layout.

    Convolution and transposed-convolution weights are transposed to MLX's
    channels-last layout; norm parameters and biases are unchanged.
    Keys resembling supported patterns must contain valid integer stage and block
    indices; malformed lookalike keys can raise ``ValueError`` during parsing.

    Returns:
        ``(mlx_key, array)``, or ``None`` if the key is an ``all_modules`` alias or
        does not match any supported ``PlainConvUNet`` / ``ResidualEncoderUNet``
        parameter pattern. The caller decides whether ``None`` is an error.

    Raises:
        ContractError: If a convolution weight is not 4-D or 5-D.
    """
    parts = torch_key.split(".")
    if "all_modules" in parts:
        return None
    if (
        torch_key.startswith("encoder.stages.")
        and len(parts) == 8
        and parts[3].isdigit()
        and parts[4] == "convs"
    ):
        stage = int(parts[2])
        conv_index = int(parts[5])
        module, parameter = parts[6], parts[7]
        key = f"enc{stage}.block{conv_index}.{module}.{parameter}"
        converted = _conv_layout(array) if module == "conv" and parameter == "weight" else array
        return key, converted
    if torch_key.startswith("encoder.stem.convs.0.") and len(parts) == 6:
        module, parameter = parts[4], parts[5]
        key = f"stem.{module}.{parameter}"
        converted = _conv_layout(array) if module == "conv" and parameter == "weight" else array
        return key, converted
    if (
        torch_key.startswith("encoder.stages.")
        and len(parts) == 8
        and parts[3] == "blocks"
        and parts[5] in ("conv1", "conv2", "conv3")
    ):
        stage, block = int(parts[2]), int(parts[4])
        path, module, parameter = parts[5], parts[6], parts[7]
        key = f"res{stage}.block{block}.{path}.{module}.{parameter}"
        converted = _conv_layout(array) if module == "conv" and parameter == "weight" else array
        return key, converted
    if (
        torch_key.startswith("encoder.stages.")
        and len(parts) == 9
        and parts[3] == "blocks"
        and parts[5] == "skip"
        and parts[6].isdigit()
    ):
        stage, block = int(parts[2]), int(parts[4])
        module, parameter = parts[7], parts[8]
        key = f"res{stage}.block{block}.proj.{module}.{parameter}"
        converted = _conv_layout(array) if module == "conv" and parameter == "weight" else array
        return key, converted
    if torch_key.startswith("decoder.stages.") and len(parts) == 7:
        stage = int(parts[2])
        conv_index = int(parts[4])
        module, parameter = parts[5], parts[6]
        key = f"dec{stage}.block{conv_index}.{module}.{parameter}"
        converted = _conv_layout(array) if module == "conv" and parameter == "weight" else array
        return key, converted
    if torch_key.startswith("decoder.transpconvs.") and len(parts) == 4:
        stage, parameter = int(parts[2]), parts[3]
        if parameter == "weight":
            return f"up{stage}.weight", _transposed_conv_layout(array)
        return f"up{stage}.{parameter}", array
    if torch_key.startswith("decoder.seg_layers.") and len(parts) == 4:
        stage, parameter = int(parts[2]), parts[3]
        converted = _conv_layout(array) if parameter == "weight" else array
        return f"seg{stage}.{parameter}", converted
    return None


def convert_state_dict(
    state: Mapping[str, object],
    *,
    allow_lesiontracer_auxiliary_head: bool = False,
) -> ConvertedState:
    """Convert an upstream torch ``state_dict`` into MLX weights.

    Source tensors are moved to the host and converted to float32 before the
    finiteness check and layout conversion. This is not a lossless conversion for
    every possible source dtype or value.

    Args:
        state: Upstream parameter name to torch tensor (any object exposing
            ``detach().cpu().numpy()``).
        allow_lesiontracer_auxiliary_head: Drop ``decoder.organ_seg_layers.*`` instead
            of rejecting it. Each such head must have 11 output channels and match its
            ``decoder.seg_layers`` companion's remaining dimensions.

    Returns:
        The converted weights and the keys that were dropped.

    Raises:
        ContractError: If a value is not a tensor or has non-finite entries, a key is
            unsupported, two keys map to the same MLX name, or nothing converts.
    """
    converted: dict[str, HostArray] = {}
    dropped: list[str] = []
    for key, tensor in state.items():
        array = _host_tensor(tensor, name=key)
        if not np.isfinite(array).all():
            raise ContractError(f"nnU-Net parameter {key!r} contains non-finite values")
        if allow_lesiontracer_auxiliary_head and key.startswith("decoder.organ_seg_layers."):
            parts = key.split(".")
            if len(parts) != 4 or not parts[2].isdigit() or parts[3] not in {"weight", "bias"}:
                raise ContractError(f"unsupported LesionTracer auxiliary key {key!r}")
            companion = state.get(f"decoder.seg_layers.{parts[2]}.{parts[3]}")
            shape = _host_tensor(companion, name=key).shape
            if not shape or array.shape != (11, *shape[1:]):
                raise ContractError(f"invalid LesionTracer auxiliary head shape for {key!r}")
            # The pinned decoder's default organ=False never executes these heads.
            dropped.append(key)
            continue
        mapped = map_key_and_layout(key, array)
        if mapped is None:
            # ``decoder.encoder.*`` re-registers the encoder's parameters inside the
            # decoder module, so they are already covered by the ``encoder.*`` keys.
            if key.startswith("decoder.encoder.") or "all_modules" in key:
                dropped.append(key)
                continue
            raise ContractError(f"unsupported nnU-Net checkpoint key {key!r}")
        mlx_key, mlx_array = mapped
        if mlx_key in converted:
            raise ContractError(f"duplicate converted key {mlx_key!r}")
        converted[mlx_key] = np.ascontiguousarray(mlx_array, dtype=np.float32)
    if not converted:
        raise ContractError("checkpoint did not contain convertible weights")
    return ConvertedState(converted, tuple(sorted(dropped)))


def load_converted_state(model: Module, state: ConvertedState) -> None:
    """Load converted weights into ``model`` with ``strict=True`` and materialize them."""
    from nnunet_mlx.runtime import import_mlx

    mx, _ = import_mlx()
    weights = [(key, mx.array(value)) for key, value in state.weights.items()]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())


def _host_tensor(value: object, *, name: str) -> HostArray:
    """Detach a source tensor and convert it to a host NumPy array."""
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise ContractError(f"nnU-Net parameter {name!r} is not a tensor")
    cpu = getattr(detach(), "cpu", None)
    if not callable(cpu):
        raise ContractError(f"nnU-Net parameter {name!r} cannot move to CPU")
    numpy_fn = getattr(cpu(), "numpy", None)
    if not callable(numpy_fn):
        raise ContractError(f"nnU-Net parameter {name!r} cannot convert to NumPy")
    raw = numpy_fn()
    if not isinstance(raw, np.ndarray):
        raise ContractError(f"nnU-Net parameter {name!r} did not yield an ndarray")
    return cast(HostArray, raw.astype(np.float32, copy=False))
