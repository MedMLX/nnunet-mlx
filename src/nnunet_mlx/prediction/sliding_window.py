"""Synchronized MLX sliding-window inference for 2-D/3-D nnU-Net logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from importlib import import_module
from itertools import combinations, product
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError
from nnunet_mlx.runtime import InferenceRuntime, import_mlx
from nnunet_mlx.runtime_types import ScatterUpdate

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.model.types import PredictionModel

type HostArray = npt.NDArray[np.generic]
type FloatArray = npt.NDArray[np.float32]
type LogitArray = npt.NDArray[np.float16] | npt.NDArray[np.float32]


def compute_steps(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    *,
    step_fraction: float = 0.5,
) -> tuple[tuple[int, ...], ...]:
    """Return per-axis patch start offsets, as upstream's ``compute_steps_for_sliding_window``.

    ``image_size`` is the (padded) spatial shape and ``patch_size`` the patch shape
    in the same axis order. Along each axis ideal offsets are evenly spread from 0
    to ``image - patch`` with a stride of at most ``patch * step_fraction``;
    offsets are then rounded to integer voxels, so individual gaps can be larger.

    Raises:
        ContractError: If ``step_fraction`` is outside (0, 1], the ranks differ or are
            not 2/3, or an image axis is smaller than the patch.
    """
    if not 0 < step_fraction <= 1:
        raise ContractError(f"step_fraction must be in (0, 1], got {step_fraction}")
    if len(image_size) not in (2, 3) or len(patch_size) != len(image_size):
        raise ContractError("sliding-window inference requires matching 2-D or 3-D shapes")
    axes = tuple(
        _axis_steps(int(image), int(patch), step_fraction=step_fraction)
        for image, patch in zip(image_size, patch_size, strict=True)
    )
    return axes


def _axis_steps(image: int, patch: int, *, step_fraction: float) -> tuple[int, ...]:
    """Compute sliding-window start positions along one spatial axis."""
    if image < patch:
        raise ContractError(f"padded image dimension {image} is smaller than patch {patch}")
    count = int(np.ceil((image - patch) / (patch * step_fraction))) + 1
    maximum = image - patch
    actual = maximum / (count - 1) if count > 1 else 0.0
    return tuple(round(actual * index) for index in range(count))


@lru_cache(maxsize=8)
def gaussian_importance(patch_size: tuple[int, ...]) -> FloatArray:
    """Return the float32 tile weight map: a separable Gaussian scaled to a maximum of 1.

    The Gaussian is centered at ``size // 2`` with sigma ``size / 8`` per axis, as
    upstream's ``compute_gaussian``, but built analytically instead of by filtering
    an impulse, and with exact zeros replaced by the smallest positive normal float32. The
    accumulated logits are divided by the accumulated weights, so the overall scale
    cancels; ``nnunet_fp16_gaussian`` reproduces upstream's scaled float16 map for
    the half-precision path.
    """
    if any(size < 1 for size in patch_size):
        raise ContractError(f"invalid patch size {patch_size}")
    exponent = np.zeros(patch_size, dtype=np.float32)
    for axis, size in enumerate(patch_size):
        center = size // 2
        sigma = size / 8
        coordinates = np.arange(size, dtype=np.float32)
        shape = tuple(size if index == axis else 1 for index in range(len(patch_size)))
        exponent += (((coordinates - center) / sigma) ** 2).reshape(shape)
    values = np.exp(-0.5 * exponent).astype(np.float32, copy=False)
    values /= np.float32(values.max())
    values[values == 0] = np.finfo(np.float32).tiny
    return values


@lru_cache(maxsize=8)
def nnunet_fp16_gaussian(patch_size: tuple[int, ...]) -> npt.NDArray[np.float16]:
    """Match nnU-Net's scaled, rounded Gaussian, including its nonzero floor.

    An impulse at ``size // 2`` is filtered with sigma ``size / 8`` per axis, scaled
    so the maximum is 10 (upstream's ``value_scaling_factor``) and cast to float16.
    Entries that round to zero take the smallest nonzero value.
    """
    gaussian_filter = cast(
        Callable[..., npt.NDArray[np.float64]], import_module("scipy.ndimage").gaussian_filter
    )

    impulse = np.zeros(patch_size)
    impulse[tuple(size // 2 for size in patch_size)] = 1
    weights = gaussian_filter(impulse, [size / 8 for size in patch_size], mode="constant")
    rounded = (weights / (weights.max() / 10)).astype(np.float16)
    rounded[rounded == 0] = rounded[rounded != 0].min()
    return rounded


def predict_logits(
    model: PredictionModel,
    data_channels_first: HostArray,
    *,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> FloatArray:
    """Return finite host ``(num_classes, *spatial)`` logits after synchronized MLX inference.

    Args:
        model: Callable mapping a channels-last tile ``(1, *patch, C_in)`` to float32
            ``(1, *patch, num_classes)`` logits.
        data_channels_first: Finite float32 ``(C_in, *spatial)`` input, 2-D or 3-D,
            in the plans' axis order.
        patch_size: Patch shape in the same axis order as ``spatial``.
        num_classes: Number of output channels of ``model``.
        mirroring_axes: Spatial axis indices (0-based, excluding the channel axis)
            for test-time mirroring; every non-empty subset is averaged in, as in
            upstream.
        step_fraction: Tile stride as a fraction of the patch size, in (0, 1].
        accumulation: ``"float32"``, or ``"nnunet-fp16"`` to accumulate and round in
            float16 like upstream's default predictor.

    Returns:
        Float32 Gaussian-weighted average over overlapping tiles, cropped to the input
        extent. Inputs smaller than the patch are zero-padded for inference and the
        padding is removed from the result.

    Raises:
        ContractError: If the input, patch, class count or mirroring axes are invalid,
            the model output has the wrong shape or dtype, or the logits are non-finite.
    """
    logits = predict_host_logits(
        model, data_channels_first, patch_size=patch_size, num_classes=num_classes,
        mirroring_axes=mirroring_axes, step_fraction=step_fraction, accumulation=accumulation,
    )
    return np.asarray(logits, dtype=np.float32)


def predict_host_logits(
    model: PredictionModel,
    data_channels_first: HostArray,
    *,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> LogitArray:
    """Return ``predict_logits`` in the accumulation dtype: float16 for ``nnunet-fp16``.

    The float16 slabs are already rounded, so the values equal ``predict_logits``
    exactly while the host volume takes half the memory and no conversion.
    """
    patch, axes = _validate_request(data_channels_first, patch_size, num_classes, mirroring_axes)
    padded, crop = _pad_minimum(data_channels_first, patch)
    mx, _ = import_mlx()
    upstream_fp16 = accumulation == "nnunet-fp16"
    dtype = mx.float16 if upstream_fp16 else mx.float32
    cropped_shape = tuple(int(item.stop - item.start) for item in crop)
    shape = (num_classes, *cropped_shape)
    output: LogitArray = (
        np.empty(shape, dtype=np.float16) if upstream_fp16 else np.empty(shape, dtype=np.float32)
    )

    def store(start: int, stop: int, normalized: array) -> None:
        # Transpose each finalized slab into the channel-first host volume.
        """Store a completed logits slab on the host."""
        first = crop[0]
        lower, upper = max(start, first.start), min(stop, first.stop)
        if lower >= upper:
            return
        host = np.asarray(normalized, dtype=output.dtype)
        selected = host[(slice(lower - start, upper - start), *crop[1:])]
        output[:, lower - first.start : upper - first.start] = np.moveaxis(selected, -1, 0)

    # A single worker copies one slab to the host while Metal runs later tiles;
    # each normalized slab is evaluated before submission and owned by its task.
    with ThreadPoolExecutor(max_workers=1) as copier:
        pending: list[Future[None]] = []

        def finalize(start: int, stop: int, logits: array, normalizer: array) -> None:
            """Normalize and copy a completed logits slab to the output array."""
            normalized = (logits.astype(mx.float32) / normalizer.astype(mx.float32)).astype(dtype)
            mx.eval(normalized)
            while pending:
                pending.pop().result()
            pending.append(copier.submit(store, start, stop, normalized))

        _accumulate_logits(
            model, padded, patch=patch, axes=axes, num_classes=num_classes,
            step_fraction=step_fraction, accumulation=accumulation, finalize=finalize,
        )
        while pending:
            pending.pop().result()
    if not np.isfinite(output).all():
        raise ContractError("sliding-window inference produced non-finite logits")
    return output


def predict_device_logits(
    model: PredictionModel,
    data_channels_first: HostArray,
    *,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> array:
    """Return cropped channels-last `(*spatial,C)` normalized logits left on Metal.

    Values equal ``predict_logits`` transposed, in float16 for ``nnunet-fp16``
    accumulation and float32 otherwise. Finiteness is not checked here.
    """
    patch, axes = _validate_request(data_channels_first, patch_size, num_classes, mirroring_axes)
    padded, crop = _pad_minimum(data_channels_first, patch)
    mx, _ = import_mlx()
    dtype = mx.float16 if accumulation == "nnunet-fp16" else mx.float32
    # Each finished slab is written in place, so the volume, the accumulation
    # window and one slab bound device score memory.
    volume = mx.zeros((*padded.shape[1:], num_classes), dtype=dtype)

    def finalize(start: int, stop: int, logits: array, normalizer: array) -> None:
        """Collect a normalized logits slab on the device."""
        nonlocal volume
        normalized = (logits.astype(mx.float32) / normalizer.astype(mx.float32)).astype(dtype)
        mx.eval(normalized)
        volume[start:stop] = normalized
        del normalized
        mx.eval(volume)

    _accumulate_logits(
        model, padded, patch=patch, axes=axes, num_classes=num_classes,
        step_fraction=step_fraction, accumulation=accumulation, finalize=finalize,
    )
    if tuple(item.stop - item.start for item in crop) != padded.shape[1:]:
        volume = mx.contiguous(volume[crop])
        mx.eval(volume)
    return volume


def add_fold_device_logits(
    model: PredictionModel,
    data_channels_first: HostArray,
    total: array,
    *,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int],
    step_fraction: float,
    accumulation: str,
) -> None:
    """Add one fold's cropped, normalized slabs into an existing device total.

    ``total`` is the first fold's channels-last output in the accumulation dtype.
    Normalize each later slab in float32 and round before fold addition, retaining
    the same half-precision rounding as a separately materialized fold volume.
    Check only the cropped fold logits for finiteness, as ``predict_device_logits``
    followed by the ensemble's finite check does.
    """
    patch, axes = _validate_request(data_channels_first, patch_size, num_classes, mirroring_axes)
    padded, crop = _pad_minimum(data_channels_first, patch)
    mx, _ = import_mlx()
    finite = mx.array(True)

    def finalize(start: int, stop: int, logits: array, normalizer: array) -> None:
        """Add a completed fold logits slab to the device accumulator."""
        nonlocal finite
        first = crop[0]
        lower, upper = max(start, first.start), min(stop, first.stop)
        if lower >= upper:
            return
        normalized = (logits.astype(mx.float32) / normalizer.astype(mx.float32)).astype(total.dtype)
        selected = normalized[(slice(lower - start, upper - start), *crop[1:])]
        destination = slice(lower - first.start, upper - first.start)
        finite = finite & mx.all(mx.isfinite(selected))
        updated = total[destination] + selected
        # Evaluate before replacing the slice, releasing views of the total so
        # the update can reuse its storage instead of copying a complete volume.
        mx.eval(updated, finite)
        total[destination] = updated
        del normalized, selected, updated
        mx.eval(total)

    _accumulate_logits(
        model, padded, patch=patch, axes=axes, num_classes=num_classes,
        step_fraction=step_fraction, accumulation=accumulation, finalize=finalize,
    )
    if not bool(finite.item()):
        raise ContractError("sliding-window inference produced non-finite logits")


def predict_labels(
    model: PredictionModel,
    data_channels_first: FloatArray,
    *,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> npt.NDArray[np.uint16]:
    """Reduce 3-D class logits on Metal, transferring only cropped spatial labels.

    Tile order, Gaussian accumulation and final half rounding match
    ``predict_logits``. Equal normalized values choose the earliest channel.
    This path does not resample logits or average multiple folds.
    """
    from nnunet_mlx.kernels.metal_normalized_labels import INVALID_LABEL, normalized_labels

    if data_channels_first.ndim != 4 or len(patch_size) != 3:
        raise ContractError("label-only inference requires three-dimensional class data")
    patch, axes = _validate_request(data_channels_first, patch_size, num_classes, mirroring_axes)
    padded, crop = _pad_minimum(data_channels_first, patch)
    mx, _ = import_mlx()
    slabs: list[array] = []

    def finalize(start: int, stop: int, logits: array, normalizer: array) -> None:
        """Reduce normalized logits to class labels for one completed slab."""
        del start, stop
        labels, _ = normalized_labels(
            logits, normalizer, round_to_float16=accumulation == "nnunet-fp16",
        )
        mx.eval(labels)
        slabs.append(labels)

    _accumulate_logits(
        model, padded, patch=patch, axes=axes, num_classes=num_classes,
        step_fraction=step_fraction, accumulation=accumulation, finalize=finalize,
    )
    cropped = mx.concatenate(slabs, axis=0)[crop]
    # The host-logit path also excludes minimum padding before checking finiteness.
    finite = mx.all(cropped != INVALID_LABEL)
    mx.eval(cropped, finite)
    if not bool(finite.item()):
        raise ContractError("sliding-window inference produced non-finite logits")
    return np.asarray(cropped, dtype=np.uint16)


def _accumulate_logits(
    model: PredictionModel,
    padded: HostArray,
    *,
    patch: tuple[int, ...],
    axes: tuple[int, ...],
    num_classes: int,
    step_fraction: float,
    accumulation: str,
    finalize: Callable[[int, int, array, array], None],
) -> None:
    """Accumulate Gaussian-weighted tiles, finalizing first-axis slabs as they complete.

    Tiles run in ``itertools.product`` order, first spatial axis outermost. After
    the last tile starting at first-axis offset ``s_i``, no later tile touches
    planes before ``s_(i+1)``, so ``finalize(start, stop, logits, normalizer)``
    receives those padded planes as channels-last ``(stop-start,*rest,C)`` and
    ``(stop-start,*rest,1)`` views. Score storage therefore spans one patch
    along that axis instead of the whole volume, while every voxel receives the
    same additions in the same order as a full-volume accumulator.

    ``finalize`` must evaluate whatever it derives and keep no reference to
    the views: the window is replaced when it returns.
    """
    if accumulation not in {"float32", "nnunet-fp16"}:
        raise ContractError(f"unsupported sliding-window accumulation {accumulation!r}")
    spatial = padded.shape[1:]
    steps = compute_steps(spatial, patch, step_fraction=step_fraction)
    mx, _ = import_mlx()
    inference_runtime = cast(InferenceRuntime, mx)
    upstream_fp16 = accumulation == "nnunet-fp16"
    dtype = mx.float16 if upstream_fp16 else mx.float32
    window_shape = (patch[0], *spatial[1:])
    logits = mx.zeros((*window_shape, num_classes), dtype=dtype)
    normalizer = mx.zeros((*window_shape, 1), dtype=dtype)
    origin = 0
    weights = nnunet_fp16_gaussian(patch) if upstream_fp16 else gaussian_importance(patch)
    gaussian = mx.array(weights)[..., None]
    gaussian_float32 = gaussian.astype(mx.float32)
    volume = mx.array(np.transpose(padded, (*range(1, padded.ndim), 0)))
    starts = list(product(*steps))
    tiles = [
        tuple(slice(start, start + size) for start, size in zip(offsets, patch, strict=True))
        for offsets in starts
    ]

    def forward(slices: tuple[slice, ...]) -> array:
        """Run the model and requested mirroring for one input tile."""
        return mirrored_prediction(
            model, volume[slices][None], axes, inference_runtime, num_classes=num_classes,
        )

    # Keep the next tile's network queued while this tile accumulates, so Metal
    # does not idle during graph construction; at most two tile graphs are live.
    # Accumulation still applies tiles in order with unchanged arithmetic.
    prediction = forward(tiles[0])
    mx.async_eval(prediction)
    for index, slices in enumerate(tiles):
        local = (slice(slices[0].start - origin, slices[0].stop - origin), *slices[1:])
        updated = None
        if upstream_fp16:
            weighted = prediction[0] * gaussian_float32
            updated = (logits[local].astype(mx.float32) + weighted).astype(dtype)
            mx.async_eval(updated)
        upcoming = forward(tiles[index + 1]) if index + 1 < len(tiles) else None
        if upcoming is not None:
            mx.async_eval(upcoming)
        if updated is not None:
            # Finishing the tile first drops its view of the window, letting the
            # slice update reuse the buffer instead of copying the whole window.
            mx.eval(updated)
            logits[local] = updated
            del updated
        else:
            logits = cast(ScatterUpdate, logits.at[local]).add(prediction[0] * gaussian)
        normalizer = cast(ScatterUpdate, normalizer.at[local]).add(gaussian)
        mx.async_eval(logits, normalizer)
        if upcoming is not None:
            prediction = upcoming
        following = starts[index + 1][0] if index + 1 < len(tiles) else spatial[0]
        if following == starts[index][0]:
            continue
        completed = following - origin
        finalize(origin, following, logits[:completed], normalizer[:completed])
        if index + 1 == len(tiles):
            break
        # Carry the still-open overlap into a fresh window. Releasing the old
        # window before allocating the new one bounds the peak to one window
        # plus the overlap.
        carried = (logits[completed:] * 1, normalizer[completed:] * 1)
        mx.eval(*carried)
        del logits, normalizer
        overlap = patch[0] - completed
        logits = mx.zeros((*window_shape, num_classes), dtype=dtype)
        normalizer = mx.zeros((*window_shape, 1), dtype=dtype)
        if overlap:
            logits[:overlap] = carried[0]
            normalizer[:overlap] = carried[1]
        del carried
        origin = following


def _validate_request(
    data: HostArray,
    patch_size: Sequence[int],
    num_classes: int,
    mirroring_axes: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate float32 data, patch dimensions, classes and mirroring axes."""
    spatial_dims = data.ndim - 1
    if (
        data.dtype != np.dtype(np.float32)
        or spatial_dims not in (2, 3)
        or not np.isfinite(data).all()
    ):
        raise ContractError(f"expected finite float32 (C,*spatial) 2-D/3-D input, got {data.shape}")
    patch = tuple(int(value) for value in patch_size)
    if len(patch) != spatial_dims or any(value < 1 for value in patch):
        raise ContractError(f"invalid patch size {patch}")
    if num_classes < 1:
        raise ContractError("sliding-window inference requires at least one output channel")
    axes = tuple(int(axis) for axis in mirroring_axes)
    if any(axis not in range(spatial_dims) for axis in axes) or len(set(axes)) != len(axes):
        raise ContractError(f"invalid mirroring axes {axes}")
    return patch, axes


def mirrored_prediction(
    model: PredictionModel,
    tile: array,
    axes: tuple[int, ...],
    mx: InferenceRuntime,
    *,
    num_classes: int,
) -> array:
    """Average the tile and its mirrored predictions, flipped back, in upstream order.

    Mirrored inputs are made contiguous before the network; outputs stay on
    Metal and are summed lazily. Each view is queued with ``async_eval`` once
    built, so the forwards run one after another rather than interleaved in one
    graph, which bounds intermediates to about one forward at unchanged speed.
    """
    expected_shape = (*tile.shape[:-1], num_classes)
    raw_prediction = model(tile)
    _validate_prediction(raw_prediction, expected_shape, mx)
    prediction = cast("array", raw_prediction)
    if not axes:
        return prediction
    predictions = [prediction]
    mx.async_eval(prediction)
    for axis_count in range(1, len(axes) + 1):
        for selected in combinations(axes, axis_count):
            mlx_axes = tuple(axis + 1 for axis in selected)
            mirrored = mx.contiguous(mx.flip(tile, axis=mlx_axes))
            raw_prediction = model(mirrored)
            _validate_prediction(raw_prediction, expected_shape, mx)
            predictions.append(mx.flip(cast("array", raw_prediction), axis=mlx_axes))
            mx.async_eval(predictions[-1])
    return sum(predictions[1:], predictions[0]) / len(predictions)


def _validate_prediction(
    prediction: object, expected_shape: tuple[int, ...], mx: InferenceRuntime
) -> None:
    """Check each output before broadcasting or host staging can hide a mismatch."""
    if not isinstance(prediction, cast("type[array]", mx.array)) or prediction.dtype != mx.float32:
        raise ContractError("model output must be a float32 MLX array")
    if prediction.shape != expected_shape:
        raise ContractError(
            f"model output shape {prediction.shape} does not match {expected_shape}"
        )


def _pad_minimum(
    data: HostArray,
    patch: tuple[int, ...],
) -> tuple[HostArray, tuple[slice, ...]]:
    # Zero-pad axes shorter than the patch, with the odd voxel after, as upstream's
    # ``pad_nd_image``; the returned slices crop the padding back off.
    """Zero-pad undersized spatial axes and return slices that remove the padding."""
    padding: list[tuple[int, int]] = [(0, 0)]
    crop: list[slice] = []
    for size, minimum in zip(data.shape[1:], patch, strict=True):
        total = max(0, minimum - size)
        before = total // 2
        padding.append((before, total - before))
        crop.append(slice(before, before + size))
    padded = (np.pad(data, padding, mode="constant", constant_values=0)
              if any(before or after for before, after in padding) else np.ascontiguousarray(data))
    return padded, tuple(crop)
