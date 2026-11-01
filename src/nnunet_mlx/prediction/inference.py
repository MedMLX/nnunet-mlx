"""MLX-native array inference, fold averaging, and label conversion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.errors import ContractError
from nnunet_mlx.prediction.sliding_window import predict_logits
from nnunet_mlx.runtime import validate_inference_execution

if TYPE_CHECKING:
    from mlx.core import array

    from nnunet_mlx.model.types import PredictionModel

type HostArray = npt.NDArray[np.generic]
type FloatArray = npt.NDArray[np.float32]


def average_fold_logits(
    models: Sequence[PredictionModel],
    data_channels_first: HostArray,
    *,
    patch_size: Sequence[int],
    num_outputs: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> FloatArray:
    """Average per-fold sliding-window logits, following upstream's ensemble arithmetic.

    Args:
        models: One callable per fold, each mapping a channels-last tile to logits.
        data_channels_first: Finite float32 preprocessed ``(C, *spatial)`` data in the
            plans' axis order.
        patch_size: Patch shape; two entries select 2-D inference slice by slice.
        num_outputs: Number of network output channels.
        mirroring_axes: Spatial axes for test-time mirroring.
        step_fraction: Sliding-window stride as a fraction of the patch size.
        accumulation: ``"float32"`` or ``"nnunet-fp16"``; see ``predict_logits``.

    Returns:
        Float32 ``(num_outputs, *spatial)`` logits: the fold sum divided by the number
        of folds. With ``nnunet-fp16`` each fold is rounded to float16 before the
        in-order sum and division, because upstream keeps half tensors through the
        ensemble.

    Raises:
        ContractError: If ``models`` is empty or any fold or the average is non-finite.
    """

    if not models:
        raise ContractError("fold inference requires at least one model")
    patch = tuple(int(value) for value in patch_size)
    data = np.asarray(data_channels_first)
    upstream_fp16 = accumulation == "nnunet-fp16"
    total: HostArray | None = None
    for model in models:
        current = _predict_model_logits(
            model,
            data,
            patch_size=patch,
            num_outputs=num_outputs,
            mirroring_axes=mirroring_axes,
            step_fraction=step_fraction,
            accumulation=accumulation,
        )
        # Upstream retains half tensors across fold addition and division.
        values = current.astype(np.float16) if upstream_fp16 else current
        if total is None:
            # Each fold's logits are freshly allocated, so the sum can own them.
            total = values
        else:
            total += values
    if total is None:
        raise AssertionError("fold ensemble unexpectedly produced no logits")
    total /= np.float32(len(models))
    if not np.isfinite(total).all():
        raise ContractError("fold averaging produced non-finite logits")
    return np.asarray(total, dtype=np.float32)


def average_fold_device_logits(
    models: Sequence[PredictionModel],
    data_channels_first: HostArray,
    *,
    patch_size: Sequence[int],
    num_outputs: int,
    mirroring_axes: Sequence[int] = (),
    step_fraction: float = 0.5,
    accumulation: str = "float32",
) -> array:
    """Average 3-D fold logits on Metal as channels-last `(*spatial,C)` float32.

    Values equal ``average_fold_logits`` transposed: folds are summed in order in
    the accumulation dtype and divided in float32, then rounded back for
    ``nnunet-fp16``. Later folds update the first fold's volume slab by slab.
    Non-finite fold or averaged logits raise ``ContractError``.
    """
    from nnunet_mlx.prediction.sliding_window import (
        add_fold_device_logits,
        predict_device_logits,
    )
    from nnunet_mlx.runtime import import_mlx

    if not models:
        raise ContractError("fold inference requires at least one model")
    if len(patch_size) != 3:
        raise ContractError("device fold averaging requires a three-dimensional patch")
    mx, _ = import_mlx()
    data = np.asarray(data_channels_first)
    total: array | None = None
    for model in models:
        if total is None:
            total = predict_device_logits(
                model, data, patch_size=patch_size, num_classes=num_outputs,
                mirroring_axes=mirroring_axes, step_fraction=step_fraction,
                accumulation=accumulation,
            )
            if not bool(mx.all(mx.isfinite(total)).item()):
                raise ContractError("sliding-window inference produced non-finite logits")
        else:
            add_fold_device_logits(
                model, data, total, patch_size=patch_size, num_classes=num_outputs,
                mirroring_axes=mirroring_axes, step_fraction=step_fraction,
                accumulation=accumulation,
            )
    if total is None:
        raise AssertionError("fold ensemble unexpectedly produced no logits")
    result = total.astype(mx.float32)
    if len(models) > 1:
        result = (result / mx.array(len(models), dtype=mx.float32)).astype(total.dtype)
    result = result.astype(mx.float32)
    finite = mx.all(mx.isfinite(result))
    mx.eval(result, finite)
    if not bool(finite.item()):
        raise ContractError("fold averaging produced non-finite logits")
    return result


def _predict_model_logits(
    model: PredictionModel,
    data: HostArray,
    *,
    patch_size: tuple[int, ...],
    num_outputs: int,
    mirroring_axes: Sequence[int],
    step_fraction: float,
    accumulation: str,
) -> FloatArray:
    # A 2-D network on (C, S, H, W) data runs on each slice along axis 1 independently.
    """Run sliding-window logits for one model using the requested accumulation mode."""
    if len(patch_size) == 2 and data.ndim == 4:
        per_slice = [
            predict_logits(
                model,
                data[:, slice_index],
                patch_size=patch_size,
                num_classes=num_outputs,
                mirroring_axes=mirroring_axes,
                step_fraction=step_fraction,
                accumulation=accumulation,
            )
            for slice_index in range(int(data.shape[1]))
        ]
        return np.ascontiguousarray(np.stack(per_slice, axis=1), dtype=np.float32)
    return predict_logits(
        model,
        data,
        patch_size=patch_size,
        num_classes=num_outputs,
        mirroring_axes=mirroring_axes,
        step_fraction=step_fraction,
        accumulation=accumulation,
    )


def logits_to_probabilities(logits: HostArray, *, has_regions: bool) -> FloatArray:
    """Convert finite ``(C, H, W)`` or ``(C, D, H, W)`` logits to probabilities.

    Region channels use independent sigmoids; class channels use a softmax
    across axis 0. Return contiguous float32 data with the input shape.
    Malformed rank or non-finite values raise ``ContractError``.
    """
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim not in (3, 4) or not np.isfinite(values).all():
        raise ContractError(f"expected finite channel-first 2-D/3-D logits, got {values.shape}")
    if has_regions:
        probabilities = _stable_sigmoid(values)
    else:
        shifted = values - np.max(values, axis=0, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=0, keepdims=True)
    return np.ascontiguousarray(probabilities, dtype=np.float32)


def _stable_sigmoid(values: FloatArray) -> FloatArray:
    """Compute sigmoid probabilities without overflowing large-magnitude logits."""
    probabilities = np.empty_like(values)
    nonnegative = values >= 0
    probabilities[nonnegative] = 1 / (1 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = exponent / (1 + exponent)
    return probabilities


def probabilities_to_segmentation(
    probabilities: HostArray,
    *,
    regions_class_order: Sequence[int] | None,
) -> npt.NDArray[np.uint16]:
    """Decode channel-first 2D/3D probabilities to a uint16 spatial label map.

    With no region order, labels are axis-0 argmax indices. Otherwise, each
    channel exceeding 0.5 writes its corresponding positive label; later
    channels overwrite earlier labels on overlap. Unselected voxels stay zero.
    Finiteness and rank are checked; a [0, 1] value range is not enforced here.
    """
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim not in (3, 4) or not np.isfinite(values).all():
        raise ContractError(
            f"expected finite channel-first 2-D/3-D probabilities, got {values.shape}"
        )
    if regions_class_order is None:
        return np.asarray(np.argmax(values, axis=0), dtype=np.uint16)
    order = _validated_region_order(regions_class_order, int(values.shape[0]))
    segmentation = np.zeros(values.shape[1:], dtype=np.uint16)
    for channel, label in enumerate(order):
        segmentation[values[channel] > 0.5] = label
    return segmentation


def logits_to_segmentation(
    logits: HostArray,
    *,
    regions_class_order: Sequence[int] | None,
) -> npt.NDArray[np.uint16]:
    """Convert logits to labels without returning a probability volume.

    For class models, take the argmax of float32 logits directly, as nnU-Net
    does when probabilities are not requested. A float32 softmax can round
    distinct logits to equal probabilities, so its argmax can differ at near
    ties. For regions, compute float32 sigmoid probabilities and apply the
    same strict 0.5 threshold and ordered overrides as the probability route.
    """

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim not in (3, 4) or not np.isfinite(values).all():
        raise ContractError(f"expected finite channel-first 2-D/3-D logits, got {values.shape}")
    if regions_class_order is None:
        if values.shape[0] < 2:
            raise ContractError("class inference requires at least two output channels")
        return np.asarray(np.argmax(values, axis=0), dtype=np.uint16)
    order = _validated_region_order(regions_class_order, int(values.shape[0]))
    probabilities = _stable_sigmoid(values)
    segmentation = np.zeros(values.shape[1:], dtype=np.uint16)
    for channel, label in enumerate(order):
        segmentation[probabilities[channel] > 0.5] = label
    return segmentation


def _validated_region_order(regions_class_order: Sequence[int], channels: int) -> tuple[int, ...]:
    """Validate the class order used to map region predictions to labels."""
    order = tuple(int(label) for label in regions_class_order)
    if len(order) != channels or any(label < 1 for label in order):
        raise ContractError("regions_class_order does not match output channels")
    return order


@dataclass(frozen=True, slots=True)
class Prediction:
    """Label map ``(*spatial)`` plus, if requested, ``(C, *spatial)`` float32 probabilities."""

    segmentation: npt.NDArray[np.uint16]
    probabilities: FloatArray | None


class ArrayPredictor:
    """Predict from already-preprocessed ``(C, *spatial)`` arrays with a fold ensemble.

    Method names follow upstream's ``nnUNetPredictor``. Every model is checked with
    ``validate_inference_execution``, so architectures outside the supported envelope
    are rejected at construction. ``regions_class_order`` is ``None`` for class-based
    models; for region-based ones it lists the label written for each output channel.
    """

    def __init__(
        self,
        models: Sequence[PredictionModel],
        *,
        patch_size: Sequence[int],
        num_outputs: int,
        mirroring_axes: Sequence[int] = (),
        regions_class_order: Sequence[int] | None = None,
        step_fraction: float = 0.5,
        accumulation: str = "float32",
    ) -> None:
        """Bind the models, patch settings and dataset label semantics for prediction."""
        if regions_class_order is None and num_outputs < 2:
            raise ContractError("class inference requires at least two output channels")
        if not models:
            raise ContractError("predictor requires at least one fold model")
        for model in models:
            validate_inference_execution(model)
        self.models = tuple(models)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.num_outputs = num_outputs
        self.mirroring_axes = tuple(int(axis) for axis in mirroring_axes)
        self.regions_class_order = (
            None
            if regions_class_order is None
            else tuple(int(label) for label in regions_class_order)
        )
        self.step_fraction = step_fraction
        self.accumulation = accumulation

    def predict_logits_from_preprocessed_data(self, data_channels_first: HostArray) -> FloatArray:
        """Return fold-averaged host logits ``(num_outputs, *spatial)``."""
        return average_fold_logits(
            self.models,
            data_channels_first,
            patch_size=self.patch_size,
            num_outputs=self.num_outputs,
            mirroring_axes=self.mirroring_axes,
            step_fraction=self.step_fraction,
            accumulation=self.accumulation,
        )

    def predict_device_logits_from_preprocessed_data(self, data_channels_first: HostArray) -> array:
        """Return fold-averaged logits left on Metal, channels-last ``(*spatial, C)`` (3-D only)."""
        return average_fold_device_logits(
            self.models,
            data_channels_first,
            patch_size=self.patch_size,
            num_outputs=self.num_outputs,
            mirroring_axes=self.mirroring_axes,
            step_fraction=self.step_fraction,
            accumulation=self.accumulation,
        )

    def predict_single_npy_array(
        self,
        data_channels_first: HostArray,
        *,
        return_probabilities: bool = False,
    ) -> Prediction:
        """Predict one preprocessed case; probabilities are built only on request.

        With class models, requesting probabilities can change a near-tie label:
        the probability route takes argmax after float32 softmax, whereas the
        label-only route takes argmax directly on logits.
        """
        logits = self.predict_logits_from_preprocessed_data(data_channels_first)
        if return_probabilities:
            probabilities = logits_to_probabilities(
                logits, has_regions=self.regions_class_order is not None
            )
            segmentation = probabilities_to_segmentation(
                probabilities, regions_class_order=self.regions_class_order
            )
        else:
            probabilities = None
            segmentation = logits_to_segmentation(
                logits, regions_class_order=self.regions_class_order
            )
        return Prediction(
            segmentation=segmentation,
            probabilities=probabilities,
        )

    def predict_from_list_of_npy_arrays(
        self,
        arrays: Sequence[HostArray],
        *,
        return_probabilities: bool = False,
    ) -> list[Prediction]:
        """Predict each preprocessed array and optionally return probabilities."""
        return [
            self.predict_single_npy_array(array, return_probabilities=return_probabilities)
            for array in arrays
        ]

    def predict_from_data_iterator(
        self,
        iterator: Iterable[Mapping[str, object]],
        *,
        return_probabilities: bool = False,
    ) -> list[dict[str, object]]:
        """Predict the preprocessed data entries supplied by an iterator."""
        results: list[dict[str, object]] = []
        for raw_item in iterator:
            item = dict(raw_item)
            data = item.get("data")
            if not isinstance(data, np.ndarray):
                raise ContractError("prediction iterator items require NumPy 'data'")
            prediction = self.predict_single_npy_array(
                cast(HostArray, data),
                return_probabilities=return_probabilities,
            )
            results.append({**item, "prediction": prediction})
        return results
