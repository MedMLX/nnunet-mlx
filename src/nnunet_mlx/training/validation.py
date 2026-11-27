"""Held-out case validation with the production sliding-window predictor."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import numpy as np
import numpy.typing as npt
from mlx import core as mx

from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.network import Network
from nnunet_mlx.prediction.sliding_window import predict_device_logits, predict_labels
from nnunet_mlx.runtime import import_mlx
from nnunet_mlx.training.data import PreparedDataset

_mx, _ = import_mlx()


@dataclass(frozen=True, slots=True)
class ValidationScore:
    """Foreground Dice pooled over the validation cases in preprocessed space."""

    dice: float | None
    per_output: tuple[float | None, ...]


def validate_cases(model: Network, data: PreparedDataset) -> ValidationScore:
    """Evaluate whole held-out cases without mirroring and exclude ignored voxels."""
    semantics = data.semantics
    patch = data.plans.configuration.patch_size
    regions = (
        semantics.regions
        if semantics.has_regions
        else tuple((label,) for label in semantics.all_labels[1:])
    )
    intersections = [0] * len(regions)
    totals = [0] * len(regions)
    model.eval()
    try:
        for identifier in data.split.val:
            image, labels = data.arrays(identifier)
            items = ((image, labels),) if len(patch) == 3 else zip(image, labels, strict=True)
            for values, target in items:
                channels_first = np.moveaxis(values, -1, 0)
                prediction: npt.NDArray[np.uint16 | np.bool_ | np.int32]
                if not semantics.has_regions and len(patch) == 3:
                    prediction = predict_labels(
                        model,
                        channels_first,
                        patch_size=patch,
                        num_classes=semantics.output_channels,
                    )
                else:
                    logits = predict_device_logits(
                        model,
                        channels_first,
                        patch_size=patch,
                        num_classes=semantics.output_channels,
                    )
                    finite = mx.all(mx.isfinite(logits))
                    decoded = logits > 0 if semantics.has_regions else mx.argmax(logits, axis=-1)
                    _mx.eval(decoded, finite)
                    if not bool(finite.item()):
                        raise ContractError("validation produced non-finite logits")
                    prediction = np.asarray(
                        decoded, dtype=np.bool_ if semantics.has_regions else np.int32
                    )
                valid = (
                    target != semantics.ignore_label if semantics.ignore_label is not None else True
                )
                for index, region in enumerate(regions):
                    label, *other_labels = region
                    truth = np.isin(target, region) if other_labels else target == label
                    positive = (
                        prediction[..., index] if semantics.has_regions else prediction == label
                    )
                    positive = positive & valid
                    intersections[index] += int(np.count_nonzero(positive & truth))
                    totals[index] += int(np.count_nonzero(positive)) + int(np.count_nonzero(truth))
    finally:
        model.train()
    scores = tuple(
        2 * intersection / total if total else None
        for intersection, total in zip(intersections, totals, strict=True)
    )
    measured = [score for score in scores if score is not None]
    return ValidationScore(fmean(measured) if measured else None, scores)
