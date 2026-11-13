"""Label-only Metal reduction must match the independent host-logit boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from nnunet_reference.oracle.network import real_image_input

from nnunet_mlx.prediction.sliding_window import predict_labels, predict_logits

if TYPE_CHECKING:
    from mlx.core import array


@pytest.mark.parametrize(
    ("accumulation", "shape"), [("float32", (2, 3, 4)), ("nnunet-fp16", (5, 6, 7))],
)
def test_label_only_matches_normalized_host_logits(
    accumulation: str, shape: tuple[int, ...],
) -> None:
    """Match label-only inference with labels from normalized host logits."""
    from nnunet_mlx.runtime import import_mlx

    mx, _ = import_mlx()

    def model(values: array) -> array:
        """Return synthetic logits for label-only comparison."""
        return mx.concatenate((values, -values, mx.zeros(values.shape) + 1e-5), axis=-1)

    data = real_image_input(shape, 1)
    expected = predict_logits(
        model, data, patch_size=(4, 4, 4), num_classes=3,
        step_fraction=0.5, accumulation=accumulation,
    ).argmax(axis=0).astype(np.uint16)
    actual = predict_labels(
        model, data, patch_size=(4, 4, 4), num_classes=3,
        step_fraction=0.5, accumulation=accumulation,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.host
@pytest.mark.parametrize("restriction", ["ensemble", "resample", "regions", "2d"])
def test_streamed_labels_decline_cases_requiring_score_restoration(
    tiny_model_plans: dict[str, object], restriction: str,
) -> None:
    """Keep fold averaging, region thresholds and spatial restoration ahead of labels."""
    from nnunet_mlx.data.labels import parse_dataset_label_semantics
    from nnunet_mlx.plans import PlansContract
    from nnunet_mlx.prediction.export import predict_same_grid_segmentation
    from nnunet_mlx.prediction.inference import ArrayPredictor

    def model(values: array) -> array:
        raise AssertionError("ineligible streamed-label requests must not execute a fold")

    regions = restriction == "regions"
    semantics = parse_dataset_label_semantics({
        "labels": {"background": 0, "a": 1, "both": [1, 2] if regions else 2},
        **({"regions_class_order": [1, 2]} if regions else {}),
    })
    runner = ArrayPredictor(
        (model, model) if restriction == "ensemble" else (model,),
        patch_size=(4, 4) if restriction == "2d" else (4, 4, 4),
        num_outputs=semantics.output_channels, regions_class_order=semantics.regions_class_order,
    )
    crop_shape = [6, 4, 5] if restriction == "resample" else [3, 4, 5]
    result = predict_same_grid_segmentation(
        runner, real_image_input((3, 4, 5), 1),
        {
            "shape_after_cropping_and_before_resampling": crop_shape,
            "shape_before_cropping": crop_shape,
            "bbox_used_for_cropping": [[0, size] for size in crop_shape],
        },
        PlansContract.from_payload(tiny_model_plans, configuration="3d_fullres"), semantics,
    )
    assert result is None
