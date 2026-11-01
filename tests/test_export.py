"""Restoration of logits to the original grid compared with the pinned upstream export."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.export import restore_prediction


@pytest.mark.oracle
def test_five_class_restoration_matches_source_resampling_crop_and_axes(
    tiny_model_plans: dict[str, object],
) -> None:
    """Compare five-class restoration with source resampling and geometry."""
    from functools import partial
    from types import SimpleNamespace

    export = pytest.importorskip("nnunetv2.inference.export_prediction")
    source_resampling = pytest.importorskip("nnunetv2.preprocessing.resampling.default_resampling")
    labels = pytest.importorskip("nnunetv2.utilities.label_handling.label_handling")
    configurations = cast(dict[str, object], tiny_model_plans["configurations"])
    assert isinstance(configurations, dict)
    base = configurations["base"]
    assert isinstance(base, dict)
    base["spacing"] = [2.0, 1.0, 1.5]
    base["resampling_fn_probabilities_kwargs"] = {
        "is_seg": False,
        "order": 1,
        "order_z": 0,
        "force_separate_z": False,
    }
    dataset = {"labels": {"background": 0, "a": 1, "b": 2, "c": 3, "d": 4}}
    properties = {
        "spacing": [1.0, 1.5, 1.0],
        "shape_after_cropping_and_before_resampling": [8, 6, 10],
        "shape_before_cropping": [12, 10, 14],
        "bbox_used_for_cropping": [[2, 10], [1, 7], [3, 13]],
    }
    logits = np.random.default_rng(1729).normal(size=(5, 4, 6, 10)).astype(np.float32)
    expected = export.convert_predicted_logits_to_segmentation_with_correct_shape(
        logits.copy(),
        SimpleNamespace(transpose_forward=[2, 0, 1], transpose_backward=[1, 2, 0]),
        SimpleNamespace(
            spacing=[2.0, 1.0, 1.5],
            resampling_fn_probabilities=partial(
                source_resampling.resample_data_or_seg_to_shape,
                is_seg=False,
                order=1,
                order_z=0,
                force_separate_z=False,
            ),
        ),
        labels.LabelManager(dataset["labels"], regions_class_order=None),
        properties,
        num_threads_torch=1,
    )
    actual = restore_prediction(
        logits,
        properties,
        PlansContract.from_payload(tiny_model_plans, configuration="3d_fullres"),
        parse_dataset_label_semantics(dataset),
        return_probabilities=False,
    ).segmentation
    assert actual.shape == (10, 14, 12)
    np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(actual)) == {0, 1, 2, 3, 4}
