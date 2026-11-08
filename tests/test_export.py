"""Restoration of logits to the original grid compared with the pinned upstream export."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.export import restore_prediction

if TYPE_CHECKING:
    from mlx.core import array
    from torch import Tensor

FOLD_SCALES = (1.0, 0.7)


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


class _ScaledModel:
    def __init__(self, scale: float) -> None:
        """Bind the fold's float32 scale."""
        self.scale = scale

    def __call__(self, values: array) -> array:
        """Return the scaled tile, so the logits are the preprocessed channels."""
        return values * self.scale


@pytest.mark.oracle
@pytest.mark.metal
@pytest.mark.parametrize("save_probabilities", [False, True])
@pytest.mark.parametrize(
    "spacing", [(1.5, 0.8, 0.7), (4.0, 0.8, 0.7)], ids=["joint", "separate-z"]
)
def test_fp16_prediction_restores_like_the_pinned_predictor(
    spacing: tuple[float, float, float], save_probabilities: bool,
) -> None:
    """Average two folds in float16 and restore upstream's labels on host and Metal.

    Upstream's predictor returns half logits and its spline resampling rounds the
    interpolated values back to float16, so near-equal logits tie and the first
    class wins. Small logits make such ties frequent.
    """
    torch = pytest.importorskip("torch")
    prediction = pytest.importorskip("nnunetv2.inference.predict_from_raw_data")
    export = pytest.importorskip("nnunetv2.inference.export_prediction")
    plans_handler = pytest.importorskip("nnunetv2.utilities.plans_handling.plans_handler")
    from nnunet_reference.oracle.fixtures import raw_inference_fixture

    from nnunet_mlx.prediction.export import restore_device_segmentation
    from nnunet_mlx.prediction.inference import (
        average_fold_device_logits,
        average_fold_logits,
        device_logits_to_host,
    )

    plans, dataset = raw_inference_fixture()
    dataset["labels"] = {"background": 0, "a": 1, "b": 2}
    properties: dict[str, object] = {
        "spacing": spacing,
        "shape_before_cropping": (30, 34, 38),
        "bbox_used_for_cropping": [[2, 26], [3, 31], [4, 36]],
        "shape_after_cropping_and_before_resampling": (24, 28, 32),
    }
    # The fixture plans transpose by (2, 0, 1) and use 1 mm spacing.
    preprocessed_shape = tuple(
        round(size * spacing[axis])
        for size, axis in zip((24, 28, 32), (2, 0, 1), strict=True)
    )
    values = np.random.default_rng(5).normal(
        scale=0.02, size=(3, *preprocessed_shape)
    ).astype(np.float32)

    manager = plans_handler.PlansManager(plans)
    configuration = manager.get_configuration("3d_fullres")
    labels = manager.get_label_manager(dataset)
    predictor = prediction.nnUNetPredictor(
        device=torch.device("cpu"), perform_everything_on_device=False,
        use_mirroring=False, allow_tqdm=False,
    )
    class TorchScaled(torch.nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            """Return the tile scaled by the fold's only parameter."""
            return values * self.scale

    network = TorchScaled()
    network.scale = torch.nn.Parameter(torch.ones(()))
    predictor.network = network
    predictor.list_of_parameters = [
        {"scale": torch.tensor(scale)} for scale in FOLD_SCALES
    ]
    predictor.configuration_manager = configuration
    predictor.label_manager = labels
    upstream_logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(values))
    assert upstream_logits.dtype == torch.float16
    expected = export.convert_predicted_logits_to_segmentation_with_correct_shape(
        upstream_logits, manager, configuration, labels, dict(properties),
        return_probabilities=save_probabilities, num_threads_torch=1,
    )

    candidate_plans = PlansContract.from_payload(plans, configuration="3d_fullres")
    semantics = parse_dataset_label_semantics(dataset)
    models = tuple(_ScaledModel(scale) for scale in FOLD_SCALES)
    host = restore_prediction(
        average_fold_logits(
            models, values, patch_size=(8, 8, 8), num_outputs=3, accumulation="nnunet-fp16",
        ),
        properties, candidate_plans, semantics, return_probabilities=save_probabilities,
    )
    if save_probabilities:
        expected_segmentation, expected_probabilities = expected
        assert host.probabilities is not None
        np.testing.assert_allclose(host.probabilities, expected_probabilities, rtol=0, atol=1e-6)
    else:
        expected_segmentation = expected
        # Mirror the pipeline: fused restoration on Metal, host restoration if it declines.
        device_logits = average_fold_device_logits(
            models, values, patch_size=(8, 8, 8), num_outputs=3, accumulation="nnunet-fp16",
        )
        device = restore_device_segmentation(
            device_logits, properties, candidate_plans, semantics
        ) or restore_prediction(
            device_logits_to_host(device_logits), properties, candidate_plans, semantics,
            return_probabilities=False,
        )
        np.testing.assert_array_equal(device.segmentation, expected_segmentation)
    np.testing.assert_array_equal(host.segmentation, expected_segmentation)
