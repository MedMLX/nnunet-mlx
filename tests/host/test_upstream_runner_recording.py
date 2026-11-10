"""Torch-free comparisons against the pinned upstream CPU recording."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
from nnunet_reference.arrays import NumericArray
from nnunet_reference.fixture_types import UpstreamRecording

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction import inference
from nnunet_mlx.prediction.export import restore_prediction


@dataclass(frozen=True)
class RecordedFold:
    index: int

    def __call__(self, values: object) -> object:
        """Fail if a recorded-fold test unexpectedly executes a network."""
        raise AssertionError("recorded-fold lookup must replace network execution")


pytestmark = pytest.mark.host


def _original_probabilities(values: np.ndarray, has_regions: bool) -> np.ndarray:
    """Keep the pre-optimization float32 formulas as an independent regression reference."""
    if not has_regions:
        shifted = values - np.max(values, axis=0, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=0, keepdims=True)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


@pytest.mark.parametrize("regions", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "transpose", "strided"])
def test_head_conversion_preserves_input_and_memory_order_rounding(
    regions: bool, layout: str,
) -> None:
    """Preserve extreme logits and near ties across NumPy reduction layouts exactly."""
    values = np.random.default_rng(397).normal(size=(33, 3, 5, 9)).astype(np.float32)
    values[0] = np.nextafter(values[1], np.float32(np.inf))
    values[2, 0, 0, :4] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, 0, -0.0]
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "transpose":
        values = values.transpose(0, 2, 3, 1)
    elif layout == "strided":
        values = values[:, ::-1, ::2, ::-1]
    original = values.copy()
    with np.errstate(over="ignore"):
        expected = _original_probabilities(values, regions)
        actual = inference.logits_to_probabilities(values, has_regions=regions)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert actual.flags.c_contiguous and actual.dtype == np.float32


@pytest.mark.parametrize(("regions", "save"), [(False, True), (True, False), (True, True)])
def test_chunked_probability_restoration_preserves_crop_axes_and_caller_logits(
    tiny_model_plans: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    regions: bool, save: bool,
) -> None:
    """Keep final probabilities and labels exact, including singleton chunk tails."""
    from nnunet_mlx.prediction import export

    channels = 3 if regions else 33
    dataset: dict[str, object] = {
        "labels": ({"background": 0, "a": 1, "ab": [1, 2], "abc": [1, 2, 3]}
                   if regions else {"background" if index == 0 else f"class_{index}": index
                                    for index in range(channels)}),
        **({"regions_class_order": [1, 2, 3]} if regions else {}),
    }
    semantics = parse_dataset_label_semantics(dataset)
    plans = PlansContract.from_payload(tiny_model_plans, configuration="3d_fullres")
    values = np.random.default_rng(967).normal(size=(channels, 3, 5, 9)).astype(np.float32)
    values[0] = np.nextafter(values[1], np.float32(np.inf))
    original = values.copy()
    expected_crop = _original_probabilities(values, regions)
    if regions:
        expected_labels = np.zeros((3, 5, 9), dtype=np.uint16)
        for channel in range(channels):
            expected_labels[expected_crop[channel] > .5] = channel + 1
    else:
        expected_labels = np.argmax(expected_crop, axis=0).astype(np.uint16)
    shape = (7, 9, 13)
    crop = (slice(1, 4), slice(2, 7), slice(3, 12))
    expected_segmentation = np.zeros(shape, dtype=np.uint16)
    expected_segmentation[crop] = expected_labels
    expected_probabilities = np.zeros((channels, *shape), dtype=np.float32)
    if not regions:
        expected_probabilities[0] = 1
    expected_probabilities[(slice(None), *crop)] = expected_crop
    monkeypatch.setattr(export, "_PROBABILITY_CHUNK_BYTES", channels * 4 * 4)
    actual = restore_prediction(
        values,
        {"spacing": [1, 1, 1], "shape_before_cropping": shape,
         "shape_after_cropping_and_before_resampling": (3, 5, 9),
         "bbox_used_for_cropping": [[1, 4], [2, 7], [3, 12]]},
        plans, semantics, return_probabilities=save,
    )
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(
        actual.segmentation, expected_segmentation.transpose(plans.transpose_backward),
    )
    if save:
        assert actual.probabilities is not None
        np.testing.assert_array_equal(
            actual.probabilities,
            expected_probabilities.transpose((0, *(axis + 1 for axis in plans.transpose_backward))),
        )
        assert actual.probabilities.flags.c_contiguous
    else:
        assert actual.probabilities is None


def test_runner_preprocessing_matches_upstream_bitwise(
    upstream_runner_recording: UpstreamRecording,
) -> None:
    """Compare native preprocessing with the recorded upstream arrays exactly."""
    metadata, arrays = upstream_runner_recording
    plans = PlansContract.from_payload(metadata["runner"]["plans"], configuration="3d_fullres")
    actual = preprocess_raw_case(arrays["raw_image"], metadata["image_properties"], plans)
    # NoNormalization and unchanged spacing perform only deterministic copies/crops/transposes.
    np.testing.assert_array_equal(actual.data, arrays["preprocessed"])
    for name in (
        "shape_before_cropping",
        "bbox_used_for_cropping",
        "shape_after_cropping_and_before_resampling",
    ):
        assert (
            json.loads(json.dumps(actual.properties[name]))
            == (metadata["preprocessed_properties"][name])
        )


@pytest.mark.parametrize("head", ["class", "region"])
def test_runner_head_conversion_matches_upstream(
    upstream_runner_recording: UpstreamRecording,
    head: str,
) -> None:
    """Compare class and region head conversion with recorded upstream results."""
    _, arrays = upstream_runner_recording
    probabilities = inference.logits_to_probabilities(
        arrays[f"head_logits_{head}"],
        has_regions=head == "region",
    )
    # Observed maximum is 2 float32 ULP (5.96e-8): NumPy exp/divide versus Torch kernels.
    np.testing.assert_array_max_ulp(probabilities, arrays[f"head_probabilities_{head}"], maxulp=2)
    segmentation = inference.probabilities_to_segmentation(
        probabilities,
        regions_class_order=(1, 2) if head == "region" else None,
    )
    # Discrete argmax/region overwrite decisions have no tolerance.
    np.testing.assert_array_equal(segmentation, arrays[f"head_segmentation_{head}"])


@pytest.mark.parametrize("name", ["default", "no_mirroring", "fold_1"])
def test_runner_restoration_matches_upstream(
    upstream_runner_recording: UpstreamRecording,
    name: str,
) -> None:
    """Compare restored predictions and geometry with upstream fixtures."""
    metadata, arrays = upstream_runner_recording
    runner = metadata["runner"]
    plans = PlansContract.from_payload(runner["plans"], configuration="3d_fullres")
    actual = restore_prediction(
        arrays[f"logits_{name}"],
        metadata["preprocessed_properties"],
        plans,
        parse_dataset_label_semantics(runner["dataset_json"]),
        return_probabilities=True,
    )
    assert actual.probabilities is not None
    # Observed maximum is 2 float32 ULP (5.96e-8) from the inference nonlinearity.
    np.testing.assert_array_max_ulp(
        actual.probabilities,
        arrays[f"probabilities_{name}"],
        maxulp=2,
    )
    # NIfTI stores XYZ; the SimpleITK/raw-array boundary uses ZYX.
    np.testing.assert_array_equal(
        actual.segmentation.transpose(2, 1, 0),
        arrays[f"segmentation_{name}"],
    )


def test_fp16_fold_averaging_matches_upstream_bitwise(
    upstream_runner_recording: UpstreamRecording,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match two-fold FP16 averaging with the recorded upstream output."""
    _, arrays = upstream_runner_recording

    def recorded_fold(model: RecordedFold, *args: object, **kwargs: object) -> NumericArray:
        """Return recorded logits for the selected source fold."""
        return arrays[f"fold_logits_{model.index}"]

    monkeypatch.setattr(inference, "_predict_model_logits", recorded_fold)
    actual = inference.average_fold_logits(
        (RecordedFold(0), RecordedFold(1)),
        arrays["preprocessed"],
        patch_size=(8, 8, 8),
        num_outputs=2,
        accumulation="nnunet-fp16",
    )
    # IEEE half addition/division round identically to the upstream half tensors.
    np.testing.assert_array_equal(actual, arrays["logits_default"])


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_fp16_fold_rounding_matches_recorded_upstream_operations(
    upstream_runner_recording: UpstreamRecording,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
) -> None:
    """Match recorded FP16 rounding across multiple fold counts."""
    _, arrays = upstream_runner_recording

    def recorded_fold(model: RecordedFold, *args: object, **kwargs: object) -> NumericArray:
        """Return the seeded rounding input for one source fold."""
        return arrays[f"rounding_input_{model.index}"]

    monkeypatch.setattr(inference, "_predict_model_logits", recorded_fold)
    actual = inference.average_fold_logits(
        tuple(RecordedFold(index) for index in range(count)),
        arrays["preprocessed"],
        patch_size=(8, 8, 8),
        num_outputs=2,
        accumulation="nnunet-fp16",
    )
    np.testing.assert_array_equal(actual, arrays[f"rounding_output_{count}"])
