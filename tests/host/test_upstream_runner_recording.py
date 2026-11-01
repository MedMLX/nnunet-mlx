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
