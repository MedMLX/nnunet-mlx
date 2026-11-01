"""Regression evidence recorded directly from nnU-Net's selected 17cb295 source."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.data.resampling import resample_probabilities_or_data, resample_segmentation
from nnunet_mlx.plans import PlansContract, ResamplingContract
from nnunet_mlx.prediction.export import restore_device_segmentation, restore_prediction
from nnunet_mlx.prediction.restore import resample_class_logits_argmax_parallel

type NumericArray = npt.NDArray[np.float32 | np.int8 | np.int16 | np.int64 | np.uint8 | np.uint16]

RECORDING = Path(__file__).parent / "fixtures/darwin-arm64/torch_resampling.npz"


class ResamplingCase(TypedDict):
    name: str
    shape: list[int]
    spacing: list[float]
    new_spacing: list[float]
    kwargs: dict[str, object]


class PipelineCase(TypedDict):
    name: str
    plans: dict[str, object]
    dataset: dict[str, object]
    properties: dict[str, object]
    geometry: dict[str, object]
    regions: bool


with np.load(RECORDING, allow_pickle=False) as _saved:
    _metadata = json.loads(str(_saved["metadata"].item()))
    RESAMPLING = cast(list[ResamplingCase], _metadata["cases"])
    PIPELINE = cast(list[PipelineCase], _metadata["pipeline"])


@pytest.fixture(scope="module")
def recording() -> dict[str, NumericArray]:
    """Load the recorded source resampling fixture."""
    with np.load(RECORDING, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files if name != "metadata"}


def _contract(options: dict[str, object]) -> ResamplingContract:
    """Build a source-style resampling contract for the selected mode."""
    return ResamplingContract.from_configuration(
        {"fn": "resample_torch_fornnunet", "fn_kwargs": options},
        field="fn",
        expected_is_seg=options.get("is_seg", False) is True,
    )


@pytest.mark.parametrize("case", RESAMPLING, ids=[row["name"] for row in RESAMPLING])
def test_torch_resampling_matches_selected_source(
    recording: dict[str, NumericArray],
    case: ResamplingCase,
) -> None:
    """Compare resampling with the selected source interpolation mode."""
    values = recording[f"{case['name']}-input"]
    expected = recording[f"{case['name']}-expected"]
    is_seg = case["kwargs"]["is_seg"] is True
    function = resample_segmentation if is_seg else resample_probabilities_or_data
    actual = function(
        values, case["shape"], case["spacing"], case["new_spacing"], _contract(case["kwargs"])
    )
    if is_seg:
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == values.dtype
    else:
        # FP32 sampling coordinates, with SciPy rather than ATen accumulation.
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32), expected, rtol=1e-5, atol=5e-6
        )
        assert actual.dtype == np.float32
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("case", PIPELINE, ids=[row["name"] for row in PIPELINE])
def test_torch_plan_preprocessing_cascade_and_restoration(
    recording: dict[str, NumericArray],
    case: PipelineCase,
) -> None:
    """Check source-style preprocessing, cascade labels and restoration together."""
    original = copy.deepcopy(case["plans"])
    plans = PlansContract.from_payload(case["plans"], configuration="selected")
    prepared = preprocess_raw_case(
        recording["raw"],
        case["properties"],
        plans,
        segmentation_previous_stage=recording["previous"],
    )
    key = case["name"]
    np.testing.assert_allclose(prepared.data, recording[f"{key}-processed"], rtol=1e-5, atol=5e-6)
    np.testing.assert_array_equal(prepared.previous_stage, recording[f"{key}-labels"])
    for field, value in case["geometry"].items():
        np.testing.assert_array_equal(prepared.properties[field], value)
    for save in (False, True):
        restored = restore_prediction(
            recording[f"{key}-logits"],
            prepared.properties,
            plans,
            parse_dataset_label_semantics(case["dataset"]),
            return_probabilities=save,
        )
        np.testing.assert_array_equal(
            restored.segmentation,
            recording[f"{key}-segmentation" if save else f"{key}-label-only"],
        )
        if save:
            assert restored.probabilities is not None
            np.testing.assert_allclose(
                restored.probabilities, recording[f"{key}-probabilities"], rtol=1e-5, atol=5e-6
            )
    assert case["plans"] == original


@pytest.mark.metal
@pytest.mark.parametrize("case", [PIPELINE[i] for i in (0, 2, 6, 8)])
def test_torch_plan_metal_label_restoration_boundary(
    recording: dict[str, NumericArray],
    case: PipelineCase,
) -> None:
    """Check the qualified Metal label-restoration boundary for source-style plans."""
    from nnunet_mlx.runtime import import_mlx

    plans = PlansContract.from_payload(case["plans"], configuration="selected")
    properties = {**case["properties"], **case["geometry"]}
    semantics = parse_dataset_label_semantics(case["dataset"])
    logits = recording[f"{case['name']}-logits"]
    mx, _ = import_mlx()
    device_logits = mx.array(np.moveaxis(logits, 0, -1))
    restored = restore_device_segmentation(device_logits, properties, plans, semantics)
    if list(logits.shape[1:]) == case["geometry"]["shape_after_cropping_and_before_resampling"]:
        assert restored is not None  # Native shape uses the existing Metal argmax.
    else:
        assert restored is None  # The fused spline grid cannot implement the Torch grid.
        restored = restore_prediction(
            np.moveaxis(np.asarray(device_logits, dtype=np.float32), -1, 0),
            properties,
            plans,
            semantics,
            return_probabilities=False,
        )
    np.testing.assert_array_equal(restored.segmentation, recording[f"{case['name']}-label-only"])


def test_torch_parallel_restoration_does_not_use_spline_plane_shortcut(
    recording: dict[str, NumericArray],
) -> None:
    """Avoid the spline plane shortcut for source-style parallel restoration."""
    case = next(row for row in RESAMPLING if row["name"] == "axis-1-nearest-data-nearest")
    actual = resample_class_logits_argmax_parallel(
        recording[f"{case['name']}-input"],
        case["shape"],
        case["spacing"],
        case["new_spacing"],
        _contract(case["kwargs"]),
    )
    np.testing.assert_array_equal(actual, recording[f"{case['name']}-expected"].argmax(axis=0))
