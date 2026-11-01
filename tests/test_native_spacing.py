"""Native-spacing preprocessing and restoration compared with pinned upstream."""

from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from nnunet_mlx.data.labels import parse_dataset_label_semantics
from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.plans import PlansContract
from nnunet_mlx.prediction.export import (
    export_prediction,
    restore_device_segmentation,
    restore_prediction,
)

# Spacings and expected shapes are explicit in the transposed [2, 0, 1] grid.
# These include fixed-spacing controls for the existing numerical contract.
SPACING_CASES = [
    pytest.param((None, None, None), (5.0, 0.7, 1.6), (7, 4, 6), id="3d-native"),
    pytest.param((None, 1.4, 0.8), (5.0, 1.4, 0.8), (7, 2, 12), id="3d-partial"),
    pytest.param((2.5, 1.4, 0.8), (2.5, 1.4, 0.8), (14, 2, 12), id="3d-fixed"),
    pytest.param((None, None), (5.0, 0.7, 1.6), (7, 4, 6), id="2d-native"),
    pytest.param((None, 0.8), (5.0, 0.7, 0.8), (7, 4, 12), id="2d-partial"),
    pytest.param((1.4, 0.8), (5.0, 1.4, 0.8), (7, 2, 12), id="2d-fixed"),
]


def _payload(
    original: dict[str, object], spacing: tuple[float | None, ...],
    *, separate_z: bool | None = None,
) -> dict[str, object]:
    """Build plans metadata containing native-spacing markers."""
    payload = copy.deepcopy(original)
    configurations = cast(dict[str, object], payload["configurations"])
    base = cast(dict[str, object], configurations["base"])
    base["spacing"] = list(spacing)
    base["patch_size"] = [8] * len(spacing)
    for kind in ("data", "seg", "probabilities"):
        kwargs = cast(dict[str, object], base[f"resampling_fn_{kind}_kwargs"])
        kwargs["force_separate_z"] = separate_z
    # A dataset-median spacing must never replace an individual image's spacing.
    payload["original_median_spacing_after_transp"] = [99.0, 98.0, 97.0]
    return payload


def _case(
    spacing: tuple[float, float, float] = (0.7, 1.6, 5.0),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int16], dict[str, object]]:
    """Create a synthetic case with a selected native input spacing."""
    data = np.zeros((1, 6, 8, 10), dtype=np.float32)
    data[0, 1:5, 1:7, 2:9] = 1 + np.arange(168, dtype=np.float32).reshape(4, 6, 7)
    previous = np.zeros(data.shape, dtype=np.int16)
    previous[0, 1:5, 1:7, 2:9] = (np.indices((4, 6, 7)).sum(axis=0) % 2).astype(np.int16)
    properties: dict[str, object] = {
        "spacing": spacing,
        "sitk_stuff": {
            "spacing": spacing[::-1], "origin": (10.0, 20.0, 30.0),
            "direction": (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
    }
    return data, previous, properties


@pytest.mark.oracle
@pytest.mark.parametrize("configured,resolved,processed_shape", SPACING_CASES)
@pytest.mark.parametrize("save_probabilities", [False, True])
def test_native_spacing_matches_resolved_upstream_preprocessing_and_export(
    tmp_path: Path, tiny_model_plans: dict[str, object], tiny_model_dataset: dict[str, object],
    configured: tuple[float | None, ...], resolved: tuple[float, float, float],
    processed_shape: tuple[int, int, int], save_probabilities: bool,
) -> None:
    # The installed v2.8.1 oracle requires numeric spacing. Supplying independently
    # specified resolved values compares interpolation while testing our null resolution.
    """Match source preprocessing and export after resolving native spacing."""
    source_payload = _payload(tiny_model_plans, resolved[-len(configured):])
    source_plans = import_module("nnunetv2.utilities.plans_handling.plans_handler").PlansManager(
        source_payload,
    )
    source_configuration = source_plans.get_configuration("3d_fullres")
    preprocessor = import_module("nnunetv2.preprocessing.preprocessors.default_preprocessor")
    data, previous, properties = _case()
    expected_data, expected_previous, expected_properties = (
        preprocessor.DefaultPreprocessor(verbose=False).run_case_npy(
            data, previous, copy.deepcopy(properties), source_plans,
            source_configuration, tiny_model_dataset,
        )
    )
    plans = PlansContract.from_payload(
        _payload(tiny_model_plans, configured), configuration="3d_fullres",
    )
    actual = preprocess_raw_case(data, properties, plans, segmentation_previous_stage=previous)
    assert actual.data.shape == (1, *processed_shape)
    np.testing.assert_allclose(actual.data, expected_data, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual.previous_stage, expected_previous)
    for key in ("shape_before_cropping", "shape_after_cropping_and_before_resampling",
                "bbox_used_for_cropping"):
        assert actual.properties[key] == expected_properties[key]

    logits = np.random.default_rng(3076).normal(size=(2, *processed_shape)).astype(np.float32)
    source_export = import_module("nnunetv2.inference.export_prediction")
    expected = source_export.convert_predicted_logits_to_segmentation_with_correct_shape(
        logits.copy(), source_plans, source_configuration,
        source_plans.get_label_manager(tiny_model_dataset), expected_properties,
        return_probabilities=save_probabilities, num_threads_torch=1,
    )
    restored = restore_prediction(
        logits, actual.properties, plans, parse_dataset_label_semantics(tiny_model_dataset),
        return_probabilities=save_probabilities,
    )
    np.testing.assert_array_equal(
        restored.segmentation, expected[0] if save_probabilities else expected,
    )
    if save_probabilities:
        assert restored.probabilities is not None
        np.testing.assert_allclose(restored.probabilities, expected[1], rtol=1e-6, atol=1e-6)
    files = export_prediction(
        restored, actual.properties, tmp_path / "native", file_ending=".nii.gz",
        image_reader_writer="SimpleITKIO", overwrite=False,
    )
    image = import_module("SimpleITK").ReadImage(str(files.segmentation))
    assert image.GetSize() == (10, 8, 6)
    np.testing.assert_allclose(image.GetSpacing(), (5.0, 1.6, 0.7), rtol=0, atol=1e-6)
    assert image.GetOrigin() == (10.0, 20.0, 30.0)
    assert image.GetDirection() == (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@pytest.mark.metal
@pytest.mark.oracle
@pytest.mark.parametrize("configured,resolved,processed_shape", SPACING_CASES[:2])
def test_native_spacing_metal_restoration_matches_upstream(
    tiny_model_plans: dict[str, object], tiny_model_dataset: dict[str, object],
    configured: tuple[float | None, ...], resolved: tuple[float, float, float],
    processed_shape: tuple[int, int, int],
) -> None:
    """Compare native-spacing Metal restoration with the source result."""
    from nnunet_mlx.runtime import import_mlx

    _, _, properties = _case()
    properties.update({
        "shape_before_cropping": (10, 6, 8),
        "shape_after_cropping_and_before_resampling": (7, 4, 6),
        "bbox_used_for_cropping": [[2, 9], [1, 5], [1, 7]],
    })
    plans = PlansContract.from_payload(
        _payload(tiny_model_plans, configured, separate_z=False), configuration="3d_fullres",
    )
    source_plans = import_module("nnunetv2.utilities.plans_handling.plans_handler").PlansManager(
        _payload(tiny_model_plans, resolved, separate_z=False),
    )
    logits = np.random.default_rng(3076).normal(size=(2, *processed_shape)).astype(np.float32)
    source_export = import_module("nnunetv2.inference.export_prediction")
    expected = source_export.convert_predicted_logits_to_segmentation_with_correct_shape(
        logits.copy(), source_plans, source_plans.get_configuration("3d_fullres"),
        source_plans.get_label_manager(tiny_model_dataset), properties, num_threads_torch=1,
    )
    mx, _ = import_mlx()
    actual = restore_device_segmentation(
        mx.array(np.moveaxis(logits, 0, -1)), properties, plans,
        parse_dataset_label_semantics(tiny_model_dataset),
    )
    assert actual is not None
    np.testing.assert_array_equal(actual.segmentation, expected)
