"""Parity tests against the pinned upstream nnU-Net oracle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from nnunet_reference.oracle.fixtures import plan_configuration_fixture
from nnunet_reference.oracle.inference import (
    run_configuration_ensemble_parity,
    run_fold_ensemble_parity,
    run_head_conversion_parity,
    run_prediction_restoration_parity,
    run_sliding_window_parity,
)
from nnunet_reference.oracle.network import (
    run_forward_parity,
    run_residual_forward_parity,
)
from nnunet_reference.oracle.postprocessing import (
    run_postprocessing_pkl_parity,
    run_remove_largest_component_parity,
)
from nnunet_reference.oracle.preprocessing import (
    run_plan_configuration_parity,
    run_raw_preprocessing_parity,
)
from nnunet_reference.oracle.resampling import (
    run_resampling_parity,
    run_segmentation_resampling_parity,
)

from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec

pytestmark = pytest.mark.oracle


def test_plan_configuration_matrix_matches_pinned_nnunet_oracle() -> None:
    """Compare plans configuration parsing against the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.utilities.plans_handling.plans_handler")
    plans, configurations = plan_configuration_fixture()
    assert run_plan_configuration_parity(plans, configurations).passed


@pytest.mark.parametrize(
    ("normalization_scheme", "use_mask_for_norm"),
    [
        ("NoNormalization", False),
        ("ZScoreNormalization", False),
        ("ZScoreNormalization", True),
        ("CTNormalization", False),
        ("RescaleTo01Normalization", False),
        ("RGBTo01Normalization", False),
    ],
)
def test_raw_preprocessing_matches_pinned_nnunet_oracle(
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
    normalization_scheme: str,
    use_mask_for_norm: bool,
) -> None:
    """Compare raw preprocessing with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.preprocessing.preprocessors.default_preprocessor")
    assert run_raw_preprocessing_parity(
        tiny_model_plans,
        tiny_model_dataset,
        normalization_scheme=normalization_scheme,
        use_mask_for_norm=use_mask_for_norm,
    ).passed


def test_raw_cascade_preprocessing_matches_pinned_nnunet_oracle(
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> None:
    """Compare cascade preprocessing with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.preprocessing.preprocessors.default_preprocessor")
    assert run_raw_preprocessing_parity(
        tiny_model_plans,
        tiny_model_dataset,
        include_previous_stage=True,
    ).passed


def test_prediction_restoration_matches_pinned_nnunet_oracle(
    tiny_model_plans: dict[str, object],
    tiny_model_dataset: dict[str, object],
) -> None:
    """Compare prediction restoration with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.inference.export_prediction")
    assert run_prediction_restoration_parity(tiny_model_plans, tiny_model_dataset).passed


@pytest.mark.parametrize(
    ("spec_fixture", "spatial", "patch", "mirror_axes"),
    [
        ("tiny_spec_2d", (13, 21), (8, 8), (0, 1)),
        ("tiny_spec", (6, 9, 11), (8, 8, 8), (0, 1, 2)),
    ],
)
def test_fold_ensemble_matches_pinned_nnunet_oracle(
    spec_fixture: str,
    spatial: tuple[int, ...],
    patch: tuple[int, ...],
    mirror_axes: tuple[int, ...],
    request: pytest.FixtureRequest,
) -> None:
    """Compare fold ensembling with the pinned nnU-Net oracle."""
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2.inference.sliding_window_prediction")
    spec = request.getfixturevalue(spec_fixture)
    assert isinstance(spec, PlainConvUNetSpec)
    assert run_fold_ensemble_parity(
        spec,
        spatial=spatial,
        patch=patch,
        mirror_axes=mirror_axes,
    ).passed


def test_configuration_ensemble_matches_pinned_nnunet_oracle(
    tiny_two_class_spec: PlainConvUNetSpec,
    tmp_path: Path,
) -> None:
    """Compare configuration ensembling with the pinned nnU-Net oracle."""
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2.ensembling.ensemble")
    assert run_configuration_ensemble_parity(
        tiny_two_class_spec,
        work_dir=tmp_path,
        spatial=(6, 9, 11),
    ).passed


@pytest.mark.parametrize(
    "fixture_name",
    [
        "tiny_spec",
        "tiny_spec_2d",
        "tiny_residual_spec_2d",
        "tiny_residual_spec",
        "tiny_bottleneck_spec_2d",
        "tiny_bottleneck_spec",
    ],
)
def test_network_forward_matches_pinned_torch_oracle(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """Compare plain and residual 2D/3D network output with the pinned Torch oracle."""
    pytest.importorskip("torch")
    pytest.importorskip("dynamic_network_architectures.architectures.unet")
    spec = request.getfixturevalue(fixture_name)
    if isinstance(spec, PlainConvUNetSpec):
        result = run_forward_parity(spec)
    else:
        assert isinstance(spec, ResidualEncoderUNetSpec)
        result = run_residual_forward_parity(spec)
    assert result.metrics.passed
    assert result.metrics.argmax_agreement == 1.0


def test_split_decoder_convolution_matches_pinned_torch_oracle(
    tiny_spec: PlainConvUNetSpec,
) -> None:
    """Compare a 32-feature first stage, whose decoder convolution runs split per part."""
    pytest.importorskip("torch")
    pytest.importorskip("dynamic_network_architectures.architectures.unet")
    spec = replace(tiny_spec, features=(32, 64))
    result = run_forward_parity(spec, spatial_shape=(32, 32, 32))
    assert result.metrics.passed
    assert result.metrics.argmax_agreement == 1.0


def test_head_conversion_matches_pinned_nnunet_oracle() -> None:
    """Compare class and region head conversion with the pinned nnU-Net oracle."""
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2.utilities.label_handling.label_handling")
    assert run_head_conversion_parity().passed


@pytest.mark.parametrize(
    ("spec_fixture", "spatial", "patch", "mirror_axes"),
    [
        ("tiny_spec_2d", (65, 97), (32, 32), (0, 1)),
        ("tiny_spec_2d", (33, 129), (32, 32), ()),
        ("tiny_spec", (17, 33, 49), (16, 32, 32), (0, 1, 2)),
    ],
)
def test_sliding_window_matches_pinned_torch_oracle(
    spec_fixture: str,
    spatial: tuple[int, ...],
    patch: tuple[int, ...],
    mirror_axes: tuple[int, ...],
    request: pytest.FixtureRequest,
) -> None:
    """Compare 2D and 3D sliding-window inference with the pinned Torch oracle."""
    pytest.importorskip("torch")
    pytest.importorskip("nnunetv2.inference.sliding_window_prediction")
    spec = request.getfixturevalue(spec_fixture)
    assert isinstance(spec, PlainConvUNetSpec)
    result = run_sliding_window_parity(
        spec, spatial=spatial, patch=patch, mirror_axes=mirror_axes,
    )
    assert result.metrics.passed
    assert result.metrics.argmax_agreement == 1.0


@pytest.mark.parametrize("anisotropic", [False, True])
def test_floating_resampling_matches_pinned_nnunet_oracle(
    anisotropic: bool,
) -> None:
    """Compare floating-point resampling with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.preprocessing.resampling.default_resampling")
    assert run_resampling_parity(anisotropic=anisotropic).passed


@pytest.mark.parametrize(
    ("anisotropic", "order_z"), [(False, 0), (True, 0), (True, 1)]
)
def test_segmentation_resampling_matches_pinned_nnunet_oracle(
    anisotropic: bool, order_z: int
) -> None:
    """Compare segmentation resampling with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.preprocessing.resampling.default_resampling")
    assert run_segmentation_resampling_parity(
        anisotropic=anisotropic, order_z=order_z
    ).passed


@pytest.mark.parametrize(
    "labels_or_regions", [1, [1, 2, 3], (1, 2), [(1, 2), 3]]
)
def test_remove_largest_component_matches_pinned_nnunet_oracle(
    labels_or_regions: int | tuple[int, ...] | list[int | tuple[int, ...]],
) -> None:
    """Compare largest-component filtering with the pinned nnU-Net oracle."""
    pytest.importorskip("nnunetv2.postprocessing.remove_connected_components")
    pytest.importorskip("acvl_utils.morphology.morphology_helper")
    assert run_remove_largest_component_parity(
        seed=1702, labels_or_regions=labels_or_regions
    ).passed


def test_postprocessing_pkl_matches_pinned_nnunet_oracle(tmp_path: Path) -> None:
    """Compare supported postprocessing pickle behavior with the pinned oracle."""
    pytest.importorskip("nnunetv2.postprocessing.remove_connected_components")
    pytest.importorskip("acvl_utils.morphology.morphology_helper")
    assert run_postprocessing_pkl_parity(work_dir=tmp_path, seed=2601).passed
