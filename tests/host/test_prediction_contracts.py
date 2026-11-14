"""Regression checks for model metadata, cascade ownership and output generations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nnunet_mlx.data.preprocessing import preprocess_raw_case
from nnunet_mlx.errors import ContractError
from nnunet_mlx.model.topology import PlainConvUNetSpec, ResidualEncoderUNetSpec
from nnunet_mlx.plans import PlansContract, ResamplingContract
from nnunet_mlx.prediction.ensembling import merge_probability_files
from nnunet_mlx.prediction.export import RestoredPrediction, export_prediction

pytestmark = pytest.mark.host


def test_host_export_without_mlx_distribution_records_absent_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-only publication works when the optional Metal runtime is not installed."""
    import json
    from importlib import metadata

    from nnunet_mlx.prediction.export import prediction_is_complete

    installed = metadata.distribution

    def without_mlx(package: str) -> metadata.Distribution:
        if package == "mlx":
            raise metadata.PackageNotFoundError(package)
        return installed(package)

    monkeypatch.setattr(metadata, "distribution", without_mlx)
    output = tmp_path / "case"
    files = export_prediction(
        RestoredPrediction(np.zeros((2, 3, 4), dtype=np.uint16), None),
        {"spacing": (1, 1, 1), "nibabel_stuff": {"original_affine": np.eye(4)}},
        output, file_ending=".nii.gz", image_reader_writer="NibabelIO", overwrite=False,
    )
    assert prediction_is_complete(output, files, {})
    receipt = json.loads((tmp_path / "case.complete.json").read_text())
    assert receipt["execution_provenance"]["mlx_distribution_version"] is None


def test_cascade_normalization_preserves_foreground_and_caller_array(
    tiny_model_plans: dict[str, object],
) -> None:
    """Normalize over crop labels >= 0, including foreground outside image support."""
    plans = PlansContract.from_payload(tiny_model_plans, configuration="3d_fullres")
    plans = replace(plans, transpose_forward=(0, 1, 2), transpose_backward=(0, 1, 2),
                    configuration=replace(
        plans.configuration, normalization_schemes=("ZScoreNormalization",),
        use_mask_for_norm=(True,),
    ))
    image = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3)
    image[0, 0, 0, 1] = 0
    previous = np.ones(image.shape, dtype=np.int32)
    previous[0, 0, 0, 1] = 0
    before = previous.copy()
    population = image[previous > 0]
    expected = np.zeros_like(image)
    expected[previous > 0] = (population - population.mean()) / population.std()
    for _ in range(2):
        actual = preprocess_raw_case(
            image, {"spacing": (1, 1, 1)}, plans, segmentation_previous_stage=previous,
        )
        np.testing.assert_allclose(
            actual.data, expected, rtol=1e-6, atol=1e-7,
        )
        np.testing.assert_array_equal(previous, before)


@pytest.mark.parametrize("spec_name", ["plain", "residual"])
@pytest.mark.parametrize("option", ["even_kernel", "large_slope"])
def test_unqualified_topologies_are_rejected(
    tiny_spec: PlainConvUNetSpec, tiny_residual_spec: ResidualEncoderUNetSpec,
    spec_name: str, option: str,
) -> None:
    """Reject settings whose fused/padded implementation differs from upstream."""
    spec = tiny_spec if spec_name == "plain" else tiny_residual_spec
    with pytest.raises(ContractError):
        if option == "even_kernel":
            replace(spec, kernel_sizes=((2, 2, 2), *spec.kernel_sizes[1:]))
        else:
            replace(spec, leaky_slope=2.0)


def test_unknown_spline_option_is_rejected() -> None:
    """A misspelled interpolation option must not silently select another algorithm."""
    with pytest.raises(ContractError):
        ResamplingContract.from_configuration(
            {"resampling_fn_data": "resample_data_or_seg_to_shape",
             "resampling_fn_data_kwargs": {"is_seg": False, "order": 1, "order_z": 0,
                                           "force_separate_zz": True}},
            field="resampling_fn_data", expected_is_seg=False,
        )


def test_ensemble_rejects_different_probability_orientations(tmp_path: Path) -> None:
    """Equal original affines do not align native LAS and canonical RAS arrays."""
    from importlib import import_module

    from nnunet_mlx.data.labels import parse_dataset_label_semantics
    from nnunet_mlx.storage.imageio import read_images

    nib = import_module("nibabel")
    original = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = np.zeros((4, 2, 2), dtype=np.float32)
    image[0] = 1
    source = tmp_path / "input.nii.gz"
    nib.save(nib.Nifti1Image(image, original), source)
    files: list[Path] = []
    for reader in ("NibabelIO", "NibabelIOWithReorient"):
        data, properties = read_images(reader, [source])
        foreground = data[0] > 0
        result = export_prediction(
            RestoredPrediction(foreground.astype(np.uint16),
                               np.stack((~foreground, foreground)).astype(np.float32)),
            properties, tmp_path / reader, file_ending=".nii.gz", image_reader_writer=reader,
            overwrite=False,
        )
        assert result.probabilities is not None
        files.append(result.probabilities)
    with pytest.raises(ContractError):
        merge_probability_files(
            files, tmp_path / "merged", file_ending=".nii.gz", image_reader_writer="NibabelIO",
            semantics=parse_dataset_label_semantics({"labels": {"background": 0, "lesion": 1}}),
            overwrite=False, save_probabilities=False,
        )
    assert not (tmp_path / "merged.nii.gz").exists()


def test_publication_race_preserves_competitor_and_rolls_back_owned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-clobber failure partway through a generation removes only our own links."""
    import os

    target = tmp_path / "case.npz"
    link = os.link

    def competing_link(source: Path, destination: Path) -> None:
        if destination == target:
            target.write_bytes(b"competing probabilities")
        link(source, destination)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(ContractError):
        export_prediction(
            RestoredPrediction(np.zeros((2, 3, 4), dtype=np.uint16),
                               np.ones((1, 2, 3, 4), dtype=np.float32)),
            {"spacing": (1, 1, 1), "nibabel_stuff": {"original_affine": np.eye(4)}},
            tmp_path / "case", file_ending=".nii.gz", image_reader_writer="NibabelIO",
            overwrite=False,
        )
    assert target.read_bytes() == b"competing probabilities"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["case.npz"]


def test_label_only_export_preserves_existing_sidecars_without_overwrite(tmp_path: Path) -> None:
    """Every member of an existing generation is protected by overwrite=False."""
    for suffix in (".npz", ".pkl"):
        (tmp_path / f"case{suffix}").write_bytes(b"previous generation")
    with pytest.raises(ContractError):
        export_prediction(
            RestoredPrediction(np.zeros((2, 3, 4), dtype=np.uint16), None),
            {"spacing": (1, 1, 1), "nibabel_stuff": {"original_affine": np.eye(4)}},
            tmp_path / "case", file_ending=".nii.gz", image_reader_writer="NibabelIO",
            overwrite=False,
        )
    assert not (tmp_path / "case.nii.gz").exists()
    for suffix in (".npz", ".pkl"):
        assert (tmp_path / f"case{suffix}").read_bytes() == b"previous generation"
