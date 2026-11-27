"""Training data preparation and patch sampling contracts."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image

from nnunet_mlx.training.data import PatchSampler, prepare_training_data

pytestmark = pytest.mark.host


def _write_dataset_layout(tmp_path: Path, channel_names: dict[str, str]) -> tuple[Path, Path]:
    images = tmp_path / "imagesTr"
    images.mkdir()
    labels = tmp_path / "labelsTr"
    labels.mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": channel_names,
                "labels": {"background": 0, "lesion": 1, "ignore": 2},
                "file_ending": ".nii",
            }
        ),
        encoding="utf-8",
    )
    return images, labels


def test_native_ct_plan_fits_training_foreground_only(tmp_path: Path) -> None:
    """Pool equal foreground samples per training scan and reuse the fit for validation."""
    images, labels = _write_dataset_layout(tmp_path, {"1": "MRI", "0": "cT"})
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps([{"train": ["first", "negative", "second"], "val": ["validation"]}]),
        encoding="utf-8",
    )
    affine = np.eye(4)
    shape = (16, 16, 16)
    mri = np.indices(shape, dtype=np.float32)[0] + 1
    nibabel = import_module("nibabel")
    for case, intensity, width in (
        ("first", 100, 3), ("negative", -900, 0), ("second", 300, 8), ("validation", 900, 6)
    ):
        image = np.full(shape, -1000, np.float32)
        target = np.zeros(shape, dtype=np.uint8)
        image[4 : 4 + width] = intensity
        target[4 : 4 + width] = 1
        target[1, 1, 1] = 2
        image[1, 1, 1] = 1800
        nibabel.save(Nifti1Image(image, affine), images / f"{case}_0000.nii")
        nibabel.save(Nifti1Image(mri, affine), images / f"{case}_0001.nii")
        nibabel.save(Nifti1Image(target, affine), labels / f"{case}.nii")

    data = prepare_training_data(tmp_path, tmp_path / "prepared", splits_file=split_path)
    assert data.plans.configuration.normalization_schemes == (
        "CTNormalization", "ZScoreNormalization",
    )
    assert data.plans.configuration.use_mask_for_norm == (False, True)
    assert data.plans.intensity_properties == {
        "0": {"mean": 200.0, "std": 100.0, "percentile_00_5": 100.0, "percentile_99_5": 300.0}
    }
    for case, expected in (("first", -1), ("second", 1), ("validation", 1)):
        image, target = data.arrays(case)
        np.testing.assert_array_equal(image[..., 0][target == 1], expected)


@pytest.mark.parametrize("spatial_augment", [False, True])
def test_sparse_annotation_sampling_selects_annotated_slices(
    tmp_path: Path, spatial_augment: bool
) -> None:
    """Sample only annotated slices, retaining cases with annotated background only."""
    images, labels = _write_dataset_layout(tmp_path, {"0": "MRI"})
    shape = (32, 32, 24)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    image = np.indices(shape, dtype=np.float32)[0] + 1
    nibabel = import_module("nibabel")
    for case, annotated in (("background", 0), ("lesion", 1), ("unannotated", None)):
        target = np.full(shape, 2, dtype=np.uint8)
        if annotated is not None:
            target[0, 0, 12] = annotated
        nibabel.save(Nifti1Image(image, affine), images / f"{case}_0000.nii")
        nibabel.save(Nifti1Image(target, affine), labels / f"{case}.nii")

    data = prepare_training_data(
        str(tmp_path),
        str(tmp_path / "prepared"),
        configuration="2d",
        patch_size=(16, 16),
        fold="all",
    )
    sampler = PatchSampler(
        data, batch_size=1, augment=spatial_augment, spatial_augment=spatial_augment
    )
    seen: set[int] = set()
    for _, target in sampler.batches(seed=21, epoch=1, steps=32):
        annotations = target[target >= 0]
        assert annotations.size > 0
        seen.add(int(annotations[0]))
    assert seen == {0, 1}
