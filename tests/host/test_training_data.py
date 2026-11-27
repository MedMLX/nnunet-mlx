"""Patch sampling from prepared datasets with sparse annotations."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image

from nnunet_mlx.training.data import PatchSampler, prepare_training_data

pytestmark = pytest.mark.host


@pytest.mark.parametrize("spatial_augment", [False, True])
def test_sparse_annotation_sampling_selects_annotated_slices(
    tmp_path: Path, spatial_augment: bool
) -> None:
    """Sample only annotated slices, retaining cases with annotated background only."""
    images = tmp_path / "imagesTr"
    images.mkdir()
    labels = tmp_path / "labelsTr"
    labels.mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "MRI"},
                "labels": {"background": 0, "lesion": 1, "ignore": 2},
                "file_ending": ".nii",
            }
        ),
        encoding="utf-8",
    )
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
