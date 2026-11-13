"""Filled-mask cropping compared with the pinned upstream implementation."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.oracle
def test_filled_mask_crop_bounds_match_pinned_upstream() -> None:
    """Compare filled-mask crop bounds with the pinned source implementation."""
    from nnunet_mlx.data.preprocessing import (
        crop_to_nonzero,
    )

    source = pytest.importorskip("nnunetv2.preprocessing.cropping.cropping").crop_to_nonzero
    root = Path.home() / ".cache/medmlx/test-data"
    ct_path = Path(os.environ.get("MEDMLX_TEST_CT", str(root / "hcc_063_ct.nii.gz")))
    nibabel = import_module("nibabel")
    ct = nibabel.load(ct_path)
    liver = nibabel.load(root / "hcc_063_liver.nii.gz")
    assert ct.shape == liver.shape and np.allclose(ct.affine, liver.affine)
    voxels = np.asarray(ct.dataobj, dtype=np.float32)
    mask = np.asarray(liver.dataobj, dtype=np.int16) > 0
    rng = np.random.default_rng(917)
    checked = 0
    for _ in range(2000):
        x, y, z = (int(v) for v in rng.integers((90, 195, 45), (125, 235, 72)))
        region = (slice(x, x + 16), slice(y, y + 16), slice(z, z + 8))
        if not np.any(mask[region]):
            continue
        liver_ct = np.where(mask[region], voxels[region], np.float32(0))
        data = np.stack([liver_ct, liver_ct])
        expected_data, expected_mask, expected_bbox = source(data)
        actual_data, actual_mask, _, actual_bbox = crop_to_nonzero(data, None)
        np.testing.assert_array_equal(actual_data, expected_data)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        assert actual_bbox == tuple(tuple(pair) for pair in expected_bbox)
        checked += 1
        if checked == 100:
            break
    assert checked == 100
