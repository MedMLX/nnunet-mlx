"""Filled-mask cropping compared with the pinned upstream implementation."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.oracle
def test_filled_mask_crop_bounds_match_pinned_upstream() -> None:
    """Compare filled-mask crop bounds with the pinned source implementation."""
    from nnunet_mlx.data.preprocessing import (
        crop_to_nonzero,
    )

    source = pytest.importorskip("nnunetv2.preprocessing.cropping.cropping").crop_to_nonzero
    rng = np.random.default_rng(917)
    for _ in range(100):
        data = rng.integers(0, 2, size=(2, 5, 6, 7)).astype(np.float32)
        data[:, 0] = 0
        data[:, 2, 2:4, 2:4] = 0
        expected_data, expected_mask, expected_bbox = source(data)
        actual_data, actual_mask, _, actual_bbox = crop_to_nonzero(data, None)
        np.testing.assert_array_equal(actual_data, expected_data)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        assert actual_bbox == tuple(tuple(pair) for pair in expected_bbox)
