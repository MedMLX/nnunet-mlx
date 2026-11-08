"""Tests that fused Metal restoration matches the materialized reference."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_mlx.data.resampling import resample_probabilities_or_data
from nnunet_mlx.kernels.metal_restore import (
    resample_class_logits_argmax_metal,
    resample_device_logits_argmax,
)
from nnunet_mlx.plans import ResamplingContract

CONTRACT = ResamplingContract("resample_data_or_seg_to_shape", 1, 0, False)
SPACING = (1.0, 1.0, 1.0)


def _device_channels_last(
    logits: np.ndarray[tuple[int, ...], np.dtype[np.float16 | np.float32]],
    target_shape: tuple[int, int, int],
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint16]] | None:
    """Run device restoration on channels-last logits, or return None if unsupported."""
    from nnunet_mlx.runtime import import_mlx

    mx, _ = import_mlx()
    return resample_device_logits_argmax(
        mx.array(np.moveaxis(logits, 0, -1)), target_shape, SPACING, SPACING, CONTRACT,
    )


def _reference(
    logits: np.ndarray[tuple[int, ...], np.dtype[np.float16 | np.float32]],
    target_shape: tuple[int, int, int],
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint16]]:
    """Restore class logits using the materialized reference path."""
    restored = resample_probabilities_or_data(logits, target_shape, SPACING, SPACING, CONTRACT)
    return np.asarray(np.argmax(restored, axis=0), dtype=np.uint16)


@pytest.mark.metal
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    ("classes", "source_shape", "target_shape"),
    [
        pytest.param(2, (2, 3, 4), (3, 5, 7), id="tiny-upsample"),
        pytest.param(3, (5, 7, 11), (3, 4, 6), id="odd-downsample"),
        pytest.param(32, (1, 5, 7), (3, 7, 1), id="thin-source-and-target"),
        pytest.param(118, (4, 5, 6), (7, 8, 9), id="large-label-vocabulary"),
        pytest.param(3, (2, 3, 5), (2, 3, 5), id="identity"),
    ],
)
def test_fused_restore_matches_materialized_reference_across_shapes(
    classes: int,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    dtype: type[np.float16] | type[np.float32],
) -> None:
    """Compare fused class restoration with the reference across supported shapes.

    Float16 logits must round interpolated values to float16 before the argmax,
    as the reference does; small logits make the resulting ties frequent.
    """
    scale = 0.01 if dtype is np.float16 else 1.0
    logits = np.random.default_rng(classes).normal(
        scale=scale, size=(classes, *source_shape)
    ).astype(dtype)

    result = resample_class_logits_argmax_metal(logits, target_shape)
    device = _device_channels_last(logits, target_shape)

    assert result.dtype == np.uint16
    assert result.flags.c_contiguous
    expected = _reference(logits, target_shape)
    assert np.array_equal(result, expected)
    assert device is not None and np.array_equal(device, expected)
