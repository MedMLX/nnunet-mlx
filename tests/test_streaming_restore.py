"""Parallel separate-axis argmax restoration matches materialized resampling."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_mlx.data.resampling import resample_probabilities_or_data
from nnunet_mlx.plans import ResamplingContract
from nnunet_mlx.prediction.restore import resample_class_logits_argmax_parallel


def _contract(
    *, force_separate_z: bool | None, order: int = 1, order_z: int = 0
) -> ResamplingContract:
    """Build a resampling contract for streaming restoration tests."""
    return ResamplingContract("resample_data_or_seg_to_shape", order, order_z, force_separate_z)


# Plane shortcut, interpolated out-of-plane order and an upsampled separate axis all run.
@pytest.mark.parametrize(
    ("separate_axis", "class_count", "order", "order_z", "source_shape", "target_shape"),
    [
        (0, 8, 0, 0, (3, 5, 7), (5, 6, 9)),
        (1, 17, 1, 0, (3, 5, 7), (5, 6, 9)),
        (2, 17, 0, 0, (3, 5, 7), (5, 6, 9)),
        (2, 8, 1, 0, (3, 5, 7), (5, 6, 9)),
        (0, 8, 1, 0, (8, 6, 10), (5, 11, 13)),
        (0, 8, 1, 1, (6, 8, 10), (9, 11, 13)),
        (0, 8, 1, 0, (5, 6, 7), (17, 8, 9)),
    ],
)
def test_parallel_separate_z_matches_materialized_adversarial_logits(
    separate_axis: int,
    class_count: int,
    order: int,
    order_z: int,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> None:
    """Match parallel separate-axis restoration to materialized resampling of adversarial logits."""
    spacing = [1.0, 1.0, 1.0]
    spacing[separate_axis] = 5.0
    rng = np.random.default_rng(31 + separate_axis + class_count + order)
    logits = rng.normal(size=(class_count, *source_shape)).astype(np.float32)
    logits[1] = logits[0]
    logits[2, 1, 2, 3] = np.nextafter(logits[0, 1, 2, 3], np.float32(np.inf))
    logits[-2].fill(np.finfo(np.float32).max / 16)
    logits[-1].fill(np.finfo(np.float32).min / 16)
    contract = _contract(force_separate_z=True, order=order, order_z=order_z)
    arguments = (logits, target_shape, spacing, (1.0, 1.0, 1.0), contract)

    expected = np.argmax(resample_probabilities_or_data(*arguments), axis=0)
    result = resample_class_logits_argmax_parallel(*arguments)

    assert result.flags.c_contiguous
    assert np.array_equal(result, expected)
