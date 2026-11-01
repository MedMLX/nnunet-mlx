"""Host resampling compared with the pinned upstream libraries."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from nnunet_mlx.data.resampling import resample_segmentation
from nnunet_mlx.plans import ResamplingContract


def _segmentation_contract(
    mode: str | None, *, order: int = 1, order_z: int = 0,
) -> ResamplingContract:
    """Build a segmentation resampling contract for label tests."""
    kwargs: dict[str, object] = {
        "is_seg": True, "order": order, "order_z": order_z, "force_separate_z": None,
    }
    if mode is not None:
        kwargs["seg_tiebreak"] = mode
    return ResamplingContract.from_configuration(
        {"resampling_fn_seg": "resample_data_or_seg_to_shape",
         "resampling_fn_seg_kwargs": kwargs},
        field="resampling_fn_seg", expected_is_seg=True,
    )


@pytest.mark.oracle
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("mode", ["nearest", "lowest", "highest"])
def test_explicit_segmentation_modes_match_batchgenerators(
    axis: int | None, order: int, mode: str,
) -> None:
    # Thin, disconnected structures and three-label junctions exercise argmax
    # scores below 0.5. The locked upstream library is independent of this port.
    """Compare explicit segmentation modes with the source implementation."""
    resize_segmentation = import_module("batchgenerators.augmentations.utils").resize_segmentation
    segmentation = np.full((5, 6, 7), -1, dtype=np.int16)
    segmentation[1:4, 1:5, 1:6] = 2
    segmentation[:, 3, 3] = 7
    segmentation[2, :, 4] = 11
    target: list[int] = [8, 9, 10] if axis is None else list(segmentation.shape)
    spacing = [1.0, 1.0, 1.0]
    if axis is not None:
        target[axis] = 2 * target[axis] - 1
        spacing[axis] = 6.0
    expected = resize_segmentation(segmentation, target, order=order, seg_tiebreak=mode)
    result = resample_segmentation(
        segmentation[None], target, spacing, (1, 1, 1),
        _segmentation_contract(mode, order=order, order_z=order),
    )
    np.testing.assert_array_equal(result[0], expected)
    assert result.dtype == segmentation.dtype
    assert result.flags.c_contiguous


@pytest.mark.oracle
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_gather_matches_installed_scipy_half_pixel_mapping(axis: int) -> None:
    """Match axis sampling to SciPy half-pixel coordinate interpolation."""
    from nnunet_mlx.data.resampling import coordinate_map, nearest_axis_resize

    map_coordinates = import_module("scipy.ndimage").map_coordinates

    for old in range(1, 15):
        for new in range(1, 15):
            shape = [3, 4, 5]
            shape[axis] = old
            target = list(shape)
            target[axis] = new
            array = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            target_shape = (target[0], target[1], target[2])
            expected = map_coordinates(
                array, coordinate_map(shape, target_shape), order=0, mode="nearest",
            )
            np.testing.assert_array_equal(nearest_axis_resize(array, target_shape), expected)
