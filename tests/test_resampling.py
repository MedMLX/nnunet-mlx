"""Host resampling compared with the pinned upstream libraries."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
from nnunet_reference.oracle.network import real_image_input
from nnunet_reference.real_cases import anatomical_labels

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
    """Compare explicit segmentation modes with the source implementation."""
    resize_segmentation = import_module("batchgenerators.augmentations.utils").resize_segmentation
    segmentation = anatomical_labels()
    # Retain the upstream ignore-label case in a real source grid.
    segmentation[0, 0, 0] = -1
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


@pytest.mark.oracle
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize(
    ("order", "order_z", "force_separate_z"),
    [(1, 0, False), (3, 0, None), (1, 0, True), (1, 1, True), (3, 1, True)],
)
def test_float_resampling_matches_pinned_nnunet_bitwise(
    dtype: type[np.float16] | type[np.float32],
    order: int,
    order_z: int,
    force_separate_z: bool | None,
) -> None:
    """Interpolate in float64 and round once to the input dtype, as upstream does.

    Half input is what nnU-Net's float16 accumulation hands to restoration; the
    rounding back to float16 decides near-tie labels. An interpolating separate
    axis must not round the in-plane pass first.
    """
    from nnunet_mlx.data.resampling import resample_probabilities_or_data

    source = import_module("nnunetv2.preprocessing.resampling.default_resampling")
    data = real_image_input((7, 20, 22), 3).astype(dtype)
    arguments = ((13, 31, 17), (5.0, 1.0, 1.0), (1.0, 0.7, 1.3))
    expected = source.resample_data_or_seg_to_shape(
        data, *arguments, is_seg=False, order=order, order_z=order_z,
        force_separate_z=force_separate_z,
    )
    actual = resample_probabilities_or_data(
        data, *arguments,
        ResamplingContract("resample_data_or_seg_to_shape", order, order_z, force_separate_z),
    )
    assert expected.dtype == dtype
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))
