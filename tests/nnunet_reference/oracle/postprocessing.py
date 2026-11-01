"""Pinned nnU-Net oracle: connected-component postprocessing."""

from __future__ import annotations

import pickle
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from nnunet_mlx.prediction.postprocessing import (
    LabelOrRegion,
    PostprocessingStep,
    apply_postprocessing,
    load_postprocessing_pkl,
    remove_all_but_largest_component_from_segmentation,
)
from nnunet_reference.oracle.metrics import (
    ContractParityResult,
    ExactParityResult,
    exact_array_parity,
)

type IntArray = npt.NDArray[np.integer]


def run_remove_largest_component_parity(
    *, seed: int, labels_or_regions: LabelOrRegion | list[LabelOrRegion], background_label: int = 0
) -> ContractParityResult:
    """Compare one postprocessing step with pinned upstream nnU-Net v2.8.1."""

    source = import_module("nnunetv2.postprocessing.remove_connected_components")
    rng = np.random.default_rng(seed)
    shape = tuple(int(size) for size in rng.integers(4, 10, size=3))
    segmentation = cast(IntArray, rng.integers(0, 4, size=shape).astype(np.uint8))
    expected = cast(
        IntArray,
        np.asarray(
            source.remove_all_but_largest_component_from_segmentation(
                segmentation.copy(), labels_or_regions, background_label
            )
        ),
    )
    items = (
        labels_or_regions if isinstance(labels_or_regions, list) else [labels_or_regions]
    )
    actual = remove_all_but_largest_component_from_segmentation(
        segmentation.copy(), items, background_label
    )
    return ContractParityResult(
        arrays=(),
        exact=(exact_array_parity(expected, actual, name="remove_largest_component"),),
    )


def run_postprocessing_pkl_parity(*, work_dir: Path, seed: int) -> ContractParityResult:
    """Build a pinned-shaped ``postprocessing.pkl`` and compare full-pipeline parity.

    Reproduces what upstream's ``determine_postprocessing`` writes: one step
    that keeps the largest component across every foreground label merged,
    then one step per label. Only the inference-time ``apply_postprocessing``
    counterpart (``nnUNetv2_apply_postprocessing``) is under test; determining
    which steps help requires ground truth and is out of scope. ``work_dir`` must
    already exist; its ``postprocessing.pkl`` is replaced if present.
    """

    source = import_module("nnunetv2.postprocessing.remove_connected_components")
    rng = np.random.default_rng(seed)
    shape = tuple(int(size) for size in rng.integers(5, 11, size=3))
    segmentation = cast(IntArray, rng.integers(0, 4, size=shape).astype(np.uint8))
    foreground_labels: list[LabelOrRegion] = [1, 2, 3]
    reference_fn = source.remove_all_but_largest_component_from_segmentation
    pp_fns = [reference_fn, reference_fn, reference_fn, reference_fn]
    pp_fn_kwargs: list[dict[str, object]] = [
        {"labels_or_regions": list(foreground_labels)},
        {"labels_or_regions": 1},
        {"labels_or_regions": 2},
        {"labels_or_regions": 3},
    ]
    expected_steps = (
        PostprocessingStep(labels_or_regions=tuple(foreground_labels)),
        PostprocessingStep(labels_or_regions=(1,)),
        PostprocessingStep(labels_or_regions=(2,)),
        PostprocessingStep(labels_or_regions=(3,)),
    )

    pickle_path = work_dir / "postprocessing.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump((pp_fns, pp_fn_kwargs), handle)

    expected = cast(
        IntArray,
        np.asarray(source.apply_postprocessing(segmentation.copy(), pp_fns, pp_fn_kwargs)),
    )
    steps = load_postprocessing_pkl(pickle_path)
    actual = apply_postprocessing(segmentation.copy(), steps)
    return ContractParityResult(
        arrays=(),
        exact=(
            exact_array_parity(expected, actual, name="postprocessing_pkl_pipeline"),
            ExactParityResult(
                name="postprocessing_pkl_steps", expected=expected_steps, actual=steps
            ),
        ),
    )


__all__ = ["run_postprocessing_pkl_parity", "run_remove_largest_component_parity"]
