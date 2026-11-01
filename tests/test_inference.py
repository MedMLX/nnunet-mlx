"""Label-only decisions compared with the pinned upstream label manager."""

from __future__ import annotations

import numpy as np
import pytest

from nnunet_mlx.prediction.inference import logits_to_segmentation


@pytest.mark.oracle
def test_label_only_decision_matches_original_label_manager() -> None:
    """Match label-only decisions with the source label manager."""
    labels = pytest.importorskip("nnunetv2.utilities.label_handling.label_handling")

    logits = np.array([0, 1e-8], dtype=np.float32).reshape(2, 1, 1)
    manager = labels.LabelManager({"background": 0, "target": 1}, None)
    expected = manager.convert_logits_to_segmentation(logits)
    np.testing.assert_array_equal(
        logits_to_segmentation(logits, regions_class_order=None), expected
    )
