"""Label-only Metal reduction must match the independent host-logit boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from nnunet_mlx.prediction.sliding_window import predict_labels, predict_logits

if TYPE_CHECKING:
    from mlx.core import array


@pytest.mark.parametrize(
    ("accumulation", "shape"), [("float32", (2, 3, 4)), ("nnunet-fp16", (5, 6, 7))],
)
def test_label_only_matches_normalized_host_logits(
    accumulation: str, shape: tuple[int, ...],
) -> None:
    """Match label-only inference with labels from normalized host logits."""
    from nnunet_mlx.runtime import import_mlx

    mx, _ = import_mlx()

    def model(values: array) -> array:
        """Return synthetic logits for label-only comparison."""
        return mx.concatenate((values, -values, mx.zeros(values.shape) + 1e-5), axis=-1)

    data = np.linspace(-1, 1, int(np.prod(shape)), dtype=np.float32).reshape((1, *shape))
    expected = predict_logits(
        model, data, patch_size=(4, 4, 4), num_classes=3,
        step_fraction=0.5, accumulation=accumulation,
    ).argmax(axis=0).astype(np.uint16)
    actual = predict_labels(
        model, data, patch_size=(4, 4, 4), num_classes=3,
        step_fraction=0.5, accumulation=accumulation,
    )
    np.testing.assert_array_equal(actual, expected)
